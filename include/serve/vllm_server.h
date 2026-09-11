/* ================================================================
 * vllm_server.h - OpenAI-compatible API server for vllm_kestrel
 *
 * Zero-dependency HTTP endpoints on top of vllm_http.c:
 *   GET  /v1/models
 *   GET  /health                    (liveness + queue/inference stats)
 *   POST /v1/chat/completions       (JSON + SSE streaming)
 *   POST /v1/completions            (legacy prompt completion)
 *
 * The model must be loaded by the caller (main.c) and handed in via
 * VLLMServerCtx. The engine holds a single inference state, so
 * inference-critical sections are serialized with an internal mutex:
 * concurrent clients queue up (bounded; overflow gets HTTP 429) and are
 * served one at a time. Metadata endpoints (/v1/models, /health) run
 * concurrently without blocking on the inference lock.
 * ================================================================ */
#ifndef VLLM_SERVER_H
#define VLLM_SERVER_H

#include "vllm_safetensors.h"
#include "vllm_tokenizer_qwen.h"
#include "vllm_http.h"
#include "vllm_vision.h"
#include "vllm_media.h"
#include <pthread.h>

/* Per-user L3 eviction isolation (--l3-evict): each user (the request JSON
 * "user" field, "anon" fallback) gets its own L3 cache file. Files whose
 * owner has been idle for --l3-user-ttl minutes are deleted on the next
 * request (lazy offline cleanup). The table below is guarded by inf_lock
 * (touched only while the inference lock is held, i.e. single-threaded). */
typedef struct {
    char user[128];        /* user id */
    char fname[256];       /* per-user L3 cache file path */
    long last_active;      /* unix seconds of the user's last request */
} L3UserEntry;

/* ================================================================
 * 内存驻留策略（档位阶梯，serve 动态调整 / 空闲逐级降级 / 软硬水位）
 *
 * 档位（level，0..4，数值越低内存占用越省）：
 *   L4 全程全层驻留（默认）：KV 全驻留（g_l3_evict=0）+ 权重不主动释放；
 *   L3 KV 逐层淘汰：下轮全量 prefill 后按 importance 逐层淘汰冷块到 L3
 *     磁盘 Q4（g_l3_evict=1；前缀复用随之关闭）；
 *   L2 权重窗口驻留：KV 淘汰保持 + 释放 layer>w_keep 的权重段物理页
 *     （仅明文 VQF + 启动 VLLM_VQF_STREAM 分层模式可用；缺页自动回驻）；
 *   L1 初始层驻留：再释放到只剩 layer 0..0（"从高向低最后到初始层"）；
 *   L0 KV落盘善后 + 整模卸载（load_state=0，彻底退出）。
 *
 * 触发源（三者可独立开）：
 *   - auto_idle：模型级全局空闲（ctx->last_infer_s 之后无推理请求）超过
 *     idle_s[level] 秒 → 降一档；idle_s[0]=0 表示在 L0 已卸载不再动作。
 *   - mem_soft_mb：可用内存低于软水位 → 自动降一档（15s 冷却），最低降到
 *     mem_low_lvl；动作不足时由 ctx->min_free_mb 硬门在请求入口 503。
 *   - 手动：/admin/api/policy/set 立即升降档。
 *
 * 能力边界（vllm_res_level 会做档位钳制）：
 *   权重档 L2/L1 需要 vqf_stream_active()（明文 VQF + VLLM_VQF_STREAM）；
 *   未启用时 L2/L1 无实际动作 → 可执行档只有 {4,3,0}。
 * ================================================================ */
typedef struct {
    int   auto_idle;        /* 空闲自动逐级降级开关（0=关，兼容旧 auto-unload） */
    long  idle_s[5];        /* idle_s[l]：在档位 l 停留 idle_s[l] 秒无推理 → 降档
                             * （0 = 在该档永不自动降）。索引 = 档位号。 */
    int   mem_soft_mb;      /* 软水位（MB，0=off）：可用内存低于此 → 自动降一档 */
    int   mem_low_lvl;      /* 自动动作允许的最低档（默认 0 = 允许最终卸载；
                             * 设 3 可禁止 KV 淘汰以上的降级、设 1 保留加载） */
    int   w_keep;           /* L2 权重窗口：保留层 0..w_keep（默认 8） */
    int   l3_forced_sparse; /* 内部：进入 L3 档时把 g_sparse_attn 从 0 强开为 1
                             * （L3 依赖 sparse decode）；离开时恢复 0。 */
    /* ---- 内部运行状态（tick / set 维护） ---- */
    int   level;            /* 当前档位（0..4；load_state!=2 时视为 0） */
    long  entered_s;        /* 进入当前档位的时刻（epoch 秒，空闲计时起点） */
    long  last_mem_s;       /* 上次"内存水位触发降档"时刻（15s 冷却） */
    int   steps;            /* 自动（空闲/水位）降档累计次数（观测） */
} VLLMResPolicy;

/* Tagged struct：vllm_attest.h 用 `typedef struct VLLMServerCtx VLLMServerCtx`
 * 前向声明，无需 include 本头即可在 API 签名中使用该类型。 */
typedef struct VLLMServerCtx {
    STModelWeights *w;         /* loaded model weights */
    STModelConfig  *cfg;       /* model config */
    QwenTokenizer  *tok;       /* loaded tokenizer */
    STQwenInferenceState *ist; /* persistent inference state (KV cache) */
    const char *model_id;      /* advertised model id */
    STVisionState *vis;        /* vision encoder state (NULL = text-only) */

    /* Robustness knobs (set by caller before vllm_server_run). */
    int max_queued;            /* max waiting inference requests (0 = unlimited) */
    int min_free_mb;           /* refuse inference below this free RAM (MB, 0 = off) */

    /* Continuous batching (--batch-max N, 0/1 = off/serial). When enabled the
     * per-request decode steps are merged via st_qwen_model_forward_batch so
     * the layer weights are read once per step for all concurrent requests
     * (block_matrix_assoc_natural: batch weight reuse, design doc §9.1). */
    int batch_max;             /* max concurrent decode requests (0 = off) */
    struct VBatchSched *batch; /* scheduler (created lazily after model load) */

    /* Admin/management metadata (mirrored from the command line by the
     * caller so /admin can report and persist the engine's core config
     * without reaching into main.c statics). */
    const char *model_dir;     /* --model path (may be repointed to model_dir_buf) */
    char   model_dir_buf[1024];/* writable copy set by /admin/api/model/load (portable
                                * manual-load mode: the packaged exe may start with no
                                * model on disk; the admin page supplies the path) */
    char   model_name[128];    /* real model identity filled after load (e.g.
                                * "Qwen3-VL-2B (dim=2048 layers=28)"); "" until then */
    int   port;                /* serving port */
    int   threads;             /* HTTP worker threads */
    int   npu_enabled;         /* --npu offload requested */
    int   prefix_cache;        /* --prefix-cache */
    int   prefix_kv;           /* serve: KV prefix reuse (default on;
                                * --no-prefix-kv disables) */
    double min_p;              /* min-p sampling filter (0 = off); request body
                                * min_p overrides. Applied before top-p. */
    int   top_k;               /* top-k sampling truncation (0 = off); request
                                * body top_k overrides. Qwen3 thinking 档建议 20。 */
    long  started_s;           /* epoch seconds at serve start (uptime) */
    const char *config_path;   /* where /admin config/save persists JSON */
    const char *log_path;      /* engine stdout/stderr log file (log tail) */

    /* Device profile (x86-pc / arm-rk3588-opi5 / arm-generic ...), set by
     * the caller from vllm_device.h detection; reported by /admin and used
     * to pre-fill the weight-adaptation recommendations. */
    const char *device_id;     /* "arm-rk3588-opi5" */
    const char *device_name;
    const char *device_arch;   /* "x86"/"arm64" */
    const char *device_class;  /* "pc"/"embedded" */
    const char *device_npu;    /* "rk3588-direct" or "-" */

    /* Verifiable-inference attestation (方案 2, vllm_attest.c): when
     * VLLM_ATTEST=1 the engine SM2-signs each completed response transcript.
     * attest_fp = model fingerprint (hex) pinned by vatt_model_ready after
     * every successful load; attest_dir = VLLM_ATTEST_DIR key directory. */
    int  attest_on;
    char attest_fp[65];        /* 64-hex model fingerprint, "" until load */
    char attest_dir[512];      /* device key directory (VLLM_ATTEST_DIR) */

    /* Model-load lifecycle (manual-load mode: the HTTP server starts with
     * the model not loaded; /admin/api/model/load runs the load and sets
     * load_state=2 on success). 0=idle, 1=loading, 2=ready, 3=failed. */
    volatile int  load_state;
    char load_error[256];
    void *load_arg;            /* opaque load context (main.c ServeModel) */
    int load_format;           /* 加载格式（--load-format，CLI 专属；v1.0 纯 VQF
                                * 运行时下 0=auto 与 1=vqf 等价，保留兼容） */

    /* Load-on-use / unload-when-idle lifecycle (VQF 等自研格式：用时加载、
     * 不用时卸载，释放权重+KV 内存；卸载后上下文状态（KV/prefix/batch）清零，
     * 下次加载从头开始）。 */
    volatile int  unload_pending;  /* 1 = unload in progress (new inference -> 503) */
    int   auto_load;            /* 1 = 请求到达且模型未加载时自动加载（用时加载） */
    long  auto_unload_s;        /* 空闲 N 秒后自动卸载（0 = 关闭，不用时卸载） */
    long  last_infer_s;         /* 上次推理完成时刻（epoch 秒） */
    long  last_load_ms;         /* 最近一次模型加载耗时（毫秒，启动时间统计） */
    int   unload_count;         /* 已完成的卸载次数（显式+自动） */
    int   auto_unload_count;    /* 自动（空闲）卸载次数 */
    volatile int  stopping;     /* 服务关闭标志（停止空闲卸载监控线程） */
    pthread_t idle_thr;         /* 空闲卸载监控线程 */
    int   idle_thr_started;

    /* Robustness state (internal, initialized by vllm_server_run). */
    VHttpMutex *inf_lock;      /* serializes inference (one at a time) */
    VHttpMutex *stat_lock;     /* protects the counters below */
    long total_requests;       /* inference requests accepted since start */
    long active_requests;      /* queued + running */
    long queued_requests;      /* waiting for the inference lock */
    int  busy;                 /* 1 = inference running right now */
    volatile int inflight_handlers; /* 1 = any HTTP handler executing (unload gate:
                                     * prevents freeing ctx->ist/cfg/w while a worker
                                     * is between ensure_model_ready and infer_gate) */
    long next_chunk_id;        /* SSE chunk id counter */

    L3UserEntry *l3_users;     /* per-user L3 session table (grown on demand) */
    int  l3_users_n;
    int  l3_users_cap;
    char l3_cur_fname[256];    /* L3 file for the request being served */

    /* KV prefix reuse state: the FULL token sequence (prompt + generated)
     * whose KV currently sits in ctx->ist. The next request computes its
     * longest common prefix against this; the shared prefix KV is kept and
     * only the remainder is prefilled (ds4.c disk-KV-cache idea, in-RAM).
     *
     * 注意（P4 实测 2026-09-07）：前缀复用是"确定性"的（同路径可复现）但
     * **并非全量重算的位级克隆**——对抗性双轮输入下贪心输出可在生成中段分叉
     * （即使把复用边界截断到纯 prefill 行也不消除，见方案文档 §7.2 P4）。 */
    int  *last_ids;            /* previous request's full token sequence */
    int   last_n;              /* length (0 = none yet) */
    int   last_cap;            /* allocated capacity of last_ids */

    /* P4 multimodal prefix reuse: the previous request was multimodal, so
     * last_ids/last_mark hold its prompt+generated token sequence AND a
     * per-token media-content mark (0 = text/generated row; non-zero =
     * 64-bit FNV of the image/video bytes occupying that KV row). The next
     * multimodal request compares BOTH token ids and marks so a reused KV
     * prefix is only honored when the same image bytes sit at the same rows
     * (token-id equality alone would be fooled by equal pad counts). Text
     * requests never reuse across last_was_mm=1 and vice versa. */
    int  last_was_mm;          /* 1 = last saved request was multimodal */
    unsigned long long *last_mark; /* aligned 1:1 with last_ids */
    int  last_mark_cap;

    /* Disk KV persistence (--disk-kv DIR): F32 KV snapshots survive a process
     * restart. Each completed text request writes kv_<fnv64>.kv (token ids +
     * F32 KV, see st_kv_disk_save/load). On a request the checkpoint whose
     * tokens share the longest prefix is loaded into the cache and only the
     * remainder is prefilled. dkv_list is rebuilt by vllm_server_diskkv_scan
     * after every model load and after each save. */
    int   disk_kv;             /* 1 = disk KV persistence enabled */
    char  kvdir[512];          /* checkpoint directory */
    struct DiskKVEntry *dkv;   /* index: [dkv_n] entries (tokens+path) */
    int   dkv_n;
    int   dkv_cap;

    /* Speculative decode (--spec): n-gram draft + parallel prefill
     * verification. Greedy-only (temperature<=0, top_p>=1, min_p<=0) and
     * text-only; the verified draft tokens are emitted together (the prefill
     * reads the weights once for all K draft positions, ~Kx fewer weight
     * loads than K single-token decodes when the draft hits). */
    int   spec;                /* 1 = enable n-gram speculative decode */
    int   spec_k;              /* draft length (default 4) */

    /* 内存驻留策略（档位阶梯）：空闲逐级降级 + 软水位主动降级 + 管理页手动
     * 升降档。默认值由 vllm_res_defaults 装配；动作在 policy tick / admin
     * API 中以 inf_lock 串行执行（与推理互斥）。 */
    VLLMResPolicy res;
} VLLMServerCtx;

/* One on-disk KV checkpoint registered in ctx->dkv. */
typedef struct DiskKVEntry {
    char  path[512];           /* full checkpoint file path */
    int   n_tokens;            /* tokens stored in the checkpoint */
    int  *tokens;              /* [n_tokens] token ids (malloc'd) */
    long  mtime;               /* file mtime (unix seconds, LRU eviction) */
} DiskKVEntry;

/* Rebuild ctx->dkv by scanning ctx->kvdir for valid KV checkpoints whose
 * geometry matches the loaded model. Safe to call after each model load;
 * frees the previous list first. No-op when disk_kv is off. */
void vllm_server_diskkv_scan(VLLMServerCtx *ctx);

/* Blocking: loads nothing, serves until killed. n_threads = HTTP worker
 * threads. Returns 0 on clean exit. */
int vllm_server_run(VLLMServerCtx *ctx, int port, int n_threads,
                    void (*on_start)(int actual_port, void *ud));

/* Start an asynchronous model load (manual-load mode, main.c). The HTTP
 * server keeps running; progress is reported via ctx->load_state and the
 * g_model_load_layer/total globals. Returns 0 = started, 1 = already
 * loading, 2 = already ready. */
int vllm_serve_start_load(VLLMServerCtx *ctx);

/* Unload the loaded model (VQF): frees weights, tokenizer,
 * inference state (KV cache), batch scheduler and the in-RAM prefix-KV
 * context (last_ids) so the next load starts from a clean slate. Returns
 * 0 = unloaded, -1 = no model loaded, -2 = still loading, -3 = busy (an
 * inference request is running or queued). Impl in main.c (owns ServeModel).
 * Safe to call from an HTTP worker or the idle-unload monitor thread. */
int vllm_serve_unload_model(VLLMServerCtx *ctx);

/* Same as vllm_serve_unload_model but callable from an HTTP handler / policy
 * request thread: temporarily subtracts the caller's own inflight_handlers
 * count (server_handler counts the caller before dispatch, which would make
 * the unload's busy check reject itself with -3 forever). The caller must
 * not touch ctx->ist/cfg/w after a successful unload. Impl in main.c. */
int vllm_serve_unload_model_self(VLLMServerCtx *ctx);

/* Load-on-use: ensure a model is loaded before serving a request. Starts a
 * background load if needed and waits for it to finish. Returns 0 = ready,
 * -1 = cannot load (no model_dir / load failed). Impl in main.c. */
int vllm_serve_load_on_use(VLLMServerCtx *ctx);

/* Free the server-side session state tied to the loaded model: continuous
 * batching scheduler, the in-RAM prefix-KV sequence (last_ids) and the disk
 * KV index. Call on unload so no stale context survives a reload. */
void vllm_server_reset_session(VLLMServerCtx *ctx);

/* Point the server at a model directory (used by /admin/api/model/load in
 * portable manual-load mode, where the exe may have started without a model
 * on disk). Copies dir into ctx->model_dir_buf and repoints ctx->model_dir. */
void vllm_serve_set_model_dir(VLLMServerCtx *ctx, const char *dir);

/* Per-request inference metrics, filled by the completion path and returned
 * to the chat client in the response body ("metrics") or as a final SSE
 * event (object=chat.completion.metrics). All times in milliseconds. */
typedef struct {
    double start_s;         /* monotonic seconds at request start */
    int    prompt_tokens;   /* input token count (prefill) = total context length */
    int    last_user_tokens;/* token count of the last user message (text only) */
    int    n_tokens;        /* generated tokens */
    double prefill_ms;      /* start -> prefill done */
    double ttft_ms;         /* start -> first token produced */
    double tpot_ms;         /* avg per-token decode time after the first */
    double total_ms;        /* start -> last token */
} VLLMMetrics;

/* ================================================================
 * 内存驻留策略 API（实现于 vllm_server.c；动作持 inf_lock 与推理互斥）
 * ================================================================ */

/* 装配策略默认值：level=L4、auto_idle 关、idle_s 默认空闲计划
 * {5m,15m,30m,60m}（L4→L3→L2→L1→L0）、w_keep=8、mem_soft_mb=0。
 * CLI/env 覆盖值应在调用本函数后写入。 */
void vllm_res_defaults(VLLMServerCtx *ctx);

/* 当前有效档位（0..4）：load_state!=2 视为 0；无权重分层能力（vqf_stream
 * 未启用）时 L2/L1 钳制为 3（不可执行的档不落地）。返回 0..4。 */
int  vllm_res_level(const VLLMServerCtx *ctx);

/* 档位显示名（"L4-full-resident" ... "L0-unloaded"）。 */
const char *vllm_res_level_name(int lvl);

/* 立即切换策略档位到 target（0..4）：逐档执行升/降动作（动作见模块顶部
 * 注释），拒绝推理进行中的卸载（返回 -3 语义沿用 unload）。成功返回新档位；
 * 目标与当前相同返回当前档。 */
int  vllm_res_set_level(VLLMServerCtx *ctx, int target);

/* 策略周期心跳（由 serve policy 线程每秒调用；无该线程时由请求门内联触发
 * 亦可）：空闲超时逐级降档 + 软水位降档 + 档位与 load_state 同步。 */
void vllm_res_tick(VLLMServerCtx *ctx);

/* 卸载前的 KV 善后：文本会话已完成即已落盘（dkv_save），此处仅作兜底日志/
 * 可选显式保存（disk_kv 开且上下文未落盘时）。返回 0。 */
int  vllm_res_persist_session(VLLMServerCtx *ctx);

#endif /* VLLM_SERVER_H */

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
    int load_wmode;            /* wmode requested for the next load (-1 = keep) */
    int load_format;           /* 加载格式优先级: 0=自动(VQF>GGUF>safetensors),
                                * 1=VQF优先, 2=safetensors优先, 3=GGUF优先 */
    char load_format_str[16];  /* 管理页回显用（"auto"/"vqf"/"safetensors"/"gguf"） */

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
     * only the remainder is prefilled (ds4.c disk-KV-cache idea, in-RAM). */
    int  *last_ids;            /* previous request's full token sequence */
    int   last_n;              /* length (0 = none yet) */
    int   last_cap;            /* allocated capacity of last_ids */

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

/* Unload the loaded model (VQF/GGUF/safetensors): frees weights, tokenizer,
 * inference state (KV cache), batch scheduler and the in-RAM prefix-KV
 * context (last_ids) so the next load starts from a clean slate. Returns
 * 0 = unloaded, -1 = no model loaded, -2 = still loading, -3 = busy (an
 * inference request is running or queued). Impl in main.c (owns ServeModel).
 * Safe to call from an HTTP worker or the idle-unload monitor thread. */
int vllm_serve_unload_model(VLLMServerCtx *ctx);

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

#endif /* VLLM_SERVER_H */

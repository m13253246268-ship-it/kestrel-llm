/* ================================================================
 * vllm_admin.c - Web management API for vllm_kestrel (/admin/*)
 *
 * A small zero-dependency management surface served from the SAME HTTP
 * listener as the OpenAI API:
 *   GET  /admin/                  management single-page app (admin.html)
 *   GET  /admin/api/status        engine + system status (JSON)
 *   GET  /admin/api/config        current core configuration (JSON)
 *   POST /admin/api/config/save   persist configuration (JSON body)
 *   GET  /admin/api/log?n=200     tail of the engine log file (JSON)
 *   POST /admin/api/shutdown      graceful shutdown (vhttp_stop)
 *
 * Process start/stop supervision lives in tools/ops/vllm_mgr.py (a stopped
 * engine cannot start itself); the admin page drives it over a second
 * tiny HTTP port (default 8082).
 *
 * Config persisted by /admin/api/config/save is a plain JSON file that
 * tools/ops/vllm_mgr.py reads to build the engine's command line on start.
 * ================================================================ */
#include "vllm_server.h"
#include "vllm_batch.h"      /* vllm_batch_active (continuous batching status) */
#include "vqf.h"             /* vqf_stream_active (res policy 权重档能力) */
#include "vllm_http.h"
#include "vllm_device.h"
#include "vllm_i18n.h"       /* vllm_tr：少量混排文案按 VLLM_LANG 二选一 */
#include "embedded_web.h"
#include "vllm_util.h"       /* st_util_utf8：JSON 写入口的 UTF-8 规整 */
#include "vllm_rulebook.h"   /* 规则包注册中心（管理面：列表/删除） */
#include "vllm_session.h"    /* 会话存储（管理面：列表/详情/删除） */
#include "vllm_audit.h"      /* 管理面审计日志（SM3 哈希链） */

/* VQF 分层驻留运行状态（vqf.c；避免在 serve 侧引入完整 VQF 头） */
extern int vqf_stream_state(int *keep, int *nl, int *nseg,
                            unsigned long long *per_layer, int *cold);

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>

#include <ctype.h>

#include <unistd.h>
#include <sys/stat.h>
#ifdef _WIN32
/* Windows 无 sys/wait.h（fork/waitpid 端点仅 POSIX 编译侧需要）；
 * direct.h/io.h 提供 MSVC 兼容入口，S_ISDIR/S_ISREG 在此手动补足。 */
#include <direct.h>
#include <io.h>
#define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
#else
#include <sys/wait.h>
#endif
#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <pthread.h>

/* ---------- paths (env-overridable, board defaults) ---------- */

static const char *admin_config_path(const VLLMServerCtx *ctx) {
    if (ctx->config_path && ctx->config_path[0]) return ctx->config_path;
    const char *e = getenv("VLLM_ADMIN_CONFIG");
    if (e && e[0]) return e;
    return "/NewVLLM/webmgr/config.json";
}

static const char *admin_log_path(const VLLMServerCtx *ctx) {
    if (ctx->log_path && ctx->log_path[0]) return ctx->log_path;
    const char *e = getenv("VLLM_ADMIN_LOG");
    if (e && e[0]) return e;
    return "/NewVLLM/logs/engine.log";
}

static const char *admin_html_path(void) {
    const char *e = getenv("VLLM_ADMIN_HTML");
    if (e && e[0]) return e;
    /* try the working directory first, then the board default */
    FILE *f = fopen("./admin.html", "rb");
    if (f) { fclose(f); return "./admin.html"; }
    return "/NewVLLM/admin.html";   /* deployed next to the engine */
}

/* 审查台独立页（/review）的磁盘候选路径；VLLM_REVIEW_HTML 可覆盖。 */
static const char *review_html_path(void) {
    const char *e = getenv("VLLM_REVIEW_HTML");
    if (e && e[0]) return e;
    FILE *f = fopen("./review.html", "rb");
    if (f) { fclose(f); return "./review.html"; }
    return "/NewVLLM/review.html";
}

/* ---------- system readings (Linux /proc; null elsewhere) ---------- */

static long proc_rss_mb(void) {
#ifdef __linux__
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            kb = atol(line + 6);
            break;
        }
    }
    fclose(f);
    return kb > 0 ? kb / 1024 : -1;
#else
    return -1;
#endif
}

static long proc_mem_mb(const char *key) {
#ifdef __linux__
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, key, strlen(key)) == 0) {
            kb = atol(line + strlen(key));
            break;
        }
    }
    fclose(f);
    return kb > 0 ? kb / 1024 : -1;
#else
    return -1;
#endif
}

/* Process CPU usage vs the last sample (relative to one core, e.g.
 * 300 = three cores saturated). First call returns -1 (no delta yet). */
static double proc_cpu_pct(void) {
#ifdef __linux__
    static unsigned long long last_total = 0, last_proc = 0;
    static int have_last = 0;
    unsigned long long total = 0, proc = 0;
    FILE *f = fopen("/proc/stat", "r");
    if (f) {
        char line[512];
        if (fgets(line, sizeof(line), f) && strncmp(line, "cpu ", 4) == 0) {
            char *p = line + 4;
            for (int i = 0; i < 8; i++) {
                while (*p == ' ') p++;
                total += strtoull(p, &p, 10);
            }
        }
        fclose(f);
    }
    f = fopen("/proc/self/stat", "r");
    if (f) {
        char line[1024];
        if (fgets(line, sizeof(line), f)) {
            /* fields after the comm field: skip past last ')' */
            char *p = strrchr(line, ')');
            if (p) {
                p++;   /* skip ')' */
                int field = 2;   /* next token is field 3 */
                char *tok = NULL;
                for (; field <= 15; field++) {
                    while (*p == ' ') p++;
                    tok = p;
                    while (*p && *p != ' ') p++;
                    if (*p) { *p = '\0'; p++; }
                    if (field == 14 || field == 15) proc += strtoull(tok, NULL, 10);
                }
            }
        }
        fclose(f);
    }
    if (total == 0 || proc == 0) return -1;
    if (!have_last) {
        last_total = total; last_proc = proc; have_last = 1;
        return -1;
    }
    unsigned long long dt = total - last_total;
    unsigned long long dp = proc - last_proc;
    last_total = total; last_proc = proc;
    if (dt == 0) return 0;
    long ncpu = 1;
#ifdef _SC_NPROCESSORS_ONLN
    ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu < 1) ncpu = 1;
#endif
    return (double)dp / (double)dt * 100.0 * (double)ncpu;
#else
    return -1;
#endif
}

/* ---------- config helpers ---------- */

static const char *wmode_str(int w) {
    switch (w) {
    case 1:  return "q4";
    case 2:  return "q8";
    case 3:  return "q4i";
    case 4:  return "g256";
    case 5:  return "q2mix";
    default: return "dual";
    }
}

static const char *getenv_int_str(const char *name, const char *dflt) {
    const char *e = getenv(name);
    return (e && e[0]) ? e : dflt;
}

/* ---------- VQF 文件内固化布局（只读展示） ----------
 * 量化模式在转换期已由 vqf_convert --wmode 固化进 VQF 文件，加载侧不可更改；
 * 管理页不再提供"量化模式/加载格式"选择，改为回读文件头 flags 做只读展示。 */
static void lappend(char *buf, size_t cap, size_t *n, const char *s) {
    if (cap == 0 || *n >= cap) return;
    int k = snprintf(buf + *n, cap - *n, "%s", s);
    if (k > 0) {
        *n += (size_t)k;
        if (*n >= cap) *n = cap - 1;
    }
}

static void vqf_layout_str(uint32_t flags, uint32_t version,
                           char *buf, size_t cap) {
    static const struct { uint32_t bit; const char *name; } items[] = {
        { VQF_FLAG_Q8BUF_Q4, "q4i" },        /* q8_* 存 pre-unpacked Q4 int8 */
        { VQF_FLAG_G256,     "g256" },
        { VQF_FLAG_Q8_8X8,   "q8-8x8" },
        { VQF_FLAG_Q4_4X4,   "q4-4x4" },
        { VQF_FLAG_X8,       "x8-8x8l" },
        { VQF_FLAG_EMB_F16,  "emb-f16" },
        { VQF_FLAG_VISION,   "vision" },
        { VQF_FLAG_ENC,      "enc" },
        { VQF_FLAG_SIGNED,   "signed" },
        { VQF_FLAG_MOE,      "moe" },
    };
    char tmp[48];
    size_t n = 0;
    if (cap == 0) return;
    snprintf(tmp, sizeof(tmp), "v%u", version);
    lappend(buf, cap, &n, tmp);
    for (size_t i = 0; i < sizeof(items) / sizeof(items[0]); i++) {
        if (!(flags & items[i].bit)) continue;
        snprintf(tmp, sizeof(tmp), " · %s", items[i].name);
        lappend(buf, cap, &n, tmp);
    }
    if (n == 0) snprintf(buf, cap, "-");
}

static void json_put_str(VJson *obj, const char *key, const char *val) {
    /* 值可能来自 Windows argv（ANSI/GBK，如中文模型路径），直接进 JSON 会让
     * 整个回包/配置文件不是合法 UTF-8：Python supervisor 的 json.load 与
     * 审查页都会读不了。写 JSON 前统一转成 UTF-8（已是 UTF-8 的原样通过）。 */
    char tmp[4096];
    st_util_utf8(val ? val : "", tmp, sizeof(tmp));
    vjson_obj_set(obj, key, vjson_new_string(tmp));
}

/* Build the current effective configuration as JSON. */
static VJson *build_config_json(const VLLMServerCtx *ctx) {
    VJson *o = vjson_new_object();
    json_put_str(o, "model_dir", ctx->model_dir);
    vjson_obj_set(o, "kv_q4", vjson_new_bool(g_kv_q4));
    vjson_obj_set(o, "prefix_cache", vjson_new_bool(ctx->prefix_cache));
    vjson_obj_set(o, "prefill_q8", vjson_new_bool(g_st_prefill_q8));
    vjson_obj_set(o, "sparse_attn", vjson_new_bool(g_sparse_attn));
    vjson_obj_set(o, "sparse_k", vjson_new_number((double)g_sparse_k));
    vjson_obj_set(o, "l3_evict", vjson_new_bool(g_l3_evict));
    vjson_obj_set(o, "l3_ratio", vjson_new_number((double)g_l3_ratio));
    vjson_obj_set(o, "l3_min_seq", vjson_new_number((double)g_l3_min_seq));
    json_put_str(o, "l3_path", (g_l3_path && g_l3_path[0]) ? g_l3_path : "kv_l3.bin");
    vjson_obj_set(o, "l3_max_size", vjson_new_number((double)g_l3_max_size));
    vjson_obj_set(o, "l3_user_ttl", vjson_new_number((double)g_l3_user_ttl));
    vjson_obj_set(o, "npu", vjson_new_bool(ctx->npu_enabled));
    vjson_obj_set(o, "npu_load",
        vjson_new_number((double)atoi(getenv_int_str("VLLM_NPU_LOAD", "0"))));
    vjson_obj_set(o, "npu_infer",
        vjson_new_number((double)atoi(getenv_int_str("VLLM_NPU_INFER", "1"))));
    vjson_obj_set(o, "npu_timing",
        vjson_new_number((double)atoi(getenv_int_str("VLLM_NPU_TIMING", "0"))));
    vjson_obj_set(o, "port", vjson_new_number((double)ctx->port));
    vjson_obj_set(o, "threads", vjson_new_number((double)ctx->threads));
    vjson_obj_set(o, "max_queued", vjson_new_number((double)ctx->max_queued));
    /* 连续批处理（--batch-max N，0/1 = 关）：多用户并发档，管理页组合⑥ 用 */
    vjson_obj_set(o, "batch_max", vjson_new_number((double)ctx->batch_max));
    vjson_obj_set(o, "min_free_mb", vjson_new_number((double)ctx->min_free_mb));
    /* Conversation features (KV prefix reuse / disk persistence / spec decode /
     * min-p), see vllm_server.h. disk_kv_dir is the checkpoint directory. */
    vjson_obj_set(o, "prefix_kv", vjson_new_bool(ctx->prefix_kv));
    vjson_obj_set(o, "disk_kv", vjson_new_bool(ctx->disk_kv));
    json_put_str(o, "disk_kv_dir", ctx->kvdir);
    /* 工业边缘：预置命名上下文（规则包）/ 会话存储 / 管理面审计 的目录与上限。
     * 对应 CLI --rulebook-dir / --session-dir / --session-limit / --audit-log
     * （由 tools/ops/vllm_mgr.py 组装）；留空 = 关闭该功能。 */
    json_put_str(o, "rulebook_dir", ctx->rulebook_dir);
    json_put_str(o, "session_dir", ctx->session_dir);
    vjson_obj_set(o, "session_limit", vjson_new_number((double)ctx->session_limit));
    json_put_str(o, "audit_log", ctx->audit_log);
    vjson_obj_set(o, "spec", vjson_new_bool(ctx->spec));
    vjson_obj_set(o, "spec_k", vjson_new_number((double)ctx->spec_k));
    vjson_obj_set(o, "min_p", vjson_new_number(ctx->min_p));
    vjson_obj_set(o, "top_k", vjson_new_number((double)ctx->top_k));
    /* Load-on-use / unload-when-idle lifecycle config */
    vjson_obj_set(o, "auto_load", vjson_new_bool(ctx->auto_load));
    vjson_obj_set(o, "auto_unload_s", vjson_new_number((double)ctx->auto_unload_s));
    /* 模型加载格式：v1.0 起引擎为纯 VQF 运行时，无格式可选（加载时自动定位
     * 模型目录内的 model.vqf），故不再持久化 format 字段。 */
    json_put_str(o, "device", ctx->device_id);
    VJson *env = vjson_new_object();
    json_put_str(env, "OMP_NUM_THREADS",
                 getenv_int_str("OMP_NUM_THREADS", "4"));
    /* NEON kernel / threadpool optimization switches (admin.html 内核优化
     * card) + MoE 优化 card (VLLM_MOE_BATCH / VLLM_ACTQ / VLLM_ACTQ16 /
     * VLLM_MOE_PAR / VLLM_MOE_Q4SIMD). Only echo vars that are explicitly
     * set; fillForm treats absent keys as the engine default.
     * VLLM_ACTQ16：管理页「精度/速度轨」三选一写出（0=精确轨 / 1=s16 近似轨），
     * 与 VLLM_ACTQ 互斥且优先；不回填的话页面会误判当前轨。 */
    static const char *opt_envs[] = {
        "VLLM_Q8_8X8", "VLLM_DISABLE_Q8_REPACK", "VLLM_DISABLE_Q4_REPACK",
        "VLLM_ENABLE_8X8L", "VLLM_PB_HEAP", "VLLM_TP_BIND", "VLLM_TP_SPIN",
        "VLLM_ROW_SLICE", "VLLM_VQF_KEY",
        "VLLM_MOE_BATCH", "VLLM_ACTQ", "VLLM_ACTQ16",
        "VLLM_MOE_PAR", "VLLM_MOE_Q4SIMD",
        /* P3 前缀复用门：--l3-evict 开启时 prefix-kv 会被静默打掉，必须靠这个
         * env 才能让 L3 驱逐与前缀复用共存。管理页有对应勾选框，故需回填。 */
        "VLLM_L3_PREFIX_REUSE",
        /* 可验证推理开关（管理页有勾选框，需回填）：VLLM_ATTEST=1 时引擎对
         * 每条响应附 SM2 出证凭证；VLLM_ATTEST_DIR 为其密钥/工作目录。 */
        "VLLM_ATTEST", "VLLM_ATTEST_DIR",
        NULL
    };
    for (int i = 0; opt_envs[i]; i++) {
        const char *e = getenv(opt_envs[i]);
        if (e && e[0])
            json_put_str(env, opt_envs[i], e);
    }
    vjson_obj_set(o, "env", env);
    return o;
}

/* Ensure every path component of `dir` exists (mkdir -p; mkdir only
 * creates one level). */
static void config_mkdirs(char *dir) {
    if (!dir || !dir[0]) return;
    for (char *p = dir + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            char save = *p;
            *p = '\0';
            st_mkdir(dir, 0755);
            *p = save;
        }
    }
    st_mkdir(dir, 0755);
}

/* Persist a config JSON object to the config file (mkdir -p the dir). */
static int config_write_file(const char *path, const VJson *obj, char *err, size_t errcap) {
    char buf[8192];
    size_t n = vjson_serialize(obj, buf, sizeof(buf));
    if (n == 0 || n >= sizeof(buf)) {
        snprintf(err, errcap, "config too large to serialize");
        return -1;
    }
    /* ensure the directory exists */
    char dir[1024];
    snprintf(dir, sizeof(dir), "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash) {
        *slash = '\0';
        if (dir[0]) config_mkdirs(dir);
    }
    FILE *f = fopen(path, "w");
    if (!f) {
        snprintf(err, errcap, "cannot open %s for writing", path);
        return -1;
    }
    fputs(buf, f);
    fclose(f);
    return 0;
}

/* ---------- log tail ---------- */

static size_t read_file_bounded(const char *path, char *buf, size_t cap) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    size_t n = fread(buf, 1, cap - 1, f);
    fclose(f);
    buf[n] = '\0';
    return n;
}

static void tail_lines(const char *data, size_t n, int max_lines,
                       char **lines, int *nlines) {
    /* start = position after the max_lines-th newline counted from the end
     * (so the final max_lines whole lines are kept); 0 when fewer lines. */
    size_t start = 0;
    int nl_count = 0;
    for (size_t k = n; k > 0; k--) {
        if (data[k - 1] == '\n') {
            nl_count++;
            if (nl_count > max_lines) {
                start = k;
                break;
            }
        }
    }
    int nout = 0;
    const char *p = data + start;
    const char *end = data + n;
    while (p < end && nout < max_lines) {
        const char *nl = memchr(p, '\n', (size_t)(end - p));
        size_t len = nl ? (size_t)(nl - p) : (size_t)(end - p);
        while (len > 0 && p[len - 1] == '\r') len--;
        char *line = (char *)malloc(len + 1);
        if (line) {
            memcpy(line, p, len);
            line[len] = '\0';
            lines[nout++] = line;
        }
        p = nl ? nl + 1 : end;
    }
    *nlines = nout;
}

/* ---------- handlers ---------- */

/* 按需容量的 JSON 回包（会话详情/列表可能远超 16KB 栈缓冲）。 */
static void admin_json_reply_cap(VHttpResponse *resp, VJson *root, size_t cap) {
    if (!root) {
        resp->status = 500;
        resp->content_type = "application/json";
        resp->body = "{\"error\":\"out of memory\"}";
        resp->body_len = strlen(resp->body);
        return;
    }
    if (cap < 4096) cap = 4096;
    char *buf = (char *)malloc(cap);
    if (!buf) {
        vjson_free(root);
        resp->status = 500;
        resp->content_type = "application/json";
        resp->body = "{\"error\":\"out of memory\"}";
        resp->body_len = strlen(resp->body);
        return;
    }
    size_t n = vjson_serialize(root, buf, cap);
    vjson_free(root);
    if (n == 0) {
        free(buf);
        resp->status = 500;
        resp->content_type = "application/json";
        resp->body = "{\"error\":\"response too large (raise the cap)\"}";
        resp->body_len = strlen(resp->body);
        return;
    }
    /* 保留调用方已显式设置的错误状态：handler 通常先置 400/403/404/500
     * 再调本函数回包，若在此无条件写 200 会把错误码全部吞掉（客户端只能
     * 靠 body 里的 "error" 字段判断）。仅在仍是成功态时归一为 200。 */
    if (resp->status < 400) resp->status = 200;
    resp->content_type = "application/json";
    resp->body_owned = buf;
    resp->body = buf;
    resp->body_len = n;
}

static void admin_json_reply(VHttpResponse *resp, VJson *root) {
    admin_json_reply_cap(resp, root, 16384);
}

static void handle_admin_status(const VLLMServerCtx *ctx, VHttpResponse *resp) {
    VJson *o = vjson_new_object();
    vjson_obj_set(o, "engine", vjson_new_string("vllm_kestrel"));
    json_put_str(o, "model_id", ctx->model_id);
    json_put_str(o, "model_name", ctx->model_name[0] ? ctx->model_name : NULL);
    json_put_str(o, "model_dir", ctx->model_dir);
    json_put_str(o, "wmode", wmode_str(g_st_wmode));
    /* VQF 文件内固化布局（只读）：量化在转换期由 vqf_convert --wmode 固化进
     * 文件，加载侧不可更改。未加载 / 加载失败时不下发该字段，管理页显示 "-"。 */
    {
        uint32_t vqf_flags = 0, vqf_ver = 0;
        if (vqf_layout_state(&vqf_flags, &vqf_ver)) {
            char lbuf[160];
            vqf_layout_str(vqf_flags, vqf_ver, lbuf, sizeof(lbuf));
            vjson_obj_set(o, "vqf_flags", vjson_new_number((double)vqf_flags));
            vjson_obj_set(o, "vqf_version", vjson_new_number((double)vqf_ver));
            json_put_str(o, "layout", lbuf);
        }
    }
    /* Device profile */
    json_put_str(o, "device_id", ctx->device_id);
    json_put_str(o, "device_name", ctx->device_name);
    {   /* 设备说明的英文名：管理页按当前语言二选一（切换语言无需重启引擎） */
        const VDevProfile *dp = ctx->device_id ? vdev_lookup(ctx->device_id) : NULL;
        if (dp && dp->name_en) json_put_str(o, "device_name_en", dp->name_en);
    }
    json_put_str(o, "device_arch", ctx->device_arch);
    json_put_str(o, "device_class", ctx->device_class);
    json_put_str(o, "device_npu", ctx->device_npu);
    vjson_obj_set(o, "npu", vjson_new_bool(ctx->npu_enabled));
    vjson_obj_set(o, "port", vjson_new_number((double)ctx->port));
    vjson_obj_set(o, "threads", vjson_new_number((double)ctx->threads));
    vjson_obj_set(o, "uptime_s",
        vjson_new_number(ctx->started_s ? (double)(time(NULL) - ctx->started_s) : 0));
    vjson_obj_set(o, "busy", vjson_new_bool(ctx->busy));
    vjson_obj_set(o, "active", vjson_new_number((double)ctx->active_requests));
    vjson_obj_set(o, "queued", vjson_new_number((double)ctx->queued_requests));
    vjson_obj_set(o, "max_queued", vjson_new_number((double)ctx->max_queued));
    vjson_obj_set(o, "total_requests", vjson_new_number((double)ctx->total_requests));
    /* Continuous batching (--batch-max N) */
    if (ctx->batch_max >= 2) {
        vjson_obj_set(o, "batch_max", vjson_new_number((double)ctx->batch_max));
        vjson_obj_set(o, "batch_active", vjson_new_number((double)vllm_batch_active(ctx->batch)));
    }
    /* Model-load lifecycle + live progress */
    vjson_obj_set(o, "load_state", vjson_new_number((double)ctx->load_state));
    vjson_obj_set(o, "load_layer", vjson_new_number((double)g_model_load_layer));
    vjson_obj_set(o, "load_total", vjson_new_number((double)g_model_load_total));
    if (ctx->load_error[0]) json_put_str(o, "load_error", ctx->load_error);
    /* Load-on-use / unload-when-idle lifecycle (启动时间 + 卸载统计) */
    vjson_obj_set(o, "last_load_ms", vjson_new_number((double)ctx->last_load_ms));
    vjson_obj_set(o, "unload_count", vjson_new_number((double)ctx->unload_count));
    vjson_obj_set(o, "auto_unload_count", vjson_new_number((double)ctx->auto_unload_count));
    vjson_obj_set(o, "auto_load", vjson_new_bool(ctx->auto_load));
    vjson_obj_set(o, "auto_unload_s", vjson_new_number((double)ctx->auto_unload_s));
    vjson_obj_set(o, "unload_pending", vjson_new_bool(ctx->unload_pending));
    /* KV 前缀复用运行态（--no-prefix-kv 关闭；P4 多模态/文本共用） */
    vjson_obj_set(o, "prefix_kv", vjson_new_bool(ctx->prefix_kv));
    /* VQF 分层驻留（AirLLM 型 layer streaming，VLLM_VQF_STREAM=N）状态 */
    {
        int k = 0, nl = 0, ns = 0, cd = 0;
        unsigned long long pl = 0;
        if (vqf_stream_state(&k, &nl, &ns, &pl, &cd)) {
            VJson *vs = vjson_new_object();
            vjson_obj_set(vs, "on", vjson_new_bool(1));
            vjson_obj_set(vs, "keep", vjson_new_number((double)k));
            vjson_obj_set(vs, "n_layers", vjson_new_number((double)nl));
            vjson_obj_set(vs, "n_segments", vjson_new_number((double)ns));
            vjson_obj_set(vs, "per_layer_mb",
                          vjson_new_number(pl ? (double)pl / 1048576.0 : 0.0));
            vjson_obj_set(vs, "cold", vjson_new_bool(cd != 0));
            vjson_obj_set(o, "vqf_stream", vs);
        }
    }
    /* 内存驻留策略（档位阶梯）：当前档 + 可调项 + 能力/观测。
     * idle_s = [L4→L3, L3→L2, L2→L1, L1→L0] 各档空闲停留秒数。 */
    {
        int cur = vllm_res_level(ctx);
        VJson *rp = vjson_new_object();
        vjson_obj_set(rp, "cur_level", vjson_new_number((double)cur));
        vjson_obj_set(rp, "cur_name",
                      vjson_new_string(vllm_res_level_name(cur)));
        vjson_obj_set(rp, "cfg_level", vjson_new_number((double)ctx->res.level));
        vjson_obj_set(rp, "auto_idle", vjson_new_bool(ctx->res.auto_idle != 0));
        VJson *arr = vjson_new_array();
        for (int k = 4; k >= 1; k--)
            vjson_array_push(arr, vjson_new_number((double)ctx->res.idle_s[k]));
        vjson_obj_set(rp, "idle_s", arr);
        vjson_obj_set(rp, "w_keep", vjson_new_number((double)ctx->res.w_keep));
        vjson_obj_set(rp, "mem_soft_mb",
                      vjson_new_number((double)ctx->res.mem_soft_mb));
        vjson_obj_set(rp, "mem_low_lvl",
                      vjson_new_number((double)ctx->res.mem_low_lvl));
        vjson_obj_set(rp, "w_capable", vjson_new_bool(vqf_stream_active() != 0));
        vjson_obj_set(rp, "l3_evict", vjson_new_bool(g_l3_evict != 0));
        vjson_obj_set(rp, "steps", vjson_new_number((double)ctx->res.steps));
        vjson_obj_set(o, "res", rp);
    }
    /* Current user connections / queue */
    vjson_obj_set(o, "connections", vjson_new_number((double)vhttp_active_conns()));
    long rss = proc_rss_mb();
    vjson_obj_set(o, "rss_mb", vjson_new_number(rss >= 0 ? (double)rss : -1));
    long ma = proc_mem_mb("MemAvailable:");
    long mt = proc_mem_mb("MemTotal:");
    vjson_obj_set(o, "mem_avail_mb", vjson_new_number(ma >= 0 ? (double)ma : -1));
    vjson_obj_set(o, "mem_total_mb", vjson_new_number(mt >= 0 ? (double)mt : -1));
    double cpu = proc_cpu_pct();
    if (cpu >= 0)
        vjson_obj_set(o, "cpu_pct", vjson_new_number(cpu));
    else
        vjson_obj_set(o, "cpu_pct", vjson_new_null());
    json_put_str(o, "config_path", admin_config_path(ctx));
    json_put_str(o, "log_path", admin_log_path(ctx));
    /* 工业边缘：预置命名上下文 / 会话存储 / 审计（审查台 /review 的"审查范围"用） */
    json_put_str(o, "rulebook_dir", ctx->rulebook_dir);
    json_put_str(o, "session_dir", ctx->session_dir);
    json_put_str(o, "audit_log", ctx->audit_log);
    vjson_obj_set(o, "rulebook_hits", vjson_new_number((double)ctx->rulebook_hits));
    vjson_obj_set(o, "session_turns", vjson_new_number((double)ctx->session_turns));
    admin_json_reply(resp, o);
}

/* Device list + current profile + weight-adaptation recommendations, so the
 * admin page can show the device card, a device dropdown and a "apply device
 * recommendations" action (whitelisted wmodes + prefilled parameters). */
static void add_device_profile_json(VJson *o, const VDevProfile *p,
                                    const char *arch_str, const char *cls_str) {
    VJson *d = vjson_new_object();
    json_put_str(d, "id", p->id);
    json_put_str(d, "name", p->name);
    json_put_str(d, "name_en", p->name_en);
    json_put_str(d, "arch", arch_str);
    json_put_str(d, "class", cls_str);
    json_put_str(d, "npu", p->npu ? p->npu : "-");
    vjson_obj_set(o, "device", d);

    VJson *wm = vjson_new_array();
    if (p->wmodes) {
        char copy[128];
        snprintf(copy, sizeof(copy), "%s", p->wmodes);
        char *tok = strtok(copy, ",");
        while (tok) {
            vjson_array_push(wm, vjson_new_string(tok));
            tok = strtok(NULL, ",");
        }
    }
    vjson_obj_set(o, "device_wmodes", wm);

    VJson *rec = vjson_new_object();
    json_put_str(rec, "wmode", p->default_wmode);
    vjson_obj_set(rec, "omp", vjson_new_number((double)p->default_omp));
    vjson_obj_set(rec, "threads", vjson_new_number((double)p->default_threads));
    vjson_obj_set(rec, "mem_limit_mb", vjson_new_number((double)p->mem_limit_mb));
    vjson_obj_set(rec, "npu", vjson_new_bool(p->npu != NULL));
    vjson_obj_set(rec, "npu_load", vjson_new_number((double)p->default_npu_load));
    vjson_obj_set(rec, "prefill_q8", vjson_new_bool(p->default_prefill_q8 != 0));
    json_put_str(rec, "model_dir", p->default_model_dir);
    vjson_obj_set(o, "device_recommend", rec);
}

static void add_device_list(VJson *o) {
    VJson *list = vjson_new_array();
    for (const VDevProfile *q = vdev_all(); q->id; q++) {
        VJson *e = vjson_new_object();
        json_put_str(e, "id", q->id);
        json_put_str(e, "name", q->name);
        json_put_str(e, "name_en", q->name_en);
        json_put_str(e, "arch", vdev_arch_str(q->arch));
        json_put_str(e, "class", vdev_class_str(q->cls));
        json_put_str(e, "npu", q->npu ? q->npu : "-");
        vjson_array_push(list, e);
    }
    vjson_obj_set(o, "devices", list);
}

static void add_device_info(VJson *o, const VLLMServerCtx *ctx) {
    const VDevProfile *p = ctx->device_id ? vdev_lookup(ctx->device_id) : NULL;
    add_device_list(o);
    if (p)
        add_device_profile_json(o, p, ctx->device_arch, ctx->device_class);
}

/* GET /admin/api/device[?id=xxx] - profile of any known device (for the
 * dropdown preview without selecting it as the running device). */
static void handle_admin_device(const VHttpRequest *req, VHttpResponse *resp) {
    const VDevProfile *p = NULL;
    if (req->query && req->query[0]) {
        const char *q = strstr(req->query, "id=");
        if (q) {
            char id[64];
            snprintf(id, sizeof(id), "%s", q + 3);
            char *amp = strchr(id, '&');
            if (amp) *amp = '\0';
            p = vdev_lookup(id);
        }
    }
    VJson *o = vjson_new_object();
    add_device_list(o);
    if (p)
        add_device_profile_json(o, p, vdev_arch_str(p->arch), vdev_class_str(p->cls));
    else
        json_put_str(o, "error", "device not found");
    admin_json_reply(resp, o);
}

static void handle_admin_config(const VLLMServerCtx *ctx, VHttpResponse *resp) {
    VJson *o = vjson_new_object();
    vjson_obj_set(o, "config", build_config_json(ctx));
    add_device_info(o, ctx);
    json_put_str(o, "config_path", admin_config_path(ctx));
    admin_json_reply(resp, o);
}

static void handle_admin_config_save(const VLLMServerCtx *ctx,
                                     const VHttpRequest *req, VHttpResponse *resp) {
    VJson *root = NULL;
    const char *err = NULL;
    char errbuf[512];
    if (!req->body || req->body_len == 0) {
        err = "empty JSON body";
    } else {
        char *copy = (char *)malloc(req->body_len + 1);
        if (!copy) {
            err = "out of memory";
        } else {
            memcpy(copy, req->body, req->body_len);
            copy[req->body_len] = '\0';
            root = vjson_parse(copy);
            free(copy);
            if (!root || root->type != VJ_OBJECT) {
                err = "invalid JSON object";
                vjson_free(root);
                root = NULL;
            }
        }
    }
    if (err) {
        resp->status = 400;
        resp->content_type = "application/json";
        snprintf(errbuf, sizeof(errbuf), "{\"error\":\"%s\"}", err);
        resp->body_owned = (char *)malloc(strlen(errbuf) + 1);
        if (resp->body_owned) {
            strcpy(resp->body_owned, errbuf);
            resp->body = resp->body_owned;
        }
        return;
    }
    const char *path = admin_config_path(ctx);
    char werr[256];
    int rc = config_write_file(path, root, werr, sizeof(werr));
    vjson_free(root);
    if (rc != 0) {
        resp->status = 500;
        resp->content_type = "application/json";
        char out[512];
        snprintf(out, sizeof(out), "{\"error\":\"%s\"}", werr);
        resp->body_owned = (char *)malloc(strlen(out) + 1);
        if (resp->body_owned) {
            strcpy(resp->body_owned, out);
            resp->body = resp->body_owned;
        }
        return;
    }
    VJson *o = vjson_new_object();
    vjson_obj_set(o, "ok", vjson_new_bool(1));
    json_put_str(o, "path", path);
    json_put_str(o, "note",
        "saved. Restart the engine (via vllm_mgr / supervisor) to apply.");
    admin_json_reply(resp, o);
}

static void handle_admin_log(const VLLMServerCtx *ctx,
                             const VHttpRequest *req, VHttpResponse *resp) {
    int n = 200;
    if (req->query && req->query[0]) {
        const char *p = strstr(req->query, "n=");
        if (p) n = atoi(p + 2);
        if (n < 1) n = 1;
        if (n > 2000) n = 2000;
    }
    const char *path = admin_log_path(ctx);
    VJson *o = vjson_new_object();
    char *buf = (char *)malloc(4 * 1024 * 1024 + 1);
    if (!buf) {
        json_put_str(o, "error", "out of memory");
        admin_json_reply(resp, o);
        return;
    }
    size_t sz = read_file_bounded(path, buf, 4 * 1024 * 1024 + 1);
    if (sz == 0) {
        char msg[512];
        snprintf(msg, sizeof(msg), "log file not readable: %s", path);
        json_put_str(o, "error", msg);
        free(buf);
        admin_json_reply(resp, o);
        return;
    }
    char **lines = (char **)malloc((size_t)n * sizeof(char *));
    int nlines = 0;
    if (lines) {
        tail_lines(buf, sz, n, lines, &nlines);
        VJson *arr = vjson_new_array();
        for (int i = 0; i < nlines; i++) {
            vjson_array_push(arr, vjson_new_string(lines[i]));
            free(lines[i]);
        }
        free(lines);
        vjson_obj_set(o, "lines", arr);
    }
    json_put_str(o, "path", path);
    vjson_obj_set(o, "bytes", vjson_new_number((double)sz));
    free(buf);
    admin_json_reply(resp, o);
}

static void handle_admin_shutdown(const VLLMServerCtx *ctx, VHttpResponse *resp) {
    VJson *o = vjson_new_object();
    vjson_obj_set(o, "ok", vjson_new_bool(1));
    json_put_str(o, "note", "engine is shutting down");
    admin_json_reply(resp, o);
    vhttp_stop();
}

/* ---------- per-user L3 cache clear (/admin/api/l3/clear) ---------- */

/* True when name matches the per-user L3 pattern "<stem>_u<8 hex>.<ext>"
 * (or plain "<stem>_u<8 hex>" when the base file has no extension). */
static int l3_is_user_fname(const char *name, const char *stem, const char *ext) {
    size_t sl = strlen(stem);
    if (strncmp(name, stem, sl) != 0) return 0;
    if (strncmp(name + sl, "_u", 2) != 0) return 0;
    const char *h = name + sl + 2;
    for (int i = 0; i < 8; i++)
        if (!isxdigit((unsigned char)h[i])) return 0;
    const char *e = h + 8;
    if (ext[0]) {
        if (e[0] != '.' || strcmp(e + 1, ext) != 0) return 0;
    } else if (e[0] != '\0') {
        return 0;
    }
    return 1;
}

/* Delete every per-user L3 file on disk (base path -> "<stem>_u<hash><ext>"
 * in the same directory), including files left over from before a restart.
 * Returns the number of files removed. */
static int l3_user_glob_remove(void) {
    const char *base = (g_l3_path && g_l3_path[0]) ? g_l3_path : "kv_l3.bin";
    char buf[512];
    snprintf(buf, sizeof(buf), "%s", base);
    char *slash = strrchr(buf, '/');
    char *nm = slash ? slash + 1 : buf;
    char *dot = strrchr(nm, '.');
    char ext[64] = "";
    if (dot) {
        snprintf(ext, sizeof(ext), "%s", dot + 1);
        *dot = '\0';
    }
    char stem[256];
    snprintf(stem, sizeof(stem), "%s", nm);
    char dir[512] = "";
    if (slash) {
        *slash = '\0';
        snprintf(dir, sizeof(dir), "%s", buf);
    }
    int removed = 0;
    {
        DIR *d = opendir(dir[0] ? dir : ".");
        if (d) {
            struct dirent *de;
            while ((de = readdir(d)) != NULL) {
                if (l3_is_user_fname(de->d_name, stem, ext)) {
                    char full[600];
                    snprintf(full, sizeof(full), "%s/%s",
                             dir[0] ? dir : ".", de->d_name);
                    if (unlink(full) == 0) removed++;
                }
            }
            closedir(d);
        }
    }
    return removed;
}

/* Clear every user's L3 cache: close the bound file, delete the per-user
 * files on disk, drop the session table. Called under the inference lock so
 * it cannot race the request path (ist_reset / l3_user_apply). The next
 * request lazily re-creates the current user's file. */
static void handle_admin_l3_clear(VLLMServerCtx *ctx, VHttpResponse *resp) {
    VJson *o = vjson_new_object();
    int removed = 0;
    vhttp_mutex_lock(ctx->inf_lock);
    STQwenInferenceState *st = ctx->ist;
    if (st && st->l3.fp) {
        fprintf(stderr, "[L3] clear: closed bound file %s\n", st->l3.path);
        l3_state_free(&st->l3);
    }
    removed += l3_user_glob_remove();
    ctx->l3_users_n = 0;
    ctx->l3_cur_fname[0] = '\0';
    g_l3_cur_path = NULL;
    vhttp_mutex_unlock(ctx->inf_lock);
    fprintf(stderr, "[L3] clear: removed %d per-user cache file(s)\n", removed);
    vjson_obj_set(o, "ok", vjson_new_bool(1));
    vjson_obj_set(o, "removed", vjson_new_number((double)removed));
    admin_json_reply(resp, o);
}

static void handle_admin_model_load(VLLMServerCtx *ctx, const VHttpRequest *req,
                                    VHttpResponse *resp) {
    VJson *o = vjson_new_object();
    /* Optional body: {"model_dir": "<path>"}. model_dir points the load at a
     * model directory that may not have been resolvable at startup (portable
     * manual-load mode — the packaged exe can run from any directory and the
     * admin page supplies the path). The load always targets the VQF single
     * file in that directory (model.vqf / weights.vqf / vllm.vqf). */
    if (req->body && req->body_len > 0) {
        char *copy = (char *)malloc(req->body_len + 1);
        if (copy) {
            memcpy(copy, req->body, req->body_len);
            copy[req->body_len] = '\0';
            VJson *root = vjson_parse(copy);
            if (root && root->type == VJ_OBJECT) {
                const VJson *md = vjson_obj_get(root, "model_dir");
                if (md && md->type == VJ_STRING) {
                    const char *s = vjson_str(md);
                    if (s && s[0]) vllm_serve_set_model_dir(ctx, s);
                }
            }
            vjson_free(root);
            free(copy);
        }
    }
    int rc = vllm_serve_start_load(ctx);
    if (rc == 0) {
        vjson_obj_set(o, "ok", vjson_new_bool(1));
        json_put_str(o, "note", "model load started (see /admin/api/status progress)");
    } else if (rc == 1) {
        vjson_obj_set(o, "ok", vjson_new_bool(1));
        json_put_str(o, "note", "model is already loading");
    } else if (rc == 2) {
        vjson_obj_set(o, "ok", vjson_new_bool(1));
        json_put_str(o, "note", "model is already loaded");
    } else {
        vjson_obj_set(o, "ok", vjson_new_bool(0));
        json_put_str(o, "error", "failed to start model load");
    }
    admin_json_reply(resp, o);
}

/* POST /admin/api/model/unload —— 卸载当前模型（释放权重+KV 内存）。
 * 仅空闲时允许；加载中或推理请求运行/排队时拒绝。 */
static void handle_admin_model_unload(VLLMServerCtx *ctx,
                                      const VHttpRequest *req,
                                      VHttpResponse *resp) {
    (void)req;
    VJson *o = vjson_new_object();
    int rc = vllm_serve_unload_model_self(ctx);   /* self 豁免：handler 内卸载 */
    if (rc == 0) {
        vjson_obj_set(o, "ok", vjson_new_bool(1));
        json_put_str(o, "note", vllm_tr(
                    "model unloaded (weights/KV freed; next request or 点「加载模型」 will reload)",
                    "model unloaded (weights/KV freed; the next request or Load Model will reload)"));
    } else if (rc == -2) {
        vjson_obj_set(o, "ok", vjson_new_bool(0));
        json_put_str(o, "error", "model is still loading");
    } else if (rc == -3) {
        vjson_obj_set(o, "ok", vjson_new_bool(0));
        json_put_str(o, "error", "inference in progress, retry when idle");
    } else {
        vjson_obj_set(o, "ok", vjson_new_bool(0));
        json_put_str(o, "error", "no model loaded");
    }
    admin_json_reply(resp, o);
}

/* POST /admin/api/policy/config —— 运行期调整内存驻留策略参数（即刻生效，
 * 无需重启）。body（缺省字段保持不变）：
 *   {"auto_idle":bool, "idle_s":[L4→L3,L3→L2,L2→L1,L1→L0],
 *    "w_keep":N, "mem_soft_mb":N, "mem_low_lvl":N} */
static void handle_admin_policy_config(VLLMServerCtx *ctx,
                                       const VHttpRequest *req,
                                       VHttpResponse *resp) {
    VJson *o = vjson_new_object();
    VLLMResPolicy *rp = &ctx->res;
    int changed = 0;
    if (req->body && req->body_len > 0) {
        char *copy = (char *)malloc(req->body_len + 1);
        if (copy) {
            memcpy(copy, req->body, req->body_len);
            copy[req->body_len] = '\0';
            VJson *root = vjson_parse(copy);
            if (root && root->type == VJ_OBJECT) {
                const VJson *v;
                vhttp_mutex_lock(ctx->stat_lock);   /* tick 同字段读，锁内整块改 */
                if ((v = vjson_obj_get(root, "auto_idle")) &&
                    (v->type == VJ_BOOL || v->type == VJ_NUMBER)) {
                    int b = (v->type == VJ_BOOL) ? vjson_bool(v) : (int)vjson_num(v);
                    if ((rp->auto_idle ? 1 : 0) != (b ? 1 : 0)) { rp->auto_idle = b ? 1 : 0; changed = 1; }
                }
                if ((v = vjson_obj_get(root, "idle_s")) && v->type == VJ_ARRAY) {
                    size_t n = vjson_array_len(v);
                    for (size_t i = 0; i < n && i < 4; i++) {
                        VJson *e = vjson_array_get(v, i);
                        long t = (long)vjson_num(e);
                        if (t < 0) t = 0;
                        if (rp->idle_s[4 - (int)i] != t) { rp->idle_s[4 - (int)i] = t; changed = 1; }
                    }
                }
                if ((v = vjson_obj_get(root, "w_keep")) && v->type == VJ_NUMBER) {
                    int t = (int)vjson_num(v);
                    if (t < 0) t = 0;
                    if (rp->w_keep != t) { rp->w_keep = t; changed = 1; }
                }
                if ((v = vjson_obj_get(root, "mem_soft_mb")) && v->type == VJ_NUMBER) {
                    int t = (int)vjson_num(v);
                    if (t < 0) t = 0;
                    if (rp->mem_soft_mb != t) { rp->mem_soft_mb = t; changed = 1; }
                }
                if ((v = vjson_obj_get(root, "mem_low_lvl")) && v->type == VJ_NUMBER) {
                    int t = (int)vjson_num(v);
                    if (t < 0) t = 0; if (t > 4) t = 4;
                    if (rp->mem_low_lvl != t) { rp->mem_low_lvl = t; changed = 1; }
                }
                vhttp_mutex_unlock(ctx->stat_lock);
            }
            vjson_free(root);
            free(copy);
        }
    }
    vjson_obj_set(o, "ok", vjson_new_bool(1));
    if (changed) {
        fprintf(stderr, "[RES] admin policy config updated: auto_idle=%d "
                        "idle_s=%ld/%ld/%ld/%ld w_keep=%d soft=%dMB low_lvl=%d\n",
                rp->auto_idle, rp->idle_s[4], rp->idle_s[3],
                rp->idle_s[2], rp->idle_s[1], rp->w_keep,
                rp->mem_soft_mb, rp->mem_low_lvl);
        fflush(stderr);
        json_put_str(o, "note", "policy config applied (runtime)");
    }
    admin_json_reply(resp, o);
}

/* POST /admin/api/policy/set —— 立即切换驻留档位。
 * body: {"level":0..4}（L4 全驻留 → L0 卸载）；返回新档位与动作结果。 */
static void handle_admin_policy_set(VLLMServerCtx *ctx,
                                    const VHttpRequest *req,
                                    VHttpResponse *resp) {
    VJson *o = vjson_new_object();
    int level = -1;
    if (req->body && req->body_len > 0) {
        char *copy = (char *)malloc(req->body_len + 1);
        if (copy) {
            memcpy(copy, req->body, req->body_len);
            copy[req->body_len] = '\0';
            VJson *root = vjson_parse(copy);
            if (root && root->type == VJ_OBJECT) {
                const VJson *v = vjson_obj_get(root, "level");
                if (v && v->type == VJ_NUMBER) level = (int)vjson_num(v);
            }
            vjson_free(root);
            free(copy);
        }
    }
    if (level < 0 || level > 4) {
        vjson_obj_set(o, "ok", vjson_new_bool(0));
        json_put_str(o, "error", "body {\"level\":0..4} required");
        admin_json_reply(resp, o);
        return;
    }
    int prev = vllm_res_level(ctx);
    int rc = vllm_res_set_level(ctx, level);
    if (rc >= 0) {
        vjson_obj_set(o, "ok", vjson_new_bool(1));
        vjson_obj_set(o, "level", vjson_new_number((double)rc));
        json_put_str(o, "level_name", vllm_res_level_name(rc));
        json_put_str(o, "note", "policy level switched");
    } else if (rc == -3) {
        vjson_obj_set(o, "ok", vjson_new_bool(0));
        vjson_obj_set(o, "level", vjson_new_number((double)prev));
        json_put_str(o, "error", "inference in progress (unload refused), "
                    "retry when idle");
    } else {
        vjson_obj_set(o, "ok", vjson_new_bool(0));
        vjson_obj_set(o, "level", vjson_new_number((double)prev));
        json_put_str(o, "error", "level switch failed (need model dir / load "
                    "for upgrade, or model loaded for downgrade)");
    }
    admin_json_reply(resp, o);
}

/* 通用静态页下发：内嵌副本兜底，磁盘副本优先（便于现场改 UI 而不重编）。
 * emb!=NULL 时用内嵌；path 为磁盘候选（不存在则回退内嵌）。 */
static void serve_html_page(VHttpResponse *resp, const char *(*emb)(size_t *),
                            const char *path, const char *name,
                            const char *env_hint) {
    size_t elen = 0;
    const char *ehtml = emb ? emb(&elen) : NULL;
    FILE *f = fopen(path, "rb");
    if (!f) {
        if (ehtml && elen) {
            resp->status = 200;
            resp->content_type = "text/html; charset=utf-8";
            resp->body = ehtml;
            resp->body_len = elen;
            return;
        }
        char *msg = (char *)malloc(768);
        if (msg) {
            snprintf(msg, 768,
                     "<html><body><h3>%s not found</h3>"
                     "<p>Set %s or deploy %s next to the engine "
                     "(e.g. /NewVLLM/%s).</p></body></html>",
                     name, env_hint, name, name);
            resp->body_owned = msg;
            resp->body = msg;
            resp->body_len = strlen(msg);
        } else {
            resp->body = "not found";
            resp->body_len = 9;
        }
        resp->status = 500;
        resp->content_type = "text/html; charset=utf-8";
        return;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0 || sz > 8 * 1024 * 1024) {
        fclose(f);
        resp->status = 500;
        resp->content_type = "text/html; charset=utf-8";
        resp->body = "page too large";
        resp->body_len = strlen(resp->body);
        return;
    }
    char *html = (char *)malloc((size_t)sz + 1);
    if (!html) {
        fclose(f);
        resp->status = 500;
        resp->content_type = "text/html; charset=utf-8";
        resp->body = "out of memory";
        resp->body_len = strlen(resp->body);
        return;
    }
    size_t rd = fread(html, 1, (size_t)sz, f);
    fclose(f);
    html[rd] = '\0';
    resp->status = 200;
    resp->content_type = "text/html; charset=utf-8";
    resp->body_owned = html;
    resp->body = html;
    resp->body_len = rd;
}

static void handle_admin_page(VHttpResponse *resp) {
    serve_html_page(resp, embedded_admin_html, admin_html_path(),
                    "admin.html", "VLLM_ADMIN_HTML");
}

/* 审查台独立页（/review）：与 /admin/ 分离；数据仍走 /admin/api/*（同源）。 */
static void handle_review_page(VHttpResponse *resp) {
    serve_html_page(resp, embedded_review_html, review_html_path(),
                    "review.html", "VLLM_REVIEW_HTML");
}

void vllm_review_page(VHttpResponse *resp) {
    handle_review_page(resp);
}

/* ================================================================
 * 第二步：会话 / 规则包 / 审计 管理面（工业边缘审查）
 *
 * 诚实边界（与 SECURITY.md 一致，UI 上也明示）：
 *   - 本管理面**无身份认证**，前提是"部署在内网/可信网络"。
 *   - actor/role 取自请求头 X-Audit-User / X-Audit-Role，只用于**审计追溯**与
 *     粗粒度写权限门（env VLLM_ADMIN_ROLES 逗号白名单）；**不是**真实鉴权。
 *   - 查看/导出/删除会话均写入 SM3 哈希链审计日志（篡改可发现）。
 *   - at-rest 加密/home 级脱敏未实现：会话明文落盘，脱敏只是审查视图上的
 *     掩码（mask=1）并记入审计。需要落盘加密请把会话目录放在加密卷上。
 * ================================================================ */

static int adm_hexv(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static size_t adm_utf8_enc(unsigned cp, char *out, size_t cap) {
    if (cp < 0x80) { if (cap < 1) return 0; out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        if (cap < 2) return 0;
        out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cap < 3) return 0;
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
}

/* 取 query 参数（支持 %XX 与 '+' 解码）。找到返回 1。 */
static int adm_qs(const VHttpRequest *req, const char *key, char *out, size_t cap) {
    if (out && cap) out[0] = '\0';
    if (!req || !req->query || !req->query[0] || !key || !key[0] || !out || cap < 2)
        return 0;
    size_t kl = strlen(key);
    const char *p = req->query;
    while (*p) {
        const char *amp = strchr(p, '&');
        size_t seg = amp ? (size_t)(amp - p) : strlen(p);
        if (seg > kl && p[kl] == '=' && strncmp(p, key, kl) == 0) {
            const char *v = p + kl + 1;
            size_t vl = seg - kl - 1, w = 0;
            for (size_t i = 0; i < vl && w + 1 < cap; i++) {
                if (v[i] == '+') { out[w++] = ' '; }
                else if (v[i] == '%' && i + 2 < vl) {
                    int hi = adm_hexv((unsigned char)v[i + 1]);
                    int lo = adm_hexv((unsigned char)v[i + 2]);
                    if (hi >= 0 && lo >= 0) { out[w++] = (char)((hi << 4) | lo); i += 2; }
                    else out[w++] = v[i];
                } else {
                    out[w++] = v[i];
                }
            }
            out[w] = '\0';
            return 1;
        }
        if (!amp) break;
        p = amp + 1;
    }
    return 0;
}

/* 管理面身份（见本段顶部"诚实边界"）：复制到调用方缓冲，避免连续取头时被
 * vhttp_req_header 的线程局部缓冲互相覆盖。 */
static void adm_actor(const VHttpRequest *req, char *out, size_t cap) {
    const char *v = vhttp_req_header(req, "X-Audit-User");
    snprintf(out, cap, "%s", (v && v[0]) ? v : "anon");
}
static void adm_role(const VHttpRequest *req, char *out, size_t cap) {
    const char *v = vhttp_req_header(req, "X-Audit-Role");
    snprintf(out, cap, "%s", (v && v[0]) ? v : "");
}

/* 写操作门：env VLLM_ADMIN_ROLES 未配置 = 放行（内网无鉴权前提）；
 * 配置后要求 X-Audit-Role ∈ 逗号白名单。 */
static int adm_write_allowed(const VHttpRequest *req) {
    const char *env = getenv("VLLM_ADMIN_ROLES");
    if (!env || !env[0]) return 1;
    char role[96];
    adm_role(req, role, sizeof(role));
    if (!role[0]) return 0;
    size_t rl = strlen(role);
    const char *p = env;
    while (*p) {
        const char *c = strchr(p, ',');
        size_t l = c ? (size_t)(c - p) : strlen(p);
        if (l == rl && strncmp(p, role, l) == 0) return 1;
        if (!c) break;
        p = c + 1;
    }
    return 0;
}

/* 会话文件名白名单：sess_<16hex>.json（只允许 ASCII/hex，杜绝路径穿越）。 */
static int adm_sess_file_ok(const char *f) {
    if (!f || strncmp(f, "sess_", 5) != 0) return 0;
    const char *h = f + 5;
    for (int i = 0; i < 16; i++) if (adm_hexv((unsigned char)h[i]) < 0) return 0;
    return strcmp(h + 16, ".json") == 0;
}

/* 预览截断（不切断 UTF-8 字符）；mask=1 时整体掩码。 */
static void adm_preview(const char *s, char *out, size_t cap, size_t maxb, int mask) {
    if (!out || cap == 0) return;
    out[0] = '\0';
    if (mask) { snprintf(out, cap, "***"); return; }
    if (!s || !s[0]) return;
    size_t n = strlen(s);
    if (n > maxb) {
        n = maxb;
        while (n > 0 && ((unsigned char)s[n] & 0xC0) == 0x80) n--;
    }
    if (n >= cap) n = cap - 1;
    memcpy(out, s, n);
    out[n] = '\0';
}

/* 在会话 JSON 文本里取字符串字段（我们自己的写入格式是确定的前缀布局）。 */
static int adm_sess_str(const char *txt, const char *key, char *out, size_t cap) {
    if (out && cap) out[0] = '\0';
    if (!txt || !key || !out || cap < 2) return 0;
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(txt, pat);
    if (!p) return 0;
    p = strchr(p + strlen(pat), ':');
    if (!p) return 0;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '"') return 0;
    p++;
    size_t w = 0;
    while (*p && *p != '"' && w + 1 < cap) {
        if (*p == '\\' && p[1]) {
            p++;
            switch (*p) {
            case 'n': out[w++] = '\n'; break;
            case 'r': out[w++] = '\r'; break;
            case 't': out[w++] = '\t'; break;
            case 'b': out[w++] = '\b'; break;
            case 'f': out[w++] = '\f'; break;
            case '"': out[w++] = '"'; break;
            case '\\': out[w++] = '\\'; break;
            case '/': out[w++] = '/'; break;
            case 'u': {
                unsigned cp = 0; int ok = 1;
                for (int i = 0; i < 4; i++) {
                    int hv = adm_hexv((unsigned char)p[1 + i]);
                    if (hv < 0) { ok = 0; break; }
                    cp = (cp << 4) | (unsigned)hv;
                }
                if (ok) { p += 4; w += adm_utf8_enc(cp, out + w, cap - w); }
                else out[w++] = '?';
                break;
            }
            default: out[w++] = *p;
            }
            p++;
        } else {
            out[w++] = *p++;
        }
    }
    out[w] = '\0';
    return 1;
}

static long adm_sess_num(const char *txt, const char *key) {
    if (!txt || !key) return 0;
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(txt, pat);
    if (!p) return 0;
    p = strchr(p + strlen(pat), ':');
    if (!p) return 0;
    return atol(p + 1);
}

#define ADM_SESS_HEAD 65536   /* 会话文件仅需前 64KB 即可取全元数据（messages 在尾部） */

typedef struct {
    char file[64];
    char sid[VLLM_SESS_ID_MAX];
    char tenant[VLLM_SESS_PART_MAX];
    char user[VLLM_SESS_PART_MAX];
    char rb[VLLM_RB_ID_MAX];
    long created, updated, size;
    int  n_msgs, n_tokens;
    int  have_preview;
    char first_user[200];
    char last_asst[200];
} AdmSessRow;

static int adm_sess_cmp(const void *a, const void *b) {
    const AdmSessRow *x = (const AdmSessRow *)a, *y = (const AdmSessRow *)b;
    if (x->updated != y->updated) return (x->updated < y->updated) ? 1 : -1;
    return strcmp(x->sid, y->sid);
}

static void handle_admin_sessions(VLLMServerCtx *ctx, const VHttpRequest *req,
                                  VHttpResponse *resp) {
    VJson *o = vjson_new_object();
    json_put_str(o, "dir", ctx->session_dir);
    vjson_obj_set(o, "enabled",
                  vjson_new_bool(ctx->session_on && ctx->session_dir[0]));
    if (!ctx->session_on || !ctx->session_dir[0]) {
        json_put_str(o, "error", "session dir not configured (--session-dir)");
        admin_json_reply(resp, o);
        return;
    }

    char f_tenant[VLLM_SESS_PART_MAX] = "", f_user[VLLM_SESS_PART_MAX] = "";
    char f_rb[VLLM_RB_ID_MAX] = "", f_q[192] = "", tmp[64] = "";
    adm_qs(req, "tenant", f_tenant, sizeof(f_tenant));
    adm_qs(req, "user", f_user, sizeof(f_user));
    adm_qs(req, "rulebook", f_rb, sizeof(f_rb));
    adm_qs(req, "q", f_q, sizeof(f_q));
    long since = 0, until = 0;
    if (adm_qs(req, "since", tmp, sizeof(tmp))) since = atol(tmp);
    if (adm_qs(req, "until", tmp, sizeof(tmp))) until = atol(tmp);
    int limit = 100, mask = 0;
    if (adm_qs(req, "limit", tmp, sizeof(tmp))) {
        limit = atoi(tmp);
        if (limit < 1) limit = 1;
        if (limit > 1000) limit = 1000;
    }
    if (adm_qs(req, "mask", tmp, sizeof(tmp))) mask = (tmp[0] == '1');

    int cap = 128, n = 0, scanned = 0;
    AdmSessRow *rows = (AdmSessRow *)malloc((size_t)cap * sizeof(AdmSessRow));
    if (!rows) { json_put_str(o, "error", "out of memory"); admin_json_reply(resp, o); return; }

    DIR *d = opendir(ctx->session_dir);
    if (d) {
        struct dirent *de;
        while ((de = readdir(d)) != NULL) {
            if (!adm_sess_file_ok(de->d_name)) continue;
            char path[1600];
            snprintf(path, sizeof(path), "%s/%s", ctx->session_dir, de->d_name);
            struct stat sb;
            if (stat(path, &sb) != 0 || !S_ISREG(sb.st_mode)) continue;
            scanned++;
            if (scanned > 5000) break;   /* 单次管理请求的扫描上限 */

            long size = (long)sb.st_size;
            size_t want = (size > ADM_SESS_HEAD) ? ADM_SESS_HEAD : (size_t)size;
            FILE *f = fopen(path, "rb");
            if (!f) continue;
            char *txt = (char *)malloc(want + 1);
            if (!txt) { fclose(f); continue; }
            size_t got = fread(txt, 1, want, f);
            fclose(f);
            txt[got] = '\0';
            int fully = (got == (size_t)size);

            AdmSessRow r;
            memset(&r, 0, sizeof(r));
            snprintf(r.file, sizeof(r.file), "%s", de->d_name);
            adm_sess_str(txt, "session_id", r.sid, sizeof(r.sid));
            adm_sess_str(txt, "tenant_id", r.tenant, sizeof(r.tenant));
            adm_sess_str(txt, "user_id", r.user, sizeof(r.user));
            adm_sess_str(txt, "rulebook_id", r.rb, sizeof(r.rb));
            r.created = adm_sess_num(txt, "created_s");
            r.updated = adm_sess_num(txt, "updated_s");
            r.n_msgs  = (int)adm_sess_num(txt, "n_messages");
            r.n_tokens = (int)adm_sess_num(txt, "n_tokens");
            r.size = size;

            if (fully) {   /* 完整读入：用真解析取首/末轮预览 */
                VJson *root = vjson_parse(txt);
                const VJson *msgs = root ? vjson_obj_get(root, "messages") : NULL;
                size_t nm = vjson_array_len(msgs);
                for (size_t i = 0; i < nm; i++) {
                    const VJson *m = vjson_array_get(msgs, i);
                    const char *role = vjson_str(vjson_obj_get(m, "role"));
                    const char *ct = vjson_str(vjson_obj_get(m, "content"));
                    if (!role) continue;
                    if (strcmp(role, "user") == 0) {
                        adm_preview(ct, r.first_user, sizeof(r.first_user), 120, mask);
                        r.have_preview = 1;
                        break;
                    }
                }
                for (size_t i = nm; i-- > 0; ) {
                    const VJson *m = vjson_array_get(msgs, i);
                    const char *role = vjson_str(vjson_obj_get(m, "role"));
                    const char *ct = vjson_str(vjson_obj_get(m, "content"));
                    if (!role) continue;
                    if (strcmp(role, "assistant") == 0) {
                        adm_preview(ct, r.last_asst, sizeof(r.last_asst), 120, mask);
                        break;
                    }
                }
                vjson_free(root);
            }
            free(txt);

            /* 过滤 */
            if (f_tenant[0] && strcmp(f_tenant, r.tenant) != 0) continue;
            if (f_user[0] && strcmp(f_user, r.user) != 0) continue;
            if (f_rb[0] && strcmp(f_rb, r.rb) != 0) continue;
            if (since > 0 && r.updated < since) continue;
            if (until > 0 && r.updated > until) continue;
            if (f_q[0] && !strstr(r.sid, f_q) && !strstr(r.tenant, f_q) &&
                !strstr(r.user, f_q) && !strstr(r.rb, f_q) &&
                !strstr(r.first_user, f_q) && !strstr(r.last_asst, f_q))
                continue;

            if (n == cap) {
                int nc = cap * 2;
                AdmSessRow *nr = (AdmSessRow *)realloc(rows, (size_t)nc * sizeof(AdmSessRow));
                if (!nr) break;
                rows = nr; cap = nc;
            }
            rows[n++] = r;
        }
        closedir(d);
    }

    qsort(rows, (size_t)n, sizeof(AdmSessRow), adm_sess_cmp);
    int shown = n < limit ? n : limit;
    VJson *arr = vjson_new_array();
    for (int i = 0; i < shown; i++) {
        AdmSessRow *r = &rows[i];
        VJson *e = vjson_new_object();
        json_put_str(e, "file", r->file);
        json_put_str(e, "session_id", r->sid);
        json_put_str(e, "tenant_id", r->tenant);
        json_put_str(e, "user_id", r->user);
        json_put_str(e, "rulebook_id", r->rb);
        vjson_obj_set(e, "created_s", vjson_new_number((double)r->created));
        vjson_obj_set(e, "updated_s", vjson_new_number((double)r->updated));
        vjson_obj_set(e, "n_messages", vjson_new_number((double)r->n_msgs));
        vjson_obj_set(e, "n_tokens", vjson_new_number((double)r->n_tokens));
        vjson_obj_set(e, "size", vjson_new_number((double)r->size));
        vjson_obj_set(e, "have_preview", vjson_new_bool(r->have_preview));
        json_put_str(e, "first_user", r->first_user);
        json_put_str(e, "last_assistant", r->last_asst);
        vjson_array_push(arr, e);
    }
    free(rows);

    vjson_obj_set(o, "sessions", arr);
    vjson_obj_set(o, "total", vjson_new_number((double)n));
    vjson_obj_set(o, "shown", vjson_new_number((double)shown));
    vjson_obj_set(o, "masked", vjson_new_bool(mask));
    admin_json_reply_cap(resp, o, 4u << 20);
}

static void handle_admin_session(VLLMServerCtx *ctx, const VHttpRequest *req,
                                 VHttpResponse *resp) {
    char actor[96], role[64], file[64] = "", tmp[64] = "";
    adm_actor(req, actor, sizeof(actor));
    adm_role(req, role, sizeof(role));
    adm_qs(req, "file", file, sizeof(file));
    int mask = 0, export_it = 0;
    if (adm_qs(req, "mask", tmp, sizeof(tmp))) mask = (tmp[0] == '1');
    if (adm_qs(req, "export", tmp, sizeof(tmp))) export_it = (tmp[0] == '1');

    VJson *o = vjson_new_object();
    if (!ctx->session_on || !ctx->session_dir[0]) {
        vllm_audit_add(ctx, actor, role, "session_view", file, "error",
                       "session dir not configured");
        json_put_str(o, "error", "session dir not configured (--session-dir)");
        admin_json_reply(resp, o);
        return;
    }
    if (!adm_sess_file_ok(file)) {
        vllm_audit_add(ctx, actor, role, "session_view", file, "error",
                       "bad session file name");
        resp->status = 400;
        json_put_str(o, "error", "bad 'file' (expect sess_<16hex>.json)");
        admin_json_reply(resp, o);
        return;
    }

    char path[1600];
    snprintf(path, sizeof(path), "%s/%s", ctx->session_dir, file);
    struct stat sb;
    if (stat(path, &sb) != 0 || sb.st_size <= 0 ||
        (unsigned long long)sb.st_size > (8u << 20)) {
        vllm_audit_add(ctx, actor, role, "session_view", file, "error",
                       "not found or too large");
        resp->status = 404;
        json_put_str(o, "error", "session not found (or > 8 MB)");
        admin_json_reply(resp, o);
        return;
    }
    FILE *f = fopen(path, "rb");
    if (!f) {
        vllm_audit_add(ctx, actor, role, "session_view", file, "error", "open failed");
        resp->status = 500;
        json_put_str(o, "error", "cannot open session file");
        admin_json_reply(resp, o);
        return;
    }
    size_t sz = (size_t)sb.st_size;
    char *txt = (char *)malloc(sz + 1);
    if (!txt) { fclose(f); resp->status = 500; json_put_str(o, "error", "out of memory"); admin_json_reply(resp, o); return; }
    size_t got = fread(txt, 1, sz, f);
    fclose(f);
    txt[got] = '\0';

    VJson *root = vjson_parse(txt);
    free(txt);
    if (!root) {
        vllm_audit_add(ctx, actor, role, "session_view", file, "error",
                       "unparseable session JSON");
        resp->status = 500;
        json_put_str(o, "error", "session file is not valid JSON");
        admin_json_reply(resp, o);
        return;
    }

    const VJson *msgs = vjson_obj_get(root, "messages");
    if (mask && msgs && msgs->type == VJ_ARRAY) {
        size_t nm = vjson_array_len(msgs);
        for (size_t i = 0; i < nm; i++) {
            VJson *m = vjson_array_get(msgs, i);
            if (m && m->type == VJ_OBJECT)
                vjson_obj_set(m, "content", vjson_new_string("***"));
        }
    }

    vjson_obj_set(o, "file", vjson_new_string(file));
    vjson_obj_set(o, "masked", vjson_new_bool(mask));
    vjson_obj_set(o, "export", vjson_new_bool(export_it));
    const char *keys[] = { "session_id", "tenant_id", "user_id", "rulebook_id" };
    for (int i = 0; i < 4; i++) {
        const VJson *v = vjson_obj_get(root, keys[i]);
        vjson_obj_set(o, keys[i], v ? vjson_clone(v) : vjson_new_string(""));
    }
    const char *nums[] = { "created_s", "updated_s", "n_tokens", "n_messages",
                           "last_ms" };
    for (int i = 0; i < 5; i++) {
        const VJson *v = vjson_obj_get(root, nums[i]);
        vjson_obj_set(o, nums[i], vjson_new_number(v ? vjson_num(v) : 0.0));
    }
    vjson_obj_set(o, "messages", msgs ? vjson_clone(msgs) : vjson_new_array());
    vjson_free(root);

    /* 引擎侧上下文（供审查页对照：模型身份/规则包/前缀复用计数） */
    json_put_str(o, "engine_model", ctx->model_name[0] ? ctx->model_name
                                                       : (ctx->model_id ? ctx->model_id : ""));
    vjson_obj_set(o, "rulebook_hits", vjson_new_number((double)ctx->rulebook_hits));
    vjson_obj_set(o, "session_turns", vjson_new_number((double)ctx->session_turns));

    vllm_audit_add(ctx, actor, role, export_it ? "session_export" : "session_view",
                   file, "ok", mask ? "masked=1" : "masked=0");
    admin_json_reply_cap(resp, o, 16u << 20);
}

static void handle_admin_session_delete(VLLMServerCtx *ctx, const VHttpRequest *req,
                                        VHttpResponse *resp) {
    char actor[96], role[64];
    adm_actor(req, actor, sizeof(actor));
    adm_role(req, role, sizeof(role));

    char file[64] = "";
    char *copy = NULL;
    if (req->body && req->body_len) {
        copy = (char *)malloc(req->body_len + 1);
        if (copy) {
            memcpy(copy, req->body, req->body_len);
            copy[req->body_len] = '\0';
            VJson *b = vjson_parse(copy);
            if (b) {
                const char *fv = vjson_str(vjson_obj_get(b, "file"));
                if (fv) snprintf(file, sizeof(file), "%s", fv);
                vjson_free(b);
            }
        }
    }

    VJson *o = vjson_new_object();
    if (!ctx->session_on || !ctx->session_dir[0]) {
        json_put_str(o, "error", "session dir not configured (--session-dir)");
        admin_json_reply(resp, o);
        free(copy);
        return;
    }
    if (!adm_sess_file_ok(file)) {
        vllm_audit_add(ctx, actor, role, "session_delete", file, "error",
                       "bad session file name");
        resp->status = 400;
        json_put_str(o, "error", "bad 'file' (expect sess_<16hex>.json)");
        admin_json_reply(resp, o);
        free(copy);
        return;
    }
    if (!adm_write_allowed(req)) {
        vllm_audit_add(ctx, actor, role, "session_delete", file, "denied",
                       "RBAC: role not in VLLM_ADMIN_ROLES");
        resp->status = 403;
        json_put_str(o, "error", "forbidden: role not allowed to delete "
                                 "(set X-Audit-Role / VLLM_ADMIN_ROLES)");
        admin_json_reply(resp, o);
        free(copy);
        return;
    }

    char path[1600];
    snprintf(path, sizeof(path), "%s/%s", ctx->session_dir, file);
    int rc = remove(path);
    vllm_audit_add(ctx, actor, role, "session_delete", file,
                   rc == 0 ? "ok" : "error",
                   rc == 0 ? "deleted" : "remove failed / not found");
    vjson_obj_set(o, "ok", vjson_new_bool(rc == 0));
    if (rc != 0) json_put_str(o, "error", "delete failed (not found?)");
    admin_json_reply(resp, o);
    free(copy);
}

static void handle_admin_rulebooks(VLLMServerCtx *ctx, VHttpResponse *resp) {
    VJson *o = vjson_new_object();
    json_put_str(o, "dir", ctx->rulebook_dir);
    vjson_obj_set(o, "enabled", vjson_new_bool(ctx->rulebook_on && ctx->rulebook_dir[0]));
    if (!ctx->rulebook_on || !ctx->rulebook_dir[0]) {
        json_put_str(o, "error", "rulebook dir not configured (--rulebook-dir)");
        admin_json_reply(resp, o);
        return;
    }
    char cur_fp[VLLM_RB_HASH_HEX];
    vllm_rb_model_fp(ctx, cur_fp);

    VJson *arr = vjson_new_array();
    DIR *d = opendir(ctx->rulebook_dir);
    if (d) {
        struct dirent *de;
        while ((de = readdir(d)) != NULL) {
            const char *nm = de->d_name;
            size_t ln = strlen(nm);
            if (ln < 9 || strncmp(nm, "rb_", 3) != 0 ||
                strcmp(nm + ln - 5, ".json") != 0)
                continue;
            char id[VLLM_RB_ID_MAX];
            size_t idl = ln - 5 - 3;
            if (idl >= sizeof(id)) continue;
            memcpy(id, nm + 3, idl);
            id[idl] = '\0';

            VLLMRulebook rb;
            if (vllm_rb_load_meta(ctx, id, &rb) != 0) continue;
            VJson *e = vjson_new_object();
            json_put_str(e, "rulebook_id", rb.id);
            json_put_str(e, "name", rb.name);
            json_put_str(e, "version", rb.version);
            json_put_str(e, "tenant_id", rb.tenant_id);
            json_put_str(e, "scope", rb.scope);
            json_put_str(e, "hash", rb.hash);
            json_put_str(e, "model_fp", rb.model_fp);
            vjson_obj_set(e, "n_tokens", vjson_new_number((double)rb.n_tokens));
            vjson_obj_set(e, "n_chunks", vjson_new_number((double)rb.n_chunks));
            vjson_obj_set(e, "created_s", vjson_new_number((double)rb.created_s));
            vjson_obj_set(e, "updated_s", vjson_new_number((double)rb.updated_s));
            /* 模型指纹是否与当前加载的模型一致（不一致 = 快照不参与复用）。 */
            int fp_match = rb.model_fp[0] && cur_fp[0] &&
                           strcmp(rb.model_fp, cur_fp) == 0;
            vjson_obj_set(e, "model_fp_match", vjson_new_bool(fp_match));
            json_put_str(e, "current_model_fp", cur_fp);

            /* 该规则包的 KV 快照清单 rbk_<id>_*.kv */
            VJson *snaps = vjson_new_array();
            char prefix[VLLM_RB_ID_MAX + 16];
            snprintf(prefix, sizeof(prefix), "rbk_%s_", id);
            size_t plen = strlen(prefix);
            DIR *d2 = opendir(ctx->rulebook_dir);
            if (d2) {
                struct dirent *de2;
                while ((de2 = readdir(d2)) != NULL) {
                    const char *n2 = de2->d_name;
                    size_t l2 = strlen(n2);
                    if (l2 < 4 || strcmp(n2 + l2 - 3, ".kv") != 0) continue;
                    if (strncmp(n2, prefix, plen) != 0) continue;
                    char p2[1800];
                    snprintf(p2, sizeof(p2), "%s/%s", ctx->rulebook_dir, n2);
                    struct stat sb2;
                    VJson *s = vjson_new_object();
                    json_put_str(s, "file", n2);
                    vjson_obj_set(s, "size", vjson_new_number(
                        (stat(p2, &sb2) == 0) ? (double)sb2.st_size : 0.0));
                    vjson_array_push(snaps, s);
                }
                closedir(d2);
            }
            vjson_obj_set(e, "kv_snapshots", snaps);
            vjson_obj_set(e, "n_kv_snapshots",
                          vjson_new_number((double)vjson_array_len(snaps)));
            vjson_array_push(arr, e);
        }
        closedir(d);
    }
    vjson_obj_set(o, "rulebooks", arr);
    vjson_obj_set(o, "total", vjson_new_number((double)vjson_array_len(arr)));
    admin_json_reply_cap(resp, o, 4u << 20);
}

static void handle_admin_rulebook_delete(VLLMServerCtx *ctx, const VHttpRequest *req,
                                         VHttpResponse *resp) {
    char actor[96], role[64], id[VLLM_RB_ID_MAX] = "";
    adm_actor(req, actor, sizeof(actor));
    adm_role(req, role, sizeof(role));
    char *copy = NULL;
    if (req->body && req->body_len) {
        copy = (char *)malloc(req->body_len + 1);
        if (copy) {
            memcpy(copy, req->body, req->body_len);
            copy[req->body_len] = '\0';
            VJson *b = vjson_parse(copy);
            if (b) {
                const char *v = vjson_str(vjson_obj_get(b, "rulebook_id"));
                if (v) snprintf(id, sizeof(id), "%s", v);
                vjson_free(b);
            }
        }
    }

    VJson *o = vjson_new_object();
    if (!ctx->rulebook_on || !ctx->rulebook_dir[0]) {
        json_put_str(o, "error", "rulebook dir not configured (--rulebook-dir)");
        admin_json_reply(resp, o);
        free(copy);
        return;
    }
    if (!id[0]) {
        vllm_audit_add(ctx, actor, role, "rulebook_delete", "", "error",
                       "missing rulebook_id");
        resp->status = 400;
        json_put_str(o, "error", "'rulebook_id' is required");
        admin_json_reply(resp, o);
        free(copy);
        return;
    }
    if (!adm_write_allowed(req)) {
        vllm_audit_add(ctx, actor, role, "rulebook_delete", id, "denied",
                       "RBAC: role not in VLLM_ADMIN_ROLES");
        resp->status = 403;
        json_put_str(o, "error", "forbidden: role not allowed to delete "
                                 "(set X-Audit-Role / VLLM_ADMIN_ROLES)");
        admin_json_reply(resp, o);
        free(copy);
        return;
    }
    /* ID 白名单门：与 sessions/delete 的文件名门对称，把"非法 ID"（客户端
     * 缺陷）与"不存在的 ID"分开，避免 200 + ok:false 让调用方无法归因。 */
    if (!vllm_rb_id_ok(id)) {
        vllm_audit_add(ctx, actor, role, "rulebook_delete", id, "error",
                       "bad rulebook id");
        resp->status = 400;
        json_put_str(o, "error", "bad 'rulebook_id' (expect [A-Za-z0-9._-])");
        admin_json_reply(resp, o);
        free(copy);
        return;
    }
    int removed = vllm_rb_delete(ctx, id);
    vllm_audit_add(ctx, actor, role, "rulebook_delete", id,
                   removed > 0 ? "ok" : "error",
                   removed > 0 ? "files removed" : "nothing removed (not found)");
    vjson_obj_set(o, "ok", vjson_new_bool(removed > 0));
    vjson_obj_set(o, "removed", vjson_new_number((double)removed));
    if (removed <= 0) {
        resp->status = 404;
        json_put_str(o, "error", "rulebook not found");
    }
    admin_json_reply(resp, o);
    free(copy);
}

static void handle_admin_audit(VLLMServerCtx *ctx, const VHttpRequest *req,
                               VHttpResponse *resp) {
    char actor[96], role[64], tmp[64] = "";
    adm_actor(req, actor, sizeof(actor));
    adm_role(req, role, sizeof(role));
    int n = 200, do_verify = 1;
    if (adm_qs(req, "n", tmp, sizeof(tmp))) {
        n = atoi(tmp);
        if (n < 1) n = 1;
        if (n > 2000) n = 2000;
    }
    tmp[0] = '\0';
    if (adm_qs(req, "verify", tmp, sizeof(tmp))) do_verify = (tmp[0] != '0');

    VJson *o = vjson_new_object();
    vjson_obj_set(o, "enabled", vjson_new_bool(vllm_audit_on(ctx)));
    json_put_str(o, "path", ctx->audit_log);
    vjson_obj_set(o, "size", vjson_new_number((double)vllm_audit_size(ctx)));

    int cnt = 0;
    char *arr = vllm_audit_tail(ctx, n, &cnt);
    if (arr) {
        VJson *entries = vjson_parse(arr);
        vjson_obj_set(o, "entries", entries ? entries : vjson_new_array());
        free(arr);
    } else {
        vjson_obj_set(o, "entries", vjson_new_array());
    }
    vjson_obj_set(o, "count", vjson_new_number((double)cnt));

    if (do_verify) {
        int checked = 0, full = 0, genesis = 0, empty = 0;
        long bad = 0, first_seq = 0;
        int ok = vllm_audit_verify(ctx, &checked, &bad, &full, &genesis,
                                   &first_seq, &empty);
        VJson *v = vjson_new_object();
        vjson_obj_set(v, "ok", vjson_new_bool(ok));
        vjson_obj_set(v, "checked", vjson_new_number((double)checked));
        vjson_obj_set(v, "bad_seq", vjson_new_number((double)bad));
        vjson_obj_set(v, "full", vjson_new_bool(full));
        /* genesis=0 且 full=1：链在文件内自洽，但首行不是 seq=1（前缀被归档/删除，
         * 与"篡改"不可区分）——UI 必须用黄色提示，不得冒充绿色通过。
         * empty=1：日志为空/不存在（刚启用或刚轮转），也不是篡改。 */
        vjson_obj_set(v, "genesis", vjson_new_bool(genesis));
        vjson_obj_set(v, "first_seq", vjson_new_number((double)first_seq));
        vjson_obj_set(v, "empty", vjson_new_bool(empty));
        vjson_obj_set(o, "verify", v);
    }

    vllm_audit_add(ctx, actor, role, "audit_view", "",
                   vllm_audit_on(ctx) ? "ok" : "error",
                   vllm_audit_on(ctx) ? "" : "audit log disabled");
    admin_json_reply_cap(resp, o, 4u << 20);
}

/* ---------- router (called from vllm_server.c) ---------- */

void vllm_admin_route(VLLMServerCtx *ctx, const VHttpRequest *req,
                      VHttpResponse *resp) {
    if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/admin/") == 0) {
        handle_admin_page(resp);
    } else if (strcmp(req->method, "GET") == 0 &&
               strcmp(req->path, "/admin/api/status") == 0) {
        handle_admin_status(ctx, resp);
    } else if (strcmp(req->method, "GET") == 0 &&
               strcmp(req->path, "/admin/api/config") == 0) {
        handle_admin_config(ctx, resp);
    } else if (strcmp(req->method, "GET") == 0 &&
               strcmp(req->path, "/admin/api/device") == 0) {
        handle_admin_device(req, resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/admin/api/config/save") == 0) {
        handle_admin_config_save(ctx, req, resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/admin/api/model/load") == 0) {
        handle_admin_model_load(ctx, req, resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/admin/api/model/unload") == 0) {
        handle_admin_model_unload(ctx, req, resp);
    } else if (strcmp(req->method, "GET") == 0 &&
               strcmp(req->path, "/admin/api/log") == 0) {
        handle_admin_log(ctx, req, resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/admin/api/shutdown") == 0) {
        handle_admin_shutdown(ctx, resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/admin/api/l3/clear") == 0) {
        handle_admin_l3_clear(ctx, resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/admin/api/policy/config") == 0) {
        handle_admin_policy_config(ctx, req, resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/admin/api/policy/set") == 0) {
        handle_admin_policy_set(ctx, req, resp);
    } else if (strcmp(req->method, "GET") == 0 &&
               strcmp(req->path, "/admin/api/rulebooks") == 0) {
        handle_admin_rulebooks(ctx, resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/admin/api/rulebooks/delete") == 0) {
        handle_admin_rulebook_delete(ctx, req, resp);
    } else if (strcmp(req->method, "GET") == 0 &&
               strcmp(req->path, "/admin/api/sessions") == 0) {
        handle_admin_sessions(ctx, req, resp);
    } else if (strcmp(req->method, "GET") == 0 &&
               strcmp(req->path, "/admin/api/session") == 0) {
        handle_admin_session(ctx, req, resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/admin/api/sessions/delete") == 0) {
        handle_admin_session_delete(ctx, req, resp);
    } else if (strcmp(req->method, "GET") == 0 &&
               strcmp(req->path, "/admin/api/audit") == 0) {
        handle_admin_audit(ctx, req, resp);
    } else {
        static const char nf[] = "{\"error\":\"admin: not found\"}";
        resp->status = 404;
        resp->content_type = "application/json";
        resp->body = nf;
        resp->body_len = sizeof(nf) - 1;
    }
}

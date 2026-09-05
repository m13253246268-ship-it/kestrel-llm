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
 * Process start/stop supervision lives in tools/vllm_mgr.py (a stopped
 * engine cannot start itself); the admin page drives it over a second
 * tiny HTTP port (default 8082).
 *
 * Config persisted by /admin/api/config/save is a plain JSON file that
 * vllm_mgr.py reads to build the engine's command line on start.
 * ================================================================ */
#include "vllm_server.h"
#include "vllm_batch.h"      /* vllm_batch_active (continuous batching status) */
#include "vllm_http.h"
#include "vllm_device.h"
#include "embedded_web.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <ctype.h>

#include <unistd.h>
#include <sys/stat.h>
#include <sys/wait.h>
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

static void json_put_str(VJson *obj, const char *key, const char *val) {
    vjson_obj_set(obj, key, vjson_new_string(val ? val : ""));
}

/* Build the current effective configuration as JSON. */
static VJson *build_config_json(const VLLMServerCtx *ctx) {
    VJson *o = vjson_new_object();
    json_put_str(o, "model_dir", ctx->model_dir);
    json_put_str(o, "wmode", wmode_str(g_st_wmode));
    vjson_obj_set(o, "kv_q4", vjson_new_bool(g_kv_q4));
    vjson_obj_set(o, "prefix_cache", vjson_new_bool(ctx->prefix_cache));
    vjson_obj_set(o, "prefill_q8", vjson_new_bool(g_st_prefill_q8));
    vjson_obj_set(o, "sparse_attn", vjson_new_bool(g_sparse_attn));
    vjson_obj_set(o, "sparse_k", vjson_new_number((double)g_sparse_k));
    vjson_obj_set(o, "l3_evict", vjson_new_bool(g_l3_evict));
    vjson_obj_set(o, "l3_ratio", vjson_new_number((double)g_l3_ratio));
    vjson_obj_set(o, "l3_min_seq", vjson_new_number((double)g_l3_min_seq));
    vjson_obj_set(o, "l3_path", vjson_new_string(
        (g_l3_path && g_l3_path[0]) ? g_l3_path : "kv_l3.bin"));
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
    vjson_obj_set(o, "min_free_mb", vjson_new_number((double)ctx->min_free_mb));
    /* Conversation features (KV prefix reuse / disk persistence / spec decode /
     * min-p), see vllm_server.h. disk_kv_dir is the checkpoint directory. */
    vjson_obj_set(o, "prefix_kv", vjson_new_bool(ctx->prefix_kv));
    vjson_obj_set(o, "disk_kv", vjson_new_bool(ctx->disk_kv));
    json_put_str(o, "disk_kv_dir", ctx->kvdir);
    vjson_obj_set(o, "spec", vjson_new_bool(ctx->spec));
    vjson_obj_set(o, "spec_k", vjson_new_number((double)ctx->spec_k));
    vjson_obj_set(o, "min_p", vjson_new_number(ctx->min_p));
    /* Load-on-use / unload-when-idle lifecycle config */
    vjson_obj_set(o, "auto_load", vjson_new_bool(ctx->auto_load));
    vjson_obj_set(o, "auto_unload_s", vjson_new_number((double)ctx->auto_unload_s));
    /* 模型加载格式优先级（管理页"加载格式"下拉） */
    json_put_str(o, "format", ctx->load_format_str[0] ? ctx->load_format_str
                                                      : "auto");
    json_put_str(o, "device", ctx->device_id);
    VJson *env = vjson_new_object();
    json_put_str(env, "OMP_NUM_THREADS",
                 getenv_int_str("OMP_NUM_THREADS", "4"));
    /* NEON kernel / threadpool optimization switches (admin.html 内核优化
     * card). Only echo vars that are explicitly set; fillForm treats absent
     * keys as the engine default. */
    static const char *opt_envs[] = {
        "VLLM_Q8_8X8", "VLLM_DISABLE_Q8_REPACK", "VLLM_DISABLE_Q4_REPACK",
        "VLLM_ENABLE_8X8L", "VLLM_PB_HEAP", "VLLM_TP_BIND", "VLLM_TP_SPIN",
        "VLLM_ROW_SLICE", "VLLM_VQF_KEY", NULL
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
            mkdir(dir, 0755);
            *p = save;
        }
    }
    mkdir(dir, 0755);
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

static void admin_json_reply(VHttpResponse *resp, VJson *root) {
    char buf[16384];
    size_t n = vjson_serialize(root, buf, sizeof(buf));
    vjson_free(root);
    if (n == 0 || n >= sizeof(buf)) {
        resp->status = 500;
        resp->content_type = "application/json";
        resp->body = "{\"error\":\"response too large\"}";
        resp->body_len = strlen(resp->body);
        return;
    }
    resp->status = 200;
    resp->content_type = "application/json";
    resp->body_owned = (char *)malloc(n + 1);
    if (resp->body_owned) {
        memcpy(resp->body_owned, buf, n + 1);
        resp->body = resp->body_owned;
        resp->body_len = n;
    } else {
        resp->status = 500;
        resp->body = "{\"error\":\"out of memory\"}";
        resp->body_len = strlen(resp->body);
    }
}

static void handle_admin_status(const VLLMServerCtx *ctx, VHttpResponse *resp) {
    VJson *o = vjson_new_object();
    vjson_obj_set(o, "engine", vjson_new_string("vllm_kestrel"));
    json_put_str(o, "model_id", ctx->model_id);
    json_put_str(o, "model_name", ctx->model_name[0] ? ctx->model_name : NULL);
    json_put_str(o, "model_dir", ctx->model_dir);
    json_put_str(o, "wmode", wmode_str(g_st_wmode));
    /* Device profile */
    json_put_str(o, "device_id", ctx->device_id);
    json_put_str(o, "device_name", ctx->device_name);
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
    vjson_obj_set(o, "config_path", vjson_new_string(admin_config_path(ctx)));
    vjson_obj_set(o, "log_path", vjson_new_string(admin_log_path(ctx)));
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
    vjson_obj_set(o, "config_path", vjson_new_string(admin_config_path(ctx)));
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

/* ================================================================
 * VQF 模型转换（独立子进程执行 --convert-vqf）
 *
 * 转换 = 引擎自己跑一遍完整加载序列（alloc → 逐层量化 → repack →
 * lm_head → vision）后 dump 成 VQF 单文件，峰值内存 ~4.5GB+，且会写
 * 全局 wmode/repack 状态 —— 必须在独立子进程执行，不能进 serve 进程
 * （避免与正在服务的模型争内存/污染全局状态）。子进程 stdout/stderr
 * 重定向到 g_conv_log，前端轮询 /admin/api/convert 拿状态 + 日志尾部。
 * ================================================================ */
#define CONV_IDLE     0
#define CONV_RUNNING  1
#define CONV_DONE     2
#define CONV_FAILED   3

static volatile int  g_conv_state = CONV_IDLE;
static pid_t         g_conv_pid   = -1;
static time_t        g_conv_start = 0;
static int           g_conv_rc    = -1;
static char          g_conv_model[512] = "";
static char          g_conv_wmode[32]  = "";
static char          g_conv_out[512]   = "";
static char          g_conv_log[512]   = "/tmp/vqf_convert.log";

static void *conv_wait_thread(void *arg) {
    (void)arg;
    int st = 0;
    pid_t pid = g_conv_pid;
    if (pid > 0) waitpid(pid, &st, 0);
    g_conv_rc = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
    g_conv_state = (g_conv_rc == 0) ? CONV_DONE : CONV_FAILED;
    g_conv_pid = -1;
    return NULL;
}

static const char *conv_state_name(int s) {
    switch (s) {
    case CONV_RUNNING: return "running";
    case CONV_DONE:    return "done";
    case CONV_FAILED:  return "failed";
    default:           return "idle";
    }
}

/* POST /admin/api/convert
 * {"model_dir","wmode","out","src_type"} —— 启动转换子进程
 * src_type: "gguf" = llama.cpp .gguf 文件输入（--convert-gguf）；
 *           其它/缺省 = safetensors 目录输入（--convert-vqf）。
 *           也可不传，后端按 model_dir 是否以 .gguf 结尾自动判定。 */
static void handle_admin_convert_start(VLLMServerCtx *ctx,
                                       const VHttpRequest *req,
                                       VHttpResponse *resp) {
    VJson *o = vjson_new_object();
    if (g_conv_state == CONV_RUNNING) {
        vjson_obj_set(o, "ok", vjson_new_bool(0));
        json_put_str(o, "error", "a conversion is already running");
        admin_json_reply(resp, o);
        return;
    }
    char md_buf[600] = "", wm_buf[32] = "dual", out_buf[600] = "";
    char st_buf[16] = "", enc_buf[600] = "";
    if (req->body && req->body_len > 0) {
        char *copy = (char *)malloc((size_t)req->body_len + 1);
        if (copy) {
            memcpy(copy, req->body, req->body_len);
            copy[req->body_len] = '\0';
            VJson *root = vjson_parse(copy);
            if (root && root->type == VJ_OBJECT) {
                /* 立即拷贝到局部缓冲：vjson 字符串指针在 vjson_free 后悬空 */
                const VJson *md = vjson_obj_get(root, "model_dir");
                if (md && md->type == VJ_STRING && vjson_str(md)[0])
                    snprintf(md_buf, sizeof(md_buf), "%s", vjson_str(md));
                const VJson *wm = vjson_obj_get(root, "wmode");
                if (wm && wm->type == VJ_STRING && vjson_str(wm)[0])
                    snprintf(wm_buf, sizeof(wm_buf), "%s", vjson_str(wm));
                const VJson *oo = vjson_obj_get(root, "out");
                if (oo && oo->type == VJ_STRING && vjson_str(oo)[0])
                    snprintf(out_buf, sizeof(out_buf), "%s", vjson_str(oo));
                const VJson *st = vjson_obj_get(root, "src_type");
                if (st && st->type == VJ_STRING && vjson_str(st)[0])
                    snprintf(st_buf, sizeof(st_buf), "%s", vjson_str(st));
                /* VQF-Enc：口令透传给转换子进程（setenv VLLM_VQF_KEY），
                 * 非空即加密导出，空则明文 VQF。 */
                const VJson *ek = vjson_obj_get(root, "enc_key");
                if (ek && ek->type == VJ_STRING && vjson_str(ek)[0])
                    snprintf(enc_buf, sizeof(enc_buf), "%s", vjson_str(ek));
            }
            vjson_free(root);
            free(copy);
        }
    }
    const char *model_dir = md_buf[0] ? md_buf : NULL;
    const char *wmode = wm_buf;
    const char *out = out_buf[0] ? out_buf : NULL;
    if (!model_dir && ctx->model_dir && ctx->model_dir[0]) model_dir = ctx->model_dir;
    if (!model_dir) {
        vjson_obj_set(o, "ok", vjson_new_bool(0));
        json_put_str(o, "error", "model_dir is required");
        admin_json_reply(resp, o);
        return;
    }
    /* 判定输入类型：显式 src_type 优先，其次按扩展名 .gguf 自动识别 */
    size_t mdl = strlen(model_dir);
    int is_gguf = (st_buf[0] && strcmp(st_buf, "gguf") == 0) ||
                  (mdl >= 5 && (strcmp(model_dir + mdl - 5, ".gguf") == 0 ||
                                strcmp(model_dir + mdl - 5, ".GGUF") == 0));
    static char outbuf[600];
    if (is_gguf) {
        struct stat gst;
        if (stat(model_dir, &gst) != 0 || !S_ISREG(gst.st_mode)) {
            vjson_obj_set(o, "ok", vjson_new_bool(0));
            char e[600];
            snprintf(e, sizeof(e), "GGUF file not found: %s", model_dir);
            json_put_str(o, "error", e);
            admin_json_reply(resp, o);
            return;
        }
        /* 默认输出：同目录同名 .vqf（去掉 .gguf 扩展名） */
        if (out && out[0]) {
            struct stat ost;
            if (stat(out, &ost) == 0 && S_ISDIR(ost.st_mode)) {
                /* out 是已存在目录：补默认文件名（同模型名 .vqf），避免转换
                 * 跑完后 fopen 目录失败（rc=255 / cannot open）。 */
                const char *slash = strrchr(model_dir, '/');
                const char *base = slash ? slash + 1 : model_dir;
                size_t blen = strlen(base);
                size_t stem = (blen >= 5 &&
                               (strcmp(base + blen - 5, ".gguf") == 0 ||
                                strcmp(base + blen - 5, ".GGUF") == 0)) ? blen - 5 : blen;
                if (stem > sizeof(outbuf) - 2) stem = sizeof(outbuf) - 2;
                snprintf(outbuf, sizeof(outbuf), "%s/%.*s.vqf",
                         out, (int)stem, base);
            } else {
                snprintf(outbuf, sizeof(outbuf), "%s", out);
            }
        } else {
            size_t n = mdl >= 5 ? mdl - 5 : 0;
            if (n + 5 > sizeof(outbuf) - 1) n = sizeof(outbuf) - 6;
            memcpy(outbuf, model_dir, n);
            memcpy(outbuf + n, ".vqf", 5);
        }
    } else {
        char cfgp[600];
        snprintf(cfgp, sizeof(cfgp), "%s/config.json", model_dir);
        struct stat stt;
        if (stat(cfgp, &stt) != 0) {
            vjson_obj_set(o, "ok", vjson_new_bool(0));
            char e[600];
            snprintf(e, sizeof(e), "config.json not found: %s", model_dir);
            json_put_str(o, "error", e);
            admin_json_reply(resp, o);
            return;
        }
        if (out && out[0]) {
            struct stat ost;
            if (stat(out, &ost) == 0 && S_ISDIR(ost.st_mode)) {
                /* out 是已存在目录：补默认文件名 model.vqf（同上，避免 fopen 目录失败） */
                snprintf(outbuf, sizeof(outbuf), "%s/model.vqf", out);
            } else {
                snprintf(outbuf, sizeof(outbuf), "%s", out);
            }
        } else               snprintf(outbuf, sizeof(outbuf), "%s/model.vqf", model_dir);
    }
    /* 本进程可执行文件路径（/proc/self/exe，Linux 板端） */
    char exe[1024];
    ssize_t el = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (el <= 0) {
        vjson_obj_set(o, "ok", vjson_new_bool(0));
        json_put_str(o, "error", "cannot resolve engine executable path");
        admin_json_reply(resp, o);
        return;
    }
    exe[el] = '\0';
    fflush(NULL);
    pid_t pid = fork();
    if (pid < 0) {
        vjson_obj_set(o, "ok", vjson_new_bool(0));
        json_put_str(o, "error", "fork failed");
        admin_json_reply(resp, o);
        return;
    }
    if (pid == 0) {
        int fd = open(g_conv_log, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
            dup2(fd, 1); dup2(fd, 2);
            close(fd);
        }
        const char *conv_flag = is_gguf ? "--convert-gguf" : "--convert-vqf";
        /* VQF-Enc：设置口令后，转换子进程的 vqf_write 读 VLLM_VQF_KEY 生成加密 VQF；
         * 留空则显式清除，避免继承父进程环境导致非预期加密导出 */
        if (enc_buf[0]) setenv("VLLM_VQF_KEY", enc_buf, 1);
        else unsetenv("VLLM_VQF_KEY");
        char *argv[] = { exe, "--model", (char *)model_dir,
                         "--wmode", (char *)wmode,
                         (char *)conv_flag, outbuf, NULL };
        execv(exe, argv);
        fprintf(stderr, "[CONV] execv failed: %s\n", exe);
        _exit(127);
    }
    snprintf(g_conv_model, sizeof(g_conv_model), "%s", model_dir);
    snprintf(g_conv_wmode, sizeof(g_conv_wmode), "%s", wmode);
    snprintf(g_conv_out,   sizeof(g_conv_out),   "%s", outbuf);
    g_conv_pid   = pid;
    g_conv_start = time(NULL);
    g_conv_rc    = -1;
    g_conv_state = CONV_RUNNING;
    pthread_t th;
    if (pthread_create(&th, NULL, conv_wait_thread, NULL) == 0)
        pthread_detach(th);
    vjson_obj_set(o, "ok", vjson_new_bool(1));
    json_put_str(o, "note", "conversion started (see /admin/api/convert)");
    json_put_str(o, "model_dir", model_dir);
    json_put_str(o, "wmode", wmode);
    json_put_str(o, "out", outbuf);
    json_put_str(o, "src_type", is_gguf ? "gguf" : "safetensors");
    admin_json_reply(resp, o);
}

/* GET /admin/api/convert —— 状态 + 日志尾部 + 输出文件大小 */
static void handle_admin_convert_status(VHttpResponse *resp) {
    VJson *o = vjson_new_object();
    json_put_str(o, "state", conv_state_name(g_conv_state));
    if (g_conv_state == CONV_RUNNING) {
        json_put_str(o, "model_dir", g_conv_model);
        json_put_str(o, "wmode", g_conv_wmode);
        json_put_str(o, "out", g_conv_out);
        vjson_obj_set(o, "pid", vjson_new_number((double)g_conv_pid));
        vjson_obj_set(o, "elapsed_s", vjson_new_number((double)(time(NULL) - g_conv_start)));
    } else if (g_conv_state == CONV_DONE || g_conv_state == CONV_FAILED) {
        json_put_str(o, "model_dir", g_conv_model);
        json_put_str(o, "wmode", g_conv_wmode);
        json_put_str(o, "out", g_conv_out);
        vjson_obj_set(o, "rc", vjson_new_number((double)g_conv_rc));
        if (g_conv_start) {
            double el = (double)(time(NULL) - g_conv_start);
            if (el < 0) el = 0;
            vjson_obj_set(o, "elapsed_s", vjson_new_number(el));
        }
    }
    struct stat st;
    if (stat(g_conv_out, &st) == 0)
        vjson_obj_set(o, "out_bytes", vjson_new_number((double)st.st_size));
    char *buf = (char *)malloc(512 * 1024 + 1);
    if (buf) {
        size_t sz = read_file_bounded(g_conv_log, buf, 512 * 1024 + 1);
        if (sz > 0) {
            /* 只回最后一段（避免响应过大）；行尾用原样文本即可 */
            const char *tail = sz > 8192 ? buf + sz - 8192 : buf;
            json_put_str(o, "log_tail", tail);
        }
        free(buf);
    }
    admin_json_reply(resp, o);
}

/* POST /admin/api/convert/cancel —— 杀掉转换子进程 */
static void handle_admin_convert_cancel(VHttpResponse *resp) {
    VJson *o = vjson_new_object();
    if (g_conv_state == CONV_RUNNING && g_conv_pid > 0) {
        kill(g_conv_pid, SIGKILL);
        vjson_obj_set(o, "ok", vjson_new_bool(1));
        json_put_str(o, "note", "conversion process killed");
    } else {
        vjson_obj_set(o, "ok", vjson_new_bool(0));
        json_put_str(o, "error", "no conversion running");
    }
    admin_json_reply(resp, o);
}

/* wmode string ("q4"/"q4i"/"q8"/"g256"/"q2mix"/"dual") -> g_st_wmode int. */
static int wmode_parse(const char *s) {
    if (!s || !s[0]) return -1;
    if (s[0] == 'q' || s[0] == 'Q') {
        if (s[1] == '2') return 5;   /* q2mix */
        if (s[1] == '4' && s[2] == 'i') return 3;   /* q4i */
        if (s[1] == '4') return 1;
        if (s[1] == '8') return 2;
    }
    if (s[0] == 'g' || s[0] == 'G') return 4;       /* g256 */
    if (s[0] == 'd' || s[0] == 'D') return 0;       /* dual */
    return -1;
}

static void handle_admin_model_load(VLLMServerCtx *ctx, const VHttpRequest *req,
                                    VHttpResponse *resp) {
    VJson *o = vjson_new_object();
    /* Optional body: {"wmode": "q4|q8|g256|...", "model_dir": "<path>"}.
     * wmode selects the quantization mode for THIS load (overrides the CLI
     * --wmode); model_dir points the load at a model directory that may not
     * have been resolvable at startup (portable manual-load mode — the
     * packaged exe can run from any directory and the admin page supplies
     * the path). */
    if (req->body && req->body_len > 0) {
        char *copy = (char *)malloc(req->body_len + 1);
        if (copy) {
            memcpy(copy, req->body, req->body_len);
            copy[req->body_len] = '\0';
            VJson *root = vjson_parse(copy);
            if (root && root->type == VJ_OBJECT) {
                const VJson *wm = vjson_obj_get(root, "wmode");
                if (wm && wm->type == VJ_STRING) {
                    int w = wmode_parse(vjson_str(wm));
                    if (w >= 0) ctx->load_wmode = w;
                }
                const VJson *md = vjson_obj_get(root, "model_dir");
                if (md && md->type == VJ_STRING) {
                    const char *s = vjson_str(md);
                    if (s && s[0]) vllm_serve_set_model_dir(ctx, s);
                }
                /* 加载格式优先级（"auto"/"vqf"/"safetensors"/"gguf"） */
                const VJson *fm = vjson_obj_get(root, "format");
                if (fm && fm->type == VJ_STRING) {
                    const char *s = vjson_str(fm);
                    if (s && s[0]) {
                        snprintf(ctx->load_format_str,
                                 sizeof(ctx->load_format_str), "%s", s);
                        if (strcmp(s, "vqf") == 0) ctx->load_format = 1;
                        else if (strcmp(s, "safetensors") == 0) ctx->load_format = 2;
                        else if (strcmp(s, "gguf") == 0) ctx->load_format = 3;
                        else ctx->load_format = 0;   /* auto */
                    }
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
    int rc = vllm_serve_unload_model(ctx);
    if (rc == 0) {
        vjson_obj_set(o, "ok", vjson_new_bool(1));
        json_put_str(o, "note", "model unloaded (weights/KV freed; next request "
                    "or 加载模型 will reload)");
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

static void handle_admin_page(VHttpResponse *resp) {
    /* Embedded admin.html (self-contained exe): used when no file is deployed
     * next to the engine. VLLM_ADMIN_HTML overrides to a custom file. */
    size_t elen = 0;
    const char *ehtml = embedded_admin_html(&elen);
    FILE *f = fopen(admin_html_path(), "rb");
    if (!f) {
        if (ehtml && elen) {
            resp->status = 200;
            resp->content_type = "text/html; charset=utf-8";
            resp->body = ehtml;
            resp->body_len = elen;
            return;
        }
        resp->status = 500;
        resp->content_type = "text/html; charset=utf-8";
        const char *msg =
            "<html><body><h3>admin.html not found</h3>"
            "<p>Set VLLM_ADMIN_HTML or deploy admin.html next to the engine "
            "(e.g. /NewVLLM/admin.html).</p></body></html>";
        resp->body = msg;
        resp->body_len = strlen(msg);
        return;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0 || sz > 8 * 1024 * 1024) {
        fclose(f);
        resp->status = 500;
        resp->body = "admin.html too large";
        return;
    }
    char *html = (char *)malloc((size_t)sz + 1);
    if (!html) {
        fclose(f);
        resp->status = 500;
        resp->body = "out of memory";
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
               strcmp(req->path, "/admin/api/convert") == 0) {
        handle_admin_convert_start(ctx, req, resp);
    } else if (strcmp(req->method, "GET") == 0 &&
               strcmp(req->path, "/admin/api/convert") == 0) {
        handle_admin_convert_status(resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/admin/api/convert/cancel") == 0) {
        handle_admin_convert_cancel(resp);
    } else {
        static const char nf[] = "{\"error\":\"admin: not found\"}";
        resp->status = 404;
        resp->content_type = "application/json";
        resp->body = nf;
        resp->body_len = sizeof(nf) - 1;
    }
}

/* ================================================================
 * vllm_server.c - OpenAI-compatible API server for vllm_kestrel
 *
 * Routes /v1/models, /health and /v1/chat/completions (with SSE
 * streaming) against the engine's Qwen inference state.
 *
 * The engine keeps a single inference state, so inference-critical
 * sections are serialized: clients queue on an internal mutex (bounded,
 * overflow -> 429) and are served one at a time. Metadata endpoints
 * (/v1/models, /health) run concurrently and never wait on inference.
 * Robustness guards: prompt-length check (400), free-RAM check (503),
 * slow-client recv timeout, bounded connection queue (vllm_http).
 * ================================================================ */
#include "vllm_server.h"
#include "vllm_batch.h"      /* continuous batching (--batch-max N) */
#include "vllm_superpos.h"   /* sample_token(): argmax/top-p sampler */
#include "vllm_l3.h"         /* STL3State / l3_state_free (per-user L3 files) */
#include "vllm_attest.h"     /* verifiable-inference attestation (方案 2) */
#include "embedded_web.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* Portable manual-load mode: /admin/api/model/load may supply a model
 * directory that was not present at startup (the packaged exe can be run
 * from any directory; the model path is chosen on the admin page). */
void vllm_serve_set_model_dir(VLLMServerCtx *ctx, const char *dir) {
    if (!ctx || !dir || !dir[0]) return;
    snprintf(ctx->model_dir_buf, sizeof(ctx->model_dir_buf), "%s", dir);
    ctx->model_dir = ctx->model_dir_buf;
}

#include <unistd.h>  /* unlink */
#include <sys/stat.h>
#include <dirent.h>

/* ---------- small growable string buffer ---------- */

typedef struct { char *s; size_t len, cap; } StrBuf;

static void sb_init(StrBuf *b) { b->s = NULL; b->len = 0; b->cap = 0; }
static void sb_free(StrBuf *b) { free(b->s); b->s = NULL; b->len = b->cap = 0; }

static void sb_put(StrBuf *b, const char *s, size_t n) {
    if (b->len + n + 1 > b->cap) {
        size_t nc = b->cap ? b->cap : 256;
        while (nc < b->len + n + 1) nc *= 2;
        char *ns = (char *)realloc(b->s, nc);
        if (!ns) return;
        b->s = ns; b->cap = nc;
    }
    memcpy(b->s + b->len, s, n);
    b->len += n;
    b->s[b->len] = '\0';
}
static void sb_str(StrBuf *b, const char *s) { sb_put(b, s, strlen(s)); }

/* ---------- robustness helpers ---------- */

/* Free physical RAM in MB; -1 if unknown. Used to refuse new inference
 * when the machine is about to run out of memory (503 instead of a
 * hard-to-diagnose allocation crash mid-prefill).
 * On Linux read /proc/meminfo MemAvailable: sysconf(_SC_AVPHYS_PAGES)
 * only counts free pages and EXCLUDES reclaimable page cache, so on a
 * fully-loaded box (e.g. a 14 GB model on 16 GB RAM) it reports ~170 MB
 * while MemAvailable is ~1.7 GB -> spurious 503s for every request. */
static long mem_avail_mb(void) {
    /* Prefer MemAvailable from /proc/meminfo (free -m "available"). */
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        long long avail_kb = -1;
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "MemAvailable:", 13) == 0) {
                avail_kb = atoll(line + 13);
                break;
            }
        }
        fclose(f);
        if (avail_kb >= 0)
            return (long)(avail_kb / 1024);
    }
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long psz = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || psz <= 0) return -1;
    return (long)(((double)pages * psz) / (1024.0 * 1024.0));
}

/* Enter the inference critical section. Applies the admission policy:
 *   -1 -> reject with 429 (queue full)
 *   -2 -> reject with 503 (not enough free RAM)
 *    0 -> inference may proceed (queued + lock held; busy=1)
 * Rejection happens before taking the inference lock so a full/stressed
 * server answers fast instead of accumulating blocked workers. */
static int infer_gate(VLLMServerCtx *ctx) {
    if (ctx->min_free_mb > 0) {
        long avail = mem_avail_mb();
        if (avail >= 0 && avail < ctx->min_free_mb) {
            fprintf(stderr, "[SRV] 503: free RAM %ld MB < min_free_mb=%d MB "
                    "(refusing inference; lower --min-free-mb or use a smaller "
                    "model on this machine)\n", avail, ctx->min_free_mb);
            fflush(stderr);
            return -2;
        }
    }
    vhttp_mutex_lock(ctx->stat_lock);
    if (ctx->unload_pending) {      /* 模型卸载中：拒绝新推理（503） */
        vhttp_mutex_unlock(ctx->stat_lock);
        return -3;
    }
    if (ctx->max_queued > 0 && ctx->queued_requests >= ctx->max_queued) {
        vhttp_mutex_unlock(ctx->stat_lock);
        return -1;
    }
    ctx->total_requests++;
    ctx->active_requests++;
    ctx->queued_requests++;
    vhttp_mutex_unlock(ctx->stat_lock);

    vhttp_mutex_lock(ctx->inf_lock);   /* wait for the running request */
    vhttp_mutex_lock(ctx->stat_lock);
    ctx->queued_requests--;
    ctx->busy = 1;
    ctx->last_infer_s = (long)time(NULL);   /* 请求开始推理即刷新空闲计时 */
    vhttp_mutex_unlock(ctx->stat_lock);
    return 0;
}

static void infer_done(VLLMServerCtx *ctx) {
    vhttp_mutex_lock(ctx->stat_lock);
    ctx->busy = 0;
    ctx->active_requests--;
    ctx->last_infer_s = (long)time(NULL);
    vhttp_mutex_unlock(ctx->stat_lock);
    vhttp_mutex_unlock(ctx->inf_lock);
}

/* Batch-mode admission: identical 429/503 policy but WITHOUT blocking on the
 * inference lock (the batch scheduler owns engine serialization). */
static int infer_gate_batch(VLLMServerCtx *ctx) {
    if (ctx->min_free_mb > 0) {
        long avail = mem_avail_mb();
        if (avail >= 0 && avail < ctx->min_free_mb) {
            fprintf(stderr, "[SRV] 503: free RAM %ld MB < min_free_mb=%d MB "
                    "(refusing inference; lower --min-free-mb or use a smaller "
                    "model on this machine)\n", avail, ctx->min_free_mb);
            fflush(stderr);
            return -2;
        }
    }
    vhttp_mutex_lock(ctx->stat_lock);
    if (ctx->unload_pending) {      /* 模型卸载中：拒绝新推理（503） */
        vhttp_mutex_unlock(ctx->stat_lock);
        return -3;
    }
    if (ctx->max_queued > 0 && ctx->queued_requests >= ctx->max_queued) {
        vhttp_mutex_unlock(ctx->stat_lock);
        return -1;
    }
    ctx->total_requests++;
    ctx->active_requests++;
    ctx->queued_requests++;
    ctx->busy = 1;
    ctx->last_infer_s = (long)time(NULL);   /* 请求开始推理即刷新空闲计时 */
    vhttp_mutex_unlock(ctx->stat_lock);
    return 0;
}

static void infer_done_batch(VLLMServerCtx *ctx) {
    vhttp_mutex_lock(ctx->stat_lock);
    ctx->queued_requests--;
    if (ctx->queued_requests < 0) ctx->queued_requests = 0;
    ctx->active_requests--;
    ctx->busy = ctx->active_requests > 0;
    ctx->last_infer_s = (long)time(NULL);
    vhttp_mutex_unlock(ctx->stat_lock);
}

/* Lazy scheduler creation once the model is loaded (works for both
 * --auto-load and admin manual-load). No-op when batching is disabled.
 * Skipped while a model unload is in progress (the unload destroys the
 * scheduler; creating a fresh one against a freed ist would race). */
static void batch_ensure(VLLMServerCtx *ctx) {
    if (ctx->batch_max < 2 || ctx->batch || !ctx->ist) return;
    vhttp_mutex_lock(ctx->stat_lock);
    if (ctx->unload_pending) { vhttp_mutex_unlock(ctx->stat_lock); return; }
    if (!ctx->batch && ctx->ist) {
        ctx->batch = vllm_batch_create(&ctx->ist->weights, ctx->tok,
                                       ctx->model_id, ctx->batch_max);
        fprintf(stderr, "[BATCH] continuous batching on (batch_max=%d)\n",
                ctx->batch_max);
        fflush(stderr);
    }
    vhttp_mutex_unlock(ctx->stat_lock);
}

/* ---------- Qwen chat template ---------- */

/* Build the Qwen im_chat prompt from OpenAI messages.
 * content may be a plain string or a part array ([{type:"text",...}, ...]);
 * the chat client always sends arrays, so text parts are extracted here.
 * Media parts never reach this path (detect_media routes them to the
 * multimodal builder). Returns malloc'd NUL-terminated string. */
static char *build_chat_prompt(VLLMServerCtx *ctx, const VJson *messages) {
    StrBuf b; sb_init(&b);
    size_t n = vjson_array_len(messages);
    for (size_t i = 0; i < n; i++) {
        const VJson *m = vjson_array_get(messages, i);
        if (!m) continue;
        const char *role = vjson_str(vjson_obj_get(m, "role"));
        const VJson *content = vjson_obj_get(m, "content");
        if (!role || !content) continue;
        if (strcmp(role, "system") == 0) {
            sb_str(&b, "<|im_start|>system\n");
        } else if (strcmp(role, "user") == 0) {
            sb_str(&b, "<|im_start|>user\n");
        } else if (strcmp(role, "assistant") == 0) {
            sb_str(&b, "<|im_start|>assistant\n");
        } else {
            sb_str(&b, "<|im_start|>"); sb_str(&b, role); sb_str(&b, "\n");
        }
        if (content->type == VJ_STRING) {
            sb_str(&b, vjson_str(content));
        } else if (content->type == VJ_ARRAY) {
            size_t np = vjson_array_len(content);
            for (size_t p = 0; p < np; p++) {
                const VJson *part = vjson_array_get(content, p);
                if (!part) continue;
                const char *ptype = vjson_str(vjson_obj_get(part, "type"));
                if (ptype && strcmp(ptype, "text") == 0) {
                    const char *t = vjson_str(vjson_obj_get(part, "text"));
                    if (t) sb_str(&b, t);
                }
                /* image_url / video_frames parts are handled by the
                 * multimodal path (detect_media) and never reach here. */
            }
        } else {
            continue;
        }
        sb_str(&b, "<|im_end|>\n");
    }
    sb_str(&b, "<|im_start|>assistant\n");
    if (!b.s) b.s = strdup("");
    return b.s;
}

/* Legacy /v1/completions: wrap prompt in a plain user turn. */
static char *build_completion_prompt(VLLMServerCtx *ctx, const char *prompt) {
    (void)ctx;
    StrBuf b; sb_init(&b);
    sb_str(&b, "<|im_start|>user\n"); sb_str(&b, prompt); sb_str(&b, "<|im_end|>\n");
    sb_str(&b, "<|im_start|>assistant\n");
    if (!b.s) b.s = strdup("");
    return b.s;
}

/* ---------- response helpers ---------- */

static long unix_now(void) { return (long)time(NULL); }

/* Fill resp with a heap-allocated JSON error (thread-safe; the server
 * frees resp->body_owned after sending). */
static void json_error(VHttpResponse *resp, int status, const char *message) {
    char *body = (char *)malloc(512);
    if (body) {
        vhttp_json_error(message, body, 512);
        resp->body = body;
        resp->body_len = strlen(body);
        resp->body_owned = body;
    } else {
        static const char oom[] =
            "{\"error\":{\"message\":\"Out of memory\",\"type\":\"server_error\"}}";
        resp->body = oom;
        resp->body_len = sizeof(oom) - 1;
    }
    resp->status = status;
    resp->content_type = "application/json";
}

/* ---------- per-user L3 eviction isolation ----------
 *
 * --l3-evict + --l3-user-ttl: every user (the request JSON "user" field,
 * "anon" fallback) gets its own L3 cache file. The file is derived from the
 * base --l3-path by inserting "_u<hex8>" (FNV-1a hash of the user id, no
 * path chars) before the extension, e.g. <kv-dir>/kv_l3.bin ->
 * <kv-dir>/kv_l3_u1a2b3c4d.bin. A user whose last request is older than
 * --l3-user-ttl minutes is considered offline and its file is deleted on the
 * next request (lazy cleanup; st->l3 is closed first if still bound to it).
 * The session table is touched only while the inference lock is held
 * (infer_gate passed), so no locking is needed. */

/* FNV-1a 64-bit -> low 32 bits (deterministic, hex-safe). */
static unsigned long l3_user_hash(const char *s) {
    unsigned long long h = 1469598103934665603ull;
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211ull; }
    return (unsigned long)(h & 0xFFFFFFFFul);
}

/* Per-user L3 file path from the base path (g_l3_path or "kv_l3.bin"). */
static void l3_user_fname(const char *base, const char *user,
                          char *out, size_t cap) {
    const char *b = (base && base[0]) ? base : "kv_l3.bin";
    const char *slash = strrchr(b, '/');
    const char *dot = strrchr(b, '.');
    if (dot && (!slash || dot > slash)) {
        snprintf(out, cap, "%.*s_u%08lx%s", (int)(dot - b), b,
                 l3_user_hash(user), dot);
    } else {
        snprintf(out, cap, "%s_u%08lx", b, l3_user_hash(user));
    }
}

/* Update (or create) the user's session entry and set the current request's
 * L3 file. Falls back to the base file when the table cannot grow. */
static void l3_user_touch(VLLMServerCtx *ctx, const char *user) {
    long now = (long)unix_now();
    for (int i = 0; i < ctx->l3_users_n; i++) {
        if (strcmp(ctx->l3_users[i].user, user) == 0) {
            ctx->l3_users[i].last_active = now;
            snprintf(ctx->l3_cur_fname, sizeof(ctx->l3_cur_fname), "%s",
                     ctx->l3_users[i].fname);
            return;
        }
    }
    if (ctx->l3_users_n >= ctx->l3_users_cap) {
        int nc = ctx->l3_users_cap ? ctx->l3_users_cap * 2 : 8;
        L3UserEntry *ne = (L3UserEntry *)realloc(
            ctx->l3_users, (size_t)nc * sizeof(*ne));
        if (!ne) {
            l3_user_fname(g_l3_path, user, ctx->l3_cur_fname,
                          sizeof(ctx->l3_cur_fname));
            return;
        }
        ctx->l3_users = ne;
        ctx->l3_users_cap = nc;
    }
    L3UserEntry *e = &ctx->l3_users[ctx->l3_users_n++];
    memset(e, 0, sizeof(*e));
    snprintf(e->user, sizeof(e->user), "%s", user);
    l3_user_fname(g_l3_path, user, e->fname, sizeof(e->fname));
    e->last_active = now;
    snprintf(ctx->l3_cur_fname, sizeof(ctx->l3_cur_fname), "%s", e->fname);
}

/* Delete the L3 files of users idle longer than --l3-user-ttl minutes.
 * Runs with the inference lock held (single-threaded vs st->l3). */
static void l3_user_cleanup(VLLMServerCtx *ctx) {
    STQwenInferenceState *st = ctx->ist;
    long now = (long)unix_now();
    long ttl = g_l3_user_ttl * 60L;
    if (ttl <= 0 || ctx->l3_users_n == 0) return;
    for (int i = 0; i < ctx->l3_users_n; i++) {
        L3UserEntry *e = &ctx->l3_users[i];
        if (now - e->last_active < ttl) continue;
        if (st && st->l3.fp && strcmp(st->l3.path, e->fname) == 0) {
            fprintf(stderr, "[L3] user idle: close bound file %s\n", e->fname);
            l3_state_free(&st->l3);
        }
        if (unlink(e->fname) == 0)
            fprintf(stderr, "[L3] user '%s' offline: removed %s\n",
                    e->user, e->fname);
        else
            fprintf(stderr, "[L3] user '%s' offline: %s absent\n",
                    e->user, e->fname);
        ctx->l3_users[i] = ctx->l3_users[ctx->l3_users_n - 1];
        ctx->l3_users_n--;
        i--;
    }
}

/* Per-request entry point (called with the inference lock held): resolve the
 * user, reap offline files, set the current user's L3 file. */
static void l3_user_apply(VLLMServerCtx *ctx, const char *user) {
    if (!g_l3_evict) {
        ctx->l3_cur_fname[0] = '\0';
        g_l3_cur_path = NULL;
        return;
    }
    l3_user_cleanup(ctx);
    l3_user_touch(ctx, (user && user[0]) ? user : "anon");
    g_l3_cur_path = ctx->l3_cur_fname;
}

/* KV prefix reuse: minimum shared-prefix length (tokens) worth skipping. */
#define PREFIX_KV_MIN_LCP 16

/* Longest common prefix of two token sequences (token-exact, so a mismatch
 * anywhere safely degrades to a full prefill). */
static int kv_lcp(const int *a, int na, const int *b, int nb) {
    int n = na < nb ? na : nb;
    for (int i = 0; i < n; i++) if (a[i] != b[i]) return i;
    return n;
}

/* Reset inference state for a fresh request. keep_lcp > 0 asks to KEEP the
 * KV of the first keep_lcp tokens (the request's common prefix with the
 * previous one - either left in RAM by the previous request or restored from
 * a disk KV checkpoint by st_kv_disk_load) so prefill only computes the
 * remainder. The reuse is only honored when those KV rows are actually
 * present (cache_len[0] >= keep_lcp) and L3 eviction is off (L3 frees prefix
 * blocks to NULL, which the prefill pack path cannot read). */
static void ist_reset(VLLMServerCtx *ctx, int keep_lcp) {
    STQwenInferenceState *st = ctx->ist;
    int nl = ctx->w->n_layers_allocated;
    int reuse = (ctx->prefix_kv || ctx->disk_kv) &&
                keep_lcp >= PREFIX_KV_MIN_LCP && !g_l3_evict &&
                keep_lcp <= st->cache_len[0];
    if (reuse) {
        for (int l = 0; l < nl; l++) st->cache_len[l] = keep_lcp;
        st->seq_len = keep_lcp;
        st->mrope_pos = 0;   /* text-only; prefill_ex sets it for multimodal */
        fprintf(stderr, "[KV-PREFIX] reuse %d-token KV prefix, prefill rest\n",
                keep_lcp);
        fflush(stderr);
        return;
    }
    for (int l = 0; l < nl; l++) st->cache_len[l] = 0;
    st->seq_len = 0;
    st->mrope_pos = 0;   /* text-only default; prefill_ex sets it for multimodal */
    /* Phase-2 L3 eviction physically frees the cold KV blocks (pointers set
     * NULL). Re-back them so this request's prefill can write the new KV. */
    size_t rb = st_qwen_kv_rebuild_freed(st);
    if (rb > 0)
        fprintf(stderr, "[KV] rebuilt %.1f MB of L3-freed KV blocks\n",
                (double)rb / 1048576.0);
    /* Per-user L3 files: if st->l3 is still bound to another user's file
     * (previous request), close it so this request's eviction opens the
     * right file. Same-user requests reuse (overwrite) their own file. */
    if (st->l3.fp && ctx->l3_cur_fname[0] &&
        strcmp(st->l3.path, ctx->l3_cur_fname) != 0) {
        fprintf(stderr, "[L3] user switch: closed %s (now %s)\n",
                st->l3.path, ctx->l3_cur_fname);
        l3_state_free(&st->l3);
    }
}

/* ================================================================
 * Disk KV persistence (--disk-kv DIR)
 *
 * After each successful text request its F32 KV snapshot (token ids + K/V
 * rows, see st_kv_disk_save) is written to DIR/kv_<fnv64>.kv. On a later
 * request - including after a process restart - the checkpoint whose tokens
 * share the longest common prefix is loaded into the KV cache and only the
 * remainder is prefilled. F32 storage keeps the restore bit-exact with a
 * fresh prefill (same guarantee as the in-RAM prefix reuse); files whose
 * model geometry does not match the loaded model are skipped.
 * ================================================================ */
#define DISKKV_MAX_TOKENS 8192  /* snapshot cap raised to max_seq window
                                 * (4K/8K conversations) so a process restart
                                 * can restore long-context KV. F32 snapshot
                                 * ~224 KB/token => up to ~1.8 GB per file. */
#define DISKKV_MAX_FILES  8     /* LRU cap on checkpoint files */
#define DISKKV_SEP '/'

static uint64_t dkv_hash(const int *tokens, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) {
        uint32_t x = (uint32_t)tokens[i];
        for (int b = 0; b < 4; b++) {
            h ^= (x & 0xFFu);
            h *= 1099511628211ULL;
            x >>= 8;
        }
    }
    return h;
}

/* mkdir -p: mkdir only creates one level. */
static void dkv_mkdirs(char *dir) {
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

/* Read one checkpoint header + tokens. Geometry must match the loaded model
 * or the file is ignored (stale model = invalid cache). Returns 0 on success
 * with e filled (e->tokens malloc'd). */
static int dkv_read_entry(VLLMServerCtx *ctx, const char *path, DiskKVEntry *e) {
    memset(e, 0, sizeof(*e));
    if (!ctx->ist || !ctx->cfg) return -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    uint8_t hdr[64];
    int rc = -1;
    if (fread(hdr, 1, 64, fp) == 64) {
        uint32_t magic, ver, nl, nkv, hd, bs, ftok, fdim;
        memcpy(&magic, hdr + 0, 4); memcpy(&ver, hdr + 4, 4);
        memcpy(&nl, hdr + 8, 4); memcpy(&nkv, hdr + 12, 4);
        memcpy(&hd, hdr + 16, 4); memcpy(&bs, hdr + 20, 4);
        memcpy(&ftok, hdr + 24, 4); memcpy(&fdim, hdr + 28, 4);
        int want_nl = ctx->ist->weights.n_layers_allocated;
        if (want_nl <= 0) want_nl = ctx->cfg->n_layers;
        if (magic == 0x564C4B56u && ver == 1 &&
            (int)nl == want_nl && (int)nkv == ctx->cfg->n_kv_heads &&
            (int)hd == ctx->cfg->head_dim && (int)bs == ctx->ist->kv_bs &&
            (int)fdim == ctx->cfg->n_kv_heads * ctx->cfg->head_dim &&
            (int)ftok >= PREFIX_KV_MIN_LCP && (int)ftok <= DISKKV_MAX_TOKENS) {
            int *toks = (int *)malloc((size_t)ftok * sizeof(int));
            if (toks && fread(toks, sizeof(int), (size_t)ftok, fp) == (size_t)ftok) {
                snprintf(e->path, sizeof(e->path), "%s", path);
                e->n_tokens = (int)ftok;
                e->tokens = toks;
                struct stat sb;
                if (stat(path, &sb) == 0) e->mtime = (long)sb.st_mtime;
                rc = 0;
            } else {
                free(toks);
            }
        }
    }
    fclose(fp);
    return rc;
}

static void dkv_push(VLLMServerCtx *ctx, const DiskKVEntry *e) {
    if (ctx->dkv_n >= ctx->dkv_cap) {
        int nc = ctx->dkv_cap ? ctx->dkv_cap * 2 : 8;
        DiskKVEntry *n = (DiskKVEntry *)realloc(ctx->dkv, (size_t)nc * sizeof(DiskKVEntry));
        if (!n) return;
        ctx->dkv = n;
        ctx->dkv_cap = nc;
    }
    ctx->dkv[ctx->dkv_n++] = *e;
}

static void dkv_free_list(VLLMServerCtx *ctx) {
    for (int i = 0; i < ctx->dkv_n; i++) free(ctx->dkv[i].tokens);
    free(ctx->dkv);
    ctx->dkv = NULL;
    ctx->dkv_n = 0;
    ctx->dkv_cap = 0;
}

void vllm_server_diskkv_scan(VLLMServerCtx *ctx) {
    dkv_free_list(ctx);
    if (!ctx->disk_kv || !ctx->kvdir[0]) return;
    DIR *d = opendir(ctx->kvdir);
    if (d) {
        struct dirent *de;
        while ((de = readdir(d)) != NULL) {
            size_t ln = strlen(de->d_name);
            if (ln < 4 || strcmp(de->d_name + ln - 3, ".kv") != 0) continue;
            char p[576];
            snprintf(p, sizeof(p), "%s/%s", ctx->kvdir, de->d_name);
            DiskKVEntry e;
            if (dkv_read_entry(ctx, p, &e) == 0) dkv_push(ctx, &e);
        }
        closedir(d);
    }
    fprintf(stderr, "[KV-DISK] index: %d checkpoint(s) in %s\n",
            ctx->dkv_n, ctx->kvdir[0] ? ctx->kvdir : "(none)");
    fflush(stderr);
}

/* Free the server-side session state tied to the loaded model: continuous
 * batching scheduler, the in-RAM prefix-KV sequence (last_ids) and the disk
 * KV index. Called on unload so no stale context (KV/prefix/batch) survives
 * a reload — the "上下文状态清零" half of the load-on-use lifecycle. */
void vllm_server_reset_session(VLLMServerCtx *ctx) {
    if (ctx->batch) {
        vllm_batch_destroy(ctx->batch);
        ctx->batch = NULL;
    }
    free(ctx->last_ids);
    ctx->last_ids = NULL;
    ctx->last_n = 0;
    ctx->last_cap = 0;
    dkv_free_list(ctx);
}

/* Longest shared prefix of the request tokens with any checkpoint; returns
 * the best LCP (>= PREFIX_KV_MIN_LCP) and copies its file path to `path`. */
static int dkv_best_lcp(VLLMServerCtx *ctx, const int *ids, int n_ids,
                        char *path, int path_cap) {
    int best = 0;
    const char *bp = NULL;
    for (int i = 0; i < ctx->dkv_n; i++) {
        const DiskKVEntry *e = &ctx->dkv[i];
        int l = kv_lcp(e->tokens, e->n_tokens, ids, n_ids);
        /* Skip checkpoints that fully cover the request (l == n_ids): keep
         * = n_ids would prefill 0 tokens, which cannot refresh logits and
         * decode would sample stale logits (predicts eos) -> empty response.
         * Prefer the longest proper prefix strictly below n_ids so the
         * caller's `l < n_ids` guard never starves the whole disk-kv path
         * (falling back to the full prefill). */
        if (l > best && l < n_ids) {
            best = l;
            bp = e->path;
        }
    }
    if (best >= PREFIX_KV_MIN_LCP && bp) {
        snprintf(path, path_cap, "%s", bp);
        return best;
    }
    return 0;
}

/* Snapshot the finished conversation (token ids + KV for those tokens) into
 * kv_<fnv64>.kv, enforce the LRU file cap, and refresh the index. */
static void dkv_save(VLLMServerCtx *ctx, const int *ids, int n_ids) {
    if (!ctx->disk_kv || !ctx->kvdir[0] || !ctx->ist) return;
    if (n_ids <= 0 || n_ids > DISKKV_MAX_TOKENS) return;
    if (n_ids > ctx->ist->cache_len[0]) return;   /* KV not computed yet */
    char dirbuf[512];
    snprintf(dirbuf, sizeof(dirbuf), "%s", ctx->kvdir);
    dkv_mkdirs(dirbuf);
    char path[576];
    snprintf(path, sizeof(path), "%s%c%s%016llx%s", ctx->kvdir, DISKKV_SEP,
             "kv_", (unsigned long long)dkv_hash(ids, n_ids), ".kv");
    if (st_kv_disk_save(ctx->ist, path, ids, n_ids) != 0) return;
    fprintf(stderr, "[KV-DISK] saved %d-token KV -> %s\n", n_ids, path);
    fflush(stderr);
    /* LRU: drop the oldest file (never the one just written) over the cap. */
    char del_path[576] = "";
    long oldest = 0;
    int cnt = 0;
    DIR *d = opendir(ctx->kvdir);
    if (d) {
        struct dirent *de;
        while ((de = readdir(d)) != NULL) {
            size_t ln = strlen(de->d_name);
            if (ln < 4 || strcmp(de->d_name + ln - 3, ".kv") != 0) continue;
            cnt++;
            char p2[576];
            snprintf(p2, sizeof(p2), "%s/%s", ctx->kvdir, de->d_name);
            if (strcmp(p2, path) != 0) {
                struct stat sb;
                if (stat(p2, &sb) == 0 && (oldest == 0 || sb.st_mtime < oldest)) {
                    oldest = (long)sb.st_mtime;
                    snprintf(del_path, sizeof(del_path), "%s", p2);
                }
            }
        }
        closedir(d);
    }
    if (cnt > DISKKV_MAX_FILES && del_path[0]) {
        remove(del_path);
        fprintf(stderr, "[KV-DISK] LRU evicted %s\n", del_path);
        fflush(stderr);
    }
    vllm_server_diskkv_scan(ctx);
}

/* ---------- /v1/models ---------- */

static void handle_models(VLLMServerCtx *ctx, VHttpResponse *resp) {
    char *body = (char *)malloc(2048);
    if (!body) { json_error(resp, 500, "Out of memory"); return; }
    VJson *root = vjson_new_object();
    vjson_obj_set(root, "object", vjson_new_string("list"));
    VJson *data = vjson_new_array();
    VJson *m = vjson_new_object();
    vjson_obj_set(m, "id", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
    vjson_obj_set(m, "object", vjson_new_string("model"));
    vjson_obj_set(m, "created", vjson_new_number((double)unix_now()));
    vjson_obj_set(m, "owned_by", vjson_new_string("vllm_kestrel"));
    vjson_array_push(data, m);
    vjson_obj_set(root, "data", data);
    size_t n = vjson_serialize(root, body, 2048);
    vjson_free(root);
    resp->status = 200;
    resp->content_type = "application/json";
    resp->body = body;
    resp->body_len = n;
    resp->body_owned = body;
}

/* ---------- inference core ---------- */

/* ================================================================
 * Incremental UTF-8 decoder.
 *
 * vocab.bin stores RAW bytes (bytes_to_unicode inverse). Fragment tokens
 * (mid-BPE emoji states) are NOT valid UTF-8 on their own 鈥?e.g. the 3
 * bytes F0 9F 98 are the first 3 bytes of a 4-byte emoji. Decoding each
 * token in isolation is what produced "冒艁暮" garbage. This decoder keeps an
 * incomplete multi-byte sequence in `pending` and splices it with the next
 * token's bytes, so fragments reassemble into a correct emoji.
 * ================================================================ */
typedef struct {
    unsigned char pending[4];
    int expect;   /* 0 = none; 2/3/4 = expected total sequence length */
    int npend;    /* bytes currently held in pending */
} Utf8Incr;

static void utf8_incr_init(Utf8Incr *d) {
    d->expect = 0;
    d->npend = 0;
}

/* Feed raw bytes; append decoded (complete) chars to out; returns out len. */
static int utf8_incr_feed(Utf8Incr *d, const unsigned char *buf, int len,
                          char *out, int out_cap) {
    int w = 0;
    for (int i = 0; i < len; i++) {
        unsigned char b = buf[i];
        if (d->expect == 0) {
            if (b < 0x80) {
                if (w < out_cap - 1) out[w++] = (char)b;
            } else if (b >= 0xC2 && b <= 0xDF) {
                d->expect = 2; d->pending[0] = b; d->npend = 1;
            } else if (b >= 0xE0 && b <= 0xEF) {
                d->expect = 3; d->pending[0] = b; d->npend = 1;
            } else if (b >= 0xF0 && b <= 0xF4) {
                d->expect = 4; d->pending[0] = b; d->npend = 1;
            } else {
                /* stray continuation byte or invalid lead -> U+FFFD */
                if (w < out_cap - 3) { out[w++] = (char)0xEF; out[w++] = (char)0xBF; out[w++] = (char)0xBD; }
            }
        } else {
            d->pending[d->npend++] = b;
            if (d->npend == d->expect) {
                int ok = 1;
                for (int k = 1; k < d->expect; k++)
                    if ((d->pending[k] & 0xC0) != 0x80) { ok = 0; break; }
                if (ok) {
                    for (int k = 0; k < d->expect; k++)
                        if (w < out_cap - 1) out[w++] = (char)d->pending[k];
                } else {
                    if (w < out_cap - 3) { out[w++] = (char)0xEF; out[w++] = (char)0xBF; out[w++] = (char)0xBD; }
                }
                d->expect = 0; d->npend = 0;
            }
        }
    }
    out[w] = '\0';
    return w;
}

/* Replace byte-level BPE markers 臓(U+0120, C4 A0) -> space and 膴(U+010A,
 * C4 8A) -> newline inside an already-decoded UTF-8 string. A piece may
 * contain several markers (e.g. "臓膴" from multi-token BPE output). */
static const char *gj_replace(const char *s, char *tmp, size_t tmplen) {
    if (!s || !s[0]) return s;
    const unsigned char *u = (const unsigned char *)s;
    int len = (int)strlen(s);
    int has_marker = 0;
    for (int i = 0; i + 1 < len; i++)
        if (u[i] == 0xC4 && u[i + 1] >= 0x80) {
            has_marker = 1;
            break;
        }
    if (!has_marker) return s;
    int w = 0;
    for (int i = 0; i < len && w < (int)tmplen - 1; i++) {
        if (u[i] == 0xC4 && i + 1 < len && u[i + 1] >= 0x80) {
            unsigned char b = (unsigned char)(u[i + 1] - 0x80);
            /* byte-level BPE 伪字符 U+0100+ -> 真实字节。\n(0x0A)、空格(0x20)
             * 与可打印/多字节字节还原为真实字符（修复含英文/数字/标点的回复
             * 在客户端被丢弃显示的问题）；其他控制字符(0x00-0x1F 除 \n)保留
             * 伪字符形式由客户端兜底丢弃，避免裸控制字符破坏 JSON。 */
            if (b == 0x0A || b == 0x20 || b >= 0x21) { tmp[w++] = (char)b; i++; continue; }
            i++;
            continue;
        }
        tmp[w++] = s[i];
    }
    tmp[w] = '\0';
    return tmp;
}

/* Shared autoregressive decode loop (sample + forward + optional SSE).
 * Collects text into out (must be non-NULL). When spec is enabled the loop
 * first tries an n-gram speculative decode round (--spec): a draft of K
 * tokens is generated from an earlier repetition of the current context
 * suffix and verified in ONE batched prefill (weights read once for all K
 * positions); accepted drafts are emitted together and only the rejected /
 * bonus token is forwarded. Greedy-only, text-only, KV prefix/L3 disabled. */

/* The current request's full token sequence = prompt_ids + gids. */
static int spec_seq_at(const int *prompt, int n_prompt,
                       const int *gids, int gn, int i) {
    return i < n_prompt ? prompt[i] : gids[i - n_prompt];
}

/* n-gram draft: find an earlier occurrence of the current context's tail
 * (window size longest-first, 8 down to 3 tokens) and take up to `k` tokens
 * that followed that occurrence. Longer windows win (more reliable
 * continuation); exact-repetition output (refrains, fixed-form lists, code
 * templates) drafts fully, while counter/varies-text patterns systematically
 * diverge and degrade to normal decode. Returns the draft length. */
static int spec_build_draft(const int *prompt, int n_prompt,
                            const int *gids, int gn, int k, int *draft) {
    int len = n_prompt + gn;
    if (len < 6 || k < 1 || k > SPEC_DRAFT_MAX) return 0;
    for (int w = 8; w >= 3; w--) {
        if (len < w + 2) continue;
        int lo = len - w;              /* window = [lo, len) */
        for (int s = 0; s + w + 1 <= lo; s++) {   /* candidate window start */
            int match = 1;
            for (int j = 0; j < w; j++) {
                if (spec_seq_at(prompt, n_prompt, gids, gn, s + j) !=
                    spec_seq_at(prompt, n_prompt, gids, gn, lo + j)) { match = 0; break; }
            }
            if (!match) continue;
            int avail = len - (s + w);
            int n = avail < k ? avail : k;
            for (int i = 0; i < n; i++)
                draft[i] = spec_seq_at(prompt, n_prompt, gids, gn, s + w + i);
            return n;
        }
    }
    return 0;
}

/* Emit one generated token: record id, decode to text, append to the output
 * (SSE chunk or buffer), update gen / first-token time. Returns 1 if the
 * token is an end token (stop decoding), -1 if the streaming client went
 * away (abort), 0 on success. */
static int emit_token(VLLMServerCtx *ctx, QwenTokenizer *tok, Utf8Incr *dec,
                      int id, int stream, VHttpConn *conn, StrBuf *out,
                      VLLMMetrics *met, int *gen, double *t_first,
                      const char *chunk_id, long created, const char *model,
                      char *dbuf, int dbuf_cap, int *gids, int gcap, int *gn) {
    if (id == tok->eos_id || id == tok->im_end_id) return 1;
    if (gids && *gn < gcap) gids[(*gn)++] = id;
    const char *raw = qwen_tokenizer_decode(tok, id);
    int rawlen = tok->str_lens[id];
    char piece[2048];
    int plen = utf8_incr_feed(dec, (const unsigned char *)raw, rawlen,
                              piece, (int)sizeof(piece));
    char pbuf[2048];
    const char *text = gj_replace(piece, pbuf, sizeof(pbuf));
    if (plen > 0 && text && text[0]) {
        if (stream) {
            VJson *root = vjson_new_object();
            vjson_obj_set(root, "id", vjson_new_string(chunk_id));
            vjson_obj_set(root, "object", vjson_new_string("chat.completion.chunk"));
            vjson_obj_set(root, "created", vjson_new_number((double)created));
            vjson_obj_set(root, "model", vjson_new_string(model));
            VJson *choices = vjson_new_array();
            VJson *ch = vjson_new_object();
            vjson_obj_set(ch, "index", vjson_new_number(0));
            VJson *delta = vjson_new_object();
            vjson_obj_set(delta, "content", vjson_new_string(text));
            vjson_obj_set(ch, "delta", delta);
            vjson_obj_set(ch, "finish_reason", vjson_new_null());
            vjson_array_push(choices, ch);
            vjson_obj_set(root, "choices", choices);
            size_t n = vjson_serialize(root, dbuf, (size_t)dbuf_cap);
            vjson_free(root);
            char line[4300];
            int ln = snprintf(line, sizeof(line), "data: %s\n\n", dbuf);
            if (ln > 0 && vhttp_stream_write(conn, line, (size_t)ln) < 0)
                return -1;
            /* Stream 模式也累计输出文本：流式出证需要全量 text。 */
            if (out) sb_str(out, text);
        } else {
            sb_str(out, text);
        }
    }
    (*gen)++;
    if (met && *gen == 1) *t_first = st_now_sec();
    return 0;
}

static int decode_loop(VLLMServerCtx *ctx, double temperature, double top_p,
                       double min_p, int max_tokens, int stream,
                       VHttpConn *conn, StrBuf *out,
                       VLLMMetrics *met, int **gen_ids_out, int *gen_n_out,
                       const int *prompt_ids, int n_prompt,
                       const char *att_user, const char *att_bodyhex) {
    STQwenInferenceState *st = ctx->ist;
    QwenTokenizer *tok = ctx->tok;
    int vc = ctx->cfg->vocab_size;

    /* Optional collector of the generated token ids (for KV prefix reuse:
     * they extend the previous request's KV sequence). */
    int *gids = NULL;
    int gn = 0, gcap = 0;
    if (gen_ids_out) {
        *gen_ids_out = NULL;
        *gen_n_out = 0;
        if (max_tokens > 0) {
            gids = (int *)malloc((size_t)max_tokens * sizeof(int));
            if (gids) gcap = max_tokens;
        }
    }

    /* SSE chunk id (thread-safe: workers may stream concurrently). */
    vhttp_mutex_lock(ctx->stat_lock);
    long my_id = ctx->next_chunk_id++;
    vhttp_mutex_unlock(ctx->stat_lock);
    char chunk_id[64];
    snprintf(chunk_id, sizeof(chunk_id), "chatcmpl-%08lx", my_id);

    long created = unix_now();
    const char *model = ctx->model_id ? ctx->model_id : "qwen3-vl-8b";
    char dbuf[4096];
    int done = 0;
    const char *finish = "stop";
    double t_first = 0.0;
    int gen = 0;
    Utf8Incr dec;
    utf8_incr_init(&dec);

    /* Speculative decode setup (--spec). Greedy-only (temp<=0/top_p>=1/
     * min_p<=0), text-only (prompt ids known), and off under L3 eviction or
     * continuous batching (they share the same inference state). */
    int spec_on = ctx->spec && prompt_ids && n_prompt > 0 &&
                  temperature <= 0.0 && top_p >= 1.0 && min_p <= 0.0 &&
                  !g_l3_evict && !g_sparse_attn && ctx->batch_max < 2;
    int spec_k = ctx->spec_k;
    if (spec_k < 1) spec_k = 1;
    if (spec_k > SPEC_DRAFT_MAX) spec_k = SPEC_DRAFT_MAX;
    float *vlog = NULL;
    if (spec_on) {
        /* +1 extra batch slot: slot K holds the pre-verify context logits
         * backup for the lossless decode-path replay (see spec block). */
        vlog = (float *)malloc((size_t)(SPEC_DRAFT_MAX + 1) * vc * sizeof(float));
        if (!vlog) spec_on = 0;
    }
    int spec_tries = 0, spec_hits = 0, spec_kv_rolls = 0;

    while (gen < max_tokens) {
        int id = -1;
        int emit_n = 0;         /* draft tokens to emit this round */
        int emit_extra = -1;    /* rejected/bonus token after the drafts */
        int draft[SPEC_DRAFT_MAX];
        int nl = st->weights.n_layers_allocated;

        if (spec_on && gn < gcap) {
            int K = spec_build_draft(prompt_ids, n_prompt, gids, gn, spec_k, draft);
            if (K > 0) {
                spec_tries++;
                /* Draft starts at the current KV length (multimodal prompts
                 * include visual tokens, so cache_len may exceed the text
                 * token count n_prompt + gn). */
                int L = st->cache_len[0];
                /* Context logits predict the next token; save the greedy pick
                 * before the verify clobbers st->logits. */
                int q_top = 0;
                float qb = -1.0e30f;
                for (int t = 0; t < vc; t++)
                    if (st->logits[t] > qb) { qb = st->logits[t]; q_top = t; }
                /* Backup the pre-verify context logits (decode path) BEFORE
                 * the batched verify clobbers st->logits; the lossless replay
                 * below restores it to re-derive the draft picks. */
                float *saved_logits = vlog + (size_t)K * vc;
                memcpy(saved_logits, st->logits, (size_t)vc * sizeof(float));
                for (int i = 0; i < K; i++) g_verify_draft[i] = draft[i];
                g_verify_invalid = 0;
                /* Verify positions the draft at the decode position base
                 * (mrope_pos for multimodal, seq_len for text-only). */
                int gv_pos = st->mrope_pos > 0 ? st->mrope_pos : st->seq_len;
                g_verify_pos = gv_pos;
                g_verify_logits = vlog;
                int pf = st_qwen_model_prefill_batch(st, draft, K);
                g_verify_logits = NULL;
                g_verify_pos = -1;
                /* Acceptance probe: draft[0] must match the context pick,
                 * then each position's prediction must match the next draft
                 * token. This ONLY probes the upper bound - the batched
                 * verify runs a different numeric path than single-token
                 * decode (fused-batched GEMM vs single-row GEMM), so its KV
                 * and top-1 can drift from decode near stop boundaries
                 * (2026-08-31: repeat_same produced 90 vs 86 tokens). The
                 * accepted prefix is therefore replayed through the decode
                 * path below for a lossless, KV-exact confirmation. */
                int acc_probe = (pf == 0 && !g_verify_invalid &&
                                 q_top == draft[0]) ? 1 : 0;
                while (acc_probe < K &&
                       g_verify_pred[acc_probe - 1] == draft[acc_probe]) acc_probe++;
                /* Lossless replay: roll KV back to L, restore the context
                 * logits, then confirm each draft token through the decode
                 * path (st_qwen_model_forward). This keeps KV bit-exact with
                 * normal decoding; the batched verify above only caps how
                 * far the replay may go. */
                for (int l = 0; l < nl; l++) st->cache_len[l] = L;
                st->seq_len = L;
                if (st->mrope_pos > 0) st->mrope_pos = gv_pos;
                memcpy(st->logits, saved_logits, (size_t)vc * sizeof(float));
                int real_acc = 0;
                for (int i = 0; i < acc_probe; i++) {
                    int top = 0;
                    float tb = -1.0e30f;
                    for (int t = 0; t < vc; t++)
                        if (st->logits[t] > tb) { tb = st->logits[t]; top = t; }
                    if (top == tok->eos_id || top == tok->im_end_id) break;
                    if (top != draft[i]) break;
                    st_qwen_model_forward(st, draft[i]);
                    real_acc++;
                }
                if (getenv("VLLM_SPEC_DBG")) {
                    int q2 = 0;
                    float qv2 = -1.0e30f;
                    for (int t = 0; t < vc; t++)
                        if (t != q_top && st->logits[t] > qv2) { qv2 = st->logits[t]; q2 = t; }
                    fprintf(stderr, "[SPEC-DBG] L=%d q_top=%d(q2=%d,m=%.3f) K=%d draft=[",
                            L, q_top, q2, qb - qv2, K);
                    for (int i = 0; i < K; i++) fprintf(stderr, "%d%s", draft[i], i + 1 < K ? "," : "");
                    fprintf(stderr, "] gvp=[");
                    for (int i = 0; i < K; i++) fprintf(stderr, "%d(%.2f)%s", g_verify_pred[i],
                                                        g_verify_margin[i], i + 1 < K ? "," : "");
                    fprintf(stderr, "] probe=%d replay=%d\n", acc_probe, real_acc);
                    fflush(stderr);
                }
                if (real_acc >= 1) {
                    /* KV now holds the decode-path-accepted prefix (bit-exact
                     * with normal decode). The bonus token is the decode
                     * path's own prediction at L+real_acc. */
                    spec_kv_rolls += K - real_acc;
                    spec_hits += real_acc;
                    emit_n = real_acc;
                    emit_extra = 0;
                    float eb = -1.0e30f;
                    for (int t = 0; t < vc; t++)
                        if (st->logits[t] > eb) { eb = st->logits[t]; emit_extra = t; }
                } else {
                    /* First draft token mismatches the decode path: keep the
                     * context's own pick (KV already rolled back to L). */
                    spec_kv_rolls += K;
                    id = q_top;
                }
            }
        }

        if (emit_n > 0) {
            /* Emit the accepted draft tokens, then the rejected-position or
             * bonus token (all greedy-verified). */
            int stop = 0;
            int extra_emitted = 0;
            for (int i = 0; i < emit_n && gen < max_tokens && stop == 0; i++) {
                int r = emit_token(ctx, tok, &dec, draft[i], stream, conn, out,
                                   met, &gen, &t_first, chunk_id, created, model,
                                   dbuf, (int)sizeof(dbuf), gids, gcap, &gn);
                if (r != 0) stop = r;
            }
            if (emit_extra >= 0 && gen < max_tokens && stop == 0) {
                int r = emit_token(ctx, tok, &dec, emit_extra, stream, conn, out,
                                   met, &gen, &t_first, chunk_id, created, model,
                                   dbuf, (int)sizeof(dbuf), gids, gcap, &gn);
                if (r != 0) stop = r;
                else extra_emitted = 1;
            }
            if (stop < 0) { done = -1; break; }   /* client went away */
            if (stop == 1) { done = 1; break; }   /* eos/im_end */
            /* Restore the model state after the accepted prefix: the extra
             * token (rejected-position pick or bonus) is the real next token
             * and must be forwarded to rebuild KV + logits at L+acc (same
             * contract as a normal decode step: even the max_tokens-th token
             * is forwarded so KV stays consistent with the token history). */
            if (extra_emitted) st_qwen_model_forward(st, emit_extra);
        } else if (id >= 0) {
            /* Spec round fully rejected: emit the context pick and forward. */
            int r = emit_token(ctx, tok, &dec, id, stream, conn, out,
                               met, &gen, &t_first, chunk_id, created, model,
                               dbuf, (int)sizeof(dbuf), gids, gcap, &gn);
            if (r < 0) { done = -1; break; }
            if (r == 1) { done = 1; break; }
            st_qwen_model_forward(st, id);
        } else {
            /* Normal single-token decode step. */
            id = sample_token_p(st->logits, vc, (float)temperature,
                                (float)top_p, (float)min_p);
            if (id == tok->eos_id || id == tok->im_end_id) { finish = "stop"; done = 1; break; }
            if (gids && gn < gcap) gids[gn++] = id;
            const char *raw = qwen_tokenizer_decode(tok, id);
            int rawlen = tok->str_lens[id];
            char piece[2048];
            int plen = utf8_incr_feed(&dec, (const unsigned char *)raw, rawlen,
                                       piece, (int)sizeof(piece));
            char pbuf[2048];
            const char *text = gj_replace(piece, pbuf, sizeof(pbuf));
            if (plen > 0 && text && text[0]) {
                if (stream) {
                    VJson *root = vjson_new_object();
                    vjson_obj_set(root, "id", vjson_new_string(chunk_id));
                    vjson_obj_set(root, "object", vjson_new_string("chat.completion.chunk"));
                    vjson_obj_set(root, "created", vjson_new_number((double)created));
                    vjson_obj_set(root, "model", vjson_new_string(model));
                    VJson *choices = vjson_new_array();
                    VJson *ch = vjson_new_object();
                    vjson_obj_set(ch, "index", vjson_new_number(0));
                    VJson *delta = vjson_new_object();
                    vjson_obj_set(delta, "content", vjson_new_string(text));
                    vjson_obj_set(ch, "delta", delta);
                    vjson_obj_set(ch, "finish_reason", vjson_new_null());
                    vjson_array_push(choices, ch);
                    vjson_obj_set(root, "choices", choices);
                    size_t n = vjson_serialize(root, dbuf, sizeof(dbuf));
                    vjson_free(root);
                    char line[4300];
                    int ln = snprintf(line, sizeof(line), "data: %s\n\n", dbuf);
                    /* Client went away: stop decoding now so the inference
                     * lock is released for the next queued request. */
                    if (ln > 0 && vhttp_stream_write(conn, line, (size_t)ln) < 0) {
                        done = -1;
                        break;
                    }
                    /* Stream 累计输出文本（出证需要全量 text）。 */
                    if (out) sb_str(out, text);
                } else {
                    sb_str(out, text);
                }
            }
            gen++;
            if (met && gen == 1) t_first = st_now_sec();   /* first token produced */
            st_qwen_model_forward(st, id);
        }
    }
    if (done != 1) finish = "length";
    if (spec_on && spec_tries > 0) {
        fprintf(stderr, "[SPEC] tries=%d hits=%d kv_rolls=%d (%.0f%% draft hit)\n",
                spec_tries, spec_hits, spec_kv_rolls,
                spec_tries ? 100.0 * (double)spec_hits / (double)(spec_tries * spec_k) : 0.0);
        fflush(stderr);
    }
    free(vlog);

    if (met) {
        double t_end = st_now_sec();
        met->n_tokens = gen;
        if (t_first > 0) met->ttft_ms = (t_first - met->start_s) * 1000.0;
        met->total_ms = (t_end - met->start_s) * 1000.0;
        if (gen > 1) met->tpot_ms = (t_end - t_first) / (double)(gen - 1) * 1000.0;
        else if (t_first > 0) met->tpot_ms = met->ttft_ms;
    }

    if (stream && done != -1) {
        VJson *root = vjson_new_object();
        vjson_obj_set(root, "id", vjson_new_string(chunk_id));
        vjson_obj_set(root, "object", vjson_new_string("chat.completion.chunk"));
        vjson_obj_set(root, "created", vjson_new_number((double)created));
        vjson_obj_set(root, "model", vjson_new_string(model));
        VJson *choices = vjson_new_array();
        VJson *ch = vjson_new_object();
        vjson_obj_set(ch, "index", vjson_new_number(0));
        vjson_obj_set(ch, "delta", vjson_new_object());
        vjson_obj_set(ch, "finish_reason", vjson_new_string(finish));
        vjson_array_push(choices, ch);
        vjson_obj_set(root, "choices", choices);
        size_t n = vjson_serialize(root, dbuf, sizeof(dbuf));
        vjson_free(root);
        char line[4300];
        int ln = snprintf(line, sizeof(line), "data: %s\n\n", dbuf);
        if (ln > 0) vhttp_stream_write(conn, line, (size_t)ln);
        /* Final event: per-request metrics for the chat client. */
        if (met) {
            char mline[512];
            double tok_s = met->total_ms > 0.0
                           ? (double)met->n_tokens * 1000.0 / met->total_ms : 0.0;
            snprintf(mline, sizeof(mline),
                     "data: {\"object\":\"chat.completion.metrics\",\"metrics\":"
                     "{\"prompt_tokens\":%d,\"last_user_tokens\":%d,\"n_tokens\":%d,"
                     "\"ttft_ms\":%.1f,\"tpot_ms\":%.1f,\"total_ms\":%.1f,\"prefill_ms\":%.1f,\"tok_s\":%.2f}}\n\n",
                     met->prompt_tokens, met->last_user_tokens, met->n_tokens,
                     met->ttft_ms, met->tpot_ms, met->total_ms, met->prefill_ms, tok_s);
            vhttp_stream_write(conn, mline, (size_t)strlen(mline));
        }
        /* history_tokens 事件：完整 token 序列（prompt + generated），客户端
         * 保存供下轮 context_tokens 精确复用。 */
        {
            int total = n_prompt + gn;
            size_t cap = (size_t)total * 8 + 128;
            char *tbuf = (char *)malloc(cap);
            if (tbuf) {
                int off = 0;
                off += snprintf(tbuf + off, cap - (size_t)off,
                                "data: {\"object\":\"chat.completion.tokens\",\"tokens\":[");
                int idx = 0;
                for (int i = 0; i < n_prompt; i++) {
                    int w = snprintf(tbuf + off, cap - (size_t)off, "%s%d",
                                     idx ? "," : "", prompt_ids[i]);
                    if (w <= 0 || off + w >= (int)cap - 2) break;
                    off += w; idx++;
                }
                for (int i = 0; i < gn; i++) {
                    int w = snprintf(tbuf + off, cap - (size_t)off, "%s%d",
                                     idx ? "," : "", gids[i]);
                    if (w <= 0 || off + w >= (int)cap - 2) break;
                    off += w; idx++;
                }
                off += snprintf(tbuf + off, cap - (size_t)off, "]}\n\n");
                vhttp_stream_write(conn, tbuf, (size_t)off);
                free(tbuf);
            }
        }
        /* 方案 2：流式出证。对所有文本已累计（out）的流式响应附一个
         * chat.completion.attest 事件（含 SM2 签名凭证）。 */
        if (vatt_active() && out && out->s) {
            VAttestReq ar;
            memset(&ar, 0, sizeof(ar));
            ar.user = att_user;
            ar.body_sha = att_bodyhex;   /* SM3(原始请求体) 64-hex */
            ar.text = out->s;
            ar.text_len = out->len;
            ar.n_prompt_tokens = (met && met->prompt_tokens > 0)
                                     ? met->prompt_tokens : n_prompt;
            ar.n_gen_tokens = gen;
            ar.finish = finish;
            ar.temperature = temperature;
            ar.top_p = top_p;
            ar.min_p = min_p;
            ar.max_tokens = max_tokens;
            ar.ts = created;
            char *aj = vatt_seal_json(&ar);
            if (aj) {
                char aline[4500];
                int ln = snprintf(aline, sizeof(aline),
                                  "data: {\"object\":\"chat.completion.attest\","
                                  "\"attest\":%s}\n\n", aj);
                if (ln > 0 && ln < (int)sizeof(aline))
                    vhttp_stream_write(conn, aline, (size_t)ln);
                free(aj);
            }
        }
        vhttp_stream_done(conn);
    }
    if (gen_ids_out) { *gen_ids_out = gids; *gen_n_out = gn; }
    return 0;
}

/* Text-only completion: encode prompt, prefill, decode.
 * ctx_tokens/n_ctx: 客户端回传的完整历史 token 序列（上一响应 history_tokens）。
 *   若与上一请求序列完全一致，则只重新编码 tail_prompt（本轮新增 user 段）并
 *   prefill 增量，历史 KV 完整复用；任何不一致回退到完整 prompt 全量 prefill。
 * hist_ids/hist_n (非流式): 输出本轮完整序列（prompt+generated），供响应
 *   history_tokens 字段；流式由 decode_loop 内单独 data 事件发送。
 * Returns 0 on success, -1 on engine failure, -2 if the prompt exceeds
 * the KV cache span (client must get 400). */
static int run_completion(VLLMServerCtx *ctx, const char *prompt,
                          const char *tail_prompt,
                          const int *ctx_tokens, int n_ctx,
                          double temperature, double top_p, double min_p,
                          int max_tokens, int stream, VHttpConn *conn,
                          StrBuf *out, VLLMMetrics *met,
                          int **hist_ids, int *hist_n,
                          const char *att_user, const char *att_bodyhex) {
    STQwenInferenceState *st = ctx->ist;
    QwenTokenizer *tok = ctx->tok;

    if (met) met->start_s = st_now_sec();

    /* Cap prompt length to the KV cache span. ids is heap-allocated so a
     * huge prompt cannot overflow a stack array. */
    int max_ids = st->max_kv_slots;
    if (max_ids < 1024) max_ids = 1024;
    int *ids = (int *)malloc(((size_t)max_ids + 16) * sizeof(int));
    if (!ids) return -1;
    int n_ids = 0;

    /* KV prefix reuse: the shared prefix with the previous request is kept
     * in the KV cache and only the remainder is prefilled. Two sources:
     * 1) the in-RAM KV of the previous text request (ctx->last_ids);
     * 2) a disk KV checkpoint (--disk-kv) whose tokens share the longest
     *    prefix - this survives process restarts. L3 eviction (which frees
     *    prefix blocks) and multimodal prompts (visual KV) disable both.
     * context_tokens 是其上层的精确变体：客户端回传的序列与 last_ids 完全
     * 一致（LCP==n_ctx）时 KV 行真实存在，可直接 keep。 */
    int keep = 0;
    if (ctx_tokens && n_ctx > 0 && ctx->prefix_kv && !g_l3_evict &&
        tail_prompt && tail_prompt[0] && n_ctx < max_ids) {
        int matched = 0;
        /* 1) 进程内 last_ids 精确匹配（正常轮次复用）。 */
        if (ctx->last_n > 0 && n_ctx <= st->cache_len[0] &&
            kv_lcp(ctx->last_ids, ctx->last_n, ctx_tokens, n_ctx) == n_ctx)
            matched = 1;
        /* 2) 模型卸载/引擎重启后 last_ids 失效：disk_kv 精确恢复历史 KV，
         *    使多轮对话上下文跨卸载/重启延续（页面只需重发 context_tokens）。 */
        if (!matched && ctx->disk_kv && !g_l3_evict) {
            for (int i = 0; i < ctx->dkv_n; i++) {
                const DiskKVEntry *e = &ctx->dkv[i];
                if (e->n_tokens >= n_ctx &&
                    kv_lcp(e->tokens, e->n_tokens, ctx_tokens, n_ctx) == n_ctx) {
                    if (st_kv_disk_load(st, e->path, n_ctx) == 0) {
                        fprintf(stderr, "[KV-DISK] context restore %d-token KV prefix\n",
                                n_ctx);
                        fflush(stderr);
                        matched = 1;
                    }
                    break;
                }
            }
        }
        if (matched) {
            memcpy(ids, ctx_tokens, (size_t)n_ctx * sizeof(int));
            int tn = qwen_tokenizer_encode(tok, tail_prompt, ids + n_ctx,
                                           max_ids + 16 - n_ctx);
            if (tn > 0) {
                n_ids = n_ctx + tn;
                keep = n_ctx;
            }
        }
    }
    if (n_ids == 0) {
        n_ids = qwen_tokenizer_encode(tok, prompt, ids, max_ids + 16);
        if (n_ids <= 0) { free(ids); return -1; }
        if (n_ids > max_ids) { free(ids); return -2; }
        if (ctx->prefix_kv && !g_l3_evict && ctx->last_n > 0) {
            int l = kv_lcp(ctx->last_ids, ctx->last_n, ids, n_ids);
            /* 必须 l < n_ids: 若新 prompt 被上次 KV 完全覆盖(l == n_ids), prefill
             * rest=0 不会刷新 logits, decode 首步会采样到上次请求末尾的陈旧 logits
             * (预测 eos) → 立即停止 → 空响应。回退全量 prefill 保证正确。 */
            if (l >= PREFIX_KV_MIN_LCP && l < n_ids && l <= st->cache_len[0])
                keep = l;
        }
        if (keep == 0 && ctx->disk_kv && !g_l3_evict) {
            char dk_path[576];
            int l = dkv_best_lcp(ctx, ids, n_ids, dk_path, sizeof(dk_path));
            if (l >= PREFIX_KV_MIN_LCP && l < n_ids &&
                st_kv_disk_load(st, dk_path, l) == 0) {
                keep = l;
                fprintf(stderr, "[KV-DISK] loaded %d-token KV prefix from %s\n",
                        l, dk_path);
                fflush(stderr);
            }
        }
    }
    ist_reset(ctx, keep);
    if (st_qwen_model_prefill_batch(st, ids + keep, n_ids - keep) != 0) {
        free(ids); return -1;
    }
    if (keep == 0) {
        /* Phase 2a: L3 cold-block Q4 disk eviction (--l3-evict). */
        l3_evict_after_prefill(st);
    }
    if (met) {
        met->prompt_tokens = n_ids;
        met->prefill_ms = (st_now_sec() - met->start_s) * 1000.0;
    }

    int *gids = NULL; int gn = 0;
    int rc = decode_loop(ctx, temperature, top_p, min_p, max_tokens, stream,
                         conn, out, met, &gids, &gn, ids, n_ids,
                         att_user, att_bodyhex);
    if (rc == 0) {
        /* Remember this request's full KV token sequence (prompt + generated)
         * so the next request can reuse the shared prefix. */
        int total = n_ids + gn;
        if (total > ctx->last_cap) {
            int nc = ctx->last_cap ? ctx->last_cap * 2 : 512;
            while (nc < total) nc *= 2;
            int *nbuf = (int *)realloc(ctx->last_ids, (size_t)nc * sizeof(int));
            if (nbuf) { ctx->last_ids = nbuf; ctx->last_cap = nc; }
        }
        if (ctx->last_cap >= total) {
            memcpy(ctx->last_ids, ids, (size_t)n_ids * sizeof(int));
            memcpy(ctx->last_ids + n_ids, gids, (size_t)gn * sizeof(int));
            ctx->last_n = total;
            /* Disk KV persistence: snapshot the finished conversation (token
             * ids + KV rows) so a later process can restore this prefix. */
            if (ctx->disk_kv) dkv_save(ctx, ctx->last_ids, ctx->last_n);
        }
        /* 客户端历史 token 序列（prompt + generated）回传，供下轮 context_tokens
         * 精确复用（流式经 decode_loop 内 data 事件，非流式经响应字段）。 */
        if (hist_ids && hist_n) {
            int total = n_ids + gn;
            int *h = (int *)malloc((size_t)total * sizeof(int));
            if (h) {
                memcpy(h, ids, (size_t)n_ids * sizeof(int));
                if (gn > 0) memcpy(h + n_ids, gids, (size_t)gn * sizeof(int));
                *hist_ids = h;
                *hist_n = total;
            } else {
                *hist_ids = NULL;
                *hist_n = 0;
            }
        }
    } else if (rc != 0) {
        /* prefill 失败：KV 缓存状态未定义，历史序列失效，下轮回退全量 prefill。 */
        ctx->last_n = 0;
    }
    free(gids);
    free(ids);
    return rc;
}

/* Multimodal completion: prompt tokens + concatenated visual tokens already
 * built and length-checked by the route layer (build_multimodal_prompt). */
static int run_completion_mm(VLLMServerCtx *ctx,
                             const int *prompt_tokens, int n_prompt,
                             const float *vis_tokens, int n_vis,
                             const int *grids, int n_regions,
                             const float *const *ds_features, int n_ds,
                             double temperature, double top_p, double min_p,
                             int max_tokens, int stream, VHttpConn *conn,
                             StrBuf *out, VLLMMetrics *met,
                             const char *att_user, const char *att_bodyhex) {
    STQwenInferenceState *st = ctx->ist;
    if (met) met->start_s = st_now_sec();
    ist_reset(ctx, 0);
    if (st_qwen_model_multimodal_prefill_ex(st, prompt_tokens, n_prompt,
                                            vis_tokens, n_vis,
                                            grids, n_regions,
                                            ds_features, n_ds) != 0)
        return -1;
    /* Phase 2a: L3 cold-block Q4 disk eviction (--l3-evict). */
    l3_evict_after_prefill(st);
    if (met) {
        met->prompt_tokens = n_prompt + n_vis;
        met->prefill_ms = (st_now_sec() - met->start_s) * 1000.0;
    }
    /* Decode with speculative decoding: the multimodal prompt's TEXT token
     * sequence (visual tokens are positional, not part of the n-gram history)
     * plus the generated ids feed the draft builder; verification runs on the
     * decode stage where only text tokens are produced. */
    int *gids = NULL; int gn = 0;
    int rc = decode_loop(ctx, temperature, top_p, min_p, max_tokens, stream,
                         conn, out, met, &gids, &gn,
                         prompt_tokens, n_prompt,
                         att_user, att_bodyhex);
    free(gids);
    /* Multimodal prompts (visual KV, mrope positions) must never leak into
     * the text-only prefix-reuse history: drop the saved sequence. */
    ctx->last_n = 0;
    return rc;
}

/* ---------- multimodal (image/video) input ---------- */

#define MM_MAX_REGIONS 16
#define MM_MAX_FRAMES  32

/* DeepStack accumulation across media regions. rows[d] = [written[d], dim],
 * aligned with the concatenated visual-token rows in vis_all. */
typedef struct {
    float *rows[3];
    int    written[3];
    int    cap[3];
} DSAccum;

static void ds_accum_init(DSAccum *a) { memset(a, 0, sizeof(*a)); }
static void ds_accum_free(DSAccum *a) {
    for (int i = 0; i < 3; i++) free(a->rows[i]);
    memset(a, 0, sizeof(*a));
}
/* Append the vision state's DeepStack features (n rows of dim) for all
 * deepstack layers. Returns 0 ok, -1 on OOM. */
static int ds_accum_append(DSAccum *a, STVisionState *vis, int n, int dim) {
    for (int d = 0; d < vis->cfg->vis_ds_count; d++) {
        int need = a->written[d] + n;
        if (need > a->cap[d]) {
            int nc = a->cap[d] ? a->cap[d] : 1024;
            while (nc < need) nc *= 2;
            float *nb = (float *)realloc(a->rows[d], (size_t)nc * dim * sizeof(float));
            if (!nb) return -1;
            a->rows[d] = nb;
            a->cap[d] = nc;
        }
        memcpy(a->rows[d] + (size_t)a->written[d] * dim,
               vis->ds_features[d], (size_t)n * dim * sizeof(float));
        a->written[d] += n;
    }
    return 0;
}

/* ---- 图片视觉 token 缓存（重复图跳过 ViT 编码） ----
 * 键：原始图像字节的 128-bit FNV-1a（两次不同初值的 64-bit）。命中时回放
 * 与 st_vision_encode_image 完全一致的视觉 token/网格/DeepStack 行，输出与
 * 重新编码逐字节相同。容量 VVIS_CACHE_CAP 条目，每条 512 图 ≈ 17MB（vis
 * 4.2MB + ds 3×4.2MB），LRU 淘汰。请求已在 inf_lock 内串行，无需加锁。 */
#define VVIS_CACHE_CAP 4
typedef struct {
    unsigned long long h1, h2;   /* 128-bit FNV-1a 键 */
    long last_use;               /* LRU 时间戳 */
    int n;                       /* n_vis */
    int dim;                     /* vis_out_dim == LLM dim */
    float *vis;                  /* [n][dim] */
    int grid_t, grid_h, grid_w;  /* 合并后网格（region） */
    int n_ds;                    /* deepstack 行数 */
    float *ds[3];                /* [n][dim] × n_ds */
} VisCacheEntry;

static VisCacheEntry g_vis_cache[VVIS_CACHE_CAP];
static int g_vis_cache_n = 0;

/* FNV-1a 64-bit over bytes（见 l3_user_hash 同族）。 */
static unsigned long long fnv1a64_buf(const void *p, size_t n,
                                      unsigned long long h) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

/* 把任意 DeepStack 行追加进累积器（缓存回放用，与 ds_accum_append 逐字节一致）。 */
static int ds_accum_append_rows(DSAccum *a, float *const *rows, int n_ds,
                                int n, int dim) {
    for (int d = 0; d < n_ds; d++) {
        int need = a->written[d] + n;
        if (need > a->cap[d]) {
            int nc = a->cap[d] ? a->cap[d] : 1024;
            while (nc < need) nc *= 2;
            float *nb = (float *)realloc(a->rows[d], (size_t)nc * dim * sizeof(float));
            if (!nb) return -1;
            a->rows[d] = nb;
            a->cap[d] = nc;
        }
        memcpy(a->rows[d] + (size_t)a->written[d] * dim,
               rows[d], (size_t)n * dim * sizeof(float));
        a->written[d] += n;
    }
    return 0;
}

/* 查找并回放缓存；返回 1 命中（vis_all/grids/dsa 已填充）、0 未命中。 */
static int vis_cache_apply(VLLMServerCtx *ctx, unsigned long long h1,
                           unsigned long long h2,
                           float **vis_all, int *vis_total, int *vis_cap,
                           int *grids, int *n_regions, DSAccum *dsa,
                           int *out_n_pads) {
    const int d = ctx->cfg->dim;
    for (int ci = 0; ci < g_vis_cache_n; ci++) {
        VisCacheEntry *e = &g_vis_cache[ci];
        if (e->h1 != h1 || e->h2 != h2) continue;
        e->last_use = unix_now();
        if (e->dim != d || e->n < 1) return 0;   /* 维度不匹配：当作未命中 */
        if (*vis_total + e->n > *vis_cap) {
            int nc = *vis_cap ? *vis_cap : 4096;
            while (nc < *vis_total + e->n) nc *= 2;
            float *nb = (float *)realloc(*vis_all, (size_t)nc * d * sizeof(float));
            if (!nb) return 0;
            *vis_all = nb;
            *vis_cap = nc;
        }
        memcpy(*vis_all + (size_t)(*vis_total) * d, e->vis,
               (size_t)e->n * d * sizeof(float));
        *vis_total += e->n;
        grids[(*n_regions) * 3 + 0] = e->grid_t;
        grids[(*n_regions) * 3 + 1] = e->grid_h;
        grids[(*n_regions) * 3 + 2] = e->grid_w;
        (*n_regions)++;
        if (dsa && ds_accum_append_rows(dsa, e->ds, e->n_ds, e->n, d) != 0)
            return 0;
        *out_n_pads = e->n;
        return 1;
    }
    return 0;
}

/* 编码成功后写入缓存（LRU 淘汰最久未用）。 */
static void vis_cache_store(unsigned long long h1, unsigned long long h2,
                            STVisionState *vis, int n, int d,
                            int grid_t, int grid_h, int grid_w) {
    int slot;
    if (g_vis_cache_n < VVIS_CACHE_CAP) {
        slot = g_vis_cache_n;
    } else {
        long oldest = g_vis_cache[0].last_use;
        slot = 0;
        for (int ci = 1; ci < VVIS_CACHE_CAP; ci++)
            if (g_vis_cache[ci].last_use < oldest) {
                oldest = g_vis_cache[ci].last_use;
                slot = ci;
            }
    }
    VisCacheEntry *e = &g_vis_cache[slot];
    if (e->vis) free(e->vis);
    for (int k = 0; k < 3; k++) if (e->ds[k]) free(e->ds[k]);
    memset(e, 0, sizeof(*e));
    e->h1 = h1; e->h2 = h2;
    e->last_use = unix_now();
    e->n = n; e->dim = d;
    e->grid_t = grid_t; e->grid_h = grid_h; e->grid_w = grid_w;
    e->n_ds = vis->cfg->vis_ds_count;
    if (e->n_ds > 3) e->n_ds = 3;
    e->vis = (float *)malloc((size_t)n * d * sizeof(float));
    if (!e->vis) { e->n = 0; return; }
    memcpy(e->vis, st_vision_get_tokens(vis), (size_t)n * d * sizeof(float));
    for (int k = 0; k < e->n_ds; k++) {
        e->ds[k] = (float *)malloc((size_t)n * d * sizeof(float));
        if (e->ds[k])
            memcpy(e->ds[k], vis->ds_features[k], (size_t)n * d * sizeof(float));
    }
    if (g_vis_cache_n < VVIS_CACHE_CAP) g_vis_cache_n++;
}

/* True if any message content carries image_url / video_frames parts. */
static int detect_media(const VJson *messages) {
    size_t n = vjson_array_len(messages);
    for (size_t m = 0; m < n; m++) {
        const VJson *msg = vjson_array_get(messages, m);
        if (!msg) continue;
        const VJson *content = vjson_obj_get(msg, "content");
        if (!content || content->type != VJ_ARRAY) continue;
        size_t np = vjson_array_len(content);
        for (size_t p = 0; p < np; p++) {
            const VJson *part = vjson_array_get(content, p);
            if (!part) continue;
            const char *ptype = vjson_str(vjson_obj_get(part, "type"));
            if (ptype && (strcmp(ptype, "image_url") == 0 ||
                          strcmp(ptype, "video_frames") == 0 ||
                          strcmp(ptype, "vision_tokens") == 0))
                return 1;
        }
    }
    return 0;
}

/* Decode a "data:<mime>;base64,...." URL into raw bytes (malloc'd). */
static uint8_t *decode_data_url(const char *url, size_t *out_len) {
    const char *p = strstr(url, "base64,");
    if (!p) return NULL;
    p += 7;   /* skip "base64," */
    return media_base64_decode(p, out_len);
}

/* Encode one media part (image_url / video_frames) with the vision encoder,
 * append its visual tokens to *vis_all and its grid to grids[].
 * out_kind: 1=image, 2=video; out_n_pads: visual tokens emitted. */
static int encode_media_item(VLLMServerCtx *ctx, const VJson *part,
                             float **vis_all, int *vis_total, int *vis_cap,
                             int *grids, int *n_regions,
                             int *out_kind, int *out_n_pads, DSAccum *dsa) {
    STVisionState *vis = ctx->vis;
    const int d = ctx->cfg->dim;

    /* 客户端预处理模式（vision_tokens part）：浏览器在本机算好的视觉特征，
     * 直接接收（跳过 ViT 编码）。协议 v1：tokens_b64 = base64(fp32 LE, n*d)，
     * ds_b64（可选）= base64 数组，每条 n*d；grid = [gt, gh, gw] 合并后网格。 */
    const VJson *vt = vjson_obj_get(part, "vision_tokens");
    if (vt) {
        const VJson *g = vjson_obj_get(vt, "grid");
        if (!g || g->type != VJ_ARRAY || vjson_array_len(g) < 3) return -1;
        long gt = (long)vjson_array_get(g, 0)->u.number;
        long gh = (long)vjson_array_get(g, 1)->u.number;
        long gw = (long)vjson_array_get(g, 2)->u.number;
        if (gt < 1 || gh < 1 || gw < 1 || gt * gh * gw > 4096) return -1;
        long n = gt * gh * gw;
        const char *tb64 = vjson_str(vjson_obj_get(vt, "tokens_b64"));
        if (!tb64) return -1;
        size_t tlen = 0;
        uint8_t *traw = media_base64_decode(tb64, &tlen);
        if (!traw) return -1;
        size_t need = (size_t)n * d * sizeof(float);
        if (tlen != need) { free(traw); return -1; }
        if (*n_regions >= MM_MAX_REGIONS) { free(traw); return -1; }
        if (*vis_total + n > *vis_cap) {
            int nc = *vis_cap ? *vis_cap : 4096;
            while (nc < *vis_total + n) nc *= 2;
            float *nb = (float *)realloc(*vis_all, (size_t)nc * d * sizeof(float));
            if (!nb) { free(traw); return -1; }
            *vis_all = nb;
            *vis_cap = nc;
        }
        memcpy(*vis_all + (size_t)(*vis_total) * d, traw, need);
        *vis_total += n;
        free(traw);
        grids[(*n_regions) * 3 + 0] = (int)gt;
        grids[(*n_regions) * 3 + 1] = (int)gh;
        grids[(*n_regions) * 3 + 2] = (int)gw;
        (*n_regions)++;
        /* DeepStack 特征（可选）：ds_b64 数组，每条 n*d 个 fp32。缺失则 n_ds=0
         * （prefill 跳过注入，输出与带 ds 路径不一致，客户端须自行负责）。 */
        float *drows[3] = { NULL, NULL, NULL };
        int n_ds = 0;
        const VJson *ds = vjson_obj_get(vt, "ds_b64");
        if (ds && ds->type == VJ_ARRAY) {
            size_t nds = vjson_array_len(ds);
            if (nds > 3) nds = 3;
            int bad = 0;
            for (size_t k = 0; k < nds; k++) {
                const char *db64 = vjson_str(vjson_array_get(ds, k));
                if (!db64) { bad = 1; break; }
                size_t dlen = 0;
                uint8_t *draw = media_base64_decode(db64, &dlen);
                if (!draw || dlen != need) { free(draw); bad = 1; break; }
                drows[k] = (float *)draw;
            }
            if (bad) {
                for (int k = 0; k < 3; k++) free(drows[k]);
                return -1;
            }
            n_ds = (int)nds;
        }
        if (dsa && n_ds > 0) {
            if (ds_accum_append_rows(dsa, drows, n_ds, (int)n, d) != 0) {
                for (int k = 0; k < n_ds; k++) free(drows[k]);
                return -1;
            }
        }
        for (int k = 0; k < n_ds; k++) free(drows[k]);
        fprintf(stderr, "[VIS-TOKENS] client-preprocess n=%ld d=%d ds=%d grid=[%ld,%ld,%ld]\n",
                n, d, n_ds, gt, gh, gw);
        fflush(stderr);
        *out_kind = 1;
        *out_n_pads = (int)n;
        return 0;
    }

    const VJson *iu = vjson_obj_get(part, "image_url");
    if (iu) {
        const char *url = vjson_str(vjson_obj_get(iu, "url"));
        if (!url) return -1;
        size_t blen = 0;
        uint8_t *raw = decode_data_url(url, &blen);
        if (!raw) return -1;
        /* 128-bit FNV-1a 图片键：相同上传字节命中缓存，跳过 ViT 编码。 */
        unsigned long long h1 = fnv1a64_buf(raw, blen, 1469598103934665603ull);
        unsigned long long h2 = fnv1a64_buf(raw, blen,
                                            1469598103934665603ull ^ 0x9E3779B97F4A7C15ull);
        int w = 0, h = 0;
        uint8_t *rgb = media_load_image_memory(raw, blen, &w, &h);
        free(raw);
        if (!rgb) return -1;
        if (vis_cache_apply(ctx, h1, h2, vis_all, vis_total, vis_cap,
                            grids, n_regions, dsa, out_n_pads)) {
            fprintf(stderr, "[VIS-CACHE] hit (%dx%d, %d vis tokens, skip ViT)\n",
                    w, h, *out_n_pads);
            fflush(stderr);
            free(rgb);
            *out_kind = 1;
            return 0;
        }
        int n = st_vision_encode_image(vis, rgb, w, h);
        free(rgb);
        if (n < 0) return -1;
        if (*n_regions >= MM_MAX_REGIONS) return -1;
        int gt = 1;
        int gh = vis->grid_h / ctx->cfg->vis_merge;
        int gw = vis->grid_w / ctx->cfg->vis_merge;
        grids[(*n_regions) * 3 + 0] = gt;
        grids[(*n_regions) * 3 + 1] = gh;
        grids[(*n_regions) * 3 + 2] = gw;
        (*n_regions)++;
        if (*vis_total + n > *vis_cap) {
            int nc = *vis_cap ? *vis_cap : 4096;
            while (nc < *vis_total + n) nc *= 2;
            float *nb = (float *)realloc(*vis_all, (size_t)nc * d * sizeof(float));
            if (!nb) return -1;
            *vis_all = nb;
            *vis_cap = nc;
        }
        memcpy(*vis_all + (size_t)(*vis_total) * d, st_vision_get_tokens(vis),
               (size_t)n * d * sizeof(float));
        *vis_total += n;
        if (dsa && ds_accum_append(dsa, vis, n, d) != 0) return -1;
        vis_cache_store(h1, h2, vis, n, d, gt, gh, gw);
        *out_kind = 1;
        *out_n_pads = n;
        return 0;
    }

    const VJson *vf = vjson_obj_get(part, "video_frames");
    if (vf && vf->type == VJ_ARRAY) {
        size_t nf = vjson_array_len(vf);
        if (nf == 0) return -1;
        if (nf > MM_MAX_FRAMES) nf = MM_MAX_FRAMES;
        uint8_t **frames = (uint8_t **)calloc(nf, sizeof(uint8_t *));
        if (!frames) return -1;
        int fw = 0, fh = 0, loaded = 0;
        for (size_t i = 0; i < nf; i++) {
            const char *url = vjson_str(vjson_array_get(vf, i));
            if (!url) continue;
            size_t blen = 0;
            uint8_t *raw = decode_data_url(url, &blen);
            if (!raw) continue;
            int w = 0, h = 0;
            uint8_t *rgb = media_load_image_memory(raw, blen, &w, &h);
            free(raw);
            if (!rgb) continue;
            if (loaded == 0) { fw = w; fh = h; }
            else if (w != fw || h != fh) { free(rgb); continue; }
            frames[loaded++] = rgb;
        }
        if (loaded == 0) { free(frames); return -1; }
        int n = st_vision_encode_video(vis, (const uint8_t **)frames, fw, fh, loaded);
        for (int i = 0; i < loaded; i++) free(frames[i]);
        free(frames);
        if (n < 0) return -1;
        if (*n_regions >= MM_MAX_REGIONS) return -1;
        grids[(*n_regions) * 3 + 0] = vis->n_frames_eff;
        grids[(*n_regions) * 3 + 1] = vis->grid_h / ctx->cfg->vis_merge;
        grids[(*n_regions) * 3 + 2] = vis->grid_w / ctx->cfg->vis_merge;
        (*n_regions)++;
        if (*vis_total + n > *vis_cap) {
            int nc = *vis_cap ? *vis_cap : 4096;
            while (nc < *vis_total + n) nc *= 2;
            float *nb = (float *)realloc(*vis_all, (size_t)nc * d * sizeof(float));
            if (!nb) return -1;
            *vis_all = nb;
            *vis_cap = nc;
        }
        memcpy(*vis_all + (size_t)(*vis_total) * d, st_vision_get_tokens(vis),
               (size_t)n * d * sizeof(float));
        *vis_total += n;
        if (dsa && ds_accum_append(dsa, vis, n, d) != 0) return -1;
        *out_kind = 2;
        *out_n_pads = n;
        return 0;
    }
    return -1;
}

/* Append helpers (bounded). Return -1 if the prompt buffer is full. */
static int app_tok(int *buf, int cap, int *n, int tid) {
    if (*n >= cap) return -1;
    buf[(*n)++] = tid;
    return 0;
}
static int app_text_tok(int *buf, int cap, int *n, QwenTokenizer *tok, const char *text) {
    if (!text || !*text) return 0;
    int used = qwen_tokenizer_encode(tok, text, buf + *n, cap - *n);
    if (used <= 0) return -1;
    *n += used;
    return 0;
}

/* Build the multimodal chat prompt (im_chat template) and encode all media.
 * prompt_tokens/max_tok: output buffer (must hold the whole sequence).
 * vis_all/vis_total: concatenated visual tokens (realloc'd as needed).
 * grids/n_regions: per-region {grid_t, grid_h, grid_w}.
 * Returns 0 ok, -1 media/decode failure, -2 prompt exceeds max_tok. */
static int build_multimodal_prompt(VLLMServerCtx *ctx, const VJson *messages,
                                   int *prompt_tokens, int max_tok, int *pn,
                                   float **vis_all, int *vis_total, int *vis_cap,
                                   int *grids, int *n_regions, DSAccum *dsa) {
    QwenTokenizer *tok = ctx->tok;
    STModelConfig *cfg = ctx->cfg;
    size_t n = vjson_array_len(messages);

    for (size_t m = 0; m < n; m++) {
        const VJson *msg = vjson_array_get(messages, m);
        if (!msg) continue;
        const char *role = vjson_str(vjson_obj_get(msg, "role"));
        const VJson *content = vjson_obj_get(msg, "content");
        if (!content) continue;
        if (!role) role = "user";

        if (app_text_tok(prompt_tokens, max_tok, pn, tok, "<|im_start|>") < 0) return -2;
        if (app_text_tok(prompt_tokens, max_tok, pn, tok, role) < 0) return -2;
        if (app_text_tok(prompt_tokens, max_tok, pn, tok, "\n") < 0) return -2;

        if (content->type == VJ_STRING) {
            if (app_text_tok(prompt_tokens, max_tok, pn, tok, vjson_str(content)) < 0) return -2;
        } else if (content->type == VJ_ARRAY) {
            size_t np = vjson_array_len(content);
            for (size_t p = 0; p < np; p++) {
                const VJson *part = vjson_array_get(content, p);
                if (!part) continue;
                const char *ptype = vjson_str(vjson_obj_get(part, "type"));
                if (ptype && strcmp(ptype, "text") == 0) {
                    if (app_text_tok(prompt_tokens, max_tok, pn, tok,
                                     vjson_str(vjson_obj_get(part, "text"))) < 0) return -2;
                } else if (ptype && (strcmp(ptype, "image_url") == 0 ||
                                     strcmp(ptype, "video_frames") == 0 ||
                                     strcmp(ptype, "vision_tokens") == 0)) {
                    int kind = 0, npads = 0;
                    if (encode_media_item(ctx, part, vis_all, vis_total, vis_cap,
                                          grids, n_regions, &kind, &npads, dsa) != 0)
                        return -1;
                    /* <|vision_start|> <|image_pad|>/<|video_pad|> xN <|vision_end|> */
                    if (app_tok(prompt_tokens, max_tok, pn, cfg->vision_start_id) < 0) return -2;
                    int pad_id = (kind == 2) ? cfg->video_token_id : cfg->image_token_id;
                    for (int i = 0; i < npads; i++)
                        if (app_tok(prompt_tokens, max_tok, pn, pad_id) < 0) return -2;
                    if (app_tok(prompt_tokens, max_tok, pn, cfg->vision_end_id) < 0) return -2;
                } else {
                    /* Unknown part type: ignore. */
                }
            }
        }
        if (app_text_tok(prompt_tokens, max_tok, pn, tok, "<|im_end|>\n") < 0) return -2;
    }
    if (app_text_tok(prompt_tokens, max_tok, pn, tok, "<|im_start|>assistant\n") < 0) return -2;
    return 0;
}

/* Token count of the LAST user message's text content (the question just
 * asked). Used for the "涓婁竴闂暱搴? metric. Media parts are not counted
 * here (visual tokens are folded into prompt_tokens). */
static int last_user_text_tokens(VLLMServerCtx *ctx, const VJson *messages) {
    QwenTokenizer *tok = ctx->tok;
    size_t n = vjson_array_len(messages);
    for (size_t m = n; m-- > 0; ) {
        const VJson *msg = vjson_array_get(messages, m);
        if (!msg) continue;
        const char *role = vjson_str(vjson_obj_get(msg, "role"));
        if (!role || strcmp(role, "user") != 0) continue;
        const VJson *content = vjson_obj_get(msg, "content");
        if (!content) return 0;

        StrBuf sb; sb_init(&sb);
        if (content->type == VJ_STRING) {
            const char *t = vjson_str(content);
            if (t) sb_str(&sb, t);
        } else if (content->type == VJ_ARRAY) {
            size_t np = vjson_array_len(content);
            for (size_t p = 0; p < np; p++) {
                const VJson *part = vjson_array_get(content, p);
                if (!part) continue;
                const char *ptype = vjson_str(vjson_obj_get(part, "type"));
                if (ptype && strcmp(ptype, "text") == 0) {
                    const char *t = vjson_str(vjson_obj_get(part, "text"));
                    if (t) sb_str(&sb, t);
                }
            }
        }
        int cnt = 0;
        if (sb.s && sb.s[0]) {
            int cap = ctx->ist->max_kv_slots;
            if (cap < 1024) cap = 1024;
            int *ids = (int *)malloc(((size_t)cap + 16) * sizeof(int));
            if (ids) {
                cnt = qwen_tokenizer_encode(tok, sb.s, ids, cap + 16);
                free(ids);
            }
        }
        sb_free(&sb);
        return cnt;
    }
    return 0;
}

/* 返回最后一条 user 消息的文本（malloc，调用方 free）；无则 NULL。
 * 供 context_tokens 模式拼接本轮新增的 user 模板段。 */
static char *last_user_text_str(const VJson *messages) {
    size_t n = vjson_array_len(messages);
    for (size_t m = n; m-- > 0; ) {
        const VJson *msg = vjson_array_get(messages, m);
        if (!msg) continue;
        const char *role = vjson_str(vjson_obj_get(msg, "role"));
        if (!role || strcmp(role, "user") != 0) continue;
        const VJson *content = vjson_obj_get(msg, "content");
        if (!content) return NULL;
        StrBuf sb; sb_init(&sb);
        if (content->type == VJ_STRING) {
            const char *t = vjson_str(content);
            if (t) sb_str(&sb, t);
        } else if (content->type == VJ_ARRAY) {
            size_t np = vjson_array_len(content);
            for (size_t p = 0; p < np; p++) {
                const VJson *part = vjson_array_get(content, p);
                if (!part) continue;
                const char *ptype = vjson_str(vjson_obj_get(part, "type"));
                if (ptype && strcmp(ptype, "text") == 0) {
                    const char *t = vjson_str(vjson_obj_get(part, "text"));
                    if (t) sb_str(&sb, t);
                }
            }
        }
        char *r = sb.s;   /* transfer ownership to caller */
        return r;
    }
    return NULL;
}

/* ---------- /v1/chat/completions ---------- */

/* Ensure a model is loaded before serving inference. With --auto-load the
 * first request triggers a load-on-use (用时加载) and waits for it; without
 * it an unloaded model returns 503 (use /admin/api/model/load). */
static int ensure_model_ready(VLLMServerCtx *ctx, VHttpResponse *resp) {
    if (ctx->load_state == 2) return 0;
    if (ctx->auto_load && ctx->load_state != 1) {
        if (vllm_serve_load_on_use(ctx) != 0) {
            json_error(resp, 503, "model load failed (see engine log)");
            return -1;
        }
    }
    if (ctx->load_state != 2) {
        json_error(resp, 503, ctx->load_state == 1
                   ? "model is loading" : "model not loaded (use /admin/api/model/load)");
        return -1;
    }
    return 0;
}

/* 方案 2：给非流式响应对象附加 "attest" 出证字段。
 * body_sha = SM3(原始请求体) 的 64-hex（schema=2，绑定"收到什么证明什么"）；
 * text 为输出文本（绑定进摘要）。params 为本次请求实际采样参数。 */
static void attest_attach(VJson *root, const char *user, const char *body_sha,
                          const char *text, int npt, int ngt,
                          const char *finish, double temperature, double top_p,
                          double min_p, int max_tokens) {
    if (!vatt_active() || !root || !text) return;
    VAttestReq ar;
    memset(&ar, 0, sizeof(ar));
    ar.user = user;
    ar.body_sha = body_sha;
    ar.text = text;
    ar.text_len = strlen(text);
    ar.n_prompt_tokens = npt;
    ar.n_gen_tokens = ngt;
    ar.finish = finish;
    ar.temperature = temperature;
    ar.top_p = top_p;
    ar.min_p = min_p;
    ar.max_tokens = max_tokens;
    ar.ts = unix_now();
    char *aj = vatt_seal_json(&ar);
    if (aj) {
        VJson *av = vjson_parse(aj);
        if (av) {
            vjson_obj_set(root, "attest", av);
        }
        free(aj);
    }
}

static void handle_chat(VLLMServerCtx *ctx, const VHttpRequest *req,
                        VHttpResponse *resp, VHttpConn *conn) {
    /* Inference endpoints are unavailable until the model is loaded
     * (manual-load mode; see /admin/api/model/load). */
    if (ensure_model_ready(ctx, resp) != 0) return;
    VJson *body = vjson_parse(req->body);
    if (!body) {
        fprintf(stderr, "[SRV] JSON parse failed, body[0..128]: %.*s\n",
                (int)(req->body_len < 128 ? req->body_len : 128), req->body);
        json_error(resp, 400, "Invalid JSON body");
        return;
    }

    const VJson *messages = vjson_obj_get(body, "messages");
    if (!messages || messages->type != VJ_ARRAY || vjson_array_len(messages) == 0) {
        json_error(resp, 400, "'messages' array is required");
        vjson_free(body);
        return;
    }

    /* 方案 2：原始请求体 SM3（逐字节绑定，出证/验证均基于它）。 */
    char bodyhash[65];
    vatt_body_digest(req->body, req->body_len, bodyhash);

    double temperature = 0.7;
    double top_p = 0.9;
    double min_p = ctx->min_p;
    int max_tokens = 64;
    int stream = 0;
    VJson *jv;
    if ((jv = vjson_obj_get(body, "temperature")) && jv->type == VJ_NUMBER)
        temperature = jv->u.number;
    if ((jv = vjson_obj_get(body, "top_p")) && jv->type == VJ_NUMBER)
        top_p = jv->u.number;
    if ((jv = vjson_obj_get(body, "min_p")) && jv->type == VJ_NUMBER)
        min_p = jv->u.number;
    if ((jv = vjson_obj_get(body, "max_tokens")) && jv->type == VJ_NUMBER) {
        max_tokens = (int)jv->u.number;
        if (max_tokens <= 0) max_tokens = 1;
        if (max_tokens > 4096) max_tokens = 4096;
    }
    if ((jv = vjson_obj_get(body, "stream")) && jv->type == VJ_BOOL)
        stream = jv->u.boolean;
    const char *user = vjson_str(vjson_obj_get(body, "user"));

    /* context_mode (per-request context budget):
     *   "full"  (default) : whole conversation history every turn
     *   "last"  / "none"  : keep system + the final user turn only, so every
     *                       prefill stays short and TTFT does not grow across
     *                       a multi-turn chat
     *   "lastN" (e.g. "last10") : keep system + the last N messages (a
     *                       sliding window of ~N/2 turns)
     * The clipped array is a deep copy owned by `body`, so the existing
     * vjson_free(body) paths stay correct. */
    if ((jv = vjson_obj_get(body, "context_mode")) && jv->type == VJ_STRING) {
        const char *cm = vjson_str(jv);
        size_t keep = 0;   /* 0 = full history */
        if (strcmp(cm, "last") == 0 || strcmp(cm, "none") == 0) {
            keep = 1;
        } else if (strncmp(cm, "last", 4) == 0) {
            long n = atol(cm + 4);
            if (n > 0) keep = (size_t)n;
        }
        if (keep > 0) {
            VJson *clipped = vjson_new_array();
            size_t nm = vjson_array_len(messages);
            /* system (if any) always survives at the front. */
            for (size_t i = 0; i < nm; i++) {
                const VJson *m = vjson_array_get(messages, i);
                const char *role = vjson_str(vjson_obj_get(m, "role"));
                if (role && strcmp(role, "system") == 0) {
                    vjson_array_push(clipped, vjson_clone(m));
                    break;
                }
            }
            size_t start = (nm >= keep) ? (nm - keep) : 0;
            for (size_t i = start; i < nm; i++) {
                const VJson *m = vjson_array_get(messages, i);
                const char *role = vjson_str(vjson_obj_get(m, "role"));
                if (role && strcmp(role, "system") == 0) continue; /* at front */
                vjson_array_push(clipped, vjson_clone(m));
            }
            vjson_obj_set(body, "messages", clipped);
            messages = vjson_obj_get(body, "messages");
        }
    }

    int max_ids = ctx->ist->max_kv_slots;
    if (max_ids < 1024) max_ids = 1024;

    /* "涓婁竴闂撮暱搴? metric: token count of the last user message (text). */
    int last_user_tokens = last_user_text_tokens(ctx, messages);

    /* Continuous batching mode (--batch-max >= 2): concurrent requests step
     * together via the batch scheduler instead of serializing on inf_lock. */
    const int batch_mode = (ctx->batch_max >= 2);
    if (batch_mode) batch_ensure(ctx);

    /* context_tokens：客户端回传的完整历史 token 序列（上轮响应 history_tokens
     * 提供）。仅串行文本路径支持；batch 模式与多模态请求忽略（各自回退全量）。 */
    const VJson *ct = vjson_obj_get(body, "context_tokens");
    int *ctx_tokens = NULL; int n_ctx = 0;
    char *tail_prompt = NULL;
    if (ct && ct->type == VJ_ARRAY && !batch_mode && !detect_media(messages)) {
        size_t cn = vjson_array_len(ct);
        if (cn > 0 && cn <= (size_t)max_ids) {
            ctx_tokens = (int *)malloc(cn * sizeof(int));
            if (ctx_tokens) {
                int bad = 0;
                for (size_t i = 0; i < cn; i++) {
                    const VJson *e = vjson_array_get(ct, i);
                    if (!e || e->type != VJ_NUMBER) { bad = 1; break; }
                    ctx_tokens[i] = (int)e->u.number;
                }
                if (!bad) n_ctx = (int)cn;
                else { free(ctx_tokens); ctx_tokens = NULL; }
            }
        }
        if (n_ctx > 0) {
            char *lt = last_user_text_str(messages);
            if (lt && lt[0]) {
                size_t tl = strlen(lt) + 64;
                tail_prompt = (char *)malloc(tl);
                if (tail_prompt)
                    snprintf(tail_prompt, tl,
                             "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
                             lt);
            }
            free(lt);
        }
    }

    /* ---- multimodal path (image_url / video_frames) ---- */
    if (detect_media(messages)) {
        if (!ctx->vis || !ctx->cfg->has_vision) {
            json_error(resp, 400, "Vision encoder not loaded on this server");
            vjson_free(body);
            return;
        }
        int *pt = (int *)malloc(((size_t)max_ids + 16) * sizeof(int));
        if (!pt) {
            json_error(resp, 500, "Out of memory");
            vjson_free(body);
            return;
        }
        /* Vision encoding + multimodal prefill share per-request state and
         * the single inference state: hold the inference lock throughout. */
        int g = batch_mode ? infer_gate_batch(ctx) : infer_gate(ctx);
        if (g == -1) {
            json_error(resp, 429, "Too many queued requests, retry later");
            free(pt);
            vjson_free(body);
            return;
        }
        if (g == -2) {
            json_error(resp, 503, "Insufficient memory, retry later");
            free(pt);
            vjson_free(body);
            return;
        }
        if (g == -3) {
            json_error(resp, 503, "Model is unloading, retry later");
            free(pt);
            vjson_free(body);
            return;
        }

        int pn = 0;
        float *vis_all = NULL;
        int vis_total = 0, vis_cap = 0;
        int grids[MM_MAX_REGIONS * 3];
        int n_regions = 0;
        DSAccum dsa; ds_accum_init(&dsa);
        /* Vision encoding (st_vision_encode_*) uses the engine thread pool:
         * serialize it against batch forwards in batch mode (the serial path
         * relies on inf_lock for the same). */
        int rc;
        if (batch_mode) vllm_batch_engine_lock(ctx->batch);
        rc = build_multimodal_prompt(ctx, messages, pt, max_ids + 8, &pn,
                                     &vis_all, &vis_total, &vis_cap,
                                     grids, &n_regions, &dsa);
        if (batch_mode) vllm_batch_engine_unlock(ctx->batch);

        if (batch_mode) {
            /* ---- continuous-batching multimodal path ---- */
            VLLMMetrics met; memset(&met, 0, sizeof(met));
            met.last_user_tokens = last_user_tokens;
            VBatchReq r; memset(&r, 0, sizeof(r));
            r.is_mm = 1;
            r.pt = pt; r.pn = pn;
            r.vis_all = vis_all; r.vis_total = vis_total;
            r.grids = grids; r.n_regions = n_regions;
            const float *ds_rows[3] = { NULL, NULL, NULL };
            int n_ds = ctx->vis->cfg->vis_ds_count;
            for (int i = 0; i < n_ds && i < 3; i++) ds_rows[i] = dsa.rows[i];
            const float *const *ds_arg = (dsa.written[0] > 0) ? ds_rows : NULL;
            r.ds_features = ds_arg; r.n_ds = n_ds;
            r.temperature = temperature;
            r.top_p = top_p;
            r.min_p = min_p;
            r.max_tokens = max_tokens;
            r.stream = stream;
            r.conn = conn;
            r.met = &met;
            /* 流式也累积输出文本：batch 流式出证需要全量 text（vb 末尾 SSE
             * attest 事件前读取）。非流式本就写入 out_txt。 */
            char *out_txt = NULL; int out_len = 0;
            r.out_text = &out_txt;
            r.out_len = &out_len;
            r.body_sha = bodyhash;
            int brc = 0;
            if (rc != 0 || pn > max_ids) {
                brc = rc == -2 ? -2 : -1;
            } else if (stream) {
                resp->status = 200;
                resp->content_type = "text/event-stream";
                resp->stream = 1;
                vhttp_stream_begin(conn, 200, "text/event-stream");
                static const char role_preamble[] =
                    "data: {\"id\":\"chatcmpl\",\"object\":\"chat.completion.chunk\",\"choices\":"
                    "[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n";
                vhttp_stream_write(conn, role_preamble, sizeof(role_preamble) - 1);
                vllm_batch_run(ctx->batch, &r);
            } else {
                brc = vllm_batch_run(ctx->batch, &r);
                if (brc == -2) {
                    json_error(resp, 400, "Prompt exceeds the context limit");
                } else if (brc != 0) {
                    json_error(resp, 500, "Inference failed");
                } else {
                    char *body = (char *)malloc(1 << 20);
                    if (!body) {
                        json_error(resp, 500, "Out of memory");
                    } else {
                        VJson *root = vjson_new_object();
                        vjson_obj_set(root, "id", vjson_new_string("chatcmpl-00000000"));
                        vjson_obj_set(root, "object", vjson_new_string("chat.completion"));
                        vjson_obj_set(root, "created", vjson_new_number((double)unix_now()));
                        vjson_obj_set(root, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
                        VJson *choices = vjson_new_array();
                        VJson *ch = vjson_new_object();
                        vjson_obj_set(ch, "index", vjson_new_number(0));
                        VJson *msg = vjson_new_object();
                        vjson_obj_set(msg, "role", vjson_new_string("assistant"));
                        vjson_obj_set(msg, "content", vjson_new_string(out_txt ? out_txt : ""));
                        vjson_obj_set(ch, "message", msg);
                        vjson_obj_set(ch, "finish_reason", vjson_new_string("stop"));
                        vjson_array_push(choices, ch);
                        vjson_obj_set(root, "choices", choices);
                        vjson_obj_set(root, "usage", vjson_new_object());
                        VJson *mj = vjson_new_object();
                        vjson_obj_set(mj, "prompt_tokens", vjson_new_number((double)met.prompt_tokens));
                        vjson_obj_set(mj, "last_user_tokens", vjson_new_number((double)met.last_user_tokens));
                        vjson_obj_set(mj, "n_tokens", vjson_new_number((double)met.n_tokens));
                        vjson_obj_set(mj, "ttft_ms", vjson_new_number(met.ttft_ms));
                        vjson_obj_set(mj, "tpot_ms", vjson_new_number(met.tpot_ms));
                        vjson_obj_set(mj, "total_ms", vjson_new_number(met.total_ms));
                        vjson_obj_set(mj, "prefill_ms", vjson_new_number(met.prefill_ms));
                        vjson_obj_set(mj, "tok_s", vjson_new_number(
                            met.total_ms > 0.0 ? (double)met.n_tokens * 1000.0 / met.total_ms : 0.0));
                        vjson_obj_set(root, "metrics", mj);
                        /* 方案 2：多模态批处理非流式出证。 */
                        attest_attach(root, user, bodyhash, out_txt ? out_txt : "",
                                      met.prompt_tokens, met.n_tokens, "stop",
                                      temperature, top_p, min_p, max_tokens);
                        size_t n = vjson_serialize(root, body, 1 << 20);
                        vjson_free(root);
                        resp->status = 200;
                        resp->content_type = "application/json";
                        resp->body = body;
                        resp->body_len = n;
                        resp->body_owned = body;
                    }
                }
            }
            free(out_txt);
            ds_accum_free(&dsa);
            infer_done_batch(ctx);
            free(pt);
            free(vis_all);
            vjson_free(body);
            return;
        }

        if (rc != 0 || pn > max_ids) {
            json_error(resp, 400,
                       rc == -2 ? "Prompt exceeds the context limit"
                                : "Media decode/encode failed (base64 data URLs required)");
            ds_accum_free(&dsa);
            infer_done(ctx);
        } else {
            /* Serial path only: per-user L3 isolation (batch mode skips L3). */
            l3_user_apply(ctx, user);
            /* DeepStack features 鈫?[n_ds][vis_total, dim] (NULL if no media) */
            const float *ds_rows[3] = { NULL, NULL, NULL };
            int n_ds = ctx->vis->cfg->vis_ds_count;
            for (int i = 0; i < n_ds && i < 3; i++) ds_rows[i] = dsa.rows[i];
            const float *const *ds_arg = (dsa.written[0] > 0) ? ds_rows : NULL;
            if (stream) {
                resp->status = 200;
                resp->content_type = "text/event-stream";
                resp->stream = 1;
                vhttp_stream_begin(conn, 200, "text/event-stream");
                static const char role_preamble[] =
                    "data: {\"id\":\"chatcmpl\",\"object\":\"chat.completion.chunk\",\"choices\":"
                    "[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n";
                vhttp_stream_write(conn, role_preamble, sizeof(role_preamble) - 1);
                StrBuf sink; sb_init(&sink);
                VLLMMetrics met; memset(&met, 0, sizeof(met));
                met.last_user_tokens = last_user_tokens;
                run_completion_mm(ctx, pt, pn, vis_all, vis_total, grids, n_regions,
                                  ds_arg, n_ds, temperature, top_p, min_p,
                                  max_tokens, 1, conn, &sink, &met, user, bodyhash);
                sb_free(&sink);
                ds_accum_free(&dsa);
                infer_done(ctx);
            } else {
                StrBuf out; sb_init(&out);
                VLLMMetrics met; memset(&met, 0, sizeof(met));
                met.last_user_tokens = last_user_tokens;
                int rc2 = run_completion_mm(ctx, pt, pn, vis_all, vis_total, grids,
                                            n_regions, ds_arg, n_ds, temperature,
                                            top_p, min_p, max_tokens, 0, NULL,
                                            &out, &met, user, bodyhash);
                ds_accum_free(&dsa);
                infer_done(ctx);
                if (rc2 != 0) {
                    json_error(resp, 500, "Inference failed");
                } else {
                    char *body = (char *)malloc(1 << 20);
                    if (!body) {
                        json_error(resp, 500, "Out of memory");
                    } else {
                        VJson *root = vjson_new_object();
                        vjson_obj_set(root, "id", vjson_new_string("chatcmpl-00000000"));
                        vjson_obj_set(root, "object", vjson_new_string("chat.completion"));
                        vjson_obj_set(root, "created", vjson_new_number((double)unix_now()));
                        vjson_obj_set(root, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
                        VJson *choices = vjson_new_array();
                        VJson *ch = vjson_new_object();
                        vjson_obj_set(ch, "index", vjson_new_number(0));
                        VJson *msg = vjson_new_object();
                        vjson_obj_set(msg, "role", vjson_new_string("assistant"));
                        vjson_obj_set(msg, "content", vjson_new_string(out.s ? out.s : ""));
                        vjson_obj_set(ch, "message", msg);
                        vjson_obj_set(ch, "finish_reason", vjson_new_string("stop"));
                        vjson_array_push(choices, ch);
                        vjson_obj_set(root, "choices", choices);
                        vjson_obj_set(root, "usage", vjson_new_object());
                        /* Per-request inference metrics. */
                        VJson *mj = vjson_new_object();
                        vjson_obj_set(mj, "prompt_tokens", vjson_new_number((double)met.prompt_tokens));
                        vjson_obj_set(mj, "last_user_tokens", vjson_new_number((double)met.last_user_tokens));
                        vjson_obj_set(mj, "n_tokens", vjson_new_number((double)met.n_tokens));
                        vjson_obj_set(mj, "ttft_ms", vjson_new_number(met.ttft_ms));
                        vjson_obj_set(mj, "tpot_ms", vjson_new_number(met.tpot_ms));
                        vjson_obj_set(mj, "total_ms", vjson_new_number(met.total_ms));
                        vjson_obj_set(mj, "prefill_ms", vjson_new_number(met.prefill_ms));
                        vjson_obj_set(mj, "tok_s", vjson_new_number(
                            met.total_ms > 0.0 ? (double)met.n_tokens * 1000.0 / met.total_ms : 0.0));
                        vjson_obj_set(root, "metrics", mj);
                        /* 方案 2：多模态串行非流式出证。 */
                         attest_attach(root, user, bodyhash, out.s ? out.s : "",
                                      met.prompt_tokens, met.n_tokens, "stop",
                                      temperature, top_p, min_p, max_tokens);
                        size_t n = vjson_serialize(root, body, 1 << 20);
                        vjson_free(root);
                        resp->status = 200;
                        resp->content_type = "application/json";
                        resp->body = body;
                        resp->body_len = n;
                        resp->body_owned = body;
                    }
                }
                sb_free(&out);
            }
        }
        free(pt);
        free(vis_all);
        vjson_free(body);
        return;
    }

    /* ---- text-only path ---- */
    char *prompt = build_chat_prompt(ctx, messages);
    if (!prompt) {
        json_error(resp, 500, "Prompt build failed");
        vjson_free(body);
        return;
    }

    /* Prompt-length pre-check BEFORE any 200/SSE header goes out, so an
     * over-long context gets a clean 400 instead of a mid-prefill crash. */
    int *ids = (int *)malloc(((size_t)max_ids + 16) * sizeof(int));
    if (!ids) {
        json_error(resp, 500, "Out of memory");
        free(prompt);
        vjson_free(body);
        return;
    }
    int n_ids = qwen_tokenizer_encode(ctx->tok, prompt, ids, max_ids + 16);
    free(ids);
    if (n_ids <= 0) {
        json_error(resp, 400, "Prompt tokenization failed");
        free(prompt);
        vjson_free(body);
        return;
    }
    if (n_ids > max_ids) {
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "Prompt of %d tokens exceeds the %d-token context limit",
                 n_ids, max_ids);
        json_error(resp, 400, msg);
        free(prompt);
        vjson_free(body);
        return;
    }

    /* Admission control: bounded queue (429) + free-RAM check (503).
     * Serial: infer_gate blocks until the running request finishes, then
     * holds the inference lock for this request (queuing semantics).
     * Batch: infer_gate_batch never blocks; the batch scheduler owns the
     * engine critical section (continuous batching). */
    int g = batch_mode ? infer_gate_batch(ctx) : infer_gate(ctx);
    if (g == -1) {
        json_error(resp, 429, "Too many queued requests, retry later");
        free(prompt);
        vjson_free(body);
        return;
    }
    if (g == -2) {
        json_error(resp, 503, "Insufficient memory, retry later");
        free(prompt);
        vjson_free(body);
        return;
    }
    if (g == -3) {
        json_error(resp, 503, "Model is unloading, retry later");
        free(prompt);
        vjson_free(body);
        return;
    }
    if (batch_mode) {
        /* ---- continuous-batching path (shared weights, private KV) ---- */
        VLLMMetrics met; memset(&met, 0, sizeof(met));
        met.last_user_tokens = last_user_tokens;
        VBatchReq r; memset(&r, 0, sizeof(r));
        r.prompt = prompt;
        r.temperature = temperature;
        r.top_p = top_p;
        r.min_p = min_p;
        r.max_tokens = max_tokens;
        r.stream = stream;
        r.conn = conn;
        r.met = &met;
        /* 流式也累积输出文本：batch 流式出证需要全量 text。 */
        char *out_txt = NULL; int out_len = 0;
        r.out_text = &out_txt;
        r.out_len = &out_len;
        r.body_sha = bodyhash;
        if (stream) {
            resp->status = 200;
            resp->content_type = "text/event-stream";
            resp->stream = 1;
            vhttp_stream_begin(conn, 200, "text/event-stream");
            static const char role_preamble[] =
                "data: {\"id\":\"chatcmpl\",\"object\":\"chat.completion.chunk\",\"choices\":"
                "[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n";
            vhttp_stream_write(conn, role_preamble, sizeof(role_preamble) - 1);
            vllm_batch_run(ctx->batch, &r);
        } else {
            int rc = vllm_batch_run(ctx->batch, &r);
            if (rc == -2) {
                json_error(resp, 400, "Prompt exceeds the context limit");
            } else if (rc != 0) {
                json_error(resp, 500, "Inference failed");
            } else {
                char *body = (char *)malloc(1 << 20);
                if (!body) {
                    json_error(resp, 500, "Out of memory");
                } else {
                    VJson *root = vjson_new_object();
                    vjson_obj_set(root, "id", vjson_new_string("chatcmpl-00000000"));
                    vjson_obj_set(root, "object", vjson_new_string("chat.completion"));
                    vjson_obj_set(root, "created", vjson_new_number((double)unix_now()));
                    vjson_obj_set(root, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
                    VJson *choices = vjson_new_array();
                    VJson *ch = vjson_new_object();
                    vjson_obj_set(ch, "index", vjson_new_number(0));
                    VJson *msg = vjson_new_object();
                    vjson_obj_set(msg, "role", vjson_new_string("assistant"));
                    vjson_obj_set(msg, "content", vjson_new_string(out_txt ? out_txt : ""));
                    vjson_obj_set(ch, "message", msg);
                    vjson_obj_set(ch, "finish_reason", vjson_new_string("stop"));
                    vjson_array_push(choices, ch);
                    vjson_obj_set(root, "choices", choices);
                    vjson_obj_set(root, "usage", vjson_new_object());
                    VJson *mj = vjson_new_object();
                    vjson_obj_set(mj, "prompt_tokens", vjson_new_number((double)met.prompt_tokens));
                    vjson_obj_set(mj, "last_user_tokens", vjson_new_number((double)met.last_user_tokens));
                    vjson_obj_set(mj, "n_tokens", vjson_new_number((double)met.n_tokens));
                    vjson_obj_set(mj, "ttft_ms", vjson_new_number(met.ttft_ms));
                    vjson_obj_set(mj, "tpot_ms", vjson_new_number(met.tpot_ms));
                    vjson_obj_set(mj, "total_ms", vjson_new_number(met.total_ms));
                    vjson_obj_set(mj, "prefill_ms", vjson_new_number(met.prefill_ms));
                    vjson_obj_set(mj, "tok_s", vjson_new_number(
                        met.total_ms > 0.0 ? (double)met.n_tokens * 1000.0 / met.total_ms : 0.0));
                    vjson_obj_set(root, "metrics", mj);
                    /* 方案 2：文本批处理非流式出证。 */
                     attest_attach(root, user, bodyhash, out_txt ? out_txt : "",
                                  met.prompt_tokens, met.n_tokens, "stop",
                                  temperature, top_p, min_p, max_tokens);
                    size_t n = vjson_serialize(root, body, 1 << 20);
                    vjson_free(root);
                    resp->status = 200;
                    resp->content_type = "application/json";
                    resp->body = body;
                    resp->body_len = n;
                    resp->body_owned = body;
                }
            }
        }
        free(out_txt);
        infer_done_batch(ctx);
        free(prompt);
        vjson_free(body);
        return;
    }
    /* Per-user L3 isolation (reap offline users, bind this request's file). */
    l3_user_apply(ctx, user);

    if (stream) {
        resp->status = 200;
        resp->content_type = "text/event-stream";
        resp->stream = 1;
        /* Must send the HTTP response header before any SSE payload. */
        vhttp_stream_begin(conn, 200, "text/event-stream");
        /* First chunk: role delta. */
        static const char role_preamble[] =
            "data: {\"id\":\"chatcmpl\",\"object\":\"chat.completion.chunk\",\"choices\":"
            "[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n";
        vhttp_stream_write(conn, role_preamble, sizeof(role_preamble) - 1);
        StrBuf sink; sb_init(&sink);
        VLLMMetrics met; memset(&met, 0, sizeof(met));
        met.last_user_tokens = last_user_tokens;
        int rc = run_completion(ctx, prompt, tail_prompt, ctx_tokens, n_ctx,
                                temperature, top_p, min_p,
                                max_tokens, 1, conn, &sink, &met, NULL, NULL,
                                user, bodyhash);
        sb_free(&sink);
        (void)rc;   /* streaming errors surface as a dropped connection */
        infer_done(ctx);
    } else {
        StrBuf out; sb_init(&out);
        VLLMMetrics met; memset(&met, 0, sizeof(met));
        met.last_user_tokens = last_user_tokens;
        int *hist_ids = NULL; int hist_n = 0;
        int rc = run_completion(ctx, prompt, tail_prompt, ctx_tokens, n_ctx,
                                temperature, top_p, min_p,
                                max_tokens, 0, NULL, &out, &met,
                                &hist_ids, &hist_n, user, bodyhash);
        infer_done(ctx);
        if (rc == -2) {
            json_error(resp, 400, "Prompt exceeds the context limit");
        } else if (rc != 0) {
            json_error(resp, 500, "Inference failed");
        } else {
            char *body = (char *)malloc(1 << 20);
            if (!body) {
                json_error(resp, 500, "Out of memory");
            } else {
                VJson *root = vjson_new_object();
                vjson_obj_set(root, "id", vjson_new_string("chatcmpl-00000000"));
                vjson_obj_set(root, "object", vjson_new_string("chat.completion"));
                vjson_obj_set(root, "created", vjson_new_number((double)unix_now()));
                vjson_obj_set(root, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
                VJson *choices = vjson_new_array();
                VJson *ch = vjson_new_object();
                vjson_obj_set(ch, "index", vjson_new_number(0));
                VJson *msg = vjson_new_object();
                vjson_obj_set(msg, "role", vjson_new_string("assistant"));
                vjson_obj_set(msg, "content", vjson_new_string(out.s ? out.s : ""));
                vjson_obj_set(ch, "message", msg);
                vjson_obj_set(ch, "finish_reason", vjson_new_string("stop"));
                vjson_array_push(choices, ch);
                vjson_obj_set(root, "choices", choices);
                vjson_obj_set(root, "usage", vjson_new_object());
                if (hist_ids && hist_n) {
                    VJson *ht = vjson_new_array();
                    for (int i = 0; i < hist_n; i++)
                        vjson_array_push(ht, vjson_new_number((double)hist_ids[i]));
                    vjson_obj_set(root, "history_tokens", ht);
                }
                /* Per-request inference metrics for the chat client. */
                VJson *mj = vjson_new_object();
                vjson_obj_set(mj, "prompt_tokens", vjson_new_number((double)met.prompt_tokens));
                vjson_obj_set(mj, "last_user_tokens", vjson_new_number((double)met.last_user_tokens));
                vjson_obj_set(mj, "n_tokens", vjson_new_number((double)met.n_tokens));
                vjson_obj_set(mj, "ttft_ms", vjson_new_number(met.ttft_ms));
                vjson_obj_set(mj, "tpot_ms", vjson_new_number(met.tpot_ms));
                vjson_obj_set(mj, "total_ms", vjson_new_number(met.total_ms));
                vjson_obj_set(mj, "prefill_ms", vjson_new_number(met.prefill_ms));
                vjson_obj_set(mj, "tok_s", vjson_new_number(
                    met.total_ms > 0.0 ? (double)met.n_tokens * 1000.0 / met.total_ms : 0.0));
                vjson_obj_set(root, "metrics", mj);
                /* 方案 2：文本串行非流式出证。 */
                 attest_attach(root, user, bodyhash, out.s ? out.s : "",
                              met.prompt_tokens, met.n_tokens, "stop",
                              temperature, top_p, min_p, max_tokens);
                size_t n = vjson_serialize(root, body, 1 << 20);
                vjson_free(root);
                resp->status = 200;
                resp->content_type = "application/json";
                resp->body = body;
                resp->body_len = n;
                resp->body_owned = body;
            }
        }
        sb_free(&out);
        if (hist_ids) free(hist_ids);
    }
    free(prompt);
    free(ctx_tokens);
    free(tail_prompt);
    vjson_free(body);
}

/* ---------- /v1/completions (legacy) ---------- */

static void handle_completions(VLLMServerCtx *ctx, const VHttpRequest *req,
                               VHttpResponse *resp, VHttpConn *conn) {
    (void)conn;
    if (ensure_model_ready(ctx, resp) != 0) return;
    VJson *body = vjson_parse(req->body);
    if (!body) {
        json_error(resp, 400, "Invalid JSON body");
        return;
    }
    const char *prompt = vjson_str(vjson_obj_get(body, "prompt"));
    if (!prompt) {
        json_error(resp, 400, "'prompt' is required");
        vjson_free(body);
        return;
    }
    /* 方案 2：原始请求体 SM3（出证/验证均基于它）。 */
    char bodyhash[65];
    vatt_body_digest(req->body, req->body_len, bodyhash);
    double temperature = 0.7, top_p = 0.9, min_p = ctx->min_p;
    int max_tokens = 64;
    VJson *jv;
    if ((jv = vjson_obj_get(body, "temperature")) && jv->type == VJ_NUMBER) temperature = jv->u.number;
    if ((jv = vjson_obj_get(body, "top_p")) && jv->type == VJ_NUMBER) top_p = jv->u.number;
    if ((jv = vjson_obj_get(body, "min_p")) && jv->type == VJ_NUMBER) min_p = jv->u.number;
    if ((jv = vjson_obj_get(body, "max_tokens")) && jv->type == VJ_NUMBER) {
        max_tokens = (int)jv->u.number;
        if (max_tokens <= 0) max_tokens = 1;
        if (max_tokens > 4096) max_tokens = 4096;
    }
    const char *user = vjson_str(vjson_obj_get(body, "user"));
    char *full = build_completion_prompt(ctx, prompt);
    if (!full) {
        json_error(resp, 500, "Prompt build failed");
        vjson_free(body);
        return;
    }

    /* Same admission control as chat: bounded queue (429) + RAM (503).
     * Batch mode: never blocks on inf_lock; the scheduler owns the engine. */
    const int batch_mode = (ctx->batch_max >= 2);
    if (batch_mode) batch_ensure(ctx);
    int g = batch_mode ? infer_gate_batch(ctx) : infer_gate(ctx);
    if (g == -1) {
        json_error(resp, 429, "Too many queued requests, retry later");
        free(full);
        vjson_free(body);
        return;
    }
    if (g == -2) {
        json_error(resp, 503, "Insufficient memory, retry later");
        free(full);
        vjson_free(body);
        return;
    }
    if (g == -3) {
        json_error(resp, 503, "Model is unloading, retry later");
        free(full);
        vjson_free(body);
        return;
    }
    if (batch_mode) {
        VLLMMetrics met; memset(&met, 0, sizeof(met));
        VBatchReq r; memset(&r, 0, sizeof(r));
        r.prompt = full;
        r.temperature = temperature;
        r.top_p = top_p;
        r.min_p = min_p;
        r.max_tokens = max_tokens;
        r.stream = 0;
        r.met = &met;
        char *out_txt = NULL; int out_len = 0;
        r.out_text = &out_txt;
        r.out_len = &out_len;
        r.body_sha = bodyhash;
        int rc = vllm_batch_run(ctx->batch, &r);
        infer_done_batch(ctx);
        if (rc == -2) {
            json_error(resp, 400, "Prompt exceeds the context limit");
        } else if (rc != 0) {
            json_error(resp, 500, "Inference failed");
        } else {
            char *body = (char *)malloc(1 << 20);
            if (!body) {
                json_error(resp, 500, "Out of memory");
            } else {
                VJson *root = vjson_new_object();
                vjson_obj_set(root, "id", vjson_new_string("cmpl-00000000"));
                vjson_obj_set(root, "object", vjson_new_string("text_completion"));
                vjson_obj_set(root, "created", vjson_new_number((double)unix_now()));
                vjson_obj_set(root, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
                VJson *choices = vjson_new_array();
                VJson *ch = vjson_new_object();
                vjson_obj_set(ch, "index", vjson_new_number(0));
                VJson *msg = vjson_new_object();
                vjson_obj_set(msg, "role", vjson_new_string("assistant"));
                vjson_obj_set(msg, "content", vjson_new_string(out_txt ? out_txt : ""));
                vjson_obj_set(ch, "message", msg);
                vjson_obj_set(ch, "finish_reason", vjson_new_string("stop"));
                vjson_array_push(choices, ch);
                vjson_obj_set(root, "choices", choices);
                vjson_obj_set(root, "usage", vjson_new_object());
                VJson *mj = vjson_new_object();
                vjson_obj_set(mj, "prompt_tokens", vjson_new_number((double)met.prompt_tokens));
                vjson_obj_set(mj, "n_tokens", vjson_new_number((double)met.n_tokens));
                vjson_obj_set(mj, "ttft_ms", vjson_new_number(met.ttft_ms));
                vjson_obj_set(mj, "tpot_ms", vjson_new_number(met.tpot_ms));
                vjson_obj_set(mj, "total_ms", vjson_new_number(met.total_ms));
                vjson_obj_set(mj, "prefill_ms", vjson_new_number(met.prefill_ms));
                vjson_obj_set(mj, "tok_s", vjson_new_number(
                    met.total_ms > 0.0 ? (double)met.n_tokens * 1000.0 / met.total_ms : 0.0));
                vjson_obj_set(root, "metrics", mj);
                /* 方案 2：legacy 批处理非流式出证。 */
                attest_attach(root, user, bodyhash, out_txt ? out_txt : "",
                              met.prompt_tokens, met.n_tokens, "stop",
                              temperature, top_p, min_p, max_tokens);
                size_t n = vjson_serialize(root, body, 1 << 20);
                vjson_free(root);
                resp->status = 200;
                resp->content_type = "application/json";
                resp->body = body;
                resp->body_len = n;
                resp->body_owned = body;
            }
        }
        free(out_txt);
        free(full);
        vjson_free(body);
        return;
    }
    /* Per-user L3 isolation (reap offline users, bind this request's file). */
    l3_user_apply(ctx, user);

    StrBuf out; sb_init(&out);
    VLLMMetrics met; memset(&met, 0, sizeof(met));
    int rc = run_completion(ctx, full, NULL, NULL, 0, temperature, top_p, min_p,
                            max_tokens, 0, NULL, &out, &met, NULL, NULL, user,
                            bodyhash);
    infer_done(ctx);
    if (rc == -2) {
        json_error(resp, 400, "Prompt exceeds the context limit");
    } else if (rc != 0) {
        json_error(resp, 500, "Inference failed");
    } else {
        char *body = (char *)malloc(1 << 20);
        if (!body) {
            json_error(resp, 500, "Out of memory");
        } else {
            VJson *root = vjson_new_object();
            vjson_obj_set(root, "id", vjson_new_string("cmpl-00000000"));
            vjson_obj_set(root, "object", vjson_new_string("text_completion"));
            vjson_obj_set(root, "created", vjson_new_number((double)unix_now()));
            vjson_obj_set(root, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
            VJson *choices = vjson_new_array();
            VJson *ch = vjson_new_object();
            vjson_obj_set(ch, "index", vjson_new_number(0));
            vjson_obj_set(ch, "text", vjson_new_string(out.s ? out.s : ""));
            vjson_obj_set(ch, "finish_reason", vjson_new_string("stop"));
            vjson_array_push(choices, ch);
            vjson_obj_set(root, "choices", choices);
            /* 方案 2：legacy 串行非流式出证。 */
            attest_attach(root, user, bodyhash, out.s ? out.s : "",
                          met.prompt_tokens, met.n_tokens, "stop",
                          temperature, top_p, min_p, max_tokens);
            size_t n = vjson_serialize(root, body, 1 << 20);
            vjson_free(root);
            resp->status = 200;
            resp->content_type = "application/json";
            resp->body = body;
            resp->body_len = n;
            resp->body_owned = body;
        }
    }
    sb_free(&out);
    free(full);
    vjson_free(body);
}

/* ---------- /v1/messages (Anthropic-compatible) ---------- */

/* Extract plain text from an Anthropic content field (string or
 * [{type:"text",text:...}, ...]); appends to sb. */
static void anthropic_content_to_text(const VJson *content, StrBuf *sb) {
    if (!content) return;
    if (content->type == VJ_STRING) {
        const char *t = vjson_str(content);
        if (t) sb_str(sb, t);
        return;
    }
    if (content->type == VJ_ARRAY) {
        size_t n = vjson_array_len(content);
        for (size_t i = 0; i < n; i++) {
            const VJson *part = vjson_array_get(content, i);
            if (!part) continue;
            const char *ptype = vjson_str(vjson_obj_get(part, "type"));
            if (ptype && strcmp(ptype, "text") == 0) {
                const char *t = vjson_str(vjson_obj_get(part, "text"));
                if (t) sb_str(sb, t);
            }
        }
    }
}

/* Convert the OpenAI chat response body into an Anthropic message body.
 * Returns a malloc'd string (caller frees) or NULL. */
static char *anthropic_from_openai(VLLMServerCtx *ctx, const char *openai_body) {
    VJson *j = vjson_parse(openai_body);
    if (!j) return NULL;
    const VJson *choices = vjson_obj_get(j, "choices");
    const VJson *ch = (choices && choices->type == VJ_ARRAY && vjson_array_len(choices) > 0)
                      ? vjson_array_get(choices, 0) : NULL;
    const char *text = "";
    if (ch) {
        const VJson *msg = vjson_obj_get(ch, "message");
        if (msg) {
            const char *c = vjson_str(vjson_obj_get(msg, "content"));
            if (c) text = c;
        }
    }
    const char *fr = ch ? vjson_str(vjson_obj_get(ch, "finish_reason")) : NULL;
    const char *stop_reason = (fr && strcmp(fr, "length") == 0) ? "max_tokens" : "end_turn";
    int in_tok = 0, out_tok = 0;
    /* vllm_kestrel 的 OpenAI 响应把指标放在顶层 "metrics"（usage 是空对象）。 */
    const VJson *met = vjson_obj_get(j, "metrics");
    if (met) {
        const VJson *mj = vjson_obj_get(met, "prompt_tokens");
        if (mj && mj->type == VJ_NUMBER) in_tok = (int)mj->u.number;
        mj = vjson_obj_get(met, "n_tokens");
        if (mj && mj->type == VJ_NUMBER) out_tok = (int)mj->u.number;
    } else {
        const VJson *usage = vjson_obj_get(j, "usage");
        if (usage) {
            const VJson *mj = vjson_obj_get(usage, "prompt_tokens");
            if (mj && mj->type == VJ_NUMBER) in_tok = (int)mj->u.number;
            mj = vjson_obj_get(usage, "n_tokens");
            if (mj && mj->type == VJ_NUMBER) out_tok = (int)mj->u.number;
        }
    }
    VJson *root = vjson_new_object();
    vjson_obj_set(root, "id", vjson_new_string("msg_vllm_kestrel"));
    vjson_obj_set(root, "type", vjson_new_string("message"));
    vjson_obj_set(root, "role", vjson_new_string("assistant"));
    vjson_obj_set(root, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
    VJson *content = vjson_new_array();
    VJson *block = vjson_new_object();
    vjson_obj_set(block, "type", vjson_new_string("text"));
    vjson_obj_set(block, "text", vjson_new_string(text));
    vjson_array_push(content, block);
    vjson_obj_set(root, "content", content);
    vjson_obj_set(root, "stop_reason", vjson_new_string(stop_reason));
    vjson_obj_set(root, "stop_sequence", vjson_new_null());
    VJson *u = vjson_new_object();
    vjson_obj_set(u, "input_tokens", vjson_new_number(in_tok));
    vjson_obj_set(u, "output_tokens", vjson_new_number(out_tok));
    vjson_obj_set(root, "usage", u);
    char *buf = (char *)malloc(1 << 20);
    if (!buf) { vjson_free(root); vjson_free(j); return NULL; }
    size_t n = vjson_serialize(root, buf, 1 << 20);
    vjson_free(root);
    vjson_free(j);
    buf[n] = '\0';
    return buf;
}

/* Anthropic /v1/messages -> internal OpenAI chat core -> Anthropic response.
 * Non-streaming only for now; system/messages text parts are supported. */
static void handle_anthropic(VLLMServerCtx *ctx, const VHttpRequest *req,
                             VHttpResponse *resp, VHttpConn *conn) {
    VJson *body = vjson_parse(req->body);
    if (!body) { json_error(resp, 400, "Invalid JSON body"); return; }

    const VJson *messages = vjson_obj_get(body, "messages");
    if (!messages || messages->type != VJ_ARRAY || vjson_array_len(messages) == 0) {
        json_error(resp, 400, "'messages' array is required");
        vjson_free(body); return;
    }
    VJson *jv;
    double temperature = 0.7, top_p = 0.9;
    int max_tokens = 0, stream = 0;
    if ((jv = vjson_obj_get(body, "max_tokens")) && jv->type == VJ_NUMBER)
        max_tokens = (int)jv->u.number;
    if ((jv = vjson_obj_get(body, "temperature")) && jv->type == VJ_NUMBER)
        temperature = jv->u.number;
    if ((jv = vjson_obj_get(body, "top_p")) && jv->type == VJ_NUMBER)
        top_p = jv->u.number;
    if ((jv = vjson_obj_get(body, "stream")) && jv->type == VJ_BOOL)
        stream = jv->u.boolean;
    if (stream) {
        json_error(resp, 501, "Anthropic streaming not supported yet (use stream=false)");
        vjson_free(body); return;
    }
    if (max_tokens <= 0) {
        json_error(resp, 400, "'max_tokens' is required");
        vjson_free(body); return;
    }

    /* Convert to an OpenAI-format body so the existing chat core runs it. */
    VJson *o = vjson_new_object();
    vjson_obj_set(o, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
    VJson *omsg = vjson_new_array();
    const VJson *sys = vjson_obj_get(body, "system");
    if (sys) {
        StrBuf sb; sb_init(&sb);
        anthropic_content_to_text(sys, &sb);
        if (sb.s && sb.s[0]) {
            VJson *m = vjson_new_object();
            vjson_obj_set(m, "role", vjson_new_string("system"));
            vjson_obj_set(m, "content", vjson_new_string(sb.s));
            vjson_array_push(omsg, m);
        }
        sb_free(&sb);
    }
    size_t n = vjson_array_len(messages);
    for (size_t i = 0; i < n; i++) {
        const VJson *msg = vjson_array_get(messages, i);
        if (!msg) continue;
        const char *role = vjson_str(vjson_obj_get(msg, "role"));
        if (!role) continue;
        const VJson *content = vjson_obj_get(msg, "content");
        if (!content) continue;
        StrBuf sb; sb_init(&sb);
        anthropic_content_to_text(content, &sb);
        VJson *m = vjson_new_object();
        vjson_obj_set(m, "role", vjson_new_string(role));
        vjson_obj_set(m, "content", vjson_new_string(sb.s ? sb.s : ""));
        vjson_array_push(omsg, m);
        sb_free(&sb);
    }
    vjson_obj_set(o, "messages", omsg);
    vjson_obj_set(o, "max_tokens", vjson_new_number(max_tokens));
    vjson_obj_set(o, "temperature", vjson_new_number(temperature));
    vjson_obj_set(o, "top_p", vjson_new_number(top_p));
    vjson_obj_set(o, "stream", vjson_new_bool(0));

    char *obuf = (char *)malloc(4 << 20);
    if (!obuf) { json_error(resp, 500, "Out of memory"); vjson_free(o); vjson_free(body); return; }
    size_t olen = vjson_serialize(o, obuf, 4 << 20);
    vjson_free(o);
    if (olen == 0) {
        json_error(resp, 500, "body convert failed");
        free(obuf); vjson_free(body); return;
    }

    VHttpRequest fake;
    memset(&fake, 0, sizeof(fake));
    fake.method = "POST";
    fake.path = "/v1/chat/completions";
    fake.body = obuf;
    fake.body_len = olen;
    handle_chat(ctx, &fake, resp, conn);
    free(obuf);
    vjson_free(body);

    /* Translate the OpenAI response into the Anthropic message format. */
    if (resp->status == 200 && resp->body && resp->body_owned) {
        char *conv = anthropic_from_openai(ctx, resp->body);
        if (conv) {
            free((void *)resp->body);
            resp->body = conv;
            resp->body_owned = conv;   /* 必须同步：http 层发送后 free(body_owned) */
            resp->body_len = strlen(conv);
        }
    }
}

/* ---------- /health ---------- */

static void handle_health(VLLMServerCtx *ctx, VHttpResponse *resp) {
    char *body = (char *)malloc(1024);
    if (!body) { json_error(resp, 500, "Out of memory"); return; }
    VJson *root = vjson_new_object();
    vjson_obj_set(root, "status", vjson_new_string("ok"));
    vjson_obj_set(root, "model", vjson_new_string(ctx->model_id ? ctx->model_id : "qwen3-vl-8b"));
    vhttp_mutex_lock(ctx->stat_lock);
    vjson_obj_set(root, "busy", vjson_new_bool(ctx->busy));
    vjson_obj_set(root, "queued", vjson_new_number((double)ctx->queued_requests));
    vjson_obj_set(root, "active", vjson_new_number((double)ctx->active_requests));
    vjson_obj_set(root, "total_requests", vjson_new_number((double)ctx->total_requests));
    vhttp_mutex_unlock(ctx->stat_lock);
    /* Model may not be loaded yet (manual-load mode): cfg is NULL then. */
    if (ctx->cfg)
        vjson_obj_set(root, "max_seq", vjson_new_number((double)ctx->cfg->max_seq_len));
    else
        vjson_obj_set(root, "max_seq", vjson_new_number(0));
    vjson_obj_set(root, "max_queued", vjson_new_number((double)ctx->max_queued));
    size_t n = vjson_serialize(root, body, 1024);
    vjson_free(root);
    resp->status = 200;
    resp->content_type = "application/json";
    resp->body = body;
    resp->body_len = n;
    resp->body_owned = body;
}

/* ---------- /chat/ (inference client page) ---------- */

static const char *chat_html_path(void) {
    const char *e = getenv("VLLM_CHAT_HTML");
    if (e && e[0]) return e;
    FILE *f = fopen("./chat.html", "rb");
    if (f) { fclose(f); return "./chat.html"; }
    return "/NewVLLM/chat.html";
}

static void handle_chat_page(VHttpResponse *resp) {
    /* Embedded chat.html (self-contained exe): used when no file is deployed
     * next to the engine. VLLM_CHAT_HTML overrides to a custom file. */
    size_t elen = 0;
    const char *ehtml = embedded_chat_html(&elen);
    FILE *f = fopen(chat_html_path(), "rb");
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
            "<html><body><h3>chat.html not found</h3>"
            "<p>Deploy chat.html next to the engine (e.g. /NewVLLM/chat.html) "
            "or set VLLM_CHAT_HTML.</p></body></html>";
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
        resp->body = "chat.html too large";
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

/* ---------- /convert/ (VQF conversion tool page) ---------- */

static const char *convert_html_path(void) {
    const char *e = getenv("VLLM_CONVERT_HTML");
    if (e && e[0]) return e;
    FILE *f = fopen("./convert.html", "rb");
    if (f) { fclose(f); return "./convert.html"; }
    return "/NewVLLM/convert.html";
}

static void handle_convert_page(VHttpResponse *resp) {
    /* Embedded convert.html (self-contained exe): used when no file is
     * deployed next to the engine. VLLM_CONVERT_HTML overrides to a file. */
    size_t elen = 0;
    const char *ehtml = embedded_convert_html(&elen);
    FILE *f = fopen(convert_html_path(), "rb");
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
            "<html><body><h3>convert.html not found</h3>"
            "<p>Deploy convert.html next to the engine (e.g. /NewVLLM/convert.html) "
            "or set VLLM_CONVERT_HTML.</p></body></html>";
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
        resp->body = "convert.html too large";
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

/* ---------- router ---------- */

/* Defined in vllm_admin.c (web management surface under /admin). */
void vllm_admin_route(VLLMServerCtx *ctx, const VHttpRequest *req,
                      VHttpResponse *resp);

static void server_handler(const VHttpRequest *req, VHttpResponse *resp,
                           VHttpConn *conn, void *ud) {
    VLLMServerCtx *ctx = (VLLMServerCtx *)ud;
    resp->status = 404;
    resp->content_type = "application/json";

    /* Unload gate: every HTTP handler is counted (inflight_handlers). The
     * unload waits for this to reach 0 before freeing ctx->ist/cfg/w — a
     * worker may be between ensure_model_ready and infer_gate where it reads
     * ctx->ist/cfg without holding inf_lock, and freeing under it would be a
     * use-after-free. Inference endpoints are additionally rejected (503)
     * while an unload is in progress so no NEW handler can enter that window. */
    const int is_infer_path =
        strcmp(req->path, "/v1/chat/completions") == 0 ||
        strcmp(req->path, "/v1/completions") == 0 ||
        strcmp(req->path, "/v1/messages") == 0;
    vhttp_mutex_lock(ctx->stat_lock);
    ctx->inflight_handlers++;
    int reject_unload = is_infer_path && ctx->unload_pending;
    vhttp_mutex_unlock(ctx->stat_lock);
    if (reject_unload) {
        static const char msg[] =
            "{\"error\":{\"message\":\"Model is unloading, retry later\","
            "\"type\":\"unavailable\",\"code\":503}}";
        resp->status = 503;
        resp->content_type = "application/json";
        resp->body = msg;
        resp->body_len = sizeof(msg) - 1;
        vhttp_mutex_lock(ctx->stat_lock);
        ctx->inflight_handlers--;
        vhttp_mutex_unlock(ctx->stat_lock);
        return;
    }

    if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/v1/models") == 0) {
        handle_models(ctx, resp);
    } else if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/health") == 0) {
        handle_health(ctx, resp);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/v1/chat/completions") == 0) {
        handle_chat(ctx, req, resp, conn);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/v1/messages") == 0) {
        handle_anthropic(ctx, req, resp, conn);
    } else if (strcmp(req->method, "POST") == 0 &&
               strcmp(req->path, "/v1/completions") == 0) {
        handle_completions(ctx, req, resp, conn);
    } else if (strncmp(req->path, "/admin", 6) == 0) {
        vllm_admin_route(ctx, req, resp);
    } else if (strcmp(req->path, "/chat") == 0 || strcmp(req->path, "/chat/") == 0) {
        handle_chat_page(resp);
    } else if (strcmp(req->path, "/convert") == 0 || strcmp(req->path, "/convert/") == 0) {
        handle_convert_page(resp);
    } else if (strcmp(req->path, "/") == 0) {
        static const char info[] = "{\"status\":\"ok\",\"server\":\"vllm_kestrel\"}";
        resp->status = 200;
        resp->content_type = "application/json";
        resp->body = info;
        resp->body_len = sizeof(info) - 1;
    } else {
        static const char nf[] =
            "{\"error\":{\"message\":\"Not Found\",\"type\":\"invalid_request_error\",\"code\":404}}";
        resp->status = 404;
        resp->content_type = "application/json";
        resp->body = nf;
        resp->body_len = sizeof(nf) - 1;
    }

    vhttp_mutex_lock(ctx->stat_lock);
    ctx->inflight_handlers--;
    vhttp_mutex_unlock(ctx->stat_lock);
}

/* Idle-unload monitor: with --auto-unload N the model is unloaded after N
 * seconds without an inference request (不用时卸载). Unload only happens
 * when the engine is fully idle; vllm_serve_unload_model refuses busy
 * states and blocks new inference while freeing. */
static void *idle_unload_thread(void *arg) {
    VLLMServerCtx *ctx = (VLLMServerCtx *)arg;
    while (!ctx->stopping) {
        sleep(1);
        if (ctx->auto_unload_s > 0 && ctx->load_state == 2 &&
            (time(NULL) - ctx->last_infer_s) > ctx->auto_unload_s) {
            int rc = vllm_serve_unload_model(ctx);
            if (rc == 0) {
                ctx->auto_unload_count++;
                fprintf(stderr, "[UNLOAD] idle %lds > %lds -> auto-unloaded "
                        "(total unloads=%d)\n",
                        (long)(time(NULL) - ctx->last_infer_s),
                        ctx->auto_unload_s, ctx->unload_count);
                fflush(stderr);
            }
        }
    }
    return NULL;
}

int vllm_server_run(VLLMServerCtx *ctx, int port, int n_threads,
                    void (*on_start)(int actual_port, void *ud)) {
    ctx->inf_lock = vhttp_mutex_new();
    ctx->stat_lock = vhttp_mutex_new();
    if (!ctx->inf_lock || !ctx->stat_lock) {
        fprintf(stderr, "[SRV] mutex init failed\n");
        return 1;
    }
    ctx->total_requests = 0;
    ctx->active_requests = 0;
    ctx->queued_requests = 0;
    ctx->busy = 0;
    ctx->inflight_handlers = 0;
    ctx->next_chunk_id = 0;
    ctx->last_infer_s = (long)time(NULL);   /* never auto-unload before first use */
    ctx->stopping = 0;
    ctx->idle_thr_started = 0;
    if (ctx->auto_unload_s > 0) {
        if (pthread_create(&ctx->idle_thr, NULL, idle_unload_thread, ctx) == 0)
            ctx->idle_thr_started = 1;
        else
            fprintf(stderr, "[SRV] idle-unload monitor thread create failed "
                    "(auto-unload disabled)\n");
    }
    int rc = vhttp_serve_ex(port, server_handler, ctx, on_start, n_threads);
    ctx->stopping = 1;
    if (ctx->idle_thr_started) {
        pthread_join(ctx->idle_thr, NULL);
        ctx->idle_thr_started = 0;
    }
    vllm_server_reset_session(ctx);
    vhttp_mutex_free(ctx->inf_lock);
    vhttp_mutex_free(ctx->stat_lock);
    return rc;
}

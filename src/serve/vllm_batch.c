/**
 * vllm_batch.c - server-side continuous batching (batch decode)
 *
 * See vllm_batch.h for the axiom mapping and design notes.
 *
 * Engine thread: the only caller of st_qwen_model_forward_batch. It polls
 * the node table for want_step nodes, runs ONE forward for the whole set
 * (weights read once per step), then bumps each node's done_epoch.
 *
 * Worker thread (per HTTP request): acquires a node (inference state),
 * prefills under the engine mutex, then per step: sample -> submit token ->
 * wait for done_epoch -> stream the decoded chunk. Requests join/leave the
 * batch freely; finished requests free their state back to the pool.
 */
#include "vllm_batch.h"
#include "vllm_superpos.h"   /* sample_token() */
#include "vllm_platform.h"   /* st_now_sec() */
#include "vllm_attest.h"     /* 方案 2：batch 流式出证 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <pthread.h>
#include <unistd.h>

#define VB_MAX_BATCH 32
#define VB_SLOT_WAIT_MS 120000   /* max wait for a free batch slot */

/* ---------- tiny helpers (self-contained; mirror vllm_server.c) ---------- */

static long vb_unix_now(void) { return (long)time(NULL); }

static void vb_sleep_ms(int ms) {
    usleep((useconds_t)ms * 1000);
}

/* Replace BPE markers Ġ(C4 A0)->space, Ċ(C4 8A)->newline. */
static const char *vb_gj_replace(const char *s, char *tmp, size_t tmplen) {
    if (!s || !s[0]) return s;
    const unsigned char *u = (const unsigned char *)s;
    int len = (int)strlen(s);
    int has = 0;
    for (int i = 0; i + 1 < len; i++)
        if (u[i] == 0xC4 && (u[i + 1] == 0xA0 || u[i + 1] == 0x8A)) { has = 1; break; }
    if (!has) return s;
    int w = 0;
    for (int i = 0; i < len && w < (int)tmplen - 1; i++) {
        if (u[i] == 0xC4 && i + 1 < len) {
            if (u[i + 1] == 0xA0) { tmp[w++] = ' '; i++; continue; }
            if (u[i + 1] == 0x8A) { tmp[w++] = '\n'; i++; continue; }
        }
        tmp[w++] = s[i];
    }
    tmp[w] = '\0';
    return tmp;
}

/* Incremental UTF-8 decoder (reassembles split multi-byte tokens). */
typedef struct { unsigned char pending[4]; int expect, npend; } VBUtf8;

static void vb_utf8_init(VBUtf8 *d) { d->expect = 0; d->npend = 0; }

static int vb_utf8_feed(VBUtf8 *d, const unsigned char *buf, int len,
                        char *out, int out_cap) {
    int w = 0;
    for (int i = 0; i < len; i++) {
        unsigned char b = buf[i];
        if (d->expect == 0) {
            if (b < 0x80) {
                if (w < out_cap - 1) out[w++] = (char)b;
            } else if (b >= 0xC2 && b <= 0xDF) { d->expect = 2; d->pending[0] = b; d->npend = 1; }
            else if (b >= 0xE0 && b <= 0xEF) { d->expect = 3; d->pending[0] = b; d->npend = 1; }
            else if (b >= 0xF0 && b <= 0xF4) { d->expect = 4; d->pending[0] = b; d->npend = 1; }
            else { if (w < out_cap - 3) { out[w++] = (char)0xEF; out[w++] = (char)0xBF; out[w++] = (char)0xBD; } }
        } else {
            d->pending[d->npend++] = b;
            if (d->npend == d->expect) {
                int ok = 1;
                for (int k = 1; k < d->expect; k++)
                    if ((d->pending[k] & 0xC0) != 0x80) { ok = 0; break; }
                if (ok) { for (int k = 0; k < d->expect; k++) if (w < out_cap - 1) out[w++] = (char)d->pending[k]; }
                else { if (w < out_cap - 3) { out[w++] = (char)0xEF; out[w++] = (char)0xBF; out[w++] = (char)0xBD; } }
                d->expect = 0; d->npend = 0;
            }
        }
    }
    out[w] = '\0';
    return w;
}

/* ---------- scheduler ---------- */

typedef struct VBNode {
    int used;
    STQwenInferenceState st;   /* per-request state (weights shared) */
    volatile int want_step;    /* a token is waiting for a forward */
    int token;
    long wait_epoch;           /* want done_epoch > wait_epoch */
    long done_epoch;
    VBatchReq *req;
    int *ids;
    int ids_cap;
    struct VBNode *next;       /* free list */
} VBNode;

struct VBatchSched {
    VHttpMutex *lock;          /* guards nodes/free list/epoch/chunk id */
    VHttpMutex *engine;        /* serializes prefill + vision-encode + forwards */
    pthread_mutex_t cv_mu;
    pthread_cond_t cv;
    STModelWeights *weights;
    QwenTokenizer *tok;
    const char *model_id;
    int batch_max;
    VBNode nodes[VB_MAX_BATCH];
    VBNode *free_list;
    long epoch;                /* completed batch-step counter */
    long next_chunk_id;
    volatile int stop;
    int thread_started;
    pthread_t thread;
};

static void *vb_engine_main(void *arg);

VBatchSched *vllm_batch_create(const STModelWeights *w, QwenTokenizer *tok,
                               const char *model_id, int batch_max) {
    if (batch_max < 2) batch_max = 2;
    if (batch_max > VB_MAX_BATCH) batch_max = VB_MAX_BATCH;
    VBatchSched *b = (VBatchSched *)calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->lock = vhttp_mutex_new();
    b->engine = vhttp_mutex_new();
    if (!b->lock || !b->engine) {
        if (b->lock) vhttp_mutex_free(b->lock);
        if (b->engine) vhttp_mutex_free(b->engine);
        free(b);
        return NULL;
    }
    pthread_mutex_init(&b->cv_mu, NULL);
    pthread_cond_init(&b->cv, NULL);
    b->weights = (STModelWeights *)w;
    b->tok = tok;
    b->model_id = model_id;
    b->batch_max = batch_max;
    for (int i = 0; i < batch_max; i++) {
        b->nodes[i].next = b->free_list;
        b->free_list = &b->nodes[i];
    }
    if (pthread_create(&b->thread, NULL, vb_engine_main, b) == 0)
        b->thread_started = 1;
    return b;
}

void vllm_batch_destroy(VBatchSched *b) {
    if (!b) return;
    b->stop = 1;
    pthread_mutex_lock(&b->cv_mu);
    pthread_cond_broadcast(&b->cv);
    pthread_mutex_unlock(&b->cv_mu);
    if (b->thread_started) {
        pthread_join(b->thread, NULL);
    }
    pthread_mutex_destroy(&b->cv_mu);
    pthread_cond_destroy(&b->cv);
    vhttp_mutex_free(b->lock);
    vhttp_mutex_free(b->engine);
    for (int i = 0; i < b->batch_max; i++) {
        free(b->nodes[i].ids);
        b->nodes[i].ids = NULL;
        b->nodes[i].ids_cap = 0;
    }
    free(b);
}

void vllm_batch_engine_lock(VBatchSched *b) { vhttp_mutex_lock(b->engine); }
void vllm_batch_engine_unlock(VBatchSched *b) { vhttp_mutex_unlock(b->engine); }

int vllm_batch_active(const VBatchSched *b) {
    if (!b) return 0;
    int n = 0;
    vhttp_mutex_lock(b->lock);
    for (int i = 0; i < b->batch_max; i++) if (b->nodes[i].used) n++;
    vhttp_mutex_unlock(b->lock);
    return n;
}

/* The single batch-forward driver. */
static void *vb_engine_main(void *arg) {
    VBatchSched *b = (VBatchSched *)arg;
    STQwenInferenceState *sts[VB_MAX_BATCH];
    int toks[VB_MAX_BATCH];
    VBNode *nodes[VB_MAX_BATCH];
    while (!b->stop) {
        int n = 0;
        vhttp_mutex_lock(b->lock);
        for (int i = 0; i < b->batch_max; i++) {
            if (b->nodes[i].used && b->nodes[i].want_step && n < VB_MAX_BATCH) {
                sts[n] = &b->nodes[i].st;
                toks[n] = b->nodes[i].token;
                nodes[n] = &b->nodes[i];
                n++;
            }
        }
        vhttp_mutex_unlock(b->lock);
        if (n == 0) {
            pthread_mutex_lock(&b->cv_mu);
            if (!b->stop) pthread_cond_wait(&b->cv, &b->cv_mu);
            pthread_mutex_unlock(&b->cv_mu);
            continue;
        }
        /* One forward for the whole set: weights read once for all tokens
         * (block_matrix_assoc_natural: shared-input fusion). */
        vhttp_mutex_lock(b->engine);
        if (!b->stop) st_qwen_model_forward_batch(sts, n, toks);
        vhttp_mutex_unlock(b->engine);
        vhttp_mutex_lock(b->lock);
        for (int i = 0; i < n; i++) {
            nodes[i]->want_step = 0;
            long e = ++b->epoch;
            __atomic_store_n(&nodes[i]->done_epoch, e, __ATOMIC_RELEASE);
        }
        vhttp_mutex_unlock(b->lock);
        pthread_mutex_lock(&b->cv_mu);
        pthread_cond_broadcast(&b->cv);
        pthread_mutex_unlock(&b->cv_mu);
    }
    return NULL;
}

static VBNode *vb_acquire(VBatchSched *b) {
    long deadline = st_now_sec() * 1000.0 + VB_SLOT_WAIT_MS;
    for (;;) {
        vhttp_mutex_lock(b->lock);
        if (b->free_list) {
            VBNode *n = b->free_list;
            b->free_list = n->next;
            n->used = 1;
            n->want_step = 0;
            __atomic_store_n(&n->done_epoch, 0, __ATOMIC_RELAXED);
            n->req = NULL;
            vhttp_mutex_unlock(b->lock);
            return n;
        }
        vhttp_mutex_unlock(b->lock);
        if (b->stop) return NULL;
        if (st_now_sec() * 1000.0 >= deadline) return NULL;
        pthread_mutex_lock(&b->cv_mu);
        if (!b->stop) pthread_cond_wait(&b->cv, &b->cv_mu);
        pthread_mutex_unlock(&b->cv_mu);
    }
}

static void vb_release(VBatchSched *b, VBNode *n) {
    st_qwen_inference_free(&n->st);
    vhttp_mutex_lock(b->lock);
    n->used = 0;
    n->req = NULL;
    n->next = b->free_list;
    b->free_list = n;
    vhttp_mutex_unlock(b->lock);
    pthread_mutex_lock(&b->cv_mu);
    pthread_cond_broadcast(&b->cv);
    pthread_mutex_unlock(&b->cv_mu);
}

/* Submit one decode token and block until its batch step completed. */
static void vb_step(VBatchSched *b, VBNode *n, int token) {
    vhttp_mutex_lock(b->lock);
    n->token = token;
    n->want_step = 1;
    n->wait_epoch = b->epoch;
    vhttp_mutex_unlock(b->lock);
    pthread_mutex_lock(&b->cv_mu);
    pthread_cond_signal(&b->cv);
    pthread_mutex_unlock(&b->cv_mu);
    for (;;) {
        long done_epoch = __atomic_load_n(&n->done_epoch, __ATOMIC_ACQUIRE);
        if (done_epoch > n->wait_epoch) break;
        pthread_mutex_lock(&b->cv_mu);
        pthread_cond_wait(&b->cv, &b->cv_mu);
        pthread_mutex_unlock(&b->cv_mu);
    }
}

/* ---------- per-request decode loop (streaming, mirrors decode_loop) ---------- */

static void vb_append_text(char **out, int *out_len, const char *text) {
    if (!out) return;
    size_t tlen = strlen(text);
    size_t cur = *out ? (size_t)*out_len : 0;
    char *np = (char *)realloc(*out, cur + tlen + 1);
    if (!np) return;
    memcpy(np + cur, text, tlen);
    np[cur + tlen] = '\0';
    *out = np;
    *out_len = (int)(cur + tlen);
}

int vllm_batch_run(VBatchSched *b, VBatchReq *req) {
    if (!b || !req) return -1;
    VBNode *n = vb_acquire(b);
    if (!n) { req->rc = -3; return -3; }
    n->req = req;
    if (st_qwen_inference_init(&n->st, b->weights) != 0) {
        vb_release(b, n);
        req->rc = -1;
        return -1;
    }
    STQwenInferenceState *st = &n->st;
    int vc = st->cfg.vocab_size;
    VLLMMetrics *met = req->met;
    if (met) met->start_s = st_now_sec();

    /* ---- prefill (engine-serialized; multimodal prompt already built) ---- */
    int rc = 0;
    vhttp_mutex_lock(b->engine);
    if (req->is_mm) {
        rc = st_qwen_model_multimodal_prefill_ex(st, req->pt, req->pn,
                                                 req->vis_all, req->vis_total,
                                                 req->grids, req->n_regions,
                                                 req->ds_features, req->n_ds,
                                                 0);
        if (rc == 0 && met) met->prompt_tokens = req->pn + req->vis_total;
    } else {
        int max_ids = st->max_kv_slots;
        if (max_ids < 1024) max_ids = 1024;
        int need = max_ids + 16;
        if (n->ids_cap < need) {
            int *np = (int *)realloc(n->ids, (size_t)need * sizeof(int));
            if (!np) { rc = -1; }
            else { n->ids = np; n->ids_cap = need; }
        }
        if (rc == 0 && !n->ids) { rc = -1; }
        else {
            int n_ids = qwen_tokenizer_encode(b->tok, req->prompt, n->ids, need);
            if (n_ids <= 0) { rc = -1; }
            else if (n_ids > max_ids) { rc = -2; }
            else {
                rc = st_qwen_model_prefill_batch(st, n->ids, n_ids);
                if (rc == 0 && met) { met->prompt_tokens = n_ids; }
            }
        }
    }
    if (rc == 0 && met)
        met->prefill_ms = (st_now_sec() - met->start_s) * 1000.0;
    vhttp_mutex_unlock(b->engine);
    if (rc != 0) {
        vb_release(b, n);
        req->rc = rc;
        return rc;
    }

    /* ---- batched decode ---- */
    long my_id;
    vhttp_mutex_lock(b->lock);
    my_id = b->next_chunk_id++;
    vhttp_mutex_unlock(b->lock);
    char chunk_id[64];
    snprintf(chunk_id, sizeof(chunk_id), "chatcmpl-%08lx", my_id);
    long created = vb_unix_now();
    const char *model = b->model_id ? b->model_id : "qwen3-vl-8b";
    char dbuf[4096];
    int done = 0;
    const char *finish = "stop";
    double t_first = 0.0;
    int gen = 0;
    VBUtf8 dec;
    vb_utf8_init(&dec);

    for (int step = 0; step < req->max_tokens; step++) {
        int id = sample_token_pk(st->logits, vc, (float)req->temperature,
                                 (float)req->top_p, (float)req->min_p,
                                 req->top_k);
        if (id == b->tok->eos_id || id == b->tok->im_end_id) { finish = "stop"; done = 1; break; }
        const char *raw = qwen_tokenizer_decode(b->tok, id);
        int rawlen = b->tok->str_lens[id];
        char piece[2048];
        int plen = vb_utf8_feed(&dec, (const unsigned char *)raw, rawlen, piece, (int)sizeof(piece));
        char pbuf[2048];
        const char *text = vb_gj_replace(piece, pbuf, sizeof(pbuf));
        if (plen > 0 && text && text[0]) {
            if (req->stream) {
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
                vjson_serialize(root, dbuf, sizeof(dbuf));
                vjson_free(root);
                char line[4300];
                int ln = snprintf(line, sizeof(line), "data: %s\n\n", dbuf);
                if (ln > 0 && vhttp_stream_write(req->conn, line, (size_t)ln) < 0) {
                    done = -1;
                    break;
                }
            }
            /* 流式 + 非流式统一累积输出文本：流式供末尾 attest 事件出证。 */
            if (req->out_text) vb_append_text(req->out_text, req->out_len, text);
        }
        gen++;
        if (met && gen == 1) t_first = st_now_sec();
        if (step + 1 < req->max_tokens) vb_step(b, n, id);
    }
    if (done != 1) finish = "length";

    if (met) {
        double t_end = st_now_sec();
        met->n_tokens = gen;
        if (t_first > 0) met->ttft_ms = (t_first - met->start_s) * 1000.0;
        met->total_ms = (t_end - met->start_s) * 1000.0;
        if (gen > 1) met->tpot_ms = (t_end - t_first) / (double)(gen - 1) * 1000.0;
        else if (t_first > 0) met->tpot_ms = met->ttft_ms;
    }

    if (req->stream && done != -1) {
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
        vjson_serialize(root, dbuf, sizeof(dbuf));
        vjson_free(root);
        char line[4300];
        int ln = snprintf(line, sizeof(line), "data: %s\n\n", dbuf);
        if (ln > 0) vhttp_stream_write(req->conn, line, (size_t)ln);
        /* per-request metrics event */
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
            vhttp_stream_write(req->conn, mline, (size_t)strlen(mline));
        }
        /* 方案 2：batch 流式出证（与串行 decode_loop 相同的 schema=3）。 */
        if (vatt_active() && req->out_text && req->out_text[0]) {
            VAttestReq ar;
            memset(&ar, 0, sizeof(ar));
            ar.body_sha = req->body_sha;
            ar.text = req->out_text[0];
            ar.text_len = strlen(req->out_text[0]);
            ar.n_prompt_tokens = (met && met->prompt_tokens > 0)
                                     ? met->prompt_tokens : 0;
            ar.n_gen_tokens = gen;
            ar.finish = finish;
            ar.temperature = req->temperature;
            ar.top_p = req->top_p;
            ar.min_p = req->min_p;
            ar.top_k = req->top_k;
            ar.thinking = req->thinking;
            ar.max_tokens = req->max_tokens;
            ar.ts = created;
            char *aj = vatt_seal_json(&ar);
            if (aj) {
                char aline[4500];
                int ln = snprintf(aline, sizeof(aline),
                                  "data: {\"object\":\"chat.completion.attest\","
                                  "\"attest\":%s}\n\n", aj);
                if (ln > 0 && ln < (int)sizeof(aline))
                    vhttp_stream_write(req->conn, aline, (size_t)ln);
                free(aj);
            }
        }
        vhttp_stream_done(req->conn);
    }
    vb_release(b, n);
    req->rc = 0;
    return 0;
}

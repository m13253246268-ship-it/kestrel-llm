/**
 * vllm_transformer.c - Real Transformer Decoder Implementation
 *
 * Implements a full LLaMA-style transformer decoder:
 *   - RMS LayerNorm
 *   - Rotary Position Embedding (RoPE)
 *   - Multi-Head Causal Attention with KV-Cache
 *   - SwiGLU Feed-Forward Network
 *   - Residual connections
 *   - Temperature + Top-P sampling
 */

#include "vllm_superpos.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "vllm_util.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================
 * RMS LayerNorm: y = x * w / sqrt(mean(x^2) + eps)
 * ================================================================ */
void rms_norm(float *out, const float *x, const float *weight,
              int dim, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    float rms = 1.0f / sqrtf(ss / (float)dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * rms * weight[i];
}

/* ================================================================
 * Matrix-vector multiply: y = W @ x
 * W stored row-major: W[row * cols + col], x is input vector
 * y[row] = sum_col(W[row*cols + col] * x[col])
 * ================================================================ */
void matvec(float *out, const float *W, const float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            sum += W[r * cols + c] * x[c];
        }
        out[r] = sum;
    }
}

/* ================================================================
 * RoPE: Rotary Position Embedding
 * Applies rotation to pairs of dimensions in Q and K.
 * For dim pair (2i, 2i+1): rotate by angle pos * theta^(-2i/d)
 * ================================================================ */
void rope_apply(float *q, float *k, int head_dim, int pos, float theta) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(theta, (float)i / (float)head_dim);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        /* Apply to Q */
        float q0 = q[i];
        float q1 = q[i + 1];
        q[i]     = q0 * cos_a - q1 * sin_a;
        q[i + 1] = q1 * cos_a + q0 * sin_a;

        /* Apply to K */
        float k0 = k[i];
        float k1 = k[i + 1];
        k[i]     = k0 * cos_a - k1 * sin_a;
        k[i + 1] = k1 * cos_a + k0 * sin_a;
    }
}

/* ================================================================
 * Scaled Dot-Product Attention with Causal Mask & KV-Cache
 *
 * For position `cur_pos` (the token we're generating):
 *   1. Store Q[cur_pos], K[cur_pos], V[cur_pos] into cache
 *   2. Compute scores = Q[cur_pos] @ K[0..cur_pos]^T / sqrt(head_dim)
 *   3. Apply causal mask (already implicit from cache range)
 *   4. softmax(scores)
 *   5. output = scores @ V[0..cur_pos]
 * ================================================================ */
void causal_attention(float *output, const float *q, float *k_cache,
                       float *v_cache, int cache_len, int cur_pos,
                       int num_heads, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int seq_len = cache_len; /* number of cached positions */
    int kv_dim = num_heads * head_dim;

    /* Q should be at current position, so cache_len is the number
       of previously cached tokens. We compute attention over all seq_len tokens. */
    for (int h = 0; h < num_heads; h++) {
        const float *qh = q + h * head_dim;

        float scores[TINY_MAX_SEQ_LEN];
        float max_score = -1e9f;
        for (int t = 0; t < seq_len; t++) {
            const float *kh = k_cache + t * kv_dim + h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += qh[d] * kh[d];
            }
            scores[t] = dot * scale;
            if (scores[t] > max_score) max_score = scores[t];
        }

        /* Softmax */
        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        for (int t = 0; t < seq_len; t++) scores[t] /= sum_exp;

        /* Weighted sum of V */
        for (int d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                const float *vh = v_cache + t * kv_dim + h * head_dim;
                val += scores[t] * vh[d];
            }
            output[h * head_dim + d] = val;
        }
    }
}

/* ================================================================
 * SwiGLU FFN: down @ (silu(gate @ x) * (up @ x))
 * ================================================================ */
static float silu(float x) {
    return x / (1.0f + expf(-x));
}

void swiglu_ffn(float *out, const float *x,
                const float *gate_W, const float *up_W, const float *down_W,
                int hidden_dim, int ffn_dim) {
    float *gate = malloc(ffn_dim * sizeof(float));
    float *up   = malloc(ffn_dim * sizeof(float));

    matvec(gate, gate_W, x, ffn_dim, hidden_dim);
    matvec(up,   up_W,   x, ffn_dim, hidden_dim);

    for (int i = 0; i < ffn_dim; i++) {
        gate[i] = silu(gate[i]) * up[i];
    }

    matvec(out, down_W, gate, hidden_dim, ffn_dim);

    free(gate);
    free(up);
}

/* ================================================================
 * Forward one transformer layer
 *
 * Structure:
 *   residual = x
 *   x = rms_norm(x, attn_norm)
 *   q = Q_proj(x), k = K_proj(x), v = V_proj(x)
 *   rope(q, k, pos)
 *   store k, v into cache
 *   attn_out = causal_attention(q, k_cache, v_cache, pos)
 *   attn_out = O_proj(attn_out)
 *   x = residual + attn_out
 *   residual = x
 *   x = rms_norm(x, ffn_norm)
 *   x = swiglu(x, gate, up, down)
 *   x = residual + x
 * ================================================================ */
void transformer_layer_forward(LayerState *st, LayerWeights *w,
                                int pos, int num_heads, int head_dim,
                                int hidden_dim, int ffn_dim) {
    int kv_dim = num_heads * head_dim;
    float *x = st->hidden;
    float *residual = malloc(hidden_dim * sizeof(float));
    float *normed = malloc(hidden_dim * sizeof(float));

    /* --- Attention Block --- */
    memcpy(residual, x, hidden_dim * sizeof(float));
    rms_norm(normed, x, w->attn_norm, hidden_dim, 1e-6f);

    /* Q, K, V projections */
    matvec(st->q, w->q_weight, normed, kv_dim, hidden_dim);
    matvec(st->k, w->k_weight, normed, kv_dim, hidden_dim);
    matvec(st->v, w->v_weight, normed, kv_dim, hidden_dim);

    /* RoPE */
    for (int h = 0; h < num_heads; h++) {
        rope_apply(st->q + h * head_dim, st->k + h * head_dim,
                   head_dim, pos, TINY_ROPE_THETA);
    }

    /* Store K, V into cache */
    int cache_off = st->cache_len * kv_dim;
    memcpy(st->k_cache + cache_off, st->k, kv_dim * sizeof(float));
    memcpy(st->v_cache + cache_off, st->v, kv_dim * sizeof(float));
    int cur_cache_len = st->cache_len;
    st->cache_len++;

    /* Causal attention with cached K,V */
    causal_attention(st->attn_out, st->q, st->k_cache, st->v_cache,
                     cur_cache_len, pos, num_heads, head_dim);

    /* Output projection */
    matvec(normed, w->o_weight, st->attn_out, hidden_dim, kv_dim);

    /* Residual */
    for (int i = 0; i < hidden_dim; i++) x[i] = residual[i] + normed[i];

    /* --- FFN Block --- */
    memcpy(residual, x, hidden_dim * sizeof(float));
    rms_norm(normed, x, w->ffn_norm, hidden_dim, 1e-6f);

    swiglu_ffn(st->ffn_down, normed,
               w->gate_weight, w->up_weight, w->down_weight,
               hidden_dim, ffn_dim);

    /* Residual */
    for (int i = 0; i < hidden_dim; i++) x[i] = residual[i] + st->ffn_down[i];

    free(residual);
    free(normed);
}

/* ================================================================
 * Forward entire model for next token
 * ================================================================ */
void model_forward(InferenceState *st, int token_id) {
    int dim = TINY_HIDDEN_DIM;

    /* Token embedding */
    const float *emb = st->weights.token_embed + token_id * dim;
    memcpy(st->layers[0].hidden, emb, dim * sizeof(float));
    /* Copy embedding to all layer hidden states */
    for (int l = 1; l < TINY_NUM_LAYERS; l++) {
        memcpy(st->layers[l].hidden, emb, dim * sizeof(float));
    }

    /* Forward through layers */
    for (int l = 0; l < TINY_NUM_LAYERS; l++) {
        transformer_layer_forward(&st->layers[l], &st->weights.layers[l],
                                  st->seq_len,
                                  TINY_NUM_HEADS, TINY_HEAD_DIM,
                                  TINY_HIDDEN_DIM, TINY_FFN_DIM);
        /* Pass hidden state to next layer */
        if (l + 1 < TINY_NUM_LAYERS) {
            memcpy(st->layers[l + 1].hidden, st->layers[l].hidden,
                   dim * sizeof(float));
        }
    }

    /* Final RMS norm */
    float *final_hidden = st->layers[TINY_NUM_LAYERS - 1].hidden;
    float normed[TINY_HIDDEN_DIM];
    rms_norm(normed, final_hidden, st->weights.final_norm, dim, 1e-6f);

    /* LM head: logits = W_lm_head @ x */
    matvec(st->logits, st->weights.lm_head, normed, TINY_VOCAB_SIZE, dim);

    st->seq_len++;
}

/* ================================================================
 * Temperature + Top-P Sampling
 * ================================================================ */

static int compare_floats_desc(const void *a, const void *b) {
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa < fb) - (fa > fb);
}

/* (index, probability) pair used by the top-p sampling heap */
typedef struct { int idx; float prob; } IdxProb;

/* Sift-down for the top-p max-heap (P2, 2026-08-29). Priority: probability
 * descending, then index ascending (deterministic tie-break). */
static void idxprob_sift_down(IdxProb *a, int root, int n) {
    for (;;) {
        int l = root * 2 + 1;
        if (l >= n) return;
        int r = l + 1;
        int m = l;
        if (r < n) {
            int take_r = (a[r].prob > a[m].prob) ||
                         (a[r].prob == a[m].prob && a[r].idx < a[m].idx);
            if (take_r) m = r;
        }
        int swap = (a[m].prob > a[root].prob) ||
                   (a[m].prob == a[root].prob && a[m].idx < a[root].idx);
        if (!swap) return;
        IdxProb t = a[root]; a[root] = a[m]; a[m] = t;
        root = m;
    }
}

int sample_token(const float *logits, int vocab_size,
                  float temperature, float top_p) {
    return sample_token_p(logits, vocab_size, temperature, top_p, 0.0f);
}

int sample_token_p(const float *logits, int vocab_size,
                   float temperature, float top_p, float min_p) {
    return sample_token_pk(logits, vocab_size, temperature, top_p, min_p, 0);
}

/* Top-K 截断：只保留概率最高的 K 个 token（其余置 0）。
 * top_k <= 0 或 >= vocab_size 时为空操作。用容量 K 的最小堆，
 * 复杂度 O(vocab + vocab·logK)。Qwen3 thinking 档推荐 top_k=20；
 * 默认 0（关闭）以保持既有行为不变。 */
static void apply_top_k(float *probs, int vocab_size, int top_k) {
    if (top_k <= 0 || top_k >= vocab_size) return;

    static _Thread_local IdxProb *s_heap = NULL;
    static _Thread_local int s_cap = 0;
    if (top_k > s_cap) {
        IdxProb *nh = (IdxProb *)realloc(s_heap, (size_t)top_k * sizeof(IdxProb));
        if (!nh) return;                 /* 分配失败：退化为不做 top-k 截断 */
        s_heap = nh;
        s_cap = top_k;
    }

    int n = 0;                           /* 堆内元素个数（<= top_k） */
    for (int i = 0; i < vocab_size; i++) {
        float p = probs[i];
        if (p <= 0.0f) continue;         /* 已被抑制的 token 不参与 */
        if (n < top_k) {
            int c = n++;
            s_heap[c].idx = i; s_heap[c].prob = p;
            while (c > 0) {              /* sift up（最小堆） */
                int par = (c - 1) / 2;
                if (s_heap[par].prob <= s_heap[c].prob) break;
                IdxProb t = s_heap[par]; s_heap[par] = s_heap[c]; s_heap[c] = t;
                c = par;
            }
        } else if (p > s_heap[0].prob) {
            s_heap[0].idx = i; s_heap[0].prob = p;
            for (int r = 0;;) {          /* sift down（最小堆） */
                int l = r * 2 + 1;
                if (l >= top_k) break;
                int rr = l + 1, m = l;
                if (rr < top_k && s_heap[rr].prob < s_heap[m].prob) m = rr;
                if (s_heap[m].prob >= s_heap[r].prob) break;
                IdxProb t = s_heap[m]; s_heap[m] = s_heap[r]; s_heap[r] = t;
                r = m;
            }
        }
    }

    /* 堆内即被保留的 K 个：全部清零后回填（n 可能 < top_k）。 */
    for (int i = 0; i < vocab_size; i++) probs[i] = 0.0f;
    for (int i = 0; i < n; i++) probs[s_heap[i].idx] = s_heap[i].prob;
}

int sample_token_pk(const float *logits, int vocab_size,
                    float temperature, float top_p, float min_p, int top_k) {
    /* Greedy fast path (OpenAI semantics: temperature <= 0 = deterministic
     * argmax; top_p <= 0 = no nucleus sampling). Avoids the O(n log n) qsort
     * over the whole vocabulary on every decode step. */
    if (temperature <= 0.0f || top_p <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    /* Reusable per-thread scratch: allocating/freeing the full-vocabulary
     * buffers on every decode step is expensive (large CRT heap allocs on
     * first touch measured ~5.6 s/token). Serving workers and CLI decode
     * reuse these across steps. */
#define ST_TLS _Thread_local
    static ST_TLS float *s_probs = NULL;
    static ST_TLS IdxProb *s_sorted = NULL;
    static ST_TLS int s_cap = 0;
    if (vocab_size > s_cap) {
        free(s_probs);
        free(s_sorted);
        s_probs = (float *)malloc((size_t)vocab_size * sizeof(float));
        s_sorted = (IdxProb *)malloc((size_t)vocab_size * sizeof(IdxProb));
        if (!s_probs || !s_sorted) {
            s_cap = 0;
            int best = 0;
            for (int i = 1; i < vocab_size; i++)
                if (logits[i] > logits[best]) best = i;
            return best;
        }
        s_cap = vocab_size;
    }
    float *probs = s_probs;

    /* Softmax with temperature */
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_logit) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; i++) probs[i] /= sum;

    /* Suppress special tokens in output */
    probs[0] = 0.0f;  /* <unk> */
    probs[1] = 0.0f;  /* <s> */
    probs[3] = 0.0f;  /* <pad> */

    /* Top-K 截断（Qwen3 thinking 档推荐 top_k=20；默认关闭）。放在 min-p /
     * top-p 之前，与 vLLM 的 top_k → top_p → min_p 顺序一致。 */
    apply_top_k(probs, vocab_size, top_k);

    /* Min-P filtering (min_p > 0): keep only tokens at or above
     * min_p * max_prob. Uses the special-suppressed probabilities, applied
     * before nucleus top-p so both filters compose (min-p first, then top-p). */
    if (min_p > 0.0f) {
        float maxp = probs[0];
        for (int i = 1; i < vocab_size; i++) if (probs[i] > maxp) maxp = probs[i];
        float thresh = min_p * maxp;
        for (int i = 0; i < vocab_size; i++) if (probs[i] < thresh) probs[i] = 0.0f;
    }

    /* Top-P filtering */
    if (top_p < 1.0f) {
        /* P2 (2026-08-29): bounded max-heap partial sort replaces the full
         * qsort. On RK3588 the qsort over vocab=151936 measured ~24 ms/token
         * (TPOT 74.1 vs 49.8 greedy); the heap is O(n + K log n), ~2 ms.
         * Same top-p semantics: the descending-probability prefix whose
         * cumulative sum first reaches top_p is kept, everything else is
         * zeroed. Ties break by index (deterministic); sampling is stochastic
         * so the tie-order difference vs glibc qsort is unobservable. */
        IdxProb *sorted = s_sorted;
        for (int i = 0; i < vocab_size; i++) {
            sorted[i].idx = i;
            sorted[i].prob = probs[i];
        }
        /* heapify (max-heap by prob desc, idx asc) */
        for (int i = vocab_size / 2 - 1; i >= 0; i--)
            idxprob_sift_down(sorted, i, vocab_size);

        /* Extract the top prefix until the cumulative mass reaches top_p;
         * each extraction moves the current max to the back (kept region). */
        float cumsum = 0.0f;
        int n_left = vocab_size;
        while (n_left > 0 && cumsum < top_p) {
            IdxProb mx = sorted[0];
            sorted[0] = sorted[n_left - 1];
            sorted[n_left - 1] = mx;
            cumsum += mx.prob;
            n_left--;
            idxprob_sift_down(sorted, 0, n_left);
        }
        /* Zero out the rejected tail (sorted[0..n_left) still holds them) */
        for (int i = 0; i < n_left; i++)
            probs[sorted[i].idx] = 0.0f;
    }

    /* Renormalize and sample */
    sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) sum += probs[i];

    float r = (float)rand() / (float)RAND_MAX * sum;
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (r <= cumsum) {
            return i;
        }
    }

    /* Fallback: argmax */
    int best = 0;
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

/* ================================================================
 * Weight Initialization - Deterministic but meaningful
 * Uses sinusoidal patterns and Glorot-like scaling to produce
 * non-trivial weights that produce interesting token sequences.
 * ================================================================ */

static float init_weight(int row, int col, int fan_in, int fan_out, int seed) {
    /* Mix multiple frequencies for pseudo-random but deterministic weights */
    float a = sinf((float)(row * 127 + col * 311 + seed) * 0.0174533f);
    float b = sinf((float)(row * 269 + col * 173 + seed * 7) * 0.0314159f);
    float c = cosf((float)(row * 419 + col * 251 + seed * 13) * 0.0098696f);
    float val = (a * 0.6f + b * 0.3f + c * 0.1f);
    /* Glorot uniform scaling */
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    return val * limit;
}

void weights_init(ModelWeights *weights) {
    memset(weights, 0, sizeof(ModelWeights));

    int dim = TINY_HIDDEN_DIM;
    int ffn = TINY_FFN_DIM;
    int vocab = TINY_VOCAB_SIZE;

    /* Token embeddings: character n-gram based initialization.
     * Tokens sharing character substrings get correlated embeddings,
     * creating a structured embedding space even without training. */
    for (int i = 0; i < vocab; i++) {
        const char *ts = tokenizer_decode_direct ? tokenizer_decode_direct(i) : NULL;
        float norm = 0.0f;

        for (int j = 0; j < dim; j++) {
            float val = 0.0f;
            if (ts) {
                /* Hash character bigrams into embedding dimensions */
                int slen = (int)strlen(ts);
                for (int k = 0; k < slen; k++) {
                    int bi = (k < slen - 1) ?
                        ((int)ts[k] * 131 + (int)ts[k+1] * 271 + j * 37) :
                        ((int)ts[k] * 379 + j * 83);
                    val += sinf((float)(bi & 0xFFFF) * 0.001533f +
                                cosf((float)(bi >> 8) * 0.002711f) * 0.3f);
                }
                val /= (float)(slen + 1);
            } else {
                val = sinf((float)(i * 739 + j * 457) * 0.01234f) *
                      cosf((float)(i * 167 + j * 293) * 0.02345f);
            }
            weights->token_embed[i * dim + j] = val * 0.03f;
            norm += val * val;
        }

        /* L2 normalize embeddings */
        norm = sqrtf(norm);
        if (norm > 1e-8f) {
            for (int j = 0; j < dim; j++)
                weights->token_embed[i * dim + j] /= norm;
        }
    }

    /* Final norm: initialize to 1.0 */
    for (int i = 0; i < dim; i++) weights->final_norm[i] = 1.0f;

    /* LM head: tied with embedding but with perturbation */
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < vocab; j++) {
            weights->lm_head[i * vocab + j] =
                weights->token_embed[j * dim + i] * 0.95f +
                sinf((float)(i * 839 + j * 211) * 0.04321f) * 0.005f;
        }
    }

    /* Per-layer weights */
    for (int l = 0; l < TINY_NUM_LAYERS; l++) {
        LayerWeights *w = &weights->layers[l];
        int seed = l * 1000;

        /* Attention weights */
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                w->q_weight[i * dim + j] = init_weight(i, j, dim, dim, seed + 1);
                w->k_weight[i * dim + j] = init_weight(i, j, dim, dim, seed + 2);
                w->v_weight[i * dim + j] = init_weight(i, j, dim, dim, seed + 3);
                w->o_weight[i * dim + j] = init_weight(i, j, dim, dim, seed + 4);
            }
        }

        /* Norms */
        for (int i = 0; i < dim; i++) {
            w->attn_norm[i] = 1.0f;
            w->ffn_norm[i]  = 1.0f;
        }

        /* SwiGLU weights */
        for (int i = 0; i < ffn; i++) {
            for (int j = 0; j < dim; j++) {
                w->gate_weight[i * dim + j] = init_weight(i, j, dim, ffn, seed + 5);
                w->up_weight[i * dim + j]   = init_weight(i, j, dim, ffn, seed + 6);
            }
        }
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < ffn; j++) {
                w->down_weight[i * ffn + j] = init_weight(i, j, ffn, dim, seed + 7);
            }
        }
    }
}

/* ================================================================
 * Weight Serialization
 * ================================================================ */
int weights_save(const ModelWeights *weights, const char *filename) {
    FILE *f = st_fopen(filename, "wb");
    if (!f) return -1;
    size_t written = fwrite(weights, sizeof(ModelWeights), 1, f);
    fclose(f);
    return (written == 1) ? 0 : -1;
}

int weights_load(ModelWeights *weights, const char *filename) {
    FILE *f = st_fopen(filename, "rb");
    if (!f) return -1;
    size_t read = fread(weights, sizeof(ModelWeights), 1, f);
    fclose(f);
    return (read == 1) ? 0 : -1;
}

/* ================================================================
 * Full Inference Pipeline
 * ================================================================ */

InferenceState* inference_create(void) {
    InferenceState *st = malloc(sizeof(InferenceState));
    if (!st) return NULL;
    memset(st, 0, sizeof(*st));
    weights_init(&st->weights);
    st->seq_len = 0;
    return st;
}

void inference_free(InferenceState *st) {
    if (st) free(st);
}

char* inference_generate(InferenceState *st, Tokenizer *tok,
                          const char *prompt, int max_new_tokens,
                          float temperature, float top_p) {
    /* Reset KV caches */
    for (int l = 0; l < TINY_NUM_LAYERS; l++) {
        st->layers[l].cache_len = 0;
    }
    st->seq_len = 0;

    /* Encode prompt */
    int prompt_ids[TINY_MAX_SEQ_LEN];
    int num_prompt = tokenizer_encode(tok, prompt, prompt_ids, TINY_MAX_SEQ_LEN);

    /* Prefill: process prompt tokens through the model */
    for (int i = 0; i < num_prompt; i++) {
        model_forward(st, prompt_ids[i]);
    }

    /* Generate new tokens autoregressively */
    int output_ids[TINY_MAX_SEQ_LEN];
    int num_output = 0;

    for (int t = 0; t < max_new_tokens && st->seq_len < TINY_MAX_SEQ_LEN; t++) {
        int next_token = sample_token(st->logits, TINY_VOCAB_SIZE,
                                       temperature, top_p);

        /* Stop if EOS */
        if (next_token == 2) { /* </s> */
            break;
        }

        output_ids[num_output++] = next_token;
        model_forward(st, next_token);
    }

    /* Decode output tokens to string */
    /* Allocate generously */
    int bufsize = num_output * 16 + 1;
    char *result = malloc(bufsize);
    if (!result) return NULL;
    result[0] = '\0';

    for (int i = 0; i < num_output; i++) {
        const char *token_str = tokenizer_decode(tok, output_ids[i]);

        /* Handle special tokens */
        if (output_ids[i] < 4) continue; /* skip <unk>, <s>, </s>, <pad> */

        /* Check if this token looks like a continuation (starts with lowercase
           or is punctuation that should be attached) */
        int is_punct = (strlen(token_str) == 1 &&
                        (token_str[0] == '.' || token_str[0] == ',' ||
                         token_str[0] == '!' || token_str[0] == '?' ||
                         token_str[0] == ';' || token_str[0] == ':' ||
                         token_str[0] == '\'' || token_str[0] == '"' ||
                         token_str[0] == ')' || token_str[0] == ']' ||
                         token_str[0] == '}'));

        int is_suffix = (token_str[0] == 'i' && token_str[1] == 'n' &&
                         token_str[2] == 'g') ||
                        (token_str[0] == 'e' && token_str[1] == 'd') ||
                        (token_str[0] == 'l' && token_str[1] == 'y') ||
                        (token_str[0] == 's' && token_str[1] == '\0');

        if (is_punct || is_suffix) {
            /* Attach directly */
            strcat(result, token_str);
        } else {
            /* Add space before */
            if (strlen(result) > 0) strcat(result, " ");
            strcat(result, token_str);
        }
    }

    return result;
}

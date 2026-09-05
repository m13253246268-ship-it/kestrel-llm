/* ================================================================
 * vllm_gguf.c - GGUF (llama.cpp Unified Format) loader
 *
 * 文件结构（小端）：
 *   magic[4]="GGUF" | u32 version | u64 n_tensors | u64 n_kv
 *   KV 区    : n_kv × { key( u64 len + bytes ), type(i32), value }
 *   张量目录 : n_tensors × { name( u64 len + bytes ), u32 n_dims,
 *              i64 dims[n_dims], i32 type, u64 offset }
 *   数据区   : 从 alignment 对齐（general.alignment KV 或默认 32）
 *
 * 反量化覆盖 Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 + Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K
 * （block 布局与 llama.cpp ggml-common.h 对齐）；F16/F32/BF16 直接转。
 * IQ 系列 / TQ / MXFP4 极稀有，不支持（明确报错）。
 *
 * 架构：qwen2/qwen3/llama/mistral → STModelConfig；张量名 llama 风格
 * （token_embd / blk.N.attn_q / blk.N.ffn_gate / output_norm / output）。
 * llama/mistral 无 q_norm/mrope → has_q_norm=0 / has_mrope=0，引擎内核
 * 按条件分支已支持（零内核改动）。
 *
 * 加载布局与 safetensors 路径完全一致：st_weights_alloc_layers_q8ffn →
 * 逐张量 dequant(F32) → f32_to_q8_0/q4_0 → repack（8x8 tiled / 4x4）。
 * 大张量（token_embd / lm_head）按行 chunked，避免大 F32 临时缓冲。
 * ================================================================ */
#include "vllm_gguf.h"
#include "vllm_platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define GGUF_MAGIC0 0x46554747u   /* "GGUF" LE */
#define QK_K 256
#define K_SCALE_SIZE 12

/* 引擎 Q8_0/Q4_0 块字节（34/18B per 32 元素），与 vllm_safetensors.c 同式 */
#define Q8_BYTES(n) ((size_t)((n) + 31) / 32 * 34)
#define Q4_BYTES(n) ((size_t)((n) + 31) / 32 * 18)

extern int g_st_q8_repack;
extern int g_st_q4_repack;

/* ---------------- f16/bf16 -> f32 ---------------- */
static inline float gf_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t u;
    if (exp == 0) {
        /* 非规格化 f16：值 = mant * 2^-24。必须规格化到 f32 普通数，
         * 否则 k-quant 超块 scale（常为 ~1e-5 的非规格化数）会变 0 */
        if (mant == 0) u = sign;
        else {
            uint32_t m = mant;
            int shift = 0;
            while (m < 0x400) { m <<= 1; shift++; }
            u = sign | ((113u - (uint32_t)shift) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1F) u = sign | 0x7F800000u | (mant << 13);
    else u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float f; memcpy(&f, &u, 4); return f;
}
static inline float gf_bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f; memcpy(&f, &u, 4); return f;
}

/* ---------------- GGUF 文件结构 ---------------- */
typedef struct {
    st_mmap_t     m;
    uint64_t      n_kv, n_tensors;
    uint64_t      data_offset;
    size_t        alignment;

    struct GgufKV_ {
        char   *key;
        int32_t type;
        union { uint8_t u8; int8_t i8; uint16_t u16; int16_t i16;
                uint32_t u32; int32_t i32; float f32; uint8_t b;
                uint64_t u64; int64_t i64; double f64; } v;
        char   *str;
        int32_t arr_type;
        uint64_t arr_n;
        void   *arr_data;
        char  **arr_str;
    } *kv;

    struct GgufTens_ {
        char    *name;
        uint32_t n_dims;
        int64_t  ne[4];
        int32_t  type;
        uint64_t offset;
    } *tens;
} GgufFile;

static const uint8_t *gf_ptr(const GgufFile *f, uint64_t off) {
    return (const uint8_t *)f->m.data + off;
}

static char *gf_strndup(const char *s, size_t n) {
    char *r = (char *)malloc(n + 1);
    if (!r) return NULL;
    memcpy(r, s, n);
    r[n] = '\0';
    return r;
}

static void gguf_free(GgufFile *f);

int gguf_is_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    uint32_t m = 0;
    int ok = (fread(&m, 1, 4, f) == 4);
    fclose(f);
    return ok && m == GGUF_MAGIC0;
}

static int gguf_parse(const char *path, GgufFile *f) {
    memset(f, 0, sizeof(*f));
    if (st_mmap_open(&f->m, 0, path) != 0) {
        fprintf(stderr, "[GGUF] mmap failed: %s\n", path);
        return -1;
    }
    const uint8_t *p = (const uint8_t *)f->m.data;
    uint64_t pos = 0;
    if (f->m.len < 24) { st_mmap_close(&f->m); return -1; }
    uint32_t magic; memcpy(&magic, p, 4); pos = 4;
    if (magic != GGUF_MAGIC0) { st_mmap_close(&f->m); return -1; }
    uint32_t ver; memcpy(&ver, p + pos, 4); pos += 4;
    if (ver > 3) {
        fprintf(stderr, "[GGUF] version %u unsupported\n", ver);
        st_mmap_close(&f->m); return -1;
    }
    memcpy(&f->n_tensors, p + pos, 8); pos += 8;
    memcpy(&f->n_kv,      p + pos, 8); pos += 8;
    if (f->n_kv > 4096 || f->n_tensors > 8192) {
        fprintf(stderr, "[GGUF] implausible n_kv=%llu n_tensors=%llu\n",
                (unsigned long long)f->n_kv, (unsigned long long)f->n_tensors);
        st_mmap_close(&f->m); return -1;
    }
    f->kv   = calloc((size_t)f->n_kv, sizeof(*f->kv));
    f->tens = calloc((size_t)f->n_tensors, sizeof(*f->tens));
    if (!f->kv || !f->tens) { st_mmap_close(&f->m); return -1; }
    f->alignment = 32;

#define NEED(n) do { if (pos + (n) > f->m.len) goto fail; } while (0)
    for (uint64_t i = 0; i < f->n_kv; i++) {
        uint64_t klen; NEED(8); memcpy(&klen, p + pos, 8); pos += 8;
        NEED(klen); f->kv[i].key = gf_strndup((const char *)p + pos, (size_t)klen);
        pos += klen;
        NEED(4); int32_t t; memcpy(&t, p + pos, 4); pos += 4;
        f->kv[i].type = t;
        switch (t) {
        case 0:  NEED(1);  memcpy(&f->kv[i].v.u8,  p + pos, 1); pos += 1; break;
        case 1:  NEED(1);  memcpy(&f->kv[i].v.i8,  p + pos, 1); pos += 1; break;
        case 2:  NEED(2);  memcpy(&f->kv[i].v.u16, p + pos, 2); pos += 2; break;
        case 3:  NEED(2);  memcpy(&f->kv[i].v.i16, p + pos, 2); pos += 2; break;
        case 4:  NEED(4);  memcpy(&f->kv[i].v.u32, p + pos, 4); pos += 4; break;
        case 5:  NEED(4);  memcpy(&f->kv[i].v.i32, p + pos, 4); pos += 4; break;
        case 6:  NEED(4);  memcpy(&f->kv[i].v.f32, p + pos, 4); pos += 4; break;
        case 7:  NEED(1);  memcpy(&f->kv[i].v.b,   p + pos, 1); pos += 1; break;
        case 10: NEED(8);  memcpy(&f->kv[i].v.u64, p + pos, 8); pos += 8; break;
        case 11: NEED(8);  memcpy(&f->kv[i].v.i64, p + pos, 8); pos += 8; break;
        case 12: NEED(8);  memcpy(&f->kv[i].v.f64, p + pos, 8); pos += 8; break;
        case 8: {
            uint64_t slen; NEED(8); memcpy(&slen, p + pos, 8); pos += 8;
            NEED(slen);
            f->kv[i].str = gf_strndup((const char *)p + pos, (size_t)slen);
            pos += slen;
            break;
        }
        case 9: {
            NEED(4); memcpy(&f->kv[i].arr_type, p + pos, 4); pos += 4;
            NEED(8); memcpy(&f->kv[i].arr_n, p + pos, 8); pos += 8;
            uint64_t an = f->kv[i].arr_n;
            if (an > 4 * 1024 * 1024) goto fail;
            if (f->kv[i].arr_type == 8) {
                f->kv[i].arr_str = calloc((size_t)an, sizeof(char *));
                if (!f->kv[i].arr_str) goto fail;
                for (uint64_t j = 0; j < an; j++) {
                    uint64_t slen; NEED(8); memcpy(&slen, p + pos, 8); pos += 8;
                    NEED(slen);
                    f->kv[i].arr_str[j] = gf_strndup((const char *)p + pos, (size_t)slen);
                    pos += slen;
                }
            } else {
                size_t esz = 0;
                switch (f->kv[i].arr_type) {
                case 0: case 1: case 7: esz = 1; break;
                case 2: case 3: esz = 2; break;
                case 4: case 5: case 6: esz = 4; break;
                case 10: case 11: case 12: esz = 8; break;
                default: goto fail;
                }
                size_t bytes = (size_t)an * esz;
                f->kv[i].arr_data = malloc(bytes);
                if (!f->kv[i].arr_data) goto fail;
                NEED(bytes);
                memcpy(f->kv[i].arr_data, p + pos, bytes);
                pos += bytes;
            }
            break;
        }
        default: goto fail;
        }
        if (strcmp(f->kv[i].key, "general.alignment") == 0 && f->kv[i].type == 4)
            f->alignment = (size_t)f->kv[i].v.u32;
    }
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        uint64_t nlen; NEED(8); memcpy(&nlen, p + pos, 8); pos += 8;
        NEED(nlen); f->tens[i].name = gf_strndup((const char *)p + pos, (size_t)nlen);
        pos += nlen;
        NEED(4); uint32_t nd; memcpy(&nd, p + pos, 4); pos += 4;
        f->tens[i].n_dims = nd;
        if (nd > 4) goto fail;
        for (uint32_t d = 0; d < nd; d++) {
            NEED(8); memcpy(&f->tens[i].ne[d], p + pos, 8); pos += 8;
        }
        for (uint32_t d = nd; d < 4; d++) f->tens[i].ne[d] = 1;
        NEED(4); memcpy(&f->tens[i].type, p + pos, 4); pos += 4;
        NEED(8); memcpy(&f->tens[i].offset, p + pos, 8); pos += 8;
    }
    f->data_offset = (pos + f->alignment - 1) & ~(uint64_t)(f->alignment - 1);
    return 0;
fail:
    fprintf(stderr, "[GGUF] parse error in %s\n", path);
    gguf_free(f);
    return -1;
}

static void gguf_free(GgufFile *f) {
    for (uint64_t i = 0; i < f->n_kv; i++) {
        free(f->kv[i].key);
        free(f->kv[i].str);
        free(f->kv[i].arr_data);
        if (f->kv[i].arr_str) {
            for (uint64_t j = 0; j < f->kv[i].arr_n; j++) free(f->kv[i].arr_str[j]);
            free(f->kv[i].arr_str);
        }
    }
    free(f->kv);
    for (uint64_t i = 0; i < f->n_tensors; i++) free(f->tens[i].name);
    free(f->tens);
    st_mmap_close(&f->m);
    memset(f, 0, sizeof(*f));
}

/* ---------------- KV helpers ---------------- */
static const struct GgufKV_ *gf_find_kv(const GgufFile *f, const char *key) {
    for (uint64_t i = 0; i < f->n_kv; i++)
        if (strcmp(f->kv[i].key, key) == 0) return &f->kv[i];
    return NULL;
}
static int64_t gf_kv_i64(const GgufFile *f, const char *key, int64_t dflt) {
    const struct GgufKV_ *k = gf_find_kv(f, key);
    if (!k) return dflt;
    switch (k->type) {
    case 4: return k->v.u32;
    case 5: return k->v.i32;
    case 10: return (int64_t)k->v.u64;
    case 11: return k->v.i64;
    case 6:  return (int64_t)k->v.f32;
    case 7:  return k->v.b ? 1 : 0;
    case 0:  return k->v.u8;
    case 1:  return k->v.i8;
    case 2:  return k->v.u16;
    case 3:  return k->v.i16;
    default: return dflt;
    }
}
static float gf_kv_f32(const GgufFile *f, const char *key, float dflt) {
    const struct GgufKV_ *k = gf_find_kv(f, key);
    if (!k) return dflt;
    if (k->type == 6) return k->v.f32;
    if (k->type == 5) return (float)k->v.i32;
    if (k->type == 11) return (float)k->v.i64;
    if (k->type == 4) return (float)k->v.u32;
    return dflt;
}
static const char *gf_kv_str(const GgufFile *f, const char *key) {
    const struct GgufKV_ *k = gf_find_kv(f, key);
    return (k && k->type == 8) ? k->str : NULL;
}
static uint64_t gf_kv_arr_count(const GgufFile *f, const char *key) {
    const struct GgufKV_ *k = gf_find_kv(f, key);
    return (k && k->type == 9) ? k->arr_n : 0;
}
static const char *gf_kv_arr_str(const GgufFile *f, const char *key, uint64_t i) {
    const struct GgufKV_ *k = gf_find_kv(f, key);
    if (!k || k->type != 9 || k->arr_type != 8 || i >= k->arr_n) return NULL;
    return k->arr_str[i];
}
static int gf_find_tensor(const GgufFile *f, const char *name) {
    for (uint64_t i = 0; i < f->n_tensors; i++)
        if (strcmp(f->tens[i].name, name) == 0) return (int)i;
    return -1;
}

/* ---------------- 反量化器（块布局与 llama.cpp 对齐） ---------------- */
typedef struct { uint16_t d; uint8_t qs[16]; }                        gf_block_q4_0;
typedef struct { uint16_t d, m; uint8_t qs[16]; }                     gf_block_q4_1;
typedef struct { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; }         gf_block_q5_0;
typedef struct { uint16_t d, m; uint8_t qh[4]; uint8_t qs[16]; }      gf_block_q5_1;
typedef struct { uint16_t d; int8_t qs[32]; }                         gf_block_q8_0;
typedef struct { uint16_t d, dmin; uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4]; } gf_block_q2_K;
typedef struct { uint8_t hmask[QK_K/8]; uint8_t qs[QK_K/4]; uint8_t scales[12]; uint16_t d; } gf_block_q3_K;
typedef struct { uint16_t d, dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; } gf_block_q4_K;
typedef struct { uint16_t d, dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qh[QK_K/8]; uint8_t qs[QK_K/2]; } gf_block_q5_K;
typedef struct { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; uint16_t d; } gf_block_q6_K;

static inline void gf_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4); }
}

/* 反量化一整行（n 个元素，src 指向该行数据起点）。qtype 见 GGUF_T_*。 */
static void gguf_dequant_row(int qtype, const uint8_t *src, float *dst, int64_t n) {
    switch (qtype) {
    case GGUF_T_F32:
        memcpy(dst, src, (size_t)n * 4);
        break;
    case GGUF_T_F16: {
        const uint16_t *p = (const uint16_t *)src;
        for (int64_t i = 0; i < n; i++) dst[i] = gf_f16_to_f32(p[i]);
        break;
    }
    case GGUF_T_BF16: {
        const uint16_t *p = (const uint16_t *)src;
        for (int64_t i = 0; i < n; i++) dst[i] = gf_bf16_to_f32(p[i]);
        break;
    }
    case GGUF_T_Q4_0: {
        const gf_block_q4_0 *x = (const gf_block_q4_0 *)src;
        int64_t nb = n / 32;
        for (int64_t i = 0; i < nb; i++) {
            const float d = gf_f16_to_f32(x[i].d);
            for (int j = 0; j < 16; ++j) {
                const int x0 = (x[i].qs[j] & 0x0F) - 8;
                const int x1 = (x[i].qs[j] >> 4) - 8;
                dst[i*32 + j]      = x0 * d;
                dst[i*32 + j + 16] = x1 * d;
            }
        }
        break;
    }
    case GGUF_T_Q4_1: {
        const gf_block_q4_1 *x = (const gf_block_q4_1 *)src;
        int64_t nb = n / 32;
        for (int64_t i = 0; i < nb; i++) {
            const float d = gf_f16_to_f32(x[i].d);
            const float m = gf_f16_to_f32(x[i].m);
            for (int j = 0; j < 16; ++j) {
                const int x0 = (x[i].qs[j] & 0x0F);
                const int x1 = (x[i].qs[j] >> 4);
                dst[i*32 + j]      = x0 * d + m;
                dst[i*32 + j + 16] = x1 * d + m;
            }
        }
        break;
    }
    case GGUF_T_Q5_0: {
        const gf_block_q5_0 *x = (const gf_block_q5_0 *)src;
        int64_t nb = n / 32;
        for (int64_t i = 0; i < nb; i++) {
            const float d = gf_f16_to_f32(x[i].d);
            uint32_t qh; memcpy(&qh, x[i].qh, 4);
            for (int j = 0; j < 16; ++j) {
                const int32_t x0 = ((x[i].qs[j] & 0x0F) | (((qh >> (j + 0)) << 4) & 0x10)) - 16;
                const int32_t x1 = ((x[i].qs[j] >> 4) | ((qh >> (j + 12)) & 0x10)) - 16;
                dst[i*32 + j]      = x0 * d;
                dst[i*32 + j + 16] = x1 * d;
            }
        }
        break;
    }
    case GGUF_T_Q5_1: {
        const gf_block_q5_1 *x = (const gf_block_q5_1 *)src;
        int64_t nb = n / 32;
        for (int64_t i = 0; i < nb; i++) {
            const float d = gf_f16_to_f32(x[i].d);
            const float m = gf_f16_to_f32(x[i].m);
            uint32_t qh; memcpy(&qh, x[i].qh, 4);
            for (int j = 0; j < 16; ++j) {
                const int32_t x0 = (x[i].qs[j] & 0x0F) | (((qh >> (j + 0)) << 4) & 0x10);
                const int32_t x1 = (x[i].qs[j] >> 4) | ((qh >> (j + 12)) & 0x10);
                dst[i*32 + j]      = x0 * d + m;
                dst[i*32 + j + 16] = x1 * d + m;
            }
        }
        break;
    }
    case GGUF_T_Q8_0: {
        const gf_block_q8_0 *x = (const gf_block_q8_0 *)src;
        int64_t nb = n / 32;
        for (int64_t i = 0; i < nb; i++) {
            const float d = gf_f16_to_f32(x[i].d);
            for (int j = 0; j < 32; j++) dst[i*32 + j] = x[i].qs[j] * d;
        }
        break;
    }
    case GGUF_T_Q8_1: {
        const uint16_t *dp = (const uint16_t *)src;
        const int8_t *qs = (const int8_t *)(src + 4);
        int64_t nb = n / 32;
        for (int64_t i = 0; i < nb; i++) {
            const float d = gf_f16_to_f32(dp[i * 2]);
            for (int j = 0; j < 32; j++) dst[i*32 + j] = qs[i*32 + j] * d;
        }
        break;
    }
    case GGUF_T_Q2_K: {
        const gf_block_q2_K *x = (const gf_block_q2_K *)src;
        int64_t nb = n / QK_K;
        for (int64_t i = 0; i < nb; i++) {
            const float d   = gf_f16_to_f32(x[i].d);
            const float min = gf_f16_to_f32(x[i].dmin);
            const uint8_t *q = x[i].qs;
            float *y = dst + i * QK_K;
            int is = 0;
            for (int m = 0; m < QK_K; m += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    uint8_t sc = x[i].scales[is++];
                    float dl = d * (sc & 0xF), ml = min * (sc >> 4);
                    for (int l = 0; l < 16; ++l) *y++ = dl * (float)(int8_t)((q[l] >> shift) & 3) - ml;
                    sc = x[i].scales[is++];
                    dl = d * (sc & 0xF); ml = min * (sc >> 4);
                    for (int l = 0; l < 16; ++l) *y++ = dl * (float)(int8_t)((q[l+16] >> shift) & 3) - ml;
                    shift += 2;
                }
                q += 32;
            }
        }
        break;
    }
    case GGUF_T_Q3_K: {
        const gf_block_q3_K *x = (const gf_block_q3_K *)src;
        int64_t nb = n / QK_K;
        const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
        for (int64_t i = 0; i < nb; i++) {
            const float d_all = gf_f16_to_f32(x[i].d);
            const uint8_t *q = x[i].qs, *hm = x[i].hmask;
            uint32_t aux[4];
            memcpy(aux, x[i].scales, 12);
            uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            const int8_t *scales = (const int8_t *)aux;
            float *y = dst + i * QK_K;
            int is = 0;
            uint8_t m = 1;
            for (int k = 0; k < QK_K; k += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    float dl = d_all * (scales[is++] - 32);
                    for (int l = 0; l < 16; ++l)
                        *y++ = dl * (float)((int8_t)((q[l+0] >> shift) & 3) - ((hm[l+0] & m) ? 0 : 4));
                    dl = d_all * (scales[is++] - 32);
                    for (int l = 0; l < 16; ++l)
                        *y++ = dl * (float)((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                    shift += 2;
                    m <<= 1;
                }
                q += 32;
            }
        }
        break;
    }
    case GGUF_T_Q4_K: {
        const gf_block_q4_K *x = (const gf_block_q4_K *)src;
        int64_t nb = n / QK_K;
        for (int64_t i = 0; i < nb; i++) {
            const uint8_t *q = x[i].qs;
            const float d   = gf_f16_to_f32(x[i].d);
            const float min = gf_f16_to_f32(x[i].dmin);
            float *y = dst + i * QK_K;
            int is = 0;
            uint8_t sc, m;
            for (int j = 0; j < QK_K; j += 64) {
                gf_get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
                const float d1 = d * sc, m1 = min * m;
                gf_get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
                const float d2 = d * sc, m2 = min * m;
                for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
                for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
                q += 32; is += 2;
            }
        }
        break;
    }
    case GGUF_T_Q5_K: {
        const gf_block_q5_K *x = (const gf_block_q5_K *)src;
        int64_t nb = n / QK_K;
        for (int64_t i = 0; i < nb; i++) {
            const uint8_t *ql = x[i].qs, *qh = x[i].qh;
            const float d   = gf_f16_to_f32(x[i].d);
            const float min = gf_f16_to_f32(x[i].dmin);
            float *y = dst + i * QK_K;
            int is = 0;
            uint8_t sc, m, u1 = 1, u2 = 2;
            for (int j = 0; j < QK_K; j += 64) {
                gf_get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
                const float d1 = d * sc, m1 = min * m;
                gf_get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
                const float d2 = d * sc, m2 = min * m;
                for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
                for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
                ql += 32; is += 2;
                u1 <<= 2; u2 <<= 2;
            }
        }
        break;
    }
    case GGUF_T_Q6_K: {
        const gf_block_q6_K *x = (const gf_block_q6_K *)src;
        int64_t nb = n / QK_K;
        for (int64_t i = 0; i < nb; i++) {
            const float d = gf_f16_to_f32(x[i].d);
            const uint8_t *ql = x[i].ql, *qh = x[i].qh;
            const int8_t *sc = x[i].scales;
            float *y = dst + i * QK_K;
            for (int m = 0; m < QK_K; m += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l / 16;
                    const float q1 = (float)((int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                    const float q2 = (float)((int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                    const float q3 = (float)((int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                    const float q4 = (float)((int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
                    y[l + 0]  = d * sc[is + 0] * q1;
                    y[l + 32] = d * sc[is + 2] * q2;
                    y[l + 64] = d * sc[is + 4] * q3;
                    y[l + 96] = d * sc[is + 6] * q4;
                }
                y += 128; ql += 64; qh += 32; sc += 8;
            }
        }
        break;
    }
    case GGUF_T_Q8_K: {
        const float *dp = (const float *)src;
        const int8_t *qs = (const int8_t *)(src + 4);
        int64_t nb = n / QK_K;
        for (int64_t i = 0; i < nb; i++) {
            const float d = dp[i * (QK_K / 16 + 1)];
            for (int j = 0; j < QK_K; j++) dst[i*QK_K + j] = qs[i*QK_K + j] * d;
        }
        break;
    }
    default:
        fprintf(stderr, "[GGUF] unsupported quant type %d\n", qtype);
        memset(dst, 0, (size_t)n * 4);
        break;
    }
}

/* 每 32 元素块（k-quants 为 QK_K=256 超块）的字节数 */
static size_t gguf_block_bytes(int qtype) {
    switch (qtype) {
    case GGUF_T_F32:  return 4;
    case GGUF_T_F16:
    case GGUF_T_BF16: return 2;
    case GGUF_T_Q4_0: return 18;
    case GGUF_T_Q4_1: return 20;
    case GGUF_T_Q5_0: return 22;
    case GGUF_T_Q5_1: return 24;
    case GGUF_T_Q8_0: return 34;
    case GGUF_T_Q8_1: return 36;
    case GGUF_T_Q2_K: return 84;
    case GGUF_T_Q3_K: return 110;
    case GGUF_T_Q4_K: return 144;
    case GGUF_T_Q5_K: return 176;
    case GGUF_T_Q6_K: return 210;
    case GGUF_T_Q8_K: return 324;
    default:          return 0;
    }
}
static int gguf_block_size(int qtype) {
    if (qtype >= GGUF_T_Q2_K && qtype <= GGUF_T_Q8_K) return QK_K;
    return 32;
}

/* 反量化张量 [elem0, elem0+nelem) 区间到 dst（dst 容纳 nelem*4 字节）。 */
static void gguf_dequant_range(int qtype, const uint8_t *data,
                               int64_t elem0, int64_t nelem, float *dst) {
    if (qtype == GGUF_T_F32) {
        memcpy(dst, data + elem0 * 4, (size_t)nelem * 4);
        return;
    }
    if (qtype == GGUF_T_F16) {
        for (int64_t i = 0; i < nelem; i++)
            dst[i] = gf_f16_to_f32(((const uint16_t *)(data + elem0 * 2))[i]);
        return;
    }
    if (qtype == GGUF_T_BF16) {
        for (int64_t i = 0; i < nelem; i++)
            dst[i] = gf_bf16_to_f32(((const uint16_t *)(data + elem0 * 2))[i]);
        return;
    }
    const int bs = gguf_block_size(qtype);
    const size_t bbytes = gguf_block_bytes(qtype);
    int64_t end = elem0 + nelem, cur = elem0;
    while (cur < end) {
        int64_t blk = cur / bs;
        int64_t within = cur - blk * bs;
        int64_t n = bs - within;
        if (n > end - cur) n = end - cur;
        float tmp[QK_K];
        gguf_dequant_row(qtype, data + (size_t)blk * bbytes, tmp, bs);
        memcpy(dst + (cur - elem0), tmp + within, (size_t)n * 4);
        cur += n;
    }
}

/* Q8 量化变体（与 safetensors 路径完全一致：q4i / g256 / q8_0） */
static void gguf_quant_q8(uint8_t *dst, const float *f32, int n, int q8_buf_q4) {
    if (q8_buf_q4) f32_to_q4i8(dst, f32, n);
    else if (st_wmode_effective() == 4) f32_to_g256q8(dst, f32, n);
    else f32_to_q8_0(dst, f32, n);
}

/* ---------------- 权重槽映射（llama 风格张量名 → 引擎权重槽） ---------------- */
enum {
    SLOT_EMBED = 0, SLOT_FINAL_NORM, SLOT_ATTN_NORM, SLOT_FFN_NORM,
    SLOT_Q, SLOT_K, SLOT_V, SLOT_O, SLOT_GATE, SLOT_UP, SLOT_DOWN,
    SLOT_QNORM, SLOT_KNORM, SLOT_LM, SLOT_NONE
};

static int gguf_slot(const char *name, int *layer) {
    if (strcmp(name, "token_embd.weight") == 0)  return SLOT_EMBED;
    if (strcmp(name, "output_norm.weight") == 0) return SLOT_FINAL_NORM;
    if (strcmp(name, "output.weight") == 0)      return SLOT_LM;
    if (strncmp(name, "blk.", 4) != 0)           return SLOT_NONE;
    int l;
    if (sscanf(name + 4, "%d.", &l) != 1) return SLOT_NONE;
    const char *dot = strchr(name + 4, '.');
    const char *tail = dot ? dot + 1 : NULL;
    if (!tail) return SLOT_NONE;
    *layer = l;
    if      (strcmp(tail, "attn_norm.weight")   == 0) return SLOT_ATTN_NORM;
    else if (strcmp(tail, "ffn_norm.weight")    == 0) return SLOT_FFN_NORM;
    else if (strcmp(tail, "attn_q.weight")      == 0) return SLOT_Q;
    else if (strcmp(tail, "attn_k.weight")      == 0) return SLOT_K;
    else if (strcmp(tail, "attn_v.weight")      == 0) return SLOT_V;
    else if (strcmp(tail, "attn_output.weight") == 0) return SLOT_O;
    else if (strcmp(tail, "attn_q_norm.weight") == 0) return SLOT_QNORM;
    else if (strcmp(tail, "attn_k_norm.weight") == 0) return SLOT_KNORM;
    else if (strcmp(tail, "ffn_gate.weight")    == 0) return SLOT_GATE;
    else if (strcmp(tail, "ffn_up.weight")      == 0) return SLOT_UP;
    else if (strcmp(tail, "ffn_down.weight")    == 0) return SLOT_DOWN;
    return SLOT_NONE;
}

/* ---------------- GGUF 权重加载 ---------------- */
int gguf_load_model(const char *path, STModelWeights *w) {
    GgufFile f;
    if (gguf_parse(path, &f) != 0) return -1;

    const char *arch = gf_kv_str(&f, "general.architecture");
    if (!arch) { fprintf(stderr, "[GGUF] no general.architecture\n"); gguf_free(&f); return -1; }
    int is_qwen = (strstr(arch, "qwen") != NULL);
    int is_llama = (strcmp(arch, "llama") == 0 || strcmp(arch, "mistral") == 0 ||
                    strcmp(arch, "falcon") == 0);
    if (!is_qwen && !is_llama) {
        fprintf(stderr, "[GGUF] unsupported architecture: %s\n", arch);
        gguf_free(&f); return -1;
    }
    char pfx[96], kbuf[160];
    snprintf(pfx, sizeof(pfx), "%s.", arch);
#define GK(sub) (snprintf(kbuf, sizeof(kbuf), "%s%s", pfx, sub), kbuf)

    STModelConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_layers   = (int)gf_kv_i64(&f, GK("block_count"), 0);
    cfg.dim        = (int)gf_kv_i64(&f, GK("embedding_length"), 0);
    cfg.n_heads    = (int)gf_kv_i64(&f, GK("attention.head_count"), 0);
    cfg.n_kv_heads = (int)gf_kv_i64(&f, GK("attention.head_count_kv"), cfg.n_heads);
    cfg.ffn_dim    = (int)gf_kv_i64(&f, GK("feed_forward_length"), 0);
    cfg.norm_eps   = gf_kv_f32(&f, GK("attention.layer_norm_rms_epsilon"), 1e-6f);
    cfg.rope_theta = gf_kv_f32(&f, GK("rope.freq_base"), 10000.0f);
    cfg.max_seq_len = (int)gf_kv_i64(&f, GK("context_length"), 32768);
    int hd_override = (int)gf_kv_i64(&f, GK("attention.key_length"), 0);
    cfg.head_dim = hd_override ? hd_override : (cfg.dim / (cfg.n_heads > 0 ? cfg.n_heads : 1));
    cfg.head_dim_full = cfg.head_dim;
    cfg.vocab_size = (int)gf_kv_arr_count(&f, "tokenizer.ggml.tokens");
    cfg.bos_id = (int)gf_kv_i64(&f, "tokenizer.ggml.bos_token_id", -1);
    cfg.eos_id = (int)gf_kv_i64(&f, "tokenizer.ggml.eos_token_id", -1);
    cfg.has_q_norm = (gf_find_tensor(&f, "blk.0.attn_q_norm.weight") >= 0);
    cfg.has_mrope = is_qwen && (strstr(arch, "qwen3") != NULL);
    if (cfg.has_mrope) {
        /* 从 GGUF 读 MRoPE sections（qwen3vl.rope.dimension_sections） */
        const struct GgufKV_ *sec = gf_find_kv(&f, GK("rope.dimension_sections"));
        cfg.mrope_n_sec = 0;
        if (sec && sec->type == 9) {
            for (uint64_t i = 0; i < sec->arr_n && cfg.mrope_n_sec < 4; i++) {
                int64_t v = 0;
                if (sec->arr_type == 5) v = ((const int32_t *)sec->arr_data)[i];
                else if (sec->arr_type == 11) v = ((const int64_t *)sec->arr_data)[i];
                if (v <= 0) break;
                cfg.mrope_sections[cfg.mrope_n_sec++] = (int)v;
            }
        }
        if (cfg.mrope_n_sec == 0) {   /* 兜底（Qwen3 文本标准） */
            cfg.mrope_n_sec = 3;
            cfg.mrope_sections[0] = 24; cfg.mrope_sections[1] = 20; cfg.mrope_sections[2] = 20;
        }
    }
    if (cfg.max_seq_len > 8192) cfg.max_seq_len = 8192;
    if (cfg.dim <= 0 || cfg.n_layers <= 0 || cfg.n_heads <= 0 ||
        cfg.ffn_dim <= 0 || cfg.vocab_size <= 0) {
        fprintf(stderr, "[GGUF] implausible config: dim=%d layers=%d heads=%d ff=%d vocab=%d\n",
                cfg.dim, cfg.n_layers, cfg.n_heads, cfg.ffn_dim, cfg.vocab_size);
        gguf_free(&f); return -1;
    }
    int dim = cfg.dim, nl = cfg.n_layers, hd = cfg.head_dim;
    int ff = cfg.ffn_dim, vc = cfg.vocab_size;
    int q_out = cfg.n_heads * hd, k_out = cfg.n_kv_heads * hd;
    fprintf(stderr, "[GGUF] arch=%s dim=%d layers=%d heads=%d kv=%d hd=%d ff=%d "
                    "vocab=%d rope_theta=%.0f eps=%g q_norm=%d mrope=%d\n",
            arch, dim, nl, cfg.n_heads, cfg.n_kv_heads, hd, ff, vc,
            cfg.rope_theta, cfg.norm_eps, cfg.has_q_norm, cfg.has_mrope);
    if (cfg.has_mrope)
        fprintf(stderr, "[GGUF] mrope sections: %d %d %d %d (n=%d)\n",
                cfg.mrope_sections[0], cfg.mrope_sections[1],
                cfg.mrope_sections[2], cfg.mrope_sections[3], cfg.mrope_n_sec);
#undef GK

    st_weights_alloc_layers_q8ffn(w, &cfg, nl);
    if (!w->is_allocated) { gguf_free(&f); return -1; }
    const int q8b4 = w->q8_buf_q4;

    /* token_embed（F32；按行 chunked，行=dim 为 32 倍数 → 块对齐） */
    {
        int te = gf_find_tensor(&f, "token_embd.weight");
        if (te < 0) {
            fprintf(stderr, "[GGUF] missing token_embd.weight\n");
            st_weights_free(w); gguf_free(&f); return -1;
        }
        const uint8_t *td = gf_ptr(&f, f.data_offset + (uint64_t)f.tens[te].offset);
        const int qt = f.tens[te].type;
        int64_t chunk_rows = 8192, ce = chunk_rows * dim;
        float *tmp = (float *)malloc((size_t)ce * 4);
        if (!tmp) { st_weights_free(w); gguf_free(&f); return -1; }
        for (int64_t r0 = 0; r0 < vc; r0 += chunk_rows) {
            int64_t rows = (vc - r0 < chunk_rows) ? (vc - r0) : chunk_rows;
            gguf_dequant_range(qt, td, r0 * dim, rows * dim, tmp);
            memcpy(w->token_embed + (size_t)r0 * dim, tmp, (size_t)(rows * dim) * 4);
        }
        free(tmp);
    }

    /* 逐张量加载（张量目录扫描 → 权重槽） */
    {
        int64_t loaded = 0;
        for (uint64_t i = 0; i < f.n_tensors; i++) {
            int layer = 0;
            int slot = gguf_slot(f.tens[i].name, &layer);
            if (slot == SLOT_NONE || slot == SLOT_EMBED || slot == SLOT_LM) continue;
            if (layer < 0 || layer >= nl) {
                fprintf(stderr, "[GGUF] layer %d out of range for %s\n", layer, f.tens[i].name);
                continue;
            }
            const uint8_t *td = gf_ptr(&f, f.data_offset + (uint64_t)f.tens[i].offset);
            const int qt = f.tens[i].type;
            int rows, cols, elems;
            float *f32dst = NULL;
            uint8_t *q8dst = NULL, *q4dst = NULL;
            int rep8 = 0, rep4 = 0;
            switch (slot) {
            case SLOT_ATTN_NORM: f32dst = w->attn_norm + (size_t)layer * dim; rows = dim; cols = 1; elems = dim; break;
            case SLOT_FFN_NORM:  f32dst = w->ffn_norm  + (size_t)layer * dim; rows = dim; cols = 1; elems = dim; break;
            case SLOT_QNORM:     f32dst = w->q_norm    + (size_t)layer * hd;  rows = hd;  cols = 1; elems = hd;  break;
            case SLOT_KNORM:     f32dst = w->k_norm    + (size_t)layer * hd;  rows = hd;  cols = 1; elems = hd;  break;
            /* 权重槽 rows/cols = 引擎布局 [输出行, 输入列]（与 safetensors 原始
             * 布局一致）。GGUF 张量为列主序存储（ne0=输入连续维, ne1=输出），
             * 行主序视图恰好 = [输出][输入]，故直接按 rows=输出维、cols=输入维
             * 读取量化即可，无需转置。 */
            case SLOT_Q:  q8dst = w->q8_q_weight  + Q8_BYTES((size_t)layer * q_out * dim);
                          q4dst = w->q4_q_weight  + Q4_BYTES((size_t)layer * q_out * dim);
                          rows = q_out; cols = dim; elems = q_out * dim;
                          rep8 = rep4 = 1; break;
            case SLOT_K:  q8dst = w->q8_k_weight  + Q8_BYTES((size_t)layer * k_out * dim);
                          q4dst = w->q4_k_weight  + Q4_BYTES((size_t)layer * k_out * dim);
                          rows = k_out; cols = dim; elems = k_out * dim;
                          rep8 = rep4 = 1; break;
            case SLOT_V:  q8dst = w->q8_v_weight  + Q8_BYTES((size_t)layer * k_out * dim);
                          q4dst = w->q4_v_weight  + Q4_BYTES((size_t)layer * k_out * dim);
                          rows = k_out; cols = dim; elems = k_out * dim;
                          rep8 = rep4 = 1; break;
            case SLOT_O:  q8dst = w->q8_o_weight  + Q8_BYTES((size_t)layer * dim * q_out);
                          q4dst = w->q4_o_weight  + Q4_BYTES((size_t)layer * dim * q_out);
                          rows = dim; cols = q_out; elems = dim * q_out;
                          rep8 = rep4 = 1; break;
            case SLOT_GATE: q8dst = w->q8_gate_weight + Q8_BYTES((size_t)layer * ff * dim);
                            q4dst = w->q4_gate_weight + Q4_BYTES((size_t)layer * ff * dim);
                            rows = ff; cols = dim; elems = ff * dim;
                            rep8 = rep4 = 1; break;
            case SLOT_UP:   q8dst = w->q8_up_weight   + Q8_BYTES((size_t)layer * ff * dim);
                            q4dst = w->q4_up_weight   + Q4_BYTES((size_t)layer * ff * dim);
                            rows = ff; cols = dim; elems = ff * dim;
                            rep8 = rep4 = 1; break;
            case SLOT_DOWN: q8dst = w->q8_down_weight + Q8_BYTES((size_t)layer * dim * ff);
                            q4dst = w->q4_down_weight + Q4_BYTES((size_t)layer * dim * ff);
                            rows = dim; cols = ff; elems = dim * ff;
                            rep8 = rep4 = 1; break;
            default: continue;
            }
            /* wmode 下部分缓冲未分配（q8-only 无 q4、q4-only 无 q8）：基指针为
             * NULL 时 `base + offset` 是指针算术 UB 且结果非 NULL，会绕过下面的
             * `if (q8dst/q4dst)` 判空对 ~1MB 非法地址写入（layer>=1 必现崩溃）。
             * 基指针族要么全 NULL 要么全非 NULL，用 k 作代理即可。 */
            if (q4dst && !w->q4_k_weight) q4dst = NULL;
            if (q8dst && !w->q8_k_weight) q8dst = NULL;
            if ((int64_t)f.tens[i].ne[0] * f.tens[i].ne[1] != elems) {
                fprintf(stderr, "[GGUF] shape mismatch %s: %lldx%lld != %dx%d\n",
                        f.tens[i].name, (long long)f.tens[i].ne[0],
                        (long long)f.tens[i].ne[1], rows, cols);
                continue;
            }
            if (f32dst) {
                gguf_dequant_range(qt, td, 0, elems, f32dst);
            } else {
                int chunk_rows = 2048;
                if (chunk_rows > rows) chunk_rows = rows;
                float *tmp = (float *)malloc((size_t)chunk_rows * cols * 4);
                if (!tmp) { st_weights_free(w); gguf_free(&f); return -1; }
                for (int r0 = 0; r0 < rows; r0 += chunk_rows) {
                    int nrows = (rows - r0 < chunk_rows) ? (rows - r0) : chunk_rows;
                    int64_t ne = (int64_t)nrows * cols;
                    gguf_dequant_range(qt, td, (int64_t)r0 * cols, ne, tmp);
                    /* 块偏移必须用 Q8/Q4_BYTES(r0*cols)：写成 (r0*(cols+31))/32 会
                     * 对 r0>=2048 的分块多算约 r0*31/32 个块，把数据写进下一层
                     * 甚至越出分配（ffn_gate/up 多块张量必现）。 */
                    if (q8dst)
                        gguf_quant_q8(q8dst + Q8_BYTES((size_t)r0 * cols),
                                      tmp, (int)ne, q8b4);
                    if (q4dst)
                        f32_to_q4_0(q4dst + Q4_BYTES((size_t)r0 * cols),
                                    tmp, (int)ne);
                }
                free(tmp);
                if (rep8 && q8dst && g_st_q8_repack)
                    repack_q8_0_tiled_inplace(q8dst, rows, cols);
                if (rep4 && q4dst && g_st_q4_repack)
                    repack_q4_0_4x4_inplace(q4dst, rows, cols);
            }
            loaded++;
        }
        fprintf(stderr, "[GGUF] loaded %lld weight tensors\n", (long long)loaded);
    }

    /* final_norm */
    {
        int on = gf_find_tensor(&f, "output_norm.weight");
        if (on < 0) {
            fprintf(stderr, "[GGUF] missing output_norm.weight\n");
            st_weights_free(w); gguf_free(&f); return -1;
        }
        gguf_dequant_range(f.tens[on].type,
                           gf_ptr(&f, f.data_offset + (uint64_t)f.tens[on].offset),
                           0, dim, w->final_norm);
    }

    /* lm_head：output.weight（chunked）或 tied（复用 token_embed） */
    {
        int ow = gf_find_tensor(&f, "output.weight");
        if (ow >= 0 && f.tens[ow].ne[0] * f.tens[ow].ne[1] == (int64_t)vc * dim) {
            const uint8_t *td = gf_ptr(&f, f.data_offset + (uint64_t)f.tens[ow].offset);
            const int qt = f.tens[ow].type;
            int chunk_rows = 4096;
            float *tmp = (float *)malloc((size_t)chunk_rows * dim * 4);
            if (!tmp) { st_weights_free(w); gguf_free(&f); return -1; }
            for (int64_t r0 = 0; r0 < vc; r0 += chunk_rows) {
                int64_t rows = (vc - r0 < chunk_rows) ? (vc - r0) : chunk_rows;
                int64_t ne = rows * dim;
                gguf_dequant_range(qt, td, r0 * dim, ne, tmp);
                if (w->q8_lm_weight)
                    gguf_quant_q8(w->q8_lm_weight + (size_t)(r0 * dim / 32) * 34, tmp, (int)ne, q8b4);
                if (w->q4_lm_weight)
                    f32_to_q4_0(w->q4_lm_weight + (size_t)(r0 * dim / 32) * 18, tmp, (int)ne);
            }
            free(tmp);
            if (w->q8_lm_weight && g_st_q8_repack)
                repack_q8_0_tiled_inplace(w->q8_lm_weight, vc, dim);
            if (w->q4_lm_weight && g_st_q4_repack)
                repack_q4_0_4x4_inplace(w->q4_lm_weight, vc, dim);
        } else {
            fprintf(stderr, "[GGUF] output.weight absent -> tied embeddings\n");
            int chunk_rows = 4096;
            float *tmp = (float *)malloc((size_t)chunk_rows * dim * 4);
            if (!tmp) { st_weights_free(w); gguf_free(&f); return -1; }
            for (int64_t r0 = 0; r0 < vc; r0 += chunk_rows) {
                int64_t rows = (vc - r0 < chunk_rows) ? (vc - r0) : chunk_rows;
                int64_t ne = rows * dim;
                memcpy(tmp, w->token_embed + (size_t)r0 * dim, (size_t)ne * 4);
                if (w->q8_lm_weight)
                    gguf_quant_q8(w->q8_lm_weight + (size_t)(r0 * dim / 32) * 34, tmp, (int)ne, q8b4);
                if (w->q4_lm_weight)
                    f32_to_q4_0(w->q4_lm_weight + (size_t)(r0 * dim / 32) * 18, tmp, (int)ne);
            }
            free(tmp);
            if (w->q8_lm_weight && g_st_q8_repack)
                repack_q8_0_tiled_inplace(w->q8_lm_weight, vc, dim);
            if (w->q4_lm_weight && g_st_q4_repack)
                repack_q4_0_4x4_inplace(w->q4_lm_weight, vc, dim);
        }
    }

    w->has_q8 = (w->q8_q_weight) ? 1 : 0;
    w->has_q4 = (w->q4_q_weight) ? 1 : 0;
    w->has_x8 = 0;
    w->emb_f16 = 0;
    w->n_layers_allocated = nl;
    w->is_allocated = 1;
    gguf_free(&f);
    return 0;
}

/* ---------------- GGUF vocab -> tokenizer ---------------- */

/* GPT-2 bytes_to_unicode 反向映射：llama.cpp 转换 Qwen3 等 byte-level BPE
 * vocab 时，把每个 token 按 GPT-2 字节映射（0x00-0xFF -> U+0100+ 等）展开成
 * unicode 字符序列再按 UTF-8 写入 GGUF（例如字节 E4 BD A0 "你" 存为
 * "ä½Ġ" = C3 A4 C2 BD C4 A0）。引擎前缀匹配输入的是原始 UTF-8 字节，必须把
 * token 字符串还原回原始字节才能匹配（与 export_vocab.py 的 vocab.bin 一致）。
 * 返回写入 out 的字节数；in 为 GGUF 里的 UTF-8 字符串。 */
static size_t gguf_gpt2_unmap(const char *in, size_t inlen, unsigned char *out) {
    size_t o = 0, i = 0;
    while (i < inlen) {
        unsigned char c = (unsigned char)in[i];
        uint32_t cp; int blen;
        if (c < 0x80) { cp = c; blen = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; blen = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; blen = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; blen = 4; }
        else { out[o++] = c; i++; continue; }
        if (i + (size_t)blen > inlen) { out[o++] = c; i++; continue; }
        int ok = 1;
        for (int k = 1; k < blen; k++) {
            unsigned char cc = (unsigned char)in[i + k];
            if ((cc & 0xC0) != 0x80) { ok = 0; break; }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (!ok) { out[o++] = c; i++; continue; }
        unsigned char b;
        if (cp >= 0x21 && cp <= 0x7E)      b = (unsigned char)cp;            /* ASCII 原样 */
        else if (cp >= 0xA1 && cp <= 0xAC) b = (unsigned char)cp;            /* 字节原样 */
        else if (cp >= 0xAE && cp <= 0xFF) b = (unsigned char)cp;            /* 字节原样 */
        else if (cp >= 0x100 && cp <= 0x120) b = (unsigned char)(cp - 0x100);              /* 0x00-0x20 */
        else if (cp >= 0x121 && cp <= 0x142) b = (unsigned char)(cp - 0x121 + 0x7F);        /* 0x7F-0xA0 */
        else if (cp == 0x143)               b = 0xAD;                        /* 0xAD 例外 */
        else b = (unsigned char)(cp & 0xFF);
        out[o++] = b;
        i += (size_t)blen;
    }
    return o;
}

int gguf_load_tokenizer(const char *path, QwenTokenizer *tok) {
    GgufFile f;
    if (gguf_parse(path, &f) != 0) return -1;
    memset(tok, 0, sizeof(*tok));
    tok->bos_id = -1; tok->eos_id = -1; tok->pad_id = -1;
    tok->im_start_id = -1; tok->im_end_id = -1;
    tok->vision_start_id = -1; tok->vision_end_id = -1;
    tok->image_token_id = -1; tok->video_token_id = -1;

    uint64_t n = gf_kv_arr_count(&f, "tokenizer.ggml.tokens");
    if (n == 0 || n > QWEN_TOKENIZER_MAX_TOKENS) {
        fprintf(stderr, "[TOK] invalid GGUF vocab size: %llu\n", (unsigned long long)n);
        gguf_free(&f); return -1;
    }
    tok->vocab_size = (int)n;
    tok->strings  = calloc((size_t)n, sizeof(char *));
    tok->str_lens = calloc((size_t)n, sizeof(int));
    if (!tok->strings || !tok->str_lens) { gguf_free(&f); return -1; }

    size_t total = 0;
    for (uint64_t i = 0; i < n; i++) {
        const char *s = gf_kv_arr_str(&f, "tokenizer.ggml.tokens", i);
        total += (s ? strlen(s) : 0) + 1;
    }
    tok->str_data = calloc(total ? total : 1, 1);
    if (!tok->str_data) { gguf_free(&f); return -1; }

    char *p = tok->str_data;
    tok->max_str_len = 0;
    unsigned char tmp[512];
    for (uint64_t i = 0; i < n; i++) {
        const char *s = gf_kv_arr_str(&f, "tokenizer.ggml.tokens", i);
        size_t len = s ? strlen(s) : 0;
        if (len > sizeof(tmp)) len = sizeof(tmp);  /* 防御：超长 token 截断（不应出现） */
        size_t blen = 0;
        if (s && len > 0) blen = gguf_gpt2_unmap(s, len, tmp);
        memcpy(p, tmp, blen);
        p[blen] = '\0';
        tok->strings[(int)i] = p;
        tok->str_lens[(int)i] = (int)blen;
        if ((int)blen > tok->max_str_len) tok->max_str_len = (int)blen;
        p += blen + 1;
    }
    tok->bos_id = (int)gf_kv_i64(&f, "tokenizer.ggml.bos_token_id", -1);
    tok->eos_id = (int)gf_kv_i64(&f, "tokenizer.ggml.eos_token_id", -1);
    tok->pad_id = (int)gf_kv_i64(&f, "tokenizer.ggml.pad_token_id", -1);
    tok->is_loaded = 1;
    fprintf(stderr, "[TOK] Loaded %d tokens from GGUF vocab (max_len=%d)\n",
            tok->vocab_size, tok->max_str_len);
    gguf_free(&f);
    return 0;
}

/* ---------------- 诊断：打印 GGUF 结构 ---------------- */
int gguf_dump_info(const char *path) {
    GgufFile f;
    if (gguf_parse(path, &f) != 0) return -1;
    printf("GGUF %s: %llu KV, %llu tensors, data_offset=%llu align=%zu\n",
           path, (unsigned long long)f.n_kv, (unsigned long long)f.n_tensors,
           (unsigned long long)f.data_offset, f.alignment);
    const char *arch = gf_kv_str(&f, "general.architecture");
    printf("  architecture: %s\n", arch ? arch : "(none)");
    for (uint64_t i = 0; i < f.n_kv; i++) {
        const struct GgufKV_ *k = &f.kv[i];
        printf("  KV[%llu] %s (type %d)", (unsigned long long)i, k->key, k->type);
        if (k->type == 8) printf(" = \"%s\"", k->str);
        else if (k->type == 5) printf(" = %d", k->v.i32);
        else if (k->type == 11) printf(" = %lld", (long long)k->v.i64);
        else if (k->type == 4) printf(" = %u", k->v.u32);
        else if (k->type == 6) printf(" = %g", k->v.f32);
        else if (k->type == 9) printf(" = [%s x %llu]", k->arr_type == 8 ? "str" : "num",
                                      (unsigned long long)k->arr_n);
        printf("\n");
    }
    for (uint64_t i = 0; i < f.n_tensors; i++) {
        const struct GgufTens_ *t = &f.tens[i];
        printf("  T[%llu] %-40s type=%d ne=[%lld,%lld,%lld,%lld] off=%llu\n",
               (unsigned long long)i, t->name, t->type,
               (long long)t->ne[0], (long long)t->ne[1],
               (long long)t->ne[2], (long long)t->ne[3],
               (unsigned long long)t->offset);
    }
    gguf_free(&f);
    return 0;
}

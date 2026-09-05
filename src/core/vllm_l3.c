/**
 * vllm_l3.c - Phase 2: cold-block Q4 disk eviction for the KV cache
 *
 * See vllm_l3.h for the format and the honest-scope design notes.
 * Determinism red line (axiom_arith_gumbel_argmax_001): quantization, the
 * dot kernel and the eviction ranking are all deterministic for identical
 * inputs (fixed rounding, fixed tie-break, fixed file offsets).
 */

#include "vllm_l3.h"
#include "vllm_platform.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>   /* fileno / lseek / read / fseeko on Linux/RK3588 */

/* 单平台（RK3588/aarch64）：Q4 点积内核走 NEON，不再保留 AVX2 分支。 */

/* L3_HEADER_SIZE now defined in vllm_l3.h (shared with main.c) */

/* ================================================================
 * Q4 kernels
 * ================================================================ */

/* Round half away from zero, deterministically (no libm dependency). */
static int q4_round(float x) {
    float t = x >= 0.0f ? x + 0.5f : x - 0.5f;
    return (int)t;
}

/* Pack 32 floats into [16B][scale] using the axiom nibble layout:
 * nibble = q + 8, even index in the low nibble, q in [-8, +7].
 * Returns the block scale (amax/7, saturated). */
static float pack32(uint8_t *dst, const float *v) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float a = fabsf(v[i]);
        if (a > amax) amax = a;
    }
    float scale = amax / 7.0f;
    if (!(scale > 0.0f)) scale = 1.0f;   /* all-zero block: q=0 */
    for (int i = 0; i < 16; i++) {
        int q0 = q4_round(v[i * 2] / scale);
        int q1 = q4_round(v[i * 2 + 1] / scale);
        if (q0 > 7) q0 = 7; else if (q0 < -8) q0 = -8;
        if (q1 > 7) q1 = 7; else if (q1 < -8) q1 = -8;
        dst[i] = (uint8_t)((q0 + 8) | ((q1 + 8) << 4));
    }
    return scale;
}

void l3_q4_pack64(uint8_t *payload, const float *src) {
    float sa = pack32(payload + 4, src);
    float sb = pack32(payload + 24, src + 32);
    memcpy(payload, &sa, 4);
    memcpy(payload + 20, &sb, 4);
}

/* Dequantize one 16B packed block + scale into dst[32]. */
static void dequant32(float *dst, const uint8_t *packed, float scale) {
    for (int i = 0; i < 16; i++) {
        uint8_t b = packed[i];
        dst[i * 2]     = (float)((int)(b & 0x0F) - 8) * scale;
        dst[i * 2 + 1] = (float)((int)((b >> 4) & 0x0F) - 8) * scale;
    }
}

void l3_q4_dequant64(float *dst, const uint8_t *payload) {
    float scale_a, scale_b;
    memcpy(&scale_a, payload, 4);
    memcpy(&scale_b, payload + 20, 4);
    dequant32(dst, payload + 4, scale_a);
    dequant32(dst + 32, payload + 24, scale_b);
}

/* Scalar reference dot (also the non-AVX2 fallback). */
static float q4_dot64_scalar(const uint8_t *payload, const float *act) {
    float sum = 0.0f;
    float scale_a, scale_b;
    memcpy(&scale_a, payload, 4);
    memcpy(&scale_b, payload + 20, 4);
    for (int i = 0; i < 16; i++) {
        uint8_t b = payload[4 + i];
        sum += (float)((int)(b & 0x0F) - 8) * scale_a * act[i * 2];
        sum += (float)((int)((b >> 4) & 0x0F) - 8) * scale_a * act[i * 2 + 1];
    }
    for (int i = 0; i < 16; i++) {
        uint8_t b = payload[24 + i];
        sum += (float)((int)(b & 0x0F) - 8) * scale_b * act[32 + i * 2];
        sum += (float)((int)((b >> 4) & 0x0F) - 8) * scale_b * act[32 + i * 2 + 1];
    }
    return sum;
}

/* NEON (aarch64/RK3588) Q4 dot: 64 elements -> 40-byte payload.
 * Element order: byte i holds element 2i (low nibble) and 2i+1 (high nibble),
 * matching the scalar reference and the AVX2 unpack (unpacklo/hi + interleave).
 * fp32 accumulate in the same per-block order as the AVX2 kernel. */
#if ST_HAVE_NEON
static float l3_q4_dot64_neon(const uint8_t *payload, const float *act) {
    float scale_a, scale_b;
    memcpy(&scale_a, payload, 4);
    memcpy(&scale_b, payload + 20, 4);

    const uint8x16_t mask = vdupq_n_u8(0x0F);
    /* 16 packed bytes -> 32 int8 in [-8,+7], interleaved element order. */
#define L3_NEON_UNPACK(src, e0, e1) do { \
        uint8x16_t v_ = vld1q_u8(src); \
        uint8x16_t lo = vandq_u8(v_, mask); \
        uint8x16_t hi = vandq_u8(vshrq_n_u8(v_, 4), mask); \
        e0 = vsubq_s8(vreinterpretq_s8_u8(vzip1q_u8(lo, hi)), vdupq_n_s8(8)); \
        e1 = vsubq_s8(vreinterpretq_s8_u8(vzip2q_u8(lo, hi)), vdupq_n_s8(8)); \
    } while (0)
    /* one int8x16 -> four fp32x4 lanes (elements 0-3, 4-7, 8-11, 12-15). */
#define L3_NEON_CONV(v, f0, f1, f2, f3) do { \
        int16x8_t l_ = vmovl_s8(vget_low_s8(v));  \
        int16x8_t h_ = vmovl_s8(vget_high_s8(v)); \
        f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(l_))); \
        f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(l_))); \
        f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(h_))); \
        f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(h_))); \
    } while (0)

    int8x16_t a0, a1, b0, b1;
    L3_NEON_UNPACK(payload + 4,  a0, a1);
    L3_NEON_UNPACK(payload + 24, b0, b1);

    float32x4_t acc[8], f0, f1, f2, f3;
    for (int i = 0; i < 8; i++) acc[i] = vdupq_n_f32(0.0f);

    const float32x4_t sa = vdupq_n_f32(scale_a);
    L3_NEON_CONV(a0, f0, f1, f2, f3);
    acc[0] = vfmaq_f32(acc[0], vmulq_f32(f0, sa), vld1q_f32(act + 0));
    acc[1] = vfmaq_f32(acc[1], vmulq_f32(f1, sa), vld1q_f32(act + 4));
    acc[2] = vfmaq_f32(acc[2], vmulq_f32(f2, sa), vld1q_f32(act + 8));
    acc[3] = vfmaq_f32(acc[3], vmulq_f32(f3, sa), vld1q_f32(act + 12));
    L3_NEON_CONV(a1, f0, f1, f2, f3);
    acc[4] = vfmaq_f32(acc[4], vmulq_f32(f0, sa), vld1q_f32(act + 16));
    acc[5] = vfmaq_f32(acc[5], vmulq_f32(f1, sa), vld1q_f32(act + 20));
    acc[6] = vfmaq_f32(acc[6], vmulq_f32(f2, sa), vld1q_f32(act + 24));
    acc[7] = vfmaq_f32(acc[7], vmulq_f32(f3, sa), vld1q_f32(act + 28));

    const float32x4_t sb = vdupq_n_f32(scale_b);
    L3_NEON_CONV(b0, f0, f1, f2, f3);
    acc[0] = vfmaq_f32(acc[0], vmulq_f32(f0, sb), vld1q_f32(act + 32));
    acc[1] = vfmaq_f32(acc[1], vmulq_f32(f1, sb), vld1q_f32(act + 36));
    acc[2] = vfmaq_f32(acc[2], vmulq_f32(f2, sb), vld1q_f32(act + 40));
    acc[3] = vfmaq_f32(acc[3], vmulq_f32(f3, sb), vld1q_f32(act + 44));
    L3_NEON_CONV(b1, f0, f1, f2, f3);
    acc[4] = vfmaq_f32(acc[4], vmulq_f32(f0, sb), vld1q_f32(act + 48));
    acc[5] = vfmaq_f32(acc[5], vmulq_f32(f1, sb), vld1q_f32(act + 52));
    acc[6] = vfmaq_f32(acc[6], vmulq_f32(f2, sb), vld1q_f32(act + 56));
    acc[7] = vfmaq_f32(acc[7], vmulq_f32(f3, sb), vld1q_f32(act + 60));

#undef L3_NEON_UNPACK
#undef L3_NEON_CONV

    float32x4_t s01 = vaddq_f32(acc[0], acc[1]);
    float32x4_t s23 = vaddq_f32(acc[2], acc[3]);
    float32x4_t s45 = vaddq_f32(acc[4], acc[5]);
    float32x4_t s67 = vaddq_f32(acc[6], acc[7]);
    float32x2_t lo = vadd_f32(vget_low_f32(vaddq_f32(s01, s23)),
                              vget_high_f32(vaddq_f32(s01, s23)));
    float32x2_t hi = vadd_f32(vget_low_f32(vaddq_f32(s45, s67)),
                              vget_high_f32(vaddq_f32(s45, s67)));
    float32x2_t t = vpadd_f32(vadd_f32(lo, hi), vadd_f32(lo, hi));
    return vget_lane_f32(t, 0);
}
#endif /* ST_HAVE_NEON */

float l3_q4_dot64(const uint8_t *payload, const float *act) {
#if ST_HAVE_NEON
    return l3_q4_dot64_neon(payload, act);
#else
    return q4_dot64_scalar(payload, act);
#endif
}

void l3_q4_vacc64(float *acc, const uint8_t *payload, float wgt) {
    float scale_a, scale_b;
    memcpy(&scale_a, payload, 4);
    memcpy(&scale_b, payload + 20, 4);
    for (int i = 0; i < 16; i++) {
        uint8_t b = payload[4 + i];
        acc[i * 2]     += wgt * (float)((int)(b & 0x0F) - 8) * scale_a;
        acc[i * 2 + 1] += wgt * (float)((int)((b >> 4) & 0x0F) - 8) * scale_a;
    }
    for (int i = 0; i < 16; i++) {
        uint8_t b = payload[24 + i];
        acc[32 + i * 2]     += wgt * (float)((int)(b & 0x0F) - 8) * scale_b;
        acc[32 + i * 2 + 1] += wgt * (float)((int)((b >> 4) & 0x0F) - 8) * scale_b;
    }
}

/* ================================================================
 * L3 state + disk layout
 * ================================================================ */

#define L3_FSEEK fseeko
#define L3_FILENO(fp) fileno(fp)
#define L3_SEEK64(fd, off) lseek((fd), (off_t)(off), SEEK_SET)
#define L3_READ(fd, buf, n) read((fd), (buf), (size_t)(n))

static int write_header(STL3State *s) {
    uint8_t hdr[L3_HEADER_SIZE];
    memset(hdr, 0, sizeof(hdr));
    uint32_t magic = L3_HEADER_MAGIC;
    uint32_t ver   = 1;
    memcpy(hdr + 0,  &magic, 4);
    memcpy(hdr + 4,  &ver, 4);
    memcpy(hdr + 8,  &s->nl, 4);
    memcpy(hdr + 12, &s->nkv, 4);
    memcpy(hdr + 16, &s->hd, 4);
    memcpy(hdr + 20, &s->bs, 4);
    memcpy(hdr + 24, &s->max_blocks, 4);
    memcpy(hdr + 28, &s->block_bytes, 8);
    if (L3_FSEEK(s->fp, 0, SEEK_SET) != 0) return -1;
    return fwrite(hdr, 1, L3_HEADER_SIZE, s->fp) == L3_HEADER_SIZE ? 0 : -1;
}

int l3_state_init(STL3State *s, const char *path, int nl, int nkv, int hd,
                  int bs, int max_blocks) {
    memset(s, 0, sizeof(*s));
    s->fp = NULL;
    s->blocks = NULL;
    if (!path || nl < 1 || nkv < 1 || hd < 64 || (hd % 64) != 0 || bs < 1 || max_blocks < 1)
        return -1;

    int n_payloads = hd / 64;
    size_t payloads_per_head = (size_t)bs * (size_t)n_payloads;
    size_t k_area = (size_t)nkv * payloads_per_head * L3_Q4_PAYLOAD64;
    s->block_bytes = 2 * k_area;

    s->fp = fopen(path, "wb+");
    if (!s->fp) return -1;
    s->nl = nl; s->nkv = nkv; s->hd = hd; s->bs = bs; s->max_blocks = max_blocks;
    s->path[0] = '\0';
    if (path) {
        strncpy(s->path, path, sizeof(s->path) - 1);
        s->path[sizeof(s->path) - 1] = '\0';
    }

    s->blocks = calloc((size_t)nl * (size_t)max_blocks, sizeof(STL3Block));
    if (!s->blocks) { fclose(s->fp); s->fp = NULL; return -1; }

    if (write_header(s) != 0) { l3_state_free(s); return -1; }
    fflush(s->fp);
    return 0;
}

void l3_state_free(STL3State *s) {
    if (!s) return;
    if (s->fp) { fclose(s->fp); s->fp = NULL; }
    free(s->mem);
    s->mem = NULL;
    s->mem_size = 0;
    free(s->blocks);
    s->blocks = NULL;
    s->evicted = 0;
}

/* Mirror the whole logical payload region into RAM so decode-time reads never
 * touch the FILE (the decode hot path historically corrupted the heap when
 * driven through fseek/fread on the shared fp). The file may be shorter than
 * the full logical extent (only evicted blocks are written); zero-fill the
 * remainder so an unwritten region reads as a benign zero payload.
 * The read itself goes through the raw fd (_lseeki64/_read) instead of the
 * CRT stream: mixing fwrite with fseek/fread on the same buffered FILE* is
 * exactly the shared-fp churn that corrupted the heap. Pending fwrite output
 * is flushed first so the raw fd observes the complete file. */
int l3_load_to_mem(STL3State *s) {
    if (!s || !s->fp || s->mem) return 0;
    size_t total = (size_t)L3_HEADER_SIZE
                 + (size_t)s->nl * (size_t)s->max_blocks * s->block_bytes;
    uint8_t *m = (uint8_t *)malloc(total ? total : 1);
    if (!m) return -1;
    memset(m, 0, total);
    if (fflush(s->fp) != 0) { free(m); return -1; }
    int fd = L3_FILENO(s->fp);
    if (L3_SEEK64(fd, 0) < 0) { free(m); return -1; }
    size_t got = 0;
    while (got < total) {
        long long n = L3_READ(fd, m + got, total - got);
        if (n <= 0) break;   /* partial/EOF: tail stays zero-filled */
        got += (size_t)n;
    }
    s->mem = m;
    s->mem_size = total;
    return 0;
}

int l3_evict_block(STL3State *s, int layer, int block,
                   const float *const *k_cache, const float *const *v_cache,
                   int kv_dim, int bs) {
    if (!s->fp) return -1;
    if (layer < 0 || block < 0 || layer >= s->nl || block >= s->max_blocks) return -1;
    if (!k_cache[block] || !v_cache[block]) return -1;   /* already evicted */

    int nkv = s->nkv, hd = s->hd;
    int n_payloads = hd / 64;
    size_t payloads_per_head = (size_t)bs * (size_t)n_payloads;
    size_t k_area = (size_t)nkv * payloads_per_head * L3_Q4_PAYLOAD64;

    size_t base = (size_t)L3_HEADER_SIZE +
                  ((size_t)layer * (size_t)s->max_blocks + (size_t)block) * s->block_bytes;
    const float *kbase = k_cache[block];   /* [bs * kv_dim] */
    const float *vbase = v_cache[block];
    uint8_t tmp[L3_Q4_PAYLOAD64];

    /* K area then V area: [head][token][hd/64 payloads] */
    for (int area = 0; area < 2; area++) {
        const float *src = area == 0 ? kbase : vbase;
        for (int h = 0; h < nkv; h++) {
            for (int t = 0; t < bs; t++) {
                const float *row = src + (size_t)t * (size_t)kv_dim + (size_t)h * (size_t)hd;
                for (int g = 0; g < n_payloads; g++) {
                    l3_q4_pack64(tmp, row + (size_t)g * 64);
                    size_t foff = base + (size_t)area * k_area
                                  + (size_t)h * payloads_per_head * L3_Q4_PAYLOAD64
                                  + ((size_t)t * (size_t)n_payloads + (size_t)g) * L3_Q4_PAYLOAD64;
                    if (L3_FSEEK(s->fp, (long long)foff, SEEK_SET) != 0) return -1;
                    if (fwrite(tmp, 1, L3_Q4_PAYLOAD64, s->fp) != L3_Q4_PAYLOAD64) return -1;
                }
            }
        }
    }
    fflush(s->fp);

    STL3Block *bm = &s->blocks[(size_t)layer * (size_t)s->max_blocks + (size_t)block];
    bm->on_disk = 1;
    bm->disk_off = (uint64_t)base;
    s->evicted++;
    return 0;
}

/* Eviction ranking: importance mass desc, block index asc (deterministic). */
typedef struct { float mass; int idx; } Rank;
static int rank_cmp(const void *a, const void *b) {
    const Rank *ra = (const Rank *)a, *rb = (const Rank *)b;
    if (ra->mass > rb->mass) return -1;
    if (ra->mass < rb->mass) return 1;
    return ra->idx - rb->idx;
}

int l3_evict_layer(STL3State *s, int layer,
                   const float *const *k_cache, const float *const *v_cache,
                   int kv_dim, int seq_len, const float *importance,
                   float ratio, int bs) {
    if (!s->fp || seq_len <= 0 || !k_cache || !v_cache) return 0;
    int n_blocks = (seq_len + bs - 1) / bs;
    if (n_blocks <= 1) return 0;

    Rank *r = (Rank *)malloc((size_t)n_blocks * sizeof(Rank));
    uint8_t *keep = (uint8_t *)calloc((size_t)n_blocks, 1);
    if (!r || !keep) { free(r); free(keep); return 0; }

    for (int b = 0; b < n_blocks; b++) {
        float m = 0.0f;
        int p0 = b * bs, p1 = p0 + bs;
        if (p1 > seq_len) p1 = seq_len;
        for (int t = p0; t < p1; t++) m += importance[t];
        r[b].mass = m;
        r[b].idx = b;
    }
    qsort(r, (size_t)n_blocks, sizeof(Rank), rank_cmp);

    int keep_n = (int)((1.0f - ratio) * (float)n_blocks);
    if (keep_n < 1) keep_n = 1;
    if (keep_n >= n_blocks) keep_n = n_blocks;
    for (int i = 0; i < keep_n; i++) keep[r[i].idx] = 1;
    keep[n_blocks - 1] = 1;   /* recency insurance: last block stays in RAM */

    int evicted = 0;
    for (int b = 0; b < n_blocks - 1; b++) {   /* never evict the last block */
        if (!keep[b]) {
            if (l3_evict_block(s, layer, b, k_cache, v_cache, kv_dim, bs) == 0)
                evicted++;
        }
    }
    free(r);
    free(keep);
    return evicted;
}

/* ================================================================
 * Disk read-back (compressed-state serving for sparse decode)
 * ================================================================ */

/* Byte offset of the (head, token) chunk within a block's K or V area. */
static size_t l3_chunk_off(int head, int token, int n_payloads, int bs,
                           size_t payloads_per_head) {
    return (size_t)head * payloads_per_head * L3_Q4_PAYLOAD64
         + ((size_t)token * (size_t)n_payloads) * L3_Q4_PAYLOAD64;
}

float l3_read_k_dot(const STL3State *s, const STL3Block *bm, int head, int token,
                    const float *q, int hd, int bs, int kv_dim) {
    (void)kv_dim;
    if (!s->fp || !bm || !bm->on_disk) return 0.0f;
    int n_payloads = hd / 64;
    size_t payloads_per_head = (size_t)bs * (size_t)n_payloads;
    size_t k_area = (size_t)s->nkv * payloads_per_head * L3_Q4_PAYLOAD64;
    size_t off = (size_t)bm->disk_off
               + l3_chunk_off(head, token, n_payloads, bs, payloads_per_head);
    uint8_t buf[8 * L3_Q4_PAYLOAD64];
    if (n_payloads > 8) n_payloads = 8;
    if (s->mem) {
        /* RAM mirror: no FILE I/O in the decode hot path. */
        if (off + (size_t)n_payloads * L3_Q4_PAYLOAD64 > s->mem_size) return 0.0f;
        memcpy(buf, s->mem + off, (size_t)n_payloads * L3_Q4_PAYLOAD64);
    } else {
        int fd = L3_FILENO(s->fp);
        if (L3_SEEK64(fd, off) < 0) return 0.0f;
        if (L3_READ(fd, buf, (size_t)n_payloads * L3_Q4_PAYLOAD64)
                != (int)((size_t)n_payloads * L3_Q4_PAYLOAD64)) return 0.0f;
    }
    float sum = 0.0f;
    for (int g = 0; g < n_payloads; g++)
        sum += l3_q4_dot64(buf + (size_t)g * L3_Q4_PAYLOAD64, q + (size_t)g * 64);
    return sum;
}

void l3_read_v_acc(const STL3State *s, const STL3Block *bm, int head, int token,
                   float wgt, float *acc, int hd, int bs, int kv_dim) {
    (void)kv_dim;
    if (!s->fp || !bm || !bm->on_disk) return;
    int n_payloads = hd / 64;
    size_t payloads_per_head = (size_t)bs * (size_t)n_payloads;
    size_t k_area = (size_t)s->nkv * payloads_per_head * L3_Q4_PAYLOAD64;
    size_t off = (size_t)bm->disk_off + k_area
               + l3_chunk_off(head, token, n_payloads, bs, payloads_per_head);
    uint8_t buf[8 * L3_Q4_PAYLOAD64];
    if (n_payloads > 8) n_payloads = 8;
    if (s->mem) {
        if (off + (size_t)n_payloads * L3_Q4_PAYLOAD64 > s->mem_size) return;
        memcpy(buf, s->mem + off, (size_t)n_payloads * L3_Q4_PAYLOAD64);
    } else {
        int fd = L3_FILENO(s->fp);
        if (L3_SEEK64(fd, off) < 0) return;
        if (L3_READ(fd, buf, (size_t)n_payloads * L3_Q4_PAYLOAD64)
                != (int)((size_t)n_payloads * L3_Q4_PAYLOAD64)) return;
    }
    for (int g = 0; g < n_payloads; g++)
        l3_q4_vacc64(acc + (size_t)g * 64, buf + (size_t)g * L3_Q4_PAYLOAD64, wgt);
}

/* ---- Phase-2c: batched (zero-copy / block-granular) read path ---- */

const uint8_t *l3_payload_at(const STL3State *s, const STL3Block *bm,
                             int head, int token, int area,
                             int hd, int bs, int kv_dim) {
    (void)kv_dim;
    if (!s || !bm || !bm->on_disk || !s->mem) return NULL;
    int n_payloads = hd / 64;
    size_t payloads_per_head = (size_t)bs * (size_t)n_payloads;
    size_t k_area = (size_t)s->nkv * payloads_per_head * L3_Q4_PAYLOAD64;
    size_t off = (size_t)bm->disk_off
               + (area ? k_area : 0)
               + (size_t)head * payloads_per_head * L3_Q4_PAYLOAD64
               + (size_t)token * (size_t)n_payloads * L3_Q4_PAYLOAD64;
    size_t nbytes = (size_t)n_payloads * L3_Q4_PAYLOAD64;
    if (off + nbytes > s->mem_size) return NULL;
    return s->mem + off;
}

int l3_fetch_block_area(const STL3State *s, const STL3Block *bm,
                        int head, int area, uint8_t *dst,
                        int hd, int bs, int kv_dim) {
    (void)kv_dim;
    if (!s || !bm || !bm->on_disk || !dst) return -1;
    int n_payloads = hd / 64;
    size_t payloads_per_head = (size_t)bs * (size_t)n_payloads;
    size_t k_area = (size_t)s->nkv * payloads_per_head * L3_Q4_PAYLOAD64;
    size_t off = (size_t)bm->disk_off
               + (area ? k_area : 0)
               + (size_t)head * payloads_per_head * L3_Q4_PAYLOAD64;
    size_t nbytes = (size_t)bs * (size_t)n_payloads * L3_Q4_PAYLOAD64;
    if (s->mem) {
        if (off + nbytes > s->mem_size) return -1;
        memcpy(dst, s->mem + off, nbytes);
        return 0;
    }
    /* No RAM mirror: bulk-read straight from the file (one seek + full read). */
    int fd = L3_FILENO(s->fp);
    if (L3_SEEK64(fd, (long long)off) < 0) return -1;
    size_t got = 0;
    while (got < nbytes) {
        long long n = L3_READ(fd, dst + got, nbytes - got);
        if (n <= 0) return -1;
        got += (size_t)n;
    }
    return 0;
}

/* ================================================================
 * Self-test
 * ================================================================ */

static uint32_t l3_rng = 0x9E3779B9u;
static uint32_t l3_rand(void) {
    l3_rng = l3_rng * 1664525u + 1013904223u;
    return l3_rng;
}

int l3_self_test(void) {
    int fail = 0;
    printf("\n=== L3 (Phase-2) self-test ===\n");

    /* ---- (1) pack/dequant round-trip + saturation ---- */
    {
        float src[64], dst[64];
        uint8_t p[L3_Q4_PAYLOAD64];
        for (int i = 0; i < 64; i++) src[i] = ((float)((int)(l3_rand() >> 8) % 2000) / 1000.0f - 1.0f) * 3.0f;
        l3_q4_pack64(p, src);
        l3_q4_dequant64(dst, p);
        float scale_a; memcpy(&scale_a, p, 4);
        float scale_b; memcpy(&scale_b, p + 20, 4);
        float bound = scale_a / 2.0f, err = 0.0f;
        for (int i = 0; i < 32; i++) { float e = fabsf(dst[i] - src[i]); if (e > err) err = e; }
        for (int i = 32; i < 64; i++) { float e = fabsf(dst[i] - src[i]); if (e > err) err = e; }
        if (scale_b > scale_a) bound = scale_b / 2.0f;
        printf("  [%s] pack/dequant round-trip (max err %.5f <= %.5f)\n",
               err <= bound + 1e-4f ? "PASS" : "FAIL", err, bound);
        if (err > bound + 1e-4f) fail++;

        /* saturation: values well beyond +/-7 scale must clamp */
        for (int i = 0; i < 64; i++) src[i] = (i % 2 == 0) ? 100.0f : -100.0f;
        l3_q4_pack64(p, src);
        l3_q4_dequant64(dst, p);
        float maxd = 0.0f;
        for (int i = 0; i < 64; i++) { float e = fabsf(fabsf(dst[i]) - 100.0f); if (e > maxd) maxd = e; }
        printf("  [%s] saturation clamps to +/-100 (max dev %.5f)\n",
               maxd < 0.5f ? "PASS" : "FAIL", maxd);
        if (maxd >= 0.5f) fail++;
    }

    /* ---- (2) dot64 vs scalar reference ---- */
    {
        float src[64], act[64];
        uint8_t p[L3_Q4_PAYLOAD64];
        for (int i = 0; i < 64; i++) {
            src[i] = ((float)((int)(l3_rand() >> 8) % 2000) / 1000.0f - 1.0f) * 4.0f;
            act[i] = ((float)((int)(l3_rand() >> 8) % 2000) / 1000.0f - 1.0f) * 2.0f;
        }
        l3_q4_pack64(p, src);
        float got = l3_q4_dot64(p, act);
        float ref = q4_dot64_scalar(p, act);
        float rel = ref != 0.0f ? fabsf(got - ref) / fabsf(ref) : fabsf(got);
        printf("  [%s] q4 dot vs scalar reference (rel %.2e)\n",
               rel < 1e-5f ? "PASS" : "FAIL", rel);
        if (rel >= 1e-5f) fail++;

        /* determinism: identical inputs -> bitwise identical dot */
        float got2 = l3_q4_dot64(p, act);
        uint32_t b1, b2; memcpy(&b1, &got, 4); memcpy(&b2, &got2, 4);
        printf("  [%s] q4 dot bitwise deterministic\n", b1 == b2 ? "PASS" : "FAIL");
        if (b1 != b2) fail++;

        /* vacc64 vs manual accumulation */
        float acc[64], refacc[64];
        memset(acc, 0, sizeof(acc)); memset(refacc, 0, sizeof(refacc));
        float wgt = 0.37f;
        l3_q4_vacc64(acc, p, wgt);
        float scale_a, scale_b;
        memcpy(&scale_a, p, 4); memcpy(&scale_b, p + 20, 4);
        for (int i = 0; i < 16; i++) {
            uint8_t b = p[4 + i];
            refacc[i * 2]     = wgt * (float)((int)(b & 0x0F) - 8) * scale_a;
            refacc[i * 2 + 1] = wgt * (float)((int)((b >> 4) & 0x0F) - 8) * scale_a;
        }
        for (int i = 0; i < 16; i++) {
            uint8_t b = p[24 + i];
            refacc[32 + i * 2]     = wgt * (float)((int)(b & 0x0F) - 8) * scale_b;
            refacc[32 + i * 2 + 1] = wgt * (float)((int)((b >> 4) & 0x0F) - 8) * scale_b;
        }
        float vacc_err = 0.0f;
        for (int i = 0; i < 64; i++) { float e = fabsf(acc[i] - refacc[i]); if (e > vacc_err) vacc_err = e; }
        printf("  [%s] q4 vacc vs manual accumulation (max err %.3e)\n",
               vacc_err < 1e-6f ? "PASS" : "FAIL", vacc_err);
        if (vacc_err >= 1e-6f) fail++;
    }

    /* ---- (3) eviction: determinism + hot-block retention + disk round-trip ---- */
    {
        const char *path = "kv_l3_selftest.bin";
        int nl = 1, nkv = 2, hd = 128, bs = 32, max_blocks = 64;
        int kv_dim = nkv * hd;
        int seq_len = 512;
        STL3State s1, s2;

        float *k1 = (float *)malloc((size_t)seq_len * kv_dim * sizeof(float));
        float *v1 = (float *)malloc((size_t)seq_len * kv_dim * sizeof(float));
        float *imp = (float *)malloc((size_t)seq_len * sizeof(float));
        float *k2 = (float *)malloc((size_t)seq_len * kv_dim * sizeof(float));
        float *v2 = (float *)malloc((size_t)seq_len * kv_dim * sizeof(float));
        if (!k1 || !v1 || !imp || !k2 || !v2) {
            printf("  [FAIL] eviction test OOM\n");
            free(k1); free(v1); free(imp); free(k2); free(v2);
            return 1;
        }
        /* Per-block views over the contiguous test buffers */
        int n_blk = seq_len / bs;
        float **k1b = (float **)malloc((size_t)n_blk * sizeof(float *));
        float **v1b = (float **)malloc((size_t)n_blk * sizeof(float *));
        float **k2b = (float **)malloc((size_t)n_blk * sizeof(float *));
        float **v2b = (float **)malloc((size_t)n_blk * sizeof(float *));
        if (!k1b || !v1b || !k2b || !v2b) {
            printf("  [FAIL] eviction test OOM (block views)\n");
            free(k1); free(v1); free(imp); free(k2); free(v2);
            free(k1b); free(v1b); free(k2b); free(v2b);
            return 1;
        }
        for (int b = 0; b < n_blk; b++) {
            k1b[b] = k1 + (size_t)b * bs * kv_dim;
            v1b[b] = v1 + (size_t)b * bs * kv_dim;
            k2b[b] = k2 + (size_t)b * bs * kv_dim;
            v2b[b] = v2 + (size_t)b * bs * kv_dim;
        }
        for (int t = 0; t < seq_len * kv_dim; t++) {
            k1[t] = ((float)((int)(l3_rand() >> 8) % 2000) / 1000.0f - 1.0f) * 3.0f;
            v1[t] = ((float)((int)(l3_rand() >> 8) % 2000) / 1000.0f - 1.0f) * 3.0f;
            k2[t] = k1[t]; v2[t] = v1[t];
        }
        for (int t = 0; t < seq_len; t++) imp[t] = ((float)((int)(l3_rand() >> 8) % 1000)) / 1000.0f;
        imp[100] = 100.0f;   /* needle spike inside block 3 */

        int e1 = 0, e2 = 0;
        if (l3_state_init(&s1, path, nl, nkv, hd, bs, max_blocks) == 0) {
            e1 = l3_evict_layer(&s1, 0, k1b, v1b, kv_dim, seq_len, imp, 0.75f, bs);
            l3_state_free(&s1);
        }
        if (l3_state_init(&s2, path, nl, nkv, hd, bs, max_blocks) == 0) {
            e2 = l3_evict_layer(&s2, 0, k2b, v2b, kv_dim, seq_len, imp, 0.75f, bs);
            l3_state_free(&s2);
        }
        int n_blocks = (seq_len + bs - 1) / bs;   /* 16 */
        printf("  [%s] eviction deterministic (%d == %d blocks evicted)\n",
               (e1 == e2 && e1 > 0) ? "PASS" : "FAIL", e1, e2);
        if (!(e1 == e2 && e1 > 0)) fail++;

        /* hot needle block (3) and last block (15) must never be evicted */
        uint8_t on_disk[32];
        memset(on_disk, 0, sizeof(on_disk));
        if (l3_state_init(&s1, path, nl, nkv, hd, bs, max_blocks) == 0) {
            l3_evict_layer(&s1, 0, k1b, v1b, kv_dim, seq_len, imp, 0.75f, bs);
            for (int b = 0; b < n_blocks; b++)
                on_disk[b] = s1.blocks[b].on_disk;
            l3_state_free(&s1);
        }
        printf("  [%s] hot needle block + last block retained (on_disk: b3=%d b15=%d)\n",
               (on_disk[3] == 0 && on_disk[n_blocks - 1] == 0) ? "PASS" : "FAIL",
               on_disk[3], on_disk[n_blocks - 1]);
        if (!(on_disk[3] == 0 && on_disk[n_blocks - 1] == 0)) fail++;

        /* disk round-trip: reopen, read one evicted block's K area, dequant,
         * compare against the original float with the quantization bound. */
        if (l3_state_init(&s1, path, nl, nkv, hd, bs, max_blocks) == 0) {
            l3_evict_layer(&s1, 0, k1b, v1b, kv_dim, seq_len, imp, 0.75f, bs);
            if (fflush(s1.fp) != 0) { fail++; printf("  [FAIL] fflush\n"); }
            int evicted_b = -1;
            for (int b = 0; b < n_blocks; b++)
                if (s1.blocks[b].on_disk) { evicted_b = b; break; }
            if (evicted_b >= 0) {
                uint8_t p[L3_Q4_PAYLOAD64];
                float worst = 0.0f;
                for (int h = 0; h < nkv && !fail; h++) {
                    for (int t = 0; t < bs; t++) {
                        for (int g = 0; g < hd / 64; g++) {
                            size_t foff = s1.blocks[evicted_b].disk_off
                                          + (size_t)h * ((size_t)bs * (hd / 64)) * L3_Q4_PAYLOAD64
                                          + ((size_t)t * (hd / 64) + (size_t)g) * L3_Q4_PAYLOAD64;
                            int sfd = L3_FILENO(s1.fp);
                            if (L3_SEEK64(sfd, (long long)foff) < 0 ||
                                L3_READ(sfd, p, L3_Q4_PAYLOAD64) != L3_Q4_PAYLOAD64) {
                                fail++;
                                printf("  [FAIL] disk read back\n");
                                break;
                            }
                            float dst[64];
                            l3_q4_dequant64(dst, p);
                            const float *orig = k1 + ((size_t)(evicted_b * bs + t)) * kv_dim
                                                + (size_t)h * hd + (size_t)g * 64;
                            float scale; memcpy(&scale, p, 4);
                            for (int i = 0; i < 64; i++) {
                                float e = fabsf(dst[i] - orig[i]);
                                if (e > worst) worst = e;
                            }
                        }
                    }
                }
                printf("  [%s] evicted K block disk round-trip (max err %.5f vs float)\n",
                       worst < 0.5f ? "PASS" : "FAIL", worst);
                if (worst >= 0.5f) fail++;
            }
            l3_state_free(&s1);
        }
        /* Phase-2c batching: bulk-fetch + kernels must be bitwise identical to
         * the per-token read path (same Q4 payloads, same dot/vacc kernels). */
        if (l3_state_init(&s1, path, nl, nkv, hd, bs, max_blocks) == 0) {
            l3_evict_layer(&s1, 0, k1b, v1b, kv_dim, seq_len, imp, 0.75f, bs);
            if (l3_load_to_mem(&s1) != 0) {
                printf("  [FAIL] phase-2c mirror\n");
                fail++;
            } else {
                int evicted_b = -1;
                for (int b = 0; b < n_blocks; b++)
                    if (s1.blocks[b].on_disk) { evicted_b = b; break; }
                int batched_ok = evicted_b >= 0;
                if (evicted_b >= 0) {
                    size_t chunk = (size_t)bs * (size_t)(hd / 64) * L3_Q4_PAYLOAD64;
                    uint8_t *kbuf = (uint8_t *)malloc(chunk);
                    uint8_t *vbuf = (uint8_t *)malloc(chunk);
                    if (!kbuf || !vbuf) { batched_ok = 0; }
                    if (batched_ok) {
                        float q[128];
                        for (int i = 0; i < hd; i++)
                            q[i] = ((float)((int)(l3_rand() >> 8) % 2000) / 1000.0f - 1.0f) * 2.0f;
                        for (int h = 0; h < nkv && batched_ok; h++) {
                            if (l3_fetch_block_area(&s1, &s1.blocks[evicted_b], h, 0, kbuf, hd, bs, kv_dim) != 0 ||
                                l3_fetch_block_area(&s1, &s1.blocks[evicted_b], h, 1, vbuf, hd, bs, kv_dim) != 0) {
                                batched_ok = 0; break;
                            }
                            float acc_ref[128], acc_bat[128];
                            memset(acc_ref, 0, sizeof(acc_ref));
                            memset(acc_bat, 0, sizeof(acc_bat));
                            for (int t = 0; t < bs && batched_ok; t++) {
                                float d_ref = l3_read_k_dot(&s1, &s1.blocks[evicted_b], h, t, q, hd, bs, kv_dim);
                                const uint8_t *kp = kbuf + (size_t)t * (size_t)(hd / 64) * L3_Q4_PAYLOAD64;
                                float d_bat = 0.0f;
                                for (int g = 0; g < hd / 64; g++)
                                    d_bat += l3_q4_dot64(kp + (size_t)g * L3_Q4_PAYLOAD64, q + (size_t)g * 64);
                                if (d_ref != d_bat) { batched_ok = 0; break; }
                                float w = ((float)((int)(l3_rand() >> 8) % 1000)) / 1000.0f;
                                l3_read_v_acc(&s1, &s1.blocks[evicted_b], h, t, w, acc_ref, hd, bs, kv_dim);
                                const uint8_t *vp = vbuf + (size_t)t * (size_t)(hd / 64) * L3_Q4_PAYLOAD64;
                                for (int g = 0; g < hd / 64; g++)
                                    l3_q4_vacc64(acc_bat + (size_t)g * 64, vp + (size_t)g * L3_Q4_PAYLOAD64, w);
                            }
                            if (batched_ok)
                                for (int i = 0; i < hd; i++)
                                    if (acc_ref[i] != acc_bat[i]) { batched_ok = 0; break; }
                        }
                    }
                    free(kbuf); free(vbuf);
                }
                printf("  [%s] phase-2c batch == per-token read (bitwise)\n",
                       batched_ok ? "PASS" : "FAIL");
                if (!batched_ok) fail++;
            }
            l3_state_free(&s1);
        }
        remove(path);
        free(k1); free(v1); free(imp); free(k2); free(v2);
        free(k1b); free(v1b); free(k2b); free(v2b);
    }

    printf("=== L3 self-test %s (%d failures) ===\n", fail == 0 ? "PASSED" : "FAILED", fail);
    return fail;
}

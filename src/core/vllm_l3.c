/**
 * vllm_l3.c - Phase 2: cold-block Q4 disk eviction for the KV cache
 *
 * See vllm_l3.h for the format and the honest-scope design notes.
 * Determinism red line (axiom_arith_gumbel_argmax_001): quantization, the
 * dot kernel and the eviction ranking are all deterministic for identical
 * inputs (fixed rounding, fixed tie-break, cursor-allocated offsets in
 * deterministic eviction order).
 */

#include "vllm_l3.h"
#include "vllm_platform.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>   /* fileno / lseek / read / fseeko on Linux/RK3588 */
#ifdef __linux__
#include <sys/mman.h> /* madvise(MADV_DONTNEED)：归还镜像的驻留页 */
#endif

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

/* 公共实现：逐字段显式初始化（不再 memset 整个结构），以便把调用方交回的
 * 跨轮镜像挂上去（P1）。mirror = NULL / cap = 0 即全新状态，与旧 memset
 * 语义等价；任何失败路径都把 mirror 归还给堆，调用方失败后不再持有它。 */
static int l3_state_init_common(STL3State *s, const char *path, int nl, int nkv,
                                int hd, int bs, int max_blocks,
                                uint8_t *mirror, size_t mirror_cap) {
    if (!s) return -1;
    s->fp = NULL;
    s->nl = 0; s->nkv = 0; s->hd = 0; s->bs = 0; s->max_blocks = 0;
    s->block_bytes = 0;
    s->blocks = NULL;
    s->evicted = 0;
    s->wcursor = 0;
    s->mem = mirror;
    s->mem_cap = mirror_cap;
    s->mem_size = 0;   /* 本轮尚未写入：load 之前镜像读一律被边界挡掉 */
    s->path[0] = '\0';

    if (!path || nl < 1 || nkv < 1 || hd < 64 || (hd % 64) != 0 || bs < 1 || max_blocks < 1) {
        free(mirror);
        s->mem = NULL; s->mem_cap = 0;
        return -1;
    }

    int n_payloads = hd / 64;
    size_t payloads_per_head = (size_t)bs * (size_t)n_payloads;
    size_t k_area = (size_t)nkv * payloads_per_head * L3_Q4_PAYLOAD64;
    s->block_bytes = 2 * k_area;

    s->fp = fopen(path, "wb+");
    if (!s->fp) { free(mirror); s->mem = NULL; s->mem_cap = 0; return -1; }
    s->nl = nl; s->nkv = nkv; s->hd = hd; s->bs = bs; s->max_blocks = max_blocks;
    strncpy(s->path, path, sizeof(s->path) - 1);
    s->path[sizeof(s->path) - 1] = '\0';
    s->wcursor = L3_HEADER_SIZE;   /* compact layout: pack evicted blocks contiguously */

    s->blocks = calloc((size_t)nl * (size_t)max_blocks, sizeof(STL3Block));
    if (!s->blocks) { l3_state_free(s); return -1; }

    if (write_header(s) != 0) { l3_state_free(s); return -1; }
    fflush(s->fp);
    return 0;
}

int l3_state_init(STL3State *s, const char *path, int nl, int nkv, int hd,
                  int bs, int max_blocks) {
    return l3_state_init_common(s, path, nl, nkv, hd, bs, max_blocks, NULL, 0);
}

int l3_state_init_reuse(STL3State *s, const char *path, int nl, int nkv, int hd,
                        int bs, int max_blocks,
                        uint8_t *reuse_mem, size_t reuse_cap) {
    return l3_state_init_common(s, path, nl, nkv, hd, bs, max_blocks,
                                reuse_mem, reuse_cap);
}

void l3_state_free(STL3State *s) {
    if (!s) return;
    if (s->fp) { fclose(s->fp); s->fp = NULL; }
    free(s->mem);
    s->mem = NULL;
    s->mem_size = 0;
    s->mem_cap = 0;
    free(s->blocks);
    s->blocks = NULL;
    s->evicted = 0;
}

/* P1：保留镜像的重置。与 l3_state_free 的差别只在于 mirror 的归属——这里把
 * 它摘出来交给调用方，供下一轮 l3_state_init_reuse 覆盖写复用，省掉每轮
 * free+malloc+memset 以及随之而来的全量触页（D1 的 +51.7MB/轮 残余）。 */
void l3_state_rewind(STL3State *s, uint8_t **mirror_out, size_t *cap_out) {
    if (mirror_out) *mirror_out = NULL;
    if (cap_out) *cap_out = 0;
    if (!s) return;
    if (s->fp) { fclose(s->fp); s->fp = NULL; }
    free(s->blocks);
    s->blocks = NULL;
    s->evicted = 0;
    if (mirror_out) *mirror_out = s->mem;
    if (cap_out) *cap_out = s->mem_cap;
    s->mem = NULL;
    s->mem_size = 0;
    s->mem_cap = 0;
}

/* P3 增量：见 vllm_l3.h。要求 mem 已就绪 —— 镜像没建起来就退回每轮重建。 */
int l3_state_matches(const STL3State *s, const char *path, int nl, int nkv,
                     int hd, int bs, int max_blocks) {
    if (!s || !s->fp || !s->blocks || !s->mem) return 0;
    if (s->nl != nl || s->nkv != nkv || s->hd != hd || s->bs != bs ||
        s->max_blocks != max_blocks) return 0;
    if (!path || !path[0] || !s->path[0]) return 0;
    if (strcmp(s->path, path) != 0) return 0;
    return 1;
}

/* 把 [off, off+len) 覆盖到的完整页归还内核（Linux MADV_DONTNEED）。匿名私有
 * 映射被丢弃后再读回零页，与"未写区域读作 0"的既有语义一致；非 Linux 平台
 * 空实现（主力目标端是 RK3588/Linux，Windows 仅编译与冒烟）。 */
static void l3_mirror_discard(STL3State *s, size_t off, size_t len) {
#ifdef __linux__
    if (!s || !s->mem || len == 0) return;
    long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0) return;
    uintptr_t a = (uintptr_t)s->mem + off;
    uintptr_t lo = (a + (uintptr_t)pg - 1) & ~((uintptr_t)pg - 1); /* 起点上对齐到页 */
    uintptr_t hi = (a + len) & ~((uintptr_t)pg - 1);               /* 终点下对齐到页 */
    if (hi > lo) madvise((void *)lo, (size_t)(hi - lo), MADV_DONTNEED);
#else
    (void)s; (void)off; (void)len;
#endif
}

/* Mirror the whole logical payload region into RAM so decode-time reads never
 * touch the FILE (the decode hot path historically corrupted the heap when
 * driven through fseek/fread on the shared fp). The file may be shorter than
 * the full logical extent (only evicted blocks are written); zero-fill the
 * remainder so an unwritten region reads as a benign zero payload.
 * The read itself goes through the raw fd (_lseeki64/_read) instead of the
 * CRT stream: mixing fwrite with fseek/fread on the same buffered FILE* is
 * exactly the shared-fp churn that corrupted the heap. Pending fwrite output
 * is flushed first so the raw fd observes the complete file.
 *
 * P1 跨轮复用：容量足够时不再 free+malloc+memset，而是复用调用方交回的同一
 * 缓冲覆盖写 [0, extent)，并把 [extent, mem_cap) 的驻留页归还内核——这两处
 * 正是"每轮映射重建 + write-touch"残余（+51.7MB/轮）的来源。可读范围仍由
 * mem_size = extent 界定，本轮 extent 内全部被覆盖读，故复用不改变可见内容。 */
int l3_load_to_mem(STL3State *s) {
    if (!s || !s->fp) return 0;
    /* Mirror only the actual written extent (the highest evicted block's end
     * offset), not the full logical capacity nl*max_blocks*block_bytes.
     * Full-capacity sizing malloc+memset'd every page of the whole KV window
     * on EVERY eviction round (~280 MiB at 8192 max_seq) and the per-round
     * re-init never freed the previous mirror, so serve RSS ratcheted
     * ~+309 MB per round (diag 2026-09-07, doc §5.1). All evicted blocks are
     * marked on_disk before this runs, so scanning the metadata yields the
     * exact byte range decode can ever read (hole/sparse regions and any
     * short-file tail stay zero-filled below). */
    size_t extent = 0;
    int n_meta = s->nl * s->max_blocks;
    for (int i = 0; i < n_meta; i++) {
        if (s->blocks[i].on_disk) {
            size_t end = (size_t)s->blocks[i].disk_off + s->block_bytes;
            if (end > extent) extent = end;
        }
    }
    /* 本轮无驱逐：镜像不再承载可读数据。归还全部驻留页但保留缓冲，
     * 供下轮继续复用（旧实现此处直接返回，把重建成本留给了下一轮）。 */
    if (extent == 0) {
        l3_mirror_discard(s, 0, s->mem_cap);
        s->mem_size = 0;
        return 0;
    }
    /* 只在容量不足时重新分配，并留出余量：长上下文逐轮递增时避免每轮
     * 都因差一个块而重建。未触页的余量不占物理内存。 */
    if (!s->mem || s->mem_cap < extent) {
        size_t need = extent + extent / 4 + (size_t)4 * s->block_bytes;
        if (need < extent) need = extent;   /* 溢出兜底 */
        free(s->mem);
        s->mem = (uint8_t *)malloc(need);
        if (!s->mem) { s->mem_cap = 0; return -1; }
        s->mem_cap = need;
    }
    /* 本轮读不到的尾部：把上轮遗留的驻留页交还内核，避免跨轮累积 */
    l3_mirror_discard(s, extent, s->mem_cap - extent);

    if (fflush(s->fp) != 0) { s->mem_size = 0; return -1; }
    int fd = L3_FILENO(s->fp);
    if (L3_SEEK64(fd, 0) < 0) { s->mem_size = 0; return -1; }
    size_t got = 0;
    while (got < extent) {
        long long n = L3_READ(fd, s->mem + got, extent - got);
        if (n <= 0) break;   /* partial/EOF: tail must be zero-filled below */
        got += (size_t)n;
    }
    /* 短读（文件短于 extent）必须显式清零：复用缓冲时残留的是上一轮数据，
     * 不能漏给下游（未写 payload 的既有语义是全零）。 */
    if (got < extent) memset(s->mem + got, 0, extent - got);
    s->mem_size = extent;
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

    /* Compact layout: (layer, block) maps to its metadata slot by index, but
     * the disk offset is cursor-allocated in eviction order so the file and
     * the RAM mirror track the ACTUAL evicted payload (~ratio of max_seq)
     * instead of the full KV window (see l3_state_init wcursor).
     * P3 增量：块已有 slot（on_disk）时**原地覆盖**、游标不前进 —— 否则每轮
     * 重打包都新占一段，文件与镜像无界膨胀。 */
    STL3Block *bm = &s->blocks[(size_t)layer * (size_t)s->max_blocks + (size_t)block];
    size_t base;
    if (bm->on_disk) {
        base = (size_t)bm->disk_off;
    } else {
        base = s->wcursor;
        if (s->wcursor > (size_t)-1 - s->block_bytes) return -1;
        s->wcursor += s->block_bytes;
    }
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

    bm->on_disk = 1;
    bm->stale = 0;            /* 刚按当前内容重打包：载荷重新有效 */
    bm->disk_off = (uint64_t)base;
    s->evicted++;
    return 0;
}

/* Eviction ranking: importance mass desc, block index asc (deterministic).
 * Fills keep[b]=1 for the blocks that STAY in RAM (top (1-ratio) by mass,
 * plus the most recent block). n_blocks must be > 1 (last block kept). */
typedef struct { float mass; int idx; } Rank;
static int rank_cmp(const void *a, const void *b) {
    const Rank *ra = (const Rank *)a, *rb = (const Rank *)b;
    if (ra->mass > rb->mass) return -1;
    if (ra->mass < rb->mass) return 1;
    return ra->idx - rb->idx;
}
static void l3_pick_evict_blocks(int n_blocks, int seq_len,
                                 const float *importance, float ratio,
                                 int bs, uint8_t *keep) {
    Rank *r = (Rank *)malloc((size_t)n_blocks * sizeof(Rank));
    if (!r) { memset(keep, 1, (size_t)n_blocks); return; }   /* OOM: keep all */
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
    free(r);
}

int l3_evict_layer(STL3State *s, int layer,
                   const float *const *k_cache, const float *const *v_cache,
                   int kv_dim, int seq_len, const float *importance,
                   float ratio, int bs) {
    if (!s->fp || seq_len <= 0 || !k_cache || !v_cache) return 0;
    int n_blocks = (seq_len + bs - 1) / bs;
    if (n_blocks <= 1) return 0;

    uint8_t *keep = (uint8_t *)calloc((size_t)n_blocks, 1);
    if (!keep) return 0;
    l3_pick_evict_blocks(n_blocks, seq_len, importance, ratio, bs, keep);

    int evicted = 0;
    for (int b = 0; b < n_blocks; b++) {
        STL3Block *bm = &s->blocks[(size_t)layer * (size_t)s->max_blocks + (size_t)b];
        bm->evict = 0;
        if (b >= n_blocks - 1 || keep[b]) continue;   /* never evict the last block */
        bm->evict = 1;
        /* P3 增量：载荷仍有效（该块未被改写）→ 不重打包，只按 evict 释放 RAM。
         * 这一分支就是消掉 R1「每轮 Q4→反量化→Q4」重复量化的关键。 */
        if (bm->on_disk && !bm->stale) continue;
        if (l3_evict_block(s, layer, b, k_cache, v_cache, kv_dim, bs) == 0)
            evicted++;
    }
    free(keep);
    return evicted;
}

/* P2 (nof32): q8-source eviction. No f32 canonical rows exist, so the Q4
 * mirrors are packed from the dequantized q8 values (q*(scale/127), same
 * formula as kv_dequant_roundtrip) — the mirror then matches what q8 sparse
 * decode would read from RAM, and is identical to the f32-world mirror when
 * the f32 rows carry the q8 dequant (P0 ON 档). */
int l3_evict_block_q8(STL3State *s, int layer, int block,
                      const int8_t *const *k_q8, const int8_t *const *v_q8,
                      const float *kscale, const float *vscale,
                      int kv_dim, int bs, int nkv) {
    if (!s->fp) return -1;
    if (layer < 0 || block < 0 || layer >= s->nl || block >= s->max_blocks) return -1;
    if (!k_q8 || !k_q8[block] || !v_q8 || !v_q8[block]) return -1;  /* already evicted */

    int hd = s->hd;
    int n_payloads = hd / 64;
    size_t payloads_per_head = (size_t)bs * (size_t)n_payloads;
    size_t k_area = (size_t)nkv * payloads_per_head * L3_Q4_PAYLOAD64;

    /* Compact layout (same cursor scheme as l3_evict_block): disk offsets are
     * allocated in eviction order so file + mirror track the evicted payload
     * rather than the full KV window. P3 增量：已有 slot 的块原地覆盖。 */
    STL3Block *bm = &s->blocks[(size_t)layer * (size_t)s->max_blocks + (size_t)block];
    size_t base;
    if (bm->on_disk) {
        base = (size_t)bm->disk_off;
    } else {
        base = s->wcursor;
        if (s->wcursor > (size_t)-1 - s->block_bytes) return -1;
        s->wcursor += s->block_bytes;
    }
    const int8_t *kbase = k_q8[block];   /* [bs * kv_dim] */
    const int8_t *vbase = v_q8[block];
    uint8_t tmp[L3_Q4_PAYLOAD64];
    float  row64[64];
    /* Per-token per-head q8 scale (token-major, contiguous [max_seq*nkv]). */
    const float *ksc = kscale + (size_t)block * bs * (size_t)nkv;
    const float *vsc = vscale + (size_t)block * bs * (size_t)nkv;

    /* K area then V area: [head][token][hd/64 payloads] */
    for (int area = 0; area < 2; area++) {
        const int8_t *src = area == 0 ? kbase : vbase;
        const float *sc  = area == 0 ? ksc : vsc;
        for (int h = 0; h < nkv; h++) {
            for (int t = 0; t < bs; t++) {
                const int8_t *row = src + (size_t)t * (size_t)kv_dim + (size_t)h * (size_t)hd;
                float srow = sc[(size_t)t * (size_t)nkv + (size_t)h] * (1.0f / 127.0f);
                for (int g = 0; g < n_payloads; g++) {
                    const int8_t *p64 = row + (size_t)g * 64;
                    for (int i = 0; i < 64; i++) row64[i] = (float)p64[i] * srow;
                    l3_q4_pack64(tmp, row64);
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

    bm->on_disk = 1;
    bm->stale = 0;            /* 刚按当前内容重打包：载荷重新有效 */
    bm->disk_off = (uint64_t)base;
    s->evicted++;
    return 0;
}

int l3_evict_layer_q8(STL3State *s, int layer,
                      const int8_t *const *k_q8, const int8_t *const *v_q8,
                      const float *kscale, const float *vscale,
                      int kv_dim, int seq_len, const float *importance,
                      float ratio, int bs, int nkv) {
    if (!s->fp || seq_len <= 0 || !k_q8 || !v_q8 || !kscale || !vscale) return 0;
    int n_blocks = (seq_len + bs - 1) / bs;
    if (n_blocks <= 1) return 0;

    uint8_t *keep = (uint8_t *)calloc((size_t)n_blocks, 1);
    if (!keep) return 0;
    l3_pick_evict_blocks(n_blocks, seq_len, importance, ratio, bs, keep);

    int evicted = 0;
    for (int b = 0; b < n_blocks; b++) {
        STL3Block *bm = &s->blocks[(size_t)layer * (size_t)s->max_blocks + (size_t)b];
        bm->evict = 0;
        if (b >= n_blocks - 1 || keep[b]) continue;   /* never evict the last block */
        bm->evict = 1;
        /* P3 增量：载荷仍有效 → 不重打包（消 R1）。 */
        if (bm->on_disk && !bm->stale) continue;
        if (l3_evict_block_q8(s, layer, b, k_q8, v_q8,
                              kscale, vscale, kv_dim, bs, nkv) == 0)
            evicted++;
    }
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

/* P3: 见 vllm_l3.h。把 on_disk 块的内容反量化回调用方已分配好的 K/V 行。
 * 纯解压路径：无 GEMM、无注意力、按 (head, token) 顺序逐 64 元素载荷展开。 */
int l3_fill_block_from_disk(const STL3State *s, const STL3Block *bm, int block,
                            float **kf, float **vf, int8_t **k8, int8_t **v8,
                            const float *kscale, const float *vscale,
                            int kv_dim, int bs, int hd) {
    if (!s || !bm || !bm->on_disk) return -1;
    if (block < 0 || block >= s->max_blocks) return -1;
    if (bs <= 0 || hd <= 0 || hd % 64 != 0) return -1;
    if (s->nkv <= 0 || kv_dim != s->nkv * hd) return -1;
    int has_f = (kf != NULL) && (vf != NULL);
    int has_i = (k8 != NULL) && (v8 != NULL);
    if (!has_f && !has_i) return -1;
    if (has_f && (!kf[block] || !vf[block])) return -1;   /* 目标未分配 */
    if (has_i && (!k8[block] || !v8[block])) return -1;
    if (has_i && (!kscale || !vscale)) return -1;

    int n_payloads = hd / 64;
    size_t area_bytes = (size_t)bs * (size_t)n_payloads * L3_Q4_PAYLOAD64;
    uint8_t *chunk = (uint8_t *)malloc(area_bytes);
    if (!chunk) return -1;

    /* 两个世界都从同一份 Q4 载荷派生，保证 f32 与 q8 的一致性（见头注释）。 */
    for (int area = 0; area < 2; area++) {
        float       *frow_base = has_f ? (area ? vf[block] : kf[block]) : NULL;
        int8_t      *irow_base = has_i ? (area ? v8[block] : k8[block]) : NULL;
        const float *sc        = has_i ? (area ? vscale : kscale) : NULL;
        for (int h = 0; h < s->nkv; h++) {
            if (l3_fetch_block_area(s, bm, h, area, chunk, hd, bs, kv_dim) != 0) {
                free(chunk);
                return -1;
            }
            for (int t = 0; t < bs; t++) {
                float  *fdst = frow_base ? frow_base
                               + (size_t)t * (size_t)kv_dim + (size_t)h * (size_t)hd
                               : NULL;
                int8_t *idst = irow_base ? irow_base
                               + (size_t)t * (size_t)kv_dim + (size_t)h * (size_t)hd
                               : NULL;
                float maxk = 0.0f;
                if (sc) {
                    maxk = sc[(size_t)t * (size_t)s->nkv + (size_t)h];
                    if (!(maxk > 1e-6f)) maxk = 1.0f;  /* 与 kv_quantize_per_head 同口径 */
                }
                float ik = L3_Q8_INV_SCALE / maxk;
                float sk = maxk * (1.0f / L3_Q8_INV_SCALE);
                const uint8_t *cp = chunk
                    + (size_t)t * (size_t)n_payloads * L3_Q4_PAYLOAD64;
                for (int g = 0; g < n_payloads; g++) {
                    float row64[64];
                    l3_q4_dequant64(row64, cp + (size_t)g * L3_Q4_PAYLOAD64);
                    if (!idst) {   /* 纯 f32 世界：直接落 Q4 反量化值 */
                        memcpy(fdst + (size_t)g * 64, row64, 64 * sizeof(float));
                        continue;
                    }
                    for (int i = 0; i < 64; i++) {
                        int q = (int)floorf(row64[i] * ik + 0.5f);
                        if (q > 127) q = 127; else if (q < -128) q = -128;
                        idst[(size_t)g * 64 + (size_t)i] = (int8_t)q;
                        if (fdst) fdst[(size_t)g * 64 + (size_t)i] = (float)q * sk;
                    }
                }
            }
        }
    }
    free(chunk);
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
        /* ---- (10) P1 跨轮镜像复用：复用缓冲读出的内容必须与每轮全新缓冲
         * 逐字节一致。三轮覆盖三条路径：容量足够（直接复用）→ extent 变大
         * （复用后扩容）→ extent 变小（复用且尾部页被归还）。每轮都用旧路径
         * （state_free + init + load，缓冲每轮重建）做对照，比较 [0, mem_size)；
         * 若复用残留了上一轮数据，memcmp 必然不同。 */
        {
            const float ratios[3] = { 0.30f, 0.85f, 0.60f };
            STL3State R, N;
            memset(&R, 0, sizeof(R));
            memset(&N, 0, sizeof(N));
            uint8_t *pm = NULL;
            size_t pc = 0;
            int reuse_ok = 1;
            for (int round = 0; round < 3 && reuse_ok; round++) {
                /* 对照 N：每轮重建缓冲 */
                l3_state_free(&N);
                if (l3_state_init(&N, path, nl, nkv, hd, bs, max_blocks) != 0) { reuse_ok = 0; break; }
                l3_evict_layer(&N, 0, k1b, v1b, kv_dim, seq_len, imp, ratios[round], bs);
                if (l3_load_to_mem(&N) != 0) { reuse_ok = 0; break; }
                /* 被测 R：首轮全新，其后 rewind 摘出镜像交回 init_reuse */
                if (round == 0) {
                    if (l3_state_init(&R, path, nl, nkv, hd, bs, max_blocks) != 0) { reuse_ok = 0; break; }
                } else {
                    l3_state_rewind(&R, &pm, &pc);
                    if (l3_state_init_reuse(&R, path, nl, nkv, hd, bs, max_blocks,
                                            pm, pc) != 0) { pm = NULL; reuse_ok = 0; break; }
                    pm = NULL; pc = 0;
                }
                l3_evict_layer(&R, 0, k1b, v1b, kv_dim, seq_len, imp, ratios[round], bs);
                if (l3_load_to_mem(&R) != 0) { reuse_ok = 0; break; }
                if (R.mem_size != N.mem_size || R.evicted != N.evicted ||
                    R.mem_cap < R.mem_size || memcmp(R.mem, N.mem, R.mem_size) != 0)
                    reuse_ok = 0;
            }
            free(pm);   /* 中途失败时镜像尚在手上 */
            l3_state_free(&R);
            l3_state_free(&N);
            printf("  [%s] P1 cross-round mirror reuse == per-round rebuild (bytewise)\n",
                   reuse_ok ? "PASS" : "FAIL");
            if (!reuse_ok) fail++;
        }
        /* ---- (11) P3 前缀重建：l3_fill_block_from_disk 把某块从 Q4 载荷
         * 反量化回**调用方已分配**的行。三条要求：
         *   ① 值确实来自该块（误差在 Q4 半格 + q8 半步内）；
         *   ② 重复填充逐位一致（确定性）；
         *   ③ 同时给 f32 与 q8 时 f32 == (float)q8 * (scale/127) 位级成立
         *      （与 kv_dequant_roundtrip 同一表达式，禁止另起口径）。 */
        {
            int bound_ok = 1, det_ok = 1, bit_ok = 1, ran_ok = 1;
            float *imp2 = (float *)malloc((size_t)seq_len * sizeof(float));
            if (imp2) {
                for (int t = 0; t < seq_len; t++) imp2[t] = (t < bs) ? 0.0f : 1.0f;
            } else {
                ran_ok = 0;
            }
            float *kf = NULL, *vf = NULL, *kf2 = NULL, *vf2 = NULL;
            int8_t *k8 = NULL, *v8 = NULL;
            float *ksc = NULL, *vsc = NULL;
            size_t nrow = (size_t)bs * (size_t)kv_dim;
            if (!imp2) { ran_ok = 0; }
            if (ran_ok) {
                kf  = (float *)calloc(nrow, sizeof(float));
                vf  = (float *)calloc(nrow, sizeof(float));
                kf2 = (float *)calloc(nrow, sizeof(float));
                vf2 = (float *)calloc(nrow, sizeof(float));
                k8  = (int8_t *)calloc(nrow, 1);
                v8  = (int8_t *)calloc(nrow, 1);
                ksc = (float *)calloc((size_t)bs * nkv, sizeof(float));
                vsc = (float *)calloc((size_t)bs * nkv, sizeof(float));
                if (!kf || !vf || !kf2 || !vf2 || !k8 || !v8 || !ksc || !vsc) ran_ok = 0;
            }
            if (!ran_ok) {
                printf("  [FAIL] P3 restore OOM\n");
                fail++;
            } else if (l3_state_init(&s1, path, nl, nkv, hd, bs, max_blocks) != 0 ||
                       l3_evict_layer(&s1, 0, k1b, v1b, kv_dim, seq_len, imp2,
                                      0.75f, bs) <= 0 ||
                       l3_load_to_mem(&s1) != 0 ||
                       !s1.blocks[0].on_disk) {
                printf("  [FAIL] P3 restore fixture (block 0 未被驱逐)\n");
                fail++;
                l3_state_free(&s1);
            } else {
                /* q8 scale 按 kv_quantize_per_head 口径取该 (token, head) 行 amax */
                for (int t = 0; t < bs; t++)
                    for (int h = 0; h < nkv; h++) {
                        float mk = 0.0f, mv = 0.0f;
                        for (int i = 0; i < hd; i++) {
                            float ak = fabsf(k1b[0][(size_t)t * kv_dim + (size_t)h * hd + i]);
                            float av = fabsf(v1b[0][(size_t)t * kv_dim + (size_t)h * hd + i]);
                            if (ak > mk) mk = ak;
                            if (av > mv) mv = av;
                        }
                        ksc[t * nkv + h] = mk;
                        vsc[t * nkv + h] = mv;
                    }
                float *kfp[1] = { kf };  float *vfp[1] = { vf };
                int8_t *k8p[1] = { k8 }; int8_t *v8p[1] = { v8 };
                float *kfp2[1] = { kf2 }; float *vfp2[1] = { vf2 };
                int8_t *k8p2[1] = { k8 }; int8_t *v8p2[1] = { v8 };
                if (l3_fill_block_from_disk(&s1, &s1.blocks[0], 0,
                                            kfp, vfp, k8p, v8p, ksc, vsc,
                                            kv_dim, bs, hd) != 0 ||
                    l3_fill_block_from_disk(&s1, &s1.blocks[0], 0,
                                            kfp2, vfp2, k8p2, v8p2, ksc, vsc,
                                            kv_dim, bs, hd) != 0) {
                    printf("  [FAIL] P3 restore call\n");
                    fail++;
                } else {
                    /* ① 误差界：Q4 半格(max/7/2) + q8 半步(max/127/2) */
                    for (int t = 0; t < bs; t++)
                        for (int h = 0; h < nkv; h++) {
                            float mx = 0.0f;
                            for (int i = 0; i < hd; i++) {
                                float a = fabsf(k1b[0][(size_t)t * kv_dim + (size_t)h * hd + i]);
                                if (a > mx) mx = a;
                            }
                            float bound = mx * (0.5f / 7.0f + 0.5f / L3_Q8_INV_SCALE) * 1.01f
                                        + 1e-5f;
                            for (int i = 0; i < hd; i++) {
                                size_t off = (size_t)t * kv_dim + (size_t)h * hd + i;
                                if (fabsf(kf[off] - k1b[0][off]) > bound) bound_ok = 0;
                                if (fabsf(vf[off] - v1b[0][off]) > bound) bound_ok = 0;
                            }
                        }
                    /* ② 确定性：两次填充逐位一致 */
                    det_ok = (memcmp(kf, kf2, nrow * sizeof(float)) == 0 &&
                              memcmp(vf, vf2, nrow * sizeof(float)) == 0);
                    /* ③ f32 == (float)q8 * (scale/127)（同一表达式，位级） */
                    for (int t = 0; t < bs && bit_ok; t++)
                        for (int h = 0; h < nkv && bit_ok; h++) {
                            float sk = ksc[t * nkv + h] * (1.0f / L3_Q8_INV_SCALE);
                            float sv = vsc[t * nkv + h] * (1.0f / L3_Q8_INV_SCALE);
                            for (int i = 0; i < hd; i++) {
                                size_t off = (size_t)t * kv_dim + (size_t)h * hd + i;
                                float ek = (float)k8[off] * sk;
                                float ev = (float)v8[off] * sv;
                                if (memcmp(&ek, &kf[off], sizeof(float)) != 0 ||
                                    memcmp(&ev, &vf[off], sizeof(float)) != 0) { bit_ok = 0; break; }
                            }
                        }
                    printf("  [%s] P3 restore error bound (Q4 half-cell + q8 half-step)\n",
                           bound_ok ? "PASS" : "FAIL");
                    if (!bound_ok) fail++;
                    printf("  [%s] P3 restore deterministic (bitwise)\n", det_ok ? "PASS" : "FAIL");
                    if (!det_ok) fail++;
                    printf("  [%s] P3 restore f32 == q8*(scale/127) (bitwise)\n", bit_ok ? "PASS" : "FAIL");
                    if (!bit_ok) fail++;
                }
                l3_state_free(&s1);
            }
            free(imp2); free(kf); free(vf); free(kf2); free(vf2);
            free(k8); free(v8); free(ksc); free(vsc);
        }
        /* ---- (12) P3 增量驱逐：① 载荷有效则不重打包（消 R1）；
         * ② stale 才重打包，且**原地覆盖**（wcursor 不前进 → 文件不膨胀）；
         * ③ 重打包后的载荷确实是新内容（读回比对）。 */
        {
            float *imp3 = (float *)malloc((size_t)seq_len * sizeof(float));
            STL3State S;
            memset(&S, 0, sizeof(S));
            if (!imp3) {
                printf("  [FAIL] P3 incremental OOM\n");
                fail++;
            } else {
                for (int t = 0; t < seq_len; t++) imp3[t] = (t < bs) ? 0.0f : 1.0f;
                if (l3_state_init(&S, path, nl, nkv, hd, bs, max_blocks) != 0) {
                    printf("  [FAIL] P3 incremental init\n");
                    fail++;
                } else {
                    int e1 = l3_evict_layer(&S, 0, k1b, v1b, kv_dim, seq_len,
                                            imp3, 0.75f, bs);
                    int ev1 = S.evicted;
                    size_t cur1 = S.wcursor;
                    /* 第二次驱逐：block 0 的内容没变（模拟"重建回 RAM 后 RAM
                     * 指针又非空"），载荷仍有效 → 不该有任何新打包。 */
                    int e2 = l3_evict_layer(&S, 0, k1b, v1b, kv_dim, seq_len,
                                            imp3, 0.75f, bs);
                    int skip_ok = (e1 > 0 && e2 == 0 && S.evicted == ev1 &&
                                   S.wcursor == cur1 && S.blocks[0].evict == 1);
                    printf("  [%s] P3 incremental: valid payload not re-packed "
                           "(e1=%d e2=%d)\n", skip_ok ? "PASS" : "FAIL", e1, e2);
                    if (!skip_ok) fail++;

                    /* 改写 block 0 的内容 → stale → 必须重打包，且原地覆盖 */
                    for (int i = 0; i < bs * kv_dim; i++) k1b[0][i] += 0.5f;
                    S.blocks[0].stale = 1;
                    int e3 = l3_evict_layer(&S, 0, k1b, v1b, kv_dim, seq_len,
                                            imp3, 0.75f, bs);
                    int slot_ok = (e3 > 0 && S.evicted == ev1 + e3 &&
                                   S.wcursor == cur1 && S.blocks[0].stale == 0);
                    printf("  [%s] P3 incremental: stale re-packed in place "
                           "(e3=%d, cursor %s)\n", slot_ok ? "PASS" : "FAIL", e3,
                           S.wcursor == cur1 ? "unchanged" : "MOVED");
                    if (!slot_ok) fail++;

                    /* 读回：载荷必须是改写后的内容（不是旧载荷） */
                    if (l3_load_to_mem(&S) == 0) {
                        int n_payloads = hd / 64;
                        size_t nb = (size_t)bs * (size_t)n_payloads * L3_Q4_PAYLOAD64;
                        uint8_t *dst = (uint8_t *)malloc(nb);
                        float r64[64];
                        int cont_ok = 1;
                        if (!dst) cont_ok = 0;
                        if (cont_ok &&
                            l3_fetch_block_area(&S, &S.blocks[0], 0, 0, dst, hd, bs,
                                                kv_dim) != 0) cont_ok = 0;
                        if (cont_ok) {
                            float mx = 0.0f;
                            for (int i = 0; i < bs * kv_dim; i++) {
                                float a = fabsf(k1b[0][i]);
                                if (a > mx) mx = a;
                            }
                            float bound = mx * (0.5f / 7.0f) * 1.02f + 1e-5f;
                            for (int t = 0; t < bs && cont_ok; t++)
                                for (int g = 0; g < n_payloads && cont_ok; g++) {
                                    l3_q4_dequant64(r64,
                                        dst + ((size_t)t * n_payloads + g)
                                              * L3_Q4_PAYLOAD64);
                                    for (int i = 0; i < 64; i++) {
                                        float src = k1b[0][(size_t)t * kv_dim
                                                          + (size_t)g * 64 + i];
                                        if (fabsf(r64[i] - src) > bound) { cont_ok = 0; break; }
                                    }
                                }
                        }
                        printf("  [%s] P3 incremental: re-packed payload == new content\n",
                               cont_ok ? "PASS" : "FAIL");
                        if (!cont_ok) fail++;
                        free(dst);
                    } else {
                        printf("  [FAIL] P3 incremental: load_to_mem\n");
                        fail++;
                    }
                    l3_state_free(&S);
                }
                free(imp3);
            }
        }
        remove(path);
        free(k1); free(v1); free(imp); free(k2); free(v2);
        free(k1b); free(v1b); free(k2b); free(v2b);
    }

    printf("=== L3 self-test %s (%d failures) ===\n", fail == 0 ? "PASSED" : "FAILED", fail);
    return fail;
}

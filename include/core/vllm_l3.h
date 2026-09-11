/**
 * vllm_l3.h - Phase 2: cold-block Q4 disk eviction for the KV cache
 *
 * Long-context memory strategy (honest scope, no O(1) claims):
 *   - The prefill pass keeps a per-token attention-importance potential
 *     (vllm_safetensors.h: prefill_importance).
 *   - After prefill, l3_evict_layer() ranks blocks by that potential, keeps
 *     the hot (1-ratio) blocks plus the most recent block in RAM, packs the
 *     cold blocks into the axiomDLL-compatible Q4 payload format and writes
 *     them to a fixed-offset disk file (kv_l3.bin).
 *   - Decode re-reads evicted blocks straight from the Q4 payload (phase 2b);
 *     the compressed dot kernel (l3_q4_dot64) computes Q·K without
 *     dequantizing to memory, i.e. ~6.4x fewer bytes moved than float.
 *
 * Q4 payload format (64 elements -> 40 bytes), byte-compatible with the
 * axiomDLL q4_dot kernels:
 *   [0..3]    scale_a  (f32)   block A = src[0..31]
 *   [4..19]   16B packed A     nibble = q + 8; even idx = low nibble
 *   [20..23]  scale_b  (f32)   block B = src[32..63]
 *   [24..39]  16B packed B
 * Per 32-element block: scale = amax/7 so max|v| maps to +/-7 (max
 * resolution, saturating, per axiom_arith_fixedpoint_quantize_001).
 */

#ifndef VLLM_L3_H
#define VLLM_L3_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define L3_Q4_PAYLOAD64  40   /* bytes per 64-element Q4 payload */
#define L3_HEADER_MAGIC  0x564C4C4D  /* "VLLM" */
#define L3_HEADER_SIZE   64   /* fixed file header bytes (magic/ver/dims) */

/* ---- Q4 kernels ---- */

/* Pack 64 floats into a 40-byte payload (block-local scale, saturating). */
void l3_q4_pack64(uint8_t *payload, const float *src);

/* Dequantize a 40-byte payload back to 64 floats (round-trip tests). */
void l3_q4_dequant64(float *dst, const uint8_t *payload);

/* Q·K dot: 40-byte payload x 64-element activation vector (AVX2, scalar
 * fallback). Deterministic for identical inputs (gumbel red line). */
float l3_q4_dot64(const uint8_t *payload, const float *act);

/* V accumulation: dequant payload, scale by wgt, add into acc[64]. */
void l3_q4_vacc64(float *acc, const uint8_t *payload, float wgt);

/* ---- L3 disk layout ----
 * file = header(64B) + one chunk per EVICTED (layer, block), packed
 * contiguously in eviction order (wcursor, see STL3State):
 *   K area [nkv][bs][hd/64 payloads] + V area [nkv][bs][hd/64 payloads]
 * chunk bytes = 2 * nkv * bs * (hd/64) * L3_Q4_PAYLOAD64.
 * Decode locates a block through blocks[i].disk_off (never by index math), so
 * the packed layout keeps file + RAM mirror sized to the evicted payload.    */

typedef struct {
    uint8_t  on_disk;    /* 1 = disk_off 处存在**有效**载荷（decode 可直接读）。
                          * P3 增量后它跨轮保留，"有载荷"≠"本轮被驱逐"。 */
    uint8_t  stale;      /* 1 = 载荷作废：RAM 里的内容已与之不一致（该块被新一轮
                          * prefill/decode 改写）。此时**绝不能**只释放 RAM ——
                          * 那样 decode 会读到错的 KV；必须先重打包。 */
    uint8_t  evict;      /* 本轮判定要移出 RAM 的标记（每轮由调用方清零，由
                          * l3_evict_layer 置位）。main.c 只按它释放 RAM。
                          * 与 on_disk 的分工：on_disk=载荷有效（跨轮），
                          * evict=本轮移出（瞬态）。没有 evict 就区分不了
                          * "上一轮冷、这一轮热"的块，会把它误释放。 */
    uint64_t disk_off;   /* 载荷在文件中的偏移。块一旦分配即固定，重打包原地
                          * 覆盖 —— 否则每轮 wcursor 前移会让文件/镜像无界增长。 */
} STL3Block;

typedef struct {
    FILE      *fp;       /* open l3 file (rb+); NULL if disabled */
    int        nl;       /* number of layers */
    int        nkv;      /* KV heads */
    int        hd;       /* head dim (multiple of 64) */
    int        bs;       /* block size in positions */
    int        max_blocks; /* capacity = max_seq / bs */
    size_t     block_bytes; /* bytes per (layer, block) K+V areas */
    STL3Block *blocks;   /* [nl * max_blocks] */
    int        evicted;  /* total blocks evicted */
    size_t     wcursor;  /* compact write cursor: next evicted block's disk_off
                          * (= L3_HEADER_SIZE after init). Blocks are packed
                          * contiguously in eviction order instead of at the
                          * fixed (layer*max_blocks+block)*block_bytes offset,
                          * so the mirror / file size track the ACTUAL evicted
                          * payload (~ratio of max_seq), not the full KV window. */
    uint8_t   *mem;      /* RAM mirror of the payload (set by l3_load_to_mem;
                         * decode reads from here instead of the file) */
    size_t     mem_size; /* 本轮可读 extent（0 = 无镜像）；全部边界检查用它，
                         * 因此它可以小于 mem_cap */
    size_t     mem_cap;  /* mem 的容量（>= mem_size）。跨轮复用的依据：
                         * 容量够就不再 free/malloc，直接覆盖写（P1） */
    char       path[256]; /* file path this state is bound to ("" if none);
                          * lets the serve layer detect per-user L3 file
                          * switches and close the old file on user change */
} STL3State;

/* Open/create the l3 file and allocate the block metadata table.
 * Returns 0 on success, -1 on failure (fp left NULL = L3 disabled). */
int l3_state_init(STL3State *s, const char *path, int nl, int nkv, int hd,
                  int bs, int max_blocks);

/* P1 跨轮镜像复用变体：调用方把上一轮的 mirror（由 l3_state_rewind 摘出）
 * 交回，容量足够时直接复用同一缓冲覆盖写，避免每轮 free+malloc+memset 的
 * 重建开销（实测残余 +51.7MB/轮，见《内存分页优化方案》D1）。
 * 语义与 l3_state_init 等价：mem_size 仍只等于本轮 extent，所有越界读都被
 * 边界检查挡住；本轮 [0, extent) 会被全量覆盖读，故复用不改变可读内容。
 * reuse_mem 传 NULL / reuse_cap 传 0 即等同 l3_state_init。 */
int l3_state_init_reuse(STL3State *s, const char *path, int nl, int nkv, int hd,
                        int bs, int max_blocks,
                        uint8_t *reuse_mem, size_t reuse_cap);

/* 保留 mirror 的重置：关闭文件、释放块元数据表，但把 mirror 摘出来交给
 * 调用方（mirror_out/cap_out 任一可为 NULL，不需要镜像时自行 free 返回值）。
 * 与 l3_state_free 的区别只在 mirror 的归属，用于跨轮复用。 */
void l3_state_rewind(STL3State *s, uint8_t **mirror_out, size_t *cap_out);

/* P3 增量：现有状态能否承接本轮的增量驱逐 —— 同一文件路径、同几何
 * （nl/nkv/hd/bs/max_blocks）、且 fp/blocks/mem 都在。返回 1 = 可保留
 * （调用方**不要** rewind，blocks[] 与载荷 slot 跨轮续用）；0 = 需重建。 */
int l3_state_matches(const STL3State *s, const char *path, int nl, int nkv,
                     int hd, int bs, int max_blocks);

/* Evict one (layer, block): pack K/V from the per-block float caches
 * (k_cache[block] / v_cache[block] are bs*kv_dim rows) and write to the
 * block's fixed offset. */
int l3_evict_block(STL3State *s, int layer, int block,
                   const float *const *k_cache, const float *const *v_cache,
                   int kv_dim, int bs);

/* Static post-prefill eviction for one layer: rank blocks by the prefill
 * importance mass, keep the top (1-ratio) blocks plus the most recent block,
 * evict the rest (deterministic: tie-break by block index).
 * Returns the number of blocks evicted this layer. */
int l3_evict_layer(STL3State *s, int layer,
                   const float *const *k_cache, const float *const *v_cache,
                   int kv_dim, int seq_len, const float *importance,
                   float ratio, int bs);

/* P2 (nof32): q8-source eviction variants — pack the Q4 mirrors from the
 * dequantized q8+scale rows (q*(scale/127)) when no f32 canonical array is
 * allocated. k_q8/v_q8: per-block int8 rows ([block] -> bs*kv_dim);
 * kscale/vscale: layer-per-token per-head scale, token-major [max_seq*nkv]. */
int l3_evict_block_q8(STL3State *s, int layer, int block,
                      const int8_t *const *k_q8, const int8_t *const *v_q8,
                      const float *kscale, const float *vscale,
                      int kv_dim, int bs, int nkv);
int l3_evict_layer_q8(STL3State *s, int layer,
                      const int8_t *const *k_q8, const int8_t *const *v_q8,
                      const float *kscale, const float *vscale,
                      int kv_dim, int seq_len, const float *importance,
                      float ratio, int bs, int nkv);

/* Read one (head, token) K chunk (hd/64 payloads) from an evicted block and
 * compute the compressed-state Q·K dot against q[hd]. Used by the sparse
 * decode path to serve disk-resident blocks without loading them to RAM. */
float l3_read_k_dot(const STL3State *s, const STL3Block *bm, int head, int token,
                    const float *q, int hd, int bs, int kv_dim);

/* Read one (head, token) V chunk from an evicted block and accumulate
 * wgt * dequant(v) into acc[hd]. */
void l3_read_v_acc(const STL3State *s, const STL3Block *bm, int head, int token,
                   float wgt, float *acc, int hd, int bs, int kv_dim);

/* Zero-copy payload pointer: address of one (head, token) Q4 chunk in the RAM
 * mirror (area 0 = K, 1 = V). Returns NULL when the mirror is absent or the
 * range falls outside it (caller then falls back to l3_read_k_dot /
 * l3_read_v_acc). Phase-2c batching: the decode hot path dot/vacc kernels run
 * directly over the returned pointer, skipping the per-token memcpy. */
const uint8_t *l3_payload_at(const STL3State *s, const STL3Block *bm,
                             int head, int token, int area,
                             int hd, int bs, int kv_dim);

/* Bulk-fetch one block's whole K (area=0) or V (area=1) chunk for a single
 * head into dst: bs * (hd/64) * L3_Q4_PAYLOAD64 contiguous bytes in one
 * copy. Sparse decode fetches a selected block once and then runs all its
 * token dots / V accumulations over dst instead of issuing per-token reads.
 * Returns 0 on success, nonzero if the block is not on disk or unreadable. */
int l3_fetch_block_area(const STL3State *s, const STL3Block *bm,
                        int head, int area, uint8_t *dst,
                        int hd, int bs, int kv_dim);

/* q8 量化口径。必须与 vllm_safetensors.c 的 KVQ_INV_SCALE 一致（那里有
 * _Static_assert 兜底）——vllm_l3.c 不依赖 model 层头文件，故在此独立声明。 */
#define L3_Q8_INV_SCALE 127.0f

/* P3: 从已驱逐块（bm->on_disk）的 Q4 载荷反量化填充目标 K/V 行。
 * 只解压、不重算、**不分配**：目标块由调用方按既有口径（kv_block_alloc_raw
 * + KV_GUARD，见 st_qwen_kv_rebuild_freed）分配好，把指针传入即可。
 *   kf/vf : f32 行块数组（NULL = 该世界无 f32 正典，如 nof32 档）
 *   k8/v8 : q8 行块数组（NULL = 不用 q8）
 *   kscale/vscale : 该层 per-token per-head scale（token-major，已偏移到
 *                   block*bs*nkv）；使用 q8 时必需
 *   block : 目标块号（定位 kf[block] / k8[block]）
 * 语义：payload 反量化成 v 后按目标表示落值——
 *   - 仅 f32：f32 行 = v（纯 Q4 近似）；
 *   - 同时有 q8：q8 = clamp((int)floorf(v*(127/scale)+0.5))，
 *     f32 = (float)q8*(scale/127)。量化/反量化表达式与 kv_quantize_per_head /
 *     kv_dequant_roundtrip 逐字一致，故 f32 与 kv_dequant_roundtrip(q8) 位级
 *     一致；相对纯 f32 多出的误差 <= q8 步长一半（约 Q4 误差的 1/18）。
 * 返回 0 成功；非 0 = 参数非法、块不可读或读取失败。 */
int l3_fill_block_from_disk(const STL3State *s, const STL3Block *bm, int block,
                            float **kf, float **vf, int8_t **k8, int8_t **v8,
                            const float *kscale, const float *vscale,
                            int kv_dim, int bs, int hd);

/* Close the file and free the metadata. */
void l3_state_free(STL3State *s);

/* After the eviction writes are complete, mirror the actually-written payload
 * extent into a RAM buffer (s->mem, sized to the highest evicted block, not
 * the full logical capacity). Decode-time reads then come from the mirror, so
 * the decode hot path never touches the FILE. Returns 0 on success. */
int l3_load_to_mem(STL3State *s);

/* Self-test: pack/dequant round-trip (incl. saturation), dot and vacc vs
 * float reference, bitwise determinism, eviction-decision determinism.
 * Returns 0 on success, nonzero on failure. */
int l3_self_test(void);

#ifdef __cplusplus
}
#endif
#endif /* VLLM_L3_H */

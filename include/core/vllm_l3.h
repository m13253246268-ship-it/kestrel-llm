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
 * file = header(64B) + per (layer, block) chunk:
 *   K area [nkv][bs][hd/64 payloads] + V area [nkv][bs][hd/64 payloads]
 * chunk bytes = 2 * nkv * bs * (hd/64) * L3_Q4_PAYLOAD64.                 */

typedef struct {
    uint8_t  on_disk;    /* 1 = evicted to disk */
    uint64_t disk_off;   /* byte offset of the block's K area in the file */
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
    uint8_t   *mem;      /* RAM mirror of the payload (set by l3_load_to_mem;
                         * decode reads from here instead of the file) */
    size_t     mem_size; /* bytes allocated for mem (0 = no mirror) */
    char       path[256]; /* file path this state is bound to ("" if none);
                          * lets the serve layer detect per-user L3 file
                          * switches and close the old file on user change */
} STL3State;

/* Open/create the l3 file and allocate the block metadata table.
 * Returns 0 on success, -1 on failure (fp left NULL = L3 disabled). */
int l3_state_init(STL3State *s, const char *path, int nl, int nkv, int hd,
                  int bs, int max_blocks);

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

/* Close the file and free the metadata. */
void l3_state_free(STL3State *s);

/* After the eviction writes are complete, mirror the whole payload region into
 * a RAM buffer (s->mem). Decode-time reads then come from the mirror, so the
 * decode hot path never touches the FILE. Returns 0 on success. */
int l3_load_to_mem(STL3State *s);

/* Self-test: pack/dequant round-trip (incl. saturation), dot and vacc vs
 * float reference, bitwise determinism, eviction-decision determinism.
 * Returns 0 on success, nonzero on failure. */
int l3_self_test(void);

#ifdef __cplusplus
}
#endif
#endif /* VLLM_L3_H */

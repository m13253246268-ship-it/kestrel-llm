/**
 * vllm_attention.c - Exact Multi-Head Attention and Paged KV-Cache
 *
 * Implements exact scaled dot-product attention over the paged KV cache,
 * along with the block allocator used by the scheduler.
 */

#include "vllm_superpos.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ================================================================
 * softmax_normalize: in-place softmax normalization
 * ================================================================ */
static void softmax_normalize(float *scores, int32_t n) {
    /* Find max for numerical stability */
    float max_val = scores[0];
    for (int32_t i = 1; i < n; i++) {
        if (scores[i] > max_val) max_val = scores[i];
    }

    /* Exp and sum */
    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }

    /* Normalize */
    if (sum > 0.0f) {
        for (int32_t i = 0; i < n; i++) {
            scores[i] /= sum;
        }
    }
}

/* ================================================================
 * exact_attention: exact multi-head scaled dot-product attention
 *
 * For each head, compute Q·K^T scores against every cached token, apply
 * softmax, then form the score-weighted sum of the cached value vectors.
 * ================================================================ */
void exact_attention(AttentionLayer *layer,
                     KVCacheManager *kvcache,
                     const int32_t *block_table,
                     int32_t num_blocks,
                     int32_t seq_pos) {
    int32_t total_ctx = num_blocks * VLLM_KV_BLOCK_SIZE;
    if (total_ctx > seq_pos + 1) total_ctx = seq_pos + 1;
    if (total_ctx <= 0) return;

    float scores[VLLM_MAX_SEQ_LEN];

    for (int h = 0; h < VLLM_NUM_HEADS; h++) {
        /* --- Phase 1: compute Q·K^T scores --- */
        int32_t score_count = 0;
        for (int32_t pos = 0; pos < total_ctx; pos++) {
            int32_t block_id = block_table[pos / VLLM_KV_BLOCK_SIZE];
            int32_t slot = pos % VLLM_KV_BLOCK_SIZE;
            const KVSlot *kv = kvcache_read(kvcache, block_id, slot);
            if (!kv) continue;

            float dot = 0.0f;
            for (int d = 0; d < VLLM_HEAD_DIM; d++) {
                dot += (float)layer->q_heads[h].data[d] * (float)kv->key[d];
            }
            scores[score_count++] = dot / sqrtf((float)VLLM_HEAD_DIM);
        }

        if (score_count == 0) continue;

        softmax_normalize(scores, score_count);

        /* --- Phase 2: score-weighted value sum --- */
        for (int d = 0; d < VLLM_HEAD_DIM; d++) {
            layer->o_heads[h].data[d] = 0;
        }

        int32_t idx = 0;
        for (int32_t pos = 0; pos < total_ctx; pos++) {
            int32_t block_id = block_table[pos / VLLM_KV_BLOCK_SIZE];
            int32_t slot = pos % VLLM_KV_BLOCK_SIZE;
            const KVSlot *kv = kvcache_read(kvcache, block_id, slot);
            if (!kv) continue;

            float w = scores[idx++];
            for (int d = 0; d < VLLM_HEAD_DIM; d++) {
                layer->o_heads[h].data[d] +=
                    (uint32_t)(w * (float)kv->value[d]);
            }
        }
    }
}

/* ================================================================
 * KV-Cache Operations (PagedAttention semantics)
 * ================================================================ */

void kvcache_init(KVCacheManager *mgr, uint32_t block_size) {
    memset(mgr, 0, sizeof(*mgr));
    mgr->block_size = block_size;

    for (int32_t i = 0; i < VLLM_MAX_BLOCKS; i++) {
        mgr->blocks[i].block_id = i;
        mgr->blocks[i].is_free = true;
        mgr->blocks[i].ref_count = 0;
        mgr->free_list[i] = i;
    }
    mgr->free_count = VLLM_MAX_BLOCKS;
}

int32_t kvcache_alloc_block(KVCacheManager *mgr) {
    if (mgr->free_count <= 0) return -1;
    int32_t id = mgr->free_list[--mgr->free_count];
    mgr->blocks[id].is_free = false;
    mgr->blocks[id].ref_count = 1;
    return id;
}

void kvcache_free_block(KVCacheManager *mgr, int32_t block_id) {
    if (block_id < 0 || block_id >= VLLM_MAX_BLOCKS) return;
    if (mgr->blocks[block_id].is_free) return;
    mgr->blocks[block_id].ref_count--;
    if (mgr->blocks[block_id].ref_count <= 0) {
        mgr->blocks[block_id].is_free = true;
        mgr->free_list[mgr->free_count++] = block_id;
    }
}

void kvcache_write(KVCacheManager *mgr, int32_t block_id,
                    int slot_idx, const KVSlot *kv) {
    if (block_id < 0 || block_id >= VLLM_MAX_BLOCKS) return;
    if (mgr->blocks[block_id].is_free) return;
    if (slot_idx < 0 || slot_idx >= (int)VLLM_KV_BLOCK_SIZE) return;
    memcpy(&mgr->blocks[block_id].slots[slot_idx], kv, sizeof(KVSlot));
}

const KVSlot* kvcache_read(const KVCacheManager *mgr,
                            int32_t block_id, int slot_idx) {
    if (block_id < 0 || block_id >= VLLM_MAX_BLOCKS) return NULL;
    if (mgr->blocks[block_id].is_free) return NULL;
    if (slot_idx < 0 || slot_idx >= (int)VLLM_KV_BLOCK_SIZE) return NULL;
    return &mgr->blocks[block_id].slots[slot_idx];
}

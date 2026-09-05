/**
 * vllm_scheduler.c - Request Scheduler with SHS-Axiom-Driven Selection
 *
 * Uses probabilistic_selection_nc and adaptive_selection_based_on_features
 * axioms for dynamic batch scheduling.
 * 
 * Axiom alignment:
 *   probabilistic_selection_nc (id: probabilistic_selection_nc_38aaa102):
 *     - Random element selection with collapse
 *     - Top-k by weight
 *     - Filter by condition
 *   adaptive_selection_based_on_features (id: adaptive_selection_based_on_features_07597cca):
 *     - Evaluate feature functions on current state
 *     - Apply selection policy to choose sequence
 */

#include "vllm_superpos.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ================================================================
 * Simple PRNG (for probabilistic selection)
 * Maps to: probabilistic_choice_operator (id: probabilistic_choice_operator_ae1e2d74)
 * ================================================================ */
static uint32_t lcg_state = 0xDEADBEEF;

static uint32_t prng_next(void) {
    lcg_state = lcg_state * 1103515245 + 12345;
    return lcg_state;
}

static float prng_float(void) {
    return (float)(prng_next() & 0x7FFFFFFF) / 2147483648.0f;
}

/* ================================================================
 * scheduler_select_batch:
 * 
 * probabilistic_selection_nc:
 *   1. Filter by priority (top-k by weight)
 *   2. Probabilistic collapse to running set
 *   3. Maintain superposition of pending requests
 * ================================================================ */
void scheduler_select_batch(RequestScheduler *sched) {
    sched->running_mask = 0;

    /* Collect waiting requests */
    int32_t waiting[VLLM_MAX_BATCH_SIZE];
    int32_t num_waiting = 0;
    for (int32_t i = 0; i < sched->num_requests; i++) {
        if (sched->requests[i].status == REQ_WAITING) {
            waiting[num_waiting++] = i;
        }
    }

    if (num_waiting == 0) {
        /* No waiting requests: keep running ones */
        for (int32_t i = 0; i < sched->num_requests; i++) {
            if (sched->requests[i].status == REQ_RUNNING) {
                sched->running_mask |= (1 << i);
            }
        }
        return;
    }

    /* probabilistic_selection_nc: Top-k by priority (non-classical selection) */
    /* Sort waiting requests by priority descending */
    for (int32_t i = 0; i < num_waiting - 1; i++) {
        for (int32_t j = i + 1; j < num_waiting; j++) {
            if (sched->requests[waiting[i]].priority <
                sched->requests[waiting[j]].priority) {
                int32_t tmp = waiting[i];
                waiting[i] = waiting[j];
                waiting[j] = tmp;
            }
        }
    }

    /* Select top-k where k is bounded by max_batch_size */
    int32_t available_slots = sched->max_batch_size;
    for (int32_t i = 0; i < sched->num_requests; i++) {
        if (sched->requests[i].status == REQ_RUNNING) {
            available_slots--;
        }
    }

    /* Probabilistic selection: with some randomness for fairness
     * This maps to the non-deterministic aspect of probabilistic_selection_nc */
    int32_t to_select = num_waiting;
    if (to_select > available_slots) to_select = available_slots;

    for (int32_t i = 0; i < to_select; i++) {
        /* Probabilistic collapse: with probability p, select this request
         * Axiom: probabilistic_choice_operator with discrete distribution */
        float prob = 0.9f - (0.3f * (float)i / (float)num_waiting);
        if (prng_float() < prob || i < (available_slots / 2)) {
            int32_t idx = waiting[i];
            sched->requests[idx].status = REQ_RUNNING;
            sched->running_mask |= (1 << idx);
        }
    }
}

/* ================================================================
 * scheduler_adaptive_route:
 * 
 * adaptive_selection_based_on_features:
 *   1. Evaluate feature functions on current state
 *   2. Apply selection_policy to route requests
 *   3. Preempt low-priority requests if needed
 * ================================================================ */
void scheduler_adaptive_route(RequestScheduler *sched) {
    /* Count running requests */
    int32_t running_count = 0;
    for (int32_t i = 0; i < sched->num_requests; i++) {
        if (sched->requests[i].status == REQ_RUNNING) {
            running_count++;
        }
    }

    /* Feature 1: Batch utilization ratio */
    float utilization = (float)running_count / (float)sched->max_batch_size;

    /* Feature 2: Average sequence completion */
    float avg_progress = 0.0f;
    int32_t active_count = 0;
    for (int32_t i = 0; i < sched->num_requests; i++) {
        if (sched->requests[i].status == REQ_RUNNING ||
            sched->requests[i].status == REQ_WAITING) {
            avg_progress += (float)sched->requests[i].current_pos /
                            (float)sched->requests[i].seq_len;
            active_count++;
        }
    }
    if (active_count > 0) avg_progress /= (float)active_count;

    /* Adaptive policy: preempt if overloaded */
    if (utilization > 0.9f && running_count > sched->max_batch_size / 2) {
        /* Find lowest priority running request to preempt
         * Maps to: probabilistic_selection_nc with "filter by condition" */
        float min_priority = 1e9f;
        int32_t preempt_idx = -1;
        for (int32_t i = 0; i < sched->num_requests; i++) {
            if (sched->requests[i].status == REQ_RUNNING &&
                sched->requests[i].priority < min_priority) {
                min_priority = sched->requests[i].priority;
                preempt_idx = i;
            }
        }
        if (preempt_idx >= 0 && sched->requests[preempt_idx].priority < 0.5f) {
            sched->requests[preempt_idx].status = REQ_PREEMPTED;
            sched->running_mask &= ~(1 << preempt_idx);
        }
    }

    /* Non-classical routing: maintain superposition of possible schedules
     * Maps to: fuzzy_equivalence_operator + conditional_operator */
    if (avg_progress > 0.8f && utilization < 0.5f) {
        /* System is nearly done: aggressively admit new requests */
        scheduler_select_batch(sched);
    }
}

/* ================================================================
 * Full Inference Pipeline
 * ================================================================ */

AttentionLayer *layers = NULL; /* heap-allocated layers */
KVCacheManager g_kvcache;
RequestScheduler g_scheduler;
static ModelConfig g_config;
static bool g_initialized = false;

void vllm_kestrel_init(const ModelConfig *config) {
    memcpy(&g_config, config, sizeof(ModelConfig));
    layers = calloc(VLLM_NUM_LAYERS, sizeof(AttentionLayer));
    kvcache_init(&g_kvcache, config->kv_block_size);
    memset(&g_scheduler, 0, sizeof(g_scheduler));
    g_scheduler.max_batch_size = VLLM_MAX_BATCH_SIZE;
    g_initialized = true;

    printf("[vLLM-Kestrel] Initialized with %u layers, %u heads, dim=%u\n",
           config->num_layers, config->num_heads, config->head_dim);
    printf("[vLLM-Kestrel] KV-Cache: %u blocks × %u slots\n",
           VLLM_MAX_BLOCKS, config->kv_block_size);
}

void vllm_kestrel_forward(KVCacheManager *kvcache, RequestScheduler *sched) {
    if (!g_initialized) return;

    /* Select the active batch using SHS scheduler */
    scheduler_select_batch(sched);
    scheduler_adaptive_route(sched);

    /* Process each running request */
    for (int32_t r = 0; r < sched->num_requests; r++) {
        if (!(sched->running_mask & (1 << r))) continue;

        InferenceRequest *req = &sched->requests[r];
        if (req->status != REQ_RUNNING) continue;

        /* Process through all layers */
        for (uint32_t l = 0; l < g_config.num_layers; l++) {
            exact_attention(&layers[l], kvcache,
                            req->block_table, req->num_blocks,
                            (int32_t)req->current_pos);
        }
    }
}

void vllm_kestrel_generate(KVCacheManager *kvcache, RequestScheduler *sched,
                        uint32_t *output_tokens, int32_t *num_outputs) {
    if (!g_initialized) return;

    vllm_kestrel_forward(kvcache, sched);

    *num_outputs = 0;

    for (int32_t r = 0; r < sched->num_requests; r++) {
        if (!(sched->running_mask & (1 << r))) continue;

        InferenceRequest *req = &sched->requests[r];
        if (req->status != REQ_RUNNING) continue;

        /* Extract output from last layer's attention output
         * (simplified: use SHS collapse to extract token) */
        SHS_SetState logits;
        setstate_init(&logits, SHS_MODULUS_N);

        /* Collect outputs from each head as superposition states */
        for (int h = 0; h < VLLM_NUM_HEADS; h++) {
            uint32_t seed = 0;
            for (int d = 0; d < VLLM_HEAD_DIM; d++) {
                seed = seed * 1103515245 + layers[g_config.num_layers - 1]
                       .o_heads[h].data[d];
            }
            setstate_add_element(&logits, seed % SHS_MODULUS_N,
                                 (int64_t)h, 1.0f / (float)VLLM_NUM_HEADS);
        }

        /* Apply interference compression */
        shs_selective_interference_compress(&logits);

        /* Collapse to final token */
        uint32_t token;
        if (shs_collapse_priority(&logits, &token)) {
            output_tokens[*num_outputs] = token % g_config.vocab_size;
        } else if (logits.count > 0) {
            output_tokens[*num_outputs] =
                logits.states[0].value % g_config.vocab_size;
        } else {
            output_tokens[*num_outputs] = 0;
        }
        (*num_outputs)++;

        /* Update request state */
        if (req->current_pos < req->seq_len) {
            req->token_ids[req->current_pos] = output_tokens[*num_outputs - 1];
            req->current_pos++;

            /* Allocate KV-cache block if needed */
            if (req->current_pos % VLLM_KV_BLOCK_SIZE == 0 &&
                req->num_blocks < (int32_t)(VLLM_MAX_SEQ_LEN / VLLM_KV_BLOCK_SIZE)) {
                int32_t block = kvcache_alloc_block(kvcache);
                if (block >= 0) {
                    req->block_table[req->num_blocks++] = block;
                }
            }
        }

        if (req->current_pos >= req->seq_len) {
            req->status = REQ_FINISHED;
        }

        setstate_free(&logits);
    }
}

void vllm_kestrel_cleanup(void) {
    g_initialized = false;
    free(layers);
    layers = NULL;
    printf("[vLLM-Kestrel] Cleanup complete.\n");
}

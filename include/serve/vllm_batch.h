/**
 * vllm_batch.h - server-side continuous batching (batch decode)
 *
 * Axiom mapping (axiom_registry.json blas_acceleration layer):
 *   block_matrix_assoc_natural (shared-input fusion / weight reuse):
 *   N concurrent decode requests are stepped in lockstep via
 *   st_qwen_model_forward_batch, so the layer weights are read ONCE per step
 *   and reused across all B tokens -> decode per-token weight DRAM ÷B
 *   (design doc §5.4 / §9.1 step 1).
 *
 * Design: one engine thread owns every st_qwen_model_forward_batch call;
 * HTTP worker threads register per-request inference states (weights shared,
 * KV private) and wait on a step-completion counter. Prefill (text or
 * multimodal) runs on the worker under the engine mutex, serialized against
 * batch steps. No condition-variable dependency: waiters poll with a 1-2ms
 * backoff (decode steps are tens of ms, so polling adds negligible latency).
 */
#ifndef VLLM_BATCH_H
#define VLLM_BATCH_H

#include "vllm_server.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One batch-decode request. Owned by the caller; the scheduler only reads
 * it while vllm_batch_run is active. */
typedef struct {
    int  is_mm;                 /* 1 = multimodal prefill (pt/vis/grids/ds) */
    /* text prompt (is_mm == 0) */
    const char *prompt;
    /* multimodal (is_mm == 1) */
    const int *pt; int pn;
    const float *vis_all; int vis_total;
    const int *grids; int n_regions;
    const float *const *ds_features; int n_ds;
    /* generation */
    double temperature, top_p, min_p;
    int top_k;                  /* top-k 截断（0 = 关闭）；Qwen3 thinking 档建议 20 */
    int thinking;               /* enable_thinking（0/1）；prompt 由调用方按此渲染，
                                 * 仅用于出证时把实际档位绑进摘要 */
    int max_tokens, stream;
    VHttpConn *conn;            /* streaming output (stream=1); the caller
                                 * must have begun the SSE stream and written
                                 * the role preamble; vllm_batch_run writes
                                 * content chunks + finish + metrics + done */
    VLLMMetrics *met;           /* caller-zeroed metrics output */
    /* non-stream output (stream=0): heap buffer filled by vllm_batch_run,
     * caller must free(). NULL pointers = skipped. 流式(stream=1)且非 NULL
     * 时同样累积全量输出文本（供末尾 attest 出证事件使用）。 */
    char **out_text;
    int   *out_len;
    /* 方案 2：原始请求体 SM3 的 64-hex（vllm_attest 出证用；NULL=不出证）。 */
    const char *body_sha;
    /* result */
    int rc;                     /* 0 ok; -1 engine/prefill failure; -2 prompt
                                 * exceeds context; -3 no batch slot (timeout) */
} VBatchReq;

/* Opaque scheduler. */
typedef struct VBatchSched VBatchSched;

/* Create the scheduler. batch_max is clamped to [2, 32]. The weights struct
 * is SHARED (read-only) across the per-request inference states. */
VBatchSched *vllm_batch_create(const STModelWeights *w, QwenTokenizer *tok,
                               const char *model_id, int batch_max);
void vllm_batch_destroy(VBatchSched *b);

/* Run one request to completion (blocking, up to a slot-wait timeout).
 * Returns req->rc. */
int vllm_batch_run(VBatchSched *b, VBatchReq *req);

/* Serialize the vision-encode / prompt-build step against batch forwards
 * (multimodal path runs st_vision_encode_* before vllm_batch_run). */
void vllm_batch_engine_lock(VBatchSched *b);
void vllm_batch_engine_unlock(VBatchSched *b);

/* Current number of requests in the decode batch. */
int vllm_batch_active(const VBatchSched *b);

#ifdef __cplusplus
}
#endif

#endif /* VLLM_BATCH_H */

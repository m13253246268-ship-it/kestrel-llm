/*
 * vllm_tp.h — self-contained thread pool replacing the OpenMP runtime.
 *
 * Design notes (axiom-guided: independent work parallel, dependent work
 * sequential; persistent workers eliminate per-region fork/join):
 *
 *  - Persistent worker threads are created once (lazy, on first parallel
 *    call) and reuse a spin-then-event wait between jobs, so small parallel
 *    regions pay ~µs of dispatch instead of OpenMP's thread-pool wake cost.
 *  - The CALLING thread participates: it executes its own chunk and joins the
 *    workers with a wait, so the common case adds no idle-thread wake latency.
 *  - Static contiguous chunking matches the OpenMP "schedule(static)" split
 *    the codebase already relies on (uniform per-row cost).
 *  - Workers bind to physical cores at creation (NUMA/affinity axioms:
 *    skip SMT siblings on x86, pin to the A76 cluster on RK3588).
 *
 * The API intentionally mirrors the OpenMP surface actually used by the
 * engine, so each call site converts 1:1:
 *   #pragma omp parallel for schedule(static)  ->  vllm_tp_parfor(...)
 *   #pragma omp parallel  (manual tid splitting) -> vllm_tp_parcall(...)
 *   omp_get_thread_num()                        ->  vllm_tp_worker_id()
 *   omp_get_max_threads() / omp_set_num_threads ->  vllm_tp_threads() / vllm_tp_init()
 *   omp_get_wtime()                             ->  vllm_tp_wtime()
 *
 * Thread-safety: vllm_tp_parfor/parcall are NOT re-entrant; nested parallel
 * regions from a worker are not supported (the engine does not use them).
 */
#ifndef VLLM_TP_H
#define VLLM_TP_H

#ifdef __cplusplus
extern "C" {
#endif

/* (Re)configure the pool. nthreads <= 0 keeps the current/default count.
 * Default: one worker per physical core (SMT siblings skipped on x86;
 * 4 A76 cores on RK3588, matching the previous OpenMP default). */
int vllm_tp_init(int nthreads);

/* Worker count including the caller (>= 1). */
int vllm_tp_threads(void);

/* Parallel for over [start, end): static contiguous chunks, caller takes the
 * last chunk. Falls back to a serial loop on the caller when the pool has
 * one thread or the range is trivially small. */
void vllm_tp_parfor(int start, int end,
                    void (*body)(void *ctx, int idx), void *ctx);

/* Run fn(ctx) on every worker AND the caller (manual per-thread splitting). */
void vllm_tp_parcall(void (*fn)(void *ctx), void *ctx);

/* Pool id of the calling thread: workers 0..nworkers-1, caller = nthreads-1
 * inside a parcall/parfor, -1 outside. */
int vllm_tp_worker_id(void);

/* Monotonic seconds (QueryPerformanceCounter / CLOCK_MONOTONIC). */
double vllm_tp_wtime(void);

/* Shut the pool down (join workers). Idempotent. */
void vllm_tp_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* VLLM_TP_H */

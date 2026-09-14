/*
 * vllm_tp.c — self-contained thread pool (replaces the OpenMP runtime).
 *
 * Mechanism: persistent worker threads; the caller publishes a job by
 * writing the descriptor, bumping a generation counter (release barrier) and
 * signalling the workers; workers spin for a bounded time (keeping them hot
 * across the back-to-back parallel regions inside one inference step) and
 * then sleep on an event/condvar between jobs. The caller executes its own
 * chunk and joins the workers. No external runtime DLLs.
 *
 * Thread-count policy replicates the previous OpenMP behaviour exactly:
 *   OMP_NUM_THREADS env wins; otherwise default to 4 threads (the engine's
 *   st_default_threads() choice for the RK3588 A76 cluster and the fallback
 *   used on x86). VLLM_THREADS is accepted as an alias.
 *
 * Affinity: each worker is pinned to its OWN logical processor
 * (slot % n_logical — never double-pinned, which collapsed throughput on
 * SMT boxes), and the caller is pinned to the last logical CPU during a
 * region so the spinning workers cannot starve it. RK3588: A76 cluster
 * (cores 4-7). The spin budget is tunable via VLLM_TP_SPIN (default 20000
 * pause iterations, ~0.4 ms): short enough that parked workers release the
 * job_gen cache line, long enough that back-to-back regions inside one
 * inference step never pay a wake.
 */
#include "vllm_tp.h"
#include "vllm_platform.h"   /* st_num_cpus() / st_bind_cpu()（x86 感知） */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#define VLLM_TP_MAX 64
/* Spin budget: workers spin this many pause iterations before parking on the
 * event. 20000 pauses ~= 0.4 ms on Zen5 — short enough that parked workers
 * leave the job_gen cache line (long spins keep 15 sharers on the line and
 * slow every caller write to it), long enough that the back-to-back parallel
 * regions inside one inference step are picked up while still spinning.
 * Tunable via VLLM_TP_SPIN. */
#define VLLM_TP_SPIN_ITERS_DEFAULT 200000
#define VLLM_TP_JOIN_SPIN 2000
static long g_spin_iters = VLLM_TP_SPIN_ITERS_DEFAULT;

/* join wait: after a short spin the caller yields (Sleep(0)) instead of
 * spinning for the whole join. Measured on Zen5: a caller that keeps spinning
 * while workers wake delays them up to a full scheduler quantum (~13 ms);
 * yielding gives a ~30 us worker wake instead. */
#define VLLM_TP_JOIN_YIELD_ITERS 256

/* ---- platform shims -------------------------------------------------- */
#if defined(__GNUC__) || defined(__clang__)
#define TP_TLS __thread
#define tp_barrier() __sync_synchronize()
#define tp_atomic_inc(p) __sync_add_and_fetch((p), 1)
#if defined(__aarch64__)
#define tp_pause() __asm__ __volatile__("yield" ::: "memory")
#else
#define tp_pause() __builtin_ia32_pause()
#endif
#else
#error "vllm_tp: unsupported compiler"
#endif
#ifndef tp_pause
#define tp_pause() ((void)0)
#endif

#define tp_yield() sched_yield()
#define tp_sleep_ms(ms) usleep((ms) * 1000)

/* ---- pool state ------------------------------------------------------ */
typedef struct vllm_tp_worker {
    pthread_t th;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    struct vllm_tp *g;
    int slot;    /* 0..nworkers-1 */
    volatile long asleep;   /* 1 while parked on the event (hot-path hint) */
    long last_gen;          /* 最后已执行过的 job_gen（代际不变量防护） */
} vllm_tp_worker;

typedef struct vllm_tp {
    volatile int running;
    volatile long job_gen;
    volatile long done;              /* workers finished current job */
    volatile int  kind;              /* 0 idle, 1 parfor, 2 parcall */
    void (*body)(void *ctx, int idx);
    void (*pfn)(void *ctx);
    void *ctx;
    int chunk_lo[VLLM_TP_MAX];       /* indexed by worker slot; caller=nt-1 */
    int chunk_hi[VLLM_TP_MAX];
    int nthreads;                    /* incl. caller */
    int nworkers;                    /* nthreads - 1 */
    vllm_tp_worker w[VLLM_TP_MAX - 1];
} vllm_tp;

static vllm_tp *g_tp = NULL;
static TP_TLS int t_tid = -1;

/* ---- affinity -------------------------------------------------------- */
static int tp_caller_cpu = -1;   /* dedicated logical CPU for the caller */
static int tp_bind_state = -1;   /* VLLM_TP_BIND=1: enable affinity pinning */

/* Worker/caller pinning is OFF by default (measured faster on the Zen5 box);
 * VLLM_TP_BIND=1 restores it. */
static int tp_bind_disabled(void) {
    if (tp_bind_state < 0) {
        const char *e = getenv("VLLM_TP_BIND");
        tp_bind_state = (e && e[0] == '1') ? 0 : 1;
    }
    return tp_bind_state;
}

static void tp_bind_worker(int slot, int nthreads) {
    (void)nthreads;
    if (tp_bind_disabled()) return;
    long n_cpus = st_num_cpus();
    if (n_cpus <= 0) n_cpus = 8;
    int core = (n_cpus > 4) ? (4 + (slot % 4)) : (slot % n_cpus); /* A76 cluster */
    st_bind_cpu(core);
}

/* The caller must not share a logical CPU with a spinning worker: a pool of
 * 15 workers spinning at 100% starves an unpinned caller for entire quanta
 * (~10-26 ms per region, measured). Give the caller the last logical CPU,
 * which no worker occupies (workers use slots 0..nworkers-1). */
static void tp_bind_caller(void) {
    if (tp_bind_disabled()) return;
    if (tp_caller_cpu < 0) {
        long n = st_num_cpus();
        tp_caller_cpu = (n > 1) ? (int)(n - 1) : 0;
    }
    st_bind_cpu(tp_caller_cpu);
}
static void tp_unbind_caller(void) {
    if (tp_bind_disabled()) return;
#ifdef _WIN32
    return;   /* no-op: st_bind_cpu is per-CPU and unused on Windows */
#else
    cpu_set_t cs;
    CPU_ZERO(&cs);
    for (int i = 0; i < tp_caller_cpu + 1; i++) CPU_SET(i, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
#endif
}

/* ---- worker main ----------------------------------------------------- */
static void *tp_worker_main(void *arg)
{
    vllm_tp_worker *w = (vllm_tp_worker *)arg;
    vllm_tp *g = w->g;
    t_tid = w->slot;
    tp_bind_worker(w->slot, g->nthreads);
    long last = 0;
    for (;;) {
        long spins = 0;
        while (g->job_gen == last) {
            if (++spins > g_spin_iters) {
                /* Park: publish the asleep flag, re-check the generation, then
                 * wait. The caller only signals workers whose asleep flag is
                 * set (hot workers stay spinning and see job_gen directly), so
                 * the common back-to-back case costs zero syscalls.
                 *
                 * The wait is TIMED (not INFINITE): under heavy load a worker
                 * can park on a stale/consumed event or a wake that was
                 * published while it was still spinning; the timeout makes the
                 * worker re-check job_gen and self-heal instead of parking
                 * forever. 50 ms is invisible to the caller (jobs are sub-ms;
                 * parked workers are woken by SetEvent well before the
                 * timeout), it only bounds the worst-case wake latency. */
                w->asleep = 1;
                tp_barrier();
                if (g->job_gen != last) { w->asleep = 0; break; }
                if (g->running) return 0;
                struct timespec ts;
                clock_gettime(CLOCK_REALTIME, &ts);
                ts.tv_nsec += 50 * 1000000L;
                if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
                pthread_mutex_lock(&w->mu);
                while (g->job_gen == last && !g->running)
                    pthread_cond_timedwait(&w->cv, &w->mu, &ts);
                pthread_mutex_unlock(&w->mu);
                w->asleep = 0;
                /* Re-check the condition: a wake can be spurious (e.g. an
                 * auto-reset event left signaled by a job published while this
                 * worker was still spinning). Falling through would re-run the
                 * SAME job with a body/ctx the caller may have already
                 * overwritten by a later job - mismatched execution, crash. */
                continue;
            }
            tp_pause();
        }
        if (g->running) return 0;
        long gen = g->job_gen;
        tp_barrier();                 /* acquire: job descriptor visible */
        int k = g->kind;
        if (gen <= w->last_gen) {
            /* 失醒/自愈边缘防护（M1：down 并行化浮出的确定性坏吸引子排查）：
             * 本 worker 已执行过该代 —— 调用方早已完成 join（done==nw 由首次执行
             * 满足）并离开，现在不得再执行（否则 body 把输出重复累加一次 → 确定
             * 性结果污染）。只对齐代际记账并跳过；done 不再 +1。 */
            fprintf(stderr, "[TP] slot=%d repeat-gen %ld (last=%ld) SKIPPED\n",
                    w->slot, gen, w->last_gen);
            fflush(stderr);
            w->last_gen = gen;
            last = gen;
            continue;
        }
        w->last_gen = gen;            /* 先记账后执行：一次 job 只执行一次 */
        if (k == 1) {
            int lo = g->chunk_lo[w->slot], hi = g->chunk_hi[w->slot];
            for (int i = lo; i < hi; i++) g->body(g->ctx, i);
        } else if (k == 2) {
            g->pfn(g->ctx);
        }
        /* Order matters: publish `last = gen` BEFORE signalling done. If the
         * caller finishes its join (done == nworkers), resets done and
         * publishes the next job between our done++ and last=gen, we would
         * re-enter the loop with a stale `last`, re-run the next job a second
         * time and inflate done — the caller then stops waiting and starts
         * overwriting job fields under our feet (mismatched execution/crash).
         * With last set first, done==nworkers implies every worker already
         * adopted the current generation, so the next publication runs each
         * worker exactly once. */
        tp_barrier();
        last = gen;
        tp_atomic_inc(&g->done);
    }
}

/* ---- wake helper ----------------------------------------------------- */
static void tp_wake_workers(vllm_tp *g) {
    tp_barrier();                       /* make asleep flags stable */
    for (int t = 0; t < g->nworkers; t++) {
        /* Only signal parked workers. A spinning worker sees the bumped
         * job_gen itself, and an unconditional SetEvent would leave its
         * auto-reset event signaled: when the worker later spins out and
         * parks it wakes instantly -> a busy loop that burns a full core and,
         * under heavy load, starves the caller/other workers (deadlock
         * contributor). Parked workers still get their event; a worker that
         * parks just after this loop (asleep=1, not yet in WaitForSingleObject)
         * self-heals via the 50 ms timed wait. */
        if (!g->w[t].asleep) continue;
        pthread_mutex_lock(&g->w[t].mu);
        pthread_cond_signal(&g->w[t].cv);
        pthread_mutex_unlock(&g->w[t].mu);
    }
}

static void tp_join_workers(vllm_tp *g) {
    long nw = g->nworkers;
    long spins = 0, yields = 0, force = 0;
    while (g->done < nw) {
        if (++spins > VLLM_TP_JOIN_SPIN) {
            /* Short spin exhausted: yield so waking workers on our CPU (and
             * on other CPUs) can run. A pure spin here was measured to delay
             * worker wake by up to a full quantum. */
            if (++yields > VLLM_TP_JOIN_YIELD_ITERS) {
                yields = 0;
                tp_sleep_ms(1);
                /* Safety net: if the join drags on (~2 s of 1 ms sleeps) a
                 * worker may have lost its wake under heavy load. Re-signal
                 * all workers; a parked worker re-checks job_gen and either
                 * executes or self-heals via the timed wait. Without this, a
                 * single lost wake deadlocks the caller forever. */
                if (++force > 2000) {
                    force = 0;
                    tp_wake_workers(g);
                }
            } else {
                tp_yield();
            }
            spins = 0;
        }
        tp_pause();
    }
    g->done = 0;
}

/* ---- public API ------------------------------------------------------ */
int vllm_tp_threads(void) {
    if (!g_tp) vllm_tp_init(0);   /* lazy default init (matches OpenMP) */
    return g_tp ? g_tp->nthreads : 1;
}

int vllm_tp_worker_id(void) { return t_tid; }

int vllm_tp_init(int nthreads) {
    if (g_tp && (nthreads <= 0 || nthreads == g_tp->nthreads))
        return g_tp->nthreads;
    if (nthreads <= 0) {
        const char *env = getenv("OMP_NUM_THREADS");
        if (!env || !env[0]) env = getenv("VLLM_THREADS");
        if (env && env[0]) {
            int v = atoi(env);
            if (v > 0) nthreads = v;
        }
        if (nthreads <= 0) {
            long nc = st_num_cpus();
#if ST_ARCH_X86
            /* §9.45：x86 默认线程数 = **逻辑处理器数 − 2**（不再沿用板端的 4 线程上限）。
             * 取 nc 是依据「本内核停顿/端口受限」——IPC≈1.4、`_mm_prefetch` 与真载入式
             * 预取均无效、只吃到内存屋顶（实测 46–55 GB/s）的 8.6% —— 所以 SMT 逻辑核
             * 有实打实收益；再 −2 是给调用者/OS 留余量，见下。
             * 实测（Ryzen 7 9800X3D 8C/16T，同二进制交错，每轮通配清场）：
             * ① 批式/流式（--stream-n 16 --stream-ctx 512 --warmup，prefill 总时 ms）：
             *    t4 15970 / t8 9355 / t12 7189 / t14 6750 / t16 6732 / t24 10142
             *    ⇒ t14 与 t16 无差别（0.3%），t12 起进入平台期；t24 因超订反而退化 48%。
             * ② 服务端（--serve --batch-max 8，8 并发 mt=24，tools/moe_serve_bench.py，
             *    预热后 r2–r4 均值 tok/s）：
             *    t4 12.9 / t8 17.0 / t10 19.0 / t12 20.9 / t14 22.6 / t16 15.8
             *    ⇒ **占满全部逻辑核会断崖**：15 个池线程 + 调用者（提交 batch 的 HTTP worker）
             *    挤满同一批逻辑核，调用者在两个 region 之间被抢占整段调度量子（与上面
             *    「spin 饿死 caller，10–26 ms/region」同机理）；留 2 个核即恢复。
             * 两条路径的共同最优 = nc−2，故此处取单一默认值，不做按入口分支。
             * 各档 `text md5` + TOKIDS **逐位相同**（并行按 idx 静态切分，不改任何浮点步序）。
             * 板端（RK3588 4×A76+4×A55）保持 4：A55 小核会拖慢 GEMM。 */
            nthreads = (nc > 2) ? (int)(nc - 2) : (int)nc;
#else
            nthreads = (nc > 4) ? 4 : (int)nc;   /* RK3588 A76-only default */
#endif
            if (nthreads <= 0) nthreads = 4;
        }
    }
    if (nthreads > VLLM_TP_MAX) nthreads = VLLM_TP_MAX;
    if (nthreads < 1) nthreads = 1;

    /* VLLM_TP_SPIN: worker spin budget before parking (iterations of pause).
     * Tune for the gap between back-to-back parallel regions: a budget that
     * covers the serial work between regions keeps workers hot (no wake cost);
     * a budget too short makes every region pay the ~30-360 us park-wake. */
    {
        const char *env = getenv("VLLM_TP_SPIN");
        if (env && env[0]) {
            long v = atol(env);
            if (v > 0) g_spin_iters = v;
        }
    }

    if (g_tp && nthreads != g_tp->nthreads) vllm_tp_shutdown();

    if (!g_tp) {
        vllm_tp *g = (vllm_tp *)calloc(1, sizeof(vllm_tp));
        if (!g) return 1;
        g_tp = g;
    }
    vllm_tp *g = g_tp;
    g->nthreads = nthreads;
    g->nworkers = nthreads - 1;
    g->running = 0;
    g->job_gen = 0;
    g->done = 0;
    g->kind = 0;
    for (int t = 0; t < g->nworkers; t++) {
        vllm_tp_worker *w = &g->w[t];
        w->g = g;
        w->slot = t;
        pthread_mutex_init(&w->mu, NULL);
        pthread_cond_init(&w->cv, NULL);
        if (pthread_create(&w->th, NULL, tp_worker_main, w) != 0) break;
    }
    return g->nthreads;
}

void vllm_tp_parfor(int start, int end,
                    void (*body)(void *ctx, int idx), void *ctx) {
    vllm_tp *g = g_tp;
    if (!g) { vllm_tp_init(0); g = g_tp; }
    /* t_tid >= 0 => already inside a parallel region (caller's own chunk or a
     * worker): run serial, matching OpenMP's disabled nesting (an inner
     * parallel-for in the old code ran with one thread too). */
    if (!g || g->nthreads <= 1 || end <= start || t_tid >= 0) {
        if (body) for (int i = start; i < end; i++) body(ctx, i);
        return;
    }
    int nt = g->nthreads;
    int n = end - start;
    int base = n / nt, rem = n % nt;
    int acc = 0;
    for (int t = 0; t < nt; t++) {
        int len = base + (t < rem);
        g->chunk_lo[t] = start + acc;
        g->chunk_hi[t] = start + acc + len;
        acc += len;
    }
    g->body = body;
    g->ctx = ctx;
    g->kind = 1;
    tp_barrier();                     /* publish fields before gen bump */
    tp_atomic_inc(&g->job_gen);
    tp_wake_workers(g);
    /* caller takes the last chunk (slot nt-1), pinned to its own CPU so the
     * spinning workers cannot starve it */
    int old = t_tid;
    t_tid = nt - 1;
    tp_bind_caller();
    for (int i = g->chunk_lo[nt - 1]; i < g->chunk_hi[nt - 1]; i++)
        body(ctx, i);
    tp_join_workers(g);
    tp_unbind_caller();
    t_tid = old;
}

void vllm_tp_parcall(void (*fn)(void *ctx), void *ctx) {
    vllm_tp *g = g_tp;
    if (!g) { vllm_tp_init(0); g = g_tp; }
    if (!g || g->nthreads <= 1 || t_tid >= 0) {
        if (fn) fn(ctx);
        return;
    }
    g->pfn = fn;
    g->ctx = ctx;
    g->kind = 2;
    tp_barrier();
    tp_atomic_inc(&g->job_gen);
    tp_wake_workers(g);
    int old = t_tid;
    t_tid = g->nthreads - 1;
    tp_bind_caller();
    fn(ctx);
    tp_join_workers(g);
    tp_unbind_caller();
    t_tid = old;
}

double vllm_tp_wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void vllm_tp_shutdown(void) {
    vllm_tp *g = g_tp;
    if (!g) return;
    g->kind = 0;                          /* no stale job to replay */
    g->body = NULL;
    g->pfn = NULL;
    g->running = 1;
    tp_barrier();
    tp_atomic_inc(&g->job_gen);       /* wake any sleeping worker */
    tp_wake_workers(g);
    for (int t = 0; t < g->nworkers; t++) {
        vllm_tp_worker *w = &g->w[t];
        if (w->th) { pthread_join(w->th, NULL); w->th = 0; }
        pthread_mutex_destroy(&w->mu);
        pthread_cond_destroy(&w->cv);
    }
    g->nworkers = 0;
    g->nthreads = 1;
    free(g);
    g_tp = NULL;
}

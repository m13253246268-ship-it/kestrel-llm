/**
 * vllm_platform.h - 单平台移植层（RK3588 / aarch64-Linux）
 *
 * 本引擎专注边缘设备（RK3588, aarch64 Linux），不再维护 x86/Windows 分支。
 * 提供：
 *   - 编译工具宏（ST_INLINE / ST_THREAD / ST_ALIGN）
 *   - 架构开关（ST_ARCH_ARM64 / ST_ARCH_X86 / ST_HAVE_NEON）
 *   - 64 字节对齐分配
 *   - 高精度单调计时（仅用于上报，不进入计算路径）
 *   - 64 位文件寻址、mmap 张量加载
 *   - CPU 亲和（A76 簇绑定）、栈提升、页保护调试分配
 *
 * 确定性红线：计时仅用于报告；对齐分配 64 字节对齐；NEON 内核在此
 * 架构内必须位级一致（gumbel determinism line）。
 */

#ifndef VLLM_PLATFORM_H
#define VLLM_PLATFORM_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * 1. Compiler utilities (GCC/Clang only)
 * ================================================================ */
#if defined(__GNUC__) || defined(__clang__)
    #define ST_INLINE static inline
    #define ST_THREAD __thread
    #define ST_ALIGN(n) __attribute__((aligned(n)))
#else
    #define ST_INLINE static inline
    #define ST_THREAD
    #define ST_ALIGN(n)
#endif

/* ================================================================
 * 2. Architecture detection
 *
 *    单平台：RK3588 (aarch64, Cortex-A76/A55, NEON 恒可用)。
 *    ST_ARCH_X86 / ST_HAVE_AVX512_VNNI 保留为 0，便于逐文件迁移；
 *    迁移完成后由各调用点直接删除 x86 分支。
 * ================================================================ */
#if defined(__aarch64__) || defined(_M_ARM64)
    #define ST_ARCH_ARM64 1
    #define ST_ARCH_X86   0
    #define ST_HAVE_NEON  1
    #define ST_HAVE_AVX512_VNNI 0
    #include <arm_neon.h>
#else
    #error "vllm_kestrel targets aarch64 (RK3588) only"
#endif

/* 遗留别名：现有源码沿用两个拼写。 */
#ifndef ST_HAVE_NEON
    #define ST_HAVE_NEON 0
#endif

/* Prefetch: aarch64 temporal prefetch. */
#define ST_PREFETCH(p) __builtin_prefetch((p), 0, 3)

/* ================================================================
 * 3. Aligned allocation (64-byte, matches NEON lanes)
 * ================================================================ */
ST_INLINE void *st_aligned_alloc(size_t size, size_t align) {
    void *p = NULL;
    if (align < sizeof(void *)) align = sizeof(void *);
    if (posix_memalign(&p, align, size ? size : 1) != 0) return NULL;
    return p;
}
ST_INLINE void st_aligned_free(void *p) {
    free(p);
}

/* ================================================================
 * 4. High-resolution monotonic timing (reporting only, not compute)
 * ================================================================ */
#include <time.h>
ST_INLINE double st_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
ST_INLINE double st_ticks_to_sec(unsigned long long ticks,
                                 unsigned long long freq) {
    return (double)ticks / (double)freq;
}
ST_INLINE unsigned long long st_tick_freq(void) {
    return 1000000000ULL; /* clock_gettime ns resolution */
}
ST_INLINE unsigned long long st_now_ticks(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ULL +
           (unsigned long long)ts.tv_nsec;
}

/* ================================================================
 * 5. Heap-integrity check (debug triage only; no-op on aarch64)
 * ================================================================ */
ST_INLINE int st_heapchk(void) { return 0; }
#define ST_HEAP_OK 0

/* ================================================================
 * 6. 64-bit file seek / tell
 * ================================================================ */
#include <sys/types.h>
#define ST_FSEEK fseeko
#define ST_FTELL ftello

/* ================================================================
 * 7. Portable mmap (safetensors tensor loading fast-path)
 *    POSIX: open + mmap, whole file read-only.
 * ================================================================ */
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
typedef struct {
    int    fi;
    void  *data;
    size_t len;
    int    fd;
} st_mmap_t;

ST_INLINE void st_mmap_close(st_mmap_t *m) {
    if (m->fi < 0) return; /* never mapped: all fields are in initial state */
    if (m->data && m->len) munmap(m->data, m->len);
    m->data = NULL; m->len = 0;
    if (m->fd >= 0) { close(m->fd); m->fd = -1; }
    m->fi = -1;
}

/* Returns 0 on success; -1 on failure (m left unmapped). */
ST_INLINE int st_mmap_open(st_mmap_t *m, int fi, const char *path) {
    m->fi = -1; m->data = NULL; m->len = 0; m->fd = -1;
    m->fd = open(path, O_RDONLY);
    if (m->fd < 0) return -1;
    struct stat sb;
    if (fstat(m->fd, &sb) != 0 || sb.st_size == 0) { st_mmap_close(m); return -1; }
    m->len = (size_t)sb.st_size;
    m->data = mmap(NULL, m->len, PROT_READ, MAP_PRIVATE, m->fd, 0);
    if (m->data == MAP_FAILED) { m->data = NULL; st_mmap_close(m); return -1; }
    m->fi = fi;
    return 0;
}

/* ================================================================
 * 8. Process affinity (RK3588 big.LITTLE: bind OpenMP workers to CPUs)
 *    Returns 0 on success; non-zero when unavailable (caller tolerates).
 * ================================================================ */
#include <sched.h>
ST_INLINE int st_bind_cpu(int cpu_id) {
    if (cpu_id < 0) return -1;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_id, &set);
    return sched_setaffinity(0, sizeof(set), &set);
}

/* ================================================================
 * 9. Stack size: RK3588/Linux main-thread stack is raised to match the
 *    large local arrays (scores[4096]...). aarch64 must NOT rely on
 *    -Wl,--stack.
 * ================================================================ */
#include <sys/resource.h>
ST_INLINE void st_raise_stack_limit(size_t bytes) {
    struct rlimit rl;
    if (getrlimit(RLIMIT_STACK, &rl) == 0 && rl.rlim_cur < bytes) {
        rl.rlim_cur = bytes;
        setrlimit(RLIMIT_STACK, &rl);
    }
}

/* ================================================================
 * 10. Page-guarded debug allocation (heap-corruption triage)
 *
 *     allocs N bytes plus one PROT_NONE guard page, so an over-write
 *     past the payload AVs at the exact kernel instead of corrupting
 *     the heap free-list. POSIX: mmap. Used by the g_xq_pages debug
 *     mode; callers must pair pg_free. Note: xq_pages_mode() returns
 *     0 (CRT heap) on aarch64, so this debug path is off in production.
 * ================================================================ */
/* munmap needs the total length; keep a small static registry (debug-only).
 * The registry lives at file scope so st_pg_alloc/st_pg_free share it. */
#define ST_PG_REG_MAX 256
static void   *st_pg_reg_p[ST_PG_REG_MAX];
static size_t  st_pg_reg_l[ST_PG_REG_MAX];
static int     st_pg_reg_n = 0;
ST_INLINE void *st_pg_alloc(size_t bytes) {
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    if (page == 0) page = 4096;
    size_t np = (bytes + page - 1) / page;
    size_t total = (np + 1) * page;
    void *p = mmap(NULL, total, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
    if (mprotect((uint8_t *)p + np * page, page, PROT_NONE) != 0) {
        munmap(p, total);
        return NULL;
    }
    if (st_pg_reg_n < ST_PG_REG_MAX) {
        st_pg_reg_p[st_pg_reg_n] = p;
        st_pg_reg_l[st_pg_reg_n] = total;
        st_pg_reg_n++;
    }
    return p;
}
ST_INLINE void st_pg_free(void *p) {
    if (!p) return;
    for (int i = 0; i < st_pg_reg_n; i++) {
        if (st_pg_reg_p[i] == p) {
            munmap(p, st_pg_reg_l[i]);
            st_pg_reg_p[i] = st_pg_reg_p[st_pg_reg_n - 1];
            st_pg_reg_l[i] = st_pg_reg_l[st_pg_reg_n - 1];
            st_pg_reg_n--;
            return;
        }
    }
    /* Not found in the registry: cannot recover the length; leave it
     * (debug path only, g_xq_pages is off in production). */
}

#endif /* VLLM_PLATFORM_H */

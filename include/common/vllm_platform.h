/**
 * vllm_platform.h - 双平台移植层（RK3588 / aarch64-Linux  +  x86-64 / Windows）
 *
 * 主线引擎以边缘设备（RK3588, aarch64 Linux）为一级目标；本头文件在保留
 * ARM-Linux 全部原语义的前提下，增加 x86-64（GCC/MinGW 与 MSVC）分支，使
 * 同一份源码可在两类设备上构建。提供：
 *   - 编译工具宏（ST_INLINE / ST_THREAD / ST_ALIGN）
 *   - 架构开关（ST_ARCH_ARM64 / ST_ARCH_X86 / ST_HAVE_NEON / ST_HAVE_AVX512_VNNI）
 *   - 64 字节对齐分配
 *   - 高精度单调计时（仅用于上报，不进入计算路径）
 *   - 64 位文件寻址、mmap 张量加载（Windows: CreateFileMapping 等效实现）
 *   - CPU 亲和（A76 簇绑定 / x86 感知 no-op）、栈提升、页保护调试分配
 *
 * 确定性红线：计时仅用于报告；对齐分配 64 字节对齐；同架构内内核位级一致
 * （gumbel determinism line）。x86 分支的映射采用写时复制（FILE_MAP_COPY，
 * 等价 POSIX MAP_PRIVATE），保证 VQF-Enc 原位解密路径与 ARM 语义一致。
 */

#ifndef VLLM_PLATFORM_H
#define VLLM_PLATFORM_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * 1. Compiler utilities
 * ================================================================ */
#if defined(_MSC_VER)
    #define ST_INLINE static __forceinline
    #define ST_THREAD __declspec(thread)
    #define ST_ALIGN(n) __declspec(align(n))
#elif defined(__GNUC__) || defined(__clang__)
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
 *    ST_ARCH_ARM64: aarch64（RK3588: Cortex-A76/A55, NEON 恒可用）。
 *    ST_ARCH_X86  : x86-64（MSVC /arch:AVX512 或 GCC -march=...）。
 *    ST_HAVE_NEON / ST_HAVE_AVX512_VNNI: 内核选择开关。
 * ================================================================ */
#if defined(__aarch64__) || defined(_M_ARM64)
    #define ST_ARCH_ARM64 1
    #define ST_ARCH_X86   0
    #define ST_HAVE_NEON  1
    #define ST_HAVE_AVX512_VNNI 0
    #include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64) || \
      defined(_M_IX86) || defined(__i386__)
    #define ST_ARCH_ARM64 0
    #define ST_ARCH_X86   1
    #define ST_HAVE_NEON  0
    #if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
        #define ST_HAVE_AVX512_VNNI 1
    #else
        #define ST_HAVE_AVX512_VNNI 0
    #endif
#else
    #error "vllm_kestrel supports aarch64 (RK3588) or x86-64 only"
#endif

/* 遗留别名：现有源码沿用两个拼写。 */
#ifndef ST_HAVE_NEON
    #define ST_HAVE_NEON 0
#endif

/* Prefetch: x86 用 T0 提示，aarch64 用时间性预取。 */
#if ST_ARCH_X86
    #if defined(_MSC_VER)
        #include <immintrin.h>
        #define ST_PREFETCH(p) _mm_prefetch((const char *)(p), _MM_HINT_T0)
    #else
        #include <xmmintrin.h>
        #define ST_PREFETCH(p) _mm_prefetch((const char *)(p), _MM_HINT_T0)
    #endif
#else
    #define ST_PREFETCH(p) __builtin_prefetch((p), 0, 3)
#endif

/* ================================================================
 * 3. Aligned allocation (64-byte, matches both NEON and AVX lanes)
 * ================================================================ */
#if defined(_WIN32)
    #include <malloc.h>
    ST_INLINE void *st_aligned_alloc(size_t size, size_t align) {
        if (align < sizeof(void *)) align = sizeof(void *);
        return _aligned_malloc(size ? size : 1, align);
    }
    ST_INLINE void st_aligned_free(void *p) {
        if (p) _aligned_free(p);
    }
#else
    ST_INLINE void *st_aligned_alloc(size_t size, size_t align) {
        void *p = NULL;
        if (align < sizeof(void *)) align = sizeof(void *);
        if (posix_memalign(&p, align, size ? size : 1) != 0) return NULL;
        return p;
    }
    ST_INLINE void st_aligned_free(void *p) {
        free(p);
    }
#endif

/* ================================================================
 * 4. High-resolution monotonic timing (reporting only, not compute)
 * ================================================================ */
#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    ST_INLINE double st_now_sec(void) {
        static LARGE_INTEGER freq = { {0} };
        if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
        LARGE_INTEGER c;
        QueryPerformanceCounter(&c);
        return (double)c.QuadPart / (double)freq.QuadPart;
    }
    ST_INLINE double st_ticks_to_sec(unsigned long long ticks,
                                     unsigned long long freq) {
        return (double)ticks / (double)freq;
    }
    ST_INLINE unsigned long long st_tick_freq(void) {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        return (unsigned long long)freq.QuadPart;
    }
    ST_INLINE unsigned long long st_now_ticks(void) {
        LARGE_INTEGER c;
        QueryPerformanceCounter(&c);
        return (unsigned long long)c.QuadPart;
    }
#else
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
#endif

/* ================================================================
 * 5. Heap-integrity check (debug triage only; no-op off MSVC)
 * ================================================================ */
#if defined(_MSC_VER)
    ST_INLINE int st_heapchk(void) {
        return _heapchk();
    }
    #define ST_HEAP_OK _HEAPOK
#else
    ST_INLINE int st_heapchk(void) { return 0; }
    #define ST_HEAP_OK 0
#endif

/* ================================================================
 * 6. 64-bit file seek / tell
 * ================================================================ */
#if defined(_WIN32)
    #include <io.h>
    #define ST_FSEEK _fseeki64
    #define ST_FTELL _ftelli64
#else
    #include <sys/types.h>
    #define ST_FSEEK fseeko
    #define ST_FTELL ftello
#endif

/* ================================================================
 * 7. Portable mmap (safetensors/VQF tensor loading fast-path)
 *
 *    POSIX  : open + mmap (MAP_PRIVATE, 整文件只读映射)。
 *    Windows: CreateFileMappingW + MapViewOfFile(FILE_MAP_COPY)。
 *             FILE_MAP_COPY 与 POSIX MAP_PRIVATE 等价：写时复制、磁盘不变，
 *             因此 VQF-Enc 的原位解密（st_mmap_write_enable 后直写映射）
 *             在双平台语义一致。
 *
 *    结构体双平台共用 .fi/.data/.len/.fd 布局：既有调用点（vqf.c）
 *    直接读 .len、预置 .fd=-1，无需改动；Windows 分支在 open/close 内自行
 *    维护句柄字段。
 * ================================================================ */
#if defined(_WIN32)
typedef struct {
    int    fi;          /* file index in the config, -1 = unmapped */
    void  *data;        /* mapped base */
    size_t len;         /* mapped length (bytes) */
    int    fd;          /* no POSIX fd on Windows; kept -1 for layout compat */
    HANDLE hMap;
    HANDLE hFile;
} st_mmap_t;

ST_INLINE void st_mmap_close(st_mmap_t *m) {
    if (m->fi < 0) return; /* never mapped: all fields are in initial state */
    if (m->data) { UnmapViewOfFile(m->data); m->data = NULL; }
    if (m->hMap) { CloseHandle(m->hMap); m->hMap = NULL; }
    if (m->hFile != INVALID_HANDLE_VALUE) { CloseHandle(m->hFile); m->hFile = INVALID_HANDLE_VALUE; }
    m->len = 0;
    m->fi = -1;
}

/* Returns 0 on success; -1 on failure (m left unmapped). */
ST_INLINE int st_mmap_open(st_mmap_t *m, int fi, const char *utf8_path) {
    m->fi = -1; m->data = NULL; m->len = 0; m->fd = -1;
    m->hMap = NULL; m->hFile = INVALID_HANDLE_VALUE;
    int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8_path, -1, NULL, 0);
    if (wlen <= 0) return -1;
    wchar_t *wpath = (wchar_t *)malloc((size_t)wlen * sizeof(wchar_t));
    if (!wpath) return -1;
    MultiByteToWideChar(CP_UTF8, 0, utf8_path, -1, wpath, wlen);
    m->hFile = CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (m->hFile == INVALID_HANDLE_VALUE) {
        /* MinGW CRT argv 为 ANSI（GBK 区域）：上一步按 UTF-8 解码非 ASCII
         * 路径会 mangled → CreateFileW 失败。把原始字节按 ANSI 再试一次。 */
        free(wpath);
        int alen = MultiByteToWideChar(CP_ACP, 0, utf8_path, -1, NULL, 0);
        if (alen <= 0) return -1;
        wpath = (wchar_t *)malloc((size_t)alen * sizeof(wchar_t));
        if (!wpath) return -1;
        MultiByteToWideChar(CP_ACP, 0, utf8_path, -1, wpath, alen);
        m->hFile = CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    }
    free(wpath);
    if (m->hFile == INVALID_HANDLE_VALUE) return -1;
    m->hMap = CreateFileMappingA(m->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!m->hMap) { st_mmap_close(m); return -1; }
    /* 默认只读共享映射（FILE_MAP_READ）：与 llama.cpp 同构 —— COW 视图
     * (FILE_MAP_COPY) 会把整段文件计入进程私有提交(commit charge)，明文 VQF
     * 只读推理路径导致 ~19GB(30B q4)/~33GB(q8) 的私有内存虚高（Windows 实测
     * PrivateMemorySize64 ≈ 全文件 + 2.3GB，即使页从未写入）。只读映射不写页
     * 即零 COW，私有提交回落到 ~2.3GB。需要写（VQF-Enc 解密）的路径在
     * st_mmap_write_enable 按需重映射。 */
    m->data = MapViewOfFile(m->hMap, FILE_MAP_READ, 0, 0, 0);
    if (!m->data) { st_mmap_close(m); return -1; }
    LARGE_INTEGER sz;
    if (GetFileSizeEx(m->hFile, &sz) == 0) { st_mmap_close(m); return -1; }
    m->len = (size_t)sz.QuadPart;
    if (m->len == 0) { st_mmap_close(m); return -1; }
    m->fi = fi;
    return 0;
}

/* Make the mapping writable (VQF-Enc in-place decrypt). POSIX: mprotect
 * on a MAP_PRIVATE view. Windows 只读映射无写能力 → 重映射为写时复制视图
 * (FILE_MAP_COPY)。调用方约定：本函数须在任何张量指针挂载(alias)之前调用
 * （vqf_load 的 ENC 解密分支），否则重映射换基址会使既有指针失效。 */
ST_INLINE int st_mmap_write_enable(st_mmap_t *m) {
#ifdef _WIN32
    if (!m->data || !m->hMap) return -1;
    void *p = MapViewOfFile(m->hMap, FILE_MAP_COPY, 0, 0, 0);
    if (!p) return -1;
    UnmapViewOfFile(m->data);
    m->data = p;
    return 0;
#else
    (void)m;
    return 0;
#endif
}

/* Release a mapped view that was opened earlier and whose st_mmap_t handles
 * are no longer reachable (VQF weights alias one whole-file mapping kept as a
 * bare data pointer). POSIX: munmap needs the length; Windows UnmapViewOfFile
 * only needs the view base (mapping handles leak for the process lifetime —
 * acceptable: model weights map for the whole serve session). */
ST_INLINE void st_mmap_release_view(void *data, size_t len) {
#if defined(_WIN32)
    (void)len;
    if (data) UnmapViewOfFile(data);
#else
    if (data && len) munmap(data, len);
#endif
}
#else
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

ST_INLINE int st_mmap_write_enable(st_mmap_t *m) {
    if (mprotect(m->data, m->len, PROT_READ | PROT_WRITE) != 0) return -1;
    return 0;
}

/* Release a bare mapped view (see the _WIN32 twin above). */
ST_INLINE void st_mmap_release_view(void *data, size_t len) {
    if (data && len) munmap(data, len);
}
#endif

/* ================================================================
 * 8. Process affinity (RK3588 big.LITTLE: bind OpenMP workers to CPUs;
 *    x86 感知 no-op——调用方容错非零返回值)
 *    Returns 0 on success; non-zero when unavailable (caller tolerates).
 * ================================================================ */
/* Online CPU count (affinity sizing / OpenMP fallback). */
#if defined(_WIN32)
#include <windows.h>
ST_INLINE long st_num_cpus(void) {
    DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    return (n > 0) ? (long)n : 1;
}
#else
#include <unistd.h>
ST_INLINE long st_num_cpus(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? n : 1;
}
#endif

#if defined(_WIN32)
ST_INLINE int st_bind_cpu(int cpu_id) {
    (void)cpu_id;
    return -1; /* Windows affinity handled per-thread elsewhere */
}
#else
#include <sched.h>
ST_INLINE int st_bind_cpu(int cpu_id) {
    if (cpu_id < 0) return -1;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_id, &set);
    return sched_setaffinity(0, sizeof(set), &set);
}
#endif

/* ================================================================
 * 9. Stack size: RK3588/Linux main-thread stack is raised to match the
 *    large local arrays (scores[4096]...). aarch64 must NOT rely on
 *    -Wl,--stack. Windows/MinGW: 链接期已由 --stack 或默认栈保证。
 * ================================================================ */
#if !defined(_WIN32)
#include <sys/resource.h>
ST_INLINE void st_raise_stack_limit(size_t bytes) {
    struct rlimit rl;
    if (getrlimit(RLIMIT_STACK, &rl) == 0 && rl.rlim_cur < bytes) {
        rl.rlim_cur = bytes;
        setrlimit(RLIMIT_STACK, &rl);
    }
}
#else
ST_INLINE void st_raise_stack_limit(size_t bytes) { (void)bytes; }
#endif

/* ================================================================
 * 9b. mkdir / free-RAM helpers（serve 层 503 门控与 KV 目录创建）
 * ================================================================ */
#if defined(_WIN32)
#include <direct.h>   /* _mkdir */
ST_INLINE int st_mkdir(const char *path, int mode) {
    (void)mode;
    return _mkdir(path);
}
ST_INLINE long st_avail_mem_mb(void) {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) return -1;
    return (long)(ms.ullAvailPhys / (1024ull * 1024ull));
}
#else
#include <sys/stat.h>
#include <unistd.h>
ST_INLINE int st_mkdir(const char *path, int mode) {
    return mkdir(path, mode);
}
ST_INLINE long st_avail_mem_mb(void) {
    /* Prefer MemAvailable from /proc/meminfo（reclaimable page cache 计入）;
     * 回退 sysconf（仅自由页，未计入可回收缓存）。 */
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        long long avail_kb = -1;
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "MemAvailable:", 13) == 0) {
                avail_kb = atoll(line + 13);
                break;
            }
        }
        fclose(f);
        if (avail_kb >= 0) return (long)(avail_kb / 1024);
    }
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long psz = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || psz <= 0) return -1;
    return (long)(((double)pages * psz) / (1024.0 * 1024.0));
}
#endif

/* ================================================================
 * 10. Page-guarded debug allocation (heap-corruption triage)
 *
 *     allocs N bytes plus one PROT_NONE / PAGE_NOACCESS guard page, so an
 *     over-write past the payload AVs at the exact kernel instead of
 *     corrupting the heap free-list. Windows: VirtualAlloc；POSIX: mmap。
 *     Used by the g_xq_pages debug mode; callers must pair pg_free.
 *     Note: xq_pages_mode() returns 0 (CRT heap) on aarch64, so this
 *     debug path is off in production.
 * ================================================================ */
#if defined(_WIN32)
ST_INLINE void *st_pg_alloc(size_t bytes) {
    size_t np = (bytes + 4095) / 4096;
    void *p = VirtualAlloc(NULL, (np + 1) * 4096,
                           MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!p) return NULL;
    DWORD old = 0;
    VirtualProtect((uint8_t *)p + np * 4096, 4096, PAGE_NOACCESS, &old);
    return p;
}
ST_INLINE void st_pg_free(void *p) {
    if (p) VirtualFree(p, 0, MEM_RELEASE);
}
#else
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
#endif

#endif /* VLLM_PLATFORM_H */

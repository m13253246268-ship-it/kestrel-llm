/* ================================================================
 * conv_platform.h - vqf_convert 平台移植层（x86_64 Linux / Windows x64）
 *
 * 提供引擎抽取代码所需的同名符号（st_mmap_* / st_fopen / 宏），
 * 独立自足，不依赖引擎 vllm_platform.h（其强制 aarch64）。
 * 与引擎语义一致；Windows 分支提供 CreateFileW/MapViewOfFile 等价。
 * ================================================================ */
#ifndef CONV_PLATFORM_H
#define CONV_PLATFORM_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define ST_PLAT_WIN 1
#include <windows.h>
#else
#define ST_PLAT_WIN 0
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#define ST_INLINE static inline

/* 64-bit seek/tell */
#if ST_PLAT_WIN
#define ST_FSEEK _fseeki64
#define ST_FTELL _ftelli64
#else
#define ST_FSEEK fseeko
#define ST_FTELL ftello
#endif

#define ST_PREFETCH(p) ((void)(p))

/* ---- mmap wrapper（与引擎 st_mmap_t 同名同字段语义） ---- */
typedef struct {
    int    fi;
    void  *data;
    size_t len;
    int    fd;            /* posix fd；win 下为 HANDLE 截断 int（<=32 个并发文件） */
#if ST_PLAT_WIN
    void  *hfile, *hmap;
#endif
} st_mmap_t;

ST_INLINE void st_mmap_close(st_mmap_t *m) {
    if (!m || m->fi < 0) return;
#if ST_PLAT_WIN
    if (m->hmap) { CloseHandle((HANDLE)m->hmap); m->hmap = NULL; }
    if (m->data) { UnmapViewOfFile(m->data); m->data = NULL; }
    if (m->hfile) { CloseHandle((HANDLE)m->hfile); m->hfile = NULL; }
#else
    if (m->data && m->len) munmap(m->data, m->len);
    m->data = NULL; m->len = 0;
    if (m->fd >= 0) { close(m->fd); m->fd = -1; }
#endif
    m->fi = -1;
}

/* 0 ok; -1 fail (m left unmapped). path 必须是 UTF-8。 */
ST_INLINE int st_mmap_open(st_mmap_t *m, int fi, const char *path) {
    memset(m, 0, sizeof(*m));
    m->fi = -1; m->fd = -1;
#if ST_PLAT_WIN
    {
        HANDLE hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hf == INVALID_HANDLE_VALUE) return -1;
        LARGE_INTEGER sz; GetFileSizeEx(hf, &sz);
        if (sz.QuadPart <= 0) { CloseHandle(hf); return -1; }
        HANDLE hm = CreateFileMappingA(hf, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hm) { CloseHandle(hf); return -1; }
        void *p = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
        if (!p) { CloseHandle(hm); CloseHandle(hf); return -1; }
        m->hfile = hf; m->hmap = hm; m->data = p;
        m->len = (size_t)sz.QuadPart; m->fi = fi;
        return 0;
    }
#else
    m->fd = open(path, O_RDONLY);
    if (m->fd < 0) return -1;
    struct stat sb;
    if (fstat(m->fd, &sb) != 0 || sb.st_size == 0) { st_mmap_close(m); return -1; }
    m->len = (size_t)sb.st_size;
    m->data = mmap(NULL, m->len, PROT_READ, MAP_PRIVATE, m->fd, 0);
    if (m->data == MAP_FAILED) { m->data = NULL; st_mmap_close(m); return -1; }
    m->fi = fi;
    return 0;
#endif
}

/* 引擎抽取代码引用 st_fopen（safetensors 内小 header 读取） */
ST_INLINE FILE *st_fopen(const char *path, const char *mode) {
#if ST_PLAT_WIN
    /* 二进制模式统一：避免 CRLF/CTRL-Z 干扰 */
    char mb[8];
    snprintf(mb, sizeof(mb), "%s", mode);
    if (strchr(mb, 'b') == NULL) strncat(mb, "b", sizeof(mb) - strlen(mb) - 1);
    return fopen(path, mb);
#else
    return fopen(path, mode);
#endif
}

ST_INLINE void *st_aligned_alloc(size_t size, size_t align) {
#if ST_PLAT_WIN
    return _aligned_malloc(size ? size : 1, align);
#else
    void *p = NULL;
    if (align < sizeof(void *)) align = sizeof(void *);
    if (posix_memalign(&p, align, size ? size : 1) != 0) return NULL;
    return p;
#endif
}
ST_INLINE void st_aligned_free(void *p) {
#if ST_PLAT_WIN
    _aligned_free(p);
#else
    free(p);
#endif
}

/* 引擎抽取代码 st_access(path, mode)：文件存在性 */
#if ST_PLAT_WIN
#include <io.h>
ST_INLINE int st_access(const char *path, int mode) { return _access(path, mode); }
#else
ST_INLINE int st_access(const char *path, int mode) { return access(path, mode); }
#endif

/* ---- 可写共享映射（--stream：输出文件 ftruncate 后 mmap 直写） ---- */
typedef struct {
    void  *data;      /* 映射基址（起始于文件 0 偏移，含头部/目录区） */
    size_t len;       /* 映射总长 */
    int    fd;
#if ST_PLAT_WIN
    void  *hfile, *hmap;
#endif
} st_wmap_t;

/* 建文件并映射 RW shared（0 ok；-1 fail）。路径 UTF-8。 */
ST_INLINE int st_wmap_create(st_wmap_t *m, const char *path, uint64_t size) {
    memset(m, 0, sizeof(*m));
    m->fd = -1;
#if ST_PLAT_WIN
    HANDLE hf = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0, NULL,
                            CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hf == INVALID_HANDLE_VALUE) return -1;
    {
        LARGE_INTEGER li; li.QuadPart = (LONGLONG)size;
        SetFilePointerEx(hf, li, NULL, FILE_BEGIN);
        if (!SetEndOfFile(hf)) { CloseHandle(hf); return -1; }
    }
    HANDLE hm = CreateFileMappingA(hf, NULL, PAGE_READWRITE, 0, 0, NULL);
    if (!hm) { CloseHandle(hf); return -1; }
    void *p = MapViewOfFile(hm, FILE_MAP_WRITE | FILE_MAP_READ, 0, 0, (SIZE_T)size);
    if (!p) { CloseHandle(hm); CloseHandle(hf); return -1; }
    m->hfile = hf; m->hmap = hm; m->data = p; m->len = (size_t)size;
    return 0;
#else
    m->fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (m->fd < 0) return -1;
    if (ftruncate(m->fd, (off_t)size) != 0) { close(m->fd); m->fd = -1; return -1; }
    void *p = mmap(NULL, (size_t)size, PROT_READ | PROT_WRITE, MAP_SHARED,
                   m->fd, 0);
    if (p == MAP_FAILED) { close(m->fd); m->fd = -1; return -1; }
    m->data = p; m->len = (size_t)size;
    return 0;
#endif
}

/* 将映射内 [off..off+len) 落盘（posix MS_SYNC / win FlushViewOfFile） */
ST_INLINE void st_wmap_flush(st_wmap_t *m, size_t off, size_t len) {
    if (!m || !m->data) return;
#if ST_PLAT_WIN
    FlushViewOfFile((char *)m->data + off, len);
#else
    msync((char *)m->data + off, len, MS_SYNC);
#endif
}

/* 交回 [off..off+len) 驻留页（posix MADV_DONTNEED；win 无等价 → no-op） */
ST_INLINE void st_wmap_discard(st_wmap_t *m, size_t off, size_t len) {
#if ST_PLAT_WIN
    (void)m; (void)off; (void)len;
#else
    if (m && m->data) madvise((char *)m->data + off, len, MADV_DONTNEED);
#endif
}

ST_INLINE void st_wmap_close(st_wmap_t *m) {
    if (!m) return;
#if ST_PLAT_WIN
    if (m->hmap) { CloseHandle((HANDLE)m->hmap); m->hmap = NULL; }
    if (m->data) { UnmapViewOfFile(m->data); m->data = NULL; }
    if (m->hfile) { CloseHandle((HANDLE)m->hfile); m->hfile = NULL; }
#else
    if (m->data && m->len) { msync(m->data, m->len, MS_SYNC); munmap(m->data, m->len); }
    m->data = NULL; m->len = 0;
    if (m->fd >= 0) { close(m->fd); m->fd = -1; }
#endif
    memset(m, 0, sizeof(*m));
}

#endif /* CONV_PLATFORM_H */

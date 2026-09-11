/**
 * vllm_util.h - Utility functions for cross-platform compatibility
 *
 * Linux/aarch64 (RK3588) 的 fopen/access 原生按 UTF-8 字节流处理路径，故
 * st_fopen/st_access 直接映射到 libc。Windows 的 CRT 窄字符接口按 ANSI 代码页
 * 解析路径（MinGW/GCC 与 MSVC 皆然），UTF-8 中文路径会打开失败——因此这里提供
 * UTF-8 → 宽字符包装（_wfopen/_waccess），并保留 ANSI 回退以兼容 MinGW 下
 * 由 ANSI 代码页（如 GBK）编码的 argv 原始字节。
 */

#ifndef VLLM_UTIL_H
#define VLLM_UTIL_H

#include <stdio.h>

#if defined(_WIN32)

#include <windows.h>
#include <io.h>
#include <stdlib.h>
#include <wchar.h>

/* 按代码页 cp 把字符串转成宽字符（含结尾 NUL）；失败返回 NULL。 */
static inline wchar_t *st_util_wide(const char *s, UINT cp) {
    if (!s) return NULL;
    int n = MultiByteToWideChar(cp, 0, s, -1, NULL, 0);
    if (n <= 0) return NULL;
    wchar_t *w = (wchar_t *)malloc((size_t)n * sizeof(wchar_t));
    if (!w) return NULL;
    MultiByteToWideChar(cp, 0, s, -1, w, n);
    return w;
}

/* fopen 包装：先按 UTF-8 解码路径；若打开失败（MinGW 下 argv 可能是 ANSI
 * 原始字节）再按 ACP 重试一次。 */
static inline FILE *st_fopen(const char *path, const char *mode) {
    wchar_t wmode[16] = {0};
    if (MultiByteToWideChar(CP_UTF8, 0, mode, -1, wmode, 16) <= 0) return NULL;
    for (int i = 0; i < 2; i++) {
        wchar_t *wp = st_util_wide(path, i == 0 ? CP_UTF8 : CP_ACP);
        if (!wp) continue;
        FILE *f = _wfopen(wp, wmode);
        free(wp);
        if (f) return f;
    }
    return NULL;
}

/* access 包装（文件存在性判断），回退策略同 st_fopen。 */
static inline int st_access(const char *path, int mode) {
    for (int i = 0; i < 2; i++) {
        wchar_t *wp = st_util_wide(path, i == 0 ? CP_UTF8 : CP_ACP);
        if (!wp) continue;
        int r = _waccess(wp, mode);
        free(wp);
        if (r == 0) return 0;
    }
    return -1;
}

#else /* POSIX: fopen/access 已原生处理 UTF-8 路径 */

#include <unistd.h>

#define st_fopen   fopen
#define st_access  access

#endif /* _WIN32 */

#endif /* VLLM_UTIL_H */

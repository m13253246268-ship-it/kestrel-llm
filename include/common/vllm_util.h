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

/* ================================================================
 * UTF-8 规整（JSON / 配置文件的写入口）
 *
 * 必要性：Windows 上 argv 与环境变量是 ANSI 代码页（GBK 等）的字节串，直接
 * 塞进 JSON 会产生非法多字节序列——读回时 vjson_parse 会拒绝**整个**文件，
 * Python supervisor 的 json.load 同样报错（中文模型路径即触发）。项目内
 * 路径/名称统一按 UTF-8 处理，故在写 JSON 前规整一次。
 * 策略：已是合法 UTF-8 的原样复制（Linux 常态、UTF-8 输入常态）；否则在
 * Windows 上按 ACP 无损转成 UTF-8（保留中文路径），再退化为把非法字节替换
 * 为 '?'。输出恒为合法 UTF-8。
 * ================================================================ */

/* 字节串是否为合法 UTF-8。 */
static inline int st_util_is_utf8(const char *s) {
    const unsigned char *p = (const unsigned char *)(s ? s : "");
    while (*p) {
        unsigned char c = *p;
        int extra;
        if (c < 0x80)                    extra = 0;
        else if (c >= 0xF0 && c <= 0xF4) extra = 3;
        else if (c >= 0xE0 && c <= 0xEF) extra = 2;
        else if (c >= 0xC2 && c <= 0xDF) extra = 1;
        else return 0;
        for (int i = 1; i <= extra; i++)
            if ((p[i] & 0xC0) != 0x80) return 0;
        p += extra + 1;
    }
    return 1;
}

/* 兜底规整：非法/截断字节以 '?' 替换（不切割多字节字符）。 */
static inline void st_util_utf8_scrub(const char *in, char *out, size_t cap) {
    size_t w = 0;
    const unsigned char *p = (const unsigned char *)(in ? in : "");
    if (!out || cap == 0) return;
    while (*p && w + 1 < cap) {
        unsigned char c = *p;
        int extra;
        if (c < 0x80)                    extra = 0;
        else if (c >= 0xF0 && c <= 0xF4) extra = 3;
        else if (c >= 0xE0 && c <= 0xEF) extra = 2;
        else if (c >= 0xC2 && c <= 0xDF) extra = 1;
        else { out[w++] = '?'; p++; continue; }
        int ok = 1;
        for (int i = 1; i <= extra; i++)
            if ((p[i] & 0xC0) != 0x80) { ok = 0; break; }
        if (!ok) { out[w++] = '?'; p++; continue; }
        for (int i = 0; i <= extra && w + 1 < cap; i++) out[w++] = (char)p[i];
        p += extra + 1;
    }
    out[w] = '\0';
}

#if defined(_WIN32)

#include <windows.h>
#include <io.h>
#include <stdlib.h>
#include <wchar.h>

/* ANSI(ACP) -> UTF-8；已是合法 UTF-8 则原样复制。 */
static inline void st_util_utf8(const char *in, char *out, size_t cap) {
    const char *s = in ? in : "";
    if (!out || cap == 0) return;
    out[0] = '\0';
    if (st_util_is_utf8(s)) { snprintf(out, cap, "%s", s); return; }
    int wn = MultiByteToWideChar(CP_ACP, 0, s, -1, NULL, 0);
    if (wn > 0) {
        wchar_t *w = (wchar_t *)malloc((size_t)wn * sizeof(wchar_t));
        if (w) {
            MultiByteToWideChar(CP_ACP, 0, s, -1, w, wn);
            int un = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
            if (un > 0 && (size_t)un <= cap) {
                WideCharToMultiByte(CP_UTF8, 0, w, -1, out, un, NULL, NULL);
                free(w);
                return;
            }
            free(w);
        }
    }
    st_util_utf8_scrub(s, out, cap);
}

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

/* POSIX 下 argv/路径本即 UTF-8，仅需过滤模型输出可能产生的半个字符。 */
static inline void st_util_utf8(const char *in, char *out, size_t cap) {
    st_util_utf8_scrub(in, out, cap);
}

#endif /* _WIN32 */

#endif /* VLLM_UTIL_H */

/* SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 裴晓光 and contributors */
/* ================================================================
 * vllm_audit.c - 管理面审计日志（追加写 + SM3 哈希链）
 *
 * 记录格式/校验语义/诚实边界见 vllm_audit.h。本文件只做：路径与目录、追加、
 * 回看（tail）、链校验（逐行重算 SM3 + prev 链 + seq 连续性）。
 *
 * 与引擎其余部分的关系：**不参与任何数值路径**，纯 I/O 与管理面留痕。
 * ================================================================ */
#include "vllm_audit.h"
#include "vllm_crypto.h"     /* SM3（零依赖国密） */
#include "vllm_http.h"       /* VHttpMutex（跨平台锁） */
#include "vllm_platform.h"   /* st_mkdir */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>

static int            g_on = 0;
static char           g_path[1024];
static long           g_seq = 0;              /* 已写条数（= 末行 seq） */
static unsigned char  g_prev[32];             /* 末行 hash（链头） */
static VHttpMutex    *g_lock = NULL;
static int            g_dir_ready = 0;
static int            g_warned = 0;
static long           g_written = 0;          /* 本进程已知的文件字节数（越权写检测） */
static int            g_tamper_warned = 0;

/* ---------- 小工具 ---------- */

static void aud_hex(const unsigned char *h, size_t n, char *out) {
    static const char *H = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[i * 2]     = H[(h[i] >> 4) & 0xF];
        out[i * 2 + 1] = H[h[i] & 0xF];
    }
    out[n * 2] = '\0';
}

/* JSON 字符串转义 + UTF-8 规整（非法/截断字节替换为 '?'）。
 * 规整是必要的：非 UTF-8 会让管理页的 JSON.parse 整体失败（审查面不可用）。 */
static void aud_esc(const char *s, char *out, size_t cap) {
    size_t w = 0;
    const unsigned char *p = (const unsigned char *)(s ? s : "");
    if (!out || cap < 8) return;
    while (*p && w + 8 < cap) {
        unsigned char c = *p;
        if (c < 0x80) {
            switch (c) {
            case '"':  out[w++] = '\\'; out[w++] = '"';  p++; continue;
            case '\\': out[w++] = '\\'; out[w++] = '\\'; p++; continue;
            case '\n': out[w++] = '\\'; out[w++] = 'n';  p++; continue;
            case '\r': out[w++] = '\\'; out[w++] = 'r';  p++; continue;
            case '\t': out[w++] = '\\'; out[w++] = 't';  p++; continue;
            default:
                if (c < 0x20) {
                    int n = snprintf(out + w, cap - w, "\\u%04x", (unsigned)c);
                    if (n <= 0) { out[w] = '\0'; return; }
                    w += (size_t)n;
                } else {
                    out[w++] = (char)c;
                }
                p++;
                continue;
            }
        }
        int extra;
        if (c >= 0xF0 && c <= 0xF4)      extra = 3;
        else if (c >= 0xE0 && c <= 0xEF) extra = 2;
        else if (c >= 0xC2 && c <= 0xDF) extra = 1;
        else { out[w++] = '?'; p++; continue; }
        int ok = 1;
        for (int i = 1; i <= extra; i++)
            if ((p[i] & 0xC0) != 0x80) { ok = 0; break; }
        if (!ok) { out[w++] = '?'; p++; continue; }
        for (int i = 0; i <= extra && w + 8 < cap; i++) out[w++] = (char)p[i];
        p += extra + 1;
    }
    out[w] = '\0';
}

static void aud_mkdirs(char *dir) {
    if (!dir || !dir[0]) return;
    for (char *p = dir + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            char save = *p;
            *p = '\0';
            st_mkdir(dir, 0755);
            *p = save;
        }
    }
    st_mkdir(dir, 0755);
}

/* 只对**目录部分**做 mkdir -p（g_path 是文件路径；直接把文件路径喂给
 * aud_mkdirs 会把文件本身建成同名目录，导致后续 fopen 追加全部失败）。 */
static void aud_ensure_parent(void) {
    char d[1024];
    snprintf(d, sizeof(d), "%s", g_path);
    char *cut = NULL;
    for (char *q = d; *q; q++)
        if (*q == '/' || *q == '\\') cut = q;
    if (cut) {
        *cut = '\0';
        if (d[0]) aud_mkdirs(d);
    }
}

/* 读出日志内容（超大时只读尾部窗口，并丢弃首个不完整行）。
 * *out_partial = 1 表示内容被窗口截断。返回 malloc'd（NUL 结尾）。 */
static char *aud_read_all(long *out_size, int *out_partial) {
    if (out_size) *out_size = 0;
    if (out_partial) *out_partial = 0;
    if (!g_path[0]) return NULL;
    struct stat sb;
    if (stat(g_path, &sb) != 0 || sb.st_size <= 0) return NULL;
    long sz = (long)sb.st_size;
    long off = 0;
    int partial = 0;
    if ((unsigned long long)sz > (unsigned long long)VLLM_AUDIT_READ_CAP) {
        off = sz - (long)VLLM_AUDIT_READ_CAP;
        partial = 1;
    }
    FILE *f = fopen(g_path, "rb");
    if (!f) return NULL;
    if (off > 0 && fseek(f, off, SEEK_SET) != 0) { fclose(f); return NULL; }
    size_t want = (size_t)(sz - off);
    char *buf = (char *)malloc(want + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, want, f);
    fclose(f);
    buf[got] = '\0';
    if (partial && got > 0) {
        char *nl = strchr(buf, '\n');
        if (nl) {
            size_t skip = (size_t)(nl - buf) + 1;
            memmove(buf, nl + 1, got - skip);
            got -= skip;
            buf[got] = '\0';
        }
    }
    if (out_size) *out_size = (long)got;
    if (out_partial) *out_partial = partial;
    return buf;
}

/* 最后一处 needle（NULL = 无）。 */
static char *aud_last_str(char *hay, const char *needle) {
    if (!hay || !needle) return NULL;
    size_t nl = strlen(needle);
    char *found = NULL, *p = hay;
    while ((p = strstr(p, needle)) != NULL) { found = p; p += nl; }
    return found;
}

/* ---------- 对外 API ---------- */

void vllm_audit_init(VLLMServerCtx *ctx) {
    g_on = 0;
    g_seq = 0;
    g_path[0] = '\0';
    g_dir_ready = 0;
    g_warned = 0;
    memset(g_prev, 0, sizeof(g_prev));
    if (!ctx) return;

    const char *env = getenv("VLLM_AUDIT_LOG");
    const char *p = (env && env[0]) ? env
                                    : (ctx->audit_log[0] ? ctx->audit_log : NULL);
    if (!p || !p[0]) {
        ctx->audit_on = 0;
        return;
    }
    snprintf(g_path, sizeof(g_path), "%s", p);
    if (!g_lock) g_lock = vhttp_mutex_new();

    /* 从末行恢复 seq 与链头 hash（变更后仍能续链）。 */
    long sz = 0;
    char *buf = aud_read_all(&sz, NULL);
    if (buf) {
        char *p2 = buf;
        char *last = NULL;
        long first_seq = -1;
        int lines = 0;
        while (*p2) {
            char *e = strchr(p2, '\n');
            size_t l = e ? (size_t)(e - p2) : strlen(p2);
            if (l > 0 && p2[l - 1] == '\r') l--;
            if (l > 0) {
                lines++;
                if (first_seq < 0) {
                    char *sp0 = strstr(p2, "\"seq\":");
                    first_seq = sp0 ? atol(sp0 + 6) : 0;
                }
                last = p2;
            }
            if (!e) break;
            p2 = e + 1;
        }
        if (last) {
            char *hp = aud_last_str(last, ",\"hash\":\"");
            char *sp = strstr(last, "\"seq\":");
            if (sp) g_seq = atol(sp + 6);
            if (hp && strlen(hp + 9) >= 64) {
                char hx[65];
                memcpy(hx, hp + 9, 64);
                hx[64] = '\0';
                if (vc_hex_decode(hx, g_prev, 32) < 0) memset(g_prev, 0, 32);
            }
        }
        /* 前缀不在文件内（被归档或删除）：链只能在文件范围内自洽，提前告警。 */
        if (lines > 0 && (first_seq > 1 || (long)lines != g_seq)) {
            fprintf(stderr, "[AUDIT] WARN: chain prefix absent (first_seq=%ld "
                            "lines=%d last_seq=%ld) — archived or removed "
                            "out-of-band; the chain is verifiable only within "
                            "this file\n", first_seq, lines, g_seq);
            fflush(stderr);
        }
        free(buf);
    }
    ctx->audit_on = 1;
    g_on = 1;
    /* 记录已知文件长度：后续 append 前比对，可发现"越权改动日志"（改写/截断/
     * 追加）并告警——此时链大概率已断，verify 会给出 bad_seq。 */
    {
        struct stat sb;
        g_written = (stat(g_path, &sb) == 0) ? (long)sb.st_size : 0;
    }
    fprintf(stderr, "[AUDIT] enabled: %s (resume seq=%ld)\n", g_path, g_seq);
    fflush(stderr);
}

int vllm_audit_on(const VLLMServerCtx *ctx) {
    return ctx && g_on;
}

void vllm_audit_add(VLLMServerCtx *ctx, const char *actor, const char *role,
                    const char *action, const char *target,
                    const char *result, const char *detail) {
    if (!ctx || !g_on) return;
    if (!g_lock) g_lock = vhttp_mutex_new();

    char a[160], r[96], ac[96], tg[768], rs[24], dt[640];
    aud_esc(actor, a, sizeof(a));
    aud_esc(role, r, sizeof(r));
    aud_esc(action, ac, sizeof(ac));
    aud_esc(target, tg, sizeof(tg));
    aud_esc(result, rs, sizeof(rs));
    aud_esc(detail, dt, sizeof(dt));

    char prevh[65];
    aud_hex(g_prev, 32, prevh);

    size_t cap = 512 + strlen(a) + strlen(r) + strlen(ac) + strlen(tg) +
                 strlen(rs) + strlen(dt) + strlen(prevh);
    char *line = (char *)malloc(cap);
    if (!line) return;
    int n = snprintf(line, cap,
        "{\"seq\":%ld,\"ts\":%ld,\"actor\":\"%s\",\"role\":\"%s\",\"action\":\"%s\","
        "\"target\":\"%s\",\"result\":\"%s\",\"detail\":\"%s\",\"prev\":\"%s\"",
        g_seq + 1, (long)time(NULL), a, r, ac, tg, rs, dt, prevh);
    if (n <= 0 || (size_t)n >= cap) { free(line); return; }

    /* hash = SM3(prev 原始 32B || 本行去掉 ,"hash":... 后的规范文本) */
    vc_sm3_ctx c;
    vc_sm3_init(&c);
    vc_sm3_update(&c, g_prev, 32);
    vc_sm3_update(&c, line, (size_t)n);
    unsigned char h[32];
    vc_sm3_final(&c, h);
    char hh[65];
    aud_hex(h, 32, hh);

    vhttp_mutex_lock(g_lock);
    if (!g_dir_ready) {
        aud_ensure_parent();
        g_dir_ready = 1;
    }
    /* 越权改动检测：文件长度与"本进程已知长度"不符 → 日志被外部改写/截断/追加。
     * 只告警一次（不自动修复、不重播种链头）：继续沿用内存中的链头写入，使断点
     * 保持可被 vllm_audit_verify 发现（宁可让尾部可疑，也不静默掩盖）。 */
    if (!g_tamper_warned) {
        struct stat sb;
        long cur = (stat(g_path, &sb) == 0) ? (long)sb.st_size : 0;
        if (cur != g_written) {
            fprintf(stderr, "[AUDIT] WARN: log size changed out-of-band "
                            "(%ld -> %ld): %s may have been modified; "
                            "run the chain check\n", g_written, cur, g_path);
            fflush(stderr);
            g_tamper_warned = 1;
            g_written = cur;
        }
    }
    FILE *f = fopen(g_path, "a");
    if (f) {
        fprintf(f, "%s,\"hash\":\"%s\"}\n", line, hh);
        long end = ftell(f);
        fclose(f);
        if (end > 0) g_written = end;
        memcpy(g_prev, h, sizeof(g_prev));
        g_seq++;
        /* 超告警水位提醒归档（不自动轮转：轮转会破坏链的连续可校验性）。 */
        if (!g_warned) {
            struct stat sb;
            if (stat(g_path, &sb) == 0 &&
                (unsigned long long)sb.st_size > VLLM_AUDIT_WARN_SIZE) {
                fprintf(stderr, "[AUDIT] WARN: %s exceeds %u MB — archive it "
                                "(no auto-rotation keeps the chain verifiable)\n",
                        g_path, VLLM_AUDIT_WARN_SIZE >> 20);
                fflush(stderr);
                g_warned = 1;
            }
        }
    } else {
        fprintf(stderr, "[AUDIT] append failed: %s\n", g_path);
        fflush(stderr);
    }
    vhttp_mutex_unlock(g_lock);
    free(line);
}

char *vllm_audit_tail(VLLMServerCtx *ctx, int n, int *out_count) {
    if (out_count) *out_count = 0;
    if (!ctx || !g_on || n < 1) return NULL;
    if (n > 2000) n = 2000;

    char *buf = aud_read_all(NULL, NULL);
    if (!buf) return NULL;

    /* 环形收集最后 n 行的起始指针（避免为长日志分配行指针数组）。 */
    char **ring = (char **)malloc((size_t)n * sizeof(char *));
    int *lens = (int *)malloc((size_t)n * sizeof(int));
    if (!ring || !lens) { free(ring); free(lens); free(buf); return NULL; }
    int total = 0;
    char *p = buf;
    while (*p) {
        char *e = strchr(p, '\n');
        int l = e ? (int)(e - p) : (int)strlen(p);
        if (l > 0 && p[l - 1] == '\r') l--;
        if (l > 0) {
            ring[total % n] = p;
            lens[total % n] = l;
            total++;
        }
        if (!e) break;
        p = e + 1;
    }
    int keep = total < n ? total : n;

    size_t cap = 64;
    for (int i = 0; i < keep; i++) cap += (size_t)lens[i] + 2;
    char *out = (char *)malloc(cap);
    if (!out) { free(ring); free(lens); free(buf); return NULL; }
    size_t w = 0;
    out[w++] = '[';
    int start = total - keep;
    for (int i = 0; i < keep; i++) {
        int idx = (start + i) % n;
        if (i) out[w++] = ',';
        memcpy(out + w, ring[idx], (size_t)lens[idx]);
        w += (size_t)lens[idx];
    }
    out[w++] = ']';
    out[w] = '\0';

    free(ring);
    free(lens);
    free(buf);
    if (out_count) *out_count = keep;
    return out;
}

int vllm_audit_verify(VLLMServerCtx *ctx, int *checked, long *bad_seq, int *full,
                      int *genesis, long *first_seq, int *empty) {
    if (checked) *checked = 0;
    if (bad_seq) *bad_seq = 0;
    if (full) *full = 0;
    if (genesis) *genesis = 0;
    if (first_seq) *first_seq = 0;
    if (empty) *empty = 0;
    if (!ctx || !g_on) return 0;

    int partial = 0;
    char *buf = aud_read_all(NULL, &partial);
    if (!buf) {
        /* 日志不存在或为空（刚启用/刚轮转）：不是篡改，中性报告。 */
        if (full) *full = 1;
        if (empty) *empty = 1;
        return 1;
    }
    if (full) *full = partial ? 0 : 1;

    unsigned char prev[32];
    memset(prev, 0, sizeof(prev));
    long exp = 1;
    int n = 0, ok = 1, first = 1;

    char *p = buf;
    while (*p) {
        char *e = strchr(p, '\n');
        size_t len = e ? (size_t)(e - p) : strlen(p);
        if (len > 0 && p[len - 1] == '\r') len--;
        if (len == 0) {
            if (!e) break;
            p = e + 1;
            continue;
        }
        char save = p[len];
        p[len] = '\0';
        n++;

        char *hp = strstr(p, ",\"hash\":\"");
        char *pp = strstr(p, ",\"prev\":\"");
        char *sp = strstr(p, "\"seq\":");
        long seq = 0;
        int bad = 0;
        if (!hp || !pp || !sp || strlen(hp + 9) < 64 || strlen(pp + 9) < 64) {
            bad = 1;
        } else {
            seq = atol(sp + 6);
            char prevs[65], stored[65];
            memcpy(prevs, pp + 9, 64); prevs[64] = '\0';
            memcpy(stored, hp + 9, 64); stored[64] = '\0';
            unsigned char prevb[32];
            if (vc_hex_decode(prevs, prevb, 32) < 0) {
                bad = 1;
            } else {
                if (first) {
                    if (first_seq) *first_seq = seq;
                    if (seq == 1 && !partial) {
                        if (genesis) *genesis = 1;   /* 从创世行起完整 */
                    } else {
                        /* 读窗口截断，或文件前缀已被归档/删除：只以首行 prev
                         * 为基准校验内部一致性（genesis 保持 0，绝不冒充完整）。 */
                        memcpy(prev, prevb, 32);
                        exp = seq;
                    }
                }
                if (seq != exp) bad = 1;
                if (!bad && memcmp(prevb, prev, 32) != 0) bad = 1;
                if (!bad) {
                    unsigned char h[32];
                    char hex[65];
                    vc_sm3_ctx c;
                    vc_sm3_init(&c);
                    vc_sm3_update(&c, prev, 32);
                    vc_sm3_update(&c, p, (size_t)(hp - p));
                    vc_sm3_final(&c, h);
                    aud_hex(h, 32, hex);
                    if (strcmp(hex, stored) != 0) bad = 1;
                    if (!bad) { memcpy(prev, h, 32); exp = seq + 1; }
                }
            }
        }
        p[len] = save;
        first = 0;
        if (bad) {
            ok = 0;
            if (bad_seq) *bad_seq = seq > 0 ? seq : (long)n;
            break;
        }
        if (!e) break;
        p = e + 1;
    }
    if (checked) *checked = n;
    if (n == 0 && empty) *empty = 1;
    free(buf);
    return ok ? 1 : 0;
}

long vllm_audit_size(VLLMServerCtx *ctx) {
    if (!ctx || !g_on || !g_path[0]) return 0;
    struct stat sb;
    if (stat(g_path, &sb) != 0) return 0;
    return (long)sb.st_size;
}

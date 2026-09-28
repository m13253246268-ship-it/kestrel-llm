/* SPDX-License-Identifier: Apache-2.0
 * Copyright (C) 2026 裴晓光 and contributors */
/* ================================================================
 * vllm_session.c - 服务端会话存储（工业边缘：会话分区 + 历史审查）
 *
 * 纯数据层：会话文件的读/写/枚举与 LRU 清理。**不含推理路径**。
 * 文件名取 tenant_id+user_id+session_id 三元组的 FNV-1a（无路径字符），
 * 三字段同时明文写入文件，供管理页按租户/用户/规则过滤与审查。
 *
 * 内存策略：只落盘消息时间线，不在内存常驻多份 KV（见 vllm_session.h）。
 * ================================================================ */
#include "vllm_session.h"
#include "vllm_http.h"
#include "vllm_platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#define VLLM_SESS_DEFAULT_LIMIT 1024
#define VLLM_SESS_MAX_FILE      (8u << 20)   /* 单会话文件上限 8MB */

/* ---------- 小工具 ---------- */

static unsigned long long sess_fnv(const char *s, unsigned long long h) {
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211ULL; }
    return h;
}

static void sess_mkdirs(char *dir) {
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

/* 逐字符转义写 JSON 字符串（内容可为长文本，故流式写，不用固定缓冲）。 */
static void sess_wesc(FILE *f, const char *s) {
    if (!s) s = "";
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
        case '"':  fputs("\\\"", f); break;
        case '\\': fputs("\\\\", f); break;
        case '\n': fputs("\\n", f); break;
        case '\r': fputs("\\r", f); break;
        case '\t': fputs("\\t", f); break;
        case '\b': fputs("\\b", f); break;
        case '\f': fputs("\\f", f); break;
        default:
            if (*p < 0x20) fprintf(f, "\\u%04x", (unsigned)*p);
            else fputc((int)*p, f);
        }
    }
    fputc('"', f);
}

/* 把 VJson 子树序列化为 malloc'd 文本（缓冲按需放大）。失败 NULL。 */
static char *sess_ser_heap(const VJson *v) {
    size_t cap = 1024;
    for (int attempt = 0; attempt < 6; attempt++) {
        char *buf = (char *)malloc(cap);
        if (!buf) return NULL;
        size_t n = vjson_serialize(v, buf, cap);
        if (n > 0) return buf;
        free(buf);
        cap *= 8;
    }
    return NULL;
}

/* 规整为合法 UTF-8 的副本（非法/截断字节替换为 '?'）。
 * 必要性：模型输出经 byte-BPE 可能产生半个字符的字节序列，原样写进会话 JSON
 * 会让下次 vllm_sess_parse 拒绝整个文件（历史静默丢失）。 */
static char *sess_utf8_dup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    size_t w = 0;
    const unsigned char *p = (const unsigned char *)s;
    while (*p) {
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
        for (int i = 0; i <= extra; i++) out[w++] = (char)p[i];
        p += extra + 1;
    }
    out[w] = '\0';
    return out;
}

/* ---------- 路径 / 读写 ---------- */

void vllm_sess_path(const VLLMServerCtx *ctx, const char *tenant,
                    const char *user, const char *sid, char *out, size_t cap) {
    if (!out || cap == 0) return;
    out[0] = '\0';
    if (!ctx || !ctx->session_on || !ctx->session_dir[0] || !sid || !sid[0])
        return;
    const char *t = (tenant && tenant[0]) ? tenant : "anon";
    const char *u = (user && user[0]) ? user : "anon";
    unsigned long long h = 1469598103934665603ULL;
    h = sess_fnv(t, h); h = sess_fnv("\x1f", h);
    h = sess_fnv(u, h); h = sess_fnv("\x1f", h);
    h = sess_fnv(sid, h);
    snprintf(out, cap, "%s/sess_%016llx.json", ctx->session_dir, h);
}

char *vllm_sess_read(const VLLMServerCtx *ctx, const char *tenant,
                     const char *user, const char *sid) {
    char p[1400];
    vllm_sess_path(ctx, tenant, user, sid, p, sizeof(p));
    if (!p[0]) return NULL;
    /* 先取实际大小再分配（边缘设备上按 8MB 固定分配过于浪费；文件不存在时
     * stat 失败即提前返回，也省掉一次 fopen）。 */
    struct stat sb;
    if (stat(p, &sb) != 0 || sb.st_size <= 0 ||
        (unsigned long long)sb.st_size > VLLM_SESS_MAX_FILE)
        return NULL;
    size_t sz = (size_t)sb.st_size;
    FILE *f = fopen(p, "rb");
    if (!f) return NULL;
    char *buf = (char *)malloc(sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, sz, f);
    fclose(f);
    buf[n] = '\0';
    return buf;
}

VJson *vllm_sess_parse(const char *json_text) {
    if (!json_text || !json_text[0]) return NULL;
    VJson *o = vjson_parse(json_text);
    if (!o || o->type != VJ_OBJECT) { vjson_free(o); return NULL; }
    return o;
}

int vllm_sess_write(const VLLMServerCtx *ctx, const char *tenant,
                    const char *user, const char *sid, const char *rulebook_id,
                    const VJson *messages, int n_tokens, double last_ms,
                    char *err, size_t errcap) {
    if (!ctx || !ctx->session_on || !ctx->session_dir[0]) {
        if (err) snprintf(err, errcap, "session dir not configured");
        return -1;
    }
    if (!sid || !sid[0]) {
        if (err) snprintf(err, errcap, "empty session_id");
        return -1;
    }
    char p[1400];
    vllm_sess_path(ctx, tenant, user, sid, p, sizeof(p));
    if (!p[0]) { if (err) snprintf(err, errcap, "bad session path"); return -1; }

    /* created_s 从旧文件继承（会话创建时间不随每轮刷新）。 */
    long created = (long)time(NULL);
    char *old = vllm_sess_read(ctx, tenant, user, sid);
    if (old) {
        VJson *oj = vllm_sess_parse(old);
        if (oj) {
            double c = vjson_num(vjson_obj_get(oj, "created_s"));
            if (c > 0) created = (long)c;
            vjson_free(oj);
        }
        free(old);
    }

    char dir[1024];
    snprintf(dir, sizeof(dir), "%s", ctx->session_dir);
    sess_mkdirs(dir);

    FILE *f = fopen(p, "w");
    if (!f) { if (err) snprintf(err, errcap, "cannot write %s", p); return -1; }

    const char *t = (tenant && tenant[0]) ? tenant : "anon";
    const char *u = (user && user[0]) ? user : "anon";
    fputs("{\n", f);
    fputs("  \"session_id\": ", f);   sess_wesc(f, sid);      fputs(",\n", f);
    fputs("  \"tenant_id\": ", f);    sess_wesc(f, t);        fputs(",\n", f);
    fputs("  \"user_id\": ", f);      sess_wesc(f, u);        fputs(",\n", f);
    fputs("  \"rulebook_id\": ", f);  sess_wesc(f, rulebook_id ? rulebook_id : "");
    fputs(",\n", f);
    fprintf(f, "  \"created_s\": %ld,\n", created);
    fprintf(f, "  \"updated_s\": %ld,\n", (long)time(NULL));
    fprintf(f, "  \"n_tokens\": %d,\n", n_tokens);
    fprintf(f, "  \"last_ms\": %.1f,\n", last_ms > 0.0 ? last_ms : 0.0);
    fprintf(f, "  \"n_messages\": %d,\n",
            messages ? (int)vjson_array_len(messages) : 0);
    fputs("  \"messages\": [", f);
    size_t nm = messages ? vjson_array_len(messages) : 0;
    for (size_t i = 0; i < nm; i++) {
        const VJson *m = vjson_array_get(messages, i);
        char *ms = m ? sess_ser_heap(m) : NULL;
        if (!ms) continue;
        fputs(i ? ", " : "", f);
        fputs(ms, f);
        free(ms);
    }
    fputs("]\n}\n", f);
    if (fclose(f) != 0) {
        if (err) snprintf(err, errcap, "short write on %s", p);
        return -1;
    }
    return 0;
}

VJson *vllm_sess_append_turns(const VJson *old, const char *query,
                              const char *answer) {
    VJson *arr = vjson_new_array();
    size_t nm = old ? vjson_array_len(old) : 0;
    for (size_t i = 0; i < nm; i++) {
        const VJson *m = vjson_array_get(old, i);
        if (m) vjson_array_push(arr, vjson_clone(m));
    }
    if (query) {
        char *cq = sess_utf8_dup(query);
        VJson *um = vjson_new_object();
        vjson_obj_set(um, "role", vjson_new_string("user"));
        vjson_obj_set(um, "content", vjson_new_string(cq ? cq : ""));
        vjson_array_push(arr, um);
        free(cq);
    }
    if (answer && answer[0]) {
        char *ca = sess_utf8_dup(answer);
        VJson *am = vjson_new_object();
        vjson_obj_set(am, "role", vjson_new_string("assistant"));
        vjson_obj_set(am, "content", vjson_new_string(ca ? ca : ""));
        vjson_array_push(arr, am);
        free(ca);
    }
    return arr;
}

/* ---------- LRU 清理 ---------- */

typedef struct { char path[1400]; long mtime; } SessFile;

static int sess_cmp_mtime(const void *a, const void *b) {
    long x = ((const SessFile *)a)->mtime, y = ((const SessFile *)b)->mtime;
    return (x > y) - (x < y);
}

void vllm_sess_gc(const VLLMServerCtx *ctx) {
    if (!ctx || !ctx->session_on || !ctx->session_dir[0]) return;
    int limit = ctx->session_limit > 0 ? ctx->session_limit
                                       : VLLM_SESS_DEFAULT_LIMIT;
    DIR *d = opendir(ctx->session_dir);
    if (!d) return;
    int cap = 256, n = 0;
    SessFile *files = (SessFile *)malloc((size_t)cap * sizeof(SessFile));
    if (!files) { closedir(d); return; }
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        const char *nm = de->d_name;
        if (strncmp(nm, "sess_", 5) != 0) continue;
        size_t ln = strlen(nm);
        if (ln < 6 || strcmp(nm + ln - 5, ".json") != 0) continue;
        if (n == cap) {
            int nc = cap * 2;
            SessFile *nf = (SessFile *)realloc(files, (size_t)nc * sizeof(SessFile));
            if (!nf) break;
            files = nf; cap = nc;
        }
        snprintf(files[n].path, sizeof(files[n].path), "%s/%s",
                 ctx->session_dir, nm);
        struct stat sb;
        files[n].mtime = (stat(files[n].path, &sb) == 0) ? (long)sb.st_mtime : 0;
        n++;
    }
    closedir(d);

    if (n > limit) {
        qsort(files, (size_t)n, sizeof(SessFile), sess_cmp_mtime);
        int drop = n - limit;
        for (int i = 0; i < drop; i++) {
            if (remove(files[i].path) == 0)
                fprintf(stderr, "[SESS] LRU evicted %s\n", files[i].path);
        }
    }
    free(files);
}

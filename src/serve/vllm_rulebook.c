/* SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 裴晓光 and contributors */
/* ================================================================
 * vllm_rulebook.c - 规则包注册中心（工业边缘：预置命名上下文）
 *
 * 纯数据层：目录读写、内容/模型指纹、system 前导块渲染、分块检索选择。
 * **不含任何数值/推理路径**：prefill 与 KV 落盘的编排在 vllm_server.c 的
 * vllm_rb_build()（复用既有 ist_reset + st_kv_disk_save），本文件只负责
 * 元数据/文本与"选哪一段文本注入"。
 *
 * 口径（与项目既有前缀复用一致）：复用为"确定性但近似"，非位级克隆。
 * ================================================================ */
#include "vllm_rulebook.h"
#include "vllm_http.h"
#include "vllm_platform.h"
#include "vllm_util.h"       /* st_util_utf8：元数据 JSON 写入口的 UTF-8 规整 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

/* ---------- 小工具 ---------- */

static unsigned long long fnv_init(void) { return 1469598103934665603ULL; }

static unsigned long long fnv_bytes(unsigned long long h, const void *p, size_t n) {
    const unsigned char *q = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) { h ^= q[i]; h *= 1099511628211ULL; }
    return h;
}

static unsigned long long fnv_u64(unsigned long long h, unsigned long long v) {
    for (int b = 0; b < 8; b++) { h ^= (v & 0xFFu); h *= 1099511628211ULL; v >>= 8; }
    return h;
}

void vllm_rb_hash_hex(const void *data, size_t n, char out[VLLM_RB_HASH_HEX]) {
    if (!out) return;
    unsigned long long h = fnv_bytes(fnv_init(), data, n);
    snprintf(out, VLLM_RB_HASH_HEX, "%016llx", h);
}

/* mkdir -p（与 vllm_admin.c 的 config_mkdirs 同义）。 */
static void rb_mkdirs(char *dir) {
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

/* 读整个文件（按实际大小分配，上限 cap 字节）。返回 malloc'd（NUL 结尾）；
 * 不存在 / 超上限 / 空文件返回 NULL。 */
static char *rb_read_file(const char *path, size_t cap) {
    struct stat sb;
    if (stat(path, &sb) != 0 || sb.st_size <= 0 ||
        (unsigned long long)sb.st_size > cap)
        return NULL;
    size_t sz = (size_t)sb.st_size;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char *buf = (char *)malloc(sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, sz, f);
    fclose(f);
    buf[n] = '\0';
    return buf;
}

/* 只允许 [A-Za-z0-9._-]，且长度受限：规则包 ID 直接进文件名，必须防注入。
 * 显式按 ASCII 判定（不用 isalnum：后者受 locale 影响，非 C locale 下可能把
 * 高位字节判为字母，破坏"文件名只用 ASCII"的前提）。 */
static int rb_id_safe(const char *id) {
    if (!id || !id[0] || strlen(id) >= VLLM_RB_ID_MAX) return 0;
    for (const char *p = id; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.'))
            return 0;
    }
    return 1;
}

int vllm_rb_id_ok(const char *id) { return rb_id_safe(id); }

static void rb_path(const VLLMServerCtx *ctx, const char *id, const char *ext,
                    char *out, size_t cap) {
    snprintf(out, cap, "%s/rb_%s%s", ctx->rulebook_dir, id, ext);
}

/* 把任意字节串规整为合法 UTF-8（见 vllm_util.h）。
 * 必要性：Windows 上 argv 为 ANSI 代码页（GBK 等），原样写进 JSON 会让读回时
 * 的 vjson_parse 因非法多字节序列拒绝**整个**元数据文件；模型输出经 byte-BPE
 * 也可能产生半个字符的字节序列。规整后 JSON 恒可解析。 */
static void json_put_str(VJson *o, const char *k, const char *v) {
    char tmp[VLLM_RB_SCOPE_MAX + 64];
    st_util_utf8(v, tmp, sizeof(tmp));
    vjson_obj_set(o, k, vjson_new_string(tmp));
}

static void json_get_str(const VJson *o, const char *k, char *out, size_t cap) {
    const char *s = vjson_str(vjson_obj_get(o, k));
    snprintf(out, cap, "%s", s ? s : "");
}

/* ---------- 模型指纹 ---------- */

void vllm_rb_model_fp(const VLLMServerCtx *ctx, char out[VLLM_RB_HASH_HEX]) {
    if (!out) return;
    out[0] = '\0';
    if (!ctx || !ctx->cfg || ctx->load_state != 2) return;

    unsigned long long h = fnv_init();
    const char *mn = (ctx->model_name && ctx->model_name[0])
                         ? ctx->model_name
                         : (ctx->model_id ? ctx->model_id : "");
    h = fnv_bytes(h, mn, strlen(mn));
    const char *md = ctx->model_dir ? ctx->model_dir : "";
    h = fnv_bytes(h, md, strlen(md));

    char cfgp[1200];
    if (md[0]) {
        snprintf(cfgp, sizeof(cfgp), "%s/config.json", md);
        FILE *f = fopen(cfgp, "rb");
        if (f) {
            char buf[4096];
            size_t n;
            while ((n = fread(buf, 1, sizeof(buf), f)) > 0) h = fnv_bytes(h, buf, n);
            fclose(f);
        }
    }

    /* KV 几何：与 st_kv_disk_load 的头校验同源（层数/头数/head_dim/kv_bs）。 */
    h = fnv_u64(h, (unsigned long long)ctx->cfg->n_layers);
    h = fnv_u64(h, (unsigned long long)ctx->cfg->n_kv_heads);
    h = fnv_u64(h, (unsigned long long)ctx->cfg->head_dim);
    if (ctx->ist) h = fnv_u64(h, (unsigned long long)ctx->ist->kv_bs);

    /* 权重文件 (name, size, mtime)：权重一变，旧的 KV 快照即失效。 */
    if (md[0]) {
        DIR *d = opendir(md);
        if (d) {
            struct dirent *de;
            while ((de = readdir(d)) != NULL) {
                const char *nm = de->d_name;
                const char *dot = strrchr(nm, '.');
                if (!dot) continue;
                if (strcmp(dot, ".vqf") != 0 && strcmp(dot, ".safetensors") != 0 &&
                    strcmp(dot, ".gguf") != 0 && strcmp(dot, ".bin") != 0) continue;
                char p[1400];
                snprintf(p, sizeof(p), "%s/%s", md, nm);
                struct stat sb;
                if (stat(p, &sb) != 0) continue;
                h = fnv_bytes(h, nm, strlen(nm));
                h = fnv_u64(h, (unsigned long long)sb.st_size);
                h = fnv_u64(h, (unsigned long long)sb.st_mtime);
            }
            closedir(d);
        }
    }
    snprintf(out, VLLM_RB_HASH_HEX, "%016llx", h);
}

/* ---------- system 前导块 ---------- */

char *vllm_rb_render_block(const char *text) {
    if (!text) text = "";
    size_t n = strlen(text);
    char *b = (char *)malloc(n + 64);
    if (!b) return NULL;
    snprintf(b, n + 64, "<|im_start|>system\n%s<|im_end|>\n", text);
    return b;
}

/* ---------- 注册中心（磁盘） ---------- */

int vllm_rb_write_meta(const VLLMServerCtx *ctx, const VLLMRulebook *rb,
                       const char *text, char *err, size_t errcap) {
    if (!ctx || !ctx->rulebook_on || !ctx->rulebook_dir[0] || !rb ||
        !rb_id_safe(rb->id)) {
        if (err) snprintf(err, errcap, "rulebook dir not configured or bad id");
        return -1;
    }
    if (!text) text = "";

    char dirbuf[1024];
    snprintf(dirbuf, sizeof(dirbuf), "%s", ctx->rulebook_dir);
    rb_mkdirs(dirbuf);

    char tp[1400], jp[1400];
    rb_path(ctx, rb->id, ".txt", tp, sizeof(tp));
    rb_path(ctx, rb->id, ".json", jp, sizeof(jp));

    FILE *f = fopen(tp, "wb");
    if (!f) { if (err) snprintf(err, errcap, "cannot write %s", tp); return -1; }
    size_t tn = strlen(text);
    if (tn > 0 && fwrite(text, 1, tn, f) != tn) {
        fclose(f);
        if (err) snprintf(err, errcap, "short write on %s", tp);
        return -1;
    }
    fclose(f);

    VJson *o = vjson_new_object();
    json_put_str(o, "rulebook_id", rb->id);
    json_put_str(o, "name", rb->name);
    json_put_str(o, "version", rb->version);
    json_put_str(o, "tenant_id", rb->tenant_id);
    json_put_str(o, "scope", rb->scope);
    json_put_str(o, "hash", rb->hash);
    json_put_str(o, "model_fp", rb->model_fp);
    vjson_obj_set(o, "n_tokens", vjson_new_number((double)rb->n_tokens));
    vjson_obj_set(o, "n_chunks", vjson_new_number((double)rb->n_chunks));
    vjson_obj_set(o, "created_s", vjson_new_number((double)rb->created_s));
    vjson_obj_set(o, "updated_s", vjson_new_number((double)rb->updated_s));
    vjson_obj_set(o, "text_len", vjson_new_number((double)tn));
    vjson_obj_set(o, "text_sha_hex", vjson_new_string(rb->hash));

    char buf[4096];
    size_t n = vjson_serialize(o, buf, sizeof(buf));
    vjson_free(o);
    if (n == 0) { if (err) snprintf(err, errcap, "meta serialize failed"); return -1; }

    f = fopen(jp, "w");
    if (!f) { if (err) snprintf(err, errcap, "cannot write %s", jp); return -1; }
    fputs(buf, f);
    fclose(f);
    return 0;
}

static int rb_load_common(const VLLMServerCtx *ctx, const char *id,
                          VLLMRulebook *out, int want_text) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!ctx || !ctx->rulebook_on || !ctx->rulebook_dir[0] || !rb_id_safe(id))
        return -1;

    char jp[1400];
    rb_path(ctx, id, ".json", jp, sizeof(jp));
    char *jbuf = rb_read_file(jp, 1 << 20);
    if (!jbuf) return -1;
    VJson *o = vjson_parse(jbuf);
    free(jbuf);
    if (!o || o->type != VJ_OBJECT) { vjson_free(o); return -1; }

    json_get_str(o, "rulebook_id", out->id, sizeof(out->id));
    json_get_str(o, "name", out->name, sizeof(out->name));
    json_get_str(o, "version", out->version, sizeof(out->version));
    json_get_str(o, "tenant_id", out->tenant_id, sizeof(out->tenant_id));
    json_get_str(o, "scope", out->scope, sizeof(out->scope));
    json_get_str(o, "hash", out->hash, sizeof(out->hash));
    json_get_str(o, "model_fp", out->model_fp, sizeof(out->model_fp));
    out->n_tokens  = (int)vjson_num(vjson_obj_get(o, "n_tokens"));
    out->n_chunks  = (int)vjson_num(vjson_obj_get(o, "n_chunks"));
    out->created_s = (long)vjson_num(vjson_obj_get(o, "created_s"));
    out->updated_s = (long)vjson_num(vjson_obj_get(o, "updated_s"));
    vjson_free(o);
    if (!out->id[0]) snprintf(out->id, sizeof(out->id), "%s", id);

    if (!want_text) return 0;   /* 管理面列表：不读正文 */
    char tp[1400];
    rb_path(ctx, id, ".txt", tp, sizeof(tp));
    out->text = rb_read_file(tp, VLLM_RB_MAX_TEXT);
    return 0;
}

int vllm_rb_load(const VLLMServerCtx *ctx, const char *id, VLLMRulebook *out) {
    return rb_load_common(ctx, id, out, 1);
}

int vllm_rb_load_meta(const VLLMServerCtx *ctx, const char *id,
                      VLLMRulebook *out) {
    return rb_load_common(ctx, id, out, 0);
}

void vllm_rb_free(VLLMRulebook *rb) {
    if (!rb) return;
    free(rb->text);
    rb->text = NULL;
}

int vllm_rb_delete(const VLLMServerCtx *ctx, const char *id) {
    if (!ctx || !ctx->rulebook_on || !ctx->rulebook_dir[0] || !rb_id_safe(id))
        return 0;
    int n = 0;
    char p[1400];
    rb_path(ctx, id, ".json", p, sizeof(p));
    if (remove(p) == 0) n++;
    rb_path(ctx, id, ".txt", p, sizeof(p));
    if (remove(p) == 0) n++;

    /* 该 id 的全部 KV 快照：rbk_<id>_*.kv（前缀匹配，避免前缀串扰用 '_' 分界）。 */
    char prefix[VLLM_RB_ID_MAX + 16];
    snprintf(prefix, sizeof(prefix), "rbk_%s_", id);
    size_t plen = strlen(prefix);
    DIR *d = opendir(ctx->rulebook_dir);
    if (d) {
        struct dirent *de;
        while ((de = readdir(d)) != NULL) {
            const char *nm = de->d_name;
            size_t ln = strlen(nm);
            if (ln < 4 || strcmp(nm + ln - 3, ".kv") != 0) continue;
            if (strncmp(nm, prefix, plen) != 0) continue;
            char fp[1800];
            snprintf(fp, sizeof(fp), "%s/%s", ctx->rulebook_dir, nm);
            if (remove(fp) == 0) n++;
        }
        closedir(d);
    }
    return n;
}

void vllm_rb_kv_path(const VLLMServerCtx *ctx, const VLLMRulebook *rb,
                     const int *ids, int n, char *out, size_t cap) {
    if (!out || cap == 0) return;
    out[0] = '\0';
    if (!ctx || !rb || !ids || n <= 0) return;
    char ah[VLLM_RB_HASH_HEX];
    vllm_rb_hash_hex(ids, (size_t)n * sizeof(int), ah);
    const char *ver = rb->version[0] ? rb->version : "v0";
    snprintf(out, cap, "%s/rbk_%s_%s_%.8s.kv", ctx->rulebook_dir, rb->id, ver, ah);
}

/* ---------- 分块检索 ---------- */

/* 段落切分：以行为单位累积到 ~RB_CHUNK_CHARS 字符为一块（token 数由
 * 编码器实测，字符数只用于控制块数与检索开销）。 */
#define RB_CHUNK_CHARS 1200

typedef struct { const char *p; size_t len; } RbChunk;

static int rb_split_chunks(const char *text, RbChunk *out, int max_n) {
    int n = 0;
    const char *p = text;
    while (*p && n < max_n) {
        while (*p == '\n' || *p == '\r') p++;
        if (!*p) break;
        const char *s = p;
        const char *q = p;
        size_t acc = 0;
        while (*q && acc < RB_CHUNK_CHARS) {
            const char *nl = strchr(q, '\n');
            size_t l = nl ? (size_t)(nl - q) : strlen(q);
            acc += l + 1;
            q = nl ? nl + 1 : q + l;
            if (!nl) break;
        }
        size_t len = (size_t)(q - s);
        while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r')) len--;
        if (len > 0) { out[n].p = s; out[n].len = len; n++; }
        p = q;
    }
    return n;
}

static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static int cmp_scored(const void *a, const void *b) {
    /* 按分数降序；同分按块序升序（确定性）。 */
    const int *x = (const int *)a, *y = (const int *)b;
    if (x[1] != y[1]) return y[1] - x[1];
    return x[0] - y[0];
}

/* 返回 malloc'd 文本（调用方 free）；*out_n_tokens = 选中文本的 token 数。 */
char *vllm_rb_select_text(VLLMServerCtx *ctx, const VLLMRulebook *rb,
                          const char *query, int budget_tokens,
                          int *out_n_tokens) {
    if (out_n_tokens) *out_n_tokens = 0;
    if (!ctx || !ctx->tok || !rb || !rb->text) return NULL;

    int max_ids = ctx->ist ? ctx->ist->max_kv_slots : 8192;
    if (max_ids < 1024) max_ids = 1024;
    /* 渲染包裹 "<|im_start|>system\n" + text + "<|im_end|>\n" 的固定开销。 */
    int text_budget = budget_tokens - 8;
    if (text_budget < 64) text_budget = 64;

    int *scratch = (int *)malloc(((size_t)max_ids + 16) * sizeof(int));
    if (!scratch) return NULL;

    int full = qwen_tokenizer_encode(ctx->tok, rb->text, scratch, max_ids + 16);
    if (full <= text_budget) {
        /* 手册整体塞得下：等价"全塞"，保持最优前缀复用。返回 NULL 表示
         * "直接用 rb->text"（调用方无需再拷贝一份）。 */
        free(scratch);
        if (out_n_tokens) *out_n_tokens = full;
        return NULL;
    }

    /* query token 集合（排序后二分查找，避免哈希表）。 */
    int qcap = 2048;
    int *qtok = (int *)malloc((size_t)qcap * sizeof(int));
    if (!qtok) { free(scratch); return NULL; }
    int qn = qwen_tokenizer_encode(ctx->tok, query ? query : "", qtok, qcap);
    if (qn < 1) {
        /* 无 query：退回"按预算取头部"（保底可用，确定性）。 */
        free(qtok);
        size_t take = (size_t)((double)text_budget / (double)(full > 0 ? full : 1) *
                               (double)strlen(rb->text));
        if (take == 0) take = strlen(rb->text);
        char *r = (char *)malloc(take + 1);
        if (!r) { free(scratch); return NULL; }
        memcpy(r, rb->text, take);
        r[take] = '\0';
        if (out_n_tokens)
            *out_n_tokens = qwen_tokenizer_encode(ctx->tok, r, scratch, max_ids + 16);
        free(scratch);
        return r;
    }
    qsort(qtok, (size_t)qn, sizeof(int), cmp_int);

    /* 去重 query token（二分查找时重复无害，但统计去重更稳）。 */
    int qu = 0;
    for (int i = 0; i < qn; i++)
        if (i == 0 || qtok[i] != qtok[i - 1]) qtok[qu++] = qtok[i];

    int max_paras = VLLM_RB_MAX_CHUNKS;
    RbChunk *chunks = (RbChunk *)malloc((size_t)max_paras * sizeof(RbChunk));
    int *score = (int *)malloc((size_t)max_paras * sizeof(int));
    int *order = (int *)malloc((size_t)max_paras * 2 * sizeof(int));
    int *len_tok = (int *)malloc((size_t)max_paras * sizeof(int));
    if (!chunks || !score || !order || !len_tok) {
        free(chunks); free(score); free(order); free(len_tok);
        free(qtok); free(scratch);
        return NULL;
    }
    int nc = rb_split_chunks(rb->text, chunks, max_paras);

    /* 逐块：编码、统计与 query 的 token 重叠、记录长度。 */
    for (int i = 0; i < nc; i++) {
        char *tmp = (char *)malloc(chunks[i].len + 1);
        if (!tmp) { chunks[i].len = 0; score[i] = 0; len_tok[i] = 0; continue; }
        memcpy(tmp, chunks[i].p, chunks[i].len);
        tmp[chunks[i].len] = '\0';
        int k = qwen_tokenizer_encode(ctx->tok, tmp, scratch, max_ids + 16);
        free(tmp);
        len_tok[i] = k;
        int ov = 0;
        for (int j = 0; j < k; j++) {
            int *hit = (int *)bsearch(&scratch[j], qtok, (size_t)qu,
                                      sizeof(int), cmp_int);
            if (hit) ov++;
        }
        score[i] = ov;
    }

    /* 按分数选块直到预算耗尽。 */
    for (int i = 0; i < nc; i++) { order[i * 2] = i; order[i * 2 + 1] = score[i]; }
    qsort(order, (size_t)nc, 2 * sizeof(int), cmp_scored);

    int used = 0;
    char *sel = (char *)calloc((size_t)nc, 1);   /* 每块选中标记（按原序输出） */
    if (!sel) {
        free(chunks); free(score); free(order); free(len_tok);
        free(qtok); free(scratch);
        return NULL;
    }
    for (int i = 0; i < nc; i++) {
        int idx = order[i * 2];
        if (len_tok[idx] <= 0) continue;
        if (used + len_tok[idx] > text_budget) continue;
        sel[idx] = 1;
        used += len_tok[idx];
    }
    /* 全都没选中（单块就超预算）：取首块并硬截到预算（保底，确定性）。 */
    if (used == 0 && nc > 0) sel[0] = 1;

    /* 按原序拼回，块间以空行分隔。 */
    size_t cap = strlen(rb->text) + (size_t)nc * 2 + 4;
    char *out = (char *)malloc(cap);
    if (!out) {
        free(sel); free(chunks); free(score); free(order); free(len_tok);
        free(qtok); free(scratch);
        return NULL;
    }
    size_t w = 0;
    for (int i = 0; i < nc; i++) {
        if (!sel[i]) continue;
        if (w > 0) { out[w++] = '\n'; out[w++] = '\n'; }
        memcpy(out + w, chunks[i].p, chunks[i].len);
        w += chunks[i].len;
    }
    out[w] = '\0';
    if (out_n_tokens)
        *out_n_tokens = qwen_tokenizer_encode(ctx->tok, out, scratch, max_ids + 16);

    free(sel); free(chunks); free(score); free(order); free(len_tok);
    free(qtok); free(scratch);
    return out;
}

int vllm_rb_chunk_count(const char *text) {
    if (!text || !text[0]) return 0;
    RbChunk *chunks = (RbChunk *)malloc((size_t)VLLM_RB_MAX_CHUNKS * sizeof(RbChunk));
    if (!chunks) return 0;
    int n = rb_split_chunks(text, chunks, VLLM_RB_MAX_CHUNKS);
    free(chunks);
    return n;
}

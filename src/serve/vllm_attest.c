/* ================================================================
 * vllm_attest.c - 可验证推理背书（方案 2：基于密码学签名的硬件可信证明）。
 *
 * 设计要点：
 *   - 默认关闭：VLLM_ATTEST=1 才启用（不回退任何默认行为/性能）。
 *   - 每次请求出证 = 请求结束时一次 SM3（转录）+ 一次 SM2 签名。
 *   - 摘要结构见 vllm_attest.h：可被验证方用公钥离线复算校验。
 *   - 设备密钥对：目录（VLLM_ATTEST_DIR，默认 "."）下 vllm_attest.priv
 *     （64 hex，0600）/ vllm_attest.pub（128 hex）。缺失则自动生成。
 *   - SM2 可辨别标识 ID = "VLLM-ATTEST-1"（验证方必须一致）。
 * ================================================================ */
/* 注意：vllm_server.h 须先于 vllm_attest.h（后者原型使用 VLLMServerCtx）。 */
#include "vllm_server.h"
#include "vllm_attest.h"
#include "vllm_crypto.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#if !defined(_WIN32)
#include <unistd.h>
#endif
#include <pthread.h>   /* batch 多 worker 并发出证互斥 */

/* SM2 用户可辨别标识（ZA 计算用；验证方需使用相同 ID）。 */
static const uint8_t VATT_ID[] = VATT_USER_ID;
#define VATT_ID_LEN (sizeof(VATT_ID) - 1)

static const char VATT_MAGIC[] = VATT_MAGIC_V3;

/* 出证互斥：vatt_digest/vatt_sign 使用静态缓冲，连续批处理（--batch-max≥2）
 * 下多个 HTTP worker 会在各自线程同时出证，必须串行化。 */
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

/* ---------------- 模块状态（进程内单服务 ctx） ---------------- */
static VLLMServerCtx *g_ctx = NULL;   /* vatt_init 时绑定；推理在 inf_lock 串行 */
static uint8_t  g_priv[32];
static uint8_t  g_pub[64];
static int      g_key_ok = 0;         /* 私钥/公钥已就绪 */
static char     g_pub_hex[129];
/* 最近一次出证的摘要/签名（供 vatt_verify_own 逐请求健康回读）。 */
static char     g_last_digest[65];
static char     g_last_sig[257];

/* ---------------- 小工具 ---------------- */
static void vatt_hex(const uint8_t *in, size_t n, char *out) {
    static const char *d = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[2 * i]     = d[in[i] >> 4];
        out[2 * i + 1] = d[in[i] & 0xf];
    }
    out[2 * n] = '\0';
}

static void vatt_u32be(uint8_t *b, uint32_t v) {
    b[0] = (uint8_t)(v >> 24); b[1] = (uint8_t)(v >> 16);
    b[2] = (uint8_t)(v >> 8);  b[3] = (uint8_t)v;
}

/* F(x) = u32be(len) || bytes；len 0 也写入。 */
static void vatt_frame(vc_sm3_ctx *c, const void *p, size_t n) {
    uint8_t h[4];
    vatt_u32be(h, (uint32_t)n);
    vc_sm3_update(c, h, 4);
    if (n) vc_sm3_update(c, p, n);
}

static int vatt_hex2bin(const char *hex, uint8_t *out, size_t n) {
    return vc_hex_decode(hex, out, n);
}

/* ---------------- 模型指纹 ---------------- */
/* SM3 读一个文件的完整内容（config.json 规模，小文件）。返回 0 成功。 */
static int vatt_sm3_file(const char *path, uint8_t out[32]) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    vc_sm3_ctx c;
    vc_sm3_init(&c);
    uint8_t buf[8192];
    size_t r;
    while ((r = fread(buf, 1, sizeof(buf), f)) > 0)
        vc_sm3_update(&c, buf, r);
    fclose(f);
    vc_sm3_final(&c, out);
    return 0;
}

static int vatt_name_cmp(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

/* strdup 替代（不依赖 _GNU_SOURCE / POSIX 宏）。 */
static char *vatt_strdup(const char *s) {
    size_t n = strlen(s);
    char *d = (char *)malloc(n + 1);
    if (d) memcpy(d, s, n + 1);
    return d;
}

/* 目录文件清单 (name,size) 归并进哈希 —— 权重被替换但字节数相同的场景
 * 由 VQF 供应链签名兜底（见头文件信任链说明）。 */
static void vatt_feed_dir(vc_sm3_ctx *c, const char *dir) {
    DIR *d = opendir(dir);
    if (!d) { vatt_frame(c, "NODIR", 5); return; }
    char **names = NULL;
    size_t n = 0, cap = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        char full[2048];
        snprintf(full, sizeof(full), "%s/%s", dir, e->d_name);
        struct stat st;
        if (stat(full, &st) != 0 || !S_ISREG(st.st_mode)) continue;
        if (n == cap) {
            cap = cap ? cap * 2 : 32;
            char **nb = (char **)realloc(names, cap * sizeof(char *));
            if (!nb) { closedir(d); free(names); return; }
            names = nb;
        }
        names[n++] = vatt_strdup(e->d_name);
    }
    closedir(d);
    qsort(names, n, sizeof(char *), vatt_name_cmp);
    char line[2304];
    for (size_t i = 0; i < n; i++) {
        char full[2048];
        snprintf(full, sizeof(full), "%s/%s", dir, names[i]);
        struct stat st;
        if (stat(full, &st) == 0) {
            int ln = snprintf(line, sizeof(line), "%s:%ld", names[i], (long)st.st_size);
            if (ln > 0) vatt_frame(c, line, (size_t)ln);
        }
        free(names[i]);
    }
    free(names);
}

void vatt_model_ready(VLLMServerCtx *ctx) {
    if (!ctx || !ctx->attest_on) return;
    vc_sm3_ctx c;
    vc_sm3_init(&c);
    vatt_frame(&c, "MODEL-FP-1", 10);
    /* model_name（真实身份，fill_model_name 已生成） */
    const char *mn = ctx->model_name && ctx->model_name[0]
                         ? ctx->model_name : (ctx->model_id ? ctx->model_id : "");
    vatt_frame(&c, mn, strlen(mn));
    /* model_dir */
    const char *md = ctx->model_dir ? ctx->model_dir : "";
    vatt_frame(&c, md, strlen(md));
    /* config.json 内容 */
    if (md[0]) {
        char cfg[2048];
        snprintf(cfg, sizeof(cfg), "%s/config.json", md);
        uint8_t d[32];
        if (vatt_sm3_file(cfg, d) == 0) vatt_frame(&c, d, 32);
    }
    /* 目录文件清单 */
    if (md[0]) vatt_feed_dir(&c, md);
    uint8_t fp[32];
    vc_sm3_final(&c, fp);
    vatt_hex(fp, 32, ctx->attest_fp);
}

/* ---------------- 密钥管理 ---------------- */
static int vatt_key_paths(char *privp, size_t pcap, char *pubp, size_t ucap) {
    const char *dir = g_ctx && g_ctx->attest_dir[0] ? g_ctx->attest_dir : ".";
    snprintf(privp, pcap, "%s/vllm_attest.priv", dir);
    snprintf(pubp, ucap, "%s/vllm_attest.pub", dir);
    return 0;
}

static int vatt_write_restrict(const char *path, const char *data) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t n = strlen(data);
    int ok = fwrite(data, 1, n, f) == n;
    fclose(f);
    if (ok) chmod(path, 0600);
    return ok ? 0 : -1;
}

/* 自动生成设备密钥对（私钥由内核 CSPRNG 提供；超范围重试）。 */
static int vatt_keygen(void) {
    uint8_t priv[32], pub[64];
    for (int attempt = 0; attempt < 8; attempt++) {
        if (vc_secure_rand(priv) != 0) return -1;
        if (vc_sm2_pub_from_priv(priv, pub) == 0) {
            char priv_hex[65], pub_hex[129];
            vatt_hex(priv, 32, priv_hex);
            vatt_hex(pub, 64, pub_hex);
            char privp[2048], pubp[2048];
            vatt_key_paths(privp, sizeof(privp), pubp, sizeof(pubp));
            /* 已存在则不改写（防覆盖线上密钥）。 */
            struct stat st;
            if (stat(privp, &st) == 0 || stat(pubp, &st) == 0) return -1;
            if (vatt_write_restrict(privp, priv_hex) != 0) return -1;
            if (vatt_write_restrict(pubp, pub_hex) != 0) return -1;
            memcpy(g_priv, priv, 32);
            memcpy(g_pub, pub, 64);
            strcpy(g_pub_hex, pub_hex);
            g_key_ok = 1;
            if (getenv("VLLM_ATTEST_LOG")) {
                fprintf(stderr, "[ATTEST] keygen: %s (0600), pub=%s\n",
                        privp, g_pub_hex);
            }
            return 0;
        }
    }
    return -1;
}

/* 读取既有密钥；公钥与私钥推导不一致视为坏密钥。 */
static int vatt_keyload(void) {
    char privp[2048], pubp[2048];
    vatt_key_paths(privp, sizeof(privp), pubp, sizeof(pubp));
    FILE *fp = fopen(privp, "rb");
    if (!fp) return -1;
    char priv_hex[65] = {0};
    size_t got = fread(priv_hex, 1, 64, fp);
    fclose(fp);
    if (got != 64) return -1;
    priv_hex[64] = '\0';
    FILE *pu = fopen(pubp, "rb");
    if (!pu) return -1;
    char pub_hex[129] = {0};
    got = fread(pub_hex, 1, 128, pu);
    fclose(pu);
    if (got != 128) return -1;
    pub_hex[128] = '\0';
    uint8_t priv[32], pub[64], p2[64];
    if (vatt_hex2bin(priv_hex, priv, 32) != 32) return -1;
    if (vatt_hex2bin(pub_hex, pub, 64) != 64) return -1;
    if (vc_sm2_pub_from_priv(priv, p2) != 0) return -1;
    if (memcmp(pub, p2, 64) != 0) return -1;   /* 公私钥不匹配 */
    memcpy(g_priv, priv, 32);
    memcpy(g_pub, pub, 64);
    strcpy(g_pub_hex, pub_hex);
    g_key_ok = 1;
    return 0;
}

/* ---------------- 摘要与签名 ---------------- */
/* 采样参数字符串（摘要帧与 JSON 字段共用同一格式）。
 * k=top_k 截断（0=关闭）、th=enable_thinking；二者可能来自 serve 级默认
 * （--top-k / VLLM_ENABLE_THINKING），请求原文里看不到，必须进摘要。 */
static void vatt_params_str(const VAttestReq *r, char *out, size_t cap) {
    snprintf(out, cap, "t=%.4g,p=%.4g,m=%.4g,k=%d,mt=%d,th=%d",
             r->temperature, r->top_p, r->min_p, r->top_k, r->max_tokens,
             r->thinking ? 1 : 0);
}

void vatt_body_digest(const void *body, size_t n, char hex[65]) {
    uint8_t d[32];
    if (body && n) vc_sm3(body, n, d);
    else memset(d, 0, sizeof(d));
    vatt_hex(d, 32, hex);
}

/* 按公开帧规范计算转录摘要。返回摘要 hex（65 字节缓冲）。
 * VLLM_ATTEST_DUMP=1 时把完整预映像逐字节 hex 打到 stderr（排障/对拍用）。 */
static const char *vatt_digest(const VAttestReq *r) {
    static char hexbuf[65];
    const int dump = getenv("VLLM_ATTEST_DUMP") ? 1 : 0;
    char *dbuf = NULL;
    size_t dn = 0, dcap = 0;
    vc_sm3_ctx c;
    vc_sm3_init(&c);
#define DF(p, n) do { \
        vc_sm3_update(&c, (p), (n)); \
        if (dump) { \
            if (dn + (n) > dcap) { \
                size_t nc = dcap ? dcap * 2 : 1024; \
                while (nc < dn + (n)) nc *= 2; \
                char *nb = (char *)realloc(dbuf, nc); \
                if (nb) { dbuf = nb; dcap = nc; } else { dump_safe = 0; } \
            } \
            if (dump_safe && (n)) { memcpy(dbuf + dn, (p), (n)); dn += (n); } \
        } \
    } while (0)
#define DFLEN(p, n) do { uint8_t _h[4]; vatt_u32be(_h, (uint32_t)(n)); DF(_h, 4); DF((p), (n)); } while (0)
    int dump_safe = 1;
    DF(VATT_MAGIC, strlen(VATT_MAGIC));
    DFLEN(g_ctx->attest_fp, strlen(g_ctx->attest_fp));
    const char *mid = g_ctx->model_id ? g_ctx->model_id : "";
    DFLEN(mid, strlen(mid));
    const char *dev = g_ctx->device_id ? g_ctx->device_id : "";
    DFLEN(dev, strlen(dev));
    const char *u = r->user ? r->user : "";
    DFLEN(u, strlen(u));
    char params[128];
    vatt_params_str(r, params, sizeof(params));
    DFLEN(params, strlen(params));
    char npt[24], ngt[24], ts[24];
    snprintf(npt, sizeof(npt), "%d", r->n_prompt_tokens);
    snprintf(ngt, sizeof(ngt), "%d", r->n_gen_tokens);
    snprintf(ts, sizeof(ts), "%ld", r->ts);
    DFLEN(npt, strlen(npt));
    DFLEN(ngt, strlen(ngt));
    const char *fin = r->finish ? r->finish : "stop";
    DFLEN(fin, strlen(fin));
    DFLEN(ts, strlen(ts));
    /* 请求原文绑定：body_sha = SM3(客户端原始请求体) 的 64-hex（schema=3）。
     * 验证方用自己保存的请求原文复算 SM3 即可比对，无需 tokenizer/模板。 */
    const char *bs = r->body_sha ? r->body_sha : "";
    DFLEN(bs, strlen(bs));
    if (r->text && r->text_len) DFLEN(r->text, r->text_len);
    else DFLEN("", 0);
    uint8_t d[32];
    vc_sm3_final(&c, d);
    vatt_hex(d, 32, hexbuf);
    if (dump && dump_safe) {
        char *hx = (char *)malloc(dn * 2 + 1);
        if (hx) {
            vatt_hex((const uint8_t *)dbuf, dn, hx);
            fprintf(stderr, "[ATTEST] dump preimage=%s\n", hx);
            free(hx);
        }
    }
    free(dbuf);
    return hexbuf;
#undef DF
#undef DFLEN
}

/* 随机数 k（SM2 签名）。非 Linux 失败返回 -1。 */
static int vatt_rand_k(uint8_t k[32]) {
    return vc_secure_rand(k);
}

/* 用 r_hex||s_hex（128 hex）校验 digest_hex（64 hex）。返回 1 有效。 */
static int vatt_verify(const char *digest_hex, const char *sig_hex) {
    uint8_t digest[32], r[32], s[32];
    if (vatt_hex2bin(digest_hex, digest, 32) != 32) return 0;
    if (strlen(sig_hex) != 128) return 0;
    if (vatt_hex2bin(sig_hex, r, 32) != 32) return 0;
    if (vatt_hex2bin(sig_hex + 64, s, 32) != 32) return 0;
    return vc_sm2_verify(g_pub, digest, 32, VATT_ID, VATT_ID_LEN, r, s);
}

static int vatt_sign(const char *digest_hex, char *sig_hex /* 257 */) {
    uint8_t digest[32], k[32];
    if (vatt_hex2bin(digest_hex, digest, 32) != 32) return -1;
    if (vatt_rand_k(k) != 0) return -1;
    uint8_t r[32], s[32];
    if (vc_sm2_sign(g_priv, digest, 32, VATT_ID, VATT_ID_LEN, k, r, s) != 0)
        return -1;
    vatt_hex(r, 32, sig_hex);
    vatt_hex(s, 32, sig_hex + 64);
    return 0;
}

/* JSON 字符串转义（user 等自由文本字段用）。 */
static void vatt_json_escape(const char *in, char *out, size_t cap) {
    size_t o = 0;
    for (const unsigned char *p = (const unsigned char *)in;
         *p && o + 6 < cap; p++) {
        switch (*p) {
        case '"':  out[o++] = '\\'; out[o++] = '"';  break;
        case '\\': out[o++] = '\\'; out[o++] = '\\'; break;
        case '\n': out[o++] = '\\'; out[o++] = 'n';  break;
        case '\r': out[o++] = '\\'; out[o++] = 'r';  break;
        case '\t': out[o++] = '\\'; out[o++] = 't';  break;
        default:   out[o++] = (char)*p; break;
        }
    }
    out[o] = '\0';
}

char *vatt_seal_json(const VAttestReq *r) {
    if (!vatt_active() || !r || !r->text) return NULL;
    pthread_mutex_lock(&g_lock);
    const char *digest = vatt_digest(r);
    char sig[257];
    if (vatt_sign(digest, sig) != 0) {
        pthread_mutex_unlock(&g_lock);
        return NULL;
    }
    strcpy(g_last_digest, digest);
    strcpy(g_last_sig, sig);

    const char *mid = g_ctx->model_id ? g_ctx->model_id : "";
    const char *dev = g_ctx->device_id ? g_ctx->device_id : "";
    char model_esc[512], dev_esc[128], user_esc[512];
    vatt_json_escape(mid, model_esc, sizeof(model_esc));
    vatt_json_escape(dev, dev_esc, sizeof(dev_esc));
    vatt_json_escape(r->user ? r->user : "", user_esc, sizeof(user_esc));
    const char *fin = r->finish ? r->finish : "stop";
    char params[128];
    vatt_params_str(r, params, sizeof(params));
    const char *bs = r->body_sha ? r->body_sha : "";

    /* 固定大小缓冲：内容全部是受控字段。 */
    char *out = (char *)malloc(4096);
    if (!out) {
        pthread_mutex_unlock(&g_lock);
        return NULL;
    }
    int n = snprintf(out, 4096,
        "{\"schema\":3,\"algo\":\"SM2-SM3\",\"ts\":%ld,"
        "\"model\":\"%s\",\"model_fp\":\"%s\",\"device\":\"%s\",\"user\":\"%s\","
        "\"params\":\"%s\",\"prompt_tokens\":%d,\"n_tokens\":%d,\"finish\":\"%s\","
        "\"body_sha\":\"%s\",\"digest\":\"%s\",\"signature\":\"%s\"}",
        r->ts, model_esc, g_ctx->attest_fp, dev_esc, user_esc, params,
        r->n_prompt_tokens, r->n_gen_tokens, fin, bs, digest, sig);
    if (n <= 0 || n >= 4096) {
        free(out);
        pthread_mutex_unlock(&g_lock);
        return NULL;
    }

    /* 设备侧健康回读：逐请求自动 PASS/FAIL 日志。 */
    int vok = vatt_verify(digest, sig);
    if (getenv("VLLM_ATTEST_LOG")) {
        fprintf(stderr, "[ATTEST] seal digest=%s verify=%s\n",
                digest, vok ? "PASS" : "FAIL");
    }
    pthread_mutex_unlock(&g_lock);
    return out;
}

int vatt_verify_own(const char *digest_hex) {
    if (!g_key_ok || !digest_hex) return 0;
    pthread_mutex_lock(&g_lock);
    int r = vatt_verify(digest_hex, g_last_sig);
    pthread_mutex_unlock(&g_lock);
    return r;
}

/* ---------------- 自检（自动化 PASS/FAIL 门） ---------------- */
int vatt_selftest(void) {
    int pass = 1;
    /* 1) 密钥一致性 */
    uint8_t p2[64];
    if (g_key_ok && vc_sm2_pub_from_priv(g_priv, p2) == 0 &&
        memcmp(g_pub, p2, 64) == 0) {
        fprintf(stderr, "[ATTEST] keypair        : PASS (pub=%s)\n", g_pub_hex);
    } else {
        fprintf(stderr, "[ATTEST] keypair        : FAIL (no usable key)\n");
        return 0;
    }
    /* 2) 签名→验签往返 + 摘要确定性（vatt_digest 复用静态缓冲，先拷贝） */
    VAttestReq r;
    memset(&r, 0, sizeof(r));
    r.user = "selftest-user";
    r.body_sha = "5c14ba5ed25e9ce7f1a1235e6f2d9e8b6f9c7d8e9a0b1c2d3e4f5a6b7c8d9e0f";
    r.text = "2";
    r.text_len = 1;
    r.n_prompt_tokens = 5;
    r.n_gen_tokens = 1;
    r.finish = "stop";
    r.temperature = 0.6;
    r.top_p = 0.95;
    r.min_p = 0.0;
    r.top_k = 20;
    r.thinking = 1;
    r.max_tokens = 128;
    r.ts = 1700000000L;
    char d1[65], d2[65], d3[65];
    strcpy(d1, vatt_digest(&r));
    char s1[129];
    if (vatt_sign(d1, s1) == 0 && vatt_verify(d1, s1)) {
        fprintf(stderr, "[ATTEST] sign/verify    : PASS\n");
    } else {
        fprintf(stderr, "[ATTEST] sign/verify    : FAIL\n");
        pass = 0;
    }
    strcpy(d2, vatt_digest(&r));
    if (strcmp(d1, d2) == 0) {
        fprintf(stderr, "[ATTEST] digest-determ  : PASS\n");
    } else {
        fprintf(stderr, "[ATTEST] digest-determ  : FAIL\n");
        pass = 0;
    }
    /* 3) 篡改检测：改一个字节 → 摘要必须不同（验签亦须失败） */
    VAttestReq r2 = r;
    r2.text = "3";
    strcpy(d3, vatt_digest(&r2));
    char s2[129];
    int tamper_ok = strcmp(d1, d3) != 0;
    if (tamper_ok && vatt_sign(d3, s2) == 0) {
        tamper_ok = (vatt_verify(d1, s2) == 0);   /* 用错摘要验签必须失败 */
    }
    if (tamper_ok) {
        fprintf(stderr, "[ATTEST] tamper-detect   : PASS\n");
    } else {
        fprintf(stderr, "[ATTEST] tamper-detect   : FAIL\n");
        pass = 0;
    }
    /* 4) schema=3 新增绑定：只改 top_k 或 thinking 也必须改摘要
     * （否则"实际采样档位"就没被真正绑住）。 */
    VAttestReq r3 = r;
    r3.top_k = 21;
    char d4[65];
    strcpy(d4, vatt_digest(&r3));
    VAttestReq r4 = r;
    r4.thinking = 0;
    char d5[65];
    strcpy(d5, vatt_digest(&r4));
    if (strcmp(d1, d4) != 0 && strcmp(d1, d5) != 0 && strcmp(d4, d5) != 0) {
        fprintf(stderr, "[ATTEST] bind-k/think   : PASS\n");
    } else {
        fprintf(stderr, "[ATTEST] bind-k/think   : FAIL\n");
        pass = 0;
    }
    return pass;
}

/* ---------------- 初始化 ---------------- */
int vatt_init(VLLMServerCtx *ctx) {
    if (!ctx) return -1;
    ctx->attest_on = 0;
    memset(ctx->attest_fp, 0, sizeof(ctx->attest_fp));
    g_ctx = ctx;

    const char *en = getenv("VLLM_ATTEST");
    if (!en || !en[0] || en[0] == '0') return -1;

    const char *dir = getenv("VLLM_ATTEST_DIR");
    if (dir && dir[0]) {
        snprintf(ctx->attest_dir, sizeof(ctx->attest_dir), "%s", dir);
    } else {
        snprintf(ctx->attest_dir, sizeof(ctx->attest_dir), ".");
    }
    struct stat st;
    if (stat(ctx->attest_dir, &st) != 0 || !S_ISDIR(st.st_mode)) {
        fprintf(stderr, "[ATTEST] dir not found: %s (attestation disabled)\n",
                ctx->attest_dir);
        return -1;
    }

    if (vatt_keyload() != 0 && vatt_keygen() != 0) {
        fprintf(stderr, "[ATTEST] key unavailable (load+gen failed); disabled\n");
        return -1;
    }
    ctx->attest_on = 1;
    if (getenv("VLLM_ATTEST_LOG")) {
        fprintf(stderr, "[ATTEST] enabled dir=%s pub=%s\n",
                ctx->attest_dir, g_pub_hex);
    }
    /* 自动化自检门：VLLM_ATTEST_SELFTEST=1 */
    if (getenv("VLLM_ATTEST_SELFTEST")) {
        int ok = vatt_selftest();
        fprintf(stderr, "[ATTEST] selftest       : %s\n",
                ok ? "PASS" : "FAIL");
    }
    return 0;
}

int vatt_active(void) {
    return g_ctx && g_ctx->attest_on && g_key_ok && g_ctx->attest_fp[0];
}

int vatt_pub_hex(char *out, size_t cap) {
    if (!out || cap < sizeof(g_pub_hex)) return 0;
    if (!g_key_ok) { out[0] = '\0'; return 0; }
    memcpy(out, g_pub_hex, sizeof(g_pub_hex));
    return 1;
}

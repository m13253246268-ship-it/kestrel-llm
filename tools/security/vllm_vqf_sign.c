/* ================================================================
 * vllm_vqf_sign.c - 离线 SM2 签名 / 密钥管理工具（P4，供应链完整性）。
 *
 * 编译（Linux 或 Windows MinGW）：
 *   gcc -O2 -Iinclude -Iinclude/common -Iinclude/model \
 *       src/common/vllm_crypto.c tools/security/vllm_vqf_sign.c \
 *       -o vllm_vqf_sign [-lbcrypt]        # Windows 生成密钥需要 -lbcrypt
 *
 * 用法：
 *   vllm_vqf_sign --genkey                         生成密钥对（stdout 输出）
 *   vllm_vqf_sign --pub <privhex>                 由私钥推导公钥（64 hex）
 *   vllm_vqf_sign --sign <model.vqf> --priv <privhex>  对 VQF 签名 / 重签
 *   vllm_vqf_sign --verify <model.vqf> --pub <pubhex>  离线验签（对照引擎判定）
 *
 * 安全边界（诚实清单）：
 *   - 私钥只在离线环境使用，绝不进入设备；设备只存内嵌/运行时公钥。
 *   - 签名随机数 k 由 CSPRNG 生成：Linux getrandom（vc_secure_rand），
 *     Windows 用 BCryptGenRandom；k 复用会泄露私钥，禁止手工注入。
 *   - 签的是「明文逻辑内容」摘要 D = SM3(header_canonical ‖ dir ‖ plaintext)，
 *     与加密口令解耦：同一签名对明文版与任意口令加密版均成立。
 *   - 对加密文件签名/验签必须提供 VLLM_VQF_KEY（需解密数据区计算明文摘要）。
 *   - 设备信任根是安全上限：若攻击者可替换引擎二进制（连同内嵌公钥），
 *     本方案无法防护（需配合安全启动/固件签名）。
 *
 * 退出码：0 成功；1 失败（--verify 时 1 = 签名无效）。
 * ================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vqf_format.h"      /* VQF 磁盘格式（三方共用，无引擎依赖） */
#include "vllm_crypto.h"     /* SM3 / SM4-CTR / HMAC-SM3 / SM2 */

#ifdef _WIN32
#define _FILE_OFFSET_BITS 64
#include <windows.h>
#include <bcrypt.h>
#endif

#define CHUNK (1u << 20)

/* ---------------- 小工具 ---------------- */

static int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static void hex_encode(const uint8_t *b, size_t n, char *out) {
    static const char *H = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[2 * i]     = H[b[i] >> 4];
        out[2 * i + 1] = H[b[i] & 0xf];
    }
    out[2 * n] = '\0';
}

/* 密码学安全随机 32B：Linux 走 vc_secure_rand（getrandom）；
 * Windows 回退 BCryptGenRandom（本工具跨平台密钥生成的兜底）。
 * 返回 0 成功；-1 失败（无 CSPRNG，拒绝生成）。 */
static int tool_secure_rand32(uint8_t out[32]) {
    if (vc_secure_rand(out) == 0) return 0;
#ifdef _WIN32
    BCRYPT_ALG_HANDLE h = NULL;
    if (BCryptOpenAlgorithmProvider(&h, BCRYPT_RNG_ALGORITHM, NULL, 0) == 0) {
        NTSTATUS st = BCryptGenRandom(h, out, 32, 0);
        BCryptCloseAlgorithmProvider(h, 0);
        return st == 0 ? 0 : -1;
    }
    return -1;
#else
    return -1;
#endif
}

/* 由 CSPRNG 生成合法 SM2 私钥（1 <= d <= n-1），越界重取。 */
static int tool_gen_priv(uint8_t priv[32]) {
    for (int tries = 0; tries < 16; tries++) {
        if (tool_secure_rand32(priv) != 0) return -1;
        if (vc_sm2_pub_from_priv(priv, (uint8_t[64]){0}) == 0) return 0;
    }
    return -1;
}

/* ---------------- 摘要计算（与 vqf.c / vqf_load 严格同口径） ----------------
 * D = SM3( header_canonical ‖ directory ‖ plaintext_data )
 * header_canonical = VQFHeader 中 file_len=0 且 sig 全 0（与 HMAC 一致）；
 * directory = n_tensors × VQFTensor（明文）；data = data_offset..file_len
 * 的明文（含张量间 64B 对齐 padding，不含头/目录区前的对齐 padding）。
 * 加密文件需 VLLM_VQF_KEY 解密；同时（如需重写 tag）流式计算 HMAC(密文)。 */
typedef struct {
    VQFHeader h;
    FILE     *f;
    uint8_t   buf[CHUNK];       /* 读缓冲（密文/明文） */
    vc_sm3_ctx dig;
    vc_hmac_ctx hmac;           /* 仅 ENC 时启用 */
    int enc;
    int want_tag;               /* 1 = 需要重算 HMAC tag（ENC 且头部 flags 将变化） */
    uint8_t   sm4key[16], hmackey[16];
    vc_sm4_ctx sm4;
    uint8_t ctr[16];
} SignCtx;

static int vqf_compute_digest(SignCtx *c, uint8_t digest[32], uint8_t tag[32]) {
    const VQFHeader *h = &c->h;
    size_t dir_off = (sizeof(VQFHeader) + 63) & ~(size_t)63;
    if (h->n_tensors > 100000) return -1;   /* 目录越界兜底 */

    VQFHeader hdr_canon = *h;
    hdr_canon.file_len = 0;
    /* flags 按「最终已签名」口径参与摘要：写侧 vqf.c 是先 flags |= VQF_FLAG_SIGNED 再
     * 算摘要；离线补签工具打开的是未签名文件（flags 无 SIGNED），必须置位后再摘要，
     * 否则验签/引擎加载侧（header 已带 SIGNED）重算的摘要与签名时不一致 →
     * 永远 digest mismatch。2026-09-07 板端真机踩坑后修复。 */
    hdr_canon.flags |= VQF_FLAG_SIGNED;
    memset(&hdr_canon.sig, 0, sizeof(hdr_canon.sig));

    vc_sm3_init(&c->dig);
    vc_sm3_update(&c->dig, &hdr_canon, sizeof(hdr_canon));
    /* 目录 */
    if (fseek(c->f, (long)dir_off, SEEK_SET) != 0) return -1;
    uint8_t dirbuf[64 * 8];
    size_t dirmax = sizeof(dirbuf);
    long dleft = (long)h->n_tensors * (long)sizeof(VQFTensor);
    while (dleft > 0) {
        size_t r = fread(dirbuf, 1, dleft < (long)dirmax ? (size_t)dleft : dirmax, c->f);
        if (!r) return -1;
        vc_sm3_update(&c->dig, dirbuf, r);
        dleft -= (long)r;
    }
    /* 数据区 */
    if (fseek(c->f, (long)h->data_offset, SEEK_SET) != 0) return -1;
    long dlen = (long)(h->file_len - h->data_offset);
    if (dlen < 0) return -1;

    if (c->enc) {
        if (c->want_tag) vc_hmac_init(&c->hmac, c->hmackey, 16);
        vc_sm4_setkey_enc(&c->sm4, c->sm4key);
        memcpy(c->ctr, h->enc_iv, 16);
        while (dlen > 0) {
            size_t k = dlen < (long)CHUNK ? (size_t)dlen : (size_t)CHUNK;
            if (fread(c->buf, 1, k, c->f) != k) return -1;
            if (c->want_tag) vc_hmac_update(&c->hmac, c->buf, k);   /* 密文 */
            vc_sm4_ctr_crypt(&c->sm4, c->ctr, c->buf, c->buf, k);   /* 原位解密 */
            vc_sm3_update(&c->dig, c->buf, k);                      /* 明文 */
            dlen -= (long)k;
        }
        if (c->want_tag) vc_hmac_final(&c->hmac, tag);
    } else {
        while (dlen > 0) {
            size_t k = dlen < (long)CHUNK ? (size_t)dlen : (size_t)CHUNK;
            if (fread(c->buf, 1, k, c->f) != k) return -1;
            vc_sm3_update(&c->dig, c->buf, k);
            dlen -= (long)k;
        }
    }
    vc_sm3_final(&c->dig, digest);
    return 0;
}

/* ---------------- 子命令 ---------------- */

static int cmd_genkey(void) {
    uint8_t priv[32], pub[64];
    char hex[160];
    if (tool_gen_priv(priv) != 0 || vc_sm2_pub_from_priv(priv, pub) != 0) {
        fprintf(stderr, "[vllm_vqf_sign] key generation failed "
                        "(need getrandom on Linux or BCrypt on Windows)\n");
        return 1;
    }
    hex_encode(priv, 32, hex);
    printf("PRIV=%s\n", hex);
    hex_encode(pub, 64, hex);
    printf("PUB=%s\n", hex);
    printf("# 发布流程：把 PUB(128 hex) 填入 vllm_sign_keys.h 的 "
           "VLLM_SM2_PUB_HEX 并重编引擎；\n"
           "# 转换时用 VLLM_VQF_SIGN_PRIV=<PRIV> 签名。私钥妥善离线保管。\n");
    return 0;
}

static int cmd_pub(const char *privhex) {
    uint8_t priv[32], pub[64];
    char hex[160];
    if (vc_hex_decode(privhex, priv, 32) != 32 ||
        vc_sm2_pub_from_priv(priv, pub) != 0) {
        fprintf(stderr, "[vllm_vqf_sign] invalid private key\n");
        return 1;
    }
    hex_encode(pub, 64, hex);
    printf("PUB=%s\n", hex);
    return 0;
}

/* 打开并校验 VQF 文件；返回 0 成功（ctx 就绪）。 */
static int open_vqf(SignCtx *c, const char *path) {
    memset(c, 0, sizeof(*c));
    c->f = fopen(path, "rb+");
    if (!c->f) { fprintf(stderr, "[vllm_vqf_sign] cannot open %s\n", path); return -1; }
    if (fread(&c->h, 1, sizeof(c->h), c->f) != sizeof(c->h)) {
        fprintf(stderr, "[vllm_vqf_sign] short header\n"); return -1;
    }
    if (c->h.magic != VQF_MAGIC) { fprintf(stderr, "[vllm_vqf_sign] not a VQF file\n"); return -1; }
    if (c->h.version != VQF_VERSION) {
        fprintf(stderr, "[vllm_vqf_sign] version %u unsupported\n", c->h.version); return -1;
    }
    c->enc = (c->h.flags & VQF_FLAG_ENC) ? 1 : 0;
    return 0;
}

static int cmd_sign(const char *path, const uint8_t priv[32]) {
    SignCtx c;
    if (open_vqf(&c, path) != 0) return 1;

    /* 加密文件必须提供口令以解密数据区计算明文摘要 */
    if (c.enc) {
        const char *pass = getenv("VLLM_VQF_KEY");
        uint8_t mat[32];
        if (!pass || !pass[0]) {
            fprintf(stderr, "[vllm_vqf_sign] encrypted VQF requires VLLM_VQF_KEY\n");
            fclose(c.f); return 1;
        }
        vc_sm3(pass, strlen(pass), mat);
        memcpy(c.sm4key, mat, 16);
        memcpy(c.hmackey, mat + 16, 16);
    }
    /* 头部 flags 将变化（unsigned→signed）时，ENC 文件需重写 HMAC tag */
    c.want_tag = c.enc && !(c.h.flags & VQF_FLAG_SIGNED);

    /* 单遍摘要：header_canonical + 目录 + 明文数据 */
    uint8_t digest[32], tag[32];
    if (vqf_compute_digest(&c, digest, tag) != 0) {
        fprintf(stderr, "[vllm_vqf_sign] digest computation failed\n");
        fclose(c.f); return 1;
    }

    /* 生成安全随机 k，SM2 签名（Z_A 绑定 ID=VQF_SM2_ID） */
    uint8_t k[32], r[32], s[32], pub[64];
    if (tool_secure_rand32(k) != 0) {
        fprintf(stderr, "[vllm_vqf_sign] secure random k failed\n");
        fclose(c.f); return 1;
    }
    if (vc_sm2_pub_from_priv(priv, pub) != 0) {
        fprintf(stderr, "[vllm_vqf_sign] invalid private key\n");
        fclose(c.f); return 1;
    }
    size_t idlen = strlen(VQF_SM2_ID);
    if (vc_sm2_sign(priv, digest, 32, (const uint8_t *)VQF_SM2_ID, idlen,
                    k, r, s) != 0) {
        fprintf(stderr, "[vllm_vqf_sign] SM2 sign failed\n");
        fclose(c.f); return 1;
    }

    /* 填充 sig 块 + SIGNED flag，回写 header */
    VQFHeader h = c.h;
    h.flags |= VQF_FLAG_SIGNED;
    memcpy(h.sig.pub, pub, 64);
    memcpy(h.sig.r, r, 32);
    memcpy(h.sig.s, s, 32);
    memcpy(h.sig.digest, digest, 32);
    memset(h.sig.id, 0, sizeof(h.sig.id));
    memcpy(h.sig.id, VQF_SM2_ID, idlen);
    h.sig.id_len = (uint32_t)idlen;
    h.sig.rsvd = 0;

    if (fseek(c.f, 0, SEEK_SET) != 0 || fwrite(&h, 1, sizeof(h), c.f) != sizeof(h)) {
        fprintf(stderr, "[vllm_vqf_sign] header write failed\n");
        fclose(c.f); return 1;
    }
    /* ENC 且头部 flags 变化：重写 HMAC tag（覆盖 头部+目录+密文） */
    if (c.want_tag) {
        if (fseek(c.f, 0, SEEK_END) != 0 || fwrite(tag, 1, 32, c.f) != 32) {
            fprintf(stderr, "[vllm_vqf_sign] tag write failed\n");
            fclose(c.f); return 1;
        }
    }
    fclose(c.f);

    char hex[160];
    hex_encode(pub, 64, hex);
    fprintf(stderr, "[vllm_vqf_sign] signed %s (flags=0x%x, pub=%s)\n",
            path, h.flags, hex);
    return 0;
}

static int cmd_verify(const char *path, const uint8_t pub[64]) {
    SignCtx c;
    if (open_vqf(&c, path) != 0) return 1;
    if (!(c.h.flags & VQF_FLAG_SIGNED)) {
        fprintf(stderr, "[vllm_vqf_sign] %s is NOT signed\n", path);
        fclose(c.f); return 1;
    }
    if (c.enc) {
        const char *pass = getenv("VLLM_VQF_KEY");
        uint8_t mat[32];
        if (!pass || !pass[0]) {
            fprintf(stderr, "[vllm_vqf_sign] encrypted VQF requires VLLM_VQF_KEY\n");
            fclose(c.f); return 1;
        }
        vc_sm3(pass, strlen(pass), mat);
        memcpy(c.sm4key, mat, 16);
        memcpy(c.hmackey, mat + 16, 16);
    }
    uint8_t digest[32], tag[32];
    if (vqf_compute_digest(&c, digest, tag) != 0) {
        fprintf(stderr, "[vllm_vqf_sign] digest computation failed\n");
        fclose(c.f); return 1;
    }
    fclose(c.f);

    if (memcmp(digest, c.h.sig.digest, 32) != 0) {
        fprintf(stderr, "[vllm_vqf_sign] VERIFY FAIL: digest mismatch (tampered)\n");
        return 1;
    }
    size_t idlen = strlen(VQF_SM2_ID);
    if (!vc_sm2_verify(pub, digest, 32, (const uint8_t *)VQF_SM2_ID, idlen,
                       c.h.sig.r, c.h.sig.s)) {
        fprintf(stderr, "[vllm_vqf_sign] VERIFY FAIL: SM2 signature invalid\n");
        return 1;
    }
    if (memcmp(c.h.sig.pub, pub, 64) != 0) {
        fprintf(stderr, "[vllm_vqf_sign] VERIFY FAIL: sig.pub != trusted pub\n");
        return 1;
    }
    fprintf(stderr, "[vllm_vqf_sign] VERIFY OK\n");
    return 0;
}

static void usage(void) {
    printf(
        "vllm_vqf_sign - VQF SM2 离线签名/密钥工具\n"
        "用法:\n"
        "  vllm_vqf_sign --genkey\n"
        "  vllm_vqf_sign --pub <privhex>\n"
        "  vllm_vqf_sign --sign <model.vqf> --priv <privhex>\n"
        "  vllm_vqf_sign --verify <model.vqf> --pub <pubhex>\n"
        "加密文件需要环境变量 VLLM_VQF_KEY（解密数据区计算明文摘要）。\n"
        "退出码: 0 成功, 1 失败（verify 失败=签名无效）。\n");
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }
    if (strcmp(argv[1], "--genkey") == 0) return cmd_genkey();
    if (strcmp(argv[1], "--pub") == 0 && argc >= 3) return cmd_pub(argv[2]);
    const char *file = NULL, *privhex = NULL, *pubhex = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sign") == 0 && i + 1 < argc) file = argv[++i];
        else if (strcmp(argv[i], "--verify") == 0 && i + 1 < argc) file = argv[++i];
        else if (strcmp(argv[i], "--priv") == 0 && i + 1 < argc) privhex = argv[++i];
        else if (strcmp(argv[i], "--pub") == 0 && i + 1 < argc) pubhex = argv[++i];
    }
    if (privhex) {
        uint8_t priv[32];
        if (vc_hex_decode(privhex, priv, 32) != 32) {
            fprintf(stderr, "[vllm_vqf_sign] invalid --priv hex\n"); return 1;
        }
        if (!file) { usage(); return 1; }
        return cmd_sign(file, priv);
    }
    if (pubhex) {
        uint8_t pub[64];
        if (vc_hex_decode(pubhex, pub, 64) != 64) {
            fprintf(stderr, "[vllm_vqf_sign] invalid --pub hex\n"); return 1;
        }
        if (!file) { usage(); return 1; }
        return cmd_verify(file, pub);
    }
    usage();
    return 1;
}

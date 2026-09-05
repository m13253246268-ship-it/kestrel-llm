/* ================================================================
 * vllm_crypto.h - 自研零依赖国密密码学原语（SM3 / SM4 / SM2）。
 *
 * 用途（VQF-Enc 公理 at_rest_encryption_load_tradeoff）：
 *   - 权重文件存储加密：SM4-CTR 逐张量加密（密文长度==明文长度，
 *     流式即可，mmap/直挂布局不变，仅多一次加载期解密）。
 *   - 完整性/认证：SM3-HMAC tag（防篡改）+ 目录区 dir_tag。
 *   - 密钥派生：SM3-KDF（GM/T 0003.4 计数器 KDF）。
 *   - 公钥签名/验签：SM2 椭圆曲线（供应链完整性，防模型被替换）。
 *
 * 标准依据：
 *   GB/T 32905-2016 (SM3)   GB/T 32907-2016 (SM4)
 *   GM/T 0003-2012  (SM2)   GM/T 0004-2012  (SM3, 早期行标)
 *
 * 实现：纯 C11 零外部依赖、无内联汇编（aarch64/x86 均可编译）。
 * 位级正确性由 vllm_crypto_test 的国密标准 KAT 向量逐字节验证（red line）。
 * ================================================================ */
#ifndef VLLM_CRYPTO_H
#define VLLM_CRYPTO_H

#include <stddef.h>
#include <stdint.h>

/* ---------------- SM3 密码杂凑 (GB/T 32905-2016) ---------------- */
typedef struct {
    uint32_t s[8];
    uint64_t n;          /* 已处理字节数 */
    uint8_t  buf[64];
    size_t   buflen;
} vc_sm3_ctx;

void vc_sm3_init(vc_sm3_ctx *c);
void vc_sm3_update(vc_sm3_ctx *c, const void *p, size_t n);
void vc_sm3_final(vc_sm3_ctx *c, uint8_t out[32]);
void vc_sm3(const void *p, size_t n, uint8_t out[32]);

/* HMAC-SM3（GB/T 15852.2 采用专用杂凑函数的 MAC / RFC 2104）。
 * 流式接口用于大消息（VQF-Enc 数据区可达数 GB），边流边算，避免整块缓冲。 */
typedef struct {
    vc_sm3_ctx ctx;
    uint8_t     opad[64];
} vc_hmac_ctx;

void vc_hmac_init(vc_hmac_ctx *c, const uint8_t *key, size_t klen);
void vc_hmac_update(vc_hmac_ctx *c, const void *p, size_t n);
void vc_hmac_final(vc_hmac_ctx *c, uint8_t out[32]);

/* 一次性 HMAC-SM3（小消息便利接口） */
void vc_hmac_sm3(const uint8_t *key, size_t klen,
                 const uint8_t *msg, size_t mlen, uint8_t out[32]);

/* SM3-KDF：GM/T 0003.4 计数器密钥派生。由共享秘密 z 派生 klen 字节。 */
void vc_sm3_kdf(const uint8_t *z, size_t zlen, size_t klen, uint8_t *out);

/* ---------------- SM4 分组密码 (GB/T 32907-2016) ---------------- */
typedef struct { uint32_t rk[32]; } vc_sm4_ctx;

void vc_sm4_setkey_enc(vc_sm4_ctx *c, const uint8_t key[16]);
void vc_sm4_setkey_dec(vc_sm4_ctx *c, const uint8_t key[16]);
void vc_sm4_crypt_block(const vc_sm4_ctx *c, const uint8_t in[16], uint8_t out[16]);

/* SM4-CTR 流式加解密（加解密同函数；密文长度==明文长度）。
 * counter 为 16 字节大端计数器，函数内部递增。 */
void vc_sm4_ctr_crypt(const vc_sm4_ctx *c, uint8_t counter[16],
                      const uint8_t *in, uint8_t *out, size_t len);

/* ---------------- SM2 椭圆曲线公钥密码 (GM/T 0003-2012) ---------------- */

/* 由私钥推导公钥：pub = [priv]G，输出 64 字节 (x||y)，各 32 字节大端。
 * 返回 0 成功；-1 私钥非法（0 或 >= n）。 */
int vc_sm2_pub_from_priv(const uint8_t priv[32], uint8_t pub[64]);

/* SM2 数字签名。k 为随机数（32 字节，1 <= k <= n-1），调用方负责生成；
 * 为便于 KAT 复现，测试可传入标准固定 k。
 * id 为用户标识（可辨别标识 ID），签名验签双方必须一致。
 * 返回 0 成功；r/s 各 32 字节大端。 */
int vc_sm2_sign(const uint8_t priv[32], const uint8_t *msg, size_t msglen,
                const uint8_t *id, size_t idlen,
                const uint8_t k[32], uint8_t r[32], uint8_t s[32]);

/* SM2 验签。返回 1=有效，0=无效（含 r/s 越界）。 */
int vc_sm2_verify(const uint8_t pub[64], const uint8_t *msg, size_t msglen,
                  const uint8_t *id, size_t idlen,
                  const uint8_t r[32], const uint8_t s[32]);

/* ---------------- 安全随机数 / hex 解码 ---------------- */

/* 密码学安全随机数（32B）。SM2 签名的随机数 k 必须由此生成（不可复用、
 * 不可预测；k 复用会泄露私钥）。Linux 用 getrandom(2)，失败回退
 * /dev/urandom；非 Linux 平台返回 -1。 */
int vc_secure_rand(uint8_t out[32]);

/* hex 解码：输入须为 2*outlen 个十六进制字符（可含尾部 '\0'），输出
 * outlen 字节。返回 outlen 成功；-1 长度不足或含非法字符。 */
int vc_hex_decode(const char *hex, uint8_t *out, size_t outlen);

#endif /* VLLM_CRYPTO_H */

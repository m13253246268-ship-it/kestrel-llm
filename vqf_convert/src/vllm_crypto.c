/* ================================================================
 * vllm_crypto.c - 国密密码学原语自研实现（SM3 / SM4 / SM2）。
 *
 * 纯 C11 零外部依赖、无内联汇编。位级正确性由 vllm_crypto_test
 * 用国密标准 KAT 向量逐字节验证（red line）。
 * ================================================================ */
#include "vllm_crypto.h"
#include <string.h>
#include <stdio.h>
#ifdef __linux__
#include <sys/random.h>
#endif
#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>       /* BCryptGenRandom（CSPRNG）；构建需链接 -lbcrypt */
#endif

/* ================================================================
 * 通用小工具
 * ================================================================ */
static inline uint32_t rotl32(uint32_t x, int n) {
    n &= 31;
    return (x << n) | (x >> (32 - n));
}

#define GET_U32_BE(b) ( \
    ((uint32_t)(b)[0] << 24) | ((uint32_t)(b)[1] << 16) | \
    ((uint32_t)(b)[2] << 8)  |  (uint32_t)(b)[3])

#define PUT_U32_BE(v, b) do { \
    (b)[0] = (uint8_t)((v) >> 24); (b)[1] = (uint8_t)((v) >> 16); \
    (b)[2] = (uint8_t)((v) >> 8);  (b)[3] = (uint8_t)(v); } while (0)

/* ================================================================
 * SM3 密码杂凑 (GB/T 32905-2016)
 * ================================================================ */
static const uint32_t SM3_IV[8] = {
    0x7380166f, 0x4914b2b9, 0x172442d7, 0xda8a0600,
    0xa96f30bc, 0x163138aa, 0xe38dee4d, 0xb0fb0e4e
};

static inline uint32_t sm3_p0(uint32_t x) {
    return x ^ rotl32(x, 9) ^ rotl32(x, 17);
}
static inline uint32_t sm3_p1(uint32_t x) {
    return x ^ rotl32(x, 15) ^ rotl32(x, 23);
}

static void sm3_compress(uint32_t s[8], const uint8_t block[64]) {
    uint32_t w[68], wp[64];
    for (int i = 0; i < 16; i++)
        w[i] = GET_U32_BE(block + 4 * i);
    for (int j = 16; j < 68; j++)
        w[j] = sm3_p1(w[j - 16] ^ w[j - 9] ^ rotl32(w[j - 3], 15)) ^
               rotl32(w[j - 13], 7) ^ w[j - 6];
    for (int j = 0; j < 64; j++)
        wp[j] = w[j] ^ w[j + 4];

    uint32_t a = s[0], b = s[1], c = s[2], d = s[3];
    uint32_t e = s[4], f = s[5], g = s[6], h = s[7];

    for (int j = 0; j < 64; j++) {
        uint32_t tj = (j < 16) ? 0x79cc4519u : 0x7a879d8au;
        uint32_t ss1 = rotl32(rotl32(a, 12) + e + rotl32(tj, j), 7);
        uint32_t ss2 = ss1 ^ rotl32(a, 12);
        uint32_t ff, gg;
        if (j < 16) {
            ff = a ^ b ^ c;
            gg = e ^ f ^ g;
        } else {
            ff = (a & b) | (a & c) | (b & c);
            gg = (e & f) | ((~e) & g);
        }
        uint32_t tt1 = ff + d + ss2 + wp[j];
        uint32_t tt2 = gg + h + ss1 + w[j];
        d = c; c = rotl32(b, 9); b = a; a = tt1;
        h = g; g = rotl32(f, 19); f = e; e = sm3_p0(tt2);
    }

    s[0] ^= a; s[1] ^= b; s[2] ^= c; s[3] ^= d;
    s[4] ^= e; s[5] ^= f; s[6] ^= g; s[7] ^= h;
}

void vc_sm3_init(vc_sm3_ctx *c) {
    for (int i = 0; i < 8; i++) c->s[i] = SM3_IV[i];
    c->n = 0;
    c->buflen = 0;
}

void vc_sm3_update(vc_sm3_ctx *c, const void *p, size_t n) {
    const uint8_t *in = (const uint8_t *)p;
    c->n += n;
    while (n > 0) {
        if (c->buflen == 0 && n >= 64) {
            sm3_compress(c->s, in);
            in += 64; n -= 64;
            continue;
        }
        size_t take = 64 - c->buflen;
        if (take > n) take = n;
        memcpy(c->buf + c->buflen, in, take);
        c->buflen += take;
        in += take; n -= take;
        if (c->buflen == 64) {
            sm3_compress(c->s, c->buf);
            c->buflen = 0;
        }
    }
}

void vc_sm3_final(vc_sm3_ctx *c, uint8_t out[32]) {
    uint64_t bits = c->n * 8;
    uint8_t pad = 0x80;
    vc_sm3_update(c, &pad, 1);
    uint8_t zero = 0;
    while (c->buflen != 56)
        vc_sm3_update(c, &zero, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; i++)
        len[i] = (uint8_t)(bits >> (56 - 8 * i));
    vc_sm3_update(c, len, 8);
    for (int i = 0; i < 8; i++)
        PUT_U32_BE(c->s[i], out + 4 * i);
}

void vc_sm3(const void *p, size_t n, uint8_t out[32]) {
    vc_sm3_ctx c;
    vc_sm3_init(&c);
    vc_sm3_update(&c, p, n);
    vc_sm3_final(&c, out);
}

void vc_hmac_init(vc_hmac_ctx *c, const uint8_t *key, size_t klen) {
    uint8_t k[64];
    if (klen > 64) {
        vc_sm3(key, klen, k);
        memset(k + 32, 0, 32);
    } else {
        memcpy(k, key, klen);
        memset(k + klen, 0, 64 - klen);
    }
    uint8_t ipad[64];
    for (int i = 0; i < 64; i++) {
        ipad[i] = k[i] ^ 0x36;
        c->opad[i] = k[i] ^ 0x5c;
    }
    vc_sm3_init(&c->ctx);
    vc_sm3_update(&c->ctx, ipad, 64);
}

void vc_hmac_update(vc_hmac_ctx *c, const void *p, size_t n) {
    vc_sm3_update(&c->ctx, p, n);
}

void vc_hmac_final(vc_hmac_ctx *c, uint8_t out[32]) {
    uint8_t inner[32];
    vc_sm3_final(&c->ctx, inner);
    vc_sm3_ctx c2;
    vc_sm3_init(&c2);
    vc_sm3_update(&c2, c->opad, 64);
    vc_sm3_update(&c2, inner, 32);
    vc_sm3_final(&c2, out);
}

void vc_hmac_sm3(const uint8_t *key, size_t klen,
                 const uint8_t *msg, size_t mlen, uint8_t out[32]) {
    vc_hmac_ctx c;
    vc_hmac_init(&c, key, klen);
    vc_hmac_update(&c, msg, mlen);
    vc_hmac_final(&c, out);
}

/* GM/T 0003.4 计数器 KDF */
void vc_sm3_kdf(const uint8_t *z, size_t zlen, size_t klen, uint8_t *out) {
    uint32_t ct = 1;
    size_t off = 0;
    while (off < klen) {
        vc_sm3_ctx ctx;
        vc_sm3_init(&ctx);
        vc_sm3_update(&ctx, z, zlen);
        uint8_t c[4];
        PUT_U32_BE(ct, c);
        vc_sm3_update(&ctx, c, 4);
        uint8_t h[32];
        vc_sm3_final(&ctx, h);
        size_t n = klen - off;
        if (n > 32) n = 32;
        memcpy(out + off, h, n);
        off += n;
        ct++;
    }
}

/* ================================================================
 * SM4 分组密码 (GB/T 32907-2016)
 * ================================================================ */
static const uint8_t SM4_SBOX[256] = {
    0xd6,0x90,0xe9,0xfe,0xcc,0xe1,0x3d,0xb7,0x16,0xb6,0x14,0xc2,0x28,0xfb,0x2c,0x05,
    0x2b,0x67,0x9a,0x76,0x2a,0xbe,0x04,0xc3,0xaa,0x44,0x13,0x26,0x49,0x86,0x06,0x99,
    0x9c,0x42,0x50,0xf4,0x91,0xef,0x98,0x7a,0x33,0x54,0x0b,0x43,0xed,0xcf,0xac,0x62,
    0xe4,0xb3,0x1c,0xa9,0xc9,0x08,0xe8,0x95,0x80,0xdf,0x94,0xfa,0x75,0x8f,0x3f,0xa6,
    0x47,0x07,0xa7,0xfc,0xf3,0x73,0x17,0xba,0x83,0x59,0x3c,0x19,0xe6,0x85,0x4f,0xa8,
    0x68,0x6b,0x81,0xb2,0x71,0x64,0xda,0x8b,0xf8,0xeb,0x0f,0x4b,0x70,0x56,0x9d,0x35,
    0x1e,0x24,0x0e,0x5e,0x63,0x58,0xd1,0xa2,0x25,0x22,0x7c,0x3b,0x01,0x21,0x78,0x87,
    0xd4,0x00,0x46,0x57,0x9f,0xd3,0x27,0x52,0x4c,0x36,0x02,0xe7,0xa0,0xc4,0xc8,0x9e,
    0xea,0xbf,0x8a,0xd2,0x40,0xc7,0x38,0xb5,0xa3,0xf7,0xf2,0xce,0xf9,0x61,0x15,0xa1,
    0xe0,0xae,0x5d,0xa4,0x9b,0x34,0x1a,0x55,0xad,0x93,0x32,0x30,0xf5,0x8c,0xb1,0xe3,
    0x1d,0xf6,0xe2,0x2e,0x82,0x66,0xca,0x60,0xc0,0x29,0x23,0xab,0x0d,0x53,0x4e,0x6f,
    0xd5,0xdb,0x37,0x45,0xde,0xfd,0x8e,0x2f,0x03,0xff,0x6a,0x72,0x6d,0x6c,0x5b,0x51,
    0x8d,0x1b,0xaf,0x92,0xbb,0xdd,0xbc,0x7f,0x11,0xd9,0x5c,0x41,0x1f,0x10,0x5a,0xd8,
    0x0a,0xc1,0x31,0x88,0xa5,0xcd,0x7b,0xbd,0x2d,0x74,0xd0,0x12,0xb8,0xe5,0xb4,0xb0,
    0x89,0x69,0x97,0x4a,0x0c,0x96,0x77,0x7e,0x65,0xb9,0xf1,0x09,0xc5,0x6e,0xc6,0x84,
    0x18,0xf0,0x7d,0xec,0x3a,0xdc,0x4d,0x20,0x79,0xee,0x5f,0x3e,0xd7,0xcb,0x39,0x48
};

static const uint32_t SM4_FK[4] = {
    0xa3b1bac6, 0x56aa3350, 0x677d9197, 0xb27022dc
};

/* CK_i = ((4i+j)*7 mod 256)，大端拼成 32 位 */
static uint32_t sm4_ck(int i) {
    uint32_t c = 0;
    for (int j = 0; j < 4; j++)
        c = (c << 8) | ((uint32_t)((4 * i + j) * 7) & 0xff);
    return c;
}

static uint32_t sm4_tau(uint32_t x) {
    return ((uint32_t)SM4_SBOX[(x >> 24) & 0xff] << 24) |
           ((uint32_t)SM4_SBOX[(x >> 16) & 0xff] << 16) |
           ((uint32_t)SM4_SBOX[(x >> 8) & 0xff] << 8)  |
           ((uint32_t)SM4_SBOX[x & 0xff]);
}

static uint32_t sm4_l(uint32_t b) {
    return b ^ rotl32(b, 2) ^ rotl32(b, 10) ^ rotl32(b, 18) ^ rotl32(b, 24);
}
static uint32_t sm4_lp(uint32_t b) {
    return b ^ rotl32(b, 13) ^ rotl32(b, 23);
}
static uint32_t sm4_t(uint32_t x) { return sm4_l(sm4_tau(x)); }
static uint32_t sm4_tp(uint32_t x) { return sm4_lp(sm4_tau(x)); }

static void sm4_key_expand(uint32_t rk[32], const uint8_t key[16]) {
    uint32_t mk[4], k[36];
    for (int i = 0; i < 4; i++)
        mk[i] = GET_U32_BE(key + 4 * i);
    for (int i = 0; i < 4; i++)
        k[i] = mk[i] ^ SM4_FK[i];
    for (int i = 0; i < 32; i++) {
        k[i + 4] = k[i] ^ sm4_tp(k[i + 1] ^ k[i + 2] ^ k[i + 3] ^ sm4_ck(i));
        rk[i] = k[i + 4];
    }
}

void vc_sm4_setkey_enc(vc_sm4_ctx *c, const uint8_t key[16]) {
    sm4_key_expand(c->rk, key);
}
void vc_sm4_setkey_dec(vc_sm4_ctx *c, const uint8_t key[16]) {
    uint32_t rk[32];
    sm4_key_expand(rk, key);
    for (int i = 0; i < 32; i++)
        c->rk[i] = rk[31 - i];
}

void vc_sm4_crypt_block(const vc_sm4_ctx *c, const uint8_t in[16], uint8_t out[16]) {
    uint32_t x[36];
    for (int i = 0; i < 4; i++)
        x[i] = GET_U32_BE(in + 4 * i);
    for (int i = 0; i < 32; i++)
        x[i + 4] = x[i] ^ sm4_t(x[i + 1] ^ x[i + 2] ^ x[i + 3] ^ c->rk[i]);
    for (int i = 0; i < 4; i++)
        PUT_U32_BE(x[35 - i], out + 4 * i);
}

void vc_sm4_ctr_crypt(const vc_sm4_ctx *c, uint8_t counter[16],
                      const uint8_t *in, uint8_t *out, size_t len) {
    uint8_t ks[16];
    while (len >= 16) {
        vc_sm4_crypt_block(c, counter, ks);
        for (int i = 0; i < 16; i++) out[i] = in[i] ^ ks[i];
        for (int i = 15; i >= 0; i--) { counter[i]++; if (counter[i]) break; }
        in += 16; out += 16; len -= 16;
    }
    if (len) {
        vc_sm4_crypt_block(c, counter, ks);
        for (size_t i = 0; i < len; i++) out[i] = in[i] ^ ks[i];
    }
}

/* ================================================================
 * SM2 椭圆曲线公钥密码 (GM/T 0003-2012)
 *   —— 256 位大数（uint32[8]，小端）+ 素域 F_p 上椭圆曲线
 * ================================================================ */
typedef uint32_t bn_t[8];

/* 曲线参数（小端 uint32） */
static const bn_t SM2_P = { 0xffffffff,0xffffffff,0x00000000,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xfffffffe };
static const bn_t SM2_A = { 0xfffffffc,0xffffffff,0x00000000,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xfffffffe };
static const bn_t SM2_B = { 0x4d940e93,0xddbcbd41,0x15ab8f92,0xf39789f5,0xcf6509a7,0x4d5a9e4b,0x9d9f5e34,0x28e9fa9e };
static const bn_t SM2_N = { 0x39d54123,0x53bbf409,0x21c6052b,0x7203df6b,0xffffffff,0xffffffff,0xffffffff,0xfffffffe };
static const bn_t SM2_GX = { 0x334c74c7,0x715a4589,0xf2660be1,0x8fe30bbf,0x6a39c994,0x5f990446,0x1f198119,0x32c4ae2c };
static const bn_t SM2_GY = { 0x2139f0a0,0x02df32e5,0xc62a4740,0xd0a9877c,0x6b692153,0x59bdcee3,0xf4f6779c,0xbc3736a2 };

static void bn_set_zero(bn_t r) { memset(r, 0, sizeof(bn_t)); }
static void bn_set_one(bn_t r) { memset(r, 0, sizeof(bn_t)); r[0] = 1; }

static int bn_is_zero(const bn_t a) {
    for (int i = 0; i < 8; i++) if (a[i]) return 0;
    return 1;
}
static int bn_is_one(const bn_t a) {
    if (a[0] != 1) return 0;
    for (int i = 1; i < 8; i++) if (a[i]) return 0;
    return 1;
}
static int bn_cmp(const bn_t a, const bn_t b) {
    for (int i = 7; i >= 0; i--) {
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    }
    return 0;
}
static int bn_eq(const bn_t a, const bn_t b) { return bn_cmp(a, b) == 0; }

/* r = (a + b) mod m，a,b < m */
static void bn_add_mod(bn_t r, const bn_t a, const bn_t b, const bn_t m) {
    uint32_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t s = (uint64_t)a[i] + b[i] + carry;
        r[i] = (uint32_t)s;
        carry = (uint32_t)(s >> 32);
    }
    if (carry || bn_cmp(r, m) >= 0) {
        uint32_t borrow = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t d = (uint64_t)r[i] - m[i] - borrow;
            r[i] = (uint32_t)d;
            borrow = (uint32_t)(d >> 63);
        }
    }
}

/* r = (a - b) mod m，a,b < m */
static void bn_sub_mod(bn_t r, const bn_t a, const bn_t b, const bn_t m) {
    if (bn_cmp(a, b) >= 0) {
        uint32_t borrow = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t d = (uint64_t)a[i] - b[i] - borrow;
            r[i] = (uint32_t)d;
            borrow = (uint32_t)(d >> 63);
        }
    } else {
        bn_t t;
        uint32_t borrow = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t d = (uint64_t)b[i] - a[i] - borrow;
            t[i] = (uint32_t)d;
            borrow = (uint32_t)(d >> 63);
        }
        borrow = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t d = (uint64_t)m[i] - t[i] - borrow;
            r[i] = (uint32_t)d;
            borrow = (uint32_t)(d >> 63);
        }
    }
}

/* t[16] = a * b（512 位，小端） */
static void bn_mul(uint32_t t[16], const bn_t a, const bn_t b) {
    memset(t, 0, 16 * sizeof(uint32_t));
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t cur = (uint64_t)t[i + j] + (uint64_t)a[i] * b[j] + carry;
            t[i + j] = (uint32_t)cur;
            carry = cur >> 32;
        }
        t[i + 8] = (uint32_t)carry;
    }
}

/* r = t mod m（t 为 512 位，二进制长除法） */
static void bn_mod(bn_t r, const uint32_t t[16], const bn_t m) {
    uint32_t acc[9] = {0};
    for (int bit = 511; bit >= 0; bit--) {
        for (int i = 8; i >= 1; i--)
            acc[i] = (acc[i] << 1) | (acc[i - 1] >> 31);
        acc[0] <<= 1;
        acc[0] |= (t[bit >> 5] >> (bit & 31)) & 1;
        if (acc[8] || bn_cmp(acc, m) >= 0) {
            uint32_t borrow = 0;
            for (int i = 0; i < 8; i++) {
                uint64_t d = (uint64_t)acc[i] - m[i] - borrow;
                acc[i] = (uint32_t)d;
                borrow = (uint32_t)(d >> 63);
            }
            acc[8] = 0;
        }
    }
    for (int i = 0; i < 8; i++) r[i] = acc[i];
}

static void bn_mod_mul(bn_t r, const bn_t a, const bn_t b, const bn_t m) {
    uint32_t t[16];
    bn_mul(t, a, b);
    bn_mod(r, t, m);
}

/* 右移 1 位 */
static void bn_shr1(bn_t x) {
    for (int i = 0; i < 7; i++)
        x[i] = (x[i] >> 1) | (x[i + 1] << 31);
    x[7] >>= 1;
}
/* x = (x + m) >> 1，x < m 且 x、m 均为奇数 */
static void bn_add_m_shr1(bn_t x, const bn_t m) {
    uint32_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t s = (uint64_t)x[i] + m[i] + carry;
        x[i] = (uint32_t)s;
        carry = (uint32_t)(s >> 32);
    }
    for (int i = 0; i < 7; i++)
        x[i] = (x[i] >> 1) | (x[i + 1] << 31);
    x[7] = (x[7] >> 1) | (carry << 31);
}

/* r = a^(-1) mod m（m 为奇数，二进制扩展欧几里得） */
static void bn_mod_inv(bn_t r, const bn_t a, const bn_t m) {
    bn_t u, v, x1, x2;
    memcpy(u, a, sizeof(bn_t));
    memcpy(v, m, sizeof(bn_t));
    bn_set_one(x1);
    bn_set_zero(x2);

    while (!bn_is_one(u) && !bn_is_one(v)) {
        while (!(u[0] & 1)) {
            bn_shr1(u);
            if (x1[0] & 1) bn_add_m_shr1(x1, m);
            else bn_shr1(x1);
        }
        while (!(v[0] & 1)) {
            bn_shr1(v);
            if (x2[0] & 1) bn_add_m_shr1(x2, m);
            else bn_shr1(x2);
        }
        if (bn_cmp(u, v) >= 0) {
            uint32_t borrow = 0;
            for (int i = 0; i < 8; i++) {
                uint64_t d = (uint64_t)u[i] - v[i] - borrow;
                u[i] = (uint32_t)d;
                borrow = (uint32_t)(d >> 63);
            }
            bn_sub_mod(x1, x1, x2, m);
        } else {
            uint32_t borrow = 0;
            for (int i = 0; i < 8; i++) {
                uint64_t d = (uint64_t)v[i] - u[i] - borrow;
                v[i] = (uint32_t)d;
                borrow = (uint32_t)(d >> 63);
            }
            bn_sub_mod(x2, x2, x1, m);
        }
    }
    if (bn_is_one(u)) memcpy(r, x1, sizeof(bn_t));
    else memcpy(r, x2, sizeof(bn_t));
}

/* 字节转换：bn_t（小端 uint32） <-> 32 字节大端 */
static void bytes_to_bn(bn_t r, const uint8_t b[32]) {
    for (int i = 0; i < 8; i++)
        r[i] = ((uint32_t)b[28 - 4 * i] << 24) | ((uint32_t)b[29 - 4 * i] << 16) |
               ((uint32_t)b[30 - 4 * i] << 8)  |  (uint32_t)b[31 - 4 * i];
}
static void bn_to_bytes(uint8_t b[32], const bn_t a) {
    for (int i = 0; i < 8; i++) {
        b[28 - 4 * i] = (uint8_t)(a[i] >> 24);
        b[29 - 4 * i] = (uint8_t)(a[i] >> 16);
        b[30 - 4 * i] = (uint8_t)(a[i] >> 8);
        b[31 - 4 * i] = (uint8_t)a[i];
    }
}

/* 椭圆曲线点，仿射坐标；无穷远点用 x==0 && y==0 表示（SM2 曲线 b!=0，无 (0,0)） */
typedef struct { bn_t x, y; } ec_point;
static const ec_point SM2_G = {
    { 0x334c74c7,0x715a4589,0xf2660be1,0x8fe30bbf,0x6a39c994,0x5f990446,0x1f198119,0x32c4ae2c },
    { 0x2139f0a0,0x02df32e5,0xc62a4740,0xd0a9877c,0x6b692153,0x59bdcee3,0xf4f6779c,0xbc3736a2 }
};

static int ec_is_inf(const ec_point *P) {
    return bn_is_zero(P->x) && bn_is_zero(P->y);
}

static void ec_double(ec_point *R, const ec_point *P) {
    if (ec_is_inf(P) || bn_is_zero(P->y)) {
        bn_set_zero(R->x); bn_set_zero(R->y);
        return;
    }
    /* 复制输入坐标：R 与 P 可能指向同一内存（原地倍点），
     * 必须在写 R->x/R->y 之前保留原 x/y，否则 y3 计算读脏值。 */
    bn_t px, py;
    memcpy(px, P->x, sizeof(bn_t));
    memcpy(py, P->y, sizeof(bn_t));

    bn_t x2, t3, num, den, inv, lam;
    bn_mod_mul(x2, px, px, SM2_P);
    /* 3*x^2 */
    bn_add_mod(t3, x2, x2, SM2_P);
    bn_add_mod(t3, t3, x2, SM2_P);
    bn_add_mod(num, t3, SM2_A, SM2_P);
    bn_add_mod(den, py, py, SM2_P);
    bn_mod_inv(inv, den, SM2_P);
    bn_mod_mul(lam, num, inv, SM2_P);

    bn_t lam2, t;
    bn_mod_mul(lam2, lam, lam, SM2_P);
    bn_add_mod(t, px, px, SM2_P);
    bn_sub_mod(R->x, lam2, t, SM2_P);

    bn_t dx, t2;
    bn_sub_mod(dx, px, R->x, SM2_P);
    bn_mod_mul(t2, lam, dx, SM2_P);
    bn_sub_mod(R->y, t2, py, SM2_P);
}

static void ec_add(ec_point *R, const ec_point *P, const ec_point *Q) {
    if (ec_is_inf(P)) { *R = *Q; return; }
    if (ec_is_inf(Q)) { *R = *P; return; }
    if (bn_eq(P->x, Q->x)) {
        if (bn_eq(P->y, Q->y)) {
            ec_double(R, P);
        } else {
            bn_set_zero(R->x); bn_set_zero(R->y);
        }
        return;
    }
    /* 复制输入坐标：R 可能指向 P（原地累加），写 R->x 后再读 P->x 会读脏值。 */
    bn_t px, py, qx, qy;
    memcpy(px, P->x, sizeof(bn_t));
    memcpy(py, P->y, sizeof(bn_t));
    memcpy(qx, Q->x, sizeof(bn_t));
    memcpy(qy, Q->y, sizeof(bn_t));

    bn_t num, den, inv, lam;
    bn_sub_mod(num, qy, py, SM2_P);
    bn_sub_mod(den, qx, px, SM2_P);
    bn_mod_inv(inv, den, SM2_P);
    bn_mod_mul(lam, num, inv, SM2_P);

    bn_t lam2, t;
    bn_mod_mul(lam2, lam, lam, SM2_P);
    bn_sub_mod(t, lam2, px, SM2_P);
    bn_sub_mod(R->x, t, qx, SM2_P);

    bn_t dx, t2;
    bn_sub_mod(dx, px, R->x, SM2_P);
    bn_mod_mul(t2, lam, dx, SM2_P);
    bn_sub_mod(R->y, t2, py, SM2_P);
}

/* R = [k]G（double-and-add，高位到低位） */
static void ec_scalar_mul(ec_point *R, const bn_t k, const ec_point *G) {
    bn_set_zero(R->x); bn_set_zero(R->y);
    for (int bit = 255; bit >= 0; bit--) {
        ec_double(R, R);
        if ((k[bit >> 5] >> (bit & 31)) & 1)
            ec_add(R, R, G);
    }
}

/* ZA = SM3(ENTL || ID || a || b || xG || yG || xA || yA) */
static void sm2_compute_z(uint8_t z[32], const uint8_t *id, size_t idlen,
                          const bn_t xA, const bn_t yA) {
    vc_sm3_ctx ctx;
    vc_sm3_init(&ctx);
    uint8_t entl[2] = { (uint8_t)((idlen * 8) >> 8), (uint8_t)(idlen * 8) };
    vc_sm3_update(&ctx, entl, 2);
    vc_sm3_update(&ctx, id, idlen);
    uint8_t tmp[32];
    bn_to_bytes(tmp, SM2_A); vc_sm3_update(&ctx, tmp, 32);
    bn_to_bytes(tmp, SM2_B); vc_sm3_update(&ctx, tmp, 32);
    bn_to_bytes(tmp, SM2_GX); vc_sm3_update(&ctx, tmp, 32);
    bn_to_bytes(tmp, SM2_GY); vc_sm3_update(&ctx, tmp, 32);
    bn_to_bytes(tmp, xA); vc_sm3_update(&ctx, tmp, 32);
    bn_to_bytes(tmp, yA); vc_sm3_update(&ctx, tmp, 32);
    vc_sm3_final(&ctx, z);
}

int vc_sm2_pub_from_priv(const uint8_t priv[32], uint8_t pub[64]) {
    bn_t d;
    bytes_to_bn(d, priv);
    if (bn_is_zero(d) || bn_cmp(d, SM2_N) >= 0) return -1;
    ec_point P;
    ec_scalar_mul(&P, d, &SM2_G);
    bn_to_bytes(pub, P.x);
    bn_to_bytes(pub + 32, P.y);
    return 0;
}

int vc_sm2_sign(const uint8_t priv[32], const uint8_t *msg, size_t msglen,
                const uint8_t *id, size_t idlen,
                const uint8_t k[32], uint8_t r_out[32], uint8_t s_out[32]) {
    bn_t d, ka;
    bytes_to_bn(d, priv);
    bytes_to_bn(ka, k);
    if (bn_is_zero(d) || bn_cmp(d, SM2_N) >= 0) return -1;
    if (bn_is_zero(ka) || bn_cmp(ka, SM2_N) >= 0) return -1;

    ec_point P;
    ec_scalar_mul(&P, d, &SM2_G);

    uint8_t z[32];
    sm2_compute_z(z, id, idlen, P.x, P.y);

    vc_sm3_ctx ctx;
    vc_sm3_init(&ctx);
    vc_sm3_update(&ctx, z, 32);
    vc_sm3_update(&ctx, msg, msglen);
    uint8_t eb[32];
    vc_sm3_final(&ctx, eb);
    bn_t e;
    bytes_to_bn(e, eb);

    ec_point K;
    ec_scalar_mul(&K, ka, &SM2_G);

    bn_t r;
    bn_add_mod(r, e, K.x, SM2_N);

    bn_t one, d1, d1inv, rd, k_rd, s;
    bn_set_one(one);
    bn_add_mod(d1, one, d, SM2_N);
    bn_mod_inv(d1inv, d1, SM2_N);
    bn_mod_mul(rd, r, d, SM2_N);
    bn_sub_mod(k_rd, ka, rd, SM2_N);
    bn_mod_mul(s, d1inv, k_rd, SM2_N);

    bn_to_bytes(r_out, r);
    bn_to_bytes(s_out, s);
    return 0;
}

int vc_sm2_verify(const uint8_t pub[64], const uint8_t *msg, size_t msglen,
                  const uint8_t *id, size_t idlen,
                  const uint8_t r_bytes[32], const uint8_t s_bytes[32]) {
    bn_t r, s;
    bytes_to_bn(r, r_bytes);
    bytes_to_bn(s, s_bytes);
    if (bn_is_zero(r) || bn_cmp(r, SM2_N) >= 0) return 0;
    if (bn_is_zero(s) || bn_cmp(s, SM2_N) >= 0) return 0;

    ec_point P;
    bytes_to_bn(P.x, pub);
    bytes_to_bn(P.y, pub + 32);

    uint8_t z[32];
    sm2_compute_z(z, id, idlen, P.x, P.y);

    vc_sm3_ctx ctx;
    vc_sm3_init(&ctx);
    vc_sm3_update(&ctx, z, 32);
    vc_sm3_update(&ctx, msg, msglen);
    uint8_t eb[32];
    vc_sm3_final(&ctx, eb);
    bn_t e;
    bytes_to_bn(e, eb);

    bn_t t;
    bn_add_mod(t, r, s, SM2_N);
    if (bn_is_zero(t)) return 0;

    ec_point sG, tP, sum;
    ec_scalar_mul(&sG, s, &SM2_G);
    ec_scalar_mul(&tP, t, &P);
    ec_add(&sum, &sG, &tP);

    bn_t R;
    bn_add_mod(R, e, sum.x, SM2_N);
    return bn_eq(R, r);
}

/* ---------------- 安全随机数 / hex 解码 ---------------- */

int vc_secure_rand(uint8_t out[32]) {
#ifdef __linux__
    size_t got = 0;
    while (got < 32) {
        ssize_t r = getrandom(out + got, 32 - got, 0);
        if (r < 0) {
            /* getrandom 不可用（极老内核）时回退 /dev/urandom */
            FILE *f = fopen("/dev/urandom", "rb");
            if (!f) return -1;
            size_t n = fread(out, 1, 32, f);
            fclose(f);
            return n == 32 ? 0 : -1;
        }
        got += (size_t)r;
    }
    return 0;
#elif defined(_WIN32)
    /* Windows：BCryptGenRandom（内核 CSPRNG）。原本这里直接 return -1，
     * 导致 x86/Windows 上出证与 VQF 签名一路被判为「密钥不可用」而静默
     * 关闭；与 tools/vllm_vqf_sign.c 的兜底口径保持一致。 */
    BCRYPT_ALG_HANDLE h = NULL;
    if (BCryptOpenAlgorithmProvider(&h, BCRYPT_RNG_ALGORITHM, NULL, 0) != 0)
        return -1;
    NTSTATUS st = BCryptGenRandom(h, out, 32, 0);
    BCryptCloseAlgorithmProvider(h, 0);
    return st == 0 ? 0 : -1;
#else
    (void)out;
    return -1;   /* 其他平台：签名侧需调用方自备 CSPRNG */
#endif
}

static int vc_hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

int vc_hex_decode(const char *hex, uint8_t *out, size_t outlen) {
    if (!hex || !out) return -1;
    for (size_t i = 0; i < outlen; i++) {
        int hi = vc_hex_nibble(hex[2 * i]);
        int lo = vc_hex_nibble(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) return -1;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return (int)outlen;
}

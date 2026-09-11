/* ================================================================
 * vqf.c - VQF (vLLM Quantized Format) writer + mmap loader.
 *
 * 转换（vqf_write）在引擎加载完成后调用：权重已在内存中完成量化+repack
 * （Q8 8x8 tiled / Q4 4x4 / Q4 8x8l），VQF 直接固化该最终布局字节。
 * 加载（vqf_load）= mmap 整文件 + 元数据校验 + 指针直挂权重结构体，
 * 零转换零计算，加载耗时从 ~23s 降到 ~2s（映射 + 校验）。
 *
 * 位级一致性（fixedpoint_quantize_saturate red line）：转换与推理共用
 * 引擎同一套 f32_to_q8_0/f32_to_q4_0/repack_* 代码路径，因此 VQF 权重
 * 与内存加载结果逐位一致；加载时校验布局 flags 与当前引擎配置一致。
 * ================================================================ */
#include "vqf.h"
#include "vllm_platform.h"
#include "vllm_vision.h"   /* STVisionWeights: VQF vision 张量收集/挂载 */
#include "vllm_crypto.h"   /* VQF-Enc: SM4-CTR + HMAC-SM3；SM2 签名 */
#include "vllm_sign_keys.h" /* SM2 验签信任公钥（编译期内嵌） */

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>   /* GetProcessMemoryInfo（WorkingSet=RSS 采样） */
#ifndef MADV_DONTNEED
#define MADV_DONTNEED 0
#endif
#ifndef MADV_WILLNEED
#define MADV_WILLNEED 1
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define Q8_BYTES(n) ((size_t)((n) + 31) / 32 * 34)
#define Q4_BYTES(n) ((size_t)((n) + 31) / 32 * 18)

/* 分层驻留：vqf_load 成功路径传入当前 VQF 的 fd（setup/advise 共用） */
static int g_vqf_stream_fd = -1;

/* 已加载 VQF 的文件内固化布局（只读展示，/admin 用）：量化在转换期已固化，
 * 加载侧不可更改，故此处只回填文件头的 flags/version 供读取。 */
static uint32_t g_vqf_loaded_flags = 0;
static uint32_t g_vqf_loaded_version = 0;

int vqf_layout_state(uint32_t *flags, uint32_t *version) {
    if (!g_vqf_loaded_flags) return 0;
    if (flags)   *flags   = g_vqf_loaded_flags;
    if (version) *version = g_vqf_loaded_version;
    return 1;
}

/* 引擎全局布局开关（vllm_safetensors.c）：几何门控失败会被清零，转换时
 * 以实际状态为准。 */
extern int g_st_q8_repack;
extern int g_st_q4_repack;


static uint64_t vqf_fnv_update(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

/* ---------------- VQF-Enc（SM4-CTR + HMAC-SM3） ----------------
 * 存储态加密：数据区（data_offset..file_len）用 SM4-CTR 逐字节加密（密文
 * 长度==明文长度，mmap 布局不变），尾部以 HMAC-SM3 tag 取代 FNV 做完整性/
 * 认证（防篡改）。密钥来自环境变量 VLLM_VQF_KEY 口令，经 SM3 派生 32 字节：
 *   [0..16) SM4 密钥，[16..32) HMAC-SM3 密钥。
 * 认证范围（彻底防篡改）：头部（VQFHeader 全体，file_len 视为 0，含 arch/
 * flags/enc_iv/data_offset/n_tensors）+ 目录（明文）+ 数据区（密文）。
 * file_len 因回填晚于数据写出而不纳入，由加载侧 size mismatch 校验兜底。
 * 口令为空 → 明文 VQF（向后兼容，零成本）。 */

/* 由口令派生 SM4/HMAC 密钥；返回 1=加密，0=明文（无口令）。 */
static int vqf_enc_derive(const char *pass, uint8_t sm4key[16], uint8_t hmackey[16]) {
    if (!pass || !pass[0]) return 0;
    uint8_t mat[32];
    vc_sm3(pass, strlen(pass), mat);
    memcpy(sm4key, mat, 16);
    memcpy(hmackey, mat + 16, 16);
    return 1;
}


/* 取验签信任公钥：优先运行时 VLLM_VQF_SIGN_PUB（128 hex），其次编译期内嵌
 * VLLM_SM2_PUB_HEX。返回 0 成功；-1 无信任公钥（fail-closed）。 */
static int vqf_trusted_pub(uint8_t pub[64]) {
    const char *e = getenv("VLLM_VQF_SIGN_PUB");
    if (e && e[0] && vc_hex_decode(e, pub, 64) == 64) return 0;
#ifdef VLLM_SM2_PUB_HEX
    if (vc_hex_decode(VLLM_SM2_PUB_HEX, pub, 64) == 64) return 0;
#endif
    return -1;
}


/* ---------------- 加载 ---------------- */
int vqf_is_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    uint32_t m = 0;
    int ok = (fread(&m, 1, 4, f) == 4);
    fclose(f);
    return ok && m == VQF_MAGIC;
}

/* 当前引擎布局 flags（加载时由环境 + wmode 决定） */
static uint32_t vqf_cur_flags(const STModelWeights *w) {
    uint32_t f = 0;
    if (g_st_q8_repack) {
        const char *e = getenv("VLLM_Q8_8X8");
        if (!(e && e[0] == '0')) f |= VQF_FLAG_Q8_8X8;
    }
    if (g_st_q4_repack) f |= VQF_FLAG_Q4_4X4;
    if (w->q8_buf_q4)   f |= VQF_FLAG_Q8BUF_Q4;
    if (st_wmode_effective() == 4) f |= VQF_FLAG_G256;
    return f;
}

static void *vqf_mount(const void *base, uint64_t off) {
    return (uint8_t *)base + (size_t)off;
}

/* ---------------- 旧版 41 字段 VQFArch 头兼容 ----------------
 * 成因：MoE 支持在 VQFArch 尾部追加 n_experts/moe_ffn/top_k/shared_experts
 * 四字段（arch 164B→180B、header 432B→448B），但稠密文件的 version 仍写 2，
 * 故同一 version=2 存在「旧 41 字段头」与「新 45 字段头」。两者 arch 前 41 字段
 * 偏移一致、目录偏移（sizeof 对齐到 64B）=448 也一致，仅 arch 之后的
 * data_offset/file_len/enc_iv/sig 整体错位 16B（旧头 184/192，新头 200/208）。
 *
 * 判据：分别按新、旧偏移取候选 (data_offset,file_len)，候选有效须同时满足
 *   1) data_offset 非 0、64B(VQF_ALIGN) 对齐、>= 目录末尾 dir_end、< 文件大小；
 *   2) file_len = 文件大小 - tag_len（tag：明文 FNV 8B / VQF_FLAG_ENC HMAC 32B），
 *      与真实文件长度严格自洽；
 *   3) file_len > data_offset（数据区非空）。
 * 恰好一种布局满足 → 采用；两种都满足（歧义）或都不满足 → 拒绝并报错，
 * 绝不静默按错误偏移继续。 */

/* 候选 (data_offset,file_len) 自洽性校验 */
static int vqf_cand_valid(uint64_t d_off, uint64_t f_len, size_t file_size,
                          size_t tag_len, uint64_t dir_end) {
    if (d_off == 0 || f_len == 0) return 0;
    if ((d_off & (uint64_t)(VQF_ALIGN - 1)) != 0) return 0;   /* 64B 对齐 */
    if (d_off < dir_end) return 0;                            /* 不覆盖头+目录 */
    if (d_off >= (uint64_t)file_size) return 0;
    if (f_len + (uint64_t)tag_len != (uint64_t)file_size) return 0;  /* 尾部 tag 自洽 */
    if (f_len <= d_off) return 0;                             /* 数据区非空 */
    return 1;
}

/* 判定头布局并归一化到新 45 字段布局；返回 0 成功（*out 可直接当 VQFHeader 用），
 * -1 失败（几何非法 / 布局无法判定）。 */
static int vqf_header_normalize(const uint8_t *raw, size_t file_size,
                                VQFHeader *out) {
    const VQFHeader *h = (const VQFHeader *)raw;
    size_t tag_len = (h->flags & VQF_FLAG_ENC) ? 32 : 8;
    uint64_t dir_end = (uint64_t)((sizeof(VQFHeader) + 63) & ~(size_t)63) +
                       (uint64_t)h->n_tensors * sizeof(VQFTensor);

    /* arch 前 41 字段在新旧头偏移一致：先做语义门，剔除明显非法头 */
    const VQFArch *a = &h->arch;
    if (!(a->dim > 0 && a->n_layers > 0 && a->n_heads > 0 && a->head_dim > 0 &&
          a->ffn_dim > 0 && a->vocab_size > 0)) {
        fprintf(stderr, "[VQF] arch geometry invalid (dim=%u layers=%u heads=%u "
                        "head_dim=%u ffn=%u vocab=%u)\n",
                a->dim, a->n_layers, a->n_heads, a->head_dim, a->ffn_dim,
                a->vocab_size);
        return -1;
    }

    uint64_t d_new, f_new, d_old, f_old;
    memcpy(&d_new, raw + offsetof(VQFHeader, data_offset), 8);
    memcpy(&f_new, raw + offsetof(VQFHeader, file_len), 8);
    memcpy(&d_old, raw + offsetof(VQFHeaderLegacy, data_offset), 8);
    memcpy(&f_old, raw + offsetof(VQFHeaderLegacy, file_len), 8);
    int ok_new = vqf_cand_valid(d_new, f_new, file_size, tag_len, dir_end);
    int ok_old = vqf_cand_valid(d_old, f_old, file_size, tag_len, dir_end);

    *out = *h;   /* 整体拷贝：magic/version/flags/n_tensors/arch 前 41 字段均正确 */
    if (ok_new && !ok_old) return 0;                    /* 新 45 字段头：无需调整 */

    if (ok_old && !ok_new) {                            /* 旧 41 字段头 */
        if (h->flags & (VQF_FLAG_ENC | VQF_FLAG_SIGNED)) {
            fprintf(stderr, "[VQF] legacy 41-field header with ENC/SIGNED flag "
                            "unsupported (authentication accounting undefined)\n");
            return -1;
        }
        /* arch 尾部 4 个 MoE 字段在旧头中不存在（该偏移实为 data_offset/
         * file_len），清零；尾部字段按旧偏移（整体前移 16B）读入。 */
        out->arch.n_experts = 0;
        out->arch.moe_ffn = 0;
        out->arch.top_k = 0;
        out->arch.shared_experts = 0;
        memcpy(&out->data_offset, raw + offsetof(VQFHeaderLegacy, data_offset), 8);
        memcpy(&out->file_len,    raw + offsetof(VQFHeaderLegacy, file_len), 8);
        memcpy(out->enc_iv,       raw + offsetof(VQFHeaderLegacy, enc_iv), 16);
        memcpy(out->enc_rsvd,     raw + offsetof(VQFHeaderLegacy, enc_rsvd), 16);
        memcpy(&out->sig,         raw + offsetof(VQFHeaderLegacy, sig), sizeof(VQFSig));
        fprintf(stderr, "[VQF] legacy 41-field VQFArch header detected "
                        "(data_offset@%zu file_len@%zu), normalized\n",
                offsetof(VQFHeaderLegacy, data_offset),
                offsetof(VQFHeaderLegacy, file_len));
        return 0;
    }

    fprintf(stderr, "[VQF] header layout undetermined "
                    "(new: data_offset=%llu file_len=%llu valid=%d; "
                    "legacy: data_offset=%llu file_len=%llu valid=%d; file=%zu)\n",
            (unsigned long long)d_new, (unsigned long long)f_new, ok_new,
            (unsigned long long)d_old, (unsigned long long)f_old, ok_old,
            file_size);
    return -1;
}

int vqf_load(STModelWeights *w, const char *path) {
    st_mmap_t m;
    g_vqf_loaded_flags = 0;      /* 布局展示：加载失败/未加载时归零 */
    g_vqf_loaded_version = 0;
    m.fi = -1; m.data = NULL; m.len = 0; m.fd = -1;
    if (st_mmap_open(&m, 0, path) != 0) {
        fprintf(stderr, "[VQF] mmap failed: %s\n", path);
        return -1;
    }
    if (m.len < sizeof(VQFHeader) + sizeof(VQFTensor)) {
        fprintf(stderr, "[VQF] file too small\n");
        st_mmap_close(&m); return -1;
    }
    const VQFHeader *raw = (const VQFHeader *)m.data;
    if (raw->magic != VQF_MAGIC) { st_mmap_close(&m); return -1; }
    if (raw->version != VQF_VERSION && raw->version != VQF_VERSION_MOE) {
        fprintf(stderr, "[VQF] version %u unsupported (need %u/%u)\n",
                raw->version, VQF_VERSION, VQF_VERSION_MOE);
        st_mmap_close(&m); return -1;
    }
    /* 兼容旧版 41 字段 VQFArch 头：判定布局并归一化为新头（新头原样拷贝） */
    VQFHeader hdr;
    if (vqf_header_normalize((const uint8_t *)m.data, m.len, &hdr) != 0) {
        st_mmap_close(&m); return -1;
    }
    const VQFHeader *h = &hdr;
    int enc = (h->flags & VQF_FLAG_ENC) ? 1 : 0;
    size_t tag_len = enc ? 32 : 8;
    if (h->file_len != 0 && (uint64_t)m.len != h->file_len + tag_len) {
        fprintf(stderr, "[VQF] size mismatch: file=%zu header=%llu\n",
                m.len, (unsigned long long)h->file_len);
        st_mmap_close(&m); return -1;
    }
    size_t d_off = (sizeof(VQFHeader) + 63) & ~(size_t)63;
    int signed_ = (h->flags & VQF_FLAG_SIGNED) ? 1 : 0;
    /* 验签决策：签名模型默认走加密（验签随解密单遍完成，默认开）；明文模型
     * 验签需全量读数据区（抵消 mmap 收益），仅 VLLM_VQF_VERIFY=1 显式开启。 */
    const char *vq = getenv("VLLM_VQF_VERIFY");
    int do_verify = signed_ && (enc || (vq && vq[0] == '1'));
    vc_sm3_ctx vdig;
    uint8_t vdig_out[32];
    if (do_verify) {
        vc_sm3_init(&vdig);
        VQFHeader hsig = *h;
        hsig.file_len = 0;
        memset(&hsig.sig, 0, sizeof(hsig.sig));
        vc_sm3_update(&vdig, &hsig, sizeof(hsig));
        vc_sm3_update(&vdig, (const uint8_t *)m.data + d_off,
                      (size_t)h->n_tensors * sizeof(VQFTensor));
    }
    /* VQF-Enc：SM4-CTR 加密文件。验证 HMAC-SM3 tag（覆盖 头部+目录+数据区），
     * 通过后原位解密（mprotect 加写，MAP_PRIVATE COW，磁盘文件不变）。
     * 无口令/口令错 → 拒绝加载。 */
    if (enc) {
        const char *pass = getenv("VLLM_VQF_KEY");
        uint8_t sm4key[16], hmackey[16];
        if (!vqf_enc_derive(pass, sm4key, hmackey)) {
            fprintf(stderr, "[VQF] encrypted model requires VLLM_VQF_KEY\n");
            st_mmap_close(&m); return -1;
        }
        vc_hmac_ctx hc;
        vc_hmac_init(&hc, hmackey, 16);
        /* 头部认证：file_len 与 sig 块均回填后 ≠ 写入侧（写入侧 HMAC 时
         * file_len=0 且 sig 全 0），故复制并按 0 喂入；file_len 本身由上方
         * size mismatch 校验兜底，sig 由 SM2 验签单独覆盖。 */
        VQFHeader hdr_auth = *h;
        hdr_auth.file_len = 0;
        memset(&hdr_auth.sig, 0, sizeof(hdr_auth.sig));
        vc_hmac_update(&hc, &hdr_auth, sizeof(hdr_auth));
        vc_hmac_update(&hc, (const uint8_t *)m.data + d_off,
                       (size_t)h->n_tensors * sizeof(VQFTensor));
        /* 单遍：HMAC（密文）+ 原位解密。先使整映射可写（POSIX: mprotect；
         * Windows FILE_MAP_COPY 视图本就 COW 可写，no-op 成功）。 */
        if (st_mmap_write_enable(&m) != 0) {
            fprintf(stderr, "[VQF] mprotect failed\n");
            st_mmap_close(&m); return -1;
        }
        vc_sm4_ctx sm4ctx;
        uint8_t ctr[16];
        memcpy(ctr, h->enc_iv, 16);
        vc_sm4_setkey_enc(&sm4ctx, sm4key);
        uint8_t *dp = (uint8_t *)m.data + h->data_offset;
        size_t dlen = (size_t)(h->file_len - h->data_offset);
        size_t done = 0;
        while (done < dlen) {
            size_t k = dlen - done;
            if (k > (1u << 20)) k = 1u << 20;
            vc_hmac_update(&hc, dp + done, k);                     /* 密文 */
            vc_sm4_ctr_crypt(&sm4ctx, ctr, dp + done, dp + done, k); /* 原位解密 */
            if (do_verify) vc_sm3_update(&vdig, dp + done, k);      /* 明文摘要 */
            done += k;
        }
        uint8_t tag[32];
        vc_hmac_final(&hc, tag);
        const uint8_t *stored = (const uint8_t *)m.data + m.len - 32;
        if (memcmp(tag, stored, 32) != 0) {
            fprintf(stderr, "[VQF] HMAC-SM3 mismatch (tampered or wrong key)\n");
            st_mmap_close(&m); return -1;
        }
        fprintf(stderr, "[VQF] decrypted %s (SM4-CTR, HMAC-SM3 ok)\n", path);
    }
    /* 尾部 FNV-1a checksum（仅明文文件；enc 文件由 HMAC-SM3 覆盖）。
     * （覆盖 目录 + 数据区，与 vqf_write 边写边算一致）。
     * 默认跳过整文件校验：转换时已 self-check 通过，位级一致由构造保证；
     * mmap 页由内核按需加载，整文件校验会把 4GB 从 eMMC 全量读一遍
     * （~20s），直接抵消 mmap 的加载收益。设 VLLM_VQF_CHECK=1 才校验
     * （完整性诊断用）。 */
    const char *chk = getenv("VLLM_VQF_CHECK");
    if (!enc && chk && chk[0] == '1') {
        uint64_t want_sum = 0;
        if (m.len >= 8) memcpy(&want_sum, (const uint8_t *)m.data + m.len - 8, 8);
        if (want_sum != 0) {
            uint64_t calc = 0xcbf29ce484222325ull;
            calc = vqf_fnv_update(calc, (const uint8_t *)m.data + d_off,
                                  (size_t)h->n_tensors * sizeof(VQFTensor));
            calc = vqf_fnv_update(calc, (const uint8_t *)m.data + h->data_offset,
                                  (size_t)(h->file_len - h->data_offset));
            if (calc != want_sum) {
                fprintf(stderr, "[VQF] checksum mismatch\n");
                st_mmap_close(&m); return -1;
            }
        }
    }
    /* SM2 验签：签名模型按 do_verify 判定（加密默认开，明文 VLLM_VQF_VERIFY=1）。
     * 加密模型已在解密循环单遍喂入明文摘要；明文模型在此全量读数据区。 */
    if (do_verify && !enc) {
        const uint8_t *vdp = (const uint8_t *)m.data + h->data_offset;
        size_t vlen = (size_t)(h->file_len - h->data_offset);
        size_t vdone = 0;
        while (vdone < vlen) {
            size_t k = vlen - vdone;
            if (k > (1u << 20)) k = 1u << 20;
            vc_sm3_update(&vdig, vdp + vdone, k);
            vdone += k;
        }
    }
    if (do_verify) {
        vc_sm3_final(&vdig, vdig_out);
        /* 快速自检：摘要与头内 digest 不等 → 直接拒绝（省一次 ECC 标量乘） */
        if (memcmp(vdig_out, h->sig.digest, 32) != 0) {
            fprintf(stderr, "[VQF] SM2 digest mismatch (tampered)\n");
            st_mmap_close(&m); return -1;
        }
        uint8_t vpub[64];
        if (vqf_trusted_pub(vpub) != 0) {
            fprintf(stderr, "[VQF] signed model requires trusted public key "
                            "(VLLM_VQF_SIGN_PUB or embedded VLLM_SM2_PUB_HEX)\n");
            st_mmap_close(&m); return -1;
        }
        /* 验签必须用可信公钥（非文件内 sig.pub），ID 以内嵌固定 ID 为准 */
        if (!vc_sm2_verify(vpub, vdig_out, 32, (const uint8_t *)VQF_SM2_ID,
                           strlen(VQF_SM2_ID), h->sig.r, h->sig.s)) {
            fprintf(stderr, "[VQF] SM2 verify failed (model replaced/tampered)\n");
            st_mmap_close(&m); return -1;
        }
        /* 交叉校验：文件内 sig.pub 与可信公钥不一致 → 拒绝（默认） */
        if (memcmp(h->sig.pub, vpub, 64) != 0) {
            fprintf(stderr, "[VQF] SM2 pub mismatch (sig.pub != trusted key)\n");
            st_mmap_close(&m); return -1;
        }
        fprintf(stderr, "[VQF] SM2 verify ok\n");
    }
    /* 布局 flags 一致性：文件固化布局必须与当前引擎配置一致 */
    uint32_t cur = vqf_cur_flags(w);
    uint32_t want = h->flags & (VQF_FLAG_Q8_8X8 | VQF_FLAG_Q4_4X4 |
                                VQF_FLAG_X8 | VQF_FLAG_Q8BUF_Q4 | VQF_FLAG_G256);
    uint32_t curm = cur & (VQF_FLAG_Q8_8X8 | VQF_FLAG_Q4_4X4 |
                           VQF_FLAG_X8 | VQF_FLAG_Q8BUF_Q4 | VQF_FLAG_G256);
    /* 非默认档直挂适配（serve/--stream-test 的 VQF 直挂路径不经过 alloc，
     * w 为 calloc 全零，q8_buf_q4/repack 状态缺失导致 flags 校验误判）：
     *   - 文件声明 Q8BUF_Q4（q4i 档，q8_* 缓冲存 pre-unpacked Q4 int8）时，
     *     引擎必须关闭 Q8 8x8 repack（q4i 无 8x8 tile），并把该语义同步到 w；
     *   - 若 w 已由 alloc 预置（is_allocated），保持其状态不变。 */
    if (!w->is_allocated && (h->flags & VQF_FLAG_Q8BUF_Q4)) {
        w->q8_buf_q4 = 1;
        g_st_q8_repack = 0;   /* 与 st_weights_alloc wmode==3 分支一致 */
        cur = vqf_cur_flags(w);
        curm = cur & (VQF_FLAG_Q8_8X8 | VQF_FLAG_Q4_4X4 |
                      VQF_FLAG_X8 | VQF_FLAG_Q8BUF_Q4 | VQF_FLAG_G256);
    }
    if (want != curm) {
        fprintf(stderr, "[VQF] layout mismatch: file=0x%x engine=0x%x "
                        "(check VLLM_Q8_8X8 / wmode match the conversion)\n",
                want, curm);
        st_mmap_close(&m); return -1;
    }

    memset(w, 0, sizeof(*w));
    const VQFArch *a = &h->arch;
    STModelConfig *c = &w->cfg;
    c->dim = (int)a->dim; c->n_layers = (int)a->n_layers;
    c->n_heads = (int)a->n_heads; c->n_kv_heads = (int)a->n_kv_heads;
    c->head_dim = (int)a->head_dim; c->ffn_dim = (int)a->ffn_dim;
    c->vocab_size = (int)a->vocab_size; c->max_seq_len = (int)a->max_seq_len;
    c->rope_theta = a->rope_theta; c->norm_eps = a->norm_eps;
    c->bos_id = (int)a->bos_id; c->eos_id = (int)a->eos_id;
    c->has_q_norm = a->has_q_norm; c->has_mrope = a->has_mrope;
    c->head_dim_full = (int)a->head_dim_full; c->kv_lora_rank = (int)a->kv_lora_rank;
    c->has_vision = (int)a->has_vision; c->vis_depth = (int)a->vis_depth;
    c->vis_hidden = (int)a->vis_hidden; c->vis_heads = (int)a->vis_heads;
    c->vis_ffn = (int)a->vis_ffn; c->vis_patch = (int)a->vis_patch;
    c->vis_temporal = (int)a->vis_temporal; c->vis_merge = (int)a->vis_merge;
    c->vis_out_dim = (int)a->vis_out_dim; c->vis_in_chan = (int)a->vis_in_chan;
    c->vis_max_pos = (int)a->vis_max_pos; c->vis_ds_count = (int)a->vis_ds_count;
    c->mrope_n_sec = (int)a->mrope_n_sec;
    for (int i = 0; i < 4; i++) {
        c->vis_ds_idx[i] = (int)a->vis_ds_idx[i];
        c->mrope_sections[i] = (int)a->mrope_sections[i];
    }
    c->vision_start_id = (int)a->vision_start_id;
    c->vision_end_id = (int)a->vision_end_id;
    c->image_token_id = (int)a->image_token_id;
    c->video_token_id = (int)a->video_token_id;
    /* MoE arch tail（v3；dense v2 文件该区恒 0 → is_moe 判定取 flags/version） */
    c->is_moe = (h->flags & VQF_FLAG_MOE) || h->version == VQF_VERSION_MOE;
    c->n_experts = (int)a->n_experts;
    c->moe_ffn = (int)a->moe_ffn;
    c->top_k = (int)a->top_k;
    c->shared_experts = (int)a->shared_experts;

    int nl = c->n_layers;
    int q_out = c->n_heads * c->head_dim;
    int k_out = c->n_kv_heads * c->head_dim;
    int ff = c->ffn_dim, vc = c->vocab_size, dim = c->dim;

    size_t dir_off = (sizeof(VQFHeader) + 63) & ~(size_t)63;
    uint64_t data_end = h->file_len;   /* file_len = 数据区绝对结束位置（含 data_offset） */
    for (uint32_t i = 0; i < h->n_tensors; i++) {
        const VQFTensor *e = (const VQFTensor *)((const uint8_t *)m.data + dir_off +
                                                (size_t)i * sizeof(VQFTensor));
        if ((uint64_t)e->offset + e->bytes > data_end) {
            fprintf(stderr, "[VQF] tensor %s out of range\n", e->name);
            st_mmap_close(&m); return -1;
        }
        void *p = vqf_mount(m.data, e->offset);
        const char *name = e->name;
        if (strcmp(name, "token_embed") == 0)      w->token_embed = (float *)p;
        else if (strcmp(name, "final_norm") == 0)  w->final_norm  = (float *)p;
        else if (strcmp(name, "moe_router") == 0)  w->moe_router  = (float *)p;
        else if (strcmp(name, "attn_norm") == 0)   w->attn_norm   = (float *)p;
        else if (strcmp(name, "ffn_norm") == 0)    w->ffn_norm    = (float *)p;
        else if (strcmp(name, "q_norm") == 0)      w->q_norm      = (float *)p;
        else if (strcmp(name, "k_norm") == 0)      w->k_norm      = (float *)p;
        else if (strcmp(name, "q8_q") == 0)        w->q8_q_weight    = (uint8_t *)p;
        else if (strcmp(name, "q8_k") == 0)        w->q8_k_weight    = (uint8_t *)p;
        else if (strcmp(name, "q8_v") == 0)        w->q8_v_weight    = (uint8_t *)p;
        else if (strcmp(name, "q8_o") == 0)        w->q8_o_weight    = (uint8_t *)p;
        else if (strcmp(name, "q8_gate") == 0)     w->q8_gate_weight = (uint8_t *)p;
        else if (strcmp(name, "q8_up") == 0)       w->q8_up_weight   = (uint8_t *)p;
        else if (strcmp(name, "q8_down") == 0)     w->q8_down_weight = (uint8_t *)p;
        else if (strcmp(name, "q8_lm") == 0)       w->q8_lm_weight   = (uint8_t *)p;
        else if (strcmp(name, "q4_q") == 0)        w->q4_q_weight    = (uint8_t *)p;
        else if (strcmp(name, "q4_k") == 0)        w->q4_k_weight    = (uint8_t *)p;
        else if (strcmp(name, "q4_v") == 0)        w->q4_v_weight    = (uint8_t *)p;
        else if (strcmp(name, "q4_o") == 0)        w->q4_o_weight    = (uint8_t *)p;
        else if (strcmp(name, "q4_gate") == 0)     w->q4_gate_weight = (uint8_t *)p;
        else if (strcmp(name, "q4_up") == 0)       w->q4_up_weight   = (uint8_t *)p;
        else if (strcmp(name, "q4_down") == 0)     w->q4_down_weight = (uint8_t *)p;
        else if (strcmp(name, "q4_lm") == 0)       w->q4_lm_weight   = (uint8_t *)p;
        else if (strcmp(name, "x8_q") == 0)        w->x8_q_weight    = (uint8_t *)p;
        else if (strcmp(name, "x8_k") == 0)        w->x8_k_weight    = (uint8_t *)p;
        else if (strcmp(name, "x8_v") == 0)        w->x8_v_weight    = (uint8_t *)p;
        else if (strcmp(name, "x8_o") == 0)        w->x8_o_weight    = (uint8_t *)p;
        else if (strcmp(name, "x8_gate") == 0)     w->x8_gate_weight = (uint8_t *)p;
        else if (strcmp(name, "x8_up") == 0)       w->x8_up_weight   = (uint8_t *)p;
        else if (strcmp(name, "x8_down") == 0)     w->x8_down_weight = (uint8_t *)p;
        /* Vision 张量（VQF_FLAG_VISION）：mmap 直挂。指针 alias vqf_map，
         * 结构体本身由 vqf_load calloc，st_weights_free 的 vqf_map 分支释放。 */
        else if (name[0] == 'v' && name[1] == '_') {
            struct STVisionWeights *vw = w->vision;
            if (!vw) {
                vw = (struct STVisionWeights *)calloc(1, sizeof(*vw));
                if (!vw) { st_mmap_close(&m); return -1; }
                w->vision = vw;
            }
            if      (strcmp(name, "v_q8_qkv") == 0) vw->q8_attn_qkv_weight  = (uint8_t *)p;
            else if (strcmp(name, "v_q8_proj") == 0) vw->q8_attn_proj_weight = (uint8_t *)p;
            else if (strcmp(name, "v_q8_fc1") == 0)  vw->q8_mlp_fc1_weight   = (uint8_t *)p;
            else if (strcmp(name, "v_q8_fc2") == 0)  vw->q8_mlp_fc2_weight   = (uint8_t *)p;
            else if (strcmp(name, "v_patch_embed") == 0) vw->patch_embed_weight = (float *)p;
            else if (strcmp(name, "v_patch_bias") == 0)  vw->patch_embed_bias   = (float *)p;
            else if (strcmp(name, "v_pos_embed") == 0)   vw->pos_embed          = (float *)p;
            else if (strcmp(name, "v_qkv_bias") == 0)    vw->attn_qkv_bias      = (float *)p;
            else if (strcmp(name, "v_proj_bias") == 0)   vw->attn_proj_bias     = (float *)p;
            else if (strcmp(name, "v_norm1_w") == 0)     vw->norm1_weight       = (float *)p;
            else if (strcmp(name, "v_norm1_b") == 0)     vw->norm1_bias         = (float *)p;
            else if (strcmp(name, "v_norm2_w") == 0)     vw->norm2_weight       = (float *)p;
            else if (strcmp(name, "v_norm2_b") == 0)     vw->norm2_bias         = (float *)p;
            else if (strcmp(name, "v_fc1_bias") == 0)    vw->mlp_fc1_bias       = (float *)p;
            else if (strcmp(name, "v_fc2_bias") == 0)    vw->mlp_fc2_bias       = (float *)p;
            else if (strcmp(name, "v_merger_nw") == 0)   vw->merger_norm_weight = (float *)p;
            else if (strcmp(name, "v_merger_nb") == 0)   vw->merger_norm_bias   = (float *)p;
            else if (strcmp(name, "v_merger_f1w") == 0)  vw->merger_fc1_weight  = (float *)p;
            else if (strcmp(name, "v_merger_f1b") == 0)  vw->merger_fc1_bias    = (float *)p;
            else if (strcmp(name, "v_merger_f2w") == 0)  vw->merger_fc2_weight  = (float *)p;
            else if (strcmp(name, "v_merger_f2b") == 0)  vw->merger_fc2_bias    = (float *)p;
            else if (strcmp(name, "v_ds_nw") == 0)   vw->ds_norm_weight = (float *)p;
            else if (strcmp(name, "v_ds_nb") == 0)   vw->ds_norm_bias   = (float *)p;
            else if (strcmp(name, "v_ds_f1w") == 0)  vw->ds_fc1_weight  = (float *)p;
            else if (strcmp(name, "v_ds_f1b") == 0)  vw->ds_fc1_bias    = (float *)p;
            else if (strcmp(name, "v_ds_f2w") == 0)  vw->ds_fc2_weight  = (float *)p;
            else if (strcmp(name, "v_ds_f2b") == 0)  vw->ds_fc2_bias    = (float *)p;
            vw->is_allocated = 1;
        }
        /* 未知名字忽略（向前兼容） */
    }
    (void)nl; (void)q_out; (void)k_out; (void)ff; (void)vc; (void)dim;

    if (!w->token_embed || (!w->q8_q_weight && !w->q4_q_weight)) {
        fprintf(stderr, "[VQF] missing core tensors (embed / q/k/v)\n");
        st_mmap_close(&m); return -1;
    }
    if (c->is_moe) {
        /* 专家可 q4 nibble 或 q8_0（A：wmode=q8 文件无 q4_gate）；
         * 解码内核在 st_moe_ffn_sparse 内按可用权重分流。 */
        if (!w->moe_router || (!w->q4_gate_weight && !w->q8_gate_weight) ||
            c->n_experts <= 0) {
            fprintf(stderr, "[VQF] MoE file missing moe_router/expert weights\n");
            st_mmap_close(&m); return -1;
        }
        fprintf(stderr, "[VQF] MoE v3: experts=%d top_k=%d moe_ffn=%d shared=%d "
                        "ffn_dim=%d router=%.1fMB (f32 resident) experts=%s\n",
                c->n_experts, c->top_k, c->moe_ffn, c->shared_experts, c->ffn_dim,
                (double)c->n_layers * c->n_experts * c->dim * 4.0 / 1048576.0,
                w->q4_gate_weight ? "q4" : "q8");
    }
    w->has_q8  = w->q8_q_weight ? 1 : 0;
    w->has_q4  = w->q4_q_weight ? 1 : 0;
    w->has_x8  = (h->flags & VQF_FLAG_X8) ? 1 : 0;
    w->q8_buf_q4 = (h->flags & VQF_FLAG_Q8BUF_Q4) ? 1 : 0;
    w->emb_f16 = (h->flags & VQF_FLAG_EMB_F16) ? 1 : 0;
    g_vqf_loaded_flags = h->flags;      /* /admin 只读展示"文件内固化布局" */
    g_vqf_loaded_version = h->version;
    w->n_layers_allocated = c->n_layers;
    w->is_allocated = 1;
    w->vqf_map = m.data;
    w->vqf_map_len = m.len;
    g_vqf_stream_fd = m.fd;      /* 成功路径持有 fd（st_weights_free 仅 munmap） */
    vqf_stream_setup(w);   /* 分层驻留：解析 VLLM_VQF_STREAM（明文 VQF 专属） */
    return 0;
}

/* ---------------- 预热（可选，VLLM_VQF_PREWARM=1） ---------------- */
#ifndef _WIN32
#include <pthread.h>
#include <sys/mman.h>

static void *vqf_prewarm_worker(void *arg) {
    const STModelWeights *w = (const STModelWeights *)arg;
    const uint8_t *p = (const uint8_t *)w->vqf_map;
    size_t len = w->vqf_map_len;
    double t0 = st_now_sec();
    madvise((void *)p, len, MADV_WILLNEED);   /* 内核异步预读提示 */
    /* 逐页 touch：同步把页 fault 进 RAM（WILLNEED 是异步提示，逐页读确保完成）。
     * 只读首字节即触发整页 fault；稀疏跳读降低 touch 开销。 */
    volatile uint8_t sink = 0;
    for (size_t off = 0; off < len; off += 4096)
        sink ^= p[off];
    sink ^= p[len - 1];
    (void)sink;
    fprintf(stderr, "[VQF] prewarm done: %.2f s (%zu MB)\n",
            st_now_sec() - t0, len / 1048576);
    return NULL;
}
#endif

void vqf_prewarm(const STModelWeights *w) {
    if (!w || !w->vqf_map || !w->vqf_map_len) return;
#ifdef _WIN32
    (void)w;   /* Windows 无 madvise/忽略（开发机不需要预热） */
    return;
#else
    pthread_t th;
    if (pthread_create(&th, NULL, vqf_prewarm_worker, (void *)w) == 0)
        pthread_detach(th);
#endif
}

/* ================================================================
 * 分层驻留（AirLLM 型 layer streaming）—— 仅明文 VQF
 *
 * 数据区布局（vqf_write/vqf_collect）：per-layer 张量 rows=n_layers*R 整层
 * 堆叠连续存放，层 l 切片 = [tensor.offset + l*step, +step)，
 * step = e->bytes / n_layers（层内几何整除时恒定，repack 层内闭合保证）。
 * 常驻段 = embed/final_norm/lm_head/vision 等 rows 非 n_layers 倍数的张量。
 * 驱逐只对"干净文件页"（MAP_PRIVATE 只读 mmap）做 MADV_DONTNEED：
 * 丢 PTE，再访问从 eMMC 重新 fault → 字节由文件重建，位级一致不受影响。
 * ================================================================ */
#define VQF_STREAM_MAX_SEG 32

typedef struct {
    uint64_t off;    /* layer-0 起始（文件内偏移 == 映射内偏移，mmap 全文件） */
    uint64_t step;   /* 单层字节数（本张量恒定） */
} VQFStreamSeg;

static struct {
    const void *map;     /* 当前 w->vqf_map（模型换载后失效） */
    int         keep;    /* VLLM_VQF_STREAM=N */
    int         log;
    int         cold;    /* VLLM_VQF_STREAM_COLD=1：驱逐时连带丢页缓存（fadvise） */
    int         fd;      /* 当前 VQF 的文件描述符（vqf_load 成功路径持有） */
    int         nl;
    int         nseg;
    VQFStreamSeg seg[VQF_STREAM_MAX_SEG];
    uint64_t    per_layer;   /* 每层可驱逐字节合计 */
    uint64_t    total;       /* 数据区字节（报告/日志用） */
} g_vqf_stream;

static long vqf_stream_rss_kb_impl(void) {
#ifndef _WIN32
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    long kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) { sscanf(line + 6, "%ld", &kb); break; }
    }
    fclose(f);
    return kb;
#else
    /* Windows：Working Set（当前驻留物理页）≈ POSIX VmRSS。GetProcessMemoryInfo
     * 需 psapi.lib（构建脚本 -lpsapi 已提供；MinGW 亦可在运行时经 K32GetProcessMemoryInfo
     * kernel32 解析）。返回 kB。 */
    PROCESS_MEMORY_COUNTERS pmc;
    SIZE_T cb = sizeof(pmc);
    HANDLE h = GetCurrentProcess();
    BOOL ok = FALSE;
    ok = GetProcessMemoryInfo(h, &pmc, cb);
    return ok ? (long)(pmc.WorkingSetSize / 1024) : 0;
#endif
}

long vqf_stream_rss_kb(void) {
    return vqf_stream_rss_kb_impl();
}

int vqf_stream_state(int *keep, int *nl, int *nseg,
                     unsigned long long *per_layer, int *cold) {
    if (!g_vqf_stream.map || g_vqf_stream.keep <= 0) return 0;
    if (keep) *keep = g_vqf_stream.keep;
    if (nl) *nl = g_vqf_stream.nl;
    if (nseg) *nseg = g_vqf_stream.nseg;
    if (per_layer) *per_layer = g_vqf_stream.per_layer;
    if (cold) *cold = g_vqf_stream.cold;
    return 1;
}

/* setup：在 vqf_load 挂载完成后调用（幂等；新加载会重建表）。
 * 只接受明文 VQF；加密（VQF_FLAG_ENC）与整文件单遍 SM4-CTR/HMAC 语义冲突，
 * 逐层驱逐会丢失 COW 解密页 → 拒绝并告警。 */
/* 专家窗口（EW，定义见下）：setup 中优先尝试接管。前置声明。 */
static int vqf_ew_setup(STModelWeights *w, const VQFHeader *h);

void vqf_stream_setup(STModelWeights *w) {
    memset(&g_vqf_stream, 0, sizeof(g_vqf_stream));
    g_vqf_stream.nl = w && w->cfg.n_layers > 0 ? w->cfg.n_layers : 0;
    if (!w || !w->vqf_map || w->vqf_map_len < sizeof(VQFHeader)) return;
    const VQFHeader *h = (const VQFHeader *)w->vqf_map;
    if (h->flags & VQF_FLAG_ENC) {
        fprintf(stderr, "[VQF-STREAM/EW] REFUSED: encrypted VQF (SM4-CTR whole-file "
                        "single-pass + HMAC) cannot be layer/experts-evicted; "
                        "plaintext VQF only\n");
        return;
    }
    /* 专家窗口（EW）优先：VLLM_EW / VLLM_EW_MASK 激活后接管层入口钩子 */
    if (vqf_ew_setup(w, h)) {
        g_vqf_stream.map  = w->vqf_map;   /* advance 的 map 校验走 EW 分支 */
        g_vqf_stream.keep = 0;
        return;
    }
    const char *e = getenv("VLLM_VQF_STREAM");
    int keep = (e && e[0]) ? atoi(e) : 0;
    if (keep <= 0 || g_vqf_stream.nl <= 0) return;          /* 默认关闭 */

    g_vqf_stream.map  = w->vqf_map;
    g_vqf_stream.keep = keep;
    g_vqf_stream.fd   = g_vqf_stream_fd;
    const char *lg = getenv("VLLM_VQF_STREAM_LOG");
    g_vqf_stream.log = (lg && lg[0] == '1');
    const char *cd = getenv("VLLM_VQF_STREAM_COLD");
    g_vqf_stream.cold = (cd && cd[0] == '1');

    int nl = g_vqf_stream.nl;
    size_t dir_off = (sizeof(VQFHeader) + 63) & ~(size_t)63;
    int n = 0;
    uint64_t pl = 0;
    for (uint32_t i = 0; i < h->n_tensors && n < VQF_STREAM_MAX_SEG; i++) {
        const VQFTensor *te = (const VQFTensor *)((const uint8_t *)w->vqf_map +
                                                  dir_off +
                                                  (size_t)i * sizeof(VQFTensor));
        const char *nm = te->name;
        /* 允许集 = 逐层 RMSNorm + 逐层量化投影；lm/embed/final_norm/vision 常驻 */
        int evict = 0;
        if (strcmp(nm, "attn_norm") == 0 || strcmp(nm, "ffn_norm") == 0 ||
            strcmp(nm, "q_norm") == 0 || strcmp(nm, "k_norm") == 0) evict = 1;
        else if (strncmp(nm, "q8_", 3) == 0 && strcmp(nm, "q8_lm") != 0) evict = 1;
        else if (strncmp(nm, "q4_", 3) == 0 && strcmp(nm, "q4_lm") != 0) evict = 1;
        else if (strncmp(nm, "x8_", 3) == 0) evict = 1;
        if (!evict) continue;
        /* 几何门控：整层堆叠且单层字节恒定（rows%nl==0 且 bytes%nl==0） */
        if ((uint64_t)te->rows % (uint64_t)nl != 0) continue;
        if ((uint64_t)te->rows / (uint64_t)nl == 0) continue;
        if (te->bytes % (uint64_t)nl != 0) continue;
        g_vqf_stream.seg[n].off  = te->offset;
        g_vqf_stream.seg[n].step = te->bytes / (uint64_t)nl;
        pl += g_vqf_stream.seg[n].step;
        n++;
    }
    if (n == 0) {
        fprintf(stderr, "[VQF-STREAM] REFUSED: no per-layer tensors found "
                        "(format geometry mismatch)\n");
        g_vqf_stream.map = NULL;
        return;
    }
    g_vqf_stream.nseg = n;
    g_vqf_stream.per_layer = pl;
    g_vqf_stream.total = h->file_len > h->data_offset
                         ? (uint64_t)(h->file_len - h->data_offset) : 0;
    fprintf(stderr,
            "[VQF-STREAM] enabled keep=%d nl=%d segs=%d per-layer=%.1fMB "
            "data=%.1fMB resident~%.1fMB rss=%ldkB\n",
            keep, nl, n, pl / 1048576.0, g_vqf_stream.total / 1048576.0,
            (g_vqf_stream.total > (uint64_t)nl * pl)
                ? (g_vqf_stream.total - (uint64_t)nl * pl) / 1048576.0 : 0.0,
            vqf_stream_rss_kb_impl());
}

/* 页对齐的 madvise/VirtualUnlock（start 下取整 / end 上取整；文件偏移 == 映射偏移） */
static void vqf_stream_advise(int advice, uint64_t lo, uint64_t hi,
                              const void *map) {
    if (hi <= lo) return;
#ifndef _WIN32
    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) ps = 4096;
    uintptr_t psz = (uintptr_t)ps;
#else
    uintptr_t psz = 4096;
#endif
    uintptr_t a = (uintptr_t)lo / psz * psz;
    uintptr_t b = ((uintptr_t)hi + psz - 1) / psz * psz;
    if (b <= a) return;

#ifndef _WIN32
    madvise((void *)((const uint8_t *)map + a), (size_t)(b - a), advice);
#else
    if (advice == MADV_DONTNEED) {
        /* Windows x86 内存释放方案：VirtualUnlock 会把指定的虚拟内存页从当前工作集(Working Set)中移除，
         * 从而让系统认为该部分内存已被释放，实现了等效于 Linux MADV_DONTNEED 的可测物理内存收益。 */
        VirtualUnlock((void *)((const uint8_t *)map + a), (size_t)(b - a));
    } else if (advice == MADV_WILLNEED) {
        /* Windows 无 madvise(WILLNEED) 等价物：动态解析 PrefetchVirtualMemory
         * （Win8+，kernel32），把页从 standby/文件预取进工作集（≈ Linux WILLNEED）。
         * 需 SeIncreaseQuota 类特权，缺失/不可用则静默跳过（退回按需 fault，行为
         * 与未启用一致）。函数指针静态缓存，只探测一次（与 moe_q4_simd_ok 同款）。 */
        typedef BOOL (WINAPI *pf_pvm_t)(HANDLE, ULONG_PTR, const void *, ULONG);
        static pf_pvm_t pf_pvm = (pf_pvm_t)0;
        if (pf_pvm == (pf_pvm_t)0) {
            pf_pvm = (pf_pvm_t)(uintptr_t)GetProcAddress(
                GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
            if (!pf_pvm) pf_pvm = (pf_pvm_t)-1;   /* -1 = 不可用（探测一次） */
        }
        if (pf_pvm != (pf_pvm_t)-1) {
            struct { void *va; SIZE_T n; } rng;   /* 布局 == WIN32_MEMORY_RANGE_ENTRY */
            rng.va = (void *)((const uint8_t *)map + a);
            rng.n  = (SIZE_T)(b - a);
            pf_pvm(GetCurrentProcess(), 1, &rng, 0);
        }
    }
#endif
}

/* ================================================================
 * 专家窗口驻留（EW，expert window）——热专家常驻 + 冷专家页级流转
 *
 * 把分层驻留的粒度从"整层"降到"专家页"（Kestrel 正轨，§10.17 原型落地）：
 *   VLLM_EW=<H>          启用，热集 = 每层 rank<H 的专家（合成占位热集）
 *   VLLM_EW_MASK=<path>  显式热掩码文件，每行 "l:e1,e2,..."（真实热度喂入）
 *   VLLM_EW_LOG=1        层边界日志（rss + 热专家数）
 *
 * 有界窗口语义（§10.17 关键结论：必须驱逐层内"全体非热页"才能收敛）：
 *   - 常驻永不驱逐：router/attn/norm/lm/embed；
 *   - 热专家 gate/up 常驻（掩码内不驱逐）；
 *   - 进入层 l 时（层循环入口钩子，先于本层权重读取）：
 *       a) gate/up：驱逐本层全部非热专家段（连续冷段合并，每段 1 次 PTE 操作）；
 *       b) down  ：整层驱逐——q4 4x4 / q8 8x8 下专家沿行组列窗碎片散布、页被
 *                  邻专家共享，整层清除最省 syscall；热 down 由本层 FFN 计算自然重驻。
 *   - A≡C：驱逐只改 PTE，页由文件/页缓存按字节重建；FFN 零改动、数值不变。
 *
 * 几何（q4 4x4 tiled / q8 8x8 tiled 双支持，2026-09-09 扩展）：
 *   gate/up：rows/层 = ff = ne×moe_ffn，cols = d；专家段 = moe_ffn 行 →
 *            连续字节 expB = bytes(moe_ffn×d)（q4 = 0.844MB/216 页、q8 = 1.67MB/
 *            408 页，均整页对齐、行组不跨专家段 → 1 段驱逐）；
 *   down  ：rows/层 = d，cols = ff；行组字节 = (ff/32)×组块（q4 72B/4行组、
 *            q8 272B/8行组）；专家列窗碎片化 → 整层驱逐。
 * ================================================================ */
typedef struct {
    int active, nl, ne, moe_ffn, d, ff;
    int is_q8;                     /* 1=q8 8x8 tiled；0=q4 4x4 tiled */
    uint64_t layerB, expB;         /* gate/up/down 每层字节 / gate-up 每专家段字节 */
    uint64_t g_off, u_off, d_off;  /* q4/q8_gate/up/down 张量文件偏移 */
    uint8_t *mask;                 /* nl×ne bits（按层 128 位） */
    int log;
} VQFEW;
static VQFEW g_ew;

static uint64_t vqf_q4_bytes(uint64_t n) { return (n + 31) / 32 * 18; }
static uint64_t vqf_q8_bytes(uint64_t n) { return (n + 31) / 32 * 34; }

static int vqf_ew_hot(const VQFEW *e, int l, int x) {
    /* mask 按位打包：每层 ne 位 = ne/8 字节（vqf_ew_setup 分配 nl*ne/8 字节） */
    return (e->mask[(size_t)l * (e->ne / 8) + (size_t)x / 8] >> (x & 7)) & 1;
}

static const VQFTensor *vqf_ew_find(const VQFHeader *h, const char *want) {
    const VQFTensor *dir = (const VQFTensor *)((const uint8_t *)h + sizeof(VQFHeader));
    for (uint32_t i = 0; i < h->n_tensors; i++)
        if (strcmp(dir[i].name, want) == 0) return &dir[i];
    return NULL;
}

/* 解析 VLLM_EW 掩码；几何/格式校验通过才激活。返回 1 = EW 接管。
 * 布局：q4_4x4（VQF_FLAG_Q4_4X4）与 q8_8x8（VQF_FLAG_Q8_8X8）双支持——
 * 两种 repack 均不改变每元素字节（q4 18B/32elt、q8 34B/32elt），专家段几何
 * 同构：gate/up 每专家 = me 行 × d 列，行组（q4=4 / q8=8）不跨专家段
 * （me%行组==0，§4 整除实证），专家段整页对齐（q4 gate/up 216 页 / q8 408 页）。
 * q4 族张量优先（与引擎权重加载同序），无则回落 q8 族。 */
static int vqf_ew_setup(STModelWeights *w, const VQFHeader *h) {
    memset(&g_ew, 0, sizeof(g_ew));
    const char *env = getenv("VLLM_EW");
    const char *mf  = getenv("VLLM_EW_MASK");
    if (!(env && env[0]) && !(mf && mf[0])) return 0;   /* 默认关 */
    if (h->flags & VQF_FLAG_ENC) {
        fprintf(stderr, "[VQF-EW] REFUSED: 加密 VQF（EW 需明文按层驱逐）\n");
        return 0;
    }
    const VQFArch *a = &h->arch;
    int ne = (int)a->n_experts, me = (int)a->moe_ffn;
    int d = (int)a->dim, ff = (int)a->ffn_dim, nl = (int)a->n_layers;
    if (ne <= 0 || me <= 0 || d <= 0 || ff <= 0 || nl <= 0 || (uint64_t)ne * me != (uint64_t)ff) {
        fprintf(stderr, "[VQF-EW] REFUSED: 非 MoE 或几何不成立\n");
        return 0;
    }
    const VQFTensor *G = vqf_ew_find(h, "q4_gate");
    const VQFTensor *U = vqf_ew_find(h, "q4_up");
    const VQFTensor *D = vqf_ew_find(h, "q4_down");
    int is_q8 = 0;
    if (!G || !U || !D) {
        G = vqf_ew_find(h, "q8_gate");
        U = vqf_ew_find(h, "q8_up");
        D = vqf_ew_find(h, "q8_down");
        is_q8 = 1;
    }
    if (!G || !U || !D) { fprintf(stderr, "[VQF-EW] REFUSED: 无 q4/q8_gate/up/down\n"); return 0; }
    uint32_t need = is_q8 ? VQF_FLAG_Q8_8X8 : VQF_FLAG_Q4_4X4;
    if (!(h->flags & need)) {
        fprintf(stderr, "[VQF-EW] REFUSED: %s 请求但文件缺布局 flag 0x%x\n",
                is_q8 ? "q8_8x8" : "q4_4x4", (unsigned)need);
        return 0;
    }
    uint64_t layerB = is_q8 ? vqf_q8_bytes((uint64_t)ff * (uint64_t)d)
                            : vqf_q4_bytes((uint64_t)ff * (uint64_t)d);
    uint64_t expB   = is_q8 ? vqf_q8_bytes((uint64_t)me * (uint64_t)d)
                            : vqf_q4_bytes((uint64_t)me * (uint64_t)d);
    /* 几何自检：整层堆叠 + 专家段整页对齐（q4 gate/up 216 页 / q8 408 页） */
    if ((uint64_t)G->rows != (uint64_t)nl * ff || (uint64_t)D->rows != (uint64_t)nl * d ||
        (uint64_t)G->bytes != layerB * nl || (uint64_t)D->bytes != layerB * nl ||
        (expB & 4095) != 0) {
        fprintf(stderr, "[VQF-EW] REFUSED: 张量几何与整层堆叠假设不符\n");
        return 0;
    }
    g_ew.is_q8 = is_q8;
    g_ew.nl = nl; g_ew.ne = ne; g_ew.moe_ffn = me; g_ew.d = d; g_ew.ff = ff;
    g_ew.layerB = layerB; g_ew.expB = expB;
    g_ew.g_off = G->offset; g_ew.u_off = U->offset; g_ew.d_off = D->offset;
    const char *lg = getenv("VLLM_EW_LOG");
    g_ew.log = (lg && lg[0] == '1');
    g_ew.mask = (uint8_t *)calloc(1, ((size_t)nl * ne + 7) / 8);
    if (!g_ew.mask) { fprintf(stderr, "[VQF-EW] OOM mask\n"); return 0; }

    /* 掩码来源：显式文件优先（每行 "l:e1,e2,..."），否则 VLLM_EW=H → rank<H */
    if (mf && mf[0]) {
        FILE *f = fopen(mf, "r");
        if (!f) { fprintf(stderr, "[VQF-EW] mask 文件打不开: %s\n", mf); free(g_ew.mask); g_ew.mask = NULL; return 0; }
        char line[512]; int nlines = 0;
        while (fgets(line, sizeof(line), f)) {
            char *c = strchr(line, '#'); if (c) *c = 0;
            int l = 0;
            if (sscanf(line, " %d:", &l) != 1) continue;
            if (l < 0 || l >= nl) continue;
            char *p = strchr(line, ':'); if (!p) continue;
            p++;
            char *tk = strtok(p, ",; \t\r\n");
            while (tk) {
                int e = atoi(tk);
                if (e >= 0 && e < ne) g_ew.mask[(size_t)l * (ne / 8) + (size_t)e / 8] |= (uint8_t)(1u << (e & 7));
                tk = strtok(NULL, ",; \t\r\n");
            }
            nlines++;
        }
        fclose(f);
        if (nlines == 0) { fprintf(stderr, "[VQF-EW] mask 文件无有效行: %s\n", mf); free(g_ew.mask); g_ew.mask = NULL; return 0; }
    } else {
        int H = atoi(env);
        if (H <= 0 || H > ne) { fprintf(stderr, "[VQF-EW] 非法 H=%d\n", H); free(g_ew.mask); g_ew.mask = NULL; return 0; }
        for (int l = 0; l < nl; l++)
            for (int e = 0; e < H; e++) g_ew.mask[(size_t)l * (ne / 8) + (size_t)e / 8] |= (uint8_t)(1u << (e & 7));
    }
    g_ew.active = 1;
    fprintf(stderr, "[VQF-EW] enabled %s nl=%d ne=%d moe_ffn=%d per-layer=%.1fMB "
                    "exp=%.3fMB(%llu页) data_off g=%llu u=%llu d=%llu\n",
            is_q8 ? "q8_8x8" : "q4_4x4", nl, ne, me, layerB / 1048576.0,
            expB / 1048576.0,
            (unsigned long long)(expB / 4096),
            (unsigned long long)g_ew.g_off, (unsigned long long)g_ew.u_off,
            (unsigned long long)g_ew.d_off);
    return 1;
}

/* 层入口：驱逐本层全体非热专家页（gate/up 按专家段；down 整层）。幂等。 */
static void vqf_ew_layer(const STModelWeights *w, int layer) {
    VQFEW *e = &g_ew;
    if (layer < 0 || layer >= e->nl) return;
    const uint8_t *map = (const uint8_t *)w->vqf_map;
    int ne = e->ne;
    /* a) gate/up：连续冷段合并驱逐 */
    for (int t = 0; t < 2; t++) {
        uint64_t base = (t == 0 ? e->g_off : e->u_off) + (uint64_t)layer * e->layerB;
        int rs = -1;
        for (int x = 0; x <= ne; x++) {
            int hot = (x < ne) ? vqf_ew_hot(e, layer, x) : 1;
            if (!hot) { if (rs < 0) rs = x; }
            else if (rs >= 0) {
                uint64_t off = base + (uint64_t)rs * e->expB;
                vqf_stream_advise(MADV_DONTNEED, off,
                                  off + (uint64_t)(x - rs) * e->expB, map);
                rs = -1;
            }
        }
    }
    /* b) down：整层（专家列窗碎片化，页共享 → 整层清除） */
    vqf_stream_advise(MADV_DONTNEED, e->d_off + (uint64_t)layer * e->layerB,
                      e->d_off + (uint64_t)(layer + 1) * e->layerB, map);
    /* c) 跨层软流水（可选，VLLM_EW_PREFETCH_LAYER=1，默认关）：层 l 计算期预读
     * l+1 整层专家页（gate/up/down 共 3×layerB），把 l+1 的缺页与 l 的计算重叠。
     * 注意：router 未跑、选路未知 → 只能整层过取（= §10.23 down 整层带同判据，
     * 稀疏窗口下过取有害）；Windows 代理（同步回填）预期负收益，x86 仅作对拍；
     * 真异步内核读入（Linux madvise WILLNEED）的目标端才可能有益。 */
    if (layer + 1 < e->nl) {
        static int pf_layer = -1;
        if (pf_layer < 0) {
            const char *pe = getenv("VLLM_EW_PREFETCH_LAYER");
            pf_layer = (pe && pe[0] == '1') ? 1 : 0;
        }
        if (pf_layer) {
            uint64_t lb1 = (uint64_t)(layer + 1) * e->layerB;
            vqf_stream_advise(MADV_WILLNEED, e->g_off + lb1,
                              e->g_off + lb1 + e->layerB, map);
            vqf_stream_advise(MADV_WILLNEED, e->u_off + lb1,
                              e->u_off + lb1 + e->layerB, map);
            vqf_stream_advise(MADV_WILLNEED, e->d_off + lb1,
                              e->d_off + lb1 + e->layerB, map);
        }
    }
    if (e->log && (layer == 0 || layer % 8 == 0 || layer == e->nl - 1)) {
        int cnt = 0;
        for (int x = 0; x < ne; x++) cnt += vqf_ew_hot(e, layer, x);
        fprintf(stderr, "[VQF-EW] l=%d/%d rss=%ldkB hot=%d\n",
                layer, e->nl, vqf_stream_rss_kb_impl(), cnt);
    }
}

/* 层循环钩子：进入 layer 时调用。
 *   keep=1：逐出 layer-1 → 任意时刻 ≈ 当前层 + 预读窗；
 *   keep=N：逐出 layer-N（前 N 层仍常驻）。
 * 幂等/无副作用：未启用、模型换载、Windows 下直接返回。 */
void vqf_stream_layer_advance(STModelWeights *w, int layer) {
    const void *map = g_vqf_stream.map;
    if (!map) return;
    if (!w || w->vqf_map != map) { g_vqf_stream.map = NULL; return; }
    if (g_ew.active) { vqf_ew_layer(w, layer); return; }   /* 专家窗口优先 */
    int nl = g_vqf_stream.nl, keep = g_vqf_stream.keep, nseg = g_vqf_stream.nseg;
    if (nl <= 0 || nseg <= 0 || keep <= 0) return;

    int evict_l = layer - keep;
    if (evict_l >= 0 && evict_l < nl) {
        for (int s = 0; s < nseg; s++) {
            uint64_t lo = g_vqf_stream.seg[s].off +
                          (uint64_t)evict_l * g_vqf_stream.seg[s].step;
            vqf_stream_advise(MADV_DONTNEED, lo, lo + g_vqf_stream.seg[s].step, map);
        }
#ifndef _WIN32
        /* COLD 档：连带丢弃页缓存（posix_fadvise DONTNEED），使驱逐后的再
         * 访问必须从 eMMC 真正重读 → 度量"数据来源换盘"的最坏代价。 */
        if (g_vqf_stream.cold && g_vqf_stream.fd >= 0) {
            long ps = sysconf(_SC_PAGESIZE);
            if (ps <= 0) ps = 4096;
            uintptr_t psz = (uintptr_t)ps;
            for (int s = 0; s < nseg; s++) {
                uint64_t lo = g_vqf_stream.seg[s].off +
                              (uint64_t)evict_l * g_vqf_stream.seg[s].step;
                uint64_t hi = lo + g_vqf_stream.seg[s].step;
                uintptr_t a = (uintptr_t)lo / psz * psz;
                uintptr_t b = ((uintptr_t)hi + psz - 1) / psz * psz;
                if (b > a)
                    posix_fadvise(g_vqf_stream.fd, (off_t)a, (off_t)(b - a),
                                  POSIX_FADV_DONTNEED);
            }
        }
#endif
    }
    if (layer + 1 < nl) {
        for (int s = 0; s < nseg; s++) {
            uint64_t lo = g_vqf_stream.seg[s].off +
                          (uint64_t)(layer + 1) * g_vqf_stream.seg[s].step;
            vqf_stream_advise(MADV_WILLNEED, lo, lo + g_vqf_stream.seg[s].step, map);
        }
    }

    if (g_vqf_stream.log) {
        int evict_l = layer - keep;
        if (layer == 0 || layer == nl - 1 || layer % 4 == 0) {
            fprintf(stderr, "[VQF-STREAM] l=%d/%d keep=%d evict_l=%d "
                            "rss=%ldkB per-layer=%.1fMB\n",
                    layer, nl, keep, evict_l, vqf_stream_rss_kb_impl(),
                    g_vqf_stream.per_layer / 1048576.0);
        }
    }
}

/* ================================================================
 * 选路后专家预取（gate-first，VLLM_EW_PREFETCH=1）——仅明文 VQF q4_4x4
 *
 * 动机：EW 分层驻留下，本层冷专家页在层入口被 DONTNEED 逐出，FFN 要等
 * top-k 选路后才确定读哪 8 个专家 → 若等并行 worker 逐页访问再 fault，缺页
 * 延迟全部落在 gate/up 计算关键路径上（§10.22 结论：gu 已由计算型转为缺页/
 * 带宽主导）。本钩子在选路（top-k + softmax）完成后立刻对"本层实际选中"的
 * 页发 WILLNEED，让缺页重驻与下方并行 gu 计算（≈11ms/层）重叠：
 *   - gate/up：每个选中专家整段（expB 连续、216 页整页对齐）→ 2×tk 次提示；
 *     预取字节 100% 被 gu 消费（0% 过取，推荐默认档）。
 *   - down  ：整层列带 1 次提示可选（VLLM_EW_PREFETCH_DOWN=1 才开，默认关）：
 *     q4 4x4 down 专家沿 4 行组列窗碎片化、页被邻专家共享 → 整层 WILLNEED 约
 *     3.4× 过取（108MB/层 vs top-8 实际消费 ~30MB），Windows 实测为负收益；
 *     仅在目标端 async 内核读入（Linux madvise）足够便宜时按需打开。
 * 原语：Linux = madvise(MADV_WILLNEED)；Windows = PrefetchVirtualMemory（动态
 * 解析，特权缺失静默降级）。只改页表/页缓存、浮点序零改动 → A≡C 位级一致。
 * 几何门控同 EW：整层堆叠 + expB 整页对齐；q8 布局 / 加密 VQF 自动拒绝。
 * ================================================================ */
typedef struct {
    const void *map;            /* 已验证的 w->vqf_map（模型换载后失效重解） */
    int nl, ne;
    int log, down;              /* down: 整层 down 带 WILLNEED（默认关） */
    uint64_t layerB, expB;      /* 每层字节 / 每专家 gate-up 段字节 */
    uint64_t g_off, u_off, d_off;
} VQFPF;
static VQFPF g_pf;

/* 惰性解析几何（首调 / 模型换载后）。失败置空 → 调用侧 no-op。 */
static void vqf_ffn_prefetch_resolve(const STModelWeights *w) {
    g_pf.map = NULL;
    g_pf.log = 0;
    g_pf.down = 0;
    const char *e = getenv("VLLM_EW_PREFETCH");
    if (!(e && e[0] == '1')) return;              /* 默认关 */
    const char *lg = getenv("VLLM_EW_PREFETCH_LOG");
    g_pf.log = (lg && lg[0] == '1');
    const char *dn = getenv("VLLM_EW_PREFETCH_DOWN");
    g_pf.down = (dn && dn[0] == '1');             /* 默认仅 gate/up（0 过取） */
    if (!w || !w->vqf_map || w->vqf_map_len < sizeof(VQFHeader)) return;
    const VQFHeader *h = (const VQFHeader *)w->vqf_map;
    if ((h->flags & VQF_FLAG_ENC) || !(h->flags & VQF_FLAG_Q4_4X4)) {
        if (g_pf.log) fprintf(stderr, "[VQF-PF] REFUSED: 仅明文 q4_4x4 VQF\n");
        return;
    }
    const VQFTensor *G = vqf_ew_find(h, "q4_gate");
    const VQFTensor *U = vqf_ew_find(h, "q4_up");
    const VQFTensor *D = vqf_ew_find(h, "q4_down");
    if (!G || !U || !D) {
        if (g_pf.log) fprintf(stderr, "[VQF-PF] REFUSED: 无 q4_gate/up/down\n");
        return;
    }
    const VQFArch *a = &h->arch;
    int nl = (int)a->n_layers, ne = (int)a->n_experts;
    int me = (int)a->moe_ffn, d = (int)a->dim, ff = (int)a->ffn_dim;
    uint64_t layerB = vqf_q4_bytes((uint64_t)ff * (uint64_t)d);
    uint64_t expB   = vqf_q4_bytes((uint64_t)me * (uint64_t)d);
    if (ne <= 0 || me <= 0 || d <= 0 || ff <= 0 || nl <= 0 ||
        (uint64_t)ne * me != (uint64_t)ff || (expB & 4095) != 0 ||
        (uint64_t)G->rows != (uint64_t)nl * ff ||
        (uint64_t)G->bytes != layerB * nl ||
        (uint64_t)D->rows != (uint64_t)nl * d) {
        if (g_pf.log) fprintf(stderr, "[VQF-PF] REFUSED: 几何与整层堆叠假设不符\n");
        return;
    }
    g_pf.map = w->vqf_map; g_pf.nl = nl; g_pf.ne = ne;
    g_pf.layerB = layerB; g_pf.expB = expB;
    g_pf.g_off = G->offset; g_pf.u_off = U->offset; g_pf.d_off = D->offset;
    if (g_pf.log)
        fprintf(stderr, "[VQF-PF] enabled nl=%d ne=%d exp=%.3fMB(%llu页) "
                        "down=%d\n",
                nl, ne, expB / 1048576.0,
                (unsigned long long)(expB / 4096), g_pf.down);
}

/* MoE FFN top-k 选路完成后调用（vllm_safetensors.c st_moe_ffn_sparse_q4）。 */
void vqf_ffn_prefetch(const STModelWeights *w, int layer,
                      const int *sel, int tk) {
    if (!w || w->vqf_map != g_pf.map) vqf_ffn_prefetch_resolve(w);
    if (!g_pf.map || layer < 0 || layer >= g_pf.nl || !sel || tk <= 0) return;
    const uint8_t *map = (const uint8_t *)w->vqf_map;
    const uint64_t lb = (uint64_t)layer * g_pf.layerB;
    for (int j = 0; j < tk; j++) {
        int e = sel[j];
        if (e < 0 || e >= g_pf.ne) continue;
        const uint64_t eb = (uint64_t)e * g_pf.expB;
        vqf_stream_advise(MADV_WILLNEED, g_pf.g_off + lb + eb,
                          g_pf.g_off + lb + eb + g_pf.expB, map);
        vqf_stream_advise(MADV_WILLNEED, g_pf.u_off + lb + eb,
                          g_pf.u_off + lb + eb + g_pf.expB, map);
    }
    if (g_pf.down)
        vqf_stream_advise(MADV_WILLNEED, g_pf.d_off + lb,
                          g_pf.d_off + lb + g_pf.layerB, map);
    if (g_pf.log && (layer == 0 || layer % 8 == 0 || layer == g_pf.nl - 1)) {
        fprintf(stderr, "[VQF-PF] l=%d/%d sel0=%d tk=%d rss=%ldkB\n",
                layer, g_pf.nl, sel[0], tk, vqf_stream_rss_kb_impl());
    }
}

/* ================================================================
 * 运行时驻留控制（serve 内存驻留策略 vllm_res_policy 调用）
 *
 * vqf_stream_setup 建好的 seg 表按层覆盖全部可驱逐权重段；本组 API 把
 * "空闲/内存紧张时从高向低逐层释放、只剩初始层"变成可直接调用的动作：
 *   1) 只改 PTE（MADV_DONTNEED），页由文件按字节重建 → 推理位级一致不变；
 *   2) 可逆：升档/再次访问自然缺页回驻，无需显式恢复；
 *   3) 依赖 setup（明文 VQF + 启动开 VLLM_VQF_STREAM），未启用时为 no-op。
 * ================================================================ */

int vqf_stream_active(void) {
    return (g_vqf_stream.map != NULL && g_vqf_stream.nl > 0 &&
            g_vqf_stream.nseg > 0);
}

/* 显式释放 layer > keep 的全部权重段物理页（保留 0..keep 层 + embed/lm/
 * final_norm/vision 等常驻段）。keep=0 即"只剩初始层"，空闲降级的终点档。 */
void vqf_stream_evict_above(int keep) {
    const void *map = g_vqf_stream.map;
    if (!map) return;
    int nl = g_vqf_stream.nl, nseg = g_vqf_stream.nseg;
    if (nl <= 0 || nseg <= 0) return;
    if (keep < 0) keep = 0;
    if (keep >= nl - 1) return;

    for (int s = 0; s < nseg; s++) {
        const VQFStreamSeg *sg = &g_vqf_stream.seg[s];
        for (int l = keep + 1; l < nl; l++) {
            uint64_t lo = sg->off + (uint64_t)l * sg->step;
            vqf_stream_advise(MADV_DONTNEED, lo, lo + sg->step, map);
        }
    }

    fprintf(stderr, "[VQF-STREAM] runtime evict above layer %d (nl=%d) "
                    "rss=%ldkB per-layer=%.1fMB\n",
            keep, nl, vqf_stream_rss_kb_impl(),
            g_vqf_stream.per_layer / 1048576.0);
    fflush(stderr);
}

/* 预读 0..keep 层权重段（MADV_WILLNEED）。升档后的可选加速：缺省不调用
 * 也可（访问自然缺页回驻，数值不变），供"恢复全驻留"档作为提示性预读。 */
void vqf_stream_willneed_to(int keep) {
    const void *map = g_vqf_stream.map;
    if (!map) return;
    int nl = g_vqf_stream.nl, nseg = g_vqf_stream.nseg;
    if (nl <= 0 || nseg <= 0) return;
    if (keep < 0) keep = 0;
    if (keep >= nl) keep = nl - 1;

    for (int s = 0; s < nseg; s++) {
        const VQFStreamSeg *sg = &g_vqf_stream.seg[s];
        for (int l = 0; l <= keep; l++) {
            uint64_t lo = sg->off + (uint64_t)l * sg->step;
            vqf_stream_advise(MADV_WILLNEED, lo, lo + sg->step, map);
        }
    }

}

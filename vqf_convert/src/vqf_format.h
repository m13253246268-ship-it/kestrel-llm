/* ================================================================
 * vqf_format.h - VQF 磁盘格式定义（平台无关，零引擎依赖）。
 *
 * 独立头的原因：离线签名工具 vllm_vqf_sign 与 vllm_crypto_test 需要引用
 * 文件格式结构（VQFHeader/VQFTensor/VQFSig），但不能引入引擎头
 * （vllm_safetensors.h → vllm_platform.h 仅 aarch64）。所有三方
 * （引擎 vqf.c / 离线工具 / KAT 测试）共用本头，格式无漂移。
 *
 * 布局（v2 稠密 / v3 MoE，头部同为 448B、目录偏移不变）：
 *   [0..448)  VQFHeader（含 VQFSig 签名块；v3=MoE 时 arch 尾部多 4 字段）
 *   [448..)   目录 entries（VQFTensor × n_tensors，64B/个）
 *   [data_offset..file_len)  数据区（64B 对齐；VQF_FLAG_ENC 时为 SM4-CTR 密文）
 *   [file_len..)  尾部：ENC→HMAC-SM3 tag(32B)；明文→FNV-1a checksum(8B)
 * ================================================================ */
#ifndef VQF_FORMAT_H
#define VQF_FORMAT_H

#include <stdint.h>
#include <stddef.h>

#define VQF_MAGIC        0x57465146u   /* "VQFW" */
#define VQF_VERSION      2             /* v2: 头部追加 VQFSig（SM2 签名） */
#define VQF_VERSION_MOE  3             /* v3: MoE（qwen3_moe）——arch 尾部追加
                                        * n_experts/moe_ffn/top_k/shared_experts；
                                        * 仅用于含 MoE 字段的文件，旧版加载器（v2
                                        * 检查）会拒绝，引擎 M2 起按 version 分支 */
#define VQF_HEADER_BYTES 4096          /* 头+目录固定 4KB，页对齐 */
#define VQF_ALIGN        64            /* 数据区内张量 64B 对齐 */

/* 布局 flags：转换时固化的内核布局变体，加载时须与当前引擎一致 */
#define VQF_FLAG_Q8_8X8   (1u<<0)      /* q8_* 为 8x8 tiled 布局 (VLLM_Q8_8X8=1) */
#define VQF_FLAG_Q4_4X4   (1u<<1)      /* q4_* 为 4x4 布局 (repack_q4_0_4x4) */
#define VQF_FLAG_X8       (1u<<2)      /* x8_* 8x8l decode 副本存在 */
#define VQF_FLAG_Q8BUF_Q4 (1u<<3)      /* q8_* 实际存 pre-unpacked Q4 int8 (q4i) */
#define VQF_FLAG_G256     (1u<<4)      /* q8_* 为 G=256 分组量化 */
#define VQF_FLAG_EMB_F16  (1u<<5)      /* token_embed 存 F16（省 ~0.6GB，读取时转 F32） */
#define VQF_FLAG_VISION   (1u<<6)      /* 含 vision 张量（多模态）：v_q8_ 系列 */
#define VQF_FLAG_ENC      (1u<<7)      /* 数据区 SM4-CTR 加密 + 尾部 HMAC-SM3 tag（VQF-Enc） */
#define VQF_FLAG_SIGNED   (1u<<8)      /* 含 SM2 签名（sig 块有效，供应链防替换） */
#define VQF_FLAG_MOE      (1u<<9)      /* MoE 模型族（qwen3_moe）：arch.n_experts>0 */

/* SM2 签名用户标识 ID_A（Z_A 绑定，签名/验签双方必须一致） */
#define VQF_SIG_ID_MAX    32
#define VQF_SM2_ID        "vllm-shs-vqf"

/* 张量量化类型（qtype） */
#define VQF_QT_F32     0x01
#define VQF_QT_Q8_0    0x02
#define VQF_QT_Q4_0    0x03
#define VQF_QT_Q4_8X8L 0x04
#define VQF_QT_F16     0x05          /* IEEE 754 半精度（embed） */

/* 推理所需模型属性（VQF 自包含，不再依赖 config.json/safetensors index） */
typedef struct {
    uint32_t dim, n_layers, n_heads, n_kv_heads, head_dim, ffn_dim, vocab_size;
    uint32_t max_seq_len;
    float    rope_theta, norm_eps;
    uint32_t bos_id, eos_id;
    uint32_t has_q_norm, has_mrope, head_dim_full, kv_lora_rank;
    uint32_t has_vision, vis_depth, vis_hidden, vis_heads, vis_ffn;
    uint32_t vis_patch, vis_temporal, vis_merge, vis_out_dim, vis_in_chan;
    uint32_t vis_max_pos, vis_ds_count, mrope_n_sec;
    int32_t  vis_ds_idx[4], mrope_sections[4];
    uint32_t vision_start_id, vision_end_id, image_token_id, video_token_id;
    /* MoE（version==VQF_VERSION_MOE / VQF_FLAG_MOE 时有效；稠密恒 0）。
     * ffn_dim 语义不变：= n_experts*moe_ffn（gate/up 每层堆叠行数/列数），
     * 使"整层堆叠→量化→repack"链路对 MoE 逐字节复用（专家行区段整除 8/4）。 */
    uint32_t n_experts, moe_ffn, top_k, shared_experts;
} VQFArch;

/* SM2 签名块（VQF_FLAG_SIGNED 时有效）。
 * digest 为「明文逻辑内容」的 SM3 摘要 D：
 *   D = SM3( header_canonical ‖ directory ‖ plaintext_data )
 * header_canonical = VQFHeader 中 file_len=0 且 sig 块全 0 的头（与 HMAC 一致）。
 * 签名 = SM2(priv, D)（Z_A 绑定 ID_A=VQF_SM2_ID），r/s 各 32B。 */
typedef struct {
    uint8_t  pub[64];              /* 签名者 SM2 公钥 (x||y)，大端，诊断用 */
    uint8_t  r[32];                /* SM2 签名分量 r */
    uint8_t  s[32];                /* SM2 签名分量 s */
    uint8_t  digest[32];           /* 被签名的摘要 D */
    uint8_t  id[VQF_SIG_ID_MAX];   /* SM2 用户标识 ID_A（自描述） */
    uint32_t id_len;               /* id 实际字节数 */
    uint32_t rsvd;                 /* 保留（对齐） */
} VQFSig;                          /* 64+32+32+32+32+4+4 = 200 B */

typedef struct {
    uint32_t magic;        /* VQF_MAGIC */
    uint32_t version;      /* VQF_VERSION */
    uint32_t flags;        /* VQF_FLAG_* */
    uint32_t n_tensors;
    VQFArch  arch;         /* 模型属性（自包含） */
    uint64_t data_offset;  /* 数据区起始（页对齐） */
    uint64_t file_len;     /* 文件总长（不含尾部 checksum/tag） */
    uint8_t  enc_iv[16];   /* VQF-Enc：SM4-CTR 初始计数器（VQF_FLAG_ENC 时有效） */
    uint8_t  enc_rsvd[16]; /* 保留 */
    VQFSig   sig;          /* SM2 签名块（VQF_FLAG_SIGNED 时有效） */
} VQFHeader;               /* v2 约 432B，目录 entries 随后，共 4096B */

typedef struct {
    char     name[32];     /* 内部张量名：q8_q / q4_gate / x8_o / token_embed ... */
    uint32_t qtype;        /* VQF_QT_* */
    uint32_t rows, cols;   /* 几何（诊断用） */
    uint64_t offset;       /* 文件内数据偏移（64B 对齐） */
    uint64_t bytes;
} VQFTensor;

/* 布局守卫：引擎/工具/测试三方共用本头，若字段改动导致尺寸漂移立即编译失败。
 * v2 头 432B → 追加 MoE 4×u32 后 448B；448 仍 64B 对齐 → 目录偏移 dir_off 不变，
 * 老稠密文件（v2）在新旧加载器下目录/数据偏移一致，天然向后兼容。 */
_Static_assert(sizeof(VQFSig)   == 200, "VQFSig layout drift");
_Static_assert(sizeof(VQFHeader) == 448, "VQFHeader layout drift (expect 448 after MoE fields)");
_Static_assert(sizeof(VQFTensor) == 64,  "VQFTensor layout drift");

#endif /* VQF_FORMAT_H */

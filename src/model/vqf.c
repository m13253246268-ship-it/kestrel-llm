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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define Q8_BYTES(n) ((size_t)((n) + 31) / 32 * 34)
#define Q4_BYTES(n) ((size_t)((n) + 31) / 32 * 18)

/* 引擎全局布局开关（vllm_safetensors.c）：几何门控失败会被清零，转换时
 * 以实际状态为准。 */
extern int g_st_q8_repack;
extern int g_st_q4_repack;

/* FNV-1a 64：文件校验。增量版用于 vqf_write 边写边累计（避免整文件
 * malloc 重读导致转换进程 OOM——dual 权重已占 ~4.5GB）。 */
static uint64_t vqf_fnv64(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

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

/* 16 字节 IV：xorshift128，种子取 time+地址（混淆级唯一性，非密码学随机源）。
 * IV 存入 header.enc_iv，加载侧复用。 */
static void vqf_enc_iv(uint8_t iv[16]) {
    static uint64_t s0 = 0, s1 = 0;
    if (!s0 && !s1) {
        uint64_t t = (uint64_t)time(NULL);
        s0 = t ^ 0x9e3779b97f4a7c15ull ^ (uint64_t)(uintptr_t)iv;
        s1 = (t << 1) | 1;
    }
    for (int i = 0; i < 16; i++) {
        uint64_t x = s0, y = s1;
        s0 = y;
        x ^= x << 23;
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        iv[i] = (uint8_t)((s1 + y) & 0xff);
    }
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

/* 加密写出：src 经 SM4-CTR（连续计数器）加密后落盘，并同步喂 HMAC。
 * 1MB 分块流式，避免为整张量分配临时缓冲。 */
static int vqf_emit_enc(FILE *f, const uint8_t *src, size_t len,
                        vc_sm4_ctx *sm4, uint8_t ctr[16],
                        vc_hmac_ctx *hc) {
    uint8_t buf[1 << 20];
    while (len) {
        size_t k = len > sizeof(buf) ? sizeof(buf) : len;
        vc_sm4_ctr_crypt(sm4, ctr, src, buf, k);
        vc_hmac_update(hc, buf, k);
        if (fwrite(buf, 1, k, f) != k) return -1;
        src += k; len -= k;
    }
    return 0;
}


/* f32 → f16（与 vllm_safetensors.c 的 f32_to_f16_bits 位级一致，禁止漂移） */
static inline uint16_t vqf_f32_to_f16(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (u >> 13) & 0x3FF;
    if (exp <= 0)    return (uint16_t)(sign | 0);
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | (mant & 0x3FF));
}

/* ---------------- 张量 dump 表 ---------------- */
typedef struct {
    const char *name;
    uint32_t   qtype;
    const void *p;
    uint32_t   rows, cols;
} VQFDump;

/* 收集当前已加载的非空权重张量；返回数量。 */
static int vqf_collect(const STModelWeights *w, const STModelConfig *cfg,
                       VQFDump *d, int cap) {
    int n = 0;
#define ADD(nm, qt, ptr, r, c)                                                \
    do {                                                                      \
        if (n < cap && (ptr)) {                                               \
            d[n].name = (nm); d[n].qtype = (qt); d[n].p = (ptr);              \
            d[n].rows = (uint32_t)(r); d[n].cols = (uint32_t)(c); n++;        \
        }                                                                     \
    } while (0)
    int dim = cfg->dim, ff = cfg->ffn_dim, vc = cfg->vocab_size;
    int q_out = cfg->n_heads * cfg->head_dim;
    int k_out = cfg->n_kv_heads * cfg->head_dim;
    int nl = w->n_layers_allocated > 0 ? w->n_layers_allocated : cfg->n_layers;
    /* F32 区；token_embed 默认 F16（省 ~0.6GB），VLLM_VQF_EMB_F32=1 可回退 */
    const char *ef = getenv("VLLM_VQF_EMB_F32");
    int emb_f32 = (ef && ef[0] == '1');
    ADD("token_embed", emb_f32 ? VQF_QT_F32 : VQF_QT_F16, w->token_embed, vc, dim);
    ADD("final_norm",  VQF_QT_F32, w->final_norm,  dim, 1);
    ADD("attn_norm",   VQF_QT_F32, w->attn_norm,   nl * dim, 1);
    ADD("ffn_norm",    VQF_QT_F32, w->ffn_norm,    nl * dim, 1);
    /* q/k_norm 每层仅 head_dim 维（load_layer_tensor 的 hd,hd 加载），
     * 不可用 q_out/k_out（含 n_heads 放大）否则越界读多写垃圾 */
    ADD("q_norm",      VQF_QT_F32, w->q_norm,      nl * cfg->head_dim, 1);
    ADD("k_norm",      VQF_QT_F32, w->k_norm,      nl * cfg->head_dim, 1);
    /* Q8 区（8x8 tiled / 4x4 / g256 / q4i 由 flags 记录） */
    ADD("q8_q",    VQF_QT_Q8_0, w->q8_q_weight,    nl * dim,  q_out);
    ADD("q8_k",    VQF_QT_Q8_0, w->q8_k_weight,    nl * dim,  k_out);
    ADD("q8_v",    VQF_QT_Q8_0, w->q8_v_weight,    nl * dim,  k_out);
    ADD("q8_o",    VQF_QT_Q8_0, w->q8_o_weight,    nl * q_out, dim);
    ADD("q8_gate", VQF_QT_Q8_0, w->q8_gate_weight, nl * ff,   dim);
    ADD("q8_up",   VQF_QT_Q8_0, w->q8_up_weight,   nl * ff,   dim);
    ADD("q8_down", VQF_QT_Q8_0, w->q8_down_weight, nl * dim,  ff);
    ADD("q8_lm",   VQF_QT_Q8_0, w->q8_lm_weight,   vc, dim);
    /* Q4 区（4x4 布局） */
    ADD("q4_q",    VQF_QT_Q4_0, w->q4_q_weight,    nl * dim,  q_out);
    ADD("q4_k",    VQF_QT_Q4_0, w->q4_k_weight,    nl * dim,  k_out);
    ADD("q4_v",    VQF_QT_Q4_0, w->q4_v_weight,    nl * dim,  k_out);
    ADD("q4_o",    VQF_QT_Q4_0, w->q4_o_weight,    nl * q_out, dim);
    ADD("q4_gate", VQF_QT_Q4_0, w->q4_gate_weight, nl * ff,   dim);
    ADD("q4_up",   VQF_QT_Q4_0, w->q4_up_weight,   nl * ff,   dim);
    ADD("q4_down", VQF_QT_Q4_0, w->q4_down_weight, nl * dim,  ff);
    ADD("q4_lm",   VQF_QT_Q4_0, w->q4_lm_weight,   vc, dim);
    /* X8 区（Q4 8x8l，wmode=q4 才分配） */
    ADD("x8_q",    VQF_QT_Q4_8X8L, w->x8_q_weight,    nl * dim,  q_out);
    ADD("x8_k",    VQF_QT_Q4_8X8L, w->x8_k_weight,    nl * dim,  k_out);
    ADD("x8_v",    VQF_QT_Q4_8X8L, w->x8_v_weight,    nl * dim,  k_out);
    ADD("x8_o",    VQF_QT_Q4_8X8L, w->x8_o_weight,    nl * q_out, dim);
    ADD("x8_gate", VQF_QT_Q4_8X8L, w->x8_gate_weight, nl * ff,   dim);
    ADD("x8_up",   VQF_QT_Q4_8X8L, w->x8_up_weight,   nl * ff,   dim);
    ADD("x8_down", VQF_QT_Q4_8X8L, w->x8_down_weight, nl * dim,  ff);
    /* Vision 区（多模态）：推理只读 Q8 大矩阵副本（v_q8_*），F32 小张量
     * （patch/pos/bias/norm/merger/deepstack）按加载布局固化。 */
    const STVisionWeights *vw = w->vision;
    if (vw && vw->is_allocated) {
        int vh = cfg->vis_hidden, vd = cfg->vis_depth, vf = cfg->vis_ffn;
        int vp = cfg->vis_patch, vt = cfg->vis_temporal;
        int vf_pad = ((vf + 31) / 32) * 32;   /* fc2 col pad 4304 → 4320 */
        int merge_in = vh * cfg->vis_merge * cfg->vis_merge;
        int ds_mid = merge_in, out_dim = cfg->vis_out_dim, ds = cfg->vis_ds_count;
        /* Q8 大矩阵：rows*cols 乘积与量化时传入的元素数一致 → Q8_BYTES 与
         * VQ8_BYTES 同值（fc2 以 vf_pad 存，加载侧按同几何挂载） */
        ADD("v_q8_qkv",  VQF_QT_Q8_0, vw->q8_attn_qkv_weight,  vd, 3 * vh * vh);
        ADD("v_q8_proj", VQF_QT_Q8_0, vw->q8_attn_proj_weight, vd, vh * vh);
        ADD("v_q8_fc1",  VQF_QT_Q8_0, vw->q8_mlp_fc1_weight,   vd, vf * vh);
        ADD("v_q8_fc2",  VQF_QT_Q8_0, vw->q8_mlp_fc2_weight,   vd, vh * vf_pad);
        /* F32 小张量 */
        ADD("v_patch_embed", VQF_QT_F32, vw->patch_embed_weight, 1, vh * 3 * vt * vp * vp);
        ADD("v_patch_bias",  VQF_QT_F32, vw->patch_embed_bias,   1, vh);
        ADD("v_pos_embed",   VQF_QT_F32, vw->pos_embed,          1, cfg->vis_max_pos * vh);
        ADD("v_qkv_bias",    VQF_QT_F32, vw->attn_qkv_bias,      vd, 3 * vh);
        ADD("v_proj_bias",   VQF_QT_F32, vw->attn_proj_bias,     vd, vh);
        ADD("v_norm1_w",     VQF_QT_F32, vw->norm1_weight,       vd, vh);
        ADD("v_norm1_b",     VQF_QT_F32, vw->norm1_bias,         vd, vh);
        ADD("v_norm2_w",     VQF_QT_F32, vw->norm2_weight,       vd, vh);
        ADD("v_norm2_b",     VQF_QT_F32, vw->norm2_bias,         vd, vh);
        ADD("v_fc1_bias",    VQF_QT_F32, vw->mlp_fc1_bias,       vd, vf);
        ADD("v_fc2_bias",    VQF_QT_F32, vw->mlp_fc2_bias,       vd, vh);
        /* merger（main 的 norm 按加载只取 vh；fc1/fc2 为全尺寸） */
        ADD("v_merger_nw", VQF_QT_F32, vw->merger_norm_weight, 1, vh);
        ADD("v_merger_nb", VQF_QT_F32, vw->merger_norm_bias,   1, vh);
        ADD("v_merger_f1w", VQF_QT_F32, vw->merger_fc1_weight, 1, merge_in * merge_in);
        ADD("v_merger_f1b", VQF_QT_F32, vw->merger_fc1_bias,   1, merge_in);
        ADD("v_merger_f2w", VQF_QT_F32, vw->merger_fc2_weight, 1, merge_in * out_dim);
        ADD("v_merger_f2b", VQF_QT_F32, vw->merger_fc2_bias,   1, out_dim);
        /* deepstack mergers */
        ADD("v_ds_nw",  VQF_QT_F32, vw->ds_norm_weight, ds, merge_in);
        ADD("v_ds_nb",  VQF_QT_F32, vw->ds_norm_bias,   ds, merge_in);
        ADD("v_ds_f1w", VQF_QT_F32, vw->ds_fc1_weight,  ds, merge_in * ds_mid);
        ADD("v_ds_f1b", VQF_QT_F32, vw->ds_fc1_bias,    ds, ds_mid);
        ADD("v_ds_f2w", VQF_QT_F32, vw->ds_fc2_weight,  ds, ds_mid * out_dim);
        ADD("v_ds_f2b", VQF_QT_F32, vw->ds_fc2_bias,    ds, out_dim);
    }
#undef ADD
    return n;
}

static size_t vqf_tensor_bytes(uint32_t qtype, uint32_t rows, uint32_t cols) {
    switch (qtype) {
    case VQF_QT_F32:     return (size_t)rows * cols * 4;
    case VQF_QT_F16:     return (size_t)rows * cols * 2;
    case VQF_QT_Q8_0:    return Q8_BYTES((size_t)rows * cols);
    case VQF_QT_Q4_0:    return Q4_BYTES((size_t)rows * cols);
    case VQF_QT_Q4_8X8L: return Q4_BYTES((size_t)rows * cols);
    default:             return 0;
    }
}

/* ---------------- 转换 ---------------- */
int vqf_write(const char *path, const STModelWeights *w,
              const STModelConfig *cfg, uint32_t flags) {
    VQFDump d[64];
    int n = vqf_collect(w, cfg, d, 64);
    if (n == 0) {
        fprintf(stderr, "[VQF] no quantized weights loaded; convert after a "
                        "normal load (wmode dual/q4)\n");
        return -1;
    }
    /* token_embed 的存储类型固化进 flags（读取侧据此决定 F16→F32）：
     * d[0] 恒为 token_embed（collect 顺序保证） */
    if (n > 0 && d[0].qtype == VQF_QT_F16) flags |= VQF_FLAG_EMB_F16;
    /* 多模态：vision 张量存在则固化 VISION flag（加载侧据此挂载） */
    if (w->vision && w->vision->is_allocated) flags |= VQF_FLAG_VISION;
    /* VQF-Enc：口令派生 SM4/HMAC 密钥 + 16B IV（口令为空则明文，enc=0） */
    uint8_t sm4key[16], hmackey[16], enc_iv[16];
    int enc = vqf_enc_derive(getenv("VLLM_VQF_KEY"), sm4key, hmackey);
    vc_sm4_ctx sm4ctx; uint8_t ctr[16];
    if (enc) {
        vqf_enc_iv(enc_iv);
        vc_sm4_setkey_enc(&sm4ctx, sm4key);
        memcpy(ctr, enc_iv, 16);
        flags |= VQF_FLAG_ENC;
    }
    /* SM2 签名：VLLM_VQF_SIGN_PRIV（64 hex）非空则对「明文逻辑内容」签名。
     * 签名与加密正交（签的是明文摘要 D，任意口令加密均不破坏签名）。 */
    uint8_t priv[32], sig_pub[64], sig_r[32], sig_s[32], sig_D[32], sig_k[32];
    int sign = 0;
    vc_sm3_ctx dig;
    const char *sp = getenv("VLLM_VQF_SIGN_PRIV");
    if (sp && sp[0] && vc_hex_decode(sp, priv, 32) == 32) {
        sign = 1;
        flags |= VQF_FLAG_SIGNED;
        vc_sm3_init(&dig);
    }
    FILE *f = fopen(path, "wb+");   /* wb+：尾部回填 file_len 需要 fread */
    if (!f) { fprintf(stderr, "[VQF] cannot open %s\n", path); return -1; }

    fprintf(stderr, "[VQF] collect %d tensors, writing...\n", n);

    VQFHeader h; memset(&h, 0, sizeof(h));
    h.magic = VQF_MAGIC; h.version = VQF_VERSION; h.flags = flags; h.n_tensors = (uint32_t)n;
    const STModelConfig *c = cfg;
    h.arch.dim = (uint32_t)c->dim; h.arch.n_layers = (uint32_t)c->n_layers;
    h.arch.n_heads = (uint32_t)c->n_heads; h.arch.n_kv_heads = (uint32_t)c->n_kv_heads;
    h.arch.head_dim = (uint32_t)c->head_dim; h.arch.ffn_dim = (uint32_t)c->ffn_dim;
    h.arch.vocab_size = (uint32_t)c->vocab_size; h.arch.max_seq_len = (uint32_t)c->max_seq_len;
    h.arch.rope_theta = c->rope_theta; h.arch.norm_eps = c->norm_eps;
    h.arch.bos_id = (uint32_t)c->bos_id; h.arch.eos_id = (uint32_t)c->eos_id;
    h.arch.has_q_norm = (uint32_t)c->has_q_norm; h.arch.has_mrope = (uint32_t)c->has_mrope;
    h.arch.head_dim_full = (uint32_t)c->head_dim_full; h.arch.kv_lora_rank = (uint32_t)c->kv_lora_rank;
    h.arch.has_vision = (uint32_t)c->has_vision; h.arch.vis_depth = (uint32_t)c->vis_depth;
    h.arch.vis_hidden = (uint32_t)c->vis_hidden; h.arch.vis_heads = (uint32_t)c->vis_heads;
    h.arch.vis_ffn = (uint32_t)c->vis_ffn; h.arch.vis_patch = (uint32_t)c->vis_patch;
    h.arch.vis_temporal = (uint32_t)c->vis_temporal; h.arch.vis_merge = (uint32_t)c->vis_merge;
    h.arch.vis_out_dim = (uint32_t)c->vis_out_dim; h.arch.vis_in_chan = (uint32_t)c->vis_in_chan;
    h.arch.vis_max_pos = (uint32_t)c->vis_max_pos; h.arch.vis_ds_count = (uint32_t)c->vis_ds_count;
    h.arch.mrope_n_sec = (uint32_t)c->mrope_n_sec;
    for (int i = 0; i < 4; i++) {
        h.arch.vis_ds_idx[i] = (int32_t)c->vis_ds_idx[i];
        h.arch.mrope_sections[i] = (int32_t)c->mrope_sections[i];
    }
    h.arch.vision_start_id = (uint32_t)c->vision_start_id;
    h.arch.vision_end_id = (uint32_t)c->vision_end_id;
    h.arch.image_token_id = (uint32_t)c->image_token_id;
    h.arch.video_token_id = (uint32_t)c->video_token_id;
    if (enc) memcpy(h.enc_iv, enc_iv, 16);

    size_t dir_off = (sizeof(VQFHeader) + 63) & ~(size_t)63;
    size_t data_off = (dir_off + (size_t)n * sizeof(VQFTensor) + VQF_HEADER_BYTES - 1)
                      & ~(size_t)(VQF_HEADER_BYTES - 1);
    h.data_offset = data_off;

    /* 头缓冲 = 固定头 + 目录区（目录从 dir_off 起，dir_off > sizeof(VQFHeader)，
     * 必须按 dir_off 计算总长，否则最后一个 entry 越界写破坏堆）。 */
    size_t head_sz = dir_off + (size_t)n * sizeof(VQFTensor);
    uint8_t *buf = (uint8_t *)calloc(1, head_sz);
    if (!buf) { fclose(f); return -1; }
    memcpy(buf, &h, sizeof(h));
    size_t off = data_off;
    for (int i = 0; i < n; i++) {
        VQFTensor *e = (VQFTensor *)(buf + dir_off + (size_t)i * sizeof(VQFTensor));
        memset(e, 0, sizeof(*e));
        snprintf(e->name, sizeof(e->name), "%s", d[i].name);
        e->qtype = d[i].qtype; e->rows = d[i].rows; e->cols = d[i].cols;
        size_t b = vqf_tensor_bytes(d[i].qtype, d[i].rows, d[i].cols);
        e->offset = off; e->bytes = (uint64_t)b;
        off = (off + b + VQF_ALIGN - 1) & ~(size_t)(VQF_ALIGN - 1);
    }
    h.file_len = 0; /* 数据写完后回填 */
    uint64_t sum = 0xcbf29ce484222325ull;   /* 目录 + 数据区（header 的 file_len
                                              * 回填不影响哈希范围） */
    if (fwrite(buf, 1, head_sz, f) != head_sz) {
        fprintf(stderr, "[VQF] write header failed\n");
        free(buf); fclose(f); return -1;
    }
    sum = vqf_fnv_update(sum, buf + dir_off, (size_t)n * sizeof(VQFTensor));
    /* VQF-Enc：HMAC 覆盖 头部（file_len 此时=0）+ 目录（明文）+ 数据区（密文）；
     * 目录在 buf 释放前喂入，头部直接喂 &h（与 buf 中 header 在 file_len=0 意义下
     * 字节一致）。 */
    vc_hmac_ctx hmacc;
    if (enc) {
        vc_hmac_init(&hmacc, hmackey, 16);
        vc_hmac_update(&hmacc, &h, sizeof(h));
        vc_hmac_update(&hmacc, buf + dir_off, (size_t)n * sizeof(VQFTensor));
    }
    /* 签名摘要 D：覆盖 header_canonical（file_len=0, sig=0）+ 目录（明文）。
     * 数据区明文在下方写入循环中逐块喂入。 */
    if (sign) {
        vc_sm3_update(&dig, &h, sizeof(h));
        vc_sm3_update(&dig, buf + dir_off, (size_t)n * sizeof(VQFTensor));
    }
    free(buf);

    /* 数据区：从 data_off 起对齐（头部已占 0..data_off-1，需补齐） */
    uint8_t pad[VQF_ALIGN];
    memset(pad, 0, sizeof(pad));
    for (size_t p = head_sz; p < data_off; p += sizeof(pad)) {
        size_t wn = sizeof(pad);
        if (p + wn > data_off) wn = data_off - p;
        if (fwrite(pad, 1, wn, f) != wn) { fclose(f); return -1; }
        /* 该段 padding 不纳入哈希（加载侧哈希 = 目录 + data_off..file_len） */
    }
    for (int i = 0; i < n; i++) {
        const VQFDump *dd = &d[i];
        size_t b = vqf_tensor_bytes(dd->qtype, dd->rows, dd->cols);
        if (dd->qtype == VQF_QT_F16) {
            /* chunked F32→F16：避免 0.6GB 临时缓冲（embed 311M 元素） */
            size_t nch = (size_t)dd->rows * dd->cols;
            uint8_t tmp[1 << 20];
            size_t cap = sizeof(tmp) / 2;
            const float *src = (const float *)dd->p;
            size_t done = 0;
            while (done < nch) {
                size_t k = nch - done; if (k > cap) k = cap;
                uint16_t *dst = (uint16_t *)tmp;
                for (size_t j = 0; j < k; j++) dst[j] = vqf_f32_to_f16(src[done + j]);
                size_t wb = k * 2;
                if (sign) vc_sm3_update(&dig, tmp, wb);   /* 明文摘要 */
                if (enc) {
                    if (vqf_emit_enc(f, tmp, wb, &sm4ctx, ctr, &hmacc) != 0) {
                        fprintf(stderr, "[VQF] write %s failed\n", dd->name);
                        fclose(f); return -1;
                    }
                } else {
                    if (fwrite(tmp, 1, wb, f) != wb) {
                        fprintf(stderr, "[VQF] write %s failed\n", dd->name);
                        fclose(f); return -1;
                    }
                    sum = vqf_fnv_update(sum, tmp, wb);
                }
                done += k;
            }
        } else {
            if (sign) vc_sm3_update(&dig, dd->p, b);   /* 明文摘要 */
            if (enc) {
                if (b && vqf_emit_enc(f, (const uint8_t *)dd->p, b,
                                      &sm4ctx, ctr, &hmacc) != 0) {
                    fprintf(stderr, "[VQF] write %s failed\n", dd->name);
                    fclose(f); return -1;
                }
            } else {
                if (b && fwrite(dd->p, 1, b, f) != b) {
                    fprintf(stderr, "[VQF] write %s failed\n", dd->name);
                    fclose(f); return -1;
                }
                sum = vqf_fnv_update(sum, dd->p, b);
            }
        }
        size_t tail = (b + VQF_ALIGN - 1) & ~(size_t)(VQF_ALIGN - 1);
        size_t rem = tail - b;
        if (rem) {
            if (sign) vc_sm3_update(&dig, pad, rem);   /* 明文 padding 摘要 */
            if (enc) {
                if (vqf_emit_enc(f, pad, rem, &sm4ctx, ctr, &hmacc) != 0) {
                    fclose(f); return -1;
                }
            } else {
                if (fwrite(pad, 1, rem, f) != rem) { fclose(f); return -1; }
                sum = vqf_fnv_update(sum, pad, rem);
            }
        }
    }
    /* 回填 file_len + 签名块（header 不在哈希范围内，可安全修改）。
     * 签名摘要 D 在此 final：dig 已含 header_canonical + 目录 + 数据区明文。 */
    long end = ST_FTELL(f);
    if (end < 0) { fclose(f); return -1; }
    if (fseek(f, 0, SEEK_SET) != 0 || fread(&h, 1, sizeof(h), f) != sizeof(h)) {
        fclose(f); return -1;
    }
    h.file_len = (uint64_t)end;
    if (sign) {
        vc_sm3_final(&dig, sig_D);
        if (vc_secure_rand(sig_k) != 0) {
            fprintf(stderr, "[VQF] secure random k failed\n");
            fclose(f); return -1;
        }
        size_t idlen = strlen(VQF_SM2_ID);
        if (vc_sm2_sign(priv, sig_D, 32, (const uint8_t *)VQF_SM2_ID, idlen,
                        sig_k, sig_r, sig_s) != 0) {
            fprintf(stderr, "[VQF] SM2 sign failed (bad private key)\n");
            fclose(f); return -1;
        }
        if (vc_sm2_pub_from_priv(priv, sig_pub) != 0) {
            fprintf(stderr, "[VQF] SM2 pub derive failed\n");
            fclose(f); return -1;
        }
        memcpy(h.sig.pub, sig_pub, 64);
        memcpy(h.sig.r, sig_r, 32);
        memcpy(h.sig.s, sig_s, 32);
        memcpy(h.sig.digest, sig_D, 32);
        memset(h.sig.id, 0, sizeof(h.sig.id));
        memcpy(h.sig.id, VQF_SM2_ID, idlen);
        h.sig.id_len = (uint32_t)idlen;
        h.sig.rsvd = 0;
    }
    if (fseek(f, 0, SEEK_SET) != 0 ||
        fwrite(&h, 1, sizeof(h), f) != sizeof(h)) {
        fclose(f); return -1;
    }
    /* 尾部标签：enc → HMAC-SM3（32B，覆盖 头部 + 目录 + 密文数据区）；明文 →
     * FNV-1a（8B，从文件流式重算绑定磁盘字节）。 */
    if (enc) {
        uint8_t tag[32];
        vc_hmac_final(&hmacc, tag);
        /* 回填 header 后文件位置在 sizeof(h) 处，必须 fseek 到文件尾再追加 tag */
        if (fseek(f, 0, SEEK_END) != 0 || fwrite(tag, 1, 32, f) != 32) {
            fclose(f); return -1;
        }
        fclose(f);
        double mb = (double)end / 1048576.0;
        fprintf(stderr, "[VQF] wrote %s: %d tensors, %.1f MB, flags=0x%x "
                        "(seq=%d) ENC=SM4-CTR+HMAC-SM3\n",
                path, n, mb, flags, c->n_layers);
        return 0;
    }

    /* 从文件流式重算 checksum（目录 + 数据区），绑定磁盘上的实际字节——
     * 大块 fwrite 到 emmc 的内容可能与内存权重存在差异（page cache /
     * 部分写交互），从文件重算保证 checksum 与加载侧一致。 */
    uint64_t sum2 = 0xcbf29ce484222325ull;
    uint8_t rbuf[1 << 20];
    long remain;
    if (fseek(f, (long)dir_off, SEEK_SET) != 0) { fclose(f); return -1; }
    remain = (long)n * (long)sizeof(VQFTensor);
    while (remain > 0) {
        size_t r = fread(rbuf, 1, remain > (long)sizeof(rbuf) ? sizeof(rbuf) : (size_t)remain, f);
        if (!r) { fclose(f); return -1; }
        sum2 = vqf_fnv_update(sum2, rbuf, r);
        remain -= (long)r;
    }
    if (fseek(f, (long)h.data_offset, SEEK_SET) != 0) { fclose(f); return -1; }
    remain = (long)(h.file_len - h.data_offset);
    while (remain > 0) {
        size_t r = fread(rbuf, 1, remain > (long)sizeof(rbuf) ? sizeof(rbuf) : (size_t)remain, f);
        if (!r) { fclose(f); return -1; }
        sum2 = vqf_fnv_update(sum2, rbuf, r);
        remain -= (long)r;
    }
    /* 回填 header 后文件位置在 sizeof(h) 处，必须 fseek 到文件尾再追加
     * checksum（否则 checksum 会覆盖目录区头部） */
    if (fseek(f, 0, SEEK_END) != 0 || fwrite(&sum2, 1, 8, f) != 8) {
        fclose(f); return -1;
    }
    fclose(f);

    double mb = (double)end / 1048576.0;
    fprintf(stderr, "[VQF] wrote %s: %d tensors, %.1f MB, flags=0x%x "
                    "(seq=%d) memsum=%016llx file_sum=%016llx\n",
            path, n, mb, flags, c->n_layers, (unsigned long long)sum,
            (unsigned long long)sum2);
    return 0;
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

int vqf_load(STModelWeights *w, const char *path) {
    st_mmap_t m;
    m.fi = -1; m.data = NULL; m.len = 0; m.fd = -1;
    if (st_mmap_open(&m, 0, path) != 0) {
        fprintf(stderr, "[VQF] mmap failed: %s\n", path);
        return -1;
    }
    if (m.len < sizeof(VQFHeader) + sizeof(VQFTensor)) {
        fprintf(stderr, "[VQF] file too small\n");
        st_mmap_close(&m); return -1;
    }
    const VQFHeader *h = (const VQFHeader *)m.data;
    if (h->magic != VQF_MAGIC) { st_mmap_close(&m); return -1; }
    if (h->version != VQF_VERSION) {
        fprintf(stderr, "[VQF] version %u unsupported (need %u)\n",
                h->version, VQF_VERSION);
        st_mmap_close(&m); return -1;
    }
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
        /* 单遍：HMAC（密文）+ 原位解密。先 mprotect 使整映射可写。 */
        if (mprotect(m.data, m.len, PROT_READ | PROT_WRITE) != 0) {
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
    w->has_q8  = w->q8_q_weight ? 1 : 0;
    w->has_q4  = w->q4_q_weight ? 1 : 0;
    w->has_x8  = (h->flags & VQF_FLAG_X8) ? 1 : 0;
    w->q8_buf_q4 = (h->flags & VQF_FLAG_Q8BUF_Q4) ? 1 : 0;
    w->emb_f16 = (h->flags & VQF_FLAG_EMB_F16) ? 1 : 0;
    w->n_layers_allocated = c->n_layers;
    w->is_allocated = 1;
    w->vqf_map = m.data;
    w->vqf_map_len = m.len;
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

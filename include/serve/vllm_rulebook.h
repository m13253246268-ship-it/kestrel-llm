/* SPDX-License-Identifier: Apache-2.0
 * Copyright (C) 2026 裴晓光 and contributors */
/* ================================================================
 * vllm_rulebook.h - 规则包注册中心（工业边缘：预置命名上下文）
 *
 * 定位：把某项"规则手册"预先预处理成一个**命名上下文**（rulebook_id +
 * 名称 + 版本 + 内容哈希 + 租户 + 适用范围），推理时请求体带 rulebook_id
 * 即可直接借用该上下文，只对"规则尾部 + 本轮对话"做增量 prefill，从而
 * 减少上下文计算；同时为每次请求打上 trace_id，并把会话按
 * tenant_id / user_id / session_id 分区，历史互不干扰。
 *
 * 目录布局（<rulebook_dir> 由 --rulebook-dir 指定，默认关）：
 *   rb_<id>.json   元数据（id/名称/版本/hash/租户/scope/模型指纹/token 数）
 *   rb_<id>.txt     源文本（原始字节，避免超长 JSON 字符串的序列化截断）
 *   rbk_<id>_<ver>_<asm>.kv  已预处理 KV 快照（分块选择不同则 asm 不同）
 *
 * 口径（与项目既有前缀复用完全一致，见 vllm_server.h:179-201 的 P4 结论）：
 * 复用是"确定性但近似"——复用路径与全量重算的舍入顺序不同，**不是**全量
 * 重算的位级克隆。禁止对外宣称无损/位级一致；`--no-prefix-kv` 仍是回到
 * 全量位级结果的逃生舱。
 *
 * 失效规则：模型指纹（vllm_rb_model_fp）或版本/内容哈希变化 → 旧 KV 快照
 * 不参与复用（仍注入文本，回退全量 prefill）。规则包版本变更请用新
 * --rulebook-version 覆盖，旧快照按文件名自然隔离，可 vllm_rb_delete 清理。
 * ================================================================ */
#ifndef VLLM_RULEBOOK_H
#define VLLM_RULEBOOK_H

#include "vllm_server.h"

#define VLLM_RB_ID_MAX     64
#define VLLM_RB_NAME_MAX   160
#define VLLM_RB_VER_MAX    32
#define VLLM_RB_TENANT_MAX 64
#define VLLM_RB_SCOPE_MAX  256
#define VLLM_RB_HASH_HEX   17        /* FNV-1a 64 -> 16 hex + NUL */
#define VLLM_RB_MAX_TEXT   (256 * 1024)
#define VLLM_RB_MAX_CHUNKS 512       /* 分块检索的段落数上限（超出按预算截断） */

/* 一个规则包的元数据 + 源文本（text 由 vllm_rb_load 分配，调用方用
 * vllm_rb_free 释放）。带 tag 以便 vllm_server.h 前向声明。 */
typedef struct VLLMRulebook {
    char  id[VLLM_RB_ID_MAX];
    char  name[VLLM_RB_NAME_MAX];
    char  version[VLLM_RB_VER_MAX];
    char  tenant_id[VLLM_RB_TENANT_MAX];
    char  scope[VLLM_RB_SCOPE_MAX];
    char  hash[VLLM_RB_HASH_HEX];        /* 源文本 FNV-1a（16 hex） */
    char  model_fp[VLLM_RB_HASH_HEX];    /* 预处理时的模型指纹（防混用旧检查点） */
    int   n_tokens;                      /* 源文本 token 数（不含模板包裹） */
    int   n_chunks;                      /* 分块数（>0 表示手册超出单次窗口） */
    long  created_s;
    long  updated_s;
    char *text;                          /* malloc'd 源文本；无则 NULL */
} VLLMRulebook;

/* ---------- 摘要/指纹 ---------- */

/* FNV-1a 64 的 16-hex 表示（含结尾 NUL）。 */
void vllm_rb_hash_hex(const void *data, size_t n, char out[VLLM_RB_HASH_HEX]);

/* 当前已加载模型的指纹：model_name + model_dir + config.json 内容 + KV 几何
 * （层数/头数/head_dim/kv_bs）+ 权重文件 (size, mtime)。用于判定规则包 KV
 * 快照是否可用于本次加载的模型（规避"重建引擎后混用旧检查点"这一项目自陈
 * 的唯一真实风险）。模型未加载时输出空串（""）。 */
void vllm_rb_model_fp(const VLLMServerCtx *ctx, char out[VLLM_RB_HASH_HEX]);

/* ---------- 渲染 ---------- */

/* 把规则文本渲染成 Qwen im_chat 的 system 前导块：
 *   "<|im_start|>system\n" + text + "<|im_end|>\n"
 * 返回 malloc'd 字符串（调用方 free）。该块在同一模型下编码稳定，因此可以
 * 作为请求 prompt 的固定前缀被 KV 复用。 */
char *vllm_rb_render_block(const char *text);

/* ---------- 注册中心（磁盘） ---------- */

/* 读 <rulebook_dir>/rb_<id>.json + .txt 到一个 VLLMRulebook。
 * 返回 0 成功；-1 未启用/不存在/解析失败（out 保持清零）。 */
int  vllm_rb_load(const VLLMServerCtx *ctx, const char *id, VLLMRulebook *out);

/* 同上但**不读源文本**（out->text 保持 NULL）：管理面列规则包时避免把每份
 * 手册正文都读进内存。 */
int  vllm_rb_load_meta(const VLLMServerCtx *ctx, const char *id,
                       VLLMRulebook *out);

/* 释放 vllm_rb_load 分配的内容（幂等，可重复调用）。 */
void vllm_rb_free(VLLMRulebook *rb);

/* 只写元数据（rb_<id>.json）与源文本（rb_<id>.txt）。text 必须已就绪。
 * 供 vllm_server.c 的 vllm_rb_build 在 prefill/KV 落盘之后调用。 */
int  vllm_rb_write_meta(const VLLMServerCtx *ctx, const VLLMRulebook *rb,
                        const char *text, char *err, size_t errcap);

/* 删除规则包（元数据 + 源文本 + 该 id 的全部 KV 快照）。返回删除的文件数。
 * id 非法（不在白名单内）时返回 0，且不会触碰任何文件。 */
int  vllm_rb_delete(const VLLMServerCtx *ctx, const char *id);

/* 规则包 ID 白名单判定（[A-Za-z0-9._-]，长度受限）。ID 直接进文件名，管理面
 * 用它把"非法 ID"与"不存在的 ID"区分开（前者 400，后者 404/ok:false）。 */
int  vllm_rb_id_ok(const char *id);

/* ---------- KV 快照路径 ---------- */

/* <rulebook_dir>/rbk_<id>_<version>_<asm>.kv，其中 asm = FNV-1a(实际注入的
 * token 序列) 的低 8 hex。同一规则 + 同一分块选择 → 同一路径 → 可复用。 */
void vllm_rb_kv_path(const VLLMServerCtx *ctx, const VLLMRulebook *rb,
                     const int *ids, int n, char *out, size_t cap);

/* ---------- 分块检索（手册超出窗口时"不要全塞"） ---------- */

/* 按 query 与各段落的 token 重叠度选择要注入的规则文本，使其 token 数不超过
 * budget_tokens（budget 为**渲染后整块**的预算，内部扣除模板包裹开销）。
 * 手册整体不超预算时**返回 NULL** 表示"直接用 rb->text 全塞"（等价于全塞，
 * 保持最优前缀复用），此时仍会填 *out_n_tokens；否则返回 malloc'd 文本
 * （调用方 free）。ctx 需已加载 tokenizer。 */
char *vllm_rb_select_text(VLLMServerCtx *ctx, const VLLMRulebook *rb,
                          const char *query, int budget_tokens,
                          int *out_n_tokens);

/* 分块数（=1 表示手册可整份注入；>1 表示超出单次窗口、会走分块检索）。
 * 只依赖文本切分，不调用 tokenizer。 */
int vllm_rb_chunk_count(const char *text);

#endif /* VLLM_RULEBOOK_H */

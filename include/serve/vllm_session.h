/* SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 裴晓光 and contributors */
/* ================================================================
 * vllm_session.h - 服务端会话存储（工业边缘：会话分区 + 历史审查）
 *
 * 定位：请求体带 session_id 时，服务端为该会话保存消息时间线（JSON 落盘），
 * 客户端只需发本轮 query，历史由服务端重建 —— 既减少客户端上行，也让每轮
 * 的 prompt 前缀与上一轮对齐（配合既有前缀 KV 复用，减少上下文计算）。
 *
 * 用法约定（首轮 vs 后续轮）：
 *   首轮：发完整 messages（含 system）+ session_id；服务端以客户端数组为准并
 *         落盘，无需 query。
 *   后续轮：发 session_id + query（可省略 messages）；服务端取回时间线、拼上
 *         本轮 user 再推理，成功后追加 assistant 并落盘。
 * 会话绑定的 rulebook_id 会被后续轮继承（本次未显式指定时）。
 *
 * 分区隔离：会话以 tenant_id + user_id + session_id 三元组为键；文件名取
 * 三元组的 FNV-1a 哈希（不含任何路径字符，杜绝路径注入），三字段同时明文
 * 写入文件内容，供管理页按租户/用户/规则过滤与审查。
 *
 * 内存策略（对齐 vllm_server.h 的"仅持久化 token 历史"约定）：会话只落盘
 * 消息时间线，**不在内存里常驻多份 KV**；跨会话复用默认隔离 —— 复用只发生
 * 在本会话自己的前缀仍驻留时（ctx->last_ids 的既有 LCP 路径）。
 *
 * 目录：<session_dir> 由 --session-dir 指定（默认关）。
 * ================================================================ */
#ifndef VLLM_SESSION_H
#define VLLM_SESSION_H

#include "vllm_server.h"

#define VLLM_SESS_ID_MAX     96
#define VLLM_SESS_PART_MAX   64
#define VLLM_SESS_RBID_MAX   64

/* 会话文件路径：<dir>/sess_<fnv64(tenant\x1fuser\x1fsid)>.json。
 * 任一为空则用 "anon" 占位（与 L3 用户隔离的 anon 约定一致）。 */
void vllm_sess_path(const VLLMServerCtx *ctx, const char *tenant,
                    const char *user, const char *sid, char *out, size_t cap);

/* 读取会话文件全文（malloc'd，调用方 free）；不存在/未启用返回 NULL。 */
char *vllm_sess_read(const VLLMServerCtx *ctx, const char *tenant,
                     const char *user, const char *sid);

/* 解析会话 JSON 文本为根对象（调用方 vjson_free）。失败返回 NULL。
 * 供调用方一次解析后读取 messages / rulebook_id / updated_s 等字段。 */
VJson *vllm_sess_parse(const char *json_text);

/* 写回会话文件（mkdir -p；覆盖写）。messages 为完整消息时间线数组。
 * created_s 从旧文件继承（无则取当前时间）。last_ms = 本轮推理总耗时（毫秒），
 * 供审查页的"延迟"对照。返回 0 / -1（err 填原因）。 */
int vllm_sess_write(const VLLMServerCtx *ctx, const char *tenant,
                    const char *user, const char *sid, const char *rulebook_id,
                    const VJson *messages, int n_tokens, double last_ms,
                    char *err, size_t errcap);

/* 在既有时间线尾部追加一轮（user=query / assistant=answer），返回新数组
 * （调用方 vjson_free）。old 可为 NULL（新建）。 */
VJson *vllm_sess_append_turns(const VJson *old, const char *query,
                              const char *answer);

/* LRU 清理：目录内 sess_*.json 超过 --session-limit 时删除最旧的若干。 */
void vllm_sess_gc(const VLLMServerCtx *ctx);

#endif /* VLLM_SESSION_H */

/* SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 裴晓光 and contributors */
/* ================================================================
 * vllm_audit.h - 管理面审计日志（追加写 + SM3 哈希链）
 *
 * 定位：会话/规则包管理面的"谁看了、谁导出了、谁删了"留痕。每行一条 JSON：
 *
 *   {"seq":N,"ts":SEC,"actor":"..","role":"..","action":"..","target":"..",
 *    "result":"ok|denied|error","detail":"..","prev":"<64hex>","hash":"<64hex>"}
 *
 *   hash = SM3( 前一行 hash 原始 32 字节 || 本行去掉 ,"hash":... 后的规范文本 )
 *
 * 于是任何一行被改写、插入或删除都会让其后所有行的链路校验失败
 * （vllm_audit_verify 逐行重算 + 比对 prev 链 + seq 连续性），即**篡改可发现**。
 *
 * 诚实边界（务必与部署方式一起理解）：
 *   1) 只提供"篡改可发现性"，**不提供不可否认性**。actor/role 取自请求头
 *      X-Audit-User / X-Audit-Role，本管理面**无身份认证**（见 SECURITY.md：
 *      默认无鉴权的设计前提是"部署在内网/可信网络"）。要真实身份请在**前置
 *      代理**做认证并注入这两个头。
 *   2) 校验覆盖有窗口：文件超过 VLLM_AUDIT_READ_CAP 时只校验尾部窗口，
 *      verify 用 full/partial 明确报告，绝不假装全量通过。
 *   3) 不自动轮转（避免破坏链的连续性）；超过告警水位时打日志提示运维归档。
 * ================================================================ */
#ifndef VLLM_AUDIT_H
#define VLLM_AUDIT_H

#include "vllm_server.h"

/* 读入内存的窗口上限（校验/回看用）。 */
#define VLLM_AUDIT_READ_CAP (8u << 20)
/* 超过此大小打一次告警（提示归档）。 */
#define VLLM_AUDIT_WARN_SIZE (4u << 20)

/* 初始化：从已有日志末行恢复 seq 与链头 hash；路径为空则审计关闭。
 * 须在服务启动前单线程调用。 */
void vllm_audit_init(VLLMServerCtx *ctx);

/* 是否已启用。 */
int vllm_audit_on(const VLLMServerCtx *ctx);

/* 追加一条记录（内部加锁；写失败只打日志，不影响调用方响应）。 */
void vllm_audit_add(VLLMServerCtx *ctx, const char *actor, const char *role,
                    const char *action, const char *target,
                    const char *result, const char *detail);

/* 取最近 n 条（新→旧）拼成 JSON 数组文本（行本身即 JSON 对象，直接拼接）。
 * 返回 malloc'd（调用方 free）；*out_count 填入条数。 */
char *vllm_audit_tail(VLLMServerCtx *ctx, int n, int *out_count);

/* 校验哈希链。
 * 返回 1 = 链内部一致（逐行 SM3 重算 + prev 链 + seq 连续性全部通过）；
 *      0 = 未启用 / 读取失败 / 发现篡改。
 *
 * 输出：
 *   *checked   已校验行数
 *   *bad_seq   首个失败行的 seq（0 = 无）
 *   *full      1 = 校验覆盖了整个文件（未受 VLLM_AUDIT_READ_CAP 截断）
 *   *genesis   1 = 文件首行就是 seq=1（从创世行起完整）
 *   *first_seq 实际校验到的首行 seq
 *   *empty     1 = 日志不存在或没有任何记录（刚启用/刚轮转）——此时返回 1 且
 *              checked=0，**不是**篡改，UI 应显示中性提示而非红色失败。
 *
 * 语义边界（务必区分）：引擎只能证明"**文件内可见的**这条链自洽"。
 *   - 行被改写/插入/删除（非首行）→ 返回 0 并给出 bad_seq；
 *   - 文件**前缀**被整段归档或删除 → 无法与"正常轮转"区分，此时
 *     genesis=0（首行 seq>1）、full=1、返回仍为 1（内部自洽），UI 以黄色
 *     提示"前缀不在文件内"，绝不冒充绿色通过。
 *   要能发现前缀删除，需要外部锚点（如在别处留存 {seq, hash} 摘要）。 */
int vllm_audit_verify(VLLMServerCtx *ctx, int *checked, long *bad_seq, int *full,
                      int *genesis, long *first_seq, int *empty);

/* 日志文件的字节数（0 = 不存在/未启用）。 */
long vllm_audit_size(VLLMServerCtx *ctx);

#endif /* VLLM_AUDIT_H */

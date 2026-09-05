/* ================================================================
 * vllm_sign_keys.h - VQF SM2 供应链签名信任根（公钥）。
 *
 * 设备侧只保存公钥（验签用），签名私钥永不进入设备。
 * 公钥由离线签名工具 vllm_vqf_sign --genkey 生成的私钥推导得到。
 *
 * 信任根解析优先级（vqf_load 验签时）：
 *   1. 运行时环境变量 VLLM_VQF_SIGN_PUB（128 hex，x||y 各 64 hex）；
 *   2. 编译期内嵌 VLLM_SM2_PUB_HEX（本文件，留空=禁用）；
 *   3. 都没有 → 拒绝加载签名 VQF（fail-closed）。
 *
 * 部署发布时：用 vllm_vqf_sign --pub <privhex> 得到公钥，填入下方宏并重编
 * 引擎；或部署后通过 VLLM_VQF_SIGN_PUB 注入（需配合固件/环境变量保护）。
 * ================================================================ */
#ifndef VLLM_SIGN_KEYS_H
#define VLLM_SIGN_KEYS_H

/* 编译期内嵌公钥（128 hex）。留空（未定义）= 仅支持运行时 VLLM_VQF_SIGN_PUB。 */
#ifndef VLLM_SM2_PUB_HEX
/* #define VLLM_SM2_PUB_HEX "0000...0000" */
#endif

#endif /* VLLM_SIGN_KEYS_H */

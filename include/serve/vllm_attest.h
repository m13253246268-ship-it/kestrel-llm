/* ================================================================
 * vllm_attest.h - 可验证推理背书（Verifiable Inference Attestation,
 * 方案 2：基于密码学签名的硬件可信证明）。
 *
 * 场景：医疗诊断、法务合同等高敏感场景需要“责任可追溯、防篡改”的
 * 推理记录。本模块为每次完成的推理响应生成一份 SM2 签名凭证（Proof）。
 *
 * 摘要绑定内容（schema=3，规范见下；验证方无需 tokenizer/聊天模板，
 * 只需请求原文 + 响应原文即可离线复算验签）：
 *
 *   digest = SM3( VLLM-AT-3
 *              || F(model_fp) || F(model_id) || F(device_id)
 *              || F(user) || F(params) || F(n_prompt_tokens)
 *              || F(n_gen_tokens) || F(finish) || F(ts)
 *              || F(body_sha) || F(output_text) )
 *
 *   F(x) = u32be(len(x)) || x（按字节）
 *   body_sha = SM3(客户端原始请求体字节) 的 64-hex —— 逐字节绑定“收到什么
 *             就证明什么”，取证方可对自己保存的请求原文复算比对；
 *   params   = "t=%.4g,p=%.4g,m=%.4g,k=%d,mt=%d,th=%d"（引擎实际采样参数）。
 *             与 schema=2 相比新增 k（top_k 截断，0=关闭）与 th
 *             （enable_thinking）—— 这两项目前既可能来自请求体，也可能来自
 *             serve 级默认（--top-k / VLLM_ENABLE_THINKING），后者不体现在
 *             请求原文里，只绑 body_sha 无法覆盖，故必须进 params；
 *   model_fp 在模型加载成功后固定（SM3(config.json 内容 + 模型目录文件
 *   清单(name+size))），防止“响应属于哪套权重”被事后偷换。
 *
 * 信任链（诚实边界）：
 *   1. 权重锚点：若模型目录含签名 VQF（VLLM_VQF_VERIFY=1），权重级信任根
 *      由 VQF 的 SM2 供应链签名提供；本模块的 model_fp 锚定目录/配置层。
 *   2. 设备背书：设备侧保存的 SM2 私钥对响应摘要签名；验证方持公钥
 *      （vllm_attest.pub）即可离线验证（SM2 ID = "VLLM-ATTEST-1"）。
 *   3. 私钥保护边界：私钥在 POSIX 平台上按 0600 写入并回读校验（Windows 无
 *      该语义，访问控制由目录 ACL 继承提供）。VLLM_ATTEST_DIR 必须位于支持
 *      POSIX 权限的文件系统（ext4/f2fs 等）——在 vfat/exfat 上 chmod 静默
 *      失效、私钥会变成世界可读；此时启动日志打印
 *      "[ATTEST] WARN: ... 同机其它用户可读"，运维须把密钥目录迁到
 *      rootfs/ext4 分区后重新生成密钥。
 *   4. 防伪范围：请求原文、输出文本、模型指纹、参数、计数、finish、ts
 *      任一字段被事后改写都会导致验签失败。
 *
 * 默认关闭（VLLM_ATTEST=1 开启）→ 不影响未开启时的任何性能与行为
 * （A-B-C-D 回退：去掉该环境变量即恢复原状）。
 * 性能影响（开启时）：请求结束各一次 SM3(请求体) + SM3(转录) + SM2 签名
 * （A76 上毫秒级），相对 TTFT(>10s) 可忽略。
 * ================================================================ */
#ifndef VLLM_ATTEST_H
#define VLLM_ATTEST_H

#include <stddef.h>

/* VLLMServerCtx 定义见 vllm_server.h（已改为 tagged struct，此 typedef
 * 只是前向别名；vllm_server.h 中重复 typedef 到同一类型，C11 允许）。 */
typedef struct VLLMServerCtx VLLMServerCtx;

/* 一次待背书请求的转录内容。 */
typedef struct {
    const char *user;        /* 请求 user 字段（可能为 NULL） */
    const char *body_sha;    /* 原始请求体 SM3 的 64-hex（schema=3 必填） */
    const char *text;        /* 输出文本（必须非 NULL） */
    size_t      text_len;
    int         n_prompt_tokens;   /* prompt token 数 */
    int         n_gen_tokens;      /* 生成 token 数 */
    const char *finish;            /* "stop"/"length"（与 API 返回一致） */
    double      temperature, top_p, min_p;   /* 采样参数（绑定进摘要） */
    int         top_k;                     /* top-k 截断（0=关闭），绑定进摘要 */
    int         thinking;                  /* enable_thinking（0/1），绑定进摘要 */
    int         max_tokens;
    long        ts;              /* unix 秒（绑定进摘要，防重放旧证） */
} VAttestReq;

/* 服务启动时调用一次：解析 VLLM_ATTEST / VLLM_ATTEST_DIR，加载或自动生成
 * 设备密钥对（.priv 0600），设置 ctx->attest_on。返回 0 = 已启用；非 0 = 关闭
 * 或初始化失败（调用方应保持 ctx->attest_on=0）。 */
int  vatt_init(VLLMServerCtx *ctx);

/* 每次模型加载成功后调用：把当前模型的指纹固化进 ctx->attest_fp。 */
void vatt_model_ready(VLLMServerCtx *ctx);

/* 对请求体字节计算 body_sha（SM3，输出 64-hex 进 hex[65]）。 */
void vatt_body_digest(const void *body, size_t n, char hex[65]);

/* 1 = 当前可出证（已启用 + 密钥就绪 + 模型指纹已固化）。 */
int  vatt_active(void);

/* 可验证推理的公开参数（供 GET /v1/attest 与管理页展示）：把设备公钥
 * （128 hex + NUL）拷进 out。返回 1 = 密钥就绪（off 会被填 ""），0 = 缓冲区
 * 太小或密钥不可用。公钥是公开信息，可安全下发；私钥始终留在设备。 */
int  vatt_pub_hex(char *out, size_t cap);

/* 验证方必须与设备侧一致的公开约定（写进 /v1/attest 便于离线复算）。 */
#define VATT_ALGO     "SM2-SM3"
#define VATT_SCHEMA   3
#define VATT_USER_ID  "VLLM-ATTEST-1"
#define VATT_MAGIC_V3 "VLLM-AT-3"

/* 对一次请求转录出证：返回 malloc 的 JSON 对象字符串（含 schema/params/
 * body_sha/digest/signature 等全部字段）；未启用或失败返回 NULL。调用方用
 * vjson_parse 嵌入响应 JSON，或包装为 SSE 事件。 */
char *vatt_seal_json(const VAttestReq *r);

/* 设备侧自检：签名→验签往返 + 篡改检测。打印 PASS/FAIL 行，返回 1 = 通过。
 * 用于自动化回归门（VLLM_ATTEST_SELFTEST=1 时由 vatt_init 触发）。 */
int  vatt_selftest(void);

/* 本地自验一条已出证的摘要（sign+verify 回读，逐请求健康检查）。 */
int  vatt_verify_own(const char *digest_hex);

#endif /* VLLM_ATTEST_H */

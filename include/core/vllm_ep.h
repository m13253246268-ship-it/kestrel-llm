/* ================================================================
 * vllm_ep.h — MoE 专家并行（EP）transport 抽象层
 *
 * 阶段一：同机多进程（环回 TCP）→ 阶段二：跨机协同（LAN/IP）。
 * 方案锚点：资料/分布式专家提取_实施方案.md
 *
 * 拓扑（星型）：rank0 = 协调者（listener），rank 1..N-1 = 工作者（connector）。
 * 语义：仅 协调者 ↔ 工作者 两两通信；工作者之间不直接通信（本方案不需要）。
 *
 * 抽象意图（方案硬前提）：上层 EP 逻辑只依赖本头。
 *   - 阶段一（同机多进程）：host = 127.0.0.1；
 *   - 阶段二（跨机协同）  ：host = 板 IP —— 只改一个参数，协议与协议栈零改动。
 *   - 后续若要 shm 零拷贝后端，只需在 vllm_ep.c 内新增 backend，不动本头/上层。
 *
 * 零第三方依赖：仅 winsock2（Windows）/ POSIX socket（Linux）。
 * 阻塞语义：所有原语同步完成，收/发循环到整帧完成（stream 语义，非消息边界）。
 * 端序：握手与载荷按本机端序（本方案各端均为 little-endian；跨端序不在范围）。
 *
 * 错误处理：任一原语失败返回 -1；open 失败返回 NULL。root 侧 accept 有超时
 * （避免工作者未启动时永久悬挂），数据 socket 有 IO 超时。
 * ================================================================ */
#ifndef VLLM_EP_H
#define VLLM_EP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* transport 后端。阶段一/二均为 TCP（差别只在 host），保留枚举作为替换缝。 */
typedef enum {
    VLLM_EP_BACKEND_TCP = 0,
} VllmEpBackend;

/* ---- 协调者（rank 0）---- */
typedef struct VllmEpRoot VllmEpRoot;

/* 监听 port 并接受 nranks-1 个工作者（每个先送 int32 rank 握手）。
 * 成功返回句柄；超时/失败返回 NULL。 */
VllmEpRoot *vllm_ep_root_open(int port, int nranks, VllmEpBackend be);

/* 广播 n 字节到全部工作者（阻塞至全部写完）。0 = 成功。 */
int vllm_ep_root_bcast(VllmEpRoot *r, const void *buf, size_t n);

/* 从指定 rank（1..nranks-1）收 n 字节。0 = 成功。 */
int vllm_ep_root_recv(VllmEpRoot *r, int rank, void *buf, size_t n);

void vllm_ep_root_close(VllmEpRoot *r);

/* ---- 工作者（rank 1..nranks-1）---- */
typedef struct VllmEpWorker VllmEpWorker;

/* 连接协调者（带有限重试，容忍协调者尚未就绪），随后发送 rank 握手。 */
VllmEpWorker *vllm_ep_worker_open(const char *host, int port, int rank,
                                  int nranks, VllmEpBackend be);

int vllm_ep_worker_recv(VllmEpWorker *w, void *buf, size_t n);
int vllm_ep_worker_send(VllmEpWorker *w, const void *buf, size_t n);

void vllm_ep_worker_close(VllmEpWorker *w);

/* 本机端序标记（诊断用）：1 = little-endian。 */
int vllm_ep_little_endian(void);

/* 通信量计数器（进程级累计，含握手；任一指针可传 NULL）。
 * tx/rx 为本进程视角的载荷字节数，msgs 为 send/recv 调用次数。
 * 用途：M4 验收项② "通信量/延迟实测报告"。 */
void vllm_ep_stats(uint64_t *tx_bytes, uint64_t *rx_bytes, uint64_t *msgs);

#ifdef __cplusplus
}
#endif

#endif /* VLLM_EP_H */

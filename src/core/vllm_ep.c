/* ================================================================
 * vllm_ep.c — MoE 专家并行（EP）transport 实现（TCP 后端，零第三方依赖）
 * 见 include/core/vllm_ep.h 的抽象说明与 资料/分布式专家提取_实施方案.md。
 *
 * 设计要点：
 *  - 星型拓扑：root 监听 / worker 连接，握手 = worker 先送 int32 rank。
 *  - stream 语义：send/recv 循环到整帧完成（TCP 无消息边界，故用长度驱动协议）。
 *  - 超时防线：root accept 用 select 限时（避免工作者未起时永久悬挂）；
 *    数据 socket 设 SO_RCVTIMEO/SO_SNDTIMEO，对端异常时返回错误而非死等。
 *  - winsock 初始化幂等（WSAStartup 只做一次，避免与 vllm_http 重复计数）。
 * ================================================================ */
#include "vllm_ep.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET ep_fd;
#define EP_INVALID  INVALID_SOCKET
#define ep_closesock(f) closesocket(f)
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>   /* TCP_NODELAY */
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
typedef int ep_fd;
#define EP_INVALID  (-1)
#define ep_closesock(f) close(f)
#endif

#define VLLM_EP_MAX_RANKS        32
#define VLLM_EP_ACCEPT_TIMEOUT_S 60   /* root 等待全部工作者的总/单次预算 */
#define VLLM_EP_IO_TIMEOUT_S     300  /* 数据 socket 读写超时 */
#define VLLM_EP_CONNECT_RETRY    900  /* worker 连接重试次数（×200ms = 180s）
                                       * 跨机部署时各节点启动时间不同步，窗口要
                                       * 足够宽；同机由驱动脚本保证时序。 */

static int g_wsa_ready = 0;

static int ep_net_init(void) {
#ifdef _WIN32
    if (g_wsa_ready) return 0;
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return -1;
    g_wsa_ready = 1;
#endif
    return 0;
}

int vllm_ep_little_endian(void) {
    uint16_t v = 1;
    return (*(const uint8_t *)&v) == 1;
}

static void ep_set_timeout(ep_fd fd) {
#ifdef _WIN32
    DWORD tv = (DWORD)VLLM_EP_IO_TIMEOUT_S * 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char *)&tv, sizeof(tv));
#else
    struct timeval tv;
    tv.tv_sec = VLLM_EP_IO_TIMEOUT_S;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
}

/* 关闭 Nagle（TCP_NODELAY）。跨机必需：本协议是"每层一次往返"的小消息
 * request/response（广播 4 次小写 + 回传），Nagle 会把第 2..4 次小写压到
 * 首次 ACK 之后（与 delayed-ACK 叠加可达 ~40ms）→ 48 层/token 可累积秒级。
 * 同机环回无此问题，故本地测试看不出差异，跨机才暴露。 */
static void ep_set_nodelay(ep_fd fd) {
    int one = 1;
#ifdef _WIN32
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof(one));
#else
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#endif
}

/* ---- 通信量计数器（M4 验收项②：通信量实测）---- */
static uint64_t g_ep_tx_bytes = 0;
static uint64_t g_ep_rx_bytes = 0;
static uint64_t g_ep_msgs     = 0;

void vllm_ep_stats(uint64_t *tx_bytes, uint64_t *rx_bytes, uint64_t *msgs) {
    if (tx_bytes) *tx_bytes = g_ep_tx_bytes;
    if (rx_bytes) *rx_bytes = g_ep_rx_bytes;
    if (msgs)     *msgs     = g_ep_msgs;
}

/* ---- stream 收发：循环到整帧完成 ---- */
static int ep_send_all(ep_fd fd, const void *buf, size_t n) {
    const char *p = (const char *)buf;
    size_t off = 0;
    while (off < n) {
#ifdef _WIN32
        int chunk = (n - off > (size_t)(1 << 30)) ? (1 << 30) : (int)(n - off);
        int r = send(fd, p + off, chunk, 0);
#else
        ssize_t r = send(fd, p + off, n - off, 0);
#endif
        if (r <= 0) return -1;
        off += (size_t)r;
    }
    g_ep_tx_bytes += (uint64_t)n;
    g_ep_msgs++;
    return 0;
}

static int ep_recv_all(ep_fd fd, void *buf, size_t n) {
    char *p = (char *)buf;
    size_t off = 0;
    while (off < n) {
#ifdef _WIN32
        int chunk = (n - off > (size_t)(1 << 30)) ? (1 << 30) : (int)(n - off);
        int r = recv(fd, p + off, chunk, 0);
#else
        ssize_t r = recv(fd, p + off, n - off, 0);
#endif
        if (r <= 0) return -1;
        off += (size_t)r;
    }
    g_ep_rx_bytes += (uint64_t)n;
    g_ep_msgs++;
    return 0;
}

/* ================================================================
 * 协调者
 * ================================================================ */
struct VllmEpRoot {
    ep_fd listener;
    int   nranks;
    ep_fd peer[VLLM_EP_MAX_RANKS];
};

VllmEpRoot *vllm_ep_root_open(int port, int nranks, VllmEpBackend be) {
    (void)be;
    if (ep_net_init() != 0) return NULL;
    if (nranks < 2 || nranks > VLLM_EP_MAX_RANKS) {
        fprintf(stderr, "[EP] root: 非法 nranks=%d (需 2..%d)\n", nranks, VLLM_EP_MAX_RANKS);
        return NULL;
    }
    VllmEpRoot *r = (VllmEpRoot *)calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->nranks = nranks;
    for (int i = 0; i < VLLM_EP_MAX_RANKS; i++) r->peer[i] = EP_INVALID;

    r->listener = socket(AF_INET, SOCK_STREAM, 0);
    if (r->listener == EP_INVALID) { free(r); return NULL; }
    {
        int one = 1;
#ifdef _WIN32
        setsockopt(r->listener, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));
#else
        setsockopt(r->listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
    }
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons((unsigned short)port);
    if (bind(r->listener, (struct sockaddr *)&a, sizeof(a)) != 0) {
        fprintf(stderr, "[EP] root: bind :%d 失败\n", port);
        ep_closesock(r->listener); free(r); return NULL;
    }
    if (listen(r->listener, VLLM_EP_MAX_RANKS) != 0) {
        fprintf(stderr, "[EP] root: listen 失败\n");
        ep_closesock(r->listener); free(r); return NULL;
    }
    fprintf(stderr, "[EP] root: listening :%d, 等待 %d 个工作者\n", port, nranks - 1);

    int accepted = 0;
    while (accepted < nranks - 1) {
        fd_set rf;
        FD_ZERO(&rf);
        FD_SET(r->listener, &rf);
        struct timeval tv;
        tv.tv_sec = VLLM_EP_ACCEPT_TIMEOUT_S;
        tv.tv_usec = 0;
        int sr = select((int)r->listener + 1, &rf, NULL, NULL, &tv);
        if (sr <= 0) {
            fprintf(stderr, "[EP] root: accept 超时（已接受 %d/%d）\n",
                    accepted, nranks - 1);
            vllm_ep_root_close(r);
            return NULL;
        }
        ep_fd c = accept(r->listener, NULL, NULL);
        if (c == EP_INVALID) continue;
        int32_t rk = -1;
        if (ep_recv_all(c, &rk, sizeof(rk)) != 0) {
            fprintf(stderr, "[EP] root: 握手读取失败\n");
            ep_closesock(c);
            continue;
        }
        if (rk < 1 || rk >= nranks || r->peer[rk] != EP_INVALID) {
            fprintf(stderr, "[EP] root: 非法/重复 rank=%d\n", (int)rk);
            ep_closesock(c);
            continue;
        }
        ep_set_timeout(c);
        ep_set_nodelay(c);
        r->peer[rk] = c;
        accepted++;
        fprintf(stderr, "[EP] root: 接入 rank %d (%d/%d)\n", (int)rk, accepted, nranks - 1);
    }
    fprintf(stderr, "[EP] root: 全部工作者就绪\n");
    return r;
}

int vllm_ep_root_bcast(VllmEpRoot *r, const void *buf, size_t n) {
    if (!r) return -1;
    for (int rk = 1; rk < r->nranks; rk++) {
        if (r->peer[rk] == EP_INVALID) return -1;
        if (ep_send_all(r->peer[rk], buf, n) != 0) {
            fprintf(stderr, "[EP] root: bcast → rank %d 失败\n", rk);
            return -1;
        }
    }
    return 0;
}

int vllm_ep_root_recv(VllmEpRoot *r, int rank, void *buf, size_t n) {
    if (!r || rank < 1 || rank >= r->nranks) return -1;
    if (r->peer[rank] == EP_INVALID) return -1;
    return ep_recv_all(r->peer[rank], buf, n);
}

void vllm_ep_root_close(VllmEpRoot *r) {
    if (!r) return;
    for (int i = 0; i < VLLM_EP_MAX_RANKS; i++)
        if (r->peer[i] != EP_INVALID) ep_closesock(r->peer[i]);
    if (r->listener != EP_INVALID) ep_closesock(r->listener);
    free(r);
}

/* ================================================================
 * 工作者
 * ================================================================ */
struct VllmEpWorker {
    ep_fd fd;
    int   rank, nranks;
};

VllmEpWorker *vllm_ep_worker_open(const char *host, int port, int rank,
                                  int nranks, VllmEpBackend be) {
    (void)be;
    if (ep_net_init() != 0) return NULL;
    if (rank < 1 || rank >= nranks || nranks > VLLM_EP_MAX_RANKS) {
        fprintf(stderr, "[EP] worker: 非法 rank=%d/nranks=%d\n", rank, nranks);
        return NULL;
    }
    if (!host || !host[0]) host = "127.0.0.1";
    VllmEpWorker *w = (VllmEpWorker *)calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->rank = rank; w->nranks = nranks; w->fd = EP_INVALID;

    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_port = htons((unsigned short)port);
#ifdef _WIN32
    a.sin_addr.s_addr = inet_addr(host);
    if (a.sin_addr.s_addr == INADDR_NONE) {
        fprintf(stderr, "[EP] worker: 非法 host=%s\n", host);
        free(w); return NULL;
    }
#else
    if (inet_pton(AF_INET, host, &a.sin_addr) != 1) {
        fprintf(stderr, "[EP] worker: 非法 host=%s\n", host);
        free(w); return NULL;
    }
#endif

    for (int attempt = 0; attempt < VLLM_EP_CONNECT_RETRY; attempt++) {
        ep_fd fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd == EP_INVALID) break;
        if (connect(fd, (struct sockaddr *)&a, sizeof(a)) == 0) { w->fd = fd; break; }
        ep_closesock(fd);
#ifdef _WIN32
        Sleep(200);
#else
        usleep(200 * 1000);
#endif
    }
    if (w->fd == EP_INVALID) {
        fprintf(stderr, "[EP] worker(rank %d): 连接 %s:%d 失败\n", rank, host, port);
        free(w); return NULL;
    }
    ep_set_timeout(w->fd);
    int32_t rk = (int32_t)rank;
    if (ep_send_all(w->fd, &rk, sizeof(rk)) != 0) {
        fprintf(stderr, "[EP] worker(rank %d): 握手发送失败\n", rank);
        vllm_ep_worker_close(w);
        return NULL;
    }
    fprintf(stderr, "[EP] worker(rank %d): 已连接 %s:%d\n", rank, host, port);
    return w;
}

int vllm_ep_worker_recv(VllmEpWorker *w, void *buf, size_t n) {
    if (!w || w->fd == EP_INVALID) return -1;
    return ep_recv_all(w->fd, buf, n);
}

int vllm_ep_worker_send(VllmEpWorker *w, const void *buf, size_t n) {
    if (!w || w->fd == EP_INVALID) return -1;
    return ep_send_all(w->fd, buf, n);
}

void vllm_ep_worker_close(VllmEpWorker *w) {
    if (!w) return;
    if (w->fd != EP_INVALID) ep_closesock(w->fd);
    free(w);
}

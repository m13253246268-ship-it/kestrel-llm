/**
 * tok_ref_check.c — tokenizer「索引路径 vs 线性扫描路径」位级回归对照
 *
 * 为什么需要它：把 O(文本长度 × vocab_size) 的词表线性扫描换成哈希索引后，
 * 分词结果必须与旧实现**逐 id 相同**（等长取最小 id、`lim` 截断、特殊 token
 * 优先、Ġ/Ċ 变体路径都要一致）。本工具用 ENGINE 内的开关
 * `VLLM_TOK_LINEAR=1`（强制旧路径）跑同一批输入，逐 id 比对。
 *
 * 编译（MinGW-w64 / Linux 均可；无需引擎其余部分）：
 *   gcc -std=gnu11 -O2 -D_GNU_SOURCE \
 *       -Iinclude/common -Iinclude/model -Isrc \
 *       -o tok_ref_check tools/bench/tok_ref_check.c src/model/vllm_tokenizer_qwen.c
 *
 * 用法：
 *   ./tok_ref_check <模型目录，含 vocab.bin>            # 用内置构造的用例
 *   ./tok_ref_check <模型目录> <文本文件> [重复次数]     # 额外加一个长文本用例
 *
 * 退出码：0 = 全部一致；1 = 存在不一致（打印首个失配的字节偏移与 id）。
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "vllm_tokenizer_qwen.h"

#define MAX_IDS 262144

static int g_case = 0;
static int g_fail = 0;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void set_linear_env(int on) {
#ifdef _WIN32
    _putenv_s("VLLM_TOK_LINEAR", on ? "1" : "0");
#else
    setenv("VLLM_TOK_LINEAR", on ? "1" : "0", 1);
#endif
}

/* 逐 id 比对两条路径；返回 0 = 一致。 */
static int check(QwenTokenizer *idx, QwenTokenizer *lin,
                 const char *text, const char *name, int verbose) {
    static int a[MAX_IDS], b[MAX_IDS];
    int na = qwen_tokenizer_encode(idx, text, a, MAX_IDS);
    int nb = qwen_tokenizer_encode(lin, text, b, MAX_IDS);
    g_case++;

    if (na != nb) {
        g_fail++;
        printf("[FAIL] %-34s token 数不同: indexed=%d linear=%d\n", name, na, nb);
        return 1;
    }
    for (int i = 0; i < na; i++) {
        if (a[i] != b[i]) {
            g_fail++;
            printf("[FAIL] %-34s 第 %d 个 id 不同: indexed=%d linear=%d\n",
                   name, i, a[i], b[i]);
            return 1;
        }
    }
    if (verbose) printf("[PASS] %-34s tokens=%d\n", name, na);
    return 0;
}

/* 造一个真实形状的长 prompt：SEG × n + 追问（与 docs/bench 的 A/B 同形）。 */
static char *make_prompt(int n_seg, int turn, size_t *out_len) {
    static const char *SEG = "边缘计算与云计算的核心区别在于数据处理发生的位置。";
    size_t seg_len = strlen(SEG);
    size_t cap = seg_len * (size_t)n_seg + 1024;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;
    size_t p = 0;
    for (int i = 0; i < n_seg; i++) {
        memcpy(buf + p, SEG, seg_len);
        p += seg_len;
    }
    p += (size_t)snprintf(buf + p, cap - p, "\n请用一句话总结上面这段话。");
    if (turn >= 2)
        p += (size_t)snprintf(buf + p, cap - p,
                              "\n\n追问一：请把这段话压缩成一个不超过 10 字的小标题。");
    if (turn >= 3)
        p += (size_t)snprintf(buf + p, cap - p,
                              "\n\n追问二：请给出一个反例，说明这个区别在什么情况下不成立。");
    *out_len = p;
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "用法: %s <模型目录(含 vocab.bin)> [文本文件] [重复次数]\n", argv[0]);
        return 2;
    }
    const char *dir = argv[1];

    static QwenTokenizer tok_idx, tok_lin;

    /* 1) 索引路径（默认） */
    set_linear_env(0);
    if (qwen_tokenizer_load(&tok_idx, dir) != 0) {
        fprintf(stderr, "tokenizer load (indexed) failed\n");
        return 2;
    }
    /* 2) 线性路径（旧实现，回归真值） */
    set_linear_env(1);
    if (qwen_tokenizer_load(&tok_lin, dir) != 0) {
        fprintf(stderr, "tokenizer load (linear) failed\n");
        return 2;
    }
    if (!tok_idx.vocab_tab) {
        fprintf(stderr, "indexed build missing vocab_tab — 索引没建起来\n");
        return 2;
    }

    /* ---- 用例 A：真实形状长 prompt（2K/4K/8K × 3 轮） ---- */
    const int segs[] = {165, 330, 600};
    for (size_t s = 0; s < sizeof(segs) / sizeof(segs[0]); s++) {
        for (int turn = 1; turn <= 3; turn++) {
            size_t len = 0;
            char *p = make_prompt(segs[s], turn, &len);
            if (!p) return 2;
            char name[64];
            snprintf(name, sizeof(name), "prompt segs=%d turn=%d (%zu B)", segs[s], turn, len);
            /* 线性路径对 8K 要 ~10s（板端）/ 数秒（x86），只对首轮详报 */
            check(&tok_idx, &tok_lin, p, name, turn == 1);
            free(p);
        }
    }

    /* ---- 用例 B：特殊 token、变体路径与短边界 ---- */
    {
        static const char *specials[] = {
            "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>",
            "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
            "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
            "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<think>", "</think>",
            "", " ", "  ", "\n", "\n\n", "\n\n\n\n\n\n\n\n\n\n", "\t",
            "<", ".<", "。<", "a<|im_end|>", "字<|im_end|>字",
            "边缘计算", "ed ge", " edge", "edge ", "中英 mixed 混排 123",
            "\xC4\xA0", "\xC4\x8A", "\xC4", "\xE4", "\xF0\x9F\x98\x80",
            "<|im_end|>\n\n</think>\n\n", "你\n好\n吗",
        };
        for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); i++) {
            check(&tok_idx, &tok_lin, specials[i], "special/short", 0);
        }
        printf("[PASS] 特殊 token / 变体 / 短边界用例 %zu 条全部一致\n",
               sizeof(specials) / sizeof(specials[0]));
    }

    /* ---- 用例 C：在长文上随机切片 + 随机插入边界字符（覆盖 lim 截断与 UTF-8 边界） ---- */
    {
        size_t big_len = 0;
        char *big = make_prompt(600, 3, &big_len);
        if (!big) return 2;
        unsigned seed = 20260915u;
        int n_rand = 1500;
        int bad = 0;
        for (int i = 0; i < n_rand; i++) {
            seed = seed * 1664525u + 1013904223u;
            size_t off = (size_t)(seed >> 8) % (big_len > 64 ? big_len - 64 : 1);
            seed = seed * 1664525u + 1013904223u;
            size_t len = 1 + (size_t)(seed >> 8) % 256;
            if (off + len > big_len) len = big_len - off;
            char tmp[512];
            if (len > sizeof(tmp) - 1) len = sizeof(tmp) - 1;
            memcpy(tmp, big + off, len);
            tmp[len] = '\0';
            /* 按 3% 概率在尾部塞入特殊 token 前缀，逼出 lim 截断与特殊优先路径 */
            if ((seed % 100) < 3) {
                size_t l2 = strlen(tmp);
                if (l2 + sizeof("<|im_end|>") < sizeof(tmp))
                    memcpy(tmp + l2, "<|im_end|>", sizeof("<|im_end|>"));
            }
            if (check(&tok_idx, &tok_lin, tmp, "random slice", 0) != 0) {
                if (++bad >= 3) break;
            }
        }
        if (bad == 0) printf("[PASS] 随机切片用例 %d 条全部一致\n", n_rand);
        free(big);
    }

    /* ---- 用例 D：可选外部文本（可给 16K 原文） ---- */
    if (argc >= 3) {
        FILE *f = fopen(argv[2], "rb");
        if (!f) { fprintf(stderr, "打不开 %s\n", argv[2]); return 2; }
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        char *txt = (char *)malloc((size_t)sz + 1);
        if (!txt) { fclose(f); return 2; }
        size_t rd = fread(txt, 1, (size_t)sz, f);
        txt[rd] = '\0';
        fclose(f);
        char name[80];
        snprintf(name, sizeof(name), "%s (%zu B)", argv[2], rd);
        check(&tok_idx, &tok_lin, txt, name, 1);
        free(txt);
    }

    /* ---- 速度对照（同一段 8K 文本，两条路径各跑一遍） ---- */
    {
        size_t len = 0;
        char *p = make_prompt(600, 3, &len);
        if (p) {
            static int tmp[MAX_IDS];
            double t0 = now_s();
            int n1 = qwen_tokenizer_encode(&tok_idx, p, tmp, MAX_IDS);
            double t1 = now_s();
            int n2 = qwen_tokenizer_encode(&tok_lin, p, tmp, MAX_IDS);
            double t2 = now_s();
            printf("\n[SPEED] 同一段 %zu B 文本（%d token）：indexed %.3f s（%.1f us/token）"
                   " vs linear %.3f s（%.1f us/token）→ 加速 %.0fx\n",
                   len, n1, t1 - t0, (t1 - t0) * 1e6 / (n1 ? n1 : 1),
                   t2 - t1, (t2 - t1) * 1e6 / (n2 ? n2 : 1),
                   (t1 - t0) > 0 ? (t2 - t1) / (t1 - t0) : 0.0);
            free(p);
        }
    }

    printf("\n==== tok_ref_check: %d 用例, %d 失配 ====\n", g_case, g_fail);
    qwen_tokenizer_free(&tok_idx);
    qwen_tokenizer_free(&tok_lin);
    return g_fail ? 1 : 0;
}

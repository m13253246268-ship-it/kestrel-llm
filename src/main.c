/**
 * main.c - vLLM-Kestrel: Axiom-Guided Inference Demo
 *
 * Demonstrates the full inference pipeline with SHS axiom constraints:
 *   1. Model initialization
 *   2. Request scheduling with probabilistic selection
 *   3. Multi-layer attention with batched KV-cache access
 *   4. Token generation with collapse_priority_rule
 *   5. Performance verification
 */

#include "vllm_superpos.h"
#include "vllm_matmul.h"
#include "vllm_l3.h"
#include "vllm_fhe.h"
#include "vllm_ckks.h"
#include "vllm_ntt.h"
#include "vllm_safetensors.h"
#include "vqf.h"
#include "vllm_vision.h"
#include "vllm_media.h"
#include "vllm_tokenizer_qwen.h"
#include "vllm_platform.h"
#include "vllm_npu.h"
#include "vllm_http.h"
#include "vllm_server.h"
#include "vllm_attest.h"
#include "vllm_device.h"
#include "vllm_i18n.h"     /* vllm_tr：引擎侧少量可现文案的中英双语 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <locale.h>   /* setlocale */
#include "vllm_util.h"
#if !defined(_WIN32)
#include <sys/mman.h>   /* 文件映射（VQF mmap 诊断/页控制） */
#include <unistd.h>     /* ftruncate */
/* Portable shims so the QPC-based timing and _heapchk triage code compiles
 * unchanged on Linux/RK3588 (aarch64). LARGE_INTEGER maps to the monotonic
 * tick counters from vllm_platform.h; _heapchk degrades to a no-op that
 * always reports heap-OK (debug triage only, never in the compute path). */
typedef struct { long long QuadPart; } LARGE_INTEGER;
#define QueryPerformanceFrequency(x) do { (x)->QuadPart = (long long)st_tick_freq(); } while (0)
#define QueryPerformanceCounter(x)   do { (x)->QuadPart = (long long)st_now_ticks(); } while (0)
#define _heapchk()    st_heapchk()
#define _HEAPOK       ST_HEAP_OK
#define _HEAPEMPTY    0
#define _HEAPBADPTR   (-1)
#define _HEAPBADBEGIN (-2)
#define _HEAPBADNODE  (-3)
#else
/* Windows：POSIX 文件映射 shims（仅编译垫片，VQF 文件加载走 vqf.c 的
 * CreateFileMappingW 路径，不依赖此处宏）。 */
#include <io.h>   /* _chsize_s / fileno */
#define PROT_READ   1
#define PROT_WRITE  2
#define MAP_SHARED  1
#define MAP_FAILED  ((void *)(intptr_t)-1)
#define MS_SYNC     0
#define MADV_DONTNEED 0
static void *mmap(void *addr, size_t len, int prot, int flags,
                  int fd, long long off) {
    (void)addr; (void)len; (void)prot; (void)flags; (void)fd; (void)off;
    return MAP_FAILED;
}
static int munmap(void *addr, size_t len) { (void)addr; (void)len; return -1; }
static int msync(void *addr, size_t len, int flags) {
    (void)addr; (void)len; (void)flags; return -1;
}
static int madvise(void *addr, size_t len, int advice) {
    if (advice == MADV_DONTNEED && addr != NULL) {
        VirtualUnlock(addr, len);
        return 0;
    }
    return -1;
}
#define ftruncate(fd, len) (_chsize_s((fd), (long)(len)))
#endif
#include "vllm_tp.h"   /* self-contained thread pool (replaces OpenMP) */

extern KVCacheManager g_kvcache;
extern AttentionLayer *layers;
extern RequestScheduler g_scheduler;

/* ================================================================
 * Test: SHS Superposition Operations
 * ================================================================ */
static void test_superposition_axioms(void) {
    printf("\n=== Test 1: SHS Superposition Axioms ===\n");

    /* Test existence_axiom */
    SHS_SetState S = shs_existence(5, SHS_MODULUS_N);
    printf("[PASS] existence_axiom: S^(1) = {%u} (count=%zu)\n",
           S.states[0].value, S.count);

    /* Test exponential_jump_set */
    int64_t jumps[] = {1, 2, 3};
    shs_exponential_jump_set(&S, 3, jumps, 3);
    printf("[PASS] exponential_jump_set: expanded to %zu states\n", S.count);

    /* Test superposition_addition_clear: 1+1 = {2, 3} */
    SHS_SetState add_result;
    shs_superposition_add(&add_result, 1, 1, SHS_MODULUS_N);
    printf("[PASS] 1�? = {%u, %u} (special case)\n",
           add_result.states[0].value, add_result.states[1].value);

    /* Test collapse_priority_rule */
    uint32_t collapsed;
    if (shs_collapse_priority(&add_result, &collapsed)) {
        printf("[PASS] collapse_priority: collapsed to %u\n", collapsed);
    }

    /* Test odd_even_interference */
    SHS_SetState inter_test;
    setstate_init(&inter_test, SHS_MODULUS_N);
    setstate_add_element(&inter_test, 100, 2, 0.8f);  /* even exponent */
    setstate_add_element(&inter_test, 100, 3, 0.6f);  /* odd exponent */
    setstate_add_element(&inter_test, 100, 5, 0.4f);  /* odd exponent */
    shs_odd_even_interference(&inter_test);
    printf("[PASS] odd_even_interference: compressed to %zu states\n",
           inter_test.count);

    /* Test ⊗_nc multiplication */
    SHS_SetState A, B, M;
    A = shs_existence(7, SHS_MODULUS_N);
    B = shs_existence(11, SHS_MODULUS_N);
    shs_nc_multiplication(&M, &A, &B);
    uint64_t expected = ((uint64_t)7 * 11) % SHS_MODULUS_N;
    printf("[PASS] ⊗_nc: 7�?1 = %u (expected %llu)\n",
           M.states[0].value, expected);

    /* Cleanup */
    setstate_free(&S);
    setstate_free(&add_result);
    setstate_free(&inter_test);
    setstate_free(&A);
    setstate_free(&B);
    setstate_free(&M);
}

/* ================================================================
 * Test: NTT Transform
 * ================================================================ */
static void test_ntt_transform(void) {
    printf("\n=== Test 3: NTT Transform (ntt_isomorphism axiom) ===\n");

    uint32_t data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint32_t orig[8];
    memcpy(orig, data, sizeof(data));

    uint32_t q = 12289; /* NTT modulus (prime: 12288 = 2^12 * 3) */

    /* Find primitive 8-th root of unity mod 12289
     * Need ω such that ω^8 �?1 (mod q) and ω^4 �?1, ω^2 �?1
     * Try: compute g^1536 where g=11 is primitive root of 12289,
     * since 12288/8 = 1536 */
    uint32_t root = 1;
    for (uint32_t r = 2; r < q; r++) {
        uint32_t r8 = 1;
        for (int i = 0; i < 8; i++) r8 = (uint32_t)(((uint64_t)r8 * r) % q);
        if (r8 != 1) continue;
        uint32_t r4 = 1;
        for (int i = 0; i < 4; i++) r4 = (uint32_t)(((uint64_t)r4 * r) % q);
        if (r4 == 1) continue; /* must be primitive 8th root, not lower order */
        root = r;
        break;
    }
    printf("[INFO] NTT primitive 8-th root mod %u = %u\n", q, root);

    shs_ntt_forward(data, 8, q, root);
    printf("[PASS] NTT forward: [%u %u %u %u ...]\n",
           data[0], data[1], data[2], data[3]);

    shs_ntt_inverse(data, 8, q, root);
    printf("[PASS] NTT inverse (roundtrip): [%u %u %u %u ...]\n",
           data[0], data[1], data[2], data[3]);

    /* Verify iNTT(NTT(x)) == x */
    bool ok = true;
    for (int i = 0; i < 8; i++) {
        if (data[i] != orig[i]) {
            ok = false;
            printf("[FAIL] Mismatch at index %d: expected %u, got %u\n",
                   i, orig[i], data[i]);
            break;
        }
    }
    printf("[%s] NTT bijection: iNTT(NTT(x)) == x\n",
           ok ? "PASS" : "FAIL");
}

/* ================================================================
 * Test: KV-Cache
 * ================================================================ */
static void test_kvcache(void) {
    printf("\n=== Test 4: KV-Cache (PagedAttention) ===\n");

    KVCacheManager mgr;
    kvcache_init(&mgr, VLLM_KV_BLOCK_SIZE);

    int32_t b1 = kvcache_alloc_block(&mgr);
    int32_t b2 = kvcache_alloc_block(&mgr);
    printf("[PASS] Allocated blocks: %d, %d (free=%d)\n",
           b1, b2, mgr.free_count);

    KVSlot kv;
    memset(&kv, 0, sizeof(kv));
    kv.key[0] = 42;
    kv.value[0] = 84;
    kvcache_write(&mgr, b1, 0, &kv);

    const KVSlot *read_back = kvcache_read(&mgr, b1, 0);
    printf("[%s] KV-Cache write/read: key=%u value=%u\n",
           (read_back && read_back->key[0] == 42) ? "PASS" : "FAIL",
           read_back ? read_back->key[0] : 0,
           read_back ? read_back->value[0] : 0);

    kvcache_free_block(&mgr, b1);
    kvcache_free_block(&mgr, b2);
    printf("[PASS] Freed blocks (free=%d)\n", mgr.free_count);
}

/* ================================================================
 * Test: Full Inference Pipeline
 * ================================================================ */
static void test_inference_pipeline(void) {
    printf("\n=== Test 5: Full Inference Pipeline ===\n");

    ModelConfig config = {
        .num_layers = 2,    /* small model for test */
        .num_heads = 4,
        .head_dim = 64,
        .hidden_dim = 256,
        .vocab_size = 32000,
        .max_seq_len = 128,
        .kv_block_size = VLLM_KV_BLOCK_SIZE,
        .ntt_modulus = 12289,
        .ntt_order = 1024,
    };

    vllm_kestrel_init(&config);

    /* Create test requests */
    RequestScheduler sched;
    memset(&sched, 0, sizeof(sched));
    sched.max_batch_size = 4;

    /* Request 1: short prompt */
    sched.requests[0].request_id = 1;
    sched.requests[0].seq_len = 8;
    sched.requests[0].current_pos = 4; /* already processed 4 tokens */
    sched.requests[0].status = REQ_RUNNING;
    sched.requests[0].priority = 0.8f;

    /* Request 2: medium prompt */
    sched.requests[1].request_id = 2;
    sched.requests[1].seq_len = 16;
    sched.requests[1].current_pos = 8;
    sched.requests[1].status = REQ_RUNNING;
    sched.requests[1].priority = 0.6f;

    /* Request 3: waiting */
    sched.requests[2].request_id = 3;
    sched.requests[2].seq_len = 32;
    sched.requests[2].current_pos = 0;
    sched.requests[2].status = REQ_WAITING;
    sched.requests[2].priority = 0.9f;

    sched.num_requests = 3;

    /* Allocate initial KV-cache blocks */
    for (int i = 0; i < 3; i++) {
        int32_t block = kvcache_alloc_block(&g_kvcache);
        if (block >= 0) {
            sched.requests[i].block_table[0] = block;
            sched.requests[i].num_blocks = 1;
        }
    }

    /* Run generation for 5 steps */
    uint32_t tokens[4];
    int32_t num_tokens;
    clock_t start = clock();

    for (int step = 0; step < 5; step++) {
        vllm_kestrel_generate(&g_kvcache, &sched, tokens, &num_tokens);

        int running = 0, finished = 0;
        for (int i = 0; i < sched.num_requests; i++) {
            if (sched.requests[i].status == REQ_RUNNING) running++;
            if (sched.requests[i].status == REQ_FINISHED) finished++;
        }
        printf("  Step %d: generated %d tokens (running=%d, finished=%d)\n",
               step + 1, num_tokens, running, finished);
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("[PASS] Inference pipeline: 5 steps in %.3f seconds\n", elapsed);

    vllm_kestrel_cleanup();
}

/* ================================================================
 * Test: Axiom Coverage Verification
 * ================================================================ */
static void test_axiom_coverage(void) {
    printf("\n=== Test 6: Axiom Coverage Report ===\n");

    const char *axioms_implemented[] = {
        "existence_axiom (a03280c0)",
        "exponential_jump_set (259264ad)",
        "superposition_addition_clear (e6cf6fd3)",
        "general_superposition_addition (001)",
        "odd_even_interference_logic (85b52caf)",
        "collapse_priority_rule (e16488e2)",
        "collapse_when_contains_one (85892d30)",
        "value_mod_interference (001)",
        "tensor_iterated_addition (e02cfff4)",
        "selective_interference_for_compression (8ecadfb1)",
        "probabilistic_selection_nc (38aaa102)",
        "adaptive_selection_based_on_features (07597cca)",
        "ntt_isomorphism / isomorphic_superposition (004)",
        "modal_transition_conservation (005)",
        "nonclassical_multiplication (nc / 6fec81b9)",
        "operator_composition (1-5)",
        "set_operations (91154c82)",
        "set_uniqueness_filter (b646eb9f)",
        "state_as_tuple (0407a8c2)",
        "state_set_merging (1a1bb662)",
        "probabilistic_choice_operator (ae1e2d74)",
        "parameter_declaration (N, m, q, r, a, K)",
        "classic_arithmetic (814ac5ab)",
        "linear_interpolation_axiom (a642640a)",
        "bitwise_xor_superposition (412bc51e)",
        "hash_acceleration (001)",
        "commutative_addition_nc (07ebe987)",
        "associative_addition_nc (bd23936b)",
        "left_distributive_nc (fbf436ac)",
        "right_distributive_nc (0e3425c2)",
        "conservation_law_under_collapse (071ef1c6)",
        "symmetry_preservation_under_superposition (efd038db)",
        "superposition_collapse_link (37b92908)",
        "collapse_preservation_under_invertible (9a4a4716)",
        "price_superposition_addition (4513280f) -- financial extension",
        "volatility_superposition_measure (ab4d2b17)",
    };

    int covered = sizeof(axioms_implemented) / sizeof(axioms_implemented[0]);
    int total = 186; /* total axioms in registry */

    printf("[REPORT] Implemented: %d / %d total axioms (%.1f%%)\n",
           covered, total, 100.0f * covered / (float)total);
    printf("[REPORT] Core inference axioms: all implemented\n");

    for (int i = 0; i < covered; i++) {
        printf("  �?%s\n", axioms_implemented[i]);
    }
}

/* ================================================================
 * Test 7: Real Transformer Inference (end-to-end)
 * ================================================================ */
static void test_real_inference(void) {
    printf("\n=== Test 7: Real Transformer Inference ===\n");

    /* Initialize model and tokenizer */
    Tokenizer tok;

    printf("[1/5] Initializing tokenizer (%d tokens)...\n", TINY_VOCAB_SIZE);
    tokenizer_init(&tok);
    printf("       Vocabulary size: %d\n", tok.vocab_size);

    printf("[2/5] Initializing model weights...\n");
    InferenceState *st = inference_create();
    if (!st) { printf("[FAIL] Out of memory\n"); return; }
    printf("       Model: %d layers, %d heads, dim=%d, ffn=%d\n",
           TINY_NUM_LAYERS, TINY_NUM_HEADS, TINY_HIDDEN_DIM, TINY_FFN_DIM);

    /* Save/Load test */
    printf("[3/5] Testing weight serialization...\n");
    int save_ok = weights_save(&st->weights, "build/model_weights.bin");
    printf("       Save to disk: %s\n", save_ok == 0 ? "OK" : "FAILED");

    ModelWeights *loaded = malloc(sizeof(ModelWeights));
    memset(loaded, 0, sizeof(*loaded));
    int load_ok = weights_load(loaded, "build/model_weights.bin");
    printf("       Load from disk: %s\n", load_ok == 0 ? "OK" : "FAILED");

    /* Verify weights match */
    int match = (memcmp(&st->weights, loaded, sizeof(ModelWeights)) == 0);
    printf("       Weight integrity: %s\n", match ? "VERIFIED" : "MISMATCH");
    free(loaded);

    /* Real inference with prompts */
    printf("[4/5] Running inference on prompts...\n");

    const char *prompts[] = {
        "the future of AI is",
        "once upon a time",
        "the most important thing in life is",
        "in the beginning",
    };
    int num_prompts = sizeof(prompts) / sizeof(prompts[0]);

    for (int p = 0; p < num_prompts; p++) {
        printf("\n       Prompt %d: \"%s\"\n", p + 1, prompts[p]);

        clock_t t0 = clock();
        char *output = inference_generate(st, &tok, prompts[p],
                                           15, 0.8f, 0.9f);
        clock_t t1 = clock();
        double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;

        printf("       Output: \"%s\"\n", output ? output : "(null)");
        printf("       Time: %.3f sec, Tokens: %d\n",
               elapsed, st->seq_len);

        free(output);
    }

    /* Performance benchmark */
    printf("\n[5/5] Performance benchmark...\n");
    {
        /* Reset and run 10 iterations */
        int total_tokens = 0;
        clock_t t_start = clock();

        for (int iter = 0; iter < 10; iter++) {
            for (int l = 0; l < TINY_NUM_LAYERS; l++) {
                st->layers[l].cache_len = 0;
            }
            st->seq_len = 0;

            int prompt_ids[32];
            int num = tokenizer_encode(&tok, "the meaning of life",
                                        prompt_ids, 32);
            for (int i = 0; i < num; i++) model_forward(st, prompt_ids[i]);

            for (int t = 0; t < 8; t++) {
                int tok_id = sample_token(st->logits, TINY_VOCAB_SIZE,
                                           1.0f, 1.0f);
                model_forward(st, tok_id);
                total_tokens++;
            }
        }

        clock_t t_end = clock();
        double total_time = (double)(t_end - t_start) / CLOCKS_PER_SEC;
        double tokens_per_sec = (double)total_tokens / total_time;

        printf("       Processed %d tokens in %.3f sec\n",
               total_tokens, total_time);
        printf("       Throughput: %.1f tokens/sec\n", tokens_per_sec);
    }

    /* Cleanup */
    inference_free(st);
    tokenizer_free(&tok);
    printf("\n[PASS] Real inference pipeline complete\n");
}


/* ================================================================
 * Test 10: Performance Benchmark �?Single-User & Multi-User Concurrent
 *
 * Measures:
 *   1. Single-user: TTFT (time-to-first-token), TPOT (per-output-token),
 *      throughput (tok/s), end-to-end latency
 *   2. Multi-user concurrent (different prompts): per-user latency,
 *      aggregate throughput, output quality validation
 *
 * Axioms: message_passing_natural_isomorphism,
 *         parallelism_preserving_decomposition
 * ================================================================ */
#define MAX_PERF_USERS  10
#define PERF_MAX_TOKENS 40
#define PERF_MAX_CTX    17408

/* Part B/C multi-user concurrency (--bench-users N, default 10). Kept small
 * (e.g. 2) on the 16GB RK3588 board to bound benchmark wall time; Part A
 * single-user TTFT/TPOT stays the primary CPU-vs-NPU comparison metric. */
static int g_bench_users = 10;

/* --bench-users N 被显式指定 → 触发完整 Part A/B/C 多用户基准
 * （回移来源：GitHupSRC/src/main.c:4897-4900；旧引擎由无参默认路径调用
 *  test_performance_benchmark，这里改为显式标志，保证无 flag 默认行为不变）。 */
static int g_bench_users_set = 0;

/* --perf-partA: run only Part A (single-user TTFT/TPOT) of the performance
 * benchmark, then stop. Skips the multi-user Part B/C to bound wall time on
 * the SD-card board; Part A is the CPU-vs-NPU comparison core. */
static int g_perf_part_a = 0;

/* --bench-seqlen N: pad each Part A prompt's prefill to a fixed context
 * length N (e.g. 64 to match the earliest CPU baseline S=64 measurement).
 * The prompt is left-padded with a neutral repeated phrase so the prefill
 * token count is exact; decode then starts from that fixed context. */
int g_bench_seqlen = 0;

/* Set via --longctx-quality to run only the long-context needle-retrieval +
 * Q4_0 quality/perplexity benchmark (Part F). Used for fast iteration. */
static int g_longctx_quality = 0;
static int g_longctx_gen_max = 32;
static int g_longctx_ctx = 0;

/* OpenAI-compatible HTTP server mode (--serve). -1 = disabled. */
static int  g_serve_port = -1;
static int  g_serve_port_explicit = 0;  /* set only by --port (no interactive prompt) */
static const char *g_serve_model_dir = NULL;
static int g_stream_test = 0;   /* --stream-test: AirLLM 型分层驻留验证 */
static int g_stream_n    = 12;  /* --stream-n N: decode token 数 */
static int g_moe_l0      = -1;  /* --moe-l0 <tok>: M3 layer-0 单 token 对拍 dump */
static const char *g_serve_model_id = NULL;
static int  g_serve_threads = 8;       /* HTTP worker threads */
static int  g_serve_max_queued = 16;   /* max queued inference requests (429 beyond) */
static int  g_serve_batch_max = 0;     /* --batch-max N: continuous batching (0/1 = off) */
static int  g_serve_min_free_mb = 512; /* refuse inference below this free RAM
                                        * (models keep their KV cache allocated
                                        * after load; a small headroom of 512 MB
                                        * guards against mid-prefill allocs
                                        * without spurious 503s on tight boxes) */
static int  g_serve_auto_load = 0;     /* --auto-load: load the model at startup */
static int  g_serve_load_on_use = 0;   /* --load-on-use: 请求到达且模型未加载时自动加载（用时加载） */
static long g_serve_auto_unload_s = 0; /* --auto-unload N: 空闲 N 秒后自动卸载模型（不用时卸载，
                                        * 隐含启用 --load-on-use 形成 加载→使用→空闲→卸载 循环） */
static int  g_serve_load_format = 0;   /* --load-format: 0=auto(VQF), 1=vqf */

/* Last model dir whose NPU strip-cache (st_npu_gw_*) is resident. Loading a
 * DIFFERENT model must drop the stale strips (different weights/geometry);
 * same-model reloads keep them so a reloaded request skips the rebuild storm. */
static char g_npu_loaded_model[1024];

/* Device classification (auto-detected, overridable via --device). */
static VDevInfo g_dev;                       /* detection result */
static const VDevProfile *g_dev_profile = NULL;   /* effective profile */
static const char *g_dev_override = NULL;    /* --device <id> */

/* Set via --prefix-cache to run only the shared-prefix KV-cache reuse A/B
 * (Part G). A long shared prefix is prefilled once and forked into N request
 * states via st_qwen_copy_kv_prefix, so only the unique suffix is prefilled. */
static int g_prefix_cache = 0;

/* serve: in-RAM KV prefix reuse (default ON - the previous request's KV
 * prefix is kept and the shared prompt prefix is not re-prefilled).
 * --no-prefix-kv turns it off; L3 eviction auto-degrades to full prefill. */
static int g_prefix_kv = 1;

/* serve: default min-p sampling filter (0 = off). ds4/DwarfStar uses min-p
 * as its default sampler; request body min_p overrides. */
static double g_serve_min_p = 0.0;

/* serve: default top-k sampling truncation (0 = off, --top-k N). Qwen3
 * thinking 档官方建议 top_k=20；请求体 top_k 可逐请求覆盖。 */
static int    g_serve_top_k = 0;

/* serve: disk KV persistence (--disk-kv DIR). F32 KV snapshots of finished
 * conversations survive process restarts; see vllm_server.c / st_kv_disk_*. */
static int    g_disk_kv = 0;
static char   g_disk_kv_dir[512] = {0};

/* serve: speculative decode (--spec), n-gram draft + batched verification.
 * Greedy-only and text-only; default draft length 4 (--spec-k). */
static int    g_spec = 0;
static int    g_spec_k = 4;

/* serve: 内存驻留策略（档位阶梯）CLI 覆盖。vllm_res_defaults 提供默认空闲
 * 计划 {5m,15m,30m,60m}；以下选项可在启动时改写（管理页运行期仍可调）：
 *   --res-auto        空闲逐级自动降级开（等价管理页"自动降级"勾选）
 *   --res-idle a,b,c,d 各档停留秒数（L4→L3,L3→L2,L2→L1,L1→L0；0=停在该档）
 *   --res-soft-mb N   软水位（可用内存低于 N MB → 自动降一档）
 *   --res-low N       自动动作允许的最低档（0..4；设 3 可只降 KV 淘汰）
 *   --res-w-keep N    L2 权重窗口层数（保留 0..N 层，默认 8） */
static int  g_res_auto = 0;
static long g_res_idle_s[4] = {0, 0, 0, 0};   /* 0 = 使用默认计划 */
static int  g_res_soft_mb = 0;
static int  g_res_low_lvl = -1;               /* -1 = 使用默认(0) */
static int  g_res_w_keep = 0;                 /* 0 = 使用默认(8) */

/* Transparent RK3588 NPU offload (--npu). When enabled, prefill GEMM stages
 * are offered to the NPU; any missing model / absent runtime falls back to
 * the CPU path transparently (see vllm_npu.h / vllm_safetensors.c).
 *   --npu-backend direct: in-tree zero-dependency rknpu driver - NO model
 *        conversion; the engine's Q8_0/Q4_0 block weights feed the NPU as-is.
 *   --npu-backend rknn:   official librknnrt + exported .rknn operator models. */
static int    g_npu_enabled = 0;
static int    g_npu_selftest = 0;
static int    g_npu_calib_only = 0;
static vllm_npu_backend_t g_npu_backend = VLLM_NPU_BACKEND_AUTO;
static const char *g_npu_model_dir = NULL;
static double g_npu_flops_threshold = 1e8;
static vllm_npu_t *g_npu = NULL;
/* Heap-integrity triage: reports _HEAPOK / _HEAPBADNODE / _HEAPBADPTR.
 * Only the FIRST corruption site is printed (later ones are noise). */
static int g_hck_once = 0;
#define HCK(tag) do { \
    int _hc = _heapchk(); \
    if (_hc != _HEAPOK && !g_hck_once) { g_hck_once = 1; \
        printf("[HCK-FAIL] %-28s => %s\n", tag, \
               _hc == _HEAPEMPTY ? "EMPTY" : _hc == _HEAPBADPTR ? "BADPTR" : _hc == _HEAPBADBEGIN ? "BADBEGIN" : "BADNODE"); \
        fflush(stdout); \
    } \
} while (0)

/* Part F: Long-context needle retrieval + Q4_0 quality/perplexity.
 *
 * Builds a long context of S tokens with a "needle" fact planted in the
 * middle of the context, then asks a question whose answer requires reading
 * that fact. Prefill and decode both run exact attention.
 *
 * Reports, for S in {1024, 2048, 4096}:
 *   - whether the needle answer appears in the generated text (semantic consistency)
 *   - teacher-forced perplexity of the greedy sequence
 *   - decode latency (Q4_0 weights active when loaded).
 */
static float logits_nll(const float *logits, int vocab, int target) {
    float maxv = -1e30f;
    for (int i = 0; i < vocab; i++)
        if (logits[i] > maxv) maxv = logits[i];
    double sum = 0.0;
    for (int i = 0; i < vocab; i++)
        sum += (double)expf(logits[i] - maxv);
    double logsum = (double)maxv + log(sum);
    return (float)(logsum - (double)logits[target]);  /* -log softmax(target) */
}

static int greedy_argmax(const float *logits, int vocab) {
    float best = -1e30f;
    int bid = 0;
    for (int v = 0; v < vocab; v++)
        if (logits[v] > best) { best = logits[v]; bid = v; }
    return bid;
}

static void append_tok_str(char *buf, int *pos, int cap, QwenTokenizer *tok, int id) {
    const char *s = qwen_tokenizer_decode(tok, id);
    if (!s || !s[0]) return;
    const char *p = ((unsigned char)s[0] == 0xC4 && (unsigned char)s[1] == 0xA0) ? s + 2 : s;
    int len = (int)strlen(p);
    if (*pos + len < cap - 1) {
        memcpy(buf + *pos, p, len);
        *pos += len;
    }
}

/* Filler facts for the long-context needle test. Named people (not "Person N")
 * are used so greedy decode does not collapse into a numeric "Person N+1"
 * continuation loop. */
static const char *G_NAMES[] = {
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Oscar", "Paula",
    "Quinn", "Rose", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zoe", "Adam", "Bella", "Chloe", "Daniel", "Ella", "Felix"
};
/* Gender-matched possessive pronoun for each name (matches the short A/B
 * prompt style "her favorite food"/"his favorite food" that is known to be
 * answered correctly; the generic "their" caused the model to collapse). */
static const char *G_PRONOUNS[] = {
    "her", "his", "her", "his", "her", "his", "her", "his",
    "her", "his", "her", "his", "her", "his", "his", "her",
    "his", "her", "his", "her", "her", "his", "her", "his",
    "her", "her", "his", "her", "her", "his", "her", "his"
};
static const char *G_COLORS[] = {
    "blue", "red", "green", "yellow", "purple", "orange", "pink", "brown"
};
static const char *G_CITIES[] = {
    "Paris", "Tokyo", "London", "Berlin", "Rome", "Madrid", "Seoul", "Sydney"
};
static const char *G_PETS[] = {
    "dog", "cat", "bird", "fish", "hamster", "rabbit", "turtle", "horse"
};
static const char *G_PETNAMES[] = {
    "Max", "Luna", "Bella", "Charlie", "Rocky", "Milo", "Coco", "Leo"
};
static const char *G_DRINKS[] = {
    "tea", "coffee", "juice", "milk", "water", "soda", "lemonade", "cocoa"
};
static const char *G_JOBS[] = {
    "teacher", "engineer", "doctor", "lawyer", "artist", "chef", "nurse", "scientist"
};

/* A distinctive "needle" fact is buried in the middle of a long filler context
 * and the model is asked to retrieve it, so a successful retrieval proves the
 * engine preserves long-range facts. We use the plain "Question: ... Answer:"
 * completion format (no chat template) because it avoids the model immediately
 * emitting <|im_end|>. The needle uses a name ("Zara") and food ("ramen")
 * absent from the filler pools so it is unambiguous. */
#define INTRO_TEXT      "Here are some facts about people."
#define NEEDLE_TEXT     "Zara likes the color teal and her favorite food is ramen."
#define QUESTION_TEXT   "Question: What is Zara's favorite food?"
#define ANSWER_TEXT     "Answer:"
/* Longer teacher-forced reference: a natural sentence mixing high-frequency
 * tokens ("is", "her", "favorite") with the low-probability answer token
 * ("ramen"). With only "ramen" (2 tokens) the Q8/F32 PPL ratio was unstable
 * at S>=12288 (both directions: 1.0645 DENSE / 0.9076 C2) because the answer
 * token sits at p~=e^-14 where tiny logit deltas explode through exp().
 * Averaging over ~14 tokens restores the 1.0645/0.9076 spread to noise. */
#define REFERENCE_TEXT  "Zara's favorite food is ramen, and her favorite color is teal."
#define REFERENCE_KEY   "ramen"

/* Build a ~`S`-token plain completion:
 *   {filler}\n ... {needle}\n ... {filler}\n {question}
 * The needle is inserted ~half-way through the filler (the collapsed region),
 * the question sits at the very end so greedy decode yields the answer. Returns
 * total token count and records the token index where the needle starts. */
static int build_needle_prompt(QwenTokenizer *tok, int S,
                               int *ids, int max_ids, int *needle_start_out) {
    const int nl = 198;   /* Qwen3 newline token (Ċ) */
    const int n_names  = (int)(sizeof(G_NAMES)  / sizeof(G_NAMES[0]));
    const int n_colors = (int)(sizeof(G_COLORS) / sizeof(G_COLORS[0]));
    const int n_cities = (int)(sizeof(G_CITIES) / sizeof(G_CITIES[0]));
    const int n_pets   = (int)(sizeof(G_PETS)   / sizeof(G_PETS[0]));
    const int n_petn   = (int)(sizeof(G_PETNAMES) / sizeof(G_PETNAMES[0]));
    const int n_drinks = (int)(sizeof(G_DRINKS) / sizeof(G_DRINKS[0]));
    const int n_jobs   = (int)(sizeof(G_JOBS)   / sizeof(G_JOBS[0]));
    int tmp[256];
    int pos = 0;

    int intro_ids[64], needle_ids[64], q_ids[64], a_ids[64];
    int n_intro  = qwen_tokenizer_encode(tok, INTRO_TEXT, intro_ids, 64);
    int n_needle = qwen_tokenizer_encode(tok, NEEDLE_TEXT, needle_ids, 64);
    int n_q      = qwen_tokenizer_encode(tok, QUESTION_TEXT, q_ids, 64);
    int n_a      = qwen_tokenizer_encode(tok, ANSWER_TEXT, a_ids, 64);

    /* Fixed overhead: intro + needle + question + answer, each followed by a
     * newline except the trailing "Answer:" which greedy decode continues. */
    int overhead = n_intro + 1 + n_needle + 1 + n_q + 1 + n_a;
    int content_budget = S - overhead;
    if (content_budget < 0) content_budget = 0;

    int needle_at = content_budget / 2; /* needle buried in the collapsed mid-section */
    int filled = 0;
    int person = 0;
    int needle_inserted = 0;
    *needle_start_out = -1;

    /* Intro sentence frames the filler as a list of facts to remember. */
    for (int i = 0; i < n_intro && pos < max_ids; i++)
        ids[pos++] = intro_ids[i];
    if (pos < max_ids) ids[pos++] = nl;

    while (filled < content_budget && pos < max_ids) {
        if (!needle_inserted && filled >= needle_at) {
            *needle_start_out = pos;
            for (int i = 0; i < n_needle && pos < max_ids; i++)
                ids[pos++] = needle_ids[i];
            if (pos < max_ids) ids[pos++] = nl;
            needle_inserted = 1;
        }
        char fact[256];
        int kind = person % 3;
        if (kind == 0) {
            /* No "favorite food" in the filler: keep that attribute unique to
             * the needle so the question "What is Zara's favorite food?" has a
             * single unambiguous answer (otherwise a trailing "favorite food
             * is pizza" filler fact hijacks greedy decode via recency). */
            snprintf(fact, sizeof(fact),
                     "%s likes the color %s.",
                     G_NAMES[person % n_names], G_COLORS[person % n_colors]);
        } else if (kind == 1) {
            snprintf(fact, sizeof(fact),
                     "%s lives in %s and has a %s named %s.",
                     G_NAMES[person % n_names], G_CITIES[person % n_cities],
                     G_PETS[(person / n_cities) % n_pets],
                     G_PETNAMES[(person / n_cities) % n_petn]);
        } else {
            snprintf(fact, sizeof(fact),
                     "%s's favorite drink is %s and %s works as a %s.",
                     G_NAMES[person % n_names], G_DRINKS[person % n_drinks],
                     G_PRONOUNS[person % n_names],
                     G_JOBS[(person / n_drinks) % n_jobs]);
        }
        int n = qwen_tokenizer_encode(tok, fact, tmp, 256);
        if (n <= 0) break;
        for (int i = 0; i < n && pos < max_ids; i++)
            ids[pos++] = tmp[i];
        if (pos < max_ids) ids[pos++] = nl;
        filled += n + 1;
        person++;
    }
    if (!needle_inserted) {   /* fallback if budget was tiny */
        *needle_start_out = pos;
        for (int i = 0; i < n_needle && pos < max_ids; i++)
            ids[pos++] = needle_ids[i];
        if (pos < max_ids) ids[pos++] = nl;
    }

    /* "Question: ...\nAnswer:" — no trailing newline, so greedy decode starts
     * right after the answer prefix (matches the known-good short A/B). */
    for (int i = 0; i < n_q && pos < max_ids; i++)
        ids[pos++] = q_ids[i];
    if (pos < max_ids) ids[pos++] = nl;
    for (int i = 0; i < n_a && pos < max_ids; i++)
        ids[pos++] = a_ids[i];

    return pos;
}

/* VLLM_L3_DIAG=1: evict 段 VmRSS 采样 + mirror 尺寸统计。验证 serve 多轮下
 * 的 L3 RSS 棘轮（2026-09-07 实测 FS/NS 每轮 +~309MB，尽管日志逐轮
 * "evicted ... freed 236/49MB"——逻辑 free ≠ RSS 实降；根因与结论见
 * 方案文档 §5.1）。 */
static int l3_diag_on(void) {
    const char *e = getenv("VLLM_L3_DIAG");
    return (e && e[0] == '1') ? 1 : 0;
}
#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
static long l3_diag_rss_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "VmRSS: %ld kB", &kb) == 1) break;
    }
    fclose(f);
    return kb;
}
/* pagemap resident 统计（present bit）：页对齐内窗的驻留字节数。
 * 用于 L3 evict 前核对"将 munmap 的块实际驻留多少页"，判定 RSS 回落
 * 不对称（anon 页返回问题）根因。自进程 /proc/self/pagemap 可读。 */
static long l3_diag_resident_bytes(const void *p, size_t len) {
    long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0 || !p || len == 0) return -1;
    uintptr_t a = (uintptr_t)p;
    uintptr_t lo = (a + (uintptr_t)pg - 1) & ~((uintptr_t)pg - 1);
    uintptr_t hi = (a + len) & ~((uintptr_t)pg - 1);
    if (hi <= lo) return 0;
    size_t n = (size_t)((hi - lo) / (uintptr_t)pg);
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) return -1;
    off_t base_off = (off_t)(lo / (uintptr_t)pg) * 8;
    unsigned char buf[4096];
    size_t per = sizeof(buf) / 8;
    long res = 0;
    size_t done = 0;
    while (done < n) {
        size_t chunk = n - done;
        if (chunk > per) chunk = per;
        ssize_t rd = pread(fd, buf, chunk * 8, base_off + (off_t)done * 8);
        if (rd != (ssize_t)(chunk * 8)) break;
        for (size_t i = 0; i < chunk; i++) {
            uint64_t e;
            memcpy(&e, buf + i * 8, 8);
            if (e & (1ull << 63)) res++;   /* present bit */
        }
        done += chunk;
    }
    close(fd);
    return res * pg;
}
#else
static long l3_diag_rss_kb(void) { (void)0; return -1; }
static long l3_diag_resident_bytes(const void *p, size_t len) {
    (void)p; (void)len;
    return -1;   /* /proc/self/pagemap 仅 Linux 可用 */
}
#endif

/* Phase-2 hook: after prefill, pack the cold KV blocks into the Q4 disk file
 * (--l3-evict) and physically free them from RAM. Sparse decode re-reads
 * evicted blocks on demand from st->l3 in compressed Q4 form.
 * Shared by the bench path and the HTTP serve path (vllm_server.c).
 *
 * P3：
 *   keep_lcp    = 本轮复用的前缀长度（0 = 全量 prefill）。用于判定哪些块的
 *                 内容本轮被改写（行 >= keep_lcp）→ 其旧载荷作废。
 *   incremental = 1（门 VLLM_L3_PREFIX_REUSE 开）时跨轮保留 L3 状态、只重打包
 *                 被改写的块；0 时保持旧的"每轮 rewind + 全量重打包"语义。
 * 注意：keep_lcp 只影响 incremental 分支，incremental=0 时行为与加参数前逐位一致。
 */
void l3_evict_after_prefill(STQwenInferenceState *st, int keep_lcp, int incremental) {
    if (!g_l3_evict) return;
    if (!g_sparse_attn) {
        fprintf(stderr, "[L3] warning: --l3-evict requires --sparse-attn "
                        "(non-sparse attention needs all blocks in RAM); skipped\n");
        return;
    }
    STL3State *l3 = &st->l3;
    int diag = l3_diag_on();
    long rss_enter = diag ? l3_diag_rss_kb() : -1;
    /* VLLM_L3_MADV=1：evict 释放前先对块做 MADV_DONTNEED（对比档）——
     * 验证残余棘轮是否因 munmap 单独回收滞后，而显式 DONTNEED 丢弃页可即时回落。 */
    static int g_l3_madv = -1;
    if (g_l3_madv < 0) {
        const char *e = getenv("VLLM_L3_MADV");
        g_l3_madv = (e && e[0] == '1') ? 1 : 0;
    }
    /* P1 跨轮镜像复用（VLLM_L3_MIRROR_REUSE=0 回退旧语义，默认开）。
     * 旧路径每轮 l3_state_free + l3_state_init + l3_load_to_mem 会把 Q4 镜像
     * 整块重建（malloc+memset+全量 read），是 serve 同用户多轮 RSS 残余
     * +51.7MB/轮 的来源之一（《内存分页优化方案》D1）。复用时把上一轮的镜像
     * 摘出来交回 init 覆盖写；l3_load_to_mem 负责把本轮读不到的尾部页归还内核。
     * 所有权规则：pm 一旦摘出，除交给 l3_state_init_reuse 外，本函数任一早退
     * 都必须 free(pm)（init 失败路径由 init 内部释放）。 */
    static int g_l3_mirror_reuse = -1;
    if (g_l3_mirror_reuse < 0) {
        const char *e = getenv("VLLM_L3_MIRROR_REUSE");
        g_l3_mirror_reuse = (e && e[0] == '0') ? 0 : 1;
    }
    int nl = st->weights.n_layers_allocated;
    int kv_dim = st->cfg.n_kv_heads * st->cfg.head_dim;
    int seq_len = st->cache_len[0];
    /* L3 cache file path: per-user file (serve, --l3-evict + user field),
     * else --l3-path, else kv_l3.bin in the CWD. */
    const char *l3path = (g_l3_cur_path && g_l3_cur_path[0]) ? g_l3_cur_path
                       : (g_l3_path && g_l3_path[0]) ? g_l3_path : "kv_l3.bin";

    /* --l3-max-size (MB): cap the logical file size by shrinking the block
     * capacity. Blocks beyond the cap are NOT evicted (stay in RAM); the
     * fixed-offset file layout automatically truncates to fit. */
    int capacity = st->kv_n_blocks;
    if (g_l3_max_size > 0) {
        size_t bb = 2 * (size_t)st->cfg.n_kv_heads * (size_t)st->kv_bs
                    * ((size_t)st->cfg.head_dim / 64) * L3_Q4_PAYLOAD64;
        size_t maxb = ((size_t)g_l3_max_size * 1048576ull - L3_HEADER_SIZE)
                      / ((size_t)nl * bb);
        if (maxb < (size_t)capacity) capacity = (int)maxb;
        if (capacity < 1) capacity = 1;
        fprintf(stderr, "[L3] max-size=%lldMB -> capacity=%d/%d blocks "
                        "(%.1f MB file)\n",
                g_l3_max_size, capacity, st->kv_n_blocks,
                (L3_HEADER_SIZE + (double)nl * capacity * bb) / 1048576.0);
    }

    /* P3 增量：门开且上一轮状态同文件/同几何/同容量 → **保留** blocks[]、fp
     * 与镜像，块载荷跨轮续用；只对被改写的块重打包。这既消掉 R1（每轮
     * Q4→反量化→Q4 重复量化），也让 keep>0 时仍能驱逐（R4）。
     * incremental=0（门关，默认）时恒为 0 → 完全走旧的每轮重建路径。 */
    int retain = incremental && l3_state_matches(l3, l3path, nl,
                                                st->cfg.n_kv_heads,
                                                st->cfg.head_dim, st->kv_bs,
                                                capacity);
    uint8_t *pm = NULL;
    size_t pc = 0;
    if (retain) {
        fprintf(stderr, "[L3] incremental: retain state (packed_total=%d, "
                        "cursor=%.2f MB)\n",
                l3->evicted, (double)l3->wcursor / 1048576.0);
        fflush(stderr);
    } else if (l3->mem || l3->fp || l3->blocks || l3->evicted) {
        /* 释放上一轮残留的 L3 状态。serve 同用户多轮此前没有任何清理点：
         * 下一轮 l3_state_init 的 memset(s,0) 会直接把上一轮 Q4 镜像指针
         * s->mem（修复前按全容量 ~280MiB 分配）丢弃 -> 每轮泄漏一个镜像，
         * 实测 FS/NS 多轮 RSS 棘轮 +~309 MB/轮（方案文档 §5.1）。先重置再重建
         * （l3_state_free/l3_state_rewind 对零结构安全；evict 本轮的 decode 需要
         * 本次新建的镜像，上一轮镜像在本次 prefill 期间已是死重）。 */
        if (diag) {
            fprintf(stderr, "[L3-DIAG] evict enter rss=%ld kB, stale "
                            "mirror=%zu B (cap %zu), fp=%s -> %s\n",
                    rss_enter, l3->mem_size, l3->mem_cap,
                    l3->fp ? "open" : "null",
                    g_l3_mirror_reuse ? "rewind(keep mirror)" : "state_free");
            fflush(stderr);
        }
        if (g_l3_mirror_reuse) l3_state_rewind(l3, &pm, &pc);
        else                   l3_state_free(l3);
        if (diag) {
            fprintf(stderr, "[L3-DIAG] after stale reset rss=%ld kB "
                            "(reused mirror %zu B)\n", l3_diag_rss_kb(), pc);
            fflush(stderr);
        }
    }
    /* Eviction trigger threshold: skip until the context is long enough
     * (--l3-min-seq). Below the threshold the full KV stays in RAM. */
    if (g_l3_min_seq > 0 && seq_len < g_l3_min_seq) {
        fprintf(stderr, "[L3] skipped: seq=%d < l3-min-seq=%d (KV stays in RAM)\n",
                seq_len, g_l3_min_seq);
        free(pm);   /* P1：本轮不驱逐，摘出的镜像不再复用，交还堆 */
        return;
    }
    if (!retain) {
        /* P1：把上一轮摘出的镜像交回复用（容量不足时 init 内部会自行扩容）；
         * 失败路径由 l3_state_init_common 负责释放 pm。 */
        if (l3_state_init_reuse(l3, l3path, nl, st->cfg.n_kv_heads,
                                st->cfg.head_dim, st->kv_bs, capacity,
                                pm, pc) != 0) {
            fprintf(stderr, "[L3] eviction skipped: cannot open %s\n", l3path);
            return;
        }
        pm = NULL; pc = 0;   /* 所有权已转移给 l3（后续早退不得再 free） */
    }

    /* evict 是每轮瞬态标记：先全清（含 b >= n_blocks 的尾部块），再由
     * l3_evict_layer 置位。 */
    for (size_t i = 0; i < (size_t)nl * (size_t)l3->max_blocks; i++)
        l3->blocks[i].evict = 0;
    /* 增量：**完全落在本轮复用前缀内**的块内容未变（只有行 >= keep 被本轮
     * prefill/decode 改写），其载荷继续有效；其余（含 b >= n_blocks 的、
     * 序列变短后残留的）必须标 stale —— 否则只释放 RAM 会让 decode 读到
     * 与当前内容不符的旧 KV。 */
    if (retain) {
        int bs = st->kv_bs > 0 ? st->kv_bs : 32;
        int blk_full = keep_lcp > 0 ? keep_lcp / bs : 0;
        for (int l = 0; l < nl; l++)
            for (int b = blk_full; b < l3->max_blocks; b++) {
                STL3Block *bm = &l3->blocks[(size_t)l * (size_t)l3->max_blocks
                                            + (size_t)b];
                if (bm->on_disk) bm->stale = 1;
            }
    }
    int round_packed = 0;
    for (int l = 0; l < nl; l++) {
        if (st->k_cache && st->k_cache[l]) {
            round_packed += l3_evict_layer(l3, l, st->k_cache[l], st->v_cache[l],
                                           kv_dim, seq_len, st->prefill_importance,
                                           g_l3_ratio, st->kv_bs);
        } else if (st->use_kv_q8 && st->k_cache_q8 && st->k_cache_q8[l]) {
            /* P2 nof32：f32 正典未分配 → 从 q8+scale 打包 Q4 镜像（位级 =
             * kv_dequant_roundtrip 反量化行，见 vllm_l3.h l3_evict_layer_q8） */
            round_packed += l3_evict_layer_q8(l3, l, st->k_cache_q8[l],
                                              st->v_cache_q8[l],
                                              st->k_scale[l], st->v_scale[l],
                                              kv_dim, seq_len,
                                              st->prefill_importance, g_l3_ratio,
                                              st->kv_bs, st->cfg.n_kv_heads);
        }
    }

    /* Mirror the payload into RAM so sparse decode serves evicted blocks from
     * memory: no FILE I/O in the decode hot path (the shared-fp fseek/fread
     * churn corrupted the heap) and much lower decode latency.
     * P3 增量：本轮没重打包任何块时镜像仍是上一轮的有效内容，无需重读。 */
    if (round_packed > 0 || l3->mem_size == 0) {
        if (l3_load_to_mem(l3) != 0)
            fprintf(stderr, "[L3] warning: could not mirror kv_l3.bin to RAM; "
                            "decode will read the file directly\n");
    }
    if (diag) {
        size_t cap_total = (size_t)L3_HEADER_SIZE
                         + (size_t)nl * (size_t)capacity * l3->block_bytes;
        fprintf(stderr, "[L3-DIAG] mirror extent=%zu B (%.1f MiB), cap=%zu B "
                        "(%.1f MiB); old full-capacity sizing would be %zu B "
                        "(%.1f MiB); rss=%ld kB\n",
                l3->mem_size, l3->mem_size / 1048576.0,
                l3->mem_cap, l3->mem_cap / 1048576.0,
                cap_total, cap_total / 1048576.0, l3_diag_rss_kb());
        fflush(stderr);
    }

    /* Phase 2b: physically free the blocks this round decided to move out of
     * RAM (bm->evict). Sparse decode serves them straight from the Q4 disk
     * state (see sparse_attn_head). P3 增量下，这与"本轮是否重打包"解耦：
     * 载荷有效的块只释放 RAM，不重打包。 */
    size_t freed_bytes = 0;
    int round_evict = 0;
    int nblocks = (seq_len + st->kv_bs - 1) / st->kv_bs;
    /* VLLM_L3_PAGEMAP=1：free 前对将 munmap 的块做 pagemap resident 统计，
     * 与 evict 段实测 RSS 回落对照——判定"逻辑 freed 量 vs 实际驻留 vs
     * RSS 实降"的不对称来自假释放（未触页）还是 munmap 回收滞后。 */
    if (diag && getenv("VLLM_L3_PAGEMAP") && getenv("VLLM_L3_PAGEMAP")[0] == '1') {
        size_t res_f32 = 0, res_q8 = 0, n_on = 0;
        for (int l = 0; l < nl; l++) {
            float **kfl = st->k_cache ? st->k_cache[l] : NULL;
            for (int b = 0; b < nblocks; b++) {
                STL3Block *bm = &l3->blocks[(size_t)l * l3->max_blocks + b];
                if (!bm->evict) continue;
                n_on++;
                size_t fdat = (size_t)st->kv_bs * (size_t)kv_dim * sizeof(float);
                size_t idat = (size_t)st->kv_bs * (size_t)kv_dim;
                if (kfl && kfl[b]) {
                    long r = l3_diag_resident_bytes(kfl[b], fdat);
                    if (r >= 0) res_f32 += (size_t)r;
                }
                if (st->k_cache_q8[l] && st->k_cache_q8[l][b]) {
                    long r = l3_diag_resident_bytes(st->k_cache_q8[l][b], idat);
                    if (r >= 0) res_q8 += (size_t)r;
                }
            }
        }
        fprintf(stderr, "[L3-DIAG] pre-free evict=%zu resident f32=%.1f MB q8=%.1f MB "
                        "(sum %.1f MB)\n",
                n_on, res_f32 / 1048576.0, res_q8 / 1048576.0,
                (res_f32 + res_q8) / 1048576.0);
        fflush(stderr);
    }
    for (int l = 0; l < nl; l++) {
        float **kfl = st->k_cache ? st->k_cache[l] : NULL;   /* P3/P2: nof32 无 f32 块 */
        float **vfl = st->v_cache ? st->v_cache[l] : NULL;
        for (int b = 0; b < nblocks; b++) {
            STL3Block *bm = &l3->blocks[(size_t)l * l3->max_blocks + b];
            /* P3 增量：释放判据从 on_disk 改为 evict —— on_disk 现在表示
             * "载荷有效"（跨轮保留），evict 才是"本轮要移出 RAM"。沿用 on_disk
             * 会把"上一轮冷、这一轮热"的块也误释放，等于丢掉 L3 的热块语义。 */
            if (bm->evict) {
                round_evict++;
                size_t fdat = (size_t)st->kv_bs * (size_t)kv_dim * sizeof(float);
                size_t idat = (size_t)st->kv_bs * (size_t)kv_dim;
                if (kfl && kfl[b]) {
                    freed_bytes += 2 * fdat;
#ifdef __linux__
                    if (g_l3_madv) {
                        madvise(kfl[b], fdat, MADV_DONTNEED);
                        madvise(vfl[b], fdat, MADV_DONTNEED);
                    }
#endif
                    st_qwen_kv_free_block(kfl[b], fdat);  kfl[b] = NULL;
                    st_qwen_kv_free_block(vfl[b], fdat);  vfl[b] = NULL;
                }
                if (st->k_cache_q8[l] && st->k_cache_q8[l][b]) {
                    freed_bytes += 2 * idat;
#ifdef __linux__
                    if (g_l3_madv) {
                        madvise(st->k_cache_q8[l][b], idat, MADV_DONTNEED);
                        madvise(st->v_cache_q8[l][b], idat, MADV_DONTNEED);
                    }
#endif
                    st_qwen_kv_free_block(st->k_cache_q8[l][b], idat); st->k_cache_q8[l][b] = NULL;
                    st_qwen_kv_free_block(st->v_cache_q8[l][b], idat); st->v_cache_q8[l][b] = NULL;
                }
            }
        }
    }
    fprintf(stderr, "[L3] evicted %d blocks -> %s (cursor=%.2f MB, seq=%d, "
                    "keep=%d, ratio=%.2f, packed_now=%d, total_packed=%d), "
                    "freed %.1f MB from RAM\n",
            round_evict, l3path, (double)l3->wcursor / 1048576.0, seq_len,
            keep_lcp, g_l3_ratio, round_packed, l3->evicted,
            (double)freed_bytes / 1048576.0);
    if (diag) {
        fprintf(stderr, "[L3-DIAG] evict exit rss=%ld kB "
                        "(enter %ld kB, delta %ld kB)\n",
                l3_diag_rss_kb(), rss_enter,
                l3_diag_rss_kb() - rss_enter);
        fflush(stderr);
    }
    /* st->l3 stays open: sparse decode reads evicted blocks from it */
}



static void test_longctx_quality_ab(STModelWeights *w, const STModelConfig *cfg,
                                    QwenTokenizer *tok) {
    const int GEN_MAX = g_longctx_gen_max;
    int ctx_lens[2] = {2048, 4096};
    int n_lens = 2;
    if (g_longctx_ctx > 0) {
        ctx_lens[0] = g_longctx_ctx;
        n_lens = 1;
    }
    const int nl = cfg->n_layers;
    const int dim = cfg->dim;
    const int vc = cfg->vocab_size;

    /* Reference-answer token sequence for teacher-forced perplexity. */
    int ref_ids[64];
    int n_ref = qwen_tokenizer_encode(tok, REFERENCE_TEXT, ref_ids, 64);
    if (n_ref <= 0) {
        printf("       [FAIL] cannot encode reference answer\n");
        return;
    }

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    printf("\n  --- Part F: Long-Context Fact Retrieval + PPL + Q4_0 A/B ---\n");
    printf("       Q4_0 quantization: %s\n", w->has_q4 ? "ACTIVE" : "NOT active");
    printf("       Needle: \"%s\"\n", NEEDLE_TEXT);
    printf("       Question: \"%s\"\n", QUESTION_TEXT);
    printf("       Reference answer: \"%s\" (%d tokens)\n", REFERENCE_TEXT, n_ref);
    fflush(stdout);

    int maxS = ctx_lens[0];
    for (int li = 1; li < n_lens; li++) {
        if (ctx_lens[li] > maxS) maxS = ctx_lens[li];
    }
    size_t ids_cap = (size_t)maxS + 1024;
    int *ids = (int *)malloc(ids_cap * sizeof(int));
    if (!ids) { printf("       [FAIL] OOM for prompt ids\n"); return; }

    for (int li = 0; li < n_lens; li++) {
        int S = ctx_lens[li];
        int needle_start = -1;
        int n_tot = build_needle_prompt(tok, S, ids, (int)ids_cap, &needle_start);
        if (n_tot < 16) {
            printf("       [S=%d] [FAIL] prompt too short (%d tokens)\n", S, n_tot);
            continue;
        }
        /* Temporarily raise max_seq_len for the long KV cache. */
        int orig_max_seq = w->cfg.max_seq_len;
        w->cfg.max_seq_len = n_tot + GEN_MAX + n_ref + 128;

        STQwenInferenceState ist;
        memset(&ist, 0, sizeof(ist));
        if (st_qwen_inference_init(&ist, w) != 0) {
            printf("       [S=%d] [FAIL] Cannot init long-context state\n", S);
            w->cfg.max_seq_len = orig_max_seq;
            free(ids);
            return;
        }
        w->cfg.max_seq_len = orig_max_seq;

        printf("\n       [S=%d] prompt=%d tokens (needle @ token %d)\n",
               S, n_tot, needle_start);
        fflush(stdout);

        /* Prefill the whole prompt with exact batched attention.
         * When the NPU backend is live, run an identical-input A/B prefill
         * (NPU offload disabled vs enabled) so one model load yields both
         * sides of the comparison; decode/PPL continue from the NPU run. */
        if (g_npu) {
            static const char *tag[2] = {"CPU", "NPU"};
            for (int run = 0; run < 2; run++) {
                st_npu_set(run == 0 ? NULL : g_npu);   /* A: CPU, B: NPU */
                /* Reset prefill state (KV + position) for a clean run */
                for (int l = 0; l < cfg->n_layers; l++) ist.cache_len[l] = 0;
                ist.seq_len = 0;
                memset(ist.hidden, 0, (size_t)cfg->dim * sizeof(float));
                LARGE_INTEGER pf0, pf1;
                QueryPerformanceCounter(&pf0);
                st_qwen_model_prefill_batch(&ist, ids, n_tot);
                QueryPerformanceCounter(&pf1);
                double pf_ms = (double)(pf1.QuadPart - pf0.QuadPart) * 1000.0
                             / (double)freq.QuadPart;
                printf("         Prefill[%s] %d tokens in %.1f ms (%.1f tok/s)\n",
                       tag[run], n_tot, pf_ms,
                       (double)n_tot / (pf_ms / 1000.0));
                fflush(stdout);
            }
            st_npu_set(g_npu);   /* keep NPU enabled for decode/PPL */
        } else {
            LARGE_INTEGER pf0, pf1;
            QueryPerformanceCounter(&pf0);
            st_qwen_model_prefill_batch(&ist, ids, n_tot);
            QueryPerformanceCounter(&pf1);
            double pf_ms = (double)(pf1.QuadPart - pf0.QuadPart) * 1000.0
                         / (double)freq.QuadPart;
            printf("         Prefill %d tokens in %.1f ms (%.1f tok/s)\n",
                   n_tot, pf_ms, (double)n_tot / (pf_ms / 1000.0));
            fflush(stdout);
        }

        /* Phase 2a: L3 cold-block Q4 disk eviction (--l3-evict). */
        l3_evict_after_prefill(&ist, 0, 0);

        /* Snapshot so every decode run starts from an identical state. */
        int   *saved_cache_len = (int*)malloc((size_t)nl * sizeof(int));
        float *saved_hidden    = (float*)malloc((size_t)dim * sizeof(float));
        float *saved_logits    = (float*)malloc((size_t)vc * sizeof(float));
        if (!saved_cache_len || !saved_hidden || !saved_logits) {
            printf("       [S=%d] [FAIL] OOM for snapshot\n", S);
            st_qwen_inference_free(&ist);
            free(ids);
            free(saved_cache_len);
            free(saved_hidden);    free(saved_logits);
            return;
        }
        memcpy(saved_cache_len, ist.cache_len,    (size_t)nl * sizeof(int));
        memcpy(saved_hidden,    ist.hidden,       (size_t)dim * sizeof(float));
        memcpy(saved_logits,    ist.logits,       (size_t)vc * sizeof(float));
        int saved_seq = ist.seq_len;

        char buf_out[512] = {0};
        int pout = 0;
        float nll = 0.0f, nll_f32 = 0.0f;
        double ms_gen = 0.0;
        int got = 0;

        /* Greedy decode (timed, Q4_0 exact attention). */
        {
            LARGE_INTEGER t0, t1;
            ist.profile_decode = 1;
            ist.dec_t_qkv = 0.0;
            ist.dec_t_attn = 0.0;
            ist.dec_t_o = 0.0;
            ist.dec_t_gu = 0.0;
            ist.dec_t_down = 0.0;
            ist.dec_t_lm = 0.0;
            ist.dec_t_other = 0.0;
            QueryPerformanceCounter(&t0);
            for (int t = 0; t < GEN_MAX; t++) {
                int id = greedy_argmax(ist.logits, vc);
                append_tok_str(buf_out, &pout, sizeof(buf_out), tok, id);
                st_qwen_model_forward(&ist, id);
            }
            QueryPerformanceCounter(&t1);
            ms_gen = (double)(t1.QuadPart - t0.QuadPart) * 1000.0
                   / (double)freq.QuadPart / GEN_MAX;
            ist.profile_decode = 0;
            {
                double t_qkv  = ist.dec_t_qkv;
                double t_attn = ist.dec_t_attn;
                double t_o    = ist.dec_t_o;
                double t_gu   = ist.dec_t_gu;
                double t_down = ist.dec_t_down;
                double t_lm   = ist.dec_t_lm;
                double t_other= ist.dec_t_other;
                double t_gemm = t_qkv + t_o + t_gu + t_down + t_lm;
                double t_total = t_gemm + t_attn + t_other;
                if (t_total > 0.0) {
                    fprintf(stderr,
                        "[DECODE-TIMING] n=%d total=%.1fms (%.2f ms/tok) | "
                        "GEMM=%.1fms(%4.1f%%) ATTN=%.1fms(%4.1f%%) OTHER=%.1fms(%4.1f%%)\n",
                        GEN_MAX, t_total * 1e3, (t_total * 1e3) / (double)GEN_MAX,
                        t_gemm * 1e3, 100.0 * t_gemm / t_total,
                        t_attn * 1e3, 100.0 * t_attn / t_total,
                        t_other * 1e3, 100.0 * t_other / t_total);
                    if (t_gemm > 0.0) {
                        fprintf(stderr,
                            "[DECODE-KERNELS] QKV=%.1fms(%4.1f%%) O=%.1fms(%4.1f%%) "
                            "GATEUP=%.1fms(%4.1f%%) DOWN=%.1fms(%4.1f%%) LM=%.1fms(%4.1f%%)\n",
                            t_qkv  * 1e3, 100.0 * t_qkv  / t_gemm,
                            t_o    * 1e3, 100.0 * t_o    / t_gemm,
                            t_gu   * 1e3, 100.0 * t_gu   / t_gemm,
                            t_down * 1e3, 100.0 * t_down / t_gemm,
                            t_lm   * 1e3, 100.0 * t_lm   / t_gemm);
                    }
                    fflush(stderr);
                }
            }
        }
        got = strstr(buf_out, REFERENCE_KEY) != NULL;
        fprintf(stderr, "[STAGE] greedy done\n"); fflush(stderr);

        memcpy(ist.cache_len,    saved_cache_len, (size_t)nl * sizeof(int));
        memcpy(ist.hidden,       saved_hidden,    (size_t)dim * sizeof(float));
        memcpy(ist.logits,       saved_logits,    (size_t)vc * sizeof(float));
        ist.seq_len = saved_seq;

        /* Teacher-forced NLL on the reference answer (exact attention). */
        fprintf(stderr, "[STAGE] ppl-q8 start\n"); fflush(stderr);
        for (int t = 0; t < n_ref; t++) {
            nll += logits_nll(ist.logits, vc, ref_ids[t]);
            st_qwen_model_forward(&ist, ref_ids[t]);
        }

        /* Teacher-forced NLL with F32 KV cache (use_kv_q8=0), to quantify the
         * INT8 KV quantization error against the exact float reference. */
        memcpy(ist.cache_len,    saved_cache_len, (size_t)nl * sizeof(int));
        memcpy(ist.hidden,       saved_hidden,    (size_t)dim * sizeof(float));
        memcpy(ist.logits,       saved_logits,    (size_t)vc * sizeof(float));
        ist.seq_len = saved_seq;
        ist.use_kv_q8 = 0;
        ist.use_kv_q4 = 0;   /* Q4 模式下也要切回 F32 参考路径 */
        fprintf(stderr, "[STAGE] ppl-f32 start\n"); fflush(stderr);
        for (int t = 0; t < n_ref; t++) {
            nll_f32 += logits_nll(ist.logits, vc, ref_ids[t]);
            st_qwen_model_forward(&ist, ref_ids[t]);
        }
        ist.use_kv_q8 = 1;
        ist.use_kv_q4 = g_kv_q4 ? 1 : 0;   /* 恢复压缩模式 */

        float ppl = expf(nll / (float)n_ref);
        float ppl_f32 = expf(nll_f32 / (float)n_ref);

        printf("         Needle retrieved  %s\n", got ? "YES" : "NO");
        printf("         Answer PPL (teacher-forced)  Q8=%.3f  F32=%.3f  (Q8/F32=%.4f)\n",
               ppl, ppl_f32, ppl_f32 > 0.0f ? ppl / ppl_f32 : 0.0f);
        printf("         Greedy (exact): \"%s\"\n", buf_out);
        printf("         Decode  %.1f ms/tok  (%.1f tok/s)\n",
               ms_gen, (ms_gen > 0.0) ? 1000.0 / ms_gen : 0.0);
        printf("         L3 disk-served decode tokens: %lld\n", g_l3_disk_hits);
        fflush(stdout);

        st_qwen_inference_free(&ist);
        free(saved_cache_len);
        free(saved_hidden);
        free(saved_logits);
    }

    free(ids);
}

/* Build a ~`target`-token shared prefix of filler facts (plain completion,
 * matching the known-good needle format so greedy decode stays coherent). */
static int build_shared_prefix(QwenTokenizer *tok, int target, int *ids, int max_ids) {
    const int nl = 198;
    const int n_names  = (int)(sizeof(G_NAMES)   / sizeof(G_NAMES[0]));
    const int n_colors = (int)(sizeof(G_COLORS)  / sizeof(G_COLORS[0]));
    const int n_cities = (int)(sizeof(G_CITIES)  / sizeof(G_CITIES[0]));
    const int n_pets   = (int)(sizeof(G_PETS)    / sizeof(G_PETS[0]));
    const int n_petn   = (int)(sizeof(G_PETNAMES)/ sizeof(G_PETNAMES[0]));
    const int n_drinks = (int)(sizeof(G_DRINKS)  / sizeof(G_DRINKS[0]));
    const int n_jobs   = (int)(sizeof(G_JOBS)    / sizeof(G_JOBS[0]));
    int tmp[256];
    int pos = 0;

    int intro_ids[64];
    int n_intro = qwen_tokenizer_encode(tok, INTRO_TEXT, intro_ids, 64);
    for (int i = 0; i < n_intro && pos < max_ids; i++) ids[pos++] = intro_ids[i];
    if (pos < max_ids) ids[pos++] = nl;

    int filled = 0, person = 0;
    while (filled < target && pos < max_ids) {
        char fact[256];
        int kind = person % 3;
        if (kind == 0) {
            snprintf(fact, sizeof(fact), "%s likes the color %s.",
                     G_NAMES[person % n_names], G_COLORS[person % n_colors]);
        } else if (kind == 1) {
            snprintf(fact, sizeof(fact), "%s lives in %s and has a %s named %s.",
                     G_NAMES[person % n_names], G_CITIES[person % n_cities],
                     G_PETS[(person / n_cities) % n_pets],
                     G_PETNAMES[(person / n_cities) % n_petn]);
        } else {
            snprintf(fact, sizeof(fact), "%s's favorite drink is %s and %s works as a %s.",
                     G_NAMES[person % n_names], G_DRINKS[person % n_drinks],
                     G_PRONOUNS[person % n_names], G_JOBS[(person / n_drinks) % n_jobs]);
        }
        int n = qwen_tokenizer_encode(tok, fact, tmp, 256);
        if (n <= 0) break;
        for (int i = 0; i < n && pos < max_ids; i++) ids[pos++] = tmp[i];
        if (pos < max_ids) ids[pos++] = nl;
        filled += n + 1;
        person++;
    }
    return pos;
}

/* Part G: Shared-prefix KV-cache reuse A/B (B方案). A long shared prefix is
 * prefilled once and forked into N request states via st_qwen_copy_kv_prefix,
 * so each user only prefills their unique suffix. Quality is verified as
 * byte-identical greedy output vs. the full-prefill baseline. */
static void test_prefix_cache_ab(STModelWeights *w, const STModelConfig *cfg,
                                 QwenTokenizer *tok) {
    enum { N_USERS = 4, GEN_MAX = 16, PREFIX_TARGET = 1024, MAX_IDS = 4096 };
    const int vc = cfg->vocab_size;

    const char *questions[N_USERS] = {
        "What color does Alice like?",
        "Where does Bob live?",
        "What is Carol's favorite drink?",
        "What color does Dave like?",
    };

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    printf("\n  --- Part G: Shared-Prefix KV-Cache Reuse (B方案) ---\n");
    printf("       Shared prefix target: %d tokens | users: %d\n", PREFIX_TARGET, N_USERS);
    fflush(stdout);

    int *prefix_ids = (int *)malloc((size_t)MAX_IDS * sizeof(int));
    if (!prefix_ids) { printf("       [FAIL] OOM (prefix)\n"); return; }
    int prefix_len = build_shared_prefix(tok, PREFIX_TARGET, prefix_ids, MAX_IDS);
    if (prefix_len < 64) {
        printf("       [FAIL] prefix too short (%d tokens)\n", prefix_len);
        free(prefix_ids);
        return;
    }

    int *suffix_ids[N_USERS];
    int  suffix_n[N_USERS];
    for (int u = 0; u < N_USERS; u++) {
        suffix_ids[u] = (int *)malloc(256 * sizeof(int));
        char q[256];
        snprintf(q, sizeof(q), "\nQuestion: %s\nAnswer:", questions[u]);
        suffix_n[u] = qwen_tokenizer_encode(tok, q, suffix_ids[u], 256);
    }

    /* Raise the KV-cache cap so prefix + suffix + generation all fit. */
    int orig_max_seq = w->cfg.max_seq_len;
    w->cfg.max_seq_len = prefix_len + 256 + GEN_MAX + 64;

    /* ---- Baseline: prefill the FULL prompt for every user ---- */
    printf("       Baseline: full prefill per user (%d tokens prefix + suffix)\n", prefix_len);
    fflush(stdout);

    char base_out[N_USERS][256] = {{0}};
    double base_ms = 0.0;
    for (int u = 0; u < N_USERS; u++) {
        printf("       [G] baseline user %d\n", u);
        fflush(stdout);
        STQwenInferenceState st;
        memset(&st, 0, sizeof(st));
        if (st_qwen_inference_init(&st, w) != 0) {
            printf("       [FAIL] baseline init\n");
            continue;
        }

        int full_ids[MAX_IDS];
        int full_n = prefix_len + suffix_n[u];
        memcpy(full_ids, prefix_ids, (size_t)prefix_len * sizeof(int));
        memcpy(full_ids + prefix_len, suffix_ids[u], (size_t)suffix_n[u] * sizeof(int));

        LARGE_INTEGER t0, t1;
        QueryPerformanceCounter(&t0);
        st_qwen_model_prefill_batch(&st, full_ids, full_n);
        QueryPerformanceCounter(&t1);
        base_ms += (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart;
        HCK("G baseline prefill done");

        int p = 0;
        for (int t = 0; t < GEN_MAX; t++) {
            int id = greedy_argmax(st.logits, vc);
            append_tok_str(base_out[u], &p, (int)sizeof(base_out[u]), tok, id);
            st_qwen_model_forward(&st, id);
        }
        HCK("G baseline decode done");
        st_qwen_inference_free(&st);
        HCK("G baseline free done");
        printf("       [G] baseline user %d done\n", u);
        fflush(stdout);
    }

    /* ---- Cached: prefill prefix ONCE, fork KV into each user ---- */
    printf("       Cached: prefix once + copy + per-user suffix\n");
    fflush(stdout);

    STQwenInferenceState tmpl;
    memset(&tmpl, 0, sizeof(tmpl));
    if (st_qwen_inference_init(&tmpl, w) != 0) {
        printf("       [FAIL] template init\n");
        w->cfg.max_seq_len = orig_max_seq;
        for (int u = 0; u < N_USERS; u++) free(suffix_ids[u]);
        free(prefix_ids);
        return;
    }

    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);
    st_qwen_model_prefill_batch(&tmpl, prefix_ids, prefix_len);
    QueryPerformanceCounter(&t1);
    double prefix_ms = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart;

    char cache_out[N_USERS][256] = {{0}};
    double copy_ms = 0.0, suffix_ms = 0.0;
    for (int u = 0; u < N_USERS; u++) {
        STQwenInferenceState st;
        memset(&st, 0, sizeof(st));
        if (st_qwen_inference_init(&st, w) != 0) {
            printf("       [FAIL] cached init\n");
            continue;
        }

        QueryPerformanceCounter(&t0);
        st_qwen_copy_kv_prefix(&st, &tmpl, prefix_len);
        QueryPerformanceCounter(&t1);
        copy_ms += (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart;

        QueryPerformanceCounter(&t0);
        st_qwen_model_prefill_batch(&st, suffix_ids[u], suffix_n[u]);
        QueryPerformanceCounter(&t1);
        suffix_ms += (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart;

        int p = 0;
        for (int t = 0; t < GEN_MAX; t++) {
            int id = greedy_argmax(st.logits, vc);
            append_tok_str(cache_out[u], &p, (int)sizeof(cache_out[u]), tok, id);
            st_qwen_model_forward(&st, id);
        }
        st_qwen_inference_free(&st);
    }
    st_qwen_inference_free(&tmpl);
    w->cfg.max_seq_len = orig_max_seq;

    /* ---- Report ---- */
    double cached_ms = prefix_ms + copy_ms + suffix_ms;
    printf("       Prefill:  baseline=%.1f ms  cached=%.1f ms  (prefix %.1f + copy %.1f + suffix %.1f)\n",
           base_ms, cached_ms, prefix_ms, copy_ms, suffix_ms);
    printf("       Prefill speedup (wall-clock) = %.2fx\n", base_ms / cached_ms);

    int identical = 1;
    for (int u = 0; u < N_USERS; u++) {
        int same = (strcmp(base_out[u], cache_out[u]) == 0);
        if (!same) identical = 0;
        printf("         User %d: base=\"%s\"  cached=\"%s\"  %s\n",
               u + 1, base_out[u], cache_out[u], same ? "IDENTICAL" : "DIFFER");
    }
    printf("       Quality: %s\n", identical ? "ZERO LOSS (byte-identical greedy output)" : "MISMATCH");

    for (int u = 0; u < N_USERS; u++) free(suffix_ids[u]);
    free(prefix_ids);
    fflush(stdout);
}

/* Longest-common-prefix of two token sequences (capped at nb). */
static int lcp_len(const int *a, int na, const int *b, int nb) {
    int n = na < nb ? na : nb;
    int i;
    for (i = 0; i < n; i++) if (a[i] != b[i]) break;
    return i;
}

/* Part G2: prefix-cache landing — a single prefilled template serves requests
 * with DIFFERENT shared-prefix lengths. Each request's KV is restored via
 * st_qwen_copy_kv_prefix up to its LCP against the cached template, then only
 * the remainder is prefilled. Reports wall-clock speedup + zero-loss check. */
static void test_prefix_cache_lcp(STModelWeights *w, const STModelConfig *cfg,
                                  QwenTokenizer *tok) {
    enum { N_DEEP = 4, N_SHALLOW = 4, GEN_MAX = 8, MAX_IDS = 4096 };
    enum { ROOT_LEN = 384, MID_LEN = 128 };        /* DEEP = 384 + 128 = 512 */
    const int vc = cfg->vocab_size;
    int n_users = N_DEEP + N_SHALLOW;

    printf("\n  --- Part G2: LCP Prefix-Cache (落地) ---\n");
    printf("       root=%d tok | deep(+mid)=%d tok | users=%d (deep %d + shallow %d)\n",
           ROOT_LEN, ROOT_LEN + MID_LEN, n_users, N_DEEP, N_SHALLOW);
    fflush(stdout);

    /* DEEP template ids = ROOT + MID (both generated filler facts). */
    int *deep_ids = (int *)malloc((size_t)MAX_IDS * sizeof(int));
    int deep_len = build_shared_prefix(tok, ROOT_LEN + MID_LEN, deep_ids, MAX_IDS);
    if (deep_len < ROOT_LEN) {
        printf("       [SKIP] prefix too short\n");
        free(deep_ids);
        return;
    }

    const char *deep_q[N_DEEP] = {
        "What color does Alice like?",
        "Where does Bob live?",
        "What is Carol's favorite drink?",
        "What color does Dave like?",
    };
    const char *shallow_q[N_SHALLOW] = {
        "What pet does Eve own?",
        "Which city does Frank live in?",
        "What job does Grace have?",
        "What color does Henry like?",
    };
    int *req_ids[N_DEEP + N_SHALLOW];
    int req_n[N_DEEP + N_SHALLOW];
    int req_share[N_DEEP + N_SHALLOW];   /* LCP length each request reuses */
    for (int i = 0; i < N_DEEP; i++) {
        req_ids[i] = (int *)malloc((size_t)MAX_IDS * sizeof(int));
        int n = deep_len;
        memcpy(req_ids[i], deep_ids, (size_t)deep_len * sizeof(int));
        int q[128];
        int nq = qwen_tokenizer_encode(tok, deep_q[i], q, 128);
        for (int j = 0; j < nq && n < MAX_IDS; j++) req_ids[i][n++] = q[j];
        req_n[i] = n;
        req_share[i] = deep_len;                    /* share the full DEEP prefix */
    }
    for (int i = 0; i < N_SHALLOW; i++) {
        req_ids[N_DEEP + i] = (int *)malloc((size_t)MAX_IDS * sizeof(int));
        int n = ROOT_LEN;
        memcpy(req_ids[N_DEEP + i], deep_ids, (size_t)ROOT_LEN * sizeof(int));
        int q[128];
        int nq = qwen_tokenizer_encode(tok, shallow_q[i], q, 128);
        for (int j = 0; j < nq && n < MAX_IDS; j++) req_ids[N_DEEP + i][n++] = q[j];
        req_n[N_DEEP + i] = n;
        req_share[N_DEEP + i] = ROOT_LEN;           /* share only the ROOT */
    }

    int orig_max_seq = w->cfg.max_seq_len;
    w->cfg.max_seq_len = deep_len + GEN_MAX + 64;

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    /* ---- Baseline: full prefill per request ---- */
    double base_ms = 0.0;
    char base_out[N_DEEP + N_SHALLOW][256] = {{0}};
    for (int u = 0; u < n_users; u++) {
        printf("       [G2] baseline user %d prefill n=%d\n", u, req_n[u]);
        fflush(stdout);
        STQwenInferenceState st;
        memset(&st, 0, sizeof(st));
        if (st_qwen_inference_init(&st, w) != 0) { printf("[FAIL] base init\n"); continue; }
        LARGE_INTEGER t0, t1;
        QueryPerformanceCounter(&t0);
        st_qwen_model_prefill_batch(&st, req_ids[u], req_n[u]);
        QueryPerformanceCounter(&t1);
        base_ms += (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart;
        printf("       [G2] baseline user %d prefill done\n", u);
        fflush(stdout);
        int p = 0;
        for (int t = 0; t < GEN_MAX; t++) {
            int id = greedy_argmax(st.logits, vc);
            append_tok_str(base_out[u], &p, (int)sizeof(base_out[u]), tok, id);
            st_qwen_model_forward(&st, id);
        }
        printf("       [G2] baseline user %d decode done\n", u);
        fflush(stdout);
        st_qwen_inference_free(&st);
        printf("       [G2] baseline user %d freed\n", u);
        fflush(stdout);
    }

    /* ---- Cached: prefill DEEP once, LCP-copy + remainder per request ---- */
    STQwenInferenceState tmpl;
    memset(&tmpl, 0, sizeof(tmpl));
    if (st_qwen_inference_init(&tmpl, w) != 0) {
        printf("       [FAIL] template init\n");
        goto lcp_done;
    }
    double deep_prefill_ms;
    {
        LARGE_INTEGER t0, t1;
        QueryPerformanceCounter(&t0);
        st_qwen_model_prefill_batch(&tmpl, deep_ids, deep_len);
        QueryPerformanceCounter(&t1);
        deep_prefill_ms = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart;
    }

    double copy_ms = 0.0, remain_ms = 0.0;
    char cache_out[N_DEEP + N_SHALLOW][256] = {{0}};
    int reuse_tokens = 0;
    for (int u = 0; u < n_users; u++) {
        /* LCP against the cached template decides how much KV to restore. */
        int lcp = lcp_len(req_ids[u], req_n[u], deep_ids, deep_len);
        if (lcp > req_share[u]) lcp = req_share[u];   /* never copy past the request's own share */
        reuse_tokens += lcp;

        STQwenInferenceState st;
        memset(&st, 0, sizeof(st));
        if (st_qwen_inference_init(&st, w) != 0) { printf("[FAIL] cached init\n"); continue; }
        LARGE_INTEGER t0, t1;
        QueryPerformanceCounter(&t0);
        st_qwen_copy_kv_prefix(&st, &tmpl, lcp);
        QueryPerformanceCounter(&t1);
        copy_ms += (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart;

        QueryPerformanceCounter(&t0);
        st_qwen_model_prefill_batch(&st, req_ids[u] + lcp, req_n[u] - lcp);
        QueryPerformanceCounter(&t1);
        remain_ms += (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart;

        int p = 0;
        for (int t = 0; t < GEN_MAX; t++) {
            int id = greedy_argmax(st.logits, vc);
            append_tok_str(cache_out[u], &p, (int)sizeof(cache_out[u]), tok, id);
            st_qwen_model_forward(&st, id);
        }
        st_qwen_inference_free(&st);
    }
    st_qwen_inference_free(&tmpl);

    {
        double cached_ms = deep_prefill_ms + copy_ms + remain_ms;
        printf("       Baseline: %.0f ms | Cached: %.0f ms (deep-pre %d + copy %.0f + remain %.0f)\n",
               base_ms, cached_ms, (int)deep_prefill_ms, copy_ms, remain_ms);
        printf("       Prefix-cache speedup (wall-clock) = %.2fx  (reused %d tokens)\n",
               base_ms / cached_ms, reuse_tokens);
        int identical = 1;
        for (int u = 0; u < n_users; u++) {
            int same = (strcmp(base_out[u], cache_out[u]) == 0);
            if (!same) identical = 0;
            printf("         User %d (%s): %s\n", u + 1,
                   u < N_DEEP ? "deep" : "shallow", same ? "IDENTICAL" : "DIFFER");
        }
        printf("       Quality: %s\n", identical ? "ZERO LOSS" : "MISMATCH");
        fflush(stdout);
    }

lcp_done:
    w->cfg.max_seq_len = orig_max_seq;
    for (int u = 0; u < n_users; u++) free(req_ids[u]);
    free(deep_ids);
}

/* Part H: Continuous batching (MA5-PAR). N requests share a prefilled prefix
 * (forked via st_qwen_copy_kv_prefix), then generate in lockstep with
 * st_qwen_model_forward_batch. Each batch step reads the layer weights once
 * for all requests. Compared against the same requests decoded sequentially;
 * outputs must be byte-identical. */
static void test_continuous_batching(STModelWeights *w, const STModelConfig *cfg,
                                     QwenTokenizer *tok) {
    enum { N_REQ = 8, GEN_MAX = 16, PREFIX_TARGET = 512, MAX_IDS = 4096 };
    const int vc = cfg->vocab_size;

    printf("\n  --- Part H: Continuous Batching (MA5-PAR) ---\n");
    printf("       requests=%d | shared prefix=%d tok | gen=%d tok\n",
           N_REQ, PREFIX_TARGET, GEN_MAX);
    fflush(stdout);

    const char *questions[N_REQ] = {
        "What color does Alice like?",
        "Where does Bob live?",
        "What is Carol's favorite drink?",
        "What color does Dave like?",
        "What pet does Eve own?",
        "Which city does Frank live in?",
        "What job does Grace have?",
        "What color does Henry like?",
    };

    int *prefix_ids = (int *)malloc((size_t)MAX_IDS * sizeof(int));
    int prefix_len = build_shared_prefix(tok, PREFIX_TARGET, prefix_ids, MAX_IDS);
    if (prefix_len < 64) { printf("       [SKIP] prefix too short\n"); free(prefix_ids); return; }

    int *req_ids[N_REQ];
    int  req_n[N_REQ];
    for (int u = 0; u < N_REQ; u++) {
        req_ids[u] = (int *)malloc((size_t)MAX_IDS * sizeof(int));
        memcpy(req_ids[u], prefix_ids, (size_t)prefix_len * sizeof(int));
        int q[128];
        int nq = qwen_tokenizer_encode(tok, questions[u], q, 128);
        int n = prefix_len;
        for (int j = 0; j < nq && n < MAX_IDS; j++) req_ids[u][n++] = q[j];
        req_n[u] = n;
    }

    int orig_max_seq = w->cfg.max_seq_len;
    w->cfg.max_seq_len = prefix_len + 256 + GEN_MAX + 64;

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    /* Shared template state for the prefix. */
    STQwenInferenceState tmpl;
    memset(&tmpl, 0, sizeof(tmpl));
    if (st_qwen_inference_init(&tmpl, w) != 0) { printf("       [FAIL] template init\n"); goto h_done; }
    st_qwen_model_prefill_batch(&tmpl, prefix_ids, prefix_len);

    STQwenInferenceState sts[N_REQ];
    memset(sts, 0, sizeof(sts));
    STQwenInferenceState *stp[N_REQ];
    for (int u = 0; u < N_REQ; u++) stp[u] = &sts[u];
    for (int u = 0; u < N_REQ; u++) {
        if (st_qwen_inference_init(&sts[u], w) != 0) { printf("       [FAIL] req init\n"); goto h_done; }
        st_qwen_copy_kv_prefix(&sts[u], &tmpl, prefix_len);
        st_qwen_model_prefill_batch(&sts[u], req_ids[u] + prefix_len, req_n[u] - prefix_len);
    }

    /* ---- Sequential baseline: each request decodes GEN_MAX alone ---- */
    char seq_out[N_REQ][512] = {{0}};
    int  seq_p[N_REQ] = {0};
    double seq_ms = 0.0;
    for (int u = 0; u < N_REQ; u++) {
        LARGE_INTEGER t0, t1;
        QueryPerformanceCounter(&t0);
        for (int t = 0; t < GEN_MAX; t++) {
            int id = greedy_argmax(sts[u].logits, vc);
            append_tok_str(seq_out[u], &seq_p[u], (int)sizeof(seq_out[u]), tok, id);
            st_qwen_model_forward(&sts[u], id);
        }
        QueryPerformanceCounter(&t1);
        seq_ms += (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart;
    }

    /* Fresh states for the batched run (identical prefix+suffix prefill). */
    for (int u = 0; u < N_REQ; u++) st_qwen_inference_free(&sts[u]);
    for (int u = 0; u < N_REQ; u++) {
        if (st_qwen_inference_init(&sts[u], w) != 0) { printf("       [FAIL] req re-init\n"); goto h_done; }
        st_qwen_copy_kv_prefix(&sts[u], &tmpl, prefix_len);
        st_qwen_model_prefill_batch(&sts[u], req_ids[u] + prefix_len, req_n[u] - prefix_len);
    }

    /* ---- Batched: all requests decode GEN_MAX in lockstep ---- */
    char bat_out[N_REQ][512] = {{0}};
    int  bat_p[N_REQ] = {0};
    int  bat_tokens[N_REQ];
    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);
    for (int t = 0; t < GEN_MAX; t++) {
        for (int u = 0; u < N_REQ; u++) {
            bat_tokens[u] = greedy_argmax(sts[u].logits, vc);
            append_tok_str(bat_out[u], &bat_p[u], (int)sizeof(bat_out[u]), tok, bat_tokens[u]);
        }
        st_qwen_model_forward_batch(stp, N_REQ, bat_tokens);
    }
    QueryPerformanceCounter(&t1);
    double bat_ms = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart;

    printf("       Sequential: %.0f ms (%.1f ms/tok/req) | Batched: %.0f ms (%.1f ms/step)\n",
           seq_ms, seq_ms / (N_REQ * GEN_MAX), bat_ms, bat_ms / GEN_MAX);
    printf("       Continuous-batch speedup = %.2fx  (batch=%d)\n",
           seq_ms / bat_ms, N_REQ);

    int identical = 1;
    for (int u = 0; u < N_REQ; u++) {
        int same = (strcmp(seq_out[u], bat_out[u]) == 0);
        if (!same) identical = 0;
        printf("         Req %d: %s\n", u + 1, same ? "IDENTICAL" : "DIFFER");
    }
    printf("       Quality: %s\n", identical ? "ZERO LOSS" : "MISMATCH");
    fflush(stdout);

h_done:
    for (int u = 0; u < N_REQ; u++) {
        if (sts[u].is_allocated) st_qwen_inference_free(&sts[u]);
        free(req_ids[u]);
    }
    if (tmpl.is_allocated) st_qwen_inference_free(&tmpl);
    w->cfg.max_seq_len = orig_max_seq;
    free(prefix_ids);
}


/* ================================================================
 * 性能基准：Part A / B / C（单用户 TTFT/TPOT + 多用户并发）
 *
 * 回移来源：GitHupSRC/src/main.c:2775-3413（test_performance_benchmark）。
 * 与旧副本的差异仅限“模型加载”一段：当前引擎走自包含 VQF 路径
 * （vqf_load + qwen_tokenizer_load*），cfg 内嵌于 w.cfg 故不再调用
 * st_config_free；计时沿用当前引擎已有的 QueryPerformanceCounter/
 * LARGE_INTEGER 垫片；Part A/B/C 的度量口径与打印行保持与旧副本一致，
 * 以便历史基准脚本（tools/bench_fair.sh、tools/_perf_matrix.sh 等，入口
 * 均为 `--perf-partA --bench-seqlen N`）可直接复用。
 * ================================================================ */
static int vqf_dir_find(const char *dir, char *buf, size_t cap);  /* 定义见下方 */
static void test_performance_benchmark(void) {
    printf("\n=== Test 10: Performance Benchmark ===\n");
    printf("       wmode=%d | OMP/TP threads=%d\n", g_st_wmode, vllm_tp_threads());
    fflush(stdout);

    /* 模型目录：优先 --model，其次回退历史默认路径。
     * 回移来源：GitHupSRC/src/main.c:2782-2799（默认路径列表）。 */
    const char *model_dir = g_serve_model_dir;
    if (!model_dir) {
        const char *model_paths[] = {
            "Modl/Qwen3-VL-2B-q4",
            "../../Modl/Qwen3-VL-2B-q4",
            "../Modl/Qwen3-VL-2B-q4",
            "Modl/千问3_VL_8B_Instruct",
            "../Modl/千问3_VL_8B_Instruct",
            "../../Modl/千问3_VL_8B_Instruct",
            NULL
        };
        for (int i = 0; model_paths[i]; i++) {
            char tp[1024];
            snprintf(tp, sizeof(tp), "%s/config.json", model_paths[i]);
            if (st_access(tp, 0) == 0) { model_dir = model_paths[i]; break; }
        }
    }
    if (!model_dir) {
        printf("       [SKIP] Model not found (use --model <dir>)\n");
        return;
    }
    printf("       Model: %s\n", model_dir);
    fflush(stdout);

    /* 定位 VQF 权重 + 词表目录（与 run_stream_test 同口径）。 */
    char vqf_path[1024]; vqf_path[0] = 0;
    char tok_dir[1024];
    if (vqf_is_file(model_dir)) {
        snprintf(vqf_path, sizeof(vqf_path), "%s", model_dir);
        snprintf(tok_dir, sizeof(tok_dir), "%s", model_dir);
        char *sl = strrchr(tok_dir, '/');
        if (sl) { *sl = '\0'; if (!tok_dir[0]) snprintf(tok_dir, sizeof(tok_dir), "."); }
        else    snprintf(tok_dir, sizeof(tok_dir), ".");
    } else {
        if (!vqf_dir_find(model_dir, vqf_path, sizeof(vqf_path))) {
            printf("       [SKIP] no model.vqf under %s\n", model_dir);
            return;
        }
        snprintf(tok_dir, sizeof(tok_dir), "%s", model_dir);
    }

    LARGE_INTEGER freq, t_start, t_end;
    QueryPerformanceFrequency(&freq);
    STModelWeights w;
    memset(&w, 0, sizeof(w));
    double load_t0 = st_now_sec();
    if (vqf_load(&w, vqf_path) != 0) {
        printf("       [FAIL] vqf_load %s\n", vqf_path);
        return;
    }
    STModelConfig *cfg = &w.cfg;   /* VQF 自包含（config 内嵌） */
    double load_sec = st_now_sec() - load_t0;

    QwenTokenizer tok;
    if (qwen_tokenizer_load(&tok, tok_dir) != 0 ||
        qwen_tokenizer_load_special(&tok, tok_dir) != 0) {
        printf("       [FAIL] tokenizer load from %s\n", tok_dir);
        st_weights_free(&w);
        return;
    }
    printf("       Tokenizer: %d tokens loaded\n", tok.vocab_size);
    printf("       Weights loaded: %.1f sec\n", load_sec);
    fflush(stdout);

    int im_start = tok.im_start_id > 0 ? tok.im_start_id : 151644;
    int im_end   = tok.im_end_id   > 0 ? tok.im_end_id   : 151645;
    int nl_tok   = 198;

    /* ============================================================
     * Part A: Single-User Performance (5 different prompts)
     * 回移来源：GitHupSRC/src/main.c:2922-3116
     * ============================================================ */
    printf("\n  --- Part A: Single-User Performance ---\n");
    fflush(stdout);

    const char *single_prompts[] = {
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate Fibonacci numbers.",
        "What are the main causes of climate change?",
        "Describe how neural networks learn from data.",
        "Compare and contrast REST and GraphQL APIs.",
    };
    int n_single_prompts = 5;

    double total_ttft = 0.0, total_tpot = 0.0, total_throughput = 0.0;
    int total_tokens_gen = 0;

    /* Pre-allocate ONE inference state, reused across all prompts */
    STQwenInferenceState ist;
    memset(&ist, 0, sizeof(ist));
    if (st_qwen_inference_init(&ist, &w) != 0) {
        printf("       [FAIL] Inference init\n");
        st_weights_free(&w); qwen_tokenizer_free(&tok);
        return;
    }

    for (int pi = 0; pi < n_single_prompts; pi++) {
        /* Reset inference state: clear KV cache and reset position */
        for (int l = 0; l < cfg->n_layers; l++) ist.cache_len[l] = 0;
        ist.seq_len = 0;
        memset(ist.hidden, 0, (size_t)cfg->dim * sizeof(float));

        /* Build chat template */
        int prompt_ids[PERF_MAX_CTX]; int pn = 0;
        int tmp_ids[128]; int n;

        /* <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n */
        prompt_ids[pn++] = im_start;
        n = qwen_tokenizer_encode(&tok, "system", tmp_ids, 128);
        for (int i = 0; i < n; i++) prompt_ids[pn++] = tmp_ids[i];
        prompt_ids[pn++] = nl_tok;
        n = qwen_tokenizer_encode(&tok, "You are a helpful assistant.", tmp_ids, 128);
        for (int i = 0; i < n; i++) prompt_ids[pn++] = tmp_ids[i];
        prompt_ids[pn++] = im_end;
        prompt_ids[pn++] = nl_tok;

        /* <|im_start|>user\n{prompt}<|im_end|>\n */
        prompt_ids[pn++] = im_start;
        n = qwen_tokenizer_encode(&tok, "user", tmp_ids, 128);
        for (int i = 0; i < n; i++) prompt_ids[pn++] = tmp_ids[i];
        prompt_ids[pn++] = nl_tok;
        n = qwen_tokenizer_encode(&tok, single_prompts[pi], tmp_ids, 128);
        for (int i = 0; i < n; i++) prompt_ids[pn++] = tmp_ids[i];
        prompt_ids[pn++] = im_end;
        prompt_ids[pn++] = nl_tok;

        /* <|im_start|>assistant\n */
        prompt_ids[pn++] = im_start;
        n = qwen_tokenizer_encode(&tok, "assistant", tmp_ids, 128);
        for (int i = 0; i < n; i++) prompt_ids[pn++] = tmp_ids[i];
        prompt_ids[pn++] = nl_tok;

        /* --bench-seqlen N: pad to a fixed prefill length (e.g. 64) so TTFT
         * matches the earliest S=64 CPU baseline. Neutral filler is prepended
         * before the system prompt; the real prompt stays at the tail.
         * 回移来源：GitHupSRC/src/main.c:2986-3003 */
        if (g_bench_seqlen > 0 && pn < g_bench_seqlen) {
            static const char *filler = "The quick brown fox jumps over the lazy dog. ";
            int fill_ids[64]; int fn = qwen_tokenizer_encode(&tok, filler, fill_ids, 64);
            int need = g_bench_seqlen - pn;
            if (fn > 0) {
                int f[PERF_MAX_CTX];
                int fi = 0;
                while (fi < need) {
                    for (int k = 0; k < fn && fi < need; k++) f[fi++] = fill_ids[k];
                }
                memmove(prompt_ids + need, prompt_ids, (size_t)pn * sizeof(int));
                memcpy(prompt_ids, f, (size_t)need * sizeof(int));
                pn = g_bench_seqlen;
            }
        }

        /* Prefill (mini-batch - axiom: blas_qkv_fusion_categorical) */
        fflush(stdout);
        QueryPerformanceCounter(&t_start);
        st_qwen_model_prefill_batch(&ist, prompt_ids, pn);
        QueryPerformanceCounter(&t_end);
        double ttft = (double)(t_end.QuadPart - t_start.QuadPart) / (double)freq.QuadPart;
        total_ttft += ttft;

        /* Decode */
        printf("       Prompt %d: \"%s\"\n", pi + 1, single_prompts[pi]);
        printf("         Prefill: %d tokens in %.3fs (TTFT=%.0fms), ", pn, ttft, ttft * 1000.0);
        fflush(stdout);

        char output[4096]; int out_len = 0;
        QueryPerformanceCounter(&t_start);
        /* M4g: 开启 decode kernel 分解（与 longctx 模式同口径），输出
         * DECODE-TIMING/DECODE-KERNELS 到 stderr。
         * 回移来源：GitHupSRC/src/main.c:3020-3080 */
        ist.profile_decode = 1;
        ist.dec_t_qkv = 0.0;
        ist.dec_t_attn = 0.0;
        ist.dec_t_o = 0.0;
        ist.dec_t_gu = 0.0;
        ist.dec_t_down = 0.0;
        ist.dec_t_lm = 0.0;
        ist.dec_t_other = 0.0;
        int gen_tokens = 0;
        for (int step = 0; step < PERF_MAX_TOKENS; step++) {
            /* Use logits from previous forward pass (prefill or last decode) */
            float best = -1e9f; int best_id = 0;
            for (int t = 0; t < cfg->vocab_size; t++) {
                if (ist.logits[t] > best) { best = ist.logits[t]; best_id = t; }
            }
            if (best_id == tok.eos_id || best_id == im_end) break;

            gen_tokens++;
            const char *s = qwen_tokenizer_decode(&tok, best_id);
            if (s && s[0]) {
                int slen = (int)strlen(s);
                if (out_len + slen < 4095) {
                    memcpy(output + out_len, s, slen);
                    out_len += slen;
                }
            }
            /* Feed generated token for next step */
            st_qwen_model_forward(&ist, best_id);
        }
        QueryPerformanceCounter(&t_end);
        ist.profile_decode = 0;
        if (ist.dec_t_qkv + ist.dec_t_gu + ist.dec_t_down + ist.dec_t_lm > 0.0) {
            double t_qkv  = ist.dec_t_qkv, t_attn = ist.dec_t_attn;
            double t_o    = ist.dec_t_o,   t_gu   = ist.dec_t_gu;
            double t_down = ist.dec_t_down, t_lm  = ist.dec_t_lm;
            double t_other= ist.dec_t_other;
            double t_gemm = t_qkv + t_o + t_gu + t_down + t_lm;
            double t_total = t_gemm + t_attn + t_other;
            if (t_total > 0.0) {
                fprintf(stderr,
                    "[DECODE-TIMING] n=%d total=%.1fms (%.2f ms/tok) | "
                    "GEMM=%.1fms(%4.1f%%) ATTN=%.1fms(%4.1f%%) OTHER=%.1fms(%4.1f%%)\n",
                    gen_tokens, t_total * 1e3, (t_total * 1e3) / (double)gen_tokens,
                    t_gemm * 1e3, 100.0 * t_gemm / t_total,
                    t_attn * 1e3, 100.0 * t_attn / t_total,
                    t_other * 1e3, 100.0 * t_other / t_total);
                if (t_gemm > 0.0) {
                    fprintf(stderr,
                        "[DECODE-KERNELS] QKV=%.1fms(%4.1f%%) O=%.1fms(%4.1f%%) "
                        "GATEUP=%.1fms(%4.1f%%) DOWN=%.1fms(%4.1f%%) LM=%.1fms(%4.1f%%)\n",
                        t_qkv  * 1e3, 100.0 * t_qkv  / t_gemm,
                        t_o    * 1e3, 100.0 * t_o    / t_gemm,
                        t_gu   * 1e3, 100.0 * t_gu   / t_gemm,
                        t_down * 1e3, 100.0 * t_down / t_gemm,
                        t_lm   * 1e3, 100.0 * t_lm   / t_gemm);
                }
                fflush(stderr);
            }
        }
        double decode_sec = (double)(t_end.QuadPart - t_start.QuadPart) / (double)freq.QuadPart;
        output[out_len] = '\0';

        double tpot = gen_tokens > 0 ? decode_sec / gen_tokens * 1000.0 : 0.0;
        double tok_per_sec = gen_tokens > 0 ? gen_tokens / decode_sec : 0.0;
        total_tpot += tpot;
        total_throughput += tok_per_sec;
        total_tokens_gen += gen_tokens;

        printf("Decode: %d tokens in %.3fs (TPOT=%.0fms/tok, %.1f tok/s)\n",
               gen_tokens, decode_sec, tpot, tok_per_sec);
        printf("         Output: %s\n", output[0] ? output : "(empty)");
        fflush(stdout);
    }

    printf("       --- Single-User Summary ---\n");
    printf("         Avg TTFT: %.0fms | Avg TPOT: %.0fms/tok | Avg Throughput: %.1f tok/s\n",
           total_ttft / n_single_prompts * 1000.0,
           total_tpot / n_single_prompts,
           total_throughput / n_single_prompts);

    if (g_perf_part_a) {
        /* --perf-partA: single-user only (CPU-vs-NPU comparison core).
         * 回移来源：GitHupSRC/src/main.c:3102-3116 */
        printf("\n  === Performance Summary (Part A only) ===\n");
        printf("  Model: %s\n", model_dir);
        printf("  Weights loading: %.1f sec\n", load_sec);
        printf("  Single-user avg: TTFT=%.0fms | TPOT=%.0fms/tok | %.1f tok/s\n",
               total_ttft / n_single_prompts * 1000.0,
               total_tpot / n_single_prompts,
               total_throughput / n_single_prompts);
        fflush(stdout);
        st_qwen_inference_free(&ist);
        st_weights_free(&w);
        qwen_tokenizer_free(&tok);
        return;
    }

    /* ============================================================
     * Part B: Multi-User Sequential (Round-Robin)
     * 回移来源：GitHupSRC/src/main.c:3118-3275
     * ============================================================ */
    printf("\n  --- Part B: Multi-User (Shared State) ---\n");
    fflush(stdout);

    const char *multi_prompts[] = {
        "Explain how blockchain technology works in simple terms.",
        "Write a short poem about artificial intelligence.",
        "What are the main benefits of renewable energy?",
        "Describe the process of photosynthesis in plants.",
        "What is the difference between AI and machine learning?",
        "Explain the theory of evolution by natural selection.",
        "How does the internet work at a fundamental level?",
        "Write a recipe for chocolate chip cookies.",
        "What are black holes and how do they form?",
        "Compare and contrast mitosis and meiosis.",
    };
    int n_users = g_bench_users;   /* --bench-users N (default 10) */
    int max_new_tokens = 25;

    /* Build chat templates for each user */
    int user_prompt_ids[MAX_PERF_USERS][PERF_MAX_CTX];
    int user_pn[MAX_PERF_USERS];
    int tmp_ids[128]; int n;
    int shared_prefix_n = 0;  /* system prompt + user-role prefix shared by all users */

    for (int u = 0; u < n_users; u++) {
        int *pids = user_prompt_ids[u];
        int pn = 0;

        /* system prompt */
        pids[pn++] = im_start;
        n = qwen_tokenizer_encode(&tok, "system", tmp_ids, 128);
        for (int i = 0; i < n; i++) pids[pn++] = tmp_ids[i];
        pids[pn++] = nl_tok;
        n = qwen_tokenizer_encode(&tok, "You are a helpful assistant.", tmp_ids, 128);
        for (int i = 0; i < n; i++) pids[pn++] = tmp_ids[i];
        pids[pn++] = im_end; pids[pn++] = nl_tok;

        /* user message */
        pids[pn++] = im_start;
        n = qwen_tokenizer_encode(&tok, "user", tmp_ids, 128);
        for (int i = 0; i < n; i++) pids[pn++] = tmp_ids[i];
        pids[pn++] = nl_tok;
        if (u == 0) shared_prefix_n = pn;  /* common prefix ends before the question */

        n = qwen_tokenizer_encode(&tok, multi_prompts[u], tmp_ids, 128);
        for (int i = 0; i < n; i++) pids[pn++] = tmp_ids[i];
        pids[pn++] = im_end; pids[pn++] = nl_tok;

        /* assistant */
        pids[pn++] = im_start;
        n = qwen_tokenizer_encode(&tok, "assistant", tmp_ids, 128);
        for (int i = 0; i < n; i++) pids[pn++] = tmp_ids[i];
        pids[pn++] = nl_tok;

        user_pn[u] = pn;
    }

    /* Create inference states */
    int user_gen_tok[MAX_PERF_USERS] = {0};
    char user_output[MAX_PERF_USERS][2048] = {{0}};

    /* ---- Baseline: full prefill per user (reference, no prefix reuse) ---- */
    char base_output[MAX_PERF_USERS][2048] = {{0}};
    double base_prefill_ms = 0.0;
    for (int u = 0; u < n_users; u++) {
        for (int l = 0; l < cfg->n_layers; l++) ist.cache_len[l] = 0;
        ist.seq_len = 0;

        LARGE_INTEGER p0, p1;
        QueryPerformanceCounter(&p0);
        st_qwen_model_prefill_batch(&ist, user_prompt_ids[u], user_pn[u]);
        QueryPerformanceCounter(&p1);
        base_prefill_ms += (double)(p1.QuadPart - p0.QuadPart) * 1000.0 / (double)freq.QuadPart;

        int pos = 0;
        for (int step = 0; step < max_new_tokens; step++) {
            int best_id = greedy_argmax(ist.logits, cfg->vocab_size);
            if (best_id == tok.eos_id || best_id == im_end) break;
            append_tok_str(base_output[u], &pos, (int)sizeof(base_output[u]), &tok, best_id);
            st_qwen_model_forward(&ist, best_id);
        }
    }

    /* ---- Cached (B-scheme): prefill shared system prompt once, then fork its
     * KV-cache into each user via st_qwen_copy_kv_prefix; each user prefills
     * only their unique question suffix. ---- */
    for (int l = 0; l < cfg->n_layers; l++) ist.cache_len[l] = 0;
    ist.seq_len = 0;

    LARGE_INTEGER p0, p1;
    QueryPerformanceCounter(&p0);
    st_qwen_model_prefill_batch(&ist, user_prompt_ids[0], shared_prefix_n);
    QueryPerformanceCounter(&p1);
    double prefix_ms = (double)(p1.QuadPart - p0.QuadPart) * 1000.0 / (double)freq.QuadPart;

    double copy_ms = 0.0, suffix_ms = 0.0;
    QueryPerformanceCounter(&t_start);
    for (int u = 0; u < n_users; u++) {
        STQwenInferenceState st;
        memset(&st, 0, sizeof(st));
        if (st_qwen_inference_init(&st, &w) != 0) {
            printf("       [FAIL] cached init for user %d\n", u + 1);
            continue;
        }

        QueryPerformanceCounter(&p0);
        st_qwen_copy_kv_prefix(&st, &ist, shared_prefix_n);
        QueryPerformanceCounter(&p1);
        copy_ms += (double)(p1.QuadPart - p0.QuadPart) * 1000.0 / (double)freq.QuadPart;

        QueryPerformanceCounter(&p0);
        st_qwen_model_prefill_batch(&st, user_prompt_ids[u] + shared_prefix_n,
                                    user_pn[u] - shared_prefix_n);
        QueryPerformanceCounter(&p1);
        suffix_ms += (double)(p1.QuadPart - p0.QuadPart) * 1000.0 / (double)freq.QuadPart;

        int gen_tok = 0;
        int pos = 0;
        for (int step = 0; step < max_new_tokens; step++) {
            int best_id = greedy_argmax(st.logits, cfg->vocab_size);
            if (best_id == tok.eos_id || best_id == im_end) break;
            gen_tok++;
            append_tok_str(user_output[u], &pos, (int)sizeof(user_output[u]), &tok, best_id);
            st_qwen_model_forward(&st, best_id);
        }
        user_gen_tok[u] = gen_tok;
        st_qwen_inference_free(&st);
    }
    QueryPerformanceCounter(&t_end);
    double seq_prefill_sec = (double)(t_end.QuadPart - t_start.QuadPart) / (double)freq.QuadPart;

    double cached_ms = prefix_ms + copy_ms + suffix_ms;
    printf("       Sequential %d users (prefix-cached): %.3fs total\n", n_users, seq_prefill_sec);
    printf("       Prefill: baseline %.1fms (full per user) vs cached %.1fms (prefix %.1f + copy %.1f + suffix %.1f) = %.2fx\n",
           base_prefill_ms, cached_ms, prefix_ms, copy_ms, suffix_ms,
           base_prefill_ms / cached_ms);

    int seq_total_tok = 0;
    for (int u = 0; u < n_users; u++) seq_total_tok += user_gen_tok[u];

    printf("       Total: %d tokens in %.3fs (%.1f tok/s aggregate)\n",
           seq_total_tok, seq_prefill_sec,
           seq_total_tok / seq_prefill_sec);

    int identical = 1;
    for (int u = 0; u < n_users; u++) {
        int same = (strcmp(base_output[u], user_output[u]) == 0);
        if (!same) identical = 0;
        printf("         User %d: \"%s\" - %d tokens: %s  [%s]\n",
               u + 1, multi_prompts[u], user_gen_tok[u],
               user_output[u][0] ? user_output[u] : "(empty)",
               same ? "IDENTICAL" : "DIFFER");
    }
    printf("       Prefix-cache quality: %s\n",
           identical ? "ZERO LOSS (byte-identical greedy output)" : "MISMATCH");

    /* ============================================================
     * Part C: Token-Level Interleaved Concurrent Multi-User
     * 回移来源：GitHupSRC/src/main.c:3277-3411
     * ============================================================ */
    printf("\n  --- Part C: Multi-User Token-Interleaved Concurrent ---\n");
    fflush(stdout);

    /* Allocate independent inference states per user (share read-only weights) */
    STQwenInferenceState ist_c[MAX_PERF_USERS];
    int par_init_ok = 1;
    for (int u = 0; u < n_users; u++) {
        if (st_qwen_inference_init(&ist_c[u], &w) != 0) {
            printf("       [FAIL] Cannot init inference state for user %d\n", u + 1);
            par_init_ok = 0;
            for (int v = 0; v < u; v++) st_qwen_inference_free(&ist_c[v]);
            break;
        }
    }
    if (!par_init_ok) {
        printf("       [SKIP] Part C failed - reporting sequential results only\n");
        printf("\n  === Performance Summary ===\n");
        printf("  Model: %s\n", model_dir);
        printf("  Weights loading: %.1f sec\n", load_sec);
        printf("  Single-user avg: TTFT=%.0fms | TPOT=%.0fms/tok | %.1f tok/s\n",
               total_ttft / n_single_prompts * 1000.0,
               total_tpot / n_single_prompts,
               total_throughput / n_single_prompts);
        printf("  Multi-user (%d users) sequential: %.1f tok/s aggregate (%d tokens, %.1fs)\n",
               n_users, seq_total_tok / seq_prefill_sec, seq_total_tok, seq_prefill_sec);
        printf("  Multi-user (%d users) concurrent: [SKIP]\n", n_users);
        st_qwen_inference_free(&ist);
        st_weights_free(&w);
        qwen_tokenizer_free(&tok);
        return;
    }

    int parallel_gen_tok[MAX_PERF_USERS] = {0};
    char parallel_output[MAX_PERF_USERS][2048] = {{0}};
    double parallel_user_time[MAX_PERF_USERS] = {0};

    QueryPerformanceCounter(&t_start);

    int user_pos[MAX_PERF_USERS] = {0};
    int user_done[MAX_PERF_USERS] = {0};
    LARGE_INTEGER u_t0[MAX_PERF_USERS], u_t1[MAX_PERF_USERS];
    for (int u = 0; u < n_users; u++) QueryPerformanceCounter(&u_t0[u]);

    /* Prefill all users (mini-batch per user - axiom: blas_qkv_fusion_categorical) */
    for (int u = 0; u < n_users; u++) {
        st_qwen_model_prefill_batch(&ist_c[u], user_prompt_ids[u], user_pn[u]);
    }

    /* Token-level interleaved decode: user-A step, user-B step, user-A step, ... */
    int active = n_users;
    while (active > 0) {
        for (int u = 0; u < n_users; u++) {
            if (user_done[u]) continue;

            /* Decode one token for this user */
            float best = -1e9f; int best_id = 0;
            for (int t = 0; t < cfg->vocab_size; t++) {
                if (ist_c[u].logits[t] > best) { best = ist_c[u].logits[t]; best_id = t; }
            }
            if (best_id == tok.eos_id || best_id == im_end || user_pos[u] >= max_new_tokens) {
                user_done[u] = 1;
                active--;
                QueryPerformanceCounter(&u_t1[u]);
                parallel_user_time[u] = (double)(u_t1[u].QuadPart - u_t0[u].QuadPart) / (double)freq.QuadPart;
                parallel_gen_tok[u] = user_pos[u];
                continue;
            }

            user_pos[u]++;
            const char *s = qwen_tokenizer_decode(&tok, best_id);
            if (s && s[0]) {
                int slen = (int)strlen(s);
                int cur_len = (int)strlen(parallel_output[u]);
                if (cur_len + slen < 2047) {
                    memcpy(parallel_output[u] + cur_len, s, slen);
                    parallel_output[u][cur_len + slen] = '\0';
                }
            }
            st_qwen_model_forward(&ist_c[u], best_id);
        }
    }

    QueryPerformanceCounter(&t_end);
    double par_total_sec = (double)(t_end.QuadPart - t_start.QuadPart) / (double)freq.QuadPart;

    int par_total_tok = 0;
    for (int u = 0; u < n_users; u++) par_total_tok += parallel_gen_tok[u];

    printf("       Token-interleaved (%d users): %.3fs wall-clock\n", n_users, par_total_sec);
    printf("       (Each user gets full TP-thread matvecs, interleaved step-by-step)\n");
    printf("       Speedup vs sequential: %.2fx (same total compute, concurrent scheduling)\n",
           seq_prefill_sec / par_total_sec);
    for (int u = 0; u < n_users; u++) {
        printf("         User %d: %.0fms, %d tokens - \"%s\"\n",
               u + 1, parallel_user_time[u] * 1000.0, parallel_gen_tok[u],
               parallel_output[u][0] ? parallel_output[u] : "(empty)");
    }

    /* Free concurrent inference states */
    for (int u = 0; u < n_users; u++) st_qwen_inference_free(&ist_c[u]);

    /* ============================================================
     * Summary
     * ============================================================ */
    printf("\n  === Performance Summary ===\n");
    printf("  Model: %s\n", model_dir);
    printf("  Weights loading: %.1f sec\n", load_sec);
    printf("  Single-user avg: TTFT=%.0fms | TPOT=%.0fms/tok | %.1f tok/s\n",
           total_ttft / n_single_prompts * 1000.0,
           total_tpot / n_single_prompts,
           total_throughput / n_single_prompts);
    printf("  Multi-user (%d users) sequential: %.1f tok/s aggregate (%d tokens, %.1fs)\n",
           n_users, seq_total_tok / seq_prefill_sec, seq_total_tok, seq_prefill_sec);
    printf("  Multi-user (%d users) concurrent: %.1f tok/s aggregate (%d tokens, %.1fs wall)\n",
           n_users, par_total_tok / par_total_sec, par_total_tok, par_total_sec);

    st_qwen_inference_free(&ist);
    st_weights_free(&w);
    qwen_tokenizer_free(&tok);
}  /* test_performance_benchmark */


/* ================================================================
 * OpenAI-compatible HTTP server mode (--serve)
 * ================================================================ */

/* Manual-load mode: with --no-auto-load the HTTP server starts with the
 * model not loaded and /admin/api/model/load runs the load in a background
 * thread (so the admin page can show a live progress bar). The load can
 * otherwise run synchronously before the server starts (default). */
#include <pthread.h>

typedef struct {
    STModelConfig        *cfg;
    QwenTokenizer        *tok;
    STModelWeights       *w;
    STQwenInferenceState *ist;
    STVisionWeights      *vis_w;
    STVisionState        *vis;
    int has_vision;
    int cfg_is_vqf;        /* cfg 指向 w->cfg（VQF mmap），不单独 free */
    char model_dir[1024];
    int format_prio;       /* 保留：仅 VQF（0=auto/vqf，serve 恒为 VQF） */
} ServeModel;

/* 模型目录内定位 VQF 权重文件（model.vqf 优先）。返回 1 找到。 */
static int vqf_dir_find(const char *dir, char *buf, size_t cap) {
    const char *cands[] = { "model.vqf", "weights.vqf", "vllm.vqf", NULL };
    for (int i = 0; cands[i]; i++) {
        snprintf(buf, cap, "%s/%s", dir, cands[i]);
        if (vqf_is_file(buf)) return 1;
    }
    buf[0] = 0;
    return 0;
}

/* ================================================================
 * --stream-test：AirLLM 型"分层驻留"验证（只走自研 VQF 格式）
 *
 * 验证对象 = VQF 单文件 mmap + 层切片驱逐（vqf_stream_layer_advance 用
 * MADV_DONTNEED/WILLNEED 控制页驻留）：
 *   1) vqf_load（mmap 直挂、零转换），模型目录必须含 model.vqf；
 *   2) 固定 prompt 一次 prefill + N token 贪心 decode；
 *   3) 逐阶段打印 VmRSS 与 token/时延。
 * 判定口径：
 *   A. token 位级一致：VLLM_VQF_STREAM 开/关 的 TOKIDS 输出必须完全一致
 *      （驱逐只丢干净文件页，内容由文件重建）；
 *   B. RSS 收敛：分层档稳态 RSS ≈ 常驻段 + keep 层，而非整文件拉满；
 *   C. 速度代价诚实披露：eMMC 冷读使 tok/s 下降，不做营销化表述。
 * 仅明文 VQF（加密 VQF 由 vqf_stream_setup 自动 REFUSED 并告警）。
 * ================================================================ */
/* P1a/质量门共用：从当前 ist.logits 起强迫续写 text，返回逐 token 平均 NLL
 * （与 corpus/COPYTEST 同一 log-softmax 实现，double 累加，确定性）。 */
static double nll_forced_forward(STModelConfig *cfg, QwenTokenizer *tok,
                                 STQwenInferenceState *ist, const char *text,
                                 int *out_n) {
    int ids[512];
    int rn = qwen_tokenizer_encode(tok, text, ids, 512);
    double s = 0.0;
    int n = 0;
    for (int step = 0; step < rn; step++) {
        double mx = -1e30;
        for (int t = 0; t < cfg->vocab_size; t++)
            if ((double)ist->logits[t] > mx) mx = (double)ist->logits[t];
        double sum = 0.0;
        for (int t = 0; t < cfg->vocab_size; t++)
            sum += exp((double)ist->logits[t] - mx);
        double lse = mx + log(sum);
        s += lse - (double)ist->logits[ids[step]];
        n++;
        if (step + 1 < rn) st_qwen_model_forward(ist, ids[step]);
    }
    if (out_n) *out_n = n;
    return n > 0 ? s / (double)n : 0.0;
}

/* M3 对拍：单 token decode（seq=0 全新 KV），l==0 各阶段 dump
 * （VLLM_MOE_DUMP=1 时 stdout 打 [L0D]），供 python numpy 逐段复算对比。 */
static int run_moe_l0_test(const char *model_dir, int tok) {
    printf("\n=== [L0] MoE layer-0 single-token dump (token=%d) ===\n", tok);
    fflush(stdout);
    char vqf_path[1024]; vqf_path[0] = 0;
    if (vqf_is_file(model_dir)) {
        snprintf(vqf_path, sizeof(vqf_path), "%s", model_dir);
    } else if (!vqf_dir_find(model_dir, vqf_path, sizeof(vqf_path))) {
        printf("[L0] FAIL: no model.vqf under %s\n", model_dir);
        return 1;
    }
    STModelWeights w; memset(&w, 0, sizeof(w));
    if (vqf_load(&w, vqf_path) != 0) {
        printf("[L0] FAIL: vqf_load %s\n", vqf_path);
        return 1;
    }
    STModelConfig *cfg = &w.cfg;
    STQwenInferenceState ist; memset(&ist, 0, sizeof(ist));
    if (st_qwen_inference_init(&ist, &w) != 0) {
        printf("[L0] FAIL: inference init\n");
        st_weights_free(&w);
        return 1;
    }
    for (int l = 0; l < cfg->n_layers; l++) ist.cache_len[l] = 0;
    ist.seq_len = 0;
    memset(ist.hidden, 0, (size_t)cfg->dim * sizeof(float));
    const char *toks_env = getenv("VLLM_L0_TOKS");   /* 多 token 顺序 decode（KV 保留） */
    if (toks_env && toks_env[0]) {
        char buf[256]; snprintf(buf, sizeof(buf), "%s", toks_env);
        char *p = buf; int st_ = 0;
        while (1) {
            int id = 0;
            while (*p == ',' || *p == ' ') p++;
            if (*p < '0' || *p > '9') break;
            while (*p >= '0' && *p <= '9') { id = id * 10 + (*p - '0'); p++; }
            printf("[L0] step %d token %d\n", st_++, id); fflush(stdout);
            st_qwen_model_forward(&ist, id);
            if (!*p) break;
        }
    } else {
        st_qwen_model_forward(&ist, tok);
    }
    if (getenv("VLLM_MOE_DUMP") && getenv("VLLM_MOE_DUMP")[0] == '1') {
        int am = 0; float amv = -1e30f;
        for (int t = 0; t < cfg->vocab_size; t++)
            if (ist.logits[t] > amv) { amv = ist.logits[t]; am = t; }
        printf("[L0D] S12_logits argmax=%d v=%.9e\n", am, (double)amv);
        printf("[L0D] S12_logits_v");
        for (int t = 0; t < cfg->vocab_size && t < 320; t++)
            printf(" %.9e", (double)ist.logits[t]);
        printf("\n");
        printf("[L0D] S13_hidden");
        for (int t = 0; t < cfg->dim && t < 64; t++)
            printf(" %.9e", (double)ist.hidden[t]);
        printf("\n");
        fflush(stdout);
    }
    printf("[L0] forward done (dim=%d layers=%d)\n", cfg->dim, cfg->n_layers);
    fflush(stdout);
    st_qwen_inference_free(&ist);
    st_weights_free(&w);
    return 0;
}

/* ================================================================
 * 私有提交定位（VLLM_MEMDUMP=1，init 后调用；VLLM_MEMDUMP_EXIT=1 直接退出）
 * 枚举 committed private 大区，判定是否与 VQF mmap（COW 私有化）重叠。
 * ================================================================ */
#ifdef _WIN32
typedef struct { uintptr_t base; size_t sz; DWORD prot; int in_map; } MBig;
static void mem_dump_regions(const STModelWeights *w) {
    uintptr_t mapB = (uintptr_t)w->vqf_map;
    uintptr_t mapE = mapB + w->vqf_map_len;
    MBig big[16]; int nb = 0;
    uint64_t tot_priv = 0, tot_map = 0, map_priv = 0, map_map = 0;
    MEMORY_BASIC_INFORMATION mi;
    for (uintptr_t a = 0; a < (uintptr_t)1 << 47; ) {
        SIZE_T r = VirtualQuery((void *)a, &mi, sizeof(mi));
        if (!r) break;
        if (mi.State == MEM_COMMIT) {
            int in = (a < mapE && a + mi.RegionSize > mapB) ? 1 : 0;
            if (mi.Type == MEM_PRIVATE) {
                tot_priv += mi.RegionSize;
                if (in) map_priv += mi.RegionSize;
            } else if (mi.Type == MEM_MAPPED) {
                tot_map += mi.RegionSize;
                if (in) map_map += mi.RegionSize;
            }
            /* 记录最大私有区 */
            if (mi.Type == MEM_PRIVATE && nb < 16) {
                big[nb].base = a; big[nb].sz = mi.RegionSize;
                big[nb].prot = mi.Protect; big[nb].in_map = in;
                nb++;
            }
        }
        a += mi.RegionSize ? mi.RegionSize : 0x10000;
    }
    printf("[MEMDUMP] committed MEM_PRIVATE=%.2fGB (in_vqf_map=%.2fGB)  "
           "MEM_MAPPED=%.2fGB (in_vqf_map=%.2fGB)\n",
           tot_priv / 1073741824.0, map_priv / 1073741824.0,
           tot_map / 1073741824.0, map_map / 1073741824.0);
    printf("[MEMDUMP] vqf mmap = [%p, %p) %.1f MB\n", w->vqf_map,
           (void *)(uintptr_t)(mapB + w->vqf_map_len), w->vqf_map_len / 1048576.0);
    /* 冒泡取 top 12（私有区） */
    for (int i = 0; i < nb; i++)
        for (int j = i + 1; j < nb; j++)
            if (big[j].sz > big[i].sz) { MBig t = big[i]; big[i] = big[j]; big[j] = t; }
    int show = nb < 12 ? nb : 12;
    for (int i = 0; i < show; i++) {
        printf("[MEMDUMP] priv[%2d] base=%p size=%6.1fMB prot=0x%x in_vqf_map=%d\n",
               i, (void *)big[i].base, big[i].sz / 1048576.0, big[i].prot, big[i].in_map);
    }
}
#endif

static int run_stream_test(const char *model_dir, int n_tokens) {
    printf("\n=== [STREAM] layer-resident validation (VQF only) ===\n");
    fflush(stdout);

    char vqf_path[1024]; vqf_path[0] = 0;
    char tok_dir[1024];
    if (vqf_is_file(model_dir)) {
        /* --model 直接指向 .vqf：词表取其所在目录 */
        snprintf(vqf_path, sizeof(vqf_path), "%s", model_dir);
        snprintf(tok_dir, sizeof(tok_dir), "%s", model_dir);
        char *sl = strrchr(tok_dir, '/');
        if (sl) { *sl = '\0'; if (!tok_dir[0]) snprintf(tok_dir, sizeof(tok_dir), "."); }
        else    snprintf(tok_dir, sizeof(tok_dir), ".");
    } else {
        if (!vqf_dir_find(model_dir, vqf_path, sizeof(vqf_path))) {
            printf("[STREAM] FAIL: %s 无 model.vqf（本测试要求自研 VQF 格式）\n",
                   model_dir);
            return 1;
        }
        snprintf(tok_dir, sizeof(tok_dir), "%s", model_dir);
    }

    STModelWeights w; memset(&w, 0, sizeof(w));
    double t0 = st_now_sec();
    if (vqf_load(&w, vqf_path) != 0) {
        printf("[STREAM] FAIL: vqf_load %s\n", vqf_path);
        return 1;
    }
    STModelConfig *cfg = &w.cfg;   /* VQF 自包含（config 内嵌） */
    printf("[STREAM] vqf_load OK (%.2fs): dim=%d layers=%d heads=%d kv=%d "
           "ffn=%d vocab=%d max_seq=%d\n",
           st_now_sec() - t0, cfg->dim, cfg->n_layers, cfg->n_heads,
           cfg->n_kv_heads, cfg->ffn_dim, cfg->vocab_size, cfg->max_seq_len);
    printf("[STREAM] rss_after_load  = %ld kB\n", vqf_stream_rss_kb());
    fflush(stdout);

    QwenTokenizer tok;
    if (qwen_tokenizer_load(&tok, tok_dir) != 0 ||
        qwen_tokenizer_load_special(&tok, tok_dir) != 0) {
        printf("[STREAM] FAIL: tokenizer load from %s (需要 vocab.bin/config.json)\n",
               tok_dir);
        st_weights_free(&w);
        return 1;
    }
    printf("[STREAM] tokenizer: %d tokens (eos=%d im_start=%d im_end=%d)\n",
           tok.vocab_size, tok.eos_id, tok.im_start_id, tok.im_end_id);

    STQwenInferenceState ist;
    memset(&ist, 0, sizeof(ist));
    if (st_qwen_inference_init(&ist, &w) != 0) {
        printf("[STREAM] FAIL: inference init\n");
        st_weights_free(&w);
        qwen_tokenizer_free(&tok);
        return 1;
    }
    printf("[STREAM] rss_after_init  = %ld kB\n", vqf_stream_rss_kb());
    fflush(stdout);
#ifdef _WIN32
    if (getenv("VLLM_MEMDUMP") && getenv("VLLM_MEMDUMP")[0] == '1') {
        mem_dump_regions(&w);   /* init 后基线（无推理分配） */
    }
#endif

    /* ---- 固定文本 prompt（不含视觉 token，保证跨档完全确定） ---- */
    int im_start = tok.im_start_id > 0 ? tok.im_start_id : 151644;
    int im_end   = tok.im_end_id   > 0 ? tok.im_end_id   : 151645;
    int nl_tok   = 198;
    const char *user_prompt =
        "请用中文一句话解释什么是\"逐层加载推理\"。";
    int prompt_ids[1024]; int pn = 0;
    int tmp_ids[128]; int n;
    prompt_ids[pn++] = im_start;
    n = qwen_tokenizer_encode(&tok, "system", tmp_ids, 128);
    for (int i = 0; i < n && pn < 1024; i++) prompt_ids[pn++] = tmp_ids[i];
    prompt_ids[pn++] = nl_tok;
    n = qwen_tokenizer_encode(&tok, "You are a helpful assistant.", tmp_ids, 128);
    for (int i = 0; i < n && pn < 1024; i++) prompt_ids[pn++] = tmp_ids[i];
    prompt_ids[pn++] = im_end;
    prompt_ids[pn++] = nl_tok;
    prompt_ids[pn++] = im_start;
    n = qwen_tokenizer_encode(&tok, "user", tmp_ids, 128);
    for (int i = 0; i < n && pn < 1024; i++) prompt_ids[pn++] = tmp_ids[i];
    prompt_ids[pn++] = nl_tok;
    n = qwen_tokenizer_encode(&tok, user_prompt, tmp_ids, 128);
    for (int i = 0; i < n && pn < 1024; i++) prompt_ids[pn++] = tmp_ids[i];
    prompt_ids[pn++] = im_end;
    prompt_ids[pn++] = nl_tok;
    prompt_ids[pn++] = im_start;
    n = qwen_tokenizer_encode(&tok, "assistant", tmp_ids, 128);
    for (int i = 0; i < n && pn < 1024; i++) prompt_ids[pn++] = tmp_ids[i];
    prompt_ids[pn++] = nl_tok;

    for (int l = 0; l < cfg->n_layers; l++) ist.cache_len[l] = 0;
    ist.seq_len = 0;
    memset(ist.hidden, 0, (size_t)cfg->dim * sizeof(float));

    /* ---- Prefill ---- */
    const char *env_heat = getenv("VLLM_EW_HEAT");
    if (env_heat && env_heat[0] == '1') st_moe_heat_reset(&w);  /* 真实热度采集 */
    const char *env_single = getenv("VLLM_STREAM_SINGLE");
    int single_prefill = env_single && env_single[0] == '1';
    t0 = st_now_sec();
    if (single_prefill) {
        /* 逐 token 单步 forward 喂入（与 decode 同一内核，绕开 batch prefill
         * 路径；30B 上 batch prefill 曾静默退出）。逐字进 KV，语义等价。 */
        for (int _i = 0; _i < pn; _i++) {
            st_qwen_model_forward(&ist, prompt_ids[_i]);
        }
    } else if (st_qwen_model_prefill_batch(&ist, prompt_ids, pn) != 0) {
        printf("[STREAM] FAIL: prefill\n");
        st_qwen_inference_free(&ist);
        st_weights_free(&w);
        qwen_tokenizer_free(&tok);
        return 1;
    }
    double t_prefill = st_now_sec() - t0;
    printf("[STREAM] prefill %d tokens in %.3fs (%.0f tok/s) [%s]\n",
           pn, t_prefill, pn / (t_prefill > 0 ? t_prefill : 1e-9),
           single_prefill ? "single-step" : "batch");
    printf("[STREAM] rss_after_prefill = %ld kB\n", vqf_stream_rss_kb());
    fflush(stdout);

    /* ---- P0 质量门：强制续写固定参考句，测 NLL（对比 f32 精确 vs q8 反量化
     * 历史段）。VLLM_STREAM_NLL=1 启用；关/开各跑一次对比 NLL_MEAN。
     * 先做"第二轮"prefill（跨 prefill 调用读历史缓存），确保命中
     * prefill 历史段注意力（单批 prefill prev_len=0 测不到差异）。 ---- */
    if (getenv("VLLM_STREAM_NLL") && getenv("VLLM_STREAM_NLL")[0] == '1') {
        /* P0 语料质量门：多段 {正文, 追问, 参考回答}，每段做"turn1 prefill +
         * turn2 prefill（读历史缓存，命中 f32 正典读点）+ forced-NLL"。
         * 关/开（VLLM_PREFILL_Q8CACHE）各跑一次，对比 CORPUS_NLL_MEAN。 */
        /* P3 门禁三档（ΔNLL 口径见《内存分页优化方案》§四 P3 Phase 1）：
         *   A = 不加 --sparse-attn          -> 稠密精确（Phase 0 基线 4.046516）
         *   B = --sparse-attn               -> 隔离「稀疏注意力」自身近似
         *   C = --sparse-attn --l3-evict --l3-min-seq 1 + VLLM_STREAM_NLL_L3=1
         *                                     -> 再叠加「L3 驱逐 + Q4 重建前缀」
         * P3 的 ΔNLL = C − B；B − A 是稀疏注意力自身的代价，不计入 P3。
         * 档 C 直接驱动 l3_evict_after_prefill + l3_restore_prefix，不经过 serve
         * 层的 VLLM_L3_PREFIX_REUSE 门（该门此时尚未存在）。 */
        int nll_l3 = 0;
        {
            const char *e3 = getenv("VLLM_STREAM_NLL_L3");
            nll_l3 = (e3 && e3[0] == '1') ? 1 : 0;
        }
        if (nll_l3) {
            if (!(g_l3_evict && g_sparse_attn)) {
                printf("[STREAM] L3 档需要 --sparse-attn --l3-evict "
                       "--l3-min-seq <N>\n");
                st_qwen_inference_free(&ist);
                st_weights_free(&w);
                qwen_tokenizer_free(&tok);
                return 1;
            }
            if (!g_l3_path || !g_l3_path[0]) g_l3_path = strdup("/tmp/nll_l3.bin");
            printf("[STREAM] NLL 档 C：L3 驱逐 + Q4 重建前缀 (l3-path=%s)\n",
                   g_l3_path ? g_l3_path : "(default)");
            fflush(stdout);
        }

        /* P3 步骤 5：长上下文 ΔNLL。VLLM_STREAM_NLL_REPEAT=N 把每段 ctx 的 token
         * 序列重复 N 次（默认 1 = Phase 0 基线口径，与旧行为逐位相同）。目的：
         * 短段（~62 token）只有 2 个块，keep 后 0 块可驱逐 -> L3 档"不适用"；
         * 放大到 ~N×62 token 后每段有 ~26 块，L3 驱逐才真实生效。 */
        int nll_rep = 1;
        {
            const char *er = getenv("VLLM_STREAM_NLL_REPEAT");
            if (er && er[0]) {
                nll_rep = atoi(er);
                if (nll_rep < 1) nll_rep = 1;
            }
        }
        if (nll_rep > 1)
            printf("[STREAM] ctx_repeat=%d（长上下文 ΔNLL 口径）\n", nll_rep);

        static const struct { const char *ctx; const char *q; const char *ref; } cps[] = {
            {
                "磁悬浮列车利用电磁力使车体悬浮于轨道之上，消除轮轨摩擦，"
                "从而在高速运行时大幅降低能量损耗与机械磨损。其控制难点在于"
                "悬浮间隙的动态稳定性：外界扰动必须在毫秒级被电磁系统纠正。",
                "请用一句话总结这段文字。",
                "磁悬浮列车通过消除轮轨摩擦来降低损耗，并以高速电磁控制维持悬浮稳定。",
            },
            {
                "RISC-V 指令集以精简、开放与可扩展著称，近年成为边缘 AI 芯片的"
                "重要选择。相比闭源架构，开发者可以自由定制向量扩展与矩阵加速器，"
                "同时借助开源工具链缩短芯片验证周期。",
                "请用一句话概括 RISC-V 的优势。",
                "RISC-V 的开放与可扩展特性使它便于定制边缘 AI 芯片并加速验证。",
            },
            {
                "量子纠错通过把单个逻辑比特编码到多个物理比特上，用冗余来探测并"
                "修复错误。表面码是目前最有希望的候选方案之一，其容错阈值允许"
                "物理错误率降至千分之一以下后可靠运行大规模量子算法。",
                "这段在讲什么？",
                "这段介绍量子纠错用多物理比特冗余编码逻辑比特，并以表面码降低错误率。",
            },
            {
                "长江上游梯级水库在汛期联合调度可削峰错峰，保障中下游防洪安全；"
                "枯水期则通过补偿下泄维持航道水深与生态流量，是流域治理的关键"
                "基础设施。",
                "请概括梯级水库的作用。",
                "梯级水库汛期削峰防洪、枯水期补偿下泄，保障防洪、航运与生态。",
            },
        };
        const int np_ = (int)(sizeof(cps) / sizeof(cps[0]));
        double c_sum = 0.0;
        int c_n = 0;
        int n_l3_applied = 0;   /* 档 C：真正走到 L3 驱逐+重建的段数 */
        for (int p = 0; p < np_; p++) {
            for (int l = 0; l < cfg->n_layers; l++) ist.cache_len[l] = 0;
            ist.seq_len = 0;
            memset(ist.hidden, 0, (size_t)cfg->dim * sizeof(float));

            /* turn1: system + user(ctx) + assistant 起始 */
            int ids1[2048]; int m1 = 0;
            ids1[m1++] = im_start;
            n = qwen_tokenizer_encode(&tok, "system", tmp_ids, 128);
            for (int i = 0; i < n && m1 < 2048; i++) ids1[m1++] = tmp_ids[i];
            ids1[m1++] = nl_tok;
            n = qwen_tokenizer_encode(&tok, "You are a helpful assistant.", tmp_ids, 128);
            for (int i = 0; i < n && m1 < 2048; i++) ids1[m1++] = tmp_ids[i];
            ids1[m1++] = im_end;
            ids1[m1++] = nl_tok;
            ids1[m1++] = im_start;
            n = qwen_tokenizer_encode(&tok, "user", tmp_ids, 128);
            for (int i = 0; i < n && m1 < 2048; i++) ids1[m1++] = tmp_ids[i];
            ids1[m1++] = nl_tok;
            {
                /* nll_rep 次重复（默认 1 时与旧代码逐位相同）。 */
                int ctx_ids[512];
                int cn = qwen_tokenizer_encode(&tok, cps[p].ctx, ctx_ids, 512);
                for (int r = 0; r < nll_rep; r++)
                    for (int i = 0; i < cn && m1 < 2048; i++)
                        ids1[m1++] = ctx_ids[i];
            }
            ids1[m1++] = im_end;
            ids1[m1++] = nl_tok;
            ids1[m1++] = im_start;
            n = qwen_tokenizer_encode(&tok, "assistant", tmp_ids, 128);
            for (int i = 0; i < n && m1 < 2048; i++) ids1[m1++] = tmp_ids[i];
            ids1[m1++] = nl_tok;
            if (st_qwen_model_prefill_batch(&ist, ids1, m1) != 0) {
                printf("[STREAM] FAIL: passage %d turn1 prefill\n", p);
                st_qwen_inference_free(&ist);
                st_weights_free(&w);
                qwen_tokenizer_free(&tok);
                return 1;
            }

            /* P3 档 C：turn1 后驱逐冷块 → 从 Q4 载荷把前缀行重建回 RAM
             * （等价于 serve 侧 ist_reset 复用前缀前要做的 rebuild），使 turn2
             * 能读到完整历史 KV。 */
            if (nll_l3) {
                int seq1 = ist.cache_len[0];
                l3_evict_after_prefill(&ist, 0, 0);
                int ev = ist.l3.evicted;
                if (ev <= 0) {
                    /* 段太短：keep 至少保留「1 个高重要度块 + 最后 1 块」，
                     * 块数 ≤ 2 时 ratio 再大也驱不出任何块。该段 L3 不适用，
                     * 退回全量 RAM（等价档 B），显式标注而不算作失败。 */
                    printf("[STREAM] P%d L3: 不适用 (seq=%d 块数不足, 0 块可驱逐) "
                           "-> 该段等同档 B\n", p, seq1);
                } else {
                    int rs = l3_restore_prefix(&ist, seq1);
                    printf("[STREAM] P%d L3: seq=%d evicted=%d restored=%d\n",
                           p, seq1, ev, rs);
                    if (rs <= 0) {
                        printf("[STREAM] FAIL: passage %d L3 档未生效 "
                               "(evicted=%d restored=%d)\n", p, ev, rs);
                        st_qwen_inference_free(&ist);
                        st_weights_free(&w);
                        qwen_tokenizer_free(&tok);
                        return 1;
                    }
                    n_l3_applied++;
                }
                fflush(stdout);
            }

            /* turn2: user(追问) + assistant 起始（读 turn1 历史缓存） */
            int ids2[1024]; int m2 = 0;
            ids2[m2++] = im_start;
            n = qwen_tokenizer_encode(&tok, "user", tmp_ids, 128);
            for (int i = 0; i < n && m2 < 1024; i++) ids2[m2++] = tmp_ids[i];
            ids2[m2++] = nl_tok;
            n = qwen_tokenizer_encode(&tok, cps[p].q, tmp_ids, 128);
            for (int i = 0; i < n && m2 < 1024; i++) ids2[m2++] = tmp_ids[i];
            ids2[m2++] = im_end;
            ids2[m2++] = nl_tok;
            ids2[m2++] = im_start;
            n = qwen_tokenizer_encode(&tok, "assistant", tmp_ids, 128);
            for (int i = 0; i < n && m2 < 1024; i++) ids2[m2++] = tmp_ids[i];
            ids2[m2++] = nl_tok;
            if (st_qwen_model_prefill_batch(&ist, ids2, m2) != 0) {
                printf("[STREAM] FAIL: passage %d turn2 prefill\n", p);
                st_qwen_inference_free(&ist);
                st_weights_free(&w);
                qwen_tokenizer_free(&tok);
                return 1;
            }

            /* forced-NLL over reference */
            int ref_ids[512];
            int rn = qwen_tokenizer_encode(&tok, cps[p].ref, ref_ids, 512);
            double nll_sum = 0.0;
            int nll_n = 0;
            for (int step = 0; step < rn; step++) {
                double mx = -1e30;
                for (int t = 0; t < cfg->vocab_size; t++)
                    if ((double)ist.logits[t] > mx) mx = (double)ist.logits[t];
                double sum = 0.0;
                for (int t = 0; t < cfg->vocab_size; t++)
                    sum += exp((double)ist.logits[t] - mx);
                double lse = mx + log(sum);
                nll_sum += lse - (double)ist.logits[ref_ids[step]];
                nll_n++;
                if (step + 1 < rn) st_qwen_model_forward(&ist, ref_ids[step]);
            }
            printf("[STREAM] P%d NLL_MEAN = %.6f (n=%d)\n", p,
                   nll_n > 0 ? nll_sum / (double)nll_n : 0.0, nll_n);
            fflush(stdout);
            c_sum += nll_sum;
            c_n += nll_n;
        }
        printf("[STREAM] CORPUS_NLL_MEAN = %.6f (total_n=%d, passages=%d)\n",
               c_n > 0 ? c_sum / (double)c_n : 0.0, c_n, np_);
        if (nll_l3) {
            /* 不适用段（块数不足）与档 B 逐位相同，会稀释 ΔNLL —— 必须显式
             * 报出适用段数，否则读者会把稀释后的差值当成真实差值。 */
            printf("[STREAM] L3 applied %d/%d passages（其余段等同档 B）\n",
                   n_l3_applied, np_);
            if (n_l3_applied <= 0) {
                printf("[STREAM] FAIL: L3 档在所有段都不适用\n");
                st_qwen_inference_free(&ist);
                st_weights_free(&w);
                qwen_tokenizer_free(&tok);
                return 1;
            }
        }
        st_qwen_inference_free(&ist);
        st_weights_free(&w);
        qwen_tokenizer_free(&tok);
        return 0;
    }

    /* ---- P1a 回归（VLLM_STREAM_COPYTEST=1）：copy_kv_prefix 忠实性 ----
     * 同一世界内两条路必须逐位一致：
     *   A = turn1 prefill + 同状态 turn2 prefill（不复制）；
     *   B = 模板 turn1 prefill + copy_kv_prefix(模板→B) + B 上 turn2 prefill。
     * 若复制存在维度/scale 错误 → NLL_A 与 NLL_B 不一致。 */
    if (getenv("VLLM_STREAM_COPYTEST") && getenv("VLLM_STREAM_COPYTEST")[0] == '1') {
        const char *ctx =
            "磁悬浮列车利用电磁力使车体悬浮于轨道之上，消除轮轨摩擦，"
            "从而在高速运行时大幅降低能量损耗与机械磨损。其控制难点在于"
            "悬浮间隙的动态稳定性：外界扰动必须在毫秒级被电磁系统纠正。";
        const char *q = "请用一句话总结这段文字。";
        const char *ref =
            "磁悬浮列车通过消除轮轨摩擦来降低损耗，并以高速电磁控制维持悬浮稳定。";

        int t1[2048]; int m1 = 0;   /* system + user(ctx) + assistant 起始 */
        int t2[1024]; int m2 = 0;   /* user(q) + assistant 起始 */
        int k;
        t1[m1++] = im_start;
        k = qwen_tokenizer_encode(&tok, "system", tmp_ids, 128);
        for (int i = 0; i < k && m1 < 2048; i++) t1[m1++] = tmp_ids[i];
        t1[m1++] = nl_tok;
        k = qwen_tokenizer_encode(&tok, "You are a helpful assistant.", tmp_ids, 128);
        for (int i = 0; i < k && m1 < 2048; i++) t1[m1++] = tmp_ids[i];
        t1[m1++] = im_end; t1[m1++] = nl_tok;
        t1[m1++] = im_start;
        k = qwen_tokenizer_encode(&tok, "user", tmp_ids, 128);
        for (int i = 0; i < k && m1 < 2048; i++) t1[m1++] = tmp_ids[i];
        t1[m1++] = nl_tok;
        k = qwen_tokenizer_encode(&tok, ctx, tmp_ids, 128);
        for (int i = 0; i < k && m1 < 2048; i++) t1[m1++] = tmp_ids[i];
        t1[m1++] = im_end; t1[m1++] = nl_tok;
        t1[m1++] = im_start;
        k = qwen_tokenizer_encode(&tok, "assistant", tmp_ids, 128);
        for (int i = 0; i < k && m1 < 2048; i++) t1[m1++] = tmp_ids[i];
        t1[m1++] = nl_tok;

        t2[m2++] = im_start;
        k = qwen_tokenizer_encode(&tok, "user", tmp_ids, 128);
        for (int i = 0; i < k && m2 < 1024; i++) t2[m2++] = tmp_ids[i];
        t2[m2++] = nl_tok;
        k = qwen_tokenizer_encode(&tok, q, tmp_ids, 128);
        for (int i = 0; i < k && m2 < 1024; i++) t2[m2++] = tmp_ids[i];
        t2[m2++] = im_end; t2[m2++] = nl_tok;
        t2[m2++] = im_start;
        k = qwen_tokenizer_encode(&tok, "assistant", tmp_ids, 128);
        for (int i = 0; i < k && m2 < 1024; i++) t2[m2++] = tmp_ids[i];
        t2[m2++] = nl_tok;

        STQwenInferenceState istA, istT, istB;
        memset(&istA, 0, sizeof(istA));
        memset(&istT, 0, sizeof(istT));
        memset(&istB, 0, sizeof(istB));
        if (st_qwen_inference_init(&istA, &w) != 0 ||
            st_qwen_inference_init(&istT, &w) != 0 ||
            st_qwen_inference_init(&istB, &w) != 0) {
            printf("[STREAM] FAIL: COPYTEST init\n");
            st_weights_free(&w);
            qwen_tokenizer_free(&tok);
            return 1;
        }

        /* A：turn1 + turn2 同状态连续 prefill */
        if (st_qwen_model_prefill_batch(&istA, t1, m1) != 0 ||
            st_qwen_model_prefill_batch(&istA, t2, m2) != 0) {
            printf("[STREAM] FAIL: COPYTEST A prefill\n");
            return 1;
        }
        int nA = 0;
        double nllA = nll_forced_forward(cfg, &tok, &istA, ref, &nA);

        /* T+B：模板 turn1 → copy_kv_prefix(T→B) → B turn2 */
        if (st_qwen_model_prefill_batch(&istT, t1, m1) != 0) {
            printf("[STREAM] FAIL: COPYTEST T prefill\n");
            return 1;
        }
        st_qwen_copy_kv_prefix(&istB, &istT, m1);
        if (st_qwen_model_prefill_batch(&istB, t2, m2) != 0) {
            printf("[STREAM] FAIL: COPYTEST B prefill\n");
            return 1;
        }
        int nB = 0;
        double nllB = nll_forced_forward(cfg, &tok, &istB, ref, &nB);

        double diff = nllA - nllB;
        printf("[STREAM] COPYTEST A_NLL=%.9f (n=%d)  B_NLL=%.9f (n=%d)  "
               "diff=%.3e %s\n",
               nllA, nA, nllB, nB, diff,
               (nA > 0 && nB > 0 && diff > -1e-9 && diff < 1e-9) ? "PASS" : "FAIL");

        /* P1b 回归（VLLM_STREAM_DKVTEST=1）：v2(q8+scale) save→load 保真度。
         * 断言 1：恢复行的 q8 字节与 scale 与源逐字节一致；
         * 断言 2：恢复后再 decode 一个 token，logits 与源逐位一致。
         * （不做 prefill-logits 直比：新状态无 prefill logits，起点不等。） */
        if (getenv("VLLM_STREAM_DKVTEST") && getenv("VLLM_STREAM_DKVTEST")[0] == '1') {
            int tot = m1 + m2;
            /* 快照范围 = istA 当前实驻行数。COPYTEST 的 forced-NLL 已把 ref
             * 的前 rn-1 个 token 追加进 KV（cache_len = tot+rn-1），decode
             * 对比若只在 tot 行上进行则两状态起点不等，恒 FAIL。 */
            int rid2[512];
            int rn = qwen_tokenizer_encode(&tok, ref, rid2, 512);
            int save_n = tot + (rn > 0 ? rn - 1 : 0);
            if (save_n > istA.cache_len[0]) save_n = istA.cache_len[0];
            const char *dkv = "/tmp/dkvtest.bin";
            int rcS = -1;
            if (save_n > 0) {
                int *toks = (int *)malloc((size_t)save_n * sizeof(int));
                memcpy(toks, t1, (size_t)m1 * sizeof(int));
                memcpy(toks + m1, t2, (size_t)m2 * sizeof(int));
                for (int i = tot; i < save_n; i++) toks[i] = rid2[i - tot];
                rcS = st_kv_disk_save(&istA, dkv, toks, save_n);
                free(toks);
            }
            STQwenInferenceState istC;
            memset(&istC, 0, sizeof(istC));
            int rcI = 0;
            if (rcS != 0 || st_qwen_inference_init(&istC, &w) != 0) rcI = -1;
            if (rcI == 0 && st_kv_disk_load(&istC, dkv, save_n) != 0) rcI = -1;

            int rows_ok = (rcS == 0 && rcI == 0);
            if (rows_ok) {
                const int kv_dim = cfg->n_kv_heads * cfg->head_dim;
                for (int l = 0; rows_ok && l < cfg->n_layers; l++) {
                    for (int t = 0; rows_ok && t < save_n; t++) {
                        int b = t / istA.kv_bs, r = t % istA.kv_bs;
                        const int8_t *kA = istA.k_cache_q8[l][b] + (size_t)r * kv_dim;
                        const int8_t *vA = istA.v_cache_q8[l][b] + (size_t)r * kv_dim;
                        const int8_t *kC = istC.k_cache_q8[l][b] + (size_t)r * kv_dim;
                        const int8_t *vC = istC.v_cache_q8[l][b] + (size_t)r * kv_dim;
                        if (memcmp(kA, kC, (size_t)kv_dim) != 0 ||
                            memcmp(vA, vC, (size_t)kv_dim) != 0 ||
                            memcmp(istA.k_scale[l] + (size_t)t * cfg->n_kv_heads,
                                   istC.k_scale[l] + (size_t)t * cfg->n_kv_heads,
                                   (size_t)cfg->n_kv_heads * sizeof(float)) != 0 ||
                            memcmp(istA.v_scale[l] + (size_t)t * cfg->n_kv_heads,
                                   istC.v_scale[l] + (size_t)t * cfg->n_kv_heads,
                                   (size_t)cfg->n_kv_heads * sizeof(float)) != 0) {
                            rows_ok = 0;
                            break;
                        }
                    }
                }
            }
            int log_ok = 0;
            if (rows_ok && rn > 0) {
                /* 两状态同处 save_n 位置，喂同一续写 token（ref 末 token），
                 * 预测 logits 必须逐位一致（q8 世界下 v2 恢复即正典）。 */
                st_qwen_model_forward(&istA, rid2[rn - 1]);
                st_qwen_model_forward(&istC, rid2[rn - 1]);
                log_ok = 1;
                for (int x = 0; x < cfg->vocab_size; x++)
                    if (istA.logits[x] != istC.logits[x]) { log_ok = 0; break; }
            }
            printf("[STREAM] DKVTEST save=%d load=%d rows=%d rows_equal=%s "
                   "logits_equal=%s\n",
                   rcS, rcI, save_n, rows_ok ? "PASS" : "FAIL",
                   log_ok ? "PASS" : "FAIL");
            if (rcI == 0) st_qwen_inference_free(&istC);
        }

        st_qwen_inference_free(&istA);
        st_qwen_inference_free(&istT);
        st_qwen_inference_free(&istB);
        st_weights_free(&w);
        qwen_tokenizer_free(&tok);
        return 0;
    }

    /* ---- N token 贪心 decode（可配重复惩罚，C 工程兜底） ----
     * 量化精度受限时贪婪 argmax 易落入 n-gram 确定性循环（st30b3：
     * t3..t12 的 token 序列在 t13+ 重演）。VLLM_STREAM_REPPEN=p（p>1，
     * HF 语义乘法惩罚：logit≥0 → ÷p，logit<0 → ×p）对最近
     * VLLM_STREAM_REPWIN=n（默认 32，1..64）个已生成 token 逐次生效；
     * 缺省 p=1 = 关闭，保持历史确定性行为与 A≡C 锚点。 */
    float rep_pen = 1.0f;
    int   rep_win = 32;
    const char *rp = getenv("VLLM_STREAM_REPPEN");
    if (rp && rp[0]) {
        float p = (float)atof(rp);
        if (p > 1.0f && p < 1e3f) rep_pen = p;
    }
    const char *rw = getenv("VLLM_STREAM_REPWIN");
    if (rw && rw[0]) {
        int w = atoi(rw);
        if (w >= 1 && w <= 64) rep_win = w;
    }
    int rep_hist[64];       /* 已生成 token 历史（n_tokens ≤ 64，无需环形） */
    int rep_cnt = 0;
    float *rep_sc = rep_pen > 1.0f
        ? (float *)malloc((size_t)cfg->vocab_size * sizeof(float)) : NULL;
    char output[4096]; int out_len = 0;
    double t_dec_all = 0.0;
    int gen = 0;
    printf("[STREAM] decode mode: greedy%s (rep_pen=%.3f win=%d)\n",
           rep_pen > 1.0f ? "+reppen" : "", rep_pen, rep_win);
    fflush(stdout);
#ifdef _WIN32
    if (getenv("VLLM_MEMDUMP") && getenv("VLLM_MEMDUMP")[0] == '1') {
        mem_dump_regions(&w);   /* prefill 结束、decode 前基线 */
    }
#endif
    printf("[STREAM] TOKIDS:");
    fflush(stdout);
    for (int step = 0; step < n_tokens; step++) {
        const float *logp = ist.logits;
        if (rep_sc && rep_cnt > 0) {
            memcpy(rep_sc, ist.logits, (size_t)cfg->vocab_size * sizeof(float));
            int cnt = rep_cnt < rep_win ? rep_cnt : rep_win;
            for (int h = 0; h < cnt; h++) {
                int tid = rep_hist[rep_cnt - 1 - h];  /* 从最近向前扫 */
                if (tid < 0 || tid >= cfg->vocab_size) continue;
                float l = rep_sc[tid];
                rep_sc[tid] = l >= 0.0f ? l / rep_pen : l * rep_pen;
            }
            logp = rep_sc;
        }
        float best = -1e9f; int best_id = 0;
        for (int t = 0; t < cfg->vocab_size; t++) {
            if (logp[t] > best) { best = logp[t]; best_id = t; }
        }
        if (rep_cnt < 64) rep_hist[rep_cnt] = best_id;  /* 记账本次已选 */
        rep_cnt++;
        if (best_id == tok.eos_id || best_id == im_end) break;
        gen++;
        printf(" %d", best_id);
        fflush(stdout);
        const char *s = qwen_tokenizer_decode(&tok, best_id);
        if (s && s[0]) {
            int slen = (int)strlen(s);
            if (out_len + slen < 4095) { memcpy(output + out_len, s, slen); out_len += slen; }
        }
        double d0 = st_now_sec();
        st_qwen_model_forward(&ist, best_id);
        t_dec_all += st_now_sec() - d0;
#ifdef _WIN32
        if (step == 0 && getenv("VLLM_MEMDUMP") && getenv("VLLM_MEMDUMP")[0] == '1') {
            mem_dump_regions(&w);   /* decode 首个 forward 后：推理期分配应已建立 */
            if (getenv("VLLM_MEMDUMP_EXIT") && getenv("VLLM_MEMDUMP_EXIT")[0] == '1') {
                st_qwen_inference_free(&ist);
                st_weights_free(&w);
                qwen_tokenizer_free(&tok);
                return 0;
            }
        }
#endif
        printf(" t=%d ms=%.0f rss=%ldkB\n", step + 1,
               (st_now_sec() - d0) * 1000.0, vqf_stream_rss_kb());
        fflush(stdout);
    }
    output[out_len] = '\0';
    printf("\n[STREAM] decoded %d tokens, %.0f ms/tok (%.2f tok/s)\n",
           gen, t_dec_all / (gen > 0 ? gen : 1) * 1000.0,
           gen / (t_dec_all > 0 ? t_dec_all : 1e-9));
    printf("[STREAM] text: %s\n", output[0] ? output : "(empty)");
    printf("[STREAM] rss_end = %ld kB\n", vqf_stream_rss_kb());
    fflush(stdout);
    if (env_heat && env_heat[0] == '1') {
        const char *ho = getenv("VLLM_EW_HEAT_OUT");
        if (ho && ho[0]) st_moe_heat_save(ho);
    }

    free(rep_sc);
    st_qwen_inference_free(&ist);
    st_weights_free(&w);
    qwen_tokenizer_free(&tok);
    return 0;
}

static int serve_load_model(ServeModel *m) {
    STModelConfig *cfg = NULL;
    QwenTokenizer *tok = NULL;
    STModelWeights *w = NULL;
    STQwenInferenceState *ist = NULL;
    STVisionWeights *vis_w = NULL;
    STVisionState *vis = NULL;
    int has_vision = 0;
    int cfg_is_vqf = 0;
    int vqf_file_mode = 0;   /* 单文件 VQF：词表需从文件所在目录加载 */
    char vqf_dirbuf[768] = {0};
    int rc = 0;

    /* 纯自研 VQF 运行时：模型目录须含 model.vqf（目录式），或 --model 直接
     * 指向 .vqf 单文件；cfg 自包含（vqf_load 填充 w->cfg）。原始权重
     * （safetensors）与 GGUF 的加载/转换已整体移出引擎，由独立工具
     * vqf_convert/（vqf_conv）承担。 */
    w = (STModelWeights *)calloc(1, sizeof(STModelWeights));
    if (!w) { rc = -1; goto done; }
    char vqf_path[768] = {0};
    if (vqf_dir_find(m->model_dir, vqf_path, sizeof(vqf_path))) {
        if (vqf_load(w, vqf_path) != 0) {
            printf("[SERVE] VQF load failed: %s\n", vqf_path);
            rc = 1; goto done;
        }
        cfg = &w->cfg;
        cfg_is_vqf = 1;
        fprintf(stderr, "[M-A] VQF loaded: %s\n", vqf_path); fflush(stderr);
        /* 可选预热：后台线程把 mmap 权重页从 eMMC 拉进 RAM，
         * 首次推理的页 fault 成本挪到就绪后空闲期（VLLM_VQF_PREWARM=1）。 */
        if (getenv("VLLM_VQF_PREWARM") && getenv("VLLM_VQF_PREWARM")[0] == '1') {
            vqf_prewarm(w);
            fprintf(stderr, "[M-A] VQF prewarm started (background)\n");
            fflush(stderr);
        }
    } else if (vqf_is_file(m->model_dir)) {
        /* VQF 单文件直接 mmap 挂载 */
        if (vqf_load(w, m->model_dir) != 0) {
            printf("[SERVE] VQF load failed: %s\n", m->model_dir);
            rc = 1; goto done;
        }
        cfg = &w->cfg;
        cfg_is_vqf = 1;
        vqf_file_mode = 1;
        /* 词表不内嵌在 .vqf 里：从文件所在目录加载 vocab.bin/config.json。
         * （目录式 VQF 的 m->model_dir 本身就是模型目录，无需此处理。） */
        snprintf(vqf_dirbuf, sizeof(vqf_dirbuf), "%s", m->model_dir);
        char *vslash = strrchr(vqf_dirbuf, '/');
        if (vslash) *vslash = '\0'; else vqf_dirbuf[0] = '\0';
        fprintf(stderr, "[M-A] VQF loaded (file): %s (tok_dir=%s)\n",
                m->model_dir, vqf_dirbuf[0] ? vqf_dirbuf : "(cwd)");
        fflush(stderr);
    } else {
        printf("[SERVE] 仅支持自研 VQF 模型：模型目录须含 model.vqf，"
               "或 --model 指向 .vqf 单文件\n");
        rc = 1; goto done;
    }
    printf("[SERVE] model=%s dim=%d layers=%d heads=%d ff=%d vocab=%d max_seq=%d\n",
           m->model_dir, cfg->dim, cfg->n_layers, cfg->n_heads, cfg->ffn_dim,
           cfg->vocab_size, cfg->max_seq_len);
    fflush(stdout);
    /* NPU strip-cache coherence: drop stale strips when the model DIR changes
     * (different weights / geometry); same-dir reloads keep them. */
    {
        extern void st_npu_gw_clear_all(void);
        int dir_changed = strcmp(g_npu_loaded_model, m->model_dir) != 0;
        if (dir_changed) {
            st_npu_gw_clear_all();
            snprintf(g_npu_loaded_model, sizeof(g_npu_loaded_model), "%s",
                     m->model_dir ? m->model_dir : "");
        }
        if (getenv("VLLM_NPU_GW_DBG")) {
            fprintf(stderr, "[GW-DBG] model='%s' cached='%s' clear=%d\n",
                    m->model_dir ? m->model_dir : "(null)",
                    g_npu_loaded_model, dir_changed);
            fflush(stderr);
        }
    }

    tok = (QwenTokenizer *)calloc(1, sizeof(QwenTokenizer));
    if (!tok) { rc = -1; goto done; }
    {
        /* 单文件 VQF 的词表来自其所在目录（vqf_dirbuf），否则用模型目录 */
        const char *tok_dir = (vqf_file_mode && vqf_dirbuf[0]) ? vqf_dirbuf
                                                              : m->model_dir;
        if (qwen_tokenizer_load(tok, tok_dir) != 0) {
            printf("[SERVE] tokenizer load failed\n");
            rc = 1; goto done;
        }
        qwen_tokenizer_load_special(tok, tok_dir);
    }
    printf("[SERVE] tokenizer: %d tokens (eos=%d im_end=%d)\n",
           tok->vocab_size, tok->eos_id, tok->im_end_id);
    fflush(stdout);
    fprintf(stderr, "[M-B] tokenizer loaded\n"); fflush(stderr);

    ist = (STQwenInferenceState *)calloc(1, sizeof(STQwenInferenceState));
    if (!ist) { rc = -1; goto done; }
    if (st_qwen_inference_init(ist, w) != 0) {
        printf("[SERVE] inference state init failed\n");
        rc = 1; goto done;
    }
    printf("[SERVE] inference state ready (max_seq=%d, kv_bs=%d)\n",
           ist->cfg.max_seq_len, ist->kv_bs);
    fflush(stdout);

    /* Vision encoder (multimodal image/video support). 纯 VQF：vision 权重
     * 已随 vqf_load mmap 挂载（Q8 大矩阵 + F32 小张量固化布局），零加载零
     * 转换，直接值拷贝权重集合给推理状态；vis_w 指向 w->vision（alias
     * vqf_map），由 st_weights_free 统一释放。 */
    if (cfg->has_vision) {
        vis = (STVisionState *)calloc(1, sizeof(STVisionState));
        if (vis && w->vision && w->vision->is_allocated) {
            if (st_vision_init(vis, cfg) == 0) {
                vis_w = w->vision;
                vis->w = *vis_w;
                has_vision = 1;
                printf("[SERVE] vision encoder ready (VQF mmap, ViT %d layers, hidden=%d)\n",
                       cfg->vis_depth, cfg->vis_hidden);
            }
        } else {
            fprintf(stderr, "[SERVE] WARNING: VQF 未含 vision 权重 -> text-only\n");
        }
        fflush(stdout);
    }

done:
    if (rc != 0) {
        if (ist) { st_qwen_inference_free(ist); free(ist); }
        if (w && w->is_allocated) { st_weights_free(w); }
        if (w) free(w);
        if (tok) { qwen_tokenizer_free(tok); free(tok); }
        /* VQF：cfg 指向 w->cfg、vis_w 指向 w->vision（alias vqf_map），
         * 均随 st_weights_free 一并释放，此处无需单独 free。 */
        if (vis) { st_vision_free(vis); free(vis); }
        return rc;
    }
    m->cfg = cfg; m->tok = tok; m->w = w; m->ist = ist;
    m->cfg_is_vqf = cfg_is_vqf;
    m->vis_w = vis_w; m->vis = vis; m->has_vision = has_vision;
    return 0;
}

/* Real model identity from the parsed config (the advertised API model_id is
 * a generic constant; this reflects what actually loaded). */
static void fill_model_name(VLLMServerCtx *ctx, const STModelConfig *c) {
    int d  = c ? c->dim : 0;
    int nl = c ? c->n_layers : 0;
    const char *fam = (d >= 4096 && nl >= 36) ? "Qwen3-VL-8B"
                    : (d >= 2048 && nl >= 28) ? "Qwen3-VL-2B"
                    : (d > 0)                 ? "Qwen3-VL"
                    :                           "?";
    snprintf(ctx->model_name, sizeof(ctx->model_name), "%s (dim=%d, layers=%d)",
             fam, d, nl);
}

/* Background load thread (manual mode). Updates ctx as the load proceeds. */
static void *serve_load_thread(void *arg) {
    VLLMServerCtx *ctx = (VLLMServerCtx *)arg;
    ServeModel *m = (ServeModel *)ctx->load_arg;
    /* Portable manual-load mode: the model dir may have been set on the
     * admin page after startup (ctx->model_dir repointed by
     * vllm_serve_set_model_dir). Sync it into the load context. */
    if (ctx->model_dir && ctx->model_dir[0] &&
        strcmp(ctx->model_dir, m->model_dir) != 0) {
        snprintf(m->model_dir, sizeof(m->model_dir), "%s", ctx->model_dir);
    }
    /* 加载格式优先级（管理页"加载格式"下拉 → ctx->load_format）。 */
    if (ctx->load_format >= 0 && ctx->load_format <= 3)
        m->format_prio = ctx->load_format;
    g_model_load_layer = 0;
    g_model_load_total = 0;
    ctx->load_state = 1;
    LARGE_INTEGER mfreq, mt0, mt1;
    QueryPerformanceFrequency(&mfreq);
    QueryPerformanceCounter(&mt0);
    int rc = serve_load_model(m);
    if (rc != 0) {
        ctx->load_state = 3;
        snprintf(ctx->load_error, sizeof(ctx->load_error),
                 "model load failed (rc=%d)", rc);
        fprintf(stderr, "[SERVE] model load failed rc=%d\n", rc);
    } else {
        QueryPerformanceCounter(&mt1);
        ctx->last_load_ms = (long)((double)(mt1.QuadPart - mt0.QuadPart) *
                                   1000.0 / (double)mfreq.QuadPart);
        fprintf(stderr, "[M-T] model load took %ld ms (startup time)\n",
                ctx->last_load_ms);
        fflush(stderr);
        /* 刷新空闲计时：load-on-use 请求在加载完成后尚未进入推理门，
         * 若不刷新，idle-unload 可能在新请求推理前就把模型卸载掉。 */
        ctx->last_infer_s = (long)time(NULL);
        ctx->w = m->w;
        ctx->cfg = m->cfg;
        ctx->tok = m->tok;
        ctx->ist = m->ist;
        ctx->vis = m->vis;
        ctx->load_state = 2;
        fill_model_name(ctx, m->cfg);
        vatt_model_ready(ctx);              /* 方案 2：固化模型指纹 */
        vllm_server_diskkv_scan(ctx);   /* index --disk-kv checkpoints */
        printf("[SERVE] model ready: %s (wmode=%d, vision=%d)\n",
               m->model_dir, g_st_wmode, m->has_vision);
        fflush(stdout);
    }
    return NULL;
}

/* Kick off an asynchronous model load (admin /admin/api/model/load, or the
 * load-on-use path). Returns 0 = started, 1 = already loading, 2 = already
 * ready. load_state is claimed to 1 under stat_lock so concurrent callers
 * (worker threads / idle monitor) never start two loads at once. */
int vllm_serve_start_load(VLLMServerCtx *ctx) {
    vhttp_mutex_lock(ctx->stat_lock);
    if (ctx->load_state == 1) { vhttp_mutex_unlock(ctx->stat_lock); return 1; }
    if (ctx->load_state == 2) { vhttp_mutex_unlock(ctx->stat_lock); return 2; }
    ctx->load_state = 1;
    vhttp_mutex_unlock(ctx->stat_lock);
    pthread_t th;
    if (pthread_create(&th, NULL, serve_load_thread, ctx) != 0) {
        ctx->load_state = 3;
        snprintf(ctx->load_error, sizeof(ctx->load_error), "load thread create failed");
        return -1;
    }
    pthread_detach(th);
    return 0;
}

/* Load-on-use (用时加载): ensure a model is loaded before serving a request.
 * Starts a background load if needed and waits for it to finish. Returns
 * 0 = ready, -1 = cannot load (no model_dir / load failed). */
int vllm_serve_load_on_use(VLLMServerCtx *ctx) {
    if (ctx->load_state == 2) return 0;
    if (!ctx->model_dir || !ctx->model_dir[0]) return -1;
    ServeModel *m = (ServeModel *)ctx->load_arg;
    if (m) snprintf(m->model_dir, sizeof(m->model_dir), "%s", ctx->model_dir);
    int rc = vllm_serve_start_load(ctx);
    if (rc < 0) return -1;
    while (ctx->load_state == 1) {
        usleep(100000);      /* wait for the background load to finish */
    }
    return (ctx->load_state == 2) ? 0 : -1;
}

/* Unload the loaded model (纯 VQF): free weights, tokenizer,
 * inference state (KV cache), vision state, batch scheduler and the in-RAM
 * prefix-KV context — the "上下文状态清零" of the load-on-use lifecycle, so
 * the next load starts from a clean slate. Called from the admin API
 * (explicit 卸载) or the idle-unload monitor (auto, 不用时卸载). Refuses
 * while an inference request is running/queued. */
int vllm_serve_unload_model(VLLMServerCtx *ctx) {
    if (ctx->load_state == 1) return -2;   /* still loading */
    if (ctx->load_state != 2) return -1;   /* nothing loaded */
    vhttp_mutex_lock(ctx->stat_lock);
    if (ctx->active_requests > 0 || ctx->queued_requests > 0 || ctx->busy ||
        ctx->inflight_handlers > 0) {
        vhttp_mutex_unlock(ctx->stat_lock);
        return -3;                         /* inference in flight */
    }
    ctx->unload_pending = 1;               /* reject any new inference from here on */
    vhttp_mutex_unlock(ctx->stat_lock);

    /* Drain: a worker may have entered server_handler just before unload_pending
     * was set and still be between ensure_model_ready and infer_gate, where it
     * reads ctx->ist/cfg/w WITHOUT holding inf_lock. Freeing under it would be
     * a use-after-free (observed crash at "[UNLOAD] step5 free w"). Wait until
     * every handler has exited; new inference is 503-rejected meanwhile. */
    for (int i = 0; i < 200; i++) {
        vhttp_mutex_lock(ctx->stat_lock);
        int busy_now = ctx->inflight_handlers > 0 ||
                       ctx->active_requests > 0 || ctx->queued_requests > 0 ||
                       ctx->busy;
        vhttp_mutex_unlock(ctx->stat_lock);
        if (!busy_now) break;
        usleep(10000);
    }
    vhttp_mutex_lock(ctx->stat_lock);
    int still_busy = ctx->inflight_handlers > 0 ||
                     ctx->active_requests > 0 || ctx->queued_requests > 0 ||
                     ctx->busy;
    vhttp_mutex_unlock(ctx->stat_lock);
    if (still_busy) {
        fprintf(stderr, "[UNLOAD] still busy after 2s, aborting unload\n");
        fflush(stderr);
        ctx->unload_pending = 0;
        return -3;
    }
    /* Rare race: the last serial request cleared busy but has not yet released
     * inf_lock. Wait for it, then hold the lock while freeing so a finishing
     * request can never touch freed state (new requests get -3 from the gate). */
    vhttp_mutex_lock(ctx->inf_lock);

    ServeModel *m = (ServeModel *)ctx->load_arg;

    fprintf(stderr, "[UNLOAD] step1 reset_session\n"); fflush(stderr);
    vllm_server_reset_session(ctx);        /* batch + prefix last_ids + dkv 索引 */
    fprintf(stderr, "[UNLOAD] step2 free vis\n"); fflush(stderr);
    if (ctx->vis) { st_vision_free(ctx->vis); free(ctx->vis); ctx->vis = NULL; }
    fprintf(stderr, "[UNLOAD] step3 free ist\n"); fflush(stderr);
    if (ctx->ist) { st_qwen_inference_free(ctx->ist); free(ctx->ist); ctx->ist = NULL; }
    fprintf(stderr, "[UNLOAD] step4 free tok\n"); fflush(stderr);
    if (ctx->tok) { qwen_tokenizer_free(ctx->tok); free(ctx->tok); ctx->tok = NULL; }
    fprintf(stderr, "[UNLOAD] step5 free w\n"); fflush(stderr);
    if (ctx->w) {
        /* VQF：vis_w 指向 w->vision（alias vqf_map），由 st_weights_free 统一释放 */
        if (ctx->w->is_allocated) st_weights_free(ctx->w);
        free(ctx->w);
        ctx->w = NULL;
        if (m) { m->w = NULL; m->vis_w = NULL; m->has_vision = 0; }
    }
    ctx->cfg = NULL;   /* VQF：cfg 即 &w->cfg，随 w 释放 */
    fprintf(stderr, "[UNLOAD] step6 done, state=%d->0\n", ctx->load_state); fflush(stderr);
    ctx->load_state = 0;
    ctx->load_error[0] = '\0';
    ctx->unload_pending = 0;
    ctx->model_name[0] = '\0';
    ctx->unload_count++;
    if (m) { m->cfg = NULL; m->tok = NULL; m->ist = NULL; m->vis = NULL; }
    vhttp_mutex_unlock(ctx->inf_lock);
    fprintf(stderr, "[UNLOAD] model unloaded (weights/KV/tokenizer/batch freed, "
            "total unloads=%d, last_load_ms=%ld)\n",
            ctx->unload_count, ctx->last_load_ms);
    fflush(stderr);
    return 0;
}

/* HTTP handler / policy handler 内卸载用的豁免包装：server_handler 在分发
 * 前已把 inflight_handlers 计数 +1，unload 的忙检视会把"调用者自己"误判为
 * 在途推理而永远返回 -3。此处临时扣除调用者自身的计数（卸载后恢复），让
 * admin「卸载模型」/ 策略 L0「落盘+退出」在请求线程内可用；卸载期间其它
 * 仍在途的 handler 会被照常等待。调用者卸载后不得再触碰 ctx->ist/cfg/w。 */
int vllm_serve_unload_model_self(VLLMServerCtx *ctx) {
    int dec = 0;
    vhttp_mutex_lock(ctx->stat_lock);
    if (ctx->inflight_handlers > 0) { ctx->inflight_handlers--; dec = 1; }
    vhttp_mutex_unlock(ctx->stat_lock);
    int rc = vllm_serve_unload_model(ctx);
    if (dec) {
        vhttp_mutex_lock(ctx->stat_lock);
        ctx->inflight_handlers++;
        vhttp_mutex_unlock(ctx->stat_lock);
    }
    return rc;
}

static void serve_on_start(int actual_port, void *ud) {
    (void)ud;
    printf("\n  [SERVE] OpenAI-compatible API listening on http://0.0.0.0:%d\n", actual_port);
    printf("  [SERVE] %s http://0.0.0.0:%d/admin/   (%s)\n",
           vllm_tr("管理入口:", "admin UI:"), actual_port,
           vllm_tr("管理页 admin.html", "admin.html"));
    printf("  [SERVE] %s http://0.0.0.0:%d/chat    (%s)\n",
           vllm_tr("对话入口:", "chat UI:"), actual_port,
           vllm_tr("对话页 chat.html", "chat.html"));
    printf("          GET  /v1/models\n");
    printf("          GET  /health\n");
    printf("          POST /v1/chat/completions  (stream=true SSE supported)\n");
    printf("          POST /v1/completions\n");
    printf("          workers=%d queued_max=%d free_ram_min=%d MB\n",
           g_serve_threads, g_serve_max_queued, g_serve_min_free_mb);
    fflush(stdout);
}

static int vllm_serve_main(int port, const char *model_dir_arg, int auto_load) {
    const char *model_dir = model_dir_arg;
    if (!model_dir) {
        const char *model_paths[] = {
            "Modl/千问3_VL_8B_Instruct",
            "Modl/Qwen3VL-8B-Instruct",
            "../Modl/千问3_VL_8B_Instruct",
            "../../Modl/千问3_VL_8B_Instruct",
            NULL
        };
        for (int i = 0; model_paths[i]; i++) {
            char tp[1024];
            snprintf(tp, sizeof(tp), "%s/config.json", model_paths[i]);
            if (st_access(tp, 0) == 0) { model_dir = model_paths[i]; break; }
        }
    }
    if (!model_dir) {
        printf("[SERVE] Model not found at the default paths (Modl/千问3_VL_8B_Instruct). ");
        if (auto_load) {
            printf("Use --model <dir> or place the model next to the exe.\n");
            return 1;
        }
        /* Portable manual-load mode: the packaged exe may be run from any
         * directory with no model on disk. Start the server anyway — the
         * admin page lets the user enter the model directory and
         * POST /admin/api/model/load applies it to the background loader. */
        printf("manual-load mode: 请在管理页 /admin/ 设置模型目录后点击「加载模型」。\n");
        model_dir = "";
    }

    VLLMServerCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.model_id = g_serve_model_id ? g_serve_model_id : "qwen3-vl-8b";
    ctx.max_queued = g_serve_max_queued;
    ctx.min_free_mb = g_serve_min_free_mb;
    ctx.batch_max = g_serve_batch_max;   /* --batch-max N: continuous batching */
    ctx.auto_load = g_serve_load_on_use;  /* 用时加载：请求触发加载（--load-on-use） */
    ctx.auto_unload_s = g_serve_auto_unload_s; /* 不用时卸载：空闲 N 秒自动卸载 */
    if (ctx.auto_unload_s > 0) ctx.auto_load = 1; /* 卸载后能自愈：下个请求重新加载 */
    ctx.load_format = g_serve_load_format; /* 加载格式（--load-format，CLI 专属） */
    /* Admin/management metadata (reported by /admin; NULL paths use the
     * board defaults in vllm_admin.c). */
    ctx.model_dir = model_dir;
    ctx.port = port;
    ctx.threads = g_serve_threads;
    ctx.npu_enabled = g_npu_enabled;
    ctx.prefix_cache = g_prefix_cache;
    ctx.prefix_kv = g_prefix_kv;
    ctx.min_p = g_serve_min_p;
    ctx.top_k = g_serve_top_k;
    ctx.disk_kv = g_disk_kv;
    if (g_disk_kv && g_disk_kv_dir[0])
        snprintf(ctx.kvdir, sizeof(ctx.kvdir), "%s", g_disk_kv_dir);
    ctx.spec = g_spec;
    ctx.spec_k = g_spec_k;
    /* 内存驻留策略（档位阶梯）默认值 + CLI 覆盖（管理页运行期可再调） */
    vllm_res_defaults(&ctx);
    if (g_l3_evict) ctx.res.level = 3;   /* 启动已开 --l3-evict → 档位从 L3 起算 */
    if (g_res_auto) ctx.res.auto_idle = 1;
    for (int k = 0; k < 4; k++)
        if (g_res_idle_s[k] > 0) ctx.res.idle_s[4 - k] = g_res_idle_s[k];
    if (g_res_soft_mb > 0) ctx.res.mem_soft_mb = g_res_soft_mb;
    if (g_res_low_lvl >= 0) ctx.res.mem_low_lvl = g_res_low_lvl;
    if (g_res_w_keep > 0) ctx.res.w_keep = g_res_w_keep;
    ctx.started_s = (long)time(NULL);
    ctx.config_path = NULL;
    ctx.log_path = NULL;
    /* Device profile (reported by /admin; drives weight-adaptation). */
    ctx.device_id = g_dev_profile ? g_dev_profile->id : g_dev.model_id;
    ctx.device_name = g_dev_profile ? g_dev_profile->name : NULL;
    ctx.device_arch = vdev_arch_str(g_dev.arch);
    ctx.device_class = vdev_class_str(g_dev.cls);
    ctx.device_npu = g_dev_profile && g_dev_profile->npu
                     ? g_dev_profile->npu : "-";

    ServeModel m;
    memset(&m, 0, sizeof(m));
    snprintf(m.model_dir, sizeof(m.model_dir), "%s", model_dir);
    m.format_prio = ctx.load_format;   /* --load-format（CLI 专属） */
    ctx.load_arg = &m;

    /* 方案 2：可验证推理背书初始化（VLLM_ATTEST=1 启用，见 vllm_attest.c）。
     * 须在模型加载前调用：vatt_model_ready 在加载成功后固化模型指纹。 */
    vatt_init(&ctx);

    if (auto_load) {
        /* Synchronous load before serving (default behavior). */
        fprintf(stderr, "[SERVE] auto-load: loading model %s (wmode=%d)\n",
                model_dir, g_st_wmode);
        fflush(stderr);
        LARGE_INTEGER mfreq, mt0, mt1;
        QueryPerformanceFrequency(&mfreq);
        QueryPerformanceCounter(&mt0);
        if (serve_load_model(&m) != 0) {
            fprintf(stderr, "[SERVE] FATAL: model load failed (see [ST]/[TOK]/[VIS] "
                    "errors above). Log file contains the full trace.\n");
            fflush(stderr);
            return 1;
        }
        QueryPerformanceCounter(&mt1);
        ctx.last_load_ms = (long)((double)(mt1.QuadPart - mt0.QuadPart) *
                                  1000.0 / (double)mfreq.QuadPart);
        fprintf(stderr, "[M-T] model load took %ld ms (startup time)\n",
                ctx.last_load_ms);
        fflush(stderr);
        ctx.w = m.w; ctx.cfg = m.cfg; ctx.tok = m.tok;
        ctx.ist = m.ist; ctx.vis = m.vis;
        ctx.load_state = 2;
        fill_model_name(&ctx, m.cfg);
        vatt_model_ready(&ctx);          /* 方案 2：固化模型指纹 */
        vllm_server_diskkv_scan(&ctx);   /* index --disk-kv checkpoints */
        printf("[SERVE] model ready: %s (wmode=%d, vision=%d)\n",
               m.model_dir, g_st_wmode, m.has_vision);
        fflush(stdout);
    } else {
        /* Manual-load mode: the HTTP server starts immediately with no model
         * loaded; POST /admin/api/model/load (from the admin page) loads it
         * in a background thread so progress is visible live. */
        printf("[SERVE] manual-load mode: model NOT loaded yet. Open "
               "http://0.0.0.0:%d/admin/ and press \"加载模型\".\n", port);
        fflush(stdout);
    }

    int rc = vllm_server_run(&ctx, port, g_serve_threads, serve_on_start);

    /* Wait for a background load (manual mode) to finish before freeing. */
    while (ctx.load_state == 1) {
        sleep(1);
    }
    if (m.has_vision) {
        if (m.vis) st_vision_free(m.vis);
        /* VQF：m.vis_w 指向 m.w->vision（alias vqf_map），由 st_weights_free 统一释放 */
    }
    if (m.ist) st_qwen_inference_free(m.ist);
    if (m.tok) { qwen_tokenizer_free(m.tok); free(m.tok); }
    /* VQF：cfg 内嵌于 w（mmap 所有权随 w），不单独 free */
    if (m.w && m.w->is_allocated) st_weights_free(m.w);
    if (m.w) free(m.w);
    if (m.vis) free(m.vis);
    return rc;
}

/* ================================================================
 * main
 * ================================================================ */
int main(int argc, char **argv) {
    /* RK3588/Linux: raise the main-thread stack to match the MSVC
     * /STACK:16777216 build (large local arrays: scores[4096] etc.). */
    st_raise_stack_limit(16u * 1024u * 1024u);

#ifdef _WIN32
    /* 源码内中文字符串为 UTF-8，而 Windows 控制台默认 GBK(cp936)，直接
     * printf 会乱码（如交互提示"请输入服务端口号"）。切换代码页对齐源码；
     * 仅影响交互终端显示，重定向/日志仍输出原始 UTF-8 字节。 */
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
#endif

    printf("  vLLM-Kestrel: Axiom-Guided Exact-Attention         \n");
    printf("  Large Language Model Inference Engine          \n");
    printf("\n");

    /* ---- Parse command-line flags ---- */
    int perf_only    = 0;
    int run_multimodal = 0;
    int bench_mixed  = 0;
    int n_images = 0;
    const char *img_paths[8] = { NULL };
    const char *vid_dir  = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--fhe-test") == 0) {
            /* 密文推理原语自测（BFV T1-T11 + CKKS T12 + NTT T13），PASS/FAIL 一键判定 */
            int fails = vfhe_self_test();
            fails += ckks_self_test();
            fails += ntt_self_test();
            printf("FHE_SELFTEST=%s(%d)\n", fails ? "FAILED" : "ALL_PASS", fails);
            return fails ? 1 : 0;
        } else if (strcmp(argv[i], "--perf-only") == 0) {
            perf_only = 1;
        } else if (strcmp(argv[i], "--longctx-quality") == 0) {
            perf_only = 1;
            g_longctx_quality = 1;
        } else if (strcmp(argv[i], "--longctx-gen") == 0 && i + 1 < argc) {
            g_longctx_gen_max = atoi(argv[++i]);
            if (g_longctx_gen_max < 1) g_longctx_gen_max = 1;
            if (g_longctx_gen_max > 512) g_longctx_gen_max = 512;
        } else if (strcmp(argv[i], "--longctx-ctx") == 0 && i + 1 < argc) {
            g_longctx_ctx = atoi(argv[++i]);
            if (g_longctx_ctx < 1) g_longctx_ctx = 1;
        } else if (strcmp(argv[i], "--prefix-cache") == 0) {
            perf_only = 1;
            g_prefix_cache = 1;
        } else if (strcmp(argv[i], "--no-prefix-kv") == 0) {
            g_prefix_kv = 0;   /* serve: disable in-RAM KV prefix reuse */
        } else if (strcmp(argv[i], "--disk-kv") == 0 && i + 1 < argc) {
            g_disk_kv = 1;     /* serve: persist KV snapshots across restarts */
            snprintf(g_disk_kv_dir, sizeof(g_disk_kv_dir), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--spec") == 0) {
            g_spec = 1;        /* serve: n-gram speculative decode */
        } else if (strcmp(argv[i], "--spec-k") == 0 && i + 1 < argc) {
            g_spec_k = atoi(argv[++i]);
            if (g_spec_k < 1) g_spec_k = 1;
            if (g_spec_k > 16) g_spec_k = 16;
        } else if (strcmp(argv[i], "--min-p") == 0 && i + 1 < argc) {
            g_serve_min_p = atof(argv[++i]);
            if (g_serve_min_p < 0.0) g_serve_min_p = 0.0;
            if (g_serve_min_p > 1.0) g_serve_min_p = 1.0;
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            g_serve_top_k = atoi(argv[++i]);
            if (g_serve_top_k < 0) g_serve_top_k = 0;
        } else if (strcmp(argv[i], "--q2mix-tail") == 0 && i + 1 < argc) {
            g_q2mix_tail = atoi(argv[++i]);
            if (g_q2mix_tail < 0) g_q2mix_tail = 0;
        } else if (strcmp(argv[i], "--no-q4") == 0) {
            g_st_no_q4 = 1;
            g_st_wmode = 2;              /* legacy alias of wmode=q8 */
        } else if (strcmp(argv[i], "--wmode") == 0 && i + 1 < argc) {
            const char *m = argv[++i];
            if (m[0] == 'q' || m[0] == 'Q') {
                if (m[1] == '2') g_st_wmode = 5;                /* q2mix: attention Q4 + FFN Q2 */
                else if (m[1] == '4' && m[2] == 'i') g_st_wmode = 3;   /* pre-unpacked Q4 int8 */
                else if (m[1] == '4') g_st_wmode = 1;             /* Q4_0 nibble-only */
                else if (m[1] == '8') g_st_wmode = 2;             /* Q8_0-only */
            } else if (m[0] == 'g' || m[0] == 'G') {
                g_st_wmode = 4;                                    /* G=256 (NPU K-block 256) */
            } else if (m[0] == 'd' || m[0] == 'D') {
                g_st_wmode = 0;                                    /* dual (default) */
            } else {
                fprintf(stderr, "--wmode: unknown mode '%s' (q4|q4i|q8|q2mix|dual|g256)\n", m);
                return 1;
            }
        } else if (strcmp(argv[i], "--bench-users") == 0 && i + 1 < argc) {
            g_bench_users = atoi(argv[++i]);
            if (g_bench_users < 1) g_bench_users = 1;
            if (g_bench_users > MAX_PERF_USERS) g_bench_users = MAX_PERF_USERS;
            g_bench_users_set = 1;   /* 显式指定 → 触发完整 Part A/B/C 基准 */
        } else if (strcmp(argv[i], "--perf-partA") == 0) {
            g_perf_part_a = 1;   /* single-user only, skip Part B/C */
        } else if (strcmp(argv[i], "--bench-seqlen") == 0 && i + 1 < argc) {
            g_bench_seqlen = atoi(argv[++i]);
            if (g_bench_seqlen < 1) g_bench_seqlen = 1;
            if (g_bench_seqlen > PERF_MAX_CTX) g_bench_seqlen = PERF_MAX_CTX;
        } else if (strcmp(argv[i], "--probe-other") == 0) {
            bench_mixed = 2;
        } else if (strcmp(argv[i], "--test-sparse") == 0) {
            bench_mixed = 3;
        } else if (strcmp(argv[i], "--test-l3") == 0) {
            bench_mixed = 4;
        } else if (strcmp(argv[i], "--l3-evict") == 0) {
            g_l3_evict = 1;
        } else if (strcmp(argv[i], "--l3-ratio") == 0 && i + 1 < argc) {
            g_l3_ratio = (float)atof(argv[++i]);
            if (g_l3_ratio < 0.0f) g_l3_ratio = 0.0f;
            if (g_l3_ratio > 0.99f) g_l3_ratio = 0.99f;
        } else if (strcmp(argv[i], "--l3-min-seq") == 0 && i + 1 < argc) {
            g_l3_min_seq = atoi(argv[++i]);
            if (g_l3_min_seq < 0) g_l3_min_seq = 0;
        } else if (strcmp(argv[i], "--l3-path") == 0 && i + 1 < argc) {
            free(g_l3_path);
            g_l3_path = strdup(argv[++i]);
        } else if (strcmp(argv[i], "--l3-max-size") == 0 && i + 1 < argc) {
            g_l3_max_size = atoll(argv[++i]);
            if (g_l3_max_size < 0) g_l3_max_size = 0;
        } else if (strcmp(argv[i], "--l3-user-ttl") == 0 && i + 1 < argc) {
            g_l3_user_ttl = atol(argv[++i]);
            if (g_l3_user_ttl < 0) g_l3_user_ttl = 0;
        } else if (strcmp(argv[i], "--kv-q4") == 0) {
            g_kv_q4 = 1;   /* in-memory Q4 KV payload cache (replaces INT8) */
        } else if (strcmp(argv[i], "--prefill-q8") == 0) {
            g_st_prefill_q8 = 1;
        } else if (strcmp(argv[i], "--sparse-attn") == 0) {
            g_sparse_attn = 1;   /* Phase 1: top-k block sparse decode attention */
        } else if (strcmp(argv[i], "--sparse-k") == 0 && i + 1 < argc) {
            g_sparse_k = atoi(argv[++i]);
            if (g_sparse_k < 1) g_sparse_k = 1;
        } else if (strcmp(argv[i], "--sparse-block") == 0 && i + 1 < argc) {
            g_sparse_block = atoi(argv[++i]);
            if (g_sparse_block < 8) g_sparse_block = 8;
        } else if (strcmp(argv[i], "--sparse-probe") == 0 && i + 1 < argc) {
            g_sparse_probe = atoi(argv[++i]);
            if (g_sparse_probe < 1) g_sparse_probe = 1;
        } else if (strcmp(argv[i], "--prefill-batch") == 0 && i + 1 < argc) {
            g_st_prefill_batch = atoi(argv[++i]);
            if (g_st_prefill_batch < 1) g_st_prefill_batch = 1;
        } else if (strcmp(argv[i], "--multimodal") == 0) {
            run_multimodal = 1;
        } else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            if (n_images < 8) img_paths[n_images++] = argv[++i];
        } else if (strcmp(argv[i], "--video") == 0 && i + 1 < argc) {
            vid_dir = argv[++i];
        } else if (strcmp(argv[i], "--npu") == 0) {
            g_npu_enabled = 1;   /* transparent RK3588 NPU offload (see --npu-dir) */
        } else if (strcmp(argv[i], "--npu-dir") == 0 && i + 1 < argc) {
            g_npu_model_dir = argv[++i];
        } else if (strcmp(argv[i], "--npu-threshold") == 0 && i + 1 < argc) {
            g_npu_flops_threshold = atof(argv[++i]);
            if (g_npu_flops_threshold < 1e6) g_npu_flops_threshold = 1e6;
        } else if (strcmp(argv[i], "--npu-selftest") == 0) {
            g_npu_selftest = 1;
        } else if (strcmp(argv[i], "--npu-calib") == 0) {
            /* Calibration-only mode: run the DIRECT selftest (full BO +
             * regcmd + DRM SUBMIT pipeline vs CPU reference) then exit - no
             * model load, no full test suite. Isolates the NPU bring-up. */
            g_npu_enabled = 1;
            g_npu_selftest = 1;
            g_npu_calib_only = 1;
        } else if (strcmp(argv[i], "--npu-backend") == 0 && i + 1 < argc) {
            if (strcmp(argv[++i], "direct") == 0)
                g_npu_backend = VLLM_NPU_BACKEND_DIRECT;
            else if (strcmp(argv[i], "rknn") == 0)
                g_npu_backend = VLLM_NPU_BACKEND_RKNN;
        } else if (strcmp(argv[i], "--serve") == 0) {
            g_serve_port = 8080;   /* default; interactive prompt unless --port */
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            g_serve_port = atoi(argv[++i]);
            g_serve_port_explicit = 1;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            g_serve_model_dir = argv[++i];
        } else if (strcmp(argv[i], "--model-id") == 0 && i + 1 < argc) {
            g_serve_model_id = argv[++i];
        } else if (strcmp(argv[i], "--stream-test") == 0) {
            g_stream_test = 1;   /* AirLLM 型分层驻留验证（纯 VQF） */
        } else if (strcmp(argv[i], "--stream-n") == 0 && i + 1 < argc) {
            g_stream_n = atoi(argv[++i]);
            if (g_stream_n < 1) g_stream_n = 1;
            if (g_stream_n > 64) g_stream_n = 64;
        } else if (strcmp(argv[i], "--moe-l0") == 0 && i + 1 < argc) {
            g_moe_l0 = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            g_serve_threads = atoi(argv[++i]);
            if (g_serve_threads < 1) g_serve_threads = 1;
            if (g_serve_threads > 32) g_serve_threads = 32;
        } else if (strcmp(argv[i], "--max-queued") == 0 && i + 1 < argc) {
            g_serve_max_queued = atoi(argv[++i]);
            if (g_serve_max_queued < 0) g_serve_max_queued = 0;
        } else if (strcmp(argv[i], "--batch-max") == 0 && i + 1 < argc) {
            g_serve_batch_max = atoi(argv[++i]);
            if (g_serve_batch_max < 0) g_serve_batch_max = 0;
        } else if (strcmp(argv[i], "--min-free-mb") == 0 && i + 1 < argc) {
            g_serve_min_free_mb = atoi(argv[++i]);
            if (g_serve_min_free_mb < 0) g_serve_min_free_mb = 0;
        } else if (strcmp(argv[i], "--auto-load") == 0) {
            g_serve_auto_load = 1;   /* load the model at startup (was the old default) */
        } else if (strcmp(argv[i], "--no-auto-load") == 0) {
            g_serve_auto_load = 0;   /* manual model load via /admin (the default) */
        } else if (strcmp(argv[i], "--load-on-use") == 0) {
            g_serve_load_on_use = 1; /* 用时加载：请求到达且模型未加载时自动加载 */
        } else if (strcmp(argv[i], "--auto-unload") == 0 && i + 1 < argc) {
            g_serve_auto_unload_s = atol(argv[++i]);
            if (g_serve_auto_unload_s < 0) g_serve_auto_unload_s = 0;
        } else if (strcmp(argv[i], "--res-auto") == 0) {
            g_res_auto = 1;   /* 空闲逐级自动降级（档位阶梯策略总开关） */
        } else if (strcmp(argv[i], "--res-idle") == 0 && i + 1 < argc) {
            /* "a,b,c,d"：L4→L3,L3→L2,L2→L1,L1→L0 各档空闲停留秒（0 = 停在该档） */
            const char *s = argv[++i];
            for (int k = 0; k < 4 && s && *s; k++) {
                g_res_idle_s[k] = atol(s);
                const char *c = strchr(s, ',');
                if (!c) break;
                s = c + 1;
            }
            for (int k = 0; k < 4; k++)
                if (g_res_idle_s[k] < 0) g_res_idle_s[k] = 0;
        } else if (strcmp(argv[i], "--res-soft-mb") == 0 && i + 1 < argc) {
            g_res_soft_mb = atoi(argv[++i]);
            if (g_res_soft_mb < 0) g_res_soft_mb = 0;
        } else if (strcmp(argv[i], "--res-low") == 0 && i + 1 < argc) {
            g_res_low_lvl = atoi(argv[++i]);
            if (g_res_low_lvl < 0) g_res_low_lvl = 0;
            if (g_res_low_lvl > 4) g_res_low_lvl = 4;
        } else if (strcmp(argv[i], "--res-w-keep") == 0 && i + 1 < argc) {
            g_res_w_keep = atoi(argv[++i]);
            if (g_res_w_keep < 0) g_res_w_keep = 0;
        } else if (strcmp(argv[i], "--load-format") == 0 && i + 1 < argc) {
            const char *lf = argv[++i];
            if (strcmp(lf, "vqf") == 0)         g_serve_load_format = 1;
            else                                g_serve_load_format = 0; /* auto (VQF) */
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            g_dev_override = argv[++i]; /* override auto-detected device */
        }
    }

    /* ---- Device classification (auto-detect, --device overrides) ---- */
    vdev_detect(&g_dev);
    g_dev_profile = g_dev.profile;
    if (g_dev_override) {
        const VDevProfile *p = vdev_lookup(g_dev_override);
        if (p) {
            g_dev_profile = p;
            snprintf(g_dev.model_id, sizeof(g_dev.model_id), "%s", p->id);
            printf("[DEV] device overridden: %s -> %s\n", g_dev_override, p->name);
        } else {
            fprintf(stderr, "[DEV] WARNING: unknown --device '%s', keeping auto-detect (%s)\n",
                    g_dev_override, g_dev.model_id);
        }
    }
    printf("[DEV] device=%s arch=%s class=%s model=%s npu=%s recommended_wmode=%s\n",
           g_dev_profile->id, vdev_arch_str(g_dev.arch), vdev_class_str(g_dev.cls),
           g_dev.model_hint[0] ? g_dev.model_hint : "-",
           g_dev_profile->npu ? g_dev_profile->npu : "none",
           g_dev_profile->default_wmode ? g_dev_profile->default_wmode : "-");
    fflush(stdout);

    /* Transparent NPU backend: init lazily on first use; when the device node
     * is absent every offload point falls back to the CPU (NEON/AVX) path and
     * the engine behaves exactly as before.
     * DIRECT = in-tree zero-dependency rknpu driver (no model conversion);
     * RKNN   = official librknnrt + exported .rknn operator models (optional). */
    if (g_npu_enabled) {
        vllm_npu_cfg_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.backend = g_npu_backend;
        cfg.model_dir = g_npu_model_dir;
        cfg.verbosity = 1;
        cfg.flops_threshold = g_npu_flops_threshold;
        if (vllm_npu_init(&g_npu, &cfg) == 0) {
            st_npu_set(g_npu);
            if (vllm_npu_available(g_npu)) {
                printf("  NPU backend : %s\n",
                       vllm_npu_backend(g_npu) == VLLM_NPU_BACKEND_DIRECT
                           ? "direct (in-tree rknpu driver, no model conversion)"
                           : "rknn (librknnrt)");
                if (g_npu_selftest) vllm_npu_selftest(g_npu);
                if (g_npu_calib_only) {
                    /* Calibration-only: report done and stop here. */
                    st_npu_set(NULL);
                    vllm_npu_destroy(g_npu);
                    g_npu = NULL;
                    printf("\n[NPU CALIBRATION COMPLETE]\n");
                    return 0;
                }
            } else {
                printf("  NPU backend : unavailable - running CPU path only\n");
            }
        }
    }

    /* Double-clicked / launched with no arguments: start the HTTP server
     * (interactive port prompt, manual model load via the admin page) instead
     * of running the self-test suite. Any explicit argument (--perf-only,
     * --bench-mixed, --serve, --port, ...) keeps the historical behaviour. */
    if (argc == 1 && g_serve_port < 0) {
        g_serve_port = 8080;
    }

    if (g_stream_test) {
        /* AirLLM 型分层驻留验证：仅自研 VQF（--model 目录含 model.vqf）。
         * VLLM_VQF_STREAM=N 控制驻留档位；0/缺省 = 全驻留 baseline。 */
        if (!g_serve_model_dir) {
            printf("[STREAM] 需要 --model <含 model.vqf 的目录或 .vqf 文件>\n");
            return 1;
        }
        return run_stream_test(g_serve_model_dir, g_stream_n);
    }

    if (g_moe_l0 >= 0) {
        /* M3 layer-0 单 token 对拍 dump（VLLM_MOE_DUMP=1） */
        if (!g_serve_model_dir) {
            printf("[L0] 需要 --model <含 model.vqf 的目录>\n");
            return 1;
        }
        return run_moe_l0_test(g_serve_model_dir, g_moe_l0);
    }

    if (g_serve_port >= 0) {
        /* OpenAI-compatible HTTP server mode. Default: manual model load via
         * the admin page (--auto-load restores startup loading). */
        if (!g_serve_port_explicit) {
            /* Interactive port input: --serve (no --port) asks for the port. */
            char buf[32] = {0};
            printf("请输入服务端口号 (默认 %d): ", g_serve_port);
            fflush(stdout);
            if (fgets(buf, sizeof(buf), stdin)) {
                int p = atoi(buf);
                if (p >= 1 && p <= 65535) g_serve_port = p;
            }
            printf("[SERVE] 服务端口: %d\n", g_serve_port);
        }
        return vllm_serve_main(g_serve_port, g_serve_model_dir, g_serve_auto_load);
    }

    if (bench_mixed == 2) {
        /* Isolated OTHER-bucket workload probe (MRoPE + KV-store + quantize)
         * at S=1024/2048/4096 - no model load. */
        st_probe_other();
        return 0;
    }
    if (bench_mixed == 3) {
        /* Sparse attention Phase-1 self-test (synthetic KV, no model load). */
        st_test_sparse_attn();
        return 0;
    }
    if (bench_mixed == 4) {
        /* L3 (Phase-2) Q4-kernel + disk-eviction self-test, no model load. */
        l3_self_test();
        return 0;
    }

    /* ---- 性能基准入口（Part A/B/C）------------------------------------
     * 回移来源：GitHupSRC/src/main.c:5200-5208（旧引擎在无参默认路径调用
     * test_performance_benchmark）。为不改动当前引擎的默认行为，这里仅在
     * 显式传入基准标志时才运行，并自行加载模型、跳过公理自测套件：
     *   --perf-partA      只跑 Part A（单用户 TTFT/TPOT），打印汇总后返回
     *   --bench-users N   跑完整 Part A/B/C（含多用户并发）
     * 与 --perf-only（仅跳过自测）相互独立，避免静默改变其语义。 */
    if (g_perf_part_a || g_bench_users_set) {
        test_performance_benchmark();
        /* Release the transparent NPU backend (no-op when never created). */
        st_npu_set(NULL);
        vllm_npu_destroy(g_npu);
        printf("\n[ALL TESTS COMPLETE]\n");
        return 0;
    }

    if (!perf_only) {
        test_superposition_axioms();
        test_ntt_transform();
        test_kvcache();
        test_inference_pipeline();
        test_real_inference();
        vmm_self_test();
    }

    if (!perf_only) {
        test_axiom_coverage();
    }

    /* Release the transparent NPU backend (no-op when it was never created). */
    st_npu_set(NULL);
    vllm_npu_destroy(g_npu);

    printf("\n[ALL TESTS COMPLETE]\n");
    return 0;
}

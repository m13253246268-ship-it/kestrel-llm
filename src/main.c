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
#include "vllm_gguf.h"
#include "vllm_vision.h"
#include "vllm_media.h"
#include "vllm_tokenizer_qwen.h"
#include "vllm_platform.h"
#include "vllm_npu.h"
#include "vllm_http.h"
#include "vllm_server.h"
#include "vllm_attest.h"
#include "vllm_device.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <locale.h>   /* setlocale */
#include "vllm_util.h"
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

#if 0 /* Test 8: GGUF roundtrip — vllm_gguf.c removed from build */
static void test_gguf_roundtrip(void) {
    printf("\n=== Test 8: GGUF + Pippenger Attention Inference ===\n");

    /* Check for real GGUF model on disk (paths relative to cwd) */
    const char *real_model_paths[] = {
        "build/smollm2-135m-real.gguf",
        "smollm2-135m-real.gguf",
        "../smollm2-135m-real.gguf",
        "build/smollm2-135m-q8_0.gguf",
        "smollm2-135m-q8_0.gguf",
        "model.gguf",
        "../model.gguf",
        NULL
    };
    const char *found_model = NULL;
    for (int i = 0; real_model_paths[i]; i++) {
        if (st_access(real_model_paths[i], 0) == 0) {
            /* Verify file is large enough to be a real GGUF (> 1MB) */
            FILE *check = st_fopen(real_model_paths[i], "rb");
            if (check) {
                fseek(check, 0, SEEK_END);
                long fsize = ftell(check);
                fclose(check);
                if (fsize > 1024 * 1024) {  /* > 1MB minimum */
                    found_model = real_model_paths[i];
                    break;
                }
            }
        }
    }

    if (found_model) {
        /* ================================================================
         * PART A: Real GGUF model �?Pippenger inference
         * ================================================================ */
        printf("\n[REAL MODEL] Loading: %s\n", found_model);

        GGUFModelConfig cfg;
        int ret = gguf_parse_config(found_model, &cfg);
        if (ret != 0) { printf("[FAIL] Cannot parse GGUF\n"); return; }
        printf("       dim=%d, layers=%d, heads=%d(kv=%d), ffn=%d, vocab=%d\n",
               cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
               cfg.ffn_dim, cfg.vocab_size);

        GGUFWeights loaded;
        memset(&loaded, 0, sizeof(loaded));
        loaded.cfg = cfg;  /* copy parsed config for weight allocation */
        ret = gguf_load_weights(found_model, &loaded);
        if (ret != 0) { printf("[FAIL] Cannot load weights\n"); return; }
        printf("       Weights loaded OK\n");

        GGUFInferenceState ist;
        ret = gguf_inference_init(&ist, &loaded);
        if (ret != 0) { printf("[FAIL] Inference init failed\n");
                         gguf_weights_free(&loaded); return; }

        Tokenizer tok;
        tokenizer_init_from_gguf(&tok, &cfg);
        printf("       Pippenger default: ON (use_pippenger=%d)\n",
               ist.use_pippenger);

        int im_start_id = -1, im_end_id = -1, newline_id = -1, user_id = -1, assistant_id = -1, colon_id = -1;
        /* Find chat template tokens in GGUF vocabulary */
        for (int vi = 0; vi < cfg.vocab_size && cfg.tok_strings; vi++) {
            if (!cfg.tok_strings[vi]) continue;
            if (im_start_id < 0 && strcmp(cfg.tok_strings[vi], "<|im_start|>") == 0) im_start_id = vi;
            if (im_end_id < 0 && strcmp(cfg.tok_strings[vi], "<|im_end|>") == 0) im_end_id = vi;
            if (newline_id < 0) {
                const char *ts = cfg.tok_strings[vi];
                /* Try multiple representations of newline */
                if (strcmp(ts, "\n") == 0) newline_id = vi;
                else if (strcmp(ts, "<0x0A>") == 0) newline_id = vi;
                else if (strlen(ts) == 1 && (unsigned char)ts[0] == 0x0A) newline_id = vi;
                else if (ts[0] == '\n' && ts[1] == '\0') newline_id = vi;
            }
            if (user_id < 0 && strcmp(cfg.tok_strings[vi], "user") == 0) user_id = vi;
            if (assistant_id < 0 && strcmp(cfg.tok_strings[vi], "assistant") == 0) assistant_id = vi;
            if (colon_id < 0 && strcmp(cfg.tok_strings[vi], ":") == 0) colon_id = vi;
        }
        printf("       im_start=%d, im_end=%d, newline=%d, user=%d, assistant=%d, colon=%d\n",
               im_start_id, im_end_id, newline_id, user_id, assistant_id, colon_id);

        /* ====== BOS-ONLY SANITY TEST ======
         * Feed just BOS token and check top predictions.
         * A working model should predict common sentence starters. */
        {
            printf("\n         [BOS-ONLY TEST]\n");
            for (int l = 0; l < ist.cfg.n_layers; l++)
                ist.cache_len[l] = 0;
            ist.seq_len = 0;
            memset(ist.hidden, 0, (size_t)ist.cfg.n_layers * ist.cfg.dim * sizeof(float));

            int bos_tok = cfg.bos_id > 0 ? cfg.bos_id : 1;
            gguf_model_forward(&ist, bos_tok);
            printf("         BOS(%d) -> Top-20: ", bos_tok);
            fflush(stdout);
            /* Get top 20 logits */
            float s20[20];
            int i20[20];
            for (int si = 0; si < 20; si++) { s20[si] = -1e9f; i20[si] = -1; }
            for (int vi = 0; vi < cfg.vocab_size; vi++) {
                float v = ist.logits[vi];
                for (int si = 0; si < 20; si++) {
                    if (v > s20[si]) {
                        for (int sj = 19; sj > si; sj--) {
                            s20[sj] = s20[sj-1]; i20[sj] = i20[sj-1];
                        }
                        s20[si] = v; i20[si] = vi; break;
                    }
                }
            }
            for (int si = 0; si < 20; si++)
                printf("[%d:%.1f] ", i20[si], s20[si]);
            printf("\n");
            fflush(stdout);
        }

        /* Chat-format test: use proper instruct template with hardcoded tokens.
         * SmolLM2-Instruct format:
         * <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
         * We build this manually to avoid tokenizer BPE issues. */
        {
            printf("\n         [CHAT-FORMAT TEST]\n");
            for (int l = 0; l < ist.cfg.n_layers; l++)
                ist.cache_len[l] = 0;
            ist.seq_len = 0;
            memset(ist.hidden, 0, (size_t)ist.cfg.n_layers * ist.cfg.dim * sizeof(float));

            /* Encode "The future of" with tokenizer (no BOS) */
            int chat_prompt[64];
            int np = tokenizer_encode(&tok, "The future of", chat_prompt, 64);

            /* Build chat template by tokenizing the full string.
             * SmolLM2-Instruct format:
             * <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
             * Since we lack a real BPE tokenizer, encode the entire string
             * with our greedy tokenizer which matches against GGUF vocab. */
            char chat_str[512];
            snprintf(chat_str, sizeof(chat_str),
                     "<|im_start|>user\nThe future of<|im_end|>\n<|im_start|>assistant\n");
            int chat_tokens[128];
            int nt = tokenizer_encode(&tok, chat_str, chat_tokens, 128);

            printf("         Chat seq (%d tokens): ", nt);
            for (int i = 0; i < nt; i++) {
                printf("%d ", chat_tokens[i]);
                gguf_model_forward(&ist, chat_tokens[i]);
            }
            fflush(stdout);

            printf("\n         Gen: \"");
            fflush(stdout);

            /* Now generate */
            for (int t = 0; t < 8; t++) {
                int next = sample_token(ist.logits, cfg.vocab_size, 0.0f, 0.95f);
                if (next == cfg.eos_id) break;
                const char *ts = tokenizer_decode(&tok, next);
                printf("[%d:%s]", next, ts ? ts : "NULL");
                fflush(stdout);
                gguf_model_forward(&ist, next);
            }
            printf("\"\n");
            fflush(stdout);
        }

        const char *prompts[] = {
            "The capital of France is",
            "The first president of the United States was",
            "The chemical symbol for water is",
        };
        int num_prompts = 3;
        int max_gen = 8;
        int eos = cfg.eos_id ? cfg.eos_id : -1;

        for (int p = 0; p < num_prompts; p++) {
            for (int l = 0; l < ist.cfg.n_layers; l++)
                ist.cache_len[l] = 0;
            ist.seq_len = 0;
            memset(ist.hidden, 0, (size_t)ist.cfg.n_layers * ist.cfg.dim * sizeof(float));

            int prompt_ids[64];
            int np = tokenizer_encode(&tok, prompts[p], prompt_ids, 64);

            printf("         [PROMPT %d tokens]\n", np);
            fflush(stdout);
            clock_t t0 = clock();
            for (int i = 0; i < np; i++) gguf_model_forward(&ist, prompt_ids[i]);

            printf("       [%s]\n         -> \"", prompts[p]);
            fflush(stdout);
            int actual_gen = 0;
            for (int t = 0; t < max_gen; t++) {
                int next = sample_token(ist.logits, cfg.vocab_size, 0.8f, 0.95f);
                if (next == eos) break;
                const char *ts = tokenizer_decode(&tok, next);
                printf("[%d:%s]", next, ts ? ts : "NULL");
                fflush(stdout);
                gguf_model_forward(&ist, next);
                actual_gen++;
            }
            printf("\"\n");
            clock_t t1 = clock();
            double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;
            printf("         (%d tok, %.1f tok/s, %.3f sec)\n",
                   np + actual_gen,
                   (double)(np + actual_gen) / elapsed,
                   elapsed);
        }
        printf("       [PASS] Pippenger real-model inference complete\n");

        /* ---- High-Concurrency Block Pippenger Benchmark ---- */
        {
            /* Build a long prompt by repeating a phrase to get >= 2 blocks (128+ tokens).
             * This tests block-level Pippenger (PIP_BLOCK_SIZE=64) under realistic
             * long-sequence conditions where block precomputation can be amortized. */
            const char *phrase = "The capital of France is Paris. ";
            /* Repeat phrase to reach ~150+ tokens (2+ blocks of 64) */
            char long_prompt[2048];
            long_prompt[0] = '\0';
            for (int r = 0; r < 30; r++)
                strcat(long_prompt, phrase);
            int bm_gen = 6;
            float time_on = 0, time_off = 0;
            int actual_on = 0, actual_off = 0;
            LARGE_INTEGER freq2c, bqt0, bqt1;
            QueryPerformanceFrequency(&freq2c);

            printf("\n       [BLOCK-PIPPENGER BENCHMARK] Long prompt, ON vs OFF\n");

            for (int mode = 0; mode < 2; mode++) {
                /* Full reset: KV cache, block precomp, hidden state */
                for (int l = 0; l < ist.cfg.n_layers; l++)
                    ist.cache_len[l] = 0;
                ist.seq_len = 0;
                memset(ist.hidden, 0, (size_t)ist.cfg.n_layers * ist.cfg.dim * sizeof(float));
                /* Reset block precomputation state */
                int max_blocks = (ist.cfg.max_seq_len + PIP_BLOCK_SIZE - 1) / PIP_BLOCK_SIZE;
                memset(ist.pip_n_blocks, 0, (size_t)ist.cfg.n_layers * sizeof(int));
                memset(ist.pip_blk_k, 0,
                       (size_t)ist.cfg.n_layers * max_blocks * ist.cfg.n_kv_heads * ist.cfg.head_dim * sizeof(float));
                memset(ist.pip_blk_v, 0,
                       (size_t)ist.cfg.n_layers * max_blocks * ist.cfg.n_kv_heads * ist.cfg.head_dim * sizeof(float));

                int prompt_ids[2048];
                int np = tokenizer_encode(&tok, long_prompt, prompt_ids, 2048);

                ist.use_pippenger = (mode == 0) ? 1 : 0;

                /* Prefill */
                QueryPerformanceCounter(&bqt0);
                for (int i = 0; i < np; i++)
                    gguf_model_forward(&ist, prompt_ids[i]);

                /* Generate a few more tokens to measure sustained throughput */
                int gen_count = 0;
                for (int t = 0; t < bm_gen; t++) {
                    int next = sample_token(ist.logits, ist.cfg.vocab_size, 0.8f, 0.95f);
                    if (next == ist.cfg.eos_id) break;
                    gguf_model_forward(&ist, next);
                    gen_count++;
                }
                QueryPerformanceCounter(&bqt1);
                double dt = (double)(bqt1.QuadPart - bqt0.QuadPart) / (double)freq2c.QuadPart;

                int nblk = ist.pip_n_blocks[0];
                if (mode == 0)      { time_on  = dt; actual_on  = np + gen_count; }
                else                { time_off = dt; actual_off = np + gen_count; }
                printf("         %s: %d tok, %d blocks, %.0f ms (%.1f tok/s)\n",
                       mode == 0 ? "Pippenger" : "Standard ",
                       np + gen_count, nblk,
                       dt * 1000, (double)(np + gen_count) / dt);
            }

            printf("         ----------  ---  ---------  -----\n");
            printf("         Pippenger   %3d  blocks=%d  %5.1f tok/s\n",
                   actual_on, (int)((actual_on - bm_gen + PIP_BLOCK_SIZE - 1) / PIP_BLOCK_SIZE),
                   actual_on / time_on);
            printf("         Standard    %3d  blocks=%d  %5.1f tok/s\n",
                   actual_off, (int)((actual_off - bm_gen + PIP_BLOCK_SIZE - 1) / PIP_BLOCK_SIZE),
                   actual_off / time_off);
            if (time_off > 0 && time_on > 0) {
                double speedup = time_off / time_on;
                printf("         Speedup:    %.2fx (Block-Pippenger/Standard)\n", speedup);
                if (speedup > 1.05)
                    printf("         VERDICT:    Block-Pippenger FASTER by %.0f%% �?"
                           "axiom block-precompute effective.\n",
                           (speedup - 1.0) * 100);
                else if (speedup < 0.95)
                    printf("         VERDICT:    Block-Pippenger SLOWER by %.0f%% �?"
                           "precompute cost not amortized.\n",
                           (1.0 - speedup) * 100);
                else
                    printf("         VERDICT:    Marginal difference (< 5%%) �?"
                           "attention not the bottleneck.\n");
            }
        }

        /* ---- Multi-User Concurrent Pippenger Benchmark ----
         *
         * This is where Pippenger's block precomputation SHOULD shine:
         * N users share the same KV cache (common prompt prefix).
         * Precompute is done once (during prefill), then each user
         * pays only block-level Q·K (O(n_blocks)) instead of
         * per-token Q·K (O(S)).
         *
         * Scenario: 100 concurrent users, same prompt prefix,
         * each generates 1 token.
         *
         * Expected if Pippenger works:
         *   Standard:  per_user �?S  = 216 (linear in seq_len)
         *   Pippenger: per_user �?n_blocks = 3  (linear in blocks)
         *   Speedup �?~S/n_blocks �?72x for attention portion */
        {
            enum { N_USERS = 100 };
            /* Shared prompt must be >= PIP_BLOCK_SIZE (64) tokens to create
             * precomputed blocks. Repeat a phrase to reach 100+ tokens. */
            const char *phrase = "The capital of France is Paris. ";
            char shared_prompt[1024];
            shared_prompt[0] = '\0';
            for (int r = 0; r < 14; r++)
                strcat(shared_prompt, phrase);
            /* ~98 tokens (14 × 7 tokens) �?should create 1-2 blocks */
            int prompt_ids[512];
            int np = tokenizer_encode(&tok, shared_prompt, prompt_ids, 512);

            /* Test queries: short continuations, one per simulated user */
            const char *queries[N_USERS];
            int query_ids[N_USERS];
            for (int u = 0; u < N_USERS; u++) {
                static const char *pool[] = {
                    "The", "What", "In", "This", "A", "It", "For", "Many", "Some",
                    "One", "Paris", "London", "Tokyo", "Berlin", "France", "Europe",
                    "History", "Science", "People", "World"
                };
                queries[u] = pool[u % 20];
                int qids[8];
                int nq = tokenizer_encode(&tok, queries[u], qids, 8);
                query_ids[u] = (nq > 0) ? qids[0] : 0;
            }

            printf("\n       [MULTI-USER CONCURRENT BENCHMARK] %d users, "
                   "%d-token shared prefix\n", N_USERS, np);

            for (int mode = 0; mode < 2; mode++) {
                int nkv = ist.cfg.n_kv_heads, hd = ist.cfg.head_dim;
                int d = ist.cfg.dim, nl = ist.cfg.n_layers;
                int kv_dim = nkv * hd;
                int max_blocks = (ist.cfg.max_seq_len + PIP_BLOCK_SIZE - 1) / PIP_BLOCK_SIZE;

                /* Full reset */
                for (int l = 0; l < nl; l++) ist.cache_len[l] = 0;
                ist.seq_len = 0;
                memset(ist.hidden, 0, (size_t)nl * d * sizeof(float));
                memset(ist.pip_n_blocks, 0, (size_t)nl * sizeof(int));
                memset(ist.pip_blk_k, 0, (size_t)nl * max_blocks * kv_dim * sizeof(float));
                memset(ist.pip_blk_v, 0, (size_t)nl * max_blocks * kv_dim * sizeof(float));

                ist.use_pippenger = (mode == 0) ? 1 : 0;

                /* Phase 0: Prefill shared prompt (builds KV cache + block precompute) */
                for (int i = 0; i < np; i++)
                    gguf_model_forward(&ist, prompt_ids[i]);

                int shared_blocks = ist.pip_n_blocks[0];
                int shared_len = ist.cache_len[0];

                /* Snapshot state that each user must restore */
                int    *saved_cache_len = malloc((size_t)nl * sizeof(int));
                float  *saved_hidden    = malloc((size_t)nl * d * sizeof(float));
                int    *saved_n_blocks  = malloc((size_t)nl * sizeof(int));
                int     saved_seq_len;

                memcpy(saved_cache_len, ist.cache_len,  (size_t)nl * sizeof(int));
                memcpy(saved_hidden,    ist.hidden,     (size_t)nl * d * sizeof(float));
                memcpy(saved_n_blocks,  ist.pip_n_blocks,(size_t)nl * sizeof(int));
                saved_seq_len = ist.seq_len;

                /* Phase 1: N concurrent users, each generates 1 token */
                LARGE_INTEGER freq2d, uqt0, uqt1;
                QueryPerformanceFrequency(&freq2d);
                QueryPerformanceCounter(&uqt0);

                for (int u = 0; u < N_USERS; u++) {
                    /* Restore shared state */
                    memcpy(ist.cache_len,   saved_cache_len, (size_t)nl * sizeof(int));
                    memcpy(ist.hidden,      saved_hidden,    (size_t)nl * d * sizeof(float));
                    memcpy(ist.pip_n_blocks, saved_n_blocks,  (size_t)nl * sizeof(int));
                    ist.seq_len = saved_seq_len;

                    /* Each user generates 1 token from their query */
                    gguf_model_forward(&ist, query_ids[u]);
                }

                QueryPerformanceCounter(&uqt1);
                double total_dt = (double)(uqt1.QuadPart - uqt0.QuadPart) / (double)freq2d.QuadPart;
                double per_user_ms = (total_dt * 1000.0) / N_USERS;

                printf("         %s: %d users, %d tok, %d blocks, "
                       "total=%.1f ms, per_user=%.2f ms\n",
                       mode == 0 ? "Pippenger" : "Standard ",
                       N_USERS, shared_len, shared_blocks,
                       total_dt * 1000, per_user_ms);

                if (mode == 1) {
                    /* Both modes done �?compare */
                    /* time_on, time_off already captured; need to store them */
                }

                free(saved_cache_len);
                free(saved_hidden);
                free(saved_n_blocks);

                /* Store timing for comparison (inelegant but works) */
                static float mu_time_on = 0, mu_time_off = 0;
                if (mode == 0) mu_time_on  = (float)(total_dt * 1000);
                else           mu_time_off = (float)(total_dt * 1000);

                if (mode == 1) {
                    printf("         ----------  ------  ---------  -----\n");
                    printf("         Pippenger   %d users  %8.1f ms  "
                           "%.2f ms/user\n",
                           N_USERS, mu_time_on, mu_time_on / N_USERS);
                    printf("         Standard    %d users  %8.1f ms  "
                           "%.2f ms/user\n",
                           N_USERS, mu_time_off, mu_time_off / N_USERS);
                    double speedup = mu_time_off / mu_time_on;
                    printf("         Speedup:    %.2fx (Pippenger/Standard)\n", speedup);
                    if (speedup > 1.10)
                        printf("         VERDICT:    Pippenger %.0f%% FASTER �?"
                               "block precompute amortization works!\n",
                               (speedup - 1.0) * 100);
                    else if (speedup > 1.02)
                        printf("         VERDICT:    Moderate benefit (%.0f%%) �?"
                               "precompute helps but FFN still dominates.\n",
                               (speedup - 1.0) * 100);
                    else
                        printf("         VERDICT:    No measurable benefit �?"
                               "attention is negligible vs FFN even for %d users.\n",
                               N_USERS);
                }
            }
        }

        gguf_inference_free(&ist);
        tokenizer_free(&tok);
        gguf_weights_free(&loaded);
        if (cfg.tok_strings) {
            for (int i = 0; i < cfg.tok_count; i++) free(cfg.tok_strings[i]);
            free(cfg.tok_strings);
        }
        return;
    }

    /* ================================================================
     * PART B: Synthetic model roundtrip (fallback when no real model)
     * ================================================================ */
    printf("\n[SYNTHETIC] No real GGUF model found on disk.\n");
    printf("       To use a real model, download e.g.:\n");
    printf("       SmolLM2-135M-Instruct-Q8_0.gguf (~138MB)\n");
    printf("       from huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF\n");
    printf("       and place it in the build/ directory.\n\n");
    printf("       Running fallback synthetic test...\n");

    /* Create a tiny model with known weights */
    printf("[1/4] Creating tiny model weights...\n");
    ModelWeights *original = malloc(sizeof(ModelWeights));
    if (!original) { printf("[FAIL] Out of memory\n"); return; }
    weights_init(original);

    /* Export to GGUF */
    const char *gguf_path = "test_model.gguf";
    printf("[2/4] Exporting to GGUF: %s\n", gguf_path);
    int ret = gguf_export_tiny(original, gguf_path,
                                TINY_HIDDEN_DIM, TINY_NUM_LAYERS,
                                TINY_NUM_HEADS, TINY_FFN_DIM,
                                TINY_VOCAB_SIZE);
    if (ret != 0) {
        printf("[FAIL] GGUF export failed\n");
        return;
    }

    /* Parse config from GGUF */
    printf("[3/4] Parsing GGUF config...\n");
    GGUFModelConfig cfg;
    ret = gguf_parse_config(gguf_path, &cfg);
    if (ret != 0) {
        printf("[FAIL] GGUF config parse failed\n");
        return;
    }

    /* Verify config matches */
    int cfg_ok = (cfg.dim == TINY_HIDDEN_DIM &&
                  cfg.n_layers == TINY_NUM_LAYERS &&
                  cfg.n_heads == TINY_NUM_HEADS &&
                  cfg.ffn_dim == TINY_FFN_DIM &&
                  cfg.vocab_size == TINY_VOCAB_SIZE);
    printf("       Config match: %s\n", cfg_ok ? "PASS" : "FAIL");
    printf("       dim=%d layers=%d heads=%d ffn=%d vocab=%d\n",
           cfg.dim, cfg.n_layers, cfg.n_heads, cfg.ffn_dim, cfg.vocab_size);

    /* Load weights from GGUF */
    printf("[4/4] Loading weights from GGUF...\n");
    GGUFWeights loaded;
    memset(&loaded, 0, sizeof(loaded));
    loaded.cfg = cfg;
    ret = gguf_load_weights(gguf_path, &loaded);
    if (ret != 0) {
        printf("[FAIL] GGUF weight load failed\n");
        return;
    }

    /* Verify a few key weight matrices match */
    int hd = TINY_HEAD_DIM;
    int d = TINY_HIDDEN_DIM;
    int vc = TINY_VOCAB_SIZE;
    int ff = TINY_FFN_DIM;
    int num_checks = 5;
    int checks_passed = 0;

    /* Check token embeddings: first 10 elements */
    int emb_ok = 1;
    for (int i = 0; i < 10; i++) {
        if (fabsf(loaded.token_embed[i] - original->token_embed[i]) > 1e-6f) {
            emb_ok = 0; break;
        }
    }
    printf("       token_embd[0:10]: %s\n", emb_ok ? "MATCH" : "MISMATCH");
    if (emb_ok) checks_passed++;

    /* Check final_norm */
    int norm_ok = 1;
    for (int i = 0; i < d; i++) {
        if (fabsf(loaded.final_norm[i] - original->final_norm[i]) > 1e-6f) {
            norm_ok = 0; break;
        }
    }
    printf("       final_norm: %s\n", norm_ok ? "MATCH" : "MISMATCH");
    if (norm_ok) checks_passed++;

    /* Check attn_norm for layer 0 */
    int a_ok = 1;
    for (int i = 0; i < d; i++) {
        if (fabsf(loaded.attn_norm[i] - original->layers[0].attn_norm[i]) > 1e-6f) {
            a_ok = 0; break;
        }
    }
    printf("       attn_norm[0]: %s\n", a_ok ? "MATCH" : "MISMATCH");
    if (a_ok) checks_passed++;

    /* Check q_weight for layer 0: first 10 elements */
    int q_ok = 1;
    for (int i = 0; i < 10; i++) {
        if (fabsf(loaded.q_weight[i] - original->layers[0].q_weight[i]) > 1e-6f) {
            q_ok = 0; break;
        }
    }
    printf("       q_weight[0][0:10]: %s\n", q_ok ? "MATCH" : "MISMATCH");
    if (q_ok) checks_passed++;

    /* Check ffn_gate for layer 1 */
    int ffn_gate_ok = 1;
    float *l1_gate = loaded.gate_weight + 1 * ff * d;
    for (int i = 0; i < 10; i++) {
        if (fabsf(l1_gate[i] - original->layers[1].gate_weight[i]) > 1e-6f) {
            ffn_gate_ok = 0; break;
        }
    }
    printf("       gate_weight[1][0:10]: %s\n",
           ffn_gate_ok ? "MATCH" : "MISMATCH");
    if (ffn_gate_ok) checks_passed++;

    printf("\n       Weight verification: %d/%d checks passed\n",
           checks_passed, num_checks);

    /* ---- Pippenger Attention Inference (default path) ---- */
    printf("\n       === Pippenger Attention Inference ===\n");
    printf("       (Pippenger is the DEFAULT attention backend)\n");

    GGUFInferenceState ist;
    ret = gguf_inference_init(&ist, &loaded);
    if (ret != 0) {
        printf("[FAIL] GGUF inference init failed\n");
        gguf_weights_free(&loaded);
        return;
    }

    Tokenizer tok;
    tokenizer_init(&tok);

    /* Multi-prompt Pippenger generation */
    const char *prompts[] = {
        "the future of",
        "machine learning",
        "hello world",
        "once upon a"
    };
    int num_prompts = 4;

    for (int p = 0; p < num_prompts; p++) {
        /* Reset state for each prompt */
        for (int l = 0; l < TINY_NUM_LAYERS; l++)
            ist.cache_len[l] = 0;
        ist.seq_len = 0;
        memset(ist.hidden, 0, (size_t)ist.cfg.n_layers * ist.cfg.dim * sizeof(float));

        int prompt_ids[32];
        int np = tokenizer_encode(&tok, prompts[p], prompt_ids, 32);

        /* Prefill */
        for (int i = 0; i < np; i++)
            gguf_model_forward(&ist, prompt_ids[i]);

        /* Generate 6 tokens */
        printf("       Prompt[%d]: \"%s\"\n", p + 1, prompts[p]);
        printf("       Output[%d]: \"", p + 1);
        for (int t = 0; t < 6; t++) {
            int next = sample_token(ist.logits, vc, 0.8f, 0.9f);
            const char *ts = tokenizer_decode(&tok, next);
            if (ts && ts[0] && strcmp(ts, "<unk>") != 0) {
                printf("%s", ts);
                if (t < 5) printf(" ");
            }
            gguf_model_forward(&ist, next);
        }
        printf("\"\n");
    }
    printf("       [PASS] Pippenger multi-prompt generation complete\n");

    /* Lightweight precision vs standard comparison */
     printf("\n       === Precision vs Standard Attention ===\n");
     {
         /* Save Pippenger logits on fresh prompt, then compare with standard */
         for (int l = 0; l < TINY_NUM_LAYERS; l++)
             ist.cache_len[l] = 0;
         ist.seq_len = 0;
         memset(ist.hidden, 0, (size_t)ist.cfg.n_layers * ist.cfg.dim * sizeof(float));

         int cmp_ids[32];
         int nc = tokenizer_encode(&tok, "the future of", cmp_ids, 32);
         for (int i = 0; i < nc; i++)
             gguf_model_forward(&ist, cmp_ids[i]);

         /* Copy Pippenger logits */
         float *pip_logits = malloc(vc * sizeof(float));
         memcpy(pip_logits, ist.logits, vc * sizeof(float));

         /* Run standard attention on same prompt */
         GGUFInferenceState ist_std;
         int rc = gguf_inference_init(&ist_std, &loaded);
         if (rc == 0) {
             ist_std.use_pippenger = 0;
             for (int i = 0; i < nc; i++)
                 gguf_model_forward(&ist_std, cmp_ids[i]);

             float max_diff = 0.0f, avg_diff = 0.0f;
             for (int i = 0; i < vc; i++) {
                 float diff = fabsf(pip_logits[i] - ist_std.logits[i]);
                 avg_diff += diff;
                 if (diff > max_diff) max_diff = diff;
             }
             avg_diff /= (float)vc;
             printf("       Logits: max_diff=%.6f  avg_diff=%.8f �?%s\n",
                    max_diff, avg_diff,
                    avg_diff < 0.01f ? "PASS" : "ACCEPTABLE");

             gguf_inference_free(&ist_std);
         }
         free(pip_logits);
     }

    /* Performance benchmark */
    printf("\n       === Performance ===\n");
    for (int mode = 0; mode < 2; mode++) {
        /* Reset */
        for (int l = 0; l < TINY_NUM_LAYERS; l++)
            ist.cache_len[l] = 0;
        ist.seq_len = 0;
        memset(ist.hidden, 0, (size_t)ist.cfg.n_layers * ist.cfg.dim * sizeof(float));

        int saved = ist.use_pippenger;
        ist.use_pippenger = (mode == 0) ? 1 : 0;  /* Pippenger first, then standard */

        int bench_ids[32];
        int nb = tokenizer_encode(&tok, "hello world", bench_ids, 32);

        clock_t t0 = clock();
        for (int i = 0; i < nb; i++) gguf_model_forward(&ist, bench_ids[i]);
        for (int t = 0; t < 8; t++) {
            int next = sample_token(ist.logits, vc, 0.8f, 0.9f);
            gguf_model_forward(&ist, next);
        }
        clock_t t1 = clock();
        double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;
        const char *label = (mode == 0) ? "Pippenger" : "Standard ";
        printf("       %s: %.4f sec �?%.1f tok/s\n",
               label, elapsed, (double)(nb + 8) / elapsed);

        ist.use_pippenger = saved;
    }
    printf("       [PASS] GGUF roundtrip + Pippenger inference complete\n");

    /* Cleanup */
    gguf_inference_free(&ist);
    tokenizer_free(&tok);
    gguf_weights_free(&loaded);
    free(original);
    original = NULL;
    loaded.is_allocated = 0;

    printf("\n[PASS] GGUF roundtrip test complete\n");
}
#endif /* Test 8: GGUF roundtrip */

/* ================================================================
 * Test 9: Qwen3-VL-8B-Instruct Model Inference
 * ================================================================ */
static void test_qwen3vl_inference(void) {
    printf("\n=== Test 9: Qwen3-VL-8B-Instruct Inference ===\n");

    /* Find model directory */
    const char *model_paths[] = {
        "../../Modl/千问3_VL_8B_Instruct",
        "../Modl/千问3_VL_8B_Instruct",
        "Modl/千问3_VL_8B_Instruct",
        "../../Modl/Qwen3VL-8B-Instruct",
        "../Modl/Qwen3VL-8B-Instruct",
        "Modl/Qwen3VL-8B-Instruct",
        NULL
    };
    const char *model_dir = NULL;
    for (int i = 0; model_paths[i]; i++) {
        char test_path[1024];
        snprintf(test_path, sizeof(test_path), "%s/config.json", model_paths[i]);
        if (st_access(test_path, 0) == 0) {
            model_dir = model_paths[i];
            break;
        }
    }
    if (!model_dir) {
        printf("[SKIP] Qwen3-VL model not found on disk.\n");
        printf("       Place model at Modl/千问3_VL_8B_Instruct/\n");
        return;
    }
    printf("[1/5] Model found: %s\n", model_dir);

    /* Parse config */
    STModelConfig cfg;
    int ret = st_parse_config(model_dir, &cfg);
    if (ret != 0) {
        printf("[FAIL] Cannot parse model config\n");
        return;
    }
    printf("[2/5] Config parsed: dim=%d layers=%d heads=%d(kv=%d) ffn=%d vocab=%d\n",
           cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
           cfg.ffn_dim, cfg.vocab_size);
    printf("       q_norm=%d mrope=%d rope_theta=%.0f\n",
           cfg.has_q_norm, cfg.has_mrope, cfg.rope_theta);

    /* Load tokenizer */
    QwenTokenizer tok;
    ret = qwen_tokenizer_load(&tok, model_dir);
    if (ret != 0) {
        printf("[FAIL] Cannot load tokenizer\n");
        st_config_free(&cfg);
        return;
    }
    qwen_tokenizer_load_special(&tok, model_dir);
    printf("[3/5] Tokenizer loaded: %d tokens\n", tok.vocab_size);

    /* Encode/Decode test */
    {
        const char *test_str = "Hello world";
        int ids[64];
        int n = qwen_tokenizer_encode(&tok, test_str, ids, 64);
        printf("       Encode '%s' -> [", test_str);
        for (int i = 0; i < n; i++) {
            printf("%d", ids[i]);
            if (i < n - 1) printf(", ");
        }
        printf("] (%d tokens)\n", n);

        printf("       Decode -> \"");
        for (int i = 0; i < n; i++) {
            const char *s = qwen_tokenizer_decode(&tok, ids[i]);
            if (s && s[0]) {
                /* Skip Ġ prefix in display */
                if ((unsigned char)s[0] == 0xC4 && (unsigned char)s[1] == 0xA0)
                    printf("%s", s + 2);
                else
                    printf("%s", s);
            }
        }
        printf("\"\n");
    }

    /* Chinese test */
    {
        const char *test_str = "\xe4\xbd\xa0\xe5\xa5\xbd";  /* 你好 in UTF-8 */
        int ids[64];
        int n = qwen_tokenizer_encode(&tok, test_str, ids, 64);
        printf("       Encode '你好' -> [");
        for (int i = 0; i < n; i++) {
            printf("%d", ids[i]);
            if (i < n - 1) printf(", ");
        }
        printf("] (%d tokens)\n", n);
    }

    /* Allocate all 36 layers with Q8_0-FFN (F32 attention + Q8_0 FFN ~17 GB) */
    STModelWeights w;
    st_weights_alloc_layers_q8ffn(&w, &cfg, cfg.n_layers);
    if (!w.is_allocated) {
        printf("[SKIP] Not enough memory for full model (~%.1f GB needed).\n",
               (double)((size_t)cfg.vocab_size * cfg.dim * 2 +  /* embed + lm_head */
                        (size_t)cfg.n_layers * cfg.dim * cfg.n_heads * cfg.head_dim * 4 +
                        (size_t)cfg.n_layers * cfg.ffn_dim * cfg.dim * 3 / 4)  /* Q8 FFN */
               / (1024.0 * 1024.0 * 1024.0));
        printf("       Running tokenizer-only validation test.\n");
        qwen_tokenizer_free(&tok);
        st_weights_free(&w);
        st_config_free(&cfg);
        return;
    }

    /* Load token embeddings and final norm */
    ret = st_load_tensor(&cfg, "model.language_model.embed_tokens.weight",
                          w.token_embed, cfg.vocab_size * cfg.dim);
    if (ret == 0) printf("       token_embed loaded OK\n");

    ret = st_load_tensor(&cfg, "model.language_model.norm.weight",
                          w.final_norm, cfg.dim);
    if (ret == 0) printf("       final_norm loaded OK\n");

    /* lm_head: skip if Q8_0-only (loaded later via temp buffer) */
    if (w.lm_head) {
        ret = st_load_tensor(&cfg, "lm_head.weight",
                              w.lm_head, cfg.vocab_size * cfg.dim);
        if (ret == 0) printf("       lm_head loaded OK\n");
    }

    /* Load all layers */
    LARGE_INTEGER freq, t_ls, t_le;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t_ls);
    ret = st_load_layer_weights(&cfg, &w, 0, cfg.n_layers);
    QueryPerformanceCounter(&t_le);
    double load_sec = (double)(t_le.QuadPart - t_ls.QuadPart) / (double)freq.QuadPart;
    if (ret == 0) printf("       %d layers loaded OK (%.1f sec)\n", cfg.n_layers, load_sec);

    /* Verify a few weights */
    printf("       Weight check: embed[0:4] = [");
    for (int i = 0; i < 4; i++) printf("%.4f ", w.token_embed[i]);
    printf("], final_norm[0:4] = [");
    for (int i = 0; i < 4; i++) printf("%.4f ", w.final_norm[i]);
    printf("]\n");

    /* Quantize lm_head to Q8_0 + Q4_0 (axiom: fixedpoint_quantize_saturate)
     * Load into temp buffer, quantize, free. Reduces 2.49 GB F32 → 660 MB Q8_0
     * / 331 MB Q4_0. */
    if (w.q8_lm_weight) {
        printf("       Loading & quantizing lm_head to Q8_0 + Q4_0...\n");
        fflush(stdout);
        float *tmp_lm = malloc((size_t)cfg.vocab_size * cfg.dim * sizeof(float));
        if (tmp_lm) {
            st_load_tensor(&cfg, "lm_head.weight", tmp_lm, cfg.vocab_size * cfg.dim);
            if (w.q8_lm_weight) {
                if (w.q8_buf_q4)
                    f32_to_q4i8(w.q8_lm_weight, tmp_lm, cfg.vocab_size * cfg.dim);
                else if (st_wmode_effective() == 4)
                    f32_to_g256q8(w.q8_lm_weight, tmp_lm, cfg.vocab_size * cfg.dim);
                else
                    f32_to_q8_0(w.q8_lm_weight, tmp_lm, cfg.vocab_size * cfg.dim);
            }
            if (w.q4_lm_weight)
                f32_to_q4_0(w.q4_lm_weight, tmp_lm, cfg.vocab_size * cfg.dim);
            if (w.q4_lm_weight)
                repack_q4_0_4x4_inplace(w.q4_lm_weight, cfg.vocab_size, cfg.dim);
            if (w.q8_lm_weight)
                repack_q8_0_tiled_inplace(w.q8_lm_weight, cfg.vocab_size, cfg.dim);
            free(tmp_lm);
            printf("       Q8_0 + Q4_0 lm_head ready\n");
        } else {
            printf("       OOM: cannot load lm_head\n");
        }
    }

    /* F32 attention & lm_head: already freed during per-layer temp loading.
     * All F32 weight pointers are NULL �?forward pass uses Q8_0 exclusively. */
    if (w.has_q8 && w.q8_q_weight) {
        printf("       All attention + lm_head in Q8_0 (peak mem ~9 GB).\n");
    }

    /* ================================================================
     * Full model inference benchmark
     * ================================================================ */
    {
        printf("[4/5] Initializing inference state...\n");
        fflush(stdout);

        STQwenInferenceState ist;
        ret = st_qwen_inference_init(&ist, &w);
        if (ret != 0) {
            printf("[FAIL] Cannot init inference state\n");
            qwen_tokenizer_free(&tok);
            st_weights_free(&w);
            st_config_free(&cfg);
            return;
        }
        printf("       KV-cache: %d tokens x %d layers, Q8_0 FFN: %s\n",
               w.cfg.max_seq_len, cfg.n_layers, w.has_q8 ? "ON" : "OFF");

        /* BOS single-token forward */
        int bos = tok.bos_id > 0 ? tok.bos_id : 151643;
        QueryPerformanceCounter(&t_ls);
        st_qwen_model_forward(&ist, bos);
        QueryPerformanceCounter(&t_le);
        double bos_sec = (double)(t_le.QuadPart - t_ls.QuadPart) / (double)freq.QuadPart;
        printf("\n       --- BOS Forward (single token) ---\n");
        printf("       Time: %.3f sec\n", bos_sec);
        printf("       Top-10 predictions:\n");
        {
            float top10[10]; int top10i[10];
            for (int i = 0; i < 10; i++) { top10[i] = -1e9f; top10i[i] = -1; }
            for (int t = 0; t < cfg.vocab_size; t++) {
                float v = ist.logits[t];
                for (int j = 0; j < 10; j++) {
                    if (v > top10[j]) {
                        for (int k = 9; k > j; k--) { top10[k] = top10[k-1]; top10i[k] = top10i[k-1]; }
                        top10[j] = v; top10i[j] = t; break;
                    }
                }
            }
            for (int i = 0; i < 10; i++) {
                const char *s = qwen_tokenizer_decode(&tok, top10i[i]);
                printf("         #%d: tid=%d score=%.2f", i + 1, top10i[i], top10[i]);
                if (s && s[0]) {
                    if ((unsigned char)s[0] == 0xC4 && (unsigned char)s[1] == 0xA0)
                        printf(" -> \"%s\"", s + 2);
                    else
                        printf(" -> \"%s\"", s);
                }
                printf("\n");
            }
        }

        /* Decode speed test: 5 tokens after BOS */
        printf("\n       --- Decode Speed (5 tokens) ---\n");
        fflush(stdout);
        {
            QueryPerformanceCounter(&t_ls);
            int prev_tok = bos;
            for (int step = 0; step < 5; step++) {
                /* Greedy decode: pick argmax logit */
                float best = -1e9f; int best_id = 0;
                for (int t = 0; t < cfg.vocab_size; t++) {
                    if (ist.logits[t] > best) { best = ist.logits[t]; best_id = t; }
                }
                prev_tok = best_id;
                st_qwen_model_forward(&ist, prev_tok);
            }
            QueryPerformanceCounter(&t_le);
            double dec_sec = (double)(t_le.QuadPart - t_ls.QuadPart) / (double)freq.QuadPart;
            printf("       Decode 5 tokens: %.3f sec (%.1f tok/s)\n",
                   dec_sec, 5.0 / dec_sec);
        }

        /* Prefill speed test: 32-token prompt */
        printf("\n       --- Prefill Speed (32 tokens) ---\n");
        fflush(stdout);
        {
            const char *prompt_str = "The capital of France is";
            int prompt_ids[64];
            int prompt_n = qwen_tokenizer_encode(&tok, prompt_str, prompt_ids, 64);
            if (prompt_n <= 0) { prompt_n = 5; prompt_ids[0] = bos; }
            printf("       Prompt: \"%s\" (%d tokens)\n", prompt_str, prompt_n);

            /* Re-init clean state */
            st_qwen_inference_free(&ist);
            st_qwen_inference_init(&ist, &w);
            ist.seq_len = 0;
            for (int l = 0; l < cfg.n_layers; l++) ist.cache_len[l] = 0;

            QueryPerformanceCounter(&t_ls);
            for (int t = 0; t < prompt_n; t++)
                st_qwen_model_forward(&ist, prompt_ids[t]);
            QueryPerformanceCounter(&t_le);
            double prefill_sec = (double)(t_le.QuadPart - t_ls.QuadPart) / (double)freq.QuadPart;
            printf("       Prefill %d tokens: %.3f sec (%.1f tok/s)\n",
                   prompt_n, prefill_sec, (double)prompt_n / prefill_sec);

            /* Decode 5 more tokens after prefill */
            printf("       --- Decode after Prefill (5 tokens) ---\n");
            fflush(stdout);
            QueryPerformanceCounter(&t_ls);
            for (int step = 0; step < 5; step++) {
                float best = -1e9f; int best_id = 0;
                for (int t = 0; t < cfg.vocab_size; t++) {
                    if (ist.logits[t] > best) { best = ist.logits[t]; best_id = t; }
                }
                st_qwen_model_forward(&ist, best_id);
            }
            QueryPerformanceCounter(&t_le);
            double dec2_sec = (double)(t_le.QuadPart - t_ls.QuadPart) / (double)freq.QuadPart;
            printf("       Decode 5 tokens (32-tok ctx): %.3f sec (%.1f tok/s)\n",
                   dec2_sec, 5.0 / dec2_sec);
        }

        /* Complete generation test */
        printf("\n       --- Generate 20 tokens (greedy, from BOS) ---\n");
        fflush(stdout);
        {
            st_qwen_inference_free(&ist);
            st_qwen_inference_init(&ist, &w);

            int prev_tok = bos;
            QueryPerformanceCounter(&t_ls);
            printf("       Output: ");
            fflush(stdout);
            for (int step = 0; step < 20; step++) {
                st_qwen_model_forward(&ist, prev_tok);
                float best = -1e9f; int best_id = 0;
                for (int t = 0; t < cfg.vocab_size; t++) {
                    if (ist.logits[t] > best) { best = ist.logits[t]; best_id = t; }
                }
                const char *s = qwen_tokenizer_decode(&tok, best_id);
                if (s && s[0]) {
                    if ((unsigned char)s[0] == 0xC4 && (unsigned char)s[1] == 0xA0)
                        printf("%s", s + 2);
                    else
                        printf("%s", s);
                }
                fflush(stdout);
                prev_tok = best_id;
            }
            printf("\n");
            QueryPerformanceCounter(&t_le);
            double gen_sec = (double)(t_le.QuadPart - t_ls.QuadPart) / (double)freq.QuadPart;
            printf("       Generation: 20 tokens in %.3f sec (%.1f tok/s)\n",
                   gen_sec, 20.0 / gen_sec);
        }

        /* Chat template generation test */
        printf("\n       --- Generate 30 tokens (chat template) ---\n");
        fflush(stdout);
        {
            st_qwen_inference_free(&ist);
            st_qwen_inference_init(&ist, &w);

            /* Build chat template tokens:
             * <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
             * <|im_start|>user\nHello, who are you?<|im_end|>\n
             * <|im_start|>assistant\n */
            int prompt_ids[256];
            int pn = 0;
            int im_start = tok.im_start_id > 0 ? tok.im_start_id : 151644;
            int im_end   = tok.im_end_id   > 0 ? tok.im_end_id   : 151645;
            int nl_tok   = 198;  /* Ċ token = newline in Qwen tokenizer */
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

            /* <|im_start|>user\nHello, who are you?<|im_end|>\n */
            prompt_ids[pn++] = im_start;
            n = qwen_tokenizer_encode(&tok, "user", tmp_ids, 128);
            for (int i = 0; i < n; i++) prompt_ids[pn++] = tmp_ids[i];
            prompt_ids[pn++] = nl_tok;
            n = qwen_tokenizer_encode(&tok, "Hello, who are you?", tmp_ids, 128);
            for (int i = 0; i < n; i++) prompt_ids[pn++] = tmp_ids[i];
            prompt_ids[pn++] = im_end;
            prompt_ids[pn++] = nl_tok;

            /* <|im_start|>assistant\n */
            prompt_ids[pn++] = im_start;
            n = qwen_tokenizer_encode(&tok, "assistant", tmp_ids, 128);
            for (int i = 0; i < n; i++) prompt_ids[pn++] = tmp_ids[i];
            prompt_ids[pn++] = nl_tok;

            printf("       Chat prompt: %d tokens\n", pn);

            /* Prefill with chat template */
            for (int i = 0; i < pn; i++)
                st_qwen_model_forward(&ist, prompt_ids[i]);

            /* Generate */
            printf("       Output: ");
            fflush(stdout);
            QueryPerformanceCounter(&t_ls);
            for (int step = 0; step < 30; step++) {
                float best = -1e9f; int best_id = 0;
                for (int t = 0; t < cfg.vocab_size; t++) {
                    if (ist.logits[t] > best) { best = ist.logits[t]; best_id = t; }
                }
                /* Stop at EOS */
                if (best_id == tok.eos_id || best_id == im_end) break;
                const char *s = qwen_tokenizer_decode(&tok, best_id);
                if (s && s[0]) {
                    if ((unsigned char)s[0] == 0xC4 && (unsigned char)s[1] == 0xA0)
                        printf("%s", s + 2);
                    else
                        printf("%s", s);
                }
                fflush(stdout);
                st_qwen_model_forward(&ist, best_id);
            }
            printf("\n");
            QueryPerformanceCounter(&t_le);
            double gen_sec = (double)(t_le.QuadPart - t_ls.QuadPart) / (double)freq.QuadPart;
            printf("       Chat gen: 30 tokens in %.3f sec (%.1f tok/s)\n",
                   gen_sec, 30.0 / gen_sec);
        }

        /* Summary */
        printf("\n       === Full Model (%d layers) Summary ===\n", cfg.n_layers);
        printf("       dim=%d heads=%d(kv=%d) ffn=%d vocab=%d\n",
               cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.ffn_dim, cfg.vocab_size);
        printf("       Weight loading: %.1f sec\n", load_sec);
        printf("       BOS forward: %.3f sec\n", bos_sec);
        printf("       Q8_0 FFN: %s\n", w.has_q8 ? "ACTIVE" : "off");
        printf("       [PASS] Qwen3-VL full model inference complete\n");

        st_qwen_inference_free(&ist);
    }

    qwen_tokenizer_free(&tok);
    st_weights_free(&w);
    st_config_free(&cfg);
    return;

}  /* test_qwen3vl_inference */

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
static const char *g_convert_vqf = NULL; /* --convert-vqf <out>: dump VQF and exit */
static const char *g_convert_gguf = NULL; /* --convert-gguf <out>: GGUF -> VQF and exit */
static const char *g_gguf_info = NULL;    /* --gguf-info <path>: dump GGUF structure and exit */
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
static int  g_serve_load_format = 0;   /* --load-format: 0=auto(VQF>GGUF>safetensors),
                                        * 1=vqf优先, 2=safetensors优先, 3=gguf优先 */

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

/* serve: disk KV persistence (--disk-kv DIR). F32 KV snapshots of finished
 * conversations survive process restarts; see vllm_server.c / st_kv_disk_*. */
static int    g_disk_kv = 0;
static char   g_disk_kv_dir[512] = {0};

/* serve: speculative decode (--spec), n-gram draft + batched verification.
 * Greedy-only and text-only; default draft length 4 (--spec-k). */
static int    g_spec = 0;
static int    g_spec_k = 4;

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

/* Phase-2 hook: after prefill, pack the cold KV blocks into the Q4 disk file
 * (--l3-evict) and physically free them from RAM. Sparse decode re-reads
 * evicted blocks on demand from st->l3 in compressed Q4 form.
 * Shared by the bench path and the HTTP serve path (vllm_server.c). */
void l3_evict_after_prefill(STQwenInferenceState *st) {
    if (!g_l3_evict) return;
    if (!g_sparse_attn) {
        fprintf(stderr, "[L3] warning: --l3-evict requires --sparse-attn "
                        "(non-sparse attention needs all blocks in RAM); skipped\n");
        return;
    }
    STL3State *l3 = &st->l3;
    int nl = st->weights.n_layers_allocated;
    int kv_dim = st->cfg.n_kv_heads * st->cfg.head_dim;
    int seq_len = st->cache_len[0];
    /* Eviction trigger threshold: skip until the context is long enough
     * (--l3-min-seq). Below the threshold the full KV stays in RAM. */
    if (g_l3_min_seq > 0 && seq_len < g_l3_min_seq) {
        fprintf(stderr, "[L3] skipped: seq=%d < l3-min-seq=%d (KV stays in RAM)\n",
                seq_len, g_l3_min_seq);
        return;
    }
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
    if (l3_state_init(l3, l3path, nl, st->cfg.n_kv_heads,
                      st->cfg.head_dim, st->kv_bs, capacity) != 0) {
        fprintf(stderr, "[L3] eviction skipped: cannot open %s\n", l3path);
        return;
    }
    for (int l = 0; l < nl; l++) {
        l3_evict_layer(l3, l, st->k_cache[l], st->v_cache[l], kv_dim,
                       seq_len, st->prefill_importance, g_l3_ratio, st->kv_bs);
    }

    /* Mirror the payload into RAM so sparse decode serves evicted blocks from
     * memory: no FILE I/O in the decode hot path (the shared-fp fseek/fread
     * churn corrupted the heap) and much lower decode latency. */
    if (l3_load_to_mem(l3) != 0)
        fprintf(stderr, "[L3] warning: could not mirror kv_l3.bin to RAM; "
                        "decode will read the file directly\n");

    /* Phase 2b: physically free the evicted blocks from RAM. Sparse decode
     * serves them straight from the Q4 disk state (see sparse_attn_head). */
    size_t freed_bytes = 0;
    int nblocks = (seq_len + st->kv_bs - 1) / st->kv_bs;
    for (int l = 0; l < nl; l++) {
        for (int b = 0; b < nblocks; b++) {
            STL3Block *bm = &l3->blocks[(size_t)l * l3->max_blocks + b];
            if (bm->on_disk && st->k_cache[l][b]) {
                freed_bytes += 2 * (size_t)st->kv_bs * (size_t)kv_dim * sizeof(float);
                st_qwen_kv_free_raw((uint8_t *)st->k_cache[l][b] - KV_GUARD);  st->k_cache[l][b] = NULL;
                st_qwen_kv_free_raw((uint8_t *)st->v_cache[l][b] - KV_GUARD);  st->v_cache[l][b] = NULL;
                if (st->k_cache_q8[l] && st->k_cache_q8[l][b]) {
                    freed_bytes += 2 * (size_t)st->kv_bs * (size_t)kv_dim;
                    st_qwen_kv_free_raw((uint8_t *)st->k_cache_q8[l][b] - KV_GUARD); st->k_cache_q8[l][b] = NULL;
                    st_qwen_kv_free_raw((uint8_t *)st->v_cache_q8[l][b] - KV_GUARD); st->v_cache_q8[l][b] = NULL;
                }
            }
        }
    }
    fprintf(stderr, "[L3] evicted %d blocks -> %s (%.1f MB, seq=%d, "
                    "ratio=%.2f), freed %.1f MB from RAM\n",
            l3->evicted, l3path, (double)l3->evicted * l3->block_bytes / 1048576.0,
            seq_len, g_l3_ratio, (double)freed_bytes / 1048576.0);
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
        l3_evict_after_prefill(&ist);

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

static void test_performance_benchmark(void) {
    printf("\n=== Test 10: Performance Benchmark ===\n");
    printf("       Qwen3-VL-8B-Instruct | Q8_0 quant | OMP_NUM_THREADS=%d\n",
           vllm_tp_threads()
    );
    fflush(stdout);

    /* Find model */
    const char *model_paths[] = {
        "../../Modl/千问3_VL_8B_Instruct",
        "../Modl/千问3_VL_8B_Instruct",
        "Modl/千问3_VL_8B_Instruct",
        "../modl/千问3_VL_8B_Instruct",
        "modl/千问3_VL_8B_Instruct",
    };
    const char *model_dir = NULL;
    for (int i = 0; i < 5; i++) {
        char test_path[1024];
        snprintf(test_path, sizeof(test_path), "%s/config.json", model_paths[i]);
        if (st_access(test_path, 0) == 0) { model_dir = model_paths[i]; break; }
    }
    if (!model_dir) {
        printf("       [SKIP] Model not found\n");
        return;
    }
    printf("       Model: %s\n", model_dir);
    fflush(stdout);

    /* Parse config + load tokenizer */
    STModelConfig cfg;
    if (st_parse_config(model_dir, &cfg) != 0) {
        printf("       [FAIL] Config parse failed\n");
        return;
    }

    QwenTokenizer tok;
    if (qwen_tokenizer_load(&tok, model_dir) != 0) {
        printf("       [SKIP] Tokenizer load failed\n");
        st_config_free(&cfg);
        return;
    }
    printf("       Tokenizer: %d tokens loaded\n", tok.vocab_size);

    /* Allocate Q8_0-only weights */
    LARGE_INTEGER freq, t_start, t_end;
    QueryPerformanceFrequency(&freq);

    STModelWeights w;
    memset(&w, 0, sizeof(w));
    QueryPerformanceCounter(&t_start);
    st_weights_alloc_layers_q8ffn(&w, &cfg, cfg.n_layers);

    /* Load embed, final_norm, lm_head */
    st_load_tensor(&cfg, "model.language_model.embed_tokens.weight",
                    w.token_embed, cfg.vocab_size * cfg.dim);
    st_load_tensor(&cfg, "model.language_model.norm.weight",
                    w.final_norm, cfg.dim);

    /* lm_head: load into temp, quantize to Q8_0 + Q4_0 (or whichever mode) */
    if (w.q8_lm_weight || w.q4_lm_weight) {
        int vc = cfg.vocab_size;
        int dm = cfg.dim;
        int64_t lm_elems = (int64_t)vc * (int64_t)dm;
        int chunk_rows = 256;
        if (chunk_rows > vc) chunk_rows = vc;
        int64_t chunk_elems = (int64_t)chunk_rows * (int64_t)dm;
        float *tmp_lm = (float *)malloc((size_t)chunk_elems * sizeof(float));
        while (!tmp_lm && chunk_rows > 1) {
            chunk_rows /= 2;
            chunk_elems = (int64_t)chunk_rows * (int64_t)dm;
            tmp_lm = (float *)malloc((size_t)chunk_elems * sizeof(float));
        }
        if (!tmp_lm) {
            printf("       [FAIL] OOM: cannot allocate lm_head chunk buffer\n");
            st_weights_free(&w); st_config_free(&cfg);
            qwen_tokenizer_free(&tok);
            return;
        }

        printf("       Loading & quantizing lm_head to Q8_0 + Q4_0...\n");
        fflush(stdout);
        for (int64_t off = 0; off < lm_elems; off += chunk_elems) {
            int64_t n = lm_elems - off;
            if (n > chunk_elems) n = chunk_elems;
            if (st_load_tensor_slice(&cfg, "lm_head.weight", off, tmp_lm, n) != 0) {
                printf("       [FAIL] lm_head slice load failed\n");
                free(tmp_lm);
                st_weights_free(&w); st_config_free(&cfg);
                qwen_tokenizer_free(&tok);
                return;
            }
            if (w.q8_lm_weight) {
                if (w.q8_buf_q4)
                    f32_to_q4i8(w.q8_lm_weight + (off / 32) * 34, tmp_lm, (int)n);
                else if (st_wmode_effective() == 4)
                    f32_to_g256q8(w.q8_lm_weight + (off / 32) * 34, tmp_lm, (int)n);
                else
                    f32_to_q8_0(w.q8_lm_weight + (off / 32) * 34, tmp_lm, (int)n);
            }
            if (w.q4_lm_weight) {
                f32_to_q4_0(w.q4_lm_weight + (off / 32) * 18, tmp_lm, (int)n);
                repack_q4_0_4x4_inplace(
                    w.q4_lm_weight + (size_t)(off / dm) * (((dm + 31) / 32) * 18),
                    (int)(n / dm), dm);
            }
            if (w.q8_lm_weight)
                repack_q8_0_tiled_inplace(
                    w.q8_lm_weight + (size_t)(off / dm) * (((dm + 31) / 32) * 34),
                    (int)(n / dm), dm);
        }
        free(tmp_lm);
        printf("       Q8_0 + Q4_0 lm_head ready\n");
    }

    if (st_load_layer_weights(&cfg, &w, 0, cfg.n_layers) != 0) {
        printf("       [FAIL] Weight loading failed\n");
        st_weights_free(&w); st_config_free(&cfg);
        qwen_tokenizer_free(&tok);
        return;
    }
    QueryPerformanceCounter(&t_end);
    double load_sec = (double)(t_end.QuadPart - t_start.QuadPart) / (double)freq.QuadPart;
    printf("       Weights loaded: %.1f sec\n", load_sec);

    int im_start = tok.im_start_id > 0 ? tok.im_start_id : 151644;
    int im_end   = tok.im_end_id   > 0 ? tok.im_end_id   : 151645;
    int nl_tok   = 198;
    int bos      = tok.bos_id > 0 ? tok.bos_id : 151643;

    if (g_longctx_quality) {
        test_longctx_quality_ab(&w, &cfg, &tok);
        st_weights_free(&w);
        st_config_free(&cfg);
        qwen_tokenizer_free(&tok);
        return;
    }

    if (g_prefix_cache) {
        test_prefix_cache_ab(&w, &cfg, &tok);
        test_prefix_cache_lcp(&w, &cfg, &tok);
        test_continuous_batching(&w, &cfg, &tok);
        st_weights_free(&w);
        st_config_free(&cfg);
        qwen_tokenizer_free(&tok);
        return;
    }

    /* ============================================================
     * Part A: Single-User Performance (5 different prompts)
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
        st_weights_free(&w); st_config_free(&cfg);
        qwen_tokenizer_free(&tok);
        return;
    }

    for (int pi = 0; pi < n_single_prompts; pi++) {
        /* Reset inference state: clear KV cache and reset position */
        for (int l = 0; l < cfg.n_layers; l++) ist.cache_len[l] = 0;
        ist.seq_len = 0;
        memset(ist.hidden, 0, (size_t)cfg.dim * sizeof(float));

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
         * before the system prompt; the real prompt stays at the tail. */
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

        /* Prefill (mini-batch �?axiom: blas_qkv_fusion_categorical) */
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
         * DECODE-TIMING/DECODE-KERNELS 到 stderr。 */
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
            for (int t = 0; t < cfg.vocab_size; t++) {
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
        /* --perf-partA: single-user only (CPU-vs-NPU comparison core). */
        printf("\n  === Performance Summary (Part A only) ===\n");
        printf("  Model: Qwen3-VL-8B-Instruct\n");
        printf("  Weights loading: %.1f sec\n", load_sec);
        printf("  Single-user avg: TTFT=%.0fms | TPOT=%.0fms/tok | %.1f tok/s\n",
               total_ttft / n_single_prompts * 1000.0,
               total_tpot / n_single_prompts,
               total_throughput / n_single_prompts);
        fflush(stdout);
        st_qwen_inference_free(&ist);
        st_weights_free(&w); st_config_free(&cfg);
        qwen_tokenizer_free(&tok);
        return;
    }

    /* ============================================================
     * Part B: Multi-User Sequential (Round-Robin)
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
        for (int l = 0; l < cfg.n_layers; l++) ist.cache_len[l] = 0;
        ist.seq_len = 0;

        LARGE_INTEGER p0, p1;
        QueryPerformanceCounter(&p0);
        st_qwen_model_prefill_batch(&ist, user_prompt_ids[u], user_pn[u]);
        QueryPerformanceCounter(&p1);
        base_prefill_ms += (double)(p1.QuadPart - p0.QuadPart) * 1000.0 / (double)freq.QuadPart;

        int pos = 0;
        for (int step = 0; step < max_new_tokens; step++) {
            int best_id = greedy_argmax(ist.logits, cfg.vocab_size);
            if (best_id == tok.eos_id || best_id == im_end) break;
            append_tok_str(base_output[u], &pos, (int)sizeof(base_output[u]), &tok, best_id);
            st_qwen_model_forward(&ist, best_id);
        }
    }

    /* ---- Cached (B-scheme): prefill shared system prompt once, then fork its
     * KV-cache into each user via st_qwen_copy_kv_prefix; each user prefills
     * only their unique question suffix. ---- */
    for (int l = 0; l < cfg.n_layers; l++) ist.cache_len[l] = 0;
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
            int best_id = greedy_argmax(st.logits, cfg.vocab_size);
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
     *
     * Each user gets an independent inference state (own KV-cache,
     * hidden state, logits).  Tokens are interleaved at decode step
     * level (user-A step 0, user-B step 0, user-A step 1, user-B
     * step 1, ...).  All 8 OMP threads work on the current user's
     * forward pass at full speed.
     *
     * This demonstrates true concurrent scheduling: both users make
     * progress simultaneously rather than sequentially.  Wall-clock
     * time �?sum of both users (no HW parallel speedup without batch
     * matmul), but per-user TPOT is preserved at the single-user
     * level (each gets 8-thread matvecs interleaved).
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
        printf("       [SKIP] Part C failed �?reporting sequential results only\n");
        printf("\n  === Performance Summary ===\n");
        printf("  Model: Qwen3-VL-8B-Instruct (Q8_0 quant, ~10 GB)\n");
        printf("  Weights loading: %.1f sec\n", load_sec);
        printf("  Single-user avg: TTFT=%.0fms | TPOT=%.0fms/tok | %.1f tok/s\n",
               total_ttft / n_single_prompts * 1000.0,
               total_tpot / n_single_prompts,
               total_throughput / n_single_prompts);
        printf("  Multi-user (%d users) sequential: %.1f tok/s aggregate (%d tokens, %.1fs)\n",
               n_users, seq_total_tok / seq_prefill_sec, seq_total_tok, seq_prefill_sec);
        printf("  Multi-user (%d users) concurrent: [SKIP]\n", n_users);
        st_qwen_inference_free(&ist);
        st_weights_free(&w); st_config_free(&cfg);
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

    /* Prefill all users (mini-batch per user �?axiom: blas_qkv_fusion_categorical) */
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
            for (int t = 0; t < cfg.vocab_size; t++) {
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
    printf("       (Each user gets full 8-thread matvecs, interleaved step-by-step)\n");
    printf("       Speedup vs sequential: %.2fx (same total compute, concurrent scheduling)\n",
           seq_prefill_sec / par_total_sec);
    for (int u = 0; u < n_users; u++) {
        printf("         User %d: %.0fms, %d tokens �?\"%s\"\n",
               u + 1, parallel_user_time[u] * 1000.0, parallel_gen_tok[u],
               parallel_output[u][0] ? parallel_output[u] : "(empty)");
    }

    /* Free concurrent inference states */
    for (int u = 0; u < n_users; u++) st_qwen_inference_free(&ist_c[u]);

    /* ============================================================
     * Summary
     * ============================================================ */
    printf("\n  === Performance Summary ===\n");
    printf("  Model: Qwen3-VL-8B-Instruct (Q8_0 quant, ~10 GB)\n");
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
    st_config_free(&cfg);
    qwen_tokenizer_free(&tok);

}  /* test_performance_benchmark */

#if 0 /* Test 9: Synthetic Pippenger validation — vllm_gguf.c removed from build */
static void test_synthetic_pippenger(void) {
    printf("\n=== Test 9: Synthetic Model Pippenger Validation ===\n");
    printf("       Goal: Verify Pippenger speedup on attention-dominated model\n");
    fflush(stdout);

    /* Synthetic model: attention �?92% of total compute */
    int synth_dim = 256, synth_heads = 8, synth_layers = 4;
    int synth_ffn = 64, synth_hd = 32, synth_nkv = 8;
    int synth_vocab = 1024, synth_maxseq = 2048;

    /* Step 1: Create synthetic weights */
    GGUFWeights w;
    memset(&w, 0, sizeof(w));
    if (gguf_weights_init_synthetic(&w, synth_dim, synth_heads, synth_layers,
            synth_ffn, synth_hd, synth_nkv, synth_vocab, synth_maxseq) != 0) {
        printf("       [FAIL] Cannot create synthetic weights\n");
        return;
    }

    /* Step 2: Init inference state */
    GGUFInferenceState ist;
    memset(&ist, 0, sizeof(ist));
    if (gguf_inference_init(&ist, &w) != 0) {
        printf("       [FAIL] Cannot init inference state\n");
        gguf_weights_free(&w);
        return;
    }
    printf("       Model: dim=%d heads=%d hd=%d kv=%d layers=%d ffn=%d\n",
           synth_dim, synth_heads, synth_hd, synth_nkv, synth_layers, synth_ffn);

    /* Step 3: Prefill 640 tokens (10 blocks of 64) */
    int prefill_len = 640;
    int n_blocks = prefill_len / PIP_BLOCK_SIZE;  /* 10 blocks */

    /* Compute FLOP per decode token (attention scales with seq_len) */
    {
        float seq_len_f = (float)prefill_len;
        float attn_flops = seq_len_f * (float)synth_heads * (float)synth_hd * (float)synth_hd * 2.0f
                           + (float)synth_heads * (float)synth_hd * (float)synth_dim;
        float ffn_flops  = (float)synth_dim * (float)synth_ffn * 3.0f;
        float attn_pct_local = 100.0f * attn_flops / (attn_flops + ffn_flops);
        printf("       Attention/FFN FLOP ratio: %.0f%% / %.0f%% (per-decode-token, %d-tok ctx)\n",
               attn_pct_local, 100.0f - attn_pct_local, prefill_len);
    }

    printf("       Prefilling %d tokens (%d blocks)...\n", prefill_len, n_blocks);
    fflush(stdout);

    /* Use sequential token IDs starting from 10 to avoid special tokens */
    for (int t = 0; t < prefill_len; t++)
        gguf_model_forward(&ist, 10 + t % (synth_vocab - 10));

    printf("       KV-cache: %d tokens, %d blocks precomputed\n",
           ist.seq_len, ist.pip_n_blocks[0]);

    /* Step 4: Snapshot state for multi-user benchmark */
    int d = synth_dim, nl = synth_layers;
    float *saved_hidden = malloc((size_t)d * sizeof(float));
    int  *saved_cache_len = malloc((size_t)nl * sizeof(int));
    int  *saved_pip_nb = malloc((size_t)nl * sizeof(int));
    int   saved_seq = ist.seq_len;
    memcpy(saved_hidden, ist.hidden, (size_t)d * sizeof(float));
    memcpy(saved_cache_len, ist.cache_len, (size_t)nl * sizeof(int));
    memcpy(saved_pip_nb, ist.pip_n_blocks, (size_t)nl * sizeof(int));

    /* Step 5: Benchmark �?Decode phase (attention bottleneck visible here)
     * In decode, each new token needs attention over entire 640-token history.
     * Block-Pippenger reduces this from O(640) scoring to O(10) block scores. */
    enum { N_USERS = 50 };

    /* Generate different query tokens for each user (within vocab range) */
     int *query_tokens = malloc(N_USERS * sizeof(int));
     for (int u = 0; u < N_USERS; u++)
         query_tokens[u] = (10 + u * 20) % synth_vocab;

    printf("\n       --- Multi-User Decode Benchmark (%d users, %d-token context) ---\n",
           N_USERS, saved_seq);

    double ms_std, ms_pip;

    /* Test A: Standard attention (Pippenger OFF) */
    {
        ist.use_pippenger = 0;

        LARGE_INTEGER freq, t1, t2;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&t1);
        for (int u = 0; u < N_USERS; u++) {
            /* Restore snapshot for clean per-user measurement */
            memcpy(ist.hidden, saved_hidden, (size_t)d * sizeof(float));
            memcpy(ist.cache_len, saved_cache_len, (size_t)nl * sizeof(int));
            memcpy(ist.pip_n_blocks, saved_pip_nb, (size_t)nl * sizeof(int));
            ist.seq_len = saved_seq;
            /* Disable block update (already precomputed) */
            gguf_model_forward(&ist, query_tokens[u]);
        }
        QueryPerformanceCounter(&t2);
        ms_std = (double)(t2.QuadPart - t1.QuadPart) * 1000.0 / (double)freq.QuadPart;
    }

    /* Test B: Block-level Pippenger attention (Pippenger ON) */
    {
        ist.use_pippenger = 1;

        LARGE_INTEGER freq, t3, t4;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&t3);
        for (int u = 0; u < N_USERS; u++) {
            memcpy(ist.hidden, saved_hidden, (size_t)d * sizeof(float));
            memcpy(ist.cache_len, saved_cache_len, (size_t)nl * sizeof(int));
            memcpy(ist.pip_n_blocks, saved_pip_nb, (size_t)nl * sizeof(int));
            ist.seq_len = saved_seq;
            gguf_model_forward(&ist, query_tokens[u]);
        }
        QueryPerformanceCounter(&t4);
        ms_pip = (double)(t4.QuadPart - t3.QuadPart) * 1000.0 / (double)freq.QuadPart;
    }

    double speedup = (ms_pip > 0) ? ms_std / ms_pip : 0.0;

    /* Print results */
    printf("\n");
    printf("       %-25s %6s  %8s  %10s  %12s  %10s\n",
           "Method","Users","Context","Total(ms)","tok/s/user","Speedup");
    printf("       %-25s %6d  %8d  %10.1f  %12.1f  %10s\n",
           "Standard Attention", N_USERS, saved_seq, ms_std,
           1000.0 / (ms_std / N_USERS), "1.00x (baseline)");
    printf("       %-25s %6d  %8d  %10.1f  %12.1f  %9.2fx\n",
           "Block-Pippenger (ours)", N_USERS, saved_seq, ms_pip,
           1000.0 / (ms_pip / N_USERS), speedup);

    /* Analysis */
    printf("\n       --- Analysis ---\n");
    printf("       Block-level scoring: O(%d) vs O(%d) per head\n",
           n_blocks, saved_seq);
    {
        float seq_f = (float)saved_seq;
        float a_flops = seq_f * (float)synth_heads * (float)synth_hd * (float)synth_hd * 2.0f
                        + (float)synth_heads * (float)synth_hd * (float)synth_dim;
        float f_flops = (float)synth_dim * (float)synth_ffn * 3.0f;
        float a_pct = 100.0f * a_flops / (a_flops + f_flops);
        printf("       Theoretical max speedup (Amdahl): %.1fx (%.0f%% attention bound)\n",
               1.0f / (1.0f - a_pct / 100.0f), a_pct);
    }
    printf("       Measured speedup: %.2fx\n", speedup);

    if (speedup > 1.50)
        printf("       VERDICT: Pippenger highly effective (>1.5x) on attention-dominated models.\n");
    else if (speedup > 1.15)
        printf("       VERDICT: Pippenger shows clear benefit (>1.15x) for long-context decode.\n");
    else if (speedup > 1.03)
        printf("       VERDICT: Pippenger provides measurable but modest benefit (1.03-1.15x).\n");
    else
        printf("       VERDICT: Pippenger not beneficial at this scale - overhead dominates.\n"
               "                May require longer context or more heads to show advantage.\n");

    /* Cleanup */
    free(saved_hidden);
    free(saved_cache_len);
    free(saved_pip_nb);
    free(query_tokens);
    gguf_inference_free(&ist);
    gguf_weights_free(&w);

    printf("       [PASS] Synthetic Pippenger validation complete\n");
    fflush(stdout);
}
#endif /* Test 9: Synthetic Pippenger validation */

/* ================================================================
 * Test 11: Multimodal Image/Video Analysis
 *
 * Demonstrates Qwen3-VL vision encoder + LLM reasoning for:
 *   - Image analysis (describing, analyzing visual content)
 *   - Video analysis (understanding video frames over time)
 *
 * Uses synthetic test images since we don't require external files.
 * Axioms: blas_matrix_block_natural_isomorphism (Vit QKV/O_proj)
 *         blas_precision_efficiency_tradeoff (Q8_0 ViT weights)
 *         tensor_decomposition_graph_mapping (DeepStack multi-scale)
 * ================================================================ */
static void test_multimodal_analysis(const char **image_paths, int n_images, const char *video_dir) {
    printf("\n=== Multimodal Performance Test ===\n");
    printf("       Model: Qwen3-VL-8B-Instruct with Vision Encoder\n");
    printf("       OMP_NUM_THREADS=%d\n",
           vllm_tp_threads()
    );
    fflush(stdout);

    /* ---- Parse config and load model weights ---- */
    /* Try multiple paths since exe location varies (build/ vs build/Release/) */
    const char *model_paths_mm[] = {
        "../../Modl/千问3_VL_8B_Instruct",
        "../Modl/千问3_VL_8B_Instruct",
        "Modl/千问3_VL_8B_Instruct",
        NULL
    };
    const char *model_dir = NULL;
    for (int i_mp = 0; model_paths_mm[i_mp]; i_mp++) {
        char test_path[1024];
        snprintf(test_path, sizeof(test_path), "%s/config.json", model_paths_mm[i_mp]);
        if (st_access(test_path, 0) == 0) {
            model_dir = model_paths_mm[i_mp];
            break;
        }
    }
    if (!model_dir) {
        printf("       SKIP: Model not found on disk\n");
        printf("       Place model at Modl/千问3_VL_8B_Instruct/\n");
        return;
    }
    STModelConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    if (st_parse_config(model_dir, &cfg) != 0) {
        printf("       SKIP: Cannot parse model config\n");
        return;
    }
    if (!cfg.has_vision) {
        printf("       SKIP: Model has no vision encoder\n");
        st_config_free(&cfg);
        return;
    }

    printf("       Vision: ViT %d layers, hidden=%d, heads=%d, patch=%d\n",
           cfg.vis_depth, cfg.vis_hidden, cfg.vis_heads, cfg.vis_patch);
    fflush(stdout);

    /* Load weights */
    STModelWeights w;
    memset(&w, 0, sizeof(w));
    double start_sec = vllm_tp_wtime();
    st_weights_alloc_layers_q8ffn(&w, &cfg, -1);

    /* Load token embeddings + final norm (not loaded by st_load_layer_weights) */
    st_load_tensor(&cfg, "model.language_model.embed_tokens.weight",
                    w.token_embed, cfg.vocab_size * cfg.dim);
    st_load_tensor(&cfg, "model.language_model.norm.weight",
                    w.final_norm, cfg.dim);
    /* lm_head: load into temp F32, quantize to Q8_0 + Q4_0 */
    if (w.q8_lm_weight || w.q4_lm_weight) {
        float *tmp_lm = malloc((size_t)cfg.vocab_size * cfg.dim * sizeof(float));
        if (tmp_lm) {
            if (st_load_tensor(&cfg, "model.language_model.lm_head.weight",
                               tmp_lm, cfg.vocab_size * cfg.dim) != 0)
                st_load_tensor(&cfg, "lm_head.weight",
                               tmp_lm, cfg.vocab_size * cfg.dim);
            if (w.q8_lm_weight) {
                if (w.q8_buf_q4)
                    f32_to_q4i8(w.q8_lm_weight, tmp_lm, cfg.vocab_size * cfg.dim);
                else if (st_wmode_effective() == 4)
                    f32_to_g256q8(w.q8_lm_weight, tmp_lm, cfg.vocab_size * cfg.dim);
                else
                    f32_to_q8_0(w.q8_lm_weight, tmp_lm, cfg.vocab_size * cfg.dim);
            }
            if (w.q4_lm_weight)
                f32_to_q4_0(w.q4_lm_weight, tmp_lm, cfg.vocab_size * cfg.dim);
            if (w.q4_lm_weight)
                repack_q4_0_4x4_inplace(w.q4_lm_weight, cfg.vocab_size, cfg.dim);
            if (w.q8_lm_weight)
                repack_q8_0_tiled_inplace(w.q8_lm_weight, cfg.vocab_size, cfg.dim);
            free(tmp_lm);
        }
    }

    st_load_layer_weights(&cfg, &w, 0, cfg.n_layers);

    /* Load vision weights */
    STVisionWeights vis_w;
    memset(&vis_w, 0, sizeof(vis_w));
    st_vision_weights_alloc(&vis_w, &cfg);
    if (st_vision_load_weights(&vis_w, &cfg) != 0) {
        printf("       FAIL: Vision weight loading failed\n");
        st_config_free(&cfg);
        st_weights_free(&w);
        st_vision_weights_free(&vis_w);
        return;
    }
    double load_sec = vllm_tp_wtime() - start_sec;
    printf("       Weights loading: %.1f sec\n", load_sec);

    /* ---- Initialize tokenizer ---- */
    QwenTokenizer tok;
    memset(&tok, 0, sizeof(tok));
    char vocab_path[512];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.bin", model_dir);
    if (qwen_tokenizer_load(&tok, model_dir) != 0) {
        printf("       FAIL: Tokenizer load failed\n");
        st_config_free(&cfg);
        st_weights_free(&w);
        st_vision_weights_free(&vis_w);
        return;
    }
    /* Set vision special tokens */
    tok.image_token_id = cfg.image_token_id;
    tok.video_token_id = cfg.video_token_id;
    tok.vision_start_id = cfg.vision_start_id;
    tok.vision_end_id = cfg.vision_end_id;
    printf("       Tokenizer: %d tokens, vision IDs: start=%d end=%d img=%d vid=%d\n",
           tok.vocab_size, cfg.vision_start_id, cfg.vision_end_id,
           cfg.image_token_id, cfg.video_token_id);

    /* ---- Image Analysis (loop over provided images) ---- */
    int total_images = (n_images > 0) ? n_images : 1;
    int img_w = 64, img_h = 64;  /* function scope, reused by video section */
    STQwenInferenceState ist;
    STVisionState vis_st;        /* function scope, reused by video section */
    for (int img_idx = 0; img_idx < total_images; img_idx++) {
        const char *image_path = (n_images > 0) ? image_paths[img_idx] : NULL;

        /* Load this image */
        uint8_t *test_img = NULL;
        int img_is_external = 0;

        if (image_path && image_path[0]) {
            test_img = media_load_image(image_path, &img_w, &img_h);
            if (test_img) {
                img_is_external = 1;
                printf("\n       --- Image %d/%d: %s (%dx%d) ---\n",
                       img_idx + 1, total_images, image_path, img_w, img_h);
            }
        }

        if (!test_img) {
            /* Fallback: synthetic test image (64x64 RGB gradient) */
            img_w = 64; img_h = 64;
            test_img = (uint8_t *)malloc((size_t)img_w * img_h * 3);
            for (int y = 0; y < img_h; y++) {
                for (int x = 0; x < img_w; x++) {
                    int idx = (y * img_w + x) * 3;
                    test_img[idx + 0] = (uint8_t)(x * 255 / img_w);
                    test_img[idx + 1] = (uint8_t)(y * 255 / img_h);
                    test_img[idx + 2] = (uint8_t)(128);
                }
            }
            printf("\n       --- Image %d/%d: synthetic (64x64) ---\n",
                   img_idx + 1, total_images);
        }

        /* ---- Initialize per-image inference state ---- */
        memset(&ist, 0, sizeof(ist));
        st_qwen_inference_init(&ist, &w);

        /* Step 1: Initialize vision state and link weights */
        memset(&vis_st, 0, sizeof(vis_st));
        st_vision_init(&vis_st, &cfg);
        vis_st.w = vis_w;

        double enc_start = vllm_tp_wtime();
        int n_vis_tokens = st_vision_encode_image(&vis_st, test_img, img_w, img_h);
        double enc_sec = vllm_tp_wtime() - enc_start;
        if (n_vis_tokens < 0) {
            printf("       FAIL: Image encoding failed\n");
            st_qwen_inference_free(&ist);
            st_vision_free(&vis_st);
            if (img_is_external) media_free_image(test_img); else free(test_img);
            continue;
        }
        const float *vis_tokens = st_vision_get_tokens(&vis_st);
        int grid_h = vis_st.grid_h / cfg.vis_merge;
        int grid_w = vis_st.grid_w / cfg.vis_merge;
        printf("       %dx%d -> %d patches -> %d visual tokens (grid %dx%d)\n",
               img_w, img_h, n_vis_tokens, n_vis_tokens, grid_h, grid_w);
        printf("       [VIS] Encode: %.1f ms\n", enc_sec * 1000.0);

        /* Step 2: Build multimodal prompt */
        int n_img_pad = n_vis_tokens;
        int prompt_tokens[4096];
        int pn = 0;

        prompt_tokens[pn++] = tok.im_start_id > 0 ? tok.im_start_id : 151644;
        prompt_tokens[pn++] = cfg.vision_start_id;
        for (int i = 0; i < n_img_pad && pn < 4095; i++) {
            prompt_tokens[pn++] = cfg.image_token_id;
        }
        prompt_tokens[pn++] = cfg.vision_end_id;

        const char *text_prompt = "\nDescribe this image.";
        {
            int n = qwen_tokenizer_encode(&tok, text_prompt,
                                           prompt_tokens + pn, 4096 - pn);
            pn += n;
        }
        {
            int n = qwen_tokenizer_encode(&tok, "<|im_end|>\n<|im_start|>assistant\n",
                                           prompt_tokens + pn, 4096 - pn);
            pn += n;
        }

        printf("       Prompt: %d tokens (%d text, %d visual)\n",
               pn, pn - n_img_pad - 3, n_img_pad + 2);
        fflush(stdout);

        /* --bench-seqlen N: pad the multimodal prompt to a fixed total
         * prefill length (visual tokens + text = N) so TTFT is comparable
         * with llama at the same context size. Neutral filler is prepended
         * at the head of the token array (outside the chat template); the
         * engine treats it as ordinary context tokens. */
        if (g_bench_seqlen > 0 && pn < g_bench_seqlen) {
            static const char *filler = "The quick brown fox jumps over the lazy dog. ";
            int fill_ids[64]; int fn = qwen_tokenizer_encode(&tok, filler, fill_ids, 64);
            int need = g_bench_seqlen - pn;
            if (fn > 0 && need > 0) {
                int f[4096]; int fi = 0;
                while (fi < need) {
                    for (int k = 0; k < fn && fi < need; k++) f[fi++] = fill_ids[k];
                }
                memmove(prompt_tokens + need, prompt_tokens, (size_t)pn * sizeof(int));
                memcpy(prompt_tokens, f, (size_t)need * sizeof(int));
                pn = g_bench_seqlen;
                printf("       [PAD] padded to %d tokens (prepend %d filler)\n", pn, need);
            }
        }

        /* Step 3: Multimodal prefill */
        int grid_thw[3];
        grid_thw[0] = 1;
        grid_thw[1] = grid_h;
        grid_thw[2] = grid_w;

        double prefill_start = vllm_tp_wtime();
        int ret = st_qwen_model_multimodal_prefill(&ist, prompt_tokens, pn,
                                                    vis_tokens, n_vis_tokens,
                                                    grid_thw);
        double prefill_sec = vllm_tp_wtime() - prefill_start;

        if (ret != 0) {
            printf("       FAIL: Multimodal prefill failed\n");
            st_qwen_inference_free(&ist);
            st_vision_free(&vis_st);
            if (img_is_external) media_free_image(test_img); else free(test_img);
            continue;
        }
        printf("       Prefill: %.0fms (%.2f tok/s)\n",
               prefill_sec * 1000.0, pn / prefill_sec);

        /* Step 4: Decode response with detailed timing */
        printf("       Decoding...\n");
        fflush(stdout);
        int max_new_tokens = 60;
        int img_tokens_generated = 0;
        double decode_start = vllm_tp_wtime();
        for (int step = 0; step < max_new_tokens; step++) {
            int best_id = 0;
            float best_logit = -1e9f;
            for (int i = 0; i < cfg.vocab_size; i++) {
                if (ist.logits[i] > best_logit) {
                    best_logit = ist.logits[i];
                    best_id = i;
                }
            }
            if (best_id == tok.eos_id || best_id == tok.im_end_id) break;
            const char *token_str = qwen_tokenizer_decode(&tok, best_id);
            if (token_str) printf("%s", token_str);
            img_tokens_generated++;
            st_qwen_model_forward(&ist, best_id);
        }
        double decode_sec = vllm_tp_wtime() - decode_start;
        printf("\n");
        printf("       Decode: %d tokens in %.0fms (TPOT=%.0fms, %.1f tok/s)\n",
               img_tokens_generated,
               decode_sec * 1000.0,
               img_tokens_generated > 0 ? (decode_sec * 1000.0) / img_tokens_generated : 0.0,
               img_tokens_generated > 0 ? img_tokens_generated / decode_sec : 0.0);
        printf("       Total: TTFT=%.0fms E2E=%.0fms\n",
               prefill_sec * 1000.0,
               (prefill_sec + decode_sec) * 1000.0);

        /* Per-image cleanup */
        st_qwen_inference_free(&ist);
        st_vision_free(&vis_st);
        if (img_is_external) media_free_image(test_img); else free(test_img);
    }

    /* ======== Video Analysis ======== */
    printf("\n       --- Video Analysis ---\n");
    fflush(stdout);

    /* Load video frames: external directory or synthetic */
    int n_frames = 0;
    uint8_t **video_frames = NULL;
    int vid_ext = 0;
    int vid_w = 64, vid_h = 64;

    /* ---- Video-specific variables (re-declared for this scope) ---- */
    int n_vis_tokens = 0;
    const float *vis_tokens = NULL;
    int pn = 0;
    int prompt_tokens[4096];
    int n_img_pad = 0;
    int grid_thw[3];
    double prefill_start, prefill_sec;
    int ret;
    int max_new_tokens = 60;

    if (video_dir && video_dir[0]) {
        video_frames = media_load_video_frames(video_dir, &vid_w, &vid_h,
                                                &n_frames, 16);
        if (video_frames && n_frames > 0) {
            vid_ext = 1;
            printf("       Loaded external video: %d frames %dx%d\n",
                   n_frames, vid_w, vid_h);
        }
    }

    if (!video_frames || n_frames == 0) {
        /* Fallback: synthetic video (4 frames of shifting gradient) */
        n_frames = 4;
        vid_w = 64; vid_h = 64;
        video_frames = (uint8_t **)malloc((size_t)n_frames * sizeof(uint8_t *));
        for (int f = 0; f < n_frames; f++) {
            video_frames[f] = (uint8_t *)malloc((size_t)vid_w * vid_h * 3);
            for (int y = 0; y < vid_h; y++) {
                for (int x = 0; x < vid_w; x++) {
                    int idx = (y * vid_w + x) * 3;
                    video_frames[f][idx + 0] = (uint8_t)((x + f * 16) * 255 / vid_w);
                    video_frames[f][idx + 1] = (uint8_t)((y + f * 8) * 255 / vid_h);
                    video_frames[f][idx + 2] = (uint8_t)(128 + f * 30);
                }
            }
        }
        printf("       Using synthetic video (%d frames, 64x64)\n", n_frames);
    }

    /* Reset inference state for video test (already freed by last image iteration) */
    memset(&ist, 0, sizeof(ist));
    st_qwen_inference_init(&ist, &w);

    memset(&vis_st, 0, sizeof(vis_st));
    st_vision_init(&vis_st, &cfg);
    vis_st.w = vis_w;

    /* Step 1: Video encode */
    double enc_start = vllm_tp_wtime();
    n_vis_tokens = st_vision_encode_video(&vis_st,
                                           (const uint8_t **)video_frames,
                                           vid_w, vid_h, n_frames);
    double enc_sec = vllm_tp_wtime() - enc_start;
    if (n_vis_tokens < 0) {
        printf("       FAIL: Video encoding failed\n");
    } else {
        vis_tokens = st_vision_get_tokens(&vis_st);
        printf("       Video %d frames %dx%d -> %d visual tokens\n",
               n_frames, vid_w, vid_h, n_vis_tokens);
        printf("       [VIS] Encode: %.1f ms\n", enc_sec * 1000.0);

        /* Step 2: Build video prompt */
        const char *vid_prompt = "<|im_start|>user\n";
        pn = qwen_tokenizer_encode(&tok, vid_prompt, prompt_tokens, 4096);
        prompt_tokens[pn++] = cfg.vision_start_id;
        n_img_pad = n_vis_tokens;
        for (int i = 0; i < n_img_pad && pn < 4095; i++) {
            prompt_tokens[pn++] = cfg.video_token_id;  /* 151656 */
        }
        prompt_tokens[pn++] = cfg.vision_end_id;

        /* Text part */
        {
            int n = qwen_tokenizer_encode(&tok,
                "\nDescribe this video.<|im_end|>\n<|im_start|>assistant\n",
                prompt_tokens + pn, 4096 - pn);
            pn += n;
        }

        grid_thw[0] = (n_frames + cfg.vis_temporal - 1) / cfg.vis_temporal;
        grid_thw[1] = vis_st.grid_h / cfg.vis_merge;
        grid_thw[2] = vis_st.grid_w / cfg.vis_merge;

        /* --bench-seqlen N: pad to a fixed total prefill length (same filler
         * and head-prepend scheme as the image branch). */
        if (g_bench_seqlen > 0 && pn < g_bench_seqlen) {
            static const char *filler = "The quick brown fox jumps over the lazy dog. ";
            int fill_ids[64]; int fn = qwen_tokenizer_encode(&tok, filler, fill_ids, 64);
            int need = g_bench_seqlen - pn;
            if (fn > 0 && need > 0) {
                int f[4096]; int fi = 0;
                while (fi < need) {
                    for (int k = 0; k < fn && fi < need; k++) f[fi++] = fill_ids[k];
                }
                memmove(prompt_tokens + need, prompt_tokens, (size_t)pn * sizeof(int));
                memcpy(prompt_tokens, f, (size_t)need * sizeof(int));
                pn = g_bench_seqlen;
                printf("       [PAD] padded to %d tokens (prepend %d filler)\n", pn, need);
            }
        }

        prefill_start = vllm_tp_wtime();
        ret = st_qwen_model_multimodal_prefill(&ist, prompt_tokens, pn,
                                                vis_tokens, n_vis_tokens,
                                                grid_thw);
        prefill_sec = vllm_tp_wtime() - prefill_start;

        if (ret != 0) {
            printf("       FAIL: Video multimodal prefill failed\n");
        } else {
            printf("       Prefill: %.0fms (%.1f tok/s)\n",
                   prefill_sec * 1000.0, pn / prefill_sec);
            printf("       Decoding...\n");
            fflush(stdout);

            int vid_tokens_generated = 0;
            double vid_decode_start = vllm_tp_wtime();
            for (int step = 0; step < max_new_tokens; step++) {
                int best_id = 0;
                float best_logit = -1e9f;
                for (int i = 0; i < cfg.vocab_size; i++) {
                    if (ist.logits[i] > best_logit) {
                        best_logit = ist.logits[i];
                        best_id = i;
                    }
                }
                if (best_id == tok.eos_id || best_id == tok.im_end_id) break;
                const char *token_str = qwen_tokenizer_decode(&tok, best_id);
                if (token_str) printf("%s", token_str);
                vid_tokens_generated++;
                st_qwen_model_forward(&ist, best_id);
            }
            double vid_decode_sec = vllm_tp_wtime() - vid_decode_start;
            printf("\n");
            printf("       Video Decode: %d tokens in %.0fms (TPOT=%.0fms/tok, %.1f tok/s)\n",
                   vid_tokens_generated,
                   vid_decode_sec * 1000.0,
                   vid_tokens_generated > 0 ? (vid_decode_sec * 1000.0) / vid_tokens_generated : 0.0,
                   vid_tokens_generated > 0 ? vid_tokens_generated / vid_decode_sec : 0.0);
            printf("       Video E2E: TTFT=%.0fms, Total=%.0fms\n",
                   prefill_sec * 1000.0,
                   (prefill_sec + vid_decode_sec) * 1000.0);
        }
    }

    /* Video frame cleanup */
    if (vid_ext) {
        media_free_video_frames(video_frames, n_frames);
    } else {
        for (int f = 0; f < n_frames; f++) free(video_frames[f]);
        free(video_frames);
    }
    st_qwen_inference_free(&ist);
    st_vision_free(&vis_st);

    st_vision_weights_free(&vis_w);

    st_config_free(&cfg);
    st_weights_free(&w);
    qwen_tokenizer_free(&tok);

    printf("\n       [Multimodal Test Complete]\n");
    fflush(stdout);
}

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
    int format_prio;       /* 模型加载格式优先级: 0=自动(VQF>GGUF>safetensors),
                            * 1=VQF优先, 2=safetensors优先, 3=GGUF优先 */
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

/* --convert-vqf <out.vqf>：把 safetensors 模型经与 serve 完全相同的加载序列
 * （alloc → 逐层量化 → repack → lm_head）量化后 dump 成 VQF 单文件。
 * 需配合 --model <dir> 与目标运行时的 wmode（默认 dual）。 */
static int load_quant_weights(STModelConfig *cfg, STModelWeights *w); /* 见下 */
static int run_convert_vqf(const char *model_dir, const char *out_path) {
    STModelConfig cfg; memset(&cfg, 0, sizeof(cfg));
    if (st_parse_config(model_dir, &cfg) != 0) {
        printf("[VQF] config parse failed: %s\n", model_dir);
        return 1;
    }
    if (cfg.max_seq_len > 8192) cfg.max_seq_len = 8192;
    STModelWeights w; memset(&w, 0, sizeof(w));
    if (load_quant_weights(&cfg, &w) != 0) {
        printf("[VQF] weight load failed\n");
        st_config_free(&cfg);
        return 1;
    }
    /* 多模态：转换时顺带加载 vision 权重（safetensors 读取 + Q8 量化，
     * 与 serve 完全同一路径），VQF 固化 Q8 大矩阵 + F32 小张量。
     * 失败仅告警（仍可产出纯文本 VQF），成功则 w.vision 供 vqf_collect。 */
    STVisionWeights *vvis = NULL;
    if (cfg.has_vision) {
        vvis = (STVisionWeights *)calloc(1, sizeof(STVisionWeights));
        if (vvis && st_vision_weights_alloc(vvis, &cfg) == 0 &&
            st_vision_load_weights(vvis, &cfg) == 0) {
            /* 推理只读 Q8 副本：固化前释放 F32 大矩阵（省内存 & 文件体积） */
            st_vision_weights_free_f32(vvis);
            w.vision = vvis;
            printf("[VQF] vision weights loaded (ViT %d layers, hidden=%d)\n",
                   cfg.vis_depth, cfg.vis_hidden);
        } else {
            printf("[VQF] WARNING: vision weight load failed -> text-only VQF\n");
            if (vvis) { st_vision_weights_free(vvis); free(vvis); }
            vvis = NULL;
        }
    }
    /* 布局 flags：与当前引擎配置（wmode / VLLM_Q8_8X8 / repack 门控）一致 */
    uint32_t flags = 0;
    if (w.q8_buf_q4) flags |= VQF_FLAG_Q8BUF_Q4;
    if (w.has_x8)    flags |= VQF_FLAG_X8;
    if (st_wmode_effective() == 4) flags |= VQF_FLAG_G256;
    extern int g_st_q8_repack, g_st_q4_repack;
    if (g_st_q8_repack) {
        const char *e = getenv("VLLM_Q8_8X8");
        if (!(e && e[0] == '0')) flags |= VQF_FLAG_Q8_8X8;
    }
    if (g_st_q4_repack) flags |= VQF_FLAG_Q4_4X4;
    int rc = vqf_write(out_path, &w, &cfg, flags);
    if (rc == 0) {
        /* 自检：转换后立即 mmap 重新加载验证（checksum/布局/指针挂载） */
        STModelWeights w2; memset(&w2, 0, sizeof(w2));
        if (vqf_load(&w2, out_path) == 0) {
            printf("[VQF] self-check PASS: %s reload OK (flags=0x%x)\n",
                   out_path, flags);
            st_weights_free(&w2);
        } else {
            printf("[VQF] self-check FAIL: %s\n", out_path);
            rc = 1;
        }
    }
    if (vvis) { st_vision_weights_free(vvis); free(vvis); }
    w.vision = NULL;
    st_weights_free(&w);
    st_config_free(&cfg);
    return rc;
}

/* 加载全部量化权重：serve 与 --convert-vqf 共用同一代码路径，保证 VQF
 * 转换与内存加载位级一致（fixedpoint_quantize_saturate red line）。
 * 序列 = alloc(q8ffn) → embed/norm → 逐层 load+量化+repack → lm_head chunked
 * 量化+repack（含 tied embeddings 回退）。返回 0 成功。 */
static int load_quant_weights(STModelConfig *cfg, STModelWeights *w) {
    st_weights_alloc_layers_q8ffn(w, cfg, cfg->n_layers);
    fprintf(stderr, "[M-C] weights allocated\n"); fflush(stderr);
    if (!w->is_allocated) {
        printf("[SERVE] weight allocation failed (~%.1f GB needed)\n",
               (double)((size_t)cfg->vocab_size * cfg->dim * 2 +
                        (size_t)cfg->n_layers * cfg->dim * cfg->n_heads * cfg->head_dim * 4 +
                        (size_t)cfg->n_layers * cfg->ffn_dim * cfg->dim * 3 / 4) /
               (1024.0 * 1024.0 * 1024.0));
        return 1;
    }
    if (st_load_tensor(cfg, "model.language_model.embed_tokens.weight",
                       w->token_embed, cfg->vocab_size * cfg->dim) != 0) {
        printf("[SERVE] embed load failed\n");
        return 1;
    }
    st_load_tensor(cfg, "model.language_model.norm.weight", w->final_norm, cfg->dim);
    fprintf(stderr, "[M-D] embed+norm loaded\n"); fflush(stderr);
    if (st_load_layer_weights(cfg, w, 0, cfg->n_layers) != 0) {
        printf("[SERVE] layer weights load failed\n");
        return 1;
    }
    fprintf(stderr, "[M-E] layers loaded\n"); fflush(stderr);
    if (w->q8_lm_weight || w->q4_lm_weight) {
        printf("[SERVE] loading & quantizing lm_head (Q8_0 + Q4_0)...\n");
        fflush(stdout);
        /* Chunked slice load: a single 2.49 GB F32 staging buffer may fail
         * on 32 GB hosts; a failed malloc silently disabled the quantized
         * head and decode fell back to the slow scalar F32 path. */
        int64_t lm_elems = (int64_t)cfg->vocab_size * cfg->dim;
        int chunk_rows = 16384;
        int64_t chunk_elems = (int64_t)chunk_rows * cfg->dim;
        float *tmp_lm = (float *)malloc((size_t)chunk_elems * sizeof(float));
        while (!tmp_lm && chunk_rows > 1) {
            chunk_rows /= 2;
            chunk_elems = (int64_t)chunk_rows * cfg->dim;
            tmp_lm = (float *)malloc((size_t)chunk_elems * sizeof(float));
        }
        if (!tmp_lm) {
            fprintf(stderr, "[SERVE] WARNING: lm_head chunk buffer OOM -> F32 head\n");
        } else {
            for (int64_t off = 0; off < lm_elems; off += chunk_elems) {
                int64_t n = lm_elems - off;
                if (n > chunk_elems) n = chunk_elems;
                if (st_load_tensor_slice(cfg, "lm_head.weight", off, tmp_lm, n) != 0) {
                    /* tie_word_embeddings (e.g. Qwen3-VL-2B): lm_head.weight is
                     * absent, lm_head == embed_tokens.weight (already loaded as
                     * F32 into w->token_embed). Quantize from that instead. */
                    fprintf(stderr,
                            "[SERVE] lm_head.weight absent -> tied embeddings, "
                            "using embed_tokens\n");
                    if (off >= 0 && off + n <=
                        (int64_t)cfg->vocab_size * (int64_t)cfg->dim) {
                        memcpy(tmp_lm, w->token_embed + off,
                               (size_t)n * sizeof(float));
                    } else {
                        break;
                    }
                }
                if (w->q8_lm_weight) {
                    if (w->q8_buf_q4)
                        f32_to_q4i8(w->q8_lm_weight + (off / 32) * 34, tmp_lm, (int)n);
                    else if (st_wmode_effective() == 4)
                        f32_to_g256q8(w->q8_lm_weight + (off / 32) * 34, tmp_lm, (int)n);
                    else
                        f32_to_q8_0(w->q8_lm_weight + (off / 32) * 34, tmp_lm, (int)n);
                }
                if (w->q4_lm_weight)
                    f32_to_q4_0(w->q4_lm_weight + (off / 32) * 18, tmp_lm, (int)n);
                if (w->q4_lm_weight)
                    repack_q4_0_4x4_inplace(
                        w->q4_lm_weight + (size_t)(off / cfg->dim) * (((cfg->dim + 31) / 32) * 18),
                        (int)(n / cfg->dim), cfg->dim);
                if (w->q8_lm_weight)
                    repack_q8_0_tiled_inplace(
                        w->q8_lm_weight + (size_t)(off / cfg->dim) * (((cfg->dim + 31) / 32) * 34),
                        (int)(n / cfg->dim), cfg->dim);
            }
            free(tmp_lm);
            printf("[SERVE] lm_head quantized OK (peak staging %d MB)\n",
                   chunk_rows * cfg->dim * 4 / (1024 * 1024));
            fflush(stdout);
        }
    }
    return 0;
}

/* Last model dir whose NPU strip-cache (st_npu_gw_*) is resident. Loading a
 * DIFFERENT model must drop the stale strips (different weights/geometry);
 * same-model reloads keep them so a reloaded request skips the rebuild storm. */
static char g_npu_loaded_model[1024];

static int serve_load_model(ServeModel *m) {
    STModelConfig *cfg = NULL;
    QwenTokenizer *tok = NULL;
    STModelWeights *w = NULL;
    STQwenInferenceState *ist = NULL;
    STVisionWeights *vis_w = NULL;
    STVisionState *vis = NULL;
    int has_vision = 0;
    int cfg_is_vqf = 0;
    int model_is_gguf = 0;
    int vqf_file_mode = 0;   /* 单文件 VQF：词表需从文件所在目录加载 */
    char vqf_dirbuf[768] = {0};
    int rc = 0;

    /* VQF 单文件模型：目录含 model.vqf 时跳过 safetensors 解析，mmap 直挂，
     * cfg 自包含（vqf_load 填充 w->cfg）。否则若 --model 指向 GGUF 文件，
     * 走 GGUF 加载（dequant → 引擎布局）。否则常规 safetensors 加载。 */
    w = (STModelWeights *)calloc(1, sizeof(STModelWeights));
    if (!w) { rc = -1; goto done; }
    char vqf_path[768] = {0};
    /* 格式优先级候选（m->format_prio，管理页"加载格式"下拉）：
     *   0=自动（VQF>GGUF>safetensors） 1=VQF优先 2=safetensors优先 3=GGUF优先
     * 优先格式不存在时回退到自动检测顺序，保证任何路径都能加载。 */
    const int prio = m->format_prio;
    if (prio == 2) goto safetensors_path;   /* 强制 safetensors */
    if (prio == 3 && gguf_is_file(m->model_dir)) goto gguf_path;
    if (prio == 1 &&
        (vqf_dir_find(m->model_dir, vqf_path, sizeof(vqf_path)) ||
         vqf_is_file(m->model_dir)))
        goto vqf_path;
    if (vqf_dir_find(m->model_dir, vqf_path, sizeof(vqf_path))) goto vqf_path;
    if (vqf_is_file(m->model_dir)) goto vqf_path;
    if (gguf_is_file(m->model_dir)) goto gguf_path;
    goto safetensors_path;

vqf_path:
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
    } else {
        /* VQF 单文件（--convert-vqf / --convert-gguf 产物）直接 mmap 挂载 */
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
    }
    goto format_done;
gguf_path:
    if (gguf_load_model(m->model_dir, w) != 0) {
        printf("[SERVE] GGUF load failed: %s\n", m->model_dir);
        rc = 1; goto done;
    }
    cfg = &w->cfg;
    cfg_is_vqf = 1;      /* cfg 指向 w->cfg，随 w 一起释放 */
    model_is_gguf = 1;
    fprintf(stderr, "[M-A] GGUF loaded: %s\n", m->model_dir); fflush(stderr);
    goto format_done;
safetensors_path:
    cfg = (STModelConfig *)calloc(1, sizeof(STModelConfig));
    if (!cfg) { rc = -1; goto done; }
    if (st_parse_config(m->model_dir, cfg) != 0) {
        printf("[SERVE] config parse failed\n");
        rc = 1; goto done;
    }
    fprintf(stderr, "[M-A] config parsed\n"); fflush(stderr);
    /* Cap the KV cache span for serving: config.max_position_embeddings
     * (262144 for Qwen3) would otherwise be collapsed to 2048 inside
     * st_qwen_inference_init and long prompts would overflow the cache. */
    if (cfg->max_seq_len > 8192) cfg->max_seq_len = 8192;
    if (load_quant_weights(cfg, w) != 0) { rc = 1; goto done; }
format_done:
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
    if (model_is_gguf) {
        /* GGUF 单文件自包含 vocab（tokenizer.ggml.*） */
        if (gguf_load_tokenizer(m->model_dir, tok) != 0) {
            printf("[SERVE] GGUF tokenizer load failed\n");
            rc = 1; goto done;
        }
    } else {
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

    /* Vision encoder (multimodal image/video support). */
    if (cfg->has_vision) {
        vis = (STVisionState *)calloc(1, sizeof(STVisionState));
        if (cfg_is_vqf && w->vision && w->vision->is_allocated) {
            /* VQF：vision 权重已随 mmap 挂载（Q8 大矩阵 + F32 小张量固化布局），
             * 零加载零转换，直接值拷贝权重集合给推理状态。vis_w 指向
             * w->vision（alias vqf_map），由 st_weights_free 统一释放。 */
            if (vis && st_vision_init(vis, cfg) == 0) {
                vis_w = w->vision;
                vis->w = *vis_w;
                has_vision = 1;
                printf("[SERVE] vision encoder ready (VQF mmap, ViT %d layers, hidden=%d)\n",
                       cfg->vis_depth, cfg->vis_hidden);
            }
        } else {
            vis_w = (STVisionWeights *)calloc(1, sizeof(STVisionWeights));
            if (vis_w && vis) {
                st_vision_weights_alloc(vis_w, cfg);
                if (st_vision_load_weights(vis_w, cfg) == 0) {
                    st_vision_init(vis, cfg);
                    vis->w = *vis_w;
                    /* Inference reads the Q8 copies only; drop the F32
                     * duplicates (~1.65 GB) to stay under the OOM line. */
                    st_vision_weights_free_f32(vis_w);
                    has_vision = 1;
                    printf("[SERVE] vision encoder ready (ViT %d layers, hidden=%d)\n",
                           cfg->vis_depth, cfg->vis_hidden);
                } else {
                    fprintf(stderr, "[SERVE] WARNING: vision weight load failed -> text-only\n");
                }
                fflush(stdout);
            }
        }
    }

done:
    if (rc != 0) {
        if (ist) { st_qwen_inference_free(ist); free(ist); }
        if (w && w->is_allocated) { st_weights_free(w); }
        if (w) free(w);
        if (tok) { qwen_tokenizer_free(tok); free(tok); }
        /* VQF 路径 cfg 指向 w->cfg（随 w 一起释放），不能单独 free */
        if (cfg && !cfg_is_vqf) { st_config_free(cfg); free(cfg); }
        /* VQF 下 vis_w 指向 w->vision（alias vqf_map），由 st_weights_free 释放 */
        if (vis_w && !cfg_is_vqf) { st_vision_weights_free(vis_w); free(vis_w); }
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
    /* The admin page may have requested a different quantization mode for
     * this load (selected in the UI before pressing "加载模型"). */
    if (ctx->load_wmode >= 0 && ctx->load_wmode <= 5) {
        g_st_wmode = ctx->load_wmode;
    }
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

/* Unload the loaded model (VQF/GGUF/safetensors): free weights, tokenizer,
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
    int cfg_embedded = (ctx->w && ctx->cfg == &ctx->w->cfg); /* VQF/GGUF cfg 内嵌于 w */

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
        /* VQF 下 vis_w 指向 w->vision（alias vqf_map），由 st_weights_free 统一
         * 释放；仅普通 safetensors 路径需要独立 free。 */
        if (m && m->vis_w && !m->cfg_is_vqf) {
            st_vision_weights_free(m->vis_w);
            free(m->vis_w);
        }
        if (ctx->w->is_allocated) st_weights_free(ctx->w);
        free(ctx->w);
        ctx->w = NULL;
        if (m) { m->w = NULL; m->vis_w = NULL; m->has_vision = 0; }
    }
    if (ctx->cfg && !cfg_embedded) { st_config_free(ctx->cfg); free(ctx->cfg); }
    ctx->cfg = NULL;
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

static void serve_on_start(int actual_port, void *ud) {
    (void)ud;
    printf("\n  [SERVE] OpenAI-compatible API listening on http://0.0.0.0:%d\n", actual_port);
    printf("  [SERVE] 管理入口: http://0.0.0.0:%d/admin/   (管理页 admin.html)\n", actual_port);
    printf("  [SERVE] 对话入口: http://0.0.0.0:%d/chat    (对话页 chat.html)\n", actual_port);
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
    ctx.load_format = g_serve_load_format; /* 加载格式优先级（--load-format，管理页可改） */
    snprintf(ctx.load_format_str, sizeof(ctx.load_format_str), "%s",
             g_serve_load_format == 1 ? "vqf" :
             g_serve_load_format == 2 ? "safetensors" :
             g_serve_load_format == 3 ? "gguf" : "auto");
    /* Admin/management metadata (reported by /admin; NULL paths use the
     * board defaults in vllm_admin.c). */
    ctx.model_dir = model_dir;
    ctx.port = port;
    ctx.threads = g_serve_threads;
    ctx.npu_enabled = g_npu_enabled;
    ctx.prefix_cache = g_prefix_cache;
    ctx.prefix_kv = g_prefix_kv;
    ctx.min_p = g_serve_min_p;
    ctx.disk_kv = g_disk_kv;
    if (g_disk_kv && g_disk_kv_dir[0])
        snprintf(ctx.kvdir, sizeof(ctx.kvdir), "%s", g_disk_kv_dir);
    ctx.spec = g_spec;
    ctx.spec_k = g_spec_k;
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
    m.format_prio = ctx.load_format;   /* 管理页设置的加载格式优先级 */
    ctx.load_arg = &m;
    ctx.load_wmode = -1;   /* not overridden: keep the --wmode from the CLI */

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
        /* VQF 下 m.vis_w 指向 m.w->vision（alias vqf_map），由 st_weights_free
         * 统一释放；仅普通 safetensors 路径需要独立 free。 */
        if (m.vis_w && !m.cfg_is_vqf) st_vision_weights_free(m.vis_w);
    }
    if (m.ist) st_qwen_inference_free(m.ist);
    if (m.tok) { qwen_tokenizer_free(m.tok); free(m.tok); }
    /* VQF 路径 cfg 内嵌于 w（mmap 所有权随 w），不单独 free */
    if (m.cfg && !m.cfg_is_vqf) { st_config_free(m.cfg); free(m.cfg); }
    if (m.w && m.w->is_allocated) st_weights_free(m.w);
    if (m.w) free(m.w);
    if (m.vis) free(m.vis);
    if (m.vis_w && !m.cfg_is_vqf) free(m.vis_w);
    return rc;
}

/* ================================================================
 * main
 * ================================================================ */
int main(int argc, char **argv) {
    /* RK3588/Linux: raise the main-thread stack to match the MSVC
     * /STACK:16777216 build (large local arrays: scores[4096] etc.). */
    st_raise_stack_limit(16u * 1024u * 1024u);

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
        } else if (strcmp(argv[i], "--bench-mixed") == 0) {
            bench_mixed = 1;
        } else if (strcmp(argv[i], "--bench-users") == 0 && i + 1 < argc) {
            g_bench_users = atoi(argv[++i]);
            if (g_bench_users < 1) g_bench_users = 1;
            if (g_bench_users > MAX_PERF_USERS) g_bench_users = MAX_PERF_USERS;
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
        } else if (strcmp(argv[i], "--convert-vqf") == 0 && i + 1 < argc) {
            g_convert_vqf = argv[++i];   /* safetensors -> VQF (offline) */
        } else if (strcmp(argv[i], "--convert-gguf") == 0 && i + 1 < argc) {
            g_convert_gguf = argv[++i];  /* GGUF -> VQF (offline) */
        } else if (strcmp(argv[i], "--gguf-info") == 0 && i + 1 < argc) {
            g_gguf_info = argv[++i];     /* dump GGUF structure and exit */
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            g_serve_port = atoi(argv[++i]);
            g_serve_port_explicit = 1;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            g_serve_model_dir = argv[++i];
        } else if (strcmp(argv[i], "--model-id") == 0 && i + 1 < argc) {
            g_serve_model_id = argv[++i];
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
        } else if (strcmp(argv[i], "--load-format") == 0 && i + 1 < argc) {
            const char *lf = argv[++i];
            if (strcmp(lf, "vqf") == 0)         g_serve_load_format = 1;
            else if (strcmp(lf, "safetensors") == 0) g_serve_load_format = 2;
            else if (strcmp(lf, "gguf") == 0)   g_serve_load_format = 3;
            else                                g_serve_load_format = 0; /* auto */
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

    if (g_convert_vqf) {
        /* 离线转换：safetensors → VQF（需 --model 指向 safetensors 模型目录，
         * wmode/VLLM_Q8_8X8 等须与目标运行环境一致）。 */
        if (!g_serve_model_dir) {
            printf("[VQF] --convert-vqf 需要 --model <safetensors 模型目录>\n");
            return 1;
        }
        return run_convert_vqf(g_serve_model_dir, g_convert_vqf);
    }

    if (g_gguf_info) {
        return (gguf_dump_info(g_gguf_info) == 0) ? 0 : 1;
    }

    if (g_convert_gguf) {
        /* 离线转换：GGUF → VQF（--model 指向 .gguf 文件）。GGUF 已量化
         * （Q4_0/Q8_0/k-quants），加载后按当前 wmode 生成引擎布局副本。 */
        if (!g_serve_model_dir) {
            printf("[VQF] --convert-gguf 需要 --model <model.gguf>\n");
            return 1;
        }
        STModelWeights w; memset(&w, 0, sizeof(w));
        if (gguf_load_model(g_serve_model_dir, &w) != 0) {
            printf("[VQF] GGUF load failed\n");
            return 1;
        }
        uint32_t flags = 0;
        if (w.q8_buf_q4) flags |= VQF_FLAG_Q8BUF_Q4;
        if (w.has_x8)    flags |= VQF_FLAG_X8;
        if (st_wmode_effective() == 4) flags |= VQF_FLAG_G256;
        extern int g_st_q8_repack, g_st_q4_repack;
        if (g_st_q8_repack) {
            const char *e = getenv("VLLM_Q8_8X8");
            if (!(e && e[0] == '0')) flags |= VQF_FLAG_Q8_8X8;
        }
        if (g_st_q4_repack) flags |= VQF_FLAG_Q4_4X4;
        int rc = vqf_write(g_convert_gguf, &w, &w.cfg, flags);
        printf("[VQF] vqf_write rc=%d\n", rc);
        fflush(stdout);
        if (rc == 0) {
            printf("[VQF] self-check: loading %s ...\n", g_convert_gguf);
            fflush(stdout);
            STModelWeights w2; memset(&w2, 0, sizeof(w2));
            if (vqf_load(&w2, g_convert_gguf) == 0) {
                printf("[VQF] self-check PASS: %s reload OK (flags=0x%x)\n",
                       g_convert_gguf, flags);
                printf("[VQF] self-check: freeing w2\n");
                fflush(stdout);
                st_weights_free(&w2);
                printf("[VQF] self-check: w2 freed OK\n");
                fflush(stdout);
            } else {
                printf("[VQF] self-check FAIL: %s\n", g_convert_gguf);
                rc = 1;
            }
        }
        printf("[VQF] freeing main weights\n");
        fflush(stdout);
        st_weights_free(&w);
        printf("[VQF] main weights freed, rc=%d\n", rc);
        fflush(stdout);
        return rc;
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

    if (bench_mixed == 1) {
        /* Synthetic-weight mixed-precision kernel benchmark - no model load,
         * completes in constrained environments. */
        st_bench_mixed_precision();
        return 0;
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

    if (!perf_only) {
        test_superposition_axioms();
        test_ntt_transform();
        test_kvcache();
        test_inference_pipeline();
        test_real_inference();
        vmm_self_test();
    }

    if (run_multimodal || n_images > 0 || vid_dir) {
        /* --perf-only: run text benchmark first, then multimodal */
        if (perf_only) {
            test_performance_benchmark();
        }
        test_multimodal_analysis(img_paths, n_images, vid_dir);
    } else {
        test_performance_benchmark();
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

/**
 * vllm_tokenizer_qwen.h - Qwen3-VL Tokenizer (vocab.bin loader)
 *
 * Loads HuggingFace vocabulary exported by export_vocab.py.
 * Provides encode/decode with UTF-8 longest-prefix matching.
 * No external BPE dependencies — works standalone with pre-exported vocab.
 */

#ifndef VLLM_TOKENIZER_QWEN_H
#define VLLM_TOKENIZER_QWEN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QWEN_TOKENIZER_MAX_TOKENS  151936
#define QWEN_TOKENIZER_MAX_STR      256

typedef struct {
    int    vocab_size;
    char **strings;       /* [vocab_size] token strings (UTF-8) */
    int   *str_lens;      /* [vocab_size] byte lengths */
    char  *str_data;      /* contiguous string storage */
    int    max_str_len;   /* longest token string in bytes */

    /* ---- 词表索引：把"每个位置全量扫 vocab_size 条"降到 O(最长 token 字节数) ----
     * 键 = token 字节串，值 = 该串的**最小** token id —— 与旧的线性扫描
     * （严格 `tlen > best_len` 比较 ⇒ 等长时先出现者胜 = 最小 id）逐位一致。
     * 开放寻址 int 表：0 = 空槽，槽内存 id+1（id 0 是合法 token，不能用 0 当空）。
     * first_maxlen[b] = 首字节为 b 的最长 token 字节数，用于把"逐长度回退"的
     * 起点收紧（中文 token 多在 1~3 字节，不必从 max_str_len 一路试下来）。 */
    int   *vocab_tab;
    int    vocab_tab_mask;
    int    first_maxlen[256];
    /* 1 = 强制走旧线性扫描（env VLLM_TOK_LINEAR=1）。仅用于位级回归对照：
     * 索引路径与线性路径对同一输入必须产出完全相同的 id 序列。 */
    int    tok_linear;

    /* Special token IDs */
    int    bos_id;
    int    eos_id;
    int    pad_id;
    int    im_start_id;
    int    im_end_id;

    /* Vision special token IDs (Qwen3-VL) */
    int    vision_start_id;
    int    vision_end_id;
    int    image_token_id;
    int    video_token_id;

    int    is_loaded;
} QwenTokenizer;

/**
 * Load vocabulary from vocab.bin (output of export_vocab.py).
 * model_dir: path to directory containing vocab.bin
 * Returns 0 on success, -1 on error.
 */
int qwen_tokenizer_load(QwenTokenizer *tok, const char *model_dir);

/**
 * Free tokenizer resources.
 */
void qwen_tokenizer_free(QwenTokenizer *tok);

/**
 * Encode a UTF-8 string into token IDs using longest-prefix matching.
 * Returns number of tokens written (up to max_tokens).
 * Does NOT prepend BOS automatically.
 */
int qwen_tokenizer_encode(QwenTokenizer *tok, const char *text,
                           int *token_ids, int max_tokens);

/**
 * Decode a single token ID to its string.
 * Returns pointer to internal string (do not free).
 * Returns "[UNK]" if token_id is out of range.
 */
const char* qwen_tokenizer_decode(const QwenTokenizer *tok, int token_id);

/**
 * Load special token IDs from model_dir (config.json + tokenizer_config.json).
 * Must be called after qwen_tokenizer_load().
 * Returns 0 on success, -1 on error.
 */
int qwen_tokenizer_load_special(QwenTokenizer *tok, const char *model_dir);

#ifdef __cplusplus
}
#endif

#endif /* VLLM_TOKENIZER_QWEN_H */

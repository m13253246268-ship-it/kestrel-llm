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

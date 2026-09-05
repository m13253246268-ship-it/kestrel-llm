/**
 * vllm_tokenizer_qwen.c - Qwen3-VL Tokenizer Implementation
 *
 * Loads vocab.bin binary format and provides UTF-8 longest-prefix
 * tokenization. Designed for the Qwen3-VL-8B-Instruct 151K vocabulary.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "vllm_tokenizer_qwen.h"
#include "vllm_util.h"

#ifdef __linux__
#include <unistd.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ================================================================
 * Load vocab.bin binary format
 * ================================================================ */

int qwen_tokenizer_load(QwenTokenizer *tok, const char *model_dir) {
    memset(tok, 0, sizeof(*tok));
    tok->bos_id = -1;
    tok->eos_id = -1;
    tok->pad_id = -1;
    tok->im_start_id = -1;
    tok->im_end_id = -1;

    char path[1024];
    snprintf(path, sizeof(path), "%s/vocab.bin", model_dir);

    FILE *f = st_fopen(path, "rb");
    if (!f) {
        /* Try model directory directly */
        snprintf(path, sizeof(path), "%s/build/vocab.bin", model_dir);
        f = st_fopen(path, "rb");
    }
    if (!f) {
        /* Fallback: 软件根目录（可执行文件所在目录）的统一 vocab.bin。
         * 多模型共享同一 Qwen 词表时，模型目录可省去该文件，由部署目录提供。 */
#ifdef __linux__
        char exe_path[1024];
        ssize_t rl = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
        if (rl > 0) {
            exe_path[rl] = '\0';
            char *slash = strrchr(exe_path, '/');
            if (slash) {
                *slash = '\0';
                snprintf(path, sizeof(path), "%s/vocab.bin", exe_path);
                f = st_fopen(path, "rb");
            }
        }
#endif
    }
    if (!f) {
        fprintf(stderr, "[TOK] Cannot open vocab.bin: %s\n", path);
        return -1;
    }

    /* Read vocab_size */
    uint32_t v32;
    if (fread(&v32, 4, 1, f) != 1) {
        fclose(f);
        return -1;
    }
    tok->vocab_size = v32;

    if (tok->vocab_size <= 0 || tok->vocab_size > QWEN_TOKENIZER_MAX_TOKENS) {
        fprintf(stderr, "[TOK] Invalid vocab_size: %d\n", tok->vocab_size);
        fclose(f);
        return -1;
    }

    /* Allocate string pointers */
    tok->strings  = calloc((size_t)tok->vocab_size, sizeof(char*));
    tok->str_lens = calloc((size_t)tok->vocab_size, sizeof(int));
    if (!tok->strings || !tok->str_lens) {
        fprintf(stderr, "[TOK] OOM allocating %d token pointers\n", tok->vocab_size);
        fclose(f);
        return -1;
    }

    /* First pass: read all entries, calculate total string data size */
    /* We need to store entries by token_id, but file may not be sorted. */
    /* vocab.bin is written in order: tid=0,1,2,...vocab_size-1 */

    size_t total_str_bytes = 0;
    long *offsets = malloc((size_t)tok->vocab_size * sizeof(long));
    int  *lens    = malloc((size_t)tok->vocab_size * sizeof(int));
    int  *ids     = malloc((size_t)tok->vocab_size * sizeof(int));
    if (!offsets || !lens || !ids) {
        fprintf(stderr, "[TOK] OOM during vocab scan\n");
        fclose(f);
        free(offsets); free(lens); free(ids);
        return -1;
    }

    tok->max_str_len = 0;
    for (int i = 0; i < tok->vocab_size; i++) {
        uint32_t tid;
        uint16_t slen;
        if (fread(&tid, 4, 1, f) != 1) break;
        if (fread(&slen, 2, 1, f) != 1) break;

        ids[i]    = tid;
        lens[i]   = (int)slen;
        offsets[i] = ftell(f);
        total_str_bytes += (size_t)slen + 1;  /* +1 for null terminator */
        if (lens[i] > tok->max_str_len) tok->max_str_len = lens[i];

        /* Skip string data */
        fseek(f, (long)slen, SEEK_CUR);
    }

    /* Allocate contiguous string storage */
    tok->str_data = calloc(total_str_bytes, 1);
    if (!tok->str_data) {
        fprintf(stderr, "[TOK] OOM allocating string data (%zu bytes)\n", total_str_bytes);
        fclose(f);
        free(offsets); free(lens); free(ids);
        return -1;
    }

    /* Second pass: read strings into str_data and set pointers */
    char *data_ptr = tok->str_data;
    for (int i = 0; i < tok->vocab_size; i++) {
        int tid = ids[i];
        if (tid < 0 || tid >= tok->vocab_size) continue;

        fseek(f, offsets[i], SEEK_SET);
        size_t nread = fread(data_ptr, 1, (size_t)lens[i], f);
        data_ptr[nread] = '\0';

        tok->strings[tid]  = data_ptr;
        tok->str_lens[tid] = lens[i];
        data_ptr += lens[i] + 1;
    }

    fclose(f);
    free(offsets);
    free(lens);
    free(ids);

    tok->is_loaded = 1;
    printf("[TOK] Loaded %d tokens from vocab.bin (max_len=%d)\n",
           tok->vocab_size, tok->max_str_len);
    return 0;
}

void qwen_tokenizer_free(QwenTokenizer *tok) {
    if (!tok->is_loaded) return;
    free(tok->strings);
    free(tok->str_lens);
    free(tok->str_data);
    memset(tok, 0, sizeof(*tok));
}

/* ================================================================
 * Encode: UTF-8 longest-prefix matching
 * ================================================================ */

/* Check if byte starts a UTF-8 continuation byte (0x80-0xBF) */
static int is_utf8_cont(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

static int find_longest_match_qwen(QwenTokenizer *tok,
                                    const char *text, int text_len,
                                    int *match_len) {
    int best_id = -1;
    int best_len = 0;

    /* Simple linear scan — could be optimized with trie */
    for (int i = 0; i < tok->vocab_size; i++) {
        int tlen = tok->str_lens[i];
        if (tlen == 0) continue;
        if (tlen > text_len) continue;
        if (memcmp(tok->strings[i], text, (size_t)tlen) == 0) {
            if (tlen > best_len) {
                best_id  = i;
                best_len = tlen;
            }
        }
    }

    *match_len = best_len;
    return best_id;
}

int qwen_tokenizer_encode(QwenTokenizer *tok, const char *text,
                           int *token_ids, int max_tokens) {
    if (!tok->is_loaded) return 0;

    int text_len = (int)strlen(text);
    int pos = 0;
    int num = 0;

    while (pos < text_len && num < max_tokens) {
        /* Skip spaces (but preserve them as position markers) */
        /* We attempt to match with leading space for BPE-style encoding */
        int had_space = 0;
        while (pos < text_len && text[pos] == ' ') {
            pos++;
            had_space = 1;
        }
        if (pos >= text_len) break;

        const char *rem = text + pos;
        int rem_len = text_len - pos;

        int match_len = 0;
        int token_id = -1;

        /* For Qwen2 tokenizer, word continuations typically use "Ġ" prefix (0xC4 0xA0).
         * Try matching with this prefix first if we had a space before this word. */
        if (had_space && num > 0) {
            char spaced_buf[512];
            int buf_len = rem_len + 2;
            if (buf_len > 511) buf_len = 511;
            spaced_buf[0] = '\xC4';
            spaced_buf[1] = '\xA0';
            if (buf_len - 2 > 0) memcpy(spaced_buf + 2, rem, (size_t)(buf_len - 2));
            spaced_buf[buf_len] = '\0';

            token_id = find_longest_match_qwen(tok, spaced_buf, buf_len, &match_len);
            if (token_id >= 0 && match_len > 2) {
                pos += (match_len - 2);  /* subtract the 2-byte Ġ prefix */
                token_ids[num++] = token_id;
                continue;
            }
        }

        /* Try matching without space prefix */
        token_id = find_longest_match_qwen(tok, rem, rem_len, &match_len);
        if (token_id >= 0 && match_len > 0) {
            token_ids[num++] = token_id;
            pos += match_len;
        } else {
            /* Fallback: encode single byte as character */
            if (rem[0] == '\n') {
                /* The Qwen BPE represents a literal newline as the "Ċ"
                 * marker (U+010A, bytes C4 8A); vocab.bin has no standalone
                 * 0x0A entry, so a raw '\n' would otherwise fall through to
                 * <unk> (id 0) and corrupt the prompt template. */
                int ml2;
                int tid2 = find_longest_match_qwen(tok, "\xC4\x8A", 2, &ml2);
                if (tid2 >= 0 && ml2 == 2) {
                    token_ids[num++] = tid2;
                    pos++;
                    continue;
                }
            }
            char c[2] = {rem[0], '\0'};
            int ml;
            int tid = find_longest_match_qwen(tok, c, 1, &ml);
            if (tid >= 0 && ml > 0) {
                token_ids[num++] = tid;
            } else {
                /* Use unk or just skip */
                token_ids[num++] = 0; /* <unk> */
            }
            pos++;
        }
    }

    return num;
}

/* ================================================================
 * Decode
 * ================================================================ */

const char* qwen_tokenizer_decode(const QwenTokenizer *tok, int token_id) {
    if (!tok->is_loaded) return "";
    if (token_id < 0 || token_id >= tok->vocab_size) return "[UNK]";
    if (!tok->strings[token_id]) return "";
    return tok->strings[token_id];
}

/* ================================================================
 * Load special token IDs from config.json
 * ================================================================ */

/* Minimal JSON integer value parser */
static int json_get_int(const char *json, const char *key, int *out) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *pos = strstr(json, search);
    if (!pos) return -1;
    pos += strlen(search);
    while (*pos == ' ' || *pos == ':' || *pos == '\t') pos++;
    if (*pos == 'n') { *out = -1; return -1; }  /* null */
    char *end;
    long v = strtol(pos, &end, 10);
    if (end == pos) return -1;
    *out = (int)v;
    return 0;
}

int qwen_tokenizer_load_special(QwenTokenizer *tok, const char *model_dir) {
    if (!tok->is_loaded) return -1;

    /* Read config.json */
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);

    FILE *f = st_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[TOK] Cannot open config.json for special tokens\n");
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json = malloc((size_t)fsize + 1);
    if (!json) { fclose(f); return -1; }
    size_t nread = fread(json, 1, (size_t)fsize, f);
    fclose(f);
    json[nread] = '\0';

    /* Qwen3-VL config has text_config nested object.
     * bos_token_id, eos_token_id are inside text_config. */
    json_get_int(json, "bos_token_id", &tok->bos_id);
    json_get_int(json, "eos_token_id", &tok->eos_id);

    /* Find im_start / im_end from tokenizer_config or hardcode known values */
    /* Qwen3-VL known specials: im_start=151644, im_end=151645, pad=151643 */
    if (tok->bos_id < 0) tok->bos_id = 151643;   /* <|endoftext|> */
    if (tok->eos_id < 0) tok->eos_id = 151645;   /* <|im_end|> */
    tok->im_start_id = 151644;  /* <|im_start|> */
    tok->im_end_id   = 151645;  /* <|im_end|> */
    tok->pad_id      = 151643;  /* <|endoftext|> */

    free(json);

    printf("[TOK] Special tokens: bos=%d eos=%d im_start=%d im_end=%d pad=%d\n",
           tok->bos_id, tok->eos_id, tok->im_start_id, tok->im_end_id, tok->pad_id);
    return 0;
}

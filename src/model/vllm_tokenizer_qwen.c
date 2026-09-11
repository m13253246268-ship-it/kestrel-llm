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

/* 在词表里找与 text 前缀相同、且长度不超过 max_len 的最长 token。
 * max_len 用来把匹配限制在"下一个特殊 token 之前"，防止贪心长匹配把特殊
 * token 的首字节吞掉（见 QWEN_SPECIAL_STR 上方注释）。 */
static int find_longest_match_qwen(QwenTokenizer *tok,
                                    const char *text, int text_len, int max_len,
                                    int *match_len) {
    int best_id = -1;
    int best_len = 0;

    if (max_len > text_len) max_len = text_len;

    /* Simple linear scan — could be optimized with trie */
    for (int i = 0; i < tok->vocab_size; i++) {
        int tlen = tok->str_lens[i];
        if (tlen == 0) continue;
        if (tlen > max_len) continue;
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

/* ---- 特殊 token 最高优先级匹配 ----------------------------------------
 * 本编码器是"逐位置取最长词表前缀"，不是真正的 BPE。当某个字符合成的二元
 * token 恰好以 '<' 结尾时（词表里存在 ".<"(15757)、"。"+"<"(89393) 这类
 * 条目），贪心匹配会把下一个特殊 token 的 '<' 一起吃掉，于是
 * <|im_end|> 被切成 5 个普通 token（"|i"+"m"+"_end"+"|"+">"）而不是单个
 * 151645。后果是 ChatML 模板被破坏：模型收到的是把 <|im_end|> 当纯文本的
 * prompt（训练时从未见过），分布外 → 开头还算连贯、随后坍缩成重复。
 * 因此在这些控制串上先做一次"整串精确匹配"，命中就直接出特殊 token id。
 * 与 HF 分词器行为一致（encode('<|im_end|>', add_special_tokens=False)
 * 同样返回 [151645]）。 */
static const char *const QWEN_SPECIAL_STR[] = {
    "<|im_start|>", "<|im_end|>", "<|endoftext|>",
    "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
    "<|object_ref_start|>", "<|object_ref_end|>",
    "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
    "<think>", "</think>",          /* Qwen3 思考段边界（151667 / 151668） */
};
#define QWEN_N_SPECIAL \
    ((int)(sizeof(QWEN_SPECIAL_STR) / sizeof(QWEN_SPECIAL_STR[0])))

/* 词表里整串精确等于 s 的 token id（长度相同且字节全同），找不到返回 -1。 */
static int find_exact_token(const QwenTokenizer *tok, const char *s, int len) {
    for (int i = 0; i < tok->vocab_size; i++) {
        if (!tok->strings[i]) continue;
        if (tok->str_lens[i] != len) continue;
        if (memcmp(tok->strings[i], s, (size_t)len) == 0) return i;
    }
    return -1;
}

/* 惰性解析并缓存特殊 token 的 id（按 tokenizer 指针失效）。 */
static int special_token_id(const QwenTokenizer *tok, int idx) {
    static const QwenTokenizer *cached_tok = NULL;
    static int cached_ids[QWEN_N_SPECIAL];
    if (cached_tok != tok) {
        for (int i = 0; i < QWEN_N_SPECIAL; i++)
            cached_ids[i] = find_exact_token(tok, QWEN_SPECIAL_STR[i],
                                             (int)strlen(QWEN_SPECIAL_STR[i]));
        cached_tok = tok;
    }
    return cached_ids[idx];
}

/* 返回 rem 里"下一个特殊 token"的起始偏移（>0）；不存在则返回 rem_len。
 * 编码时用这个上界截断普通 token 的最长匹配：否则像 "." + "<|im_end|>" 这种
 * 位置，贪心匹配会先吃掉词表里的 ".<"（id 15757），使扫描指针跳过 '<'，
 * 后面的特殊 token 就再也匹配不上了。 */
static int special_start_after(const char *rem, int rem_len) {
    const char *p = rem;
    int left = rem_len;
    while (left > 1) {
        const char *lt = (const char *)memchr(p + 1, '<', (size_t)(left - 1));
        if (!lt) break;
        int k = (int)(lt - rem);
        for (int i = 0; i < QWEN_N_SPECIAL; i++) {
            int l = (int)strlen(QWEN_SPECIAL_STR[i]);
            if (l <= rem_len - k &&
                memcmp(lt, QWEN_SPECIAL_STR[i], (size_t)l) == 0)
                return k;
        }
        left = rem_len - k;
        p = lt;
    }
    return rem_len;
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

        /* 特殊 token 优先：整串精确命中就直接产出特殊 token id，避免被下面的
         * 贪心最长匹配吞掉首字节（见 QWEN_SPECIAL_STR 上方注释）。 */
        {
            int sp_hit = -1, sp_len = 0;
            for (int i = 0; i < QWEN_N_SPECIAL; i++) {
                int l = (int)strlen(QWEN_SPECIAL_STR[i]);
                if (l > rem_len || l <= sp_len) continue;
                if (memcmp(rem, QWEN_SPECIAL_STR[i], (size_t)l) == 0) {
                    sp_hit = i;
                    sp_len = l;
                }
            }
            if (sp_hit >= 0) {
                int sid = special_token_id(tok, sp_hit);
                if (sid >= 0) {
                    token_ids[num++] = sid;
                    pos += sp_len;
                    continue;
                }
            }
        }

        /* 普通 token 的匹配上界：不得跨过后面出现的特殊 token 起始位置。 */
        int lim = special_start_after(rem, rem_len);

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

            token_id = find_longest_match_qwen(tok, spaced_buf, buf_len,
                                               buf_len, &match_len);
            if (token_id >= 0 && match_len > 2 && (match_len - 2) <= lim) {
                pos += (match_len - 2);  /* subtract the 2-byte Ġ prefix */
                token_ids[num++] = token_id;
                continue;
            }
        }

        /* Try matching without space prefix */
        token_id = find_longest_match_qwen(tok, rem, rem_len, lim, &match_len);
        if (token_id >= 0 && match_len > 0) {
            token_ids[num++] = token_id;
            pos += match_len;
        } else {
            /* Fallback: encode single byte as character */
            if (rem[0] == '\n') {
                /* The Qwen BPE represents a literal newline as the "Ċ"
                 * marker (U+010A, bytes C4 8A); vocab.bin has no standalone
                 * 0x0A entry, so a raw '\n' would otherwise fall through to
                 * <unk> (id 0) and corrupt the prompt template.
                 * 连续的换行同样要按「连续多个 Ċ」去匹配：HF 把 "\n\n"
                 * 切成单个 token 271（ĊĊ），逐个吃会变成两个 198，与参考
                 * 分词不一致（Qwen3 非 thinking 模板的 `<think>\n\n</think>\n\n`
                 * 正是这种形状）。 */
                int nl = 0;
                while (nl < rem_len && rem[nl] == '\n') nl++;
                if (nl > 8) nl = 8;
                if (nl > lim) nl = lim;
                if (nl > 0) {
                    char nb[16];
                    for (int i = 0; i < nl; i++) {
                        nb[i * 2] = '\xC4';
                        nb[i * 2 + 1] = '\x8A';
                    }
                    int ml2 = 0;
                    int tid2 = find_longest_match_qwen(tok, nb, nl * 2, nl * 2, &ml2);
                    if (tid2 >= 0 && ml2 >= 2 && (ml2 % 2) == 0) {
                        token_ids[num++] = tid2;
                        pos += ml2 / 2;
                        continue;
                    }
                }
            }
            char c[2] = {rem[0], '\0'};
            int ml;
            int tid = find_longest_match_qwen(tok, c, 1, 1, &ml);
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

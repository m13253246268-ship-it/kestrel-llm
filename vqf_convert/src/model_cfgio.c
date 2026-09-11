/* Auto-extracted from src/model/vllm_safetensors.c (sha256 da768fbff3b10063a3815f8786b536892e0b7ec68ef6efe8a10180483bd27cc3)
 * Re-run tools/extract_quant.py after engine edits; verify sha256.
 * ================================================================ */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>    /* errno（glibc 不再间接引入；MinGW 下曾侥幸可用） */
#include <dirent.h>   /* shard glob (model-N-of-N.safetensors) */

#include "conv_platform.h"
#include "vqf_st.h"
int st_load_tensor(const STModelConfig *cfg, const char *name, float *dst, int n);
int st_load_tensor_slice(const STModelConfig *cfg, const char *name, int64_t off, float *dst, int64_t n);
void st_config_free(STModelConfig *cfg);

/* json/config/tensor helpers (src 1182-1371) */
static const char* json_find_key(const char *json, const char *key) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *pos = strstr(json, search);
    if (!pos) return NULL;
    pos += strlen(search);
    while (*pos == ' ' || *pos == ':') pos++;
    return pos;
}

/* Like json_find_key but limited to the range [start, end) */
static const char* json_find_key_in_range(const char *start, const char *end, const char *key) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    size_t key_len = strlen(search);
    size_t range_len = end - start;
    const char *pos = start;
    while (pos + key_len <= end) {
        pos = memchr(pos, '"', end - pos);
        if (!pos) return NULL;
        if (pos + key_len > end) return NULL;
        if (memcmp(pos, search, key_len) == 0) {
            pos += key_len;
            while (pos < end && (*pos == ' ' || *pos == ':')) pos++;
            if (pos >= end) return NULL;
            return pos;
        }
        pos++;
    }
    return NULL;
}

/* Parse an integer value at pos. Sets *out and returns
 * pointer one past the integer, or pos if failed. */
static const char* json_parse_int(const char *pos, int *out) {
    char *end;
    long v = strtol(pos, &end, 10);
    if (end == pos) { *out = 0; return pos; }
    *out = (int)v;
    return end;
}

/* Parse a float value at pos. */
static const char* json_parse_float(const char *pos, float *out) {
    char *end;
    float v = strtof(pos, &end);
    if (end == pos) { *out = 0.0f; return pos; }
    *out = v;
    return end;
}

/* Parse a string value (quoted) at pos. Allocates and returns new string.
 * Returns NULL if not a valid quoted string. */
static char* json_parse_string(const char *pos) {
    while (*pos && *pos != '"') pos++;
    if (!*pos) return NULL;
    pos++; /* skip opening " */
    const char *start = pos;
    while (*pos && *pos != '"') pos++;
    if (!*pos) return NULL;
    size_t len = pos - start;
    char *s = malloc(len + 1);
    if (!s) return NULL;
    memcpy(s, start, len);
    s[len] = '\0';
    return s;
}

/* ================================================================
 * bfloat16 conversion
 * ================================================================ */

static float bf16_to_f32_impl(uint16_t bf16) {
    /* bf16: [b15..b0] = [sign][exp8][mant7]
     * f32:  [b31..b0] = [sign][exp8][mant23]
     * Just shift left by 16 bits */
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void bf16_to_f32_array(float *dst, const uint8_t *src, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        uint16_t bf16;
        memcpy(&bf16, src + i * 2, 2);
        dst[i] = bf16_to_f32_impl(bf16);
    }
}

static void f16_to_f32_array(float *dst, const uint8_t *src, int64_t n) {
    /* IEEE 754 f16 → f32 */
    for (int64_t i = 0; i < n; i++) {
        uint16_t h;
        memcpy(&h, src + i * 2, 2);
        int sign = (h >> 15) & 1;
        int exp  = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0) {
            if (mant == 0) { dst[i] = sign ? -0.0f : 0.0f; continue; }
            while (mant < 0x400) { mant <<= 1; exp--; }
            exp++; mant &= 0x3FF;
        } else if (exp == 0x1F) {
            dst[i] = NAN; continue;
        }
        int exp_f32 = exp + 112;
        uint32_t bits = ((uint32_t)sign << 31) | ((uint32_t)exp_f32 << 23) | ((uint32_t)mant << 13);
        memcpy(&dst[i], &bits, sizeof(float));
    }
}

/* ================================================================
 * Read entire file into memory
 * ================================================================ */

static uint8_t *read_file(const char *path, size_t *out_size) {
    FILE *f = st_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[ST] Cannot open: %s (errno=%d)\n", path, errno);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long fsize64 = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize64 <= 0) { fclose(f); return NULL; }
    /* Only read up to 1GB for full file reads; larger files need per-tensor loading */
    if (fsize64 > 1024 * 1024 * 1024) {
        fprintf(stderr, "[ST] File too large for full read: %s (%lld bytes)\n", path, fsize64);
        fclose(f);
        return NULL;
    }
    size_t fsize = (size_t)fsize64;
    uint8_t *buf = malloc(fsize + 1);
    if (!buf) { fclose(f); fprintf(stderr, "[ST] OOM reading %s (%zu bytes)\n", path, fsize); return NULL; }
    size_t nread = fread(buf, 1, fsize, f);
    fclose(f);
    if (nread != fsize) { free(buf); return NULL; }
    buf[fsize] = '\0';   /* null-terminate: callers pass buf to strstr/strchr */
    *out_size = fsize;
    return buf;
}

/**
 * Read only the JSON header portion of a safetensors file.
 * This avoids loading the multi-GB tensor data into memory.
 * Returns the header string (null-terminated) and sets data_offset.
 */
static char *read_st_header(const char *path) {
    FILE *f = st_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[ST] Cannot open: %s (errno=%d)\n", path, errno);
        return NULL;
    }
    /* Read 8-byte header size (uint64 LE) */
    uint64_t header_size = 0;
    if (fread(&header_size, 8, 1, f) != 1) {
        fprintf(stderr, "[ST] Cannot read header from: %s\n", path);
        fclose(f);
        return NULL;
    }
    if (header_size > 100 * 1024 * 1024) {  /* Sanity: max 100MB header */
        fprintf(stderr, "[ST] Header too large: %llu bytes in %s\n",
                (unsigned long long)header_size, path);
        fclose(f);
        return NULL;
    }
    char *header = malloc((size_t)header_size + 1);
    if (!header) {
        fprintf(stderr, "[ST] OOM for header %zu bytes\n", (size_t)header_size);
        fclose(f);
        return NULL;
    }
    size_t nread = fread(header, 1, (size_t)header_size, f);
    fclose(f);
    if (nread != (size_t)header_size) {
        free(header);
        return NULL;
    }
    header[header_size] = '\0';
    return header;
}

/* ================================================================
 * st_parse_config: Parse model config from HuggingFace JSON files
 * ================================================================ */

/* Count tensors and deduce file layout from index JSON */
#define MAX_SHARDS 32
#define MAX_TENSORS 65536   /* 30B-A3B: 18,867 tensors > legacy 1024 cap */

/* ---- 分片 glob：无 index.json/单文件时扫 model-NNNNN-of-NNNNN.safetensors ---- */
static int shard_no(const char *name) {
    int a = -1, b = -1;
    if (sscanf(name, "model-%d-of-%d.safetensors", &a, &b) == 2) return a;
    return -1;
}
static int cmp_shard(const void *pa, const void *pb) {
    const char *a = *(const char *const *)pa, *b = *(const char *const *)pb;
    return shard_no(a) - shard_no(b);
}
static int glob_model_shards(const char *model_dir, STModelConfig *cfg) {
    DIR *dh = opendir(model_dir);
    if (!dh) return -1;
    struct dirent *de;
    char *names[64];
    int nf = 0;
    while ((de = readdir(dh)) != NULL) {
        if (shard_no(de->d_name) < 0 || nf >= 64) continue;
        names[nf++] = strdup(de->d_name);
    }
    closedir(dh);
    if (nf == 0) { fprintf(stderr, "[ST] no model shards in %s\n", model_dir); return -1; }
    qsort(names, (size_t)nf, sizeof(char *), cmp_shard);
    cfg->n_files = nf;
    cfg->file_paths = calloc((size_t)nf, sizeof(char *));
    for (int i = 0; i < nf; i++) {
        size_t plen = strlen(model_dir) + 1 + strlen(names[i]) + 1;
        cfg->file_paths[i] = malloc(plen);
        snprintf(cfg->file_paths[i], plen, "%s/%s", model_dir, names[i]);
        free(names[i]);
    }
    fprintf(stderr, "[ST] Globbed %d shard(s) (model-*-of-*.safetensors)\n", nf);
    return 0;
}

/* st_parse_config + find_tensor + get_file_* (src 1372-1743) */
int st_parse_config(const char *model_dir, STModelConfig *cfg) {
    memset(cfg, 0, sizeof(*cfg));

    /* --- Parse config.json --- */
    char path_cfg[1024];
    snprintf(path_cfg, sizeof(path_cfg), "%s/config.json", model_dir);
    size_t cfg_size;
    uint8_t *cfg_data = read_file(path_cfg, &cfg_size);
    if (!cfg_data) return -1;

    char *json = (char*)cfg_data;
    const char *pos;

    #define GET_INT(key, field) \
        pos = json_find_key(json, key); \
        if (pos && *pos != 'n') json_parse_int(pos, &cfg->field)

    #define GET_FLOAT(key, field) \
        pos = json_find_key(json, key); \
        if (pos) json_parse_float(pos, &cfg->field)

    GET_INT("hidden_size", dim);
    GET_INT("num_hidden_layers", n_layers);
    GET_INT("num_attention_heads", n_heads);
    GET_INT("num_key_value_heads", n_kv_heads);
    GET_INT("intermediate_size", ffn_dim);
    GET_INT("vocab_size", vocab_size);
    GET_INT("max_position_embeddings", max_seq_len);
    GET_INT("bos_token_id", bos_id);
    GET_INT("eos_token_id", eos_id);
    GET_FLOAT("rope_theta", rope_theta);
    GET_FLOAT("rms_norm_eps", norm_eps);

    /* Detect head_dim, q_norm, mrope from config */
    /* Prefer the explicit "head_dim" field: Qwen3 small models set it
     * independently of hidden_size/n_heads (e.g. Qwen3-0.6B: dim=1024,
     * heads=16, head_dim=128, so dim/n_heads=64 would be WRONG). The 8B
     * config also carries head_dim=128 == dim/n_heads, so this is a no-op
     * for the production model. Fall back to dim/n_heads when absent. */
    GET_INT("head_dim", head_dim);
    if (cfg->head_dim <= 0) {
        if (cfg->n_heads > 0 && cfg->dim > 0)
            cfg->head_dim = cfg->dim / cfg->n_heads;
        else
            cfg->head_dim = 128;
    }

    cfg->has_q_norm = (strstr(json, "q_norm") != NULL);
    cfg->has_mrope = (strstr(json, "mrope") != NULL);
    cfg->head_dim_full = cfg->head_dim;
    cfg->kv_lora_rank = 0;

    /* ---- MoE (qwen3_moe) ---- */
    GET_INT("num_experts", n_experts);
    GET_INT("num_experts_per_tok", top_k);
    GET_INT("moe_intermediate_size", moe_ffn);
    GET_INT("num_shared_experts", shared_experts);
    if (cfg->n_experts > 0 && cfg->moe_ffn > 0 && cfg->top_k > 0) {
        cfg->is_moe = 1;
        /* 专家行连续堆叠后每层 gate/up/down 与稠密 ffn 同构：
         * rows(gate/up) = n_experts*moe_ffn，down cols = n_experts*moe_ffn。 */
        cfg->ffn_dim = cfg->n_experts * cfg->moe_ffn;
        fprintf(stderr, "[CFG] MoE: experts=%d top_k=%d moe_ffn=%d "
                        "ffn_dim(=nl*stack)=%d\n",
                cfg->n_experts, cfg->top_k, cfg->moe_ffn, cfg->ffn_dim);
    }

    /* ---- Parse vision config ---- */
    /* Vision special tokens */
    GET_INT("vision_start_token_id", vision_start_id);
    GET_INT("vision_end_token_id", vision_end_id);
    GET_INT("image_token_id", image_token_id);
    GET_INT("video_token_id", video_token_id);

    /* Extract vision_config sub-object scope */
    {
        const char *vis_start = strstr(json, "\"vision_config\"");
        if (vis_start) {
            vis_start = strchr(vis_start, '{');
            if (vis_start) {
                /* Find matching closing brace */
                const char *vis_end = vis_start;
                int depth = 0;
                while (*vis_end) {
                    if (*vis_end == '{') depth++;
                    else if (*vis_end == '}') { depth--; if (depth == 0) break; }
                    vis_end++;
                }
                /* Parse vision_config fields within scope */
                #define VIS_GET_INT(key, field) \
                    do { const char *p2 = json_find_key_in_range(vis_start, vis_end, key); \
                         if (p2 && *p2 != 'n') json_parse_int(p2, &cfg->field); } while(0)

                VIS_GET_INT("depth", vis_depth);
                VIS_GET_INT("hidden_size", vis_hidden);
                VIS_GET_INT("num_heads", vis_heads);
                VIS_GET_INT("intermediate_size", vis_ffn);
                VIS_GET_INT("patch_size", vis_patch);
                VIS_GET_INT("temporal_patch_size", vis_temporal);
                VIS_GET_INT("spatial_merge_size", vis_merge);
                VIS_GET_INT("out_hidden_size", vis_out_dim);
                VIS_GET_INT("in_channels", vis_in_chan);
                VIS_GET_INT("num_position_embeddings", vis_max_pos);

                cfg->has_vision = (cfg->vis_depth > 0);

                /* Parse deepstack_visual_indexes array */
                cfg->vis_ds_count = 0;
                {
                    const char *ds = json_find_key_in_range(vis_start, vis_end, "deepstack_visual_indexes");
                    if (ds && *ds == '[') {
                        ds++; /* skip [ */
                        while (*ds && cfg->vis_ds_count < 4) {
                            while (*ds == ' ' || *ds == ',') ds++;
                            if (*ds == ']') break;
                            char *end2;
                            int v = (int)strtol(ds, &end2, 10);
                            if (end2 == ds) break;
                            cfg->vis_ds_idx[cfg->vis_ds_count++] = v;
                            ds = end2;
                        }
                    }
                }
                #undef VIS_GET_INT
            }
        }
    }

    /* Parse MRoPE sections from rope_scaling */
    cfg->mrope_n_sec = 0;
    if (cfg->has_mrope) {
        const char *mrope_pos = strstr(json, "\"mrope_section\"");
        if (mrope_pos) {
            mrope_pos = strchr(mrope_pos, '[');
            if (mrope_pos) {
                mrope_pos++; /* skip [ */
                while (*mrope_pos && cfg->mrope_n_sec < 4) {
                    while (*mrope_pos == ' ' || *mrope_pos == ',') mrope_pos++;
                    if (*mrope_pos == ']') break;
                    char *end2;
                    int v = (int)strtol(mrope_pos, &end2, 10);
                    if (end2 == mrope_pos) break;
                    cfg->mrope_sections[cfg->mrope_n_sec++] = v;
                    mrope_pos = end2;
                }
            }
        }
    }

    if (cfg->has_vision) {
        printf("[ST] Vision: depth=%d hidden=%d heads=%d ffn=%d patch=%d temporal=%d merge=%d out=%d\n",
               cfg->vis_depth, cfg->vis_hidden, cfg->vis_heads, cfg->vis_ffn,
               cfg->vis_patch, cfg->vis_temporal, cfg->vis_merge, cfg->vis_out_dim);
        if (cfg->vis_ds_count > 0) {
            printf("[ST]   DeepStack layers:");
            for (int i = 0; i < cfg->vis_ds_count; i++)
                printf(" %d", cfg->vis_ds_idx[i]);
            printf("\n");
        }
        if (cfg->mrope_n_sec > 0) {
            printf("[ST]   MRoPE sections:");
            for (int i = 0; i < cfg->mrope_n_sec; i++)
                printf(" %d", cfg->mrope_sections[i]);
            printf("\n");
        }
    }

    free(cfg_data);

    if (cfg->dim == 0 || cfg->n_layers == 0) {
        fprintf(stderr, "[ST] Failed to parse config.json\n");
        return -1;
    }

    printf("[ST] Config: dim=%d layers=%d heads=%d kv_heads=%d hd=%d ffn=%d vocab=%d\n",
           cfg->dim, cfg->n_layers, cfg->n_heads, cfg->n_kv_heads,
           cfg->head_dim, cfg->ffn_dim, cfg->vocab_size);
    printf("[ST]   rope_theta=%.1f norm_eps=%.1e q_norm=%d mrope=%d\n",
           cfg->rope_theta, cfg->norm_eps, cfg->has_q_norm, cfg->has_mrope);

    /* --- Parse safetensors index --- */
    char path_idx[1024];
    snprintf(path_idx, sizeof(path_idx), "%s/model.safetensors.index.json", model_dir);
    size_t idx_size;
    uint8_t *idx_data = read_file(path_idx, &idx_size);
    if (!idx_data) {
        /* Maybe single file: model.safetensors */
        snprintf(path_idx, sizeof(path_idx), "%s/model.safetensors", model_dir);
        if (st_access(path_idx, 0) != 0) {
            /* Third form: shard set model-NNNNN-of-NNNNN.safetensors (qwen3_moe 30B) */
            if (glob_model_shards(model_dir, cfg) != 0) {
                fprintf(stderr, "[ST] No model file found\n");
                return -1;
            }
        } else {
            /* Single file mode */
            cfg->n_files = 1;
            cfg->file_paths = calloc(1, sizeof(char*));
            cfg->file_paths[0] = strdup(path_idx);
            fprintf(stderr, "[ST] Single safetensors file: %s\n", path_idx);
        }
    } else {
        /* Multi-shard mode: parse weight_map from index */
        char *idx_json = (char*)idx_data;

        /* Count unique files in weight_map */
        const char *wm_start = strstr(idx_json, "\"weight_map\"");
        if (!wm_start) { free(idx_data); return -1; }

        /* Collect unique file names */
        char *files[MAX_SHARDS] = {0};
        int nf = 0;

        const char *p = wm_start;
        while ((p = strstr(p, "\": \"")) != NULL && nf < MAX_SHARDS) {
            p += 4; /* skip ":" " " "\"" to point at value content */
            const char *end = strchr(p, '"');
            if (!end) break;
            size_t flen = end - p;
            char *fname = malloc(flen + 1);
            memcpy(fname, p, flen);
            fname[flen] = '\0';

            /* Deduplicate */
            int found = 0;
            for (int i = 0; i < nf; i++) {
                if (strcmp(files[i], fname) == 0) { found = 1; break; }
            }
            if (!found) files[nf++] = fname;
            else free(fname);
            p = end + 1;
        }

        cfg->n_files = nf;
        cfg->file_paths = calloc(nf, sizeof(char*));
        for (int i = 0; i < nf; i++) {
            size_t plen = strlen(model_dir) + 1 + strlen(files[i]) + 1;
            cfg->file_paths[i] = malloc(plen);
            snprintf(cfg->file_paths[i], plen, "%s/%s", model_dir, files[i]);
            free(files[i]);
        }

        fprintf(stderr, "[ST] Found %d safetensors shard(s)\n", nf);
        free(idx_data);
    }

    /* --- Build tensor info: open each file, parse header, collect entries --- */
    int n_tensors_total = 0;
    cfg->tensors = calloc(MAX_TENSORS, sizeof(STTensorInfo));

    for (int fi = 0; fi < cfg->n_files; fi++) {
        char *header = read_st_header(cfg->file_paths[fi]);
        if (!header) {
            fprintf(stderr, "[ST] WARN: cannot read header[%d]: %s\n", fi, cfg->file_paths[fi]);
            continue;
        }

        size_t hdr_size = strlen(header);

        /* Parse header JSON: {"weight1": {"dtype":"BF16","shape":[4096,4096],"data_offsets":[0,33554432]}, ...} */
        const char *hp = header;
        int64_t data_offset = 8 + (int64_t)hdr_size;  /* 8-byte header_size field + JSON */

        while (*hp) {
            /* Find next key (tensor name) */
            hp = strchr(hp, '"');
            if (!hp) break;
            hp++; /* skip opening " */
            const char *name_start = hp;
            hp = strchr(hp, '"');
            if (!hp) break;
            size_t name_len = hp - name_start;
            hp++; /* skip closing " */
            hp = strchr(hp, '{'); /* find value object */
            if (!hp) break;
            hp++;

            STTensorInfo *ti = &cfg->tensors[n_tensors_total];

            /* name */
            ti->name = malloc(name_len + 1);
            memcpy(ti->name, name_start, name_len);
            ti->name[name_len] = '\0';

            /* dtype */
            const char *dp = strstr(hp, "\"dtype\"");
            if (dp) {
                dp = strchr(dp + 7, '"');
                if (dp) {
                    dp++;
                    if (strncmp(dp, "F32", 3) == 0) ti->dtype = ST_DTYPE_F32;
                    else if (strncmp(dp, "F16", 3) == 0) ti->dtype = ST_DTYPE_F16;
                    else if (strncmp(dp, "BF16", 4) == 0) ti->dtype = ST_DTYPE_BF16;
                }
            }

            /* shape */
            const char *sp = strstr(hp, "\"shape\"");
            if (sp) {
                sp = strchr(sp, '[');
                if (sp) {
                    sp++;
                    const char *sep;
                    int di = 0;
                    while (*sp != ']' && di < 8) {
                        ti->ne[di] = strtoll(sp, (char**)&sep, 10);
                        di++; sp = sep;
                        while (*sp == ',' || *sp == ' ') sp++;
                    }
                    /* Skip any extra dimensions beyond 4 */
                    while (*sp != ']') {
                        strtoll(sp, (char**)&sep, 10);
                        sp = sep;
                        while (*sp == ',' || *sp == ' ') sp++;
                    }
                    ti->n_dims = di;
                }
            }

            ti->n_elems = 1;
            for (int d = 0; d < ti->n_dims; d++) ti->n_elems *= ti->ne[d];

            /* data_offsets */
            const char *op = strstr(hp, "\"data_offsets\"");
            if (op) {
                op = strchr(op, '[');
                if (op) {
                    op++;
                    char *oe;
                    ti->file_offsets[0] = strtoll(op, &oe, 10);
                    ti->file_offsets[1] = strtoll(oe + 1, NULL, 10);
                }
            }

            ti->n_bytes = ti->file_offsets[1] - ti->file_offsets[0];

            /* Store which file this tensor is in */
            /* We encode file index in the upper bits of file_offsets[0] */
            /* Actually, we need to save file index. Let's use a separate field or encode it. */
            /* For now, store file index in n_bytes's negative range... no. */
            /* Append a special field: we'll look up by name and scan files later. */
            /* Easiest: store file index as offset bias. We'll compute absolute offset later. */
            /* For now, save the file path directly. */
            /* Actually, we already have cfg->file_paths[fi]. Just need to remember fi. */
            /* Encode shard index: file_offsets[0] += FI_BASE * fi.
             * FI_BASE = 1e15: any shard data offset <= 4e9 << 1e15, and
             * 32 shards -> 3.2e16 < INT64_MAX (the old 1e18 overflowed at 10+ shards). */
            ti->file_offsets[0] += (int64_t)fi * 1000000000000000LL;

            n_tensors_total++;
            if (n_tensors_total >= MAX_TENSORS) break;

            /* Find closing "}" */
            hp = strchr(hp, '}');
            if (hp) hp++;
        }

        free(header);
    }

    cfg->n_tensors = n_tensors_total;
    fprintf(stderr, "[ST] Total %d tensors across %d file(s)\n", n_tensors_total, cfg->n_files);

    /* Detect architecture features from tensor names */
    for (int i = 0; i < n_tensors_total; i++) {
        if (strstr(cfg->tensors[i].name, "q_norm")) cfg->has_q_norm = 1;
        if (strstr(cfg->tensors[i].name, "mrope")) cfg->has_mrope = 1;
    }

    return 0;
}

const STTensorInfo *find_tensor(const STModelConfig *cfg, const char *name) {
    for (int i = 0; i < cfg->n_tensors; i++) {
        if (strcmp(cfg->tensors[i].name, name) == 0) return &cfg->tensors[i];
    }
    return NULL;
}

/* Extract file index encoded in file_offsets[0] */
static int get_file_index(const STTensorInfo *ti) {
    return (int)(ti->file_offsets[0] / 1000000000000000LL);
}

static int64_t get_file_offset(const STTensorInfo *ti) {
    return ti->file_offsets[0] % 1000000000000000LL;
}

/* ================================================================
 * st_load_tensor: Load a single tensor from safetensors files
 * ================================================================ */
/* st_load_tensor / slice (src 1745-1887) */
int st_load_tensor(const STModelConfig *cfg, const char *tensor_name,
                    float *dst, int expected_elems) {
    const STTensorInfo *ti = find_tensor(cfg, tensor_name);
    if (!ti) {
        fprintf(stderr, "[ST] Tensor not found: %s\n", tensor_name);
        return -1;
    }
    if (ti->n_elems != expected_elems && expected_elems > 0) {
        fprintf(stderr, "[ST] Size mismatch for %s: expected %d, got %lld\n",
                tensor_name, expected_elems, (long long)ti->n_elems);
    }

    int fi = get_file_index(ti);
    int64_t file_off = get_file_offset(ti);

    if (fi < 0 || fi >= cfg->n_files) {
        fprintf(stderr, "[ST] Invalid file index %d for %s\n", fi, tensor_name);
        return -1;
    }

    /* ====== mmap fast-path: map file once, reuse across tensor reads ======
     * Portable implementation (vllm_platform.h): CreateFileW+MapViewOfFile on
     * Windows, open+mmap on Linux/RK3588. */
    static st_mmap_t mc = {0};

    if (mc.fi != fi || !mc.data) {
        st_mmap_close(&mc);
        if (st_mmap_open(&mc, fi, cfg->file_paths[fi]) != 0) {
            fprintf(stderr, "[ST] mmap: Cannot open %s\n", cfg->file_paths[fi]);
            return -1;
        }
    }

    /* Use fread to get header_size first (8 bytes at file start) */
    int64_t data_offset = 0;
    {
        FILE *fh = st_fopen(cfg->file_paths[fi], "rb");
        if (!fh) return -1;
        uint64_t header_size = 0;
        int ok = (fread(&header_size, 8, 1, fh) == 1);
        fclose(fh);
        if (!ok) return -1;
        data_offset = 8 + (int64_t)header_size;
    }

    int64_t abs_off = data_offset + file_off;
    int64_t n_elems = ti->n_elems;
    const uint8_t *raw = (const uint8_t *)mc.data + abs_off;

    /* Convert and copy to destination */
    switch (ti->dtype) {
    case ST_DTYPE_F32:
        memcpy(dst, raw, (size_t)n_elems * 4);
        break;
    case ST_DTYPE_F16:
        f16_to_f32_array(dst, raw, n_elems);
        break;
    case ST_DTYPE_BF16:
        bf16_to_f32_array(dst, raw, n_elems);
        break;
    default:
        memset(dst, 0, (size_t)n_elems * 4);
        break;
    }

    return 0;
}

int st_load_tensor_slice(const STModelConfig *cfg, const char *tensor_name,
                         int64_t elem_offset, float *dst, int64_t elem_count) {
    const STTensorInfo *ti = find_tensor(cfg, tensor_name);
    if (!ti) {
        fprintf(stderr, "[ST] Tensor not found: %s\n", tensor_name);
        return -1;
    }
    if (elem_offset < 0) elem_offset = 0;
    if (elem_count < 0) elem_count = 0;
    if (elem_offset > ti->n_elems) elem_offset = ti->n_elems;
    if (elem_offset + elem_count > ti->n_elems) elem_count = ti->n_elems - elem_offset;

    int fi = get_file_index(ti);
    int64_t file_off = get_file_offset(ti);

    if (fi < 0 || fi >= cfg->n_files) {
        fprintf(stderr, "[ST] Invalid file index %d for %s\n", fi, tensor_name);
        return -1;
    }

    static st_mmap_t mc = {0};

    if (mc.fi != fi || !mc.data) {
        st_mmap_close(&mc);
        if (st_mmap_open(&mc, fi, cfg->file_paths[fi]) != 0) {
            fprintf(stderr, "[ST] mmap: Cannot open %s\n", cfg->file_paths[fi]);
            return -1;
        }
    }

    int64_t data_offset = 0;
    {
        FILE *fh = st_fopen(cfg->file_paths[fi], "rb");
        if (!fh) return -1;
        uint64_t header_size = 0;
        int ok = (fread(&header_size, 8, 1, fh) == 1);
        fclose(fh);
        if (!ok) return -1;
        data_offset = 8 + (int64_t)header_size;
    }

    int dtype_bytes = 4;
    if (ti->dtype == ST_DTYPE_F16 || ti->dtype == ST_DTYPE_BF16) dtype_bytes = 2;
    int64_t abs_off = data_offset + file_off + elem_offset * (int64_t)dtype_bytes;
    const uint8_t *raw = (const uint8_t *)mc.data + abs_off;

    switch (ti->dtype) {
    case ST_DTYPE_F32:
        memcpy(dst, raw, (size_t)elem_count * 4);
        break;
    case ST_DTYPE_F16:
        f16_to_f32_array(dst, raw, elem_count);
        break;
    case ST_DTYPE_BF16:
        bf16_to_f32_array(dst, raw, elem_count);
        break;
    default:
        memset(dst, 0, (size_t)elem_count * 4);
        break;
    }

    return 0;
}

/* ================================================================
 * st_load_layer_weights: Load weights for a specific layer
 * ================================================================ */

/* Forward declarations for Q8_0 functions (defined later) */
void f32_to_q8_0(uint8_t *q8_out, const float *f32_in, int n_elements);
void f32_to_q4_0(uint8_t *q4_out, const float *f32_in, int n_elements);
void f32_to_q2_1(uint8_t *q2_out, const float *f32_in, int n_elements);
static void quantize_row_q8_0_act(const float *__restrict x,
                                  int8_t *__restrict q, float *__restrict d, int cols);

/* st_config_free (src 2829-2835) */
void st_config_free(STModelConfig *cfg) {
    for (int i = 0; i < cfg->n_tensors; i++) free(cfg->tensors[i].name);
    free(cfg->tensors);
    for (int i = 0; i < cfg->n_files; i++) free(cfg->file_paths[i]);
    free(cfg->file_paths);
    memset(cfg, 0, sizeof(*cfg));
}


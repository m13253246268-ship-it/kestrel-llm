/**
 * vllm_media.c - Real Image & Video Frame Loading
 *
 * Image decoding via stb_image (single-file public-domain decoder):
 * JPEG, PNG, BMP, PPM, TGA, GIF, WebP, ...
 *
 * Video: loads image sequences from a directory or base64 frame arrays.
 */

#include "vllm_media.h"
#include "vllm_util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* stb_image implementation (single-file, zero external deps).
 * Kept under this TU so the decoder is compiled exactly once. */
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STBI_NO_GIF
#define STBI_NO_TGA
#include "stb_image.h"

#include <dirent.h>

/* ================================================================
 * Public API
 * ================================================================ */

uint8_t* media_load_image(const char *path, int *out_w, int *out_h) {
    int w = 0, h = 0, ch = 0;
    uint8_t *rgb = stbi_load(path, &w, &h, &ch, 3);
    if (!rgb) {
        fprintf(stderr, "[MEDIA] Failed to decode image: %s\n", path);
        return NULL;
    }
    *out_w = w;
    *out_h = h;
    return rgb;   /* stbi_free == free for standard allocator */
}

uint8_t* media_load_image_memory(const uint8_t *data, size_t len,
                                 int *out_w, int *out_h) {
    int w = 0, h = 0, ch = 0;
    uint8_t *rgb = stbi_load_from_memory(data, (int)len, &w, &h, &ch, 3);
    if (!rgb) {
        fprintf(stderr, "[MEDIA] Failed to decode image payload (%zu bytes)\n", len);
        return NULL;
    }
    *out_w = w;
    *out_h = h;
    return rgb;
}

uint8_t* media_base64_decode(const char *b64, size_t *out_len) {
    static const int8_t T[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
        52,53,54,55,56,57,58,59,60,61,-1,-1,-1, 0,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
    };
    size_t len = strlen(b64);
    if (len == 0) { if (out_len) *out_len = 0; return (uint8_t *)calloc(1, 1); }
    size_t cap = len / 4 * 3 + 3;
    uint8_t *out = (uint8_t *)malloc(cap);
    if (!out) return NULL;
    size_t o = 0, i = 0;
    uint32_t acc = 0;
    int nbits = 0;
    int bad = 0;
    for (i = 0; i < len; i++) {
        char c = b64[i];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
        if (c == '=') break;
        int8_t v = (c < 128) ? T[(uint8_t)c] : -1;
        if (v < 0) { bad = 1; break; }
        acc = (acc << 6) | (uint32_t)v;
        nbits += 6;
        if (nbits >= 8) {
            nbits -= 8;
            out[o++] = (uint8_t)((acc >> nbits) & 0xFF);
        }
    }
    if (bad) { free(out); return NULL; }
    if (out_len) *out_len = o;
    return out;
}

void media_free_image(uint8_t *data) {
    free(data);
}

/* ================================================================
 * Bilinear Resize
 * ================================================================ */

void media_resize_rgb(uint8_t *dst, const uint8_t *src,
                       int src_w, int src_h, int dst_w, int dst_h) {
    for (int y = 0; y < dst_h; y++) {
        float fy = (float)y * (float)(src_h - 1) / (float)(dst_h > 1 ? dst_h - 1 : 1);
        int iy = (int)fy;
        float dy = fy - (float)iy;
        int iy1 = iy + 1;
        if (iy1 >= src_h) iy1 = src_h - 1;

        for (int x = 0; x < dst_w; x++) {
            float fx = (float)x * (float)(src_w - 1) / (float)(dst_w > 1 ? dst_w - 1 : 1);
            int ix = (int)fx;
            float dx = fx - (float)ix;
            int ix1 = ix + 1;
            if (ix1 >= src_w) ix1 = src_w - 1;

            int dst_idx = (y * dst_w + x) * 3;
            for (int c = 0; c < 3; c++) {
                float v00 = (float)src[(iy * src_w + ix) * 3 + c];
                float v10 = (float)src[(iy * src_w + ix1) * 3 + c];
                float v01 = (float)src[(iy1 * src_w + ix) * 3 + c];
                float v11 = (float)src[(iy1 * src_w + ix1) * 3 + c];

                float v0 = v00 + (v10 - v00) * dx;
                float v1 = v01 + (v11 - v01) * dx;
                float v = v0 + (v1 - v0) * dy;

                if (v < 0.0f) v = 0.0f;
                if (v > 255.0f) v = 255.0f;
                dst[dst_idx + c] = (uint8_t)(v + 0.5f);
            }
        }
    }
}

/* ================================================================
 * Video Frame Loading
 * ================================================================ */

static int is_image_ext(const char *ext) {
    if (!ext) return 0;
    return (strcasecmp(ext, ".bmp") == 0 ||
            strcasecmp(ext, ".ppm") == 0 ||
            strcasecmp(ext, ".pgm") == 0 ||
            strcasecmp(ext, ".png") == 0 ||
            strcasecmp(ext, ".jpg") == 0 ||
            strcasecmp(ext, ".jpeg") == 0 ||
            strcasecmp(ext, ".tga") == 0);
}

uint8_t** media_load_video_frames(const char *dir,
                                   int *out_w, int *out_h, int *out_n,
                                   int max_frames) {
    DIR *d = opendir(dir);
    if (!d) {
        fprintf(stderr, "[MEDIA] Cannot open directory: %s\n", dir);
        return NULL;
    }

    char **names = NULL;
    int name_cap = 0, name_cnt = 0;

    struct dirent *entry;
    while ((entry = readdir(d)) != NULL) {
        const char *ext = strrchr(entry->d_name, '.');
        if (!is_image_ext(ext)) continue;

        if (name_cnt >= name_cap) {
            name_cap = name_cap ? name_cap * 2 : 64;
            names = (char**)realloc(names, (size_t)name_cap * sizeof(char*));
        }
        names[name_cnt] = strdup(entry->d_name);
        name_cnt++;
        if (max_frames > 0 && name_cnt >= max_frames) break;
    }
    closedir(d);

    if (name_cnt == 0) {
        fprintf(stderr, "[MEDIA] No image files found in: %s\n", dir);
        free(names);
        return NULL;
    }

    printf("[MEDIA] Found %d frame(s) in directory: %s\n", name_cnt, dir);
    fflush(stdout);

    for (int i = 0; i < name_cnt - 1; i++)
        for (int j = i + 1; j < name_cnt; j++)
            if (strcmp(names[i], names[j]) > 0) {
                char *tmp = names[i]; names[i] = names[j]; names[j] = tmp;
            }

    uint8_t **frames = (uint8_t**)calloc((size_t)name_cnt, sizeof(uint8_t*));
    int first_w = 0, first_h = 0, loaded = 0;

    for (int i = 0; i < name_cnt; i++) {
        char full_path[1024];
        snprintf(full_path, sizeof(full_path), "%s/%s", dir, names[i]);
        int w, h;
        uint8_t *data = media_load_image(full_path, &w, &h);
        if (!data) { continue; }
        if (loaded == 0) { first_w = w; first_h = h; }
        else if (w != first_w || h != first_h) { media_free_image(data); continue; }
        frames[loaded++] = data;
    }

    for (int i = 0; i < name_cnt; i++) free(names[i]);
    free(names);

    if (loaded == 0) { free(frames); return NULL; }
    *out_w = first_w; *out_h = first_h; *out_n = loaded;
    return frames;
}

void media_free_video_frames(uint8_t **frames, int n_frames) {
    if (!frames) return;
    for (int i = 0; i < n_frames; i++) {
        free(frames[i]);
    }
    free(frames);
}

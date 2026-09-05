/**
 * vllm_utf8_utils.h - UTF-8 path utilities
 *
 * On Linux/aarch64 (RK3588) fopen/access already handle UTF-8 paths, so
 * the wrappers map directly to the libc functions.
 */

#ifndef VLLM_UTF8_UTILS_H
#define VLLM_UTF8_UTILS_H

#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

#define utf8_fopen(path, mode)   fopen(path, mode)
static inline int utf8_file_exists(const char *path) {
    return access(path, 0);
}

#endif /* VLLM_UTF8_UTILS_H */

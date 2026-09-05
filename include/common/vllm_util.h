/**
 * vllm_util.h - Utility functions for cross-platform compatibility
 *
 * On Linux/aarch64 (RK3588) fopen/access already handle UTF-8 paths, so
 * st_fopen/st_access map directly to the libc functions.
 */

#ifndef VLLM_UTIL_H
#define VLLM_UTIL_H

#include <stdio.h>
#include <unistd.h>

/* fopen already handles UTF-8 paths on Linux */
#define st_fopen   fopen
#define st_access  access

#endif /* VLLM_UTIL_H */

#!/bin/bash
# ================================================================
#  vqf_convert 一键编译（Linux / x86_64 gcc）
#  产物: ./vqf_conv
# ================================================================
set -e
cd "$(dirname "$0")"

CC="${CC:-gcc}"
echo "[build] vqf_convert (x86_64) -O2 ..."
$CC -O2 -Wall -Wextra -I src \
    src/conv_main.c src/model_cfgio.c src/model_layers.c \
    src/quant_kernels.c src/qk_repack.c src/vqf_vision.c \
    src/vllm_crypto.c \
    -o vqf_conv -lm

echo "[ok] 产物: $(pwd)/vqf_conv"
./vqf_conv --help

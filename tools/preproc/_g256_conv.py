#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""后台转换 g256 VQF（2B F16 GGUF -> wmode g256），供 NPU prefill 实测。"""
import os
import paramiko

_board_host = os.environ.get('KESTREL_BOARD_HOST')
_board_pass = os.environ.get('KESTREL_BOARD_PASS')
if not _board_host or not _board_pass:
    raise SystemExit('[g256] 请先设置 KESTREL_BOARD_HOST / KESTREL_BOARD_PASS'
                     '（可选 KESTREL_BOARD_USER，默认 root）')

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(_board_host, 22, os.environ.get('KESTREL_BOARD_USER', 'root'), _board_pass, timeout=30)
sftp = c.open_sftp()
sftp.put(os.path.abspath(''), '/tmp/x') if False else None
sftp.close()
# 用板端脚本方式避免 paramiko 卡死
script = """#!/bin/bash
mkdir -p /mnt/emmc/vqf_g256
cd /mnt/emmc/vqf_g256
if [ -f model.vqf ]; then echo "EXISTS"; exit 0; fi
nohup /mnt/emmc/RK3588/vllm_shs --convert-gguf /mnt/emmc/vqf_g256/model.vqf \
  --model /mnt/emmc/llama_cpp/models/Qwen3-VL-2B-F16.gguf --wmode g256 \
  > /tmp/conv_g256.log 2>&1 < /dev/null &
disown
echo "CONV_STARTED pid=$!"
"""
sftp = c.open_sftp()
with sftp.open('/tmp/conv_g256.sh', 'w') as f:
    f.write(script)
sftp.close()
_, o, e = c.exec_command('bash /tmp/conv_g256.sh; echo RC=$?', timeout=20)
print(o.read().decode(errors='replace') + e.read().decode(errors='replace'))
c.close()

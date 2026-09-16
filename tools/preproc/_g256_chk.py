#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import paramiko

_board_host = os.environ.get('KESTREL_BOARD_HOST')
_board_pass = os.environ.get('KESTREL_BOARD_PASS')
if not _board_host or not _board_pass:
    raise SystemExit('[g256] 请先设置 KESTREL_BOARD_HOST / KESTREL_BOARD_PASS'
                     '（可选 KESTREL_BOARD_USER，默认 root）')

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(_board_host, 22, os.environ.get('KESTREL_BOARD_USER', 'root'), _board_pass, timeout=30)
_, o, e = c.exec_command("ps aux | grep convert | grep -v grep; echo ===; "
                         "tail -12 /tmp/conv_g256.log 2>/dev/null; echo ===; "
                         "ls -la /mnt/emmc/vqf_g256/ 2>/dev/null", timeout=30)
print(o.read().decode(errors='replace') + e.read().decode(errors='replace'))
c.close()

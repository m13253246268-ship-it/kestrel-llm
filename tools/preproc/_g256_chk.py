#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import paramiko
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.0.60', 22, 'root', 'orangepi', timeout=30)
_, o, e = c.exec_command("ps aux | grep convert | grep -v grep; echo ===; "
                         "tail -12 /tmp/conv_g256.log 2>/dev/null; echo ===; "
                         "ls -la /mnt/emmc/vqf_g256/ 2>/dev/null", timeout=30)
print(o.read().decode(errors='replace') + e.read().decode(errors='replace'))
c.close()

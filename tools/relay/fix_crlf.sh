#!/bin/bash
# fix_crlf.sh — 把代码树的 CRLF 统一为 LF。
#
# 场景：从 Windows 导出/拷贝到 Linux 或板端后，.sh 因 CRLF 报
#       "bad interpreter: /bin/bash^M"，脚本无法执行。
#
# 用法：
#   bash tools/relay/fix_crlf.sh              # 默认处理仓库根（脚本所在目录的上两级）
#   bash tools/relay/fix_crlf.sh <目录>        # 处理指定目录
#
# 说明：只转换「文本文件」（grep -I 会跳过二进制文件），.git/ 一律不动。
set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
D=${1:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
cd "$D" || exit 1

CR=$(printf '\r')
LIST=$(grep -rIlU --exclude-dir=.git "$CR" . 2>/dev/null || true)

echo "dir         = $(pwd)"
echo "total_files = $(find . -path ./.git -prune -o -type f -print | wc -l)"
echo "crlf_before = $(printf '%s\n' "${LIST:-}" | grep -c . || true)"

if [ -n "${LIST:-}" ]; then
    printf '%s\n' "$LIST" | while IFS= read -r f; do
        [ -n "$f" ] && sed -i 's/\r$//' "$f"
    done
fi

echo "crlf_after  = $(grep -rIlU --exclude-dir=.git "$CR" . 2>/dev/null | grep -c . || true)"
echo "--- sample (should print 'ASCII text' / 'UTF-8 text', not 'with CRLF') ---"
file build_rk3588.sh vqf_convert/build.sh tools/relay/fix_crlf.sh

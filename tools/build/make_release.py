#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_release.py —— 从仓库当前提交导出「发布门面」快照

为什么需要它：`_release_verify/{gitee,github}` 此前是**人肉导出**的
（`git archive` → 拷到板端编译自检 → 回拷本机），既没有脚本、也没有把规则写下来，
结果是两份快照世代混乱（tools 已归档、admin 三轨已在，但 ARM include 未修、
`tok_ref_check.c` 缺、README/wiki 停在 09-12），并且导出树被 Windows 侧
`core.autocrlf=true` 整树 CRLF 化，与仓库「逐文件全不同」——那才是"56/62 不同"的真因。
本脚本把规则固化成一条可复现的命令。

规则
----
1. 内容 = `git archive HEAD`，即**仅已跟踪文件**、且为**仓库内存储形态（LF）**：
   不导出 `.git/`、构建产物（`build*/`）、`__pycache__/`、以及任何未跟踪文件。
2. `gitee` 门面：原样。Gitee 是正式站点，不改写任何站点 URL。
3. `github` 门面：**只做 i18n 互换**（GitHub 侧以英文作门面）：
       README.en.md  ->  README.md
       README.md     ->  README.zh-CN.md
   `wiki/Home.md` 里的两条 README 链接同步改写。
4. **不删任何内容**：`docs/bench/` 的原始基准证据按 `.gitignore` 的注释
   「发布证据要随仓库分发（引擎日志是行为证据）」保留，`CHANGELOG.md` 同样保留。
5. **不清除** `/mnt/...` 等部署路径：那是面向板端的操作说明（本引擎跑在 RK3588 上），
   不是本机路径泄露；`tools/bench/bench_value.sh` 等历史默认路径亦照原样导出。

用法
----
    python3 tools/build/make_release.py --out <目录> --facade gitee
    python3 tools/build/make_release.py --out <目录> --facade github
    python3 tools/build/make_release.py --out <目录> --facade github --clean

    --out 指向已存在且非空的目录时，必须加 --clean 才会清空重建。
"""
import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

# i18n 互换的精确匹配（改前 -> 改后），全部为整行替换，命中数必须是 1
GH_README_EN_LINE = ("[简体中文](README.md) | **English**",
                     "[简体中文](README.zh-CN.md) | **English**")
GH_README_ZH_LINE = ("[English](README.en.md) | **简体中文**",
                     "[English](README.md) | **简体中文**")
GH_HOME_LINE = ("- [README.md](../README.md) ｜ [README.en.md](../README.en.md)",
                "- [README.md](../README.md) ｜ [简体中文](../README.zh-CN.md)")


def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def export_head(root, dest):
    """git archive HEAD -> dest（纯 stdlib 解包，LF 按仓库存储形态保留）"""
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tf:
        tar_path = tf.name
    try:
        subprocess.run(["git", "-C", root, "archive", "--format=tar",
                        "-o", tar_path, "HEAD"], check=True)
        with tarfile.open(tar_path, "r:") as t:
            try:
                t.extractall(dest, filter="data")   # Python >= 3.12
            except TypeError:
                t.extractall(dest)
    finally:
        if os.path.exists(tar_path):
            os.remove(tar_path)


def sub_once(path, old, new):
    with open(path, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    n = text.count(old)
    if n != 1:
        raise SystemExit("[make_release] 期望 %r 恰好出现 1 次，实际 %d 次：%s"
                         % (old, n, path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text.replace(old, new))
    return n


def make_github_facade(dest):
    """英文作门面：README.en.md -> README.md；README.md -> README.zh-CN.md"""
    en = os.path.join(dest, "README.en.md")
    zh = os.path.join(dest, "README.md")
    for p in (en, zh):
        if not os.path.exists(p):
            raise SystemExit("[make_release] 缺少 %s" % p)
    tmp = os.path.join(dest, "README.zh-CN.md")
    os.replace(zh, tmp)          # 中文 -> README.zh-CN.md
    os.replace(en, os.path.join(dest, "README.md"))  # 英文 -> README.md（门面）
    changed = []
    changed.append(("README.md", sub_once(os.path.join(dest, "README.md"),
                                          *GH_README_EN_LINE)))
    changed.append(("README.zh-CN.md", sub_once(tmp, *GH_README_ZH_LINE)))
    changed.append(("wiki/Home.md", sub_once(os.path.join(dest, "wiki", "Home.md"),
                                             *GH_HOME_LINE)))
    return changed


def main():
    ap = argparse.ArgumentParser(description="导出发布门面快照（gitee / github）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--facade", required=True, choices=["gitee", "github"])
    ap.add_argument("--clean", action="store_true", help="输出目录非空时先清空")
    args = ap.parse_args()

    root = repo_root()
    dest = os.path.abspath(args.out)
    if os.path.isdir(dest) and os.listdir(dest):
        if not args.clean:
            raise SystemExit("[make_release] %s 非空；加 --clean 才能清空重建" % dest)
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)

    head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    tree = subprocess.run(["git", "-C", root, "rev-parse", "HEAD^{tree}"],
                          check=True, capture_output=True, text=True).stdout.strip()
    export_head(root, dest)

    changed = []
    if args.facade == "github":
        changed = make_github_facade(dest)

    n_files = sum(len(fs) for _, _, fs in os.walk(dest))
    print("[make_release] facade = %s" % args.facade)
    print("[make_release] source = HEAD %s (tree %s)" % (head[:12], tree[:12]))
    print("[make_release] out    = %s" % dest)
    print("[make_release] files  = %d" % n_files)
    for name, cnt in changed:
        print("[make_release] i18n   = %s (替换 %d 处)" % (name, cnt))
    return 0


if __name__ == "__main__":
    sys.exit(main())

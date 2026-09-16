#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_release.py —— 从仓库当前提交导出「发布门面」快照

背景（先厘清，别搞错）：`_release_verify/{gitee,github}` **不是**导出树，而是**两份验证克隆**
（分别 clone Gitee 与 GitHub 远程），用途是核对「站点上真正发布出去的内容」。
而 GitHub 侧并非 master 的镜像 —— 它由本仓库的 **`gh` 门面分支**承载（GitHub 以英文作门面），
靠定期把 master 合并进 `gh` 来维护。门面约定此前只活在提交历史里，没有可执行定义；
本脚本把它固化成一条可复现命令：既能**生成**门面内容，也能用来**校验**
「`gh` 分支 == master 只差这一层 i18n 互换」。

规则
----
1. 内容 = `git -c core.autocrlf=false archive HEAD`，即**仅已跟踪文件**、且为
   **仓库内存储形态（LF）**；不导出 `.git/`、`build*/`、`__pycache__/`、未跟踪文件。
   （`core.autocrlf=false` 必须显式给：本仓库 `core.autocrlf=true` 且无 `.gitattributes`，
   否则 425 个文件里会有 251 个被转成 CRLF —— 那正是历史上导出树"整树 CRLF、
   与仓库逐文件全不同"的根因。）
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


def export_head(root, ref, dest):
    """git archive <ref> -> dest

    注意两件事：
    1) 必须显式 `core.autocrlf=false`。本仓库 core.autocrlf=true 且无 .gitattributes，
       否则 `git archive` 会把内容按 CRLF 导出 —— 实测 425 个文件里 251 个被转成 CRLF，
       与仓库内 blob 的 LF 形态不一致。这正是历史上 `_release_verify` 导出树"整树 CRLF、
       与仓库逐文件全不同"的根因。
    2) 必须显式给 ref。若依赖 HEAD，脚本行为会随「当前检出哪个分支」而变 ——
       在 `gh` 分支上跑 `--facade github` 会因缺 README.en.md 直接失败。
    """
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tf:
        tar_path = tf.name
    try:
        subprocess.run(["git", "-C", root, "-c", "core.autocrlf=false",
                        "archive", "--format=tar", "-o", tar_path, ref], check=True)
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
    ap.add_argument("--ref", default="master", help="导出哪个 ref（默认 master）")
    ap.add_argument("--clean", action="store_true", help="输出目录非空时先清空")
    args = ap.parse_args()

    root = repo_root()
    dest = os.path.abspath(args.out)
    if os.path.isdir(dest) and os.listdir(dest):
        if not args.clean:
            raise SystemExit("[make_release] %s 非空；加 --clean 才能清空重建" % dest)
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)

    head = subprocess.run(["git", "-C", root, "rev-parse", args.ref],
                          check=True, capture_output=True, text=True).stdout.strip()
    tree = subprocess.run(["git", "-C", root, "rev-parse", args.ref + "^{tree}"],
                          check=True, capture_output=True, text=True).stdout.strip()
    export_head(root, args.ref, dest)

    changed = []
    if args.facade == "github":
        changed = make_github_facade(dest)

    n_files = sum(len(fs) for _, _, fs in os.walk(dest))
    print("[make_release] facade = %s" % args.facade)
    print("[make_release] source = %s %s (tree %s)" % (args.ref, head[:12], tree[:12]))
    print("[make_release] out    = %s" % dest)
    print("[make_release] files  = %d" % n_files)
    for name, cnt in changed:
        print("[make_release] i18n   = %s (替换 %d 处)" % (name, cnt))
    return 0


if __name__ == "__main__":
    sys.exit(main())

<#
  wiki_facade.ps1 —— GitHub 门面变换：把指向 Gitee Wiki 的绝对链接改写为仓内相对链接

  背景：仓库 wiki/*.md 是 Gitee Wiki 的内容源，页内互链写成 Gitee 绝对地址
        https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/<页>
        这在 Gitee 上是必需的；但同一批文件也随仓库分发到 GitHub，
        在 GitHub 上这些链接会把读者送出站（且目标是中文 wiki）。

  本脚本只应在 **gh 门面分支** 上运行（master 侧链接必须保持 Gitee 绝对地址），改写规则：

      仓库内 wiki/*.md 里的链接      ->  ./<页>.md          （同目录相对链接）
      其它位置（README.zh-CN.md 等） ->  wiki/<页>.md

  用法：
      powershell -ExecutionPolicy Bypass -File tools/ops/wiki_facade.ps1
      powershell -ExecutionPolicy Bypass -File tools/ops/wiki_facade.ps1 -Check

  -Check 只校验：发现仍存在 Gitee wiki 链接即列出并以退出码 1 结束（同步后把关用）。
  幂等：改写过的文件再跑一次不会产生变化。
#>
[CmdletBinding()]
param(
    [switch]$Check,
    [switch]$Force,
    [string]$Root = (Get-Location).Path
)

$ErrorActionPreference = 'Stop'
$giteePrefix = 'https://gitee.com/pei-xiaoguang/kestrel-llm/wikis/'
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

# 分支守卫：master 侧的链接是 Gitee Wiki 的内容源，绝不能被改写
if (-not $Check -and -not $Force) {
    $branch = (& git -C $Root rev-parse --abbrev-ref HEAD 2>$null)
    if ($branch -ne 'gh') {
        throw "当前分支为 '$branch'；本脚本只应在 gh 门面分支运行（确需强制请加 -Force）。"
    }
}

# 目标文件：仓库内全部被跟踪的 .md
$tracked = & git -C $Root ls-files '*.md'
if (-not $tracked) { throw "未在 $Root 找到被跟踪的 .md 文件" }
$wikiDir = (Join-Path $Root 'wiki') + [IO.Path]::DirectorySeparatorChar

# 页面名取自 wiki/ 目录（实际上盘的页），长的先替换，避免名字互为前缀
$pages = @(Get-ChildItem (Join-Path $Root 'wiki') -Filter *.md -File |
           ForEach-Object { $_.BaseName } | Sort-Object -Property Length -Descending)
if ($pages.Count -eq 0) { throw "未在 $Root\wiki 找到 wiki 页面" }

$changed = 0
$violations = @()

foreach ($rel in $tracked) {
    $full = Join-Path $Root $rel
    if (-not (Test-Path $full)) { continue }
    $text = [IO.File]::ReadAllText($full, [Text.Encoding]::UTF8)
    if ($text.IndexOf($giteePrefix, [StringComparison]::Ordinal) -lt 0) { continue }

    if ($Check) {
        $n = ([regex]::Matches($text, [regex]::Escape($giteePrefix))).Count
        $violations += ("{0}  ({1} 处)" -f ($rel -replace '\\', '/'), $n)
        continue
    }

    $inWiki = $full.StartsWith($wikiDir, [StringComparison]::OrdinalIgnoreCase)
    $new = $text
    foreach ($p in $pages) {
        $to = if ($inWiki) { "./$p.md" } else { "wiki/$p.md" }
        $new = $new.Replace($giteePrefix + $p, $to)
    }

    if ($new -ne $text) {
        [IO.File]::WriteAllText($full, $new, $utf8NoBom)
        $changed++
        Write-Host ("rewritten: " + ($rel -replace '\\', '/'))
    }
    $left = ([regex]::Matches($new, [regex]::Escape($giteePrefix))).Count
    if ($left -gt 0) { $violations += ("{0}  (残留 {1} 处，页面名可能不在 wiki/ 内)" -f ($rel -replace '\\', '/'), $left) }
}

if ($Check) {
    if ($violations.Count -gt 0) {
        Write-Host "FAIL: 仍存在指向 Gitee Wiki 的绝对链接（GitHub 门面应改为仓内相对链接）："
        $violations | ForEach-Object { Write-Host ("  " + $_) }
        exit 1
    }
    Write-Host "OK: 仓内未发现 Gitee Wiki 绝对链接"
    exit 0
}

Write-Host ("done: 改写 {0} 个文件" -f $changed)
if ($violations.Count -gt 0) {
    Write-Host "WARN: 以下文件仍有残留（对应页面不在 wiki/ 目录内，请人工确认）："
    $violations | ForEach-Object { Write-Host ("  " + $_) }
    exit 1
}
exit 0

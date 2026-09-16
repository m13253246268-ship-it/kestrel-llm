# collect_results.ps1 — 把某一层范围的跑通结果归档到指定目录，并生成 manifest.sha256
#
# 用法（仓库根目录）：
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\relay\collect_results.ps1 `
#       -From 0 -To 4 -LogDir .tmp_tok\relay_rerun -Dest results\L0-4
#
# 归档内容：chain/u{L}_*.ct + chain/u{L}r112_*.ct、logs/、meta.txt、timing.tsv、manifest.sha256
# 退出码：0 = 归档完成；1 = 有层缺产物

param(
    [int]$From = 0,
    [int]$To = 4,
    [string]$ChainDir = ".tmp_tok\chain",
    [string]$LogDir = ".tmp_tok\relay_rerun",
    [string]$Dest = "results\L0-4"
)

$ErrorActionPreference = "Stop"
# 仓库根自定位：从脚本所在目录向上找到含 src/ 的目录（本脚本位于 tools/relay/）
$Root = $PSScriptRoot
while ($Root -and -not (Test-Path (Join-Path $Root "src"))) { $Root = Split-Path -Parent $Root }
if (-not $Root) { $Root = (Get-Location).Path }
Set-Location $Root

New-Item -ItemType Directory -Force -Path "$Dest\chain", "$Dest\logs" | Out-Null

$missing = 0
for ($L = $From; $L -le $To; $L++) {
    foreach ($pre in @("u$L", "u${L}r112")) {
        $n = 0
        for ($t = 0; $t -lt 4; $t++) { for ($h = 0; $h -lt 2; $h++) {
            $src = Join-Path $ChainDir "${pre}_${t}_${h}.ct"
            if (Test-Path $src) { Copy-Item $src "$Dest\chain\" -Force; $n++ }
        } }
        if ($n -ne 8) { Write-Output ("WARN: ${pre} only $n/8 ct"); $missing++ }
    }
    foreach ($tag in @("lay$L", "boot$L")) {
        foreach ($ext in @("out", "err")) {
            $f = Join-Path $LogDir "$tag.$ext"
            if (Test-Path $f) { Copy-Item $f "$Dest\logs\" -Force }
        }
    }
    Write-Output ("collected layer " + $L)
}
foreach ($f in @("timing.tsv", "meta.txt", "master.log")) {
    $p = Join-Path $LogDir $f
    if (Test-Path $p) { Copy-Item $p $Dest -Force }
}

# manifest（相对路径，排除自身）
$outAbs = (Resolve-Path $Dest).Path
$lines = New-Object System.Collections.Generic.List[string]
$rel = Get-ChildItem -Recurse -File $Dest |
       Where-Object { $_.Name -ne "manifest.sha256" } |
       ForEach-Object { $_.FullName.Substring($outAbs.Length + 1) } | Sort-Object
foreach ($r in $rel) {
    $h = (Get-FileHash -Algorithm SHA256 -Path (Join-Path $outAbs $r) -ErrorAction SilentlyContinue).Hash
    if ($h) { $lines.Add($h.ToLower() + "  " + $r) }
}
$lines | Set-Content -Path "$Dest\manifest.sha256" -Encoding ascii

Write-Output "----------------------------------------"
Write-Output ("dest     : " + $Dest)
Write-Output ("files    : " + $lines.Count + "  (warnings: " + $missing + ")")
Write-Output ("manifest : " + $Dest + "\manifest.sha256")
exit $(if ($missing -eq 0) { 0 } else { 1 })

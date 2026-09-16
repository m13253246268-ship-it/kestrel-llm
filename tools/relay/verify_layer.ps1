# verify_layer.ps1 — 批量"算得对不对"校验：解密各层密文，与明文参考对照
#
# 原理：调用 .tmp_tok\verify_layer.exe（独立代码路径：解密 -> 解码 -> 对明文参考算 max|err|），
#       不是复用驱动的 PASS 判定，因此可作第三方独立复核。
#
# 用法（仓库根目录）：
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\relay\verify_layer.ps1 -From 0 -To 4            # 校验 lay 输出 u{L}
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\relay\verify_layer.ps1 -From 0 -To 3 -Mode boot # 校验 boot 输出 u{L}r112
#
# 依赖：① 数据包里的明文参考 tail/l{L}_u2_ref.bin（8192 个 float32）
#       ② sk.bin  ③ 引擎源码可编译（本脚本会在 exe 缺失时自动构建）
# 退出码：失败层数（0 = 全层通过）

param(
    [int]$From = 0,
    [int]$To = 4,
    [ValidateSet("lay", "boot")][string]$Mode = "lay",
    [string]$ChainDir = ".tmp_tok\chain",
    [string]$Sk = "",
    [double]$Tol = 3e-2,
    [string]$Exe = ".tmp_tok\verify_layer.exe"
)

$ErrorActionPreference = "Stop"
# 仓库根自定位：从脚本所在目录向上找到含 src/ 的目录（本脚本位于 tools/relay/）
$Root = $PSScriptRoot
while ($Root -and -not (Test-Path (Join-Path $Root "src"))) { $Root = Split-Path -Parent $Root }
if (-not $Root) { $Root = (Get-Location).Path }
Set-Location $Root
if (-not $Sk) { $Sk = Join-Path $ChainDir "sk.bin" }

# ---------- 构建（缺失时） ----------
if (-not (Test-Path $Exe)) {
    # 输出目录必须存在（gcc -o 不会自建）；全新克隆时 .tmp_tok 还不存在，这里补上
    $exeDir = Split-Path -Parent $Exe
    if ($exeDir) { New-Item -ItemType Directory -Force -Path $exeDir | Out-Null }
    Write-Output "building $Exe ..."
    gcc -O2 -fopenmp -Wno-implicit-function-declaration `
        -I include -I include/core -I include/common `
        -DCKKS_N=2048 -DCKKS_NPRIMES=112 -DBB=32 -DGG=32 `
        src/core/vllm_ntt.c src/core/vllm_ckks.c src/core/vllm_tp.c `
        tools/relay/verify_layer.c -o $Exe -lm
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Exe)) { Write-Output "BUILD FAILED"; exit 99 }
}

Write-Output "mode=$Mode  layers=$From..$To  chain=$ChainDir  sk=$Sk  tol=$Tol"
Write-Output "--------------------------------------------------------------------------"
$failed = 0
$rows = @()
for ($L = $From; $L -le $To; $L++) {
    $pre = if ($Mode -eq "boot") { "u${L}r112" } else { "u$L" }
    $out = & $Exe -L $L -Ct $pre -Dir $ChainDir -Sk $Sk -Tol $Tol 2>&1
    $rc = $LASTEXITCODE
    $res = ($out | Select-String -Pattern 'RESULT:') | Select-Object -Last 1
    $mx  = ($out | Select-String -Pattern 'max_err=([0-9.eE+-]+)') | Select-Object -Last 1
    $maxv = if ($mx) { $mx.Matches[0].Groups[1].Value } else { "-" }
    $tag = if ($rc -eq 0) { "PASS" } else { "FAIL" }
    if ($rc -ne 0) { $failed++ }
    Write-Output ("layer $L  {0,-7}  max|err|={1}" -f $tag, $maxv)
    if ($rc -ne 0) { $out | Where-Object { $_ -match 'FAIL|LOAD_FAIL' } | ForEach-Object { Write-Output ("    " + $_) } }
    $rows += [pscustomobject]@{ Layer = $L; Status = $tag; MaxErr = $maxv; Prefix = $pre }
}
Write-Output "--------------------------------------------------------------------------"
if ($failed -eq 0) {
    Write-Output ("RESULT: ALL_LAYERS_VERIFY_PASS  (" + ($To - $From + 1) + " layers, mode=$Mode, tol=$Tol)")
} else {
    Write-Output ("RESULT: VERIFY_FAILED  (" + $failed + " of " + ($To - $From + 1) + " layers failed)")
}
exit $failed

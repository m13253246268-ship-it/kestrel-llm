# check_x64.ps1 - x86「可验证」回归：构建 + L3/Sparse 自检计数
#
# 目的：把「引擎在 x86 上能编译、且数值自检与 ARM 同源」固化成一条命令，
#       不再依赖"上次手工跑过"的口头证据。
#       期望值：--test-l3 16 PASS / 0 FAIL、--test-sparse 9/9。
#       x86 无 NEON，(1b)「P2 restore NEON vs scalar bitwise」一项打 [SKIP] 而非
#       [PASS]（不计入 PASS 数；FAIL 必须为 0）。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File tools\build\check_x64.ps1
#   powershell -ExecutionPolicy Bypass -File tools\build\check_x64.ps1 -Gcc <gcc路径> -OutDir .\build-x64
#   （-Gcc / -OutDir 省略时透传给 build_x64.ps1 的同名回退规则）
# 退出码：
#   0 = 构建成功 且 --test-l3 16 PASS/0 FAIL 且 --test-sparse 9/9
#   1 = 构建失败 或 任一项出现 FAIL / 条数不符
#
# 边界（照实标注）：本脚本只证明「可编译 + 自检通过 + 与 ARM 同为 16/9 条」，
#   不证明 x86 可作为性能基准 —— 开放项「x64 偶发整请求卡死」未解决，
#   且 L3 Q4 点积在 x86 无 AVX2 内核（走标量回退，数值对、性能无意义）。
#
# 编码注意：PowerShell 5.1 读无 BOM 脚本时按 ANSI 解码，中文会乱码 → 本文件须 UTF-8 BOM。

param(
    [string]$Gcc = $env:VLLM_GCC,
    [string]$OutDir = $env:VLLM_X64_OUTDIR
)

$ErrorActionPreference = 'Continue'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $OutDir) { $OutDir = Join-Path $root 'build-x64' }
$exe  = Join-Path $OutDir 'vllm_kestrel_x64.exe'

# 重定向到文件时避免中文按 ANSI 落盘（配合脚本自身的 UTF-8 BOM）
[Console]::OutputEncoding = [Text.Encoding]::UTF8

Write-Host '=== [1/3] build x64 ==='
$buildArgs = @('-ExecutionPolicy','Bypass','-File',(Join-Path $PSScriptRoot 'build_x64.ps1'))
if ($Gcc)    { $buildArgs += @('-Gcc', $Gcc) }
if ($OutDir) { $buildArgs += @('-OutDir', $OutDir) }
& powershell @buildArgs | Out-Host
if (-not (Test-Path $exe)) {
    Write-Host "X64 CHECK FAILED: build produced no $exe"
    exit 1
}

function Invoke-SelfTest([string]$flag, [int]$want) {
    $out = & $exe $flag 2>&1
    $p = ($out | Select-String -SimpleMatch '[PASS]' | Measure-Object).Count
    $f = ($out | Select-String -SimpleMatch '[FAIL]' | Measure-Object).Count
    $ok = ($p -eq $want) -and ($f -eq 0)
    $verdict = if ($ok) { 'OK' } else { 'MISMATCH' }
    Write-Host ("     {0,-14} PASS={1} FAIL={2}  (want {3}/0)  -> {4}" -f $flag, $p, $f, $want, $verdict)
    return $ok
}

Write-Host '=== [2/3] self-tests (x86 口径: 16 PASS + 1 SKIP / 9) ==='
$a = Invoke-SelfTest '--test-l3' 16
$b = Invoke-SelfTest '--test-sparse' 9

Write-Host '=== [3/3] artifact ==='
Get-Item $exe | Select-Object Name, Length, LastWriteTime | Format-List | Out-Host
Write-Host ('     sha256 = ' + (Get-FileHash $exe -Algorithm SHA256).Hash)

if ($a -and $b) {
    Write-Host 'X64 CHECK PASSED'
    exit 0
}
Write-Host 'X64 CHECK FAILED'
exit 1

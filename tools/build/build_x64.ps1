# build_x64.ps1 - Windows x64 (MinGW-w64) 构建脚本（x86「可验证」副产物）
#
# 定位：主树为双架构（aarch64-Linux 一级目标 + x86-64/MinGW 副分支）。x86 构建
#       只用于「可编译 + 自检与 ARM 同口径」的回归，**不作性能基准**
#       （开放项：x64 偶发整请求卡死；且 L3 Q4 点积无 AVX2 内核走标量回退）。
#       所有 x86 代码均以 #if ST_ARCH_X86 / #ifdef _WIN32 / #if ST_HAVE_NEON 门控。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File tools\build\build_x64.ps1
#   powershell -ExecutionPolicy Bypass -File tools\build\build_x64.ps1 -Gcc D:\tools\mingw64\bin\gcc.exe -OutDir .\build-x64
#
# 参数（均可省略，按顺序回退）：
#   -Gcc     MinGW-w64 gcc 路径；省略时取 $env:VLLM_GCC，再取 PATH 中的 gcc
#   -OutDir  产物目录；省略时取 $env:VLLM_X64_OUTDIR，再取 <repo>\build-x64
#   产物名：vllm_kestrel_x64.exe
#
# 依赖：MinGW-w64 x86_64 GCC（含 winpthreads、winsock2）。
#       无 NPU 后端（src/npu/vllm_npu.c 走 _WIN32 透明降级 stub）。
#       GCC16+ 将 -Wincompatible-pointer-types 等提为 error，故加 -Wno-error 旗标。
#
# 编码注意：PowerShell 5.1 读取无 BOM 脚本时按 ANSI 解码，中文会乱码；
#           本文件必须以 UTF-8 BOM 保存。
# 项目根由 $PSScriptRoot 向上解析（不硬编码绝对路径）。

param(
    [string]$Gcc = $env:VLLM_GCC,
    [string]$OutDir = $env:VLLM_X64_OUTDIR
)

$ErrorActionPreference = 'Continue'

if (-not $Gcc) {
    $cmd = Get-Command gcc -ErrorAction SilentlyContinue
    if ($cmd) { $Gcc = $cmd.Source }
}
if (-not $Gcc -or -not (Test-Path $Gcc)) {
    Write-Host "ERROR: gcc not found. Pass -Gcc <path> or set `$env:VLLM_GCC."
    exit 1
}

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $OutDir) { $OutDir = Join-Path $root 'build-x64' }
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }
$out = Join-Path $OutDir 'vllm_kestrel_x64.exe'

Push-Location $root
$src = @(
  'src/main.c',
  'src/core/vllm_superpos.c','src/core/vllm_attention.c','src/core/vllm_scheduler.c',
  'src/core/vllm_transformer.c','src/core/vllm_tokenizer.c','src/core/vllm_matmul.c',
  'src/core/vllm_tp.c','src/core/vllm_ep.c','src/core/vllm_l3.c','src/core/vllm_fhe.c','src/core/vllm_ckks.c',
  'src/core/vllm_ntt.c',
  'src/common/vllm_device.c','src/common/vllm_crypto.c',
  'src/serve/vllm_http.c','src/serve/vllm_server.c','src/serve/vllm_batch.c',
  'src/serve/vllm_admin.c','src/serve/vllm_attest.c','src/serve/embedded_web.c',
  'src/model/vllm_safetensors.c','src/model/vqf.c',
  'src/model/vllm_tokenizer_qwen.c','src/model/vllm_vision.c','src/model/vllm_media.c',
  'src/npu/vllm_npu.c'
)
$incs = @('-Iinclude/common','-Iinclude/core','-Iinclude/serve','-Iinclude/model',
          '-Iinclude/media','-Iinclude/npu','-Iinclude/npu/rk3588','-Isrc')
& $Gcc -std=gnu11 -O2 -D_GNU_SOURCE -mavx2 -mfma -ffp-contract=off `
  -Wno-error=incompatible-pointer-types -Wno-error=int-conversion `
  -Wno-error=implicit-function-declaration `
  @incs @src -o $out -lm -lws2_32 -lbcrypt 2>&1 | Tee-Object -Variable errs
$rc = $LASTEXITCODE
Write-Host "rc=$rc"
Pop-Location
if ($rc -eq 0) { Get-Item $out | Select-Object Name,Length }
exit $rc

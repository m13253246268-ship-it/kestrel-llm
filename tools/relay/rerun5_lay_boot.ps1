# rerun5_lay_boot.ps1 — 前 5 层 RNS-CKKS 密文链「从种子完整重跑」（可断点续跑）
#
# 链式协议（每层 = lay + boot 两跳）：
#   lay0 : 明文 L0 数据(.tmp_tok/l0/*) -> .tmp_tok/chain/u0_{t}_{h}.ct      (np=112)
#   boot0: u0_{t}_{h}.ct              -> .tmp_tok/chain/u0r112_{t}_{h}.ct   (刷新回满链)
#   layL : u{L-1}r112_{t}_{h}.ct      -> u{L}_{t}_{h}.ct                    (L=1..4)
#   bootL: u{L}_{t}_{h}.ct            -> u{L}r112_{t}_{h}.ct                (L=1..4)
#   t=0..3 (4 token) × h=0..1 (2 密文分量) = 每层 8 个 .ct
#
# 断点续跑：某一跳的输出 8 个 .ct 齐全则跳过；中断后重跑本脚本即可接着算。
# 日志：.tmp_tok/relay_rerun/{lay,boot}{0..4}.out/.err + master.log + timing.tsv
# 判据：lay 看末尾 RESULT=PASS；boot 看末尾 BOOT=PASS。
#
# 用法（在仓库根目录）：
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\relay\rerun5_lay_boot.ps1
# 线程：$env:T23_NT（默认 4）

param(
    [int]$From = 0,
    [int]$To = 4,
    [string]$LayExe  = ".tmp_tok\t23lay.exe",
    [string]$BootExe = ".tmp_tok\t23boot.exe",
    [string]$LogDir  = ".tmp_tok\relay_rerun"
)

$ErrorActionPreference = "Continue"
# 仓库根自定位：从脚本所在目录向上找到含 src/ 的目录（本脚本位于 tools/relay/）
$Root = $PSScriptRoot
while ($Root -and -not (Test-Path (Join-Path $Root "src"))) { $Root = Split-Path -Parent $Root }
if (-not $Root) { $Root = (Get-Location).Path }
Set-Location $Root
if (-not $env:T23_NT) { $env:T23_NT = "4" }

$LogD = $LogDir
# 工作目录必须自建：驱动用裸 fopen 写 .tmp_tok/chain/（不建目录），
# 目录不存在时会直接报 "[FAIL] sk save" / "[FAIL] ct save"；gcc 的 -o 也会报 cannot open output file。
foreach ($d in @(".tmp_tok", ".tmp_tok\chain", $LogD)) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
$MLOG = "$LogD\master.log"
$TSV  = "$LogD\timing.tsv"

function Log([string]$m) {
    $line = "$(Get-Date -Format 'HH:mm:ss') $m"
    Write-Host $line
    Add-Content -Path $MLOG -Value $line -Encoding utf8
}

# ---------- 0) 构建（缺失才编译），并落 meta ----------
function Build-All {
    if (-not (Test-Path $LayExe)) {
        Log "build lay (np=112, n=2048)"
        gcc -O2 -fopenmp -Wno-implicit-function-declaration -I include -I include/core -I include/common `
            -DCKKS_N=2048 -DCKKS_NPRIMES=112 -DBB=32 -DGG=32 `
            src/core/vllm_ntt.c src/core/vllm_ckks.c src/core/vllm_tp.c tools/drivers/t23_m3p.c -o $LayExe -lm
        if ($LASTEXITCODE -ne 0) { Log "!!! lay build FAIL"; exit 1 }
    }
    if (-not (Test-Path $BootExe)) {
        Log "build boot (np=2100, n=2048, big stack)"
        gcc -O2 -fopenmp -Wno-implicit-function-declaration '-Wl,--stack,33554432' -I include -I include/core -I include/common `
            -DCKKS_N=2048 -DCKKS_NPRIMES=2100 -DBB=32 -DGG=32 `
            src/core/vllm_ntt.c src/core/vllm_ckks.c src/core/vllm_tp.c tools/drivers/t23_chain.c -o $BootExe -lm
        if ($LASTEXITCODE -ne 0) { Log "!!! boot build FAIL"; exit 1 }
    }
}

function Write-Meta {
    $cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1)
    $cc  = (gcc --version 2>&1 | Select-Object -First 1)
    @(
      "=== RNS-CKKS Qwen3-VL-2B 前 5 层重跑 run metadata ===",
      "date       : $(Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz')",
      "host       : $env:COMPUTERNAME",
      "os         : $([System.Environment]::OSVersion.VersionString)",
      "cpu        : $($cpu.Name)",
      "cores_phys : $($cpu.NumberOfCores)  cores_log : $($cpu.NumberOfLogicalProcessors)",
      "threads    : T23_NT=$env:T23_NT",
      "cc         : $cc",
      "compile    : lay CKKS_NPRIMES=112; boot CKKS_NPRIMES=2100; n=2048; scale=2^60",
      "judge      : lay -> RESULT=PASS; boot -> BOOT=PASS"
    ) -join "`n" | Set-Content -Path "$LogD\meta.txt" -Encoding utf8
}

# ---------- 1) 断点判定 ----------
function Have-CT([string]$pre) {
    for ($t = 0; $t -lt 4; $t++) { for ($h = 0; $h -lt 2; $h++) {
        if (-not (Test-Path ".tmp_tok\chain\${pre}_${t}_${h}.ct")) { return $false }
    } }
    return $true
}

# 用 cmd 做重定向，保证字节级落盘（避免 PowerShell 管道把 UTF-8 中文按 GBK 转码）
function Run-Exe([string]$exe, [string]$out, [string]$err) {
    $line = "`"$exe`" > `"$out`" 2> `"$err`""
    cmd /c $line | Out-Null
    return $LASTEXITCODE
}

# ---------- 2) 单跳执行 ----------
$rows = @{}
function Do-Lay([int]$L) {
    $tag = "lay$L"; $op = "u$L"
    if (Have-CT $op) { Log "$tag already done ($op), skip"; $rows[$L].lay = "SKIP"; return $true }
    Log "=== $tag START"
    $env:T23_PHASE = "lay"; $env:T23_LAY = "$L"; $env:T23_E2EMODE = "1"
    $t0 = Get-Date
    $rc = Run-Exe $LayExe "$LogD\$tag.out" "$LogD\$tag.err"
    $w  = [int]((Get-Date) - $t0).TotalSeconds
    $ok = ($rc -eq 0) -and (Select-String -Path "$LogD\$tag.out" -Pattern 'RESULT=PASS' -Quiet)
    if ($ok -and (Have-CT $op)) {
        $rows[$L].lay = "PASS"; $rows[$L].lay_s = $w
        Log "=== $tag PASS wall=${w}s"
        Add-Content $MLOG (Get-Content "$LogD\$tag.out" -Tail 2) -Encoding utf8
        return $true
    }
    $rows[$L].lay = "FAIL"; $rows[$L].lay_s = $w
    Log "!!! $tag FAILED rc=$rc wall=${w}s"
    Add-Content $MLOG (Get-Content "$LogD\$tag.err" -Tail 4) -Encoding utf8
    return $false
}

function Do-Boot([int]$L) {
    $tag = "boot$L"; $ip = "u$L"; $op = "u${L}r"
    if (Have-CT "${op}112") { Log "$tag already done (${op}112), skip"; $rows[$L].boot = "SKIP"; return $true }
    Log "=== $tag START"
    $env:T23_PHASE = "boot"; $env:T23_BI = $ip; $env:T23_BO = $op
    $t0 = Get-Date
    $rc = Run-Exe $BootExe "$LogD\$tag.out" "$LogD\$tag.err"
    $w  = [int]((Get-Date) - $t0).TotalSeconds
    $ok = ($rc -eq 0) -and (Select-String -Path "$LogD\$tag.out" -Pattern 'BOOT=PASS' -Quiet)
    if ($ok -and (Have-CT "${op}112")) {
        $rows[$L].boot = "PASS"; $rows[$L].boot_s = $w
        $pf = Select-String -Path "$LogD\$tag.out" -Pattern 'total=([0-9.]+)s' -AllMatches
        if ($pf) { $rows[$L].boot_total = $pf.Matches[0].Groups[1].Value }
        Log "=== $tag PASS wall=${w}s"
        Add-Content $MLOG (Get-Content "$LogD\$tag.out" -Tail 2) -Encoding utf8
        return $true
    }
    $rows[$L].boot = "FAIL"; $rows[$L].boot_s = $w
    Log "!!! $tag FAILED rc=$rc wall=${w}s"
    Add-Content $MLOG (Get-Content "$LogD\$tag.err" -Tail 4) -Encoding utf8
    return $false
}

# ---------- 3) 主循环 ----------
Build-All
Write-Meta
Log "=== rerun START  layers $From..$To  (T23_NT=$env:T23_NT)"
$grand0 = Get-Date
for ($L = $From; $L -le $To; $L++) {
    if (-not $rows.ContainsKey($L)) { $rows[$L] = @{} }
    if (-not (Do-Lay $L))  { Log "!!! chain aborted at lay$L"; break }
    if (-not (Do-Boot $L)) { Log "!!! chain aborted at boot$L"; break }
}
$grand = [int]((Get-Date) - $grand0).TotalSeconds
Log "=== rerun END total wall=${grand}s"

# ---------- 4) 汇总 timing.tsv ----------
$hdr = "layer`tlay_s`tlay_st`tboot_s`tboot_st`tboot_total_s`tcts_ok"
Set-Content -Path $TSV -Value $hdr -Encoding utf8
for ($L = $From; $L -le $To; $L++) {
    $r = $rows[$L]
    $cts = if ((Have-CT "u$L") -and (Have-CT "u${L}r112")) { "u$L+u${L}r112" } else { "-" }
    $line = "$L`t$($r.lay_s)`t$($r.lay)`t$($r.boot_s)`t$($r.boot)`t$($r.boot_total)`t$cts"
    Add-Content -Path $TSV -Value $line -Encoding utf8
}
Log "summary: $TSV"

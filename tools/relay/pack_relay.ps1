﻿# pack_relay.ps1 — 接力者打包：把某一层范围的交接物打成可上传的 zip，供他人验证
#
# 用法（仓库根目录）：
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\relay\pack_relay.ps1 -From 5 -To 7 -Id yourname
# 产物：
#   .tmp_tok\relay_out\relay_L5-7_yourname_<时间戳>.zip
#
# 包内结构（下一个人与验证者都按这个结构读）：
#   chain/    u{L}r112_{t}_{h}.ct  ×8 / 每层      （核心交接物）
#   logs/     lay{L}.out/.err  boot{L}.out/.err   （判据证据）
#   timing.tsv  meta.txt  master.log              （耗时/机器元数据）
#   sk.bin                                        （全链共用私钥，验证解码需要）
#   manifest.sha256                               （包内全部文件的 SHA256）
#   README.txt                                    （层范围/主张边界/如何验证/如何续跑）
#
# 注意：sk.bin 是全链唯一私钥。发布即可复现/可验证，但也意味着公开该密钥。
#       本项目参数为机制验证级、非真实业务密钥；若你有顾虑，用 -NoKey 排除（接收方需自行持有同一 sk.bin）。

param(
    [Parameter(Mandatory = $true)][int]$From,
    [Parameter(Mandatory = $true)][int]$To,
    [string]$Id = "anon",
    [string]$LogDir = ".tmp_tok\relay_rerun",
    [string]$ChainDir = ".tmp_tok\chain",
    [switch]$NoKey
)

$ErrorActionPreference = "Stop"
# 仓库根自定位：从脚本所在目录向上找到含 src/ 的目录（本脚本位于 tools/relay/）
$Root = $PSScriptRoot
while ($Root -and -not (Test-Path (Join-Path $Root "src"))) { $Root = Split-Path -Parent $Root }
if (-not $Root) { $Root = (Get-Location).Path }
Set-Location $Root

$stamp = Get-Date -Format 'yyyyMMdd-HHmm'
$name  = "relay_L${From}-${To}_${Id}_${stamp}"
$stage = ".tmp_tok\relay_out\$name"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path "$stage\chain", "$stage\logs" | Out-Null
$stageAbs = (Resolve-Path $stage).Path

# ---------- 1) 收集交接物与日志 ----------
$ctCount = 0
for ($L = $From; $L -le $To; $L++) {
    for ($t = 0; $t -lt 4; $t++) { for ($h = 0; $h -lt 2; $h++) {
        $src = Join-Path $ChainDir "u${L}r112_${t}_${h}.ct"
        if (-not (Test-Path $src)) { throw ("缺少交接物: " + $src + " (该层 boot 未完成?)") }
        Copy-Item $src "$stage\chain\" -Force
        $ctCount++
    } }
    foreach ($tag in @("lay$L", "boot$L")) {
        foreach ($ext in @("out", "err")) {
            $f = Join-Path $LogDir "$tag.$ext"
            if (Test-Path $f) { Copy-Item $f "$stage\logs\" -Force }
        }
    }
    Write-Output ("collected layer " + $L)
}
$expect = ($To - $From + 1) * 8
if ($ctCount -ne $expect) { throw ("交接密文数量异常: " + $ctCount + " expected " + $expect) }

foreach ($f in @("timing.tsv", "meta.txt", "master.log")) {
    $p = Join-Path $LogDir $f
    if (Test-Path $p) { Copy-Item $p $stage -Force }
}
if (-not $NoKey) {
    $sk = Join-Path $ChainDir "sk.bin"
    if (-not (Test-Path $sk)) { throw ("缺少 sk.bin: " + $sk) }
    Copy-Item $sk $stage -Force
}

# ---------- 2) README ----------
$next = $To + 1
$readme = @"
relay package: layers $From..$To
generated : $(Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz')
packed by : $Id
handoff   : u$To`r112  (the next runner continues at layer $next)

WHAT THIS IS
  Fully homomorphic (RNS-CKKS) ciphertext inference of Qwen3-VL-2B.
  Each layer = one 'lay' hop (forward) + one 'boot' hop (bootstrapping refresh).
  The handoff artifact is 8 ciphertexts (4 tokens x 2 components) per layer.

WHAT IS CLAIMED
  Only this: the covered layers reproduce and are independently verifiable --
  per-layer judging criterion RESULT=PASS (lay) / BOOT=PASS (boot, refreshed back
  to the full modulus chain, out np=2083).
  A full 28-layer chain is NOT claimed here.

HOW TO VERIFY (third party, no trust required)
  1) integrity : hash every file and compare with manifest.sha256
  2) criteria  : logs/lay*.out must contain 'RESULT=PASS'; logs/boot*.out must contain
                 'BOOT=PASS' and 'out np=2083' once per token/component
  3) structure : each chain/u{L}r112_t_h.ct should be about 3.5 MB (np=112)
  One command:
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\relay\verify_relay.ps1 -Zip <this.zip>

HOW TO CONTINUE (next runner)
  Put chain/*.ct back into .tmp_tok/chain/ (plus sk.bin if you do not already have
  the same one), then run the chain driver starting at the next layer:
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\relay\rerun5_lay_boot.ps1 -From $next -To <end>

JUDGING CAVEAT
  Logs contain per-stage FAIL lines (A/B/C/D) AND a final RESULT=PASS. That is not a
  contradiction: stage checks compare against a plaintext reference from a non-drifting
  input, while a ciphertext chain carries drift from the previous layer. The layer-output
  criterion (RESULT=PASS / BOOT=PASS) is the one that counts.

BOUNDARY
  Not a cryptographic-security claim. Parameters are mechanism/verification level, far
  below 128-bit. Weights are plaintext constants; what is encrypted is the input and the
  intermediate activations. Speed is not a selling point.
"@
Set-Content -Path "$stage\README.txt" -Value $readme -Encoding utf8

# ---------- 3) 包内清单（不含 manifest 自身） ----------
$lines = New-Object System.Collections.Generic.List[string]
$rel = Get-ChildItem -Recurse -File $stage | Where-Object { $_.Name -ne "manifest.sha256" } |
       ForEach-Object { $_.FullName.Substring($stageAbs.Length + 1) } | Sort-Object
foreach ($r in $rel) {
    $h = (Get-FileHash -Algorithm SHA256 -Path (Join-Path $stage $r)).Hash.ToLower()
    $lines.Add("$h  $r")
}
$lines | Set-Content -Path "$stage\manifest.sha256" -Encoding ascii

# ---------- 4) 打包 ----------
New-Item -ItemType Directory -Force -Path ".tmp_tok\relay_out" | Out-Null
$zip = ".tmp_tok\relay_out\$name.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal

Write-Output "----------------------------------------"
Write-Output ("package : " + $zip)
Write-Output ("size    : {0:N1} MB" -f ((Get-Item $zip).Length / 1MB))
Write-Output ("layers  : $From..$To   cts=$ctCount   key=" + $(if ($NoKey) { "excluded" } else { "included" }))
Write-Output ("sha256  : " + (Get-FileHash -Algorithm SHA256 -Path $zip).Hash.ToLower())

# verify_relay.ps1 — 第三方验证：对收到的接力 zip 做完整性与判据校验
#
# 用法：
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\relay\verify_relay.ps1 -Zip <包路径>
# 退出码：0 = 全部通过；1 = 有不通过项（明细打印在控制台）
#
# 校验三层：
#   ① 完整性：包内每个文件的 SHA256 与 manifest.sha256 逐字节一致
#   ② 判据：logs/lay*.out 含 RESULT=PASS；logs/boot*.out 含 BOOT=PASS 且 out np=2083
#   ③ 结构：每层 8 个 u{L}r112_*.ct，单个文件 ~3.5 MB（np=112）

param(
    [Parameter(Mandatory = $true)][string]$Zip,
    [switch]$Keep
)

$ErrorActionPreference = "Continue"
if (-not (Test-Path $Zip)) { Write-Output "ERROR: zip not found: $Zip"; exit 1 }

$fails = 0
function Check([string]$name, [bool]$ok, [string]$detail) {
    if ($ok) { Write-Output ("  [PASS] " + $name + $(if ($detail) { "  -- " + $detail } else { "" })) }
    else     { Write-Output ("  [FAIL] " + $name + $(if ($detail) { "  -- " + $detail } else { "" })); $script:fails++ }
}

# ---------- 0) 解包 ----------
$tmp = Join-Path $env:TEMP ("relay_verify_" + [System.Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
try {
    Expand-Archive -Path $Zip -DestinationPath $tmp -Force
} catch {
    Write-Output ("ERROR: unzip failed: " + $_.Exception.Message); exit 1
}

$root = (Get-ChildItem -Recurse -Directory $tmp | Where-Object { Test-Path (Join-Path $_.FullName "manifest.sha256") } | Select-Object -First 1)
if (-not $root) { $root = Get-Item $tmp }
$stage = $root.FullName
Write-Output ("package  : " + (Get-Item $Zip).Name)
Write-Output ("zip sha256: " + (Get-FileHash -Algorithm SHA256 -Path $Zip).Hash.ToLower())
Write-Output ("stage    : " + $stage)
Write-Output ""

# ---------- 1) 完整性（对比 manifest） ----------
Write-Output "[1] integrity (SHA256 vs manifest.sha256)"
$mf = Join-Path $stage "manifest.sha256"
Check "manifest.sha256 present" (Test-Path $mf) ""
if (Test-Path $mf) {
    $lines = Get-Content $mf | Where-Object { $_ -match '^\s*[0-9a-f]{64}\s' }
    $bad = 0; $n = 0
    foreach ($ln in $lines) {
        $p = $ln -split '\s+', 3
        $want = $p[0].ToLower()
        $rel  = $p[-1] -replace '[\\/]', [IO.Path]::DirectorySeparatorChar
        $abs  = Join-Path $stage $rel
        if (-not (Test-Path $abs)) { Write-Output ("  [FAIL] missing: " + $rel); $bad++; $script:fails++; continue }
        $got = (Get-FileHash -Algorithm SHA256 -Path $abs).Hash.ToLower()
        if ($got -ne $want) { Write-Output ("  [FAIL] hash mismatch: " + $rel); $bad++ }
        $n++
    }
    Check "all files match manifest" ($bad -eq 0) ("checked=$n mismatched=$bad")

    # 反向检查：包内是否有 manifest 未收录的文件（防夹带）
    $onDisk = Get-ChildItem -Recurse -File $stage | Where-Object { $_.Name -ne "manifest.sha256" } |
              ForEach-Object { $_.FullName.Substring($stage.Length + 1) }
    $extra = $onDisk | Where-Object { $rel = $_; -not ($lines | Where-Object { ($_ -split '\s+', 3)[-1] -eq $rel }) }
    Check "no unlisted files (no smuggling)" (-not $extra) (($extra | Measure-Object).Count.ToString() + " extra")
}

# ---------- 2) 判据 ----------
Write-Output ""
Write-Output "[2] judging criteria"
$layOuts  = Get-ChildItem (Join-Path $stage "logs") -Filter "lay*.out"  -ErrorAction SilentlyContinue
$bootOuts = Get-ChildItem (Join-Path $stage "logs") -Filter "boot*.out" -ErrorAction SilentlyContinue
Check "lay logs present"  ($layOuts.Count  -gt 0) ("count=" + $layOuts.Count)
Check "boot logs present" ($bootOuts.Count -gt 0) ("count=" + $bootOuts.Count)
foreach ($f in $layOuts) {
    Check ("lay criterion " + $f.BaseName) ((Select-String -Path $f.FullName -Pattern 'RESULT=PASS' -Quiet) -eq $true) ""
}
foreach ($f in $bootOuts) {
    $okBoot = (Select-String -Path $f.FullName -Pattern 'BOOT=PASS' -Quiet) -eq $true
    $refresh = (Select-String -Path $f.FullName -Pattern 'out np=2083' -AllMatches).Count
    Check ("boot criterion " + $f.BaseName) ($okBoot -and $refresh -ge 8) ("BOOT=PASS=" + $okBoot + " refreshes=" + $refresh)
}

# ---------- 3) 结构 ----------
Write-Output ""
Write-Output "[3] artifact structure"
$cts = Get-ChildItem (Join-Path $stage "chain") -Filter "u*r112_*.ct" -ErrorAction SilentlyContinue
Check "ciphertexts present" ($cts.Count -gt 0) ("count=" + $cts.Count)
Check "count is multiple of 8" (($cts.Count % 8) -eq 0) ("layers=" + [int]($cts.Count / 8))
$small = $cts | Where-Object { $_.Length -lt 1MB }
Check "each ct ~3.5MB (np=112)" (($small | Measure-Object).Count -eq 0) ("suspicious(small)=" + ($small | Measure-Object).Count)

# ---------- 汇总 ----------
Write-Output ""
if ($fails -eq 0) {
    Write-Output "RESULT: ALL_CHECKS_PASS"
} else {
    Write-Output ("RESULT: FAILED  (" + $fails + " check(s) failed)")
}
if (-not $Keep) { Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue }
else { Write-Output ("extracted kept at: " + $stage) }
exit $(if ($fails -eq 0) { 0 } else { 1 })

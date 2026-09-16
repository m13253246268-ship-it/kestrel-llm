# layver_run.ps1 - numeric equivalence check for an alternative ciphertext lineage.
#
# Question this answers:
#   A boot refresh produced u{L}r112 at a DIFFERENT thread count (T23_NT) than the
#   baseline lineage, so its bytes differ. Is it still numerically equivalent when the
#   next layer consumes it? Bytes-equal is NOT the criterion here; max|err| is.
#
# Method (isolation: only the input changes, the layer's thread count stays the same):
#   1) back up chain\u{L-1}r112*.ct and chain\u{L}_*.ct
#   2) copy the alternative u{L-1}r112*.ct from -AltDir into the chain dir
#   3) run lay{L} with T23_E2EMODE=1 at T23_NT = -NT (same as the baseline run)
#   4) compare the max|err| statistics with the baseline log
#   5) restore the backup
#
# IMPORTANT: -TargetLayer must be >= 1.
#   lay0 does NOT read any chain ciphertext -- it encrypts the plaintext seed -- so
#   injecting an alternative u0r112 and running lay0 compares nothing (it is a no-op).
#   To test an alternative u0r112, run lay1.
#
# Usage (from the repo root):
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\relay\layver_run.ps1 `
#       -TargetLayer 1 -AltDir .tmp_tok\chain\_alt_in `
#       -BaselineLog .tmp_tok\relay_rerun\lay1.out
#
# ASCII only (PowerShell 5.1 decodes BOM-less files as ANSI).
param(
    [int]$TargetLayer = 1,
    [string]$AltDir = ".tmp_tok\chain\_alt_in",
    [string]$BaselineLog = "",
    [string]$ChainDir = ".tmp_tok\chain",
    [string]$Work = ".tmp_tok\relay_state",
    [string]$NT = "4",
    [string]$LayExe = ".tmp_tok\t23lay.exe"
)

$ErrorActionPreference = "Continue"

# repo root: walk up from this script until a dir containing src/ is found
$Root = $PSScriptRoot
while ($Root -and -not (Test-Path (Join-Path $Root "src"))) { $Root = Split-Path -Parent $Root }
if (-not $Root) { $Root = (Get-Location).Path }
Set-Location $Root

if ($TargetLayer -lt 1) {
    Write-Output "ERROR: -TargetLayer must be >= 1."
    Write-Output "       lay0 encrypts the plaintext seed and reads no chain ciphertext, so"
    Write-Output "       injecting an alternative u0r112 there is a no-op. Use -TargetLayer 1."
    exit 2
}

New-Item -ItemType Directory -Force -Path $Work | Out-Null
$res = Join-Path $Work "layver_result.txt"
if (-not $BaselineLog) { $BaselineLog = ".tmp_tok\relay_rerun\lay$TargetLayer.out" }

$prev = "u$($TargetLayer - 1)r112"
$cur  = "u$TargetLayer"

[IO.File]::WriteAllText($res, "layver queued $(Get-Date -Format 'HH:mm:ss')`r`n")
[IO.File]::AppendAllText($res, "inject   = $AltDir\${prev}*.ct`r`n")
[IO.File]::AppendAllText($res, "run      = lay$TargetLayer  (T23_NT=$NT, T23_E2EMODE=1)`r`n")
[IO.File]::AppendAllText($res, "baseline = $BaselineLog`r`n")

if (-not (Test-Path (Join-Path $AltDir "${prev}_0_0.ct"))) {
    [IO.File]::AppendAllText($res, "ERROR: alternative input missing: $AltDir\${prev}_0_0.ct`r`n")
    [IO.File]::AppendAllText($res, "RESULT=FAILED`r`nDONE $(Get-Date -Format 'HH:mm:ss')`r`n")
    exit 3
}

# back up everything this run touches
$bak = Join-Path $Work "_layver_bak"
New-Item -ItemType Directory -Force -Path $bak | Out-Null
Copy-Item (Join-Path $ChainDir "${prev}*.ct")  $bak -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $ChainDir "${cur}_*.ct")  $bak -Force -ErrorAction SilentlyContinue

# feed the alternative lineage in
Copy-Item (Join-Path $AltDir "${prev}*.ct") $ChainDir -Force

$env:T23_PHASE   = 'lay'
$env:T23_LAY     = "$TargetLayer"
$env:T23_E2EMODE = '1'
$env:T23_NT      = $NT
Remove-Item Env:T23_ONE      -ErrorAction SilentlyContinue
Remove-Item Env:VLLM_TP_SPIN -ErrorAction SilentlyContinue

$o = Join-Path $Work "layver_lay$TargetLayer.out"
$e = Join-Path $Work "layver_lay$TargetLayer.err"
$t0 = Get-Date
cmd /c "`"$LayExe`" > `"$o`" 2> `"$e`"" | Out-Null
$rc = $LASTEXITCODE
$w  = [int]((Get-Date) - $t0).TotalSeconds

function Get-ErrStats([string]$path) {
    $vals = New-Object System.Collections.ArrayList
    if (Test-Path $path) {
        Select-String -Path $path -Pattern 'max\|err\|=([0-9.eE+-]+)' -AllMatches -ErrorAction SilentlyContinue |
            ForEach-Object { foreach ($m in $_.Matches) { [void]$vals.Add([double]$m.Groups[1].Value) } }
    }
    if ($vals.Count -eq 0) { return "count=0" }
    $mx = ($vals | Measure-Object -Maximum).Maximum
    $mn = ($vals | Measure-Object -Minimum).Minimum
    $av = ($vals | Measure-Object -Average).Average
    return ("count={0}  max={1:E3}  min={2:E3}  mean={3:E3}" -f $vals.Count, $mx, $mn, $av)
}

$verdict = 'no-RESULT'
$m = Select-String -Path $o -Pattern 'RESULT=(\w+)' -AllMatches -ErrorAction SilentlyContinue
if ($m) { $verdict = $m.Matches[0].Groups[1].Value }

("lay{0} rc={1} wall={2}s  RESULT={3}  at {4}" -f $TargetLayer, $rc, $w, $verdict, (Get-Date -Format 'HH:mm:ss')) |
    Add-Content -Encoding utf8 $res
("  NEW  (input from $AltDir): " + (Get-ErrStats $o)) | Add-Content -Encoding utf8 $res
("  BASE ($BaselineLog): " + (Get-ErrStats $BaselineLog)) | Add-Content -Encoding utf8 $res

# restore
Copy-Item (Join-Path $bak "${prev}*.ct") $ChainDir -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $bak "${cur}_*.ct") $ChainDir -Force -ErrorAction SilentlyContinue

"DONE $(Get-Date -Format 'HH:mm:ss')" | Add-Content -Encoding utf8 $res

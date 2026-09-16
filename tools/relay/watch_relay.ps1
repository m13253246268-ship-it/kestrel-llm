# watch_relay.ps1 - unattended duty watcher for the L{From}-{To} relay.
#
# Ticks every -IntervalSec (default 15 min), appends a status snapshot, gates on the
# layver verdict, and when it passes starts the relay chain. When the chain ends it
# archives the artifacts, runs the independent decrypt-and-compare verification, and
# optionally bit-compares against a reference lineage.
#
# Why a detached process: on Windows a foreground long-blocking command can be reclaimed
# by the invoking tool, so the only reliable clock here is a detached loop. Launch with:
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','tools\relay\watch_relay.ps1' -WorkingDirectory <repo root> -WindowStyle Hidden
#
# ASCII only (PowerShell 5.1 decodes BOM-less files as ANSI).
param(
    [string]$StateDir = ".tmp_tok\relay_state",
    [string]$LogDir = ".tmp_tok\relay_rerun",
    [string]$ChainDir = ".tmp_tok\chain",
    [string]$Dest = "results\L0-4",
    [string]$LayExe = ".tmp_tok\t23lay.exe",
    [string]$BootExe = ".tmp_tok\t23boot.exe",
    [string]$RefDir = "",
    [int]$From = 0,
    [int]$To = 4,
    [int]$IntervalSec = 900,
    [int]$TimeoutHours = 16
)

# repo root: walk up from this script until a dir containing src/ is found
$Root = $PSScriptRoot
while ($Root -and -not (Test-Path (Join-Path $Root "src"))) { $Root = Split-Path -Parent $Root }
if (-not $Root) { $Root = (Get-Location).Path }
Set-Location $Root

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
$log  = Join-Path $StateDir "watch_relay_log.txt"
$decl = Join-Path $StateDir "watch_relay_decision.txt"
$sum  = Join-Path $StateDir "L$($From)_$($To)_verify_summary.txt"
$lr   = Join-Path $StateDir "layver_result.txt"
$mlog = Join-Path $LogDir "master.log"

function Snap([string]$msg) {
    $t  = (Get-Date -Format 'HH:mm:ss')
    $nb = (Get-Process -Name t23boot -ErrorAction SilentlyContinue | Measure-Object).Count
    $nl = (Get-Process -Name t23lay  -ErrorAction SilentlyContinue | Measure-Object).Count
    $lrLast = '-'
    if (Test-Path $lr) { $lrLast = ((Get-Content $lr | Select-Object -Last 1) -join '') }
    $ml = '-'
    if (Test-Path $mlog) { $ml = ((Get-Content $mlog -Tail 1) -join '') }
    "[$t] $msg | boot=$nb lay=$nl | layver:$lrLast | master:$ml" | Add-Content -Encoding utf8 $log
}

"watcher started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Add-Content -Encoding utf8 $decl

$relayStarted = $false
$relayDone    = $false
$relayT0      = $null
$deadline     = (Get-Date).AddHours($TimeoutHours)
$ticks        = [int](($TimeoutHours * 3600) / [Math]::Max($IntervalSec, 1) + 4)

for ($i = 0; $i -lt $ticks; $i++) {
    Snap "tick$i"

    # ---- phase 1: gate on the layver verdict ----
    if (-not $relayStarted -and -not $relayDone) {
        $done = (Test-Path $lr) -and (Select-String -Path $lr -Pattern 'DONE' -Quiet -ErrorAction SilentlyContinue)
        if ($done) {
            $txt = Get-Content $lr -Raw
            $verdict = 'UNKNOWN'
            if ($txt -match 'RESULT=PASS')   { $verdict = 'PASS' }
            if ($txt -match 'RESULT=FAILED') { $verdict = 'FAILED' }
            $newMax = 'n/a'
            foreach ($line in ($txt -split "`n")) {
                if ($line -match 'NEW\s+\(input') {
                    $mm = [regex]::Match($line, 'max=([0-9.eE+-]+)')
                    if ($mm.Success) { $newMax = $mm.Groups[1].Value }
                }
            }
            if ($verdict -eq 'PASS') {
                "GO $(Get-Date -Format 'HH:mm:ss') layver=PASS newMax=$newMax -> starting L$From-$To relay" |
                    Add-Content -Encoding utf8 $decl
                $relayArgs = @('-NoProfile','-ExecutionPolicy','Bypass',
                               '-File', (Join-Path $Root 'tools\relay\rerun5_lay_boot.ps1'),
                               '-From', "$From", '-To', "$To",
                               '-LayExe', $LayExe, '-BootExe', $BootExe, '-LogDir', $LogDir)
                Start-Process powershell -ArgumentList $relayArgs -WorkingDirectory $Root -WindowStyle Hidden
                $relayStarted = $true
                $relayT0 = Get-Date
                Snap 'relay launched'
            } else {
                "HOLD $(Get-Date -Format 'HH:mm:ss') layver verdict=$verdict -> relay NOT started" |
                    Add-Content -Encoding utf8 $decl
                Snap 'HOLD'
                break
            }
        }
    }

    # ---- phase 2: wait for the relay to END, then archive + verify ----
    if ($relayStarted -and -not $relayDone) {
        if ((Test-Path $mlog) -and (Select-String -Path $mlog -Pattern 'rerun END' -Quiet -ErrorAction SilentlyContinue)) {
            $relayDone = $true
            "relay END at $(Get-Date -Format 'HH:mm:ss'), wall=$([int]((Get-Date) - $relayT0).TotalSeconds)s" |
                Add-Content -Encoding utf8 $decl
            Snap 'relay END -> verifying'

            # 1) archive
            & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root 'tools\relay\collect_results.ps1') `
                -From $From -To $To -LogDir $LogDir -ChainDir $ChainDir -Dest $Dest 2>&1 |
                Out-File -Encoding utf8 (Join-Path $StateDir 'collect_out.txt')
            "collect exit=$LASTEXITCODE" | Add-Content -Encoding utf8 $decl

            # 2) independent decrypt-and-compare, lay then boot
            & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root 'tools\relay\verify_layer.ps1') `
                -From $From -To $To -Mode lay -ChainDir $ChainDir 2>&1 |
                Out-File -Encoding utf8 (Join-Path $StateDir 'verify_lay.txt')
            $rcLay = $LASTEXITCODE
            & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root 'tools\relay\verify_layer.ps1') `
                -From $From -To $To -Mode boot -ChainDir $ChainDir 2>&1 |
                Out-File -Encoding utf8 (Join-Path $StateDir 'verify_boot.txt')
            $rcBoot = $LASTEXITCODE

            # 3) optional bit-level compare against a reference lineage
            "=== L$From-$To verify summary $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Set-Content -Encoding utf8 $sum
            "relay wall = $([int]((Get-Date) - $relayT0).TotalSeconds)s" | Add-Content -Encoding utf8 $sum
            "verify_layer -Mode lay  exit=$rcLay  (0 = all layers pass)" | Add-Content -Encoding utf8 $sum
            "verify_layer -Mode boot exit=$rcBoot" | Add-Content -Encoding utf8 $sum
            (Get-Content (Join-Path $StateDir 'verify_lay.txt')  | Select-Object -Last 12) | Add-Content -Encoding utf8 $sum
            (Get-Content (Join-Path $StateDir 'verify_boot.txt') | Select-Object -Last 12) | Add-Content -Encoding utf8 $sum

            if ($RefDir -and (Test-Path $RefDir)) {
                $same = 0; $diff = 0; $miss = 0
                $cmp = Join-Path $StateDir 'ref_cmp.txt'
                Get-ChildItem $RefDir -File | ForEach-Object {
                    $mine = Join-Path $Root (Join-Path $ChainDir $_.Name)
                    if (-not (Test-Path $mine)) {
                        $miss++
                        "{0}  ref={1}  mine=MISSING" -f $_.Name, (Get-FileHash $_.FullName -Algorithm SHA256).Hash.Substring(0,16)
                    } else {
                        $ha = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
                        $hb = (Get-FileHash $mine     -Algorithm SHA256).Hash
                        if ($ha -eq $hb) { $same++ } else { $diff++ }
                        "{0}  ref={1}  mine={2}  {3}" -f $_.Name, $ha.Substring(0,16), $hb.Substring(0,16), $(if ($ha -eq $hb) { 'same' } else { 'DIFF' })
                    }
                } | Set-Content -Encoding utf8 $cmp
                "bit-compare vs $RefDir : same=$same diff=$diff missing=$miss" | Add-Content -Encoding utf8 $sum
                "raw hashes: $cmp" | Add-Content -Encoding utf8 $sum
            } else {
                "bit-compare: skipped (no -RefDir given)" | Add-Content -Encoding utf8 $sum
            }

            "DONE $(Get-Date -Format 'HH:mm:ss')" | Add-Content -Encoding utf8 $sum
            Snap 'all done'
            break
        }
    }

    if ((Get-Date) -gt $deadline) { Snap 'TIMEOUT'; break }
    Start-Sleep -Seconds $IntervalSec
}
"watcher exit $(Get-Date -Format 'HH:mm:ss')" | Add-Content -Encoding utf8 $decl

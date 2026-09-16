# _monitor.ps1 - detached progress monitor for the 5-layer rerun (ASCII only, no BOM needed)
# Repo root: walk up from this script until a dir containing src/ is found (script lives in tools/relay/).
$root = $PSScriptRoot
while ($root -and -not (Test-Path (Join-Path $root 'src'))) { $root = Split-Path -Parent $root }
if (-not $root) { $root = (Get-Location).Path }
Set-Location $root
$out = ".tmp_tok\relay_rerun\_status.txt"
for ($i = 0; $i -lt 720; $i++) {
    $o = @()
    $o += "t=" + (Get-Date -Format 'HH:mm:ss')
    $p = Get-Process -Name t23lay,t23boot -ErrorAction SilentlyContinue
    if ($p) { $o += ($p | ForEach-Object { $_.ProcessName + " pid=" + $_.Id + " cpu=" + [int]$_.CPU + " wsMB=" + [int]($_.WS/1MB) }) }
    else { $o += "no t23 process running" }
    $o += "ct_files=" + (Get-ChildItem .tmp_tok\chain\*.ct -ErrorAction SilentlyContinue).Count
    $o += "--- master.log ---"
    $o += (Get-Content .tmp_tok\relay_rerun\master.log -ErrorAction SilentlyContinue)
    $o += "--- newest lay/boot result lines ---"
    $o += (Get-ChildItem .tmp_tok\relay_rerun\*.out -ErrorAction SilentlyContinue |
           Sort-Object LastWriteTime |
           ForEach-Object { $s = (Select-String -Path $_.FullName -Pattern 'RESULT=|BOOT=' -ErrorAction SilentlyContinue); if ($s) { $_.Name + " :: " + $s[-1].Line } })
    $o += "--- current tail ---"
    $cur = (Get-ChildItem .tmp_tok\relay_rerun\*.out -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1)
    if ($cur) { $o += ($cur.Name + ": " + (Get-Content $cur.FullName -Tail 2 -ErrorAction SilentlyContinue | Select-Object -Last 1)) }
    $o | Set-Content $out -Encoding utf8
    Start-Sleep -Seconds 60
}

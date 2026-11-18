# run_scout.ps1 — weekly paper-scout run, meant for Windows Task Scheduler.
# (Unix/Mac equivalent: run_scout.sh, for crontab.)
#
# Install (weekly, Monday 6am), from an ordinary (non-admin) PowerShell prompt:
#
#   schtasks /create /tn "PaperScout" /sc weekly /d MON /st 06:00 /tr `
#     "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"C:\path\to\paper-scout\run_scout.ps1`""
#
# Or via the Task Scheduler GUI: Create Basic Task -> Trigger: Weekly, Monday,
# 6:00 AM -> Action: Start a program -> Program: powershell.exe -> Arguments:
#   -NoProfile -ExecutionPolicy Bypass -File "C:\path\to\paper-scout\run_scout.ps1"
#
# If PowerShell refuses to run this at all (execution policy), the -File
# invocation above with -ExecutionPolicy Bypass sidesteps that without you
# needing to change your global execution policy.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$venvActivate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
}

$logDir = Join-Path $PSScriptRoot "results\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$ts  = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$log = Join-Path $logDir "scout_$ts.log"

"[$(Get-Date)] starting scout run" | Out-File -FilePath $log -Append -Encoding utf8

$py = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "py" }

try {
    & $py paper_scout.py --sync-arxiv --save-vault --evidence-cards --profile microstructure *>> $log
    $rc = $LASTEXITCODE
} catch {
    "[$(Get-Date)] ERROR: $_" | Out-File -FilePath $log -Append -Encoding utf8
    $rc = 1
}

"[$(Get-Date)] finished (exit $rc)" | Out-File -FilePath $log -Append -Encoding utf8
exit $rc

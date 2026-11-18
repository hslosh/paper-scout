#!/bin/bash
# run_scout.sh — weekly paper-scout run, meant for crontab.
#
# Install (weekly, Monday 6am):
#   crontab -e
#   0 6 * * 1 /path/to/paper-scout/run_scout.sh
#
# (cron runs with a minimal environment, so this script cd's into the repo
# and activates a venv itself rather than relying on inherited PATH/cwd.)

set -euo pipefail
cd "$(dirname "$0")"

if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

mkdir -p results/logs
TS=$(date '+%Y-%m-%d_%H-%M-%S')
LOG="results/logs/scout_${TS}.log"

{
    echo "[$(date)] starting scout run"
    if python3 paper_scout.py --sync-arxiv --save-vault --evidence-cards --profile microstructure; then
        rc=0
    else
        rc=$?
    fi
    echo "[$(date)] finished (exit $rc)"
} >> "$LOG" 2>&1
exit "$rc"

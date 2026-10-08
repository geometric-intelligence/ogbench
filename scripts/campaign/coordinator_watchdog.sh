#!/usr/bin/env bash
# Restart the campaign coordinator if it died before the campaign finished.
#
# Meant for cron on the coordinator host, e.g. every 10 minutes:
#   flock -n /tmp/<campaign>_watchdog.lock bash coordinator_watchdog.sh <coordinator_root>
# The coordinator resumes from the assignment log in <coordinator_root>.
set -uo pipefail

ROOT=${1:?usage: coordinator_watchdog.sh COORDINATOR_ROOT}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-/home/gbg141/miniconda3/envs/bgbench/bin/python}
LOG="$ROOT/coordinator.log"
PID_FILE="$ROOT/coordinator.pid"

if grep -q 'campaign finished' "$LOG" 2>/dev/null; then
  exit 0
fi
pid=$(cat "$PID_FILE" 2>/dev/null)
if [[ -n "$pid" ]] && { tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null \
  | grep -q 'scripts.campaign.coordinator run'; then
  exit 0
fi

cd "$REPO" || exit 1
nohup setsid "$PYTHON" -m scripts.campaign.coordinator run >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
printf '%s coordinator_watchdog: restarted coordinator (pid %s, was %s)\n' \
  "$(date -u '+%F %T')" "$!" "${pid:-none}"

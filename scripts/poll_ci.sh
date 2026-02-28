#!/usr/bin/env bash
# scripts/poll_ci.sh
# ------------------------------------------------------------------
# Poll a GitHub Actions run until it completes, then dump strategy
# result lines from the Stage 1 job log.
#
# Usage:
#   ./scripts/poll_ci.sh [RUN_ID] [INTERVAL_SECONDS]
#
# If RUN_ID is omitted, the latest run on the current branch is used.
# Default poll interval: 60 seconds.
# ------------------------------------------------------------------
set -euo pipefail

REPO="isaacegglestone/AS-Grid"
JOB_NAME_FILTER="Stage 1"
INTERVAL="${2:-60}"

# ── Resolve run ID ──────────────────────────────────────────────────────────
if [[ -n "${1:-}" ]]; then
  RUN_ID="$1"
else
  BRANCH=$(git -C "$(dirname "$0")/.." rev-parse --abbrev-ref HEAD)
  echo "No RUN_ID supplied — looking up latest run for branch: $BRANCH"
  RUN_ID=$(gh api "repos/$REPO/actions/runs?branch=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$BRANCH")&per_page=1" \
    | python3 -c "import sys,json; runs=json.load(sys.stdin)['workflow_runs']; print(runs[0]['id']) if runs else sys.exit('No runs found')")
  echo "Using run ID: $RUN_ID"
fi

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/poll_ci_${RUN_ID}.log"
echo "Logging to: $LOG_FILE"
echo "" > "$LOG_FILE"

# ── Poll loop ───────────────────────────────────────────────────────────────
i=0
while true; do
  i=$((i + 1))
  PAYLOAD=$(gh api "repos/$REPO/actions/runs/$RUN_ID/jobs" 2>/dev/null || echo '{"jobs":[]}')
  STATUS=$(echo "$PAYLOAD" | python3 -c "
import sys, json
jobs = json.load(sys.stdin)['jobs']
s1 = [j for j in jobs if '$JOB_NAME_FILTER' in j.get('name','')]
if not s1:
    print('pending')
else:
    j = s1[0]
    concl = j['conclusion'] or 'running'
    print(j['status'] + '/' + concl)
")
  LINE="$(date +%H:%M:%S) [${i}] run=${RUN_ID} ${STATUS}"
  echo "$LINE"
  echo "$LINE" >> "$LOG_FILE"

  if [[ "$STATUS" == *"completed"* ]]; then
    break
  fi
  sleep "$INTERVAL"
done

# ── Fetch and print results ─────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  CI run $RUN_ID complete — fetching strategy results"
echo "========================================================"

JOB_ID=$(gh api "repos/$REPO/actions/runs/$RUN_ID/jobs" \
  | python3 -c "
import sys, json
jobs = json.load(sys.stdin)['jobs']
s1 = [j for j in jobs if '$JOB_NAME_FILTER' in j.get('name','')]
print(s1[0]['id'])
")

RESULTS=$(gh api "repos/$REPO/actions/jobs/$JOB_ID/logs" \
  | grep -E "Strategy:|return:|max_dd:|Best strategy|=====|v2 ADX|ADX filter|walk-forward|OOS" \
  | head -120)

echo "$RESULTS"
echo ""
echo "$RESULTS" >> "$LOG_FILE"
echo "Full log: $LOG_FILE"

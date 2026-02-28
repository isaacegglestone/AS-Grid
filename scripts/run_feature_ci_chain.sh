#!/usr/bin/env bash
# =============================================================================
# run_feature_ci_chain.sh
#
# Drives the sequential CI push-chain for feature backtests v3→v8.
# Each feature is pushed only after the previous CI run completes successfully.
#
# Usage:
#   bash scripts/run_feature_ci_chain.sh [--start SYMBOL]
#
#   --start SYMBOL   Resume from a specific symbol (default: EMA, i.e. ATR
#                    must already be running when this script starts)
#
# The script:
#   1. Waits for the *current* in-progress CI run to finish.
#   2. Updates .github/workflows/ci.yml to the next feature symbol.
#   3. Commits and pushes.
#   4. Gets the new run ID and repeats.
#
# Results are logged to logs/chain_ci_<SYMBOL>.log for each feature.
# A summary of best strategies is appended to logs/chain_results_summary.txt
# =============================================================================
set -euo pipefail

BRANCH="feat/bitunix-exchange-adapter"
REPO="isaacegglestone/AS-Grid"
POLL_INTERVAL=90  # seconds between GitHub API polls
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"
SUMMARY_FILE="${LOG_DIR}/chain_results_summary.txt"

# Feature push sequence (ATR is already pushed / running, start from EMA)
declare -a FEATURES=("EMA" "BB" "RSI" "VOL" "MS")
declare -A FEATURE_SYMBOLS=(
  ["EMA"]="XRPEMA"
  ["BB"]="XRPBB"
  ["RSI"]="XRPRSI"
  ["VOL"]="XRPVOL"
  ["MS"]="XRPMS"
)
declare -A FEATURE_NAMES=(
  ["EMA"]="v4 EMA bias filter"
  ["BB"]="v5 BB squeeze filter"
  ["RSI"]="v6 RSI filter"
  ["VOL"]="v7 Volume confirmation"
  ["MS"]="v8 Market structure filter"
)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_current_run() {
  # Poll the most recent run on the branch until it completes.
  # Returns the run_id via stdout.
  local run_id
  local status
  local conclusion
  local poll_count=0

  log "Fetching current in-progress run on ${BRANCH}..."
  run_id=$(gh api "repos/${REPO}/actions/runs?branch=${BRANCH}&per_page=1" \
    | python3 -c "import sys,json; r=json.load(sys.stdin)['workflow_runs'][0]; print(r['id'])")
  log "Monitoring run ${run_id}..."

  while true; do
    poll_count=$((poll_count + 1))
    status=$(gh api "repos/${REPO}/actions/runs/${run_id}" \
      | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['status'])")
    conclusion=$(gh api "repos/${REPO}/actions/runs/${run_id}" \
      | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('conclusion') or 'none')")

    log "[${poll_count}] run=${run_id} ${status}/${conclusion}"

    if [[ "$status" == "completed" ]]; then
      if [[ "$conclusion" != "success" ]]; then
        log "⚠️  Run ${run_id} concluded: ${conclusion} — stopping chain"
        exit 1
      fi
      log "✅ Run ${run_id} succeeded"
      echo "$run_id"
      return 0
    fi
    sleep "$POLL_INTERVAL"
  done
}

extract_results() {
  # Extract strategy results from a completed run's job logs.
  local run_id="$1"
  local feature="$2"
  local log_file="${LOG_DIR}/chain_ci_${feature}.log"

  log "Fetching job logs for run ${run_id} → ${log_file}"
  gh api "repos/${REPO}/actions/runs/${run_id}/jobs" \
    | python3 -c "
import sys, json, subprocess
jobs = json.load(sys.stdin)['jobs']
job_id = jobs[0]['id']
# Download logs for step containing 'Backtest'
for step in jobs[0]['steps']:
    if 'backtest' in step.get('name','').lower() or 'Backtest' in step.get('name',''):
        print(step['name'])
        break
print('job_id:', job_id)
" 2>/dev/null || true

  # Get the run log and grep for strategy results
  gh run view "$run_id" --repo "$REPO" --log 2>/dev/null \
    | grep -E "Strategy:|return:|Best strategy:" \
    >> "$log_file" 2>/dev/null || true

  # Extract best lines for summary
  local best_6m best_2y
  best_6m=$(grep "Best strategy:" "$log_file" | head -1 || echo "n/a")
  best_2y=$(grep "Best strategy:" "$log_file" | tail -1 || echo "n/a")

  echo "" >> "$SUMMARY_FILE"
  echo "=== ${feature} (run ${run_id}) ===" >> "$SUMMARY_FILE"
  echo "  6m OOS:    ${best_6m}" >> "$SUMMARY_FILE"
  echo "  2y:        ${best_2y}" >> "$SUMMARY_FILE"
  cat "$log_file" >> "$SUMMARY_FILE"
  echo "" >> "$SUMMARY_FILE"
}

push_next_feature() {
  local feature="$1"
  local symbol="${FEATURE_SYMBOLS[$feature]}"
  local name="${FEATURE_NAMES[$feature]}"

  log "Updating ci.yml to run ${symbol} (${name})"
  sed -i '' \
    "s/- name: Backtest XRP.*$/- name: Backtest XRP\/USDT — ${name}/" \
    .github/workflows/ci.yml
  sed -i '' \
    "s/run: python asBack\/backtest_grid_bitunix.py XRP[A-Z]*/run: python asBack\/backtest_grid_bitunix.py ${symbol}/" \
    .github/workflows/ci.yml

  git add .github/workflows/ci.yml
  git commit -m "ci: run ${name} sweep (${symbol})"
  git push

  log "Pushed ${symbol} — waiting for new run to register..."
  sleep 15

  local new_run_id
  new_run_id=$(gh api "repos/${REPO}/actions/runs?branch=${BRANCH}&per_page=1" \
    | python3 -c "import sys,json; r=json.load(sys.stdin)['workflow_runs'][0]; print(r['id'])")
  log "New run: ${new_run_id}"
  echo "$new_run_id"
}

# =============================================================================
# Main
# =============================================================================

# Parse --start argument
START_FROM="${1:-EMA}"
if [[ "$1" == "--start" && -n "${2:-}" ]]; then
  START_FROM="$2"
fi

cd "$(dirname "$0")/.."
log "Feature CI chain starting from: ${START_FROM}"
echo "Feature CI chain started at $(date)" > "$SUMMARY_FILE"

# Wait for the currently running CI (ATR or later) to complete
log "=== Waiting for current in-progress CI run to finish ==="
wait_for_current_run > "${LOG_DIR}/_current_run_id.txt"
LAST_RUN_ID=$(cat "${LOG_DIR}/_current_run_id.txt")
extract_results "$LAST_RUN_ID" "ATR"

# Now iterate through remaining features
STARTED=false
for feature in "${FEATURES[@]}"; do
  if [[ "$feature" == "$START_FROM" ]]; then
    STARTED=true
  fi
  [[ "$STARTED" == false ]] && continue

  log ""
  log "========================================================="
  log "  Pushing feature: ${feature} (${FEATURE_SYMBOLS[$feature]})"
  log "========================================================="
  push_next_feature "$feature" > "${LOG_DIR}/_new_run_id.txt"
  NEW_RUN_ID=$(cat "${LOG_DIR}/_new_run_id.txt")

  log "Waiting for ${feature} CI run ${NEW_RUN_ID} to complete..."
  wait_for_current_run > "${LOG_DIR}/_current_run_id.txt"
  LAST_RUN_ID=$(cat "${LOG_DIR}/_current_run_id.txt")
  extract_results "$LAST_RUN_ID" "$feature"
done

log ""
log "========================================================="
log "  ALL FEATURES COMPLETE"
log "========================================================="
log "Results summary: ${SUMMARY_FILE}  (relative: logs/chain_results_summary.txt)"
cat "$SUMMARY_FILE"

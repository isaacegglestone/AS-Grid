#!/usr/bin/env bash
# poll_run.sh — polls the latest AS-Grid CI run every 10 minutes
# Runs for up to MAX_HOURS hours, then prints a full XRPPM8+XRPPM9 summary.
# Usage: ./scripts/poll_run.sh [max_hours]

set -euo pipefail

OWNER="isaacegglestone"
REPO="AS-Grid"
BRANCH="feat/bitunix-exchange-adapter"
LOG_FILE="$HOME/git/tmp/latest_run_logs.txt"
SUMMARY_FILE="$HOME/git/tmp/latest_run_summary.txt"
MAX_HOURS="${1:-4}"
POLL_INTERVAL=600   # 10 minutes
MAX_POLLS=$(( MAX_HOURS * 60 / (POLL_INTERVAL / 60) ))

mkdir -p "$HOME/git/tmp"

echo "=================================================="
echo "  AS-Grid CI Poller"
echo "  Branch : $BRANCH"
echo "  Polling: every $(( POLL_INTERVAL / 60 )) min for up to $MAX_HOURS hours"
echo "  Started: $(date)"
echo "=================================================="

# ── helpers ────────────────────────────────────────────────────────────────

get_latest_run() {
    ~/bin/ghapi actions list-runs "$OWNER" "$REPO" \
        --branch "$BRANCH" --per-page 1 2>/dev/null \
    | python3 -c "
import sys, json
data = json.load(sys.stdin)
runs = data.get('workflow_runs', [])
if not runs:
    print('NO_RUNS')
    sys.exit(0)
r = runs[0]
print(r['id'], r['status'], r['conclusion'] or 'pending', r['html_url'])
" 2>/dev/null || echo "ERROR"
}

get_run_steps() {
    local run_id="$1"
    ~/bin/ghapi actions list-run-jobs "$OWNER" "$REPO" "$run_id" 2>/dev/null \
    | python3 -c "
import sys, json
data = json.load(sys.stdin)
jobs = data.get('jobs', [])
for job in jobs:
    print(f\"  Job: {job['name']} | status={job['status']} conclusion={job.get('conclusion','pending')}\")
    for step in job.get('steps', []):
        status = step['status']
        conc   = step.get('conclusion') or ''
        icon   = {'success':'OK','failure':'FAIL','skipped':'SKIP'}.get(conc, '...')
        print(f\"    [{icon}] {step['number']:>2}. {step['name']}\")
" 2>/dev/null || echo "  (could not fetch steps)"
}

fetch_and_parse_logs() {
    local job_id="$1"
    echo ""
    echo "Fetching logs for job $job_id ..."
    ~/bin/ghapi actions get-job-logs "$OWNER" "$REPO" "$job_id" > "$LOG_FILE" 2>/dev/null
    local lines
    lines=$(wc -l < "$LOG_FILE")
    echo "  Log lines: $lines  ->  $LOG_FILE"
    python3 /Users/isaac.egglestone/git/tmp/parse_run.py "$LOG_FILE" | tee "$SUMMARY_FILE"
}

notify() {
    local title="$1"
    local msg="$2"
    # macOS notification
    osascript -e "display notification \"$msg\" with title \"$title\" sound name \"Glass\"" 2>/dev/null || true
    # Also print prominently
    echo ""
    echo "=============================="
    echo "  $title"
    echo "  $msg"
    echo "=============================="
}

# ── write the log parser inline ────────────────────────────────────────────

cat > /Users/isaac.egglestone/git/tmp/parse_run.py << 'PYEOF'
#!/usr/bin/env python3
"""Parse AS-Grid backtest log and print a result summary table."""
import re, sys

log_file = sys.argv[1] if len(sys.argv) > 1 else '/Users/isaac.egglestone/git/tmp/latest_run_logs.txt'

def strip_ts(line):
    return re.sub(r'^.*?Z ', '', line).strip()

results = {}  # name -> {return, trades, max_dd}
bests   = {}  # group_key -> {name, return}

for raw in open(log_file):
    line = strip_ts(raw)

    # Return line
    m = re.search(r'Strategy: ([\w_]+)', line)
    if m:
        current = m.group(1)
        results.setdefault(current, {})
        continue

    m = re.search(r'return:\s*([-\d.]+)%\s+trades:\s*(\d+)\s+max_dd:\s*([\d.]+)%', line)
    if m and 'current' in dir():
        results[current] = {
            'return': float(m.group(1)),
            'trades': int(m.group(2)),
            'max_dd': float(m.group(3)),
        }

    m = re.search(r'Best strategy:\s*([\w_]+)\s+return:\s*([-\d.]+)%', line)
    if m:
        key = 'best_' + ('2y' if '2y' in m.group(1) else '6m')
        group = re.sub(r'^(2y_)?pm\d+_?.*', lambda x: x.group(0).split('_')[0] + ('_' + x.group(0).split('_')[1] if '_' in x.group(0) else ''), m.group(1))
        bests[m.group(1)] = float(m.group(2))

# Print tables for target groups
def table(title, prefix, names):
    rows = [(n, results[n]) for n in names if n in results and results[n]]
    if not rows:
        return
    print(f'\n{"=" * 65}')
    print(f'  {title}')
    print(f'{"=" * 65}')
    print(f'  {"Strategy":<26} {"return":>8}  {"trades":>7}  {"max_dd":>8}')
    print(f'  {"-" * 26}  {"--------":>8}  {"-------":>7}  {"--------":>8}')
    best_ret = max(r['return'] for _, r in rows)
    for name, r in sorted(rows, key=lambda x: -x[1]['return']):
        flag = ' <<' if r['return'] == best_ret else ''
        print(f'  {name:<26}  {r["return"]:>7.2f}%  {r["trades"]:>7}  {r["max_dd"]:>7.2f}%{flag}')

# XRPPM8 6m
table('XRPPM8 — 6-month OOS (Aug 2025 → Feb 2026)', 'pm8',
    ['pm8_baseline','pm8_h0','pm8_h1','pm8_sp10','pm8_sp12','pm8_sp13','pm8_h0_sp12'])

# XRPPM8 2y
table('XRPPM8 — 2-year walk-forward (Feb 2024 → Feb 2026)', '2y_pm8',
    ['2y_pm8_baseline','2y_pm8_h0','2y_pm8_h1','2y_pm8_sp10','2y_pm8_sp12','2y_pm8_sp13','2y_pm8_h0_sp12'])

# XRPPM9 6m
table('XRPPM9 — 6-month OOS (Aug 2025 → Feb 2026)', 'pm9',
    ['pm9_baseline','pm9_vas','pm9_vas_tight','pm9_vas_wide',
     'pm9_btbw','pm9_btbw_tight','pm9_btbw_xtight'])

# XRPPM9 2y
table('XRPPM9 — 2-year walk-forward (Feb 2024 → Feb 2026)', '2y_pm9',
    ['2y_pm9_baseline','2y_pm9_vas','2y_pm9_vas_tight','2y_pm9_vas_wide',
     '2y_pm9_btbw','2y_pm9_btbw_tight','2y_pm9_btbw_xtight'])

print()
PYEOF

# ── main poll loop ──────────────────────────────────────────────────────────

poll=0
last_run_id=""
last_status=""

while (( poll < MAX_POLLS )); do
    poll=$(( poll + 1 ))
    now=$(date '+%H:%M:%S')

    read -r run_id status conclusion url <<< "$(get_latest_run)"

    if [[ "$run_id" == "NO_RUNS" || "$run_id" == "ERROR" ]]; then
        echo "[$now] poll $poll/$MAX_POLLS — could not fetch run info, retrying..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Step progress
    if [[ "$run_id" != "$last_run_id" || "$status" != "$last_status" ]]; then
        echo ""
        echo "[$now] poll $poll/$MAX_POLLS — Run #$run_id  status=$status  conclusion=$conclusion"
        echo "  $url"
        get_run_steps "$run_id"
        last_run_id="$run_id"
        last_status="$status"
    else
        echo "[$now] poll $poll/$MAX_POLLS — Run #$run_id  $status ($conclusion) — no change"
    fi

    if [[ "$status" == "completed" ]]; then
        # Fetch Stage 1 job ID
        job_id=$(~/bin/ghapi actions list-run-jobs "$OWNER" "$REPO" "$run_id" 2>/dev/null \
            | python3 -c "
import sys, json
data = json.load(sys.stdin)
for j in data.get('jobs', []):
    if 'Stage 1' in j.get('name',''):
        print(j['id'])
        break
" 2>/dev/null)

        if [[ -n "$job_id" ]]; then
            fetch_and_parse_logs "$job_id"
        else
            echo "  Could not find Stage 1 job ID — logs unavailable"
        fi

        if [[ "$conclusion" == "success" ]]; then
            notify "AS-Grid Run Complete" "Run #$run_id passed — check $SUMMARY_FILE"
        else
            notify "AS-Grid Run FAILED" "Run #$run_id conclusion=$conclusion"
        fi
        echo ""
        echo "Done. Full logs: $LOG_FILE"
        echo "Summary      : $SUMMARY_FILE"
        exit 0
    fi

    sleep "$POLL_INTERVAL"
done

echo ""
echo "[$(date '+%H:%M:%S')] Timeout after $MAX_HOURS hours — run still in progress."
echo "Last known: run=$last_run_id status=$last_status"
notify "AS-Grid Poller Timeout" "No result after ${MAX_HOURS}h — check GitHub manually"
exit 1

#!/usr/bin/env python3
"""
poll_run.py — AS-Grid CI run poller
Polls GitHub Actions every 10 minutes for up to 4 hours.
When the run completes, fetches logs, parses XRPPM8 + XRPPM9 + XRPPM10 results,
and fires a macOS notification.

Usage: python3 poll_run.py [max_hours]
"""
import json, os, re, subprocess, sys, time
from datetime import datetime

OWNER          = "isaacegglestone"
REPO           = "AS-Grid"
BRANCH         = "feat/bitunix-exchange-adapter"
GHAPI          = os.path.expanduser("~/bin/ghapi")
LOG_FILE       = os.path.expanduser("~/git/tmp/latest_run_logs.txt")
SUMMARY_FILE   = os.path.expanduser("~/git/tmp/latest_run_summary.txt")
MAX_HOURS      = float(sys.argv[1]) if len(sys.argv) > 1 else 999.0
POLL_SECS      = 1200  # 20 minutes

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def ts():
    return datetime.now().strftime("%H:%M:%S")


def ghapi(*args, **kwargs):
    result = subprocess.run(
        [GHAPI] + list(args),
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def _as_list(data):
    """Normalise ghapi response — may be a list or a dict with a list field."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("workflow_runs", "jobs", "steps"):
            if key in data:
                return data[key]
    return []


def get_latest_run():
    data = ghapi("actions", "list-runs", OWNER, REPO,
                 "--branch", BRANCH, "--per-page", "1")
    if not data:
        return None
    runs = _as_list(data)
    if not runs:
        return None
    return runs[0]


def get_stage1_job_id(run_id):
    data = ghapi("actions", "list-jobs", OWNER, REPO, str(run_id))
    if not data:
        return None
    for job in _as_list(data):
        if "Stage 1" in job.get("name", ""):
            return job["id"]
    return None


def get_step_summary(run_id):
    data = ghapi("actions", "list-jobs", OWNER, REPO, str(run_id))
    if not data:
        return "  (could not fetch steps)"
    lines = []
    for job in _as_list(data):
        lines.append(f"  Job: {job['name']} | status={job['status']} conclusion={job.get('conclusion','pending')}")
        for step in job.get("steps", []):
            conc = step.get("conclusion") or ""
            icon = {"success": "OK", "failure": "FAIL", "skipped": "SKIP"}.get(conc, "..")
            lines.append(f"    [{icon}] {step['number']:>2}. {step['name']}")
    return "\n".join(lines)


def fetch_logs(job_id):
    print(f"[{ts()}] Fetching logs for job {job_id} ...")
    result = subprocess.run(
        [GHAPI, "actions", "get-job-logs", OWNER, REPO, str(job_id)],
        capture_output=True, text=True, timeout=120
    )
    with open(LOG_FILE, "w") as f:
        f.write(result.stdout)
    lines = result.stdout.count("\n")
    print(f"[{ts()}] Log saved: {lines} lines -> {LOG_FILE}")


def parse_results(log_file):
    def strip_ts(line):
        return re.sub(r"^.*?Z ", "", line).strip()

    results = {}
    current = None

    for raw in open(log_file):
        line = strip_ts(raw)
        m = re.search(r"Strategy: ([\w_]+)", line)
        if m:
            current = m.group(1)
            results.setdefault(current, {})
            continue
        m = re.search(r"return:\s*([-\d.]+)%\s+trades:\s*(\d+)\s+max_dd:\s*([\d.]+)%", line)
        if m and current:
            results[current] = {
                "return": float(m.group(1)),
                "trades": int(m.group(2)),
                "max_dd": float(m.group(3)),
            }

    def table(title, names):
        rows = [(n, results[n]) for n in names if n in results and results[n]]
        if not rows:
            return f"\n  [{title}] — no results found yet\n"
        out = [f"\n{'=' * 65}", f"  {title}", f"{'=' * 65}",
               f"  {'Strategy':<28} {'return':>8}  {'trades':>7}  {'max_dd':>8}",
               f"  {'-' * 28}  {'--------':>8}  {'-------':>7}  {'--------':>8}"]
        best = max(r["return"] for _, r in rows)
        for name, r in sorted(rows, key=lambda x: -x[1]["return"]):
            flag = " <<<" if r["return"] == best else ""
            out.append(f"  {name:<28}  {r['return']:>7.2f}%  {r['trades']:>7}  {r['max_dd']:>7.2f}%{flag}")
        return "\n".join(out)

    pm8_6m  = ["pm8_baseline","pm8_h0","pm8_h1","pm8_sp10","pm8_sp12","pm8_sp13","pm8_h0_sp12"]
    pm8_2y  = ["2y_pm8_baseline","2y_pm8_h0","2y_pm8_h1","2y_pm8_sp10","2y_pm8_sp12","2y_pm8_sp13","2y_pm8_h0_sp12"]
    pm9_6m  = ["pm9_baseline","pm9_vas","pm9_vas_tight","pm9_vas_wide","pm9_btbw","pm9_btbw_tight","pm9_btbw_xtight"]
    pm9_2y  = ["2y_pm9_baseline","2y_pm9_vas","2y_pm9_vas_tight","2y_pm9_vas_wide","2y_pm9_btbw","2y_pm9_btbw_tight","2y_pm9_btbw_xtight"]
    pm10_6m  = ["pm10_baseline","pm10_btbw","pm10_btbw_tight","pm10_btbw_xtight"]
    pm10_2y  = ["2y_pm10_baseline","2y_pm10_btbw","2y_pm10_btbw_tight"]
    pm10_mid = ["mid_pm10_baseline","mid_pm10_btbw","mid_pm10_btbw_tight"]

    summary = "\n".join([
        f"AS-Grid run results parsed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        table("XRPPM8 — 6m OOS (Aug 2025 -> Feb 2026)", pm8_6m),
        table("XRPPM8 — 2y walk-forward (Feb 2024 -> Feb 2026)", pm8_2y),
        table("XRPPM9 — 6m OOS (Aug 2025 -> Feb 2026)", pm9_6m),
        table("XRPPM9 — 2y walk-forward (Feb 2024 -> Feb 2026)", pm9_2y),
        table("XRPPM10 — 6m OOS (Aug 2025 -> Feb 2026)", pm10_6m),
        table("XRPPM10 — 2y walk-forward (Feb 2024 -> Feb 2026)", pm10_2y),
        table("XRPPM10 — mid-year (Aug 2024 -> Aug 2025)", pm10_mid),
        "",
    ])
    return summary


def notify(title, msg):
    script = f'display notification "{msg}" with title "{title}" sound name "Glass"'
    subprocess.run(["osascript", "-e", script], capture_output=True)
    print(f"\n>>> NOTIFICATION: {title} — {msg}\n")


# ── main ────────────────────────────────────────────────────────────────────

max_polls = int(MAX_HOURS * 60 / (POLL_SECS / 60))
print("=" * 60)
print("  AS-Grid CI Poller (Python)")
print(f"  Branch  : {BRANCH}")
print(f"  Interval: {POLL_SECS // 60} min  |  Timeout: {MAX_HOURS}h ({max_polls} polls)")
print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

last_run_id   = None
last_status   = None
last_step_poll = 0

for poll in range(1, max_polls + 1):
    print(f"\n[{ts()}] poll {poll}/{max_polls}", flush=True)

    run = get_latest_run()
    if not run:
        print("  Could not fetch run — retrying next poll")
        time.sleep(POLL_SECS)
        continue

    run_id     = run["id"]
    status     = run["status"]
    conclusion = run.get("conclusion") or "pending"
    url        = run["html_url"]

    changed = (run_id != last_run_id or status != last_status)
    if changed or (poll - last_step_poll) >= 3:
        print(f"  Run #{run_id}  status={status}  conclusion={conclusion}")
        print(f"  {url}")
        print(get_step_summary(run_id))
        last_step_poll = poll
    else:
        print(f"  Run #{run_id}  {status} ({conclusion}) — no change")

    last_run_id = run_id
    last_status = status

    if status == "completed":
        print(f"\n[{ts()}] Run #{run_id} completed with conclusion={conclusion}")
        job_id = get_stage1_job_id(run_id)
        if job_id:
            fetch_logs(job_id)
            summary = parse_results(LOG_FILE)
            print(summary)
            with open(SUMMARY_FILE, "w") as f:
                f.write(summary)
            print(f"Summary saved -> {SUMMARY_FILE}")
        else:
            print("  Could not find Stage 1 job ID — fetch logs manually")

        if conclusion == "success":
            notify("AS-Grid Run Complete",
                   f"Run #{run_id} passed — XRPPM8+9+10 results in {SUMMARY_FILE}")
        else:
            notify("AS-Grid Run FAILED",
                   f"Run #{run_id} conclusion={conclusion}")
        sys.exit(0)

    time.sleep(POLL_SECS)

print(f"\n[{ts()}] Timeout after {MAX_HOURS}h — run still in progress")
notify("AS-Grid Poller Timeout", f"No result after {MAX_HOURS}h — check GitHub manually")
sys.exit(1)

#!/usr/bin/env python3
"""
ci_status.py
============
Query GitHub Actions CI runs and job details for AS-Grid.

Usage:
    python3 scripts/ci_status.py                  # Show recent runs + chain status
    python3 scripts/ci_status.py --run 22531373089 # Detail a specific run
    python3 scripts/ci_status.py --logs 22531373089 # Print backtest output from a run
    python3 scripts/ci_status.py --watch           # Poll every 30s until Stage 1 completes
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = "isaacegglestone/AS-Grid"
BRANCH = "feat/bitunix-exchange-adapter"
REPO_ROOT = Path(__file__).resolve().parent.parent

# ANSI colours (disabled when not a tty)
_USE_COLOUR = sys.stdout.isatty()
GREEN  = "\033[32m" if _USE_COLOUR else ""
YELLOW = "\033[33m" if _USE_COLOUR else ""
RED    = "\033[31m" if _USE_COLOUR else ""
CYAN   = "\033[36m" if _USE_COLOUR else ""
BOLD   = "\033[1m"  if _USE_COLOUR else ""
RESET  = "\033[0m"  if _USE_COLOUR else ""


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def gh_api(path: str, method: str = "GET") -> Any:
    """Call `gh api repos/<REPO>/<path>` and return parsed JSON."""
    result = subprocess.run(
        ["gh", "api", "--method", method, f"repos/{REPO}/{path}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh api failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def gh_run_logs(run_id: int) -> str:
    """
    Fetch log text for the Stage 1 job of a run.
    Uses per-job log endpoint so it works even while the overall run is still
    in progress (e.g. waiting for EC2 manual approval on Stage 2).
    Falls back to `gh run view --log` for fully completed runs.
    """
    # Try fetching Stage 1 job logs by job ID first
    try:
        jobs_data = gh_api(f"actions/runs/{run_id}/jobs")
        stage1 = next(
            (j for j in jobs_data["jobs"]
             if "Stage 1" in j["name"] or "GitHub runner" in j["name"]),
            None,
        )
        if stage1 and stage1["status"] == "completed":
            job_id = stage1["id"]
            result = subprocess.run(
                ["gh", "api", f"repos/{REPO}/actions/jobs/{job_id}/logs"],
                capture_output=True, text=True, cwd=REPO_ROOT,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
    except Exception:
        pass

    # Fall back to whole-run log (only works when run is fully complete)
    result = subprocess.run(
        ["gh", "run", "view", str(run_id), "--repo", REPO, "--log"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    return result.stdout


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _age(iso_ts: str) -> str:
    """Return human-readable age from ISO timestamp."""
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        diff = datetime.now(timezone.utc) - dt
        mins = int(diff.total_seconds() // 60)
        if mins < 60:
            return f"{mins}m ago"
        return f"{mins // 60}h {mins % 60}m ago"
    except Exception:
        return iso_ts[:16]


def _status_icon(status: str, conclusion: str | None) -> str:
    conclusion = conclusion or ""
    if status == "completed":
        if conclusion == "success":
            return f"{GREEN}✅{RESET}"
        if conclusion == "cancelled":
            return f"{YELLOW}⊘ {RESET}"
        return f"{RED}❌{RESET}"
    if status in ("in_progress", "waiting", "queued", "pending"):
        return f"{YELLOW}🔄{RESET}"
    return "·"


def _conclusion_colour(conclusion: str | None) -> str:
    c = conclusion or "none"
    if c == "success":
        return f"{GREEN}{c}{RESET}"
    if c in ("failure", "timed_out"):
        return f"{RED}{c}{RESET}"
    if c == "cancelled":
        return f"{YELLOW}{c}{RESET}"
    return c


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def show_runs(n: int = 10) -> None:
    """Print recent runs table + chain process status."""
    data = gh_api(f"actions/runs?branch={BRANCH}&per_page={n}")
    runs = data["workflow_runs"]

    # Chain process
    chain_pids = subprocess.run(
        ["pgrep", "-f", "run_feature_ci_chain.py"], capture_output=True, text=True
    ).stdout.strip()
    chain_status = f"{GREEN}RUNNING pid={chain_pids}{RESET}" if chain_pids else f"{RED}NOT RUNNING{RESET}"

    # Chain log tail
    log_path = REPO_ROOT / "logs" / "chain_run.log"
    if log_path.exists():
        with open(log_path) as f:
            last_log_lines = f.readlines()[-3:]
        log_tail = "".join(last_log_lines).strip()
    else:
        log_tail = "(no log yet)"

    print(f"\n{BOLD}=== AS-Grid CI Status === {datetime.now().strftime('%H:%M:%S')}{RESET}\n")
    print(f"Chain: {chain_status}")
    print(f"  {log_tail.splitlines()[-1] if log_tail else ''}\n")

    print(f"{BOLD}{'Icon':<5} {'Run ID':<16} {'Status':<12} {'Conclusion':<12} {'Age':<12} {'Name'}{RESET}")
    print("─" * 72)
    for run in runs:
        icon = _status_icon(run["status"], run.get("conclusion"))
        conclusion = _conclusion_colour(run.get("conclusion"))
        print(
            f"{icon:<5} {run['id']:<16} {run['status']:<12} {conclusion:<22} "
            f"{_age(run['created_at']):<12} {run['name']}"
        )


def show_run_detail(run_id: int) -> None:
    """Print jobs and steps for a specific run."""
    run = gh_api(f"actions/runs/{run_id}")
    jobs_data = gh_api(f"actions/runs/{run_id}/jobs")
    jobs = jobs_data["jobs"]

    icon = _status_icon(run["status"], run.get("conclusion"))
    print(f"\n{BOLD}Run {run_id}{RESET} {icon}  {run['status']}/{run.get('conclusion') or 'none'}")
    print(f"  Branch:  {run['head_branch']}")
    print(f"  Commit:  {run['head_sha'][:12]}  {run['head_commit']['message'][:60]}")
    print(f"  Created: {run['created_at'][:19]}  ({_age(run['created_at'])})")
    print(f"  Updated: {run['updated_at'][:19]}")

    for job in jobs:
        icon = _status_icon(job["status"], job.get("conclusion"))
        print(f"\n  {icon} {BOLD}{job['name']}{RESET}  {job['status']}/{job.get('conclusion') or 'none'}")
        if job.get("started_at"):
            start = datetime.fromisoformat(job["started_at"].replace("Z", "+00:00"))
            end_ts = job.get("completed_at")
            if end_ts:
                end = datetime.fromisoformat(end_ts.replace("Z", "+00:00"))
                duration = int((end - start).total_seconds())
                print(f"     Duration: {duration // 60}m {duration % 60}s")
        for step in job.get("steps", []):
            s_icon = _status_icon(step["status"], step.get("conclusion"))
            print(f"       {s_icon}  {step['name']}")


def show_logs(run_id: int) -> None:
    """Print backtest result lines from a run's logs."""
    print(f"\n{BOLD}Fetching logs for run {run_id}...{RESET}")
    raw = gh_run_logs(run_id)
    if not raw:
        print("(no log output returned)")
        return

    keywords = ("Strategy:", "return:", "Best strategy:", "OOS", "walk-forward",
                "WARN", "ERROR", "Traceback", "Error:")
    lines = [l for l in raw.splitlines() if any(k in l for k in keywords)]

    if not lines:
        print("(no backtest result lines found — job may not have reached that step yet)")
        # Show last 40 lines of raw log anyway
        print(f"\n{CYAN}Raw log tail:{RESET}")
        for l in raw.splitlines()[-40:]:
            print(l)
    else:
        for l in lines:
            if "Best strategy:" in l:
                print(f"{GREEN}{l}{RESET}")
            elif "ERROR" in l or "Traceback" in l:
                print(f"{RED}{l}{RESET}")
            else:
                print(l)


def watch_stage1(run_id: int | None, interval: int = 30) -> None:
    """Poll Stage 1 of the latest (or given) run until complete."""
    if run_id is None:
        data = gh_api(f"actions/runs?branch={BRANCH}&per_page=10")
        for r in data["workflow_runs"]:
            if r.get("conclusion") != "cancelled":
                run_id = r["id"]
                break
    print(f"{BOLD}Watching Stage 1 of run {run_id} (polling every {interval}s) — Ctrl+C to stop{RESET}\n")
    poll = 0
    while True:
        poll += 1
        jobs_data = gh_api(f"actions/runs/{run_id}/jobs")
        stage1 = next(
            (j for j in jobs_data["jobs"] if "Stage 1" in j["name"] or "GitHub runner" in j["name"]),
            None,
        )
        ts = datetime.now().strftime("%H:%M:%S")
        if stage1 is None:
            print(f"[{ts}] [{poll}] Stage 1 not yet visible...")
        else:
            icon = _status_icon(stage1["status"], stage1.get("conclusion"))
            print(f"[{ts}] [{poll}] {icon} Stage1={stage1['status']}/{stage1.get('conclusion') or 'none'}")
            if stage1["status"] == "completed":
                conclusion = stage1.get("conclusion")
                if conclusion == "success":
                    print(f"\n{GREEN}✅ Stage 1 complete! Fetching results...{RESET}")
                    show_logs(run_id)
                else:
                    print(f"\n{RED}❌ Stage 1 finished: {conclusion}{RESET}")
                    show_logs(run_id)
                return
        time.sleep(interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AS-Grid CI status tool")
    parser.add_argument("--run", type=int, metavar="RUN_ID",
                        help="Show detailed job/step info for a run")
    parser.add_argument("--logs", type=int, metavar="RUN_ID",
                        help="Print backtest output lines from a run")
    parser.add_argument("--watch", action="store_true",
                        help="Poll Stage 1 of the latest run until done")
    parser.add_argument("--watch-run", type=int, metavar="RUN_ID",
                        help="Poll Stage 1 of a specific run until done")
    parser.add_argument("--interval", type=int, default=30,
                        help="Poll interval in seconds for --watch (default: 30)")
    parser.add_argument("-n", type=int, default=10,
                        help="Number of recent runs to show (default: 10)")
    args = parser.parse_args()

    if args.run:
        show_run_detail(args.run)
    elif args.logs:
        show_logs(args.logs)
    elif args.watch or args.watch_run:
        watch_stage1(args.watch_run, interval=args.interval)
    else:
        show_runs(n=args.n)


if __name__ == "__main__":
    main()

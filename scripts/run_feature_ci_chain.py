#!/usr/bin/env python3
"""
run_feature_ci_chain.py
=======================
Drives the sequential CI push-chain for feature backtests v4-v8.

Each feature is pushed only after the previous CI Stage 1 (GitHub runner)
completes successfully. Stage 2 (EC2) requires manual approval and is ignored.

Usage:
    python3 scripts/run_feature_ci_chain.py [--start FEATURE]

    --start FEATURE   Resume from a feature name: EMA BB RSI VOL MS
                      (default: EMA, meaning ATR must already be running)

Results are written to:
    logs/chain_results_summary.txt   — best strategy per feature
    logs/chain_ci_<FEATURE>.log      — full output per feature run
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO = "isaacegglestone/AS-Grid"
BRANCH = "feat/bitunix-exchange-adapter"
POLL_INTERVAL = 90  # seconds between GitHub API polls

FEATURES = ["EMA", "BB", "RSI", "VOL", "MS"]

FEATURE_SYMBOLS = {
    "EMA": "XRPEMA",
    "BB":  "XRPBB",
    "RSI": "XRPRSI",
    "VOL": "XRPVOL",
    "MS":  "XRPMS",
}

FEATURE_NAMES = {
    "EMA": "v4 EMA bias filter",
    "BB":  "v5 BB squeeze filter",
    "RSI": "v6 RSI filter",
    "VOL": "v7 Volume confirmation",
    "MS":  "v8 Market structure filter",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
SUMMARY_FILE = LOG_DIR / "chain_results_summary.txt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def gh_api(path: str, method: str = "GET", extra_args: list[str] | None = None) -> dict:
    """Call gh api and return parsed JSON. Raises on non-zero exit."""
    cmd = ["gh", "api"]
    if method != "GET":
        cmd += ["--method", method]
    if extra_args:
        cmd += extra_args
    cmd.append(f"repos/{REPO}/{path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gh api failed ({result.returncode}): {result.stderr.strip()}")
    return json.loads(result.stdout)


def git(*args: str) -> str:
    """Run a git command from REPO_ROOT. Returns stdout."""
    result = subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def gh_run_stage1_logs(run_id: int) -> str:
    """
    Fetch Stage 1 job logs by job ID — works while Stage 2 is pending/waiting.
    """
    try:
        jobs_data = gh_api(f"actions/runs/{run_id}/jobs")
        stage1 = next(
            (j for j in jobs_data["jobs"]
             if "Stage 1" in j["name"] or "GitHub runner" in j["name"]),
            None,
        )
        if stage1:
            job_id = stage1["id"]
            result = subprocess.run(
                ["gh", "api", f"repos/{REPO}/actions/jobs/{job_id}/logs"],
                capture_output=True, text=True, cwd=REPO_ROOT,
            )
            if result.returncode == 0:
                return result.stdout
    except Exception:
        pass
    # Fallback to whole-run log (only works when run fully complete)
    result = subprocess.run(
        ["gh", "run", "view", str(run_id), "--repo", REPO, "--log"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    return result.stdout


# ---------------------------------------------------------------------------
# Core: find latest non-cancelled run
# ---------------------------------------------------------------------------

def get_active_run_id() -> int:
    """Return the run_id of the most recent non-cancelled run on BRANCH."""
    data = gh_api(f"actions/runs?branch={BRANCH}&per_page=20")
    for run in data["workflow_runs"]:
        if run.get("conclusion") != "cancelled":
            return run["id"]
    raise RuntimeError("No non-cancelled run found on branch")


def cancel_run(run_id: int) -> None:
    """Cancel a run (e.g. to dismiss a stuck Stage 2 EC2 approval)."""
    subprocess.run(
        ["gh", "api", "--method", "POST",
         f"repos/{REPO}/actions/runs/{run_id}/cancel"],
        capture_output=True, text=True,
    )
    log(f"Cancelled run {run_id} (dismissed Stage 2 EC2 approval wait)")
    time.sleep(5)  # give GitHub a moment to update state


# ---------------------------------------------------------------------------
# Core: poll Stage 1 job until complete
# ---------------------------------------------------------------------------

def wait_for_stage1(run_id: int) -> None:
    """
    Poll Stage 1 (GitHub runner) job for run_id until it completes.
    Raises RuntimeError if Stage 1 fails.
    """
    poll_count = 0
    log(f"Monitoring Stage 1 of run {run_id}...")

    while True:
        poll_count += 1
        jobs_data = gh_api(f"actions/runs/{run_id}/jobs")
        jobs = jobs_data["jobs"]

        # Find Stage 1 job
        stage1 = next(
            (j for j in jobs if "Stage 1" in j["name"] or "GitHub runner" in j["name"]),
            None,
        )

        if stage1 is None:
            log(f"  [{poll_count}] run={run_id} Stage 1 job not yet visible, waiting...")
        else:
            status = stage1["status"]
            conclusion = stage1.get("conclusion") or "none"
            log(f"  [{poll_count}] run={run_id} Stage1={status}/{conclusion}")

            if status == "completed":
                if conclusion != "success":
                    raise RuntimeError(f"Stage 1 failed with conclusion: {conclusion}")
                log(f"✅ Stage 1 succeeded (run {run_id})")
                return

        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Core: extract results from a completed run
# ---------------------------------------------------------------------------

def extract_results(run_id: int, feature: str) -> None:
    """Pull run logs and write best-strategy lines to logs/."""
    log_file = LOG_DIR / f"chain_ci_{feature}.log"
    log(f"Fetching run logs for {feature} (run {run_id}) → {log_file.name}")

    raw_logs = gh_run_stage1_logs(run_id)
    relevant = [
        line for line in raw_logs.splitlines()
        if any(kw in line for kw in ("Strategy:", "return:", "Best strategy:", "OOS", "walk-forward"))
    ]

    with open(log_file, "w") as f:
        f.write(f"=== {feature} (run {run_id}) ===\n")
        f.write(raw_logs)

    best_lines = [l for l in relevant if "Best strategy:" in l]
    best_6m = best_lines[0] if len(best_lines) > 0 else "n/a"
    best_2y = best_lines[-1] if len(best_lines) > 1 else "n/a"

    with open(SUMMARY_FILE, "a") as f:
        f.write(f"\n=== {feature} (run {run_id}) ===\n")
        f.write(f"  6m OOS:  {best_6m.strip()}\n")
        f.write(f"  2y:      {best_2y.strip()}\n")
        for line in relevant:
            f.write(f"  {line.strip()}\n")
        f.write("\n")

    log(f"  Best 6m:  {best_6m.strip()}")
    log(f"  Best 2y:  {best_2y.strip()}")


# ---------------------------------------------------------------------------
# Core: push the next feature
# ---------------------------------------------------------------------------

def push_feature(feature: str) -> int:
    """
    Update ci.yml for feature, commit, push, and return the new run_id.
    """
    symbol = FEATURE_SYMBOLS[feature]
    name = FEATURE_NAMES[feature]
    ci_yml = REPO_ROOT / ".github" / "workflows" / "ci.yml"

    log(f"Updating ci.yml → {symbol} ({name})")
    content = ci_yml.read_text()
    # Replace the step name
    content = re.sub(
        r"- name: Backtest XRP.*",
        f"- name: Backtest XRP/USDT — {name}",
        content,
    )
    # Replace the run command symbol
    content = re.sub(
        r"run: python asBack/backtest_grid_bitunix\.py XRP[A-Z]*",
        f"run: python asBack/backtest_grid_bitunix.py {symbol}",
        content,
    )
    ci_yml.write_text(content)

    git("add", ".github/workflows/ci.yml")
    git("commit", "-m", f"ci: run {name} sweep ({symbol})")
    git("push")
    log(f"Pushed {symbol} — waiting 20s for run to register...")
    time.sleep(20)

    data = gh_api(f"actions/runs?branch={BRANCH}&per_page=1")
    new_run_id = data["workflow_runs"][0]["id"]
    log(f"New run: {new_run_id}")
    return new_run_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential feature CI chain")
    parser.add_argument("--start", default="EMA", choices=FEATURES,
                        help="Feature to start from (default: EMA)")
    args = parser.parse_args()

    log(f"Feature CI chain starting from: {args.start}")
    SUMMARY_FILE.write_text(f"Feature CI chain started at {datetime.now()}\n")

    # Wait for currently running CI (ATR) to finish Stage 1
    log("=== Waiting for current in-progress CI run (ATR) to finish Stage 1 ===")
    current_run_id = get_active_run_id()
    wait_for_stage1(current_run_id)    cancel_run(current_run_id)  # dismiss EC2 Stage 2 approval to free concurrency queue    extract_results(current_run_id, "ATR")

    # Iterate through remaining features
    started = False
    for feature in FEATURES:
        if feature == args.start:
            started = True
        if not started:
            continue

        log("")
        log("=" * 60)
        log(f"  Pushing feature: {feature} ({FEATURE_SYMBOLS[feature]})")
        log("=" * 60)

        new_run_id = push_feature(feature)
        log(f"Waiting for {feature} CI run {new_run_id} Stage 1 to complete...")
        wait_for_stage1(new_run_id)
        cancel_run(new_run_id)  # dismiss EC2 Stage 2 approval to free concurrency queue
        extract_results(new_run_id, feature)

    log("")
    log("=" * 60)
    log("  ALL FEATURES COMPLETE")
    log("=" * 60)
    log(f"Results summary: {SUMMARY_FILE}")
    print(SUMMARY_FILE.read_text())


if __name__ == "__main__":
    main()

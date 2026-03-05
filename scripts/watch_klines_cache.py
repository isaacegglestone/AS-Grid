"""
scripts/watch_klines_cache.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Polls the latest ``cache-klines.yml`` GitHub Actions run on the current branch
until it completes, then downloads and extracts the ``klines-cache`` artifact
to ``asBack/klines_cache/``.

Usage
-----
    python scripts/watch_klines_cache.py

Requirements: ``gh`` CLI authenticated, run from repo root.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO = "princeniu/AS-Grid"
WORKFLOW = "cache-klines.yml"
ARTIFACT_NAME = "klines-cache"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "asBack", "klines_cache")
POLL_INTERVAL_SEC = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], check=True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def gh(*args) -> dict | list:
    """Run a `gh api` command and return parsed JSON."""
    result = run(["gh", "api", "--paginate", *args])
    return json.loads(result.stdout)


def get_branch() -> str:
    r = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return r.stdout.strip()


def get_latest_run(branch: str) -> dict | None:
    """Return the most-recent run of WORKFLOW on *branch*, or None."""
    data = gh(
        f"repos/{REPO}/actions/workflows/{WORKFLOW}/runs",
        "-q", f".workflow_runs[] | select(.head_branch == \"{branch}\")",
        "--jq", f".workflow_runs | map(select(.head_branch == \"{branch}\")) | sort_by(.created_at) | last",
    )
    # gh --jq returns raw JSON scalar; re-parse
    if isinstance(data, str):
        data = json.loads(data)
    return data or None


def get_latest_run_simple(branch: str) -> dict | None:
    """Fetch runs and pick the newest on this branch without --jq (wider gh compat)."""
    result = run([
        "gh", "api",
        f"repos/{REPO}/actions/workflows/{WORKFLOW}/runs",
        "--field", f"branch={branch}",
        "--field", "per_page=5",
    ])
    data = json.loads(result.stdout)
    runs = data.get("workflow_runs", [])
    if not runs:
        return None
    # Sort descending by created_at and return newest
    runs.sort(key=lambda r: r["created_at"], reverse=True)
    return runs[0]


def download_artifact(run_id: int, dest_dir: str) -> bool:
    """Download *ARTIFACT_NAME* from *run_id* into *dest_dir*. Returns True on success."""
    # List artifacts for run
    result = run([
        "gh", "api",
        f"repos/{REPO}/actions/runs/{run_id}/artifacts",
    ])
    artifacts = json.loads(result.stdout).get("artifacts", [])
    target = next((a for a in artifacts if a["name"] == ARTIFACT_NAME), None)
    if not target:
        print(f"  [!] No artifact named '{ARTIFACT_NAME}' found in run {run_id}")
        print(f"      Available: {[a['name'] for a in artifacts]}")
        return False

    artifact_id = target["id"]
    print(f"  → Downloading artifact id={artifact_id} ({target['size_in_bytes']:,} bytes)…")

    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(tempfile.mkdtemp(), f"{ARTIFACT_NAME}.zip")

    # gh CLI download command
    dl = run([
        "gh", "run", "download", str(run_id),
        "--repo", REPO,
        "--name", ARTIFACT_NAME,
        "--dir", dest_dir,
    ])
    if dl.returncode != 0:
        print(f"  [!] Download failed:\n{dl.stderr}")
        return False

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    branch = get_branch()
    print(f"[watch_klines_cache] Repo   : {REPO}")
    print(f"[watch_klines_cache] Branch : {branch}")
    print(f"[watch_klines_cache] Output : {os.path.abspath(OUT_DIR)}")
    print(f"[watch_klines_cache] Polling every {POLL_INTERVAL_SEC}s …\n")

    run_info = None
    poll = 0

    while True:
        poll += 1
        run_info = get_latest_run_simple(branch)

        if run_info is None:
            print(f"  [{poll:03d}] No run found for '{WORKFLOW}' on branch '{branch}' — waiting…")
            time.sleep(POLL_INTERVAL_SEC)
            continue

        run_id = run_info["run_number"]
        db_id  = run_info["id"]
        status = run_info["status"]          # queued / in_progress / completed
        conclusion = run_info.get("conclusion")  # success / failure / cancelled / None

        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"  [{poll:03d}] {now}  run #{run_id} (id={db_id})  status={status}  conclusion={conclusion}")

        if status == "completed":
            if conclusion == "success":
                print(f"\n✓ Workflow succeeded! Downloading artifact '{ARTIFACT_NAME}'…")
                ok = download_artifact(db_id, OUT_DIR)
                if ok:
                    # List what was placed there
                    files = os.listdir(OUT_DIR)
                    print(f"\n✓ Files in {OUT_DIR}:")
                    for f in files:
                        fpath = os.path.join(OUT_DIR, f)
                        size_mb = os.path.getsize(fpath) / 1_048_576
                        print(f"    {f}  ({size_mb:.1f} MB)")
                    print("\nNext steps:")
                    print("  git add asBack/klines_cache/")
                    print("  git commit -m 'data: commit XRPUSDT 15min klines parquet [skip ci]'")
                    print("  git push")
                else:
                    print("[!] Download failed — check output above.")
                    sys.exit(1)
            else:
                print(f"\n✗ Workflow ended with conclusion='{conclusion}'. Nothing to download.")
                sys.exit(1)
            break

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()

"""
scripts/generate_klines_cache.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pre-fetches klines from the Bitunix public API and saves them as parquet files
in ``asBack/klines_cache/``.  Once generated, the backtest engine loads from
disk instead of hitting the API — cutting Stage 1 CI time from ~80 min to
under 2 minutes.

The klines endpoint is **public** — no API key required.

Datasets fetched (one file per symbol+interval, covering the widest date range
used by any config in backtest_grid_bitunix.py):

  XRPUSDT  15min  2022-04-20 → 2026-02-28   (covers MAX, 2Y, 6m, validate)

Usage
-----
    python scripts/generate_klines_cache.py

The script is also run automatically by the "Generate klines cache" GitHub
Actions workflow (.github/workflows/cache-klines.yml).
"""

import asyncio
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Make sure repo root is on sys.path so we can import asBack + src modules.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from asBack.backtest_grid_bitunix import fetch_klines_as_df  # noqa: E402

# ---------------------------------------------------------------------------
# Datasets to cache.  Each entry is (symbol, interval, start_dt, end_dt).
# Add new rows here when new configs with different ranges are introduced.
# ---------------------------------------------------------------------------
DATASETS = [
    (
        "XRPUSDT",
        "15min",
        datetime(2022, 4, 20),   # earliest date used (XRP_MAX_CONFIG)
        datetime(2026, 2, 28),   # latest date used   (all 2Y + MAX configs)
    ),
]


async def main() -> None:
    cache_dir = os.path.join(_REPO_ROOT, "asBack", "klines_cache")
    os.makedirs(cache_dir, exist_ok=True)

    for symbol, interval, start_dt, end_dt in DATASETS:
        out_path = os.path.join(cache_dir, f"{symbol}_{interval}.parquet")

        if os.path.exists(out_path):
            print(f"[skip] {out_path} already exists — delete it to force re-fetch.")
            continue

        print(f"\n{'='*60}")
        print(f"Fetching {symbol} {interval}  {start_dt.date()} → {end_dt.date()}")
        print(f"Output : {out_path}")
        print(f"{'='*60}")

        df = await fetch_klines_as_df(
            symbol=symbol,
            interval=interval,
            start_dt=start_dt,
            end_dt=end_dt,
        )

        df.to_parquet(out_path, index=False)
        size_mb = os.path.getsize(out_path) / 1_048_576
        print(f"  → Saved {len(df):,} rows  ({size_mb:.1f} MB)  →  {out_path}")

    print("\nAll datasets cached.")


if __name__ == "__main__":
    asyncio.run(main())

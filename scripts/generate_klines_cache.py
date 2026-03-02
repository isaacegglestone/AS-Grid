"""
scripts/generate_klines_cache.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pre-fetches klines and saves them as parquet files in ``asBack/klines_cache/``.
Once generated, the backtest engine loads from disk instead of hitting the API.

Data sources
------------
- **Bitunix public API** — XRPUSDT available from ~2022-04-20 onward.
- **Binance Data Vision** (data.binance.vision) — XRPUSDT spot data from ~2019-10-01,
  giving a full ~6.5-year history including the 2021 bull run, 2022 crash
  (LUNA/FTX) and the 2023-2025 recovery cycle.  Downloads monthly ZIP/CSV files
  with no API key and no geographic restrictions.

Both datasets are stitched from both sources:
  Binance  2019-10-01 → 2022-04-20  (pre-Bitunix history)
  Bitunix  2022-04-20 → now          (live-exchange data used by the bot)

The 1min stitched dataset is ~200–300 MB uncompressed (~60–80 MB as parquet).
It is stored in GitHub Actions cache (not committed to git).

Usage
-----
    python scripts/generate_klines_cache.py

The script is also run automatically by the "Generate klines cache" GitHub
Actions workflow (.github/workflows/cache-klines.yml).
"""

import asyncio
import io
import os
import sys
import zipfile
from calendar import monthrange
from datetime import datetime, timedelta, timezone
from typing import List

import aiohttp
import pandas as pd

# ---------------------------------------------------------------------------
# Make sure repo root is on sys.path so we can import asBack + src modules.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from asBack.backtest_grid_bitunix import fetch_klines_as_df  # noqa: E402

# ---------------------------------------------------------------------------
# Earliest date Bitunix has XRPUSDT data.  Anything before this is fetched
# from Binance USDT-M futures and stitched together.
# ---------------------------------------------------------------------------
_BITUNIX_START = datetime(2022, 4, 20, tzinfo=timezone.utc)

# Binance interval names differ from Bitunix (e.g. "15min" → "15m").
_BINANCE_INTERVAL_MAP = {
    "1min": "1m",
    "3min": "3m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}
# ---------------------------------------------------------------------------
# Datasets to cache.
# source="stitched"  — fetch pre-BITUNIX_START from Binance, rest from Bitunix.
# end_dt is evaluated at runtime so each generation always fetches up to today.
# ---------------------------------------------------------------------------
def _datasets():
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return [
        # Full ~6.5-year 15min history: Binance 2019-10-01 → 2022-04-20, then Bitunix.
        ("XRPUSDT", "15min", datetime(2019, 10, 1, tzinfo=timezone.utc), now, "stitched"),
        # Full ~6.5-year 1min history: Binance 2019-10-01 → 2022-04-20, then Bitunix.
        # Stored in Actions cache only (~60-80 MB parquet). Not committed to git.
        ("XRPUSDT", "1min",  datetime(2019, 10, 1, tzinfo=timezone.utc), now, "stitched"),
    ]


# ---------------------------------------------------------------------------
# Binance Data Vision bulk-download klines fetcher
# ---------------------------------------------------------------------------
# Uses https://data.binance.vision (public S3) instead of the REST API.
# No geographic restrictions (unlike fapi.binance.com which returns HTTP 451
# from GitHub Actions runners in AWS us-east).
#
# Downloads monthly ZIP/CSV files for complete months, plus daily ZIPs for
# the current/incomplete month.
#
# Monthly: https://data.binance.vision/data/spot/monthly/klines/{sym}/{iv}/{sym}-{iv}-{YYYY}-{MM}.zip
# Daily:   https://data.binance.vision/data/spot/daily/klines/{sym}/{iv}/{sym}-{iv}-{YYYY}-{MM}-{DD}.zip
# ---------------------------------------------------------------------------

_BINANCE_DV_BASE_MONTHLY = "https://data.binance.vision/data/spot/monthly/klines"
_BINANCE_DV_BASE_DAILY   = "https://data.binance.vision/data/spot/daily/klines"


def _parse_binance_dv_csv(csv_bytes: bytes) -> pd.DataFrame:
    """Parse a Binance Data Vision klines CSV (no header row)."""
    df = pd.read_csv(
        io.BytesIO(csv_bytes),
        header=None,
        usecols=[0, 1, 2, 3, 4, 5],
        names=["open_time", "open", "high", "low", "close", "volume"],
        dtype={
            "open_time": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        },
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


async def fetch_binance_klines_as_df(
    symbol: str,
    interval_bitunix: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """
    Fetch historical klines from **Binance Data Vision** bulk CSV downloads.

    Works from any location — no geographic restrictions.
    Downloads one ZIP per calendar month (monthly files for complete months,
    daily files for the current/incomplete month).

    Returns a DataFrame with the same columns as the Bitunix fetcher:
        open_time (datetime64[ns, UTC]), open, high, low, close, volume (float64)
    """
    binance_interval = _BINANCE_INTERVAL_MAP.get(interval_bitunix)
    if binance_interval is None:
        raise ValueError(
            f"No Binance interval mapping for '{interval_bitunix}'. "
            f"Known: {list(_BINANCE_INTERVAL_MAP)}"
        )

    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    print(
        f"  [Binance DataVision] Fetching {symbol} {interval_bitunix}  "
        f"{start_dt.date()} → {end_dt.date()} …"
    )

    all_frames: List[pd.DataFrame] = []

    async with aiohttp.ClientSession() as session:

        async def _get_zip(url: str) -> bytes | None:
            """Download a ZIP and return raw bytes, or None on 404."""
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as r:
                if r.status == 200:
                    return await r.read()
                if r.status == 404:
                    return None
                text = await r.text()
                raise RuntimeError(f"HTTP {r.status} fetching {url}: {text[:200]}")

        year, month = start_dt.year, start_dt.month
        end_year, end_month = end_dt.year, end_dt.month

        while (year, month) <= (end_year, end_month):
            last_day_of_month = monthrange(year, month)[1]
            month_end_dt = datetime(year, month, last_day_of_month, 23, 59, 59, tzinfo=timezone.utc)

            if month_end_dt < today:
                # Complete month in the past — use monthly file.
                url = (
                    f"{_BINANCE_DV_BASE_MONTHLY}/{symbol}/{binance_interval}/"
                    f"{symbol}-{binance_interval}-{year}-{month:02d}.zip"
                )
                raw = await _get_zip(url)
                if raw is not None:
                    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                        csv_bytes = zf.read(zf.namelist()[0])
                    df = _parse_binance_dv_csv(csv_bytes)
                    all_frames.append(df)
                    print(f"    {year}-{month:02d}: {len(df):,} candles (monthly)", flush=True)
                else:
                    print(f"    {year}-{month:02d}: not found — skipped", flush=True)
            else:
                # Current or future month — download available daily files.
                day_start = start_dt.day if (year == start_dt.year and month == start_dt.month) else 1
                yesterday = today - timedelta(days=1)
                day_end = (
                    yesterday.day
                    if (year == yesterday.year and month == yesterday.month)
                    else last_day_of_month
                )
                daily_count = 0
                for d in range(day_start, day_end + 1):
                    url = (
                        f"{_BINANCE_DV_BASE_DAILY}/{symbol}/{binance_interval}/"
                        f"{symbol}-{binance_interval}-{year}-{month:02d}-{d:02d}.zip"
                    )
                    raw = await _get_zip(url)
                    if raw is not None:
                        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                            csv_bytes = zf.read(zf.namelist()[0])
                        all_frames.append(_parse_binance_dv_csv(csv_bytes))
                        daily_count += 1
                if daily_count:
                    print(f"    {year}-{month:02d}: {daily_count} daily file(s) downloaded", flush=True)

            # Advance to next month.
            if month == 12:
                year, month = year + 1, 1
            else:
                month += 1

    if not all_frames:
        raise ValueError(
            f"No Binance data retrieved for {symbol} {interval_bitunix} "
            f"between {start_dt} and {end_dt}"
        )

    df = pd.concat(all_frames, ignore_index=True)
    df = (
        df[
            (df["open_time"] >= pd.Timestamp(start_dt))
            & (df["open_time"] < pd.Timestamp(end_dt))
        ]
        .drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    print(
        f"  → {len(df):,} Binance candles  "
        f"price: ${df['low'].min():.4f} – ${df['high'].max():.4f}"
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    cache_dir = os.path.join(_REPO_ROOT, "asBack", "klines_cache")
    os.makedirs(cache_dir, exist_ok=True)

    for symbol, interval, start_dt, end_dt, source in _datasets():
        out_path = os.path.join(cache_dir, f"{symbol}_{interval}.parquet")

        if os.path.exists(out_path):
            print(f"[skip] {out_path} already exists — delete it to force re-fetch.")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset : {symbol} {interval}  {start_dt.date()} → {end_dt.date()}")
        print(f"Source  : {source}")
        print(f"Output  : {out_path}")
        print(f"{'='*60}")

        if source == "stitched":
            # ------------------------------------------------------------------
            # Stitched: Binance for history before Bitunix listing date,
            # Bitunix for the remainder.
            # ------------------------------------------------------------------
            start_utc = start_dt if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc)
            end_utc   = end_dt   if end_dt.tzinfo   else end_dt.replace(tzinfo=timezone.utc)
            parts = []

            if start_utc < _BITUNIX_START:
                print(f"\n--- Part 1/2: Binance  {start_utc.date()} → {_BITUNIX_START.date()} ---")
                df_binance = await fetch_binance_klines_as_df(
                    symbol=symbol,
                    interval_bitunix=interval,
                    start_dt=start_utc,
                    end_dt=_BITUNIX_START,
                )
                parts.append(df_binance)

            bitunix_start = max(start_utc, _BITUNIX_START)
            print(f"\n--- Part 2/2: Bitunix  {bitunix_start.date()} → {end_utc.date()} ---")
            df_bitunix = await fetch_klines_as_df(
                symbol=symbol,
                interval=interval,
                start_dt=bitunix_start,
                end_dt=end_utc,
            )
            parts.append(df_bitunix)

            df = (
                pd.concat(parts, ignore_index=True)
                .drop_duplicates(subset=["open_time"])
                .sort_values("open_time")
                .reset_index(drop=True)
            )
            print(f"\nStitched total: {len(df):,} candles")
            print(f"  Range: {df['open_time'].iloc[0].date()} → {df['open_time'].iloc[-1].date()}")

        else:
            # ------------------------------------------------------------------
            # Bitunix-only
            # ------------------------------------------------------------------
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

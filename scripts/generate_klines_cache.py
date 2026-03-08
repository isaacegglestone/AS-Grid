"""
scripts/generate_klines_cache.py  (v3 — 3-source stitched cache 2026-03-08)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pre-fetches klines and saves them as parquet files in ``asBack/klines_cache/``.
Once generated, the backtest engine loads from disk instead of hitting the API.

Data sources (3-way stitch)
---------------------------
- **Bitfinex public API** — tXRPUSD available from ~2017-05-19 onward.
  Covers the 2017 mega-bubble ($0.006 → $3.84) and 2018 crash.
  Up to 10,000 candles per request, no API key required.
- **Binance Data Vision** (data.binance.vision) — XRPUSDT spot data from
  ~2018-05-01 onward.  Downloads monthly ZIP/CSV files with no API key and
  no geographic restrictions.
- **Bitunix public API** — XRPUSDT available from ~2022-04-20 onward.

Stitched timeline (~8.8 years):
  Bitfinex  2017-05-19 → 2018-05-01  (XRP/USD → converted to align with USDT)
  Binance   2018-05-01 → 2022-04-20  (XRPUSDT spot via Data Vision)
  Bitunix   2022-04-20 → now          (live-exchange data used by the bot)

The 1min stitched dataset is ~350–450 MB uncompressed (~100–120 MB as parquet).
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
import shutil
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
# Source boundary dates.  Data before each date comes from the previous source.
# ---------------------------------------------------------------------------
_BITFINEX_START = datetime(2017, 5, 19, tzinfo=timezone.utc)   # earliest tXRPUSD 1m candle
_BINANCE_START  = datetime(2018, 5, 1,  tzinfo=timezone.utc)   # earliest XRPUSDT on Data Vision
_BITUNIX_START  = datetime(2022, 4, 20, tzinfo=timezone.utc)   # earliest XRPUSDT on Bitunix

# Bitfinex interval names
_BITFINEX_INTERVAL_MAP = {
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1hour": "1h",
    "4hour": "4h",
    "1day":  "1D",
}

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
# source="stitched"  — 3-way stitch: Bitfinex → Binance → Bitunix.
# end_dt is evaluated at runtime so each generation always fetches up to today.
# ---------------------------------------------------------------------------
def _datasets():
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return [
        # Full ~8.8-year 15min history: Bitfinex → Binance → Bitunix.
        ("XRPUSDT", "15min", _BITFINEX_START, now, "stitched"),
        # Full ~8.8-year 1min history: Bitfinex → Binance → Bitunix.
        # Stored in Actions cache only (~100-120 MB parquet). Not committed to git.
        ("XRPUSDT", "1min",  _BITFINEX_START, now, "stitched"),
    ]


# ---------------------------------------------------------------------------
# Bitfinex public API klines fetcher
# ---------------------------------------------------------------------------
# Uses https://api-pub.bitfinex.com/v2/candles/ (public, no auth).
# Covers the 2017 mega-bubble era when Binance didn't yet have XRPUSDT.
# Bitfinex pair is tXRPUSD (USD, not USDT) — prices are close enough to
# USDT that stitching is seamless (XRP/USD ≈ XRP/USDT within ~0.1%).
#
# API returns up to 10,000 candles per request.
# Rate limit: 30 requests/minute (we add a 2s sleep between requests).
# ---------------------------------------------------------------------------

_BITFINEX_API_BASE = "https://api-pub.bitfinex.com/v2/candles"

# Bitfinex interval string → milliseconds per candle
_BITFINEX_INTERVAL_MS = {
    "1m":  60_000,
    "5m":  5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h":  60 * 60_000,
    "4h":  4 * 60 * 60_000,
    "1D":  24 * 60 * 60_000,
}


async def fetch_bitfinex_klines_as_df(
    symbol: str,
    interval_bitunix: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """
    Fetch historical 1-minute klines from Bitfinex public API.

    The Bitfinex pair is tXRPUSD (USD fiat pair).  Data is returned with
    columns matching the rest of the pipeline:
        open_time (datetime64[ns, UTC]), open, high, low, close, volume (float64)

    Pagination: API returns up to 10,000 candles per request.  We page
    forward using the ``start`` parameter, advancing by the window size
    each iteration.
    """
    bfx_interval = _BITFINEX_INTERVAL_MAP.get(interval_bitunix)
    if bfx_interval is None:
        raise ValueError(
            f"No Bitfinex interval mapping for '{interval_bitunix}'. "
            f"Known: {list(_BITFINEX_INTERVAL_MAP)}"
        )

    interval_ms = _BITFINEX_INTERVAL_MS[bfx_interval]
    CHUNK_SIZE = 10_000  # Bitfinex max per request
    WINDOW_MS = CHUNK_SIZE * interval_ms

    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # Bitfinex symbol mapping: XRPUSDT → tXRPUSD
    bfx_symbol = "tXRPUSD"  # Bitfinex has XRP/USD, not XRP/USDT

    print(
        f"  [Bitfinex] Fetching {bfx_symbol} {interval_bitunix}  "
        f"{start_dt.date()} → {end_dt.date()} …"
    )

    all_frames: List[pd.DataFrame] = []
    cursor = start_ms
    req_num = 0

    async with aiohttp.ClientSession() as session:
        while cursor < end_ms:
            chunk_end = min(cursor + WINDOW_MS, end_ms)
            url = (
                f"{_BITFINEX_API_BASE}/trade:{bfx_interval}:{bfx_symbol}/hist"
                f"?start={cursor}&end={chunk_end}&limit={CHUNK_SIZE}&sort=1"
            )

            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status == 429:
                    # Rate-limited — back off and retry
                    print("    ⏳ Rate-limited, waiting 60s …")
                    await asyncio.sleep(60)
                    continue
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"Bitfinex HTTP {resp.status}: {text[:200]}"
                    )
                data = await resp.json()

            req_num += 1
            if not data:
                # No data in this window — might be a gap, advance anyway
                cursor = chunk_end
                await asyncio.sleep(0.5)
                continue

            # Bitfinex candle format: [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]
            # Note: column order is MTS, O, C, H, L, V  (not OHLCV!)
            df_chunk = pd.DataFrame(data, columns=["open_time", "open", "close", "high", "low", "volume"])
            # Reorder to standard OHLCV
            df_chunk = df_chunk[["open_time", "open", "high", "low", "close", "volume"]]
            all_frames.append(df_chunk)

            if req_num % 5 == 0 or req_num == 1:
                total_candles = sum(len(f) for f in all_frames)
                pct = min(100, int((cursor - start_ms) / max(1, end_ms - start_ms) * 100))
                print(f"    req {req_num}  {total_candles:,} candles  ~{pct}%", flush=True)

            # Advance cursor past the last candle we received
            last_ts = max(row[0] for row in data)
            cursor = last_ts + interval_ms

            # Respect rate limit: 30 req/min → 2s between requests
            await asyncio.sleep(2.1)

    if not all_frames:
        raise ValueError(
            f"No Bitfinex data retrieved for {bfx_symbol} {interval_bitunix} "
            f"between {start_dt} and {end_dt}"
        )

    df = pd.concat(all_frames, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype("float64")
    df = (
        df[(df["open_time"] >= pd.Timestamp(start_dt))
           & (df["open_time"] < pd.Timestamp(end_dt))]
        .drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    print(
        f"  → {len(df):,} Bitfinex candles  "
        f"price: ${df['low'].min():.4f} – ${df['high'].max():.4f}"
    )
    return df


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

    # Local dev mirror directory — lives outside the repo so large files aren't committed.
    # The backtest engine checks this path as a fallback when the repo-local cache is absent.
    _local_dev_dir = os.path.expanduser(os.path.join("~", "git", "data", "asgrid-klines"))

    for symbol, interval, start_dt, end_dt, source in _datasets():
        out_path = os.path.join(cache_dir, f"{symbol}_{interval}.parquet")
        _local_dev_path = os.path.join(_local_dev_dir, f"{symbol}_{interval}.parquet")

        if os.path.exists(out_path) or os.path.exists(_local_dev_path):
            print(f"[skip] {symbol} {interval} already cached — delete both copies to force re-fetch.")
            print(f"         repo:  {out_path}")
            print(f"         local: {_local_dev_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset : {symbol} {interval}  {start_dt.date()} → {end_dt.date()}")
        print(f"Source  : {source}")
        print(f"Output  : {out_path}")
        print(f"{'='*60}")

        if source == "stitched":
            # ------------------------------------------------------------------
            # 3-way stitch: Bitfinex → Binance → Bitunix.
            # Each source covers a non-overlapping date window.
            # ------------------------------------------------------------------
            start_utc = start_dt if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc)
            end_utc   = end_dt   if end_dt.tzinfo   else end_dt.replace(tzinfo=timezone.utc)
            parts = []
            part_num = 0
            total_parts = (
                (1 if start_utc < _BINANCE_START else 0)
                + (1 if start_utc < _BITUNIX_START else 0)
                + 1  # Bitunix (always)
            )

            # Part A: Bitfinex (pre-Binance era)
            if start_utc < _BINANCE_START:
                part_num += 1
                bfx_end = min(_BINANCE_START, end_utc)
                print(f"\n--- Part {part_num}/{total_parts}: Bitfinex  {start_utc.date()} → {bfx_end.date()} ---")
                df_bitfinex = await fetch_bitfinex_klines_as_df(
                    symbol=symbol,
                    interval_bitunix=interval,
                    start_dt=start_utc,
                    end_dt=bfx_end,
                )
                parts.append(df_bitfinex)

            # Part B: Binance (pre-Bitunix era)
            binance_start = max(start_utc, _BINANCE_START)
            if binance_start < _BITUNIX_START and binance_start < end_utc:
                part_num += 1
                binance_end = min(_BITUNIX_START, end_utc)
                print(f"\n--- Part {part_num}/{total_parts}: Binance  {binance_start.date()} → {binance_end.date()} ---")
                df_binance = await fetch_binance_klines_as_df(
                    symbol=symbol,
                    interval_bitunix=interval,
                    start_dt=binance_start,
                    end_dt=binance_end,
                )
                parts.append(df_binance)

            # Part C: Bitunix (current exchange)
            bitunix_start = max(start_utc, _BITUNIX_START)
            if bitunix_start < end_utc:
                part_num += 1
                print(f"\n--- Part {part_num}/{total_parts}: Bitunix  {bitunix_start.date()} → {end_utc.date()} ---")
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
            print(f"\nStitched total: {len(df):,} candles  ({len(parts)} sources)")
            print(f"  Range: {df['open_time'].iloc[0].date()} → {df['open_time'].iloc[-1].date()}")

            # Validate stitch boundaries — warn if there are gaps > 5 minutes
            time_diffs = df["open_time"].diff()
            _interval_td = pd.Timedelta(milliseconds=_BITFINEX_INTERVAL_MS.get(
                _BITFINEX_INTERVAL_MAP.get(interval, "1m"), 60_000))
            max_gap = time_diffs.max()
            if max_gap > _interval_td * 5:
                gap_idx = time_diffs.idxmax()
                gap_time = df["open_time"].iloc[gap_idx]
                print(f"  ⚠️  Largest gap: {max_gap} at {gap_time} "
                      f"(>{_interval_td * 5} threshold)")
            else:
                print(f"  ✅ No significant gaps detected (max: {max_gap})")

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

        # Mirror to the local dev directory (skipped in CI where the dir won't exist).
        os.makedirs(_local_dev_dir, exist_ok=True)
        shutil.copy2(out_path, _local_dev_path)
        print(f"  → Mirrored  →  {_local_dev_path}")

    print("\nAll datasets cached.")


if __name__ == "__main__":
    asyncio.run(main())

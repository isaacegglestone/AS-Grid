"""
scripts/generate_klines_cache.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pre-fetches klines and saves them as parquet files in ``asBack/klines_cache/``.
Once generated, the backtest engine loads from disk instead of hitting the API.

Data sources
------------
- **Bitunix public API** — XRPUSDT available from ~2022-04-20 onward.
- **Binance USDT-M futures public API** — XRPUSDT available from ~2019-10-01,
  giving a full ~6.5-year history including the 2021 bull run, 2022 crash
  (LUNA/FTX) and the 2023-2025 recovery cycle.

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
import os
import sys
from datetime import datetime, timezone
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
_BINANCE_INTERVAL_MS = {
    "1m":  60_000,
    "3m":  180_000,
    "5m":  300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
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
# Binance USDT-M futures klines fetcher
# ---------------------------------------------------------------------------

async def fetch_binance_klines_as_df(
    symbol: str,
    interval_bitunix: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """
    Fetch klines for *symbol* from the **Binance USDT-M futures** public API.

    Returns a DataFrame with the same columns as Bitunix:
        open_time (datetime64[ns, UTC]), open, high, low, close, volume (float64)

    *interval_bitunix* uses the Bitunix naming convention (e.g. ``"15min"``);
    it is mapped internally to the Binance format (``"15m"``).

    Binance limit: 1,500 candles per request.  No API key required.
    """
    binance_interval = _BINANCE_INTERVAL_MAP.get(interval_bitunix)
    if binance_interval is None:
        raise ValueError(
            f"No Binance interval mapping for '{interval_bitunix}'. "
            f"Known: {list(_BINANCE_INTERVAL_MAP)}"
        )
    interval_ms = _BINANCE_INTERVAL_MS[binance_interval]

    def _to_ms(dt: datetime) -> int:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    CHUNK = 1500
    start_ms = _to_ms(start_dt)
    end_ms = _to_ms(end_dt)
    total_ms = end_ms - start_ms
    estimated_chunks = max(1, total_ms // (CHUNK * interval_ms))

    print(f"  [Binance] Fetching {symbol} {interval_bitunix}  "
          f"{start_dt.date()} → {end_dt.date()} …")
    print(f"  (estimated ~{estimated_chunks} API requests)")

    rows: List[dict] = []
    cursor = start_ms
    chunk_num = 0

    async with aiohttp.ClientSession() as session:
        while cursor < end_ms:
            url = (
                "https://fapi.binance.com/fapi/v1/klines"
                f"?symbol={symbol}&interval={binance_interval}"
                f"&startTime={cursor}&endTime={end_ms}&limit={CHUNK}"
            )
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"Binance API error {resp.status}: {text[:200]}"
                    )
                data = await resp.json()

            if not data:
                break

            chunk_num += 1
            if chunk_num % 20 == 0 or chunk_num == 1:
                pct = min(100, int(len(rows) / max(1, estimated_chunks * CHUNK) * 100))
                print(f"    chunk {chunk_num}/{estimated_chunks}  ~{pct}% …", flush=True)

            for row in data:
                open_time_ms: int = row[0]
                if open_time_ms < end_ms:
                    rows.append({
                        "open_time": open_time_ms,
                        "open":   float(row[1]),
                        "high":   float(row[2]),
                        "low":    float(row[3]),
                        "close":  float(row[4]),
                        "volume": float(row[5]),
                    })

            last_open_ms: int = data[-1][0]
            cursor = last_open_ms + interval_ms
            await asyncio.sleep(0.05)   # gentle rate limiting

    if not rows:
        raise ValueError(
            f"No Binance klines returned for {symbol} {interval_bitunix} "
            f"between {start_dt} and {end_dt}"
        )

    df = pd.DataFrame(rows)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = (
        df.drop_duplicates(subset=["open_time"])
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

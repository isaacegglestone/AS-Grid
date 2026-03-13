"""
scripts/generate_funding_cache.py  — Funding Rate Cache Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fetches historical funding rates from Binance Futures API and stores them
as parquet files in ``asBack/klines_cache/``.

Binance Futures funding rates:
  - Paid every 8 hours (00:00, 08:00, 16:00 UTC)
  - Positive rate = longs pay shorts
  - Negative rate = shorts pay longs
  - Available from contract launch:
      BTCUSDT:  2019-09-10
      XRPUSDT:  2020-03-19

API endpoint (public, no key required):
  GET https://fapi.binance.com/fapi/v1/fundingRate
  Params: symbol, startTime (ms), endTime (ms), limit (max 1000)

Output files:
  asBack/klines_cache/XRPUSDT_funding.parquet
  asBack/klines_cache/BTCUSDT_funding.parquet

Columns: timestamp (UTC datetime), rate (float, e.g. 0.0001 = 0.01%)

Usage:
    python scripts/generate_funding_cache.py           # all symbols
    python scripts/generate_funding_cache.py BTCUSDT   # single symbol
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional

import aiohttp
import pandas as pd

# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(_REPO_ROOT, "asBack", "klines_cache")

_BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
_LIMIT = 1000  # max per request

# Earliest available funding rate data per symbol on Binance Futures.
_SYMBOL_START = {
    "BTCUSDT": datetime(2019, 9, 10, tzinfo=timezone.utc),
    "XRPUSDT": datetime(2020, 3, 19, tzinfo=timezone.utc),
}


async def _fetch_funding_rates(
    session: aiohttp.ClientSession,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> List[dict]:
    """Fetch up to 1000 funding rate records from Binance."""
    params = {
        "symbol": symbol,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": _LIMIT,
    }
    for attempt in range(5):
        try:
            async with session.get(_BINANCE_FUNDING_URL, params=params) as resp:
                if resp.status == 429:
                    wait = 2 ** attempt
                    print(f"  Rate limited, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            wait = 2 ** attempt
            print(f"  Request error ({e}), retrying in {wait}s...")
            await asyncio.sleep(wait)
    raise RuntimeError(f"Failed to fetch funding rates for {symbol} after 5 retries")


async def fetch_all_funding(symbol: str) -> pd.DataFrame:
    """Fetch complete funding rate history for a symbol."""
    start_dt = _SYMBOL_START.get(symbol)
    if start_dt is None:
        print(f"  Skipping {symbol} — no start date defined")
        return pd.DataFrame()

    now = datetime.now(timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    all_records = []
    cursor_ms = start_ms

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while cursor_ms < end_ms:
            batch = await _fetch_funding_rates(session, symbol, cursor_ms, end_ms)
            if not batch:
                break
            all_records.extend(batch)
            # Move cursor past last record
            last_ts = batch[-1]["fundingTime"]
            cursor_ms = last_ts + 1
            print(f"  {symbol}: fetched {len(all_records)} records "
                  f"(up to {datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).date()})")
            # Small delay to respect rate limits
            await asyncio.sleep(0.2)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["rate"] = df["fundingRate"].astype(float)
    df = df[["timestamp", "rate"]].sort_values("timestamp").drop_duplicates("timestamp")
    df = df.reset_index(drop=True)
    return df


async def main(symbols: Optional[List[str]] = None):
    if symbols is None:
        symbols = list(_SYMBOL_START.keys())

    os.makedirs(CACHE_DIR, exist_ok=True)

    for sym in symbols:
        print(f"\n{'='*60}")
        print(f"  Fetching funding rates: {sym}")
        print(f"{'='*60}")
        df = await fetch_all_funding(sym)
        if df.empty:
            print(f"  No data for {sym}")
            continue

        out_path = os.path.join(CACHE_DIR, f"{sym}_funding.parquet")
        df.to_parquet(out_path, index=False)
        print(f"  Saved {len(df)} records → {out_path}")
        print(f"  Date range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
        print(f"  Mean rate: {df['rate'].mean():.6f} ({df['rate'].mean()*100:.4f}%)")
        print(f"  Median rate: {df['rate'].median():.6f}")


if __name__ == "__main__":
    syms = [s.upper() for s in sys.argv[1:]] if len(sys.argv) > 1 else None
    asyncio.run(main(syms))

"""
scripts/diagnose_pm.py
~~~~~~~~~~~~~~~~~~~~~~~
Counts how many candles each PM condition actually fires during the
6m (Aug 2025 → Feb 2026) and 2y (Feb 2024 → Feb 2026) windows.

Run with:
    python scripts/diagnose_pm.py
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd

from asBack.backtest_grid_bitunix import fetch_klines_as_df


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _bb_width(close: pd.Series, period: int = 20, mult: float = 2.0) -> pd.Series:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + mult * std
    lower = mid - mult * std
    return (upper - lower) / mid.replace(0, np.nan)


def _vol_avg(volume: pd.Series, period: int = 20) -> pd.Series:
    return volume.rolling(period).mean()


async def diagnose(label: str, start: datetime, end: datetime) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}  ({start.date()} → {end.date()})")
    print(f"{'='*60}")

    df = await fetch_klines_as_df(
        symbol="XRPUSDT",
        interval="15min",
        start_dt=start,
        end_dt=end,
    )

    close = df["close"]
    volume = df["volume"]

    rsi_series = _rsi(close, 14)
    bb_w_series = _bb_width(close, 20, 2.0)
    vol_avg_series = _vol_avg(volume, 20)

    # BB squeeze threshold from _pm_set
    BB_THRESHOLD = 0.02
    RSI_OB = 80.0
    RSI_OS = 20.0
    VOL_HIGH_MULT = 1.5
    VELOCITY_THRESH = 0.015   # TREND_VELOCITY_PCT in re_reentry
    LOOKBACK = 10             # trend_lookback_candles

    # ── 1. RSI tight trail ────────────────────────────────────────────────
    # Fires when trend position is open AND RSI ≥ 80 (long) or ≤ 20 (short)
    # Proxy: count candles where RSI ≥ 80 or ≤ 20 (would fire if position were open)
    rsi_ob_candles = (rsi_series >= RSI_OB).sum()
    rsi_os_candles = (rsi_series <= RSI_OS).sum()
    print(f"\nRSI tight trail conditions:")
    print(f"  RSI ≥ {RSI_OB} (overbought — long exit tighter): {rsi_ob_candles:,} candles "
          f"({100*rsi_ob_candles/len(df):.1f}%)")
    print(f"  RSI ≤ {RSI_OS} (oversold  — short exit tighter): {rsi_os_candles:,} candles "
          f"({100*rsi_os_candles/len(df):.1f}%)")
    print(f"  ⚠️  But this only fires while a trend POSITION IS OPEN.")
    print(f"  If trend positions are typically short-lived (<10 candles),")
    print(f"  the probability of RSI being ≥80 simultaneously is low.")

    # Simulate: approximate trend open windows using velocity
    price_arr = close.values
    trend_open_candles = 0
    rsi_arr = rsi_series.values
    i = LOOKBACK
    in_trend_long = False
    in_trend_short = False
    peak_long = 0.0
    peak_short = 0.0
    TRAIL = 0.04

    rsi_trail_fires = 0

    while i < len(price_arr):
        price = price_arr[i]
        past_price = price_arr[i - LOOKBACK]
        vel = (price - past_price) / past_price

        if not in_trend_long and not in_trend_short:
            if vel > VELOCITY_THRESH:
                in_trend_long = True
                peak_long = price
            elif vel < -VELOCITY_THRESH:
                in_trend_short = True
                peak_short = price
        elif in_trend_long:
            if price > peak_long:
                peak_long = price
            trail_stop = peak_long * (1 - TRAIL)
            trend_open_candles += 1
            rsi_now = rsi_arr[i]
            if rsi_now >= RSI_OB:
                rsi_trail_fires += 1
            if price <= trail_stop or vel < -VELOCITY_THRESH:
                in_trend_long = False
        elif in_trend_short:
            if price < peak_short:
                peak_short = price
            trail_stop = peak_short * (1 + TRAIL)
            trend_open_candles += 1
            rsi_now = rsi_arr[i]
            if rsi_now <= RSI_OS:
                rsi_trail_fires += 1
            if price >= trail_stop or vel > VELOCITY_THRESH:
                in_trend_short = False
        i += 1

    print(f"  Simulated trend-open candles: {trend_open_candles:,}")
    print(f"  RSI trail would fire: {rsi_trail_fires:,} times "
          f"({'NEVER' if rsi_trail_fires == 0 else 'YES — ' + str(rsi_trail_fires) + ' times'})")

    # ── 2. Volume-scaled re-entry ─────────────────────────────────────────
    vol_arr = volume.values
    vol_avg_arr = vol_avg_series.values
    # Count candles where volume < 1.5× avg (would reduce size to 0.45)
    low_vol_entries = 0
    high_vol_entries = 0
    for j in range(LOOKBACK, len(price_arr)):
        p = price_arr[j]
        pp = price_arr[j - LOOKBACK]
        v_now = vol_arr[j]
        v_avg = vol_avg_arr[j]
        vel_j = (p - pp) / pp
        is_entry = (abs(vel_j) >= VELOCITY_THRESH) and not np.isnan(v_avg) and v_avg > 0
        if is_entry:
            if v_now >= VOL_HIGH_MULT * v_avg:
                high_vol_entries += 1
            else:
                low_vol_entries += 1

    total_entries = high_vol_entries + low_vol_entries
    print(f"\nVolume-scaled re-entry conditions:")
    print(f"  Approximate trend entries: {total_entries:,}")
    print(f"  High-vol (full 90% size): {high_vol_entries:,} "
          f"({100*high_vol_entries/max(1,total_entries):.0f}%)")
    print(f"  Low-vol  (45% size):       {low_vol_entries:,} "
          f"({100*low_vol_entries/max(1,total_entries):.0f}%)")
    if low_vol_entries == 0:
        print(f"  ⚠️  ALL entries happen on high-volume candles — scale never reduces size!")
    else:
        print(f"  ✅ Would reduce size on {low_vol_entries} entries")

    # ── 3. BB squeeze boost ───────────────────────────────────────────────
    bb_w = bb_w_series.values
    just_broke_count = 0
    for j in range(1, len(bb_w)):
        if np.isnan(bb_w[j]) or np.isnan(bb_w[j-1]):
            continue
        prev = bb_w[j-1]
        curr = bb_w[j]
        expanding = curr > prev
        was_in_squeeze = prev < BB_THRESHOLD
        just_broke = was_in_squeeze and expanding
        if just_broke:
            just_broke_count += 1

    pct_squeeze = 100 * (bb_w < BB_THRESHOLD).sum() / len(df)
    print(f"\nBB squeeze boost conditions (threshold={BB_THRESHOLD}):")
    print(f"  Candles in BB squeeze (width < {BB_THRESHOLD}): "
          f"{(bb_w < BB_THRESHOLD).sum():,} ({pct_squeeze:.1f}%)")
    print(f"  'just_broke' (squeeze→expansion) fires: {just_broke_count:,} times")
    if just_broke_count == 0:
        print(f"  ⚠️  BB threshold {BB_THRESHOLD} is too tight — no squeezes detected!")
        # Show what threshold would have squeezes
        for t in [0.04, 0.06, 0.08, 0.10]:
            count = (bb_w < t).sum()
            if count > 0:
                print(f"     At threshold={t}: {count:,} squeeze candles — BB boost WOULD fire")
                break
    else:
        print(f"  ✅ Would boost size on {just_broke_count} entry candles")

    print()


async def main() -> None:
    await diagnose(
        "6m OOS (Aug 2025 → Feb 2026)",
        datetime(2025, 8, 1, tzinfo=timezone.utc),
        datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    await diagnose(
        "2y walk-forward (Feb 2024 → Feb 2026)",
        datetime(2024, 2, 28, tzinfo=timezone.utc),
        datetime(2026, 2, 28, tzinfo=timezone.utc),
    )


if __name__ == "__main__":
    asyncio.run(main())

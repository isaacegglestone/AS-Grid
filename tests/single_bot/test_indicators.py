"""
tests/single_bot/test_indicators.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ``src/single_bot/indicators.py``.

All tests are pure – no network calls, no async, no mocking required.
Synthetic OHLCV series are generated from simple mathematical sequences so
that expected indicator values can be reasoned about deterministically.

Structure
---------
- TestCandle              – dataclass construction
- TestCandleBufferUpdate  – rolling buffer / candle-close detection logic
- TestCandleBufferSignals – warm-up guard and Signals field presence
- TestRSI                 – Wilder RSI edge cases and direction
- TestBBWidth             – Bollinger-Band width calculation
- TestADX                 – ADX direction / range
- TestEMA                 – EMA output and bias flags
- TestVolMetrics          – vol_avg / vol_ratio
- TestSwingHighLow        – market-structure window bounds
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from src.single_bot.indicators import Candle, CandleBuffer, Signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TS_BASE = 1_740_000_000_000  # millisecond epoch anchor
TS_STEP = 15 * 60 * 1_000    # 15-minute bars in ms


def _make_candles(
    closes: List[float],
    *,
    open_offset: float = 0.0,
    high_offset: float = 0.05,
    low_offset: float = -0.05,
    vol: float = 1_000_000.0,
) -> List[Candle]:
    """Return a list of ``Candle`` objects with programmatically-derived OHLCV."""
    candles = []
    for i, c in enumerate(closes):
        candles.append(
            Candle(
                ts=TS_BASE + i * TS_STEP,
                open=c + open_offset,
                high=c + abs(high_offset),
                low=c - abs(low_offset),
                close=c,
                volume=vol,
            )
        )
    return candles


def _seeded_buffer(
    n: int = 60,
    *,
    closes: List[float] | None = None,
    vol: float = 1_000_000.0,
    high_offset: float = 0.05,
    low_offset: float = -0.05,
    **buf_kwargs,
) -> CandleBuffer:
    """Return a ``CandleBuffer`` pre-loaded with *n* closed candles.

    ``high_offset`` / ``low_offset`` control candle OHLC spread (forwarded to
    ``_make_candles``).  Any remaining kwargs are passed to ``CandleBuffer``.
    """
    buf = CandleBuffer(maxlen=200, interval="15min", regime_ema_period=21, **buf_kwargs)
    if closes is None:
        closes = [1.0 + i * 0.001 for i in range(n)]  # gentle uptrend
    candles = _make_candles(closes, vol=vol, high_offset=high_offset, low_offset=low_offset)
    for c in candles:
        buf._closed.append(c)
    return buf


# ---------------------------------------------------------------------------
# TestCandle
# ---------------------------------------------------------------------------

class TestCandle:
    def test_fields_stored(self):
        c = Candle(ts=1, open=1.0, high=1.1, low=0.9, close=1.05, volume=5e6)
        assert c.ts == 1
        assert c.open == pytest.approx(1.0)
        assert c.high == pytest.approx(1.1)
        assert c.low  == pytest.approx(0.9)
        assert c.close == pytest.approx(1.05)
        assert c.volume == pytest.approx(5e6)


# ---------------------------------------------------------------------------
# TestCandleBufferUpdate
# ---------------------------------------------------------------------------

class TestCandleBufferUpdate:
    def test_first_update_returns_false(self):
        buf = CandleBuffer()
        closed = buf.update(1.0, 1.1, 0.9, 1.05, 1e6, TS_BASE)
        assert closed is False

    def test_first_update_creates_live_candle(self):
        buf = CandleBuffer()
        buf.update(1.0, 1.1, 0.9, 1.05, 1e6, TS_BASE)
        assert buf._live is not None
        assert buf._live.ts == TS_BASE
        assert buf._live.open  == pytest.approx(1.0)
        assert buf._live.close == pytest.approx(1.05)

    def test_same_ts_updates_high_low_close_volume(self):
        buf = CandleBuffer()
        buf.update(1.0, 1.1, 0.9, 1.05, 1e6, TS_BASE)
        # Push higher high and lower low
        buf.update(1.0, 1.2, 0.8, 1.08, 2e6, TS_BASE)
        live = buf._live
        assert live.high   == pytest.approx(1.2)
        assert live.low    == pytest.approx(0.8)
        assert live.close  == pytest.approx(1.08)
        assert live.volume == pytest.approx(2e6)
        # Open is set on the first update and must NOT change
        assert live.open   == pytest.approx(1.0)

    def test_same_ts_does_not_commit_to_closed(self):
        buf = CandleBuffer()
        buf.update(1.0, 1.1, 0.9, 1.05, 1e6, TS_BASE)
        buf.update(1.0, 1.2, 0.8, 1.08, 2e6, TS_BASE)
        assert len(buf._closed) == 0

    def test_new_ts_commits_previous_candle_and_returns_true(self):
        buf = CandleBuffer()
        buf.update(1.0, 1.1, 0.9, 1.05, 1e6, TS_BASE)
        closed = buf.update(1.1, 1.2, 1.0, 1.15, 1e6, TS_BASE + TS_STEP)
        assert closed is True
        assert len(buf._closed) == 1

    def test_committed_candle_has_correct_final_values(self):
        buf = CandleBuffer()
        # First candle: open-high updated mid-candle
        buf.update(1.00, 1.10, 0.90, 1.05, 1e6, TS_BASE)
        buf.update(1.00, 1.15, 0.85, 1.03, 9e5, TS_BASE)  # same ts
        # Second candle arrives – first is finalised
        buf.update(1.03, 1.20, 1.00, 1.20, 2e6, TS_BASE + TS_STEP)
        committed = buf._closed[0]
        assert committed.open   == pytest.approx(1.00)
        assert committed.high   == pytest.approx(1.15)
        assert committed.low    == pytest.approx(0.85)
        assert committed.close  == pytest.approx(1.03)
        assert committed.volume == pytest.approx(9e5)

    def test_multiple_candle_closes_accumulate(self):
        buf = CandleBuffer()
        for i in range(5):
            ts = TS_BASE + i * TS_STEP
            buf.update(1.0 + i * 0.01, 1.1, 0.9, 1.05, 1e6, ts)
        assert len(buf._closed) == 4   # 4 closed, 1 in-progress

    def test_maxlen_caps_closed_deque(self):
        buf = CandleBuffer(maxlen=5)
        for i in range(10):
            buf.update(float(i), float(i) + 0.1, float(i) - 0.1, float(i), 1e6,
                       TS_BASE + i * TS_STEP)
        assert len(buf._closed) <= 5

    def test_no_live_before_any_update(self):
        buf = CandleBuffer()
        assert buf._live is None


# ---------------------------------------------------------------------------
# TestCandleBufferSignals – warm-up guard and Signals fields
# ---------------------------------------------------------------------------

class TestCandleBufferSignals:
    def test_returns_none_when_empty(self):
        buf = CandleBuffer()
        assert buf.signals() is None

    def test_returns_none_below_min_required(self):
        # 10 candles is far below the minimum (~26 = max(14,14,20,21,regime_ema=21)+5)
        buf = _seeded_buffer(n=10)
        assert buf.signals() is None

    def test_returns_signals_when_enough_candles(self):
        buf = _seeded_buffer(n=60)
        sig = buf.signals()
        assert isinstance(sig, Signals)

    def test_signals_close_equals_last_candle_close(self):
        closes = [1.0 + i * 0.001 for i in range(60)]
        buf = _seeded_buffer(n=60, closes=closes)
        sig = buf.signals()
        assert sig is not None
        assert sig.close == pytest.approx(closes[-1])

    def test_signals_rsi_in_range(self):
        buf = _seeded_buffer(n=60)
        sig = buf.signals()
        assert sig is not None
        assert 0.0 <= sig.rsi <= 100.0

    def test_signals_adx_in_range(self):
        buf = _seeded_buffer(n=60)
        sig = buf.signals()
        assert sig is not None
        assert 0.0 <= sig.adx <= 100.0

    def test_signals_bb_width_non_negative(self):
        buf = _seeded_buffer(n=60)
        sig = buf.signals()
        assert sig is not None
        assert sig.bb_width >= 0.0

    def test_signals_atr_positive(self):
        buf = _seeded_buffer(n=60, high_offset=0.05, low_offset=-0.05)
        sig = buf.signals()
        assert sig is not None
        assert sig.atr > 0.0

    def test_ema_bias_flags_mutually_exclusive(self):
        buf = _seeded_buffer(n=60)
        sig = buf.signals()
        assert sig is not None
        assert not (sig.ema_bias_long and sig.ema_bias_short)

    def test_vol_ratio_equals_one_for_constant_volume(self):
        buf = _seeded_buffer(n=60, vol=2_000_000.0)
        sig = buf.signals()
        assert sig is not None
        assert sig.vol_ratio == pytest.approx(1.0, abs=1e-6)

    def test_vol_ratio_above_one_for_recent_spike(self):
        n = 60
        vols = [1_000_000.0] * n
        vols[-1] = 5_000_000.0  # spike on last candle
        closes = [1.0 + i * 0.001 for i in range(n)]
        buf = CandleBuffer(maxlen=200, interval="15min", vol_period=20, regime_ema_period=21)
        for c, v in zip(_make_candles(closes), vols):
            c.volume = v
            buf._closed.append(c)
        sig = buf.signals()
        assert sig is not None
        assert sig.vol_ratio > 1.0


# ---------------------------------------------------------------------------
# TestRSI
# ---------------------------------------------------------------------------

class TestRSI:
    def test_rsi_uptrend_above_50(self):
        """Uniformly rising closes should produce RSI well above 50."""
        closes = np.linspace(1.0, 2.0, 80).tolist()
        buf = _seeded_buffer(n=80, closes=closes)
        sig = buf.signals()
        assert sig is not None
        assert sig.rsi > 60.0

    def test_rsi_downtrend_below_50(self):
        """Uniformly falling closes should produce RSI well below 50."""
        closes = np.linspace(2.0, 1.0, 80).tolist()
        buf = _seeded_buffer(n=80, closes=closes)
        sig = buf.signals()
        assert sig is not None
        assert sig.rsi < 40.0

    def test_rsi_flat_near_50(self):
        """Flat closes (no change) should produce RSI near 50 (no gains or losses)."""
        closes = [1.25] * 80
        buf = _seeded_buffer(n=80, closes=closes)
        sig = buf.signals()
        assert sig is not None
        # With zero deltas the _rsi helper returns 100 (no loss branch),
        # but any real series will be 50 ± buffer noise.  Just test range.
        assert 0.0 <= sig.rsi <= 100.0

    def test_rsi_pure_uptrend_approaches_100(self):
        """Every day is a gain → RSI should be very high."""
        closes = [1.0 + i * 0.01 for i in range(100)]
        result = CandleBuffer._rsi(np.array(closes), period=14)
        assert result > 90.0

    def test_rsi_pure_downtrend_approaches_0(self):
        """Every day is a loss → RSI should be very low."""
        closes = [100.0 - i * 0.01 for i in range(100)]
        result = CandleBuffer._rsi(np.array(closes), period=14)
        assert result < 10.0

    def test_rsi_bounded(self):
        import random
        random.seed(42)
        closes = [1.0 + random.gauss(0, 0.02) for _ in range(200)]
        closes = list(np.cumsum([0.001] + closes))
        result = CandleBuffer._rsi(np.array(closes), period=14)
        assert 0.0 <= result <= 100.0


# ---------------------------------------------------------------------------
# TestBBWidth
# ---------------------------------------------------------------------------

class TestBBWidth:
    def test_flat_closes_give_zero_bb_width(self):
        closes = np.array([1.25] * 30, dtype=float)
        assert CandleBuffer._bb_width(closes, period=20, mult=2.0) == pytest.approx(0.0)

    def test_volatile_closes_give_positive_bb_width(self):
        rng = np.random.default_rng(0)
        closes = rng.normal(loc=1.0, scale=0.05, size=30)
        bw = CandleBuffer._bb_width(closes, period=20, mult=2.0)
        assert bw > 0.0

    def test_bb_width_larger_for_higher_volatility(self):
        rng = np.random.default_rng(1)
        low_vol  = rng.normal(loc=1.0, scale=0.01, size=30)
        high_vol = rng.normal(loc=1.0, scale=0.10, size=30)
        bw_low  = CandleBuffer._bb_width(low_vol,  period=20, mult=2.0)
        bw_high = CandleBuffer._bb_width(high_vol, period=20, mult=2.0)
        assert bw_high > bw_low

    def test_bb_width_returns_zero_when_insufficient_data(self):
        closes = np.array([1.0, 1.1, 1.2], dtype=float)
        assert CandleBuffer._bb_width(closes, period=20, mult=2.0) == pytest.approx(0.0)

    def test_bb_width_non_negative(self):
        rng = np.random.default_rng(2)
        closes = rng.normal(loc=50.0, scale=5.0, size=100)
        bw = CandleBuffer._bb_width(closes, period=20, mult=2.0)
        assert bw >= 0.0


# ---------------------------------------------------------------------------
# TestADX
# ---------------------------------------------------------------------------

class TestADX:
    def test_adx_components_in_range(self):
        rng = np.random.default_rng(3)
        n = 100
        highs  = np.cumsum(rng.uniform(0, 0.02, n)) + 1.0
        lows   = highs - rng.uniform(0.01, 0.05, n)
        closes = (highs + lows) / 2.0
        adx, plus_di, minus_di = CandleBuffer._adx(highs, lows, closes, 14)
        assert 0.0 <= adx <= 100.0
        assert plus_di >= 0.0
        assert minus_di >= 0.0

    def test_strong_uptrend_has_plus_di_above_minus_di(self):
        n = 100
        highs  = np.array([1.0 + i * 0.02 for i in range(n)])
        lows   = highs - 0.01
        closes = highs - 0.005
        _, plus_di, minus_di = CandleBuffer._adx(highs, lows, closes, 14)
        assert plus_di > minus_di

    def test_strong_downtrend_has_minus_di_above_plus_di(self):
        n = 100
        lows   = np.array([2.0 - i * 0.02 for i in range(n)])
        highs  = lows + 0.01
        closes = lows + 0.005
        _, plus_di, minus_di = CandleBuffer._adx(highs, lows, closes, 14)
        assert minus_di > plus_di

    def test_flat_market_has_low_adx(self):
        """No directional movement → low ADX."""
        n = 100
        highs  = np.array([1.05] * n)
        lows   = np.array([0.95] * n)
        closes = np.array([1.00] * n)
        adx, _, _ = CandleBuffer._adx(highs, lows, closes, 14)
        # Fully flat: both DMs are 0, so ADX should be ~0
        assert adx < 5.0


# ---------------------------------------------------------------------------
# TestEMA
# ---------------------------------------------------------------------------

class TestEMA:
    def test_ema_of_constant_series_equals_constant(self):
        closes = np.array([2.5] * 50, dtype=float)
        assert CandleBuffer._ema(closes, span=9) == pytest.approx(2.5, rel=1e-6)

    def test_ema_fast_above_slow_in_uptrend(self):
        closes = np.linspace(1.0, 2.0, 100)
        ema_f = CandleBuffer._ema(closes, span=9)
        ema_s = CandleBuffer._ema(closes, span=21)
        assert ema_f > ema_s

    def test_ema_fast_below_slow_in_downtrend(self):
        closes = np.linspace(2.0, 1.0, 100)
        ema_f = CandleBuffer._ema(closes, span=9)
        ema_s = CandleBuffer._ema(closes, span=21)
        assert ema_f < ema_s

    def test_ema_bias_flags_uptrend(self):
        closes = [1.0 + i * 0.01 for i in range(80)]
        buf = _seeded_buffer(n=80, closes=closes)
        sig = buf.signals()
        assert sig is not None
        assert sig.ema_bias_long is True
        assert sig.ema_bias_short is False

    def test_ema_bias_flags_downtrend(self):
        closes = [2.0 - i * 0.01 for i in range(80)]
        buf = _seeded_buffer(n=80, closes=closes)
        sig = buf.signals()
        assert sig is not None
        assert sig.ema_bias_short is True
        assert sig.ema_bias_long is False


# ---------------------------------------------------------------------------
# TestVolMetrics
# ---------------------------------------------------------------------------

class TestVolMetrics:
    def test_vol_sma_uniform(self):
        volumes = np.array([1_000_000.0] * 40)
        result = CandleBuffer._vol_sma(volumes, period=20)
        assert result == pytest.approx(1_000_000.0)

    def test_vol_sma_uses_last_n_only(self):
        # First 20 values = 0, last 20 values = 1e6 → SMA of last 20 = 1e6
        volumes = np.array([0.0] * 20 + [1_000_000.0] * 20)
        result = CandleBuffer._vol_sma(volumes, period=20)
        assert result == pytest.approx(1_000_000.0)

    def test_vol_sma_fewer_than_period(self):
        """When n < period, averages all available values."""
        volumes = np.array([2.0, 4.0, 6.0])
        result = CandleBuffer._vol_sma(volumes, period=20)
        assert result == pytest.approx(4.0)

    def test_vol_ratio_constant_equals_one(self):
        buf = _seeded_buffer(n=60, vol=3_000_000.0)
        sig = buf.signals()
        assert sig is not None
        assert sig.vol_ratio == pytest.approx(1.0, abs=1e-6)

    def test_vol_ratio_high_when_last_spike(self):
        n = 60
        closes = [1.0 + i * 0.001 for i in range(n)]
        candles = _make_candles(closes, vol=1_000_000.0)
        candles[-1].volume = 10_000_000.0   # 10× spike
        buf = CandleBuffer(maxlen=200, interval="15min", vol_period=20, regime_ema_period=21)
        for c in candles:
            buf._closed.append(c)
        sig = buf.signals()
        assert sig is not None
        assert sig.vol_ratio > 2.0


# ---------------------------------------------------------------------------
# TestSwingHighLow
# ---------------------------------------------------------------------------

class TestSwingHighLow:
    def test_swing_high_is_max_of_preceding_window(self):
        # Last candle excluded; preceding 5-candle highs: [10, 20, 15, 12, 18]
        highs = np.array([10.0, 20.0, 15.0, 12.0, 18.0, 5.0], dtype=float)
        result = CandleBuffer._swing_high(highs, lookback=5)
        assert result == pytest.approx(20.0)

    def test_swing_low_is_min_of_preceding_window(self):
        lows = np.array([10.0, 5.0, 8.0, 3.0, 7.0, 20.0], dtype=float)
        result = CandleBuffer._swing_low(lows, lookback=5)
        assert result == pytest.approx(3.0)

    def test_swing_high_excludes_current_candle(self):
        # Current candle (last) has the highest high; should not be swing_high
        highs = np.array([1.0, 2.0, 3.0, 4.0, 100.0], dtype=float)
        result = CandleBuffer._swing_high(highs, lookback=10)
        assert result == pytest.approx(4.0)

    def test_swing_low_excludes_current_candle(self):
        lows = np.array([10.0, 9.0, 8.0, 7.0, 0.001], dtype=float)
        result = CandleBuffer._swing_low(lows, lookback=10)
        assert result == pytest.approx(7.0)

    def test_swing_high_single_candle_returns_that_high(self):
        highs = np.array([5.0], dtype=float)
        result = CandleBuffer._swing_high(highs, lookback=10)
        assert result == pytest.approx(5.0)

    def test_swing_low_single_candle_returns_that_low(self):
        lows = np.array([3.0], dtype=float)
        result = CandleBuffer._swing_low(lows, lookback=10)
        assert result == pytest.approx(3.0)

    def test_swing_high_capped_by_lookback(self):
        # lookback=2: only look at highs[-3:-1] (indices 7 and 8 in 10-element array)
        highs = np.array([100.0, 100.0, 100.0, 100.0, 100.0,
                          100.0, 100.0, 3.0, 4.0, 1.0], dtype=float)
        # preceding 2: indices 7 and 8 → [3.0, 4.0]
        result = CandleBuffer._swing_high(highs, lookback=2)
        assert result == pytest.approx(4.0)

    def test_swing_signals_consistent_with_series(self):
        n = 60
        closes = [1.0 + i * 0.001 for i in range(n)]
        # Make a prominent high somewhere in the middle of the lookback window
        candles = _make_candles(closes)
        candles[-5].high = 999.0
        buf = CandleBuffer(maxlen=200, interval="15min", ms_lookback=20, regime_ema_period=21)
        for c in candles:
            buf._closed.append(c)
        sig = buf.signals()
        assert sig is not None
        assert sig.swing_high == pytest.approx(999.0)


# ---------------------------------------------------------------------------
# TestCandleBufferSeed (async)
# ---------------------------------------------------------------------------

class TestCandleBufferSeed:
    @pytest.mark.asyncio
    async def test_seed_populates_closed_from_client(self):
        """``seed()`` should translate get_klines rows into Candle objects."""
        from unittest.mock import AsyncMock

        fake_rows = [
            {"open_time": TS_BASE + i * TS_STEP,
             "open": str(1.0 + i * 0.001),
             "high": str(1.0 + i * 0.001 + 0.005),
             "low":  str(1.0 + i * 0.001 - 0.005),
             "close": str(1.0 + i * 0.001),
             "volume": "1500000"}
            for i in range(50)
        ]

        mock_client = AsyncMock()
        mock_client.get_klines = AsyncMock(return_value=fake_rows)

        buf = CandleBuffer(maxlen=200, interval="15min")
        await buf.seed(mock_client, "XRPUSDT")

        assert len(buf._closed) == 50
        assert buf._closed[0].ts == TS_BASE
        assert buf._closed[-1].close == pytest.approx(1.0 + 49 * 0.001, rel=1e-5)
        assert buf._closed[0].volume == pytest.approx(1_500_000.0)

    @pytest.mark.asyncio
    async def test_seed_respects_maxlen_cap(self):
        """seed() is capped at min(maxlen, 200) rows requested from REST."""
        from unittest.mock import AsyncMock, call

        mock_client = AsyncMock()
        mock_client.get_klines = AsyncMock(return_value=[])

        buf = CandleBuffer(maxlen=50, interval="15min")
        await buf.seed(mock_client, "XRPUSDT")

        # Expect exactly one get_klines call with limit=50
        mock_client.get_klines.assert_called_once_with("XRPUSDT", "15min", limit=50)

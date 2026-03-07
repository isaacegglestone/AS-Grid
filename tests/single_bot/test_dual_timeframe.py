"""
tests/single_bot/test_dual_timeframe.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the dual-timeframe architecture added in the 1min+15min candle
implementation.  Covers:

- ``_merge_signals()``      – regime EMA overlay logic
- ``handle_kline_update``   – 15min handler stores ``_regime_signals`` only
- ``handle_kline_1m_update`` – 1min handler merges signals + drives trend
- Startup seeding of both buffers

These are complementary to the existing handler tests in
``test_handle_kline_update.py`` which focus on guards, buffer wiring, and
error swallowing.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from collections import deque

import pytest

from src.single_bot.bitunix_bot import GridTradingBot
from src.single_bot.indicators import CandleBuffer, Candle, Signals


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TS_BASE = 1_740_000_000_000
TS_1MIN = 60 * 1_000         # 1 minute in ms
TS_15MIN = 15 * 60 * 1_000   # 15 minutes in ms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot() -> GridTradingBot:
    """Minimal GridTradingBot with dummy credentials (no I/O)."""
    return GridTradingBot(
        api_key="test_key",
        api_secret="test_secret",
        coin_name="XRP",
        grid_spacing=0.01,
        initial_quantity=10,
        leverage=5,
    )


def _kline_1m(ts: int, c="1.26", **overrides) -> dict:
    """Well-formed ``market_kline_1min`` WS payload."""
    payload = {
        "ch": "market_kline_1min",
        "symbol": "XRPUSDT",
        "ts": ts,
        "data": {"o": "1.25", "h": "1.27", "l": "1.23", "c": c,
                 "b": "2500000", "q": "3150000"},
    }
    payload["data"].update(overrides)
    return payload


def _kline_15m(ts: int, c="1.26", **overrides) -> dict:
    """Well-formed ``market_kline_15min`` WS payload."""
    payload = {
        "ch": "market_kline_15min",
        "symbol": "XRPUSDT",
        "ts": ts,
        "data": {"o": "1.25", "h": "1.27", "l": "1.23", "c": c,
                 "b": "2500000", "q": "3150000"},
    }
    payload["data"].update(overrides)
    return payload


def _fake_signals(**kwargs) -> Signals:
    """Create a ``Signals`` with sensible defaults, overridable via kwargs."""
    defaults = dict(
        adx=25.0, rsi=55.0, bb_width=0.025, close=1.26, atr=0.02,
        ema_fast=1.26, ema_slow=1.25, ema_bias_long=True, ema_bias_short=False,
        vol_ratio=1.0, regime_ema=1.20, regime_ema_87=1.22, regime_ema_42=1.24,
        volume=3_000_000.0,
    )
    defaults.update(kwargs)
    return Signals(**defaults)


def _seeded_buffer(n: int = 60, interval: str = "1min",
                   regime_ema_period: int = 1) -> CandleBuffer:
    """Pre-seeded ``CandleBuffer`` with *n* synthetic candles."""
    step = TS_1MIN if interval == "1min" else TS_15MIN
    buf = CandleBuffer(maxlen=max(200, n + 50), interval=interval,
                       regime_ema_period=regime_ema_period)
    for i in range(n):
        buf._closed.append(
            Candle(
                ts=TS_BASE + i * step,
                open=1.0 + i * 0.001,
                high=1.0 + i * 0.001 + 0.05,
                low=1.0 + i * 0.001 - 0.05,
                close=1.0 + i * 0.001,
                volume=1_000_000.0,
            )
        )
    return buf


# ==========================================================================
# TestMergeSignals
# ==========================================================================

class TestMergeSignals:
    """_merge_signals() overlays 15-min regime EMAs onto a 1-min Signals obj."""

    def test_overlay_regime_from_stored_15min_signals(self):
        """When _regime_signals is populated, regime_ema/87/42 are copied."""
        bot = _make_bot()
        bot._regime_signals = _fake_signals(
            regime_ema=1.20, regime_ema_87=1.22, regime_ema_42=1.24
        )

        sig_1m = _fake_signals(regime_ema=0.0, regime_ema_87=0.0, regime_ema_42=0.0)
        merged = bot._merge_signals(sig_1m)

        assert merged.regime_ema == pytest.approx(1.20)
        assert merged.regime_ema_87 == pytest.approx(1.22)
        assert merged.regime_ema_42 == pytest.approx(1.24)

    def test_regime_zeroed_when_no_15min_data(self):
        """Before 15min buffer warms up, regime_ema fields must be 0.0."""
        bot = _make_bot()
        assert bot._regime_signals is None

        sig_1m = _fake_signals(regime_ema=99.9, regime_ema_87=99.9, regime_ema_42=99.9)
        merged = bot._merge_signals(sig_1m)

        assert merged.regime_ema == 0.0
        assert merged.regime_ema_87 == 0.0
        assert merged.regime_ema_42 == 0.0

    def test_short_period_indicators_preserved(self):
        """Merge must not overwrite ATR, ADX, RSI, etc. from the 1-min buffer."""
        bot = _make_bot()
        bot._regime_signals = _fake_signals(regime_ema=1.20)

        sig_1m = _fake_signals(adx=42.0, rsi=65.0, atr=0.035, close=1.30)
        merged = bot._merge_signals(sig_1m)

        assert merged.adx == pytest.approx(42.0)
        assert merged.rsi == pytest.approx(65.0)
        assert merged.atr == pytest.approx(0.035)
        assert merged.close == pytest.approx(1.30)

    def test_returns_mutated_input_object(self):
        """_merge_signals mutates and returns the same Signals obj (no copy)."""
        bot = _make_bot()
        bot._regime_signals = _fake_signals(regime_ema=1.20)

        sig_1m = _fake_signals()
        result = bot._merge_signals(sig_1m)
        assert result is sig_1m

    def test_regime_updated_independently_on_each_call(self):
        """Two successive merges with different regimes produce different results."""
        bot = _make_bot()

        # First merge
        bot._regime_signals = _fake_signals(regime_ema=1.10)
        sig_a = _fake_signals()
        bot._merge_signals(sig_a)
        assert sig_a.regime_ema == pytest.approx(1.10)

        # Update regime
        bot._regime_signals = _fake_signals(regime_ema=1.30)
        sig_b = _fake_signals()
        bot._merge_signals(sig_b)
        assert sig_b.regime_ema == pytest.approx(1.30)


# ==========================================================================
# TestHandleKlineUpdate15m
# ==========================================================================

class TestHandleKlineUpdate15m:
    """15-min handler must store regime signals but NOT drive trend evaluation."""

    @pytest.mark.asyncio
    async def test_stores_regime_signals_on_candle_close(self):
        """On 15min candle close, _regime_signals is set from signals()."""
        bot = _make_bot()
        assert bot._regime_signals is None

        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.return_value = True  # candle closed
        regime = _fake_signals(regime_ema=1.22, regime_ema_87=1.21, regime_ema_42=1.23)
        mock_buf.signals.return_value = regime
        bot.candle_buffer = mock_buf

        await bot.handle_kline_update(_kline_15m(ts=TS_BASE))

        assert bot._regime_signals is regime

    @pytest.mark.asyncio
    async def test_does_not_call_evaluate_trend(self):
        """15min handler must NOT call _evaluate_trend — that's 1min's job."""
        bot = _make_bot()
        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.return_value = True
        mock_buf.signals.return_value = _fake_signals()
        bot.candle_buffer = mock_buf

        with patch.object(bot, "_evaluate_trend", new_callable=AsyncMock) as mock_eval:
            await bot.handle_kline_update(_kline_15m(ts=TS_BASE))
            mock_eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_update_latest_signals(self):
        """latest_signals is driven by 1min only — 15min must not touch it."""
        bot = _make_bot()
        sentinel = object()
        bot.latest_signals = sentinel  # type: ignore[assignment]

        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.return_value = True
        mock_buf.signals.return_value = _fake_signals()
        bot.candle_buffer = mock_buf

        await bot.handle_kline_update(_kline_15m(ts=TS_BASE))
        assert bot.latest_signals is sentinel  # unchanged

    @pytest.mark.asyncio
    async def test_regime_not_set_when_candle_not_closed(self):
        """No candle close → _regime_signals stays None."""
        bot = _make_bot()
        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.return_value = False  # no close
        bot.candle_buffer = mock_buf

        await bot.handle_kline_update(_kline_15m(ts=TS_BASE))
        assert bot._regime_signals is None

    @pytest.mark.asyncio
    async def test_regime_not_set_when_signals_none(self):
        """Candle closed but buffer not warmed → signals() returns None."""
        bot = _make_bot()
        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.return_value = True
        mock_buf.signals.return_value = None  # warm-up
        bot.candle_buffer = mock_buf

        await bot.handle_kline_update(_kline_15m(ts=TS_BASE))
        assert bot._regime_signals is None


# ==========================================================================
# TestHandleKline1mUpdate
# ==========================================================================

class TestHandleKline1mUpdate:
    """1-min handler: merge signals, update latest_signals, call _evaluate_trend."""

    @pytest.mark.asyncio
    async def test_calls_evaluate_trend_on_candle_close(self):
        """1min candle close → _evaluate_trend is called."""
        bot = _make_bot()
        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.return_value = True
        sig = _fake_signals(close=1.30)
        mock_buf.signals.return_value = sig
        bot.candle_buffer_1m = mock_buf

        with patch.object(bot, "_evaluate_trend", new_callable=AsyncMock) as mock_eval:
            await bot.handle_kline_1m_update(_kline_1m(ts=TS_BASE))
            mock_eval.assert_called_once_with(pytest.approx(1.30))

    @pytest.mark.asyncio
    async def test_does_not_call_evaluate_trend_when_no_close(self):
        """No candle close → _evaluate_trend must NOT be called."""
        bot = _make_bot()
        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.return_value = False
        bot.candle_buffer_1m = mock_buf

        with patch.object(bot, "_evaluate_trend", new_callable=AsyncMock) as mock_eval:
            await bot.handle_kline_1m_update(_kline_1m(ts=TS_BASE))
            mock_eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_merges_regime_into_latest_signals(self):
        """After 1min close, latest_signals should contain 15min regime EMAs."""
        bot = _make_bot()
        bot._regime_signals = _fake_signals(
            regime_ema=1.20, regime_ema_87=1.22, regime_ema_42=1.24,
        )

        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.return_value = True
        sig_1m = _fake_signals(regime_ema=0.0, regime_ema_87=0.0, regime_ema_42=0.0,
                                adx=30.0, close=1.28)
        mock_buf.signals.return_value = sig_1m
        bot.candle_buffer_1m = mock_buf

        with patch.object(bot, "_evaluate_trend", new_callable=AsyncMock):
            await bot.handle_kline_1m_update(_kline_1m(ts=TS_BASE))

        assert bot.latest_signals.regime_ema == pytest.approx(1.20)
        assert bot.latest_signals.regime_ema_87 == pytest.approx(1.22)
        assert bot.latest_signals.regime_ema_42 == pytest.approx(1.24)
        assert bot.latest_signals.adx == pytest.approx(30.0)

    @pytest.mark.asyncio
    async def test_regime_zero_when_15min_not_warmed(self):
        """Before 15min data arrives, merged regime_ema fields are 0.0."""
        bot = _make_bot()
        assert bot._regime_signals is None

        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.return_value = True
        mock_buf.signals.return_value = _fake_signals(close=1.28)
        bot.candle_buffer_1m = mock_buf

        with patch.object(bot, "_evaluate_trend", new_callable=AsyncMock):
            await bot.handle_kline_1m_update(_kline_1m(ts=TS_BASE))

        assert bot.latest_signals.regime_ema == 0.0
        assert bot.latest_signals.regime_ema_87 == 0.0
        assert bot.latest_signals.regime_ema_42 == 0.0

    @pytest.mark.asyncio
    async def test_latest_signals_unchanged_when_buffer_not_warmed(self):
        """signals() returns None during warm-up → latest_signals unchanged."""
        bot = _make_bot()
        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.return_value = True
        mock_buf.signals.return_value = None
        bot.candle_buffer_1m = mock_buf

        with patch.object(bot, "_evaluate_trend", new_callable=AsyncMock) as mock_eval:
            await bot.handle_kline_1m_update(_kline_1m(ts=TS_BASE))
            mock_eval.assert_not_called()
        assert bot.latest_signals is None

    @pytest.mark.asyncio
    async def test_exception_swallowed(self):
        """Errors in handle_kline_1m_update must not propagate."""
        bot = _make_bot()
        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.side_effect = RuntimeError("boom")
        bot.candle_buffer_1m = mock_buf

        # Must not raise
        await bot.handle_kline_1m_update(_kline_1m(ts=TS_BASE))

    @pytest.mark.asyncio
    async def test_missing_ts_returns_early(self):
        """Payload with ts=0 → guard returns early, no update."""
        bot = _make_bot()
        mock_buf = MagicMock(spec=CandleBuffer)
        bot.candle_buffer_1m = mock_buf

        await bot.handle_kline_1m_update({"ch": "market_kline_1min", "ts": 0, "data": {}})
        mock_buf.update.assert_not_called()


# ==========================================================================
# TestDualTimeframeIntegration
# ==========================================================================

class TestDualTimeframeIntegration:
    """End-to-end scenarios with both 1min and 15min handlers working together."""

    @pytest.mark.asyncio
    async def test_15min_then_1min_produces_merged_signals(self):
        """
        Scenario: 15min candle closes (sets regime), then 1min candle closes.
        Result: latest_signals should have 1min indicators + 15min regime.
        """
        bot = _make_bot()

        # --- 15min: mock buffer with candle close ---
        mock_15m = MagicMock(spec=CandleBuffer)
        mock_15m.update.return_value = True
        regime = _fake_signals(regime_ema=1.22, regime_ema_87=1.21, regime_ema_42=1.23)
        mock_15m.signals.return_value = regime
        bot.candle_buffer = mock_15m

        await bot.handle_kline_update(_kline_15m(ts=TS_BASE))
        assert bot._regime_signals is regime

        # --- 1min: mock buffer with candle close ---
        mock_1m = MagicMock(spec=CandleBuffer)
        mock_1m.update.return_value = True
        sig_1m = _fake_signals(adx=35.0, rsi=60.0, close=1.30,
                                regime_ema=0.0, regime_ema_87=0.0, regime_ema_42=0.0)
        mock_1m.signals.return_value = sig_1m
        bot.candle_buffer_1m = mock_1m

        with patch.object(bot, "_evaluate_trend", new_callable=AsyncMock):
            await bot.handle_kline_1m_update(_kline_1m(ts=TS_BASE))

        # Merged: 1min indicators + 15min regime
        assert bot.latest_signals.adx == pytest.approx(35.0)
        assert bot.latest_signals.rsi == pytest.approx(60.0)
        assert bot.latest_signals.close == pytest.approx(1.30)
        assert bot.latest_signals.regime_ema == pytest.approx(1.22)
        assert bot.latest_signals.regime_ema_87 == pytest.approx(1.21)
        assert bot.latest_signals.regime_ema_42 == pytest.approx(1.23)

    @pytest.mark.asyncio
    async def test_1min_before_15min_gets_zero_regime(self):
        """
        Scenario: 1min candle closes before any 15min data arrives.
        Result: regime_ema fields are 0.0 — safe for all consumers.
        """
        bot = _make_bot()
        assert bot._regime_signals is None

        mock_1m = MagicMock(spec=CandleBuffer)
        mock_1m.update.return_value = True
        sig_1m = _fake_signals(adx=20.0, close=1.25)
        mock_1m.signals.return_value = sig_1m
        bot.candle_buffer_1m = mock_1m

        with patch.object(bot, "_evaluate_trend", new_callable=AsyncMock):
            await bot.handle_kline_1m_update(_kline_1m(ts=TS_BASE))

        assert bot.latest_signals.regime_ema == 0.0
        assert bot.latest_signals.adx == pytest.approx(20.0)

    @pytest.mark.asyncio
    async def test_regime_refresh_picked_up_by_next_1min_close(self):
        """
        Scenario: 1min close → regime=0, then 15min close updates regime,
        then another 1min close → regime should now be populated.
        """
        bot = _make_bot()

        # Step 1: 1min close (no regime yet)
        mock_1m = MagicMock(spec=CandleBuffer)
        mock_1m.update.return_value = True
        sig_a = _fake_signals(close=1.25, regime_ema=0.0, regime_ema_87=0.0, regime_ema_42=0.0)
        mock_1m.signals.return_value = sig_a
        bot.candle_buffer_1m = mock_1m

        with patch.object(bot, "_evaluate_trend", new_callable=AsyncMock):
            await bot.handle_kline_1m_update(_kline_1m(ts=TS_BASE))
        assert bot.latest_signals.regime_ema == 0.0

        # Step 2: 15min close (regime arrives)
        mock_15m = MagicMock(spec=CandleBuffer)
        mock_15m.update.return_value = True
        mock_15m.signals.return_value = _fake_signals(regime_ema=1.30)
        bot.candle_buffer = mock_15m

        await bot.handle_kline_update(_kline_15m(ts=TS_BASE))
        assert bot._regime_signals.regime_ema == pytest.approx(1.30)

        # Step 3: Next 1min close — should pick up regime
        sig_b = _fake_signals(close=1.28, regime_ema=0.0, regime_ema_87=0.0, regime_ema_42=0.0)
        mock_1m.signals.return_value = sig_b

        with patch.object(bot, "_evaluate_trend", new_callable=AsyncMock):
            await bot.handle_kline_1m_update(_kline_1m(ts=TS_BASE + TS_1MIN))
        assert bot.latest_signals.regime_ema == pytest.approx(1.30)

    @pytest.mark.asyncio
    async def test_consecutive_15min_closes_update_regime(self):
        """Multiple 15min closes progressively update _regime_signals."""
        bot = _make_bot()

        mock_15m = MagicMock(spec=CandleBuffer)
        mock_15m.update.return_value = True
        bot.candle_buffer = mock_15m

        # First 15min close
        mock_15m.signals.return_value = _fake_signals(regime_ema=1.10)
        await bot.handle_kline_update(_kline_15m(ts=TS_BASE))
        assert bot._regime_signals.regime_ema == pytest.approx(1.10)

        # Second 15min close — regime should update
        mock_15m.signals.return_value = _fake_signals(regime_ema=1.15)
        await bot.handle_kline_update(_kline_15m(ts=TS_BASE + TS_15MIN))
        assert bot._regime_signals.regime_ema == pytest.approx(1.15)


# ==========================================================================
# TestConstructorDualBuffers
# ==========================================================================

class TestConstructorDualBuffers:
    """Verify both candle buffers are created with correct configuration."""

    def test_1min_buffer_exists(self):
        bot = _make_bot()
        assert hasattr(bot, "candle_buffer_1m")
        assert isinstance(bot.candle_buffer_1m, CandleBuffer)

    def test_15min_buffer_exists(self):
        bot = _make_bot()
        assert hasattr(bot, "candle_buffer")
        assert isinstance(bot.candle_buffer, CandleBuffer)

    def test_1min_buffer_interval(self):
        bot = _make_bot()
        assert bot.candle_buffer_1m.interval == "1min"

    def test_15min_buffer_interval(self):
        bot = _make_bot()
        assert bot.candle_buffer.interval == "15min"

    def test_1min_regime_ema_disabled(self):
        """1min buffer should have regime_ema_period=1 (effectively disabled)."""
        bot = _make_bot()
        assert bot.candle_buffer_1m.regime_ema_period == 1

    def test_15min_has_regime_ema(self):
        """15min buffer should have a real regime_ema_period (>1)."""
        bot = _make_bot()
        assert bot.candle_buffer.regime_ema_period > 1

    def test_regime_signals_initially_none(self):
        bot = _make_bot()
        assert bot._regime_signals is None

    def test_latest_signals_initially_none(self):
        bot = _make_bot()
        assert bot.latest_signals is None

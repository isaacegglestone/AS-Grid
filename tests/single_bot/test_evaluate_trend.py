"""
tests/single_bot/test_evaluate_trend.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit & integration tests for ``GridTradingBot._evaluate_trend`` covering:

HIGH priority:
- Trailing stop (peak tracking, trail close, reversal close)
- Confirmation counter (increment, reset, threshold)
- Force-close opposing grid on confirmed trend
- Trend cooldown (increment, completion, rebound reset)

LOW priority:
- Guard conditions (buffer too small, past_price zero, no re-fire)
- End-to-end lifecycle (up + down)
"""
from __future__ import annotations

from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.single_bot.bitunix_bot import (
    ADX_MIN_TREND,
    TREND_CAP_VEL_PCT,
    TREND_CONFIRM_CANDLES,
    TREND_COOLDOWN_CANDLES,
    TREND_LOOKBACK_CANDLES,
    TREND_TRAIL_PCT,
    TREND_VELOCITY_PCT,
    GridTradingBot,
)
from src.single_bot.indicators import Candle, CandleBuffer, Signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TS_BASE = 1_740_000_000_000
TS_STEP = 15 * 60 * 1_000


def _make_bot(**overrides) -> GridTradingBot:
    bot = GridTradingBot(
        api_key="k",
        api_secret="s",
        coin_name="XRP",
        grid_spacing=0.01,
        initial_quantity=10,
        leverage=5,
    )
    for k, v in overrides.items():
        setattr(bot, k, v)
    return bot


def _seeded_buffer(n: int = 15, base_close: float = 1.0) -> CandleBuffer:
    """Buffer with *n* closed candles; gentle uptrend from *base_close*."""
    buf = CandleBuffer(maxlen=210, interval="15min", regime_ema_period=21)
    for i in range(n):
        c = base_close + i * 0.001
        buf._closed.append(
            Candle(
                ts=TS_BASE + i * TS_STEP,
                open=c,
                high=c + 0.05,
                low=c - 0.05,
                close=c,
                volume=1_000_000.0,
            )
        )
    return buf


def _append_candle(buf: CandleBuffer, close: float, idx: int) -> None:
    """Append a single closed candle to the buffer."""
    buf._closed.append(
        Candle(
            ts=TS_BASE + idx * TS_STEP,
            open=close,
            high=close + 0.01,
            low=close - 0.01,
            close=close,
            volume=1_000_000.0,
        )
    )


# ===========================================================================
# TestTrailingStopLong — step 2 long side
# ===========================================================================


class TestTrailingStopLong:
    """Trail-stop management for an existing long trend position."""

    def _setup(self, base_close: float = 1.06) -> GridTradingBot:
        """Bot with long trend_position; peak=1.10, entry=1.05."""
        bot = _make_bot()
        bot.candle_buffer = _seeded_buffer(n=15, base_close=base_close)
        bot.trend_mode = "up"
        bot.trend_position = {
            "side": "long", "entry": 1.05, "qty": 10, "peak": 1.10,
        }
        bot.trend_pending_dir = "up"
        bot.trend_confirm_counter = 0
        bot.latest_signals = Signals(adx=30.0)
        return bot

    @pytest.mark.asyncio
    async def test_peak_tracks_upward(self):
        """Price above current peak → peak updates."""
        bot = self._setup(base_close=1.10)
        with patch.object(bot, "_close_trend_trade", new_callable=AsyncMock):
            await bot._evaluate_trend(1.12)
        assert bot.trend_position["peak"] == 1.12

    @pytest.mark.asyncio
    async def test_peak_does_not_track_down(self):
        """Price below current peak → peak unchanged."""
        bot = self._setup(base_close=1.06)
        with patch.object(bot, "_close_trend_trade", new_callable=AsyncMock):
            await bot._evaluate_trend(1.08)
        assert bot.trend_position["peak"] == 1.10

    @pytest.mark.asyncio
    async def test_trail_stop_triggers_close(self):
        """Price ≤ trail_stop → _close_trend_trade('trail') called."""
        # peak=1.10, trail_stop = 1.10 * 0.96 = 1.056
        # Use base_close=1.03 so velocity is neutral
        bot = self._setup(base_close=1.03)
        with patch.object(bot, "_close_trend_trade", new_callable=AsyncMock) as mock_close:
            await bot._evaluate_trend(1.05)  # 1.05 < 1.056
        mock_close.assert_called_once_with(1.05, reason="trail")

    @pytest.mark.asyncio
    async def test_reversal_triggers_close(self):
        """trending_down (but above trail_stop) → close with reason='reversal'."""
        # peak=1.10, trail_stop = 1.056
        # base_close=1.12 → closed[-10]=1.125 → vel=(1.07-1.125)/1.125=-0.049 → trending_down
        bot = self._setup(base_close=1.12)
        with patch.object(bot, "_close_trend_trade", new_callable=AsyncMock) as mock_close:
            await bot._evaluate_trend(1.07)  # above 1.056 but trending_down
        mock_close.assert_called_once_with(1.07, reason="reversal")

    @pytest.mark.asyncio
    async def test_no_close_within_trail(self):
        """Price above trail_stop and not reversing → no close."""
        bot = self._setup(base_close=1.06)
        # price=1.08 > trail_stop(1.056), velocity neutral
        with patch.object(bot, "_close_trend_trade", new_callable=AsyncMock) as mock_close:
            await bot._evaluate_trend(1.08)
        mock_close.assert_not_called()


# ===========================================================================
# TestTrailingStopShort — step 2 short side
# ===========================================================================


class TestTrailingStopShort:
    """Trail-stop management for an existing short trend position."""

    def _setup(self, base_close: float = 0.94) -> GridTradingBot:
        """Bot with short trend_position; peak=0.95, entry=1.00."""
        bot = _make_bot()
        bot.candle_buffer = _seeded_buffer(n=15, base_close=base_close)
        bot.trend_mode = "down"
        bot.trend_position = {
            "side": "short", "entry": 1.00, "qty": 10, "peak": 0.95,
        }
        bot.trend_pending_dir = "down"
        bot.trend_confirm_counter = 0
        bot.latest_signals = Signals(adx=30.0)
        return bot

    @pytest.mark.asyncio
    async def test_short_peak_tracks_downward(self):
        """Price below current peak → peak updates."""
        bot = self._setup(base_close=0.92)
        with patch.object(bot, "_close_trend_trade", new_callable=AsyncMock):
            await bot._evaluate_trend(0.93)
        assert bot.trend_position["peak"] == 0.93

    @pytest.mark.asyncio
    async def test_short_trail_stop_triggers_close(self):
        """Price ≥ trail_stop → close by trail."""
        # peak=0.95, trail_stop = 0.95 * 1.04 = 0.988
        bot = self._setup(base_close=0.97)
        with patch.object(bot, "_close_trend_trade", new_callable=AsyncMock) as mock_close:
            await bot._evaluate_trend(0.99)  # 0.99 ≥ 0.988
        mock_close.assert_called_once_with(0.99, reason="trail")

    @pytest.mark.asyncio
    async def test_short_reversal_triggers_close(self):
        """trending_up (but below trail_stop) → close with reason='reversal'."""
        # trail_stop = 0.988;  base_close=0.92 → vel=(0.97-0.925)/0.925=0.049 → trending_up
        bot = self._setup(base_close=0.92)
        with patch.object(bot, "_close_trend_trade", new_callable=AsyncMock) as mock_close:
            await bot._evaluate_trend(0.97)  # 0.97 < 0.988 but trending_up
        mock_close.assert_called_once_with(0.97, reason="reversal")


# ===========================================================================
# TestConfirmCounter — step 3 confirmation accumulation
# ===========================================================================


class TestConfirmCounter:
    """Confirmation counter must accumulate correctly before trend fires."""

    def _neutral_bot(self) -> GridTradingBot:
        """Bot with clean state; no trend position."""
        bot = _make_bot()
        bot.candle_buffer = _seeded_buffer(n=15, base_close=1.0)
        bot.trend_mode = None
        bot.trend_position = None
        bot.trend_pending_dir = None
        bot.trend_confirm_counter = 0
        bot.short_position = 0.0
        bot.long_position = 0.0
        bot.latest_signals = Signals(adx=30.0)
        return bot

    @pytest.mark.asyncio
    async def test_increments_on_consecutive_up(self):
        """Two consecutive trending_up calls → counter=2."""
        bot = self._neutral_bot()
        # price=1.08 → velocity ≈ 0.075 > 0.04 → trending_up
        await bot._evaluate_trend(1.08)
        assert bot.trend_confirm_counter == 1
        assert bot.trend_pending_dir == "up"
        _append_candle(bot.candle_buffer, 1.08, idx=15)
        await bot._evaluate_trend(1.08)
        assert bot.trend_confirm_counter == 2

    @pytest.mark.asyncio
    async def test_increments_on_consecutive_down(self):
        """Two consecutive trending_down calls → counter=2."""
        bot = self._neutral_bot()
        await bot._evaluate_trend(0.92)
        assert bot.trend_confirm_counter == 1
        assert bot.trend_pending_dir == "down"
        _append_candle(bot.candle_buffer, 0.92, idx=15)
        await bot._evaluate_trend(0.92)
        assert bot.trend_confirm_counter == 2

    @pytest.mark.asyncio
    async def test_direction_change_resets_counter(self):
        """Switching from up to down resets counter to 1."""
        bot = self._neutral_bot()
        await bot._evaluate_trend(1.08)  # trending_up  → counter=1
        _append_candle(bot.candle_buffer, 1.08, idx=15)
        await bot._evaluate_trend(0.92)  # trending_down → dir changes, counter=1
        assert bot.trend_confirm_counter == 1
        assert bot.trend_pending_dir == "down"

    @pytest.mark.asyncio
    async def test_neutral_velocity_resets_counter(self):
        """Neutral velocity → counter=0, pending_dir=None."""
        bot = self._neutral_bot()
        await bot._evaluate_trend(1.08)   # counter=1, dir="up"
        _append_candle(bot.candle_buffer, 1.08, idx=15)
        await bot._evaluate_trend(1.01)   # neutral → counter=0
        assert bot.trend_confirm_counter == 0
        assert bot.trend_pending_dir is None

    @pytest.mark.asyncio
    async def test_confirmation_threshold_sets_mode(self):
        """Counter reaches TREND_CONFIRM_CANDLES → trend_mode set."""
        bot = self._neutral_bot()
        bot.trend_pending_dir = "up"
        bot.trend_confirm_counter = TREND_CONFIRM_CANDLES - 1
        # Set ADX below min to prevent capture from opening (isolate counter test)
        bot.latest_signals = Signals(adx=0.0)
        await bot._evaluate_trend(1.08)
        assert bot.trend_mode == "up"
        assert bot.trend_confirm_counter >= TREND_CONFIRM_CANDLES


# ===========================================================================
# TestForceCloseGrid — step 4a force-close opposing grid
# ===========================================================================


class TestForceCloseGrid:
    """On confirmed trend, the opposing grid side should be force-closed."""

    def _confirmed_up_bot(self) -> GridTradingBot:
        """Bot wired for confirmed-up on next call with price=1.08."""
        bot = _make_bot()
        bot.candle_buffer = _seeded_buffer(n=15, base_close=1.0)
        bot.trend_pending_dir = "up"
        bot.trend_confirm_counter = TREND_CONFIRM_CANDLES - 1
        bot.trend_mode = None
        bot.trend_position = None
        bot.short_position = 5.7   # opposing position to force-close
        bot.latest_signals = Signals(adx=0.0)  # ADX=0 prevents capture opening
        return bot

    def _confirmed_down_bot(self) -> GridTradingBot:
        """Bot wired for confirmed-down on next call with price=0.92."""
        bot = _make_bot()
        bot.candle_buffer = _seeded_buffer(n=15, base_close=1.0)
        bot.trend_pending_dir = "down"
        bot.trend_confirm_counter = TREND_CONFIRM_CANDLES - 1
        bot.trend_mode = None
        bot.trend_position = None
        bot.long_position = 8.3   # opposing position
        bot.latest_signals = Signals(adx=0.0)
        return bot

    @pytest.mark.asyncio
    async def test_force_close_short_on_confirmed_up(self):
        """Confirmed up + short_position > 0 → cancel + market buy."""
        bot = self._confirmed_up_bot()
        bot.cancel_orders_for_side = AsyncMock()
        bot.place_market_order = AsyncMock(return_value={"orderId": "x"})
        await bot._evaluate_trend(1.08)
        bot.cancel_orders_for_side.assert_called_once_with("XRPUSDT", "short")
        bot.place_market_order.assert_called_once_with(
            symbol="XRPUSDT",
            side="buy",
            quantity=5,          # int(5.7) = 5
            reduce_only=True,
            position_side="short",
        )

    @pytest.mark.asyncio
    async def test_force_close_long_on_confirmed_down(self):
        """Confirmed down + long_position > 0 → cancel + market sell."""
        bot = self._confirmed_down_bot()
        bot.cancel_orders_for_side = AsyncMock()
        bot.place_market_order = AsyncMock(return_value={"orderId": "x"})
        await bot._evaluate_trend(0.92)
        bot.cancel_orders_for_side.assert_called_once_with("XRPUSDT", "long")
        bot.place_market_order.assert_called_once_with(
            symbol="XRPUSDT",
            side="sell",
            quantity=8,          # int(8.3) = 8
            reduce_only=True,
            position_side="long",
        )

    @pytest.mark.asyncio
    async def test_skip_when_flag_false(self):
        """TREND_FORCE_CLOSE_GRID=False → no force-close orders."""
        bot = self._confirmed_up_bot()
        bot.cancel_orders_for_side = AsyncMock()
        bot.place_market_order = AsyncMock()
        with patch("src.single_bot.bitunix_bot.TREND_FORCE_CLOSE_GRID", False):
            await bot._evaluate_trend(1.08)
        bot.cancel_orders_for_side.assert_not_called()
        bot.place_market_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_when_no_opposing_position(self):
        """No opposing position → force-close block skipped entirely."""
        bot = self._confirmed_up_bot()
        bot.short_position = 0.0
        bot.cancel_orders_for_side = AsyncMock()
        bot.place_market_order = AsyncMock()
        await bot._evaluate_trend(1.08)
        bot.cancel_orders_for_side.assert_not_called()
        bot.place_market_order.assert_not_called()


# ===========================================================================
# TestTrendCooldown — step 5 cooldown after trend fades
# ===========================================================================


class TestTrendCooldown:
    """Cooldown should count down and reset trend state when velocity fades."""

    def _cooldown_bot(self, counter: int = 0) -> GridTradingBot:
        """Bot in trend_mode='up' with no position → ready for cooldown."""
        bot = _make_bot()
        bot.candle_buffer = _seeded_buffer(n=15, base_close=1.0)
        bot.trend_mode = "up"
        bot.trend_position = None
        bot.trend_cooldown_counter = counter
        bot.trend_pending_dir = "up"
        bot.trend_confirm_counter = 2
        bot.latest_signals = Signals(adx=20.0)
        return bot

    @pytest.mark.asyncio
    async def test_cooldown_increments_when_velocity_low(self):
        """Velocity < 0.5 × TREND_VELOCITY_PCT → cooldown_counter += 1."""
        bot = self._cooldown_bot(counter=0)
        # price=1.01 → velocity ≈ 0.005 < 0.02 → low velocity
        await bot._evaluate_trend(1.01)
        assert bot.trend_cooldown_counter == 1
        assert bot.trend_mode == "up"  # not yet reset

    @pytest.mark.asyncio
    async def test_cooldown_resets_mode_after_enough_candles(self):
        """Counter reaches TREND_COOLDOWN_CANDLES → full state reset."""
        bot = self._cooldown_bot(counter=TREND_COOLDOWN_CANDLES - 1)
        await bot._evaluate_trend(1.01)
        assert bot.trend_mode is None
        assert bot.trend_cooldown_counter == 0
        assert bot.trend_pending_dir is None
        assert bot.trend_confirm_counter == 0

    @pytest.mark.asyncio
    async def test_cooldown_resets_counter_on_velocity_rebound(self):
        """Velocity rebounds above 0.5 × threshold → counter reset to 0."""
        bot = self._cooldown_bot(counter=5)
        # price=1.035 → velocity ≈ 0.030 ≥ 0.02 → rebound (but not trending_up)
        await bot._evaluate_trend(1.035)
        assert bot.trend_cooldown_counter == 0
        assert bot.trend_mode == "up"  # still active, just reset counter

    @pytest.mark.asyncio
    async def test_no_cooldown_when_position_open(self):
        """Trend position still open → cooldown block never fires."""
        bot = self._cooldown_bot(counter=5)
        bot.trend_position = {
            "side": "long", "entry": 1.0, "qty": 10, "peak": 1.02,
        }
        with patch.object(bot, "_close_trend_trade", new_callable=AsyncMock):
            await bot._evaluate_trend(1.01)
        # cooldown_counter should be unchanged since step 5 condition fails
        assert bot.trend_cooldown_counter == 5


# ===========================================================================
# TestEvaluateTrendGuards — early return conditions
# ===========================================================================


class TestEvaluateTrendGuards:
    """Guard conditions at the top of _evaluate_trend."""

    @pytest.mark.asyncio
    async def test_returns_early_when_buffer_too_small(self):
        """< TREND_LOOKBACK_CANDLES closed candles → no state changes."""
        bot = _make_bot()
        bot.candle_buffer = _seeded_buffer(
            n=TREND_LOOKBACK_CANDLES - 1, base_close=1.0,
        )
        bot.trend_pending_dir = None
        await bot._evaluate_trend(1.08)
        assert bot.trend_pending_dir is None  # no change

    @pytest.mark.asyncio
    async def test_returns_early_when_past_price_zero(self):
        """Oldest relevant candle close ≤ 0 → returns early."""
        bot = _make_bot()
        bot.candle_buffer = _seeded_buffer(n=15, base_close=1.0)
        # Overwrite the lookback candle to have close=0
        idx = len(bot.candle_buffer._closed) - TREND_LOOKBACK_CANDLES
        old = bot.candle_buffer._closed[idx]
        bot.candle_buffer._closed[idx] = Candle(
            ts=old.ts, open=0, high=0, low=0, close=0, volume=0,
        )
        bot.trend_pending_dir = None
        await bot._evaluate_trend(1.08)
        assert bot.trend_pending_dir is None

    @pytest.mark.asyncio
    async def test_confirmed_up_does_not_refire(self):
        """Already in trend_mode='up' → confirmed_up block skipped."""
        bot = _make_bot()
        bot.candle_buffer = _seeded_buffer(n=15, base_close=1.0)
        bot.trend_mode = "up"
        bot.trend_pending_dir = "up"
        bot.trend_confirm_counter = TREND_CONFIRM_CANDLES - 1
        bot.trend_position = None
        bot.short_position = 0.0
        bot.latest_signals = Signals(adx=30.0)
        # This call will push counter to CONFIRM_CANDLES → confirmed_up=True
        # but trend_mode == "up" already → block skipped
        bot.cancel_orders_for_side = AsyncMock()
        await bot._evaluate_trend(1.08)
        bot.cancel_orders_for_side.assert_not_called()


# ===========================================================================
# TestEvaluateTrendEndToEnd — full state machine lifecycle
# ===========================================================================


class TestEvaluateTrendEndToEnd:
    """Multi-call sequence covering detect → enter → trail → close → reset."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_up(self):
        """Long lifecycle: confirm → capture → peak track → trail close."""
        bot = _make_bot()
        bot.balance = {"USDT": {"available": 1000.0, "margin": 0.0}}
        bot.candle_buffer = _seeded_buffer(n=15, base_close=1.0)
        bot.short_position = 5.0
        bot.latest_signals = Signals(adx=30.0)
        bot.place_market_order = AsyncMock(return_value={"orderId": "x"})
        bot.cancel_orders_for_side = AsyncMock()
        bot.send_telegram_message = AsyncMock()

        # Phase 1 — build confirmation (3 trending_up calls)
        for i in range(TREND_CONFIRM_CANDLES):
            price = 1.08 + i * 0.01
            await bot._evaluate_trend(price)
            _append_candle(bot.candle_buffer, price, idx=15 + i)

        assert bot.trend_mode == "up"
        assert bot.trend_position is not None
        assert bot.trend_position["side"] == "long"
        bot.cancel_orders_for_side.assert_called()  # force-close fired

        entry = bot.trend_position["entry"]

        # Phase 2 — price rises, peak tracks
        higher = entry + 0.10
        _append_candle(bot.candle_buffer, higher, idx=18)
        await bot._evaluate_trend(higher)
        assert bot.trend_position["peak"] == higher

        # Phase 3 — price drops to trail stop → close
        trail_stop = higher * (1.0 - TREND_TRAIL_PCT)
        close_price = trail_stop - 0.01
        _append_candle(bot.candle_buffer, close_price, idx=19)
        await bot._evaluate_trend(close_price)

        # TREND_REENTRY_FAST=True → mode resets immediately
        assert bot.trend_position is None
        assert bot.trend_mode is None

    @pytest.mark.asyncio
    async def test_full_lifecycle_down(self):
        """Short lifecycle: confirm → capture → peak track → trail close."""
        bot = _make_bot()
        bot.balance = {"USDT": {"available": 1000.0, "margin": 0.0}}
        bot.candle_buffer = _seeded_buffer(n=15, base_close=1.0)
        bot.long_position = 5.0
        bot.latest_signals = Signals(adx=30.0)
        bot.place_market_order = AsyncMock(return_value={"orderId": "x"})
        bot.cancel_orders_for_side = AsyncMock()
        bot.send_telegram_message = AsyncMock()

        # Phase 1 — build confirmation (3 trending_down calls)
        for i in range(TREND_CONFIRM_CANDLES):
            price = 0.92 - i * 0.01
            await bot._evaluate_trend(price)
            _append_candle(bot.candle_buffer, price, idx=15 + i)

        assert bot.trend_mode == "down"
        assert bot.trend_position is not None
        assert bot.trend_position["side"] == "short"

        entry = bot.trend_position["entry"]

        # Phase 2 — price drops further, peak tracks
        lower = entry - 0.10
        _append_candle(bot.candle_buffer, lower, idx=18)
        await bot._evaluate_trend(lower)
        assert bot.trend_position["peak"] == lower

        # Phase 3 — price rises to trail stop → close
        trail_stop = lower * (1.0 + TREND_TRAIL_PCT)
        close_price = trail_stop + 0.01
        _append_candle(bot.candle_buffer, close_price, idx=19)
        await bot._evaluate_trend(close_price)

        assert bot.trend_position is None
        assert bot.trend_mode is None


# ===========================================================================
# TestDirVelScaling — L5/L5b dynamic velocity threshold
# ===========================================================================


class TestDirVelScaling:
    """
    Tests for the directional velocity scaling mechanism (L5/L5b) that
    dynamically adjusts TREND_VELOCITY_PCT and TREND_CAP_VEL_PCT using
    the ATR/SMA ratio, optionally gated by bearish context (vel_dir_only).
    """

    def _dirvel_bot(
        self,
        vel_atr_mult: float = 0.0,
        vel_dir_only: bool = False,
        atr: float = 0.01,
        atr_sma: float = 0.01,
        htf_ema_fast: float = 1.0,
        base_close: float = 1.0,
    ) -> GridTradingBot:
        """Create a bot with dirvel params and appropriate signals."""
        bot = _make_bot()
        bot.vel_atr_mult = vel_atr_mult
        bot.vel_dir_only = vel_dir_only
        bot.candle_buffer = _seeded_buffer(n=15, base_close=base_close)
        bot.trend_pending_dir = None
        bot.trend_confirm_counter = 0
        bot.latest_signals = Signals(
            adx=30.0,
            atr=atr,
            atr_sma=atr_sma,
            htf_ema_fast=htf_ema_fast,
        )
        return bot

    # -- Defaults off (vel_atr_mult=0) → static thresholds -----------------

    @pytest.mark.asyncio
    async def test_mult_zero_uses_static_threshold(self):
        """vel_atr_mult=0.0 → static TREND_VELOCITY_PCT, no scaling."""
        bot = self._dirvel_bot(vel_atr_mult=0.0, atr=0.10, atr_sma=0.01)
        # Velocity ≈ 4.5% which is > TREND_VELOCITY_PCT (4%)
        # With static threshold, this should register as trending_up
        price = 1.0 + TREND_LOOKBACK_CANDLES * 0.001 + TREND_VELOCITY_PCT + 0.005
        await bot._evaluate_trend(price)
        assert bot.trend_pending_dir == "up"

    @pytest.mark.asyncio
    async def test_mult_zero_no_scaling_even_with_high_atr(self):
        """Even with ATR/SMA ratio of 10×, mult=0 means no scaling."""
        bot = self._dirvel_bot(vel_atr_mult=0.0, atr=0.10, atr_sma=0.01)
        # Price gives velocity ≈ 5% (just above 4% static threshold)
        price = 1.05 + TREND_LOOKBACK_CANDLES * 0.001
        await bot._evaluate_trend(price)
        assert bot.trend_pending_dir == "up"

    # -- Scaling active (vel_atr_mult > 0, no directional filter) ----------

    @pytest.mark.asyncio
    async def test_scaling_raises_threshold(self):
        """vel_atr_mult=1.0 with ATR/SMA=3.0 → threshold*3, blocking a 5% move."""
        bot = self._dirvel_bot(
            vel_atr_mult=1.0,
            vel_dir_only=False,
            atr=0.03,         # 3× the SMA
            atr_sma=0.01,
        )
        # Velocity ≈ 5%: above static 4% but below scaled 12% (4% × 3.0)
        price = 1.05 + TREND_LOOKBACK_CANDLES * 0.001
        await bot._evaluate_trend(price)
        # Should NOT be trending — threshold scaled up
        assert bot.trend_pending_dir is None

    @pytest.mark.asyncio
    async def test_scaling_allows_huge_move(self):
        """A 15% move exceeds even a 3× scaled threshold (12%)."""
        bot = self._dirvel_bot(
            vel_atr_mult=1.0,
            vel_dir_only=False,
            atr=0.03,
            atr_sma=0.01,
        )
        price = 1.15 + TREND_LOOKBACK_CANDLES * 0.001
        await bot._evaluate_trend(price)
        assert bot.trend_pending_dir == "up"

    @pytest.mark.asyncio
    async def test_scaling_floor_at_one(self):
        """ATR/SMA < 1 → scale clamped at 1.0, threshold unchanged."""
        bot = self._dirvel_bot(
            vel_atr_mult=1.0,
            vel_dir_only=False,
            atr=0.005,        # below SMA
            atr_sma=0.01,
        )
        # 5% velocity, above static 4% threshold
        price = 1.05 + TREND_LOOKBACK_CANDLES * 0.001
        await bot._evaluate_trend(price)
        assert bot.trend_pending_dir == "up"

    # -- Directional filter (L5b: vel_dir_only) ----------------------------

    @pytest.mark.asyncio
    async def test_dir_only_bullish_skips_scaling(self):
        """price ≥ EMA-36 (bullish) → scaling NOT applied, static threshold."""
        bot = self._dirvel_bot(
            vel_atr_mult=1.0,
            vel_dir_only=True,
            atr=0.03,
            atr_sma=0.01,
            htf_ema_fast=1.0,  # EMA at 1.0 — price will be above
        )
        # 5% velocity, above static 4%, below scaled 12%
        # Since price ≥ EMA → no scaling → should trend
        price = 1.05 + TREND_LOOKBACK_CANDLES * 0.001
        await bot._evaluate_trend(price)
        assert bot.trend_pending_dir == "up"

    @pytest.mark.asyncio
    async def test_dir_only_bearish_applies_scaling(self):
        """price < EMA-36 (bearish) → scaling applied, threshold raised."""
        bot = self._dirvel_bot(
            vel_atr_mult=1.0,
            vel_dir_only=True,
            atr=0.03,
            atr_sma=0.01,
            htf_ema_fast=2.0,  # EMA at 2.0 — price (≈1.05) well below
        )
        # 5% velocity, above static 4% but below scaled 12%
        price = 1.05 + TREND_LOOKBACK_CANDLES * 0.001
        await bot._evaluate_trend(price)
        # Bearish context → scaling applied → threshold raised → NOT trending
        assert bot.trend_pending_dir is None

    @pytest.mark.asyncio
    async def test_dir_only_bearish_huge_move_still_trends(self):
        """Even in bearish context, a sufficiently large move trends."""
        bot = self._dirvel_bot(
            vel_atr_mult=1.0,
            vel_dir_only=True,
            atr=0.03,
            atr_sma=0.01,
            htf_ema_fast=2.0,  # bearish
        )
        # 15% velocity exceeds 12% scaled threshold
        price = 1.15 + TREND_LOOKBACK_CANDLES * 0.001
        await bot._evaluate_trend(price)
        assert bot.trend_pending_dir == "up"

    # -- Cap velocity also scaled ------------------------------------------

    @pytest.mark.asyncio
    async def test_cap_velocity_scaled_blocks_entry(self):
        """Scaled cap threshold blocks trend capture entry."""
        bot = self._dirvel_bot(
            vel_atr_mult=1.0,
            vel_dir_only=False,
            atr=0.03,         # 3× SMA
            atr_sma=0.01,
        )
        bot.balance = {"USDT": {"available": 1000.0, "margin": 0.0}}
        bot.short_position = 0.0
        bot.place_market_order = AsyncMock(return_value={"orderId": "x"})
        bot.cancel_orders_for_side = AsyncMock()
        bot.send_telegram_message = AsyncMock()

        # Build up to confirmation threshold with very large velocity (15%)
        # to exceed the scaled vel threshold (12%) and register trending
        for i in range(TREND_CONFIRM_CANDLES):
            price = 1.15 + i * 0.01 + TREND_LOOKBACK_CANDLES * 0.001
            _append_candle(bot.candle_buffer, price, idx=15 + i)
            await bot._evaluate_trend(price)

        # trend_mode should flip — the velocity was 15% > scaled 12%
        assert bot.trend_mode == "up"

        # But capture position depends on velocity ≥ effective_cap_pct
        # effective_cap = 0.06 × 3.0 = 0.18 (18%)
        # velocity is only ≈ 15% → capture should NOT open
        assert bot.trend_position is None

    # -- Edge cases --------------------------------------------------------

    @pytest.mark.asyncio
    async def test_no_signals_uses_static(self):
        """latest_signals=None → fallback to static thresholds."""
        bot = self._dirvel_bot(vel_atr_mult=1.0, vel_dir_only=True)
        bot.latest_signals = None
        # 5% velocity above static 4% threshold
        price = 1.05 + TREND_LOOKBACK_CANDLES * 0.001
        await bot._evaluate_trend(price)
        assert bot.trend_pending_dir == "up"

    @pytest.mark.asyncio
    async def test_zero_atr_sma_uses_static(self):
        """atr_sma=0 → division guard, static thresholds used."""
        bot = self._dirvel_bot(
            vel_atr_mult=1.0, atr=0.03, atr_sma=0.0,
        )
        # 5% velocity above static 4% threshold
        price = 1.05 + TREND_LOOKBACK_CANDLES * 0.001
        await bot._evaluate_trend(price)
        assert bot.trend_pending_dir == "up"

    @pytest.mark.asyncio
    async def test_zero_ema_fast_skips_dir_filter(self):
        """htf_ema_fast=0 → directional filter skipped (EMA not warm)."""
        bot = self._dirvel_bot(
            vel_atr_mult=1.0,
            vel_dir_only=True,
            atr=0.03,
            atr_sma=0.01,
            htf_ema_fast=0.0,  # EMA not computed yet
        )
        # 5% velocity, above static 4% but below scaled 12%
        # Without a valid EMA, directional filter can't activate →
        # scaling still applies (conservative: scale when unsure)
        price = 1.05 + TREND_LOOKBACK_CANDLES * 0.001
        await bot._evaluate_trend(price)
        assert bot.trend_pending_dir is None  # scaled threshold blocks it

    @pytest.mark.asyncio
    async def test_cooldown_uses_effective_threshold(self):
        """Cooldown velocity check uses effective (scaled) threshold."""
        bot = self._dirvel_bot(
            vel_atr_mult=1.0,
            vel_dir_only=False,
            atr=0.03,         # 3× SMA → effective_vel_pct ≈ 0.12
            atr_sma=0.01,
        )
        bot.trend_mode = "up"
        bot.trend_position = None
        bot.trend_cooldown_counter = 0

        # velocity ≈ 4% which is < 0.5 × static 4% = 2%?  No — 4% > 2%
        # BUT with scaling: effective_vel_pct ≈ 12%, so 0.5 × 12% = 6%
        # 4% velocity < 6% → cooldown should increment (scaled threshold)
        price = 1.04 + TREND_LOOKBACK_CANDLES * 0.001
        await bot._evaluate_trend(price)
        assert bot.trend_cooldown_counter == 1

    @pytest.mark.asyncio
    async def test_trending_down_also_scaled(self):
        """Downward velocity threshold is also scaled symmetrically."""
        bot = self._dirvel_bot(
            vel_atr_mult=1.0,
            vel_dir_only=False,
            atr=0.03,
            atr_sma=0.01,
            base_close=1.5,   # start higher so we can drop 5%
        )
        # 5% drop, above static -4% but below -12% scaled
        past = bot.candle_buffer._closed[-TREND_LOOKBACK_CANDLES].close
        price = past * 0.95   # -5% velocity
        await bot._evaluate_trend(price)
        # -5% is past static -4% threshold but not past scaled -12%
        assert bot.trend_pending_dir is None

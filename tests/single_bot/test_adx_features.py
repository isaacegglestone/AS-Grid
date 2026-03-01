"""
tests/single_bot/test_adx_features.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the v10 ADX-gate features added to ``GridTradingBot``:

1. **ADX grid pause** (``place_long_orders`` / ``place_short_orders``)
   - Returns early (skipping grid refresh) when ``latest_signals.adx >= ADX_GRID_PAUSE``
   - Proceeds normally when ADX is below the threshold
   - Proceeds normally when ``latest_signals`` is ``None``

2. **ADX minimum trend gate** (``_evaluate_trend``)
   - Does NOT open a capture position when ADX < ``ADX_MIN_TREND``
   - DOES open a capture position when ADX >= ``ADX_MIN_TREND``
   - Gate applies for both long (confirmed_up) and short (confirmed_down) directions

3. **Fast re-entry reset** (``_close_trend_trade``)
   - After a successful close, ``trend_mode``, ``trend_confirm_counter`` and
     ``trend_pending_dir`` are all reset
   - When ``TREND_REENTRY_FAST=False``, those fields are preserved
"""
from __future__ import annotations

from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.single_bot.bitunix_bot import (
    ADX_GRID_PAUSE,
    ADX_MIN_TREND,
    TREND_CAP_VEL_PCT,
    TREND_CONFIRM_CANDLES,
    TREND_LOOKBACK_CANDLES,
    TREND_VELOCITY_PCT,
    GridTradingBot,
)
from src.single_bot.indicators import CandleBuffer, Candle, Signals

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

TS_BASE = 1_740_000_000_000
TS_STEP = 15 * 60 * 1_000


def _make_bot() -> GridTradingBot:
    return GridTradingBot(
        api_key="k",
        api_secret="s",
        coin_name="XRP",
        grid_spacing=0.01,
        initial_quantity=10,
        leverage=5,
    )


def _signals(adx: float) -> Signals:
    """Return a minimal ``Signals`` object with the specified ADX value.

    All other fields use their dataclass defaults; tests only care about ``adx``.
    """
    return Signals(adx=adx)


def _seeded_buffer(n: int = 15, base_close: float = 1.0) -> CandleBuffer:
    """
    Return a CandleBuffer with *n* closed candles.

    The oldest candle (index 0) has ``close = base_close``; subsequent candles
    increase by 0.001 each step.  This gives the caller control over the
    velocity calculation:

        velocity = (price_passed_to_evaluate - closed[-LOOKBACK].close) / closed[-LOOKBACK].close

    With ``base_close=1.0`` and passing ``price=1.08``, velocity = 0.08 which
    is above both ``TREND_VELOCITY_PCT`` and ``TREND_CAP_VEL_PCT``.
    """
    buf = CandleBuffer(maxlen=200, interval="15min")
    for i in range(n):
        buf._closed.append(
            Candle(
                ts=TS_BASE + i * TS_STEP,
                open=base_close + i * 0.001,
                high=base_close + i * 0.001 + 0.05,
                low=base_close + i * 0.001 - 0.05,
                close=base_close + i * 0.001,
                volume=1_000_000.0,
            )
        )
    return buf


# ---------------------------------------------------------------------------
# 1. ADX grid pause — place_long_orders / place_short_orders
# ---------------------------------------------------------------------------

class TestAdxGridPauseLong:
    """place_long_orders should skip the grid refresh when ADX >= ADX_GRID_PAUSE."""

    @pytest.mark.asyncio
    async def test_returns_early_when_adx_at_pause_threshold(self):
        """Exactly at the threshold (adx == ADX_GRID_PAUSE) → returns early."""
        bot = _make_bot()
        bot.latest_signals = _signals(adx=ADX_GRID_PAUSE)
        bot.get_take_profit_quantity = MagicMock()

        await bot.place_long_orders(1.0)

        bot.get_take_profit_quantity.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_early_when_adx_above_pause_threshold(self):
        """ADX well above threshold → returns early."""
        bot = _make_bot()
        bot.latest_signals = _signals(adx=ADX_GRID_PAUSE + 10.0)
        bot.get_take_profit_quantity = MagicMock()

        await bot.place_long_orders(1.0)

        bot.get_take_profit_quantity.assert_not_called()

    @pytest.mark.asyncio
    async def test_proceeds_when_adx_below_pause_threshold(self):
        """ADX below threshold → grid refresh proceeds (get_take_profit_quantity is called)."""
        bot = _make_bot()
        bot.latest_signals = _signals(adx=ADX_GRID_PAUSE - 1.0)
        bot.get_take_profit_quantity = MagicMock()

        await bot.place_long_orders(1.0)

        bot.get_take_profit_quantity.assert_called_once()

    @pytest.mark.asyncio
    async def test_proceeds_when_latest_signals_is_none(self):
        """No signals yet (cold start) → guard doesn't fire, grid proceeds."""
        bot = _make_bot()
        bot.latest_signals = None
        bot.get_take_profit_quantity = MagicMock()

        await bot.place_long_orders(1.0)

        bot.get_take_profit_quantity.assert_called_once()


class TestAdxGridPauseShort:
    """place_short_orders should mirror the same ADX gate behaviour."""

    @pytest.mark.asyncio
    async def test_returns_early_when_adx_at_pause_threshold(self):
        bot = _make_bot()
        bot.latest_signals = _signals(adx=ADX_GRID_PAUSE)
        bot.get_take_profit_quantity = MagicMock()

        await bot.place_short_orders(1.0)

        bot.get_take_profit_quantity.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_early_when_adx_above_pause_threshold(self):
        bot = _make_bot()
        bot.latest_signals = _signals(adx=ADX_GRID_PAUSE + 5.0)
        bot.get_take_profit_quantity = MagicMock()

        await bot.place_short_orders(1.0)

        bot.get_take_profit_quantity.assert_not_called()

    @pytest.mark.asyncio
    async def test_proceeds_when_adx_below_pause_threshold(self):
        bot = _make_bot()
        bot.latest_signals = _signals(adx=ADX_GRID_PAUSE - 0.1)
        bot.get_take_profit_quantity = MagicMock()

        await bot.place_short_orders(1.0)

        bot.get_take_profit_quantity.assert_called_once()

    @pytest.mark.asyncio
    async def test_proceeds_when_latest_signals_is_none(self):
        bot = _make_bot()
        bot.latest_signals = None
        bot.get_take_profit_quantity = MagicMock()

        await bot.place_short_orders(1.0)

        bot.get_take_profit_quantity.assert_called_once()


# ---------------------------------------------------------------------------
# 2. ADX minimum trend gate — _evaluate_trend
#
# Strategy: pre-seed the buffer so velocity = 0.08 (> VEL and CAP thresholds)
# and pre-set confirm counter to TREND_CONFIRM_CANDLES - 1 with the matching
# pending direction so a single evaluate call tips the counter to "confirmed".
# Then assert _open_trend_trade is/isn't called based on ADX level.
# ---------------------------------------------------------------------------

class TestAdxMinTrendLong:
    """confirmed_up scenario — open capture only when ADX >= ADX_MIN_TREND."""

    def _setup_bot_for_confirmed_up(self, adx: float) -> GridTradingBot:
        """
        Return a bot wired for a confirmed-up situation on the next
        ``_evaluate_trend`` call with price=1.08 and the given ADX.

        Buffer: 15 candles, oldest close=1.0.
        velocity = (1.08 - closed[-10].close) / closed[-10].close
        closed[-10] has index 5  (15 total, 10 from end = index 5, close=1.005)
        → velocity ≈ 0.074 > TREND_CAP_VEL_PCT(0.06) ✓
        """
        bot = _make_bot()
        bot.candle_buffer = _seeded_buffer(n=15, base_close=1.0)
        # Pre-warm confirm counter to CONFIRM_CANDLES - 1
        bot.trend_pending_dir = "up"
        bot.trend_confirm_counter = TREND_CONFIRM_CANDLES - 1
        bot.trend_mode = None
        bot.trend_position = None
        bot.latest_signals = _signals(adx=adx)
        # Suppress grid force-close side effects
        bot.short_position = 0.0
        return bot

    @pytest.mark.asyncio
    async def test_no_capture_when_adx_below_min(self):
        """ADX below ADX_MIN_TREND → _open_trend_trade not called."""
        bot = self._setup_bot_for_confirmed_up(adx=ADX_MIN_TREND - 1.0)

        with patch.object(bot, "_open_trend_trade", new_callable=AsyncMock) as mock_open:
            await bot._evaluate_trend(price=1.08)

        mock_open.assert_not_called()

    @pytest.mark.asyncio
    async def test_capture_opens_when_adx_meets_min(self):
        """ADX exactly at ADX_MIN_TREND → _open_trend_trade called with 'long'."""
        bot = self._setup_bot_for_confirmed_up(adx=ADX_MIN_TREND)

        with patch.object(bot, "_open_trend_trade", new_callable=AsyncMock) as mock_open:
            await bot._evaluate_trend(price=1.08)

        mock_open.assert_called_once_with("long", 1.08)

    @pytest.mark.asyncio
    async def test_capture_opens_when_adx_above_min(self):
        """ADX well above ADX_MIN_TREND → _open_trend_trade called."""
        bot = self._setup_bot_for_confirmed_up(adx=ADX_MIN_TREND + 15.0)

        with patch.object(bot, "_open_trend_trade", new_callable=AsyncMock) as mock_open:
            await bot._evaluate_trend(price=1.08)

        mock_open.assert_called_once_with("long", 1.08)

    @pytest.mark.asyncio
    async def test_no_capture_when_signals_none(self):
        """No signals (adx defaults to 0.0) → ADX_MIN_TREND not met → no capture."""
        bot = self._setup_bot_for_confirmed_up(adx=ADX_MIN_TREND)
        bot.latest_signals = None  # overrides the fixture

        with patch.object(bot, "_open_trend_trade", new_callable=AsyncMock) as mock_open:
            await bot._evaluate_trend(price=1.08)

        mock_open.assert_not_called()


class TestAdxMinTrendShort:
    """confirmed_down scenario — open short capture only when ADX >= ADX_MIN_TREND."""

    def _setup_bot_for_confirmed_down(self, adx: float) -> GridTradingBot:
        """
        Return a bot wired for a confirmed-down situation on the next call
        with price=0.92.

        Buffer: 15 candles, oldest close=1.0.
        closed[-10] is the 6th candle (index 5), close ≈ 1.005.
        velocity = (0.92 - 1.005) / 1.005 ≈ -0.085 < -TREND_CAP_VEL_PCT(-0.06) ✓
        """
        bot = _make_bot()
        bot.candle_buffer = _seeded_buffer(n=15, base_close=1.0)
        bot.trend_pending_dir = "down"
        bot.trend_confirm_counter = TREND_CONFIRM_CANDLES - 1
        bot.trend_mode = None
        bot.trend_position = None
        bot.latest_signals = _signals(adx=adx)
        bot.long_position = 0.0
        return bot

    @pytest.mark.asyncio
    async def test_no_capture_when_adx_below_min(self):
        bot = self._setup_bot_for_confirmed_down(adx=ADX_MIN_TREND - 1.0)

        with patch.object(bot, "_open_trend_trade", new_callable=AsyncMock) as mock_open:
            await bot._evaluate_trend(price=0.92)

        mock_open.assert_not_called()

    @pytest.mark.asyncio
    async def test_capture_opens_when_adx_meets_min(self):
        bot = self._setup_bot_for_confirmed_down(adx=ADX_MIN_TREND)

        with patch.object(bot, "_open_trend_trade", new_callable=AsyncMock) as mock_open:
            await bot._evaluate_trend(price=0.92)

        mock_open.assert_called_once_with("short", 0.92)

    @pytest.mark.asyncio
    async def test_capture_opens_when_adx_above_min(self):
        bot = self._setup_bot_for_confirmed_down(adx=ADX_MIN_TREND + 10.0)

        with patch.object(bot, "_open_trend_trade", new_callable=AsyncMock) as mock_open:
            await bot._evaluate_trend(price=0.92)

        mock_open.assert_called_once_with("short", 0.92)


# ---------------------------------------------------------------------------
# 3. Fast re-entry reset — _close_trend_trade
# ---------------------------------------------------------------------------

class TestFastReentry:
    """_close_trend_trade should reset pending-direction state when TREND_REENTRY_FAST=True."""

    def _bot_with_open_long(self) -> GridTradingBot:
        """Return a bot that has an open long trend position."""
        bot = _make_bot()
        bot.trend_position = {
            "side": "long",
            "qty": 10,
            "entry": 1.00,
            "peak": 1.10,
        }
        bot.trend_mode            = "up"
        bot.trend_confirm_counter = 5
        bot.trend_pending_dir     = "up"
        return bot

    @pytest.mark.asyncio
    async def test_fast_reentry_resets_all_state_on_successful_close(self):
        """After a successful market close, confirm/direction state is wiped."""
        bot = self._bot_with_open_long()
        bot.place_market_order    = AsyncMock(return_value={"orderId": "99"})
        bot.send_telegram_message = AsyncMock()

        await bot._close_trend_trade(price=1.09, reason="trail")

        assert bot.trend_position       is None
        assert bot.trend_mode           is None
        assert bot.trend_confirm_counter == 0
        assert bot.trend_pending_dir    is None

    @pytest.mark.asyncio
    async def test_fast_reentry_disabled_preserves_confirm_state(self):
        """When TREND_REENTRY_FAST=False, direction/counter are NOT reset."""
        bot = self._bot_with_open_long()
        bot.place_market_order    = AsyncMock(return_value={"orderId": "99"})
        bot.send_telegram_message = AsyncMock()

        with patch("src.single_bot.bitunix_bot.TREND_REENTRY_FAST", False):
            await bot._close_trend_trade(price=1.09, reason="trail")

        assert bot.trend_position       is None     # always cleared
        assert bot.trend_mode           == "up"     # NOT reset when disabled
        assert bot.trend_confirm_counter == 5       # NOT reset when disabled
        assert bot.trend_pending_dir    == "up"     # NOT reset when disabled

    @pytest.mark.asyncio
    async def test_failed_close_does_not_reset_state(self):
        """If the market order fails (returns None), all state is preserved."""
        bot = self._bot_with_open_long()
        bot.place_market_order    = AsyncMock(return_value=None)
        bot.send_telegram_message = AsyncMock()

        await bot._close_trend_trade(price=1.09, reason="trail")

        assert bot.trend_position       is not None     # position kept
        assert bot.trend_mode           == "up"         # unchanged
        assert bot.trend_confirm_counter == 5           # unchanged
        assert bot.trend_pending_dir    == "up"         # unchanged

    @pytest.mark.asyncio
    async def test_close_with_no_position_is_noop(self):
        """Calling _close_trend_trade when no position is open is a silent no-op."""
        bot = _make_bot()
        bot.trend_position        = None
        bot.trend_mode            = "up"
        bot.trend_confirm_counter = 3
        bot.trend_pending_dir     = "up"
        bot.place_market_order    = AsyncMock()
        bot.send_telegram_message = AsyncMock()

        await bot._close_trend_trade(price=1.09)

        bot.place_market_order.assert_not_called()
        assert bot.trend_mode            == "up"    # never touched
        assert bot.trend_confirm_counter == 3

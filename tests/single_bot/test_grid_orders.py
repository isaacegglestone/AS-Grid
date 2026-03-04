"""
tests/single_bot/test_grid_orders.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MEDIUM-priority unit tests for grid order management:

1. ``handle_kline_update`` → ``_evaluate_trend`` wiring
2. Single-EMA regime filter path in ``place_long_orders``
3. Lockdown mode TP placement in ``place_long_orders`` / ``place_short_orders``
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.single_bot.bitunix_bot import (
    POSITION_THRESHOLD,
    REGIME_EMA_PERIOD,
    GridTradingBot,
)
from src.single_bot.indicators import Signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _kline_data(
    ts: int = 1_740_000_900_000,
    o: str = "1.0",
    h: str = "1.1",
    l: str = "0.9",
    c: str = "1.05",
    q: str = "1000000",
) -> dict:
    """Minimal kline WS payload."""
    return {"ts": ts, "data": {"o": o, "h": h, "l": l, "c": c, "q": q}}


def _grid_bot(
    adx: float = 10.0,
    regime_ema: float = 0.0,
    latest_price: float = 2.0,
    **overrides,
) -> GridTradingBot:
    """Bot pre-configured for place_long/short_orders tests."""
    bot = _make_bot(**overrides)
    bot.latest_signals = Signals(adx=adx, regime_ema=regime_ema, close=latest_price)
    bot.latest_price = latest_price
    bot.long_position = 1.0
    bot.short_position = 1.0
    bot.sell_long_orders = 0.0
    bot.buy_short_orders = 0.0
    bot.long_initial_quantity = 10
    bot.short_initial_quantity = 10
    bot.get_take_profit_quantity = MagicMock()
    bot.update_mid_price = MagicMock()
    bot.cancel_orders_for_side = AsyncMock()
    bot.place_take_profit_order = AsyncMock()
    bot.place_order = AsyncMock()
    bot.check_and_notify_position_threshold = AsyncMock()
    for k, v in overrides.items():
        setattr(bot, k, v)
    return bot


# ===========================================================================
# TestHandleKlineUpdateTrend — wiring from kline feed to _evaluate_trend
# ===========================================================================


class TestHandleKlineUpdateTrend:
    """handle_kline_update should call _evaluate_trend on candle close."""

    @pytest.mark.asyncio
    async def test_evaluate_trend_called_on_candle_close(self):
        """Candle closes + signals valid → _evaluate_trend called."""
        bot = _make_bot()
        bot.candle_buffer = MagicMock()
        bot.candle_buffer.update = MagicMock(return_value=True)
        bot.candle_buffer.signals = MagicMock(
            return_value=Signals(close=1.50, adx=20.0),
        )
        bot._evaluate_trend = AsyncMock()

        await bot.handle_kline_update(_kline_data())

        bot._evaluate_trend.assert_called_once_with(1.50)

    @pytest.mark.asyncio
    async def test_evaluate_trend_not_called_when_signals_none(self):
        """Candle closes but signals() returns None → no evaluation."""
        bot = _make_bot()
        bot.candle_buffer = MagicMock()
        bot.candle_buffer.update = MagicMock(return_value=True)
        bot.candle_buffer.signals = MagicMock(return_value=None)
        bot._evaluate_trend = AsyncMock()

        await bot.handle_kline_update(_kline_data())

        bot._evaluate_trend.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_trend_not_called_when_candle_not_closed(self):
        """Intra-candle update → no evaluation."""
        bot = _make_bot()
        bot.candle_buffer = MagicMock()
        bot.candle_buffer.update = MagicMock(return_value=False)
        bot._evaluate_trend = AsyncMock()

        await bot.handle_kline_update(_kline_data())

        bot._evaluate_trend.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_in_evaluate_trend_swallowed(self):
        """If _evaluate_trend raises, handle_kline_update catches it."""
        bot = _make_bot()
        bot.candle_buffer = MagicMock()
        bot.candle_buffer.update = MagicMock(return_value=True)
        bot.candle_buffer.signals = MagicMock(
            return_value=Signals(close=1.50, adx=20.0),
        )
        bot._evaluate_trend = AsyncMock(side_effect=ValueError("test error"))

        # Should NOT raise
        await bot.handle_kline_update(_kline_data())


# ===========================================================================
# TestRegimeFilterLong — single-EMA path in place_long_orders
# ===========================================================================


class TestRegimeFilterLong:
    """When regime_vote_mode=False, single-EMA regime filter applies."""

    @pytest.mark.asyncio
    async def test_halts_long_when_price_below_threshold(self):
        """Price < regime_ema × (1 − hyst) → long grid halted."""
        bot = _grid_bot(
            adx=10.0,
            regime_ema=2.0,
            latest_price=1.90,
        )
        bot.regime_vote_mode = False
        bot.regime_hysteresis_pct = 0.02
        # threshold = 2.0 × 0.98 = 1.96;  price 1.90 < 1.96 → halt
        await bot.place_long_orders(1.90)
        bot.get_take_profit_quantity.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_long_when_price_above_threshold(self):
        """Price ≥ regime threshold → long grid proceeds."""
        bot = _grid_bot(
            adx=10.0,
            regime_ema=2.0,
            latest_price=2.10,
        )
        bot.regime_vote_mode = False
        bot.regime_hysteresis_pct = 0.02
        # threshold = 1.96;  price 2.10 ≥ 1.96 → proceed
        await bot.place_long_orders(2.10)
        bot.get_take_profit_quantity.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_long_when_regime_ema_zero(self):
        """Regime EMA not warmed up (0.0) → filter skipped entirely."""
        bot = _grid_bot(
            adx=10.0,
            regime_ema=0.0,
            latest_price=0.50,
        )
        bot.regime_vote_mode = False
        await bot.place_long_orders(0.50)
        bot.get_take_profit_quantity.assert_called_once()


# ===========================================================================
# TestLockdownMode — TP placement when position exceeds threshold
# ===========================================================================


class TestLockdownMode:
    """When position > POSITION_THRESHOLD, lockdown TP logic fires."""

    @pytest.mark.asyncio
    async def test_lockdown_places_tp_when_sell_orders_zero(self):
        """Long lockdown: sell_long_orders=0 → TP placed above price."""
        bot = _grid_bot(adx=10.0, regime_ema=0.0, latest_price=2.0)
        bot.long_position = POSITION_THRESHOLD + 10
        bot.short_position = 1.0
        bot.sell_long_orders = 0.0

        await bot.place_long_orders(2.0)

        bot.place_take_profit_order.assert_called_once()
        args = bot.place_take_profit_order.call_args
        assert args[0][0] == "long"
        # ratio = (int((THRESHOLD+10) / max(1,1)) / 100) + 1 → price * ratio
        assert args[0][1] > 2.0   # TP is above current price

    @pytest.mark.asyncio
    async def test_lockdown_skips_tp_when_sell_orders_nonzero(self):
        """Long lockdown: sell_long_orders > 0 → no new TP placed."""
        bot = _grid_bot(adx=10.0, regime_ema=0.0, latest_price=2.0)
        bot.long_position = POSITION_THRESHOLD + 10
        bot.sell_long_orders = 5.0

        await bot.place_long_orders(2.0)

        bot.place_take_profit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_lockdown_short_uses_division_ratio(self):
        """Short lockdown: TP placed below price using price / ratio."""
        bot = _grid_bot(adx=10.0, regime_ema=0.0, latest_price=2.0)
        bot.short_position = POSITION_THRESHOLD + 10
        bot.long_position = 1.0
        bot.buy_short_orders = 0.0

        await bot.place_short_orders(2.0)

        bot.place_take_profit_order.assert_called_once()
        args = bot.place_take_profit_order.call_args
        assert args[0][0] == "short"
        assert args[0][1] < 2.0   # TP is below current price

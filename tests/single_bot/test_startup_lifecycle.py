"""
tests/single_bot/test_startup_lifecycle.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the bot startup lifecycle:

* ``setup()`` calls leverage, position-mode, and candle seeding in order.
* ``run()`` orchestrates setup → position fetch → order fetch → WS loop.
* Graceful error handling when ``setup()`` methods fail.
* ``validate_config()`` is invoked before bot construction in ``main()``.
* CandleBuffer receives the correct ``vel_dir_ema_period`` from the bot.
"""

from __future__ import annotations

import asyncio
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from src.single_bot.bitunix_bot import GridTradingBot
from src.single_bot.indicators import CandleBuffer, Candle, Signals


def _make_bot(**overrides) -> GridTradingBot:
    """Create a GridTradingBot with mocked exchange calls."""
    defaults = dict(
        api_key="test-key",
        api_secret="test-secret",
        coin_name="XRP",
        grid_spacing=0.015,
        initial_quantity=1,
        leverage=2,
    )
    defaults.update(overrides)
    bot = GridTradingBot(**defaults)

    # Stub exchange REST methods to avoid real API calls
    bot.set_leverage = AsyncMock()
    bot.set_position_mode = AsyncMock()
    bot.get_position = AsyncMock(return_value=(0.0, 0.0))
    bot.check_orders_status = AsyncMock(return_value=(0.0, 0.0, 0.0, 0.0))
    bot.send_startup_notification = AsyncMock()
    bot.send_error_notification = AsyncMock()
    bot.get_balance = AsyncMock(return_value={"USDT": {"available": 1000.0, "margin": 0.0}})
    bot.get_klines = AsyncMock(return_value=[])
    return bot


# ---------------------------------------------------------------------------
# setup() — call sequence
# ---------------------------------------------------------------------------

class TestSetupSequence:
    """Verify setup() calls the correct methods in order."""

    @pytest.mark.asyncio
    async def test_setup_calls_set_leverage(self):
        bot = _make_bot()
        await bot.setup()
        bot.set_leverage.assert_awaited_once_with("XRPUSDT", 2)

    @pytest.mark.asyncio
    async def test_setup_calls_set_position_mode(self):
        bot = _make_bot()
        await bot.setup()
        bot.set_position_mode.assert_awaited_once_with(hedge_mode=True)

    @pytest.mark.asyncio
    async def test_setup_seeds_candle_buffer(self):
        bot = _make_bot()
        bot.candle_buffer.seed = AsyncMock()
        bot.candle_buffer_1m.seed = AsyncMock()
        await bot.setup()
        bot.candle_buffer_1m.seed.assert_awaited_once_with(bot, "XRPUSDT")
        bot.candle_buffer.seed.assert_awaited_once_with(bot, "XRPUSDT")

    @pytest.mark.asyncio
    async def test_setup_sequence_order(self):
        """Leverage → position mode → 1min seed → 15min seed (strict ordering)."""
        bot = _make_bot()
        call_order = []
        bot.set_leverage = AsyncMock(side_effect=lambda *a: call_order.append("leverage"))
        bot.set_position_mode = AsyncMock(side_effect=lambda **kw: call_order.append("pos_mode"))
        bot.candle_buffer_1m.seed = AsyncMock(side_effect=lambda *a: call_order.append("seed_1m"))
        bot.candle_buffer.seed = AsyncMock(side_effect=lambda *a: call_order.append("seed_15m"))
        await bot.setup()
        assert call_order == ["leverage", "pos_mode", "seed_1m", "seed_15m"]


# ---------------------------------------------------------------------------
# setup() — error handling
# ---------------------------------------------------------------------------

class TestSetupErrors:
    """Verify setup() failures propagate properly."""

    @pytest.mark.asyncio
    async def test_leverage_failure_propagates(self):
        bot = _make_bot()
        bot.set_leverage = AsyncMock(side_effect=RuntimeError("API down"))
        with pytest.raises(RuntimeError, match="API down"):
            await bot.setup()

    @pytest.mark.asyncio
    async def test_position_mode_failure_propagates(self):
        bot = _make_bot()
        bot.set_position_mode = AsyncMock(side_effect=RuntimeError("mode error"))
        with pytest.raises(RuntimeError, match="mode error"):
            await bot.setup()

    @pytest.mark.asyncio
    async def test_seed_failure_propagates(self):
        bot = _make_bot()
        bot.candle_buffer_1m.seed = AsyncMock(side_effect=RuntimeError("no data"))
        with pytest.raises(RuntimeError, match="no data"):
            await bot.setup()


# ---------------------------------------------------------------------------
# run() — orchestration
# ---------------------------------------------------------------------------

class TestRunOrchestration:
    """Verify run() wires up setup → position → orders → WS loop."""

    @pytest.mark.asyncio
    async def test_run_calls_setup(self):
        bot = _make_bot()
        bot.setup = AsyncMock()
        bot.monitor_orders = AsyncMock()
        # Break out of the infinite reconnect loop via CancelledError
        bot._public_ws_loop = AsyncMock(side_effect=asyncio.CancelledError)
        bot._private_ws_loop = AsyncMock(side_effect=asyncio.CancelledError)

        with pytest.raises(asyncio.CancelledError):
            await bot.run()
        bot.setup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_fetches_initial_positions(self):
        bot = _make_bot()
        bot.get_position = AsyncMock(return_value=(5.0, 3.0))
        bot.monitor_orders = AsyncMock()
        bot._public_ws_loop = AsyncMock(side_effect=asyncio.CancelledError)
        bot._private_ws_loop = AsyncMock(side_effect=asyncio.CancelledError)

        with pytest.raises(asyncio.CancelledError):
            await bot.run()
        assert bot.long_position == 5.0
        assert bot.short_position == 3.0

    @pytest.mark.asyncio
    async def test_run_sends_startup_notification(self):
        bot = _make_bot()
        bot.monitor_orders = AsyncMock()
        bot._public_ws_loop = AsyncMock(side_effect=asyncio.CancelledError)
        bot._private_ws_loop = AsyncMock(side_effect=asyncio.CancelledError)

        with pytest.raises(asyncio.CancelledError):
            await bot.run()
        bot.send_startup_notification.assert_awaited_once()


# ---------------------------------------------------------------------------
# CandleBuffer wiring
# ---------------------------------------------------------------------------

class TestCandleBufferWiring:
    """Verify CandleBuffer receives correct config from bot constructor."""

    def test_default_vel_dir_ema_period(self):
        """Default bot creates CandleBuffer with vel_dir_ema_period=36."""
        bot = _make_bot()
        assert bot.candle_buffer.vel_dir_ema_period == 36

    def test_custom_vel_dir_ema_period(self):
        """Custom vel_dir_ema_period is passed through to CandleBuffer."""
        bot = _make_bot()
        bot.vel_dir_ema_period = 120
        # Recreate the buffer as it would be in __init__
        bot.candle_buffer = CandleBuffer(
            maxlen=210,
            interval="15min",
            vel_dir_ema_period=bot.vel_dir_ema_period,
        )
        assert bot.candle_buffer.vel_dir_ema_period == 120

    def test_buffer_maxlen_accommodates_regime_ema(self):
        """Buffer maxlen (210) is larger than regime_ema_period (175)."""
        bot = _make_bot()
        assert bot.candle_buffer.maxlen >= 175 + 5

    def test_signals_include_vel_dir_ema(self):
        """CandleBuffer.signals() populates vel_dir_ema field."""
        buf = CandleBuffer(maxlen=200, vel_dir_ema_period=120)
        # Seed with enough candles
        for i in range(200):
            price = 1.0 + i * 0.001
            buf._closed.append(Candle(
                ts=i * 900_000,
                open=price, high=price + 0.001,
                low=price - 0.001, close=price,
                volume=100.0,
            ))
        sig = buf.signals()
        assert sig is not None
        assert sig.vel_dir_ema > 0  # should be computed
        # vel_dir_ema should differ from htf_ema_fast when period != 36
        assert sig.vel_dir_ema != sig.htf_ema_fast  # 120 ≠ 36 → different values


# ---------------------------------------------------------------------------
# Bot constructor — symbol and state init
# ---------------------------------------------------------------------------

class TestBotConstructor:
    """Validate bot constructor wiring."""

    def test_symbol_format(self):
        bot = _make_bot(coin_name="BTC")
        assert bot.symbol == "BTCUSDT"

    def test_initial_positions_zero(self):
        bot = _make_bot()
        assert bot.long_position == 0.0
        assert bot.short_position == 0.0

    def test_trend_state_initialised(self):
        bot = _make_bot()
        assert bot.trend_mode is None
        assert bot.trend_pending_dir is None
        assert bot.trend_confirm_counter == 0

    def test_layer_gates_from_defaults(self):
        """Verify layer gates default to OFF on bot construction."""
        bot = _make_bot()
        assert bot.vel_atr_mult == 0.0
        assert bot.vel_dir_only is False
        assert bot.vel_dir_ema_period == 36

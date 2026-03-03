"""
tests/integration/test_bot_cycle.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration tests for the ``GridTradingBot`` orchestration layer
(src/single_bot/bitunix_bot.py).

These tests exercise the bot end-to-end using a real exchange connection —
they verify that the grid logic wires correctly through
``BitunixExchange`` to produce real orders that can be read back over REST.

Test lifecycle per scenario
---------------------------
1. Instantiate ``GridTradingBot`` with live credentials.
2. Call ``setup()`` — sets leverage, switches to hedge mode, seeds candle buffer.
3. Open a tiny position via market order to give the grid logic something to act on.
4. Inject the live position + price into bot state (normally done by WS events).
5. Call ``place_long_orders()`` / ``place_short_orders()`` and verify open orders.
6. Autouse cleanup cancels all orders and closes all positions.

Cost: exchange fees on ~4 market orders for open+close of 1-XRP positions,
      plus bid–ask spread ≈ ~$0.01 total per run.

Note: ``setup()`` calls ``set_position_mode(hedge_mode=True)`` which requires
no open positions — the autouse fixture in conftest.py guarantees this.
"""

import asyncio
import sys
import os

import pytest

# Allow importing the single_bot module which does a sys.path.insert at module level
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.single_bot.bitunix_bot import GridTradingBot

from .conftest import (
    DEFAULT_LEVERAGE,
    SYMBOL,
    _API_KEY,
    _SECRET_KEY,
    get_mid_price,
    skip_if_no_creds,
)

# Grid spacing for tests — wide enough that orders never accidentally fill
TEST_GRID_SPACING = 0.30  # 30% — dramatically non-executable in production
TEST_QUANTITY = 1         # 1 XRP — minimum order size
TEST_COIN = "XRP"
MIN_QTY = 1.0


def _make_bot() -> GridTradingBot:
    """Construct a GridTradingBot with test parameters."""
    return GridTradingBot(
        api_key=_API_KEY,
        api_secret=_SECRET_KEY,
        coin_name=TEST_COIN,
        grid_spacing=TEST_GRID_SPACING,
        initial_quantity=TEST_QUANTITY,
        leverage=DEFAULT_LEVERAGE,
    )


@skip_if_no_creds
@pytest.mark.integration
class TestBotSetup:
    """GridTradingBot.setup() — initialises leverage, position mode, and candle buffer."""

    async def test_setup_completes_without_error(self) -> None:
        """setup() runs successfully: sets leverage, hedge mode, seeds candle buffer."""
        bot = _make_bot()
        await bot.setup()  # should not raise

    async def test_setup_enables_hedge_mode(self) -> None:
        """After setup(), we can verify the account is in a usable state (get_balance works)."""
        bot = _make_bot()
        await bot.setup()
        balance = await bot.get_balance("USDT")
        assert balance["total"] > 0

    async def test_setup_sets_correct_leverage(self) -> None:
        """After setup(), set_leverage has been called (no exception), bot has leverage attr."""
        bot = _make_bot()
        await bot.setup()
        assert bot.leverage == DEFAULT_LEVERAGE

    async def test_candle_buffer_seeded_after_setup(self) -> None:
        """Candle buffer has at least some closed candles after seeding."""
        bot = _make_bot()
        await bot.setup()
        n_candles = len(bot.candle_buffer._closed)
        assert n_candles > 0, (
            f"Expected closed candles in buffer after seed, got {n_candles}"
        )


@skip_if_no_creds
@pytest.mark.integration
class TestBotCheckOrdersStatus:
    """GridTradingBot.check_orders_status() — classifies open orders into 4 buckets."""

    async def test_returns_four_zero_floats_when_flat(self) -> None:
        """When no orders are open, check_orders_status returns (0, 0, 0, 0)."""
        bot = _make_bot()
        result = await bot.check_orders_status()
        assert isinstance(result, tuple) and len(result) == 4
        buy_long, sell_long, sell_short, buy_short = result
        assert buy_long == 0.0
        assert sell_long == 0.0
        assert sell_short == 0.0
        assert buy_short == 0.0

    async def test_long_entry_reflected_in_buy_long_bucket(self) -> None:
        """A BUY + not-reduce-only order increments the buy_long bucket."""
        bot = _make_bot()
        mid = await get_mid_price(bot)
        # Place a long-entry limit order far below market
        await bot.place_order(
            "buy",
            round(mid * 0.70, 4),
            TEST_QUANTITY,
            is_reduce_only=False,
        )
        await asyncio.sleep(1)

        buy_long, sell_long, sell_short, buy_short = await bot.check_orders_status()
        assert buy_long > 0, (
            f"Expected buy_long > 0 after placing long-entry, got {buy_long}"
        )
        assert sell_long == 0.0
        assert sell_short == 0.0
        assert buy_short == 0.0

    async def test_short_entry_reflected_in_sell_short_bucket(self) -> None:
        """A SELL + not-reduce-only order increments the sell_short bucket."""
        bot = _make_bot()
        mid = await get_mid_price(bot)
        await bot.place_order(
            "sell",
            round(mid * 1.30, 4),
            TEST_QUANTITY,
            is_reduce_only=False,
        )
        await asyncio.sleep(1)

        buy_long, sell_long, sell_short, buy_short = await bot.check_orders_status()
        assert sell_short > 0, (
            f"Expected sell_short > 0 after placing short-entry, got {sell_short}"
        )
        assert buy_long == 0.0


@skip_if_no_creds
@pytest.mark.integration
class TestBotUpdateMidPrice:
    """GridTradingBot.update_mid_price() — pure price calculation, no API calls."""

    def test_long_side_grid_prices_calculated_correctly(self) -> None:
        """Long grid: lower = price × (1 - spacing); upper = price × (1 + spacing)."""
        bot = _make_bot()
        price = 2.5000
        bot.update_mid_price("long", price)

        expected_lower = round(price * (1 - TEST_GRID_SPACING), 4)
        expected_upper = round(price * (1 + TEST_GRID_SPACING), 4)

        assert abs(bot.lower_price_long - expected_lower) < 0.0001, (
            f"lower_price_long={bot.lower_price_long} ≠ expected {expected_lower}"
        )
        assert abs(bot.upper_price_long - expected_upper) < 0.0001, (
            f"upper_price_long={bot.upper_price_long} ≠ expected {expected_upper}"
        )
        assert bot.mid_price_long == round(price, 4)

    def test_short_side_grid_prices_calculated_correctly(self) -> None:
        """Short grid: upper = price × (1 + spacing); lower = price × (1 - spacing)."""
        bot = _make_bot()
        price = 2.5000
        bot.update_mid_price("short", price)

        expected_upper = round(price * (1 + TEST_GRID_SPACING), 4)
        expected_lower = round(price * (1 - TEST_GRID_SPACING), 4)

        assert abs(bot.upper_price_short - expected_upper) < 0.0001
        assert abs(bot.lower_price_short - expected_lower) < 0.0001
        assert bot.mid_price_short == round(price, 4)

    def test_update_is_idempotent_for_same_price(self) -> None:
        """Calling update_mid_price twice with the same price gives identical bounds."""
        bot = _make_bot()
        bot.update_mid_price("long", 2.5000)
        lower1, upper1 = bot.lower_price_long, bot.upper_price_long
        bot.update_mid_price("long", 2.5000)
        assert bot.lower_price_long == lower1
        assert bot.upper_price_long == upper1


@skip_if_no_creds
@pytest.mark.integration
class TestBotGridOrderPlacement:
    """
    Full bot grid cycle: open position → inject state → place grid → verify orders.

    These tests open a 1-XRP position via market order, inject the live position
    size and price into bot state, then call place_long_orders / place_short_orders
    to verify that the bot correctly places both an entry and a take-profit order.
    """

    async def test_place_long_orders_creates_two_orders(self) -> None:
        """
        With a long position open, place_long_orders should place:
          - 1 SELL reduce-only (take-profit leg)
          - 1 BUY open (new entry leg)
        """
        bot = _make_bot()
        await bot.setup()

        # Open a 1-XRP long position
        await bot.place_market_order(SYMBOL, "buy", MIN_QTY, reduce_only=False)
        await asyncio.sleep(2)

        # Fetch live state and inject into bot
        long_qty, _ = await bot.get_positions(SYMBOL)
        assert long_qty > 0, f"Expected a long position, got {long_qty}"
        bot.long_position = long_qty

        mid = await get_mid_price(bot)
        bot.latest_price = mid

        # Place grid orders (cancel + entry + TP)
        await bot.place_long_orders(mid)
        await asyncio.sleep(1)

        # Verify orders exist
        buy_long, sell_long, _ss, _bs = await bot.check_orders_status()
        assert sell_long > 0, (
            f"Expected at least 1 sell_long (TP) order after place_long_orders, "
            f"got sell_long={sell_long}"
        )
        assert buy_long > 0, (
            f"Expected at least 1 buy_long (entry) order after place_long_orders, "
            f"got buy_long={buy_long}"
        )

    async def test_place_short_orders_creates_two_orders(self) -> None:
        """
        With a short position open, place_short_orders should place:
          - 1 BUY reduce-only (take-profit leg)
          - 1 SELL open (new entry leg)
        """
        bot = _make_bot()
        await bot.setup()

        # Open a 1-XRP short position
        await bot.place_market_order(SYMBOL, "sell", MIN_QTY, reduce_only=False)
        await asyncio.sleep(2)

        # Fetch live state and inject into bot
        _, short_qty = await bot.get_positions(SYMBOL)
        assert short_qty > 0, f"Expected a short position, got {short_qty}"
        bot.short_position = short_qty

        mid = await get_mid_price(bot)
        bot.latest_price = mid

        # Place grid orders
        await bot.place_short_orders(mid)
        await asyncio.sleep(1)

        _bl, _sl, sell_short, buy_short = await bot.check_orders_status()
        assert buy_short > 0, (
            f"Expected at least 1 buy_short (TP) order after place_short_orders, "
            f"got buy_short={buy_short}"
        )
        assert sell_short > 0, (
            f"Expected at least 1 sell_short (entry) order after place_short_orders, "
            f"got sell_short={sell_short}"
        )

    async def test_place_long_orders_without_position_is_noop(self) -> None:
        """place_long_orders does nothing if long_position == 0 (no position to grid)."""
        bot = _make_bot()
        bot.long_position = 0.0
        bot.latest_price = 2.50

        await bot.place_long_orders(2.50)
        await asyncio.sleep(1)

        orders = await bot.get_open_orders(SYMBOL)
        assert len(orders) == 0, (
            f"Expected no orders (position=0), but found {len(orders)}"
        )

"""
tests/integration/test_hedge_repair.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration tests for the ``HedgeRepairBot`` against a real Bitunix exchange.

These tests exercise the bot's exchange-interaction layer with real API calls:
  - setup() — leverage, hedge mode, price history seeding
  - _open_both() / _close_position() — market orders for hedge pair
  - _dca_repair() — DCA add to a losing position
  - Full cycle — IDLE → BOTH_OPEN → close one side → REPAIRING → close remaining

Cost: ~4–6 market orders per test, ~1 XRP each ≈ ~$0.02 fees total.
Pre-fund the account with ~$50 USDT on Bitunix Futures.

Note: the autouse ``_clean_exchange_state`` fixture from ``conftest.py``
cancels orders and closes all positions before/after every test.
"""

from __future__ import annotations

import asyncio
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.exchange.bitunix import BitunixExchange
from src.single_bot.hedge_repair_bot import HedgeRepairBot, HedgePosition

from .conftest import (
    DEFAULT_LEVERAGE,
    SYMBOL,
    _API_KEY,
    _SECRET_KEY,
    get_mid_price,
    skip_if_no_creds,
)

TEST_COIN = "XRP"
MIN_QTY = 1.0


def _make_hedge_bot(exchange: BitunixExchange) -> HedgeRepairBot:
    """Create a HedgeRepairBot with conservative test config."""
    config = {
        "symbol": TEST_COIN,
        "leverage": DEFAULT_LEVERAGE,
        "entry_pct": 0.05,          # 5% of balance per side
        "fee_pct": 0.0006,
        "profit_threshold": 5.0,
        "trailing_distance": 1.0,
        "lookback_candles": 5,
        "zig_zag_candles": 3,
        "zig_zag_threshold": 0.01,
        "dca_multiplier": 1.0,
        "liq_proximity_pct": 0.80,
        "dynamic_threshold": False,
        "net_profit_target": 0.10,
        "spot_reserve": 0.0,        # No spot reserve for integration tests
        "periodic_injection_usd": 0.0,
        "poll_interval_secs": 0,
        "enable_notifications": False,
    }
    return HedgeRepairBot(exchange, config)


# ═══════════════════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════════════════

@skip_if_no_creds
@pytest.mark.integration
class TestHedgeRepairSetup:
    """HedgeRepairBot.setup() with a real exchange."""

    async def test_setup_completes(self, exchange: BitunixExchange) -> None:
        """setup() sets leverage, hedge mode, and seeds price history."""
        bot = _make_hedge_bot(exchange)
        await bot.setup()

        # Price ring should have been seeded
        assert len(bot._price_ring) > 0

    async def test_setup_leverage(self, exchange: BitunixExchange) -> None:
        """After setup, the exchange should accept orders at the configured leverage."""
        bot = _make_hedge_bot(exchange)
        await bot.setup()

        # Verify we can read balance (proves exchange is connected)
        bal = await exchange.get_balance()
        assert bal["free"] > 0, "Account needs USDT balance for integration tests"


# ═══════════════════════════════════════════════════════════════════════════
# Open / Close Pair
# ═══════════════════════════════════════════════════════════════════════════

@skip_if_no_creds
@pytest.mark.integration
class TestOpenClosePair:
    """Test opening and closing a hedge pair via real market orders."""

    async def test_open_both(self, exchange: BitunixExchange) -> None:
        """_open_both() opens simultaneous long + short at market price."""
        bot = _make_hedge_bot(exchange)
        await bot.setup()

        price = await get_mid_price(exchange)
        result = await bot._open_both(price)

        assert result is True
        assert bot.long_pos is not None
        assert bot.short_pos is not None
        assert bot.long_pos.qty > 0
        assert bot.short_pos.qty > 0

        # Verify positions exist on exchange
        long_qty, short_qty = await exchange.get_positions(SYMBOL)
        assert long_qty > 0
        assert short_qty > 0

    async def test_close_position(self, exchange: BitunixExchange) -> None:
        """_close_position() closes one side of the hedge pair."""
        bot = _make_hedge_bot(exchange)
        await bot.setup()

        price = await get_mid_price(exchange)
        await bot._open_both(price)
        assert bot.long_pos is not None

        # Close the long side
        net = await bot._close_position(bot.long_pos, price, "test_close")
        assert isinstance(net, float)

        # Verify long is flat, short remains
        long_qty, short_qty = await exchange.get_positions(SYMBOL)
        assert long_qty == 0
        assert short_qty > 0

    async def test_full_open_close_cycle(self, exchange: BitunixExchange) -> None:
        """Open both → close long → close short → back to flat."""
        bot = _make_hedge_bot(exchange)
        await bot.setup()

        price = await get_mid_price(exchange)
        await bot._open_both(price)

        # Close long
        await bot._close_position(bot.long_pos, price, "cycle_long")
        bot.long_pos = None

        # Close short
        await bot._close_position(bot.short_pos, price, "cycle_short")
        bot.short_pos = None

        # Verify flat
        long_qty, short_qty = await exchange.get_positions(SYMBOL)
        assert long_qty == 0
        assert short_qty == 0


# ═══════════════════════════════════════════════════════════════════════════
# DCA Repair
# ═══════════════════════════════════════════════════════════════════════════

@skip_if_no_creds
@pytest.mark.integration
class TestDCARepair:
    """Test DCA repair with real exchange orders."""

    async def test_dca_adds_to_position(self, exchange: BitunixExchange) -> None:
        """_dca_repair() adds to the position via a real market order."""
        bot = _make_hedge_bot(exchange)
        await bot.setup()

        price = await get_mid_price(exchange)
        await bot._open_both(price)
        assert bot.long_pos is not None

        original_qty = bot.long_pos.qty

        # DCA the long side at current price
        result = await bot._dca_repair(bot.long_pos, price)
        assert result is True
        assert bot.long_pos.qty > original_qty
        assert bot.long_pos.dca_count == 1
        assert bot.total_dca_count == 1

        # Verify the position grew on the exchange
        long_qty, _ = await exchange.get_positions(SYMBOL)
        assert long_qty > original_qty


# ═══════════════════════════════════════════════════════════════════════════
# Balance / State
# ═══════════════════════════════════════════════════════════════════════════

@skip_if_no_creds
@pytest.mark.integration
class TestBalanceState:
    """Verify balance tracking with real exchange balance."""

    async def test_balance_decreases_after_open(self, exchange: BitunixExchange) -> None:
        """Opening a hedge pair decreases free balance (margin locked)."""
        bot = _make_hedge_bot(exchange)
        await bot.setup()

        bal_before = await exchange.get_balance()
        free_before = bal_before["free"]

        price = await get_mid_price(exchange)
        await bot._open_both(price)

        bal_after = await exchange.get_balance()
        free_after = bal_after["free"]

        # Free balance should have decreased by 2 × margin + fees
        assert free_after < free_before

    async def test_trade_log_populated(self, exchange: BitunixExchange) -> None:
        """Trade log captures the OPEN_PAIR action."""
        bot = _make_hedge_bot(exchange)
        await bot.setup()

        price = await get_mid_price(exchange)
        await bot._open_both(price)

        assert len(bot.trade_log) == 1
        assert bot.trade_log[0]["action"] == "OPEN_PAIR"
        assert bot.trade_log[0]["qty"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# Status Summary
# ═══════════════════════════════════════════════════════════════════════════

@skip_if_no_creds
@pytest.mark.integration
class TestStatusSummary:
    """Verify status_summary() works with real positions."""

    async def test_summary_with_live_positions(self, exchange: BitunixExchange) -> None:
        """status_summary() returns valid text after opening a hedge pair."""
        bot = _make_hedge_bot(exchange)
        await bot.setup()

        price = await get_mid_price(exchange)
        await bot._open_both(price)
        bot.state = HedgeRepairBot.BOTH_OPEN

        summary = bot.status_summary()
        assert "BOTH_OPEN" in summary
        assert "Long:" in summary
        assert "Short:" in summary


# ═══════════════════════════════════════════════════════════════════════════
# Wallet Transfers (spot ↔ futures)
# ═══════════════════════════════════════════════════════════════════════════

@skip_if_no_creds
@pytest.mark.integration
class TestWalletTransfers:
    """Test real spot ↔ futures wallet transfers on Bitunix."""

    async def test_round_trip_transfer(self, exchange: BitunixExchange) -> None:
        """Transfer $1 futures → spot, then $1 spot → futures (net zero)."""
        bal_before = await exchange.get_balance()

        # Futures → spot
        tid1 = await exchange.transfer_futures_to_spot(1.0, coin="USDT")
        assert tid1  # non-empty transfer ID

        bal_mid = await exchange.get_balance()
        assert bal_mid["free"] < bal_before["free"]

        # Spot → futures (return the $1)
        tid2 = await exchange.transfer_spot_to_futures(1.0, coin="USDT")
        assert tid2

        bal_after = await exchange.get_balance()
        # Should be approximately back to where we started (minus any processing delays)
        assert bal_after["free"] == pytest.approx(bal_before["free"], abs=0.1)

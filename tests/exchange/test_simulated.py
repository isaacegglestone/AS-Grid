"""
tests/exchange/test_simulated.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the SimulatedExchange paper-trading adapter.

Covers: balances, order placement, fill matching, position management,
cancel logic, equity calculations.
"""

from __future__ import annotations

import pytest

from src.exchange.simulated import SimulatedExchange


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sim() -> SimulatedExchange:
    """Fresh simulated exchange with $1000 and 5x leverage."""
    s = SimulatedExchange(initial_balance=1000.0, fee_pct=0.0006)
    return s


SYMBOL = "XRPUSDT"


# ---------------------------------------------------------------------------
# Balance
# ---------------------------------------------------------------------------

class TestBalance:
    async def test_initial_balance(self, sim: SimulatedExchange):
        bal = await sim.get_balance()
        assert bal["total"] == 1000.0
        assert bal["free"] == 1000.0
        assert bal["used"] == 0.0

    async def test_balance_after_open(self, sim: SimulatedExchange):
        await sim.set_leverage(SYMBOL, 5)
        sim._last_price = 2.0
        await sim.place_market_order(SYMBOL, "buy", 100)
        bal = await sim.get_balance()
        # margin = 100 * 2.0 / 5 = 40.0
        # fee = 100 * 2.0 * 0.0006 = 0.12
        assert bal["used"] == pytest.approx(40.0, abs=0.01)
        assert bal["free"] == pytest.approx(1000.0 - 40.0 - 0.12, abs=0.01)


# ---------------------------------------------------------------------------
# Leverage & Position Mode
# ---------------------------------------------------------------------------

class TestConfig:
    async def test_set_leverage(self, sim: SimulatedExchange):
        await sim.set_leverage(SYMBOL, 10)
        assert sim._leverage[SYMBOL] == 10

    async def test_set_position_mode(self, sim: SimulatedExchange):
        await sim.set_position_mode(True)
        assert sim._hedge_mode is True
        await sim.set_position_mode(False)
        assert sim._hedge_mode is False


# ---------------------------------------------------------------------------
# Market Orders
# ---------------------------------------------------------------------------

class TestMarketOrders:
    async def test_open_long_market(self, sim: SimulatedExchange):
        await sim.set_leverage(SYMBOL, 5)
        sim._last_price = 2.0
        result = await sim.place_market_order(SYMBOL, "buy", 100)
        assert result is not None
        assert "orderId" in result

        long_qty, short_qty = await sim.get_positions(SYMBOL)
        assert long_qty == 100.0
        assert short_qty == 0.0

    async def test_open_short_market(self, sim: SimulatedExchange):
        await sim.set_leverage(SYMBOL, 5)
        sim._last_price = 2.0
        await sim.place_market_order(SYMBOL, "sell", 50)

        long_qty, short_qty = await sim.get_positions(SYMBOL)
        assert long_qty == 0.0
        assert short_qty == 50.0

    async def test_close_long_market(self, sim: SimulatedExchange):
        await sim.set_leverage(SYMBOL, 5)
        sim._last_price = 2.0
        await sim.place_market_order(SYMBOL, "buy", 100)

        # Close at higher price
        sim._last_price = 2.5
        await sim.place_market_order(SYMBOL, "sell", 100, reduce_only=True)

        long_qty, _ = await sim.get_positions(SYMBOL)
        assert long_qty == 0.0

        # Should have profit
        bal = await sim.get_balance()
        assert bal["total"] > 1000.0

    async def test_close_short_market(self, sim: SimulatedExchange):
        await sim.set_leverage(SYMBOL, 5)
        sim._last_price = 2.0
        await sim.place_market_order(SYMBOL, "sell", 100)

        # Close at lower price (profit for short)
        sim._last_price = 1.5
        await sim.place_market_order(SYMBOL, "buy", 100, reduce_only=True)

        _, short_qty = await sim.get_positions(SYMBOL)
        assert short_qty == 0.0

        bal = await sim.get_balance()
        assert bal["total"] > 1000.0

    async def test_no_price_returns_none(self, sim: SimulatedExchange):
        result = await sim.place_market_order(SYMBOL, "buy", 100)
        assert result is None


# ---------------------------------------------------------------------------
# Limit Orders & Fill Matching
# ---------------------------------------------------------------------------

class TestLimitOrders:
    async def test_place_limit_order(self, sim: SimulatedExchange):
        await sim.set_leverage(SYMBOL, 5)
        result = await sim.place_limit_order(SYMBOL, "buy", 100, 1.90)
        assert result is not None

        orders = await sim.get_open_orders(SYMBOL)
        assert len(orders) == 1
        assert orders[0]["price"] == 1.90

    async def test_limit_buy_fills_on_low(self, sim: SimulatedExchange):
        await sim.set_leverage(SYMBOL, 5)
        await sim.place_limit_order(SYMBOL, "buy", 100, 1.90)

        # Price dips to 1.85 — should fill the buy at 1.90
        fills = sim.on_price_update(SYMBOL, 1.88, high=1.95, low=1.85)
        assert len(fills) == 1
        assert fills[0]["price"] == 1.90

        long_qty, _ = await sim.get_positions(SYMBOL)
        assert long_qty == 100.0

        orders = await sim.get_open_orders(SYMBOL)
        assert len(orders) == 0

    async def test_limit_sell_fills_on_high(self, sim: SimulatedExchange):
        await sim.set_leverage(SYMBOL, 5)
        await sim.place_limit_order(SYMBOL, "sell", 50, 2.10)

        fills = sim.on_price_update(SYMBOL, 2.12, high=2.15, low=2.05)
        assert len(fills) == 1
        assert fills[0]["price"] == 2.10

        _, short_qty = await sim.get_positions(SYMBOL)
        assert short_qty == 50.0

    async def test_limit_no_fill_if_not_crossed(self, sim: SimulatedExchange):
        await sim.set_leverage(SYMBOL, 5)
        await sim.place_limit_order(SYMBOL, "buy", 100, 1.90)

        fills = sim.on_price_update(SYMBOL, 2.00, high=2.05, low=1.95)
        assert len(fills) == 0

        orders = await sim.get_open_orders(SYMBOL)
        assert len(orders) == 1


# ---------------------------------------------------------------------------
# Cancel Orders
# ---------------------------------------------------------------------------

class TestCancelOrders:
    async def test_cancel_single(self, sim: SimulatedExchange):
        await sim.set_leverage(SYMBOL, 5)
        result = await sim.place_limit_order(SYMBOL, "buy", 100, 1.90)
        assert await sim.cancel_order(result["orderId"], SYMBOL) is True

        orders = await sim.get_open_orders(SYMBOL)
        assert len(orders) == 0

    async def test_cancel_nonexistent(self, sim: SimulatedExchange):
        assert await sim.cancel_order("fake_id", SYMBOL) is False

    async def test_cancel_for_side_long(self, sim: SimulatedExchange):
        await sim.set_leverage(SYMBOL, 5)
        # Long entry (BUY, not reduce_only)
        await sim.place_limit_order(SYMBOL, "buy", 100, 1.90)
        # Short entry (SELL, not reduce_only)
        await sim.place_limit_order(SYMBOL, "sell", 50, 2.10)

        await sim.cancel_orders_for_side(SYMBOL, "long")

        orders = await sim.get_open_orders(SYMBOL)
        assert len(orders) == 1
        assert orders[0]["side"] == "SELL"


# ---------------------------------------------------------------------------
# Equity & Unrealised PnL
# ---------------------------------------------------------------------------

class TestEquity:
    async def test_unrealised_pnl_long(self, sim: SimulatedExchange):
        await sim.set_leverage(SYMBOL, 5)
        sim._last_price = 2.0
        await sim.place_market_order(SYMBOL, "buy", 100)

        pnl = sim.unrealised_pnl(SYMBOL, 2.5)
        assert pnl == pytest.approx(50.0, abs=0.01)  # (2.5 - 2.0) * 100

    async def test_unrealised_pnl_short(self, sim: SimulatedExchange):
        await sim.set_leverage(SYMBOL, 5)
        sim._last_price = 2.0
        await sim.place_market_order(SYMBOL, "sell", 100)

        pnl = sim.unrealised_pnl(SYMBOL, 1.5)
        assert pnl == pytest.approx(50.0, abs=0.01)  # (2.0 - 1.5) * 100

    async def test_total_equity(self, sim: SimulatedExchange):
        await sim.set_leverage(SYMBOL, 5)
        sim._last_price = 2.0
        await sim.place_market_order(SYMBOL, "buy", 100)

        eq = sim.total_equity(SYMBOL, 2.5)
        # 1000 - 40 margin - 0.12 fee + 40 used + 50 pnl = 1049.88
        assert eq > 1000.0

    async def test_hedge_pair_zero_pnl(self, sim: SimulatedExchange):
        """Simultaneous long+short at same price → zero unrealised PnL."""
        await sim.set_leverage(SYMBOL, 5)
        sim._last_price = 2.0
        await sim.place_market_order(SYMBOL, "buy", 100)
        await sim.place_market_order(SYMBOL, "sell", 100)

        pnl = sim.unrealised_pnl(SYMBOL, 2.0)
        assert pnl == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    async def test_reset(self, sim: SimulatedExchange):
        await sim.set_leverage(SYMBOL, 5)
        sim._last_price = 2.0
        await sim.place_market_order(SYMBOL, "buy", 100)

        sim.reset(500.0)
        bal = await sim.get_balance()
        assert bal["total"] == 500.0
        long_qty, short_qty = await sim.get_positions(SYMBOL)
        assert long_qty == 0.0
        assert short_qty == 0.0


# ---------------------------------------------------------------------------
# Trade Log
# ---------------------------------------------------------------------------

class TestTradeLog:
    async def test_trade_logged(self, sim: SimulatedExchange):
        await sim.set_leverage(SYMBOL, 5)
        sim._last_price = 2.0
        await sim.place_market_order(SYMBOL, "buy", 100)

        assert len(sim.trade_log) == 1
        assert sim.trade_log[0]["side"] == "BUY"
        assert sim.trade_log[0]["trade_side"] == "OPEN"
        assert sim.trade_log[0]["qty"] == 100

    async def test_fees_tracked(self, sim: SimulatedExchange):
        await sim.set_leverage(SYMBOL, 5)
        sim._last_price = 2.0
        await sim.place_market_order(SYMBOL, "buy", 100)

        expected_fee = 100 * 2.0 * 0.0006
        assert sim.total_fees == pytest.approx(expected_fee, abs=0.001)


# ---------------------------------------------------------------------------
# DCA (weighted average entry)
# ---------------------------------------------------------------------------

class TestDCA:
    async def test_dca_updates_avg_entry(self, sim: SimulatedExchange):
        await sim.set_leverage(SYMBOL, 5)
        sim._last_price = 2.0
        await sim.place_market_order(SYMBOL, "buy", 100)

        # DCA at lower price
        sim._last_price = 1.5
        await sim.place_market_order(SYMBOL, "buy", 100)

        pos = sim.get_position_detail(SYMBOL, "LONG")
        assert pos is not None
        assert pos.qty == 200.0
        # Weighted avg: (100*2.0 + 100*1.5) / 200 = 1.75
        assert pos.avg_entry == pytest.approx(1.75, abs=0.001)


# ---------------------------------------------------------------------------
# Wallet transfers (spot ↔ futures)
# ---------------------------------------------------------------------------

class TestTransfers:
    async def test_spot_to_futures_adds_balance(self, sim: SimulatedExchange):
        bal_before = await sim.get_balance()
        tid = await sim.transfer_spot_to_futures(100.0)
        bal_after = await sim.get_balance()
        assert bal_after["free"] == pytest.approx(bal_before["free"] + 100.0)
        assert tid  # non-empty transfer ID

    async def test_futures_to_spot_deducts_balance(self, sim: SimulatedExchange):
        bal_before = await sim.get_balance()
        tid = await sim.transfer_futures_to_spot(50.0)
        bal_after = await sim.get_balance()
        assert bal_after["free"] == pytest.approx(bal_before["free"] - 50.0)
        assert tid

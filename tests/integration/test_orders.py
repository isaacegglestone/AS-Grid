"""
tests/integration/test_orders.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration tests for the full limit-order lifecycle:
  place_limit_order → get_open_orders → cancel_order → cancel_orders_for_side

Strategy to avoid accidental fills
------------------------------------
All limit orders are placed at deliberately unreachable prices:
  - Buy  orders: 30% below current market price
  - Sell orders: 30% above current market price

With XRP at ~$2.50 this gives safety margins of ~$0.75, which is far larger
than the ~1.5% grid spacing we operate at in production.

Cost: $0 — every order placed in this suite is immediately cancelled.
Expected balance impact: fees only on cancellation (usually $0 on limit orders).
"""

import asyncio

import pytest

from src.exchange.bitunix import BitunixExchange

from .conftest import SYMBOL, get_mid_price, skip_if_no_creds

# Fraction of current price used to set safe order prices
BUY_DISCOUNT = 0.30   # buy at 30% below market → will never fill in normal conditions
SELL_PREMIUM = 0.30   # sell at 30% above market

# Minimum quantity for XRP/USDT on Bitunix Futures (1 XRP)
MIN_QTY = 1.0


def _buy_price(mid: float) -> float:
    return round(mid * (1 - BUY_DISCOUNT), 4)


def _sell_price(mid: float) -> float:
    return round(mid * (1 + SELL_PREMIUM), 4)


@skip_if_no_creds
@pytest.mark.integration
class TestPlaceLimitOrder:
    """BitunixExchange.place_limit_order — places a GTC limit order."""

    async def test_place_limit_buy_returns_order_id(
        self, exchange: BitunixExchange
    ) -> None:
        """Placing a limit buy returns a dict with a non-empty orderId."""
        mid = await get_mid_price(exchange)
        result = await exchange.place_limit_order(
            SYMBOL, side="buy", quantity=MIN_QTY, price=_buy_price(mid)
        )
        assert result is not None, "place_limit_order returned None"
        assert "orderId" in result, f"No 'orderId' in result: {result}"
        assert result["orderId"], "orderId is empty"

    async def test_place_limit_sell_returns_order_id(
        self, exchange: BitunixExchange
    ) -> None:
        """Placing a limit sell (open short) returns a dict with a non-empty orderId."""
        mid = await get_mid_price(exchange)
        result = await exchange.place_limit_order(
            SYMBOL, side="sell", quantity=MIN_QTY, price=_sell_price(mid)
        )
        assert result is not None, "place_limit_order returned None"
        assert "orderId" in result, f"No 'orderId' in result: {result}"
        assert result["orderId"], "orderId is empty"

    async def test_place_limit_order_at_far_below_market_does_not_fill(
        self, exchange: BitunixExchange
    ) -> None:
        """A limit buy at 30% below market remains open (not immediately filled)."""
        mid = await get_mid_price(exchange)
        result = await exchange.place_limit_order(
            SYMBOL, side="buy", quantity=MIN_QTY, price=_buy_price(mid)
        )
        assert result is not None
        # Give the exchange a moment to process
        await asyncio.sleep(1)
        orders = await exchange.get_open_orders(SYMBOL)
        order_ids = [o["orderId"] for o in orders]
        assert result["orderId"] in order_ids, (
            "Expected order to remain open but it was already filled/cancelled. "
            "Check that BUY_DISCOUNT is large enough for current market conditions."
        )


@skip_if_no_creds
@pytest.mark.integration
class TestGetOpenOrders:
    """BitunixExchange.get_open_orders — fetches pending orders."""

    async def test_returns_list_when_no_orders(
        self, exchange: BitunixExchange
    ) -> None:
        """get_open_orders returns an empty list when no orders are pending."""
        orders = await exchange.get_open_orders(SYMBOL)
        assert isinstance(orders, list)
        assert len(orders) == 0, (
            f"Expected 0 open orders before test but found {len(orders)}. "
            "The autouse cleanup should have cancelled them."
        )

    async def test_placed_order_appears_in_open_orders(
        self, exchange: BitunixExchange
    ) -> None:
        """An order just placed appears in get_open_orders output."""
        mid = await get_mid_price(exchange)
        placed = await exchange.place_limit_order(
            SYMBOL, side="buy", quantity=MIN_QTY, price=_buy_price(mid)
        )
        assert placed

        await asyncio.sleep(1)
        orders = await exchange.get_open_orders(SYMBOL)
        ids = [o["orderId"] for o in orders]
        assert placed["orderId"] in ids

    async def test_open_order_dicts_have_required_fields(
        self, exchange: BitunixExchange
    ) -> None:
        """Each entry from get_open_orders has the normalised fields the bot uses."""
        mid = await get_mid_price(exchange)
        await exchange.place_limit_order(
            SYMBOL, side="buy", quantity=MIN_QTY, price=_buy_price(mid)
        )
        await asyncio.sleep(1)
        orders = await exchange.get_open_orders(SYMBOL)
        assert orders
        for o in orders:
            for field in (
                "orderId", "side", "price", "qty", "remaining", "reduceOnly", "status"
            ):
                assert field in o, f"Missing field '{field}' in order: {o}"

    async def test_open_order_side_is_uppercase(
        self, exchange: BitunixExchange
    ) -> None:
        """side field is normalised to uppercase 'BUY' or 'SELL'."""
        mid = await get_mid_price(exchange)
        await exchange.place_limit_order(
            SYMBOL, side="buy", quantity=MIN_QTY, price=_buy_price(mid)
        )
        await asyncio.sleep(1)
        orders = await exchange.get_open_orders(SYMBOL)
        for o in orders:
            assert o["side"] in ("BUY", "SELL"), (
                f"Unexpected side value '{o['side']}'"
            )

    async def test_multiple_orders_all_appear(
        self, exchange: BitunixExchange
    ) -> None:
        """Placing 3 orders returns all 3 in get_open_orders."""
        mid = await get_mid_price(exchange)
        placed_ids = set()
        for offset in [0.30, 0.35, 0.40]:
            r = await exchange.place_limit_order(
                SYMBOL,
                side="buy",
                quantity=MIN_QTY,
                price=round(mid * (1 - offset), 4),
            )
            assert r
            placed_ids.add(r["orderId"])

        await asyncio.sleep(1)
        orders = await exchange.get_open_orders(SYMBOL)
        open_ids = {o["orderId"] for o in orders}
        assert placed_ids.issubset(open_ids), (
            f"Not all placed orders appear in open orders. "
            f"Placed: {placed_ids}, Open: {open_ids}"
        )


@skip_if_no_creds
@pytest.mark.integration
class TestCancelOrder:
    """BitunixExchange.cancel_order — cancels a single order by ID."""

    async def test_cancel_returns_true_on_success(
        self, exchange: BitunixExchange
    ) -> None:
        """cancel_order returns True when the order is successfully cancelled."""
        mid = await get_mid_price(exchange)
        placed = await exchange.place_limit_order(
            SYMBOL, side="buy", quantity=MIN_QTY, price=_buy_price(mid)
        )
        assert placed

        await asyncio.sleep(1)
        success = await exchange.cancel_order(placed["orderId"], SYMBOL)
        assert success is True, f"Expected cancel_order to return True, got {success}"

    async def test_cancelled_order_disappears_from_open_orders(
        self, exchange: BitunixExchange
    ) -> None:
        """After cancellation the order is no longer listed in get_open_orders."""
        mid = await get_mid_price(exchange)
        placed = await exchange.place_limit_order(
            SYMBOL, side="buy", quantity=MIN_QTY, price=_buy_price(mid)
        )
        assert placed
        await asyncio.sleep(1)

        await exchange.cancel_order(placed["orderId"], SYMBOL)
        await asyncio.sleep(1)

        orders = await exchange.get_open_orders(SYMBOL)
        ids = [o["orderId"] for o in orders]
        assert placed["orderId"] not in ids, (
            f"Order {placed['orderId']} still appears in open orders after cancel"
        )

    async def test_cancel_nonexistent_order_returns_false(
        self, exchange: BitunixExchange
    ) -> None:
        """cancel_order returns False (or at least does not raise) for a bogus ID."""
        result = await exchange.cancel_order("00000000000000000000", SYMBOL)
        assert result is False


@skip_if_no_creds
@pytest.mark.integration
class TestCancelOrdersForSide:
    """BitunixExchange.cancel_orders_for_side — bulk-cancels by long/short side."""

    async def test_cancel_long_side_removes_long_entries(
        self, exchange: BitunixExchange
    ) -> None:
        """Placing 2 long-entry buys and calling cancel_orders_for_side('long') cancels both."""
        mid = await get_mid_price(exchange)
        ids = []
        for offset in [0.30, 0.35]:
            r = await exchange.place_limit_order(
                SYMBOL,
                side="buy",       # BUY + not reduce-only = long entry
                quantity=MIN_QTY,
                price=round(mid * (1 - offset), 4),
                reduce_only=False,
            )
            assert r
            ids.append(r["orderId"])

        await asyncio.sleep(1)

        await exchange.cancel_orders_for_side(SYMBOL, "long")
        await asyncio.sleep(1)

        orders = await exchange.get_open_orders(SYMBOL)
        remaining_ids = {o["orderId"] for o in orders}
        for oid in ids:
            assert oid not in remaining_ids, (
                f"Long-entry order {oid} still open after cancel_orders_for_side('long')"
            )

    async def test_cancel_short_side_removes_short_entries(
        self, exchange: BitunixExchange
    ) -> None:
        """Placing 2 short-entry sells and calling cancel_orders_for_side('short') cancels both."""
        mid = await get_mid_price(exchange)
        ids = []
        for offset in [0.30, 0.35]:
            r = await exchange.place_limit_order(
                SYMBOL,
                side="sell",      # SELL + not reduce-only = short entry
                quantity=MIN_QTY,
                price=round(mid * (1 + offset), 4),
                reduce_only=False,
            )
            assert r
            ids.append(r["orderId"])

        await asyncio.sleep(1)

        await exchange.cancel_orders_for_side(SYMBOL, "short")
        await asyncio.sleep(1)

        orders = await exchange.get_open_orders(SYMBOL)
        remaining_ids = {o["orderId"] for o in orders}
        for oid in ids:
            assert oid not in remaining_ids, (
                f"Short-entry order {oid} still open after cancel_orders_for_side('short')"
            )

    async def test_cancel_long_does_not_affect_short_orders(
        self, exchange: BitunixExchange
    ) -> None:
        """cancel_orders_for_side('long') leaves short-entry orders intact."""
        mid = await get_mid_price(exchange)

        # Place one long entry and one short entry
        long_r = await exchange.place_limit_order(
            SYMBOL, side="buy", quantity=MIN_QTY,
            price=round(mid * 0.70, 4), reduce_only=False
        )
        short_r = await exchange.place_limit_order(
            SYMBOL, side="sell", quantity=MIN_QTY,
            price=round(mid * 1.30, 4), reduce_only=False
        )
        assert long_r and short_r
        await asyncio.sleep(1)

        await exchange.cancel_orders_for_side(SYMBOL, "long")
        await asyncio.sleep(1)

        orders = await exchange.get_open_orders(SYMBOL)
        remaining_ids = {o["orderId"] for o in orders}
        assert long_r["orderId"] not in remaining_ids, "Long order should be cancelled"
        assert short_r["orderId"] in remaining_ids, "Short order should be untouched"

    async def test_cancel_for_side_on_empty_book_is_noop(
        self, exchange: BitunixExchange
    ) -> None:
        """cancel_orders_for_side does not raise when there are no matching orders."""
        await exchange.cancel_orders_for_side(SYMBOL, "long")
        await exchange.cancel_orders_for_side(SYMBOL, "short")

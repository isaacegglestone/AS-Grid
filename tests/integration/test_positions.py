"""
tests/integration/test_positions.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration tests for position management via market orders:
  place_market_order (open) → get_positions → place_market_order (close)

These tests actually execute trades on the exchange.  The positions opened are
the smallest possible size (1 XRP ≈ $2.50 notional at 5x leverage = $0.50 margin).

Cost per test cycle: exchange fees on 2 market orders (~0.04% × 2 × $2.50 ≈ $0.002).
Full suite cost across all position tests: ~$0.05 USDT in fees.

The autouse cleanup fixture in conftest.py will close any position left open by a
failed test so the account never gets stuck with unwanted exposure.
"""

import asyncio

import pytest

from src.exchange.bitunix import BitunixExchange

from .conftest import DEFAULT_LEVERAGE, SYMBOL, skip_if_no_creds

MIN_QTY = 1.0  # 1 XRP — minimum order size on Bitunix Futures


@skip_if_no_creds
@pytest.mark.integration
class TestGetPositions:
    """BitunixExchange.get_positions — reads open position quantities."""

    async def test_returns_tuple_of_two_floats(
        self, exchange: BitunixExchange
    ) -> None:
        """get_positions returns (long_qty, short_qty) as a 2-tuple of floats."""
        result = await exchange.get_positions(SYMBOL)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2-tuple, got length {len(result)}"
        long_qty, short_qty = result
        assert isinstance(long_qty, float), f"long_qty is not float: {long_qty}"
        assert isinstance(short_qty, float), f"short_qty is not float: {short_qty}"

    async def test_flat_when_no_positions(self, exchange: BitunixExchange) -> None:
        """Returns (0.0, 0.0) when no positions are open (autouse cleanup ensures this)."""
        long_qty, short_qty = await exchange.get_positions(SYMBOL)
        assert long_qty == 0.0, f"Expected 0 long qty but got {long_qty}"
        assert short_qty == 0.0, f"Expected 0 short qty but got {short_qty}"

    async def test_non_negative_quantities(self, exchange: BitunixExchange) -> None:
        """Long and short quantities are always non-negative."""
        long_qty, short_qty = await exchange.get_positions(SYMBOL)
        assert long_qty >= 0.0
        assert short_qty >= 0.0


@skip_if_no_creds
@pytest.mark.integration
class TestPlaceMarketOrder:
    """BitunixExchange.place_market_order — executes market orders immediately."""

    async def _ensure_hedge_mode(self, exchange: BitunixExchange) -> None:
        """Switch to hedge mode so long and short can coexist."""
        try:
            await exchange.set_position_mode(hedge_mode=True)
        except Exception:
            pass  # may already be in hedge mode; ignore

    async def test_market_buy_returns_order_id(
        self, exchange: BitunixExchange
    ) -> None:
        """place_market_order(buy) returns a dict with a non-empty orderId."""
        await exchange.set_leverage(SYMBOL, leverage=DEFAULT_LEVERAGE)
        result = await exchange.place_market_order(
            SYMBOL, side="buy", quantity=MIN_QTY, reduce_only=False
        )
        assert result is not None, "place_market_order returned None"
        assert "orderId" in result, f"No 'orderId' in result: {result}"
        assert result["orderId"], "orderId is empty"

    async def test_market_buy_opens_long_position(
        self, exchange: BitunixExchange
    ) -> None:
        """After a market buy, get_positions reports long_qty > 0."""
        await exchange.set_leverage(SYMBOL, leverage=DEFAULT_LEVERAGE)
        await exchange.place_market_order(
            SYMBOL, side="buy", quantity=MIN_QTY, reduce_only=False
        )
        # Give exchange a moment for position to register
        await asyncio.sleep(2)
        long_qty, short_qty = await exchange.get_positions(SYMBOL)
        assert long_qty >= MIN_QTY, (
            f"Expected long_qty >= {MIN_QTY} after market buy, got {long_qty}"
        )

    async def test_market_sell_opens_short_position(
        self, exchange: BitunixExchange
    ) -> None:
        """After a market sell (open short), get_positions reports short_qty > 0."""
        await self._ensure_hedge_mode(exchange)
        await exchange.set_leverage(SYMBOL, leverage=DEFAULT_LEVERAGE)
        await exchange.place_market_order(
            SYMBOL, side="sell", quantity=MIN_QTY, reduce_only=False
        )
        await asyncio.sleep(2)
        long_qty, short_qty = await exchange.get_positions(SYMBOL)
        assert short_qty >= MIN_QTY, (
            f"Expected short_qty >= {MIN_QTY} after market sell, got {short_qty}"
        )

    async def test_market_sell_reduce_closes_long(
        self, exchange: BitunixExchange
    ) -> None:
        """A reduce_only market sell closes an existing long position."""
        await exchange.set_leverage(SYMBOL, leverage=DEFAULT_LEVERAGE)

        # Open a long
        await exchange.place_market_order(
            SYMBOL, side="buy", quantity=MIN_QTY, reduce_only=False
        )
        await asyncio.sleep(2)
        long_qty, _ = await exchange.get_positions(SYMBOL)
        assert long_qty >= MIN_QTY, f"Long not opened, got {long_qty}"

        # Close it
        close_result = await exchange.place_market_order(
            SYMBOL, side="sell", quantity=long_qty, reduce_only=True
        )
        assert close_result is not None

        await asyncio.sleep(2)
        long_qty_after, _ = await exchange.get_positions(SYMBOL)
        assert long_qty_after < long_qty, (
            f"Expected long to decrease from {long_qty} → 0, got {long_qty_after}"
        )

    async def test_market_buy_reduce_closes_short(
        self, exchange: BitunixExchange
    ) -> None:
        """A reduce_only market buy closes an existing short position."""
        await self._ensure_hedge_mode(exchange)
        await exchange.set_leverage(SYMBOL, leverage=DEFAULT_LEVERAGE)

        # Open a short
        await exchange.place_market_order(
            SYMBOL, side="sell", quantity=MIN_QTY, reduce_only=False
        )
        await asyncio.sleep(2)
        _, short_qty = await exchange.get_positions(SYMBOL)
        assert short_qty >= MIN_QTY, f"Short not opened, got {short_qty}"

        # Close it
        close_result = await exchange.place_market_order(
            SYMBOL, side="buy", quantity=short_qty, reduce_only=True
        )
        assert close_result is not None

        await asyncio.sleep(2)
        _, short_qty_after = await exchange.get_positions(SYMBOL)
        assert short_qty_after < short_qty, (
            f"Expected short to decrease from {short_qty} → 0, got {short_qty_after}"
        )


@skip_if_no_creds
@pytest.mark.integration
class TestFullPositionLifecycle:
    """End-to-end: open → inspect → close using the exchange adapter."""

    async def test_open_long_inspect_and_close(
        self, exchange: BitunixExchange
    ) -> None:
        """Full long lifecycle: market buy → verify position → market sell (reduce)."""
        await exchange.set_leverage(SYMBOL, leverage=DEFAULT_LEVERAGE)

        # ── step 1: verify flat start ──────────────────────────────────────
        long_start, short_start = await exchange.get_positions(SYMBOL)
        assert long_start == 0.0 and short_start == 0.0, (
            f"Expected flat start, got long={long_start} short={short_start}"
        )

        # ── step 2: open long ─────────────────────────────────────────────
        open_result = await exchange.place_market_order(
            SYMBOL, "buy", MIN_QTY, reduce_only=False
        )
        assert open_result and open_result.get("orderId"), "Open market buy failed"
        await asyncio.sleep(2)

        # ── step 3: verify position is open ──────────────────────────────
        long_open, short_open = await exchange.get_positions(SYMBOL)
        assert long_open >= MIN_QTY, (
            f"Expected long_qty >= {MIN_QTY} after open, got {long_open}"
        )
        assert short_open == 0.0, f"Unexpected short position: {short_open}"

        # ── step 4: close long ────────────────────────────────────────────
        close_result = await exchange.place_market_order(
            SYMBOL, "sell", long_open, reduce_only=True
        )
        assert close_result and close_result.get("orderId"), "Close market sell failed"
        await asyncio.sleep(2)

        # ── step 5: verify flat ───────────────────────────────────────────
        long_end, short_end = await exchange.get_positions(SYMBOL)
        assert long_end == 0.0, f"Long position still open after close: {long_end}"
        assert short_end == 0.0, f"Unexpected short after long close: {short_end}"

    async def test_open_short_inspect_and_close(
        self, exchange: BitunixExchange
    ) -> None:
        """Full short lifecycle: market sell → verify position → market buy (reduce)."""
        try:
            await exchange.set_position_mode(hedge_mode=True)
        except Exception:
            pass
        await exchange.set_leverage(SYMBOL, leverage=DEFAULT_LEVERAGE)

        # ── step 1: verify flat ───────────────────────────────────────────
        long_start, short_start = await exchange.get_positions(SYMBOL)
        assert long_start == 0.0 and short_start == 0.0

        # ── step 2: open short ────────────────────────────────────────────
        open_result = await exchange.place_market_order(
            SYMBOL, "sell", MIN_QTY, reduce_only=False
        )
        assert open_result and open_result.get("orderId")
        await asyncio.sleep(2)

        # ── step 3: verify position ───────────────────────────────────────
        long_open, short_open = await exchange.get_positions(SYMBOL)
        assert short_open >= MIN_QTY, (
            f"Expected short_qty >= {MIN_QTY} after open, got {short_open}"
        )
        assert long_open == 0.0, f"Unexpected long position: {long_open}"

        # ── step 4: close short ───────────────────────────────────────────
        close_result = await exchange.place_market_order(
            SYMBOL, "buy", short_open, reduce_only=True
        )
        assert close_result and close_result.get("orderId")
        await asyncio.sleep(2)

        # ── step 5: verify flat ───────────────────────────────────────────
        long_end, short_end = await exchange.get_positions(SYMBOL)
        assert short_end == 0.0, f"Short still open after close: {short_end}"
        assert long_end == 0.0, f"Unexpected long after short close: {long_end}"

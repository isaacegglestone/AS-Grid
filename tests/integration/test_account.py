"""
tests/integration/test_account.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration tests for account-state endpoints:
  - get_balance()
  - set_leverage()
  - set_position_mode() [hedge/one-way switch]

Cost: $0 — no orders placed, no positions opened.
"""

import pytest

from src.exchange.bitunix import BitunixExchange

from .conftest import DEFAULT_LEVERAGE, SYMBOL, skip_if_no_creds


@skip_if_no_creds
@pytest.mark.integration
class TestGetBalance:
    """BitunixExchange.get_balance — reads account balance over REST."""

    async def test_returns_dict_with_required_keys(
        self, exchange: BitunixExchange
    ) -> None:
        """Response contains 'total', 'free', 'used', and the raw model."""
        balance = await exchange.get_balance("USDT")
        for key in ("total", "free", "used"):
            assert key in balance, f"Missing key '{key}' in balance: {balance}"

    async def test_values_are_floats(self, exchange: BitunixExchange) -> None:
        """total / free / used are numeric floats."""
        balance = await exchange.get_balance("USDT")
        assert isinstance(balance["total"], float)
        assert isinstance(balance["free"], float)
        assert isinstance(balance["used"], float)

    async def test_account_has_usdt(self, exchange: BitunixExchange) -> None:
        """Account has more than $0 USDT (required for integration suite to be useful)."""
        balance = await exchange.get_balance("USDT")
        assert balance["total"] > 0, (
            f"Expected non-zero USDT balance but got total={balance['total']}. "
            "Please fund the integration test account with at least $100 USDT."
        )

    async def test_free_plus_used_approx_total(
        self, exchange: BitunixExchange
    ) -> None:
        """free + used ≈ total (within rounding tolerance)."""
        balance = await exchange.get_balance("USDT")
        reconstructed = balance["free"] + balance["used"]
        # Allow 0.01 USDT rounding difference from floating-point arithmetic
        assert abs(reconstructed - balance["total"]) < 0.01, (
            f"free ({balance['free']}) + used ({balance['used']}) "
            f"!= total ({balance['total']})"
        )

    async def test_free_and_used_non_negative(
        self, exchange: BitunixExchange
    ) -> None:
        """free and used are both ≥ 0."""
        balance = await exchange.get_balance("USDT")
        assert balance["free"] >= 0, f"Negative free balance: {balance['free']}"
        assert balance["used"] >= 0, f"Negative used balance: {balance['used']}"

    async def test_sufficient_balance_for_integration_suite(
        self, exchange: BitunixExchange
    ) -> None:
        """Warn (not fail) if free balance is below the recommended $50 threshold."""
        balance = await exchange.get_balance("USDT")
        if balance["free"] < 50:
            pytest.skip(
                f"Free balance ${balance['free']:.2f} is below $50 — "
                "order/position tests may fail. Fund the account and retry."
            )

    async def test_raw_model_present(self, exchange: BitunixExchange) -> None:
        """The 'raw' key holds an AccountBalanceEntry pydantic model."""
        from src.exchange.bitunix_models import AccountBalanceEntry

        balance = await exchange.get_balance("USDT")
        assert "raw" in balance
        assert isinstance(balance["raw"], AccountBalanceEntry)


@skip_if_no_creds
@pytest.mark.integration
class TestSetLeverage:
    """BitunixExchange.set_leverage — changes leverage via REST POST."""

    async def test_set_leverage_5x(self, exchange: BitunixExchange) -> None:
        """set_leverage(5) completes without raising an exception."""
        await exchange.set_leverage(SYMBOL, leverage=5)

    async def test_set_leverage_20x(self, exchange: BitunixExchange) -> None:
        """set_leverage(20) completes without raising an exception."""
        await exchange.set_leverage(SYMBOL, leverage=20)

    async def test_set_leverage_idempotent(self, exchange: BitunixExchange) -> None:
        """Calling set_leverage twice with the same value does not raise."""
        await exchange.set_leverage(SYMBOL, leverage=DEFAULT_LEVERAGE)
        await exchange.set_leverage(SYMBOL, leverage=DEFAULT_LEVERAGE)

    async def test_set_leverage_returns_none(self, exchange: BitunixExchange) -> None:
        """set_leverage is a fire-and-forget method that returns None."""
        result = await exchange.set_leverage(SYMBOL, leverage=5)
        assert result is None

    async def test_set_leverage_with_explicit_margin_coin(
        self, exchange: BitunixExchange
    ) -> None:
        """Explicit margin_coin='USDT' parameter is accepted."""
        await exchange.set_leverage(SYMBOL, leverage=5, margin_coin="USDT")


@skip_if_no_creds
@pytest.mark.integration
class TestSetPositionMode:
    """
    BitunixExchange.set_position_mode — switches hedge ↔ one-way mode.

    NOTE: This call is rejected by the exchange if any open position or
    pending order exists.  The autouse cleanup fixture ensures the account
    starts clean, so these tests should be safe to run.
    """

    async def test_set_hedge_mode(self, exchange: BitunixExchange) -> None:
        """Switch to hedge mode succeeds when no open orders/positions exist."""
        await exchange.set_position_mode(hedge_mode=True)

    async def test_set_oneway_mode(self, exchange: BitunixExchange) -> None:
        """Switch to one-way mode succeeds when no open orders/positions exist."""
        await exchange.set_position_mode(hedge_mode=False)

    async def test_restore_hedge_mode(self, exchange: BitunixExchange) -> None:
        """Round-trip: one-way → hedge mode restores correctly."""
        await exchange.set_position_mode(hedge_mode=False)
        await exchange.set_position_mode(hedge_mode=True)

    async def test_set_position_mode_returns_none(
        self, exchange: BitunixExchange
    ) -> None:
        """set_position_mode is a fire-and-forget method that returns None."""
        result = await exchange.set_position_mode(hedge_mode=True)
        assert result is None

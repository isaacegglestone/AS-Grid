"""
tests/integration/conftest.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Shared fixtures and cleanup logic for Bitunix live-exchange integration tests.

Setup
-----
1. Create ``.env.integration`` at the repo root (gitignored) with your
   Bitunix **futures** API credentials:

       BITUNIX_INT_API_KEY=your_api_key_here
       BITUNIX_INT_SECRET_KEY=your_secret_key_here

2. Fund the account with ~$100 USDT on Bitunix Futures.

3. Run the suite:

       pytest -m integration tests/integration/ -v -s

Isolation
---------
An ``autouse`` async fixture runs **before and after** every test:
  - Before: cancel all pending XRP/USDT orders, close all XRP/USDT positions.
  - After:  same cleanup — ensures no state leaks between tests.

Tests are designed to use only XRP/USDT with minimal notional (~$2–5 per order)
so that the $100 float covers the full suite with plenty of buffer for fees.
"""

from __future__ import annotations

import os

import pytest

from dotenv import load_dotenv

from src.exchange.bitunix import BitunixExchange, BitunixRestClient

# ---------------------------------------------------------------------------
# Credentials — loaded from .env.integration (gitignored) or shell env
# ---------------------------------------------------------------------------
load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env.integration"),
    override=False,
)

SYMBOL = "XRPUSDT"
MARGIN_COIN = "USDT"
DEFAULT_LEVERAGE = 5  # low leverage for safety during tests

_API_KEY: str = os.getenv("BITUNIX_INT_API_KEY", "")
_SECRET_KEY: str = os.getenv("BITUNIX_INT_SECRET_KEY", "")
_CREDS_PRESENT: bool = bool(_API_KEY and _SECRET_KEY)

# Convenience skip decorator — apply to any test class or function that needs live creds
skip_if_no_creds = pytest.mark.skipif(
    not _CREDS_PRESENT,
    reason="BITUNIX_INT_API_KEY / BITUNIX_INT_SECRET_KEY not set — skipping live test",
)


# ---------------------------------------------------------------------------
# Session-scoped exchange client (constructed once, credentials are stateless)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def exchange() -> BitunixExchange:
    """Return a live BitunixExchange pointed at the account under test."""
    if not _CREDS_PRESENT:
        pytest.skip("BITUNIX_INT_API_KEY / BITUNIX_INT_SECRET_KEY not set")
    return BitunixExchange(api_key=_API_KEY, secret_key=_SECRET_KEY)


# ---------------------------------------------------------------------------
# Low-level cleanup helpers (called by the autouse fixture)
# ---------------------------------------------------------------------------

async def _cancel_all_orders(exch: BitunixExchange) -> None:
    """Cancel every pending order for SYMBOL in one batch call."""
    if not _CREDS_PRESENT:
        return
    try:
        orders = await exch.get_open_orders(SYMBOL)
        if not orders:
            return
        client = BitunixRestClient(exch.api_key, exch.secret_key)
        order_ids = [{"orderId": o["orderId"]} for o in orders]
        await client.post(
            "/api/v1/futures/trade/cancel_orders",
            body={"symbol": SYMBOL, "orderList": order_ids},
        )
        print(f"\n[cleanup] cancelled {len(order_ids)} open orders for {SYMBOL}")
    except Exception as exc:
        print(f"\n[cleanup] cancel_all_orders error (non-fatal): {exc}")


async def _close_all_positions(exch: BitunixExchange) -> None:
    """Flatten any open long/short positions for SYMBOL with market orders."""
    if not _CREDS_PRESENT:
        return
    try:
        long_qty, short_qty = await exch.get_positions(SYMBOL)
        if long_qty > 0:
            await exch.place_market_order(
                SYMBOL, "sell", long_qty, reduce_only=True
            )
            print(f"\n[cleanup] closed {long_qty} long {SYMBOL}")
        if short_qty > 0:
            await exch.place_market_order(
                SYMBOL, "buy", short_qty, reduce_only=True
            )
            print(f"\n[cleanup] closed {short_qty} short {SYMBOL}")
    except Exception as exc:
        print(f"\n[cleanup] close_all_positions error (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Autouse per-test cleanup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
async def _clean_exchange_state(exchange: BitunixExchange) -> None:  # type: ignore[misc]
    """
    Run before AND after every integration test:
      - cancels all pending XRP/USDT orders
      - closes all XRP/USDT positions

    This ensures each test starts with a blank slate and leaves one too.
    """
    # ── pre-test cleanup ──────────────────────────────────────────────────────
    await _cancel_all_orders(exchange)
    await _close_all_positions(exchange)

    yield  # test runs here

    # ── post-test cleanup ─────────────────────────────────────────────────────
    await _cancel_all_orders(exchange)
    await _close_all_positions(exchange)


# ---------------------------------------------------------------------------
# Shared helper: get current mid-price from a recent kline
# ---------------------------------------------------------------------------

async def get_mid_price(exch: BitunixExchange) -> float:
    """Return the latest close price for SYMBOL (from 1-min kline)."""
    candles = await exch.get_klines(SYMBOL, interval="1min", limit=1)
    assert candles, "No kline data returned for mid-price lookup"
    return float(candles[-1]["close"])

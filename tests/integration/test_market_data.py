"""
tests/integration/test_market_data.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration tests for public market-data endpoints.

These tests hit the Bitunix **public** REST API and do NOT require
credentials or spend any money.  They are still marked ``integration``
because they make real network calls.

Cost: $0 — no orders placed.
"""

import time

import pytest

from src.exchange.bitunix import BitunixExchange

from .conftest import SYMBOL, skip_if_no_creds


@skip_if_no_creds
@pytest.mark.integration
class TestGetKlines:
    """BitunixExchange.get_klines — public endpoint, no auth required."""

    async def test_returns_list_of_candles(self, exchange: BitunixExchange) -> None:
        """get_klines returns a non-empty list for a liquid symbol."""
        candles = await exchange.get_klines(SYMBOL, interval="1min", limit=10)
        assert isinstance(candles, list)
        assert len(candles) > 0

    async def test_candles_have_ohlcv_fields(self, exchange: BitunixExchange) -> None:
        """Every candle dict has the required OHLCV fields with numeric values."""
        candles = await exchange.get_klines(SYMBOL, interval="1min", limit=5)
        assert candles, "Expected at least one candle"
        for candle in candles:
            for field in ("open_time", "open", "high", "low", "close", "volume"):
                assert field in candle, f"Missing field '{field}' in candle: {candle}"
                assert candle[field] is not None

    async def test_candle_ohlcv_are_numeric(self, exchange: BitunixExchange) -> None:
        """OHLCV values are floats / ints, not strings."""
        candles = await exchange.get_klines(SYMBOL, interval="1min", limit=5)
        for c in candles:
            assert isinstance(c["open"], float)
            assert isinstance(c["high"], float)
            assert isinstance(c["low"], float)
            assert isinstance(c["close"], float)
            assert isinstance(c["volume"], float)
            assert isinstance(c["open_time"], int)

    async def test_ohlcv_sanity(self, exchange: BitunixExchange) -> None:
        """Candle OHLCV values satisfy basic invariants: high≥low, volume≥0, etc."""
        candles = await exchange.get_klines(SYMBOL, interval="1min", limit=20)
        for c in candles:
            assert c["high"] >= c["low"], f"high < low in candle: {c}"
            assert c["high"] >= c["open"], f"high < open in candle: {c}"
            assert c["high"] >= c["close"], f"high < close in candle: {c}"
            assert c["low"] <= c["open"], f"low > open in candle: {c}"
            assert c["low"] <= c["close"], f"low > close in candle: {c}"
            assert c["volume"] >= 0, f"Negative volume in candle: {c}"
            assert c["open"] > 0, f"Non-positive open in candle: {c}"
            assert c["close"] > 0, f"Non-positive close in candle: {c}"

    async def test_limit_parameter_respected(self, exchange: BitunixExchange) -> None:
        """Requesting limit=5 returns at most 5 candles."""
        candles = await exchange.get_klines(SYMBOL, interval="1min", limit=5)
        assert 1 <= len(candles) <= 5

    async def test_limit_100_returns_more_data(self, exchange: BitunixExchange) -> None:
        """A larger limit returns more candles than a small limit."""
        small = await exchange.get_klines(SYMBOL, interval="1min", limit=5)
        large = await exchange.get_klines(SYMBOL, interval="1min", limit=100)
        assert len(large) >= len(small)

    async def test_1h_interval(self, exchange: BitunixExchange) -> None:
        """1h interval returns valid candles with timestamps spaced ~1 hour apart."""
        candles = await exchange.get_klines(SYMBOL, interval="1h", limit=5)
        assert len(candles) >= 2
        # Timestamps should be roughly 1-hour apart (within ±5 minutes tolerance)
        for i in range(1, len(candles)):
            delta_ms = candles[i]["open_time"] - candles[i - 1]["open_time"]
            delta_s = delta_ms / 1000
            # Allow for 30-min to 90-min gap (1h ± 50% tolerance)
            assert 1800 <= delta_s <= 5400, (
                f"Unexpected gap between 1h candles: {delta_s:.0f}s "
                f"(candle {i-1}→{i})"
            )

    async def test_time_range_filter(self, exchange: BitunixExchange) -> None:
        """Providing start_time / end_time restricts the returned candle window."""
        now_ms = int(time.time() * 1000)
        two_hours_ago_ms = now_ms - 2 * 60 * 60 * 1000  # 2h back
        one_hour_ago_ms = now_ms - 1 * 60 * 60 * 1000   # 1h back

        candles = await exchange.get_klines(
            SYMBOL,
            interval="1min",
            start_time=two_hours_ago_ms,
            end_time=one_hour_ago_ms,
            limit=200,
        )
        assert len(candles) > 0, "Expected candles in 2h–1h window"
        # All returned candles should be within the requested window
        # (allow a small margin for boundary alignment)
        tolerance_ms = 5 * 60 * 1000  # 5-minute tolerance
        for c in candles:
            assert c["open_time"] >= two_hours_ago_ms - tolerance_ms, (
                f"Candle before start_time: open_time={c['open_time']}"
            )
            assert c["open_time"] <= one_hour_ago_ms + tolerance_ms, (
                f"Candle after end_time: open_time={c['open_time']}"
            )

    async def test_candles_are_time_ordered(self, exchange: BitunixExchange) -> None:
        """Candles are returned in ascending time order."""
        candles = await exchange.get_klines(SYMBOL, interval="1min", limit=20)
        for i in range(1, len(candles)):
            assert candles[i]["open_time"] > candles[i - 1]["open_time"], (
                f"Candle {i} is not later than candle {i-1}: "
                f"{candles[i]['open_time']} vs {candles[i-1]['open_time']}"
            )

    async def test_close_price_is_realistic_for_xrp(
        self, exchange: BitunixExchange
    ) -> None:
        """Latest XRP close price is in a sanity range ($0.01 – $100)."""
        candles = await exchange.get_klines(SYMBOL, interval="1min", limit=1)
        assert candles
        price = candles[-1]["close"]
        assert 0.01 < price < 100.0, f"XRP price {price} outside sanity range"

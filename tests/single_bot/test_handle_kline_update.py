"""
tests/single_bot/test_handle_kline_update.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ``GridTradingBot.handle_kline_update``.

``GridTradingBot.__init__`` requires no network connections (it only stores
credentials), so we can instantiate it directly.  For tests that need the
buffer pre-warmed, we replace ``bot.candle_buffer`` with a pre-seeded buffer
or a ``MagicMock``.

Structure
---------
- TestHandleKlineUpdateGuards     – early-return conditions (bad payload)
- TestHandleKlineUpdateBufferWire – correct fields forwarded to candle_buffer
- TestHandleKlineUpdateSignals    – latest_signals refreshed on candle close
- TestHandleKlineUpdateErrors     – exceptions are swallowed not propagated
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
from collections import deque

import pytest

from src.single_bot.bitunix_bot import GridTradingBot
from src.single_bot.indicators import CandleBuffer, Candle, Signals


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TS_BASE = 1_740_000_000_000
TS_STEP = 15 * 60 * 1_000


def _make_bot() -> GridTradingBot:
    """Build a minimal GridTradingBot with dummy credentials (no I/O)."""
    return GridTradingBot(
        api_key="test_key",
        api_secret="test_secret",
        coin_name="XRP",
        grid_spacing=0.01,
        initial_quantity=10,
        leverage=5,
    )


def _kline_payload(ts: int, o="1.25", h="1.27", l="1.23", c="1.26",
                   b="2500000", q="3150000") -> dict:
    """Return a well-formed market_kline_15min WS payload dict."""
    return {
        "ch": "market_kline_15min",
        "symbol": "XRPUSDT",
        "ts": ts,
        "data": {"o": o, "h": h, "l": l, "c": c, "b": b, "q": q},
    }


def _seeded_buffer(n: int = 60) -> CandleBuffer:
    """Return a ``CandleBuffer`` pre-seeded with enough candles for signals."""
    buf = CandleBuffer(maxlen=200, interval="15min")
    for i in range(n):
        buf._closed.append(
            Candle(
                ts=TS_BASE + i * TS_STEP,
                open=1.0 + i * 0.001,
                high=1.0 + i * 0.001 + 0.05,
                low=1.0 + i * 0.001 - 0.05,
                close=1.0 + i * 0.001,
                volume=1_000_000.0,
            )
        )
    return buf


# ---------------------------------------------------------------------------
# TestHandleKlineUpdateGuards
# ---------------------------------------------------------------------------

class TestHandleKlineUpdateGuards:
    @pytest.mark.asyncio
    async def test_missing_ts_returns_early_no_error(self):
        bot = _make_bot()
        original_live = bot.candle_buffer._live
        # payload with ts=0 (falsy) → should return immediately
        payload = {"ch": "market_kline_15min", "symbol": "XRPUSDT", "ts": 0,
                   "data": {"o": "1.25", "h": "1.27", "l": "1.23", "c": "1.26",
                            "b": "2500000", "q": "3150000"}}
        await bot.handle_kline_update(payload)
        assert bot.candle_buffer._live is original_live  # buffer unchanged

    @pytest.mark.asyncio
    async def test_missing_data_key_returns_early(self):
        bot = _make_bot()
        # payload with no "data" key → `kline_data` is {} → falsy guard triggers
        payload = {"ch": "market_kline_15min", "ts": TS_BASE}
        await bot.handle_kline_update(payload)
        assert bot.candle_buffer._live is None

    @pytest.mark.asyncio
    async def test_empty_payload_returns_early(self):
        bot = _make_bot()
        await bot.handle_kline_update({})
        assert bot.candle_buffer._live is None

    @pytest.mark.asyncio
    async def test_latest_signals_unchanged_on_early_return(self):
        bot = _make_bot()
        bot.latest_signals = None
        await bot.handle_kline_update({"ts": 0, "data": {}})
        assert bot.latest_signals is None


# ---------------------------------------------------------------------------
# TestHandleKlineUpdateBufferWire
# ---------------------------------------------------------------------------

class TestHandleKlineUpdateBufferWire:
    @pytest.mark.asyncio
    async def test_ohlc_and_volume_forwarded_from_payload(self):
        """update() receives o/h/l/c from fields and q as volume."""
        bot = _make_bot()
        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.return_value = False
        bot.candle_buffer = mock_buf

        payload = _kline_payload(
            ts=TS_BASE, o="1.25", h="1.30", l="1.20", c="1.27", q="3200000"
        )
        await bot.handle_kline_update(payload)

        mock_buf.update.assert_called_once_with(
            o=1.25, h=1.30, l=1.20, c=1.27,
            volume=3_200_000.0,
            ts_ms=TS_BASE,
        )

    @pytest.mark.asyncio
    async def test_uses_q_field_not_b_for_volume(self):
        """b = coin count, q = USDT notional — we always pass q as volume."""
        bot = _make_bot()
        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.return_value = False
        bot.candle_buffer = mock_buf

        payload = _kline_payload(ts=TS_BASE, b="2500000", q="4000000")
        await bot.handle_kline_update(payload)

        _, kwargs = mock_buf.update.call_args
        assert kwargs["volume"] == pytest.approx(4_000_000.0)

    @pytest.mark.asyncio
    async def test_ts_passed_as_int(self):
        bot = _make_bot()
        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.return_value = False
        bot.candle_buffer = mock_buf

        await bot.handle_kline_update(_kline_payload(ts=TS_BASE))

        _, kwargs = mock_buf.update.call_args
        assert isinstance(kwargs["ts_ms"], int)
        assert kwargs["ts_ms"] == TS_BASE

    @pytest.mark.asyncio
    async def test_update_called_once_per_message(self):
        bot = _make_bot()
        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.return_value = False
        bot.candle_buffer = mock_buf

        await bot.handle_kline_update(_kline_payload(ts=TS_BASE))
        await bot.handle_kline_update(_kline_payload(ts=TS_BASE))

        assert mock_buf.update.call_count == 2


# ---------------------------------------------------------------------------
# TestHandleKlineUpdateSignals
# ---------------------------------------------------------------------------

class TestHandleKlineUpdateSignals:
    @pytest.mark.asyncio
    async def test_latest_signals_not_set_when_candle_not_closed(self):
        """No candle close → latest_signals must remain None."""
        bot = _make_bot()
        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.return_value = False
        bot.candle_buffer = mock_buf

        await bot.handle_kline_update(_kline_payload(ts=TS_BASE))
        assert bot.latest_signals is None

    @pytest.mark.asyncio
    async def test_latest_signals_updated_when_candle_closes(self):
        """On candle close, latest_signals should be set to whatever signals() returns."""
        bot = _make_bot()
        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.return_value = True  # simulate candle close
        fake_signals = Signals(adx=30.0, rsi=55.0, bb_width=0.03, close=1.26)
        mock_buf.signals.return_value = fake_signals
        bot.candle_buffer = mock_buf

        await bot.handle_kline_update(_kline_payload(ts=TS_BASE))

        mock_buf.signals.assert_called_once()
        assert bot.latest_signals is fake_signals

    @pytest.mark.asyncio
    async def test_latest_signals_none_when_buffer_not_warmed(self):
        """signals() returns None during warm-up → latest_signals stays None."""
        bot = _make_bot()
        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.return_value = True
        mock_buf.signals.return_value = None   # warm-up period
        bot.candle_buffer = mock_buf

        initial = bot.latest_signals
        await bot.handle_kline_update(_kline_payload(ts=TS_BASE))
        assert bot.latest_signals is initial  # unchanged

    @pytest.mark.asyncio
    async def test_real_buffer_signals_populated_after_two_candles(self):
        """
        Integration flavour: use a real pre-seeded buffer; send two successive
        messages with different timestamps.  The second message triggers a
        candle close; latest_signals should be non-None (enough history).
        """
        bot = _make_bot()
        bot.candle_buffer = _seeded_buffer(n=60)

        # First message — opens in-progress candle
        await bot.handle_kline_update(_kline_payload(ts=TS_BASE + 5_000))
        assert bot.latest_signals is None  # no close yet

        # Second message — new ts → closes previous, triggers signals()
        await bot.handle_kline_update(
            _kline_payload(ts=TS_BASE + TS_STEP, c="1.28", q="2800000")
        )
        # Buffer had 60 candles; after close it has 61 → enough for signals
        assert bot.latest_signals is not None
        assert isinstance(bot.latest_signals, Signals)
        assert 0.0 <= bot.latest_signals.rsi <= 100.0

    @pytest.mark.asyncio
    async def test_signals_close_reflects_ws_close_price(self):
        """latest_signals.close should match the ``c`` field from the WS message
        that triggered the *next* candle (the committed candle's close)."""
        bot = _make_bot()
        bot.candle_buffer = _seeded_buffer(n=60)

        close_price = "1.3500"
        # Open candle 1 at ts1
        await bot.handle_kline_update(
            _kline_payload(ts=TS_BASE + 1_000, c=close_price, q="3000000")
        )
        # Open candle 2 at ts2 — commits candle 1
        await bot.handle_kline_update(
            _kline_payload(ts=TS_BASE + TS_STEP + 1_000, c="1.36", q="3100000")
        )
        assert bot.latest_signals is not None
        # The signals are computed from _closed which now includes the committed candle
        # whose close was 1.35
        assert bot.latest_signals.close == pytest.approx(float(close_price), rel=1e-4)


# ---------------------------------------------------------------------------
# TestHandleKlineUpdateErrors
# ---------------------------------------------------------------------------

class TestHandleKlineUpdateErrors:
    @pytest.mark.asyncio
    async def test_exception_in_update_does_not_propagate(self):
        """Any exception inside handle_kline_update must be swallowed."""
        bot = _make_bot()
        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.side_effect = RuntimeError("simulated crash")
        bot.candle_buffer = mock_buf

        # Should not raise
        await bot.handle_kline_update(_kline_payload(ts=TS_BASE))

    @pytest.mark.asyncio
    async def test_exception_in_signals_does_not_propagate(self):
        bot = _make_bot()
        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.return_value = True
        mock_buf.signals.side_effect = ValueError("signals broke")
        bot.candle_buffer = mock_buf

        await bot.handle_kline_update(_kline_payload(ts=TS_BASE))

    @pytest.mark.asyncio
    async def test_corrupt_ohlc_string_does_not_propagate(self):
        """Non-numeric OHLCV values → float() raises → must be caught."""
        bot = _make_bot()
        payload = _kline_payload(ts=TS_BASE, o="NaN-not-parseable")
        # float("NaN-not-parseable") fails → should be swallowed
        await bot.handle_kline_update(payload)

    @pytest.mark.asyncio
    async def test_latest_signals_unchanged_after_exception(self):
        bot = _make_bot()
        initial = object()
        bot.latest_signals = initial  # type: ignore[assignment]

        mock_buf = MagicMock(spec=CandleBuffer)
        mock_buf.update.side_effect = RuntimeError("boom")
        bot.candle_buffer = mock_buf

        await bot.handle_kline_update(_kline_payload(ts=TS_BASE))
        assert bot.latest_signals is initial  # unchanged after error

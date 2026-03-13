"""
tests/single_bot/test_hedge_repair_bot.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the HedgeRepairBot state machine.

Uses SimulatedExchange so no network calls are made.  Tests cover the full
lifecycle: IDLE → BOTH_OPEN → REPAIRING → IDLE, DCA repair, trailing stop,
dynamic threshold, zig-zag detection, and periodic injection.
"""

from __future__ import annotations

import pytest

from src.exchange.simulated import SimulatedExchange
from src.single_bot.hedge_repair_bot import HedgeRepairBot, HedgePosition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYMBOL = "XRPUSDT"


def _make_bot(
    balance: float = 1000.0,
    spot: float = 1000.0,
    leverage: int = 5,
    entry_pct: float = 0.10,
    lookback: int = 5,
    initial_price: float = 2.0,
    **overrides: object,
) -> HedgeRepairBot:
    """Create a HedgeRepairBot with a SimulatedExchange and sensible test defaults."""
    sim = SimulatedExchange(initial_balance=balance, fee_pct=0.0006)
    sim._leverage[SYMBOL] = leverage
    sim._hedge_mode = True
    sim._last_price = initial_price  # needed for place_market_order

    config = {
        "symbol": "XRP",
        "leverage": leverage,
        "entry_pct": entry_pct,
        "fee_pct": 0.0006,
        "profit_threshold": 5.0,
        "trailing_distance": 1.0,
        "lookback_candles": lookback,
        "zig_zag_candles": 3,
        "zig_zag_threshold": 0.01,
        "dca_multiplier": 1.0,
        "liq_proximity_pct": 0.80,
        "dynamic_threshold": False,
        "net_profit_target": 0.10,
        "spot_reserve": spot,
        "periodic_injection_usd": 0.0,
        "poll_interval_secs": 0,
        "enable_notifications": False,
        **overrides,
    }
    bot = HedgeRepairBot(sim, config)
    return bot


def _candle(close: float, high: float = 0.0, low: float = 0.0) -> dict:
    """Build a minimal candle dict."""
    if high <= 0:
        high = close * 1.001
    if low <= 0:
        low = close * 0.999
    return {
        "open_time": 0, "open": close,
        "high": high, "low": low,
        "close": close, "volume": 1000,
    }


def _candle_range(close: float, spread: float = 0.0) -> dict:
    """Build a candle with specified spread around close."""
    return _candle(close, high=close + spread, low=close - spread)


async def _feed(bot: HedgeRepairBot, candle: dict) -> None:
    """Feed a candle through on_price_update then process_candle (mirrors run_simulation)."""
    sim = bot.exchange
    sim.on_price_update(SYMBOL, candle["close"], high=candle["high"], low=candle["low"])
    await bot.process_candle(candle)


# ---------------------------------------------------------------------------
# State Machine — Basic Flow
# ---------------------------------------------------------------------------

class TestStateMachine:
    async def test_starts_idle(self):
        bot = _make_bot()
        assert bot.state == HedgeRepairBot.IDLE

    async def test_needs_lookback_before_entry(self):
        bot = _make_bot(lookback=5)
        # Less than lookback candles — should stay IDLE
        for i in range(3):
            await _feed(bot, _candle(2.0))
        assert bot.state == HedgeRepairBot.IDLE

    async def test_entry_on_mid_cross(self):
        # High threshold so trailing doesn't fire on the entry candle
        bot = _make_bot(lookback=3, profit_threshold=1000.0)
        # 2 candles — not yet enough for lookback=3
        await _feed(bot, _candle(2.0, high=2.1, low=1.9))
        await _feed(bot, _candle(2.0, high=2.1, low=1.9))
        assert bot.state == HedgeRepairBot.IDLE
        # 3rd candle completes lookback, mid=(2.1+1.9)/2=2.0, candle crosses mid
        await _feed(bot, _candle(2.0, high=2.1, low=1.9))
        assert bot.state == HedgeRepairBot.BOTH_OPEN
        assert bot.long_pos is not None
        assert bot.short_pos is not None

    async def test_full_cycle_to_idle(self):
        """Entry → trail one side → repair → trail remaining → IDLE."""
        # trailing_distance=10 prevents same-candle trigger
        bot = _make_bot(lookback=2, entry_pct=0.10, trailing_distance=10.0)

        # Candle 1+2: seed lookback, entry on candle 2 at mid=2.0
        # qty = 1000*0.10*5/2.0 = 250
        await _feed(bot, _candle(2.0, high=2.05, low=1.95))
        await _feed(bot, _candle(2.0, high=2.05, low=1.95))
        assert bot.state == HedgeRepairBot.BOTH_OPEN
        assert bot.long_pos is not None

        # Candle 3: pump up → long peak = (2.03-2.0)*250 = 7.5 > threshold(5.0) → trailing active
        # l_worst = (2.01-2.0)*250 = 2.5, NOT <= 7.5-10 = -2.5 → no trigger
        await _feed(bot, _candle(2.02, high=2.03, low=2.01))
        assert bot.long_pos.trailing_active is True

        # Candle 4: pullback → l_worst = (1.96-2.0)*250 = -10 <= 7.5-10 = -2.5 → triggers
        # Short builds peak = (2.0-1.96)*250 = 10.0 → trailing active, but
        # s_worst = (2.0-1.99)*250 = 2.5, NOT <= 10.0-10.0 = 0 → no trigger
        await _feed(bot, _candle(1.97, high=1.99, low=1.96))
        assert bot.state == HedgeRepairBot.REPAIRING
        assert bot.long_pos is None
        assert bot.short_pos is not None

        # Candle 5: pump down → short peak = (2.0-1.93)*250 = 17.5
        await _feed(bot, _candle(1.94, high=1.96, low=1.93))
        assert bot.short_pos.trailing_active is True

        # Candle 6: pull back up → s_worst = (2.0-2.02)*250 = -5 <= 17.5-10 = 7.5 → triggers
        await _feed(bot, _candle(2.01, high=2.02, low=1.99))
        assert bot.state == HedgeRepairBot.IDLE
        assert bot.cycles_completed == 1


# ---------------------------------------------------------------------------
# Trailing Stop Logic
# ---------------------------------------------------------------------------

class TestTrailingStop:
    def test_not_triggered_when_inactive(self):
        bot = _make_bot()
        pos = HedgePosition(side="long", qty=100, avg_entry=2.0, margin=40)
        pos.peak_profit = 10.0
        pos.trailing_active = False
        assert bot._trailing_triggered(pos, -5.0) is False

    def test_triggered_when_active_and_dropped(self):
        bot = _make_bot()
        pos = HedgePosition(side="long", qty=100, avg_entry=2.0, margin=40)
        pos.peak_profit = 10.0
        pos.trailing_active = True
        # worst profit = 8.5 < peak(10) - trail(1.0) = 9.0 ✓
        assert bot._trailing_triggered(pos, 8.5) is True

    def test_not_triggered_if_still_above(self):
        bot = _make_bot()
        pos = HedgePosition(side="long", qty=100, avg_entry=2.0, margin=40)
        pos.peak_profit = 10.0
        pos.trailing_active = True
        # worst = 9.5 > 10 - 1.0 = 9.0 → not triggered
        assert bot._trailing_triggered(pos, 9.5) is False


# ---------------------------------------------------------------------------
# Dynamic Threshold
# ---------------------------------------------------------------------------

class TestDynamicThreshold:
    def test_fixed_mode(self):
        bot = _make_bot(dynamic_threshold=False, profit_threshold=5.0)
        assert bot._effective_threshold() == 5.0

    def test_dynamic_mode_no_fees(self):
        bot = _make_bot(dynamic_threshold=True, net_profit_target=0.10)
        bot.cycle_fees = 0.0
        assert bot._effective_threshold() == pytest.approx(0.10)

    def test_dynamic_mode_with_fees(self):
        bot = _make_bot(dynamic_threshold=True, net_profit_target=0.10)
        bot.cycle_fees = 2.50
        assert bot._effective_threshold() == pytest.approx(2.60)


# ---------------------------------------------------------------------------
# Profit Calculation
# ---------------------------------------------------------------------------

class TestProfitCalc:
    def test_long_profit(self):
        bot = _make_bot()
        pos = HedgePosition(side="long", qty=100, avg_entry=2.0, margin=40)
        assert bot._calc_profit(pos, 2.5) == pytest.approx(50.0)

    def test_long_loss(self):
        bot = _make_bot()
        pos = HedgePosition(side="long", qty=100, avg_entry=2.0, margin=40)
        assert bot._calc_profit(pos, 1.5) == pytest.approx(-50.0)

    def test_short_profit(self):
        bot = _make_bot()
        pos = HedgePosition(side="short", qty=100, avg_entry=2.0, margin=40)
        assert bot._calc_profit(pos, 1.5) == pytest.approx(50.0)

    def test_short_loss(self):
        bot = _make_bot()
        pos = HedgePosition(side="short", qty=100, avg_entry=2.0, margin=40)
        assert bot._calc_profit(pos, 2.5) == pytest.approx(-50.0)


# ---------------------------------------------------------------------------
# Stop Price
# ---------------------------------------------------------------------------

class TestStopPrice:
    def test_long_stop(self):
        bot = _make_bot()
        pos = HedgePosition(side="long", qty=100, avg_entry=2.0, margin=40)
        pos.peak_profit = 10.0
        # stop = entry + (peak - trail) / qty = 2.0 + (10-1)/100 = 2.09
        assert bot._stop_price(pos) == pytest.approx(2.09)

    def test_short_stop(self):
        bot = _make_bot()
        pos = HedgePosition(side="short", qty=100, avg_entry=2.0, margin=40)
        pos.peak_profit = 10.0
        # stop = entry - (peak - trail) / qty = 2.0 - (10-1)/100 = 1.91
        assert bot._stop_price(pos) == pytest.approx(1.91)


# ---------------------------------------------------------------------------
# Lookback Midpoint
# ---------------------------------------------------------------------------

class TestLookback:
    def test_mid_returns_none_when_insufficient(self):
        bot = _make_bot(lookback=5)
        bot._price_ring = [{"high": 2.0, "low": 1.9, "close": 1.95}] * 3
        assert bot._lookback_mid() is None

    def test_mid_calculation(self):
        bot = _make_bot(lookback=3)
        bot._price_ring = [
            {"high": 2.2, "low": 1.8, "close": 2.0},
            {"high": 2.4, "low": 1.9, "close": 2.1},
            {"high": 2.3, "low": 1.7, "close": 2.0},
        ]
        # hi=2.4, lo=1.7 → mid = (2.4+1.7)/2 = 2.05
        assert bot._lookback_mid() == pytest.approx(2.05)


# ---------------------------------------------------------------------------
# Zig-Zag Detection
# ---------------------------------------------------------------------------

class TestZigZag:
    def test_flat_market_detected(self):
        bot = _make_bot()
        bot.zig_zag_candles = 3
        bot.zig_zag_threshold = 0.02
        # Tight range: high=2.01, low=1.99 → range/mid = 0.02/2.0 = 0.01 < 0.02
        bot._price_ring = [
            {"high": 2.01, "low": 1.99, "close": 2.0},
            {"high": 2.01, "low": 1.99, "close": 2.0},
            {"high": 2.01, "low": 1.99, "close": 2.0},
        ]
        assert bot._check_zig_zag() is True

    def test_volatile_market_not_detected(self):
        bot = _make_bot()
        bot.zig_zag_candles = 3
        bot.zig_zag_threshold = 0.01
        # Wide range: high=2.2, low=1.8 → range/mid = 0.4/2.0 = 0.20 > 0.01
        bot._price_ring = [
            {"high": 2.2, "low": 1.8, "close": 2.0},
            {"high": 2.2, "low": 1.8, "close": 2.0},
            {"high": 2.2, "low": 1.8, "close": 2.0},
        ]
        assert bot._check_zig_zag() is False

    def test_insufficient_data(self):
        bot = _make_bot()
        bot.zig_zag_candles = 5
        bot._price_ring = [{"high": 2.0, "low": 1.9, "close": 1.95}] * 3
        assert bot._check_zig_zag() is False


# ---------------------------------------------------------------------------
# DCA Repair
# ---------------------------------------------------------------------------

class TestDCARepair:
    async def test_dca_updates_position(self):
        bot = _make_bot(balance=1000.0, leverage=5, entry_pct=0.10, initial_price=2.0)
        sim = bot.exchange

        # Create a long position manually
        await sim.place_market_order(SYMBOL, "buy", 100)
        bot.long_pos = HedgePosition(
            side="long", qty=100, avg_entry=2.0,
            margin=40.0, dca_count=0,
        )

        # DCA at 1.5
        sim._last_price = 1.5
        result = await bot._dca_repair(bot.long_pos, 1.5)
        assert result is True
        assert bot.long_pos.qty == 200
        assert bot.long_pos.avg_entry == pytest.approx(1.75, abs=0.01)
        assert bot.long_pos.dca_count == 1
        assert bot.total_dca_count == 1

    async def test_dca_resets_trailing(self):
        bot = _make_bot(balance=1000.0, leverage=5, entry_pct=0.10, initial_price=2.0)
        sim = bot.exchange

        await sim.place_market_order(SYMBOL, "buy", 100)
        bot.long_pos = HedgePosition(
            side="long", qty=100, avg_entry=2.0,
            margin=40.0, trailing_active=True, peak_profit=10.0,
        )

        sim._last_price = 1.5
        await bot._dca_repair(bot.long_pos, 1.5)
        assert bot.long_pos.trailing_active is False

    async def test_dca_short_updates_position(self):
        bot = _make_bot(balance=1000.0, leverage=5, entry_pct=0.10, initial_price=2.0)
        sim = bot.exchange

        # Create a short position manually
        await sim.place_market_order(SYMBOL, "sell", 100)
        bot.short_pos = HedgePosition(
            side="short", qty=100, avg_entry=2.0,
            margin=40.0, dca_count=0,
        )

        # DCA at 2.5 (price moved against the short)
        sim._last_price = 2.5
        result = await bot._dca_repair(bot.short_pos, 2.5)
        assert result is True
        assert bot.short_pos.qty == 200
        # Weighted avg: (100*2.0 + 100*2.5) / 200 = 2.25
        assert bot.short_pos.avg_entry == pytest.approx(2.25, abs=0.01)
        assert bot.short_pos.dca_count == 1
        assert bot.total_dca_count == 1


# ---------------------------------------------------------------------------
# Simulation Mode
# ---------------------------------------------------------------------------

class TestSimulation:
    async def test_simulation_runs(self):
        bot = _make_bot(lookback=2, balance=1000.0)

        candles = []
        # Seed + entry
        for i in range(5):
            candles.append(_candle(2.0, high=2.05, low=1.95))
        # Pump long
        for i in range(5):
            p = 2.0 + (i + 1) * 0.02
            candles.append(_candle(p, high=p + 0.02, low=p - 0.01))
        # Pullback
        for i in range(5):
            p = 2.1 - (i + 1) * 0.02
            candles.append(_candle(p, high=p + 0.01, low=p - 0.02))
        # Flat
        for i in range(10):
            candles.append(_candle(2.0, high=2.01, low=1.99))

        result = await bot.run_simulation(candles)
        assert "return_pct" in result
        assert "total_fees" in result
        assert "liquidated" in result
        assert result["trades"] > 0

    async def test_simulation_rejects_live_exchange(self):
        """run_simulation should raise if exchange is not SimulatedExchange."""
        from unittest.mock import AsyncMock, MagicMock

        mock_exchange = MagicMock()
        bot = HedgeRepairBot(mock_exchange, {"symbol": "XRP"})

        with pytest.raises(TypeError, match="SimulatedExchange"):
            await bot.run_simulation([_candle(2.0)])


# ---------------------------------------------------------------------------
# Status Summary
# ---------------------------------------------------------------------------

class TestStatusSummary:
    def test_idle_summary(self):
        bot = _make_bot()
        s = bot.status_summary()
        assert "IDLE" in s
        assert "Cycles: 0" in s

    def test_summary_with_positions(self):
        bot = _make_bot()
        bot.state = HedgeRepairBot.BOTH_OPEN
        bot.long_pos = HedgePosition(side="long", qty=100, avg_entry=2.0, margin=40)
        bot.short_pos = HedgePosition(side="short", qty=100, avg_entry=2.0, margin=40)
        s = bot.status_summary()
        assert "Long:" in s
        assert "Short:" in s


# ---------------------------------------------------------------------------
# Periodic Injection
# ---------------------------------------------------------------------------

class TestPeriodicInjection:
    async def test_injection_adds_to_spot(self):
        bot = _make_bot(
            periodic_injection_usd=480.0,
            injection_futures_pct=0.5,
        )
        bot.injection_interval_secs = 0.001  # trigger immediately

        import time
        bot._last_injection_time = time.time() - 1  # expired

        await _feed(bot, _candle(2.0))

        assert bot.injection_count == 1
        assert bot.total_injected == 480.0
        assert bot.spot_balance > 1000.0  # spot_reserve + 240

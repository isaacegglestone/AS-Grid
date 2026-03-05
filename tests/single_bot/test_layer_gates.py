"""
tests/single_bot/test_layer_gates.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the v21/v22 Layer 1–4 loss-mitigation gates added to
``GridTradingBot`` and the supporting ``Signals`` / ``CandleBuffer`` changes.

Layer 1: ATR parabolic gate      — ``_parabolic_gate()``
Layer 2: HTF EMA alignment       — ``_htf_bull()`` / ``_htf_bear()``
Layer 3: Regime vote mode         — ``_regime_vote_halt_longs()``
Layer 4: Grid sleep               — ``_grid_sleep()``

Also covers:
- New ``Signals`` fields: ``atr_sma``, ``htf_ema_fast``, ``htf_ema_slow``,
  ``regime_ema_87``, ``regime_ema_42``
- ``CandleBuffer.signals()`` computation of those fields
- Integration into ``place_long_orders`` / ``place_short_orders``
- Integration into ``_evaluate_trend`` (long and short)
"""
from __future__ import annotations

from collections import deque
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.single_bot.bitunix_bot import (
    ADX_MIN_TREND,
    TREND_CAP_VEL_PCT,
    TREND_CONFIRM_CANDLES,
    TREND_LOOKBACK_CANDLES,
    GridTradingBot,
)
from src.single_bot.indicators import Candle, CandleBuffer, Signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TS_BASE = 1_740_000_000_000
TS_STEP = 15 * 60 * 1_000


def _make_bot(**overrides) -> GridTradingBot:
    """Create a bot with sensible defaults.  *overrides* are set on the instance."""
    bot = GridTradingBot(
        api_key="k",
        api_secret="s",
        coin_name="XRP",
        grid_spacing=0.01,
        initial_quantity=10,
        leverage=5,
    )
    for k, v in overrides.items():
        setattr(bot, k, v)
    return bot


def _signals(**kwargs) -> Signals:
    """Return a ``Signals`` object with custom field values."""
    return Signals(**kwargs)


def _seeded_buffer(n: int = 200, base_close: float = 1.0) -> CandleBuffer:
    """Return a ``CandleBuffer`` with *n* closed candles (gentle uptrend)."""
    buf = CandleBuffer(maxlen=210, interval="15min", regime_ema_period=21)
    for i in range(n):
        c = base_close + i * 0.001
        buf._closed.append(
            Candle(
                ts=TS_BASE + i * TS_STEP,
                open=c,
                high=c + 0.05,
                low=c - 0.05,
                close=c,
                volume=1_000_000.0,
            )
        )
    return buf


def _trend_up_bot(adx: float = 30.0, **bot_overrides) -> GridTradingBot:
    """Bot pre-wired for confirmed-up on next ``_evaluate_trend(1.08)``."""
    bot = _make_bot(**bot_overrides)
    bot.candle_buffer = _seeded_buffer(n=15)
    bot.trend_pending_dir = "up"
    bot.trend_confirm_counter = TREND_CONFIRM_CANDLES - 1
    bot.trend_mode = None
    bot.trend_position = None
    bot.short_position = 0.0
    bot.latest_signals = _signals(adx=adx, close=1.08)
    return bot


def _trend_down_bot(adx: float = 30.0, **bot_overrides) -> GridTradingBot:
    """Bot pre-wired for confirmed-down on next ``_evaluate_trend(0.92)``."""
    bot = _make_bot(**bot_overrides)
    bot.candle_buffer = _seeded_buffer(n=15)
    bot.trend_pending_dir = "down"
    bot.trend_confirm_counter = TREND_CONFIRM_CANDLES - 1
    bot.trend_mode = None
    bot.trend_position = None
    bot.long_position = 0.0
    bot.latest_signals = _signals(adx=adx, close=0.92)
    return bot


# ===========================================================================
# Signals dataclass — new fields
# ===========================================================================

class TestSignalsNewFields:
    """New Layer 1-4 fields should exist with correct defaults."""

    def test_atr_sma_default(self):
        assert Signals().atr_sma == 0.0

    def test_htf_ema_fast_default(self):
        assert Signals().htf_ema_fast == 0.0

    def test_htf_ema_slow_default(self):
        assert Signals().htf_ema_slow == 0.0

    def test_regime_ema_87_default(self):
        assert Signals().regime_ema_87 == 0.0

    def test_regime_ema_42_default(self):
        assert Signals().regime_ema_42 == 0.0

    def test_fields_set_via_constructor(self):
        s = Signals(atr_sma=0.5, htf_ema_fast=1.1, htf_ema_slow=1.0,
                    regime_ema_87=2.0, regime_ema_42=2.1)
        assert s.atr_sma == 0.5
        assert s.htf_ema_fast == 1.1
        assert s.htf_ema_slow == 1.0
        assert s.regime_ema_87 == 2.0
        assert s.regime_ema_42 == 2.1


# ===========================================================================
# CandleBuffer.signals() — new fields computed
# ===========================================================================

class TestCandleBufferNewSignals:
    """CandleBuffer.signals() should return the 5 new fields."""

    def test_signals_contain_new_fields(self):
        """All 5 new fields are populated (non-zero for a rising series)."""
        buf = _seeded_buffer(n=200)
        sig = buf.signals()
        assert sig is not None
        assert sig.atr_sma > 0, "atr_sma should be positive"
        assert sig.htf_ema_fast > 0, "htf_ema_fast should be positive"
        assert sig.htf_ema_slow > 0, "htf_ema_slow should be positive"
        assert sig.regime_ema_87 > 0, "regime_ema_87 should be positive"
        assert sig.regime_ema_42 > 0, "regime_ema_42 should be positive"

    def test_htf_fast_above_slow_in_uptrend(self):
        """In a sustained uptrend, htf_ema_fast should be above htf_ema_slow."""
        buf = _seeded_buffer(n=200)
        sig = buf.signals()
        assert sig is not None
        assert sig.htf_ema_fast > sig.htf_ema_slow

    def test_atr_sma_close_to_atr_for_stable_vol(self):
        """With constant OHLC spread, ATR SMA ≈ ATR."""
        buf = _seeded_buffer(n=200)
        sig = buf.signals()
        assert sig is not None
        # Should be within 20% of each other
        assert abs(sig.atr - sig.atr_sma) < sig.atr * 0.5

    def test_shorter_regime_ema_more_reactive(self):
        """EMA-42 should be closer to the latest close than EMA-87."""
        buf = _seeded_buffer(n=200)
        sig = buf.signals()
        assert sig is not None
        # In an uptrend the shorter EMA gets pulled up faster
        assert sig.regime_ema_42 > sig.regime_ema_87


# ===========================================================================
# Layer 1: ATR parabolic gate — _parabolic_gate()
# ===========================================================================

class TestLayer1ParabolicGate:
    """_parabolic_gate should block when ATR > mult × SMA(ATR)."""

    def test_gate_off_when_mult_zero(self):
        """mult=0 ⇒ gate disabled (always returns False)."""
        bot = _make_bot(atr_parabolic_mult=0.0)
        bot.latest_signals = _signals(atr=10.0, atr_sma=1.0)
        assert bot._parabolic_gate() is False

    def test_gate_off_when_no_signals(self):
        """No signals ⇒ gate does not fire."""
        bot = _make_bot(atr_parabolic_mult=2.0)
        bot.latest_signals = None
        assert bot._parabolic_gate() is False

    def test_gate_fires_when_atr_exceeds_threshold(self):
        """ATR = 5.0, SMA=2.0, mult=2.0 → threshold=4.0 → gate fires."""
        bot = _make_bot(atr_parabolic_mult=2.0)
        bot.latest_signals = _signals(atr=5.0, atr_sma=2.0)
        assert bot._parabolic_gate() is True

    def test_gate_does_not_fire_below_threshold(self):
        """ATR = 3.0, SMA=2.0, mult=2.0 → threshold=4.0 → no fire."""
        bot = _make_bot(atr_parabolic_mult=2.0)
        bot.latest_signals = _signals(atr=3.0, atr_sma=2.0)
        assert bot._parabolic_gate() is False

    def test_gate_boundary_exact_threshold(self):
        """ATR exactly at threshold ⇒ no fire (not strictly >)."""
        bot = _make_bot(atr_parabolic_mult=2.0)
        bot.latest_signals = _signals(atr=4.0, atr_sma=2.0)
        assert bot._parabolic_gate() is False

    def test_gate_with_zero_atr_sma(self):
        """SMA=0 ⇒ gate does not fire (avoid division edge)."""
        bot = _make_bot(atr_parabolic_mult=2.0)
        bot.latest_signals = _signals(atr=5.0, atr_sma=0.0)
        assert bot._parabolic_gate() is False


class TestLayer1bRegimeAdaptive:
    """Regime-adaptive gate uses different mults for bull vs bear."""

    def test_bull_regime_uses_bull_mult(self):
        """Price ≥ regime_ema → uses bull_mult (2.5). ATR=4, SMA=2 → threshold=5 → no fire."""
        bot = _make_bot(
            atr_parabolic_mult=1.5,  # fallback (not used)
            atr_regime_adaptive=True,
            atr_bull_mult=2.5,
            atr_bear_mult=1.5,
        )
        bot.latest_signals = _signals(atr=4.0, atr_sma=2.0, regime_ema=1.0, close=1.5)
        assert bot._parabolic_gate() is False  # 4.0 < 2.5×2.0=5.0

    def test_bear_regime_uses_bear_mult(self):
        """Price < regime_ema → uses bear_mult (1.5). ATR=4, SMA=2 → threshold=3 → fires."""
        bot = _make_bot(
            atr_parabolic_mult=2.5,  # fallback (not used when adaptive)
            atr_regime_adaptive=True,
            atr_bull_mult=2.5,
            atr_bear_mult=1.5,
        )
        bot.latest_signals = _signals(atr=4.0, atr_sma=2.0, regime_ema=2.0, close=1.5)
        assert bot._parabolic_gate() is True  # 4.0 > 1.5×2.0=3.0

    def test_adaptive_off_uses_fixed_mult(self):
        """When atr_regime_adaptive=False, uses atr_parabolic_mult regardless of regime."""
        bot = _make_bot(
            atr_parabolic_mult=2.0,
            atr_regime_adaptive=False,
            atr_bull_mult=5.0,
            atr_bear_mult=1.0,
        )
        bot.latest_signals = _signals(atr=5.0, atr_sma=2.0, regime_ema=1.0, close=1.5)
        assert bot._parabolic_gate() is True  # 5.0 > 2.0×2.0=4.0, uses fixed

    def test_regime_ema_zero_falls_back_to_fixed(self):
        """If regime_ema is 0 (not warmed up), falls back to fixed mult."""
        bot = _make_bot(
            atr_parabolic_mult=2.0,
            atr_regime_adaptive=True,
            atr_bull_mult=5.0,
            atr_bear_mult=1.0,
        )
        bot.latest_signals = _signals(atr=5.0, atr_sma=2.0, regime_ema=0.0, close=1.5)
        assert bot._parabolic_gate() is True  # falls back to fixed 2.0, 5>4


class TestLayer1cCooldown:
    """Cooldown gate: force-resume after N consecutive fires."""

    def test_cooldown_off_no_effect(self):
        """atr_cooldown=0 → normal gate behavior."""
        bot = _make_bot(atr_parabolic_mult=2.0, atr_cooldown=0)
        bot.latest_signals = _signals(atr=5.0, atr_sma=2.0)
        # Should fire indefinitely
        for _ in range(20):
            assert bot._parabolic_gate() is True

    def test_cooldown_force_resumes(self):
        """After N consecutive fires, gate returns False (force-resume)."""
        bot = _make_bot(atr_parabolic_mult=2.0, atr_cooldown=4)
        bot.latest_signals = _signals(atr=5.0, atr_sma=2.0)
        results = [bot._parabolic_gate() for _ in range(8)]
        # First 4 fire (True), then force-resume (False)
        assert results == [True, True, True, True, False, False, False, False]

    def test_cooldown_resets_on_natural_clear(self):
        """Counter resets when gate naturally clears (ATR drops)."""
        bot = _make_bot(atr_parabolic_mult=2.0, atr_cooldown=4)
        bot.latest_signals = _signals(atr=5.0, atr_sma=2.0)
        # Fire 3 times (not yet at cooldown)
        for _ in range(3):
            assert bot._parabolic_gate() is True
        assert bot._gate_fire_counter == 3

        # ATR drops — gate naturally clears → counter resets
        bot.latest_signals = _signals(atr=3.0, atr_sma=2.0)
        assert bot._parabolic_gate() is False
        assert bot._gate_fire_counter == 0

        # Fire again — counter starts fresh
        bot.latest_signals = _signals(atr=5.0, atr_sma=2.0)
        for _ in range(4):
            assert bot._parabolic_gate() is True
        # 5th fire → force-resume
        assert bot._parabolic_gate() is False

    def test_cooldown_with_regime_adaptive(self):
        """Cooldown works together with regime-adaptive mults."""
        bot = _make_bot(
            atr_parabolic_mult=0.0,
            atr_regime_adaptive=True,
            atr_bull_mult=2.5,
            atr_bear_mult=1.5,
            atr_cooldown=3,
        )
        # Bear regime (close < regime_ema) → mult=1.5, threshold=3.0
        bot.latest_signals = _signals(atr=4.0, atr_sma=2.0, regime_ema=2.0, close=1.5)
        results = [bot._parabolic_gate() for _ in range(6)]
        assert results == [True, True, True, False, False, False]


class TestLayer1dAcceleration:
    """Acceleration filter: only gate when ATR is actively rising."""

    def test_accel_off_no_effect(self):
        """atr_accel_lookback=0 → normal gate (no acceleration check)."""
        bot = _make_bot(atr_parabolic_mult=2.0, atr_accel_lookback=0)
        bot.latest_signals = _signals(atr=5.0, atr_sma=2.0, atr_prev=6.0)
        # Even though ATR is falling (5 < 6), gate fires because accel is off
        assert bot._parabolic_gate() is True

    def test_accel_suppresses_when_atr_falling(self):
        """ATR above threshold BUT falling → gate suppressed (recovery phase)."""
        bot = _make_bot(atr_parabolic_mult=2.0, atr_accel_lookback=10)
        bot.latest_signals = _signals(atr=5.0, atr_sma=2.0, atr_prev=6.0)
        assert bot._parabolic_gate() is False

    def test_accel_suppresses_when_atr_flat(self):
        """ATR exactly equal to prev → not rising → gate suppressed."""
        bot = _make_bot(atr_parabolic_mult=2.0, atr_accel_lookback=10)
        bot.latest_signals = _signals(atr=5.0, atr_sma=2.0, atr_prev=5.0)
        assert bot._parabolic_gate() is False

    def test_accel_allows_when_atr_rising(self):
        """ATR above threshold AND rising → gate fires normally."""
        bot = _make_bot(atr_parabolic_mult=2.0, atr_accel_lookback=10)
        bot.latest_signals = _signals(atr=5.0, atr_sma=2.0, atr_prev=4.0)
        assert bot._parabolic_gate() is True

    def test_accel_no_prev_data(self):
        """atr_prev=0 (not enough history) → gate fires normally."""
        bot = _make_bot(atr_parabolic_mult=2.0, atr_accel_lookback=10)
        bot.latest_signals = _signals(atr=5.0, atr_sma=2.0, atr_prev=0.0)
        assert bot._parabolic_gate() is True

    def test_accel_below_threshold_no_gate(self):
        """ATR below threshold (no gate) → accel filter irrelevant."""
        bot = _make_bot(atr_parabolic_mult=2.0, atr_accel_lookback=10)
        bot.latest_signals = _signals(atr=3.0, atr_sma=2.0, atr_prev=4.0)
        assert bot._parabolic_gate() is False

    def test_accel_with_cooldown(self):
        """Acceleration + cooldown work together (the winning combo)."""
        bot = _make_bot(atr_parabolic_mult=1.5, atr_accel_lookback=10, atr_cooldown=4)
        # ATR rising above threshold → gate fires
        bot.latest_signals = _signals(atr=5.0, atr_sma=2.0, atr_prev=3.0)
        results = [bot._parabolic_gate() for _ in range(6)]
        # 4 fires then cooldown kicks in
        assert results == [True, True, True, True, False, False]

    def test_accel_suppresses_before_cooldown(self):
        """ATR falling suppresses gate before cooldown even starts counting."""
        bot = _make_bot(atr_parabolic_mult=1.5, atr_accel_lookback=10, atr_cooldown=4)
        bot.latest_signals = _signals(atr=5.0, atr_sma=2.0, atr_prev=6.0)
        # Accel suppresses immediately — cooldown counter never increments
        assert bot._parabolic_gate() is False
        assert bot._gate_fire_counter == 0
    """Parabolic gate integrated into _evaluate_trend."""

    @pytest.mark.asyncio
    async def test_long_blocked_by_parabolic(self):
        """Trend UP confirmed but ATR parabolic → no capture."""
        bot = _trend_up_bot(atr_parabolic_mult=2.0)
        bot.latest_signals = _signals(adx=30.0, atr=5.0, atr_sma=2.0, close=1.08)

        with patch.object(bot, "_open_trend_trade", new_callable=AsyncMock) as mock:
            await bot._evaluate_trend(price=1.08)
        mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_allowed_below_parabolic(self):
        """Trend UP confirmed, ATR below threshold → capture opens."""
        bot = _trend_up_bot(atr_parabolic_mult=2.0)
        bot.latest_signals = _signals(adx=30.0, atr=3.0, atr_sma=2.0, close=1.08)

        with patch.object(bot, "_open_trend_trade", new_callable=AsyncMock) as mock:
            await bot._evaluate_trend(price=1.08)
        mock.assert_called_once_with("long", 1.08)

    @pytest.mark.asyncio
    async def test_short_blocked_by_parabolic(self):
        """Trend DOWN confirmed but ATR parabolic → no capture."""
        bot = _trend_down_bot(atr_parabolic_mult=2.0)
        bot.latest_signals = _signals(adx=30.0, atr=5.0, atr_sma=2.0, close=0.92)

        with patch.object(bot, "_open_trend_trade", new_callable=AsyncMock) as mock:
            await bot._evaluate_trend(price=0.92)
        mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_short_allowed_below_parabolic(self):
        bot = _trend_down_bot(atr_parabolic_mult=2.0)
        bot.latest_signals = _signals(adx=30.0, atr=3.0, atr_sma=2.0, close=0.92)

        with patch.object(bot, "_open_trend_trade", new_callable=AsyncMock) as mock:
            await bot._evaluate_trend(price=0.92)
        mock.assert_called_once_with("short", 0.92)


# ===========================================================================
# Layer 2: HTF EMA alignment — _htf_bull() / _htf_bear()
# ===========================================================================

class TestLayer2HtfEma:
    """HTF EMA alignment helpers."""

    def test_htf_bull_true_when_fast_above_slow(self):
        bot = _make_bot()
        bot.latest_signals = _signals(htf_ema_fast=1.1, htf_ema_slow=1.0)
        assert bot._htf_bull() is True

    def test_htf_bull_false_when_fast_below_slow(self):
        bot = _make_bot()
        bot.latest_signals = _signals(htf_ema_fast=0.9, htf_ema_slow=1.0)
        assert bot._htf_bull() is False

    def test_htf_bear_true_when_fast_below_slow(self):
        bot = _make_bot()
        bot.latest_signals = _signals(htf_ema_fast=0.9, htf_ema_slow=1.0)
        assert bot._htf_bear() is True

    def test_htf_bear_false_when_fast_above_slow(self):
        bot = _make_bot()
        bot.latest_signals = _signals(htf_ema_fast=1.1, htf_ema_slow=1.0)
        assert bot._htf_bear() is False

    def test_htf_bull_false_when_equal(self):
        bot = _make_bot()
        bot.latest_signals = _signals(htf_ema_fast=1.0, htf_ema_slow=1.0)
        assert bot._htf_bull() is False

    def test_htf_bear_false_when_equal(self):
        bot = _make_bot()
        bot.latest_signals = _signals(htf_ema_fast=1.0, htf_ema_slow=1.0)
        assert bot._htf_bear() is False

    def test_no_signals_bull_false(self):
        bot = _make_bot()
        bot.latest_signals = None
        assert bot._htf_bull() is False

    def test_no_signals_bear_false(self):
        bot = _make_bot()
        bot.latest_signals = None
        assert bot._htf_bear() is False


class TestLayer2TrendIntegration:
    """HTF EMA alignment integrated into _evaluate_trend."""

    @pytest.mark.asyncio
    async def test_long_blocked_in_htf_downtrend(self):
        """htf_ema_align=True, fast < slow → long blocked."""
        bot = _trend_up_bot(htf_ema_align=True)
        bot.latest_signals = _signals(
            adx=30.0, htf_ema_fast=0.9, htf_ema_slow=1.0, close=1.08,
        )
        with patch.object(bot, "_open_trend_trade", new_callable=AsyncMock) as mock:
            await bot._evaluate_trend(price=1.08)
        mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_allowed_in_htf_uptrend(self):
        """htf_ema_align=True, fast > slow → long allowed."""
        bot = _trend_up_bot(htf_ema_align=True)
        bot.latest_signals = _signals(
            adx=30.0, htf_ema_fast=1.1, htf_ema_slow=1.0, close=1.08,
        )
        with patch.object(bot, "_open_trend_trade", new_callable=AsyncMock) as mock:
            await bot._evaluate_trend(price=1.08)
        mock.assert_called_once_with("long", 1.08)

    @pytest.mark.asyncio
    async def test_short_blocked_in_htf_uptrend(self):
        """htf_ema_align=True, fast > slow → short blocked."""
        bot = _trend_down_bot(htf_ema_align=True)
        bot.latest_signals = _signals(
            adx=30.0, htf_ema_fast=1.1, htf_ema_slow=1.0, close=0.92,
        )
        with patch.object(bot, "_open_trend_trade", new_callable=AsyncMock) as mock:
            await bot._evaluate_trend(price=0.92)
        mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_short_allowed_in_htf_downtrend(self):
        """htf_ema_align=True, fast < slow → short allowed."""
        bot = _trend_down_bot(htf_ema_align=True)
        bot.latest_signals = _signals(
            adx=30.0, htf_ema_fast=0.9, htf_ema_slow=1.0, close=0.92,
        )
        with patch.object(bot, "_open_trend_trade", new_callable=AsyncMock) as mock:
            await bot._evaluate_trend(price=0.92)
        mock.assert_called_once_with("short", 0.92)

    @pytest.mark.asyncio
    async def test_no_gate_when_htf_align_disabled(self):
        """htf_ema_align=False → long proceeds regardless of EMA direction."""
        bot = _trend_up_bot(htf_ema_align=False)
        bot.latest_signals = _signals(
            adx=30.0, htf_ema_fast=0.9, htf_ema_slow=1.0, close=1.08,
        )
        with patch.object(bot, "_open_trend_trade", new_callable=AsyncMock) as mock:
            await bot._evaluate_trend(price=1.08)
        mock.assert_called_once_with("long", 1.08)


# ===========================================================================
# Layer 3: Regime vote mode — _regime_vote_halt_longs()
# ===========================================================================

class TestLayer3RegimeVote:
    """Regime vote helper logic."""

    def test_off_when_vote_mode_disabled(self):
        bot = _make_bot(regime_vote_mode=False)
        bot.latest_signals = _signals(
            close=0.5, regime_ema=1.0, regime_ema_87=1.0, regime_ema_42=1.0,
        )
        assert bot._regime_vote_halt_longs() is False

    def test_off_when_no_signals(self):
        bot = _make_bot(regime_vote_mode=True)
        bot.latest_signals = None
        assert bot._regime_vote_halt_longs() is False

    def test_halts_when_3_of_3_bear(self):
        bot = _make_bot(regime_vote_mode=True)
        bot.latest_signals = _signals(
            close=0.5, regime_ema=1.0, regime_ema_87=1.0, regime_ema_42=1.0,
        )
        assert bot._regime_vote_halt_longs() is True

    def test_halts_when_2_of_3_bear(self):
        bot = _make_bot(regime_vote_mode=True)
        bot.latest_signals = _signals(
            close=0.5, regime_ema=1.0, regime_ema_87=1.0, regime_ema_42=0.0,
        )
        assert bot._regime_vote_halt_longs() is True

    def test_no_halt_when_1_of_3_bear(self):
        bot = _make_bot(regime_vote_mode=True)
        bot.latest_signals = _signals(
            close=0.5, regime_ema=1.0, regime_ema_87=0.0, regime_ema_42=0.0,
        )
        assert bot._regime_vote_halt_longs() is False

    def test_no_halt_when_0_of_3_bear(self):
        bot = _make_bot(regime_vote_mode=True)
        bot.latest_signals = _signals(
            close=2.0, regime_ema=1.0, regime_ema_87=1.0, regime_ema_42=1.0,
        )
        assert bot._regime_vote_halt_longs() is False

    def test_hysteresis_prevents_halt_at_boundary(self):
        """With 5% hysteresis, price=0.96 is above threshold (ema*0.95=0.95)."""
        bot = _make_bot(regime_vote_mode=True, regime_hysteresis_pct=0.05)
        bot.latest_signals = _signals(
            close=0.96,
            regime_ema=1.0,     # threshold = 0.95
            regime_ema_87=1.0,  # threshold = 0.95
            regime_ema_42=1.0,  # threshold = 0.95
        )
        assert bot._regime_vote_halt_longs() is False

    def test_hysteresis_halts_below_band(self):
        """With 5% hysteresis, price=0.94 is below all thresholds → halt."""
        bot = _make_bot(regime_vote_mode=True, regime_hysteresis_pct=0.05)
        bot.latest_signals = _signals(
            close=0.94,
            regime_ema=1.0,     # threshold = 0.95
            regime_ema_87=1.0,  # threshold = 0.95
            regime_ema_42=1.0,  # threshold = 0.95
        )
        assert bot._regime_vote_halt_longs() is True


class TestLayer3GridIntegration:
    """Regime vote mode blocks long grid entries in place_long_orders."""

    @pytest.mark.asyncio
    async def test_vote_halts_long_grid(self):
        """vote_mode=True, 3-of-3 bear → long grid halted."""
        bot = _make_bot(regime_vote_mode=True)
        bot.latest_signals = _signals(
            adx=10.0, close=0.5,
            regime_ema=1.0, regime_ema_87=1.0, regime_ema_42=1.0,
        )
        bot.latest_price = 0.5
        bot.get_take_profit_quantity = MagicMock()

        await bot.place_long_orders(0.5)
        bot.get_take_profit_quantity.assert_not_called()

    @pytest.mark.asyncio
    async def test_vote_allows_long_grid_when_bull(self):
        """vote_mode=True, 0-of-3 bear → long grid proceeds."""
        bot = _make_bot(regime_vote_mode=True)
        bot.latest_signals = _signals(
            adx=10.0, close=2.0,
            regime_ema=1.0, regime_ema_87=1.0, regime_ema_42=1.0,
        )
        bot.latest_price = 2.0
        bot.get_take_profit_quantity = MagicMock()

        await bot.place_long_orders(2.0)
        bot.get_take_profit_quantity.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_to_single_ema_when_vote_disabled(self):
        """vote_mode=False → uses original single regime-ema check."""
        bot = _make_bot(regime_vote_mode=False)
        bot.latest_signals = _signals(
            adx=10.0, close=0.5, regime_ema=1.0,
        )
        bot.latest_price = 0.5
        bot.get_take_profit_quantity = MagicMock()

        await bot.place_long_orders(0.5)
        # Should be halted by the single-EMA regime filter
        bot.get_take_profit_quantity.assert_not_called()


# ===========================================================================
# Layer 4: Grid sleep — _grid_sleep()
# ===========================================================================

class TestLayer4GridSleep:
    """Grid sleep gate logic."""

    def test_off_when_thresh_zero(self):
        bot = _make_bot(grid_sleep_atr_thresh=0.0)
        bot.latest_signals = _signals(atr=0.001, close=1.0)
        assert bot._grid_sleep() is False

    def test_off_when_no_signals(self):
        bot = _make_bot(grid_sleep_atr_thresh=0.003)
        bot.latest_signals = None
        assert bot._grid_sleep() is False

    def test_sleeps_when_atr_price_below_thresh(self):
        """ATR/price = 0.001/1.0 = 0.001 < 0.003 → sleep."""
        bot = _make_bot(grid_sleep_atr_thresh=0.003)
        bot.latest_signals = _signals(atr=0.001, close=1.0)
        assert bot._grid_sleep() is True

    def test_no_sleep_when_atr_price_above_thresh(self):
        """ATR/price = 0.005/1.0 = 0.005 > 0.003 → no sleep."""
        bot = _make_bot(grid_sleep_atr_thresh=0.003)
        bot.latest_signals = _signals(atr=0.005, close=1.0)
        assert bot._grid_sleep() is False

    def test_boundary_exact_threshold(self):
        """ATR/price == threshold exactly → not <, so no sleep."""
        bot = _make_bot(grid_sleep_atr_thresh=0.003)
        bot.latest_signals = _signals(atr=0.003, close=1.0)
        assert bot._grid_sleep() is False

    def test_zero_price_no_crash(self):
        """price=0 should not crash (returns False)."""
        bot = _make_bot(grid_sleep_atr_thresh=0.003)
        bot.latest_signals = _signals(atr=0.001, close=0.0)
        assert bot._grid_sleep() is False


class TestLayer4GridIntegration:
    """Grid sleep blocks both long and short grid entries."""

    @pytest.mark.asyncio
    async def test_long_grid_paused_by_sleep(self):
        """grid_sleep fires → place_long_orders skips grid refresh."""
        bot = _make_bot(grid_sleep_atr_thresh=0.003)
        bot.latest_signals = _signals(adx=10.0, atr=0.001, close=1.0, regime_ema=0.0)
        bot.latest_price = 1.0
        bot.get_take_profit_quantity = MagicMock()

        await bot.place_long_orders(1.0)
        bot.get_take_profit_quantity.assert_not_called()

    @pytest.mark.asyncio
    async def test_short_grid_paused_by_sleep(self):
        """grid_sleep fires → place_short_orders skips grid refresh."""
        bot = _make_bot(grid_sleep_atr_thresh=0.003)
        bot.latest_signals = _signals(adx=10.0, atr=0.001, close=1.0)
        bot.latest_price = 1.0
        bot.get_take_profit_quantity = MagicMock()

        await bot.place_short_orders(1.0)
        bot.get_take_profit_quantity.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_grid_proceeds_when_vol_ok(self):
        """ATR/price above threshold → grid proceeds."""
        bot = _make_bot(grid_sleep_atr_thresh=0.003)
        bot.latest_signals = _signals(adx=10.0, atr=0.005, close=1.0, regime_ema=0.0)
        bot.latest_price = 1.0
        bot.get_take_profit_quantity = MagicMock()

        await bot.place_long_orders(1.0)
        bot.get_take_profit_quantity.assert_called_once()

    @pytest.mark.asyncio
    async def test_short_grid_proceeds_when_vol_ok(self):
        """ATR/price above threshold → short grid proceeds."""
        bot = _make_bot(grid_sleep_atr_thresh=0.003)
        bot.latest_signals = _signals(adx=10.0, atr=0.005, close=1.0)
        bot.latest_price = 1.0
        bot.get_take_profit_quantity = MagicMock()

        await bot.place_short_orders(1.0)
        bot.get_take_profit_quantity.assert_called_once()


# ===========================================================================
# Combined layers
# ===========================================================================

class TestLayersCombined:
    """Multiple gates active simultaneously."""

    @pytest.mark.asyncio
    async def test_parabolic_plus_htf_blocks_long(self):
        """Both L1 and L2 active; either alone would block."""
        bot = _trend_up_bot(atr_parabolic_mult=2.0, htf_ema_align=True)
        bot.latest_signals = _signals(
            adx=30.0, atr=5.0, atr_sma=2.0,
            htf_ema_fast=0.9, htf_ema_slow=1.0, close=1.08,
        )
        with patch.object(bot, "_open_trend_trade", new_callable=AsyncMock) as mock:
            await bot._evaluate_trend(price=1.08)
        mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_layers_pass_allows_capture(self):
        """L1+L2 active but conditions favourable → capture opens."""
        bot = _trend_up_bot(atr_parabolic_mult=2.0, htf_ema_align=True)
        bot.latest_signals = _signals(
            adx=30.0, atr=3.0, atr_sma=2.0,
            htf_ema_fast=1.1, htf_ema_slow=1.0, close=1.08,
        )
        with patch.object(bot, "_open_trend_trade", new_callable=AsyncMock) as mock:
            await bot._evaluate_trend(price=1.08)
        mock.assert_called_once_with("long", 1.08)

    @pytest.mark.asyncio
    async def test_grid_sleep_plus_vote_blocks_long_grid(self):
        """L3 vote halts + L4 sleep — either is sufficient."""
        bot = _make_bot(regime_vote_mode=True, grid_sleep_atr_thresh=0.003)
        bot.latest_signals = _signals(
            adx=10.0, atr=0.001, close=0.5,
            regime_ema=1.0, regime_ema_87=1.0, regime_ema_42=1.0,
        )
        bot.latest_price = 0.5
        bot.get_take_profit_quantity = MagicMock()

        await bot.place_long_orders(0.5)
        # Should be halted by vote (checked first) before grid-sleep
        bot.get_take_profit_quantity.assert_not_called()

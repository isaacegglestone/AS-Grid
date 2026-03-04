"""
tests/single_bot/test_btbw_spacing.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for BTBW (Bull-Tight Bear-Wide) grid spacing selection in
``GridTradingBot.update_mid_price``.

Verifies:
- Bull regime selects ``bull_spacing``
- Bear regime selects ``bear_spacing``
- Hysteresis band prevents premature bear flip
- Fallback to ``grid_spacing`` when regime EMA not warmed up
- Short side uses identical regime-based logic
"""
from __future__ import annotations

import pytest

from src.single_bot.bitunix_bot import GridTradingBot
from src.single_bot.indicators import Signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot(**overrides) -> GridTradingBot:
    """Create a bot with distinct BTBW spacing for easy verification."""
    bot = GridTradingBot(
        api_key="k",
        api_secret="s",
        coin_name="XRP",
        grid_spacing=0.015,
        initial_quantity=10,
        leverage=5,
    )
    bot.bull_spacing = 0.010
    bot.bear_spacing = 0.020
    bot.regime_hysteresis_pct = 0.02   # 2% hysteresis band
    for k, v in overrides.items():
        setattr(bot, k, v)
    return bot


# ===========================================================================
# TestBtbwSpacing — update_mid_price regime-based spacing selection
# ===========================================================================


class TestBtbwSpacing:
    """update_mid_price should select effective spacing from regime signal."""

    def test_bull_regime_uses_bull_spacing(self):
        """Price above regime_ema → bull_spacing (tight) used."""
        bot = _make_bot()
        bot.latest_signals = Signals(regime_ema=2.0)
        bot.update_mid_price("long", 2.10)
        # bull_spacing = 0.010
        assert bot.lower_price_long == pytest.approx(2.10 * 0.990, abs=1e-4)
        assert bot.upper_price_long == pytest.approx(2.10 * 1.010, abs=1e-4)

    def test_bear_regime_uses_bear_spacing(self):
        """Price below regime_ema × (1 − hyst) → bear_spacing (wide) used."""
        bot = _make_bot()
        # threshold = 2.0 × 0.98 = 1.96;  price 1.90 < 1.96 → bear
        bot.latest_signals = Signals(regime_ema=2.0)
        bot.update_mid_price("long", 1.90)
        # bear_spacing = 0.020
        assert bot.lower_price_long == pytest.approx(1.90 * 0.980, abs=1e-4)
        assert bot.upper_price_long == pytest.approx(1.90 * 1.020, abs=1e-4)

    def test_hysteresis_keeps_bull_near_ema(self):
        """Price between EMA and hysteresis band → still treated as bull."""
        bot = _make_bot()
        # threshold = 2.0 × 0.98 = 1.96;  price 1.97 ≥ 1.96 → bull
        bot.latest_signals = Signals(regime_ema=2.0)
        bot.update_mid_price("long", 1.97)
        assert bot.lower_price_long == pytest.approx(1.97 * 0.990, abs=1e-4)

    def test_hysteresis_flips_bear_below_band(self):
        """Price below hysteresis band → switches to bear_spacing."""
        bot = _make_bot()
        # threshold = 2.0 × 0.98 = 1.96;  price 1.95 < 1.96 → bear
        bot.latest_signals = Signals(regime_ema=2.0)
        bot.update_mid_price("long", 1.95)
        assert bot.lower_price_long == pytest.approx(1.95 * 0.980, abs=1e-4)

    def test_fallback_when_signals_none(self):
        """No signals yet (cold start) → grid_spacing fallback."""
        bot = _make_bot()
        bot.latest_signals = None
        bot.update_mid_price("long", 2.0)
        assert bot.lower_price_long == pytest.approx(2.0 * 0.985, abs=1e-4)
        assert bot.upper_price_long == pytest.approx(2.0 * 1.015, abs=1e-4)

    def test_fallback_when_regime_ema_zero(self):
        """Regime EMA not yet warmed up (0.0) → grid_spacing fallback."""
        bot = _make_bot()
        bot.latest_signals = Signals(regime_ema=0.0)
        bot.update_mid_price("long", 2.0)
        assert bot.lower_price_long == pytest.approx(2.0 * 0.985, abs=1e-4)

    def test_short_side_bull_spacing(self):
        """Short side in bull regime → bull_spacing applied to short prices."""
        bot = _make_bot()
        bot.latest_signals = Signals(regime_ema=2.0)
        bot.update_mid_price("short", 2.10)
        assert bot.upper_price_short == pytest.approx(2.10 * 1.010, abs=1e-4)
        assert bot.lower_price_short == pytest.approx(2.10 * 0.990, abs=1e-4)

    def test_short_side_bear_spacing(self):
        """Short side in bear regime → bear_spacing applied to short prices."""
        bot = _make_bot()
        bot.latest_signals = Signals(regime_ema=2.0)
        bot.update_mid_price("short", 1.90)
        assert bot.upper_price_short == pytest.approx(1.90 * 1.020, abs=1e-4)
        assert bot.lower_price_short == pytest.approx(1.90 * 0.980, abs=1e-4)

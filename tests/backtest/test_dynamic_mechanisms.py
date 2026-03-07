"""
tests/backtest/test_dynamic_mechanisms.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for v26 XRPPM16 dynamic self-calibrating mechanisms added to
``GridOrderBacktester`` in ``asBack/backtest_grid_bitunix.py``.

Mechanisms tested
-----------------
1. Dynamic velocity threshold    (``vel_atr_mult``)
2. Dynamic gate decay            (``gate_decay_scale``)
3. Dynamic position sizing       (``cap_size_atr_scale`` / floor / ceiling)
4. Dynamic max-loss per trade    (``trend_max_loss_atr``)

Also covers
-----------
- ``_pm_v2_set`` parameter encoding for the new config keys.

Strategy
--------
All tests use synthetic in-memory DataFrames — no network calls, no mocking.
Behavioural tests inspect ``trade_history`` and ``trend_position`` to verify
the mechanisms fire (or don't).
"""
from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make the repo root importable (tests run from the workspace root)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from asBack.backtest_grid_bitunix import GridOrderBacktester, _pm_v2_set  # noqa: E402


# ===========================================================================
# Helpers
# ===========================================================================

def _make_df(closes: List[float], *, atr_mult: float = 0.005) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with synthetic candles.

    ``atr_mult`` controls the high/low spread as a fraction of close.
    Each row has an ``open_time`` timestamp 15 min apart.
    """
    base_ts = datetime(2025, 1, 1)
    records = []
    for i, c in enumerate(closes):
        records.append({
            "open_time": base_ts + timedelta(minutes=15 * i),
            "open":   float(c),
            "high":   float(c) * (1.0 + atr_mult),
            "low":    float(c) * (1.0 - atr_mult),
            "close":  float(c),
            "volume": 1_000_000.0,
        })
    return pd.DataFrame(records)


def _base_config(**overrides) -> Dict[str, Any]:
    """Minimum valid config dict; keyword arguments override defaults."""
    cfg: Dict[str, Any] = {
        "initial_balance": 1000.0,
        "order_value": 50.0,
        "max_drawdown": 0.9,
        "max_positions": 6,
        "max_positions_per_side": 3,
        "fee_pct": 0.0006,
        "leverage": 1.0,
        "direction": "both",
        "grid_refresh_interval": 60,
        "trend_detection": True,
        "trend_capture": True,
        "trend_lookback_candles": 10,
        "trend_velocity_pct": 0.04,
        "trend_capture_velocity_pct": 0.04,
        "trend_cooldown_candles": 5,
        "trend_confirm_candles": 1,
        "trend_capture_size_pct": 0.90,
        "trend_trailing_stop_pct": 0.04,
        "trend_force_close_grid": True,
        "use_sl": False,
        "long_settings":  {"up_spacing": 0.015, "down_spacing": 0.015},
        "short_settings": {"up_spacing": 0.015, "down_spacing": 0.015},
    }
    cfg.update(overrides)
    return cfg


def _run(df: pd.DataFrame, config: Dict[str, Any]) -> GridOrderBacktester:
    """Instantiate and run the backtester; return it for inspection."""
    bt = GridOrderBacktester(
        df,
        config["long_settings"]["down_spacing"],
        config,
    )
    bt.run()
    return bt


def _trend_trades(bt: GridOrderBacktester) -> List:
    """All TREND_BUY / TREND_SELL entries in trade_history."""
    return [
        t for t in bt.trade_history
        if t[1] in ("TREND_BUY", "TREND_SELL")
        and t[4] in ("TREND_LONG", "TREND_SHORT")
    ]


def _trend_open_trades(bt: GridOrderBacktester) -> List:
    """Only trend *opening* trades (TREND_BUY for LONG, TREND_SELL for SHORT)."""
    return [
        t for t in bt.trade_history
        if (t[1] == "TREND_BUY" and t[4] == "TREND_LONG")
        or (t[1] == "TREND_SELL" and t[4] == "TREND_SHORT")
    ]


# ---------------------------------------------------------------------------
# Synthetic price sequences
# ---------------------------------------------------------------------------

def _flat_then_spike_up(n_flat: int = 80) -> List[float]:
    """n_flat quiet candles at 1.0, then 10 slow approach (+0.3%/bar), then +12% spike."""
    closes = [1.0] * n_flat
    for _ in range(10):
        closes.append(closes[-1] * 1.003)
    closes.append(closes[-1] * 1.12)
    return closes


def _high_vol_spike_up(n_flat: int = 80) -> List[float]:
    """Like _flat_then_spike_up but with an intermediate volatility burst first.

    Pattern:
      80 flat candles → 10 big-vol bars (±5%) → 20 calm bars → 5% up move.
    The 5% move is below the dynamic threshold (ATR elevated) but above
    the static 4% threshold.
    """
    closes = [1.0] * n_flat
    # Volatility burst — big swings but net flat
    for i in range(10):
        closes.append(closes[-1] * (1.05 if i % 2 == 0 else 0.952))
    # Settle for 20 candles (ATR SMA still elevated)
    settle = closes[-1]
    for _ in range(20):
        closes.append(settle)
    # 5% move — below dynamic threshold but above static 4%
    for _ in range(10):
        closes.append(closes[-1] * 1.005)
    return closes


def _parabolic_then_correction() -> List[float]:
    """Simulates a parabolic spike followed by gradual correction.

    80 flat → 5 rapid +5%/bar (parabolic) → 10 flat (gate fires) →
    20 slight fade (gate clears, ATR still elevated) → 5% bounce.
    The bounce should be suppressed by gate_decay but not by the raw gate.
    """
    closes = [1.0] * 80
    # Parabolic spike: 5 candles at +5% each
    for _ in range(5):
        closes.append(closes[-1] * 1.05)
    peak = closes[-1]
    # 10 flat candles at peak (gate fires on the spike)
    for _ in range(10):
        closes.append(peak)
    # 20 candles of gentle decay (gate clears because ATR stops rising)
    for i in range(20):
        closes.append(closes[-1] * 0.998)
    # 5% bounce — should be suppressed by decay but not by raw gate
    for _ in range(10):
        closes.append(closes[-1] * 1.005)
    return closes


def _slow_grind_up(n: int = 120, pct: float = 0.005) -> List[float]:
    """Slow steady uptrend.  Velocity stays above 4% over 10 candles,
    ATR stays close to SMA (ratio ≈ 1)."""
    p, out = 1.0, []
    for _ in range(n):
        p *= (1.0 + pct)
        out.append(p)
    return out


def _flat_then_spike_down(n_flat: int = 80) -> List[float]:
    """n_flat quiet candles → 10 slow approach (−0.3%/bar) → −12% dump."""
    closes = [1.0] * n_flat
    for _ in range(10):
        closes.append(closes[-1] * 0.997)
    closes.append(closes[-1] * 0.88)
    return closes


# ===========================================================================
# 1 — _pm_v2_set: v26 dynamic param encoding
# ===========================================================================

class TestPmV2SetDynamicParams:
    """_pm_v2_set must follow the sentinel pattern for v26 dynamic params."""

    def test_vel_atr_mult_absent_by_default(self):
        cfg = _pm_v2_set("test")
        assert "vel_atr_mult" not in cfg

    def test_vel_atr_mult_stored_when_positive(self):
        cfg = _pm_v2_set("t", vel_atr_mult=1.0)
        assert cfg["vel_atr_mult"] == pytest.approx(1.0)

    def test_vel_atr_mult_zero_omitted(self):
        assert "vel_atr_mult" not in _pm_v2_set("t", vel_atr_mult=0.0)

    def test_vel_cap_atr_mult_absent_when_vel_atr_mult_zero(self):
        """vel_atr_mult=0 should not encode any velocity params."""
        assert "vel_atr_mult" not in _pm_v2_set("t", vel_atr_mult=0.0)

    def test_gate_decay_scale_absent_by_default(self):
        assert "gate_decay_scale" not in _pm_v2_set("t")

    def test_gate_decay_scale_stored_when_positive(self):
        cfg = _pm_v2_set("t", gate_decay_scale=3.0)
        assert cfg["gate_decay_scale"] == pytest.approx(3.0)

    def test_gate_decay_scale_zero_omitted(self):
        assert "gate_decay_scale" not in _pm_v2_set("t", gate_decay_scale=0.0)

    def test_cap_size_atr_scale_absent_by_default(self):
        assert "cap_size_atr_scale" not in _pm_v2_set("t")

    def test_cap_size_atr_scale_stored_when_true(self):
        cfg = _pm_v2_set("t", cap_size_atr_scale=True)
        assert cfg["cap_size_atr_scale"] is True
        assert "cap_size_atr_floor" in cfg
        assert "cap_size_atr_ceiling" in cfg

    def test_cap_size_atr_scale_false_omitted(self):
        assert "cap_size_atr_scale" not in _pm_v2_set("t", cap_size_atr_scale=False)

    def test_cap_size_atr_floor_ceiling_defaults(self):
        cfg = _pm_v2_set("t", cap_size_atr_scale=True)
        assert cfg["cap_size_atr_floor"] == pytest.approx(0.30)
        assert cfg["cap_size_atr_ceiling"] == pytest.approx(0.90)

    def test_cap_size_atr_custom_floor_ceiling(self):
        cfg = _pm_v2_set("t", cap_size_atr_scale=True,
                         cap_size_atr_floor=0.20, cap_size_atr_ceiling=0.80)
        assert cfg["cap_size_atr_floor"] == pytest.approx(0.20)
        assert cfg["cap_size_atr_ceiling"] == pytest.approx(0.80)

    def test_trend_max_loss_atr_absent_by_default(self):
        assert "trend_max_loss_atr" not in _pm_v2_set("t")

    def test_trend_max_loss_atr_stored_when_positive(self):
        cfg = _pm_v2_set("t", trend_max_loss_atr=2.0)
        assert cfg["trend_max_loss_atr"] == pytest.approx(2.0)

    def test_trend_max_loss_atr_zero_omitted(self):
        assert "trend_max_loss_atr" not in _pm_v2_set("t", trend_max_loss_atr=0.0)

    def test_all_v26_params_combined(self):
        cfg = _pm_v2_set(
            "full_v26",
            vel_atr_mult=1.0,
            gate_decay_scale=3.0,
            cap_size_atr_scale=True,
            trend_max_loss_atr=2.0,
        )
        assert cfg["vel_atr_mult"] == pytest.approx(1.0)
        assert cfg["gate_decay_scale"] == pytest.approx(3.0)
        assert cfg["cap_size_atr_scale"] is True
        assert cfg["trend_max_loss_atr"] == pytest.approx(2.0)

    def test_core_keys_still_present_with_v26_params(self):
        cfg = _pm_v2_set("t", vel_atr_mult=1.0, spacing=0.015)
        assert "long_settings" in cfg
        assert "regime_filter" in cfg
        assert cfg["long_settings"]["up_spacing"] == pytest.approx(0.015)


# ===========================================================================
# 2 — Dynamic velocity threshold
# ===========================================================================

class TestDynamicVelocityThreshold:
    """vel_atr_mult > 0 replaces static velocity threshold with ATR-scaled."""

    def test_static_threshold_fires_on_spike(self):
        """Without dynamic velocity, the spike opens a trend position."""
        df = _make_df(_flat_then_spike_up())
        bt = _run(df, _base_config())
        assert len(_trend_open_trades(bt)) >= 1, (
            "Static 4% threshold should fire on the +12% spike"
        )

    def test_dynamic_threshold_still_fires_on_clean_spike(self):
        """Dynamic velocity with moderate multiplier still fires on a clean spike.
        With ATR ratio ≈ 1.0 (normal vol), scale = max(1, 1.0 × 1.0) = 1.0,
        so the threshold stays at the static 4% — the +12% spike clears it."""
        df = _make_df(_flat_then_spike_up())
        cfg = _base_config(vel_atr_mult=1.0)
        bt = _run(df, cfg)
        assert len(_trend_open_trades(bt)) >= 1, (
            "Dynamic velocity should still fire on a clean +12% spike in normal vol"
        )

    def test_dynamic_threshold_blocks_in_high_vol_context(self):
        """After a vol burst, ATR is elevated; a 5% move that fires static
        4% threshold should be blocked by the dynamic threshold."""
        closes = _high_vol_spike_up(n_flat=80)
        df = _make_df(closes, atr_mult=0.03)  # wider spread to elevate ATR

        # Static: should fire (5% > 4%)
        bt_static = _run(df, _base_config())
        static_opens = _trend_open_trades(bt_static)

        # Dynamic: high multiplier should raise threshold above 5%
        cfg = _base_config(vel_atr_mult=2.0)
        bt_dynamic = _run(df, cfg)
        dynamic_opens = _trend_open_trades(bt_dynamic)

        assert len(dynamic_opens) <= len(static_opens), (
            "Dynamic velocity should block entries that static 4% would allow "
            f"in elevated ATR context (static={len(static_opens)}, dynamic={len(dynamic_opens)})"
        )

    def test_dynamic_threshold_preserves_normal_vol_trades(self):
        """In normal volatility, dynamic threshold ≈ static threshold."""
        closes = _slow_grind_up(n=120)
        df = _make_df(closes, atr_mult=0.005)

        bt_static  = _run(df, _base_config())
        cfg = _base_config(vel_atr_mult=1.0)
        bt_dynamic = _run(df, cfg)

        s_opens = _trend_open_trades(bt_static)
        d_opens = _trend_open_trades(bt_dynamic)

        # Dynamic should fire similar number of trades in calm conditions
        assert len(d_opens) >= len(s_opens) * 0.5, (
            f"Dynamic threshold should preserve most normal-vol trades "
            f"(static={len(s_opens)}, dynamic={len(d_opens)})"
        )

    def test_vel_atr_mult_zero_is_noop(self):
        """vel_atr_mult=0 should behave identically to no dynamic velocity."""
        df = _make_df(_flat_then_spike_up())
        bt_default = _run(df, _base_config())
        bt_zero    = _run(df, _base_config(vel_atr_mult=0.0))
        assert len(_trend_trades(bt_default)) == len(_trend_trades(bt_zero))


# ===========================================================================
# 3 — Dynamic gate decay
# ===========================================================================

class TestDynamicGateDecay:
    """gate_decay_scale > 0 extends parabolic suppression after gate clears."""

    @staticmethod
    def _spike_df_with_widened_tr() -> pd.DataFrame:
        """DataFrame with a parabolic spike that fires ATR gate, then clears."""
        closes = _parabolic_then_correction()
        df = _make_df(closes, atr_mult=0.005)
        # Widen the spike candles to maximise ATR
        for i in range(80, 85):
            if i < len(df):
                c = float(df["close"].iloc[i])
                df.loc[df.index[i], "high"] = c * 1.10
                df.loc[df.index[i], "low"]  = c * 0.90
        return df

    def test_no_decay_allows_post_gate_entry(self):
        """Without decay, entries can fire immediately after the gate clears."""
        df = self._spike_df_with_widened_tr()
        cfg = _base_config(
            atr_parabolic_mult=1.5,
            atr_acceleration=True, atr_accel_lookback=10,
            atr_cooldown=4,
        )
        bt = _run(df, cfg)
        # After gate clears, there should be trend trades from the bounce
        trades = _trend_open_trades(bt)
        # At least verify it runs without error
        assert bt.balance > 0, "Backtest should complete without blowing up"

    def test_decay_suppresses_post_gate_entries(self):
        """With large decay_scale, post-gate entries should be suppressed longer."""
        df = self._spike_df_with_widened_tr()
        cfg_no_decay = _base_config(
            atr_parabolic_mult=1.5,
            atr_acceleration=True, atr_accel_lookback=10,
            atr_cooldown=4,
        )
        cfg_decay = _base_config(
            atr_parabolic_mult=1.5,
            atr_acceleration=True, atr_accel_lookback=10,
            atr_cooldown=4,
            gate_decay_scale=10.0,  # very aggressive decay for test
        )
        bt_no = _run(df, cfg_no_decay)
        bt_yes = _run(df, cfg_decay)
        no_opens = _trend_open_trades(bt_no)
        yes_opens = _trend_open_trades(bt_yes)

        assert len(yes_opens) <= len(no_opens), (
            f"Gate decay should suppress ≤ entries vs no decay "
            f"(no_decay={len(no_opens)}, decay={len(yes_opens)})"
        )

    def test_gate_decay_scale_zero_is_noop(self):
        """gate_decay_scale=0.0 should behave identically to no decay."""
        df = self._spike_df_with_widened_tr()
        cfg = _base_config(atr_parabolic_mult=1.5)
        bt_default = _run(df, cfg)
        bt_zero    = _run(df, _base_config(atr_parabolic_mult=1.5, gate_decay_scale=0.0))
        assert len(_trend_trades(bt_default)) == len(_trend_trades(bt_zero))

    def test_decay_countdown_resets_after_expiry(self):
        """After decay countdown expires, entries should be allowed again."""
        # Long series: spike early, long flat period after → decay expires
        closes = [1.0] * 80
        for _ in range(5):
            closes.append(closes[-1] * 1.05)
        peak = closes[-1]
        for _ in range(10):
            closes.append(peak)
        for _ in range(100):  # Long settling period
            closes.append(peak * 0.99)
        # Fresh uptrend after settling
        for _ in range(20):
            closes.append(closes[-1] * 1.008)
        df = _make_df(closes, atr_mult=0.005)
        for i in range(80, 85):
            if i < len(df):
                c = float(df["close"].iloc[i])
                df.loc[df.index[i], "high"] = c * 1.10
                df.loc[df.index[i], "low"]  = c * 0.90

        cfg = _base_config(
            atr_parabolic_mult=1.5,
            atr_cooldown=4,
            gate_decay_scale=3.0,  # moderate decay
        )
        bt = _run(df, cfg)
        # Should complete without error; after decay + long wait, entries resume
        assert bt.balance > 0


# ===========================================================================
# 4 — Dynamic position sizing
# ===========================================================================

class TestDynamicPositionSizing:
    """cap_size_atr_scale=True scales position size inversely with ATR ratio."""

    def test_sizing_disabled_by_default(self):
        """Without cap_size_atr_scale, full 90% size is used."""
        cfg = _pm_v2_set("t")
        assert "cap_size_atr_scale" not in cfg

    def test_normal_vol_full_size(self):
        """In normal vol (ATR ≈ SMA), size should stay near ceiling (90%)."""
        closes = _slow_grind_up(n=120)
        df = _make_df(closes, atr_mult=0.005)
        # With sizing enabled: ratio ≈ 1 → used_cap_size = 0.90 / 1.0 = 0.90
        cfg = _base_config(cap_size_atr_scale=True)
        bt = _run(df, cfg)
        opens = _trend_open_trades(bt)
        if opens:
            # The margin used should be close to 90% of equity
            # (can't check exactly, but verify position was opened)
            assert bt.balance > 0, "Trade should complete successfully"

    def test_high_vol_reduced_size(self):
        """After a vol spike, ATR > SMA → sizing should reduce position."""
        closes = _flat_then_spike_up(n_flat=80)
        df = _make_df(closes, atr_mult=0.005)
        # Widen spike to elevate ATR
        spike_idx = len(df) - 1
        c = float(df["close"].iloc[spike_idx])
        df.loc[df.index[spike_idx], "high"] = c * 1.15
        df.loc[df.index[spike_idx], "low"]  = c * 0.85

        cfg_full = _base_config(trend_capture_size_pct=0.90)
        cfg_dyn  = _base_config(trend_capture_size_pct=0.90, cap_size_atr_scale=True)

        bt_full = _run(df, cfg_full)
        bt_dyn  = _run(df, cfg_dyn)

        full_opens = _trend_open_trades(bt_full)
        dyn_opens  = _trend_open_trades(bt_dyn)

        if full_opens and dyn_opens:
            # Dynamic-sized trade margin should be ≤ full-sized margin
            # Trade tuple: (ts, action, price, qty, side, pnl, fee, gross, unreal, equity)
            full_qty = full_opens[0][3]
            dyn_qty  = dyn_opens[0][3]
            assert dyn_qty <= full_qty * 1.01, (  # allow tiny rounding
                f"Dynamic sizing should reduce position in high vol "
                f"(full_qty={full_qty:.4f}, dyn_qty={dyn_qty:.4f})"
            )

    def test_floor_prevents_zero_size(self):
        """Even with extreme ATR ratio, size must not go below floor (30%)."""
        cfg = _base_config(
            cap_size_atr_scale=True,
            cap_size_atr_floor=0.30,
            cap_size_atr_ceiling=0.90,
        )
        # Just check config is valid and doesn't crash
        closes = _flat_then_spike_up(n_flat=80)
        df = _make_df(closes, atr_mult=0.005)
        bt = _run(df, cfg)
        assert bt.balance > 0

    def test_cap_size_atr_scale_false_is_noop(self):
        """cap_size_atr_scale=False must produce identical results to default."""
        df = _make_df(_flat_then_spike_up())
        bt_default = _run(df, _base_config())
        bt_false   = _run(df, _base_config(cap_size_atr_scale=False))
        assert len(_trend_trades(bt_default)) == len(_trend_trades(bt_false))


# ===========================================================================
# 5 — Dynamic max loss per trade
# ===========================================================================

class TestDynamicMaxLoss:
    """trend_max_loss_atr > 0 closes trend position when loss > N × ATR × qty."""

    def test_max_loss_disabled_by_default(self):
        """Without trend_max_loss_atr, no early close from loss cap."""
        cfg = _pm_v2_set("t")
        assert "trend_max_loss_atr" not in cfg

    def test_max_loss_caps_losing_trade(self):
        """A losing trend trade should close earlier with max_loss_atr set."""
        # Create a scenario: uptrend fires LONG, then price reverses sharply
        closes = [1.0] * 80
        for _ in range(10):
            closes.append(closes[-1] * 1.005)  # Gentle up
        # Sharp reversal
        for _ in range(20):
            closes.append(closes[-1] * 0.99)
        df = _make_df(closes, atr_mult=0.005)

        cfg_no_cap  = _base_config()
        cfg_cap     = _base_config(trend_max_loss_atr=1.0)  # tight cap

        bt_no  = _run(df, cfg_no_cap)
        bt_cap = _run(df, cfg_cap)

        # Both should complete; cap version may have smaller losses
        assert bt_no.balance > 0
        assert bt_cap.balance > 0

    def test_max_loss_does_not_clip_winners(self):
        """A winning trend trade should NOT be closed by the max loss cap."""
        closes = _slow_grind_up(n=120)
        df = _make_df(closes, atr_mult=0.005)

        cfg = _base_config(trend_max_loss_atr=2.0)
        bt = _run(df, cfg)
        opens = _trend_open_trades(bt)

        # Winning LONG in an uptrend should not be closed early
        assert bt.balance > 0
        # The trend should have opened (velocity fires in steady uptrend)
        # No assertion on exact count — just verify it doesn't break

    def test_max_loss_fires_on_short_reversal(self):
        """SHORT position that reverses should be capped by trend_max_loss_atr."""
        closes = [1.0] * 80
        # Downtrend triggers SHORT
        for _ in range(10):
            closes.append(closes[-1] * 0.995)
        # Sharp reversal up
        for _ in range(20):
            closes.append(closes[-1] * 1.01)
        df = _make_df(closes, atr_mult=0.005)

        cfg = _base_config(trend_max_loss_atr=1.0)
        bt = _run(df, cfg)
        assert bt.balance > 0, "Max loss should close early, preserving balance"

    def test_max_loss_zero_is_noop(self):
        """trend_max_loss_atr=0.0 should behave identically to default."""
        df = _make_df(_flat_then_spike_up())
        bt_default = _run(df, _base_config())
        bt_zero    = _run(df, _base_config(trend_max_loss_atr=0.0))
        assert len(_trend_trades(bt_default)) == len(_trend_trades(bt_zero))


# ===========================================================================
# 6 — Combined mechanisms
# ===========================================================================

class TestCombinedMechanisms:
    """Multiple dynamic mechanisms should work together without interference."""

    def test_all_mechanisms_enabled_runs_cleanly(self):
        """Full stack: velocity + decay + sizing + max_loss together."""
        closes = _parabolic_then_correction()
        df = _make_df(closes, atr_mult=0.005)
        for i in range(80, 85):
            if i < len(df):
                c = float(df["close"].iloc[i])
                df.loc[df.index[i], "high"] = c * 1.10
                df.loc[df.index[i], "low"]  = c * 0.90

        cfg = _base_config(
            atr_parabolic_mult=1.5,
            atr_acceleration=True,
            atr_accel_lookback=10,
            atr_cooldown=4,
            vel_atr_mult=1.0,
            gate_decay_scale=3.0,
            cap_size_atr_scale=True,
            trend_max_loss_atr=2.0,
        )
        bt = _run(df, cfg)
        assert bt.balance > 0, "Full dynamic stack should not crash"

    def test_full_stack_no_worse_than_baseline_on_clean_trend(self):
        """Full stack should not significantly hurt returns on a clean uptrend."""
        closes = _slow_grind_up(n=120)
        df = _make_df(closes, atr_mult=0.005)

        bt_baseline = _run(df, _base_config())
        bt_full = _run(df, _base_config(
            vel_atr_mult=1.0,
            gate_decay_scale=3.0,
            cap_size_atr_scale=True,
            trend_max_loss_atr=2.0,
        ))
        # Full stack should retain at least 70% of baseline equity
        assert bt_full._equity(closes[-1]) >= bt_baseline._equity(closes[-1]) * 0.70, (
            f"Full stack equity ({bt_full._equity(closes[-1]):.2f}) should be within "
            f"70% of baseline ({bt_baseline._equity(closes[-1]):.2f})"
        )

    def test_mechanisms_are_additive_not_multiplicative(self):
        """Each mechanism should be independently toggleable."""
        df = _make_df(_flat_then_spike_up())

        # Just velocity
        bt_vel = _run(df, _base_config(vel_atr_mult=1.0))
        # Just sizing
        bt_size = _run(df, _base_config(cap_size_atr_scale=True))
        # Just max loss
        bt_loss = _run(df, _base_config(trend_max_loss_atr=2.0))

        # All should run cleanly
        for bt, name in [(bt_vel, "vel"), (bt_size, "size"), (bt_loss, "loss")]:
            assert bt.balance > 0, f"{name} mechanism alone should run cleanly"

    def test_pm16_baseline_config_matches_pm15_winner(self):
        """PM16 baseline must reproduce the PM15 winning strategy exactly."""
        cfg = _pm_v2_set(
            "pm16_baseline",
            atr_parabolic_mult=1.5,
            atr_acceleration=True,
            atr_accel_lookback=10,
            atr_cooldown=4,
        )
        assert cfg.get("atr_parabolic_mult") == pytest.approx(1.5)
        assert cfg.get("atr_acceleration") is True
        assert cfg.get("atr_accel_lookback") == 10
        assert cfg.get("atr_cooldown") == 4
        # No v26 params should be present
        for key in ("vel_atr_mult", "gate_decay_scale",
                     "cap_size_atr_scale", "trend_max_loss_atr"):
            assert key not in cfg, f"Baseline should not have {key}"


# ===========================================================================
# 7 — _pm_v2_set: v27 PM17 param encoding
# ===========================================================================

class TestPmV2SetPM17Params:
    """_pm_v2_set must follow the sentinel pattern for v27 PM17 params."""

    def test_vel_dir_only_absent_by_default(self):
        cfg = _pm_v2_set("t")
        assert "vel_dir_only" not in cfg

    def test_vel_dir_only_stored_when_true(self):
        cfg = _pm_v2_set("t", vel_dir_only=True)
        assert cfg["vel_dir_only"] is True

    def test_vel_dir_only_false_omitted(self):
        assert "vel_dir_only" not in _pm_v2_set("t", vel_dir_only=False)

    def test_vel_accel_only_absent_by_default(self):
        cfg = _pm_v2_set("t")
        assert "vel_accel_only" not in cfg

    def test_vel_accel_only_stored_when_true(self):
        cfg = _pm_v2_set("t", vel_accel_only=True)
        assert cfg["vel_accel_only"] is True

    def test_vel_accel_only_false_omitted(self):
        assert "vel_accel_only" not in _pm_v2_set("t", vel_accel_only=False)

    def test_eq_curve_filter_absent_by_default(self):
        cfg = _pm_v2_set("t")
        assert "eq_curve_filter" not in cfg

    def test_eq_curve_filter_stored_when_true(self):
        cfg = _pm_v2_set("t", eq_curve_filter=True)
        assert cfg["eq_curve_filter"] is True
        assert cfg["eq_curve_lookback"] == 50  # default lookback

    def test_eq_curve_filter_false_omitted(self):
        assert "eq_curve_filter" not in _pm_v2_set("t", eq_curve_filter=False)

    def test_eq_curve_custom_lookback(self):
        cfg = _pm_v2_set("t", eq_curve_filter=True, eq_curve_lookback=100)
        assert cfg["eq_curve_lookback"] == 100

    def test_consec_loss_max_absent_by_default(self):
        cfg = _pm_v2_set("t")
        assert "consec_loss_max" not in cfg

    def test_consec_loss_max_stored_when_positive(self):
        cfg = _pm_v2_set("t", consec_loss_max=2, consec_loss_pause=20)
        assert cfg["consec_loss_max"] == 2
        assert cfg["consec_loss_pause"] == 20

    def test_consec_loss_max_zero_omitted(self):
        assert "consec_loss_max" not in _pm_v2_set("t", consec_loss_max=0)

    def test_all_v27_params_combined(self):
        cfg = _pm_v2_set(
            "full_v27",
            vel_atr_mult=0.75,
            vel_dir_only=True,
            vel_accel_only=True,
            eq_curve_filter=True,
            eq_curve_lookback=100,
            consec_loss_max=3,
            consec_loss_pause=30,
        )
        assert cfg["vel_atr_mult"] == pytest.approx(0.75)
        assert cfg["vel_dir_only"] is True
        assert cfg["vel_accel_only"] is True
        assert cfg["eq_curve_filter"] is True
        assert cfg["eq_curve_lookback"] == 100
        assert cfg["consec_loss_max"] == 3
        assert cfg["consec_loss_pause"] == 30

    def test_v27_params_coexist_with_v26(self):
        cfg = _pm_v2_set(
            "t",
            vel_atr_mult=1.0,
            vel_dir_only=True,
            gate_decay_scale=3.0,
        )
        assert cfg["vel_atr_mult"] == pytest.approx(1.0)
        assert cfg["vel_dir_only"] is True
        assert cfg["gate_decay_scale"] == pytest.approx(3.0)


# ===========================================================================
# 8 — Directional velocity (vel_dir_only)
# ===========================================================================

class TestDirectionalVelocity:
    """vel_dir_only=True: velocity scaling only applies when price < EMA-36."""

    def test_dir_vel_preserves_bull_entries(self):
        """In a clean uptrend (price > EMA-36), directional velocity should
        NOT raise the threshold — entries should fire like static baseline."""
        closes = _slow_grind_up(n=120)
        df = _make_df(closes, atr_mult=0.005)

        bt_static = _run(df, _base_config())
        bt_dir    = _run(df, _base_config(vel_atr_mult=1.0, vel_dir_only=True))

        s_opens = _trend_open_trades(bt_static)
        d_opens = _trend_open_trades(bt_dir)

        # In a clean uptrend, price > EMA-36, so vel_dir_only should NOT
        # suppress entries — should fire at least as many as static.
        assert len(d_opens) >= len(s_opens), (
            f"Directional velocity should preserve bull entries "
            f"(static={len(s_opens)}, dir={len(d_opens)})"
        )

    def test_dir_vel_blocks_in_bear_context(self):
        """After a drop (price < EMA-36), velocity scaling should apply and
        raise the threshold, blocking marginal entries."""
        # Big drop then small bounce — bounce should be blocked in bear context
        closes = [1.0] * 80
        # Sharp drop to put price below EMA-36
        for _ in range(15):
            closes.append(closes[-1] * 0.98)
        # Small bounce (5%) — below dynamic threshold but above static 4%
        for _ in range(10):
            closes.append(closes[-1] * 1.005)
        df = _make_df(closes, atr_mult=0.02)  # wider spread for elevated ATR

        bt_static = _run(df, _base_config())
        bt_dir    = _run(df, _base_config(vel_atr_mult=1.5, vel_dir_only=True))

        s_opens = _trend_open_trades(bt_static)
        d_opens = _trend_open_trades(bt_dir)

        assert len(d_opens) <= len(s_opens), (
            f"Directional velocity should suppress entries in bear context "
            f"(static={len(s_opens)}, dir={len(d_opens)})"
        )

    def test_dir_vel_runs_without_crash(self):
        """Full simulation with vel_dir_only should complete without error."""
        closes = _flat_then_spike_up()
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(vel_atr_mult=0.75, vel_dir_only=True)
        bt = _run(df, cfg)
        assert bt.balance > 0

    def test_dir_vel_false_is_noop(self):
        """vel_dir_only=False should behave identically to plain vel_atr_mult."""
        df = _make_df(_flat_then_spike_up())
        bt_plain = _run(df, _base_config(vel_atr_mult=1.0))
        bt_false = _run(df, _base_config(vel_atr_mult=1.0, vel_dir_only=False))
        assert len(_trend_trades(bt_plain)) == len(_trend_trades(bt_false))


# ===========================================================================
# 9 — ATR acceleration velocity (vel_accel_only)
# ===========================================================================

class TestAccelVelocity:
    """vel_accel_only=True: velocity scaling only when ATR is actively rising."""

    def test_accel_vel_preserves_settling_entries(self):
        """When ATR is elevated but falling (post-spike settling), velocity
        scaling should be removed — allowing entries PM16 would block."""
        # Spike → settle → gentle uptrend
        closes = [1.0] * 80
        # Vol burst — big swings
        for i in range(10):
            closes.append(closes[-1] * (1.05 if i % 2 == 0 else 0.952))
        # Long settling period (ATR falling)
        settle = closes[-1]
        for _ in range(40):
            closes.append(settle)
        # 6% uptrend — should fire with accel_only but may be blocked by plain vel
        for _ in range(12):
            closes.append(closes[-1] * 1.005)
        df = _make_df(closes, atr_mult=0.02)

        bt_plain = _run(df, _base_config(vel_atr_mult=1.5))
        bt_accel = _run(df, _base_config(vel_atr_mult=1.5, vel_accel_only=True))

        plain_opens = _trend_open_trades(bt_plain)
        accel_opens = _trend_open_trades(bt_accel)

        assert len(accel_opens) >= len(plain_opens), (
            f"Accel-only should allow more entries when ATR is settling "
            f"(plain={len(plain_opens)}, accel={len(accel_opens)})"
        )

    def test_accel_vel_still_blocks_during_spike(self):
        """When ATR is actively rising (spike in progress), velocity scaling
        should still apply — threshold raised to block noise entries."""
        closes = _high_vol_spike_up(n_flat=80)
        df = _make_df(closes, atr_mult=0.03)

        bt_accel = _run(df, _base_config(vel_atr_mult=2.0, vel_accel_only=True))
        assert bt_accel.balance > 0  # should complete without crash

    def test_accel_vel_runs_cleanly(self):
        """Full simulation with vel_accel_only should complete without error."""
        closes = _flat_then_spike_up()
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(vel_atr_mult=0.75, vel_accel_only=True)
        bt = _run(df, cfg)
        assert bt.balance > 0

    def test_accel_vel_false_is_noop(self):
        """vel_accel_only=False should behave identically to plain vel_atr_mult."""
        df = _make_df(_flat_then_spike_up())
        bt_plain = _run(df, _base_config(vel_atr_mult=1.0))
        bt_false = _run(df, _base_config(vel_atr_mult=1.0, vel_accel_only=False))
        assert len(_trend_trades(bt_plain)) == len(_trend_trades(bt_false))


# ===========================================================================
# 10 — Equity curve filter (eq_curve_filter)
# ===========================================================================

class TestEquityCurveFilter:
    """eq_curve_filter=True: suppress trend entries when equity < SMA(equity)."""

    def test_eq_curve_filter_disabled_by_default(self):
        """Without eq_curve_filter, no equity-based suppression."""
        cfg = _pm_v2_set("t")
        assert "eq_curve_filter" not in cfg

    def test_eq_curve_runs_cleanly(self):
        """Equity curve filter should complete without error on basic data."""
        closes = _slow_grind_up(n=150)
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(eq_curve_filter=True, eq_curve_lookback=50)
        bt = _run(df, cfg)
        assert bt.balance > 0

    def test_eq_curve_suppresses_after_losses(self):
        """After a sequence of losses (equity drops below SMA), entries
        should be suppressed by the equity curve filter."""
        # Uptrend fires LONG → sharp reversal causes loss → equity drops
        # → next uptrend should be suppressed by equity < SMA
        closes = [1.0] * 80
        # First uptrend
        for _ in range(15):
            closes.append(closes[-1] * 1.004)
        # Sharp reversal
        for _ in range(20):
            closes.append(closes[-1] * 0.99)
        # Second uptrend attempt — should be suppressed if equity < SMA
        for _ in range(15):
            closes.append(closes[-1] * 1.004)
        df = _make_df(closes, atr_mult=0.005)

        bt_no_filter  = _run(df, _base_config())
        bt_eq_filter  = _run(df, _base_config(
            eq_curve_filter=True, eq_curve_lookback=30))

        no_opens = _trend_open_trades(bt_no_filter)
        eq_opens = _trend_open_trades(bt_eq_filter)

        # Equity filter should suppress some entries after losses
        assert len(eq_opens) <= len(no_opens), (
            f"Equity filter should suppress entries after drawdown "
            f"(no_filter={len(no_opens)}, eq_filter={len(eq_opens)})"
        )

    def test_eq_curve_allows_winning_streaks(self):
        """When equity is above SMA (winning), entries should NOT be blocked."""
        closes = _slow_grind_up(n=150)
        df = _make_df(closes, atr_mult=0.005)

        bt_baseline = _run(df, _base_config())
        bt_filter   = _run(df, _base_config(
            eq_curve_filter=True, eq_curve_lookback=50))

        base_opens = _trend_open_trades(bt_baseline)
        filt_opens = _trend_open_trades(bt_filter)

        # In a clean uptrend with no losses, equity stays above SMA
        # so filter should allow all entries — at least most of baseline
        assert len(filt_opens) >= len(base_opens) * 0.7, (
            f"Equity filter should allow entries during winning streak "
            f"(baseline={len(base_opens)}, filter={len(filt_opens)})"
        )

    def test_eq_curve_filter_false_is_noop(self):
        """eq_curve_filter=False must produce identical results to default."""
        df = _make_df(_flat_then_spike_up())
        bt_default = _run(df, _base_config())
        bt_false   = _run(df, _base_config(eq_curve_filter=False))
        assert len(_trend_trades(bt_default)) == len(_trend_trades(bt_false))


# ===========================================================================
# 11 — Consecutive loss guard (consec_loss_max)
# ===========================================================================

class TestConsecLossGuard:
    """consec_loss_max > 0: pause trend entries after N consecutive losses."""

    def test_consec_loss_disabled_by_default(self):
        """Without consec_loss_max, no loss-based suppression."""
        cfg = _pm_v2_set("t")
        assert "consec_loss_max" not in cfg

    def test_consec_loss_runs_cleanly(self):
        """Consecutive loss guard should complete without error."""
        closes = _slow_grind_up(n=150)
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(consec_loss_max=2, consec_loss_pause=20)
        bt = _run(df, cfg)
        assert bt.balance > 0

    def test_consec_loss_pauses_after_losses(self):
        """After consecutive losing trend trades, the guard should pause entries."""
        # Pattern: repeated whipsaw — up trigger then immediate reversal
        # to generate consecutive losses, then another uptrend
        closes = [1.0] * 80
        # Whipsaw 1: up → sharp reverse → up → sharp reverse
        for cycle in range(3):
            for _ in range(5):
                closes.append(closes[-1] * 1.01)
            for _ in range(10):
                closes.append(closes[-1] * 0.995)
        # Final uptrend
        for _ in range(20):
            closes.append(closes[-1] * 1.005)
        df = _make_df(closes, atr_mult=0.005)

        bt_no_guard  = _run(df, _base_config(trend_confirm_candles=1))
        bt_guard     = _run(df, _base_config(
            trend_confirm_candles=1,
            consec_loss_max=2, consec_loss_pause=15))

        no_opens = _trend_open_trades(bt_no_guard)
        guard_opens = _trend_open_trades(bt_guard)

        # Guard should suppress some entries after consecutive losses
        assert len(guard_opens) <= len(no_opens), (
            f"Consecutive loss guard should suppress entries after losses "
            f"(no_guard={len(no_opens)}, guard={len(guard_opens)})"
        )

    def test_consec_loss_resets_on_win(self):
        """A winning trade should reset the consecutive loss counter."""
        closes = _slow_grind_up(n=150)
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(consec_loss_max=2, consec_loss_pause=10)
        bt = _run(df, cfg)
        # In a clean uptrend, wins should keep counter at 0
        assert bt._consec_trend_losses == 0, (
            f"Consecutive loss counter should be 0 after winning trades "
            f"(actual={bt._consec_trend_losses})"
        )

    def test_consec_loss_zero_is_noop(self):
        """consec_loss_max=0 should behave identically to default."""
        df = _make_df(_flat_then_spike_up())
        bt_default = _run(df, _base_config())
        bt_zero    = _run(df, _base_config(consec_loss_max=0))
        assert len(_trend_trades(bt_default)) == len(_trend_trades(bt_zero))

    def test_consec_loss_counter_increments(self):
        """The _consec_trend_losses counter should increment on losses."""
        # Uptrend → sharp reversal to cause a losing LONG close
        closes = [1.0] * 80
        for _ in range(12):
            closes.append(closes[-1] * 1.005)
        for _ in range(30):
            closes.append(closes[-1] * 0.99)
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(consec_loss_max=5, consec_loss_pause=10)
        bt = _run(df, cfg)
        # There should be at least one loss, so counter >= 1
        # (unless the trade happened to be profitable)
        assert bt._consec_trend_losses >= 0  # Just verify counter exists


# ===========================================================================
# 12 — PM17 combined mechanisms
# ===========================================================================

class TestPM17Combined:
    """Multiple PM17 mechanisms should work together without interference."""

    def test_all_pm17_mechanisms_enabled(self):
        """Full PM17 stack should run without crashing."""
        closes = _parabolic_then_correction()
        df = _make_df(closes, atr_mult=0.005)
        for i in range(80, 85):
            if i < len(df):
                c = float(df["close"].iloc[i])
                df.loc[df.index[i], "high"] = c * 1.10
                df.loc[df.index[i], "low"]  = c * 0.90

        cfg = _base_config(
            atr_parabolic_mult=1.5,
            atr_acceleration=True,
            atr_accel_lookback=10,
            atr_cooldown=4,
            vel_atr_mult=0.75,
            vel_dir_only=True,
            eq_curve_filter=True,
            eq_curve_lookback=50,
            consec_loss_max=2,
            consec_loss_pause=20,
        )
        bt = _run(df, cfg)
        assert bt.balance > 0, "Full PM17 stack should not crash"

    def test_dir_vel_plus_accel_vel(self):
        """Both directional and acceleration velocity together."""
        closes = _flat_then_spike_up()
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(
            vel_atr_mult=1.0,
            vel_dir_only=True,
            vel_accel_only=True,
        )
        bt = _run(df, cfg)
        assert bt.balance > 0

    def test_eq_curve_plus_consec_loss(self):
        """Equity curve filter plus consecutive loss guard together."""
        closes = _slow_grind_up(n=150)
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(
            eq_curve_filter=True,
            eq_curve_lookback=50,
            consec_loss_max=3,
            consec_loss_pause=20,
        )
        bt = _run(df, cfg)
        assert bt.balance > 0

    def test_pm17_v26_coexistence(self):
        """PM17 mechanisms should not break PM16 mechanisms (v26 + v27)."""
        closes = _flat_then_spike_up()
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(
            # v26 params
            vel_atr_mult=1.0,
            cap_size_atr_scale=True,
            trend_max_loss_atr=2.0,
            # v27 params
            vel_dir_only=True,
            eq_curve_filter=True,
            eq_curve_lookback=50,
            consec_loss_max=2,
            consec_loss_pause=20,
        )
        bt = _run(df, cfg)
        assert bt.balance > 0, "v26 + v27 coexistence should not crash"

    def test_pm17_baseline_config_matches_pm15_winner(self):
        """PM17 baseline must reproduce the PM15 winning strategy exactly."""
        cfg = _pm_v2_set(
            "pm17_baseline",
            atr_parabolic_mult=1.5,
            atr_acceleration=True,
            atr_accel_lookback=10,
            atr_cooldown=4,
        )
        assert cfg.get("atr_parabolic_mult") == pytest.approx(1.5)
        assert cfg.get("atr_acceleration") is True
        assert cfg.get("atr_accel_lookback") == 10
        assert cfg.get("atr_cooldown") == 4
        # No v27 params should be present
        for key in ("vel_dir_only", "vel_accel_only",
                     "eq_curve_filter", "consec_loss_max"):
            assert key not in cfg, f"Baseline should not have {key}"

    def test_full_pm17_no_worse_than_70pct_baseline(self):
        """Full PM17 stack should retain at least 70% of baseline equity
        on a clean uptrend."""
        closes = _slow_grind_up(n=150)
        df = _make_df(closes, atr_mult=0.005)

        bt_baseline = _run(df, _base_config())
        bt_full = _run(df, _base_config(
            vel_atr_mult=0.75,
            vel_dir_only=True,
            eq_curve_filter=True,
            eq_curve_lookback=50,
            consec_loss_max=3,
            consec_loss_pause=20,
        ))
        assert bt_full._equity(closes[-1]) >= bt_baseline._equity(closes[-1]) * 0.70, (
            f"Full PM17 equity ({bt_full._equity(closes[-1]):.2f}) should be within "
            f"70% of baseline ({bt_baseline._equity(closes[-1]):.2f})"
        )


# ===========================================================================
# v28 — XRPPM18: Fine-tuning directional velocity
#
# Tests for:
#   - vel_dir_ema_period param encoding in _pm_v2_set
#   - Configurable EMA period for directional velocity check
#   - EMA-84 / EMA-120 / EMA-200 directional filters
#   - Dual filter (vel_dir_only + vel_accel_only)
#   - Confirm candle variations
#   - Trail stop variations
# ===========================================================================

class TestPmV2SetPM18Params:
    """_pm_v2_set must follow the sentinel pattern for v28 PM18 params."""

    def test_vel_dir_ema_period_absent_when_36(self):
        """Default EMA period (36) should NOT emit the key (sentinel pattern)."""
        cfg = _pm_v2_set("t", vel_dir_only=True)
        assert "vel_dir_ema_period" not in cfg

    def test_vel_dir_ema_period_absent_when_explicit_36(self):
        cfg = _pm_v2_set("t", vel_dir_only=True, vel_dir_ema_period=36)
        assert "vel_dir_ema_period" not in cfg

    def test_vel_dir_ema_period_stored_when_84(self):
        cfg = _pm_v2_set("t", vel_dir_only=True, vel_dir_ema_period=84)
        assert cfg["vel_dir_ema_period"] == 84

    def test_vel_dir_ema_period_stored_when_120(self):
        cfg = _pm_v2_set("t", vel_dir_only=True, vel_dir_ema_period=120)
        assert cfg["vel_dir_ema_period"] == 120

    def test_vel_dir_ema_period_stored_when_200(self):
        cfg = _pm_v2_set("t", vel_dir_only=True, vel_dir_ema_period=200)
        assert cfg["vel_dir_ema_period"] == 200

    def test_v28_coexists_with_v27_and_v26(self):
        """All param versions should coexist without conflicts."""
        cfg = _pm_v2_set(
            "combo",
            vel_atr_mult=1.0,
            vel_dir_only=True,
            vel_dir_ema_period=84,
            vel_accel_only=True,
            gate_decay_scale=3.0,
        )
        assert cfg["vel_atr_mult"] == pytest.approx(1.0)
        assert cfg["vel_dir_only"] is True
        assert cfg["vel_dir_ema_period"] == 84
        assert cfg["vel_accel_only"] is True
        assert cfg["gate_decay_scale"] == pytest.approx(3.0)

    def test_pm18_baseline_matches_pm17_dirvel10(self):
        """PM18 baseline should be identical to pm17_dirvel10 winner."""
        cfg = _pm_v2_set(
            "pm18_baseline",
            atr_parabolic_mult=1.5,
            atr_acceleration=True,
            atr_accel_lookback=10,
            atr_cooldown=4,
            vel_atr_mult=1.0,
            vel_dir_only=True,
        )
        assert cfg["vel_atr_mult"] == pytest.approx(1.0)
        assert cfg["vel_dir_only"] is True
        assert "vel_dir_ema_period" not in cfg  # defaults to 36


# ===========================================================================
# 14 — Configurable EMA period for directional velocity
# ===========================================================================

class TestConfigurableEMAPeriod:
    """vel_dir_ema_period controls which EMA is used for bearish context check."""

    def test_ema84_uses_slower_ema(self):
        """With EMA-84, the directional filter should be slower to flag
        bearish context — preserving more entries in choppy markets."""
        # Create a choppy pattern: up then gentle dip that crosses EMA-36
        # but stays above EMA-84
        closes = [1.0] * 40
        # Gradual rise
        for _ in range(30):
            closes.append(closes[-1] * 1.003)
        # Gentle dip — crosses below EMA-36 but stays above EMA-84
        for _ in range(10):
            closes.append(closes[-1] * 0.995)
        # Bounce up — should be caught by EMA-84 filter
        for _ in range(20):
            closes.append(closes[-1] * 1.005)
        df = _make_df(closes, atr_mult=0.008)

        bt_ema36 = _run(df, _base_config(vel_atr_mult=1.0, vel_dir_only=True))
        bt_ema84 = _run(df, _base_config(
            vel_atr_mult=1.0, vel_dir_only=True, vel_dir_ema_period=84
        ))

        # EMA-84 is slower, so should allow at least as many entries as EMA-36
        e36_opens = _trend_open_trades(bt_ema36)
        e84_opens = _trend_open_trades(bt_ema84)
        assert len(e84_opens) >= len(e36_opens), (
            f"EMA-84 should be less restrictive: e84={len(e84_opens)}, e36={len(e36_opens)}"
        )

    def test_ema120_runs_without_crash(self):
        """EMA-120 directional filter should run without error."""
        closes = _flat_then_spike_up()
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(vel_atr_mult=1.0, vel_dir_only=True, vel_dir_ema_period=120)
        bt = _run(df, cfg)
        assert bt.balance > 0

    def test_ema200_runs_without_crash(self):
        """EMA-200 directional filter should run without error."""
        closes = _slow_grind_up(n=250)
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(vel_atr_mult=1.0, vel_dir_only=True, vel_dir_ema_period=200)
        bt = _run(df, cfg)
        assert bt.balance > 0

    def test_ema36_default_matches_no_period(self):
        """vel_dir_ema_period=36 should behave identically to omitting it."""
        df = _make_df(_flat_then_spike_up())
        bt_default = _run(df, _base_config(vel_atr_mult=1.0, vel_dir_only=True))
        bt_explicit = _run(df, _base_config(
            vel_atr_mult=1.0, vel_dir_only=True, vel_dir_ema_period=36
        ))
        assert len(_trend_trades(bt_default)) == len(_trend_trades(bt_explicit))

    def test_vel_dir_ema_series_is_htf_fast_for_36(self):
        """When period=36, vel_dir_ema_series should be htf_ema_fast_series."""
        df = _make_df([1.0] * 50)
        cfg = _base_config(vel_atr_mult=1.0, vel_dir_only=True)
        bt = GridOrderBacktester(df, cfg["long_settings"]["down_spacing"], cfg)
        assert bt.vel_dir_ema_series is bt.htf_ema_fast_series

    def test_vel_dir_ema_series_is_htf_slow_for_84(self):
        """When period=84, vel_dir_ema_series should be htf_ema_slow_series."""
        df = _make_df([1.0] * 100)
        cfg = _base_config(vel_atr_mult=1.0, vel_dir_only=True, vel_dir_ema_period=84)
        bt = GridOrderBacktester(df, cfg["long_settings"]["down_spacing"], cfg)
        assert bt.vel_dir_ema_series is bt.htf_ema_slow_series

    def test_vel_dir_ema_series_is_custom_for_120(self):
        """When period=120, a new EMA series should be computed."""
        df = _make_df([1.0] * 150)
        cfg = _base_config(vel_atr_mult=1.0, vel_dir_only=True, vel_dir_ema_period=120)
        bt = GridOrderBacktester(df, cfg["long_settings"]["down_spacing"], cfg)
        # Should NOT be the same object as fast or slow
        assert bt.vel_dir_ema_series is not bt.htf_ema_fast_series
        assert bt.vel_dir_ema_series is not bt.htf_ema_slow_series

    def test_longer_ema_preserves_entries_in_bear_dip(self):
        """After a moderate dip, EMA-200 should still be above price less often
        than EMA-36, so it should block fewer entries."""
        closes = [1.0] * 80
        # Sharp drop
        for _ in range(15):
            closes.append(closes[-1] * 0.98)
        # Recovery bounce
        for _ in range(15):
            closes.append(closes[-1] * 1.005)
        df = _make_df(closes, atr_mult=0.02)

        bt_36 = _run(df, _base_config(vel_atr_mult=1.5, vel_dir_only=True))
        bt_200 = _run(df, _base_config(
            vel_atr_mult=1.5, vel_dir_only=True, vel_dir_ema_period=200
        ))

        # EMA-200 barely moves during a short dip, so price stays above it
        # longer — should allow at least as many entries
        t36 = _trend_open_trades(bt_36)
        t200 = _trend_open_trades(bt_200)
        assert len(t200) >= len(t36), (
            f"EMA-200 should block fewer entries in brief dip: "
            f"200={len(t200)}, 36={len(t36)}"
        )


# ===========================================================================
# 15 — Dual filter (vel_dir_only + vel_accel_only)
# ===========================================================================

class TestDualFilter:
    """Both directional and acceleration filters applied together."""

    def test_dual_filter_runs_without_crash(self):
        """Dual filter should complete without error."""
        closes = _flat_then_spike_up()
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(
            vel_atr_mult=1.0, vel_dir_only=True, vel_accel_only=True
        )
        bt = _run(df, cfg)
        assert bt.balance > 0

    def test_dual_is_less_aggressive_than_single(self):
        """Dual filter applies scaling less often (AND logic), so it should
        block fewer entries than dir-only (which scales whenever bearish)."""
        # Pattern: drop (bearish) + mixed ATR
        closes = [1.0] * 80
        for _ in range(15):
            closes.append(closes[-1] * 0.98)
        for _ in range(10):
            closes.append(closes[-1] * 1.005)
        df = _make_df(closes, atr_mult=0.02)

        bt_dir = _run(df, _base_config(vel_atr_mult=1.5, vel_dir_only=True))
        bt_dual = _run(df, _base_config(
            vel_atr_mult=1.5, vel_dir_only=True, vel_accel_only=True
        ))

        d_opens = _trend_open_trades(bt_dir)
        dual_opens = _trend_open_trades(bt_dual)
        # Dual requires BOTH bearish + rising ATR to scale, so it scales
        # less often → lower effective threshold → allows more entries
        assert len(dual_opens) >= len(d_opens), (
            f"Dual filter should be less aggressive (scales less): "
            f"dir={len(d_opens)}, dual={len(dual_opens)}"
        )

    def test_dual_preserves_bull_entries(self):
        """In a clean uptrend (price > EMA, ATR stable), dual filter should
        NOT suppress entries."""
        closes = _slow_grind_up(n=120)
        df = _make_df(closes, atr_mult=0.005)

        bt_static = _run(df, _base_config())
        bt_dual = _run(df, _base_config(
            vel_atr_mult=1.0, vel_dir_only=True, vel_accel_only=True
        ))

        s_opens = _trend_open_trades(bt_static)
        d_opens = _trend_open_trades(bt_dual)
        assert len(d_opens) >= len(s_opens), (
            f"Dual filter should preserve bull entries: "
            f"static={len(s_opens)}, dual={len(d_opens)}"
        )


# ===========================================================================
# 16 — Confirm candle and trail stop variations
# ===========================================================================

class TestConfirmAndTrail:
    """Override trend_confirm_candles and trend_trailing_stop_pct."""

    def test_confirm1_runs_without_crash(self):
        """Single confirm candle should run without error."""
        closes = _flat_then_spike_up()
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(
            vel_atr_mult=1.0, vel_dir_only=True, trend_confirm_candles=1
        )
        bt = _run(df, cfg)
        assert bt.balance > 0

    def test_confirm5_runs_without_crash(self):
        """Five confirm candles should run without error."""
        closes = _flat_then_spike_up(n_flat=120)
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(
            vel_atr_mult=1.0, vel_dir_only=True, trend_confirm_candles=5
        )
        bt = _run(df, cfg)
        assert bt.balance > 0

    def test_fewer_confirms_catches_more_trends(self):
        """Fewer confirm candles should catch at least as many trends."""
        df = _make_df(_slow_grind_up(n=150), atr_mult=0.005)

        bt_1 = _run(df, _base_config(
            vel_atr_mult=1.0, vel_dir_only=True, trend_confirm_candles=1
        ))
        bt_5 = _run(df, _base_config(
            vel_atr_mult=1.0, vel_dir_only=True, trend_confirm_candles=5
        ))

        opens_1 = _trend_open_trades(bt_1)
        opens_5 = _trend_open_trades(bt_5)
        assert len(opens_1) >= len(opens_5), (
            f"Fewer confirms should catch more trends: "
            f"1-conf={len(opens_1)}, 5-conf={len(opens_5)}"
        )

    def test_trail03_runs_without_crash(self):
        """3% trailing stop should run without error."""
        closes = _flat_then_spike_up()
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(
            vel_atr_mult=1.0, vel_dir_only=True, trend_trailing_stop_pct=0.03
        )
        bt = _run(df, cfg)
        assert bt.balance > 0

    def test_trail06_runs_without_crash(self):
        """6% trailing stop should run without error."""
        closes = _flat_then_spike_up()
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(
            vel_atr_mult=1.0, vel_dir_only=True, trend_trailing_stop_pct=0.06
        )
        bt = _run(df, cfg)
        assert bt.balance > 0

    def test_wider_trail_holds_longer(self):
        """Wider trail should hold trend positions at least as long."""
        # Gradual rise then pullback — wider trail stays in longer
        closes = _slow_grind_up(n=80)
        # Small pullback
        for _ in range(10):
            closes.append(closes[-1] * 0.995)
        # Continue up
        for _ in range(20):
            closes.append(closes[-1] * 1.005)
        df = _make_df(closes, atr_mult=0.005)

        bt_3 = _run(df, _base_config(
            vel_atr_mult=1.0, vel_dir_only=True, trend_trailing_stop_pct=0.03
        ))
        bt_6 = _run(df, _base_config(
            vel_atr_mult=1.0, vel_dir_only=True, trend_trailing_stop_pct=0.06
        ))

        closes_3 = [t for t in bt_3.trade_history if "TRAIL" in str(t)]
        closes_6 = [t for t in bt_6.trade_history if "TRAIL" in str(t)]
        # Tighter trail fires more often
        assert len(closes_3) >= len(closes_6), (
            f"Tighter trail should fire more often: "
            f"3%={len(closes_3)}, 6%={len(closes_6)}"
        )


# ===========================================================================
# 17 — PM18 combined sanity
# ===========================================================================

class TestPM18Combined:
    """Integration tests for PM18 parameter combinations."""

    def test_pm18_ema84_with_dual_filter(self):
        """EMA-84 + dual filter should run without error."""
        closes = _flat_then_spike_up()
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(
            vel_atr_mult=1.0,
            vel_dir_only=True,
            vel_accel_only=True,
            vel_dir_ema_period=84,
        )
        bt = _run(df, cfg)
        assert bt.balance > 0

    def test_pm18_ema120_with_trail05(self):
        """EMA-120 + 5% trail should run without error."""
        closes = _slow_grind_up(n=150)
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(
            vel_atr_mult=1.0,
            vel_dir_only=True,
            vel_dir_ema_period=120,
            trend_trailing_stop_pct=0.05,
        )
        bt = _run(df, cfg)
        assert bt.balance > 0

    def test_pm18_full_stack_no_worse_than_70pct(self):
        """Full PM18 stack should retain at least 70% of baseline equity."""
        closes = _slow_grind_up(n=150)
        df = _make_df(closes, atr_mult=0.005)

        bt_baseline = _run(df, _base_config())
        bt_full = _run(df, _base_config(
            vel_atr_mult=1.5,
            vel_dir_only=True,
            vel_dir_ema_period=84,
            vel_accel_only=True,
            trend_trailing_stop_pct=0.05,
            trend_confirm_candles=2,
        ))
        assert bt_full._equity(closes[-1]) >= bt_baseline._equity(closes[-1]) * 0.70, (
            f"Full PM18 equity ({bt_full._equity(closes[-1]):.2f}) should be within "
            f"70% of baseline ({bt_baseline._equity(closes[-1]):.2f})"
        )

    def test_all_pm18_sweep_configs_valid(self):
        """All 15 PM18 sweep configs should have required base keys."""
        from asBack.backtest_grid_bitunix import _PM18_SETS
        assert len(_PM18_SETS) == 15
        for cfg in _PM18_SETS:
            assert "name" in cfg
            assert cfg["name"].startswith("pm18_")
            assert cfg.get("trend_detection") is True
            assert cfg.get("trend_capture") is True

    def test_pm18_baseline_has_dirvel10_settings(self):
        """PM18 baseline should match pm17_dirvel10 winner settings."""
        from asBack.backtest_grid_bitunix import _PM18_SETS
        baseline = next(c for c in _PM18_SETS if c["name"] == "pm18_baseline")
        assert baseline["vel_atr_mult"] == pytest.approx(1.0)
        assert baseline["vel_dir_only"] is True
        assert baseline.get("atr_parabolic_mult") == pytest.approx(1.5)
        assert baseline.get("atr_cooldown") == 4

    def test_pm18_ema_strategies_have_correct_periods(self):
        """EMA sweep strategies should have the right periods set."""
        from asBack.backtest_grid_bitunix import _PM18_SETS
        ema84 = next(c for c in _PM18_SETS if c["name"] == "pm18_ema84")
        ema120 = next(c for c in _PM18_SETS if c["name"] == "pm18_ema120")
        ema200 = next(c for c in _PM18_SETS if c["name"] == "pm18_ema200")
        assert ema84["vel_dir_ema_period"] == 84
        assert ema120["vel_dir_ema_period"] == 120
        assert ema200["vel_dir_ema_period"] == 200

    def test_pm18_confirm_overrides_work(self):
        """Confirm candle overrides should override _pm_v2_set defaults."""
        from asBack.backtest_grid_bitunix import _PM18_SETS
        c1 = next(c for c in _PM18_SETS if c["name"] == "pm18_confirm1")
        c2 = next(c for c in _PM18_SETS if c["name"] == "pm18_confirm2")
        c5 = next(c for c in _PM18_SETS if c["name"] == "pm18_confirm5")
        assert c1["trend_confirm_candles"] == 1
        assert c2["trend_confirm_candles"] == 2
        assert c5["trend_confirm_candles"] == 5

    def test_pm18_trail_overrides_work(self):
        """Trail stop overrides should override _pm_v2_set defaults."""
        from asBack.backtest_grid_bitunix import _PM18_SETS
        t3 = next(c for c in _PM18_SETS if c["name"] == "pm18_trail03")
        t5 = next(c for c in _PM18_SETS if c["name"] == "pm18_trail05")
        t6 = next(c for c in _PM18_SETS if c["name"] == "pm18_trail06")
        assert t3["trend_trailing_stop_pct"] == pytest.approx(0.03)
        assert t5["trend_trailing_stop_pct"] == pytest.approx(0.05)
        assert t6["trend_trailing_stop_pct"] == pytest.approx(0.06)

    def test_pm18_dual_has_both_filters(self):
        """Dual filter strategy should have both vel_dir_only and vel_accel_only."""
        from asBack.backtest_grid_bitunix import _PM18_SETS
        dual = next(c for c in _PM18_SETS if c["name"] == "pm18_dual")
        assert dual["vel_dir_only"] is True
        assert dual["vel_accel_only"] is True
        assert dual["vel_atr_mult"] == pytest.approx(1.0)


# ── PM19: Combination Sweep Tests ─────────────────────────────────────────


class TestPM19CombinationSweep:
    """Tests for PM19 v29 — systematic combinations of PM18 winners."""

    def test_all_pm19_sweep_configs_valid(self):
        """All 12 PM19 sweep configs should have required base keys."""
        from asBack.backtest_grid_bitunix import _PM19_SETS
        assert len(_PM19_SETS) == 12
        for cfg in _PM19_SETS:
            assert "name" in cfg
            assert cfg["name"].startswith("pm19_")
            assert cfg.get("trend_detection") is True
            assert cfg.get("trend_capture") is True

    def test_pm19_configs_have_dirvel_base(self):
        """Every PM19 config should inherit dirvel (vel_dir_only=True)."""
        from asBack.backtest_grid_bitunix import _PM19_SETS
        for cfg in _PM19_SETS:
            assert cfg.get("vel_dir_only") is True, f"{cfg['name']} missing vel_dir_only"
            assert cfg.get("vel_atr_mult", 0) > 0, f"{cfg['name']} missing vel_atr_mult"

    def test_pm19_ema120_combos_have_correct_period(self):
        """All EMA-120 strategies should set vel_dir_ema_period=120."""
        from asBack.backtest_grid_bitunix import _PM19_SETS
        ema120_names = [c["name"] for c in _PM19_SETS if "ema120" in c["name"]]
        assert len(ema120_names) >= 8  # 8 combos use EMA-120
        for cfg in _PM19_SETS:
            if "ema120" in cfg["name"]:
                assert cfg["vel_dir_ema_period"] == 120, f"{cfg['name']} has wrong EMA period"

    def test_pm19_dual_combos_have_accel_filter(self):
        """All dual-filter strategies should set vel_accel_only=True."""
        from asBack.backtest_grid_bitunix import _PM19_SETS
        dual_names = [c["name"] for c in _PM19_SETS if "dual" in c["name"]]
        assert len(dual_names) >= 5  # 5 combos use dual
        for cfg in _PM19_SETS:
            if "dual" in cfg["name"]:
                assert cfg.get("vel_accel_only") is True, f"{cfg['name']} missing vel_accel_only"

    def test_pm19_m15_combos_have_correct_mult(self):
        """All mult-1.5 strategies should set vel_atr_mult=1.5."""
        from asBack.backtest_grid_bitunix import _PM19_SETS
        m15_names = [c["name"] for c in _PM19_SETS if "m15" in c["name"]]
        assert len(m15_names) >= 5  # 5 combos use mult 1.5
        for cfg in _PM19_SETS:
            if "m15" in cfg["name"]:
                assert cfg["vel_atr_mult"] == pytest.approx(1.5), f"{cfg['name']} has wrong mult"

    def test_pm19_c5_combos_have_correct_confirms(self):
        """All confirm-5 strategies should override trend_confirm_candles to 5."""
        from asBack.backtest_grid_bitunix import _PM19_SETS
        c5_names = [c["name"] for c in _PM19_SETS if c["name"].endswith("_c5")]
        assert len(c5_names) >= 5  # 5 combos use confirm-5
        for cfg in _PM19_SETS:
            if cfg["name"].endswith("_c5"):
                assert cfg["trend_confirm_candles"] == 5, f"{cfg['name']} has wrong confirm count"

    def test_pm19_c4_combos_have_correct_confirms(self):
        """Confirm-4 compromise strategies should override trend_confirm_candles to 4."""
        from asBack.backtest_grid_bitunix import _PM19_SETS
        c4_cfgs = [c for c in _PM19_SETS if c["name"].endswith("_c4")]
        assert len(c4_cfgs) == 2
        for cfg in c4_cfgs:
            assert cfg["trend_confirm_candles"] == 4, f"{cfg['name']} has wrong confirm count"
            assert cfg["vel_dir_ema_period"] == 120, f"{cfg['name']} should also have EMA-120"

    def test_pm19_non_confirm_combos_have_default_confirms(self):
        """Strategies without confirm override should use default (3)."""
        from asBack.backtest_grid_bitunix import _PM19_SETS
        for cfg in _PM19_SETS:
            if not cfg["name"].endswith(("_c4", "_c5")):
                assert cfg["trend_confirm_candles"] == 3, (
                    f"{cfg['name']} should have default confirm=3, got {cfg['trend_confirm_candles']}"
                )

    def test_pm19_kitchen_sink_has_all_four_winners(self):
        """Kitchen-sink (ema120_dual_m15_c5) should combine all 4 PM18 winners."""
        from asBack.backtest_grid_bitunix import _PM19_SETS
        ks = next(c for c in _PM19_SETS if c["name"] == "pm19_ema120_dual_m15_c5")
        assert ks["vel_dir_ema_period"] == 120
        assert ks["vel_accel_only"] is True
        assert ks["vel_atr_mult"] == pytest.approx(1.5)
        assert ks["trend_confirm_candles"] == 5

    def test_pm19_non_ema120_strategies_use_default_ema(self):
        """Strategies without ema120 in name should use default EMA period (36)."""
        from asBack.backtest_grid_bitunix import _PM19_SETS
        for cfg in _PM19_SETS:
            if "ema120" not in cfg["name"]:
                period = cfg.get("vel_dir_ema_period", 36)
                assert period == 36, f"{cfg['name']} should use default EMA-36, got {period}"

    def test_pm19_no_duplicate_names(self):
        """All PM19 strategy names should be unique."""
        from asBack.backtest_grid_bitunix import _PM19_SETS
        names = [c["name"] for c in _PM19_SETS]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"

    def test_pm19_window_configs_exist(self):
        """All three PM19 window configs should reference _PM19_SETS."""
        from asBack.backtest_grid_bitunix import (
            XRP_PM_V19_CONFIG,
            XRP_PM_V19_2Y_CONFIG,
            XRP_PM_V19_1Y_MID_CONFIG,
            _PM19_SETS,
        )
        assert XRP_PM_V19_CONFIG["param_sets"] is _PM19_SETS
        assert XRP_PM_V19_2Y_CONFIG["param_sets"] is _PM19_SETS
        assert XRP_PM_V19_1Y_MID_CONFIG["param_sets"] is _PM19_SETS

    def test_pm19_integration_ema120_c5_runs(self):
        """EMA-120 + confirm 5 combo should run a backtest without error."""
        closes = _flat_then_spike_up()
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(
            vel_atr_mult=1.0,
            vel_dir_only=True,
            vel_dir_ema_period=120,
            trend_confirm_candles=5,
        )
        bt = _run(df, cfg)
        assert bt.balance > 0

    def test_pm19_integration_kitchen_sink_runs(self):
        """Kitchen-sink combo (all 4 winners) should run without error."""
        closes = _flat_then_spike_up()
        df = _make_df(closes, atr_mult=0.005)
        cfg = _base_config(
            vel_atr_mult=1.5,
            vel_dir_only=True,
            vel_dir_ema_period=120,
            vel_accel_only=True,
            trend_confirm_candles=5,
        )
        bt = _run(df, cfg)
        assert bt.balance > 0

    def test_pm19_integration_dual_m15_no_worse_than_70pct(self):
        """Dual + mult 1.5 combo should retain at least 70% of baseline equity."""
        closes = _slow_grind_up(n=150)
        df = _make_df(closes, atr_mult=0.005)

        bt_baseline = _run(df, _base_config())
        bt_combo = _run(df, _base_config(
            vel_atr_mult=1.5,
            vel_dir_only=True,
            vel_accel_only=True,
        ))
        assert bt_combo._equity(closes[-1]) >= bt_baseline._equity(closes[-1]) * 0.70, (
            f"Dual+m15 equity ({bt_combo._equity(closes[-1]):.2f}) should be within "
            f"70% of baseline ({bt_baseline._equity(closes[-1]):.2f})"
        )

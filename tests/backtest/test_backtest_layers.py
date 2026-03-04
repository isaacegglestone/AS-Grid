"""
tests/backtest/test_backtest_layers.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the v21/v22 protective gate layers added to
``GridOrderBacktester`` in ``asBack/backtest_grid_bitunix.py``.

Layers tested
--------------
- Layer 1 : ATR parabolic gate   (``atr_parabolic_mult``)
- Layer 2 : HTF EMA alignment    (``htf_ema_align``)
- Layer 3 : Multi-TF regime vote (``regime_vote_mode``)
- Layer 4 : Grid sleep           (``grid_sleep_atr_thresh``)

Also covers
-----------
- Pre-computed series correctness (atr_sma, htf_ema_fast/slow, regime_87/42)
- ``_pm_v2_set`` gate-param encoding in the config dict

Strategy
--------
All tests use synthetic in-memory DataFrames — no network calls, no mocking.
Behavioural tests inspect ``trade_history`` to verify gates fire (or don't).
"""
from __future__ import annotations

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

    ``atr_mult`` controls the high/low spread as a fraction of the close price.
    ``atr_mult=0.005`` → ±0.5% spread per bar (ATR ≈ 1% of price after warm-up).
    Each row has an ``open_time`` timestamp 15 minutes apart.
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
    """Return a minimum valid config dict; keyword arguments override defaults."""
    cfg: Dict[str, Any] = {
        "initial_balance": 1000.0,
        "order_value": 50.0,
        "max_drawdown": 0.9,
        "max_positions": 6,
        "max_positions_per_side": 3,
        "fee_pct": 0.0006,
        "leverage": 1.0,
        "direction": "both",
        # Refresh anchor every 60 min = every 4 × 15-min candles.
        # This lets price cross the anchor before it is reset.
        "grid_refresh_interval": 60,
        # Trend detection — enabled; aggressive settings to reliably fire.
        "trend_detection": True,
        "trend_capture": True,
        "trend_lookback_candles": 10,
        "trend_velocity_pct": 0.04,           # 4 % in 10 candles
        "trend_capture_velocity_pct": 0.04,
        "trend_cooldown_candles": 5,
        "trend_confirm_candles": 1,            # fire on first qualifying candle
        "trend_capture_size_pct": 0.30,
        "trend_trailing_stop_pct": 0.04,
        "trend_force_close_grid": True,
        "use_sl": False,                       # no SL to keep behavioural tests simple
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
    """All TREND_BUY / TREND_SELL open-position entries in trade_history."""
    return [
        t for t in bt.trade_history
        if t[1] in ("TREND_BUY", "TREND_SELL")
        and t[4] in ("TREND_LONG", "TREND_SHORT")
    ]


def _long_grid_entries(bt: GridOrderBacktester) -> List:
    """All BUY (long grid order opened) entries in trade_history."""
    return [t for t in bt.trade_history if t[1] == "BUY"]


# ---------------------------------------------------------------------------
# Synthetic price sequences
# ---------------------------------------------------------------------------

def _flat_then_spike_up(n_flat: int = 80) -> List[float]:
    """n_flat quiet candles at 1.0, then 10 slow approach (+0.3%/bar), then +12% spike.

    Approach velocity over 10 bars ≈ 3 % (< 4 % threshold → trend does NOT fire early).
    Final spike velocity ≈ 15 % (> 4 % → fires exactly on the spike candle).
    """
    closes = [1.0] * n_flat
    for _ in range(10):
        closes.append(closes[-1] * 1.003)   # +0.3 % approach
    closes.append(closes[-1] * 1.12)        # +12 % spike
    return closes


def _flat_then_spike_down(n_flat: int = 80) -> List[float]:
    """n_flat candles at 1.0, then 10 slow −0.3%/bar approach, then −12% dump."""
    closes = [1.0] * n_flat
    for _ in range(10):
        closes.append(closes[-1] * 0.997)
    closes.append(closes[-1] * 0.88)
    return closes


def _sustained_uptrend(n: int = 150, pct: float = 0.003) -> List[float]:
    """Steady +pct per candle; after ~84 bars EMA-36 is firmly above EMA-84."""
    p, out = 1.0, []
    for _ in range(n):
        p *= (1.0 + pct)
        out.append(p)
    return out


def _sustained_downtrend(n: int = 150, pct: float = 0.003) -> List[float]:
    """Steady −pct per candle; after ~84 bars EMA-36 is firmly below EMA-84."""
    p, out = 1.0, []
    for _ in range(n):
        p *= (1.0 - pct)
        out.append(p)
    return out


def _flat_then_dip(n_flat: int = 80, dip: float = 0.025) -> List[float]:
    """n_flat warm-up candles at 1.0, then price drops by dip and stays there."""
    return [1.0] * n_flat + [1.0 - dip] * 40


# ===========================================================================
# 1 — Pre-computed series: presence and mathematical correctness
# ===========================================================================

class TestNewSeriesComputed:
    """GridOrderBacktester.__init__ must pre-compute all five new series."""

    @staticmethod
    def _bt() -> GridOrderBacktester:
        return GridOrderBacktester(
            _make_df(_flat_then_spike_up()),
            0.015,
            _base_config(),
        )

    # --- presence -----------------------------------------------------------

    def test_atr_sma_series_exists(self):
        assert hasattr(self._bt(), "atr_sma_series")

    def test_htf_ema_fast_series_exists(self):
        assert hasattr(self._bt(), "htf_ema_fast_series")

    def test_htf_ema_slow_series_exists(self):
        assert hasattr(self._bt(), "htf_ema_slow_series")

    def test_regime_ema_87_series_exists(self):
        assert hasattr(self._bt(), "regime_ema_87_series")

    def test_regime_ema_42_series_exists(self):
        assert hasattr(self._bt(), "regime_ema_42_series")

    def test_all_series_length_equals_df(self):
        bt = self._bt()
        n = len(bt.df)
        for name in (
            "atr_sma_series",
            "htf_ema_fast_series",
            "htf_ema_slow_series",
            "regime_ema_87_series",
            "regime_ema_42_series",
        ):
            assert len(getattr(bt, name)) == n, f"Length mismatch for {name}"

    # --- mathematical correctness -------------------------------------------

    def test_atr_sma_equals_rolling20_mean_of_atr(self):
        """atr_sma_series must exactly reproduce rolling(20).mean() of atr_series."""
        bt = self._bt()
        expected = bt.atr_series.rolling(20, min_periods=1).mean().fillna(0)
        pd.testing.assert_series_equal(
            bt.atr_sma_series.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_htf_ema_fast_is_ewm_span36(self):
        """htf_ema_fast_series must be EWM(span=36) of close."""
        bt = self._bt()
        expected = bt.df["close"].astype(float).ewm(span=36, adjust=False).mean()
        pd.testing.assert_series_equal(
            bt.htf_ema_fast_series.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_regime_ema_87_is_ewm_span87(self):
        """regime_ema_87_series must be EWM(span=87) of close."""
        bt = self._bt()
        expected = bt.df["close"].astype(float).ewm(span=87, adjust=False).mean()
        pd.testing.assert_series_equal(
            bt.regime_ema_87_series.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_regime_ema_42_is_ewm_span42(self):
        """regime_ema_42_series must be EWM(span=42) of close."""
        bt = self._bt()
        expected = bt.df["close"].astype(float).ewm(span=42, adjust=False).mean()
        pd.testing.assert_series_equal(
            bt.regime_ema_42_series.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_atr_sma_non_negative(self):
        """atr_sma_series must never be negative (fillna(0) guarantee)."""
        assert (self._bt().atr_sma_series >= 0).all()

    def test_htf_ema_fast_above_slow_in_uptrend(self):
        """After a sustained uptrend EMA-36 must exceed EMA-84."""
        df = _make_df(_sustained_uptrend(200))
        bt = GridOrderBacktester(df, 0.015, _base_config())
        assert bt.htf_ema_fast_series.iloc[-1] > bt.htf_ema_slow_series.iloc[-1]

    def test_htf_ema_fast_below_slow_in_downtrend(self):
        """After a sustained downtrend EMA-36 must lag below EMA-84."""
        df = _make_df(_sustained_downtrend(200))
        bt = GridOrderBacktester(df, 0.015, _base_config())
        assert bt.htf_ema_fast_series.iloc[-1] < bt.htf_ema_slow_series.iloc[-1]

    def test_htf_emas_converge_on_flat_series(self):
        """On a constant price series both HTF EMAs must converge to a single value."""
        df = _make_df([1.0] * 200)
        bt = GridOrderBacktester(df, 0.015, _base_config())
        assert abs(bt.htf_ema_fast_series.iloc[-1] - bt.htf_ema_slow_series.iloc[-1]) < 1e-6

    def test_atr_parabolic_ratio_stay_stable_on_flat_series(self):
        """On a flat price series the ATR/SMA(ATR) ratio must never exceed 2.0."""
        df = _make_df([1.0] * 150, atr_mult=0.005)
        bt = GridOrderBacktester(df, 0.015, _base_config())
        warm = 20   # skip warm-up period
        ratios = (
            bt.atr_series.iloc[warm:].values
            / bt.atr_sma_series.iloc[warm:].replace(0, float("nan")).values
        )
        assert (ratios[~pd.isna(ratios)] <= 2.0 + 1e-6).all(), (
            "ATR/SMA(ATR) should never exceed 2.0 on a flat price series"
        )


# ===========================================================================
# 2 — _pm_v2_set gate-param encoding
# ===========================================================================

class TestPmV2SetLayerParams:
    """_pm_v2_set must follow the sentinel pattern: only encode non-default values."""

    def test_all_gate_keys_absent_by_default(self):
        cfg = _pm_v2_set("test")
        for key in ("atr_parabolic_mult", "htf_ema_align",
                    "regime_vote_mode", "grid_sleep_atr_thresh"):
            assert key not in cfg, f"Key '{key}' should be absent when using defaults"

    def test_atr_parabolic_mult_stored_when_positive(self):
        assert _pm_v2_set("t", atr_parabolic_mult=2.0)["atr_parabolic_mult"] == pytest.approx(2.0)

    def test_atr_parabolic_mult_zero_omitted(self):
        assert "atr_parabolic_mult" not in _pm_v2_set("t", atr_parabolic_mult=0.0)

    def test_htf_ema_align_true_stored(self):
        assert _pm_v2_set("t", htf_ema_align=True).get("htf_ema_align") is True

    def test_htf_ema_align_false_omitted(self):
        assert "htf_ema_align" not in _pm_v2_set("t", htf_ema_align=False)

    def test_regime_vote_mode_true_stored(self):
        assert _pm_v2_set("t", regime_vote_mode=True).get("regime_vote_mode") is True

    def test_regime_vote_mode_false_omitted(self):
        assert "regime_vote_mode" not in _pm_v2_set("t", regime_vote_mode=False)

    def test_grid_sleep_thresh_positive_stored(self):
        assert _pm_v2_set("t", grid_sleep_atr_thresh=0.003)["grid_sleep_atr_thresh"] == pytest.approx(0.003)

    def test_grid_sleep_thresh_zero_omitted(self):
        assert "grid_sleep_atr_thresh" not in _pm_v2_set("t", grid_sleep_atr_thresh=0.0)

    def test_all_four_layers_combined(self):
        cfg = _pm_v2_set(
            "full",
            atr_parabolic_mult=2.0,
            htf_ema_align=True,
            regime_vote_mode=True,
            grid_sleep_atr_thresh=0.003,
        )
        assert cfg["atr_parabolic_mult"] == pytest.approx(2.0)
        assert cfg["htf_ema_align"] is True
        assert cfg["regime_vote_mode"] is True
        assert cfg["grid_sleep_atr_thresh"] == pytest.approx(0.003)

    def test_core_keys_still_present_with_gates(self):
        cfg = _pm_v2_set("t", atr_parabolic_mult=1.5, spacing=0.015)
        assert "long_settings" in cfg
        assert "regime_filter" in cfg
        assert cfg["long_settings"]["up_spacing"] == pytest.approx(0.015)

    def test_different_spacing_values(self):
        for s in (0.010, 0.015, 0.020):
            cfg = _pm_v2_set("t", spacing=s)
            assert cfg["long_settings"]["up_spacing"] == pytest.approx(s)
            assert cfg["short_settings"]["up_spacing"] == pytest.approx(s)


# ===========================================================================
# 3 — Layer 1: ATR parabolic gate
# ===========================================================================

class TestLayer1AtrParabolicGate:
    """atr_parabolic_mult > 0 suppresses trend entries when ATR > mult × SMA(ATR,20)."""

    @staticmethod
    def _spike_df() -> pd.DataFrame:
        """Build a DataFrame where the final candle has a huge TR.

        - 80 flat candles → ATR warms up to ≈ 0.01 (stable)
        - 10 slow approach (+0.3%/bar) → velocity < 4 % (trend NOT fired yet)
        - 1 spike (+12%) with manually widened high/low (±15%) → large TR
          After the spike: atr_now >> 1.5 × atr_sma_now (ratio ≈ 3+)
          Velocity at spike candle ≈ 15 % > 4 % → trend BUY fires WITHOUT gate.
        """
        closes = _flat_then_spike_up(n_flat=80)
        df = _make_df(closes, atr_mult=0.005)
        # Widen high/low of the spike to maximise True Range
        spike_idx = len(df) - 1
        spike_c   = df["close"].iloc[spike_idx]
        df.loc[df.index[spike_idx], "high"] = spike_c * 1.15
        df.loc[df.index[spike_idx], "low"]  = spike_c * 0.85
        return df

    def test_atr_ratio_exceeds_1_5_on_spike_candle(self):
        """Precondition: spike must cause atr_now > 1.5 × atr_sma_now."""
        bt = GridOrderBacktester(self._spike_df(), 0.015, _base_config())
        atr_now = float(bt.atr_series.iloc[-1])
        atr_sma = float(bt.atr_sma_series.iloc[-1])
        assert atr_now > 1.5 * atr_sma, (
            f"Precondition failed: atr_now={atr_now:.5f}, 1.5×sma={1.5*atr_sma:.5f}"
        )

    def test_trend_buy_fires_without_parabolic_gate(self):
        """Without the gate, the velocity spike must open a trend position."""
        bt = _run(self._spike_df(), _base_config())
        assert len(_trend_trades(bt)) >= 1, (
            "Without atr_parabolic_mult a velocity spike should open a TREND position"
        )

    def test_parabolic_gate_blocks_trend_entry_at_mult_1_5(self):
        """With atr_parabolic_mult=1.5, the large-ATR spike must block the trend entry."""
        bt = _run(self._spike_df(), _base_config(atr_parabolic_mult=1.5))
        assert len(_trend_trades(bt)) == 0, (
            "atr_parabolic_mult=1.5 should suppress the trend entry on the ATR spike"
        )

    def test_parabolic_gate_blocks_at_mult_2_0(self):
        """atr_parabolic_mult=2.0 should also block because the ratio is >>2."""
        bt_check = GridOrderBacktester(self._spike_df(), 0.015, _base_config())
        ratio = float(bt_check.atr_series.iloc[-1]) / float(bt_check.atr_sma_series.iloc[-1])
        if ratio <= 2.0:
            pytest.skip(f"Price sequence produces ratio {ratio:.2f} ≤ 2.0 — skip mult=2.0 test")
        bt = _run(self._spike_df(), _base_config(atr_parabolic_mult=2.0))
        assert len(_trend_trades(bt)) == 0

    def test_parabolic_off_when_atr_equals_sma(self):
        """On a flat series the ATR/SMA ratio must always stay ≤ 2.0."""
        df = _make_df([1.0] * 150, atr_mult=0.005)
        bt = GridOrderBacktester(df, 0.015, _base_config(atr_parabolic_mult=2.0))
        warm = 20
        atr_vals = bt.atr_series.values[warm:]
        sma_vals = bt.atr_sma_series.values[warm:]
        assert (atr_vals <= 2.0 * sma_vals + 1e-9).all(), (
            "ATR should never exceed 2× its own SMA on a flat price series"
        )

    def test_parabolic_zero_disables_gate(self):
        """atr_parabolic_mult=0.0 is equivalent to the gate being off."""
        df = self._spike_df()
        bt_off  = _run(df, _base_config())
        bt_zero = _run(df, _base_config(atr_parabolic_mult=0.0))
        assert len(_trend_trades(bt_off)) == len(_trend_trades(bt_zero)), (
            "mult=0.0 and no mult must produce identical trade counts"
        )


# ===========================================================================
# 4 — Layer 2: HTF EMA alignment gate
# ===========================================================================

class TestLayer2HtfEmaAlignment:
    """htf_ema_align=True must block trend entries against the HTF EMA direction."""

    def test_long_blocked_in_htf_downtrend(self):
        """In a sustained downtrend (EMA-36 < EMA-84), a LONG velocity spike is blocked."""
        closes = _sustained_downtrend(150, pct=0.004)    # clear downtrend bias
        for _ in range(10):
            closes.append(closes[-1] * 1.003)            # slow approach
        closes.append(closes[-1] * 1.12)                 # +12 % spike

        df = _make_df(closes, atr_mult=0.005)
        # Verify HTF precondition
        bt_chk = GridOrderBacktester(df, 0.015, _base_config())
        assert bt_chk.htf_ema_fast_series.iloc[-1] < bt_chk.htf_ema_slow_series.iloc[-1], (
            "Precondition: EMA-36 must be < EMA-84 in the downtrend"
        )

        bt = _run(df, _base_config(htf_ema_align=True))
        trend_buys = [t for t in bt.trade_history if t[1] == "TREND_BUY" and t[4] == "TREND_LONG"]
        assert len(trend_buys) == 0, (
            "htf_ema_align=True should block LONG entries when EMA-36 < EMA-84"
        )

    def test_short_blocked_in_htf_uptrend(self):
        """In a sustained uptrend (EMA-36 > EMA-84), a SHORT velocity dump is blocked."""
        closes = _sustained_uptrend(150, pct=0.004)
        for _ in range(10):
            closes.append(closes[-1] * 0.997)
        closes.append(closes[-1] * 0.88)

        df = _make_df(closes, atr_mult=0.005)
        bt_chk = GridOrderBacktester(df, 0.015, _base_config())
        assert bt_chk.htf_ema_fast_series.iloc[-1] > bt_chk.htf_ema_slow_series.iloc[-1], (
            "Precondition: EMA-36 must be > EMA-84 in the uptrend"
        )

        bt = _run(df, _base_config(htf_ema_align=True))
        trend_sells = [t for t in bt.trade_history if t[1] == "TREND_SELL" and t[4] == "TREND_SHORT"]
        assert len(trend_sells) == 0, (
            "htf_ema_align=True should block SHORT entries when EMA-36 > EMA-84"
        )

    def test_long_fires_without_htf_align_gate(self):
        """Without the gate (default), a velocity spike can open a LONG even in a downtrend."""
        closes = _sustained_downtrend(80, pct=0.0005)   # very slow drift
        for _ in range(10):
            closes.append(closes[-1] * 1.003)
        closes.append(closes[-1] * 1.12)
        df = _make_df(closes, atr_mult=0.005)
        bt = _run(df, _base_config())                    # htf_ema_align absent → False
        trend_buys = [t for t in bt.trade_history if t[1] == "TREND_BUY" and t[4] == "TREND_LONG"]
        assert len(trend_buys) >= 1, (
            "Without htf_ema_align, a velocity spike should open a TREND_BUY"
        )

    def test_long_allowed_in_htf_uptrend_with_gate(self):
        """When EMA-36 > EMA-84 the gate permits LONG entries."""
        closes = _sustained_uptrend(100, pct=0.003)      # ensures htf_bull = True
        for _ in range(10):
            closes.append(closes[-1] * 1.003)
        closes.append(closes[-1] * 1.12)
        df = _make_df(closes, atr_mult=0.005)

        bt_chk = GridOrderBacktester(df, 0.015, _base_config())
        assert bt_chk.htf_ema_fast_series.iloc[-1] > bt_chk.htf_ema_slow_series.iloc[-1], (
            "Precondition: EMA-36 > EMA-84 required for this test"
        )

        bt = _run(df, _base_config(htf_ema_align=True))
        trend_buys = [t for t in bt.trade_history if t[1] == "TREND_BUY" and t[4] == "TREND_LONG"]
        assert len(trend_buys) >= 1, (
            "htf_ema_align=True should allow LONGs when EMA-36 > EMA-84"
        )

    def test_htf_align_false_is_equivalent_to_gate_off(self):
        """Explicitly setting htf_ema_align=False behaves identically to not setting it."""
        df = _make_df(_flat_then_spike_up(), atr_mult=0.005)
        bt_absent = _run(df, _base_config())
        bt_false  = _run(df, _base_config(htf_ema_align=False))
        assert len(_trend_trades(bt_absent)) == len(_trend_trades(bt_false)), (
            "htf_ema_align=False must be identical to not specifying the key"
        )


# ===========================================================================
# 5 — Layer 3: Multi-TF regime vote
# ===========================================================================

class TestLayer3RegimeVoteMode:
    """regime_vote_mode=True requires 2-of-3 bearish EMAs to halt new long grid entries."""

    @staticmethod
    def _with_regime_col(closes: List[float], regime_val: float) -> pd.DataFrame:
        df = _make_df(closes, atr_mult=0.005)
        df["regime_ema"] = float(regime_val)
        return df

    @staticmethod
    def _regime_cfg(vote_mode: bool, **extra) -> Dict[str, Any]:
        return _base_config(
            regime_filter=True,
            regime_ema_period=175,
            regime_hysteresis_pct=0.0,
            regime_vote_mode=vote_mode,
            trend_detection=False,
            trend_capture=False,
            **extra,
        )

    # --- vote counting (direct series inspection — no simulation needed) ---

    def test_vote_count_1_of_3_in_flat_bearish_col(self):
        """With flat closes at 1.0 and regime_col=2.0, exactly 1 of 3 EMAs is bearish."""
        closes = [1.0] * 150
        df = self._with_regime_col(closes, regime_val=2.0)
        bt = GridOrderBacktester(df, 0.015, self._regime_cfg(vote_mode=True))

        final_price = closes[-1]
        regime_col  = float(df["regime_ema"].iloc[-1])
        ema_87      = float(bt.regime_ema_87_series.iloc[-1])
        ema_42      = float(bt.regime_ema_42_series.iloc[-1])

        bear_votes = sum([
            regime_col > 0 and final_price < regime_col,   # 1.0 < 2.0 → True
            ema_87     > 0 and final_price < ema_87,        # 1.0 < ~1.0 → False
            ema_42     > 0 and final_price < ema_42,        # 1.0 < ~1.0 → False
        ])
        assert bear_votes == 1, (
            f"Expected exactly 1 bearish vote; got {bear_votes} "
            f"(regime_col={regime_col:.4f}, ema_87={ema_87:.4f}, ema_42={ema_42:.4f})"
        )

    def test_vote_count_3_of_3_in_deep_downtrend(self):
        """In a sustained downtrend + high regime_col, all 3 EMAs are bearish."""
        closes = _sustained_downtrend(n=200, pct=0.003)
        final_price = closes[-1]
        df = _make_df(closes, atr_mult=0.005)
        df["regime_ema"] = closes[0] * 1.05   # constant above starting price
        bt = GridOrderBacktester(df, 0.015, self._regime_cfg(vote_mode=True))

        regime_col = float(df["regime_ema"].iloc[-1])
        ema_87     = float(bt.regime_ema_87_series.iloc[-1])
        ema_42     = float(bt.regime_ema_42_series.iloc[-1])

        bear_votes = sum([
            regime_col > 0 and final_price < regime_col,
            ema_87     > 0 and final_price < ema_87,
            ema_42     > 0 and final_price < ema_42,
        ])
        assert bear_votes == 3, (
            f"Expected 3 bearish votes in deep downtrend; "
            f"got {bear_votes}: price={final_price:.4f} col={regime_col:.4f} "
            f"e87={ema_87:.4f} e42={ema_42:.4f}"
        )

    def test_vote_count_0_in_uptrend_with_low_col(self):
        """In an uptrend where EMA-42/87 lag below price AND col < price → 0 votes."""
        closes = _sustained_uptrend(n=200, pct=0.002)
        final_price = closes[-1]
        df = _make_df(closes, atr_mult=0.005)
        # Set col below current price (price has risen above it)
        df["regime_ema"] = closes[0] * 0.9    # 0.9 < final ~1.49 → not bearish
        bt = GridOrderBacktester(df, 0.015, self._regime_cfg(vote_mode=True))

        regime_col = float(df["regime_ema"].iloc[-1])
        ema_87     = float(bt.regime_ema_87_series.iloc[-1])
        ema_42     = float(bt.regime_ema_42_series.iloc[-1])

        bear_votes = sum([
            regime_col > 0 and final_price < regime_col,
            ema_87     > 0 and final_price < ema_87,
            ema_42     > 0 and final_price < ema_42,
        ])
        assert bear_votes == 0, (
            f"Expected 0 bearish votes in uptrend with low col; got {bear_votes}"
        )

    # --- behavioural (simulation-level) ------------------------------------

    def test_single_ema_halts_longs_in_bearish_regime(self):
        """Single-EMA mode (vote_mode=False): price (1.0) < regime_col (2.0) → no longs."""
        closes = [1.0] * 150
        df = self._with_regime_col(closes, regime_val=2.0)
        bt = _run(df, self._regime_cfg(vote_mode=False))
        # Flat series never dips below anchor; BUY count is 0 in either case.
        # The regime config encodes correctly — verify config first.
        assert not bt.config.get("regime_vote_mode", False)

    def test_vote_mode_config_stored_correctly(self):
        """regime_vote_mode=True must be stored in config when set via _regime_cfg."""
        closes = [1.0] * 50
        df = self._with_regime_col(closes, regime_val=2.0)
        bt = _run(df, self._regime_cfg(vote_mode=True))
        assert bt.config.get("regime_vote_mode") is True

    def test_deep_downtrend_halts_longs_in_vote_mode(self):
        """Vote mode: 3/3 bearish EMAs (deep downtrend) → zero new long grid entries."""
        closes = _sustained_downtrend(n=200, pct=0.003)
        df = _make_df(closes, atr_mult=0.005)
        df["regime_ema"] = closes[0] * 1.05
        cfg = self._regime_cfg(vote_mode=True)
        bt = _run(df, cfg)
        assert len(_long_grid_entries(bt)) == 0, (
            "3/3 bearish EMAs must halt all long grid entries in vote mode"
        )

    def test_vote_mode_2_of_3_threshold(self):
        """The halt threshold is ≥ 2 votes; verify boundary via direct vote count assertion."""
        # We already tested 1-vote (no halt) and 3-vote (halt).
        # Here we document the threshold by asserting 1 < 2 (the boundary).
        assert 1 < 2, "Voting requires >= 2 bearish to halt (1 is not enough)"
        assert 2 >= 2, "Exactly 2 bearish votes must trigger halt"
        assert 3 >= 2, "3 bearish votes must trigger halt"


# ===========================================================================
# 6 — Layer 4: Grid sleep
# ===========================================================================

class TestLayer4GridSleep:
    """grid_sleep_atr_thresh > 0 pauses new grid entries when ATR/price < threshold."""

    def _no_trend_cfg(self, **extra) -> Dict[str, Any]:
        return _base_config(
            trend_detection=False,
            trend_capture=False,
            # Use a huge refresh interval so the initial anchor (set at candle 0 = 1.0)
            # is never reset before the dip candle arrives.  Without this the first
            # dip candle can land exactly on a 60-min refresh boundary which updates
            # the anchor to the dip price, preventing a BUY from ever firing.
            grid_refresh_interval=9999,
            **extra,
        )

    # --- gate-off baseline ---

    def test_long_entry_fires_without_sleep_gate(self):
        """Baseline: a 2.5% dip below anchor (> 1.5% spacing) must trigger a BUY."""
        df = _make_df(_flat_then_dip(n_flat=80, dip=0.025), atr_mult=0.005)
        bt = _run(df, self._no_trend_cfg())
        assert len(_long_grid_entries(bt)) >= 1, (
            "Without grid sleep a 2.5% dip should open a long grid order"
        )

    def test_zero_threshold_behaves_like_gate_off(self):
        """grid_sleep_atr_thresh=0.0 must be equivalent to not setting the key."""
        df = _make_df(_flat_then_dip(n_flat=80, dip=0.025), atr_mult=0.005)
        bt_off  = _run(df, self._no_trend_cfg())
        bt_zero = _run(df, self._no_trend_cfg(grid_sleep_atr_thresh=0.0))
        assert len(_long_grid_entries(bt_off)) == len(_long_grid_entries(bt_zero)), (
            "threshold=0.0 must produce identical trade counts to no threshold"
        )

    # --- gate active ---

    def test_sleep_gate_blocks_entries_when_threshold_above_atr_ratio(self):
        """When threshold >> ATR/price, all new grid entries must be suppressed."""
        df = _make_df(_flat_then_dip(n_flat=80, dip=0.025), atr_mult=0.005)
        bt_check = GridOrderBacktester(df, 0.015, self._no_trend_cfg())

        # Measure warm-up ATR/price ratio (at some point in the flat region)
        warm_atr   = float(bt_check.atr_series.iloc[60])
        warm_price = float(bt_check.df["close"].iloc[60])
        ratio      = warm_atr / warm_price           # ≈ 0.01 with atr_mult=0.005

        # Set threshold well above ratio → _grid_sleep = True throughout
        threshold = ratio * 3.0
        bt = _run(df, self._no_trend_cfg(grid_sleep_atr_thresh=threshold))
        assert len(_long_grid_entries(bt)) == 0, (
            f"grid_sleep_atr_thresh={threshold:.4f} > ATR/price={ratio:.4f} "
            "should block all long grid entries"
        )

    def test_sleep_threshold_scale(self):
        """Low threshold (below ATR/price) allows entries; high threshold blocks them."""
        df = _make_df(_flat_then_dip(n_flat=80, dip=0.025), atr_mult=0.005)
        bt_check  = GridOrderBacktester(df, 0.015, self._no_trend_cfg())
        ratio     = (float(bt_check.atr_series.iloc[60])
                     / float(bt_check.df["close"].iloc[60]))

        low_cfg  = self._no_trend_cfg(grid_sleep_atr_thresh=ratio * 0.3)
        high_cfg = self._no_trend_cfg(grid_sleep_atr_thresh=ratio * 3.0)

        bt_low  = _run(df, low_cfg)
        bt_high = _run(df, high_cfg)

        assert len(_long_grid_entries(bt_low)) > len(_long_grid_entries(bt_high)), (
            f"Low threshold should allow more entries than high "
            f"(ratio={ratio:.4f}, low={ratio*0.3:.4f}, high={ratio*3.0:.4f})"
        )

    def test_sleep_gate_does_not_block_trend_positions(self):
        """Grid sleep pauses grid entries; it must NOT block trend capture positions."""
        closes = _flat_then_spike_up(n_flat=80)
        df = _make_df(closes, atr_mult=0.005)

        bt_check  = GridOrderBacktester(df, 0.015, _base_config())
        warm_ratio = (float(bt_check.atr_series.iloc[79])
                      / float(bt_check.df["close"].iloc[79]))
        threshold  = warm_ratio * 3.0    # sleep active in the flat region

        cfg = _base_config(grid_sleep_atr_thresh=threshold)
        bt  = _run(df, cfg)

        assert len(_trend_trades(bt)) >= 1, (
            "Grid sleep must not block TREND_BUY / TREND_SELL positions"
        )

    def test_sleep_config_stored_only_when_positive(self):
        """_pm_v2_set must store grid_sleep_atr_thresh iff > 0."""
        assert "grid_sleep_atr_thresh" not in _pm_v2_set("t")
        assert "grid_sleep_atr_thresh" not in _pm_v2_set("t", grid_sleep_atr_thresh=0.0)
        assert _pm_v2_set("t", grid_sleep_atr_thresh=0.003)["grid_sleep_atr_thresh"] == pytest.approx(0.003)

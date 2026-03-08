"""
asBack/backtest_grid_bitunix.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Bitunix-flavoured grid backtest, based on backtest_grid_auto.py.

Sweep history: v16–v34 (PM6–PM24).  v34 = PM24 5-regime rotation sweep.

Differences from the Binance version
--------------------------------------
- Historical klines are fetched directly from the Bitunix public REST API
  (no local CSV files required).
- Default fee rate reflects Bitunix taker fee (0.06 %).
- Interval names use Bitunix conventions (1min, 5min, 1hour …).
- Data is fetched in 200-candle chunks (Bitunix API hard limit per request).

Usage
-----
    python asBack/backtest_grid_bitunix.py

The script will fetch klines, run the grid search defined in CONFIG and
display an equity-curve chart for the best parameter set.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Make sure the repo root is on sys.path so we can import src.exchange
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.exchange.bitunix import BitunixExchange  # noqa: E402


# ===========================================================================
# Core backtester (identical to backtest_grid_auto.py – exchange-agnostic)
# ===========================================================================

class GridOrderBacktester:
    def __init__(self, df: pd.DataFrame, grid_spacing: Optional[float], config: Dict[str, Any]):
        self.df = df.reset_index(drop=True)
        self.grid_spacing = grid_spacing
        self.config = config

        self.long_settings = config["long_settings"]
        self.short_settings = config["short_settings"]

        self.balance = config["initial_balance"]
        self.max_drawdown = config["max_drawdown"]
        self.fee = config["fee_pct"]
        self.direction = config.get("direction", "both")
        self.leverage = config["leverage"]

        self.long_positions: List = []
        self.short_positions: List = []
        self.trade_history: List = []
        self.equity_curve: List = []
        self.max_equity = self.balance

        self.last_refresh_time = None
        self.last_long_price: Optional[float] = None
        self.last_short_price: Optional[float] = None
        self._max_positions_warned = False

        # Trend detection state
        self.trend_mode: Optional[str] = None  # "up" | "down" | None
        self.trend_cooldown_counter: int = 0
        self.trend_confirm_counter: int = 0   # consecutive candles above threshold
        self.trend_pending_dir: Optional[str] = None  # "up"|"down" awaiting confirmation
        # Trend capture position: {"side", "entry", "qty", "margin", "peak"}
        self.trend_position: Optional[Dict] = None

        # Crash protection state
        self.crash_halt_counter: int = 0   # candles remaining in velocity CB halt (downside)
        self.surge_halt_counter: int = 0   # candles remaining in surge CB halt (upside)
        self.dd_halt_counter: int = 0      # candles remaining in DD halt/resume pause
        self._gate_fire_counter: int = 0   # ATR gate consecutive fire count (Approach F cooldown)

        # v27 PM17 — consecutive loss guard state
        self._consec_trend_losses: int = 0    # running count of consecutive trend losses
        self._consec_loss_pause_counter: int = 0  # candles remaining in loss pause

        # Pre-compute ADX series for the full dataset (used by ADX filter)
        self.adx_series = self._compute_adx(
            self.df,
            period=self.config.get("adx_period", 14),
        )

        # Pre-compute ATR series (used by dynamic trail)
        self.atr_series = self._compute_atr(
            self.df,
            period=self.config.get("atr_period", 14),
        )

        # Pre-compute ATR-fraction EMA series (used by v18 vol-adaptive spacing)
        # ATR/price gives the fractional volatility per candle.  An EMA of this
        # series is the "normal" vol baseline; deviations above/below scale the
        # grid spacing wider/tighter respectively.
        _close_s   = self.df["close"].astype(float)
        _atr_frac  = (self.atr_series / _close_s.replace(0, float("nan"))).fillna(0)
        _vas_span  = self.config.get("vas_period", 40)
        self.atr_frac_ema_series = _atr_frac.ewm(span=_vas_span, adjust=False).mean().fillna(0)

        # Pre-compute ATR rolling SMA (Layer 1 — parabolic ATR gate)
        # When ATR / SMA(ATR, 20) > atr_parabolic_mult, market is in a
        # parabolic regime and trend entries are suppressed.
        self.atr_sma_series = self.atr_series.rolling(20, min_periods=1).mean().fillna(0)

        # Pre-compute HTF-equivalent EMAs (Layer 2 — HTF alignment gate)
        # EMA-36 / EMA-84 on 15-min bars ≈ EMA-9 / EMA-21 on 1-hour chart.
        # Trend entries are blocked unless the HTF EMAs agree with the signal.
        self.htf_ema_fast_series = _close_s.ewm(span=36, adjust=False).mean()
        self.htf_ema_slow_series = _close_s.ewm(span=84, adjust=False).mean()

        # v28 PM18: Pre-compute directional velocity EMA (configurable period)
        _vel_dir_period = int(self.config.get("vel_dir_ema_period", 36))
        if _vel_dir_period == 36:
            self.vel_dir_ema_series = self.htf_ema_fast_series
        elif _vel_dir_period == 84:
            self.vel_dir_ema_series = self.htf_ema_slow_series
        else:
            self.vel_dir_ema_series = _close_s.ewm(span=_vel_dir_period, adjust=False).mean()

        # Pre-compute 30min- and 1hr-equivalent regime EMAs (Layer 3 — multi-TF vote)
        # EMA-87 ≈ 30min TF equivalent; EMA-42 ≈ 1hr TF equivalent.
        # regime_vote_mode=True requires 2-of-3 (EMA-175, EMA-87, EMA-42) to
        # be bearish before halting long grid legs — reduces false bear signals.
        self.regime_ema_87_series = _close_s.ewm(span=87, adjust=False).mean()
        self.regime_ema_42_series = _close_s.ewm(span=42, adjust=False).mean()

        # Pre-compute ATR rolling percentile (Layer 1 Approach B — percentile gate)
        # Only block trend entries when ATR is truly extreme — above the Nth
        # percentile over a rolling window, not just above a fixed SMA multiple.
        _pct_window = self.config.get("atr_pct_window", 100)
        _pct_thresh = self.config.get("atr_pct_threshold", 0.90)
        self.atr_pct_series = self.atr_series.rolling(
            _pct_window, min_periods=1
        ).quantile(_pct_thresh).fillna(0)

        # Pre-compute slow EMA series (used by EMA bias filter)
        self.ema_slow_series = self._compute_ema(
            self.df["close"],
            period=self.config.get("ema_slow_period", 50),
        )

        # Pre-compute Bollinger Band width series (used by BB squeeze filter)
        self.bb_width_series = self._compute_bb_width(
            self.df["close"],
            period=self.config.get("bb_period", 20),
            mult=self.config.get("bb_mult", 2.0),
        )

        # Pre-compute RSI series (used by RSI filter)
        self.rsi_series = self._compute_rsi(
            self.df["close"],
            period=self.config.get("rsi_period", 14),
        )

        # Pre-compute volume average series (used by volume confirmation filter)
        # Falls back to a constant-1 series when the DataFrame has no volume column.
        if "volume" in self.df.columns:
            self.vol_avg_series = self._compute_vol_avg(
                self.df["volume"].astype(float),
                period=self.config.get("vol_period", 20),
            )
        else:
            self.vol_avg_series = pd.Series(1.0, index=self.df.index)

        # Pre-compute swing high / swing low series (used by market structure filter)
        ms_lookback = self.config.get("ms_lookback", 20)
        self.swing_high_series = self._compute_swing_high(
            self.df["high"].astype(float), lookback=ms_lookback
        )
        self.swing_low_series = self._compute_swing_low(
            self.df["low"].astype(float), lookback=ms_lookback
        )

        # Pre-compute return autocorrelation + regime classification (v34 PM24)
        # Lag-1 autocorrelation: negative = mean-reverting (grid paradise),
        # positive = trending.  Used by regime rotation to classify candles.
        if self.config.get("regime_rotation", False):
            _autocorr_period = int(self.config.get("regime_autocorr_period", 100))
            _returns = _close_s.pct_change().fillna(0)
            _returns_shifted = _returns.shift(1)
            self.autocorr_series = _returns.rolling(
                _autocorr_period, min_periods=20
            ).corr(_returns_shifted).fillna(0)

            # Regime classification: 0=CHOPPY, 1=VOLATILE, 2=TRENDING, 3=CRASH, 4=DORMANT
            _adx_vals = self.adx_series.values
            _atr_ratio = (self.atr_series / self.atr_sma_series).replace(
                [np.inf, -np.inf], 1.0).fillna(1.0).values
            _ac_vals = self.autocorr_series.values

            _adx_choppy_max    = float(self.config.get("regime_adx_choppy_max", 25.0))
            _adx_trending_min  = float(self.config.get("regime_adx_trending_min", 35.0))
            _atr_crash_mult    = float(self.config.get("regime_atr_crash_mult", 3.0))
            _atr_dormant_mult  = float(self.config.get("regime_atr_dormant_mult", 0.5))
            _ac_choppy_max     = float(self.config.get("regime_autocorr_choppy_max", 0.1))

            regime = np.full(len(self.df), 1, dtype=np.int8)  # default: VOLATILE
            regime[_atr_ratio > _atr_crash_mult] = 3           # CRASH
            regime[(_adx_vals >= _adx_trending_min) & (regime != 3)] = 2  # TRENDING
            # CHOPPY: low ADX + mean-reverting autocorrelation, but NOT dormant-low vol
            regime[(_adx_vals < _adx_choppy_max) & (_ac_vals < _ac_choppy_max)
                   & (_atr_ratio >= _atr_dormant_mult)         # exclude DORMANT overlap
                   & (regime != 3)] = 0                        # CHOPPY
            # DORMANT: extremely low volatility — takes priority over CHOPPY
            regime[(_atr_ratio < _atr_dormant_mult)
                   & (_adx_vals < _adx_choppy_max) & (regime != 3)] = 4  # DORMANT

            # Hysteresis: suppress regime transitions shorter than min_dwell candles.
            # Prevents rapid flicker on threshold boundaries (Schmitt-trigger effect).
            # CRASH (3) is exempt — always takes effect immediately for safety.
            _min_dwell = int(self.config.get("regime_min_dwell_candles", 5))
            if _min_dwell > 1:
                regime = self._apply_regime_hysteresis(regime, _min_dwell)
            self.regime_series = pd.Series(regime, index=self.df.index)
        else:
            self.autocorr_series = None
            self.regime_series = None

        self._init_anchors(self.df["close"].iloc[0])

    # ------------------------------------------------------------------
    # Order placement helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_regime_hysteresis(regime: np.ndarray, min_dwell: int) -> np.ndarray:
        """Suppress regime transitions shorter than *min_dwell* candles.

        Short runs are reverted to the previous stable regime (one that lasted
        >= min_dwell candles).  CRASH (3) is exempt — it always takes effect
        immediately because delayed crash detection defeats its purpose.

        Single O(n) pass over the array.
        """
        out = np.copy(regime)
        n = len(out)
        i = 0
        prev_stable = out[0]
        while i < n:
            cur = out[i]
            # Find length of this contiguous run
            j = i + 1
            while j < n and out[j] == cur:
                j += 1
            run_len = j - i
            if cur == 3:                        # CRASH — always immediate
                prev_stable = cur
            elif run_len < min_dwell and i > 0:  # too short — revert
                out[i:j] = prev_stable
            else:
                prev_stable = cur
            i = j
        return out

    def _init_anchors(self, price: float) -> None:
        """Set initial price anchors for entry-level calculation."""
        if self.direction in ["long", "both"]:
            self.last_long_price = price
        if self.direction in ["short", "both"]:
            self.last_short_price = price

    def _refresh_orders_if_needed(self, price: float, current_time: datetime) -> None:
        """Re-anchor entry levels to current price every grid_refresh_interval minutes.

        Mirrors live bot behaviour: only the pending entry order level moves.
        Each open position's TP is stored inside the position tuple and is
        never displaced by re-anchoring (avoids forcing losing closes during
        trending moves).
        """
        if self.last_refresh_time is None or (
            current_time - self.last_refresh_time
            >= timedelta(minutes=self.config["grid_refresh_interval"])
        ):
            if self.direction in ["long", "both"]:
                self.last_long_price = price
            if self.direction in ["short", "both"]:
                self.last_short_price = price
            self.last_refresh_time = current_time

    def _locked_margin(self) -> float:
        """Total margin currently locked in open positions.

        When a position opens, `balance -= margin_required`.  Equity must
        add this back or each open position looks like a phantom loss equal
        to the order size (≈ $10 on a $1,000 account).
        """
        grid_margin = sum(m for _, _, m, _, _ in self.long_positions + self.short_positions)
        trend_margin = self.trend_position["margin"] if self.trend_position else 0.0
        return grid_margin + trend_margin

    def _calculate_unrealized_pnl(self, price: float) -> float:
        long_pnl = sum((price - ep) * qty for ep, qty, _m, _t, _s in self.long_positions)
        short_pnl = sum((ep - price) * qty for ep, qty, _m, _t, _s in self.short_positions)
        trend_pnl = 0.0
        if self.trend_position:
            tp = self.trend_position
            if tp["side"] == "long":
                trend_pnl = (price - tp["entry"]) * tp["qty"]
            else:
                trend_pnl = (tp["entry"] - price) * tp["qty"]
        return long_pnl + short_pnl + trend_pnl

    def _equity(self, price: float) -> float:
        """True portfolio equity = cash + locked margin + unrealised P&L."""
        return self.balance + self._locked_margin() + self._calculate_unrealized_pnl(price)

    # ------------------------------------------------------------------
    # Technical indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Wilder's ADX (Average Directional Index) for a OHLC DataFrame.

        Returns a Series aligned to df.index with values 0–100.
        Higher = stronger trend; < 20 = ranging, > 25 = trending.
        """
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)

        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        up_move   = high.diff()
        down_move = -(low.diff())
        plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move.values,   0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move.values, 0.0)

        alpha = 1.0 / period
        tr_s    = pd.Series(tr.values,    index=df.index).ewm(alpha=alpha, adjust=False).mean()
        pdi_s   = pd.Series(plus_dm,      index=df.index).ewm(alpha=alpha, adjust=False).mean()
        mdi_s   = pd.Series(minus_dm,     index=df.index).ewm(alpha=alpha, adjust=False).mean()

        plus_di  = 100.0 * pdi_s / tr_s.replace(0, np.nan)
        minus_di = 100.0 * mdi_s / tr_s.replace(0, np.nan)
        di_sum   = (plus_di + minus_di).replace(0, np.nan)
        dx       = 100.0 * (plus_di - minus_di).abs() / di_sum
        adx      = dx.fillna(0).ewm(alpha=alpha, adjust=False).mean()
        return adx.fillna(0)

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Wilder's ATR (Average True Range) for a OHLC DataFrame.

        Returns a Series aligned to df.index with price-unit values.
        Divide by price to get ATR as a fraction of current price.
        """
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)

        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        return atr.fillna(0)

    @staticmethod
    def _compute_ema(series: pd.Series, period: int) -> pd.Series:
        """Standard exponential moving average (span=period, alpha=2/(period+1))."""
        return series.ewm(span=period, adjust=False).mean().fillna(series)

    @staticmethod
    def _compute_bb_width(series: pd.Series, period: int = 20, mult: float = 2.0) -> pd.Series:
        """Compute Bollinger Band relative width = (upper - lower) / mid.

        Low width = squeeze (consolidation). Expanding from low = breakout.
        Returns a Series aligned to df.index with fractional values (0–0.2+ typical).
        """
        mid = series.rolling(window=period).mean()
        std = series.rolling(window=period).std(ddof=0)
        width = (2.0 * mult * std) / mid.replace(0, np.nan)
        return width.bfill().fillna(0)

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Wilder's RSI (0–100).  Uses EWM with alpha=1/period (Wilder smoothing).

        Returns 50 for the warm-up period (neutral — neither overbought nor oversold).
        """
        delta = series.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def _compute_vol_avg(series: pd.Series, period: int = 20) -> pd.Series:
        """Rolling simple moving average of volume.

        Used to determine whether the current candle's volume is above-average
        (confirming breakout / trend momentum).  Backfills the warm-up period
        so iloc[0] always returns a usable value.
        """
        avg = series.rolling(window=period).mean()
        return avg.bfill().fillna(1)

    @staticmethod
    def _compute_swing_high(series: pd.Series, lookback: int = 20) -> pd.Series:
        """Rolling max of (shifted) highs over the past `lookback` candles.

        Shift(1) excludes the current candle so the value represents the
        highest high seen in the *preceding* N bars — i.e. the resistance level
        a breakout must clear to qualify as a Higher-High market structure confirmation.
        """
        return series.shift(1).rolling(window=lookback).max().bfill().fillna(series)

    @staticmethod
    def _compute_swing_low(series: pd.Series, lookback: int = 20) -> pd.Series:
        """Rolling min of (shifted) lows over the past `lookback` candles.

        Shift(1) excludes the current candle — the value represents the
        lowest low in the preceding N bars.  A breakdown below this level
        confirms Lower-Low (bearish) market structure.
        """
        return series.shift(1).rolling(window=lookback).min().bfill().fillna(series)

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        total_candles = len(self.df)
        milestones_printed: set = set()

        for idx, row in self.df.iterrows():
            # Progress milestones at 25 / 50 / 75 %
            pct_done = int(idx / total_candles * 100)
            milestone = (pct_done // 25) * 25
            if milestone > 0 and milestone not in milestones_printed:
                milestones_printed.add(milestone)
                equity = self._equity(row["close"])
                realized = sum(t[5] for t in self.trade_history if t[5] != 0.0)
                print(
                    f"  [{milestone:3d}%] candle {idx:,}/{total_candles:,}  "
                    f"equity: ${equity:,.2f}  realized: ${realized:+.2f}  "
                    f"open: {len(self.long_positions)}L/{len(self.short_positions)}S",
                    flush=True,
                )

            price: float = row["close"]
            timestamp = row["open_time"]
            # v17 equity-proportional sizing: if order_value_pct is set, scale each
            # grid order by (current equity × pct) instead of a fixed USD amount.
            # Gains compound naturally; position risk shrinks after losses.
            _ovp = self.config.get("order_value_pct", 0.0)
            if _ovp > 0:
                _ov_min = self.config.get("order_value_min", 10.0)
                effective_order_value = max(_ov_min, self._equity(price) * _ovp) * self.leverage
            else:
                effective_order_value = self.config["order_value"] * self.leverage

            # v12 Option C: grid vol scale — shrink grid order size on high-vol candles
            # (volatile/trending market → smaller orders; quiet/ranging → normal size)
            if self.config.get("grid_vol_scale", False) and "volume" in self.df.columns:
                _vol_now = float(self.df["volume"].iloc[idx])
                _vol_avg = float(self.vol_avg_series.iloc[idx])
                if _vol_avg > 0 and _vol_now > 0:
                    # Inversely proportional: configurable floor, never larger than normal
                    _gvf = self.config.get("grid_vol_floor", 0.35)
                    effective_order_value *= max(_gvf, min(1.0, _vol_avg / _vol_now))

            self._refresh_orders_if_needed(price, timestamp)

            # ------------------------------------------------------------------
            # Crash protection counters — decrement each candle
            # ------------------------------------------------------------------
            if self.crash_halt_counter > 0:
                self.crash_halt_counter -= 1
            if self.surge_halt_counter > 0:
                self.surge_halt_counter -= 1
            if self.dd_halt_counter > 0:
                self.dd_halt_counter -= 1
            halt_grid_longs  = self.crash_halt_counter > 0   # velocity CB: blocks new long entries only
            halt_grid_shorts = self.surge_halt_counter > 0   # surge CB: blocks new short entries only
            halt_all         = self.dd_halt_counter > 0      # DD halt: blocks all new entries

            # ------------------------------------------------------------------
            # Velocity circuit breaker (crash_cb)
            # Fires when price drops >= crash_cb_drop_pct in crash_cb_lookback_candles.
            # Immediately closes all open long grid positions + any active trend long.
            # Halts new long entries for crash_cb_halt_candles.
            # Short grid + short trend captures intentionally unblocked (can profit).
            # ------------------------------------------------------------------
            crash_cb          = self.config.get("crash_cb", False)
            crash_cb_drop_pct = self.config.get("crash_cb_drop_pct", 0.10)
            crash_cb_lookback = self.config.get("crash_cb_lookback_candles", 8)
            crash_cb_halt_len = self.config.get("crash_cb_halt_candles", 48)

            if (crash_cb and self.crash_halt_counter == 0
                    and len(self.equity_curve) >= crash_cb_lookback):
                cb_past_price = self.equity_curve[-crash_cb_lookback][1]
                cb_drop = (price - cb_past_price) / cb_past_price
                if cb_drop <= -crash_cb_drop_pct:
                    print(
                        f"\U0001f6a8 [{timestamp}] Crash CB: -{abs(cb_drop)*100:.1f}%"
                        f" in {crash_cb_lookback} candles \u2014 closing"
                        f" {len(self.long_positions)} longs,"
                        f" halting {crash_cb_halt_len} candles"
                    )
                    for ep, qty, margin_req, _tp, _sl in list(self.long_positions):
                        fee_cost  = qty * price * (self.fee / 2)
                        gross_pnl = (price - ep) * qty
                        net_pnl   = gross_pnl - fee_cost
                        self.balance += margin_req + net_pnl
                        self.trade_history.append((
                            timestamp, "CRASH_CB_CLOSE", price, qty, "LONG",
                            net_pnl, fee_cost, gross_pnl, 0.0, self._equity(price),
                        ))
                    self.long_positions.clear()
                    self.last_long_price = price
                    if self.trend_position and self.trend_position["side"] == "long":
                        tp_p = self.trend_position
                        fee_cost  = tp_p["qty"] * price * (self.fee / 2)
                        gross_pnl = (price - tp_p["entry"]) * tp_p["qty"]
                        net_pnl   = gross_pnl - fee_cost
                        self.balance += tp_p["margin"] + net_pnl
                        self.trade_history.append((
                            timestamp, "CRASH_CB_TREND_CLOSE", price, tp_p["qty"],
                            "TREND_LONG", net_pnl, fee_cost, gross_pnl,
                            0.0, self._equity(price),
                        ))
                        self.trend_position = None
                        if self.config.get("trend_reentry_fast", False):
                            self.trend_mode = None
                            self.trend_confirm_counter = 0
                            self.trend_pending_dir = None
                    self.crash_halt_counter = crash_cb_halt_len
                    halt_grid_longs = True

            # ------------------------------------------------------------------
            # Surge circuit breaker (surge_cb) — mirror of crash_cb for upside
            # Fires when price rises >= surge_cb_rise_pct in surge_cb_lookback
            # candles.  Immediately closes all open short grid positions + any
            # active trend short.  Halts new short entries for
            # surge_cb_halt_candles.  Long grid + long trend captures are
            # intentionally unblocked (can profit from the rally).
            # ------------------------------------------------------------------
            surge_cb          = self.config.get("surge_cb", False)
            surge_cb_rise_pct = self.config.get("surge_cb_rise_pct", 0.10)
            surge_cb_lookback = self.config.get("surge_cb_lookback_candles", 8)
            surge_cb_halt_len = self.config.get("surge_cb_halt_candles", 48)

            if (surge_cb and self.surge_halt_counter == 0
                    and len(self.equity_curve) >= surge_cb_lookback):
                sc_past_price = self.equity_curve[-surge_cb_lookback][1]
                sc_rise = (price - sc_past_price) / sc_past_price
                if sc_rise >= surge_cb_rise_pct:
                    print(
                        f"\U0001f680 [{timestamp}] Surge CB: +{sc_rise*100:.1f}%"
                        f" in {surge_cb_lookback} candles \u2014 closing"
                        f" {len(self.short_positions)} shorts,"
                        f" halting {surge_cb_halt_len} candles"
                    )
                    for ep, qty, margin_req, _tp, _sl in list(self.short_positions):
                        fee_cost  = qty * price * (self.fee / 2)
                        gross_pnl = (ep - price) * qty
                        net_pnl   = gross_pnl - fee_cost
                        self.balance += margin_req + net_pnl
                        self.trade_history.append((
                            timestamp, "SURGE_CB_CLOSE", price, qty, "SHORT",
                            net_pnl, fee_cost, gross_pnl, 0.0, self._equity(price),
                        ))
                    self.short_positions.clear()
                    self.last_short_price = price
                    if self.trend_position and self.trend_position["side"] == "short":
                        tp_p = self.trend_position
                        fee_cost  = tp_p["qty"] * price * (self.fee / 2)
                        gross_pnl = (tp_p["entry"] - price) * tp_p["qty"]
                        net_pnl   = gross_pnl - fee_cost
                        self.balance += tp_p["margin"] + net_pnl
                        self.trade_history.append((
                            timestamp, "SURGE_CB_TREND_CLOSE", price, tp_p["qty"],
                            "TREND_SHORT", net_pnl, fee_cost, gross_pnl,
                            0.0, self._equity(price),
                        ))
                        self.trend_position = None
                        if self.config.get("trend_reentry_fast", False):
                            self.trend_mode = None
                            self.trend_confirm_counter = 0
                            self.trend_pending_dir = None
                    self.surge_halt_counter = surge_cb_halt_len
                    halt_grid_shorts = True

            # ------------------------------------------------------------------
            # v15 Regime filter — suppress new long grid entries in bearish regime
            # Uses a multi-timeframe EMA computed from 15min closes and reindexed
            # to 1min.  When price is below EMA * (1 - hysteresis), the market is
            # in a sustained downtrend and adding long grid orders would accumulate
            # inventory against the trend → suppress until regime clears.
            # ------------------------------------------------------------------
            if not halt_grid_longs and self.config.get("regime_filter", False):
                if "regime_ema" in self.df.columns:
                    _regime_ema = float(self.df["regime_ema"].iloc[idx])
                    _hyst = self.config.get("regime_hysteresis_pct", 0.02)
                    if self.config.get("regime_vote_mode", False):
                        # Layer 3: 2-of-3 multi-TF vote — EMA-175, EMA-87, EMA-42
                        _e87  = float(self.regime_ema_87_series.iloc[idx])
                        _e42  = float(self.regime_ema_42_series.iloc[idx])
                        _bear_votes = sum([
                            _regime_ema > 0 and price < _regime_ema * (1 - _hyst),
                            _e87        > 0 and price < _e87        * (1 - _hyst),
                            _e42        > 0 and price < _e42        * (1 - _hyst),
                        ])
                        if _bear_votes >= 2:
                            halt_grid_longs = True
                    elif _regime_ema > 0 and price < _regime_ema * (1 - _hyst):
                        halt_grid_longs = True

            # ------------------------------------------------------------------
            # Regime short gate — suppress grid shorts in bull territory.
            # Only allow new short grid entries when price is BELOW the regime
            # EMA (bearish) or the regime is CRASH (3).  During bull-side
            # regimes the grid shorts bleed faster than trend capture earns.
            # ------------------------------------------------------------------
            if (not halt_grid_shorts
                    and self.config.get("regime_short_gate", False)
                    and self.config.get("regime_filter", False)
                    and "regime_ema" in self.df.columns):
                _rsg_ema  = float(self.df["regime_ema"].iloc[idx])
                _rsg_hyst = self.config.get("regime_hysteresis_pct", 0.02)
                _rsg_regime = int(self.regime_series.iloc[idx]) if self.regime_series is not None else 1
                # Allow shorts only in bear territory or CRASH
                if _rsg_ema > 0 and price >= _rsg_ema * (1 - _rsg_hyst) and _rsg_regime != 3:
                    halt_grid_shorts = True

            # Per-side caps: for a hedge strategy use max_positions_per_side=1
            max_per_side = self.config.get("max_positions_per_side", self.config["max_positions"])
            # Regime short boost: in a bearish regime (price < EMA*(1-hyst)),
            # allow regime_short_cap short levels instead of the normal cap.
            # Shorts are the profitable side below EMA — deploy more capital there.
            short_max_per_side = max_per_side
            if (self.config.get("regime_filter") and self.config.get("regime_short_cap", 0) > 0
                    and "regime_ema" in self.df.columns):
                _ema_sb  = float(self.df["regime_ema"].iloc[idx])
                _hyst_sb = self.config.get("regime_hysteresis_pct", 0.02)
                if _ema_sb > 0 and price < _ema_sb * (1 - _hyst_sb):
                    short_max_per_side = int(self.config["regime_short_cap"])
            long_at_cap  = len(self.long_positions)  >= max_per_side
            short_at_cap = len(self.short_positions) >= short_max_per_side
            open_position_count = len(self.long_positions) + len(self.short_positions)
            at_max = open_position_count >= (max_per_side + short_max_per_side)

            # Per-side unrealized loss circuit breaker.
            # Prevents piling into a trending move — if the open positions on
            # one side are already down more than the threshold, stop adding.
            max_unreal_loss = self.config.get("max_unrealized_loss_per_side", float("inf"))
            sl_multiplier = self.config.get("sl_multiplier", 2.0)
            min_sl_pct    = self.config.get("min_sl_pct", 0.0)  # floor: SL never closer than this % from entry
            use_sl        = self.config.get("use_sl", True)
            long_unreal  = sum((price - ep) * qty for ep, qty, _m, _t, _s in self.long_positions)
            short_unreal = sum((ep - price) * qty for ep, qty, _m, _t, _s in self.short_positions)
            long_loss_tripped  = long_unreal  < -abs(max_unreal_loss)
            short_loss_tripped = short_unreal < -abs(max_unreal_loss)

            if (long_at_cap or short_at_cap) and not self._max_positions_warned:
                print("\u26a0\ufe0f Per-side position cap reached (new entries paused, existing positions continue)")
                self._max_positions_warned = True
            if long_loss_tripped and not getattr(self, "_long_loss_warned", False):
                print(f"\u26a0\ufe0f Long-side unrealized loss circuit tripped "
                      f"(${long_unreal:+.2f} < -${abs(max_unreal_loss):.0f}) — no new longs")
                self._long_loss_warned = True
            if short_loss_tripped and not getattr(self, "_short_loss_warned", False):
                print(f"\u26a0\ufe0f Short-side unrealized loss circuit tripped "
                      f"(${short_unreal:+.2f} < -${abs(max_unreal_loss):.0f}) — no new shorts")
                self._short_loss_warned = True
            # ------------------------------------------------------------------
            # Trend detection + capture
            # When velocity > threshold:
            #   1. Force-close all positions on the losing side immediately.
            #   2. Open a single directional "trend position" sized at
            #      trend_capture_size_pct of current balance.
            #   3. Trail a stop (trend_trailing_stop_pct) behind the peak.
            #   4. Close the trend position + resume grid when trail fires or
            #      momentum fades.
            # ------------------------------------------------------------------
            trend_detection   = self.config.get("trend_detection", False)
            trend_capture     = self.config.get("trend_capture", False)
            force_close_grid  = self.config.get("trend_force_close_grid", True)
            trend_lookback    = self.config.get("trend_lookback_candles", 15)
            vel_threshold     = self.config.get("trend_velocity_pct", 0.01)
            cap_vel_threshold = self.config.get("trend_capture_velocity_pct", vel_threshold)
            cooldown_candles  = self.config.get("trend_cooldown_candles", 30)
            confirm_candles   = self.config.get("trend_confirm_candles", 1)
            cap_size_pct      = self.config.get("trend_capture_size_pct", 0.15)
            trail_stop_pct    = self.config.get("trend_trailing_stop_pct", 0.04)
            # ── ATR dynamic trail ────────────────────────────────────
            # Replace fixed trail_stop_pct with N × ATR / price each candle.
            # Adapts to volatility: wider in high-vol (lets trend breathe),
            # tighter in low-vol (locks in gains quickly).
            atr_trail         = self.config.get("atr_trail", False)
            atr_trail_mult    = self.config.get("atr_trail_multiplier", 2.0)
            atr_trail_min     = self.config.get("atr_trail_min", 0.015)  # floor 1.5%
            atr_trail_max     = self.config.get("atr_trail_max", 0.12)   # cap 12%
            if atr_trail and price > 0:
                atr_now = float(self.atr_series.iloc[idx])
                dyn_trail = atr_now * atr_trail_mult / price
                trail_stop_pct = max(atr_trail_min, min(atr_trail_max, dyn_trail))

            # ── Layer 1: ATR parabolic gate (fixed or adaptive) ───────────
            # When ATR exceeds a threshold, price is in a parabolic spike and
            # trend entries are suppressed.  The threshold can be:
            #   Fixed:    atr_parabolic_mult × SMA(ATR,20)    (v21 original)
            #   Approach A (atr_regime_adaptive): bull→relaxed mult, bear→tight
            #   Approach B (atr_percentile_gate): block only above rolling Nth %ile
            #   Approach C (atr_adx_scale): ADX strength modulates mult continuously
            #   Hybrid:   any combination — OR logic (any trigger blocks)
            _atr_now     = float(self.atr_series.iloc[idx])
            _atr_sma_now = float(self.atr_sma_series.iloc[idx])

            # ── v26 XRPPM16: Dynamic velocity threshold ──────────────────
            # Scale the STATIC vel_threshold proportionally to the ATR spike
            # ratio (ATR / ATR_SMA).  In normal vol (ratio ≈ 1.0), the floor
            # keeps the threshold unchanged.  In high-vol (ratio 3-4×),
            # threshold rises 3-4× so post-parabolic bounces are ignored.
            #   threshold = static_vel × max(1.0, vel_atr_mult × ATR/SMA)
            #
            # v27 XRPPM17 directional modifiers:
            #   vel_dir_only:   only scale when price < EMA-36 (bearish context)
            #   vel_accel_only: only scale when ATR is actively rising
            _vel_atr_mult = float(self.config.get("vel_atr_mult", 0.0))
            if _vel_atr_mult > 0 and _atr_sma_now > 0:
                _apply_vel_scaling = True

                # v27: Directional filter — only scale in bearish context
                if self.config.get("vel_dir_only", False):
                    _ema_dir_val = float(self.vel_dir_ema_series.iloc[idx])
                    if price >= _ema_dir_val:
                        _apply_vel_scaling = False  # bullish — leave static

                # v27: Acceleration filter — only scale when ATR is rising
                if _apply_vel_scaling and self.config.get("vel_accel_only", False):
                    _accel_lb = max(1, int(self.config.get("atr_accel_lookback", 10)))
                    _prev_idx = max(0, idx - _accel_lb)
                    _atr_prev = float(self.atr_series.iloc[_prev_idx])
                    if _atr_now <= _atr_prev:
                        _apply_vel_scaling = False  # ATR flat/falling — leave static

                if _apply_vel_scaling:
                    _atr_ratio_vel = _atr_now / _atr_sma_now
                    _vel_scale = max(1.0, _vel_atr_mult * _atr_ratio_vel)
                    vel_threshold     = vel_threshold * _vel_scale
                    cap_vel_threshold = cap_vel_threshold * _vel_scale

            # Start with the fixed multiplier (0 = disabled)
            _atr_parabolic_mult = float(self.config.get("atr_parabolic_mult", 0.0))

            # Approach A: Regime-adaptive — bull/bear regime switches multiplier
            if self.config.get("atr_regime_adaptive", False) and "regime_ema" in self.df.columns:
                _regime_ema_val = float(self.df["regime_ema"].iloc[idx])
                if _regime_ema_val > 0 and price >= _regime_ema_val:
                    _atr_parabolic_mult = float(self.config.get("atr_bull_mult", 2.5))
                else:
                    _atr_parabolic_mult = float(self.config.get("atr_bear_mult", 1.5))

            # Approach C: ADX-scaled — ADX strength modulates multiplier
            # High ADX (strong trend) → higher mult → more permissive.
            # Overrides fixed mult and regime-adaptive mult.
            if self.config.get("atr_adx_scale", False):
                _adx_val = float(self.adx_series.iloc[idx])
                _adx_base = float(self.config.get("atr_adx_base_mult", 1.5))
                _adx_max  = float(self.config.get("atr_adx_max_mult", 3.0))
                _atr_parabolic_mult = _adx_base + (_adx_val / 100.0) * (_adx_max - _adx_base)

            # SMA-mult gate: fires when ATR > effective_mult × SMA(ATR,20)
            _parabolic_sma = (_atr_parabolic_mult > 0 and _atr_sma_now > 0
                              and _atr_now > _atr_parabolic_mult * _atr_sma_now)

            # Approach B: Percentile gate — block when ATR > rolling Nth percentile
            _parabolic_pct = False
            if self.config.get("atr_percentile_gate", False):
                _atr_pct_val = float(self.atr_pct_series.iloc[idx])
                _parabolic_pct = (_atr_pct_val > 0 and _atr_now > _atr_pct_val)

            # Final gate: either SMA-mult OR percentile triggers blocking
            _parabolic = _parabolic_sma or _parabolic_pct

            # Approach D: Directional — only keep gate ON if price also falling
            # Grid strategies profit from buying dips, so suppress the gate when
            # the ATR spike is caused by price RISING (healthy pump).
            if _parabolic and self.config.get("atr_directional", False):
                _dir_lb = int(self.config.get("atr_dir_lookback", 8))
                _dir_drop = float(self.config.get("atr_dir_drop_pct", 0.02))
                if idx >= _dir_lb:
                    _prev_price = float(self.df["close"].iloc[idx - _dir_lb])
                    if _prev_price > 0:
                        _price_chg = (price - _prev_price) / _prev_price
                        # Gate only fires if price actually FELL by drop threshold
                        if _price_chg > -_dir_drop:
                            _parabolic = False  # price flat/rising → allow entries

            # Approach E: ATR acceleration — only keep gate ON if ATR is rising
            # After a crash ATR stays high but flattens → gate clears faster.
            if _parabolic and self.config.get("atr_acceleration", False):
                _accel_lb = int(self.config.get("atr_accel_lookback", 10))
                if idx >= _accel_lb:
                    _prev_atr = float(self.atr_series.iloc[idx - _accel_lb])
                    if _prev_atr > 0 and _atr_now <= _prev_atr:
                        _parabolic = False  # ATR flat/falling → allow entries

            # Approach F: Cooldown — limit consecutive blocked candles
            # After gate fires for N consecutive candles, force-resume entries.
            _atr_cooldown_max = int(self.config.get("atr_cooldown", 0))
            if _atr_cooldown_max > 0:
                if _parabolic:
                    self._gate_fire_counter = getattr(self, "_gate_fire_counter", 0) + 1
                    if self._gate_fire_counter > _atr_cooldown_max:
                        _parabolic = False  # cooldown expired → force resume
                else:
                    self._gate_fire_counter = 0  # reset when gate not firing

            # ── v26 XRPPM16: Dynamic gate decay ──────────────────────────
            # After the parabolic gate fires and then clears, maintain
            # suppression for spike_ratio × scale additional candles.
            # spike_ratio = peak_atr / atr_sma during the gate period.
            # This catches the post-parabolic correction window (Dec 2024)
            # where ATR is still 3-4× normal but no longer RISING.
            #
            # State machine:  IDLE → GATE_ACTIVE → DECAYING → IDLE
            # Only track peak during real gate fire, NOT during decay.
            _gate_decay_scale = float(self.config.get("gate_decay_scale", 0.0))
            if _gate_decay_scale > 0:
                # Ensure state variables exist on first candle
                if not hasattr(self, "_gate_peak_atr_ratio"):
                    self._gate_peak_atr_ratio = 1.0
                if not hasattr(self, "_gate_decay_countdown"):
                    self._gate_decay_countdown = 0
                if not hasattr(self, "_gate_was_active"):
                    self._gate_was_active = False

                _decay_countdown = self._gate_decay_countdown

                if _parabolic and _decay_countdown == 0:
                    # GATE_ACTIVE: real gate fire → track peak ratio
                    _cur_spike = _atr_now / _atr_sma_now if _atr_sma_now > 0 else 1.0
                    self._gate_peak_atr_ratio = max(
                        self._gate_peak_atr_ratio, _cur_spike
                    )
                    self._gate_was_active = True
                elif not _parabolic and self._gate_was_active and _decay_countdown == 0:
                    # Gate just cleared → start decay countdown from peak
                    _peak = self._gate_peak_atr_ratio
                    if _peak > 1.0:
                        self._gate_decay_countdown = int(_peak * _gate_decay_scale)
                    self._gate_was_active = False

                if self._gate_decay_countdown > 0:
                    _parabolic = True  # suppress entries during decay
                    self._gate_decay_countdown -= 1
                    if self._gate_decay_countdown <= 0:
                        # Decay expired → fully reset
                        self._gate_peak_atr_ratio = 1.0

            # ── Layer 2: HTF EMA alignment gate ────────────────────────────
            # Before opening a LONG trend entry: require HTF EMA-36 > EMA-84.
            # Before opening a SHORT trend entry: require HTF EMA-36 < EMA-84.
            # This stops velocity signals fired into a counter-trend HTF bias.
            _htf_ema_align = bool(self.config.get("htf_ema_align", False))
            _htf_fast      = float(self.htf_ema_fast_series.iloc[idx])
            _htf_slow      = float(self.htf_ema_slow_series.iloc[idx])
            _htf_bull      = _htf_fast > _htf_slow
            _htf_bear      = _htf_fast < _htf_slow

            # ── Layer 4: Grid sleep (low-ATR gate) ──────────────────────────
            # When ATR/price < grid_sleep_atr_thresh, market is too flat for
            # the grid to complete round trips — pause ALL new grid entries.
            _grid_sleep_thresh = float(self.config.get("grid_sleep_atr_thresh", 0.0))
            _grid_sleep = (_grid_sleep_thresh > 0 and price > 0
                           and (_atr_now / price) < _grid_sleep_thresh)

            # ── ADX filter ─────────────────────────────────────────────
            # adx_filter=True gates trend-capture entries on ADX strength.
            # adx_grid_pause (optional): pause new grid orders entirely when
            # ADX > threshold — prevents piling in during strong trends.
            adx_filter      = self.config.get("adx_filter", False)
            adx_min_trend   = self.config.get("adx_min_trend", 25.0)
            adx_grid_pause  = self.config.get("adx_grid_pause", None)
            if adx_filter:
                current_adx = float(self.adx_series.iloc[idx])
            else:
                current_adx = 0.0
            adx_allows_trend = (not adx_filter) or (current_adx >= adx_min_trend)
            adx_pauses_grid  = (adx_filter and adx_grid_pause is not None
                                and current_adx >= adx_grid_pause)
            # ── ADX-adaptive trailing stop ───────────────────────────────────
            # When ADX is very strong (≥ adx_wide_trail_threshold), widen the
            # trailing stop so the position can breathe through a minor pullback
            # and keep riding the larger trend move.
            adx_wide_trail_threshold = self.config.get("adx_wide_trail_threshold", 9999.0)
            if adx_filter and current_adx >= adx_wide_trail_threshold:
                trail_stop_pct = self.config.get("adx_wide_trail_pct", trail_stop_pct)
            # ── Fast re-entry flag ───────────────────────────────────────────
            # When True: immediately reset trend_mode on position close so the
            # confirmation counter can re-fire and capture the next leg without
            # waiting for the full 30-candle cooldown.
            trend_reentry_fast = self.config.get("trend_reentry_fast", False)
            # ── EMA bias filter ─────────────────────────────────────────────
            # ema_bias_filter=True: long capture only when price >= slow EMA,
            # short capture only when price <= slow EMA.  Avoids counter-trend trades.
            ema_bias       = self.config.get("ema_bias_filter", False)
            ema_slow_val   = float(self.ema_slow_series.iloc[idx]) if ema_bias else price
            ema_bias_long  = (not ema_bias) or (price >= ema_slow_val)
            ema_bias_short = (not ema_bias) or (price <= ema_slow_val)
            # ── BB squeeze filter ───────────────────────────────────────────
            # bb_squeeze_gate=True: skip trend entries while bands are tight.
            # bb_squeeze_boost=True: increase position size on the first breakout
            # candle that exits a squeeze (expanding from below threshold).
            bb_squeeze_gate  = self.config.get("bb_squeeze_gate", False)
            bb_squeeze_boost = self.config.get("bb_squeeze_boost", False)
            bb_sq_threshold  = self.config.get("bb_squeeze_threshold", 0.035)
            bb_boost_mult    = self.config.get("bb_squeeze_boost_mult", 1.5)
            if bb_squeeze_gate or bb_squeeze_boost:
                current_bb_w = float(self.bb_width_series.iloc[idx])
                prev_bb_w    = float(self.bb_width_series.iloc[max(0, idx - 1)])
            else:
                current_bb_w = 1.0
                prev_bb_w    = 1.0
            in_squeeze      = current_bb_w < bb_sq_threshold
            expanding       = current_bb_w > prev_bb_w
            just_broke      = (not in_squeeze) and expanding and (prev_bb_w < bb_sq_threshold)
            bb_allows_trend = (not bb_squeeze_gate) or (not in_squeeze)
            bb_size_boost   = bb_boost_mult if (bb_squeeze_boost and just_broke) else 1.0
            # ── RSI filter ──────────────────────────────────────────────────
            # rsi_filter=True: block long entries when overbought (RSI > ob),
            # block short entries when oversold (RSI < os).
            # rsi_momentum=True: additionally require RSI > 50 for longs, < 50 for shorts.
            rsi_filter     = self.config.get("rsi_filter", False)
            rsi_overbought = self.config.get("rsi_overbought", 70.0)
            rsi_oversold   = self.config.get("rsi_oversold", 30.0)
            rsi_momentum   = self.config.get("rsi_momentum", False)
            current_rsi    = float(self.rsi_series.iloc[idx]) if rsi_filter else 50.0
            rsi_allows_long  = (
                ((not rsi_filter) or (current_rsi < rsi_overbought)) and
                ((not rsi_momentum) or (current_rsi > 50.0))
            )
            rsi_allows_short = (
                ((not rsi_filter) or (current_rsi > rsi_oversold)) and
                ((not rsi_momentum) or (current_rsi < 50.0))
            )
            # ── RSI exhaustion trail tightener (position management v10) ────
            # Shrinks trail_stop_pct when the existing trend position is in
            # over-extended territory (RSI ≥ ob for longs, ≤ os for shorts).
            # Locks in more profit during blow-off tops / capitulation lows.
            rsi_tight_trail     = self.config.get("rsi_tight_trail", False)
            rsi_tight_trail_ob  = self.config.get("rsi_tight_trail_ob", 80.0)
            rsi_tight_trail_os  = self.config.get("rsi_tight_trail_os", 20.0)
            rsi_tight_trail_pct = self.config.get("rsi_tight_trail_pct", 0.02)
            if rsi_tight_trail and self.trend_position is not None:
                rsi_pm = float(self.rsi_series.iloc[idx])
                if (self.trend_position["side"] == "long"  and rsi_pm >= rsi_tight_trail_ob) or \
                   (self.trend_position["side"] == "short" and rsi_pm <= rsi_tight_trail_os):
                    trail_stop_pct = rsi_tight_trail_pct
            # ── Volume confirmation filter ──────────────────────────────────
            # vol_filter=True: only fire trend-capture when the current candle's
            # volume exceeds vol_multiplier × the rolling average volume.
            # Filters out low-conviction false breakouts during thin markets.
            vol_filter      = self.config.get("vol_filter", False)
            vol_mult        = self.config.get("vol_multiplier", 1.5)
            if vol_filter and "volume" in self.df.columns:
                vol_now      = float(self.df["volume"].iloc[idx])
                vol_avg_now  = float(self.vol_avg_series.iloc[idx])
                vol_confirms = vol_now >= vol_mult * vol_avg_now
            else:
                vol_confirms = True
            # ── Volume-scaled re-entry size (position management v10) ───────
            # When vol_reentry_scale=True, size each trend capture at full
            # cap_size_pct only when volume confirms conviction; otherwise
            # uses vol_reentry_low_pct to limit exposure on weak breakouts.
            vol_reentry_scale     = self.config.get("vol_reentry_scale", False)
            vol_reentry_high_mult = self.config.get("vol_reentry_high_mult", 1.5)
            vol_reentry_low_pct   = self.config.get("vol_reentry_low_pct", 0.45)
            if vol_reentry_scale and "volume" in self.df.columns:
                vol_now_pm  = float(self.df["volume"].iloc[idx])
                vol_avg_pm  = float(self.vol_avg_series.iloc[idx])
                used_cap_size_pct = (
                    cap_size_pct if vol_now_pm >= vol_reentry_high_mult * vol_avg_pm
                    else vol_reentry_low_pct
                )
            else:
                used_cap_size_pct = cap_size_pct

            # ── v26 XRPPM16: Dynamic position sizing ────────────────────
            # Scale position size inversely with ATR spike ratio.
            # Normal vol (ratio≈1): full size.  3× ATR: ~30% size.
            # Reduces exposure during post-parabolic volatility (Dec 2024).
            if self.config.get("cap_size_atr_scale", False) and _atr_sma_now > 0:
                _atr_spike_ratio = _atr_now / _atr_sma_now
                _size_floor   = float(self.config.get("cap_size_atr_floor", 0.30))
                _size_ceiling = float(self.config.get("cap_size_atr_ceiling", 0.90))
                if _atr_spike_ratio > 1.0:
                    _scaled_size = used_cap_size_pct / _atr_spike_ratio
                    used_cap_size_pct = max(_size_floor, min(_size_ceiling, _scaled_size))

            # ── Market structure filter ─────────────────────────────────────
            # ms_filter=True: only fire trend-capture when the current close
            # breaks above the recent swing high (bullish HH — allow long) or
            # below the recent swing low (bearish LL — allow short).
            # Prevents entering a trend capture in the middle of a range.
            ms_filter = self.config.get("ms_filter", False)
            if ms_filter:
                swing_high_prev  = float(self.swing_high_series.iloc[idx])
                swing_low_prev   = float(self.swing_low_series.iloc[idx])
                ms_allows_long   = price > swing_high_prev   # HH breakout
                ms_allows_short  = price < swing_low_prev    # LL breakdown
            else:
                ms_allows_long   = True
                ms_allows_short  = True
            long_blocked_by_trend  = False
            short_blocked_by_trend = False

            if trend_detection and len(self.equity_curve) >= trend_lookback:
                past_price = self.equity_curve[-trend_lookback][1]
                velocity   = (price - past_price) / past_price
                trending_up   = velocity >  vel_threshold
                trending_down = velocity < -vel_threshold

                # ── Manage existing trend position (trailing stop) ──────────
                if self.trend_position is not None:
                    tp = self.trend_position
                    # ── v26 XRPPM16: Dynamic max loss per trade ──────────
                    # Close immediately if unrealised loss > N × ATR × qty.
                    # ATR-scaled so the stop widens in normal vol and
                    # tightens relative to position size in high vol.
                    _max_loss_atr = float(self.config.get("trend_max_loss_atr", 0.0))
                    _max_loss_hit = False
                    if _max_loss_atr > 0:
                        if tp["side"] == "long":
                            _unrealised = (price - tp["entry"]) * tp["qty"]
                        else:
                            _unrealised = (tp["entry"] - price) * tp["qty"]
                        _loss_cap = _max_loss_atr * _atr_now * tp["qty"]
                        if _unrealised < 0 and abs(_unrealised) >= _loss_cap:
                            _max_loss_hit = True
                    if tp["side"] == "long":
                        if price > tp["peak"]:
                            tp["peak"] = price
                        trail_stop = tp["peak"] * (1 - trail_stop_pct)
                        # Close if trail hit OR trend fully reversed OR max loss
                        close_trend = price <= trail_stop or trending_down or _max_loss_hit
                        if close_trend:
                            fee_cost = tp["qty"] * price * (self.fee / 2)
                            gross_pnl = (price - tp["entry"]) * tp["qty"]
                            net_pnl = gross_pnl - fee_cost
                            self.balance += tp["margin"] + net_pnl
                            self.trade_history.append((
                                timestamp, "TREND_SELL", price, tp["qty"], "TREND_LONG",
                                net_pnl, fee_cost, gross_pnl,
                                self._calculate_unrealized_pnl(price), self._equity(price),
                            ))
                            print(f"\U0001f3af [{timestamp}] Trend LONG closed "
                                  f"entry={tp['entry']:.4f} exit={price:.4f} pnl={net_pnl:+.2f}")
                            # v27: consecutive loss tracking
                            if net_pnl < 0:
                                self._consec_trend_losses += 1
                                _clm = int(self.config.get("consec_loss_max", 0))
                                if _clm > 0 and self._consec_trend_losses >= _clm:
                                    self._consec_loss_pause_counter = int(
                                        self.config.get("consec_loss_pause", 20))
                            else:
                                self._consec_trend_losses = 0
                            self.trend_position = None
                            if trend_reentry_fast:
                                # Reset mode so the confirmation counter can re-fire immediately
                                self.trend_mode = None
                                self.trend_confirm_counter = 0
                                self.trend_pending_dir = None
                    else:  # short
                        if price < tp["peak"]:
                            tp["peak"] = price
                        trail_stop = tp["peak"] * (1 + trail_stop_pct)
                        close_trend = price >= trail_stop or trending_up or _max_loss_hit
                        if close_trend:
                            fee_cost = tp["qty"] * price * (self.fee / 2)
                            gross_pnl = (tp["entry"] - price) * tp["qty"]
                            net_pnl = gross_pnl - fee_cost
                            self.balance += tp["margin"] + net_pnl
                            self.trade_history.append((
                                timestamp, "TREND_BUY", price, tp["qty"], "TREND_SHORT",
                                net_pnl, fee_cost, gross_pnl,
                                self._calculate_unrealized_pnl(price), self._equity(price),
                            ))
                            print(f"\U0001f3af [{timestamp}] Trend SHORT closed "
                                  f"entry={tp['entry']:.4f} exit={price:.4f} pnl={net_pnl:+.2f}")
                            # v27: consecutive loss tracking
                            if net_pnl < 0:
                                self._consec_trend_losses += 1
                                _clm = int(self.config.get("consec_loss_max", 0))
                                if _clm > 0 and self._consec_trend_losses >= _clm:
                                    self._consec_loss_pause_counter = int(
                                        self.config.get("consec_loss_pause", 20))
                            else:
                                self._consec_trend_losses = 0
                            self.trend_position = None
                            if trend_reentry_fast:
                                self.trend_mode = None
                                self.trend_confirm_counter = 0
                                self.trend_pending_dir = None

                # ── Confirmation counter ────────────────────────────────────
                # Require velocity to remain above threshold for confirm_candles
                # consecutive candles in the same direction before acting.
                if trending_up:
                    if self.trend_pending_dir == "up":
                        self.trend_confirm_counter += 1
                    else:
                        self.trend_pending_dir = "up"
                        self.trend_confirm_counter = 1
                elif trending_down:
                    if self.trend_pending_dir == "down":
                        self.trend_confirm_counter += 1
                    else:
                        self.trend_pending_dir = "down"
                        self.trend_confirm_counter = 1
                else:
                    self.trend_pending_dir = None
                    self.trend_confirm_counter = 0

                confirmed_up   = (self.trend_pending_dir == "up"   and
                                  self.trend_confirm_counter >= confirm_candles)
                confirmed_down = (self.trend_pending_dir == "down" and
                                  self.trend_confirm_counter >= confirm_candles)

                # ── v27 XRPPM17: Equity curve filter ───────────────────────
                # Suppress trend entries when equity is below its SMA.
                # Self-calibrating: based on the bot's own P&L trajectory.
                _eq_curve_blocked = False
                if self.config.get("eq_curve_filter", False):
                    _eq_lb = int(self.config.get("eq_curve_lookback", 50))
                    if len(self.equity_curve) >= _eq_lb:
                        _recent_eq = [e[2] for e in self.equity_curve[-_eq_lb:]]
                        _eq_sma = sum(_recent_eq) / len(_recent_eq)
                        _current_eq = self._equity(price)
                        if _current_eq < _eq_sma:
                            _eq_curve_blocked = True

                # ── v27 XRPPM17: Consecutive loss guard ────────────────────
                # After N consecutive losing trend trades, pause entries for
                # M candles — pure outcome-based risk management.
                _consec_blocked = False
                if self._consec_loss_pause_counter > 0:
                    self._consec_loss_pause_counter -= 1
                    _consec_blocked = True

                # ── Act on confirmed trend ──────────────────────────────────
                if confirmed_up and self.trend_mode != "up":
                    self.trend_mode = "up"
                    self.trend_cooldown_counter = 0
                    self.last_short_price = price
                    if force_close_grid and self.short_positions:
                        n = len(self.short_positions)
                        for ep, qty, margin_req, _tp, _sl in self.short_positions:
                            fee_cost = qty * price * (self.fee / 2)
                            gross_pnl = (ep - price) * qty
                            net_pnl = gross_pnl - fee_cost
                            self.balance += margin_req + net_pnl
                            self.trade_history.append((
                                timestamp, "TREND_FORCE_CLOSE", price, qty, "SHORT",
                                net_pnl, fee_cost, gross_pnl, 0.0, self._equity(price),
                            ))
                        self.short_positions.clear()
                        print(f"\U0001f4c8 [{timestamp}] Trend UP confirmed "
                              f"(vel={velocity*100:.2f}% x{self.trend_confirm_counter}) "
                              f"— force-closed {n} short(s)")
                    if (trend_capture and self.trend_position is None and not halt_all
                            and adx_allows_trend and ema_bias_long and bb_allows_trend
                            and rsi_allows_long and vol_confirms and ms_allows_long
                            and not _parabolic                            # Layer 1
                            and (not _htf_ema_align or _htf_bull)         # Layer 2
                            and not _eq_curve_blocked                     # v27 equity curve
                            and not _consec_blocked):                     # v27 consecutive loss
                        if velocity >= cap_vel_threshold:
                            current_equity = self._equity(price)
                            cap_margin = current_equity * used_cap_size_pct * bb_size_boost
                            # Respect the s100 hard cap (can't deploy more than 90% balance)
                            cap_margin = min(cap_margin, current_equity * 0.90)
                            cap_qty = (cap_margin * self.leverage) / price
                            fee_cost = cap_qty * price * (self.fee / 2)
                            if cap_margin + fee_cost <= self.balance:
                                self.balance -= cap_margin + fee_cost
                                self.trend_position = {
                                    "side": "long", "entry": price, "qty": cap_qty,
                                    "margin": cap_margin, "peak": price,
                                }
                                self.trade_history.append((
                                    timestamp, "TREND_BUY", price, cap_qty, "TREND_LONG",
                                    0.0, fee_cost, 0.0,
                                    self._calculate_unrealized_pnl(price), self._equity(price),
                                ))
                                adx_info = f" ADX={current_adx:.1f}" if adx_filter else ""
                                print(f"\U0001f3af [{timestamp}] Trend LONG opened "
                                      f"at {price:.4f} size=${cap_margin:.0f}{adx_info}")

                elif confirmed_down and self.trend_mode != "down":
                    self.trend_mode = "down"
                    self.trend_cooldown_counter = 0
                    self.last_long_price = price
                    if force_close_grid and self.long_positions:
                        n = len(self.long_positions)
                        for ep, qty, margin_req, _tp, _sl in self.long_positions:
                            fee_cost = qty * price * (self.fee / 2)
                            gross_pnl = (price - ep) * qty
                            net_pnl = gross_pnl - fee_cost
                            self.balance += margin_req + net_pnl
                            self.trade_history.append((
                                timestamp, "TREND_FORCE_CLOSE", price, qty, "LONG",
                                net_pnl, fee_cost, gross_pnl, 0.0, self._equity(price),
                            ))
                        self.long_positions.clear()
                        print(f"\U0001f4c9 [{timestamp}] Trend DOWN confirmed "
                              f"(vel={velocity*100:.2f}% x{self.trend_confirm_counter}) "
                              f"— force-closed {n} long(s)")
                    if (trend_capture and self.trend_position is None and not halt_all
                            and adx_allows_trend and ema_bias_short and bb_allows_trend
                            and rsi_allows_short and vol_confirms and ms_allows_short
                            and not _parabolic                            # Layer 1
                            and (not _htf_ema_align or _htf_bear)         # Layer 2
                            and not _eq_curve_blocked                     # v27 equity curve
                            and not _consec_blocked):                     # v27 consecutive loss
                        if velocity <= -cap_vel_threshold:
                            current_equity = self._equity(price)
                            cap_margin = current_equity * used_cap_size_pct * bb_size_boost
                            cap_margin = min(cap_margin, current_equity * 0.90)
                            cap_qty = (cap_margin * self.leverage) / price
                            fee_cost = cap_qty * price * (self.fee / 2)
                            if cap_margin + fee_cost <= self.balance:
                                self.balance -= cap_margin + fee_cost
                                self.trend_position = {
                                    "side": "short", "entry": price, "qty": cap_qty,
                                    "margin": cap_margin, "peak": price,
                                }
                                self.trade_history.append((
                                    timestamp, "TREND_SELL", price, cap_qty, "TREND_SHORT",
                                    0.0, fee_cost, 0.0,
                                    self._calculate_unrealized_pnl(price), self._equity(price),
                                ))
                                adx_info = f" ADX={current_adx:.1f}" if adx_filter else ""
                                print(f"\U0001f3af [{timestamp}] Trend SHORT opened "
                                      f"at {price:.4f} size=${cap_margin:.0f}{adx_info}")

                elif self.trend_mode is not None and self.trend_position is None:
                    # ── v26: Retry capture when mode is set but position not yet opened.
                    # With dynamic velocity the detection threshold (vel_threshold)
                    # can be lower than the capture threshold (cap_vel_threshold),
                    # so the mode fires before velocity is strong enough for a position.
                    # Retry until velocity either meets cap threshold or fades.
                    _retry_up   = (self.trend_mode == "up"   and trend_capture
                                   and velocity >= cap_vel_threshold
                                   and not halt_all and adx_allows_trend
                                   and ema_bias_long and bb_allows_trend
                                   and rsi_allows_long and vol_confirms
                                   and ms_allows_long
                                   and not _parabolic
                                   and (not _htf_ema_align or _htf_bull)
                                   and not _eq_curve_blocked
                                   and not _consec_blocked)
                    _retry_down = (self.trend_mode == "down" and trend_capture
                                   and abs(velocity) >= cap_vel_threshold
                                   and not halt_all and adx_allows_trend
                                   and ema_bias_short and bb_allows_trend
                                   and rsi_allows_short and vol_confirms
                                   and ms_allows_short
                                   and not _parabolic
                                   and (not _htf_ema_align or _htf_bear)
                                   and not _eq_curve_blocked
                                   and not _consec_blocked)
                    if _retry_up:
                        current_equity = self._equity(price)
                        cap_margin = current_equity * used_cap_size_pct * bb_size_boost
                        cap_margin = min(cap_margin, current_equity * 0.90)
                        cap_qty = (cap_margin * self.leverage) / price
                        fee_cost = cap_qty * price * (self.fee / 2)
                        if cap_margin + fee_cost <= self.balance:
                            self.balance -= cap_margin + fee_cost
                            self.trend_position = {
                                "side": "long", "entry": price, "qty": cap_qty,
                                "margin": cap_margin, "peak": price,
                            }
                            self.trade_history.append((
                                timestamp, "TREND_BUY", price, cap_qty, "TREND_LONG",
                                0.0, fee_cost, 0.0,
                                self._calculate_unrealized_pnl(price), self._equity(price),
                            ))
                            adx_info = f" ADX={current_adx:.1f}" if adx_filter else ""
                            print(f"\U0001f3af [{timestamp}] Trend LONG opened (retry) "
                                  f"at {price:.4f} size=${cap_margin:.0f}{adx_info}")
                    elif _retry_down:
                        current_equity = self._equity(price)
                        cap_margin = current_equity * used_cap_size_pct * bb_size_boost
                        cap_margin = min(cap_margin, current_equity * 0.90)
                        cap_qty = (cap_margin * self.leverage) / price
                        fee_cost = cap_qty * price * (self.fee / 2)
                        if cap_margin + fee_cost <= self.balance:
                            self.balance -= cap_margin + fee_cost
                            self.trend_position = {
                                "side": "short", "entry": price, "qty": cap_qty,
                                "margin": cap_margin, "peak": price,
                            }
                            self.trade_history.append((
                                timestamp, "TREND_SELL", price, cap_qty, "TREND_SHORT",
                                0.0, fee_cost, 0.0,
                                self._calculate_unrealized_pnl(price), self._equity(price),
                            ))
                            adx_info = f" ADX={current_adx:.1f}" if adx_filter else ""
                            print(f"\U0001f3af [{timestamp}] Trend SHORT opened (retry) "
                                  f"at {price:.4f} size=${cap_margin:.0f}{adx_info}")
                    elif abs(velocity) < vel_threshold * 0.5:
                        self.trend_cooldown_counter += 1
                        if self.trend_cooldown_counter >= cooldown_candles:
                            print(f"\U0001f504 [{timestamp}] Trend {self.trend_mode.upper()} ended "
                                  f"— resuming hedge mode")
                            self.trend_mode = None
                            self.trend_cooldown_counter = 0
                            self.last_long_price  = price
                            self.last_short_price = price
                    else:
                        self.trend_cooldown_counter = 0

            # Gate new grid entries when a trend is active or ADX signals strong directional move
            long_blocked_by_trend  = (self.trend_mode == "down") or adx_pauses_grid
            short_blocked_by_trend = (self.trend_mode == "up")  or adx_pauses_grid

            # ------------------------------------------------------------------
            # v34 PM24: 5-regime rotation override
            # Overrides binary gate variables based on current market regime.
            # Regimes: 0=CHOPPY (aggressive grid), 1=VOLATILE (default),
            #          2=TRENDING (grid off), 3=CRASH (bounce grid),
            #          4=DORMANT (micro-DCA).
            # ------------------------------------------------------------------
            _regime_spacing_mult = 1.0
            _regime_qty_mult     = 1.0
            if self.regime_series is not None:
                _regime = int(self.regime_series.iloc[idx])

                if _regime == 0:    # CHOPPY — bypass gates, grid-only
                    halt_grid_longs = False
                    long_blocked_by_trend  = (self.trend_mode == "down")
                    short_blocked_by_trend = (self.trend_mode == "up")
                    _grid_sleep = False

                elif _regime == 2:  # TRENDING — grid off, trend capture only
                    if self.config.get("regime_trending_grid_off", True):
                        _grid_sleep = True

                elif _regime == 3:  # CRASH — bounce hunting, reduced qty
                    # Only unblock longs if velocity CB is NOT active.
                    # The crash CB is a safety halt that must not be overridden.
                    if self.crash_halt_counter == 0:
                        halt_grid_longs = False
                    long_blocked_by_trend  = (self.trend_mode == "down")
                    short_blocked_by_trend = (self.trend_mode == "up")
                    _grid_sleep = False
                    _regime_qty_mult     = float(self.config.get(
                        "regime_crash_qty_mult", 0.5))
                    _regime_spacing_mult = float(self.config.get(
                        "regime_crash_spacing_mult", 0.5))

                elif _regime == 4:  # DORMANT — micro-DCA, long-only, tiny qty
                    halt_grid_longs = False
                    long_blocked_by_trend  = False
                    short_blocked_by_trend = True   # no shorts in dormant
                    _grid_sleep = False
                    _regime_qty_mult     = float(self.config.get(
                        "regime_dormant_qty_mult", 0.2))
                    _regime_spacing_mult = float(self.config.get(
                        "regime_dormant_spacing_mult", 2.0))

                # _regime == 1 (VOLATILE): no overrides — default gate behavior

            # Apply regime qty multiplier to effective order value
            effective_order_value *= _regime_qty_mult

            used_margin = sum(pos[2] for pos in self.long_positions + self.short_positions)
            available_margin = self.balance - used_margin

            # LONG SIDE — each open position carries its own pinned TP level
            if self.direction in ["long", "both"]:
                # 1. Close first position whose TP is hit (for...else: else only
                #    runs when no break fired, i.e. no TP hit this candle)
                for i, (ep, qty, margin_required, tp, sl) in enumerate(self.long_positions):
                    if price >= tp:
                        self.long_positions.pop(i)
                        fee_cost = qty * price * (self.fee / 2)
                        gross_pnl = (price - ep) * qty
                        net_pnl = gross_pnl - fee_cost
                        self.balance += margin_required + net_pnl
                        unrealized_pnl = self._calculate_unrealized_pnl(price)
                        self.trade_history.append((
                            timestamp, "SELL", price, qty, "LONG",
                            net_pnl, fee_cost, gross_pnl, unrealized_pnl,
                            self._equity(price),
                        ))
                        self.last_long_price = price  # next re-entry anchors below TP
                        break
                    elif price <= sl:
                        self.long_positions.pop(i)
                        fee_cost = qty * price * (self.fee / 2)
                        gross_pnl = (price - ep) * qty  # negative: stopped out below entry
                        net_pnl = gross_pnl - fee_cost
                        self.balance += margin_required + net_pnl
                        unrealized_pnl = self._calculate_unrealized_pnl(price)
                        self.trade_history.append((
                            timestamp, "SELL_STOP", price, qty, "LONG",
                            net_pnl, fee_cost, gross_pnl, unrealized_pnl,
                            self._equity(price),
                        ))
                        self.last_long_price = price  # re-anchor after stop
                        break
                else:
                    # 2. Open new entry only if no close happened this candle
                    _gnc_pct = self.config.get("grid_notional_cap_pct", None)
                    long_notional_capped = (
                        _gnc_pct is not None and
                        sum(m for _, _, m, _, _ in self.long_positions)
                        >= _gnc_pct * self._equity(price)
                    )
                    if (not long_at_cap and not long_loss_tripped
                            and not long_blocked_by_trend and not long_notional_capped
                            and not halt_grid_longs and not halt_all
                            and not _grid_sleep):       # Layer 4
                        # v18 — effective long spacing: vol-adaptive or regime-adaptive
                        _base_long_sp = self.long_settings["down_spacing"]
                        _eff_long_sp  = _base_long_sp
                        if self.config.get("vol_adaptive_spacing", False):
                            # Scale spacing by (current ATR/price) / (EMA of ATR/price).
                            # Calm market → tighter grid (captures small oscillations).
                            # Volatile/trending → wider grid (avoids SL cascade).
                            _atr_frac_now = (float(self.atr_series.iloc[idx]) / price) if price > 0 else 0.0
                            _atr_frac_ema = float(self.atr_frac_ema_series.iloc[idx])
                            if _atr_frac_ema > 0:
                                _vas_floor = self.config.get("vas_floor", 0.008)
                                _vas_ceil  = self.config.get("vas_ceil",  0.020)
                                _eff_long_sp = max(_vas_floor, min(_vas_ceil,
                                    _base_long_sp * (_atr_frac_now / _atr_frac_ema)))
                        elif (self.config.get("bull_spacing", 0.0) > 0
                                and self.config.get("bear_spacing", 0.0) > 0
                                and self.config.get("regime_filter")
                                and "regime_ema" in self.df.columns):
                            # Two-speed grid: tighter in bull regime, wider in bear.
                            _ema_btbw  = float(self.df["regime_ema"].iloc[idx])
                            _hyst_btbw = self.config.get("regime_hysteresis_pct", 0.02)
                            if _ema_btbw > 0:
                                _eff_long_sp = (self.config["bull_spacing"]
                                    if price >= _ema_btbw * (1 - _hyst_btbw)
                                    else self.config["bear_spacing"])
                        _eff_long_sp *= _regime_spacing_mult  # v34 regime rotation
                        buy_price = self.last_long_price * (1 - _eff_long_sp)
                        if price <= buy_price:
                            qty = effective_order_value / price
                            margin_required = qty * price / self.leverage
                            fee_cost = qty * price * (self.fee / 2)
                            if (margin_required + fee_cost) <= available_margin:
                                tp_price = price * (1 + _eff_long_sp)
                                if use_sl:
                                    long_sl_pct = max(sl_multiplier * _eff_long_sp, min_sl_pct)
                                    sl_price = price * (1 - long_sl_pct)
                                else:
                                    sl_price = 0.0  # sentinel: never triggered (price > 0 always)
                                self.balance -= margin_required + fee_cost
                                self.long_positions.append((price, qty, margin_required, tp_price, sl_price))
                                unrealized_pnl = self._calculate_unrealized_pnl(price)
                                self.trade_history.append((
                                    timestamp, "BUY", price, qty, "LONG",
                                    0.0, fee_cost, 0.0, unrealized_pnl,
                                    self._equity(price),
                                ))
                                self.last_long_price = price  # cascade next entry below

            # SHORT SIDE — each open position carries its own pinned TP level
            if self.direction in ["short", "both"]:
                for i, (ep, qty, margin_required, tp, sl) in enumerate(self.short_positions):
                    if price <= tp:
                        self.short_positions.pop(i)
                        fee_cost = qty * price * (self.fee / 2)
                        gross_pnl = (ep - price) * qty
                        net_pnl = gross_pnl - fee_cost
                        self.balance += margin_required + net_pnl
                        unrealized_pnl = self._calculate_unrealized_pnl(price)
                        self.trade_history.append((
                            timestamp, "COVER_SHORT", price, qty, "SHORT",
                            net_pnl, fee_cost, gross_pnl, unrealized_pnl,
                            self._equity(price),
                        ))
                        self.last_short_price = price  # next re-entry anchors above TP
                        break
                    elif price >= sl:
                        self.short_positions.pop(i)
                        fee_cost = qty * price * (self.fee / 2)
                        gross_pnl = (ep - price) * qty  # negative: stopped out above entry
                        net_pnl = gross_pnl - fee_cost
                        self.balance += margin_required + net_pnl
                        unrealized_pnl = self._calculate_unrealized_pnl(price)
                        self.trade_history.append((
                            timestamp, "COVER_SHORT_STOP", price, qty, "SHORT",
                            net_pnl, fee_cost, gross_pnl, unrealized_pnl,
                            self._equity(price),
                        ))
                        self.last_short_price = price  # re-anchor after stop
                        break
                else:
                    if (not short_at_cap and not short_loss_tripped
                            and not short_blocked_by_trend and not halt_all
                            and not halt_grid_shorts
                            and not _grid_sleep):       # Layer 4
                        # v17 asymmetric short spacing: below regime EMA use tighter
                        # spacing → faster SL cycling, less unrealized-loss accumulation,
                        # more fills during the downtrend regime.
                        _rss = self.config.get("regime_short_spacing", 0.0)
                        if (_rss > 0 and self.config.get("regime_filter")
                                and "regime_ema" in self.df.columns):
                            _ema_rss  = float(self.df["regime_ema"].iloc[idx])
                            _hyst_rss = self.config.get("regime_hysteresis_pct", 0.02)
                            _eff_short_sp = (_rss
                                if _ema_rss > 0 and price < _ema_rss * (1 - _hyst_rss)
                                else self.short_settings["up_spacing"])
                        else:
                            _eff_short_sp = self.short_settings["up_spacing"]

                        _eff_short_sp *= _regime_spacing_mult  # v34 regime rotation
                        sell_price = self.last_short_price * (1 + _eff_short_sp)
                        if price >= sell_price:
                            qty = effective_order_value / price
                            margin_required = qty * price / self.leverage
                            fee_cost = qty * price * (self.fee / 2)
                            if (margin_required + fee_cost) <= available_margin:
                                tp_price = price * (1 - _eff_short_sp)
                                if use_sl:
                                    short_sl_pct = max(sl_multiplier * _eff_short_sp, min_sl_pct)
                                    sl_price = price * (1 + short_sl_pct)
                                else:
                                    sl_price = float("inf")  # sentinel: never triggered
                                self.balance -= margin_required + fee_cost
                                self.short_positions.append((price, qty, margin_required, tp_price, sl_price))
                                unrealized_pnl = self._calculate_unrealized_pnl(price)
                                self.trade_history.append((
                                    timestamp, "SELL_SHORT", price, qty, "SHORT",
                                    0.0, fee_cost, 0.0, unrealized_pnl,
                                    self._equity(price),
                                ))
                                self.last_short_price = price  # cascade next entry above

            # Equity tracking
            long_pnl = sum((price - ep) * qty for ep, qty, _m, _t, _s in self.long_positions)
            short_pnl = sum((ep - price) * qty for ep, qty, _m, _t, _s in self.short_positions)
            unrealized_pnl = long_pnl + short_pnl
            equity = self.balance + self._locked_margin() + unrealized_pnl
            self.max_equity = max(self.max_equity, equity)
            drawdown = 1 - (equity / self.max_equity) if self.max_equity > 0 else 0
            realized_pnl_so_far = sum(t[5] for t in self.trade_history)
            self.equity_curve.append((
                timestamp, price, equity, realized_pnl_so_far, unrealized_pnl
            ))

            dd_halt         = self.config.get("dd_halt", False)
            dd_halt_candles = self.config.get("dd_halt_candles", 96)

            if drawdown >= self.max_drawdown and self.dd_halt_counter == 0:
                if dd_halt:
                    print(
                        f"\u26a0\ufe0f  [{timestamp}] DD halt FIRED at"
                        f" {drawdown*100:.1f}% drawdown \u2014 flattening all,"
                        f" pausing {dd_halt_candles} candles"
                    )
                    for ep, qty, margin_req, _tp, _sl in list(self.long_positions):
                        fee_cost  = qty * price * (self.fee / 2)
                        gross_pnl = (price - ep) * qty
                        net_pnl   = gross_pnl - fee_cost
                        self.balance += margin_req + net_pnl
                        self.trade_history.append((
                            timestamp, "DD_HALT_CLOSE", price, qty, "LONG",
                            net_pnl, fee_cost, gross_pnl, 0.0, self._equity(price),
                        ))
                    self.long_positions.clear()
                    for ep, qty, margin_req, _tp, _sl in list(self.short_positions):
                        fee_cost  = qty * price * (self.fee / 2)
                        gross_pnl = (ep - price) * qty
                        net_pnl   = gross_pnl - fee_cost
                        self.balance += margin_req + net_pnl
                        self.trade_history.append((
                            timestamp, "DD_HALT_CLOSE", price, qty, "SHORT",
                            net_pnl, fee_cost, gross_pnl, 0.0, self._equity(price),
                        ))
                    self.short_positions.clear()
                    if self.trend_position:
                        tp_p = self.trend_position
                        side = tp_p["side"]
                        gross_pnl = (
                            (price - tp_p["entry"]) if side == "long"
                            else (tp_p["entry"] - price)
                        ) * tp_p["qty"]
                        fee_cost = tp_p["qty"] * price * (self.fee / 2)
                        net_pnl  = gross_pnl - fee_cost
                        self.balance += tp_p["margin"] + net_pnl
                        self.trade_history.append((
                            timestamp, "DD_HALT_CLOSE", price, tp_p["qty"],
                            f"TREND_{side.upper()}",
                            net_pnl, fee_cost, gross_pnl, 0.0, self._equity(price),
                        ))
                        self.trend_position = None
                    self.last_long_price  = price
                    self.last_short_price = price
                    self.max_equity = self._equity(price)   # reset peak — prevents immediate re-trigger
                    self.dd_halt_counter = dd_halt_candles
                else:
                    print(f"\u26a0\ufe0f Max drawdown {drawdown * 100:.2f}% reached \u2013 stopping")
                    break

        return self.summary(price)

    # ------------------------------------------------------------------
    # Summary / export
    # ------------------------------------------------------------------

    def summary(self, final_price: float) -> Dict[str, Any]:
        long_pnl = sum((final_price - ep) * qty for ep, qty, _m, _t, _s in self.long_positions)
        short_pnl = sum((ep - final_price) * qty for ep, qty, _m, _t, _s in self.short_positions)
        trend_pnl = 0.0
        if self.trend_position:
            tp = self.trend_position
            trend_pnl = ((final_price - tp["entry"]) if tp["side"] == "long"
                         else (tp["entry"] - final_price)) * tp["qty"]
        unrealized_pnl = long_pnl + short_pnl + trend_pnl
        realized_pnl = sum(t[5] for t in self.trade_history if t[5] != 0.0)
        final_equity = self.balance + self._locked_margin() + unrealized_pnl
        return {
            "final_equity": final_equity,
            "return_pct": (final_equity - self.config["initial_balance"])
            / self.config["initial_balance"],
            "max_drawdown": 1 - final_equity / self.max_equity,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": realized_pnl + unrealized_pnl,
            "trades": len(self.trade_history),
            "direction": self.direction,
        }

    def export_trades(self, filename: str = "grid_orders_trades.csv") -> None:
        pd.DataFrame(
            self.trade_history,
            columns=[
                "time", "action", "price", "quantity", "direction",
                "pnl", "fee_cost", "gross_pnl", "unrealized_pnl", "total_equity",
            ],
        ).to_csv(filename, index=False)

    def export_equity_curve(self, filename: str = "equity_curve.csv") -> None:
        pd.DataFrame(
            self.equity_curve,
            columns=["time", "price", "equity", "realized_pnl", "unrealized_pnl"],
        ).to_csv(filename, index=False)


# ===========================================================================
# Bitunix data loader  (async – replaces load_data_for_date CSV reader)
# ===========================================================================

# Bitunix interval string → milliseconds per candle
_INTERVAL_MS: Dict[str, int] = {
    "1min":  60_000,
    "3min":  3 * 60_000,
    "5min":  5 * 60_000,
    "15min": 15 * 60_000,
    "30min": 30 * 60_000,
    "1hour": 60 * 60_000,
    "2hour": 2 * 60 * 60_000,
    "4hour": 4 * 60 * 60_000,
    "6hour": 6 * 60 * 60_000,
    "8hour": 8 * 60 * 60_000,
    "12hour": 12 * 60 * 60_000,
    "1day":  24 * 60 * 60_000,
    "3day":  3 * 24 * 60 * 60_000,
    "1week": 7 * 24 * 60 * 60_000,
}


async def fetch_klines_as_df(
    symbol: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    api_key: str = "",
    secret_key: str = "",
) -> pd.DataFrame:
    """
    Fetch all klines for *symbol* between *start_dt* and *end_dt* from Bitunix
    in 200-candle chunks (Bitunix API hard limit per request).

    Columns: open_time (datetime), open, high, low, close, volume (floats).

    Pagination notes (confirmed by live API tests):
    - The API returns candles DESCENDING (newest-first) within the requested window.
    - Passing only startTime causes the API to return the 200 most-recent candles
      with open_time >= startTime — making startTime-only pagination useless for
      historical data since it always returns today's candles.
    - Correct approach: pass both startTime and endTime as a sliding window of
      exactly CHUNK_SIZE candle-widths.  The API rounds startTime down to the
      nearest candle boundary, so we advance by interval_ms (not +1).

    *api_key* / *secret_key* are only required for private endpoints; the
    klines endpoint is public and they can be left as empty strings.
    """
    # ------------------------------------------------------------------
    # Disk cache: check for a pre-generated parquet file first.
    # Search order (first match wins):
    #   1. asBack/klines_cache/{symbol}_{interval}.parquet   — repo-local / CI
    #   2. ~/git/data/asgrid-klines/{symbol}_{interval}.parquet — local dev copy
    #      (69 MB 1min file lives here; not committed to git)
    # Generate / refresh via: scripts/generate_klines_cache.py
    # or the "Generate klines cache" GitHub Actions workflow.
    # ------------------------------------------------------------------
    _cache_candidates = [
        os.path.join(os.path.dirname(__file__), "klines_cache", f"{symbol}_{interval}.parquet"),
        os.path.expanduser(os.path.join("~", "git", "data", "asgrid-klines", f"{symbol}_{interval}.parquet")),
    ]
    _cache_file = next((p for p in _cache_candidates if os.path.exists(p)), None)
    if _cache_file is not None:
        df_full = pd.read_parquet(_cache_file)
        if df_full["open_time"].dt.tz is None:
            df_full["open_time"] = df_full["open_time"].dt.tz_localize("UTC")
        _start = start_dt if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc)
        _end   = end_dt   if end_dt.tzinfo   else end_dt.replace(tzinfo=timezone.utc)
        df = df_full[(df_full["open_time"] >= _start) & (df_full["open_time"] < _end)].copy().reset_index(drop=True)
        if len(df) > 0:
            print(f"  → Cache hit ({os.path.basename(os.path.dirname(_cache_file))}): {len(df):,} candles for {symbol} {interval} ({start_dt.date()} → {end_dt.date()})")
            return df
        print(f"  → Cache file found but no rows in range — falling back to API")

    exchange = BitunixExchange(api_key=api_key, secret_key=secret_key)

    def _to_ms(dt: datetime) -> int:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    interval_ms = _INTERVAL_MS.get(interval)
    if interval_ms is None:
        raise ValueError(
            f"Unknown interval '{interval}'. Expected one of: {list(_INTERVAL_MS)}"
        )

    CHUNK_SIZE = 200  # Bitunix API hard limit per request
    WINDOW_MS  = CHUNK_SIZE * interval_ms  # total time span per request

    all_candles: List[Dict[str, Any]] = []
    chunk_start = _to_ms(start_dt)
    end_ms = _to_ms(end_dt)

    # Estimate total chunks for progress reporting
    total_ms = end_ms - _to_ms(start_dt)
    estimated_chunks = max(1, total_ms // WINDOW_MS)
    chunk_num = 0

    print(f"Fetching {symbol} {interval} klines from {start_dt.date()} to {end_dt.date()} …")
    print(f"  (estimated ~{estimated_chunks} API requests)")

    seen_times: set = set()  # de-duplicate candles at chunk boundaries

    while chunk_start < end_ms:
        chunk_end = min(chunk_start + WINDOW_MS, end_ms)

        candles = await exchange.get_klines(
            symbol=symbol,
            interval=interval,
            start_time=chunk_start,
            end_time=chunk_end,
            limit=CHUNK_SIZE,
        )
        chunk_num += 1
        if chunk_num % 50 == 0 or chunk_num == 1:
            pct = min(100, int(len(all_candles) / max(1, estimated_chunks * CHUNK_SIZE) * 100))
            print(f"  chunk {chunk_num}/{estimated_chunks}  ~{pct}% …", flush=True)

        if not candles:
            # Small pause before retrying or moving on — avoids tight loops on empty windows
            await asyncio.sleep(0.1)
            break

        # API returns newest-first — sort ascending so candles[-1] is newest.
        candles.sort(key=lambda c: c["open_time"])

        # De-duplicate: the API includes the endTime boundary candle in both
        # the current and following chunk (off-by-one at minute boundaries).
        new_candles = [c for c in candles if c["open_time"] not in seen_times]
        for c in new_candles:
            seen_times.add(c["open_time"])
        all_candles.extend(new_candles)

        last_time = candles[-1]["open_time"]
        if last_time <= chunk_start:
            break  # Guard against infinite loop

        # Advance by one candle width (not +1 ms) to avoid the boundary overlap.
        chunk_start = last_time + interval_ms

    if not all_candles:
        raise ValueError(f"No klines returned for {symbol} between {start_dt} and {end_dt}")

    df = pd.DataFrame(all_candles)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    price_range = f"${df['low'].min():,.0f} – ${df['high'].max():,.0f}"
    print(f"  → {len(df):,} candles loaded  |  price range: {price_range}")
    return df


# ===========================================================================
# Visualisation helpers (identical to backtest_grid_auto.py)
# ===========================================================================

def plot_equity_curve(bt: GridOrderBacktester) -> None:
    df = pd.DataFrame(
        bt.equity_curve,
        columns=["time", "price", "equity", "realized_pnl", "unrealized_pnl"],
    )
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

    trades_df = pd.DataFrame(
        bt.trade_history,
        columns=[
            "time", "action", "price", "quantity", "direction",
            "pnl", "fee_cost", "gross_pnl", "unrealized_pnl", "total_equity",
        ],
    )
    trades_df["time"] = pd.to_datetime(trades_df["time"], errors="coerce", utc=True)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 12), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    ax1.plot(df["time"], df["price"], label="Price", color="blue", alpha=0.5)
    if not trades_df.empty:
        for action, marker, color, label in [
            ("BUY", "^", "green", "BUY"),
            ("SELL", "v", "red", "SELL"),
            ("SELL_SHORT", "v", "purple", "SELL_SHORT"),
            ("COVER_SHORT", "^", "orange", "COVER_SHORT"),
            ("TREND_CLOSE_SHORT", "X", "cyan", "TREND_CLOSE_SHORT"),
            ("TREND_CLOSE_LONG",  "X", "magenta", "TREND_CLOSE_LONG"),
        ]:
            sub = trades_df[trades_df["action"] == action]
            if not sub.empty:
                ax1.scatter(sub["time"], sub["price"], marker=marker,
                            color=color, label=label, s=60, zorder=3)
    ax1.set_ylabel("Price")
    ax1.set_title("Price with Trade Signals (Bitunix)")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2.plot(df["time"], df["equity"], color="green", label="Equity")
    ax2.plot(df["time"], df["realized_pnl"], color="blue", linestyle="--", label="Realized PnL")
    ax2.plot(df["time"], df["unrealized_pnl"], color="red", linestyle=":", label="Unrealized PnL")
    ax2.set_ylabel("Account Value")
    ax2.set_title("Equity & PnL")
    ax2.legend(loc="upper left")
    ax2.grid(True)

    if not trades_df.empty:
        ax3.plot(trades_df["time"], trades_df["total_equity"],
                 color="purple", label="Total Equity")
        ax3.set_ylabel("Total Equity")
        ax3.set_title("Total Account Equity")
        ax3.legend(loc="upper left")
        ax3.grid(True)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def visualize_results(df_results: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_results, x="strategy_name", y="return_pct",
                hue="strategy_name", palette="Blues_d", legend=False)
    plt.title("Return by Strategy (Bitunix)")
    plt.xlabel("Strategy")
    plt.ylabel("Return (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ===========================================================================
# Quarterly P&L helper
# ===========================================================================


def _print_quarterly_breakdown(bt: "GridOrderBacktester", initial_balance: float) -> None:
    """Print a quarter-by-quarter equity breakdown from the backtester's equity curve.

    Shows per-quarter and cumulative returns so it's easy to spot which
    sub-periods drove (or hurt) the overall result — especially useful for
    multi-year windows where a single aggregate return hides intra-period drawdowns.
    """
    if len(bt.equity_curve) < 2:
        return

    ec_df = pd.DataFrame(
        bt.equity_curve,
        columns=["time", "price", "equity", "realized", "unrealized"],
    )
    ec_df["time"] = pd.to_datetime(ec_df["time"])
    ec_df = ec_df.set_index("time")

    q_end = ec_df["equity"].resample("QE").last().dropna()
    if q_end.empty:
        return

    prev_eq = initial_balance
    print("     " + "\u2500" * 60)
    print("     Quarter-by-quarter breakdown:")
    for q_date, eq in q_end.items():
        q_return   = (eq - prev_eq) / prev_eq * 100
        cum_return = (eq - initial_balance) / initial_balance * 100
        period_str = f"{q_date.year} Q{q_date.quarter}"
        flag = "  \u25c4 LOSS" if q_return < -3 else ("  \u2605" if q_return > 10 else "")
        print(
            f"       {period_str}: ${prev_eq:>8,.2f} \u2192 ${eq:>8,.2f}"
            f"  {q_return:>+7.2f}%  (cum {cum_return:>+7.2f}%){flag}"
        )
        prev_eq = eq
    print("     " + "\u2500" * 60)


def _print_weekly_breakdown(bt: "GridOrderBacktester", initial_balance: float) -> None:
    """Print a week-by-week equity breakdown from the backtester's equity curve.

    Crypto regime shifts, liquidation cascades, and parabolic runs often play
    out over days rather than months.  Weekly granularity surfaces which exact
    weeks drove (or hurt) the overall result — especially useful when comparing
    strategies like BTBW where the difference concentrates in a handful of weeks.

    Flags:
      ◄ LOSS  weekly return < -3% (significant down week)
      ★       weekly return >  5% (notable bull week)
    """
    if len(bt.equity_curve) < 2:
        return

    ec_df = pd.DataFrame(
        bt.equity_curve,
        columns=["time", "price", "equity", "realized", "unrealized"],
    )
    ec_df["time"] = pd.to_datetime(ec_df["time"])
    ec_df = ec_df.set_index("time")

    # Resample to week-ending Sunday (pandas default for "W")
    w_end = ec_df["equity"].resample("W").last().dropna()
    if w_end.empty:
        return

    prev_eq = initial_balance
    print("     " + "\u2500" * 60)
    print("     Week-by-week breakdown (week ending Sunday):")
    for w_date, eq in w_end.items():
        w_return   = (eq - prev_eq) / prev_eq * 100
        cum_return = (eq - initial_balance) / initial_balance * 100
        flag = "  \u25c4 LOSS" if w_return < -3 else ("  \u2605" if w_return > 5 else "")
        print(
            f"       {w_date.strftime('%Y-%m-%d')}  {w_return:>+7.2f}%  (cum {cum_return:>+7.2f}%){flag}"
        )
        prev_eq = eq
    print("     " + "\u2500" * 60)


def _print_daily_breakdown(bt: "GridOrderBacktester", initial_balance: float) -> None:
    """Print a day-by-day equity + trade breakdown from the backtester.

    For each calendar day shows:
      - Opening / closing equity and daily % change
      - Cumulative return from initial balance
      - Number of trades executed that day
      - Realized P&L for that day

    Flags:
      \u25c4  daily return < -1% (significant down day)
      \u2605  daily return >  2% (notable up day)
      \u00b7  zero-trade day (no activity)

    Designed for the full 6.5-year backtest so we can identify exactly which
    days drive or hurt performance and find remaining gaps.
    """
    if len(bt.equity_curve) < 2:
        return

    # \u2500\u2500 Equity curve by day \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    ec_df = pd.DataFrame(
        bt.equity_curve,
        columns=["time", "price", "equity", "realized", "unrealized"],
    )
    ec_df["time"] = pd.to_datetime(ec_df["time"])
    ec_df = ec_df.set_index("time")

    d_end = ec_df["equity"].resample("D").last().dropna()
    if d_end.empty:
        return

    # \u2500\u2500 Trade counts & realized P&L by day \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if bt.trade_history:
        tr_df = pd.DataFrame(
            bt.trade_history,
            columns=[
                "time", "action", "price", "quantity", "direction",
                "pnl", "fee_cost", "gross_pnl", "unrealized_pnl", "total_equity",
            ],
        )
        tr_df["time"] = pd.to_datetime(tr_df["time"])
        tr_df["date"] = tr_df["time"].dt.date
        trades_per_day   = tr_df.groupby("date").size()
        realized_per_day = tr_df.groupby("date")["pnl"].sum()
    else:
        trades_per_day   = pd.Series(dtype=int)
        realized_per_day = pd.Series(dtype=float)

    # \u2500\u2500 Accumulators for summary stats \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    total_days      = 0
    loss_days       = 0
    zero_trade_days = 0
    best_day        = ("", -999.0)
    worst_day       = ("", 999.0)

    prev_eq = initial_balance
    header = (
        "     \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510\n"
        "     \u2502     Date     \u2502   Equity $  \u2502   Daily %   \u2502  Cum %  \u2502 Trades \u2502 Real P&L  \u2502\n"
        "     \u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524"
    )
    print("     " + "\u2500" * 76)
    print("     Day-by-day breakdown:")
    print(header)

    for d_date, eq in d_end.items():
        d_return   = (eq - prev_eq) / prev_eq * 100 if prev_eq else 0.0
        cum_return = (eq - initial_balance) / initial_balance * 100
        day_key    = d_date.date() if hasattr(d_date, "date") else d_date
        n_trades   = int(trades_per_day.get(day_key, 0))
        day_pnl    = float(realized_per_day.get(day_key, 0.0))

        flag = ""
        if d_return < -1:
            flag = " \u25c4"
            loss_days += 1
        elif d_return > 2:
            flag = " \u2605"
        if n_trades == 0:
            flag += " \u00b7"
            zero_trade_days += 1

        total_days += 1
        if d_return > best_day[1]:
            best_day = (str(day_key), d_return)
        if d_return < worst_day[1]:
            worst_day = (str(day_key), d_return)

        print(
            f"     \u2502 {str(day_key):>12s} \u2502 ${eq:>10,.2f} \u2502 {d_return:>+9.2f}%  \u2502{cum_return:>+7.1f}% \u2502 {n_trades:>5d}  \u2502 ${day_pnl:>+8.2f} \u2502{flag}"
        )
        prev_eq = eq

    print(
        "     \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518"
    )

    # \u2500\u2500 Summary stats \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    active_days = total_days - zero_trade_days
    print(f"     Summary: {total_days} days total, {active_days} active, {zero_trade_days} idle, {loss_days} loss days (< -1%)")
    print(f"     Best day:  {best_day[0]}  {best_day[1]:>+.2f}%")
    print(f"     Worst day: {worst_day[0]}  {worst_day[1]:>+.2f}%")
    print("     " + "\u2500" * 76)


# ===========================================================================
# Grid-search orchestrator
# ===========================================================================

async def grid_search_backtest_async(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Run the full grid search defined in *config* against Bitunix historical data.
    Returns a DataFrame of results, one row per param set.
    """
    # 1. Fetch klines once, reused across all param sets
    full_df = await fetch_klines_as_df(
        symbol=config["symbol"],
        interval=config["interval"],
        start_dt=config["start_date"],
        end_dt=config["end_date"],
        api_key=config.get("api_key", ""),
        secret_key=config.get("secret_key", ""),
    )

    results = []
    best_result = None
    best_bt = None

    # Pre-compute 15min EMA arrays for any param sets that use the regime filter.
    # We load 15min klines once and align them to the 1min index via forward-fill.
    _regime_ema_cache: Dict[int, np.ndarray] = {}
    _regime_param_sets = [p for p in config["param_sets"] if p.get("regime_filter")]
    if _regime_param_sets:
        df_15m = await fetch_klines_as_df(
            symbol=config["symbol"],
            interval="15min",
            start_dt=config["start_date"],
            end_dt=config["end_date"],
            api_key=config.get("api_key", ""),
            secret_key=config.get("secret_key", ""),
        )
        ts_15m = pd.to_datetime(df_15m["open_time"], unit="ms", utc=True)
        ts_1m  = pd.to_datetime(full_df["open_time"],  unit="ms", utc=True)
        for p in _regime_param_sets:
            period = int(p["regime_ema_period"])
            if period not in _regime_ema_cache:
                ema_15m = df_15m["close"].ewm(span=period, adjust=False).mean()
                ema_series = pd.Series(ema_15m.values, index=ts_15m)
                aligned = ema_series.reindex(ts_1m, method="ffill").bfill()
                _regime_ema_cache[period] = aligned.values

    for params in config["param_sets"]:
        print(f"\n🚀 Strategy: {params['name']}")
        print(
            f"  Long  | TP: {params['long_settings']['up_spacing'] * 100:.2f}%  "
            f"RE: {params['long_settings']['down_spacing'] * 100:.2f}%"
        )
        print(
            f"  Short | RE: {params['short_settings']['up_spacing'] * 100:.2f}%  "
            f"TP: {params['short_settings']['down_spacing'] * 100:.2f}%"
        )

        temp_config = {k: v for k, v in config.items() if k not in ("param_sets",)}
        temp_config.update(
            {
                "long_settings": params["long_settings"],
                "short_settings": params["short_settings"],
                "use_sl": params.get("use_sl", True),
                # Per-strategy overrides (fall back to global config value)
                **({k: params[k] for k in (
                    "trend_detection", "trend_capture",
                    "trend_force_close_grid", "trend_confirm_candles",
                    "trend_capture_size_pct", "trend_trailing_stop_pct",
                    "trend_capture_velocity_pct", "trend_velocity_pct",
                    "trend_lookback_candles",
                    # ADX filter params
                    "adx_filter", "adx_period", "adx_min_trend", "adx_grid_pause",
                    # ATR dynamic trail params
                    "atr_trail", "atr_period", "atr_trail_multiplier",
                    "atr_trail_min", "atr_trail_max",
                    # EMA bias filter params
                    "ema_bias_filter", "ema_slow_period",
                    # BB squeeze filter params
                    "bb_squeeze_gate", "bb_squeeze_boost", "bb_period", "bb_mult",
                    "bb_squeeze_threshold", "bb_squeeze_boost_mult",
                    # RSI filter params
                    "rsi_filter", "rsi_period", "rsi_overbought", "rsi_oversold",
                    "rsi_momentum",
                    # Volume confirmation params
                    "vol_filter", "vol_period", "vol_multiplier",
                    # Market structure params
                    "ms_filter", "ms_lookback",
                    # v9 re-entry + adaptive trail params
                    "trend_reentry_fast",
                    "adx_wide_trail_threshold", "adx_wide_trail_pct",
                    # v10 position management params
                    "rsi_tight_trail", "rsi_tight_trail_ob", "rsi_tight_trail_os",
                    "rsi_tight_trail_pct",
                    "vol_reentry_scale", "vol_reentry_high_mult", "vol_reentry_low_pct",
                    # v11 crash protection params
                    "crash_cb", "crash_cb_drop_pct", "crash_cb_lookback_candles",
                    "crash_cb_halt_candles",
                    "dd_halt", "max_drawdown", "dd_halt_candles",
                    "grid_notional_cap_pct",
                    # v12 grid vol scale
                    "grid_vol_scale",
                    # v13 grid vol floor + period
                    "grid_vol_floor", "vol_period",
                    # v15 regime filter
                    "regime_filter", "regime_ema_period", "regime_hysteresis_pct",
                    # v16 XRPPM6 structural params
                    "max_positions_per_side", "regime_short_cap",
                    # v17 XRPPM7 params
                    "leverage", "order_value_pct", "order_value_min", "regime_short_spacing",
                    # v18 XRPPM9 params
                    "vol_adaptive_spacing", "vas_floor", "vas_ceil", "vas_period",
                    "bull_spacing", "bear_spacing",
                    # v21 XRPPM11 — regime & parabolic protection layers
                    "atr_parabolic_mult", "htf_ema_align", "regime_vote_mode",
                    # v22 XRPPM12 — grid sleep
                    "grid_sleep_atr_thresh",
                    # v23 XRPPM13 — adaptive ATR gate
                    "atr_regime_adaptive", "atr_bull_mult", "atr_bear_mult",
                    "atr_percentile_gate", "atr_pct_threshold", "atr_pct_window",
                    "atr_adx_scale", "atr_adx_base_mult", "atr_adx_max_mult",
                    # v24 XRPPM14 — directional velocity gate
                    "atr_directional", "atr_dir_lookback", "atr_dir_drop_pct",
                    "atr_acceleration", "atr_accel_lookback",
                    "atr_cooldown",
                    # v26 XRPPM16 — dynamic self-calibrating mechanisms
                    "vel_atr_mult",
                    "gate_decay_scale",
                    "cap_size_atr_scale", "cap_size_atr_floor", "cap_size_atr_ceiling",
                    "trend_max_loss_atr",
                    # v27 XRPPM17 — directional & outcome-based mechanisms
                    "vel_dir_only", "vel_accel_only",
                    "eq_curve_filter", "eq_curve_lookback",
                    "consec_loss_max", "consec_loss_pause",
                    # v28 XRPPM18 — fine-tuning directional velocity
                    "vel_dir_ema_period",
                    # v34 XRPPM24 — 5-regime rotation
                    "regime_rotation", "regime_trending_grid_off",
                    "regime_adx_trending_min", "regime_adx_choppy_max",
                    "regime_bb_squeeze_thresh", "regime_bb_breakout_thresh",
                    "regime_rsi_ob", "regime_rsi_os",
                    # v35 XRPPM25 — regime detection quality
                    "regime_min_dwell_candles", "regime_autocorr_choppy_max",
                    # v36 — ATR-adaptive trailing stop
                    "atr_trail", "atr_trail_multiplier", "atr_trail_min", "atr_trail_max",
                    # v37 — circuit breakers & drawdown halt
                    "surge_cb", "surge_cb_rise_pct", "surge_cb_lookback_candles",
                    "surge_cb_halt_candles",
                    # v38 — regime-gated shorts
                    "regime_short_gate",
                ) if k in params}),
            }
        )

        # Per-run df: add regime_ema column if this param set uses the filter
        df_slice = full_df.copy()
        if params.get("regime_filter") and params.get("regime_ema_period") in _regime_ema_cache:
            df_slice["regime_ema"] = _regime_ema_cache[params["regime_ema_period"]]

        bt = GridOrderBacktester(df_slice, None, temp_config)
        result = bt.run()
        result.update(
            {
                "strategy_name": params["name"],
                "long_up": params["long_settings"]["up_spacing"],
                "long_down": params["long_settings"]["down_spacing"],
                "short_up": params["short_settings"]["up_spacing"],
                "short_down": params["short_settings"]["down_spacing"],
            }
        )
        results.append(result)
        open_long  = len(bt.long_positions)
        open_short = len(bt.short_positions)
        print(
            f"  → return: {result['return_pct'] * 100:.2f}%  "
            f"trades: {result['trades']}  "
            f"max_dd: {result['max_drawdown'] * 100:.2f}%"
        )
        print(
            f"     realized: ${result['realized_pnl']:+.2f}  "
            f"unrealized: ${result['unrealized_pnl']:+.2f}  "
            f"open at end: {open_long}L / {open_short}S"
        )
        _print_quarterly_breakdown(bt, config["initial_balance"])
        _print_weekly_breakdown(bt, config["initial_balance"])
        if config.get("daily_breakdown"):
            _print_daily_breakdown(bt, config["initial_balance"])

        if best_result is None or result["return_pct"] > best_result["return_pct"]:
            best_result = result
            best_bt = bt

    # 2. Persist results
    df_results = pd.DataFrame(results)
    df_results.to_csv("bitunix_grid_search_results.csv", index=False)
    print("\n✅ Results saved → bitunix_grid_search_results.csv")

    if best_bt is not None:
        print(
            f"\n🏆 Best strategy: {best_result['strategy_name']}  "
            f"return: {best_result['return_pct'] * 100:.2f}%"
        )
        best_bt.export_trades("bitunix_best_grid_trades.csv")
        best_bt.export_equity_curve("bitunix_best_equity_curve.csv")
        visualize_results(df_results)
        plot_equity_curve(best_bt)

    return df_results


def grid_search_backtest(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Synchronous wrapper around :func:`grid_search_backtest_async`."""
    return asyncio.run(grid_search_backtest_async(config or CONFIG))


# ===========================================================================
# Configuration
# ===========================================================================

CONFIG: Dict[str, Any] = {
    # ── Bitunix credentials (only needed if you add private endpoints later)
    "api_key": os.getenv("BITUNIX_API_KEY", ""),
    "secret_key": os.getenv("BITUNIX_SECRET_KEY", ""),

    # ── Market
    "symbol": "BTCUSDT",
    "interval": "1min",           # Bitunix intervals: 1min 3min 5min 15min 30min 1hour 2hour 4hour 6hour 8hour 12hour 1day 3day 1week

    # ── Date range
    # end_date is exclusive (midnight = start of that day), so use Aug 1 to
    # include all 31 days of July (31 × 1440 = 44,640 candles expected).
    "start_date": datetime(2025, 7, 1),
    "end_date": datetime(2025, 8, 1),

    # ── Account / risk
    "initial_balance": 1000,
    "order_value": 100,          # USD per grid order (~10% of balance per level)
    "max_drawdown": 0.9,
    "max_positions": 6,          # total cap (3 long + 3 short)
    "max_positions_per_side": 3, # allow 3 stacked levels per side
    "max_unrealized_loss_per_side": 30, # USD — circuit breaker: stop adding to
                                         # a trending side once it's down >$30
    "sl_multiplier": 2.0,        # SL distance = sl_multiplier × spacing (e.g. 2× TP distance)
    "min_sl_pct": 0.01,          # floor: SL never closer than 1.0% from entry (prevents noise-stops on tight spacings)
    "fee_pct": 0.0006,           # Bitunix taker fee 0.06 %
    "leverage": 1,
    # direction="both" = market-neutral: profits from oscillation in either direction
    "direction": "both",         # "long" | "short" | "both"
    "grid_refresh_interval": 10, # minutes between re-anchoring entry levels
    # ── Trend detection + side liquidation
    # When price moves >trend_velocity_pct in trend_lookback_candles candles,
    # close all positions on the losing side and ride the profitable side.
    # Resumes hedge mode after trend_cooldown_candles of subdued velocity.
    "trend_detection": True,
    "trend_lookback_candles": 15,   # rolling window for velocity calculation
    "trend_velocity_pct": 0.01,     # 1.0% move in 15min = trend signal
    "trend_cooldown_candles": 30,   # quiet candles before resuming hedge mode

    # ── Parameter sets to grid-search
    # Spacing = distance between levels as a fraction of price.
    # Tighter → more fills, smaller profit each.  Wider → fewer fills, larger profit each.
    # With 3 levels per side spacing also determines max deployed capital:
    #   3 × $100 × 2 sides = $600 max out of $1,000 balance.
    "param_sets": [
        {
            "name": "tight_0.2pct",
            "use_sl": False,  # SL disabled: rely on circuit breaker only
            "long_settings":  {"up_spacing": 0.002, "down_spacing": 0.002},
            "short_settings": {"up_spacing": 0.002, "down_spacing": 0.002},
        },
        {
            "name": "medium_0.3pct",
            "use_sl": False,
            "long_settings":  {"up_spacing": 0.003, "down_spacing": 0.003},
            "short_settings": {"up_spacing": 0.003, "down_spacing": 0.003},
        },
        {
            "name": "medium_0.5pct",
            "use_sl": False,
            "long_settings":  {"up_spacing": 0.005, "down_spacing": 0.005},
            "short_settings": {"up_spacing": 0.005, "down_spacing": 0.005},
        },
        {
            "name": "wide_0.8pct",
            "use_sl": True,   # SL enabled: 2× spacing = 1.6%, clears noise
            "long_settings":  {"up_spacing": 0.008, "down_spacing": 0.008},
            "short_settings": {"up_spacing": 0.008, "down_spacing": 0.008},
        },
        {
            "name": "wide_1.0pct",
            "use_sl": True,
            "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
            "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
        },
    ],
}

# ===========================================================================
# XRP/USDT config  (target for live deployment)
# ===========================================================================
# XRP trades at ~$2–$3 and has higher % daily volatility than BTC (often
# 3–8% swings per day vs BTC's 1–3%).  Tighter grids fill more frequently.
# Spacing is the same % concept but we sweep tighter values down to 0.1%.
XRP_CONFIG: Dict[str, Any] = {
    # ── Market
    "symbol": "XRPUSDT",
    "interval": "1min",

    # ── Date range — 6-month out-of-sample validation (Aug 2025 → Feb 2026)
    # Tuning was done on Jul 2025; this is a forward, unseen window.
    "start_date": datetime(2025, 8, 1),
    "end_date":   datetime(2026, 2, 1),

    # ── Account / risk
    "initial_balance": 1000,
    "order_value": 100,          # USD per grid order
    "max_drawdown": 0.9,
    "max_positions": 6,          # 3 long + 3 short
    "max_positions_per_side": 3,
    "max_unrealized_loss_per_side": 30, # USD — circuit breaker: stop adding to
                                         # a trending side once it's down >$30
    "sl_multiplier": 2.0,        # SL distance = sl_multiplier × spacing
    "min_sl_pct": 0.01,          # floor: SL never closer than 1.0% from entry
    "fee_pct": 0.0006,           # Bitunix taker fee 0.06 %
    "leverage": 1,
    "direction": "both",
    "grid_refresh_interval": 10, # minutes
    # ── Trend detection + capture (can be overridden per param_set)
    "trend_detection": True,        # default ON for tight strategies; wide disables it per-set
    "trend_capture": True,          # open a directional position when trend fires
    "trend_lookback_candles": 15,   # XRP moves fast — 15min window catches early
    "trend_velocity_pct": 0.04,     # 4.0% in 15min = strong trend for XRP
    "trend_cooldown_candles": 30,   # 30 quiet candles before resuming hedge mode
    "trend_capture_size_pct": 0.15,   # 15% equity per trend position (down from 30%)
    "trend_trailing_stop_pct": 0.04,   # 4% trail — gives room for XRP 2-3% retracements
    "trend_confirm_candles": 3,         # require 3 consecutive candles above threshold
    "trend_capture_velocity_pct": 0.06, # entry only on strong moves ≥6% (not every 4% blip)
    "trend_force_close_grid": False,    # default: DON'T force-close grid (override per-set)

    # ── Ceiling probe (s90 control) + find hard ceiling at s100/s110
    "param_sets": [
        {
            "name": "FINAL_s90_l10",      # ★ confirmed optimal — +38.33% on 6-month OOS
            "use_sl": True, "trend_detection": True, "trend_capture": True,
            "trend_force_close_grid": True, "trend_confirm_candles": 3,
            "trend_trailing_stop_pct": 0.04, "trend_capture_size_pct": 0.90,
            "trend_lookback_candles": 10,
            "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
            "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
        },
        {
            "name": "FINAL_s40_l10",      # conservative variant (+14.32%)
            "use_sl": True, "trend_detection": True, "trend_capture": True,
            "trend_force_close_grid": True, "trend_confirm_candles": 3,
            "trend_trailing_stop_pct": 0.04, "trend_capture_size_pct": 0.40,
            "trend_lookback_candles": 10,
            "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
            "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
        },
        {
            "name": "FINAL_trend_off",    # grid-only baseline (-3.95% on 6-month OOS)
            "use_sl": True, "trend_detection": False, "trend_capture": False,
            "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
            "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
        },
    ],
}

# ---------------------------------------------------------------------------
# Robustness check: run the best known params on an earlier, more ranging
# window (May–Aug 2025) to validate against overfitting to the Aug-Feb bull run.
# ---------------------------------------------------------------------------
XRP_VALIDATE_CONFIG = dict(XRP_CONFIG)
XRP_VALIDATE_CONFIG["start_date"] = datetime(2025, 5, 1)
XRP_VALIDATE_CONFIG["end_date"]   = datetime(2025, 8, 1)
XRP_VALIDATE_CONFIG["param_sets"] = [
    {
        "name": "val_s90_l10",
        "use_sl": True, "trend_detection": True, "trend_capture": True,
        "trend_force_close_grid": True, "trend_confirm_candles": 3,
        "trend_trailing_stop_pct": 0.04, "trend_capture_size_pct": 0.90,
        "trend_lookback_candles": 10,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
    {
        "name": "val_s40_l10",  # conservative sizing for comparison
        "use_sl": True, "trend_detection": True, "trend_capture": True,
        "trend_force_close_grid": True, "trend_confirm_candles": 3,
        "trend_trailing_stop_pct": 0.04, "trend_capture_size_pct": 0.40,
        "trend_lookback_candles": 10,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
    {
        "name": "val_trend_off",  # baseline: grid only
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]

# ---------------------------------------------------------------------------
# Long-term walk-forward: 2 years (Feb 28 2024 → Feb 28 2026)
# Covers the full 2024 bear/recovery cycle + 2025 XRP bull run.
# Uses best-known params only — data fetch is ~5,256 chunks (≈9 min of CI)
# ---------------------------------------------------------------------------
_LONGTERM_PARAM_SETS = [
    {
        "name": "lt_s90_l10",         # ★ optimised
        "use_sl": True, "trend_detection": True, "trend_capture": True,
        "trend_force_close_grid": True, "trend_confirm_candles": 3,
        "trend_trailing_stop_pct": 0.04, "trend_capture_size_pct": 0.90,
        "trend_lookback_candles": 10,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
    {
        "name": "lt_s40_l10",         # conservative
        "use_sl": True, "trend_detection": True, "trend_capture": True,
        "trend_force_close_grid": True, "trend_confirm_candles": 3,
        "trend_trailing_stop_pct": 0.04, "trend_capture_size_pct": 0.40,
        "trend_lookback_candles": 10,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
    {
        "name": "lt_trend_off",       # grid-only baseline
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]

XRP_2Y_CONFIG = dict(XRP_CONFIG)
XRP_2Y_CONFIG["start_date"]  = datetime(2024, 2, 28)
XRP_2Y_CONFIG["end_date"]    = datetime(2026, 2, 28)
XRP_2Y_CONFIG["param_sets"]  = _LONGTERM_PARAM_SETS

# ---------------------------------------------------------------------------
# Maximum-available history: Apr 20 2022 → Feb 28 2026 (~3y 10m)
# Bitunix XRPUSDT 1min data confirmed available from ~Apr 19 2022.
# Covers: 2022 crash, 2023 range, 2024 recovery, 2025 bull run.
# Data fetch ~9,800 chunks (≈17 min of CI); simulation ≈ same again per set.
# ---------------------------------------------------------------------------
XRP_MAX_CONFIG = dict(XRP_CONFIG)
XRP_MAX_CONFIG["start_date"]  = datetime(2022, 4, 20)
XRP_MAX_CONFIG["end_date"]    = datetime(2026, 2, 28)
XRP_MAX_CONFIG["param_sets"]  = [
    {
        "name": "max_s90_l10",        # ★ optimised at max history
        "use_sl": True, "trend_detection": True, "trend_capture": True,
        "trend_force_close_grid": True, "trend_confirm_candles": 3,
        "trend_trailing_stop_pct": 0.04, "trend_capture_size_pct": 0.90,
        "trend_lookback_candles": 10,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
    {
        "name": "max_trend_off",      # grid-only across full history
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]

# ===========================================================================
# v2 — ADX filter sweep
# Gate trend-capture entries on ADX strength to reduce false signals.
# Sweep: no ADX gate vs thresholds 20/25/30, plus an ADX+grid-pause variant.
# Also run the 2-year walk-forward with the same sweep to see if it fixes the
# -11.45% bear-market bleed.
# ===========================================================================

def _adx_set(name: str, adx_on: bool, adx_min: float = 25.0,
             grid_pause: float = None, size: float = 0.90) -> Dict[str, Any]:
    """Helper to build an ADX-sweep param set from the s90_l10 baseline."""
    d: Dict[str, Any] = {
        "name": name,
        "use_sl": True,
        "trend_detection": True, "trend_capture": True,
        "trend_force_close_grid": True, "trend_confirm_candles": 3,
        "trend_trailing_stop_pct": 0.04, "trend_capture_size_pct": size,
        "trend_lookback_candles": 10,
        "adx_filter": adx_on,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    }
    if adx_on:
        d["adx_min_trend"] = adx_min
    if grid_pause is not None:
        d["adx_grid_pause"] = grid_pause
    return d


XRP_ADX_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_ADX_CONFIG["param_sets"] = [
    _adx_set("adx_off",         adx_on=False),                        # baseline
    _adx_set("adx_t20",         adx_on=True,  adx_min=20.0),
    _adx_set("adx_t25",         adx_on=True,  adx_min=25.0),
    _adx_set("adx_t30",         adx_on=True,  adx_min=30.0),
    _adx_set("adx_t25_gp35",   adx_on=True,  adx_min=25.0, grid_pause=35.0),
    {   # grid-only control (no trend capture)
        "name": "adx_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]

XRP_ADX_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_ADX_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_ADX_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_ADX_2Y_CONFIG["param_sets"] = [
    _adx_set("2y_adx_off",       adx_on=False),           # 2y baseline
    _adx_set("2y_adx_t25",       adx_on=True, adx_min=25.0),
    _adx_set("2y_adx_t25_gp35",  adx_on=True, adx_min=25.0, grid_pause=35.0),
    {   # grid-only 2y baseline
        "name": "2y_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]


# ===========================================================================
# v3 — ATR dynamic trail sweep
# Replace fixed 4% trail with N × ATR / price, adapting to volatility regime.
# High-vol candles widen the stop (let trends breathe); low-vol tighten it
# (lock in gains quickly). Sweep multipliers 1×/2×/3× vs fixed 4% baseline.
# ===========================================================================

def _atr_set(name: str, atr_on: bool, mult: float = 2.0, size: float = 0.90,
             fixed_trail: float = 0.04) -> Dict[str, Any]:
    """Build a param set using the s90_l10/confirm=3/force_close baseline + ATR trail."""
    d: Dict[str, Any] = {
        "name": name,
        "use_sl": True,
        "trend_detection": True, "trend_capture": True,
        "trend_force_close_grid": True, "trend_confirm_candles": 3,
        "trend_trailing_stop_pct": fixed_trail,
        "trend_capture_size_pct": size,
        "trend_lookback_candles": 10,
        "atr_trail": atr_on,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    }
    if atr_on:
        d["atr_trail_multiplier"] = mult
    return d


XRP_ATR_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_ATR_CONFIG["param_sets"] = [
    _atr_set("atr_off_4pct",  atr_on=False, fixed_trail=0.04),   # baseline (matches s90)
    _atr_set("atr_1x",        atr_on=True,  mult=1.0),
    _atr_set("atr_2x",        atr_on=True,  mult=2.0),
    _atr_set("atr_3x",        atr_on=True,  mult=3.0),
    _atr_set("atr_2x_6pct",   atr_on=False, fixed_trail=0.06),   # fixed-6% control
    {   # grid-only control
        "name": "atr_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]

XRP_ATR_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_ATR_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_ATR_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_ATR_2Y_CONFIG["param_sets"] = [
    _atr_set("2y_atr_off",   atr_on=False, fixed_trail=0.04),
    _atr_set("2y_atr_2x",    atr_on=True,  mult=2.0),
    _atr_set("2y_atr_3x",    atr_on=True,  mult=3.0),
    {   # grid-only 2y baseline
        "name": "2y_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]


# ===========================================================================
# v4 — EMA bias filter sweep
# Gate trend-capture direction by slow EMA: long capture only when price >
# slow EMA (local uptrend), short only when price < slow EMA (downtrend).
# Prevents taking short trends during up-trending markets and vice versa.
# Sweep EMA periods: 50/100/200 candles (50-min / 100-min / 200-min bias).
# ===========================================================================

def _ema_set(name: str, ema_on: bool, slow_period: int = 50,
            size: float = 0.90) -> Dict[str, Any]:
    """Build a param set using s90_l10/confirm=3/force_close baseline + EMA bias."""
    d: Dict[str, Any] = {
        "name": name,
        "use_sl": True,
        "trend_detection": True, "trend_capture": True,
        "trend_force_close_grid": True, "trend_confirm_candles": 3,
        "trend_trailing_stop_pct": 0.04,
        "trend_capture_size_pct": size,
        "trend_lookback_candles": 10,
        "ema_bias_filter": ema_on,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    }
    if ema_on:
        d["ema_slow_period"] = slow_period
    return d


XRP_EMA_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_EMA_CONFIG["param_sets"] = [
    _ema_set("ema_off",    ema_on=False),                  # baseline
    _ema_set("ema_50",     ema_on=True,  slow_period=50),   # ~50-min bias
    _ema_set("ema_100",    ema_on=True,  slow_period=100),  # ~100-min bias
    _ema_set("ema_200",    ema_on=True,  slow_period=200),  # ~3.5h bias (classic)
    {   # grid-only control
        "name": "ema_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]

XRP_EMA_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_EMA_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_EMA_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_EMA_2Y_CONFIG["param_sets"] = [
    _ema_set("2y_ema_off",  ema_on=False),
    _ema_set("2y_ema_50",   ema_on=True, slow_period=50),
    _ema_set("2y_ema_200",  ema_on=True, slow_period=200),
    {   # grid-only 2y baseline
        "name": "2y_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]


# ===========================================================================
# v5 — Bollinger Band squeeze filter
# Two modes:
#   bb_squeeze_gate=True  — skip trend entries when bands are in a squeeze
#                           (low-vol noise → wait for expansion)
#   bb_squeeze_boost=True — multiply position size when bands are just
#                           expanding from a squeeze (cleaner breakouts)
# ===========================================================================

def _bb_set(name: str, gate: bool = False, boost: bool = False,
            boost_mult: float = 1.5, threshold: float = 0.02,
            size: float = 0.90) -> Dict[str, Any]:
    """Build a param set with the s90_l10/confirm baseline + BB squeeze options."""
    return {
        "name": name,
        "use_sl": True,
        "trend_detection": True, "trend_capture": True,
        "trend_force_close_grid": True, "trend_confirm_candles": 3,
        "trend_trailing_stop_pct": 0.04, "trend_capture_size_pct": size,
        "trend_lookback_candles": 10,
        "bb_squeeze_gate":      gate,
        "bb_squeeze_boost":     boost,
        "bb_squeeze_threshold": threshold,
        "bb_squeeze_boost_mult": boost_mult,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    }


XRP_BB_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_BB_CONFIG["param_sets"] = [
    _bb_set("bb_off",           gate=False, boost=False),         # baseline
    _bb_set("bb_gate",          gate=True,  boost=False),         # skip entries in squeeze
    _bb_set("bb_boost_1.5x",    gate=False, boost=True, boost_mult=1.5),   # boost on breakout
    _bb_set("bb_boost_2x",      gate=False, boost=True, boost_mult=2.0),
    _bb_set("bb_gate_boost",    gate=True,  boost=True, boost_mult=1.5),   # gate + boost combo
    {   # grid-only control
        "name": "bb_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]

XRP_BB_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_BB_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_BB_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_BB_2Y_CONFIG["param_sets"] = [
    _bb_set("2y_bb_off",        gate=False, boost=False),
    _bb_set("2y_bb_gate",       gate=True,  boost=False),
    _bb_set("2y_bb_boost_1.5x", gate=False, boost=True, boost_mult=1.5),
    {   # grid-only 2y baseline
        "name": "2y_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]


# ===========================================================================
# v6 — RSI filter
# Prevents chasing exhausted trends:
#   rsi_filter=True  →  long blocked if RSI > 70 (overbought),
#                        short blocked if RSI < 30 (oversold)
#   rsi_momentum=True → also requires RSI > 50 for longs, < 50 for shorts
# ===========================================================================

def _rsi_set(name: str, rsi_on: bool, ob: float = 70.0, os_: float = 30.0,
             momentum: bool = False, size: float = 0.90) -> Dict[str, Any]:
    return {
        "name": name,
        "use_sl": True,
        "trend_detection": True, "trend_capture": True,
        "trend_force_close_grid": True, "trend_confirm_candles": 3,
        "trend_trailing_stop_pct": 0.04, "trend_capture_size_pct": size,
        "trend_lookback_candles": 10,
        "rsi_filter": rsi_on, "rsi_overbought": ob, "rsi_oversold": os_,
        "rsi_momentum": momentum,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    }


XRP_RSI_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_RSI_CONFIG["param_sets"] = [
    _rsi_set("rsi_off",          rsi_on=False),                          # baseline
    _rsi_set("rsi_ob70_os30",    rsi_on=True, ob=70, os_=30),            # classic
    _rsi_set("rsi_ob65_os35",    rsi_on=True, ob=65, os_=35),            # tighter
    _rsi_set("rsi_momentum",     rsi_on=True, ob=70, os_=30, momentum=True),  # + RSI > 50 gate
    {   # grid-only control
        "name": "rsi_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]

XRP_RSI_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_RSI_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_RSI_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_RSI_2Y_CONFIG["param_sets"] = [
    _rsi_set("2y_rsi_off",      rsi_on=False),
    _rsi_set("2y_rsi_ob70",     rsi_on=True, ob=70, os_=30),
    _rsi_set("2y_rsi_momentum", rsi_on=True, ob=70, os_=30, momentum=True),
    {   # grid-only 2y baseline
        "name": "2y_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]


# ===========================================================================
# v7 — Volume confirmation filter
# Only fire trend-capture when candle volume exceeds the rolling average
# by a configurable multiple.  Filters low-conviction false breakouts.
#   vol_filter=True     →  open trend trade only when vol >= N × avg_vol
#   vol_multiplier=1.5  →  require 1.5× above-average volume (default)
#   vol_period=20       →  rolling window for average volume
# ===========================================================================

def _vol_set(name: str, vol_on: bool, mult: float = 1.5, period: int = 20,
             size: float = 0.90) -> Dict[str, Any]:
    return {
        "name": name,
        "use_sl": True,
        "trend_detection": True, "trend_capture": True,
        "trend_force_close_grid": True, "trend_confirm_candles": 3,
        "trend_trailing_stop_pct": 0.04, "trend_capture_size_pct": size,
        "trend_lookback_candles": 10,
        "vol_filter": vol_on, "vol_multiplier": mult, "vol_period": period,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    }


XRP_VOL_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_VOL_CONFIG["param_sets"] = [
    _vol_set("vol_off",       vol_on=False),                # baseline (no vol filter)
    _vol_set("vol_1.5x",      vol_on=True, mult=1.5),       # 1.5× avg volume required
    _vol_set("vol_2.0x",      vol_on=True, mult=2.0),       # 2× avg volume required
    _vol_set("vol_2.5x",      vol_on=True, mult=2.5),       # 2.5× avg volume required
    {   # grid-only control
        "name": "vol_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]

XRP_VOL_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_VOL_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_VOL_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_VOL_2Y_CONFIG["param_sets"] = [
    _vol_set("2y_vol_off",   vol_on=False),
    _vol_set("2y_vol_1.5x",  vol_on=True, mult=1.5),
    _vol_set("2y_vol_2.0x",  vol_on=True, mult=2.0),
    {   # grid-only 2y baseline
        "name": "2y_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]


# ===========================================================================
# v8 — Market structure filter (Higher-High / Lower-Low breakout)
# Only fire trend-capture when price confirms market structure:
#   ms_filter=True + confirmed_up  → open LONG only if price > swing high (HH)
#   ms_filter=True + confirmed_down→ open SHORT only if price < swing low  (LL)
#   ms_lookback=20  → look back N candles for the swing high / low
# Prevents trend trades from firing in the middle of a consolidation range.
# ===========================================================================

def _ms_set(name: str, ms_on: bool, lookback: int = 20,
            size: float = 0.90) -> Dict[str, Any]:
    return {
        "name": name,
        "use_sl": True,
        "trend_detection": True, "trend_capture": True,
        "trend_force_close_grid": True, "trend_confirm_candles": 3,
        "trend_trailing_stop_pct": 0.04, "trend_capture_size_pct": size,
        "trend_lookback_candles": 10,
        "ms_filter": ms_on, "ms_lookback": lookback,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    }


XRP_MS_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_MS_CONFIG["param_sets"] = [
    _ms_set("ms_off",      ms_on=False),           # baseline (no MS filter)
    _ms_set("ms_lb10",     ms_on=True, lookback=10),  # 10-candle swing lookback
    _ms_set("ms_lb20",     ms_on=True, lookback=20),  # 20-candle swing lookback
    _ms_set("ms_lb30",     ms_on=True, lookback=30),  # 30-candle swing lookback
    {   # grid-only control
        "name": "ms_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]

XRP_MS_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_MS_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_MS_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_MS_2Y_CONFIG["param_sets"] = [
    _ms_set("2y_ms_off",    ms_on=False),
    _ms_set("2y_ms_lb10",   ms_on=True, lookback=10),
    _ms_set("2y_ms_lb20",   ms_on=True, lookback=20),
    {   # grid-only 2y baseline
        "name": "2y_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]


# ===========================================================================
# v9 — Fast re-entry + ADX-adaptive trailing stop
#
# Two combined improvements to maximise gains during extended bull runs:
#
#   trend_reentry_fast=True
#       After a trailing stop fires, immediately reset trend_mode so the
#       confirmation counter can re-fire on the next velocity signal.
#       Replaces the 30-candle "quiet cool-down" with a 3-candle re-confirmation
#       window, capturing additional trend legs in multi-day breakouts.
#
#   adx_wide_trail_threshold + adx_wide_trail_pct
#       When ADX ≥ threshold (very strong trend), widen the trailing stop so
#       the position can breathe through minor pullbacks and ride the larger move.
#       Defaults: threshold=40, wide_pct=0.06 (6% vs 4% default).
# ===========================================================================

def _re_set(name: str, reentry: bool = False,
            wide_trail_threshold: float = 9999.0,
            wide_trail_pct: float = 0.06,
            size: float = 0.90) -> Dict[str, Any]:
    """ADX t25/gp35 baseline with optional fast re-entry and wide ADX trail."""
    return {
        "name": name,
        "use_sl": True,
        "trend_detection": True, "trend_capture": True,
        "trend_force_close_grid": True, "trend_confirm_candles": 3,
        "trend_trailing_stop_pct": 0.04, "trend_capture_size_pct": size,
        "trend_lookback_candles": 10,
        "adx_filter": True, "adx_min_trend": 25, "adx_grid_pause": 35,
        "trend_reentry_fast": reentry,
        "adx_wide_trail_threshold": wide_trail_threshold,
        "adx_wide_trail_pct": wide_trail_pct,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    }


XRP_REENTRY_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_REENTRY_CONFIG["param_sets"] = [
    _re_set("re_baseline",           reentry=False),              # ADX t25/gp35, no changes (control)
    _re_set("re_reentry",            reentry=True),               # fast re-entry only
    _re_set("re_wide_trail",         reentry=False,               # wide trail only (ADX≥40 → 6%)
            wide_trail_threshold=40, wide_trail_pct=0.06),
    _re_set("re_wide_trail_8",       reentry=False,               # wide trail only (ADX≥40 → 8%)
            wide_trail_threshold=40, wide_trail_pct=0.08),
    _re_set("re_reentry_wide_trail", reentry=True,                # combined: reentry + wide trail 6%
            wide_trail_threshold=40, wide_trail_pct=0.06),
    _re_set("re_reentry_wider_trail",reentry=True,                # combined: reentry + wide trail 8%
            wide_trail_threshold=40, wide_trail_pct=0.08),
    {   # grid-only control
        "name": "re_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]

XRP_REENTRY_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_REENTRY_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_REENTRY_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_REENTRY_2Y_CONFIG["param_sets"] = [
    _re_set("2y_re_baseline",           reentry=False),
    _re_set("2y_re_reentry",            reentry=True),
    _re_set("2y_re_reentry_wide",       reentry=True,
            wide_trail_threshold=40, wide_trail_pct=0.06),
    _re_set("2y_re_reentry_wider",      reentry=True,
            wide_trail_threshold=40, wide_trail_pct=0.08),
    {   # grid-only 2y baseline
        "name": "2y_re_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]


# ===========================================================================
# v10 — Indicator-based position management
#
# Instead of using indicators as entry gates (which proved inferior to _off
# in v5–v8), we use them to dynamically manage trend position exits and sizing:
#
#   rsi_tight_trail=True  (RSI exhaustion → tighten trail)
#       When RSI ≥ 80 on a long or ≤ 20 on a short, shrink trail from 4% → 2%.
#       Locks in gains faster during blow-off tops / capitulation lows.
#
#   vol_reentry_scale=True  (volume-scaled re-entry size)
#       When re-entering after a trail stop, size at 90% only if volume ≥ 1.5×
#       average; otherwise fall back to 45% to reduce low-conviction risk.
#
#   bb_squeeze_boost=True  (BB breakout → position size boost)
#       When BB expands from squeeze on entry candle, boost to 1.35× normal size.
#       Combined with re_reentry for maximum capture on genuine breakouts.
# ===========================================================================

def _pm_set(name: str, rsi_trail: bool = False, vol_scale: bool = False,
            bb_boost: bool = False, bb_boost_mult: float = 1.35) -> Dict[str, Any]:
    """v10 position-management param set — inherits re_reentry baseline."""
    return {
        "name": name,
        "use_sl": True,
        "trend_detection": True, "trend_capture": True,
        "trend_force_close_grid": True, "trend_confirm_candles": 3,
        "trend_trailing_stop_pct": 0.04, "trend_capture_size_pct": 0.90,
        "trend_lookback_candles": 10,
        "adx_filter": True, "adx_min_trend": 25, "adx_grid_pause": 35,
        "trend_reentry_fast": True,
        # RSI tight trail
        "rsi_tight_trail":     rsi_trail,
        "rsi_tight_trail_ob":  80.0,
        "rsi_tight_trail_os":  20.0,
        "rsi_tight_trail_pct": 0.02,
        # Volume-scaled re-entry
        "vol_reentry_scale":     vol_scale,
        "vol_reentry_high_mult": 1.5,
        "vol_reentry_low_pct":   0.45,
        # BB breakout size boost
        "bb_squeeze_boost":      bb_boost,
        "bb_squeeze_threshold":  0.02,
        "bb_squeeze_boost_mult": bb_boost_mult,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    }


XRP_PM_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_CONFIG["param_sets"] = [
    _pm_set("pm_baseline"),                                               # re_reentry control
    _pm_set("pm_rsi_trail",  rsi_trail=True),                            # RSI → tight trail
    _pm_set("pm_vol_scale",  vol_scale=True),                            # volume → scaled size
    _pm_set("pm_bb_boost",   bb_boost=True),                             # BB → 1.35× boost
    _pm_set("pm_rsi_vol",    rsi_trail=True, vol_scale=True),            # RSI + vol
    _pm_set("pm_combined",   rsi_trail=True, vol_scale=True, bb_boost=True),  # all 3
    {   # grid-only control
        "name": "pm_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]

XRP_PM_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_2Y_CONFIG["param_sets"] = [
    _pm_set("2y_pm_baseline"),
    _pm_set("2y_pm_rsi_trail",  rsi_trail=True),
    _pm_set("2y_pm_vol_scale",  vol_scale=True),
    _pm_set("2y_pm_bb_boost",   bb_boost=True),
    _pm_set("2y_pm_combined",   rsi_trail=True, vol_scale=True, bb_boost=True),
    {   # grid-only 2y baseline
        "name": "2y_pm_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]


# ===========================================================================
# v12 — PM tuning sweep
#
# v10 PM mechanics are coded correctly but had near-zero effect because
# the baseline runs ~14 trend entries in 6 months while the other ~386
# trades are grid trades.  This sweep widens the conditions so PM
# actually fires, and adds Option C which targets GRID order sizing:
#
# (A) RSI tight trail with looser threshold (ob: 80 → 70 or 75)
#     Fires more often during trend positions — tighter trail on overbought.
#     Tests: rsi_ob=70, rsi_ob=75
#
# (B) BB squeeze threshold widened (0.02 → 0.04 or 0.06)
#     More candles count as 'just_broke' → more trend entries get boosted.
#     Tests: threshold=0.04, threshold=0.06
#
# (C) Grid vol scale  (grid_vol_scale=True)
#     Scales effective_order_value by min(1.0, vol_avg/vol_now) each candle.
#     On high-vol candles (trending/crash) → smaller grid orders.
#     On low-vol candles (ranging)         → normal grid orders.
#     This directly targets the 95%+ of trades that are grid trades.
# ===========================================================================

def _pm_v2_set(
    name: str,
    rsi_trail: bool = False,
    rsi_trail_ob: float = 80.0,
    rsi_trail_os: float = 20.0,
    bb_boost: bool = False,
    bb_threshold: float = 0.02,
    grid_vol_scale: bool = False,
    grid_vol_floor: float = 0.35,
    grid_vol_period: int = 20,
    regime_filter: bool = False,
    regime_ema_period: int = 200,
    regime_hysteresis_pct: float = 0.02,
    # v16 XRPPM6 structural extensions
    spacing: float = 0.010,          # symmetric grid spacing for all 4 sides
    max_per_side: int = 0,           # 0 = inherit base config default (3)
    regime_short_cap: int = 0,       # 0 = off; >0 = short cap below regime EMA
    # v17 XRPPM7 extensions
    leverage: float = 0.0,          # 0 = inherit base config (default 1x)
    order_value_pct: float = 0.0,   # 0 = fixed order_value; >0 = % of equity per level
    order_value_min: float = 10.0,  # floor order value when equity-pct mode is active
    regime_short_spacing: float = 0.0,  # 0 = off; >0 = short spacing below regime EMA
    # v18 XRPPM9 extensions — adaptive spacing
    vol_adaptive_spacing: bool = False,  # scale spacing by (ATR/price) / EMA(ATR/price)
    vas_floor: float = 0.008,     # minimum allowed spacing when vol is very low
    vas_ceil: float = 0.020,      # maximum allowed spacing when vol is very high
    vas_period: int = 40,         # EMA window for ATR-fraction baseline
    bull_spacing: float = 0.0,    # two-speed: spacing when price >= regime EMA*(1-hyst)
    bear_spacing: float = 0.0,    # two-speed: spacing when price < regime EMA*(1-hyst)
    # v21 XRPPM11 — regime & parabolic protection layers
    atr_parabolic_mult: float = 0.0,  # Layer 1: block trend entry when ATR > mult × SMA(ATR,20)
    htf_ema_align: bool = False,      # Layer 2: require 1hr-equiv EMA alignment for trend entry
    regime_vote_mode: bool = False,   # Layer 3: 2-of-3 multi-TF regime vote (EMA-175/87/42)
    # v22 XRPPM12 — grid sleep in low-ATR
    grid_sleep_atr_thresh: float = 0.0,  # Layer 4: pause new grid entries when ATR/price < thresh
    # v23 XRPPM13 — adaptive ATR gate (replaces fixed atr_parabolic_mult)
    atr_regime_adaptive: bool = False,   # Approach A: bull→relaxed mult, bear→tight mult
    atr_bull_mult: float = 2.5,          #   multiplier when price >= regime EMA (bull regime)
    atr_bear_mult: float = 1.5,          #   multiplier when price <  regime EMA (bear regime)
    atr_percentile_gate: bool = False,   # Approach B: block when ATR > rolling percentile
    atr_pct_threshold: float = 0.90,     #   percentile level (0.90 = 90th)
    atr_pct_window: int = 100,           #   rolling window for percentile calc
    atr_adx_scale: bool = False,         # Approach C: ADX modulates ATR mult continuously
    atr_adx_base_mult: float = 1.5,      #   mult when ADX=0 (no trend)
    atr_adx_max_mult: float = 3.0,       #   mult when ADX=100 (max trend strength)
    # v24 XRPPM14 — directional velocity gate
    atr_directional: bool = False,       # Approach D: only gate when ATR spike + price falling
    atr_dir_lookback: int = 8,           #   candles to measure price direction (8 = 2 hours on 15min)
    atr_dir_drop_pct: float = 0.02,      #   min price decline to qualify as "falling" (2%)
    atr_acceleration: bool = False,      # Approach E: gate only when ATR is rising vs N candles ago
    atr_accel_lookback: int = 10,        #   candles to compare ATR growth
    atr_cooldown: int = 0,               # Approach F: resume entries N candles after gate fires
    # v26 XRPPM16 — dynamic self-calibrating mechanisms
    vel_atr_mult: float = 0.0,           # Dynamic velocity: scale = max(1, mult × ATR/SMA) (0=use static)
    gate_decay_scale: float = 0.0,       # Dynamic gate decay: after gate clears, suppress for spike_ratio × scale candles (0=off)
    cap_size_atr_scale: bool = False,    # Dynamic position sizing: size = base / atr_ratio, capped [0.30, 0.90]
    cap_size_atr_floor: float = 0.30,    # Minimum trend capture size when ATR is extreme
    cap_size_atr_ceiling: float = 0.90,  # Maximum trend capture size when ATR is normal
    trend_max_loss_atr: float = 0.0,     # Dynamic max loss: close if loss > N × ATR × qty (0=off)
    # v27 XRPPM17 — directional & outcome-based mechanisms
    vel_dir_only: bool = False,          # Only apply velocity scaling when price < EMA-36 (falling)
    vel_accel_only: bool = False,        # Only apply velocity scaling when ATR is actively rising
    eq_curve_filter: bool = False,       # Suppress trend entries when equity < SMA(equity)
    eq_curve_lookback: int = 50,         # Lookback for equity SMA
    consec_loss_max: int = 0,            # Pause trend entries after N consecutive losses (0=off)
    consec_loss_pause: int = 20,         # Candles to pause after consecutive loss threshold hit
    # v28 XRPPM18 — fine-tuning directional velocity
    vel_dir_ema_period: int = 36,        # EMA period for directional check (36=fast, 84=slow, 120/200=smoother)
    # v34 XRPPM24 — 5-regime rotation
    regime_rotation: bool = False,              # Master switch: enable regime-based gate overrides
    regime_adx_choppy_max: float = 25.0,        # ADX below this → CHOPPY regime
    regime_adx_trending_min: float = 35.0,      # ADX above this → TRENDING regime
    regime_atr_crash_mult: float = 3.0,         # ATR/SMA ratio above this → CRASH regime
    regime_atr_dormant_mult: float = 0.5,       # ATR/SMA ratio below this + low ADX → DORMANT regime
    regime_autocorr_period: int = 100,          # Lookback for lag-1 return autocorrelation
    regime_dormant_qty_mult: float = 0.2,       # Position size multiplier in DORMANT (micro-DCA)
    regime_crash_qty_mult: float = 0.5,         # Position size multiplier in CRASH (bounce hunting)
    regime_crash_spacing_mult: float = 0.5,     # Tighter spacing in CRASH (catch bounces)
    regime_dormant_spacing_mult: float = 2.0,   # Wider spacing in DORMANT (slow accumulation)
    regime_trending_grid_off: bool = True,      # Disable grid entries in TRENDING regime
    # v35 XRPPM25 — regime detection quality
    regime_min_dwell_candles: int = 5,              # Hysteresis: min candles before regime switch
    regime_autocorr_choppy_max: float = 0.1,        # Autocorrelation below this → CHOPPY eligible
    # v36 — ATR-adaptive trailing stop
    atr_trail: bool = False,                        # Replace fixed trail with ATR × mult / price
    atr_trail_multiplier: float = 2.0,              # ATR multiplier for dynamic trail
    atr_trail_min: float = 0.015,                   # Floor: 1.5% min trail width
    atr_trail_max: float = 0.12,                    # Cap: 12% max trail width
    # v38 — regime-gated shorts (suppress grid shorts in bull territory)
    regime_short_gate: bool = False,                # Only allow grid shorts below regime EMA or CRASH
    # v37 — surge circuit breaker (upside velocity CB)
    surge_cb: bool = False,                         # Fire when price rises >= rise_pct in lookback
    surge_cb_rise_pct: float = 0.10,                # Min rise to trigger (10%)
    surge_cb_lookback_candles: int = 8,              # Lookback window (8 candles = 2hr on 15min)
    surge_cb_halt_candles: int = 48,                # Halt new short entries for N candles
    # v37 — crash circuit breaker (downside velocity CB)
    crash_cb: bool = False,                         # Fire when price drops >= drop_pct in lookback
    crash_cb_drop_pct: float = 0.10,                # Min drop to trigger (10%)
    crash_cb_lookback_candles: int = 8,              # Lookback window
    crash_cb_halt_candles: int = 48,                # Halt new long entries for N candles
    # v37 — drawdown halt (enable existing mechanism)
    dd_halt: bool = False,                          # Master switch for equity drawdown halt
    dd_halt_max_drawdown: float = 0.30,             # Max allowed drawdown before halt (30%)
    dd_halt_candles: int = 384,                     # Halt duration after DD trigger (~4 days on 15min)
) -> Dict[str, Any]:
    """v12/v13/v15/v16/v17/v18/v23/v24/v26/v27/v28/v29/v34 PM-tuning param set — RSI, BB, vol scale, regime filter, spacing, leverage, vol-adaptive spacing, adaptive gates, directional gates, dynamic velocity/decay/sizing, directional velocity, equity curve, consecutive loss guard, configurable dir EMA, combination sweep, regime rotation."""
    return {
        "name": name,
        "use_sl": True,
        "trend_detection": True, "trend_capture": True,
        "trend_force_close_grid": True, "trend_confirm_candles": 3,
        "trend_trailing_stop_pct": 0.04, "trend_capture_size_pct": 0.90,
        "trend_lookback_candles": 10,
        "adx_filter": True, "adx_min_trend": 25, "adx_grid_pause": 35,
        "trend_reentry_fast": True,
        # Option A — RSI tight trail (configurable thresholds)
        "rsi_tight_trail":     rsi_trail,
        "rsi_tight_trail_ob":  rsi_trail_ob,
        "rsi_tight_trail_os":  rsi_trail_os,
        "rsi_tight_trail_pct": 0.02,
        # Option B — BB breakout boost (configurable squeeze detection threshold)
        "bb_squeeze_boost":      bb_boost,
        "bb_squeeze_threshold":  bb_threshold,
        "bb_squeeze_boost_mult": 1.35,
        # Option C — grid order size scaled inversely with volume
        "grid_vol_scale":  grid_vol_scale,
        "grid_vol_floor":  grid_vol_floor,
        "vol_period":      grid_vol_period,
        # v15 — multi-timeframe regime filter
        "regime_filter":           regime_filter,
        "regime_ema_period":       regime_ema_period,
        "regime_hysteresis_pct":   regime_hysteresis_pct,
        "long_settings":  {"up_spacing": spacing, "down_spacing": spacing},
        "short_settings": {"up_spacing": spacing, "down_spacing": spacing},
        **({"max_positions_per_side": max_per_side} if max_per_side > 0 else {}),
        **({"regime_short_cap":       regime_short_cap} if regime_short_cap > 0 else {}),
        **({"leverage":               leverage}             if leverage > 0 else {}),
        **({"order_value_pct":        order_value_pct}      if order_value_pct > 0 else {}),
        **({"order_value_min":        order_value_min}      if order_value_pct > 0 else {}),
        **({"regime_short_spacing":   regime_short_spacing} if regime_short_spacing > 0 else {}),
        # v18 vol-adaptive / regime-adaptive spacing
        **({"vol_adaptive_spacing": vol_adaptive_spacing,
            "vas_floor": vas_floor, "vas_ceil": vas_ceil, "vas_period": vas_period}
           if vol_adaptive_spacing else {}),
        **({"bull_spacing": bull_spacing, "bear_spacing": bear_spacing}
           if bull_spacing > 0 and bear_spacing > 0 else {}),
        # v21 XRPPM11 — regime & parabolic protection layers
        **({"atr_parabolic_mult": atr_parabolic_mult} if atr_parabolic_mult > 0 else {}),
        **({"htf_ema_align": True}                    if htf_ema_align else {}),
        **({"regime_vote_mode": True}                 if regime_vote_mode else {}),
        # v22 XRPPM12 — grid sleep
        **({"grid_sleep_atr_thresh": grid_sleep_atr_thresh} if grid_sleep_atr_thresh > 0 else {}),
        # v23 XRPPM13 — adaptive ATR gate
        **({"atr_regime_adaptive": True,
            "atr_bull_mult": atr_bull_mult, "atr_bear_mult": atr_bear_mult}
           if atr_regime_adaptive else {}),
        **({"atr_percentile_gate": True,
            "atr_pct_threshold": atr_pct_threshold, "atr_pct_window": atr_pct_window}
           if atr_percentile_gate else {}),
        **({"atr_adx_scale": True,
            "atr_adx_base_mult": atr_adx_base_mult, "atr_adx_max_mult": atr_adx_max_mult}
           if atr_adx_scale else {}),
        # v24 XRPPM14 — directional velocity gate
        **({"atr_directional": True,
            "atr_dir_lookback": atr_dir_lookback, "atr_dir_drop_pct": atr_dir_drop_pct}
           if atr_directional else {}),
        **({"atr_acceleration": True,
            "atr_accel_lookback": atr_accel_lookback}
           if atr_acceleration else {}),
        **({"atr_cooldown": atr_cooldown} if atr_cooldown > 0 else {}),
        # v26 XRPPM16 — dynamic self-calibrating mechanisms
        **({"vel_atr_mult": vel_atr_mult} if vel_atr_mult > 0 else {}),
        **({"gate_decay_scale": gate_decay_scale} if gate_decay_scale > 0 else {}),
        **({"cap_size_atr_scale": True,
            "cap_size_atr_floor": cap_size_atr_floor,
            "cap_size_atr_ceiling": cap_size_atr_ceiling}
           if cap_size_atr_scale else {}),
        **({"trend_max_loss_atr": trend_max_loss_atr} if trend_max_loss_atr > 0 else {}),
        # v27 XRPPM17 — directional & outcome-based mechanisms
        **({"vel_dir_only": True} if vel_dir_only else {}),
        **({"vel_accel_only": True} if vel_accel_only else {}),
        **({"eq_curve_filter": True,
            "eq_curve_lookback": eq_curve_lookback}
           if eq_curve_filter else {}),
        **({"consec_loss_max": consec_loss_max,
            "consec_loss_pause": consec_loss_pause}
           if consec_loss_max > 0 else {}),
        # v28 XRPPM18 — fine-tuning directional velocity
        **({
            "vel_dir_ema_period": vel_dir_ema_period} if vel_dir_ema_period != 36 else {}),
        # v34 XRPPM24 — 5-regime rotation
        **({
            "regime_rotation": True,
            "regime_adx_choppy_max": regime_adx_choppy_max,
            "regime_adx_trending_min": regime_adx_trending_min,
            "regime_atr_crash_mult": regime_atr_crash_mult,
            "regime_atr_dormant_mult": regime_atr_dormant_mult,
            "regime_autocorr_period": regime_autocorr_period,
            "regime_dormant_qty_mult": regime_dormant_qty_mult,
            "regime_crash_qty_mult": regime_crash_qty_mult,
            "regime_crash_spacing_mult": regime_crash_spacing_mult,
            "regime_dormant_spacing_mult": regime_dormant_spacing_mult,
            "regime_trending_grid_off": regime_trending_grid_off,
            # v35 regime detection quality
            "regime_min_dwell_candles": regime_min_dwell_candles,
            "regime_autocorr_choppy_max": regime_autocorr_choppy_max,
        } if regime_rotation else {}),
        # v38 — regime-gated shorts
        **({
            "regime_short_gate": True,
        } if regime_short_gate else {}),
        # v36 — ATR-adaptive trailing stop
        **({"atr_trail": True,
            "atr_trail_multiplier": atr_trail_multiplier,
            "atr_trail_min": atr_trail_min,
            "atr_trail_max": atr_trail_max}
           if atr_trail else {}),
        # v37 — surge circuit breaker
        **({"surge_cb": True,
            "surge_cb_rise_pct": surge_cb_rise_pct,
            "surge_cb_lookback_candles": surge_cb_lookback_candles,
            "surge_cb_halt_candles": surge_cb_halt_candles}
           if surge_cb else {}),
        # v37 — crash circuit breaker
        **({"crash_cb": True,
            "crash_cb_drop_pct": crash_cb_drop_pct,
            "crash_cb_lookback_candles": crash_cb_lookback_candles,
            "crash_cb_halt_candles": crash_cb_halt_candles}
           if crash_cb else {}),
        # v37 — drawdown halt
        **({"dd_halt": True,
            "max_drawdown": dd_halt_max_drawdown,
            "dd_halt_candles": dd_halt_candles}
           if dd_halt else {}),
    }


XRP_PM_V2_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V2_CONFIG["param_sets"] = [
    # ── Baseline (control — matches pm_baseline from v10) ───────────────────
    _pm_v2_set("pm2_baseline"),

    # ── Option A only: RSI trail with looser trigger ─────────────────────────
    _pm_v2_set("pm2_A_rsi70", rsi_trail=True, rsi_trail_ob=70.0, rsi_trail_os=30.0),
    _pm_v2_set("pm2_A_rsi75", rsi_trail=True, rsi_trail_ob=75.0, rsi_trail_os=25.0),

    # ── Option B only: BB squeeze threshold widened ───────────────────────────
    _pm_v2_set("pm2_B_bb04",  bb_boost=True, bb_threshold=0.04),
    _pm_v2_set("pm2_B_bb06",  bb_boost=True, bb_threshold=0.06),

    # ── Option C only: grid order vol scale ───────────────────────────────────
    _pm_v2_set("pm2_C_gridvol", grid_vol_scale=True),

    # ── A + B combined ────────────────────────────────────────────────────────
    _pm_v2_set("pm2_AB_rsi70_bb04",
               rsi_trail=True, rsi_trail_ob=70.0, rsi_trail_os=30.0,
               bb_boost=True, bb_threshold=0.04),

    # ── A + B + C (all three) ─────────────────────────────────────────────────
    _pm_v2_set("pm2_ABC",
               rsi_trail=True, rsi_trail_ob=70.0, rsi_trail_os=30.0,
               bb_boost=True, bb_threshold=0.04,
               grid_vol_scale=True),

    # ── Grid-only (no trend): structural floor ────────────────────────────────
    {
        "name": "pm2_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]

XRP_PM_V2_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V2_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V2_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V2_2Y_CONFIG["param_sets"] = [
    _pm_v2_set("2y_pm2_baseline"),
    _pm_v2_set("2y_pm2_A_rsi70",  rsi_trail=True, rsi_trail_ob=70.0, rsi_trail_os=30.0),
    _pm_v2_set("2y_pm2_A_rsi75",  rsi_trail=True, rsi_trail_ob=75.0, rsi_trail_os=25.0),
    _pm_v2_set("2y_pm2_B_bb04",   bb_boost=True, bb_threshold=0.04),
    _pm_v2_set("2y_pm2_B_bb06",   bb_boost=True, bb_threshold=0.06),
    _pm_v2_set("2y_pm2_C_gridvol", grid_vol_scale=True),
    _pm_v2_set("2y_pm2_AB_rsi70_bb04",
               rsi_trail=True, rsi_trail_ob=70.0, rsi_trail_os=30.0,
               bb_boost=True, bb_threshold=0.04),
    _pm_v2_set("2y_pm2_ABC",
               rsi_trail=True, rsi_trail_ob=70.0, rsi_trail_os=30.0,
               bb_boost=True, bb_threshold=0.04,
               grid_vol_scale=True),
    {
        "name": "2y_pm2_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]


# ===========================================================================
# v13 — C_gridvol parameter sweep
#
# XRPPM2 showed Option C (grid_vol_scale) is the only PM variant that
# consistently beats baseline across both the 6-month OOS and 2-year windows.
# This sweep dials in the two key parameters of that mechanism:
#
# (floor) — minimum fraction of normal order size on extreme-volume candles
#   Default 0.35 → also test 0.20 (more aggressive) and 0.50 (gentler)
#
# (period) — SMA window for computing vol_avg baseline
#   Default 20 → also test 10 (faster signal) and 40 (slower signal)
#
# Grid of variants (floor × period, plus unscaled baseline as control):
#   baseline    — no vol scaling (control)
#   C_f35_p20   — current C_gridvol (exactly replicates XRPPM2 winner)
#   C_f20_p20   — more aggressive floor, same period
#   C_f50_p20   — gentler floor, same period
#   C_f35_p10   — same floor, faster vol signal
#   C_f35_p40   — same floor, slower vol signal
#   C_f20_p10   — most aggressive: lowest floor + fastest signal
# ===========================================================================

XRP_PM_V3_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V3_CONFIG["param_sets"] = [
    _pm_v2_set("pm3_baseline"),
    _pm_v2_set("pm3_C_f35_p20", grid_vol_scale=True, grid_vol_floor=0.35, grid_vol_period=20),
    _pm_v2_set("pm3_C_f20_p20", grid_vol_scale=True, grid_vol_floor=0.20, grid_vol_period=20),
    _pm_v2_set("pm3_C_f50_p20", grid_vol_scale=True, grid_vol_floor=0.50, grid_vol_period=20),
    _pm_v2_set("pm3_C_f35_p10", grid_vol_scale=True, grid_vol_floor=0.35, grid_vol_period=10),
    _pm_v2_set("pm3_C_f35_p40", grid_vol_scale=True, grid_vol_floor=0.35, grid_vol_period=40),
    _pm_v2_set("pm3_C_f20_p10", grid_vol_scale=True, grid_vol_floor=0.20, grid_vol_period=10),
]

XRP_PM_V3_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V3_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V3_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V3_2Y_CONFIG["param_sets"] = [
    _pm_v2_set("2y_pm3_baseline"),
    _pm_v2_set("2y_pm3_C_f35_p20", grid_vol_scale=True, grid_vol_floor=0.35, grid_vol_period=20),
    _pm_v2_set("2y_pm3_C_f20_p20", grid_vol_scale=True, grid_vol_floor=0.20, grid_vol_period=20),
    _pm_v2_set("2y_pm3_C_f50_p20", grid_vol_scale=True, grid_vol_floor=0.50, grid_vol_period=20),
    _pm_v2_set("2y_pm3_C_f35_p10", grid_vol_scale=True, grid_vol_floor=0.35, grid_vol_period=10),
    _pm_v2_set("2y_pm3_C_f35_p40", grid_vol_scale=True, grid_vol_floor=0.35, grid_vol_period=40),
    _pm_v2_set("2y_pm3_C_f20_p10", grid_vol_scale=True, grid_vol_floor=0.20, grid_vol_period=10),
]


# ===========================================================================
# v14 — C_gridvol period fine-sweep
#
# XRPPM3 showed:
#  - floor value is irrelevant (vol_avg/vol_now never drops below ~0.50)
#  - period controls the result: p10=14.97%, p20=15.12%, p40=16.88% (2y)
#  - 2y return still climbing at p40 — peak not yet found
#
# This sweep refines the period axis (floor fixed at 0.35, irrelevant):
#   p20   — XRPPM3 baseline / control
#   p30   — first step toward p40
#   p40   — XRPPM3 winner (replicated as reference)
#   p50   — step beyond p40
#   p60   — longer baseline
#   p80   — much longer baseline
#   p120  — 2-hour rolling average
# ===========================================================================

XRP_PM_V4_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V4_CONFIG["param_sets"] = [
    _pm_v2_set("pm4_baseline"),
    _pm_v2_set("pm4_C_p20",  grid_vol_scale=True, grid_vol_period=20),
    _pm_v2_set("pm4_C_p30",  grid_vol_scale=True, grid_vol_period=30),
    _pm_v2_set("pm4_C_p40",  grid_vol_scale=True, grid_vol_period=40),
    _pm_v2_set("pm4_C_p50",  grid_vol_scale=True, grid_vol_period=50),
    _pm_v2_set("pm4_C_p60",  grid_vol_scale=True, grid_vol_period=60),
    _pm_v2_set("pm4_C_p80",  grid_vol_scale=True, grid_vol_period=80),
    _pm_v2_set("pm4_C_p120", grid_vol_scale=True, grid_vol_period=120),
]

XRP_PM_V4_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V4_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V4_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V4_2Y_CONFIG["param_sets"] = [
    _pm_v2_set("2y_pm4_baseline"),
    _pm_v2_set("2y_pm4_C_p20",  grid_vol_scale=True, grid_vol_period=20),
    _pm_v2_set("2y_pm4_C_p30",  grid_vol_scale=True, grid_vol_period=30),
    _pm_v2_set("2y_pm4_C_p40",  grid_vol_scale=True, grid_vol_period=40),
    _pm_v2_set("2y_pm4_C_p50",  grid_vol_scale=True, grid_vol_period=50),
    _pm_v2_set("2y_pm4_C_p60",  grid_vol_scale=True, grid_vol_period=60),
    _pm_v2_set("2y_pm4_C_p80",  grid_vol_scale=True, grid_vol_period=80),
    _pm_v2_set("2y_pm4_C_p120", grid_vol_scale=True, grid_vol_period=120),
]


# ===========================================================================
# v15 — Multi-timeframe regime filter sweep
#
# Build on the XRPPM4 winner (C_gridvol, vol_period=40).
# Add a 15-minute EMA regime filter that suppresses new long grid entries
# when price < EMA(15min close, N) × (1 - hysteresis).
# During bearish regimes the strategy stops buying the dip, avoiding the
# slow-grind losses that the ADX filter cannot catch.
#
# Variants:
#   pm5_baseline      — control: C_f35_p40, NO regime filter
#   pm5_ema100        — EMA(100) on 15min ≈ 25h lookback
#   pm5_ema200        — EMA(200) on 15min ≈ 50h lookback (expected winner)
#   pm5_ema400        — EMA(400) on 15min ≈ 100h lookback
#
# All regime-filter variants use hysteresis=2% to avoid rapid toggling.
# ===========================================================================
_V5_BASE = dict(grid_vol_scale=True, grid_vol_floor=0.35, grid_vol_period=50)  # XRPPM4 winner

XRP_PM_V5_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V5_CONFIG["param_sets"] = [
    _pm_v2_set("pm5_baseline",           **_V5_BASE),
    _pm_v2_set("pm5_ema100",             **_V5_BASE, regime_filter=True, regime_ema_period=100),
    _pm_v2_set("pm5_ema200",             **_V5_BASE, regime_filter=True, regime_ema_period=200),
    _pm_v2_set("pm5_ema400",             **_V5_BASE, regime_filter=True, regime_ema_period=400),
]

XRP_PM_V5_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V5_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V5_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V5_2Y_CONFIG["param_sets"] = [
    _pm_v2_set("2y_pm5_baseline",        **_V5_BASE),
    _pm_v2_set("2y_pm5_ema100",          **_V5_BASE, regime_filter=True, regime_ema_period=100),
    _pm_v2_set("2y_pm5_ema200",          **_V5_BASE, regime_filter=True, regime_ema_period=200),
    _pm_v2_set("2y_pm5_ema400",          **_V5_BASE, regime_filter=True, regime_ema_period=400),
]


# ===========================================================================
# v16 — XRPPM6: Comprehensive grid and regime sweep
#
# Build on XRPPM5 winner (C_gridvol, vol_period=50, 15min EMA200, h=2%).
# Six sweep groups to simultaneously address:
#
#   (A) Hysteresis — width of the band below EMA before halting longs.
#       h2% is the pm5 default; scan h0..h5 to find the optimal deadband.
#
#   (B) EMA period fine-tune around the ema200 winner (150/175/225/250).
#       pm5 showed 200 > 100 and 200 > 400 on 2y; find the true optimum.
#
#   (C) vol_period compound — p40 was the XRPPM3 2y winner.
#       Does p40 + ema200 beat the p50 + ema200 combo on 2y?
#
#   (D) Grid spacing: 0.5% / 0.7% / 1.0% (control) / 1.5%.
#       Diagnoses how much grid income is locked in spacing friction.
#       Tighter → more fills, smaller profit, tighter SLs.
#       Wider  → fewer fills, larger profit, fewer SLs.
#
#   (E) Grid depth (max_positions_per_side): 3 (control) / 4 / 5.
#       At 3 × $100 × 2 sides = $600 deployed, $400 sits idle.
#       Adding depth puts that capital to work.
#
#   (F) Short boost in bearish regime: regime_short_cap = 4 or 5.
#       Below EMA, longs are halted; shorts are the profitable side.
#       Deploying extra short levels harvests more of the downtrend.
# ===========================================================================
_V6_BASE = dict(
    grid_vol_scale=True, grid_vol_floor=0.35, grid_vol_period=50,  # XRPPM4/5 winner
    regime_filter=True, regime_ema_period=200, regime_hysteresis_pct=0.02,   # XRPPM5 winner
)

XRP_PM_V6_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V6_CONFIG["param_sets"] = [
    # ── Baseline: matches pm5_ema200 (18.93% 2y, 0.18% max_dd) ───────────────
    _pm_v2_set("pm6_baseline",  **_V6_BASE),

    # ── Group A — Hysteresis sweep (EMA200, p50) ──────────────────────────────
    _pm_v2_set("pm6_h0",        **{**_V6_BASE, "regime_hysteresis_pct": 0.00}),  # no band
    _pm_v2_set("pm6_h1",        **{**_V6_BASE, "regime_hysteresis_pct": 0.01}),  # 1% below EMA
    _pm_v2_set("pm6_h3",        **{**_V6_BASE, "regime_hysteresis_pct": 0.03}),  # 3% below EMA
    _pm_v2_set("pm6_h5",        **{**_V6_BASE, "regime_hysteresis_pct": 0.05}),  # 5% below EMA

    # ── Group B — EMA period fine-tune (h=2%, p50) ────────────────────────────
    _pm_v2_set("pm6_e150",      **{**_V6_BASE, "regime_ema_period": 150}),  # 37.5h
    _pm_v2_set("pm6_e175",      **{**_V6_BASE, "regime_ema_period": 175}),  # 43.75h
    _pm_v2_set("pm6_e225",      **{**_V6_BASE, "regime_ema_period": 225}),  # 56.25h
    _pm_v2_set("pm6_e250",      **{**_V6_BASE, "regime_ema_period": 250}),  # 62.5h

    # ── Group C — vol_period compound (XRPPM3 2y: p40 vs p50 + ema200) ───────
    _pm_v2_set("pm6_p40",       **{**_V6_BASE, "grid_vol_period": 40}),

    # ── Group D — Grid spacing (harvesting diagnostic, ema200 h2 p50) ─────────
    _pm_v2_set("pm6_sp05",      **_V6_BASE, spacing=0.005),  # 0.5% — double fill rate
    _pm_v2_set("pm6_sp07",      **_V6_BASE, spacing=0.007),  # 0.7% — intermediate
    _pm_v2_set("pm6_sp15",      **_V6_BASE, spacing=0.015),  # 1.5% — wider profit/fill

    # ── Group E — Grid depth (capital utilisation, ema200 h2 p50) ────────────
    _pm_v2_set("pm6_d4",        **_V6_BASE, max_per_side=4),  # $400/side max deployed
    _pm_v2_set("pm6_d5",        **_V6_BASE, max_per_side=5),  # $500/side max deployed

    # ── Group F — Short boost below EMA (deploy idle capital in downtrend) ────
    _pm_v2_set("pm6_sb4",       **_V6_BASE, regime_short_cap=4),  # 4 short levels below EMA
    _pm_v2_set("pm6_sb5",       **_V6_BASE, regime_short_cap=5),  # 5 short levels below EMA
]

XRP_PM_V6_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V6_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V6_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V6_2Y_CONFIG["param_sets"] = [
    _pm_v2_set("2y_pm6_baseline",  **_V6_BASE),

    _pm_v2_set("2y_pm6_h0",        **{**_V6_BASE, "regime_hysteresis_pct": 0.00}),
    _pm_v2_set("2y_pm6_h1",        **{**_V6_BASE, "regime_hysteresis_pct": 0.01}),
    _pm_v2_set("2y_pm6_h3",        **{**_V6_BASE, "regime_hysteresis_pct": 0.03}),
    _pm_v2_set("2y_pm6_h5",        **{**_V6_BASE, "regime_hysteresis_pct": 0.05}),

    _pm_v2_set("2y_pm6_e150",      **{**_V6_BASE, "regime_ema_period": 150}),
    _pm_v2_set("2y_pm6_e175",      **{**_V6_BASE, "regime_ema_period": 175}),
    _pm_v2_set("2y_pm6_e225",      **{**_V6_BASE, "regime_ema_period": 225}),
    _pm_v2_set("2y_pm6_e250",      **{**_V6_BASE, "regime_ema_period": 250}),

    _pm_v2_set("2y_pm6_p40",       **{**_V6_BASE, "grid_vol_period": 40}),

    _pm_v2_set("2y_pm6_sp05",      **_V6_BASE, spacing=0.005),
    _pm_v2_set("2y_pm6_sp07",      **_V6_BASE, spacing=0.007),
    _pm_v2_set("2y_pm6_sp15",      **_V6_BASE, spacing=0.015),

    _pm_v2_set("2y_pm6_d4",        **_V6_BASE, max_per_side=4),
    _pm_v2_set("2y_pm6_d5",        **_V6_BASE, max_per_side=5),

    _pm_v2_set("2y_pm6_sb4",       **_V6_BASE, regime_short_cap=4),
    _pm_v2_set("2y_pm6_sb5",       **_V6_BASE, regime_short_cap=5),
]


# ===========================================================================
# v17 — XRPPM7: Leverage, equity-proportional sizing, asymmetric short spacing
#
# Build on the XRPPM6 winner (TBD — update _V7_BASE once run #88 completes).
# Until then _V7_BASE inherits _V6_BASE (pm6_baseline = pm5_ema200 stack).
#
# Three sweep groups:
#
#   (L) Leverage: 1x (ctrl) / 1.5x / 2x / 3x
#       Direct P&L multiplier — returns and drawdown scale proportionally.
#       With current max_dd of 0.18%, even 3x stays well below 1% drawdown.
#
#   (P) Equity-proportional order sizing (order_value_pct)
#       Replace fixed $100/level with (equity × pct).
#       Returns compound as equity grows; position size shrinks after losses.
#       pct=0.10 → $100 at start (same absolute as baseline, but compounds).
#       pct=0.12 → $120 at start (+20% from day one, compounds faster).
#       pct=0.15 → $150 at start (+50% initial aggression + compounding).
#
#   (LP) Leverage + equity-pct combined (best of L and P stacked):
#       l2_eqp10 → 2× leverage, 10% equity-pct (effective $200/level at start)
#       l15_eqp10 → 1.5× leverage, 10% equity-pct (effective $150/level at start)
#
#   (S) Asymmetric short spacing below regime EMA:
#       When longs are halted (bearish regime), switch to tighter short grid.
#       Tighter SLs → less locked capital per stopped position → faster cycling.
#       as07 → 0.7% short spacing below EMA (vs 1.0% normal)
#       as05 → 0.5% short spacing below EMA
#
# XRPPM6 (run #88) confirmed winners applied to _V7_BASE:
#   spacing=0.015  — sp15 won 2y by +11.61pp (18.93% → 30.54%, dd=0.55%)
#   ema_period=175 — e175 won 2y by +1.39pp (18.93% → 20.32%, dd=0.15%)
#   hysteresis=0.02 kept (h0/h1 gain +1-2pp but double dd; sweep on sp15 base
#                         is a natural XRPPM8 target if leverage headroom warrants)
#   depth/short-boost: no impact (max_unrealized_loss_per_side=$30 CB prevents
#                       accumulation beyond 3 positions; raise CB to test)
# ===========================================================================
# _V7_BASE: XRPPM6 winners — sp15 + e175 on top of V6_BASE fundamentals
_V7_BASE = dict(
    grid_vol_scale=True, grid_vol_floor=0.35, grid_vol_period=50,  # XRPPM4 winner
    regime_filter=True, regime_ema_period=175, regime_hysteresis_pct=0.02,  # XRPPM6: e175 winner
    spacing=0.015,  # XRPPM6: sp15 winner (+11.61pp 2y: 18.93% → 30.54%)
)

XRP_PM_V7_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V7_CONFIG["param_sets"] = [
    # ── Baseline (control = V7_BASE: sp15 + ema175 + h2%) ────────────────────
    _pm_v2_set("pm7_baseline",    **_V7_BASE),

    # ── Group L — Leverage sweep ───────────────────────────────────────────────
    _pm_v2_set("pm7_l15",         **_V7_BASE, leverage=1.5),   # +50% returns, +50% dd
    _pm_v2_set("pm7_l2",          **_V7_BASE, leverage=2.0),   # ×2 returns,  ×2 dd
    _pm_v2_set("pm7_l3",          **_V7_BASE, leverage=3.0),   # ×3 returns,  ×3 dd

    # ── Group P — Equity-proportional order sizing ────────────────────────────
    _pm_v2_set("pm7_eqp10",       **_V7_BASE, order_value_pct=0.10),  # = $100 at start
    _pm_v2_set("pm7_eqp12",       **_V7_BASE, order_value_pct=0.12),  # = $120 at start
    _pm_v2_set("pm7_eqp15",       **_V7_BASE, order_value_pct=0.15),  # = $150 at start

    # ── Group LP — Leverage + equity-pct combined ─────────────────────────────
    _pm_v2_set("pm7_l2_eqp10",    **_V7_BASE, leverage=2.0, order_value_pct=0.10),
    _pm_v2_set("pm7_l15_eqp10",   **_V7_BASE, leverage=1.5, order_value_pct=0.10),

    # ── Group S — Asymmetric short spacing below regime EMA ───────────────────
    _pm_v2_set("pm7_as07",        **_V7_BASE, regime_short_spacing=0.007),  # 0.7% below EMA
    _pm_v2_set("pm7_as05",        **_V7_BASE, regime_short_spacing=0.005),  # 0.5% below EMA
]

XRP_PM_V7_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V7_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V7_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V7_2Y_CONFIG["param_sets"] = [
    _pm_v2_set("2y_pm7_baseline",   **_V7_BASE),
    _pm_v2_set("2y_pm7_l15",        **_V7_BASE, leverage=1.5),
    _pm_v2_set("2y_pm7_l2",         **_V7_BASE, leverage=2.0),
    _pm_v2_set("2y_pm7_l3",         **_V7_BASE, leverage=3.0),
    _pm_v2_set("2y_pm7_eqp10",      **_V7_BASE, order_value_pct=0.10),
    _pm_v2_set("2y_pm7_eqp12",      **_V7_BASE, order_value_pct=0.12),
    _pm_v2_set("2y_pm7_eqp15",      **_V7_BASE, order_value_pct=0.15),
    _pm_v2_set("2y_pm7_l2_eqp10",   **_V7_BASE, leverage=2.0, order_value_pct=0.10),
    _pm_v2_set("2y_pm7_l15_eqp10",  **_V7_BASE, leverage=1.5, order_value_pct=0.10),
    _pm_v2_set("2y_pm7_as07",       **_V7_BASE, regime_short_spacing=0.007),
    _pm_v2_set("2y_pm7_as05",       **_V7_BASE, regime_short_spacing=0.005),
]


# ===========================================================================
# v18 — XRPPM8: Hysteresis + spacing fine-tune on l2 production base
#
# XRPPM7 confirmed: leverage=2.0 is the production target (54.23% 2y, 0.99% dd).
# eqp sizing and asymmetric short spacing both degraded — abandoned.
#
# Two open questions answered here:
#
#   (H) Does h0 stack on l2?
#       pm6 showed h0 +2pp (18.93% → 21.00%) with doubled dd (0.18% → 0.36%).
#       With l2 base (0.99% dd), there's headroom.  Is the +2pp real on sp15?
#       h0=0.0 → no hysteresis band (fires on every EMA cross)
#       h1=0.01 → 1% band (midpoint between h0 and h2% current)
#
#   (S) Can spacing between sp10 and sp15 reclaim trades without 2y collapse?
#       sp15 (78 trades/6m, 30.54% 2y) vs sp10 (294 trades/6m, 18.93% 2y).
#       sp10 with l1 → -0.82% 2y (too many low-quality fills).
#       Want: sp12/sp13 with l2 — more trades than 78/6m, 2y stays stable.
#       sp10 tested too to see the full l2 curve.
#
#   (C) Hysteresis × spacing interaction:
#       h0_sp12 — does h0 compound with tighter spacing?
#
# _V8_BASE: pm7_l2 locked in as production → leverage=2.0 baked
_V8_BASE = dict(**_V7_BASE, leverage=2.0)  # 54.23% 2y, max_dd 0.99%

XRP_PM_V8_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V8_CONFIG["param_sets"] = [
    # ── Baseline (control = pm7_l2 reproduced) ────────────────────────────────
    _pm_v2_set("pm8_baseline",    **_V8_BASE),

    # ── Group H — Hysteresis sweep on l2 base ─────────────────────────────────
    _pm_v2_set("pm8_h0",          **{**_V8_BASE, "regime_hysteresis_pct": 0.00}),  # no band
    _pm_v2_set("pm8_h1",          **{**_V8_BASE, "regime_hysteresis_pct": 0.01}),  # 1% band

    # ── Group S — Spacing fill-in: sp10→sp15 curve with l2 ───────────────────
    _pm_v2_set("pm8_sp10",        **{**_V8_BASE, "spacing": 0.010}),  # high-freq baseline ref
    _pm_v2_set("pm8_sp12",        **{**_V8_BASE, "spacing": 0.012}),  # midpoint
    _pm_v2_set("pm8_sp13",        **{**_V8_BASE, "spacing": 0.013}),  # tighter midpoint

    # ── Group C — Cross: h0 × sp12 interaction ────────────────────────────────
    _pm_v2_set("pm8_h0_sp12",     **{**_V8_BASE, "regime_hysteresis_pct": 0.00, "spacing": 0.012}),
]

XRP_PM_V8_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V8_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V8_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V8_2Y_CONFIG["param_sets"] = [
    _pm_v2_set("2y_pm8_baseline",  **_V8_BASE),
    _pm_v2_set("2y_pm8_h0",        **{**_V8_BASE, "regime_hysteresis_pct": 0.00}),
    _pm_v2_set("2y_pm8_h1",        **{**_V8_BASE, "regime_hysteresis_pct": 0.01}),
    _pm_v2_set("2y_pm8_sp10",      **{**_V8_BASE, "spacing": 0.010}),
    _pm_v2_set("2y_pm8_sp12",      **{**_V8_BASE, "spacing": 0.012}),
    _pm_v2_set("2y_pm8_sp13",      **{**_V8_BASE, "spacing": 0.013}),
    _pm_v2_set("2y_pm8_h0_sp12",   **{**_V8_BASE, "regime_hysteresis_pct": 0.00, "spacing": 0.012}),
]


# ===========================================================================
# v19 — XRPPM9: Adaptive spacing — capture small oscillations without SL cascades
#
# Core insight from run #90 loss audit:
#   - The ONLY loss mechanism that matters is grid SL cycling.
#   - No crash CB, DD halt, or per-side circuit breakers ever fire.
#   - sp15 (78 trades/6m, 30.54% 2y) leaves sub-1.5% oscillations uncaptured.
#   - sp07 (1004 trades/6m) captures them profitably in bull markets but
#     collapses in 2y (-0.82%) because sustained downtrends cause SL cascades
#     at tight spacing.
#
# Goal: tight spacing in calm/bull markets, wide spacing in volatile/bear markets.
#
# Two mechanisms tested:
#
#   (V) Vol-Adaptive Spacing (VAS)
#       eff_spacing = base_spacing × (ATR/price) / EMA(ATR/price, period=40)
#       Clamp [vas_floor, vas_ceil].
#       Calm market (ATR low vs baseline) → spacing < 0.015 → captures sp07-like fills.
#       Volatile/trending (ATR high vs baseline) → spacing > 0.015 → SL protection.
#
#       Variants:
#         vas       → floor=0.010, ceil=0.020  (moderate range around sp15)
#         vas_tight → floor=0.007, ceil=0.020  (more aggressive tightening in calm)
#         vas_wide  → floor=0.010, ceil=0.025  (wider ceiling in spike events)
#
#   (B) Bull-Tight Bear-Wide (BTBW) — two-speed regime-aware grid
#       When price >= regime_EMA*(1-hyst): use bull_spacing (tighter)
#       When price <  regime_EMA*(1-hyst): use bear_spacing (=sp15 protection)
#       Simple, interpretable, zero per-candle computation overhead.
#
#       Variants:
#         btbw      → bull=0.010, bear=0.015  (sp10 in bull, sp15 in bear)
#         btbw_tight → bull=0.008, bear=0.015  (tighter bull grid)
#         btbw_xtight → bull=0.007, bear=0.015  (most aggressive)
#
# _V9_BASE: locked to _V8_BASE (h0=0.02 baseline — XRPPM8 winner was h0=0.00; see _V10_BASE)
_V9_BASE = dict(**_V8_BASE)  # = _V7_BASE + leverage=2.0, h=0.02

XRP_PM_V9_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V9_CONFIG["param_sets"] = [
    # ── Baseline (control = pm8_baseline / pm7_l2 reproduced) ─────────────────
    _pm_v2_set("pm9_baseline",     **_V9_BASE),

    # ── Group V — Vol-Adaptive Spacing ────────────────────────────────────────
    # base=0.015, scales ±ATR ratio; different floors/ceilings
    _pm_v2_set("pm9_vas",          **_V9_BASE, vol_adaptive_spacing=True,
               vas_floor=0.010, vas_ceil=0.020, vas_period=40),
    _pm_v2_set("pm9_vas_tight",    **_V9_BASE, vol_adaptive_spacing=True,
               vas_floor=0.007, vas_ceil=0.020, vas_period=40),
    _pm_v2_set("pm9_vas_wide",     **_V9_BASE, vol_adaptive_spacing=True,
               vas_floor=0.010, vas_ceil=0.025, vas_period=40),

    # ── Group B — Bull-Tight Bear-Wide (two-speed regime grid) ────────────────
    _pm_v2_set("pm9_btbw",         **_V9_BASE, bull_spacing=0.010, bear_spacing=0.015),
    _pm_v2_set("pm9_btbw_tight",   **_V9_BASE, bull_spacing=0.008, bear_spacing=0.015),
    _pm_v2_set("pm9_btbw_xtight",  **_V9_BASE, bull_spacing=0.007, bear_spacing=0.015),
]

XRP_PM_V9_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V9_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V9_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V9_2Y_CONFIG["param_sets"] = [
    _pm_v2_set("2y_pm9_baseline",    **_V9_BASE),
    _pm_v2_set("2y_pm9_vas",         **_V9_BASE, vol_adaptive_spacing=True,
               vas_floor=0.010, vas_ceil=0.020, vas_period=40),
    _pm_v2_set("2y_pm9_vas_tight",   **_V9_BASE, vol_adaptive_spacing=True,
               vas_floor=0.007, vas_ceil=0.020, vas_period=40),
    _pm_v2_set("2y_pm9_vas_wide",    **_V9_BASE, vol_adaptive_spacing=True,
               vas_floor=0.010, vas_ceil=0.025, vas_period=40),
    _pm_v2_set("2y_pm9_btbw",        **_V9_BASE, bull_spacing=0.010, bear_spacing=0.015),
    _pm_v2_set("2y_pm9_btbw_tight",  **_V9_BASE, bull_spacing=0.008, bear_spacing=0.015),
    _pm_v2_set("2y_pm9_btbw_xtight", **_V9_BASE, bull_spacing=0.007, bear_spacing=0.015),
]


# ===========================================================================
# v20 — XRPPM10: h0 locked in as new base.  Test BTBW variants on h0.
#
# XRPPM8 result (run #96):
#   h0 = regime_hysteresis_pct=0.00 wins 2y walk-forward (+5.44% over h2 baseline)
#   h0:  59.67% 2y, 1.02% dd, 1066 trades  ← NEW BASE
#   h2:  54.23% 2y, 0.99% dd, 1126 trades
#
# XRPPM9 result (run #96):
#   BTBW-tight wins 2y (+5.06%) but regresses -3% on 6m OOS.
#   Need quarterly breakdown to confirm where the gain concentrates.
#
# _V10_BASE: _V8_BASE but with h0 (hysteresis=0.00)
# All other params unchanged: ema=175, sp=0.015, lev=2.0
#
# This run tests:
#   (B) BTBW variants on h0 base — 6m OOS + 2y walk-forward
#   (M) Mid-year window Aug 2024 → Aug 2025 to isolate bull-2024 vs bear-ish-2025
#       split that drove btbw_tight's 2y gain.
# ===========================================================================

_V10_BASE = {**_V8_BASE, "regime_hysteresis_pct": 0.00}
# = regime_ema_period=175, hysteresis=0.00, spacing=0.015, leverage=2.0
# Expected baseline ~59.67% 2y (mirrors pm8_h0 from run #96)

XRP_PM_V10_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)   # 6m OOS Aug 2025 → Feb 2026
XRP_PM_V10_CONFIG["param_sets"] = [
    _pm_v2_set("pm10_baseline",     **_V10_BASE),
    _pm_v2_set("pm10_btbw",         **_V10_BASE, bull_spacing=0.010, bear_spacing=0.015),
    _pm_v2_set("pm10_btbw_tight",   **_V10_BASE, bull_spacing=0.008, bear_spacing=0.015),
    _pm_v2_set("pm10_btbw_xtight",  **_V10_BASE, bull_spacing=0.007, bear_spacing=0.015),
]

XRP_PM_V10_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V10_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V10_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V10_2Y_CONFIG["param_sets"] = [
    _pm_v2_set("2y_pm10_baseline",   **_V10_BASE),
    _pm_v2_set("2y_pm10_btbw",       **_V10_BASE, bull_spacing=0.010, bear_spacing=0.015),
    _pm_v2_set("2y_pm10_btbw_tight", **_V10_BASE, bull_spacing=0.008, bear_spacing=0.015),
]

# Mid-year window: isolate whether btbw_tight gains concentrate in bull-2024 or persist
# into Q1-2025 bear / Q2-2025 recovery.  Aug 2024 → Aug 2025 straddles the XRP parabolic
# run (Oct–Dec 2024) and the 2025 correction.
XRP_PM_V10_1Y_MID_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V10_1Y_MID_CONFIG["start_date"] = datetime(2024, 8, 1)
XRP_PM_V10_1Y_MID_CONFIG["end_date"]   = datetime(2025, 8, 1)
XRP_PM_V10_1Y_MID_CONFIG["param_sets"] = [
    _pm_v2_set("mid_pm10_baseline",   **_V10_BASE),
    _pm_v2_set("mid_pm10_btbw",       **_V10_BASE, bull_spacing=0.010, bear_spacing=0.015),
    _pm_v2_set("mid_pm10_btbw_tight", **_V10_BASE, bull_spacing=0.008, bear_spacing=0.015),
]


# ===========================================================================
# v21 — XRPPM11: 3-layer 80% fix.  ATR gate (L1) + HTF align (L2) + regime vote (L3)
#
# Loss-audit of 2y weekly breakdown (run #108) found 80% of losses concentrate in:
#   1. Parabolic pumps  (Nov–Dec 2024 XRP +300%) — trend entries fire into blow-offs
#   2. Counter-trend    (Jan–Feb 2025)            — HTF bias ignored
#   3. False bear halts (short flat chops)         — single-EMA too sensitive
#
# Layer 1: ATR parabolic gate — blocks trend entries when ATR > mult × SMA(ATR,20)
# Layer 2: HTF EMA alignment  — requires 1hr-equiv EMA agreement for trend entry
# Layer 3: Multi-TF regime vote — 2-of-3 EMAs (175/87/42) must agree before halting longs
#
# 10 strategies: baseline + L1@1.5/2.0/2.5 + L2 + L3 + L1L2 + L1L3 + L2L3 + full
# 3 windows: 6m OOS + 2y walk-forward + mid-year
# ===========================================================================

_V11_BASE = {**_V10_BASE}  # h0, spacing=1.5%, leverage=2.0

_PM11_SETS = [
    _pm_v2_set("pm11_baseline",   **_V11_BASE),
    _pm_v2_set("pm11_L1_15",      **_V11_BASE, atr_parabolic_mult=1.5),
    _pm_v2_set("pm11_L1_20",      **_V11_BASE, atr_parabolic_mult=2.0),
    _pm_v2_set("pm11_L1_25",      **_V11_BASE, atr_parabolic_mult=2.5),
    _pm_v2_set("pm11_L2",         **_V11_BASE, htf_ema_align=True),
    _pm_v2_set("pm11_L3",         **_V11_BASE, regime_vote_mode=True),
    _pm_v2_set("pm11_L1_L2",      **_V11_BASE, atr_parabolic_mult=2.0, htf_ema_align=True),
    _pm_v2_set("pm11_L1_L3",      **_V11_BASE, atr_parabolic_mult=2.0, regime_vote_mode=True),
    _pm_v2_set("pm11_L2_L3",      **_V11_BASE, htf_ema_align=True, regime_vote_mode=True),
    _pm_v2_set("pm11_full",       **_V11_BASE, atr_parabolic_mult=2.0, htf_ema_align=True, regime_vote_mode=True),
]

XRP_PM_V11_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)   # 6m OOS Aug 2025 → Feb 2026
XRP_PM_V11_CONFIG["param_sets"] = _PM11_SETS

XRP_PM_V11_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V11_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V11_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V11_2Y_CONFIG["param_sets"] = _PM11_SETS

XRP_PM_V11_1Y_MID_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V11_1Y_MID_CONFIG["start_date"] = datetime(2024, 8, 1)
XRP_PM_V11_1Y_MID_CONFIG["end_date"]   = datetime(2025, 8, 1)
XRP_PM_V11_1Y_MID_CONFIG["param_sets"] = _PM11_SETS


# ===========================================================================
# v22 — XRPPM12: v21 full + Layer 4 grid sleep.  Stacked on top of the 3-layer winner.
#
# Layer 4: Grid sleep — pauses ALL new grid entries when ATR/price < threshold.
# In flat, low-ATR markets the grid piles into positions that never complete
# round trips.  Grid sleep keeps the bot idle until volatility returns.
#
# 4 strategies: baseline (pm11_full, no sleep) + sleep_02 / sleep_03 / sleep_04
# 3 windows: 6m OOS + 2y walk-forward + mid-year
# ===========================================================================

_V12_BASE = {**_V11_BASE, "atr_parabolic_mult": 2.0, "htf_ema_align": True, "regime_vote_mode": True}

_PM12_SETS = [
    _pm_v2_set("pm12_baseline",  **_V12_BASE),
    _pm_v2_set("pm12_sleep_02",  **_V12_BASE, grid_sleep_atr_thresh=0.002),
    _pm_v2_set("pm12_sleep_03",  **_V12_BASE, grid_sleep_atr_thresh=0.003),
    _pm_v2_set("pm12_sleep_04",  **_V12_BASE, grid_sleep_atr_thresh=0.004),
]

XRP_PM_V12_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)   # 6m OOS Aug 2025 → Feb 2026
XRP_PM_V12_CONFIG["param_sets"] = _PM12_SETS

XRP_PM_V12_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V12_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V12_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V12_2Y_CONFIG["param_sets"] = _PM12_SETS

XRP_PM_V12_1Y_MID_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V12_1Y_MID_CONFIG["start_date"] = datetime(2024, 8, 1)
XRP_PM_V12_1Y_MID_CONFIG["end_date"]   = datetime(2025, 8, 1)
XRP_PM_V12_1Y_MID_CONFIG["param_sets"] = _PM12_SETS


# ===========================================================================
# v23 — XRPPM13: Adaptive ATR gate.  Replaces fixed L1 multiplier with
# regime-aware, percentile-based, and ADX-scaled approaches.
#
# PM11 showed L1 at 1.5× is a volatility smoother:
#   Bull (6m OOS): 106.77% → 29.41%  (−77pp — too aggressive in bull)
#   Bear (mid-yr): −6.92% → +24.57%  (+31pp — excellent protection)
#   2y walk-fwd:   59.67% → 60.49%   (+0.82pp — washes out)
#
# Goal: achieve L1's bear protection while preserving bull performance
# by dynamically adjusting the ATR gate threshold.
#
# Approach A (atr_regime_adaptive):
#   Bull regime (price ≥ regime EMA) → high mult (relaxed, rarely blocks)
#   Bear regime (price < regime EMA) → low mult (tight, blocks parabolic spikes)
#
# Approach B (atr_percentile_gate):
#   Block only when ATR > rolling Nth percentile of recent ATR window.
#   Naturally adapts: in a steady bull, percentile rises → fewer blocks.
#
# Approach C (atr_adx_scale):
#   ADX strength modulates the multiplier continuously.
#   Strong trend (high ADX) → higher mult → more permissive.
#   Weak/ranging (low ADX) → lower mult → more protective.
#
# Hybrid (A+B): regime-adaptive mult + percentile gate (OR logic).
#
# 8 strategies × 3 windows (6m OOS, 2y walk-forward, mid-year)
# ===========================================================================

_V13_BASE = {**_V10_BASE}  # h0, spacing=1.5%, leverage=2.0 (same as V11)

_PM13_SETS = [
    # Control — no adaptive gate, no fixed gate (matches pm11_baseline)
    _pm_v2_set("pm13_baseline",         **_V13_BASE),

    # Approach A — Regime-adaptive ATR gate
    _pm_v2_set("pm13_regime_b25_r15",   **_V13_BASE,
               atr_regime_adaptive=True, atr_bull_mult=2.5, atr_bear_mult=1.5),
    _pm_v2_set("pm13_regime_b30_r15",   **_V13_BASE,
               atr_regime_adaptive=True, atr_bull_mult=3.0, atr_bear_mult=1.5),

    # Approach B — Percentile gate (no SMA mult)
    _pm_v2_set("pm13_pct_90",           **_V13_BASE,
               atr_percentile_gate=True, atr_pct_threshold=0.90, atr_pct_window=100),
    _pm_v2_set("pm13_pct_95",           **_V13_BASE,
               atr_percentile_gate=True, atr_pct_threshold=0.95, atr_pct_window=100),

    # Approach C — ADX-scaled gate
    _pm_v2_set("pm13_adx_15_30",        **_V13_BASE,
               atr_adx_scale=True, atr_adx_base_mult=1.5, atr_adx_max_mult=3.0),

    # Hybrid A+B — Regime-adaptive AND percentile (OR logic)
    _pm_v2_set("pm13_hybrid_b25_p90",   **_V13_BASE,
               atr_regime_adaptive=True, atr_bull_mult=2.5, atr_bear_mult=1.5,
               atr_percentile_gate=True, atr_pct_threshold=0.90, atr_pct_window=100),
    _pm_v2_set("pm13_hybrid_b30_p95",   **_V13_BASE,
               atr_regime_adaptive=True, atr_bull_mult=3.0, atr_bear_mult=1.5,
               atr_percentile_gate=True, atr_pct_threshold=0.95, atr_pct_window=100),
]

XRP_PM_V13_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)   # 6m OOS Aug 2025 → Feb 2026
XRP_PM_V13_CONFIG["param_sets"] = _PM13_SETS

XRP_PM_V13_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V13_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V13_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V13_2Y_CONFIG["param_sets"] = _PM13_SETS

XRP_PM_V13_1Y_MID_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V13_1Y_MID_CONFIG["start_date"] = datetime(2024, 8, 1)
XRP_PM_V13_1Y_MID_CONFIG["end_date"]   = datetime(2025, 8, 1)
XRP_PM_V13_1Y_MID_CONFIG["param_sets"] = _PM13_SETS


# ===========================================================================
# v24 — XRPPM14: Directional velocity gate.
#
# PM13 diagnosis revealed that ALL ATR-based gates (regime-adaptive,
# percentile, ADX-scaled) cost ~77pp in bull because:
#   1. Gate fire rates are identical in pump vs calm periods (~3% for 1.5×)
#   2. The gate blocks at CRITICAL reversal points — exactly when the grid
#      strategy wants to buy dips and ride the recovery
#   3. Grid strategies profit from high-ATR dips; ATR gates block them
#
# Root cause: ATR gates are direction-blind.  An ATR spike from a healthy
# pump (price rising) is treated identically to one from a crash (falling).
#
# Approach D (atr_directional):
#   Keep original ATR gate logic, but only fire when price has actually
#   FALLEN over a lookback window.  If price is flat/rising, the gate
#   is suppressed — the ATR spike is from a bullish move, not a crash.
#   Params: atr_dir_lookback (default 8 = 2hrs), atr_dir_drop_pct (2%)
#
# Approach E (atr_acceleration):
#   Only fire the gate when ATR is actively RISING vs N candles ago.
#   After a crash, ATR stays elevated but flattens — gate clears faster
#   than the SMA-based approach.  Allows entries during the recovery.
#   Param: atr_accel_lookback (default 10)
#
# Approach F (atr_cooldown):
#   After the gate has fired for N consecutive candles, force it off.
#   Prevents prolonged blocking during recovery after a crash.
#   Param: atr_cooldown (default 0 = disabled)
#
# Combined with PM13 Approach A (regime-adaptive): D+A, E+A, D+E+A
# Also standalone: D-only, E-only, D+E, F+A
#
# 10 strategies × 3 windows (6m OOS, 2y walk-forward, mid-year)
# ===========================================================================

_V14_BASE = {**_V10_BASE}  # h0, spacing=1.5%, leverage=2.0 (same as V11/V13)

_PM14_SETS = [
    # Control — no gate at all (matches pm13_baseline)
    _pm_v2_set("pm14_baseline",           **_V14_BASE),

    # PM13 best bear protector as reference (regime-adaptive A, b25/r15)
    _pm_v2_set("pm14_regime_ref",         **_V14_BASE,
               atr_regime_adaptive=True, atr_bull_mult=2.5, atr_bear_mult=1.5),

    # Approach D standalone — directional gate with fixed 1.5× mult
    _pm_v2_set("pm14_dir_only",           **_V14_BASE,
               atr_parabolic_mult=1.5, atr_directional=True,
               atr_dir_lookback=8, atr_dir_drop_pct=0.02),

    # D + looser drop threshold (price must fall 3% to engage gate)
    _pm_v2_set("pm14_dir_3pct",           **_V14_BASE,
               atr_parabolic_mult=1.5, atr_directional=True,
               atr_dir_lookback=8, atr_dir_drop_pct=0.03),

    # Approach E standalone — acceleration gate with fixed 1.5× mult
    _pm_v2_set("pm14_accel_only",         **_V14_BASE,
               atr_parabolic_mult=1.5, atr_acceleration=True,
               atr_accel_lookback=10),

    # D + A — directional + regime-adaptive (best of both worlds?)
    _pm_v2_set("pm14_dir_regime",         **_V14_BASE,
               atr_regime_adaptive=True, atr_bull_mult=2.5, atr_bear_mult=1.5,
               atr_directional=True, atr_dir_lookback=8, atr_dir_drop_pct=0.02),

    # E + A — acceleration + regime-adaptive
    _pm_v2_set("pm14_accel_regime",       **_V14_BASE,
               atr_regime_adaptive=True, atr_bull_mult=2.5, atr_bear_mult=1.5,
               atr_acceleration=True, atr_accel_lookback=10),

    # D + E — directional + acceleration (double filter, no regime)
    _pm_v2_set("pm14_dir_accel",          **_V14_BASE,
               atr_parabolic_mult=1.5,
               atr_directional=True, atr_dir_lookback=8, atr_dir_drop_pct=0.02,
               atr_acceleration=True, atr_accel_lookback=10),

    # D + E + A — triple filter
    _pm_v2_set("pm14_dir_accel_regime",   **_V14_BASE,
               atr_regime_adaptive=True, atr_bull_mult=2.5, atr_bear_mult=1.5,
               atr_directional=True, atr_dir_lookback=8, atr_dir_drop_pct=0.02,
               atr_acceleration=True, atr_accel_lookback=10),

    # F + A — cooldown (resume after 4 candles / 1 hour) + regime-adaptive
    _pm_v2_set("pm14_cooldown_regime",    **_V14_BASE,
               atr_regime_adaptive=True, atr_bull_mult=2.5, atr_bear_mult=1.5,
               atr_cooldown=4),
]

XRP_PM_V14_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)   # 6m OOS Aug 2025 → Feb 2026
XRP_PM_V14_CONFIG["param_sets"] = _PM14_SETS

XRP_PM_V14_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V14_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V14_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V14_2Y_CONFIG["param_sets"] = _PM14_SETS

XRP_PM_V14_1Y_MID_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V14_1Y_MID_CONFIG["start_date"] = datetime(2024, 8, 1)
XRP_PM_V14_1Y_MID_CONFIG["end_date"]   = datetime(2025, 8, 1)
XRP_PM_V14_1Y_MID_CONFIG["param_sets"] = _PM14_SETS


# ===========================================================================
# v25 — XRPPM15: Cooldown sweep + F+E combinations.
#
# PM14 results revealed the COOLDOWN approach (F) is the breakthrough:
#   pm14_cooldown_regime:  106.77% bull | 119.18% 2y WF | 15.06% mid-yr
# This is 2× better than any previous strategy on the 2y walk-forward.
#
# Approach D (directional) was actively HARMFUL — it suppresses the gate
# during parabolic blow-off tops where price IS rising, causing entries
# at blow-off peaks.  2y WF dropped from 48.11% → 33.34%.
#
# Key insight: the problem isn't gate DIRECTION, it's gate DURATION.
# Brief gates protect against flash crashes; prolonged gates miss
# recoveries.  Cooldown directly controls duration.
#
# This sweep optimizes:
#   1. Cooldown duration: 2, 4, 6, 8, 12, 16 candles (0.5-4 hours)
#   2. F+E+A combo: cooldown + acceleration + regime-adaptive
#   3. F+E: cooldown + acceleration (no regime)
#
# 10 strategies × 3 windows (6m OOS, 2y walk-forward, mid-year)
# ===========================================================================

_V15_BASE = {**_V10_BASE}  # h0, spacing=1.5%, leverage=2.0

_PM15_SETS = [
    # Control — no gate (reference)
    _pm_v2_set("pm15_baseline",           **_V15_BASE),

    # Cooldown sweep: F+A with varying durations
    _pm_v2_set("pm15_cd2_regime",         **_V15_BASE,
               atr_regime_adaptive=True, atr_bull_mult=2.5, atr_bear_mult=1.5,
               atr_cooldown=2),
    _pm_v2_set("pm15_cd4_regime",         **_V15_BASE,
               atr_regime_adaptive=True, atr_bull_mult=2.5, atr_bear_mult=1.5,
               atr_cooldown=4),
    _pm_v2_set("pm15_cd6_regime",         **_V15_BASE,
               atr_regime_adaptive=True, atr_bull_mult=2.5, atr_bear_mult=1.5,
               atr_cooldown=6),
    _pm_v2_set("pm15_cd8_regime",         **_V15_BASE,
               atr_regime_adaptive=True, atr_bull_mult=2.5, atr_bear_mult=1.5,
               atr_cooldown=8),
    _pm_v2_set("pm15_cd12_regime",        **_V15_BASE,
               atr_regime_adaptive=True, atr_bull_mult=2.5, atr_bear_mult=1.5,
               atr_cooldown=12),
    _pm_v2_set("pm15_cd16_regime",        **_V15_BASE,
               atr_regime_adaptive=True, atr_bull_mult=2.5, atr_bear_mult=1.5,
               atr_cooldown=16),

    # F+E+A — cooldown + acceleration + regime-adaptive (best of E bear + F bull)
    _pm_v2_set("pm15_cd4_accel_regime",   **_V15_BASE,
               atr_regime_adaptive=True, atr_bull_mult=2.5, atr_bear_mult=1.5,
               atr_acceleration=True, atr_accel_lookback=10,
               atr_cooldown=4),

    # F+E — cooldown + acceleration, fixed 1.5× mult (no regime)
    _pm_v2_set("pm15_cd4_accel_fixed",    **_V15_BASE,
               atr_parabolic_mult=1.5,
               atr_acceleration=True, atr_accel_lookback=10,
               atr_cooldown=4),

    # Accel-only reference (E with fixed 1.5× — best bear protector from PM14)
    _pm_v2_set("pm15_accel_ref",          **_V15_BASE,
               atr_parabolic_mult=1.5, atr_acceleration=True,
               atr_accel_lookback=10),
]

XRP_PM_V15_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)   # 6m OOS Aug 2025 → Feb 2026
XRP_PM_V15_CONFIG["param_sets"] = _PM15_SETS

XRP_PM_V15_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V15_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V15_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V15_2Y_CONFIG["param_sets"] = _PM15_SETS

XRP_PM_V15_1Y_MID_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V15_1Y_MID_CONFIG["start_date"] = datetime(2024, 8, 1)
XRP_PM_V15_1Y_MID_CONFIG["end_date"]   = datetime(2025, 8, 1)
XRP_PM_V15_1Y_MID_CONFIG["param_sets"] = _PM15_SETS


# ===========================================================================
# v26 — XRPPM16: Dynamic self-calibrating mechanisms sweep.
#
# PM15 winner (pm15_cd4_accel_fixed) achieves +139.58% 2y walk-forward
# but suffers -10.58% in Dec 2024 post-parabolic correction.  All static
# gates (4% velocity, 4-candle cooldown, 90% size) fail when ATR is
# 3-5× normal because the thresholds don't scale with volatility.
#
# Three new self-calibrating mechanisms replace static parameters:
#
#   1. Dynamic velocity threshold (vel_atr_mult):
#      vel_threshold = mult × ATR / price.  Auto-scales up in high-vol,
#      so a 5% bounce in Dec 2024 (noise) doesn't trigger a LONG entry
#      that the fixed 4% threshold would accept.
#
#   2. Dynamic gate decay (gate_decay_scale):
#      After parabolic gate clears, suppress for spike_ratio × scale
#      additional candles.  Catches the 2-4 week post-parabolic window
#      where ATR is still elevated but no longer rising.
#
#   3. Dynamic position sizing (cap_size_atr_scale):
#      size = base / atr_ratio, capped [30%, 90%].  Reduces exposure
#      automatically when entering during residual high-vol.
#
#   4. Dynamic max loss per trade (trend_max_loss_atr):
#      Close if unrealised loss > N × ATR × qty.  Tighter in calm,
#      wider in volatile — ATR-proportional.
#
# Base: pm15_cd4_accel_fixed (ATR 1.5×, accel_lookback=10, cooldown=4)
# 12 strategies × 3 windows (6m OOS, 2y walk-forward, mid-year)
# ===========================================================================

_V16_BASE = {
    **_V10_BASE,                          # h0, spacing=1.5%, leverage=2.0
    "atr_parabolic_mult": 1.5,            # pm15 winner settings
    "atr_acceleration": True,
    "atr_accel_lookback": 10,
    "atr_cooldown": 4,
}

_PM16_SETS = [
    # Control — pm15 winner baseline (no dynamic mechanisms)
    _pm_v2_set("pm16_baseline",              **_V16_BASE),

    # ── Dynamic velocity only (ATR-ratio scaling) ───────────────
    # vel_atr_mult scales static threshold by max(1, mult × ATR/SMA)
    # Normal vol (ratio≈1): unchanged.  3× spike: threshold ×3.
    _pm_v2_set("pm16_vel05",                 **_V16_BASE,
               vel_atr_mult=0.5),
    _pm_v2_set("pm16_vel075",               **_V16_BASE,
               vel_atr_mult=0.75),
    _pm_v2_set("pm16_vel10",                 **_V16_BASE,
               vel_atr_mult=1.0),
    _pm_v2_set("pm16_vel15",                 **_V16_BASE,
               vel_atr_mult=1.5),

    # ── Dynamic gate decay only ───────────────────────────────
    _pm_v2_set("pm16_decay3",                **_V16_BASE,
               gate_decay_scale=3.0),
    _pm_v2_set("pm16_decay5",                **_V16_BASE,
               gate_decay_scale=5.0),

    # ── Dynamic sizing only ───────────────────────────────────
    _pm_v2_set("pm16_sizing",                **_V16_BASE,
               cap_size_atr_scale=True),

    # ── Velocity + decay combo ────────────────────────────────
    _pm_v2_set("pm16_vel10_decay3",          **_V16_BASE,
               vel_atr_mult=1.0,
               gate_decay_scale=3.0),
    _pm_v2_set("pm16_vel10_decay5",          **_V16_BASE,
               vel_atr_mult=1.0,
               gate_decay_scale=5.0),

    # ── Full stack: velocity + decay + sizing ─────────────────
    _pm_v2_set("pm16_full_v10d3",            **_V16_BASE,
               vel_atr_mult=1.0,
               gate_decay_scale=3.0,
               cap_size_atr_scale=True),
    _pm_v2_set("pm16_full_v10d5",            **_V16_BASE,
               vel_atr_mult=1.0,
               gate_decay_scale=5.0,
               cap_size_atr_scale=True),

    # ── Full stack + max loss cap ─────────────────────────────
    _pm_v2_set("pm16_full_maxloss",          **_V16_BASE,
               vel_atr_mult=1.0,
               gate_decay_scale=3.0,
               cap_size_atr_scale=True,
               trend_max_loss_atr=2.0),
]

XRP_PM_V16_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)   # 6m OOS Aug 2025 → Feb 2026
XRP_PM_V16_CONFIG["param_sets"] = _PM16_SETS

XRP_PM_V16_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V16_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V16_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V16_2Y_CONFIG["param_sets"] = _PM16_SETS

XRP_PM_V16_1Y_MID_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V16_1Y_MID_CONFIG["start_date"] = datetime(2024, 8, 1)
XRP_PM_V16_1Y_MID_CONFIG["end_date"]   = datetime(2025, 8, 1)
XRP_PM_V16_1Y_MID_CONFIG["param_sets"] = _PM16_SETS


# ===========================================================================
# v27 — XRPPM17: Directional & outcome-based mechanisms sweep.
#
# PM16 showed that magnitude-based ATR scaling (velocity, decay, sizing,
# max loss) hurts returns because high ATR is needed for profitable trends
# too.  The "ATR Paradox": restricting activity when ATR is high inevitably
# restricts profit-making entries alongside loss-making ones.
#
# PM17 addresses this with DIRECTIONAL and OUTCOME-BASED mechanisms:
#
#   1. Directional velocity (vel_dir_only):
#      Only apply velocity scaling when price < EMA-36 (bearish context).
#      In bull regimes (price >= EMA-36), velocity stays at static 4%.
#      Preserves bull-market entries while raising threshold in corrections.
#
#   2. ATR acceleration velocity (vel_accel_only):
#      Only apply velocity scaling when ATR is actively rising (vs N ago).
#      When ATR is elevated but falling (post-spike settling), scaling
#      is removed — allowing recovery entries that PM16 would block.
#
#   3. Equity curve filter (eq_curve_filter):
#      Suppress trend entries when equity < SMA(equity, N).
#      Self-calibrating: based on actual P&L, not market conditions.
#      Source: Van Tharp / Andrea Unger systematic trading.
#
#   4. Consecutive loss guard (consec_loss_max):
#      After N consecutive losing trend trades, pause entries for
#      M candles.  Pure outcome-based — detects when the bot's edge
#      has temporarily vanished regardless of market structure.
#
# Base: pm15 winner (ATR 1.5×, accel_lookback=10, cooldown=4)
# 11 strategies × 3 windows (6m OOS, 2y walk-forward, mid-year)
# ===========================================================================

_V17_BASE = {
    **_V10_BASE,                          # h0, spacing=1.5%, leverage=2.0
    "atr_parabolic_mult": 1.5,            # pm15 winner settings
    "atr_acceleration": True,
    "atr_accel_lookback": 10,
    "atr_cooldown": 4,
}

_PM17_SETS = [
    # Control — pm15 winner baseline (no PM17 mechanisms)
    _pm_v2_set("pm17_baseline",              **_V17_BASE),

    # ── Directional velocity only ─────────────────────────────
    # Scales threshold only when price < EMA-36 (bearish)
    _pm_v2_set("pm17_dirvel075",             **_V17_BASE,
               vel_atr_mult=0.75, vel_dir_only=True),
    _pm_v2_set("pm17_dirvel10",              **_V17_BASE,
               vel_atr_mult=1.0,  vel_dir_only=True),

    # ── ATR acceleration velocity only ────────────────────────
    # Scales threshold only when ATR is actively rising
    _pm_v2_set("pm17_accelvel075",           **_V17_BASE,
               vel_atr_mult=0.75, vel_accel_only=True),
    _pm_v2_set("pm17_accelvel10",            **_V17_BASE,
               vel_atr_mult=1.0,  vel_accel_only=True),

    # ── Equity curve filter only ──────────────────────────────
    _pm_v2_set("pm17_eqcurve50",             **_V17_BASE,
               eq_curve_filter=True, eq_curve_lookback=50),
    _pm_v2_set("pm17_eqcurve100",            **_V17_BASE,
               eq_curve_filter=True, eq_curve_lookback=100),

    # ── Consecutive loss guard only ───────────────────────────
    _pm_v2_set("pm17_closs2",                **_V17_BASE,
               consec_loss_max=2, consec_loss_pause=20),
    _pm_v2_set("pm17_closs3",                **_V17_BASE,
               consec_loss_max=3, consec_loss_pause=30),

    # ── Combos ────────────────────────────────────────────────
    _pm_v2_set("pm17_dirvel_eqcurve",        **_V17_BASE,
               vel_atr_mult=0.75, vel_dir_only=True,
               eq_curve_filter=True, eq_curve_lookback=50),
    _pm_v2_set("pm17_dirvel_closs",          **_V17_BASE,
               vel_atr_mult=0.75, vel_dir_only=True,
               consec_loss_max=2, consec_loss_pause=20),
]

XRP_PM_V17_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)   # 6m OOS Aug 2025 → Feb 2026
XRP_PM_V17_CONFIG["param_sets"] = _PM17_SETS

XRP_PM_V17_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V17_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V17_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V17_2Y_CONFIG["param_sets"] = _PM17_SETS

XRP_PM_V17_1Y_MID_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V17_1Y_MID_CONFIG["start_date"] = datetime(2024, 8, 1)
XRP_PM_V17_1Y_MID_CONFIG["end_date"]   = datetime(2025, 8, 1)
XRP_PM_V17_1Y_MID_CONFIG["param_sets"] = _PM17_SETS


# ===========================================================================
# v28 — XRPPM18: Push dirvel10 to its limits.
#
# PM17 confirmed directional velocity as the breakthrough mechanism:
#   pm17_dirvel10 = +93.93% 2y WF (+9.83pp), +4.74% mid-year (+16.17pp)
#
# However, the 6m OOS cost is ~26% (106.77% → 79.01%).  This cost comes
# from EMA-36 being too responsive — in choppy bull markets, price
# temporarily dips below EMA-36, triggering velocity scaling when it
# shouldn't.  PM18 explores:
#
#   1. Multiplier fine-tuning: 0.5, 1.25, 1.5, 2.0 with vel_dir_only=True
#      (we already have 0.75 and 1.0 from PM17)
#
#   2. Longer EMA periods for directional check (vel_dir_ema_period):
#      EMA-84 (slow), EMA-120, EMA-200 — smoother = fewer false bearish
#      signals in bull markets, reducing OOS cost while keeping crash guard
#
#   3. Dual filter (vel_dir_only + vel_accel_only):
#      Require BOTH conditions before scaling — most conservative filter
#
#   4. Confirm candle variations: 1, 2, 5 (base = 3)
#      Fewer confirms = catch trends earlier but more false signals
#
#   5. Trail stop variations: 3%, 5%, 6% (base = 4%)
#      Wider trail = ride trends longer but give back more on reversals
#
# Base: pm17_dirvel10 (vel_atr_mult=1.0, vel_dir_only=True + pm15 winner)
# 15 strategies × 3 windows
# ===========================================================================

_V18_BASE = {
    **_V10_BASE,                          # h0, spacing=1.5%, leverage=2.0
    "atr_parabolic_mult": 1.5,            # pm15 winner settings
    "atr_acceleration": True,
    "atr_accel_lookback": 10,
    "atr_cooldown": 4,
}

_PM18_SETS = [
    # Control — pm17_dirvel10 winner (vel_atr_mult=1.0, vel_dir_only, EMA-36)
    _pm_v2_set("pm18_baseline",              **_V18_BASE,
               vel_atr_mult=1.0, vel_dir_only=True),

    # ── Multiplier sweep with directional filter ──────────────
    _pm_v2_set("pm18_dirvel05",              **_V18_BASE,
               vel_atr_mult=0.5,  vel_dir_only=True),
    _pm_v2_set("pm18_dirvel125",             **_V18_BASE,
               vel_atr_mult=1.25, vel_dir_only=True),
    _pm_v2_set("pm18_dirvel15",              **_V18_BASE,
               vel_atr_mult=1.5,  vel_dir_only=True),
    _pm_v2_set("pm18_dirvel20",              **_V18_BASE,
               vel_atr_mult=2.0,  vel_dir_only=True),

    # ── Longer EMA periods for directional check ─────────────
    # Smoother EMA = fewer false "bearish" signals in choppy bull markets
    _pm_v2_set("pm18_ema84",                 **_V18_BASE,
               vel_atr_mult=1.0, vel_dir_only=True, vel_dir_ema_period=84),
    _pm_v2_set("pm18_ema120",                **_V18_BASE,
               vel_atr_mult=1.0, vel_dir_only=True, vel_dir_ema_period=120),
    _pm_v2_set("pm18_ema200",                **_V18_BASE,
               vel_atr_mult=1.0, vel_dir_only=True, vel_dir_ema_period=200),

    # ── Dual filter (direction + acceleration) ───────────────
    _pm_v2_set("pm18_dual",                  **_V18_BASE,
               vel_atr_mult=1.0, vel_dir_only=True, vel_accel_only=True),

    # ── Confirm candle variations ─────────────────────────────
    {**_pm_v2_set("pm18_confirm1",           **_V18_BASE,
               vel_atr_mult=1.0, vel_dir_only=True),
     "trend_confirm_candles": 1},
    {**_pm_v2_set("pm18_confirm2",           **_V18_BASE,
               vel_atr_mult=1.0, vel_dir_only=True),
     "trend_confirm_candles": 2},
    {**_pm_v2_set("pm18_confirm5",           **_V18_BASE,
               vel_atr_mult=1.0, vel_dir_only=True),
     "trend_confirm_candles": 5},

    # ── Trail stop variations ─────────────────────────────────
    {**_pm_v2_set("pm18_trail03",            **_V18_BASE,
               vel_atr_mult=1.0, vel_dir_only=True),
     "trend_trailing_stop_pct": 0.03},
    {**_pm_v2_set("pm18_trail05",            **_V18_BASE,
               vel_atr_mult=1.0, vel_dir_only=True),
     "trend_trailing_stop_pct": 0.05},
    {**_pm_v2_set("pm18_trail06",            **_V18_BASE,
               vel_atr_mult=1.0, vel_dir_only=True),
     "trend_trailing_stop_pct": 0.06},
]

XRP_PM_V18_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)   # 6m OOS Aug 2025 → Feb 2026
XRP_PM_V18_CONFIG["param_sets"] = _PM18_SETS

XRP_PM_V18_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V18_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V18_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V18_2Y_CONFIG["param_sets"] = _PM18_SETS

XRP_PM_V18_1Y_MID_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V18_1Y_MID_CONFIG["start_date"] = datetime(2024, 8, 1)
XRP_PM_V18_1Y_MID_CONFIG["end_date"]   = datetime(2025, 8, 1)
XRP_PM_V18_1Y_MID_CONFIG["param_sets"] = _PM18_SETS


# ===========================================================================
# v29 — XRPPM19: Combination sweep of PM18 top performers.
#
# PM18 identified four winning dimensions:
#   1. EMA-120:   +100.68% 2y WF, +8.35% mid-year  (vs baseline +93.93%, +4.74%)
#   2. Dual:      +98.19%  2y WF, +7.02% mid-year   (AND: vel_dir + vel_accel)
#   3. Confirm 5: +84.71%  2y WF, +21.64% mid-year  (best crash protection)
#   4. Mult 1.5:  +93.75%  2y WF, +12.05% mid-year  (higher multiplier)
#
# PM19 systematically combines these to find the optimal cocktail.
# ===========================================================================

_V19_BASE = {
    **_V18_BASE,                          # h0, spacing=1.5%, leverage=2.0, atr_parabolic_mult=1.5
}

_PM19_SETS = [
    # ── Two-way combinations ──────────────────────────────────
    # EMA-120 + each other winner
    {**_pm_v2_set("pm19_ema120_c5",          **_V19_BASE,
               vel_atr_mult=1.0, vel_dir_only=True, vel_dir_ema_period=120),
     "trend_confirm_candles": 5},
    _pm_v2_set("pm19_ema120_dual",           **_V19_BASE,
               vel_atr_mult=1.0, vel_dir_only=True, vel_dir_ema_period=120,
               vel_accel_only=True),
    _pm_v2_set("pm19_ema120_m15",            **_V19_BASE,
               vel_atr_mult=1.5, vel_dir_only=True, vel_dir_ema_period=120),

    # Other two-way combos (no EMA change — keep EMA-36 default)
    {**_pm_v2_set("pm19_dual_c5",            **_V19_BASE,
               vel_atr_mult=1.0, vel_dir_only=True, vel_accel_only=True),
     "trend_confirm_candles": 5},
    _pm_v2_set("pm19_dual_m15",              **_V19_BASE,
               vel_atr_mult=1.5, vel_dir_only=True, vel_accel_only=True),
    {**_pm_v2_set("pm19_m15_c5",             **_V19_BASE,
               vel_atr_mult=1.5, vel_dir_only=True),
     "trend_confirm_candles": 5},

    # ── Three-way combinations ────────────────────────────────
    {**_pm_v2_set("pm19_ema120_dual_c5",     **_V19_BASE,
               vel_atr_mult=1.0, vel_dir_only=True, vel_dir_ema_period=120,
               vel_accel_only=True),
     "trend_confirm_candles": 5},
    {**_pm_v2_set("pm19_ema120_m15_c5",      **_V19_BASE,
               vel_atr_mult=1.5, vel_dir_only=True, vel_dir_ema_period=120),
     "trend_confirm_candles": 5},
    _pm_v2_set("pm19_ema120_dual_m15",       **_V19_BASE,
               vel_atr_mult=1.5, vel_dir_only=True, vel_dir_ema_period=120,
               vel_accel_only=True),

    # ── Four-way kitchen sink ─────────────────────────────────
    {**_pm_v2_set("pm19_ema120_dual_m15_c5", **_V19_BASE,
               vel_atr_mult=1.5, vel_dir_only=True, vel_dir_ema_period=120,
               vel_accel_only=True),
     "trend_confirm_candles": 5},

    # ── Confirm 4 compromises (between default 3 and winner 5) ─
    {**_pm_v2_set("pm19_ema120_c4",          **_V19_BASE,
               vel_atr_mult=1.0, vel_dir_only=True, vel_dir_ema_period=120),
     "trend_confirm_candles": 4},
    {**_pm_v2_set("pm19_ema120_dual_c4",     **_V19_BASE,
               vel_atr_mult=1.0, vel_dir_only=True, vel_dir_ema_period=120,
               vel_accel_only=True),
     "trend_confirm_candles": 4},
]

XRP_PM_V19_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)   # 6m OOS Aug 2025 → Feb 2026
XRP_PM_V19_CONFIG["param_sets"] = _PM19_SETS

XRP_PM_V19_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V19_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V19_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V19_2Y_CONFIG["param_sets"] = _PM19_SETS

XRP_PM_V19_1Y_MID_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V19_1Y_MID_CONFIG["start_date"] = datetime(2024, 8, 1)
XRP_PM_V19_1Y_MID_CONFIG["end_date"]   = datetime(2025, 8, 1)
XRP_PM_V19_1Y_MID_CONFIG["param_sets"] = _PM19_SETS


# ===========================================================================
# v30 — XRPPM20: Multiplier fine-tune around the PM19 winner (EMA-120 + m1.5).
#
# PM19 showed EMA-120 + mult 1.5 = +206.04% combined (+114.84% 2y WF,
# +24.14% mid-year crash).  This sweep narrows the multiplier to find
# the exact optimal value, keeping EMA-120 fixed.
#
# 10 multiplier steps from 1.1 → 1.9 in 0.1 increments, plus two EMA
# variations (100, 140) at mult=1.5 to confirm 120 is optimal.
# ===========================================================================

_V20_BASE = {
    **_V19_BASE,                          # inherits h0, spacing=1.5%, leverage=2.0, atr_parabolic_mult=1.5
}

_PM20_SETS = [
    # ── Multiplier sweep with EMA-120 fixed ───────────────────
    _pm_v2_set("pm20_m110",   **_V20_BASE,
               vel_atr_mult=1.1, vel_dir_only=True, vel_dir_ema_period=120),
    _pm_v2_set("pm20_m120",   **_V20_BASE,
               vel_atr_mult=1.2, vel_dir_only=True, vel_dir_ema_period=120),
    _pm_v2_set("pm20_m130",   **_V20_BASE,
               vel_atr_mult=1.3, vel_dir_only=True, vel_dir_ema_period=120),
    _pm_v2_set("pm20_m140",   **_V20_BASE,
               vel_atr_mult=1.4, vel_dir_only=True, vel_dir_ema_period=120),
    _pm_v2_set("pm20_baseline", **_V20_BASE,       # control = PM19 winner
               vel_atr_mult=1.5, vel_dir_only=True, vel_dir_ema_period=120),
    _pm_v2_set("pm20_m160",   **_V20_BASE,
               vel_atr_mult=1.6, vel_dir_only=True, vel_dir_ema_period=120),
    _pm_v2_set("pm20_m170",   **_V20_BASE,
               vel_atr_mult=1.7, vel_dir_only=True, vel_dir_ema_period=120),
    _pm_v2_set("pm20_m175",   **_V20_BASE,
               vel_atr_mult=1.75, vel_dir_only=True, vel_dir_ema_period=120),
    _pm_v2_set("pm20_m180",   **_V20_BASE,
               vel_atr_mult=1.8, vel_dir_only=True, vel_dir_ema_period=120),
    _pm_v2_set("pm20_m190",   **_V20_BASE,
               vel_atr_mult=1.9, vel_dir_only=True, vel_dir_ema_period=120),

    # ── EMA period variations at mult=1.5 ─────────────────────
    _pm_v2_set("pm20_ema100", **_V20_BASE,
               vel_atr_mult=1.5, vel_dir_only=True, vel_dir_ema_period=100),
    _pm_v2_set("pm20_ema140", **_V20_BASE,
               vel_atr_mult=1.5, vel_dir_only=True, vel_dir_ema_period=140),
]

XRP_PM_V20_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)   # 6m OOS Aug 2025 → Feb 2026
XRP_PM_V20_CONFIG["param_sets"] = _PM20_SETS

XRP_PM_V20_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V20_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V20_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V20_2Y_CONFIG["param_sets"] = _PM20_SETS

XRP_PM_V20_1Y_MID_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V20_1Y_MID_CONFIG["start_date"] = datetime(2024, 8, 1)
XRP_PM_V20_1Y_MID_CONFIG["end_date"]   = datetime(2025, 8, 1)
XRP_PM_V20_1Y_MID_CONFIG["param_sets"] = _PM20_SETS


# ===========================================================================
# v31 — XRPPM21: Fine-grain multiplier cliff-mapping.
#
# PM20 showed a massive cliff between mult 1.6 (+258.28%) and 1.7 (-1.32%).
# This sweep uses 0.02 steps from 1.54 to 1.70 to pinpoint the exact edge.
# All strategies use EMA-120 + vel_dir_only=True (the PM19/PM20 base).
# ===========================================================================

_V21_BASE = {
    **_V20_BASE,
}

_PM21_SETS = [
    _pm_v2_set("pm21_m154",     **_V21_BASE,
               vel_atr_mult=1.54, vel_dir_only=True, vel_dir_ema_period=120),
    _pm_v2_set("pm21_m156",     **_V21_BASE,
               vel_atr_mult=1.56, vel_dir_only=True, vel_dir_ema_period=120),
    _pm_v2_set("pm21_m158",     **_V21_BASE,
               vel_atr_mult=1.58, vel_dir_only=True, vel_dir_ema_period=120),
    _pm_v2_set("pm21_baseline", **_V21_BASE,       # control = PM20 winner
               vel_atr_mult=1.60, vel_dir_only=True, vel_dir_ema_period=120),
    _pm_v2_set("pm21_m162",     **_V21_BASE,
               vel_atr_mult=1.62, vel_dir_only=True, vel_dir_ema_period=120),
    _pm_v2_set("pm21_m164",     **_V21_BASE,
               vel_atr_mult=1.64, vel_dir_only=True, vel_dir_ema_period=120),
    _pm_v2_set("pm21_m166",     **_V21_BASE,
               vel_atr_mult=1.66, vel_dir_only=True, vel_dir_ema_period=120),
    _pm_v2_set("pm21_m168",     **_V21_BASE,
               vel_atr_mult=1.68, vel_dir_only=True, vel_dir_ema_period=120),
    _pm_v2_set("pm21_m170",     **_V21_BASE,       # cliff reference
               vel_atr_mult=1.70, vel_dir_only=True, vel_dir_ema_period=120),
]

XRP_PM_V21_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)   # 6m OOS Aug 2025 → Feb 2026
XRP_PM_V21_CONFIG["param_sets"] = _PM21_SETS

XRP_PM_V21_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V21_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V21_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V21_2Y_CONFIG["param_sets"] = _PM21_SETS

XRP_PM_V21_1Y_MID_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V21_1Y_MID_CONFIG["start_date"] = datetime(2024, 8, 1)
XRP_PM_V21_1Y_MID_CONFIG["end_date"]   = datetime(2025, 8, 1)
XRP_PM_V21_1Y_MID_CONFIG["param_sets"] = _PM21_SETS

# ---------------------------------------------------------------------------
# PM21 FULL — only the two survivors across the full 6.5-year stitched cache
# ---------------------------------------------------------------------------
_PM21_FULL_SETS = [
    _pm_v2_set("pm21_baseline", **_V21_BASE,
               vel_atr_mult=1.60, vel_dir_only=True, vel_dir_ema_period=120),
    _pm_v2_set("pm21_m166",     **_V21_BASE,
               vel_atr_mult=1.66, vel_dir_only=True, vel_dir_ema_period=120),
]

XRP_PM_V21_FULL_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V21_FULL_CONFIG["start_date"]      = datetime(2017, 5, 19)   # Bitfinex stitched start
XRP_PM_V21_FULL_CONFIG["end_date"]        = datetime(2026, 3, 8)    # latest cache date
XRP_PM_V21_FULL_CONFIG["param_sets"]      = _PM21_FULL_SETS
XRP_PM_V21_FULL_CONFIG["daily_breakdown"] = True                    # emit day-by-day table


# ===========================================================================
# v32 — XRPPM22: PM19 combination winners re-tested at mult 1.6.
#
# PM20 showed mult 1.6 (+258%) is far superior to 1.5 (+206%).  Now re-test
# the PM19-style combinations (dual filter, confirm candles) at this new
# optimal multiplier to see if they further improve or interfere.
# ===========================================================================

_V22_BASE = {
    **_V21_BASE,
}

_PM22_SETS = [
    # ── Baseline ────────────────────────────────────────────────
    _pm_v2_set("pm22_baseline",   **_V22_BASE,       # control = PM20 winner
               vel_atr_mult=1.6, vel_dir_only=True, vel_dir_ema_period=120),

    # ── Single additions ────────────────────────────────────────
    _pm_v2_set("pm22_dual",       **_V22_BASE,
               vel_atr_mult=1.6, vel_dir_only=True, vel_dir_ema_period=120,
               vel_accel_only=True),
    {**_pm_v2_set("pm22_c2",     **_V22_BASE,
               vel_atr_mult=1.6, vel_dir_only=True, vel_dir_ema_period=120),
     "trend_confirm_candles": 2},
    {**_pm_v2_set("pm22_c4",     **_V22_BASE,
               vel_atr_mult=1.6, vel_dir_only=True, vel_dir_ema_period=120),
     "trend_confirm_candles": 4},
    {**_pm_v2_set("pm22_c5",     **_V22_BASE,
               vel_atr_mult=1.6, vel_dir_only=True, vel_dir_ema_period=120),
     "trend_confirm_candles": 5},

    # ── Dual + confirm combinations ────────────────────────────
    {**_pm_v2_set("pm22_dual_c2", **_V22_BASE,
               vel_atr_mult=1.6, vel_dir_only=True, vel_dir_ema_period=120,
               vel_accel_only=True),
     "trend_confirm_candles": 2},
    {**_pm_v2_set("pm22_dual_c4", **_V22_BASE,
               vel_atr_mult=1.6, vel_dir_only=True, vel_dir_ema_period=120,
               vel_accel_only=True),
     "trend_confirm_candles": 4},
    {**_pm_v2_set("pm22_dual_c5", **_V22_BASE,
               vel_atr_mult=1.6, vel_dir_only=True, vel_dir_ema_period=120,
               vel_accel_only=True),
     "trend_confirm_candles": 5},
]

XRP_PM_V22_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)   # 6m OOS Aug 2025 → Feb 2026
XRP_PM_V22_CONFIG["param_sets"] = _PM22_SETS

XRP_PM_V22_2Y_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V22_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_PM_V22_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_PM_V22_2Y_CONFIG["param_sets"] = _PM22_SETS

XRP_PM_V22_1Y_MID_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V22_1Y_MID_CONFIG["start_date"] = datetime(2024, 8, 1)
XRP_PM_V22_1Y_MID_CONFIG["end_date"]   = datetime(2025, 8, 1)
XRP_PM_V22_1Y_MID_CONFIG["param_sets"] = _PM22_SETS


# ===========================================================================
# v33 — XRPPM23: Gate-relaxation combined sweep.
#
# Deep idle-day analysis of the 6.5yr backtest revealed that 87.9% of the
# 1,739 idle days (74% of all days) were caused by GATE OVER-SUPPRESSION,
# not grid spacing.  Only 2.6% of idle days had volatility below the 1.5%
# grid spacing.  The ADX grid-pause (threshold=35) and regime filter
# (EMA-175, 0% hysteresis) shut the bot down even when plentiful grid-
# crossing opportunities exist (median idle-day range = 4.14%).
#
# This sweep tests three relaxation axes together:
#   Axis A — ADX grid-pause threshold:  35 (current) | 50 | off
#   Axis B — Regime hysteresis buffer:  0.00 (current) | 0.02
#   Axis C — Vol-adaptive spacing:       off (current) | on (floor=0.008, ceil=0.020)
#
# 10 combinations (smart subset of 3×2×2=12) covering all single, pairwise,
# and full-open variants.  All use PM21 m166 winner as base (vel_atr_mult=1.66,
# vel_dir_only=True, vel_dir_ema_period=120).
# ===========================================================================

_V23_BASE = {
    **_V21_BASE,
}

# Helper: build a PM23 param set with optional gate overrides
def _pm23(name: str, adx_gp=35, hyst: float = 0.00, vas: bool = False):
    """Build a PM23 sweep param set with gate-relaxation overrides."""
    # Remove keys that will be explicitly overridden to avoid duplicate kwargs
    _base = {k: v for k, v in _V23_BASE.items()
             if k not in ("regime_hysteresis_pct", "vol_adaptive_spacing",
                          "vas_floor", "vas_ceil")}
    base = _pm_v2_set(
        name, **_base,
        vel_atr_mult=1.66, vel_dir_only=True, vel_dir_ema_period=120,
        regime_hysteresis_pct=hyst,
        vol_adaptive_spacing=vas,
        vas_floor=0.008,
        vas_ceil=0.020,
    )
    if adx_gp is None:
        base["adx_grid_pause"] = None
    else:
        base["adx_grid_pause"] = adx_gp
    return base

_PM23_SETS = [
    # ── Control ─────────────────────────────────────────────────
    _pm23("pm23_baseline",        adx_gp=35,   hyst=0.00, vas=False),

    # ── Single-axis changes ─────────────────────────────────────
    _pm23("pm23_adx50",           adx_gp=50,   hyst=0.00, vas=False),
    _pm23("pm23_adx_off",         adx_gp=None, hyst=0.00, vas=False),
    _pm23("pm23_h2",              adx_gp=35,   hyst=0.02, vas=False),
    _pm23("pm23_vas",             adx_gp=35,   hyst=0.00, vas=True),

    # ── Two-axis combinations ───────────────────────────────────
    _pm23("pm23_adx50_h2",        adx_gp=50,   hyst=0.02, vas=False),
    _pm23("pm23_adx50_vas",       adx_gp=50,   hyst=0.00, vas=True),
    _pm23("pm23_h2_vas",          adx_gp=35,   hyst=0.02, vas=True),

    # ── Three-axis full open ────────────────────────────────────
    _pm23("pm23_adx50_h2_vas",    adx_gp=50,   hyst=0.02, vas=True),
    _pm23("pm23_full_open",       adx_gp=None, hyst=0.02, vas=True),
]

XRP_PM_V23_FULL_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V23_FULL_CONFIG["start_date"]      = datetime(2017, 5, 19)   # full 8.8yr (Bitfinex start)
XRP_PM_V23_FULL_CONFIG["end_date"]        = datetime(2026, 3, 8)
XRP_PM_V23_FULL_CONFIG["param_sets"]      = _PM23_SETS
XRP_PM_V23_FULL_CONFIG["daily_breakdown"] = True


# ===========================================================================
# v34 — XRPPM24: 5-regime rotation sweep.
#
# Production algo systems don't use binary gates — they classify the
# market into regimes and apply different strategy profiles per regime.
# The idle-day analysis proved 87.9% of idle days are gate-blocked, not
# volatility-constrained.  Regime rotation directly fixes this by:
#
#   CHOPPY   (ADX < choppy_max, autocorr < 0.1):
#     Bypass ADX pause + regime filter → aggressive grid, no trend capture.
#   VOLATILE (ADX between choppy_max..trending_min, normal ATR):
#     Default gated behavior (current PM21 logic).
#   TRENDING (ADX > trending_min):
#     Grid off (avoid SL cascades), trend capture only.
#   CRASH    (ATR/SMA > crash_mult):
#     Bounce-hunting grid (tight spacing, reduced qty), no trend capture.
#   DORMANT  (ATR/SMA < dormant_mult, ADX < choppy_max):
#     Micro-DCA (wide spacing, 20% qty, long-only), no trend capture.
#
# Sweep axes:
#   A — ADX choppy_max threshold:  20 / 25 (default) / 30
#   B — ADX trending_min threshold:  30 / 35 (default) / 40
#   C — Dormant qty multiplier:  0.1 / 0.2 / 0.3
# ===========================================================================

_V24_BASE = {
    **_V21_BASE,
}

# Helper: build a PM24 regime-rotation param set
def _pm24(name: str, regime_rotation: bool = True, **kw):
    """Build a PM24 sweep param set with regime rotation overrides."""
    _base = {k: v for k, v in _V24_BASE.items()
             if k not in ("regime_hysteresis_pct", "vol_adaptive_spacing",
                          "vas_floor", "vas_ceil", "regime_rotation")}
    return _pm_v2_set(
        name, **_base,
        vel_atr_mult=1.66, vel_dir_only=True, vel_dir_ema_period=120,
        regime_rotation=regime_rotation,
        atr_trail=True,
        surge_cb=True,
        crash_cb=True,
        dd_halt=True,
        regime_short_gate=True,
        **kw,
    )

_PM24_SETS = [
    # ── Control — no regime rotation (= PM21 m166 behavior) ────────────
    _pm24("pm24_baseline",        regime_rotation=False),

    # ── Default regime rotation ─────────────────────────────────────────
    _pm24("pm24_default"),

    # ── Axis A: ADX choppy threshold (below this = range-trade) ────────
    _pm24("pm24_choppy20",        regime_adx_choppy_max=20.0),
    _pm24("pm24_choppy30",        regime_adx_choppy_max=30.0),

    # ── Axis B: ADX trending threshold (above this = trend-only) ───────
    _pm24("pm24_trend30",         regime_adx_trending_min=30.0),
    _pm24("pm24_trend40",         regime_adx_trending_min=40.0),

    # ── Axis C: Dormant DCA aggressiveness ─────────────────────────────
    _pm24("pm24_dorm10",          regime_dormant_qty_mult=0.1),
    _pm24("pm24_dorm30",          regime_dormant_qty_mult=0.3),

    # ── Aggressive: wide choppy zone, narrow trending zone ─────────────
    _pm24("pm24_aggressive",      regime_adx_choppy_max=30.0, regime_adx_trending_min=40.0),

    # ── Conservative: narrow choppy zone, wide trending zone ───────────
    _pm24("pm24_conservative",    regime_adx_choppy_max=20.0, regime_adx_trending_min=30.0),
]

XRP_PM_V24_FULL_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V24_FULL_CONFIG["start_date"]      = datetime(2017, 5, 19)   # full 8.8yr (Bitfinex start)
XRP_PM_V24_FULL_CONFIG["end_date"]        = datetime(2026, 3, 8)
XRP_PM_V24_FULL_CONFIG["param_sets"]      = _PM24_SETS
XRP_PM_V24_FULL_CONFIG["daily_breakdown"] = True


# ===========================================================================
# v35 — XRPPM25: Regime detection quality sweep.
#
# PM24 identified the regime rotation framework.  PM25 improves the
# classifier itself:
#
#   1) Hysteresis (min dwell) — suppress flicker on threshold boundaries.
#      Short regime bursts (< N candles) are reverted to the previous
#      stable regime.  CRASH is exempt (always immediate for safety).
#
#   2) Autocorrelation threshold — the CHOPPY classification requires
#      autocorr < threshold.  Sweep to find optimal sensitivity.
#
#   3) CRASH/CB conflict fix — the CRASH regime no longer overrides
#      the velocity circuit breaker halt (always-on safety improvement).
#
#   4) CHOPPY/DORMANT exclusivity — CHOPPY now explicitly excludes
#      DORMANT-level volatility (always-on correctness fix).
#
# Sweep axes:
#   A — regime_min_dwell_candles: 1 (off) / 3 / 5 (default) / 10 / 20
#   B — regime_autocorr_choppy_max: 0.05 / 0.1 (default) / 0.15 / 0.2
#   C — Key combos
# ===========================================================================

_V25_BASE = {
    **_V21_BASE,
}

def _pm25(name: str, **kw):
    """Build a PM25 regime-quality sweep param set."""
    _base = {k: v for k, v in _V25_BASE.items()
             if k not in ("regime_hysteresis_pct", "vol_adaptive_spacing",
                          "vas_floor", "vas_ceil", "regime_rotation")}
    return _pm_v2_set(
        name, **_base,
        vel_atr_mult=1.66, vel_dir_only=True, vel_dir_ema_period=120,
        regime_rotation=True,
        atr_trail=True,
        surge_cb=True,
        crash_cb=True,
        dd_halt=True,
        regime_short_gate=True,
        **kw,
    )

_PM25_SETS = [
    # ── Control — PM24 default (dwell=5, autocorr=0.1) ─────────────────
    _pm25("pm25_baseline"),

    # ── Axis A: Hysteresis dwell time ──────────────────────────────────
    _pm25("pm25_dwell1",          regime_min_dwell_candles=1),    # off
    _pm25("pm25_dwell3",          regime_min_dwell_candles=3),
    _pm25("pm25_dwell10",         regime_min_dwell_candles=10),
    _pm25("pm25_dwell20",         regime_min_dwell_candles=20),

    # ── Axis B: Autocorrelation CHOPPY threshold ───────────────────────
    _pm25("pm25_ac05",            regime_autocorr_choppy_max=0.05),
    _pm25("pm25_ac15",            regime_autocorr_choppy_max=0.15),
    _pm25("pm25_ac20",            regime_autocorr_choppy_max=0.20),

    # ── Axis C: Combos — best dwell + best autocorr ───────────────────
    _pm25("pm25_dwell10_ac15",    regime_min_dwell_candles=10, regime_autocorr_choppy_max=0.15),
    _pm25("pm25_dwell10_ac05",    regime_min_dwell_candles=10, regime_autocorr_choppy_max=0.05),
    _pm25("pm25_dwell20_ac15",    regime_min_dwell_candles=20, regime_autocorr_choppy_max=0.15),
]

XRP_PM_V25_FULL_CONFIG: Dict[str, Any] = dict(XRP_CONFIG)
XRP_PM_V25_FULL_CONFIG["start_date"]      = datetime(2017, 5, 19)   # full 8.8yr (Bitfinex start)
XRP_PM_V25_FULL_CONFIG["end_date"]        = datetime(2026, 3, 8)
XRP_PM_V25_FULL_CONFIG["param_sets"]      = _PM25_SETS
XRP_PM_V25_FULL_CONFIG["daily_breakdown"] = True


# ===========================================================================
# v11 — Crash protection sweep
#
# Three independent mechanisms to limit losses in flash-crash events
# (LUNA/Terra May 2022, FTX Nov 2022, XRP SEC lawsuit Dec 2020, etc.):
#
# (A) Velocity circuit breaker  (crash_cb=True)
#     Fires when price drops >= crash_cb_drop_pct in crash_cb_lookback_candles.
#     Immediately closes all open long grid positions + any active trend long.
#     Halts new long entries for crash_cb_halt_candles.
#     Short side is intentionally unblocked — it can profit from the drop.
#
# (B) Drawdown halt + resume  (dd_halt=True)
#     Replaces the permanent hard-stop.  When equity falls >= max_drawdown
#     from peak, flatten ALL positions and pause all trading for
#     dd_halt_candles.  After the cooldown the bot resumes normally.
#
# (C) Grid notional cap  (grid_notional_cap_pct)
#     Limits total long-side grid margin to X% of current equity.
#     Prevents cascading ladder-fills ("buying a falling knife") during
#     sustained one-directional drops.
#
# Baseline is the v9 re_reentry winner (ADX t25/gp35 + fast re-entry).
# Run on the 3.9-year MAX window (Apr 2022 → Feb 2026) which covers:
#   May 2022 crash (LUNA/Terra collapse) — XRP down ~60%
#   Nov 2022 crash (FTX collapse)        — XRP down ~45%
#   2023 sideways grind + 2024–2025 bull run.
# Also run on the 2Y window for comparison with earlier features.
# Once the extended Binance cache is available (Oct 2019), re-run on
# the full 6.5-year dataset to test against the 2020 SEC lawsuit crash.
# ===========================================================================

def _cb_set(
    name: str,
    crash_cb: bool = False,
    cb_drop: float = 0.10,
    cb_lookback: int = 8,
    cb_halt: int = 48,
    dd_halt: bool = False,
    dd_halt_pct: float = 0.30,
    dd_halt_candles: int = 384,
    grid_cap: Optional[float] = None,
) -> Dict[str, Any]:
    """v11 crash-protection param set — inherits v9 re_reentry baseline."""
    d: Dict[str, Any] = {
        "name": name,
        "use_sl": True,
        "trend_detection": True, "trend_capture": True,
        "trend_force_close_grid": True, "trend_confirm_candles": 3,
        "trend_trailing_stop_pct": 0.04, "trend_capture_size_pct": 0.90,
        "trend_lookback_candles": 10,
        "adx_filter": True, "adx_min_trend": 25, "adx_grid_pause": 35,
        "trend_reentry_fast": True,
        # (A) Velocity circuit breaker
        "crash_cb":                  crash_cb,
        "crash_cb_drop_pct":         cb_drop,
        "crash_cb_lookback_candles": cb_lookback,
        "crash_cb_halt_candles":     cb_halt,
        # (B) Drawdown halt + resume: max_drawdown doubles as the halt threshold
        "dd_halt":        dd_halt,
        "max_drawdown":   dd_halt_pct if dd_halt else 0.9,  # 0.9 = effectively no hard-stop
        "dd_halt_candles": dd_halt_candles,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    }
    # (C) Grid notional cap
    if grid_cap is not None:
        d["grid_notional_cap_pct"] = grid_cap
    return d


XRP_CB_CONFIG: Dict[str, Any] = dict(XRP_MAX_CONFIG)
XRP_CB_CONFIG["param_sets"] = [
    # ── Baseline (no crash protection) ──────────────────────────────────────
    _cb_set("cb_baseline"),

    # ── (A) Velocity CB only: 3 variants ────────────────────────────────────
    # A1: 8-candle window (2 h), -10% drop, 48-candle halt (12 h)
    _cb_set("cb_vel_8_10_48",  crash_cb=True, cb_lookback=8,  cb_drop=0.10, cb_halt=48),
    # A2: 4-candle window (1 h), -8% drop,  24-candle halt (6 h) — snappier
    _cb_set("cb_vel_4_8_24",   crash_cb=True, cb_lookback=4,  cb_drop=0.08, cb_halt=24),
    # A3: 8-candle window (2 h), -12% drop, 96-candle halt (24 h) — conservative
    _cb_set("cb_vel_8_12_96",  crash_cb=True, cb_lookback=8,  cb_drop=0.12, cb_halt=96),

    # ── (B) DD halt + resume only: 2 variants ───────────────────────────────
    # B1: halt at 15% drawdown, 96-candle resume (24 h)
    _cb_set("cb_dd_15_96",  dd_halt=True, dd_halt_pct=0.15, dd_halt_candles=96),
    # B2: halt at 20% drawdown, 48-candle resume (12 h) — shallower trigger, faster resume
    _cb_set("cb_dd_20_48",  dd_halt=True, dd_halt_pct=0.20, dd_halt_candles=48),

    # ── (C) Grid notional cap only: 2 variants ──────────────────────────────
    # C1: limit long grid exposure to 25% of equity
    _cb_set("cb_cap_25", grid_cap=0.25),
    # C2: 40% cap — looser but still bounded
    _cb_set("cb_cap_40", grid_cap=0.40),

    # ── Combined: A1 + B1 + C1 ──────────────────────────────────────────────
    _cb_set("cb_combined",
            crash_cb=True, cb_lookback=8, cb_drop=0.10, cb_halt=48,
            dd_halt=True,  dd_halt_pct=0.15, dd_halt_candles=96,
            grid_cap=0.25),

    # ── Grid-only (no trend): structural floor ───────────────────────────────
    {
        "name": "cb_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]

XRP_CB_2Y_CONFIG: Dict[str, Any] = dict(XRP_CB_CONFIG)
XRP_CB_2Y_CONFIG["start_date"] = datetime(2024, 2, 28)
XRP_CB_2Y_CONFIG["end_date"]   = datetime(2026, 2, 28)
XRP_CB_2Y_CONFIG["param_sets"] = [
    _cb_set("2y_cb_baseline"),
    _cb_set("2y_cb_vel_8_10_48", crash_cb=True, cb_lookback=8, cb_drop=0.10, cb_halt=48),
    _cb_set("2y_cb_dd_15_96",    dd_halt=True,  dd_halt_pct=0.15, dd_halt_candles=96),
    _cb_set("2y_cb_cap_25",      grid_cap=0.25),
    _cb_set("2y_cb_combined",
            crash_cb=True, cb_lookback=8, cb_drop=0.10, cb_halt=48,
            dd_halt=True,  dd_halt_pct=0.15, dd_halt_candles=96,
            grid_cap=0.25),
    {
        "name": "2y_cb_trend_off",
        "use_sl": True, "trend_detection": False, "trend_capture": False,
        "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
    },
]


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import sys
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"

    if symbol in ("BTCUSDT", "BTC"):
        print("\n" + "=" * 60)
        print("  BTCUSDT backtest")
        print("=" * 60)
        grid_search_backtest(CONFIG)
    elif symbol in ("XRPUSDT", "XRP"):
        print("\n" + "=" * 60)
        print("  XRPUSDT 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_CONFIG)

        print("\n" + "=" * 60)
        print("  XRPUSDT robustness check  (May 2025 → Aug 2025)")
        print("=" * 60)
        grid_search_backtest(XRP_VALIDATE_CONFIG)

        print("\n" + "=" * 60)
        print("  XRPUSDT 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_2Y_CONFIG)

        print("\n" + "=" * 60)
        print("  XRPUSDT max history  (Apr 2022 → Feb 2026  ~3y 10m)")
        print("=" * 60)
        grid_search_backtest(XRP_MAX_CONFIG)
    elif symbol in ("XRPADX", "ADX"):
        print("\n" + "=" * 60)
        print("  v2 ADX filter sweep \u2014 6-month OOS  (Aug 2025 \u2192 Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_ADX_CONFIG)

        print("\n" + "=" * 60)
        print("  v2 ADX filter sweep \u2014 2-year walk-forward  (Feb 2024 \u2192 Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_ADX_2Y_CONFIG)
    elif symbol in ("XRPMS", "MS"):
        print("\n" + "=" * 60)
        print("  v8 Market structure filter — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_MS_CONFIG)

        print("\n" + "=" * 60)
        print("  v8 Market structure filter — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_MS_2Y_CONFIG)
    elif symbol in ("XRPATR", "ATR"):
        print("\n" + "=" * 60)
        print("  v3 ATR dynamic trail \u2014 6-month OOS  (Aug 2025 \u2192 Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_ATR_CONFIG)

        print("\n" + "=" * 60)
        print("  v3 ATR dynamic trail \u2014 2-year walk-forward  (Feb 2024 \u2192 Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_ATR_2Y_CONFIG)
    elif symbol in ("XRPEMA", "EMA"):
        print("\n" + "=" * 60)
        print("  v4 EMA bias filter — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_EMA_CONFIG)

        print("\n" + "=" * 60)
        print("  v4 EMA bias filter — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_EMA_2Y_CONFIG)
    elif symbol in ("XRPBB", "BB"):
        print("\n" + "=" * 60)
        print("  v5 BB squeeze filter — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_BB_CONFIG)

        print("\n" + "=" * 60)
        print("  v5 BB squeeze filter — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_BB_2Y_CONFIG)
    elif symbol in ("XRPRSI", "RSI"):
        print("\n" + "=" * 60)
        print("  v6 RSI filter — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_RSI_CONFIG)

        print("\n" + "=" * 60)
        print("  v6 RSI filter — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_RSI_2Y_CONFIG)
    elif symbol in ("XRPVOL", "VOL"):
        print("\n" + "=" * 60)
        print("  v7 Volume confirmation — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_VOL_CONFIG)

        print("\n" + "=" * 60)
        print("  v7 Volume confirmation — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_VOL_2Y_CONFIG)
    elif symbol in ("XRPRE", "RE"):
        print("\n" + "=" * 60)
        print("  v9 Fast re-entry + ADX-adaptive trail — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_REENTRY_CONFIG)

        print("\n" + "=" * 60)
        print("  v9 Fast re-entry + ADX-adaptive trail — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_REENTRY_2Y_CONFIG)
    elif symbol in ("XRPPM", "PM"):
        print("\n" + "=" * 60)
        print("  v10 Indicator position management — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_CONFIG)

        print("\n" + "=" * 60)
        print("  v10 Indicator position management — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_2Y_CONFIG)
    elif symbol in ("XRPPM2", "PM2"):
        print("\n" + "=" * 60)
        print("  v12 PM tuning (A/B/C) — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V2_CONFIG)

        print("\n" + "=" * 60)
        print("  v12 PM tuning (A/B/C) — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V2_2Y_CONFIG)
    elif symbol in ("XRPPM3", "PM3"):
        print("\n" + "=" * 60)
        print("  v13 C_gridvol sweep (floor × period) — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V3_CONFIG)

        print("\n" + "=" * 60)
        print("  v13 C_gridvol sweep (floor × period) — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V3_2Y_CONFIG)
    elif symbol in ("XRPPM4", "PM4"):
        print("\n" + "=" * 60)
        print("  v14 C_gridvol period fine-sweep — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V4_CONFIG)

        print("\n" + "=" * 60)
        print("  v14 C_gridvol period fine-sweep — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V4_2Y_CONFIG)
    elif symbol in ("XRPPM5", "PM5"):
        print("\n" + "=" * 60)
        print("  v15 Regime filter sweep — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V5_CONFIG)

        print("\n" + "=" * 60)
        print("  v15 Regime filter sweep — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V5_2Y_CONFIG)
    elif symbol in ("XRPPM6", "PM6"):
        print("\n" + "=" * 60)
        print("  v16 XRPPM6 comprehensive sweep — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V6_CONFIG)

        print("\n" + "=" * 60)
        print("  v16 XRPPM6 comprehensive sweep — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V6_2Y_CONFIG)
    elif symbol in ("XRPPM7", "PM7"):
        print("\n" + "=" * 60)
        print("  v17 XRPPM7 leverage + equity-pct sweep — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V7_CONFIG)

        print("\n" + "=" * 60)
        print("  v17 XRPPM7 leverage + equity-pct sweep — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V7_2Y_CONFIG)
    elif symbol in ("XRPPM8", "PM8"):
        print("\n" + "=" * 60)
        print("  v18 XRPPM8 hysteresis + spacing fine-tune on l2 base — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V8_CONFIG)

        print("\n" + "=" * 60)
        print("  v18 XRPPM8 hysteresis + spacing fine-tune on l2 base — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V8_2Y_CONFIG)
    elif symbol in ("XRPPM9", "PM9"):
        print("\n" + "=" * 60)
        print("  v19 XRPPM9 adaptive spacing (VAS + BTBW) — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V9_CONFIG)

        print("\n" + "=" * 60)
        print("  v19 XRPPM9 adaptive spacing (VAS + BTBW) — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V9_2Y_CONFIG)
    elif symbol in ("XRPPM10", "PM10"):
        print("\n" + "=" * 60)
        print("  v20 XRPPM10 h0 base + BTBW — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V10_CONFIG)

        print("\n" + "=" * 60)
        print("  v20 XRPPM10 h0 base + BTBW — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V10_2Y_CONFIG)

        print("\n" + "=" * 60)
        print("  v20 XRPPM10 h0 base + BTBW — mid-year  (Aug 2024 → Aug 2025)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V10_1Y_MID_CONFIG)
    elif symbol in ("XRPPM11", "PM11"):
        print("\n" + "=" * 60)
        print("  v21 XRPPM11 3-layer 80% fix — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V11_CONFIG)

        print("\n" + "=" * 60)
        print("  v21 XRPPM11 3-layer 80% fix — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V11_2Y_CONFIG)

        print("\n" + "=" * 60)
        print("  v21 XRPPM11 3-layer 80% fix — mid-year  (Aug 2024 → Aug 2025)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V11_1Y_MID_CONFIG)
    elif symbol in ("XRPPM12", "PM12"):
        print("\n" + "=" * 60)
        print("  v22 XRPPM12 full fix + grid sleep — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V12_CONFIG)

        print("\n" + "=" * 60)
        print("  v22 XRPPM12 full fix + grid sleep — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V12_2Y_CONFIG)

        print("\n" + "=" * 60)
        print("  v22 XRPPM12 full fix + grid sleep — mid-year  (Aug 2024 → Aug 2025)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V12_1Y_MID_CONFIG)
    elif symbol in ("XRPPM13", "PM13"):
        print("\n" + "=" * 60)
        print("  v23 XRPPM13 adaptive ATR gate — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V13_CONFIG)

        print("\n" + "=" * 60)
        print("  v23 XRPPM13 adaptive ATR gate — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V13_2Y_CONFIG)

        print("\n" + "=" * 60)
        print("  v23 XRPPM13 adaptive ATR gate — mid-year  (Aug 2024 → Aug 2025)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V13_1Y_MID_CONFIG)
    elif symbol in ("XRPPM14", "PM14"):
        print("\n" + "=" * 60)
        print("  v24 XRPPM14 directional velocity gate — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V14_CONFIG)

        print("\n" + "=" * 60)
        print("  v24 XRPPM14 directional velocity gate — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V14_2Y_CONFIG)

        print("\n" + "=" * 60)
        print("  v24 XRPPM14 directional velocity gate — mid-year  (Aug 2024 → Aug 2025)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V14_1Y_MID_CONFIG)
    elif symbol in ("XRPPM15", "PM15"):
        print("\n" + "=" * 60)
        print("  v25 XRPPM15 cooldown sweep + F+E — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V15_CONFIG)

        print("\n" + "=" * 60)
        print("  v25 XRPPM15 cooldown sweep + F+E — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V15_2Y_CONFIG)

        print("\n" + "=" * 60)
        print("  v25 XRPPM15 cooldown sweep + F+E — mid-year  (Aug 2024 → Aug 2025)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V15_1Y_MID_CONFIG)
    elif symbol in ("XRPPM16", "PM16"):
        print("\n" + "=" * 60)
        print("  v26 XRPPM16 dynamic self-calibrating — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V16_CONFIG)

        print("\n" + "=" * 60)
        print("  v26 XRPPM16 dynamic self-calibrating — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V16_2Y_CONFIG)

        print("\n" + "=" * 60)
        print("  v26 XRPPM16 dynamic self-calibrating — mid-year  (Aug 2024 → Aug 2025)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V16_1Y_MID_CONFIG)
    elif symbol in ("XRPPM17", "PM17"):
        print("\n" + "=" * 60)
        print("  v27 XRPPM17 directional & outcome-based — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V17_CONFIG)

        print("\n" + "=" * 60)
        print("  v27 XRPPM17 directional & outcome-based — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V17_2Y_CONFIG)

        print("\n" + "=" * 60)
        print("  v27 XRPPM17 directional & outcome-based — mid-year  (Aug 2024 → Aug 2025)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V17_1Y_MID_CONFIG)
    elif symbol in ("XRPPM18", "PM18"):
        print("\n" + "=" * 60)
        print("  v28 XRPPM18 push dirvel limits — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V18_CONFIG)

        print("\n" + "=" * 60)
        print("  v28 XRPPM18 push dirvel limits — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V18_2Y_CONFIG)

        print("\n" + "=" * 60)
        print("  v28 XRPPM18 push dirvel limits — mid-year  (Aug 2024 → Aug 2025)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V18_1Y_MID_CONFIG)
    elif symbol in ("XRPPM19", "PM19"):
        print("\n" + "=" * 60)
        print("  v29 XRPPM19 combination sweep — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V19_CONFIG)

        print("\n" + "=" * 60)
        print("  v29 XRPPM19 combination sweep — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V19_2Y_CONFIG)

        print("\n" + "=" * 60)
        print("  v29 XRPPM19 combination sweep — mid-year  (Aug 2024 → Aug 2025)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V19_1Y_MID_CONFIG)
    elif symbol in ("XRPPM20", "PM20"):
        print("\n" + "=" * 60)
        print("  v30 XRPPM20 mult fine-tune — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V20_CONFIG)

        print("\n" + "=" * 60)
        print("  v30 XRPPM20 mult fine-tune — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V20_2Y_CONFIG)

        print("\n" + "=" * 60)
        print("  v30 XRPPM20 mult fine-tune — mid-year  (Aug 2024 → Aug 2025)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V20_1Y_MID_CONFIG)
    elif symbol in ("XRPPM21", "PM21"):
        print("\n" + "=" * 60)
        print("  v31 XRPPM21 fine-grain cliff — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V21_CONFIG)

        print("\n" + "=" * 60)
        print("  v31 XRPPM21 fine-grain cliff — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V21_2Y_CONFIG)

        print("\n" + "=" * 60)
        print("  v31 XRPPM21 fine-grain cliff — mid-year  (Aug 2024 → Aug 2025)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V21_1Y_MID_CONFIG)
    elif symbol in ("XRPPM22", "PM22"):
        print("\n" + "=" * 60)
        print("  v32 XRPPM22 combos at m1.6 — 6-month OOS  (Aug 2025 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V22_CONFIG)

        print("\n" + "=" * 60)
        print("  v32 XRPPM22 combos at m1.6 — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V22_2Y_CONFIG)

        print("\n" + "=" * 60)
        print("  v32 XRPPM22 combos at m1.6 — mid-year  (Aug 2024 → Aug 2025)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V22_1Y_MID_CONFIG)
    elif symbol in ("XRPPM21FULL", "PM21FULL"):
        print("\n" + "=" * 60)
        print("  PM21 FULL — baseline vs m166 across 6.5yr stitched cache  (Oct 2019 → Mar 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V21_FULL_CONFIG)
    elif symbol in ("XRPPM23FULL", "PM23FULL"):
        print("\n" + "=" * 60)
        print("  v33 PM23 FULL — gate-relaxation combined sweep  (Oct 2019 → Mar 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V23_FULL_CONFIG)
    elif symbol in ("XRPPM24FULL", "PM24FULL"):
        print("\n" + "=" * 60)
        print("  v34 PM24 FULL — 5-regime rotation sweep  (May 2017 → Mar 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V24_FULL_CONFIG)
    elif symbol in ("XRPPM25FULL", "PM25FULL"):
        print("\n" + "=" * 60)
        print("  v35 PM25 FULL — regime detection quality sweep  (May 2017 → Mar 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_PM_V25_FULL_CONFIG)
    elif symbol in ("XRPCB", "CB"):
        print("\n" + "=" * 60)
        print("  v11 Crash protection — 3.9-year MAX history  (Apr 2022 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_CB_CONFIG)

        print("\n" + "=" * 60)
        print("  v11 Crash protection — 2-year walk-forward  (Feb 2024 → Feb 2026)")
        print("=" * 60)
        grid_search_backtest(XRP_CB_2Y_CONFIG)
    else:
        # Run both
        print("\n" + "=" * 60)
        print("  BTCUSDT backtest")
        print("=" * 60)
        grid_search_backtest(CONFIG)

        print("\n" + "=" * 60)
        print("  XRPUSDT backtest  ← live target")
        print("=" * 60)
        grid_search_backtest(XRP_CONFIG)

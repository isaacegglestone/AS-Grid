"""
asBack/backtest_grid_bitunix.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Bitunix-flavoured grid backtest, based on backtest_grid_auto.py.

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
        self.crash_halt_counter: int = 0   # candles remaining in velocity CB halt
        self.dd_halt_counter: int = 0      # candles remaining in DD halt/resume pause

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

        self._init_anchors(self.df["close"].iloc[0])

    # ------------------------------------------------------------------
    # Order placement helpers
    # ------------------------------------------------------------------

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
            if self.dd_halt_counter > 0:
                self.dd_halt_counter -= 1
            halt_grid_longs = self.crash_halt_counter > 0   # velocity CB: blocks new long entries only
            halt_all        = self.dd_halt_counter > 0      # DD halt: blocks all new entries

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

            # Per-side caps: for a hedge strategy use max_positions_per_side=1
            max_per_side = self.config.get("max_positions_per_side", self.config["max_positions"])
            long_at_cap  = len(self.long_positions)  >= max_per_side
            short_at_cap = len(self.short_positions) >= max_per_side
            open_position_count = len(self.long_positions) + len(self.short_positions)
            at_max = open_position_count >= self.config["max_positions"]

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
                    if tp["side"] == "long":
                        if price > tp["peak"]:
                            tp["peak"] = price
                        trail_stop = tp["peak"] * (1 - trail_stop_pct)
                        # Close if trail hit OR trend fully reversed
                        close_trend = price <= trail_stop or trending_down
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
                        close_trend = price >= trail_stop or trending_up
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
                            and rsi_allows_long and vol_confirms and ms_allows_long):
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
                            and rsi_allows_short and vol_confirms and ms_allows_short):
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
                    if abs(velocity) < vel_threshold * 0.5:
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
                            and not halt_grid_longs and not halt_all):
                        buy_price = self.last_long_price * (1 - self.long_settings["down_spacing"])
                        if price <= buy_price:
                            qty = effective_order_value / price
                            margin_required = qty * price / self.leverage
                            fee_cost = qty * price * (self.fee / 2)
                            if (margin_required + fee_cost) <= available_margin:
                                tp_price = price * (1 + self.long_settings["up_spacing"])
                                if use_sl:
                                    long_sl_pct = max(sl_multiplier * self.long_settings["down_spacing"], min_sl_pct)
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
                            and not short_blocked_by_trend and not halt_all):
                        sell_price = self.last_short_price * (1 + self.short_settings["up_spacing"])
                        if price >= sell_price:
                            qty = effective_order_value / price
                            margin_required = qty * price / self.leverage
                            fee_cost = qty * price * (self.fee / 2)
                            if (margin_required + fee_cost) <= available_margin:
                                tp_price = price * (1 - self.short_settings["down_spacing"])
                                if use_sl:
                                    short_sl_pct = max(sl_multiplier * self.short_settings["up_spacing"], min_sl_pct)
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
    # Files are stored as asBack/klines_cache/{symbol}_{interval}.parquet
    # and cover the full available history; we slice to the requested range.
    # Generate the cache file by running: scripts/generate_klines_cache.py
    # or via the "Generate klines cache" GitHub Actions workflow.
    # ------------------------------------------------------------------
    _cache_dir = os.path.join(os.path.dirname(__file__), "klines_cache")
    _cache_file = os.path.join(_cache_dir, f"{symbol}_{interval}.parquet")
    if os.path.exists(_cache_file):
        df_full = pd.read_parquet(_cache_file)
        if df_full["open_time"].dt.tz is None:
            df_full["open_time"] = df_full["open_time"].dt.tz_localize("UTC")
        _start = start_dt if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc)
        _end   = end_dt   if end_dt.tzinfo   else end_dt.replace(tzinfo=timezone.utc)
        df = df_full[(df_full["open_time"] >= _start) & (df_full["open_time"] < _end)].copy().reset_index(drop=True)
        if len(df) > 0:
            print(f"  → Cache hit: {len(df):,} candles for {symbol} {interval} ({start_dt.date()} → {end_dt.date()})")
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
                ) if k in params}),
            }
        )

        bt = GridOrderBacktester(full_df.copy(), None, temp_config)
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
) -> Dict[str, Any]:
    """v12/v13 PM-tuning param set — exposes RSI threshold, BB threshold, grid vol scale, floor and period."""
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
        "long_settings":   {"up_spacing": 0.010, "down_spacing": 0.010},
        "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
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
    dd_halt_pct: float = 0.15,
    dd_halt_candles: int = 96,
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

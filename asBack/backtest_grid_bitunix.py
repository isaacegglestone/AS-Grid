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

            self._refresh_orders_if_needed(price, timestamp)

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
                    if trend_capture and self.trend_position is None:
                        if velocity >= cap_vel_threshold:
                            current_equity = self._equity(price)
                            cap_margin = current_equity * cap_size_pct
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
                                print(f"\U0001f3af [{timestamp}] Trend LONG opened "
                                      f"at {price:.4f} size=${cap_margin:.0f}")

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
                    if trend_capture and self.trend_position is None:
                        if velocity <= -cap_vel_threshold:
                            current_equity = self._equity(price)
                            cap_margin = current_equity * cap_size_pct
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
                                print(f"\U0001f3af [{timestamp}] Trend SHORT opened "
                                      f"at {price:.4f} size=${cap_margin:.0f}")

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

            # Gate new grid entries when a trend is active
            long_blocked_by_trend  = (self.trend_mode == "down")
            short_blocked_by_trend = (self.trend_mode == "up")
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
                    if not long_at_cap and not long_loss_tripped and not long_blocked_by_trend:
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
                    if not short_at_cap and not short_loss_tripped and not short_blocked_by_trend:
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

            if drawdown >= self.max_drawdown:
                print(f"⚠️ Max drawdown {drawdown * 100:.2f}% reached – stopping")
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

    # ── 6-month validation: baseline vs protect-only vs capture-soft vs capture-hard
    "param_sets": [
        {
            "name": "xrp_trend_off",
            "use_sl": True,
            "trend_detection": False,        # baseline: grid only, no trend logic
            "trend_capture": False,
            "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
            "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
        },
        {
            "name": "xrp_protect_only",
            "use_sl": True,
            "trend_detection": True,
            "trend_capture": False,          # block entries on losing side only
            "trend_force_close_grid": False,
            "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
            "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
        },
        {
            "name": "xrp_capture_soft",
            "use_sl": True,
            "trend_detection": True,
            "trend_capture": True,
            "trend_force_close_grid": False, # keep grid open, just ADD trend position
            "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
            "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
        },
        {
            "name": "xrp_capture_hard",
            "use_sl": True,
            "trend_detection": True,
            "trend_capture": True,
            "trend_force_close_grid": True,  # force-close losing side + ride trend
            "long_settings":  {"up_spacing": 0.010, "down_spacing": 0.010},
            "short_settings": {"up_spacing": 0.010, "down_spacing": 0.010},
        },
    ],
}

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
        print("  XRPUSDT backtest")
        print("=" * 60)
        grid_search_backtest(XRP_CONFIG)
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

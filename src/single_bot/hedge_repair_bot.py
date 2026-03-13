"""
src/single_bot/hedge_repair_bot.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Live hedge-repair bot — ports the backtester's state machine to work with
a real (or simulated) exchange adapter.

State machine: ``IDLE`` → ``BOTH_OPEN`` → ``REPAIRING`` → ``IDLE``

The bot opens simultaneous long + short positions at the lookback-window
midpoint, trails profits with a configurable threshold, and DCA-repairs
the losing side when price levels off or approaches liquidation.

Supports:
 - Dynamic threshold (cycle_fees + net_profit_target)
 - DCA repair with spot borrowing
 - Periodic capital injection (DCA)
 - Simulation mode (via SimulatedExchange)
 - Telegram notifications

Usage — live::

    bot = HedgeRepairBot(exchange, config)
    await bot.run()

Usage — simulation::

    from src.exchange.simulated import SimulatedExchange
    sim = SimulatedExchange(initial_balance=1000, fee_pct=0.0006)
    bot = HedgeRepairBot(sim, config)
    result = await bot.run_simulation(candles_df)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration from env vars (with defaults matching proven PM43 5x 10c)
# ---------------------------------------------------------------------------

def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Position tracking dataclass
# ---------------------------------------------------------------------------

@dataclass
class HedgePosition:
    """Tracks one side (long or short) of the hedge pair."""

    side: str           # "long" or "short"
    qty: float = 0.0
    avg_entry: float = 0.0
    margin: float = 0.0
    dca_count: int = 0
    peak_profit: float = 0.0
    trailing_active: bool = False
    entry_time: float = 0.0
    order_id: Optional[str] = None


# ---------------------------------------------------------------------------
# HedgeRepairBot
# ---------------------------------------------------------------------------

class HedgeRepairBot:
    """
    Live hedge-repair trading bot.

    Parameters
    ----------
    exchange
        An exchange adapter (``BitunixExchange`` or ``SimulatedExchange``).
    config : dict
        Configuration dict. Falls back to env vars for missing keys.
    """

    IDLE = "IDLE"
    BOTH_OPEN = "BOTH_OPEN"
    REPAIRING = "REPAIRING"

    def __init__(self, exchange: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self.exchange = exchange
        cfg = config or {}

        # ── core params ──────────────────────────────────────────────────
        self.symbol: str = cfg.get("symbol", os.getenv("COIN_NAME", "XRP")) + "USDT"
        self.coin_name: str = cfg.get("symbol", os.getenv("COIN_NAME", "XRP"))
        self.leverage: int = cfg.get("leverage", _env_int("LEVERAGE", 5))
        self.entry_pct: float = cfg.get("entry_pct", _env_float("ENTRY_PCT", 0.05))
        self.fee_pct: float = cfg.get("fee_pct", _env_float("FEE_PCT", 0.0006))

        # ── threshold & trailing ─────────────────────────────────────────
        self.profit_threshold: float = cfg.get(
            "profit_threshold", _env_float("PROFIT_THRESHOLD", 5.0)
        )
        self.trailing_distance: float = cfg.get(
            "trailing_distance", _env_float("TRAILING_DISTANCE", 1.0)
        )
        self.dynamic_threshold: bool = cfg.get(
            "dynamic_threshold", _env_bool("DYNAMIC_THRESHOLD", False)
        )
        self.net_profit_target: float = cfg.get(
            "net_profit_target", _env_float("NET_PROFIT_TARGET", 0.10)
        )

        # ── DCA repair ───────────────────────────────────────────────────
        self.dca_multiplier: float = cfg.get(
            "dca_multiplier", _env_float("DCA_MULTIPLIER", 1.0)
        )
        self.liq_proximity_pct: float = cfg.get(
            "liq_proximity_pct", _env_float("LIQ_PROXIMITY_PCT", 0.80)
        )
        self.zig_zag_candles: int = cfg.get(
            "zig_zag_candles", _env_int("ZIG_ZAG_CANDLES", 20)
        )
        self.zig_zag_threshold: float = cfg.get(
            "zig_zag_threshold", _env_float("ZIG_ZAG_THRESHOLD", 0.01)
        )

        # ── lookback ─────────────────────────────────────────────────────
        self.lookback_candles: int = cfg.get(
            "lookback_candles", _env_int("LOOKBACK_CANDLES", 672)
        )

        # ── spot reserve ─────────────────────────────────────────────────
        self.spot_reserve: float = cfg.get(
            "spot_reserve", _env_float("SPOT_RESERVE", 1000.0)
        )
        self.spot_balance: float = self.spot_reserve
        self.spot_borrowed: float = 0.0

        # ── periodic injection ───────────────────────────────────────────
        self.periodic_injection_usd: float = cfg.get(
            "periodic_injection_usd", _env_float("PERIODIC_INJECTION_USD", 0.0)
        )
        self.injection_interval_secs: float = cfg.get(
            "injection_interval_secs",
            _env_float("INJECTION_INTERVAL_SECS", 14 * 24 * 3600),  # 2 weeks
        )
        self.injection_futures_pct: float = cfg.get(
            "injection_futures_pct", _env_float("INJECTION_FUTURES_PCT", 0.5)
        )

        # ── polling interval ─────────────────────────────────────────────
        self.poll_interval_secs: float = cfg.get(
            "poll_interval_secs", _env_float("POLL_INTERVAL_SECS", 60.0)
        )

        # ── notifications ────────────────────────────────────────────────
        self.telegram_bot_token: str = cfg.get(
            "telegram_bot_token", os.getenv("TELEGRAM_BOT_TOKEN", "")
        )
        self.telegram_chat_id: str = cfg.get(
            "telegram_chat_id", os.getenv("TELEGRAM_CHAT_ID", "")
        )
        self.enable_notifications: bool = cfg.get(
            "enable_notifications",
            _env_bool("ENABLE_NOTIFICATIONS", False),
        )

        # ── runtime state ────────────────────────────────────────────────
        self.state: str = self.IDLE
        self.long_pos: Optional[HedgePosition] = None
        self.short_pos: Optional[HedgePosition] = None
        self.cycle_fees: float = 0.0
        self.total_funding: float = 0.0

        # ── tracking ─────────────────────────────────────────────────────
        self.cycles_completed: int = 0
        self.total_dca_count: int = 0
        self.total_spot_borrows: int = 0
        self.total_fees: float = 0.0
        self.total_injected: float = 0.0
        self.injection_count: int = 0
        self.liquidated: bool = False
        self.trade_log: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []

        # ── price history (for zig-zag + lookback) ───────────────────────
        self._price_ring: List[Dict[str, float]] = []
        self._max_ring_size: int = max(self.lookback_candles, self.zig_zag_candles) + 10
        self._last_injection_time: float = 0.0

    # ══════════════════════════════════════════════════════════════════════
    # Setup
    # ══════════════════════════════════════════════════════════════════════

    async def setup(self) -> None:
        """Configure exchange: leverage + hedge mode, seed price history."""
        await self.exchange.set_leverage(self.symbol, self.leverage)
        try:
            await self.exchange.set_position_mode(True)  # hedge mode
        except Exception as exc:
            logger.warning("set_position_mode failed (may already be set): %s", exc)

        # Seed price ring with historical klines
        candles = await self.exchange.get_klines(
            self.symbol, interval="15min", limit=min(self.lookback_candles, 200)
        )
        for c in candles:
            self._price_ring.append({
                "high": c["high"], "low": c["low"], "close": c["close"]
            })
        if len(self._price_ring) > self._max_ring_size:
            self._price_ring = self._price_ring[-self._max_ring_size:]

        logger.info(
            "HedgeRepairBot setup: %s %dx entry=%s%% threshold=%s trail=%s",
            self.symbol, self.leverage,
            self.entry_pct * 100, self._effective_threshold_label(),
            self.trailing_distance,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Core logic — threshold / trailing / profit
    # ══════════════════════════════════════════════════════════════════════

    def _effective_threshold(self) -> float:
        if self.dynamic_threshold:
            return self.cycle_fees + self.net_profit_target
        return self.profit_threshold

    def _effective_threshold_label(self) -> str:
        if self.dynamic_threshold:
            return f"dynamic(fees+${self.net_profit_target:.2f})"
        return f"fixed(${self.profit_threshold:.2f})"

    def _calc_profit(self, pos: HedgePosition, price: float) -> float:
        if pos.side == "long":
            return (price - pos.avg_entry) * pos.qty
        return (pos.avg_entry - price) * pos.qty

    def _trailing_triggered(self, pos: HedgePosition, worst_profit: float) -> bool:
        if not pos.trailing_active:
            return False
        return worst_profit <= pos.peak_profit - self.trailing_distance

    def _stop_price(self, pos: HedgePosition) -> float:
        stop_profit = pos.peak_profit - self.trailing_distance
        if pos.side == "long":
            return pos.avg_entry + stop_profit / pos.qty if pos.qty > 0 else pos.avg_entry
        return pos.avg_entry - stop_profit / pos.qty if pos.qty > 0 else pos.avg_entry

    # ══════════════════════════════════════════════════════════════════════
    # Entry / lookback
    # ══════════════════════════════════════════════════════════════════════

    def _lookback_mid(self) -> Optional[float]:
        """Calculate midpoint of the lookback window from the price ring."""
        if len(self._price_ring) < self.lookback_candles:
            return None
        window = self._price_ring[-self.lookback_candles:]
        hi = max(c["high"] for c in window)
        lo = min(c["low"] for c in window)
        return (hi + lo) / 2.0

    def _check_zig_zag(self) -> bool:
        """True when recent price range is tight (price levelling off)."""
        if len(self._price_ring) < self.zig_zag_candles:
            return False
        window = self._price_ring[-self.zig_zag_candles:]
        hi = max(c["high"] for c in window)
        lo = min(c["low"] for c in window)
        mid = (hi + lo) / 2.0
        if mid <= 0:
            return False
        return (hi - lo) / mid < self.zig_zag_threshold

    # ══════════════════════════════════════════════════════════════════════
    # Exchange operations
    # ══════════════════════════════════════════════════════════════════════

    async def _open_both(self, price: float) -> bool:
        """Open equal-size long + short at *price* via the exchange."""
        bal = await self.exchange.get_balance()
        free = bal["free"]

        margin_per_side = free * self.entry_pct
        notional = margin_per_side * self.leverage
        qty = notional / price

        # Minimum quantity guard (XRP = 1 contract minimum)
        if qty < 1:
            logger.warning("Computed qty %.4f < 1 — insufficient balance", qty)
            return False

        qty = round(qty, 0)  # whole contracts for XRP
        notional = qty * price
        margin_per_side = notional / self.leverage
        fee_per_side = notional * self.fee_pct
        total_cost = 2 * (margin_per_side + fee_per_side)

        if free < total_cost:
            logger.warning("Insufficient balance: need %.2f, have %.2f", total_cost, free)
            return False

        # Place both market orders
        long_result = await self.exchange.place_market_order(
            self.symbol, "buy", qty, reduce_only=False
        )
        if not long_result:
            logger.error("Failed to open long position")
            return False

        short_result = await self.exchange.place_market_order(
            self.symbol, "sell", qty, reduce_only=False
        )
        if not short_result:
            # Rollback: close the long
            await self.exchange.place_market_order(
                self.symbol, "sell", qty, reduce_only=True
            )
            logger.error("Failed to open short position — rolled back long")
            return False

        self.cycle_fees = 2 * fee_per_side
        self.total_fees += 2 * fee_per_side

        now = time.time()
        self.long_pos = HedgePosition(
            side="long", qty=qty, avg_entry=price, margin=margin_per_side,
            entry_time=now, order_id=long_result.get("orderId"),
        )
        self.short_pos = HedgePosition(
            side="short", qty=qty, avg_entry=price, margin=margin_per_side,
            entry_time=now, order_id=short_result.get("orderId"),
        )

        self.trade_log.append({
            "action": "OPEN_PAIR", "price": price, "qty": qty,
            "margin_per_side": margin_per_side, "fee": 2 * fee_per_side,
            "time": now,
        })

        logger.info(
            "OPEN_PAIR @ %.6f qty=%.0f margin=%.2f/side fee=%.4f",
            price, qty, margin_per_side, 2 * fee_per_side,
        )
        return True

    async def _close_position(self, pos: HedgePosition, price: float, reason: str = "trailing_stop") -> float:
        """Close a position via market order and return net profit."""
        side = "sell" if pos.side == "long" else "buy"
        result = await self.exchange.place_market_order(
            self.symbol, side, pos.qty, reduce_only=True
        )

        profit = self._calc_profit(pos, price)
        fee = pos.qty * price * self.fee_pct
        self.total_fees += fee
        net = profit - fee

        self.trade_log.append({
            "action": f"CLOSE_{pos.side.upper()}", "price": price,
            "qty": pos.qty, "profit": profit, "fee": fee, "net": net,
            "reason": reason, "dca_count": pos.dca_count, "time": time.time(),
        })

        logger.info(
            "CLOSE_%s @ %.6f qty=%.0f profit=%.2f fee=%.4f net=%.2f (%s)",
            pos.side.upper(), price, pos.qty, profit, fee, net, reason,
        )
        return net

    async def _dca_repair(self, pos: HedgePosition, price: float) -> bool:
        """DCA: add dca_multiplier × current qty at current price."""
        dca_qty = round(pos.qty * self.dca_multiplier, 0)
        if dca_qty < 1:
            dca_qty = 1

        dca_notional = dca_qty * price
        dca_margin = dca_notional / self.leverage
        dca_fee = dca_notional * self.fee_pct
        needed = dca_margin + dca_fee

        # Check balance
        bal = await self.exchange.get_balance()
        available = bal["free"]

        if available < needed:
            shortfall = needed - available
            borrow = min(shortfall, self.spot_balance)
            if borrow > 0:
                await self.exchange.transfer_spot_to_futures(borrow)
                self.spot_balance -= borrow
                self.spot_borrowed += borrow
                self.total_spot_borrows += 1
                logger.info("Transferred $%.2f spot → futures (remaining: $%.2f)",
                            borrow, self.spot_balance)

            # Re-check
            new_bal = await self.exchange.get_balance()
            if new_bal["free"] + borrow < needed:
                logger.warning("Cannot DCA — insufficient funds even with spot borrow")
                return False

        # Place market order for DCA
        side = "buy" if pos.side == "long" else "sell"
        result = await self.exchange.place_market_order(
            self.symbol, side, dca_qty, reduce_only=False
        )
        if not result:
            logger.error("DCA market order failed")
            return False

        # Update position
        old_notional = pos.qty * pos.avg_entry
        new_notional = dca_qty * price
        pos.avg_entry = (old_notional + new_notional) / (pos.qty + dca_qty)
        pos.qty += dca_qty
        pos.margin += dca_margin
        pos.dca_count += 1
        pos.peak_profit = self._calc_profit(pos, price)
        pos.trailing_active = False

        self.total_fees += dca_fee
        self.cycle_fees += dca_fee
        self.total_dca_count += 1

        self.trade_log.append({
            "action": f"DCA_{pos.side.upper()}", "price": price,
            "qty": dca_qty, "new_avg": pos.avg_entry,
            "total_qty": pos.qty, "fee": dca_fee,
            "dca_number": pos.dca_count, "time": time.time(),
        })

        logger.info(
            "DCA_%s @ %.6f +%d → total %d avg=%.6f",
            pos.side.upper(), price, dca_qty, int(pos.qty), pos.avg_entry,
        )
        return True

    async def _repay_spot(self) -> None:
        """Repay any spot borrowed funds by transferring futures → spot."""
        if self.spot_borrowed > 0:
            bal = await self.exchange.get_balance()
            repay = min(self.spot_borrowed, bal["free"])
            if repay > 0:
                await self.exchange.transfer_futures_to_spot(repay)
                self.spot_balance += repay
                self.spot_borrowed -= repay
                logger.info("Transferred $%.2f futures → spot (remaining debt: $%.2f)",
                            repay, self.spot_borrowed)

    # ══════════════════════════════════════════════════════════════════════
    # Main candle processing — the state machine
    # ══════════════════════════════════════════════════════════════════════

    async def process_candle(self, candle: Dict[str, Any]) -> None:
        """
        Process a single 15-min candle through the state machine.

        This is the core tick — called from ``run()`` (live polling) or
        ``run_simulation()`` (historical replay).
        """
        price = candle["close"]
        high = candle["high"]
        low = candle["low"]

        # Update price ring
        self._price_ring.append({"high": high, "low": low, "close": price})
        if len(self._price_ring) > self._max_ring_size:
            self._price_ring.pop(0)

        if self.liquidated:
            return

        # ── IDLE: look for entry ─────────────────────────────────────────
        if self.state == self.IDLE:
            mid = self._lookback_mid()
            if mid is not None and low <= mid <= high:
                if await self._open_both(mid):
                    self.state = self.BOTH_OPEN
                    logger.info("State → BOTH_OPEN (mid=%.6f)", mid)

        # ── BOTH_OPEN: trail profits, look for close trigger ─────────────
        elif self.state == self.BOTH_OPEN:
            if self.long_pos is None or self.short_pos is None:
                # Shouldn't happen, but recover
                self.state = self.REPAIRING
                return

            lp, sp = self.long_pos, self.short_pos

            # Best/worst intra-candle profits
            l_best = (high - lp.avg_entry) * lp.qty
            s_best = (sp.avg_entry - low) * sp.qty
            l_worst = (low - lp.avg_entry) * lp.qty
            s_worst = (sp.avg_entry - high) * sp.qty

            lp.peak_profit = max(lp.peak_profit, l_best)
            sp.peak_profit = max(sp.peak_profit, s_best)

            threshold = self._effective_threshold()
            if lp.peak_profit >= threshold:
                lp.trailing_active = True
            if sp.peak_profit >= threshold:
                sp.trailing_active = True

            l_trig = self._trailing_triggered(lp, l_worst)
            s_trig = self._trailing_triggered(sp, s_worst)

            if l_trig and s_trig:
                # Both triggered — close the more profitable side
                if l_best >= s_best:
                    await self._close_position(lp, self._stop_price(lp))
                    self.long_pos = None
                else:
                    await self._close_position(sp, self._stop_price(sp))
                    self.short_pos = None
                self.state = self.REPAIRING
                logger.info("State → REPAIRING (both trails triggered)")
            elif l_trig:
                await self._close_position(lp, self._stop_price(lp))
                self.long_pos = None
                self.state = self.REPAIRING
                logger.info("State → REPAIRING (long trail triggered)")
            elif s_trig:
                await self._close_position(sp, self._stop_price(sp))
                self.short_pos = None
                self.state = self.REPAIRING
                logger.info("State → REPAIRING (short trail triggered)")

        # ── REPAIRING: trail + DCA the remaining side ────────────────────
        elif self.state == self.REPAIRING:
            remaining = self.long_pos or self.short_pos
            if remaining is None:
                await self._repay_spot()
                self.cycles_completed += 1
                self.state = self.IDLE
                logger.info("State → IDLE (cycle #%d complete)", self.cycles_completed)
                return

            if remaining.side == "long":
                best_p = (high - remaining.avg_entry) * remaining.qty
                worst_p = (low - remaining.avg_entry) * remaining.qty
            else:
                best_p = (remaining.avg_entry - low) * remaining.qty
                worst_p = (remaining.avg_entry - high) * remaining.qty

            remaining.peak_profit = max(remaining.peak_profit, best_p)
            if remaining.peak_profit >= self._effective_threshold():
                remaining.trailing_active = True

            if self._trailing_triggered(remaining, worst_p):
                await self._close_position(remaining, self._stop_price(remaining))
                if remaining.side == "long":
                    self.long_pos = None
                else:
                    self.short_pos = None
                await self._repay_spot()
                self.cycles_completed += 1
                self.state = self.IDLE
                logger.info("State → IDLE (repair complete, cycle #%d)", self.cycles_completed)
                return

            # DCA triggers
            cur_profit = self._calc_profit(remaining, price)

            # Emergency DCA — approaching liquidation
            if (remaining.margin > 0
                    and abs(cur_profit) > remaining.margin * self.liq_proximity_pct):
                logger.warning("Emergency DCA — loss %.2f exceeds %.0f%% of margin %.2f",
                               cur_profit, self.liq_proximity_pct * 100, remaining.margin)
                await self._dca_repair(remaining, price)

            # Zig-zag DCA — price levelling off while losing
            elif cur_profit < 0 and self._check_zig_zag():
                logger.info("Zig-zag DCA — price flat, position losing %.2f", cur_profit)
                await self._dca_repair(remaining, price)

        # ── Periodic injection ───────────────────────────────────────────
        now = time.time()
        if (self.periodic_injection_usd > 0
                and now - self._last_injection_time >= self.injection_interval_secs):
            if self._last_injection_time > 0:  # Skip first tick
                fut_share = self.periodic_injection_usd * self.injection_futures_pct
                spot_share = self.periodic_injection_usd - fut_share
                self.spot_balance += spot_share
                self.total_injected += self.periodic_injection_usd
                self.injection_count += 1
                logger.info(
                    "Injected $%.2f (futures=$%.2f spot=$%.2f) — total injected: $%.2f (#%d)",
                    self.periodic_injection_usd, fut_share, spot_share,
                    self.total_injected, self.injection_count,
                )
            self._last_injection_time = now

        # ── Equity tracking ──────────────────────────────────────────────
        bal = await self.exchange.get_balance()
        total_eq = bal["total"] + self.spot_balance
        self.equity_curve.append({
            "time": now, "price": price, "total_equity": total_eq,
            "futures_equity": bal["total"],
            "spot_balance": self.spot_balance, "state": self.state,
        })

    # ══════════════════════════════════════════════════════════════════════
    # Live run — polling loop
    # ══════════════════════════════════════════════════════════════════════

    async def run(self) -> None:
        """
        Main live loop: poll 15-min klines and process through state machine.
        """
        await self.setup()
        logger.info("HedgeRepairBot live loop starting for %s", self.symbol)

        last_candle_time: int = 0

        while True:
            try:
                candles = await self.exchange.get_klines(
                    self.symbol, interval="15min", limit=2
                )
                if not candles:
                    await asyncio.sleep(self.poll_interval_secs)
                    continue

                latest = candles[-1]
                ct = latest.get("open_time", 0)

                if ct > last_candle_time:
                    last_candle_time = ct
                    await self.process_candle(latest)

                await asyncio.sleep(self.poll_interval_secs)

            except KeyboardInterrupt:
                logger.info("Received stop signal — shutting down.")
                break
            except Exception as exc:
                logger.error("Error in run loop: %s — retrying in 30s", exc)
                await asyncio.sleep(30)

    # ══════════════════════════════════════════════════════════════════════
    # Simulation mode — replay historical or forward candles
    # ══════════════════════════════════════════════════════════════════════

    async def run_simulation(
        self,
        candles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run the bot against a list of candle dicts (historical or forward).

        Each candle must have keys: ``open_time``, ``open``, ``high``,
        ``low``, ``close``, ``volume``.

        Returns a results dict compatible with backtester output format.
        """
        from src.exchange.simulated import SimulatedExchange

        if not isinstance(self.exchange, SimulatedExchange):
            raise TypeError("run_simulation requires a SimulatedExchange")

        logger.info(
            "Simulation: %d candles, %s %dx, threshold=%s",
            len(candles), self.symbol, self.leverage,
            self._effective_threshold_label(),
        )

        for candle in candles:
            # Feed price to the simulated exchange (for order matching)
            self.exchange.on_price_update(
                self.symbol, candle["close"],
                high=candle["high"], low=candle["low"],
            )
            await self.process_candle(candle)

            if self.liquidated:
                break

        # ── close any open positions at end ───────────────────────────────
        if not self.liquidated:
            final_price = candles[-1]["close"] if candles else 0.0
            if self.long_pos:
                await self._close_position(self.long_pos, final_price, "end_of_sim")
                self.long_pos = None
            if self.short_pos:
                await self._close_position(self.short_pos, final_price, "end_of_sim")
                self.short_pos = None
            await self._repay_spot()

        # ── compute results ──────────────────────────────────────────────
        bal = await self.exchange.get_balance()
        total_final = bal["total"] + self.spot_balance

        initial_total = bal["total"] if not self.equity_curve else (
            self.equity_curve[0]["total_equity"] if self.equity_curve else 0
        )
        # Use the exchange's starting balance + spot reserve
        initial_total = getattr(self.exchange, '_free', 0) + getattr(self.exchange, '_used', 0) + self.spot_reserve
        # Actually, use the recorded equity curve start
        if self.equity_curve:
            initial_total = self.equity_curve[0].get("total_equity", initial_total)

        total_deployed = initial_total + self.total_injected

        # Peak equity and max drawdown from equity curve
        peak_eq = initial_total
        max_dd = 0.0
        for pt in self.equity_curve:
            eq = pt["total_equity"]
            if eq > peak_eq:
                peak_eq = eq
            dd = (peak_eq - eq) / peak_eq if peak_eq > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        return_pct = (
            (total_final - total_deployed) / total_deployed
            if total_deployed > 0 else 0.0
        )

        return {
            "return_pct": return_pct,
            "total_return": total_final - total_deployed,
            "total_deployed": total_deployed,
            "total_injected": self.total_injected,
            "injection_count": self.injection_count,
            "final_total": total_final,
            "futures_final": bal["total"],
            "spot_final": self.spot_balance,
            "trades": len(self.trade_log),
            "cycles": self.cycles_completed,
            "dca_count": self.total_dca_count,
            "max_drawdown": max_dd,
            "total_fees": self.total_fees,
            "total_funding": self.total_funding,
            "spot_borrows": self.total_spot_borrows,
            "liquidated": self.liquidated,
        }

    # ══════════════════════════════════════════════════════════════════════
    # Status / reporting
    # ══════════════════════════════════════════════════════════════════════

    def status_summary(self) -> str:
        """Human-readable status string."""
        parts = [
            f"State: {self.state}",
            f"Cycles: {self.cycles_completed}",
            f"DCAs: {self.total_dca_count}",
            f"Spot reserve: ${self.spot_balance:.2f}",
            f"Spot borrowed: ${self.spot_borrowed:.2f}",
            f"Total fees: ${self.total_fees:.4f}",
        ]
        if self.long_pos:
            lp = self.long_pos
            parts.append(
                f"Long: qty={lp.qty} avg={lp.avg_entry:.6f} "
                f"dcas={lp.dca_count} peak=${lp.peak_profit:.2f} "
                f"trailing={lp.trailing_active}"
            )
        if self.short_pos:
            sp = self.short_pos
            parts.append(
                f"Short: qty={sp.qty} avg={sp.avg_entry:.6f} "
                f"dcas={sp.dca_count} peak=${sp.peak_profit:.2f} "
                f"trailing={sp.trailing_active}"
            )
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Bootstrap the hedge-repair bot from environment variables."""
    from src.exchange.bitunix import BitunixExchange

    simulation_mode = _env_bool("SIMULATION_MODE", False)

    if simulation_mode:
        from src.exchange.simulated import SimulatedExchange

        initial_bal = _env_float("INITIAL_BALANCE", 1000.0)
        sim = SimulatedExchange(initial_balance=initial_bal, fee_pct=0.0006)
        bot = HedgeRepairBot(sim)
        logger.info("Starting hedge-repair bot in SIMULATION mode")

        # Fetch forward candles
        candles = await sim.get_klines(
            bot.symbol, interval="15min", limit=200
        )
        if not candles:
            logger.error("No candle data available for simulation")
            return

        result = await bot.run_simulation(candles)
        logger.info("Simulation result: %s", result)
    else:
        api_key = os.getenv("API_KEY", "")
        api_secret = os.getenv("API_SECRET", "")
        if not api_key or not api_secret:
            logger.error("API_KEY and API_SECRET must be set")
            return

        exchange = BitunixExchange(api_key=api_key, secret_key=api_secret)
        bot = HedgeRepairBot(exchange)
        logger.info("Starting hedge-repair bot in LIVE mode")
        await bot.run()


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())

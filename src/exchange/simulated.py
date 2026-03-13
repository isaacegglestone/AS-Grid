"""
src/exchange/simulated.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Simulated exchange adapter for paper-trading / forward-testing.

Implements the same async interface as :class:`BitunixExchange` but never
touches a real exchange.  Balances, positions and order fills are tracked
internally; orders are filled when the price feed crosses their level.

Usage
-----
::

    from src.exchange.simulated import SimulatedExchange

    sim = SimulatedExchange(initial_balance=1000.0, fee_pct=0.0006)
    # Use exactly like BitunixExchange — all methods are async.
    await sim.set_leverage("XRPUSDT", 5)
    await sim.place_market_order("XRPUSDT", "buy", 10)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal order / position models
# ---------------------------------------------------------------------------

@dataclass
class SimOrder:
    """An in-flight limit order tracked by the simulator."""

    order_id: str
    symbol: str
    side: str          # "BUY" or "SELL"
    trade_side: str    # "OPEN" or "CLOSE"
    qty: float
    price: float
    reduce_only: bool
    status: str = "NEW"
    ctime: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "orderId": self.order_id,
            "clientId": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "orderType": "LIMIT",
            "price": self.price,
            "qty": self.qty,
            "remaining": self.qty,
            "reduceOnly": self.reduce_only,
            "status": self.status,
            "ctime": self.ctime,
        }


@dataclass
class SimPosition:
    """A synthetic position tracked by the simulator."""

    symbol: str
    side: str           # "LONG" or "SHORT"
    qty: float = 0.0
    avg_entry: float = 0.0
    margin: float = 0.0


# ---------------------------------------------------------------------------
# SimulatedExchange — drop-in replacement for BitunixExchange
# ---------------------------------------------------------------------------

class SimulatedExchange:
    """
    Paper-trading exchange adapter.

    Tracks balances, positions, and orders locally.  Limit orders fill when
    :meth:`on_price_update` is called with a price that crosses their level.
    Market orders fill immediately at the supplied price.

    Parameters
    ----------
    initial_balance : float
        Starting USDT balance (free).
    fee_pct : float
        Per-trade fee rate (e.g. 0.0006 = 0.06 %).
    """

    def __init__(
        self,
        initial_balance: float = 1000.0,
        fee_pct: float = 0.0006,
    ) -> None:
        self.api_key = "sim"
        self.secret_key = "sim"
        self.fee_pct = fee_pct

        # Balances
        self._free: float = initial_balance
        self._used: float = 0.0

        # Positions keyed by (symbol, side)
        self._positions: Dict[Tuple[str, str], SimPosition] = {}

        # Pending limit orders
        self._orders: Dict[str, SimOrder] = {}

        # Per-symbol leverage
        self._leverage: Dict[str, int] = {}

        # Position mode
        self._hedge_mode: bool = True

        # Trade log (for analysis)
        self.trade_log: List[Dict[str, Any]] = []
        self.total_fees: float = 0.0

    # ── properties ────────────────────────────────────────────────────────

    @property
    def total_balance(self) -> float:
        return self._free + self._used

    # ── account / balance ────────────────────────────────────────────────

    async def get_balance(self, margin_coin: str = "USDT") -> Dict[str, Any]:
        return {
            "total": self._free + self._used,
            "free": self._free,
            "used": self._used,
            "raw": None,
        }

    # ── positions ────────────────────────────────────────────────────────

    async def get_positions(self, symbol: str) -> Tuple[float, float]:
        lp = self._positions.get((symbol, "LONG"))
        sp = self._positions.get((symbol, "SHORT"))
        return (lp.qty if lp else 0.0, sp.qty if sp else 0.0)

    def get_position_detail(
        self, symbol: str, side: str
    ) -> Optional[SimPosition]:
        """Return the internal SimPosition object (or None)."""
        return self._positions.get((symbol, side.upper()))

    # ── leverage / position mode ─────────────────────────────────────────

    async def set_leverage(
        self, symbol: str, leverage: int, margin_coin: str = "USDT"
    ) -> None:
        self._leverage[symbol] = leverage
        logger.info("sim: set_leverage %s → %dx", symbol, leverage)

    async def set_position_mode(self, hedge_mode: bool) -> None:
        self._hedge_mode = hedge_mode
        logger.info("sim: set_position_mode → %s", "HEDGE" if hedge_mode else "ONE_WAY")

    # ── helpers ──────────────────────────────────────────────────────────

    def _get_leverage(self, symbol: str) -> int:
        return self._leverage.get(symbol, 1)

    def _apply_fill(
        self, symbol: str, side: str, qty: float, price: float, is_close: bool
    ) -> None:
        """
        Apply a fill to internal state.

        For opens: increase position, deduct margin + fee from free balance.
        For closes: decrease position, credit margin + pnl - fee to free.
        """
        lev = self._get_leverage(symbol)
        notional = qty * price
        fee = notional * self.fee_pct
        self.total_fees += fee

        pos_side = "LONG" if (
            (side == "BUY" and not is_close) or (side == "SELL" and is_close)
        ) else "SHORT"

        key = (symbol, pos_side)

        if not is_close:
            # ── OPEN ─────────────────────────────────────────────────────
            margin = notional / lev
            cost = margin + fee
            if self._free < cost:
                logger.warning("sim: insufficient balance for fill (need %.2f, have %.2f)", cost, self._free)
                return
            self._free -= cost
            self._used += margin

            pos = self._positions.get(key)
            if pos is None:
                pos = SimPosition(symbol=symbol, side=pos_side)
                self._positions[key] = pos

            # Weighted average entry
            old_notional = pos.qty * pos.avg_entry
            new_notional = qty * price
            pos.avg_entry = (
                (old_notional + new_notional) / (pos.qty + qty)
                if (pos.qty + qty) > 0
                else price
            )
            pos.qty += qty
            pos.margin += margin
        else:
            # ── CLOSE ────────────────────────────────────────────────────
            pos = self._positions.get(key)
            if pos is None or pos.qty <= 0:
                logger.warning("sim: close fill but no position for %s %s", symbol, pos_side)
                return

            close_qty = min(qty, pos.qty)
            close_margin = pos.margin * (close_qty / pos.qty) if pos.qty > 0 else 0.0

            # PnL
            if pos_side == "LONG":
                pnl = (price - pos.avg_entry) * close_qty
            else:
                pnl = (pos.avg_entry - price) * close_qty

            self._used -= close_margin
            self._free += close_margin + pnl - fee

            pos.qty -= close_qty
            pos.margin -= close_margin
            if pos.qty <= 1e-12:
                del self._positions[key]

        self.trade_log.append({
            "time": time.time(),
            "symbol": symbol,
            "side": side,
            "trade_side": "CLOSE" if is_close else "OPEN",
            "pos_side": pos_side,
            "qty": qty,
            "price": price,
            "fee": fee,
            "balance_after": self._free + self._used,
        })

    # ── order management ─────────────────────────────────────────────────

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        reduce_only: bool = False,
        position_side: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        oid = str(uuid.uuid4())[:12]
        ts = "CLOSE" if reduce_only else "OPEN"
        order = SimOrder(
            order_id=oid,
            symbol=symbol,
            side=side.upper(),
            trade_side=ts,
            qty=quantity,
            price=price,
            reduce_only=reduce_only,
            ctime=int(time.time() * 1000),
        )
        self._orders[oid] = order
        logger.debug("sim: placed limit %s %s %s qty=%.4f @%.6f", symbol, side, ts, quantity, price)
        return {"orderId": oid, "clientId": oid}

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
        position_side: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Market orders fill immediately at the current sim price."""
        oid = str(uuid.uuid4())[:12]
        is_close = reduce_only
        # Market orders need a price — use avg_entry for closes, or last known
        # The caller should call on_price_update before this for accuracy
        price = self._last_price if hasattr(self, "_last_price") else 0.0

        if is_close:
            side_upper = side.upper()
            pos_side = "LONG" if side_upper == "SELL" else "SHORT"
            pos = self._positions.get((symbol, pos_side))
            if pos:
                price = price or pos.avg_entry

        if price <= 0:
            logger.warning("sim: market order with no price reference")
            return None

        self._apply_fill(symbol, side.upper(), quantity, price, is_close)
        logger.debug("sim: market fill %s %s qty=%.4f @%.6f", symbol, side, quantity, price)
        return {"orderId": oid, "clientId": oid}

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        if order_id in self._orders:
            del self._orders[order_id]
            return True
        return False

    async def cancel_orders_for_side(
        self, symbol: str, position_side: str
    ) -> None:
        to_remove = []
        for oid, order in self._orders.items():
            if order.symbol != symbol:
                continue
            if position_side == "long":
                if (not order.reduce_only and order.side == "BUY") or (
                    order.reduce_only and order.side == "SELL"
                ):
                    to_remove.append(oid)
            elif position_side == "short":
                if (not order.reduce_only and order.side == "SELL") or (
                    order.reduce_only and order.side == "BUY"
                ):
                    to_remove.append(oid)
        for oid in to_remove:
            del self._orders[oid]

    async def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        return [
            o.to_dict()
            for o in self._orders.values()
            if o.symbol == symbol and o.status == "NEW"
        ]

    # ── market data (delegates to real Bitunix public API) ───────────────

    async def get_klines(
        self,
        symbol: str,
        interval: str = "1min",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Fetch real klines from Bitunix public API (no auth needed).

        This is the one method that hits the real exchange — simulation needs
        real price data for forward testing.
        """
        from src.exchange.bitunix import BitunixRestClient

        client = BitunixRestClient("", "")
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "type": "LAST_PRICE",
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        raw = await client.get_public(
            "/api/v1/futures/market/kline", params=params
        )
        candles: List[Dict[str, Any]] = []
        for item in raw or []:
            candles.append({
                "open_time": int(item["time"]),
                "open": float(item["open"]),
                "high": float(item["high"]),
                "low": float(item["low"]),
                "close": float(item["close"]),
                "volume": float(
                    item.get("baseVol") or item.get("vol") or item.get("volume") or 0
                ),
            })
        return candles

    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Fetch real funding rate from Bitunix public API."""
        from src.exchange.bitunix import BitunixRestClient

        client = BitunixRestClient("", "")
        data = await client.get_public(
            "/api/v1/futures/market/funding_rate",
            params={"symbol": symbol},
        )
        entry = data if isinstance(data, dict) else (data[0] if data else {})
        return {
            "rate": float(entry.get("fundingRate", 0)),
            "next_time": int(entry.get("nextFundingTime", 0)),
            "raw": entry,
        }

    # ── WebSocket stubs (not used in simulation) ─────────────────────────

    def build_ws_login_payload(self) -> Dict[str, Any]:
        return {"op": "login", "args": []}

    @staticmethod
    def build_subscribe_payload(channel: str, symbol: str) -> Dict[str, Any]:
        return {"op": "subscribe", "args": [{"symbol": symbol, "ch": channel}]}

    @staticmethod
    def build_ping_payload() -> Dict[str, Any]:
        return {"op": "ping", "ping": int(time.time())}

    # ── wallet transfers (spot ↔ futures) ────────────────────────────────

    async def transfer_spot_to_futures(
        self, amount: float, coin: str = "USDT"
    ) -> str:
        """Simulate spot → futures transfer by adding to free balance."""
        self._free += amount
        tid = str(uuid.uuid4())[:12]
        logger.info("sim: transfer %.2f %s spot → futures (id=%s)", amount, coin, tid)
        return tid

    async def transfer_futures_to_spot(
        self, amount: float, coin: str = "USDT"
    ) -> str:
        """Simulate futures → spot transfer by deducting from free balance."""
        self._free -= amount
        tid = str(uuid.uuid4())[:12]
        logger.info("sim: transfer %.2f %s futures → spot (id=%s)", amount, coin, tid)
        return tid

    # ── price feed / order matching ──────────────────────────────────────

    def on_price_update(self, symbol: str, price: float, high: float = 0.0, low: float = 0.0) -> List[Dict[str, Any]]:
        """
        Feed a new price tick into the simulator.

        Checks all pending limit orders for *symbol* and fills any whose
        price has been crossed.  Uses high/low for intra-candle fills when
        available, otherwise uses ``price`` for both.

        Returns a list of fill dicts for orders that matched.
        """
        self._last_price = price
        if high <= 0:
            high = price
        if low <= 0:
            low = price

        fills: List[Dict[str, Any]] = []
        to_remove: List[str] = []

        for oid, order in self._orders.items():
            if order.symbol != symbol:
                continue
            filled = False

            if order.side == "BUY" and low <= order.price:
                filled = True
            elif order.side == "SELL" and high >= order.price:
                filled = True

            if filled:
                is_close = order.reduce_only
                self._apply_fill(symbol, order.side, order.qty, order.price, is_close)
                order.status = "FILLED"
                to_remove.append(oid)
                fills.append({
                    "orderId": oid,
                    "side": order.side,
                    "qty": order.qty,
                    "price": order.price,
                    "is_close": is_close,
                })

        for oid in to_remove:
            del self._orders[oid]

        return fills

    # ── equity helpers ───────────────────────────────────────────────────

    def unrealised_pnl(self, symbol: str, price: float) -> float:
        """Calculate total unrealised PnL for all positions in *symbol*."""
        pnl = 0.0
        lp = self._positions.get((symbol, "LONG"))
        if lp and lp.qty > 0:
            pnl += (price - lp.avg_entry) * lp.qty
        sp = self._positions.get((symbol, "SHORT"))
        if sp and sp.qty > 0:
            pnl += (sp.avg_entry - price) * sp.qty
        return pnl

    def total_equity(self, symbol: str, price: float) -> float:
        """Free + used + unrealised PnL."""
        return self._free + self._used + self.unrealised_pnl(symbol, price)

    def reset(self, initial_balance: float) -> None:
        """Reset all state for a fresh simulation run."""
        self._free = initial_balance
        self._used = 0.0
        self._positions.clear()
        self._orders.clear()
        self.trade_log.clear()
        self.total_fees = 0.0
        self._last_price = 0.0

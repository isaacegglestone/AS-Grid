"""
src/exchange/bitunix.py
~~~~~~~~~~~~~~~~~~~~~~~
Bitunix futures REST + WebSocket adapter for AS-Grid.

Authentication
--------------
REST (per https://openapidoc.bitunix.com/doc/common/sign.html):
  1. Sort all query-param keys in ascending ASCII order and concatenate key+value
     pairs without separators, e.g. ``"id1uid200"``.
  2. Stringify the JSON body with all spaces removed.
  3. digest = SHA-256(nonce + timestamp_ms + api_key + sorted_query_str + body_str)
  4. sign   = SHA-256(digest + secret_key)
  Headers: api-key, nonce, timestamp (ms), sign, Content-Type: application/json

WebSocket login (private channels):
  1. digest = SHA-256(nonce + timestamp_sec + api_key)   ← seconds, not ms
  2. sign   = SHA-256(digest + secret_key)
  Send: {"op":"login","args":[{"apiKey":..,"timestamp":<sec>,"nonce":..,"sign":..}]}

References
----------
- https://openapidoc.bitunix.com/doc/common/introduction.html
- https://openapidoc.bitunix.com/doc/common/sign.html
- https://openapidoc.bitunix.com/doc/websocket/prepare/WebSocket.html
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import aiohttp

from .bitunix_models import (
    AccountBalanceEntry,
    CancelOrdersData,
    PendingOrder,
    PendingOrdersData,
    PlaceOrderData,
    PositionEntry,
    SetLeverageData,
    SetPositionModeData,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REST_BASE = "https://fapi.bitunix.com"
WS_PUBLIC = "wss://fapi.bitunix.com/public/"
WS_PRIVATE = "wss://fapi.bitunix.com/private/"


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    """Return the lowercase hex SHA-256 digest of *text* (UTF-8 encoded)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _new_nonce() -> str:
    """Return a fresh 32-character random alphanumeric nonce."""
    return uuid.uuid4().hex  # 32 hex chars


def _rest_sign(
    api_key: str,
    secret_key: str,
    nonce: str,
    timestamp_ms: str,
    query_params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Compute the Bitunix REST request signature.

    Steps (from official Python example):
      sorted_query = sorted key+value pairs of *query_params* in ASCII key order
      body_str     = JSON-serialised *body* with no extra spacing
      digest = SHA256(nonce + timestamp_ms + api_key + sorted_query + body_str)
      sign   = SHA256(digest + secret_key)
    """
    # 1. Build sorted query string: keys in ascending ASCII order, values appended
    sorted_query = ""
    if query_params:
        for k in sorted(query_params.keys()):
            sorted_query += str(k) + str(query_params[k])

    # 2. Serialise body without spaces
    body_str = ""
    if body:
        body_str = json.dumps(body, separators=(",", ":"), ensure_ascii=False)

    # 3. Double hash
    digest = _sha256(nonce + timestamp_ms + api_key + sorted_query + body_str)
    sign = _sha256(digest + secret_key)
    return sign


def ws_login_sign(api_key: str, secret_key: str, nonce: str, timestamp_sec: int) -> str:
    """
    Compute the Bitunix WebSocket login signature.

    Steps (from official Python example in WS docs):
      digest = SHA256(nonce + str(timestamp_sec) + api_key)
      sign   = SHA256(digest + secret_key)
    """
    digest = _sha256(nonce + str(timestamp_sec) + api_key)
    return _sha256(digest + secret_key)


# ---------------------------------------------------------------------------
# Async REST client
# ---------------------------------------------------------------------------

class BitunixRestClient:
    """
    Thin async wrapper around the Bitunix futures REST API.

    Every authenticated request automatically generates a fresh nonce,
    millisecond timestamp, and double-SHA-256 signature.
    """

    def __init__(self, api_key: str, secret_key: str) -> None:
        self.api_key = api_key
        self.secret_key = secret_key

    def _auth_headers(
        self,
        nonce: str,
        timestamp_ms: str,
        sign: str,
    ) -> Dict[str, str]:
        return {
            "api-key": self.api_key,
            "nonce": nonce,
            "timestamp": timestamp_ms,
            "sign": sign,
            "Content-Type": "application/json",
        }

    async def get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Authenticated GET – parameters sent as query string."""
        nonce = _new_nonce()
        ts = str(int(time.time() * 1000))
        sign = _rest_sign(self.api_key, self.secret_key, nonce, ts, query_params=params)
        headers = self._auth_headers(nonce, ts, sign)

        url = REST_BASE + path
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as resp:
                data = await resp.json()
                if data.get("code") != 0:
                    raise RuntimeError(
                        f"Bitunix GET {path} error {data.get('code')}: {data.get('msg')}"
                    )
                return data.get("data")

    async def post(
        self, path: str, body: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Authenticated POST – payload sent as JSON body."""
        nonce = _new_nonce()
        ts = str(int(time.time() * 1000))
        sign = _rest_sign(self.api_key, self.secret_key, nonce, ts, body=body)
        headers = self._auth_headers(nonce, ts, sign)

        url = REST_BASE + path
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers) as resp:
                data = await resp.json()
                if data.get("code") != 0:
                    raise RuntimeError(
                        f"Bitunix POST {path} error {data.get('code')}: {data.get('msg')}"
                    )
                return data.get("data")

    async def get_public(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Unauthenticated GET for public endpoints (e.g. klines, tickers)."""
        url = REST_BASE + path
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                if data.get("code") != 0:
                    raise RuntimeError(
                        f"Bitunix GET {path} error {data.get('code')}: {data.get('msg')}"
                    )
                return data.get("data")


# ---------------------------------------------------------------------------
# High-level exchange adapter
# ---------------------------------------------------------------------------

class BitunixExchange:
    """
    High-level Bitunix futures exchange adapter.

    Wraps the Bitunix REST API with methods that match the calling
    conventions used by gate_bot.py and binance_multi_bot.py so the
    grid-bot layer can stay exchange-agnostic.

    REST base : https://fapi.bitunix.com
    WS public : wss://fapi.bitunix.com/public/
    WS private: wss://fapi.bitunix.com/private/
    """

    def __init__(self, api_key: str, secret_key: str) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self._client = BitunixRestClient(api_key, secret_key)

    # ------------------------------------------------------------------
    # Account / balance
    # ------------------------------------------------------------------

    async def get_balance(self, margin_coin: str = "USDT") -> Dict[str, Any]:
        """
        Fetch account balance for *margin_coin*.

        GET /api/v1/futures/account?marginCoin=USDT

        Returns a dict with keys matching ccxt conventions:
          ``{"total": float, "free": float, "used": float}``

        Response fields used:
          available → free
          frozen + margin → used
          available + frozen + margin (approx) → total
        """
        data = await self._client.get(
            "/api/v1/futures/account", params={"marginCoin": margin_coin}
        )
        # data is a list; find the entry for margin_coin
        raw_entry = next(
            (d for d in (data or []) if d.get("marginCoin") == margin_coin), {}
        )
        entry = AccountBalanceEntry.model_validate({"marginCoin": margin_coin, **raw_entry})
        free = entry.available
        frozen = entry.frozen
        margin = entry.margin
        used = frozen + margin
        total = free + used
        logger.debug(
            "get_balance %s: free=%.4f used=%.4f total=%.4f",
            margin_coin, free, used, total,
        )
        return {"total": total, "free": free, "used": used, "raw": entry}

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def get_positions(self, symbol: str) -> Tuple[float, float]:
        """
        Fetch open positions for *symbol*.

        GET /api/v1/futures/position/get_pending_positions?symbol=<symbol>

        Returns ``(long_qty, short_qty)`` as non-negative floats.
        Response ``side`` field is ``"LONG"`` or ``"SHORT"``.
        """
        data = await self._client.get(
            "/api/v1/futures/position/get_pending_positions",
            params={"symbol": symbol},
        )
        long_qty: float = 0.0
        short_qty: float = 0.0
        for raw in (data or []):
            pos = PositionEntry.model_validate(raw)
            if pos.symbol != symbol:
                continue
            if pos.side.value == "LONG":
                long_qty += pos.qty
            elif pos.side.value == "SHORT":
                short_qty += abs(pos.qty)
        logger.debug("get_positions %s: long=%.4f short=%.4f", symbol, long_qty, short_qty)
        return long_qty, short_qty

    # ------------------------------------------------------------------
    # Leverage & position mode
    # ------------------------------------------------------------------

    async def set_leverage(
        self, symbol: str, leverage: int, margin_coin: str = "USDT"
    ) -> None:
        """
        Set leverage for *symbol*.

        POST /api/v1/futures/account/change_leverage
        Body: {"marginCoin": "USDT", "symbol": "XRPUSDT", "leverage": 20}
        """
        resp = await self._client.post(
            "/api/v1/futures/account/change_leverage",
            body={"marginCoin": margin_coin, "symbol": symbol, "leverage": leverage},
        )
        validated = SetLeverageData.model_validate(resp or {})
        logger.info("set_leverage %s → %dx confirmed: %s", symbol, leverage, validated)

    async def set_position_mode(self, hedge_mode: bool) -> None:
        """
        Switch between hedge mode (``True``) and one-way mode (``False``).

        POST /api/v1/futures/account/change_position_mode
        Body: {"positionMode": "HEDGE" | "ONE_WAY"}

        Note: the API will reject this call if any open position or order
        exists on any symbol.
        """
        mode = "HEDGE" if hedge_mode else "ONE_WAY"
        resp = await self._client.post(
            "/api/v1/futures/account/change_position_mode",
            body={"positionMode": mode},
        )
        validated = SetPositionModeData.model_validate(resp or {})
        logger.info("set_position_mode → %s confirmed: %s", mode, validated)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def place_limit_order(
        self,
        symbol: str,
        side: str,           # "buy" or "sell"
        quantity: float,
        price: float,
        reduce_only: bool = False,
        position_side: Optional[str] = None,  # "long" or "short"
    ) -> Optional[Dict[str, Any]]:
        """
        Place a GTC limit order.

        POST /api/v1/futures/trade/place_order

        Bitunix hedge-mode mapping:
          open long  : side=BUY,  tradeSide=OPEN
          open short : side=SELL, tradeSide=OPEN
          close long : side=SELL, tradeSide=CLOSE  (reduceOnly=True)
          close short: side=BUY,  tradeSide=CLOSE  (reduceOnly=True)

        ``reduce_only=True`` automatically sets tradeSide to CLOSE.
        """
        bs = side.upper()           # "BUY" or "SELL"
        trade_side = "CLOSE" if reduce_only else "OPEN"

        body: Dict[str, Any] = {
            "symbol": symbol,
            "side": bs,
            "tradeSide": trade_side,
            "orderType": "LIMIT",
            "qty": str(quantity),
            "price": str(price),
            "effect": "GTC",
            "reduceOnly": reduce_only,
        }
        try:
            resp = await self._client.post(
                "/api/v1/futures/trade/place_order", body=body
            )
            order = PlaceOrderData.model_validate(resp or {})
            logger.info(
                "place_limit_order %s %s %s qty=%s @%s reduce=%s → orderId=%s",
                symbol, bs, trade_side, quantity, price, reduce_only, order.orderId,
            )
            return {"orderId": order.orderId, "clientId": order.clientId}
        except Exception as exc:
            logger.error("place_limit_order failed: %s", exc)
            return None

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
        position_side: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Place a market order.

        POST /api/v1/futures/trade/place_order  (orderType=MARKET, no price)
        """
        bs = side.upper()
        trade_side = "CLOSE" if reduce_only else "OPEN"

        body: Dict[str, Any] = {
            "symbol": symbol,
            "side": bs,
            "tradeSide": trade_side,
            "orderType": "MARKET",
            "qty": str(quantity),
            "reduceOnly": reduce_only,
        }
        try:
            resp = await self._client.post(
                "/api/v1/futures/trade/place_order", body=body
            )
            order = PlaceOrderData.model_validate(resp or {})
            logger.info(
                "place_market_order %s %s %s qty=%s reduce=%s → orderId=%s",
                symbol, bs, trade_side, quantity, reduce_only, order.orderId,
            )
            return {"orderId": order.orderId, "clientId": order.clientId}
        except Exception as exc:
            logger.error("place_market_order failed: %s", exc)
            return None

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel a single order by *order_id*.

        POST /api/v1/futures/trade/cancel_orders
        Body: {"symbol": "XRPUSDT", "orderList": [{"orderId": "<id>"}]}

        Returns ``True`` on success, ``False`` on soft failure (already filled, etc.).
        """
        try:
            resp = await self._client.post(
                "/api/v1/futures/trade/cancel_orders",
                body={"symbol": symbol, "orderList": [{"orderId": order_id}]},
            )
            result = CancelOrdersData.model_validate(resp or {})
            if result.has_failures:
                logger.warning(
                    "cancel_order %s partial failure: %s",
                    order_id,
                    [(f.orderId, f.errorMsg) for f in result.failureList],
                )
                return False
            return True
        except Exception as exc:
            logger.error("cancel_order %s failed: %s", order_id, exc)
            return False

    async def cancel_orders_for_side(
        self, symbol: str, position_side: str
    ) -> None:
        """
        Cancel all open orders for *position_side* (``'long'`` or ``'short'``).

        Fetches pending orders, classifies by side+reduceOnly using the same
        logic as gate_bot.cancel_orders_for_side(), then batch-cancels.

        Bitunix cancel_orders accepts a list of up to N order IDs in one call, so
        we cancel in one POST per run.
        """
        orders = await self.get_open_orders(symbol)
        to_cancel: List[str] = []

        for order in orders:
            o_side = order.get("side", "").upper()       # BUY / SELL
            reduce = order.get("reduceOnly", False)

            if position_side == "long":
                # long entry: BUY + not reduce-only
                # long TP:    SELL + reduce-only
                if (not reduce and o_side == "BUY") or (reduce and o_side == "SELL"):
                    to_cancel.append(order["orderId"])
            elif position_side == "short":
                # short entry: SELL + not reduce-only
                # short TP:    BUY  + reduce-only
                if (not reduce and o_side == "SELL") or (reduce and o_side == "BUY"):
                    to_cancel.append(order["orderId"])

        if not to_cancel:
            logger.info("cancel_orders_for_side %s %s: nothing to cancel", symbol, position_side)
            return

        try:
            resp = await self._client.post(
                "/api/v1/futures/trade/cancel_orders",
                body={
                    "symbol": symbol,
                    "orderList": [{"orderId": oid} for oid in to_cancel],
                },
            )
            result = CancelOrdersData.model_validate(resp or {})
            logger.info(
                "cancel_orders_for_side %s %s: cancelled %d, failed %d",
                symbol, position_side,
                len(result.successList), len(result.failureList),
            )
        except Exception as exc:
            logger.error("cancel_orders_for_side failed: %s", exc)

    async def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch all open (NEW + PART_FILLED) orders for *symbol*.

        GET /api/v1/futures/trade/get_pending_orders?symbol=<symbol>

        Returns a list of order dicts normalised to the fields used by the
        grid-bot:
          orderId, side (BUY/SELL), price (float), qty (float),
          remaining (float), reduceOnly (bool), status, ctime (int ms)
        """
        data = await self._client.get(
            "/api/v1/futures/trade/get_pending_orders",
            params={"symbol": symbol, "limit": 100},
        )
        pending = PendingOrdersData.model_validate(data or {})
        orders: List[Dict[str, Any]] = []
        for o in pending.orderList:
            orders.append(
                {
                    "orderId": o.orderId,
                    "clientId": o.clientId,
                    "symbol": o.symbol,
                    "side": o.side.value,          # "BUY" / "SELL"
                    "orderType": o.orderType.value if o.orderType else None,
                    "price": o.price,
                    "qty": o.qty,
                    "remaining": o.remaining,
                    "reduceOnly": o.reduceOnly,
                    "status": o.status.value if o.status else None,
                    "ctime": o.ctime,
                }
            )
        return orders

    # ------------------------------------------------------------------
    # Market data (public)
    # ------------------------------------------------------------------

    async def get_klines(
        self,
        symbol: str,
        interval: str = "1min",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV klines from Bitunix public endpoint.

        GET /api/v1/futures/market/kline

        Parameters
        ----------
        symbol     : e.g. ``"BTCUSDT"``
        interval   : one of ``1m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M``
        start_time : start of range in milliseconds (inclusive)
        end_time   : end of range in milliseconds (inclusive)
        limit      : number of candles to return (max 200 — Bitunix API hard limit)

        Returns
        -------
        List of dicts with keys: ``open_time``, ``open``, ``high``, ``low``,
        ``close``, ``volume`` – values are floats / int as appropriate.
        """
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

        raw = await self._client.get_public(
            "/api/v1/futures/market/kline", params=params
        )
        candles: List[Dict[str, Any]] = []
        for item in raw or []:
            candles.append(
                {
                    "open_time": int(item["time"]),
                    "open": float(item["open"]),
                    "high": float(item["high"]),
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                    "volume": float(item.get("vol") or item.get("volume") or 0),
                }
            )
        return candles

    # ------------------------------------------------------------------
    # WebSocket helpers
    # ------------------------------------------------------------------

    def build_ws_login_payload(self) -> Dict[str, Any]:
        """
        Build the WebSocket login payload for private channel authentication.

        Uses seconds-precision timestamp (not milliseconds) as per the
        Bitunix WebSocket docs.

          digest = SHA256(nonce + str(timestamp_sec) + apiKey)
          sign   = SHA256(digest + secretKey)
        """
        nonce = _new_nonce()
        ts_sec = int(time.time())
        sign = ws_login_sign(self.api_key, self.secret_key, nonce, ts_sec)
        return {
            "op": "login",
            "args": [
                {
                    "apiKey": self.api_key,
                    "timestamp": ts_sec,
                    "nonce": nonce,
                    "sign": sign,
                }
            ],
        }

    @staticmethod
    def build_subscribe_payload(channel: str, symbol: str) -> Dict[str, Any]:
        """
        Build a channel subscription payload.

        Example: ``build_subscribe_payload("ticker", "XRPUSDT")``
        → ``{"op":"subscribe","args":[{"symbol":"XRPUSDT","ch":"ticker"}]}``
        """
        return {"op": "subscribe", "args": [{"symbol": symbol, "ch": channel}]}

    @staticmethod
    def build_ping_payload() -> Dict[str, Any]:
        """Build a WebSocket keepalive ping payload (seconds timestamp)."""
        return {"op": "ping", "ping": int(time.time())}

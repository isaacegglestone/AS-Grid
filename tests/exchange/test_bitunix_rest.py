"""
tests/exchange/test_bitunix_rest.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for BitunixRestClient and BitunixExchange REST methods.

HTTP calls are intercepted by ``aioresponses`` so no real network traffic
is made.  Each test supplies a canned JSON payload matching the structure
documented at https://openapidoc.bitunix.com/.
"""
import json
import re
from unittest.mock import patch

import pytest
import pytest_asyncio
from aioresponses import aioresponses

from src.exchange.bitunix import (
    REST_BASE,
    BitunixExchange,
    BitunixRestClient,
    _rest_sign,
    _sha256,
    ws_login_sign,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

API_KEY = "test_api_key"
SECRET_KEY = "test_secret_key"
NONCE = "a" * 32          # fixed 32-char nonce for deterministic signing
TIMESTAMP_MS = "1700000000000"
TIMESTAMP_SEC = 1700000000

# Regex matching any Bitunix REST URL (used for loose URL matching)
ANY_BITUNIX_URL = re.compile(r"https://fapi\.bitunix\.com/.*")


@pytest.fixture
def exchange() -> BitunixExchange:
    return BitunixExchange(api_key=API_KEY, secret_key=SECRET_KEY)


@pytest.fixture
def client() -> BitunixRestClient:
    return BitunixRestClient(api_key=API_KEY, secret_key=SECRET_KEY)


# ---------------------------------------------------------------------------
# Signing helpers – deterministic, no I/O
# ---------------------------------------------------------------------------

class TestSigning:
    def test_sha256_produces_64_char_lowercase_hex(self):
        # Verify structural properties: 64-char lowercase hex, deterministic
        result = _sha256("abc")
        assert len(result) == 64
        assert result == result.lower()
        assert all(c in "0123456789abcdef" for c in result)

    def test_sha256_different_inputs_differ(self):
        assert _sha256("abc") != _sha256("ABC")
        assert _sha256("") != _sha256(" ")

    def test_sha256_deterministic(self):
        assert _sha256("hello world") == _sha256("hello world")

    def test_rest_sign_deterministic(self):
        """Same inputs must always produce the same signature."""
        sig1 = _rest_sign(API_KEY, SECRET_KEY, NONCE, TIMESTAMP_MS)
        sig2 = _rest_sign(API_KEY, SECRET_KEY, NONCE, TIMESTAMP_MS)
        assert sig1 == sig2

    def test_rest_sign_with_query_params(self):
        sig_with = _rest_sign(
            API_KEY, SECRET_KEY, NONCE, TIMESTAMP_MS,
            query_params={"symbol": "XRPUSDT"},
        )
        sig_without = _rest_sign(API_KEY, SECRET_KEY, NONCE, TIMESTAMP_MS)
        # Different params → different signature
        assert sig_with != sig_without

    def test_rest_sign_params_sorted(self):
        """Query params must be sorted by key; order of input dict must not matter."""
        sig_ab = _rest_sign(
            API_KEY, SECRET_KEY, NONCE, TIMESTAMP_MS,
            query_params={"marginCoin": "USDT", "limit": "100"},
        )
        sig_ba = _rest_sign(
            API_KEY, SECRET_KEY, NONCE, TIMESTAMP_MS,
            query_params={"limit": "100", "marginCoin": "USDT"},
        )
        assert sig_ab == sig_ba

    def test_rest_sign_with_body(self):
        sig_body = _rest_sign(
            API_KEY, SECRET_KEY, NONCE, TIMESTAMP_MS,
            body={"symbol": "XRPUSDT", "leverage": 20},
        )
        sig_empty = _rest_sign(API_KEY, SECRET_KEY, NONCE, TIMESTAMP_MS)
        assert sig_body != sig_empty

    def test_rest_sign_produces_64_char_hex(self):
        sig = _rest_sign(API_KEY, SECRET_KEY, NONCE, TIMESTAMP_MS)
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)

    def test_ws_login_sign_deterministic(self):
        s1 = ws_login_sign(API_KEY, SECRET_KEY, NONCE, TIMESTAMP_SEC)
        s2 = ws_login_sign(API_KEY, SECRET_KEY, NONCE, TIMESTAMP_SEC)
        assert s1 == s2

    def test_ws_login_sign_differs_from_rest_sign(self):
        # WS uses seconds; REST uses milliseconds – must produce different results
        rest = _rest_sign(API_KEY, SECRET_KEY, NONCE, str(TIMESTAMP_SEC * 1000))
        ws = ws_login_sign(API_KEY, SECRET_KEY, NONCE, TIMESTAMP_SEC)
        assert rest != ws

    def test_ws_login_sign_64_char_hex(self):
        sig = ws_login_sign(API_KEY, SECRET_KEY, NONCE, TIMESTAMP_SEC)
        assert len(sig) == 64


# ---------------------------------------------------------------------------
# BitunixRestClient.get / post – happy path & error path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBitunixRestClient:
    async def test_get_returns_data_field(self, client):
        payload = {"code": 0, "msg": "Success", "data": [{"marginCoin": "USDT"}]}
        with aioresponses() as m:
            m.get(ANY_BITUNIX_URL, payload=payload)
            result = await client.get("/api/v1/futures/account", params={"marginCoin": "USDT"})
        assert result == [{"marginCoin": "USDT"}]

    async def test_post_returns_data_field(self, client):
        payload = {"code": 0, "msg": "Success", "data": {"orderId": "999"}}
        with aioresponses() as m:
            m.post(ANY_BITUNIX_URL, payload=payload)
            result = await client.post(
                "/api/v1/futures/trade/place_order",
                body={"symbol": "XRPUSDT", "side": "BUY"},
            )
        assert result == {"orderId": "999"}

    async def test_get_raises_on_nonzero_code(self, client):
        payload = {"code": 10001, "msg": "Invalid API key", "data": None}
        with aioresponses() as m:
            m.get(ANY_BITUNIX_URL, payload=payload)
            with pytest.raises(RuntimeError, match="10001"):
                await client.get("/api/v1/futures/account")

    async def test_post_raises_on_nonzero_code(self, client):
        payload = {"code": 40001, "msg": "Insufficient balance", "data": None}
        with aioresponses() as m:
            m.post(ANY_BITUNIX_URL, payload=payload)
            with pytest.raises(RuntimeError, match="40001"):
                await client.post("/api/v1/futures/trade/place_order", body={})

    async def test_get_sends_auth_headers(self, client):
        """Verify auth headers are present in the request."""
        payload = {"code": 0, "msg": "ok", "data": []}
        captured = {}

        def callback(url, **kwargs):
            captured["headers"] = kwargs.get("headers", {})
            from aioresponses.core import CallbackResult
            return CallbackResult(payload=payload)

        with aioresponses() as m:
            m.get(ANY_BITUNIX_URL, callback=callback)
            await client.get("/api/v1/futures/account")

        assert "api-key" in captured["headers"]
        assert "nonce" in captured["headers"]
        assert "timestamp" in captured["headers"]
        assert "sign" in captured["headers"]
        assert captured["headers"]["api-key"] == API_KEY


# ---------------------------------------------------------------------------
# BitunixExchange.get_balance
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGetBalance:
    async def test_returns_correct_values(self, exchange):
        payload = {
            "code": 0,
            "msg": "Success",
            "data": [{
                "marginCoin": "USDT",
                "available": "1000",
                "frozen": "50",
                "margin": "200",
                "transfer": "800",
                "positionMode": "HEDGE",
                "crossUnrealizedPNL": "0",
                "isolationUnrealizedPNL": "0",
                "bonus": "0",
            }],
        }
        with aioresponses() as m:
            m.get(ANY_BITUNIX_URL, payload=payload)
            result = await exchange.get_balance("USDT")

        assert result["free"] == pytest.approx(1000.0)
        assert result["used"] == pytest.approx(250.0)   # frozen(50) + margin(200)
        assert result["total"] == pytest.approx(1250.0)

    async def test_missing_coin_returns_zeros(self, exchange):
        """If the requested coin is absent from the list, all values are 0."""
        payload = {"code": 0, "msg": "Success", "data": []}
        with aioresponses() as m:
            m.get(ANY_BITUNIX_URL, payload=payload)
            result = await exchange.get_balance("USDT")

        assert result["free"] == 0.0
        assert result["used"] == 0.0
        assert result["total"] == 0.0

    async def test_raw_is_account_balance_entry(self, exchange):
        from src.exchange.bitunix_models import AccountBalanceEntry
        payload = {
            "code": 0, "msg": "Success",
            "data": [{"marginCoin": "USDT", "available": "500"}],
        }
        with aioresponses() as m:
            m.get(ANY_BITUNIX_URL, payload=payload)
            result = await exchange.get_balance("USDT")

        assert isinstance(result["raw"], AccountBalanceEntry)


# ---------------------------------------------------------------------------
# BitunixExchange.get_positions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGetPositions:
    async def test_returns_long_and_short(self, exchange):
        payload = {
            "code": 0, "msg": "Success",
            "data": [
                {"positionId": "1", "symbol": "XRPUSDT", "side": "LONG",  "qty": "10", "ctime": 0},
                {"positionId": "2", "symbol": "XRPUSDT", "side": "SHORT", "qty": "5",  "ctime": 0},
            ],
        }
        with aioresponses() as m:
            m.get(ANY_BITUNIX_URL, payload=payload)
            long_qty, short_qty = await exchange.get_positions("XRPUSDT")

        assert long_qty == pytest.approx(10.0)
        assert short_qty == pytest.approx(5.0)

    async def test_filters_by_symbol(self, exchange):
        """Positions for other symbols must be ignored."""
        payload = {
            "code": 0, "msg": "Success",
            "data": [
                {"positionId": "1", "symbol": "BTCUSDT", "side": "LONG", "qty": "1", "ctime": 0},
                {"positionId": "2", "symbol": "XRPUSDT", "side": "LONG", "qty": "20", "ctime": 0},
            ],
        }
        with aioresponses() as m:
            m.get(ANY_BITUNIX_URL, payload=payload)
            long_qty, short_qty = await exchange.get_positions("XRPUSDT")

        assert long_qty == pytest.approx(20.0)
        assert short_qty == pytest.approx(0.0)

    async def test_no_positions_returns_zeros(self, exchange):
        payload = {"code": 0, "msg": "Success", "data": []}
        with aioresponses() as m:
            m.get(ANY_BITUNIX_URL, payload=payload)
            long_qty, short_qty = await exchange.get_positions("XRPUSDT")

        assert long_qty == 0.0
        assert short_qty == 0.0


# ---------------------------------------------------------------------------
# BitunixExchange.place_limit_order
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPlaceLimitOrder:
    async def test_returns_order_id_dict(self, exchange):
        payload = {
            "code": 0, "msg": "Success",
            "data": {"orderId": "55555", "clientId": "my-client"},
        }
        with aioresponses() as m:
            m.post(ANY_BITUNIX_URL, payload=payload)
            result = await exchange.place_limit_order(
                "XRPUSDT", "buy", 10, 2.50
            )

        assert result is not None
        assert result["orderId"] == "55555"
        assert result["clientId"] == "my-client"

    async def test_reduce_only_returns_order_id(self, exchange):
        payload = {
            "code": 0, "msg": "Success",
            "data": {"orderId": "66666"},
        }
        with aioresponses() as m:
            m.post(ANY_BITUNIX_URL, payload=payload)
            result = await exchange.place_limit_order(
                "XRPUSDT", "sell", 10, 2.80, reduce_only=True
            )

        assert result["orderId"] == "66666"

    async def test_api_error_returns_none(self, exchange):
        payload = {"code": 40001, "msg": "Insufficient balance", "data": None}
        with aioresponses() as m:
            m.post(ANY_BITUNIX_URL, payload=payload)
            result = await exchange.place_limit_order("XRPUSDT", "buy", 10, 2.50)

        assert result is None


# ---------------------------------------------------------------------------
# BitunixExchange.cancel_order
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCancelOrder:
    async def test_returns_true_on_success(self, exchange):
        payload = {
            "code": 0, "msg": "Success",
            "data": {
                "successList": [{"orderId": "11111", "clientId": "22222"}],
                "failureList": [],
            },
        }
        with aioresponses() as m:
            m.post(ANY_BITUNIX_URL, payload=payload)
            result = await exchange.cancel_order("11111", "XRPUSDT")

        assert result is True

    async def test_returns_false_on_failure(self, exchange):
        payload = {
            "code": 0, "msg": "Success",
            "data": {
                "successList": [],
                "failureList": [
                    {"orderId": "11112", "errorMsg": "Order status error", "errorCode": 10013}
                ],
            },
        }
        with aioresponses() as m:
            m.post(ANY_BITUNIX_URL, payload=payload)
            result = await exchange.cancel_order("11112", "XRPUSDT")

        assert result is False

    async def test_returns_false_on_api_error(self, exchange):
        payload = {"code": 500, "msg": "Internal error", "data": None}
        with aioresponses() as m:
            m.post(ANY_BITUNIX_URL, payload=payload)
            result = await exchange.cancel_order("99999", "XRPUSDT")

        assert result is False


# ---------------------------------------------------------------------------
# BitunixExchange.get_open_orders
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGetOpenOrders:
    async def test_normalises_order_fields(self, exchange):
        payload = {
            "code": 0, "msg": "Success",
            "data": {
                "orderList": [{
                    "orderId": "11111",
                    "clientId": "22222",
                    "symbol": "XRPUSDT",
                    "side": "BUY",
                    "orderType": "LIMIT",
                    "price": "2.5000",
                    "qty": "10",
                    "tradeQty": "3",
                    "fee": "0.01",
                    "reduceOnly": False,
                    "status": "PART_FILLED",
                    "ctime": 1597026383085,
                    "mtime": 1597026383085,
                }],
                "total": 1,
            },
        }
        with aioresponses() as m:
            m.get(ANY_BITUNIX_URL, payload=payload)
            orders = await exchange.get_open_orders("XRPUSDT")

        assert len(orders) == 1
        o = orders[0]
        assert o["orderId"] == "11111"
        assert o["side"] == "BUY"
        assert o["price"] == pytest.approx(2.5)
        assert o["qty"] == pytest.approx(10.0)
        assert o["remaining"] == pytest.approx(7.0)   # 10 - 3
        assert o["reduceOnly"] is False
        assert o["status"] == "PART_FILLED"
        assert o["ctime"] == 1597026383085

    async def test_empty_order_list(self, exchange):
        payload = {"code": 0, "msg": "Success", "data": {"orderList": [], "total": 0}}
        with aioresponses() as m:
            m.get(ANY_BITUNIX_URL, payload=payload)
            orders = await exchange.get_open_orders("XRPUSDT")

        assert orders == []


# ---------------------------------------------------------------------------
# BitunixExchange.cancel_orders_for_side
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCancelOrdersForSide:
    async def _pending_orders_payload(self, orders):
        return {
            "code": 0, "msg": "Success",
            "data": {"orderList": orders, "total": len(orders)},
        }

    async def test_cancels_long_side_orders(self, exchange):
        """Long side: BUY+not-reduce (entry) and SELL+reduce (TP) should be cancelled."""
        pending_payload = await self._pending_orders_payload([
            # Long entry
            {"orderId": "1", "symbol": "XRPUSDT", "side": "BUY",  "qty": "10",
             "tradeQty": "0", "reduceOnly": False, "status": "NEW", "price": "2.4",
             "orderType": "LIMIT", "ctime": 0},
            # Long TP
            {"orderId": "2", "symbol": "XRPUSDT", "side": "SELL", "qty": "10",
             "tradeQty": "0", "reduceOnly": True, "status": "NEW", "price": "2.6",
             "orderType": "LIMIT", "ctime": 0},
            # Short entry (must NOT be cancelled)
            {"orderId": "3", "symbol": "XRPUSDT", "side": "SELL", "qty": "5",
             "tradeQty": "0", "reduceOnly": False, "status": "NEW", "price": "2.7",
             "orderType": "LIMIT", "ctime": 0},
        ])
        cancel_payload = {
            "code": 0, "msg": "Success",
            "data": {
                "successList": [{"orderId": "1"}, {"orderId": "2"}],
                "failureList": [],
            },
        }

        cancelled_body = {}
        def capture_cancel(url, **kwargs):
            cancelled_body.update(kwargs.get("json", {}))
            from aioresponses.core import CallbackResult
            return CallbackResult(payload=cancel_payload)

        with aioresponses() as m:
            m.get(ANY_BITUNIX_URL, payload=pending_payload)
            m.post(ANY_BITUNIX_URL, callback=capture_cancel)
            await exchange.cancel_orders_for_side("XRPUSDT", "long")

        cancelled_ids = {item["orderId"] for item in cancelled_body.get("orderList", [])}
        assert "1" in cancelled_ids   # long entry
        assert "2" in cancelled_ids   # long TP
        assert "3" not in cancelled_ids  # short entry must be untouched

    async def test_no_orders_nothing_cancelled(self, exchange):
        """If there are no open orders, no cancel POST is made."""
        pending_payload = await self._pending_orders_payload([])
        with aioresponses() as m:
            m.get(ANY_BITUNIX_URL, payload=pending_payload)
            # No POST registered – would raise aioresponses.ConnectionError if called
            await exchange.cancel_orders_for_side("XRPUSDT", "long")
        # Test passes if no exception was raised


# ---------------------------------------------------------------------------
# WS payload builders (no I/O)
# ---------------------------------------------------------------------------

class TestWSBuilders:
    def test_build_subscribe_payload(self):
        exchange = BitunixExchange(api_key="k", secret_key="s")
        payload = exchange.build_subscribe_payload("ticker", "XRPUSDT")
        assert payload["op"] == "subscribe"
        assert payload["args"][0]["ch"] == "ticker"
        assert payload["args"][0]["symbol"] == "XRPUSDT"

    def test_build_ping_payload(self):
        exchange = BitunixExchange(api_key="k", secret_key="s")
        payload = exchange.build_ping_payload()
        assert payload["op"] == "ping"
        assert isinstance(payload["ping"], int)

    def test_build_ws_login_payload(self):
        exchange = BitunixExchange(api_key="my_key", secret_key="my_secret")
        payload = exchange.build_ws_login_payload()
        assert payload["op"] == "login"
        args = payload["args"][0]
        assert args["apiKey"] == "my_key"
        assert isinstance(args["timestamp"], int)
        assert len(args["nonce"]) == 32
        assert len(args["sign"]) == 64

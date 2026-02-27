"""
tests/exchange/test_bitunix_models.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pure unit tests for src/exchange/bitunix_models.py.

No network calls, no mocking required – all tests exercise Pydantic
parsing/coercion against JSON payloads copied directly from the Bitunix
API documentation response examples.
"""
import pytest
from pydantic import ValidationError

from src.exchange.bitunix_models import (
    AccountBalanceEntry,
    BitunixEnvelope,
    CancelOrderFailure,
    CancelOrderSuccess,
    CancelOrdersData,
    MarginMode,
    OrderEffect,
    OrderSide,
    OrderStatus,
    OrderType,
    PendingOrder,
    PendingOrdersData,
    PlaceOrderData,
    PositionEntry,
    PositionMode,
    PositionSide,
    SetLeverageData,
    SetPositionModeData,
    StopType,
    TradeSide,
    _to_float,
)


# ---------------------------------------------------------------------------
# FloatStr coercion helper
# ---------------------------------------------------------------------------

class TestToFloat:
    def test_string_integer(self):
        assert _to_float("100") == 100.0

    def test_string_decimal(self):
        assert _to_float("100.50") == pytest.approx(100.50)

    def test_already_float(self):
        assert _to_float(3.14) == pytest.approx(3.14)

    def test_already_int(self):
        assert _to_float(42) == 42.0

    def test_none_returns_zero(self):
        assert _to_float(None) == 0.0

    def test_empty_string_returns_zero(self):
        assert _to_float("") == 0.0

    def test_negative_string(self):
        assert _to_float("-0.2") == pytest.approx(-0.2)

    def test_invalid_string_returns_zero(self):
        assert _to_float("not-a-number") == 0.0


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TestEnums:
    def test_order_side_values(self):
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"

    def test_trade_side_values(self):
        assert TradeSide.OPEN.value == "OPEN"
        assert TradeSide.CLOSE.value == "CLOSE"

    def test_position_side_values(self):
        assert PositionSide.LONG.value == "LONG"
        assert PositionSide.SHORT.value == "SHORT"

    def test_position_mode_values(self):
        assert PositionMode.ONE_WAY.value == "ONE_WAY"
        assert PositionMode.HEDGE.value == "HEDGE"

    def test_margin_mode_values(self):
        assert MarginMode.ISOLATION.value == "ISOLATION"
        assert MarginMode.CROSS.value == "CROSS"

    def test_order_type_values(self):
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.MARKET.value == "MARKET"

    def test_order_effect_all_values(self):
        assert {e.value for e in OrderEffect} == {"GTC", "IOC", "FOK", "POST_ONLY"}

    def test_order_status_all_values(self):
        assert {s.value for s in OrderStatus} == {
            "INIT", "NEW", "PART_FILLED", "FILLED", "CANCELED"
        }

    def test_stop_type_values(self):
        assert StopType.MARK_PRICE.value == "MARK_PRICE"
        assert StopType.LAST_PRICE.value == "LAST_PRICE"

    def test_enum_str_equality(self):
        # str enums should compare equal to plain strings
        assert OrderSide.BUY == "BUY"
        assert OrderStatus.FILLED == "FILLED"


# ---------------------------------------------------------------------------
# BitunixEnvelope
# ---------------------------------------------------------------------------

class TestBitunixEnvelope:
    def test_is_ok_true_on_zero_code(self):
        env = BitunixEnvelope(code=0, msg="Success", data={"orderId": "123"})
        assert env.is_ok is True

    def test_is_ok_false_on_nonzero_code(self):
        env = BitunixEnvelope(code=10013, msg="Order status error", data=None)
        assert env.is_ok is False

    def test_missing_msg_defaults_empty(self):
        env = BitunixEnvelope(code=0)
        assert env.msg == ""

    def test_missing_data_defaults_none(self):
        env = BitunixEnvelope(code=0, msg="ok")
        assert env.data is None

    def test_extra_fields_allowed(self):
        # extra="allow" – unknown fields must not raise
        env = BitunixEnvelope(code=0, msg="ok", data=None, unknownField="x")
        assert env.code == 0


# ---------------------------------------------------------------------------
# AccountBalanceEntry – doc sample from get_single_account.html
# ---------------------------------------------------------------------------

ACCOUNT_SAMPLE = {
    "marginCoin": "USDT",
    "available": "1000",
    "frozen": "0",
    "margin": "10",
    "transfer": "1000",
    "positionMode": "HEDGE",
    "crossUnrealizedPNL": "2",
    "isolationUnrealizedPNL": "0",
    "bonus": "0",
}


class TestAccountBalanceEntry:
    def test_parses_doc_sample(self):
        entry = AccountBalanceEntry.model_validate(ACCOUNT_SAMPLE)
        assert entry.marginCoin == "USDT"
        assert entry.available == pytest.approx(1000.0)
        assert entry.frozen == pytest.approx(0.0)
        assert entry.margin == pytest.approx(10.0)
        assert entry.transfer == pytest.approx(1000.0)
        assert entry.positionMode == PositionMode.HEDGE
        assert entry.crossUnrealizedPNL == pytest.approx(2.0)
        assert entry.isolationUnrealizedPNL == pytest.approx(0.0)
        assert entry.bonus == pytest.approx(0.0)

    def test_numeric_strings_coerced(self):
        entry = AccountBalanceEntry.model_validate(
            {"marginCoin": "USDT", "available": "500.75", "margin": "25.5"}
        )
        assert entry.available == pytest.approx(500.75)
        assert entry.margin == pytest.approx(25.5)

    def test_missing_optional_fields_default_zero(self):
        entry = AccountBalanceEntry.model_validate({"marginCoin": "USDT"})
        assert entry.available == 0.0
        assert entry.frozen == 0.0
        assert entry.positionMode is None

    def test_position_mode_one_way(self):
        entry = AccountBalanceEntry.model_validate(
            {"marginCoin": "USDT", "positionMode": "ONE_WAY"}
        )
        assert entry.positionMode == PositionMode.ONE_WAY

    def test_extra_fields_ignored(self):
        entry = AccountBalanceEntry.model_validate(
            {"marginCoin": "USDT", "unknownField": "whatever"}
        )
        assert entry.marginCoin == "USDT"


# ---------------------------------------------------------------------------
# PositionEntry – doc sample from get_pending_positions.html
# ---------------------------------------------------------------------------

POSITION_SAMPLE = {
    "positionId": "12345678",
    "symbol": "BTCUSDT",
    "qty": "0.5",
    "entryValue": "30000",
    "side": "LONG",
    "positionMode": "HEDGE",
    "marginMode": "ISOLATION",
    "leverage": 100,
    "fee": "0.1",
    "funding": "-0.2",
    "realizedPNL": "102.9",
    "margin": "300",
    "unrealizedPNL": "1.5",
    "liqPrice": "22209",
    "marginRate": "0.01",
    "avgOpenPrice": "1.0",
    "ctime": 1691382137448,
    "mtime": 1691382137448,
}


class TestPositionEntry:
    def test_parses_doc_sample(self):
        pos = PositionEntry.model_validate(POSITION_SAMPLE)
        assert pos.positionId == "12345678"
        assert pos.symbol == "BTCUSDT"
        assert pos.side == PositionSide.LONG
        assert pos.qty == pytest.approx(0.5)
        assert pos.entryValue == pytest.approx(30000.0)
        assert pos.avgOpenPrice == pytest.approx(1.0)
        assert pos.leverage == 100
        assert pos.funding == pytest.approx(-0.2)
        assert pos.marginMode == MarginMode.ISOLATION
        assert pos.positionMode == PositionMode.HEDGE
        assert pos.liqPrice == pytest.approx(22209.0)
        assert pos.ctime == 1691382137448

    def test_short_side(self):
        pos = PositionEntry.model_validate(
            {"symbol": "XRPUSDT", "side": "SHORT", "qty": "10"}
        )
        assert pos.side == PositionSide.SHORT
        assert pos.qty == pytest.approx(10.0)

    def test_invalid_side_raises(self):
        with pytest.raises(ValidationError):
            PositionEntry.model_validate({"symbol": "XRPUSDT", "side": "FLAT"})


# ---------------------------------------------------------------------------
# PlaceOrderData – doc sample from place_order.html
# ---------------------------------------------------------------------------

class TestPlaceOrderData:
    def test_parses_doc_sample(self):
        data = PlaceOrderData.model_validate({"orderId": "11111", "clientId": "22222"})
        assert data.orderId == "11111"
        assert data.clientId == "22222"

    def test_client_id_optional(self):
        data = PlaceOrderData.model_validate({"orderId": "abc123"})
        assert data.orderId == "abc123"
        assert data.clientId is None

    def test_missing_order_id_raises(self):
        with pytest.raises(ValidationError):
            PlaceOrderData.model_validate({"clientId": "only-client"})


# ---------------------------------------------------------------------------
# CancelOrdersData – doc sample from cancel_orders.html
# ---------------------------------------------------------------------------

CANCEL_SUCCESS_SAMPLE = {
    "successList": [{"orderId": "11111", "clientId": "22222"}],
    "failureList": [
        {
            "orderId": "11112",
            "clientId": "22223",
            "errorMsg": "Order status error",
            "errorCode": 10013,
        }
    ],
}


class TestCancelOrdersData:
    def test_parses_doc_sample(self):
        result = CancelOrdersData.model_validate(CANCEL_SUCCESS_SAMPLE)
        assert len(result.successList) == 1
        assert len(result.failureList) == 1
        assert result.has_failures is True

    def test_failure_fields(self):
        result = CancelOrdersData.model_validate(CANCEL_SUCCESS_SAMPLE)
        f = result.failureList[0]
        assert f.errorMsg == "Order status error"
        assert f.errorCode == 10013

    def test_all_success_no_failures(self):
        result = CancelOrdersData.model_validate(
            {"successList": [{"orderId": "1"}], "failureList": []}
        )
        assert result.has_failures is False

    def test_empty_lists_default(self):
        result = CancelOrdersData.model_validate({})
        assert result.successList == []
        assert result.failureList == []
        assert result.has_failures is False

    def test_cancel_success_alias_id(self):
        # docs say field name is "id" in successList
        entry = CancelOrderSuccess.model_validate({"id": "99999", "clientId": "ccc"})
        assert entry.orderId == "99999"

    def test_cancel_failure_alias_id(self):
        entry = CancelOrderFailure.model_validate(
            {"id": "88888", "errorMsg": "bad", "errorCode": 1}
        )
        assert entry.orderId == "88888"


# ---------------------------------------------------------------------------
# PendingOrder + PendingOrdersData – doc sample from get_pending_orders.html
# ---------------------------------------------------------------------------

PENDING_ORDER_SAMPLE = {
    "orderId": "11111",
    "qty": "1",
    "tradeQty": "0.5",
    "price": "60000",
    "symbol": "BTCUSDT",
    "positionMode": "HEDGE",
    "marginMode": "ISOLATION",
    "leverage": 15,
    "status": "NEW",
    "fee": "0.01",
    "realizedPNL": "1.78",
    "orderType": "LIMIT",
    "effect": "GTC",
    "reduceOnly": False,
    "clientId": "22222",
    "tpPrice": "61000",
    "tpStopType": "MARK_PRICE",
    "tpOrderType": "LIMIT",
    "tpOrderPrice": "61000.1",
    "slPrice": "59000",
    "slStopType": "MARK_PRICE",
    "slOrderType": "LIMIT",
    "slOrderPrice": "59000.1",
    "side": "BUY",
    "ctime": 1597026383085,
    "mtime": 1597026383085,
}


class TestPendingOrder:
    def test_parses_doc_sample(self):
        o = PendingOrder.model_validate(PENDING_ORDER_SAMPLE)
        assert o.orderId == "11111"
        assert o.symbol == "BTCUSDT"
        assert o.side == OrderSide.BUY
        assert o.orderType == OrderType.LIMIT
        assert o.effect == OrderEffect.GTC
        assert o.status == OrderStatus.NEW
        assert o.qty == pytest.approx(1.0)
        assert o.tradeQty == pytest.approx(0.5)
        assert o.price == pytest.approx(60000.0)
        assert o.reduceOnly is False
        assert o.leverage == 15
        assert o.tpPrice == pytest.approx(61000.0)
        assert o.tpStopType == StopType.MARK_PRICE
        assert o.tpOrderType == OrderType.LIMIT
        assert o.ctime == 1597026383085

    def test_remaining_property(self):
        o = PendingOrder.model_validate(PENDING_ORDER_SAMPLE)
        # qty=1, tradeQty=0.5 → remaining=0.5
        assert o.remaining == pytest.approx(0.5)

    def test_remaining_fully_filled(self):
        o = PendingOrder.model_validate({**PENDING_ORDER_SAMPLE, "tradeQty": "1"})
        assert o.remaining == pytest.approx(0.0)

    def test_remaining_never_negative(self):
        # tradeQty > qty shouldn't produce negative remaining
        o = PendingOrder.model_validate({**PENDING_ORDER_SAMPLE, "tradeQty": "2"})
        assert o.remaining == pytest.approx(0.0)

    def test_status_part_filled(self):
        o = PendingOrder.model_validate({**PENDING_ORDER_SAMPLE, "status": "PART_FILLED"})
        assert o.status == OrderStatus.PART_FILLED

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            PendingOrder.model_validate({**PENDING_ORDER_SAMPLE, "status": "UNKNOWN_STATUS"})


class TestPendingOrdersData:
    def test_empty_list(self):
        data = PendingOrdersData.model_validate({"orderList": []})
        assert data.orderList == []

    def test_parses_order_list(self):
        data = PendingOrdersData.model_validate({"orderList": [PENDING_ORDER_SAMPLE]})
        assert len(data.orderList) == 1
        assert data.orderList[0].orderId == "11111"

    def test_missing_order_list_defaults_empty(self):
        data = PendingOrdersData.model_validate({})
        assert data.orderList == []


# ---------------------------------------------------------------------------
# SetLeverageData
# ---------------------------------------------------------------------------

class TestSetLeverageData:
    def test_all_fields_optional(self):
        data = SetLeverageData.model_validate({})
        assert data.symbol is None
        assert data.leverage is None
        assert data.marginCoin is None

    def test_parses_echo_response(self):
        data = SetLeverageData.model_validate(
            {"symbol": "XRPUSDT", "leverage": "20", "marginCoin": "USDT"}
        )
        assert data.symbol == "XRPUSDT"
        assert data.leverage == "20"

    def test_none_response(self):
        # API may return None data; calling model_validate({}) should succeed
        data = SetLeverageData.model_validate({})
        assert data.symbol is None


# ---------------------------------------------------------------------------
# SetPositionModeData
# ---------------------------------------------------------------------------

class TestSetPositionModeData:
    def test_hedge_mode(self):
        data = SetPositionModeData.model_validate({"positionMode": "HEDGE"})
        assert data.positionMode == PositionMode.HEDGE

    def test_one_way_mode(self):
        data = SetPositionModeData.model_validate({"positionMode": "ONE_WAY"})
        assert data.positionMode == PositionMode.ONE_WAY

    def test_empty_response(self):
        data = SetPositionModeData.model_validate({})
        assert data.positionMode is None

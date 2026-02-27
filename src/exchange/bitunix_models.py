"""
src/exchange/bitunix_models.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pydantic v2 response models for the Bitunix futures REST API.

Field names and enum values are taken directly from the official docs:
  https://openapidoc.bitunix.com/doc/common/introduction.html

Numeric values are sent by Bitunix as JSON strings (e.g. ``"available": "100.50"``);
the ``FloatStr`` annotated type coerces them to ``float`` transparently.
All models use ``extra="allow"`` so undocumented fields do not break validation.

Enumerations
------------
  OrderSide       – BUY | SELL
  TradeSide       – OPEN | CLOSE
  PositionSide    – LONG | SHORT
  PositionMode    – ONE_WAY | HEDGE
  MarginMode      – ISOLATION | CROSS
  OrderType       – LIMIT | MARKET
  OrderEffect     – GTC | IOC | FOK | POST_ONLY
  OrderStatus     – INIT | NEW | PART_FILLED | FILLED | CANCELED
  StopType        – MARK_PRICE | LAST_PRICE

Per-endpoint data models
------------------------
  AccountBalanceEntry   – GET  /api/v1/futures/account
  PositionEntry         – GET  /api/v1/futures/position/get_pending_positions
  PlaceOrderData        – POST /api/v1/futures/trade/place_order
  CancelOrderSuccess    – successList entry in cancel_orders response
  CancelOrderFailure    – failureList entry in cancel_orders response
  CancelOrdersData      – POST /api/v1/futures/trade/cancel_orders
  PendingOrder          – one order in get_pending_orders response
  PendingOrdersData     – GET  /api/v1/futures/trade/get_pending_orders
  SetLeverageData       – POST /api/v1/futures/account/change_leverage
  SetPositionModeData   – POST /api/v1/futures/account/change_position_mode
"""
from __future__ import annotations

import enum
from typing import Annotated, Any, Dict, Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

# ---------------------------------------------------------------------------
# Helper type: coerce numeric strings (e.g. "100.50") to float.
# Falls back gracefully if the value is already a number or None.
# ---------------------------------------------------------------------------

def _to_float(v: Any) -> float:
    if v is None or v == "":
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


FloatStr = Annotated[float, BeforeValidator(_to_float)]


# ---------------------------------------------------------------------------
# Enumerations derived from docs
# ---------------------------------------------------------------------------

class OrderSide(str, enum.Enum):
    """side field on orders and position open/close direction."""
    BUY = "BUY"
    SELL = "SELL"


class TradeSide(str, enum.Enum):
    """tradeSide: whether the order opens or closes a position."""
    OPEN = "OPEN"
    CLOSE = "CLOSE"


class PositionSide(str, enum.Enum):
    """side field on positions."""
    LONG = "LONG"
    SHORT = "SHORT"


class PositionMode(str, enum.Enum):
    """Account-level position mode."""
    ONE_WAY = "ONE_WAY"
    HEDGE = "HEDGE"


class MarginMode(str, enum.Enum):
    """Per-position margin mode."""
    ISOLATION = "ISOLATION"
    CROSS = "CROSS"


class OrderType(str, enum.Enum):
    """Order execution type."""
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class OrderEffect(str, enum.Enum):
    """Time-in-force for limit orders."""
    GTC = "GTC"           # Good Till Canceled (default)
    IOC = "IOC"           # Immediate Or Cancel
    FOK = "FOK"           # Fill Or Kill
    POST_ONLY = "POST_ONLY"


class OrderStatus(str, enum.Enum):
    """Lifecycle states of an order."""
    INIT = "INIT"               # preparing / pre-submit
    NEW = "NEW"                 # resting in the book
    PART_FILLED = "PART_FILLED" # partially executed
    FILLED = "FILLED"           # fully executed
    CANCELED = "CANCELED"       # cancelled


class StopType(str, enum.Enum):
    """Trigger price type for TP/SL orders."""
    MARK_PRICE = "MARK_PRICE"
    LAST_PRICE = "LAST_PRICE"


# ---------------------------------------------------------------------------
# Generic outer envelope – returned by every Bitunix REST endpoint
# ---------------------------------------------------------------------------

T = TypeVar("T")


class BitunixEnvelope(BaseModel, Generic[T]):
    """
    Top-level API response wrapper.

    ``code == 0`` means success; any other value is an error.
    ``data`` is the payload; its type varies by endpoint.
    """

    model_config = ConfigDict(extra="allow")

    code: int
    msg: str = ""
    data: Optional[Any] = None  # typed as Any; callers validate the inner type

    @property
    def is_ok(self) -> bool:
        return self.code == 0


# ---------------------------------------------------------------------------
# GET /api/v1/futures/account
# data: list[AccountBalanceEntry]
# ---------------------------------------------------------------------------

class AccountBalanceEntry(BaseModel):
    """
    One entry in the account balance list.

    GET /api/v1/futures/account?marginCoin=USDT
    The endpoint returns a list; we filter by ``marginCoin``.
    Fields from https://openapidoc.bitunix.com/doc/account/get_single_account.html
    """

    model_config = ConfigDict(extra="allow")

    marginCoin: str
    available: FloatStr = 0.0              # free / withdrawable
    frozen: FloatStr = 0.0                 # frozen by open orders
    margin: FloatStr = 0.0                 # locked by positions
    transfer: FloatStr = 0.0               # max transferable
    positionMode: Optional[PositionMode] = None  # ONE_WAY | HEDGE
    crossUnrealizedPNL: FloatStr = 0.0     # unrealised PnL for cross positions
    isolationUnrealizedPNL: FloatStr = 0.0 # unrealised PnL for isolated positions
    bonus: FloatStr = 0.0                  # futures bonus


# ---------------------------------------------------------------------------
# GET /api/v1/futures/position/get_pending_positions
# data: list[PositionEntry]
# ---------------------------------------------------------------------------

class PositionEntry(BaseModel):
    """
    One open position returned by get_pending_positions.

    GET /api/v1/futures/position/get_pending_positions
    Fields from https://openapidoc.bitunix.com/doc/position/get_pending_positions.html
    """

    model_config = ConfigDict(extra="allow")

    positionId: str = ""
    symbol: str
    side: PositionSide                     # LONG | SHORT
    qty: FloatStr = 0.0                    # position size (base coin)
    entryValue: FloatStr = 0.0             # notional value at entry
    avgOpenPrice: FloatStr = 0.0           # average open price
    margin: FloatStr = 0.0                 # locked position margin
    liqPrice: FloatStr = 0.0              # estimated liquidation price (<=0 means no risk)
    marginRate: FloatStr = 0.0             # margin ratio
    unrealizedPNL: FloatStr = 0.0          # unrealised PnL
    realizedPNL: FloatStr = 0.0            # realised PnL (excl. fees & funding)
    fee: FloatStr = 0.0                    # cumulative transaction fees
    funding: FloatStr = 0.0               # cumulative funding fees
    leverage: int = 1
    marginMode: Optional[MarginMode] = None   # ISOLATION | CROSS
    positionMode: Optional[PositionMode] = None  # ONE_WAY | HEDGE
    ctime: int = 0                         # creation timestamp ms
    mtime: Optional[int] = None            # last modify timestamp ms


# ---------------------------------------------------------------------------
# POST /api/v1/futures/trade/place_order
# data: PlaceOrderData
# ---------------------------------------------------------------------------

class PlaceOrderData(BaseModel):
    """Confirmation payload returned after successfully placing an order."""

    model_config = ConfigDict(extra="allow")

    orderId: str
    clientId: Optional[str] = None


# ---------------------------------------------------------------------------
# POST /api/v1/futures/trade/cancel_orders
# data: CancelOrdersData
# ---------------------------------------------------------------------------

class CancelOrderSuccess(BaseModel):
    """
    One successfully-cancelled order entry inside ``CancelOrdersData.successList``.

    Note: docs table says field name is ``id`` but the response example uses ``orderId``.
    We accept both via alias.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    orderId: Optional[str] = Field(None, alias="id")  # docs say "id", examples show "orderId"
    clientId: Optional[str] = None


class CancelOrderFailure(BaseModel):
    """
    One failed cancellation entry inside ``CancelOrdersData.failureList``.

    Note: docs table says field name is ``id`` but the response example uses ``orderId``.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    orderId: Optional[str] = Field(None, alias="id")
    clientId: Optional[str] = None
    errorMsg: Optional[str] = None
    errorCode: Optional[int] = None


class CancelOrdersData(BaseModel):
    """
    Batch cancellation result.

    POST /api/v1/futures/trade/cancel_orders
    Fields from https://openapidoc.bitunix.com/doc/trade/cancel_orders.html
    """

    model_config = ConfigDict(extra="allow")

    successList: List[CancelOrderSuccess] = []
    failureList: List[CancelOrderFailure] = []

    @property
    def has_failures(self) -> bool:
        return len(self.failureList) > 0


# ---------------------------------------------------------------------------
# GET /api/v1/futures/trade/get_pending_orders
# data: PendingOrdersData  (wrapper with orderList)
# ---------------------------------------------------------------------------

class PendingOrder(BaseModel):
    """
    One entry in the pending-orders list.

    GET /api/v1/futures/trade/get_pending_orders
    Fields from https://openapidoc.bitunix.com/doc/trade/get_pending_orders.html

    ``side``        : BUY | SELL
    ``orderType``   : LIMIT | MARKET
    ``effect``      : GTC | IOC | FOK | POST_ONLY
    ``status``      : INIT | NEW | PART_FILLED | FILLED | CANCELED
    ``reduceOnly``  : True for take-profit / close orders
    ``ctime``       : creation timestamp (milliseconds)
    ``mtime``       : last modification timestamp (milliseconds)
    """

    model_config = ConfigDict(extra="allow")

    orderId: str
    clientId: Optional[str] = None
    symbol: str
    side: OrderSide                             # BUY | SELL
    orderType: Optional[OrderType] = None       # LIMIT | MARKET
    effect: Optional[OrderEffect] = None        # GTC | IOC | FOK | POST_ONLY
    price: FloatStr = 0.0
    qty: FloatStr = 0.0
    tradeQty: FloatStr = 0.0                    # filled quantity
    tradeValue: FloatStr = 0.0                  # filled notional value
    fee: FloatStr = 0.0
    realizedPNL: FloatStr = 0.0
    reduceOnly: bool = False
    status: Optional[OrderStatus] = None        # INIT | NEW | PART_FILLED | FILLED | CANCELED
    positionMode: Optional[PositionMode] = None # ONE_WAY | HEDGE
    marginMode: Optional[MarginMode] = None     # ISOLATION | CROSS
    leverage: Optional[int] = None
    # TP/SL fields (populated when order was placed with TP/SL)
    tpPrice: FloatStr = 0.0
    tpStopType: Optional[StopType] = None       # MARK_PRICE | LAST_PRICE
    tpOrderType: Optional[OrderType] = None
    tpOrderPrice: FloatStr = 0.0
    slPrice: FloatStr = 0.0
    slStopType: Optional[StopType] = None
    slOrderType: Optional[OrderType] = None
    slOrderPrice: FloatStr = 0.0
    ctime: int = 0                              # creation time ms
    mtime: Optional[int] = None                 # last modification time ms

    @property
    def remaining(self) -> float:
        """Unfilled quantity."""
        return max(self.qty - self.tradeQty, 0.0)


class PendingOrdersData(BaseModel):
    """Wrapper returned by get_pending_orders."""

    model_config = ConfigDict(extra="allow")

    orderList: List[PendingOrder] = []


# ---------------------------------------------------------------------------
# POST /api/v1/futures/account/change_leverage
# data: SetLeverageData  (may be None or an echo of the request params)
# ---------------------------------------------------------------------------

class SetLeverageData(BaseModel):
    """
    Confirmation payload for change_leverage.

    POST /api/v1/futures/account/change_leverage
    The API may return None or an echo of the request; all fields are optional.
    """

    model_config = ConfigDict(extra="allow")

    symbol: Optional[str] = None
    leverage: Optional[str] = None   # echoed as string
    marginCoin: Optional[str] = None


# ---------------------------------------------------------------------------
# POST /api/v1/futures/account/change_position_mode
# data: SetPositionModeData
# ---------------------------------------------------------------------------

class SetPositionModeData(BaseModel):
    """
    Confirmation payload for change_position_mode.

    POST /api/v1/futures/account/change_position_mode
    """

    model_config = ConfigDict(extra="allow")

    positionMode: Optional[PositionMode] = None  # ONE_WAY | HEDGE

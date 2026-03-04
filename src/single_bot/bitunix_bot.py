"""
src/single_bot/bitunix_bot.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Bitunix grid-trading bot for AS-Grid.

Follows the same structure as gate_bot.py:
  - Module-level constants from environment variables
  - ``validate_config()`` guard
  - ``GridTradingBot`` class with:
      * REST init + leverage/position-mode setup
      * Grid price helpers (update_mid_price, get_take_profit_quantity)
      * place_order / place_take_profit_order / place_long_orders / place_short_orders
      * cancel_order / cancel_orders_for_side / check_orders_status
      * run() → connect_websocket() with per-channel subscription + dispatch
      * Per-channel WS handlers (ticker, position, order, balance)
      * Periodic REST reconciliation
      * Telegram notification helpers
  - ``main()`` + ``__main__`` guard

WebSocket channels used
-----------------------
Public  (wss://fapi.bitunix.com/public/) :  ticker
Private (wss://fapi.bitunix.com/private/): position, order, balance
  Private channels require a login frame sent immediately after connect.

References
----------
- https://openapidoc.bitunix.com/doc/common/introduction.html
- https://openapidoc.bitunix.com/doc/websocket/prepare/WebSocket.html
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import websockets
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path bootstrap so ``src.exchange`` is importable regardless of cwd.
# Mirrors the sys.path tweak used in binance_bot.py.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.exchange.bitunix import BitunixExchange, WS_PUBLIC, WS_PRIVATE  # noqa: E402
from src.single_bot.indicators import CandleBuffer, Signals  # noqa: E402

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Telegram configuration
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
ENABLE_NOTIFICATIONS: bool = os.getenv("ENABLE_NOTIFICATIONS", "true").lower() == "true"
NOTIFICATION_INTERVAL: int = int(os.getenv("NOTIFICATION_INTERVAL", "3600"))

# ---------------------------------------------------------------------------
# Bot configuration (from env)
# ---------------------------------------------------------------------------
API_KEY: str = os.getenv("API_KEY", "")
API_SECRET: str = os.getenv("API_SECRET", "")
COIN_NAME: str = os.getenv("COIN_NAME", "XRP")
GRID_SPACING: float = float(os.getenv("GRID_SPACING", "0.015"))   # backtest winner: 1.5%
INITIAL_QUANTITY: int = int(os.getenv("INITIAL_QUANTITY", "1"))
LEVERAGE: int = int(os.getenv("LEVERAGE", "2"))                    # backtest winner: 2×

# ---------------------------------------------------------------------------
# Regime filter + BTBW spacing (ported from backtest winning configs: h0 + btbw)
# ---------------------------------------------------------------------------
# REGIME_EMA_PERIOD    — EMA period for bull/bear regime detection (175 = winner)
# REGIME_HYSTERESIS_PCT — hysteresis band around the regime EMA (h0 = 0.00 = winner)
# BULL_SPACING         — grid spacing used when price ≥ regime_ema (bull regime)
# BEAR_SPACING         — grid spacing used when price < regime_ema  (bear regime)
#
# Defaults: BULL_SPACING and BEAR_SPACING both fall back to GRID_SPACING when
# not set, making this a pure regime-halt (no BTBW) unless both are explicitly
# configured.  Set BULL_SPACING=0.010 and BEAR_SPACING=0.015 for full BTBW.
# ---------------------------------------------------------------------------
REGIME_EMA_PERIOD:     int   = int(os.getenv("REGIME_EMA_PERIOD",     "175"))
REGIME_HYSTERESIS_PCT: float = float(os.getenv("REGIME_HYSTERESIS_PCT", "0.00"))
BULL_SPACING:          float = float(os.getenv("BULL_SPACING",          str(GRID_SPACING)))
BEAR_SPACING:          float = float(os.getenv("BEAR_SPACING",          str(GRID_SPACING)))

# ---------------------------------------------------------------------------
# Derived / fixed constants (mirrors gate_bot.py)
# ---------------------------------------------------------------------------
POSITION_THRESHOLD: float = 10 * INITIAL_QUANTITY / GRID_SPACING * 2 / 100
POSITION_LIMIT: float = 5 * INITIAL_QUANTITY / GRID_SPACING * 2 / 100
ORDER_COOLDOWN_TIME: int = 60   # s: pause after entering lockdown
SYNC_TIME: int = 3              # s: REST reconciliation interval
ORDER_FIRST_TIME: int = 1       # s: pause before placing first grid

# ---------------------------------------------------------------------------
# Trend-capture configuration
# ---------------------------------------------------------------------------
# These values mirror the confirmed-optimal params from XRP_CONFIG / FINAL_s90_l10.
# They will be locked in after the full backtest chain results are analysed.
# ---------------------------------------------------------------------------
TREND_LOOKBACK_CANDLES: int   = 10    # price velocity window (candles)
TREND_VELOCITY_PCT:     float = 0.04  # 4% move over lookback → trend detected
TREND_CAP_VEL_PCT:      float = 0.06  # 6% minimum to actually open a capture position
TREND_CONFIRM_CANDLES:  int   = 3     # consecutive above-threshold candles before acting
TREND_COOLDOWN_CANDLES: int   = 30    # quiet candles before resuming hedge mode
TREND_TRAIL_PCT:        float = 0.04  # trailing stop distance (4% from peak)
TREND_SIZE_PCT:         float = 0.90  # capture position size as fraction of equity
TREND_FORCE_CLOSE_GRID: bool  = True  # close opposing grid side on trend fire

# ADX gate + fast re-entry (mirrors v9 re_reentry winning config)
ADX_MIN_TREND:      float = 25.0  # min ADX to allow opening a trend capture position
ADX_GRID_PAUSE:     float = 35.0  # pause new grid legs when ADX ≥ this threshold
TREND_REENTRY_FAST: bool  = True  # skip 30-candle cooldown; re-confirm in 3 candles

# ---------------------------------------------------------------------------
# Layer 1–4 loss-mitigation gates (mirrors v21/v22 backtest configs)
# ---------------------------------------------------------------------------
# Set via env to allow per-deployment tuning without code changes.
# All default to OFF (0 / False) so existing deployments are unaffected.
# ---------------------------------------------------------------------------
ATR_PARABOLIC_MULT:    float = float(os.getenv("ATR_PARABOLIC_MULT",    "0.0"))   # L1: block trend entries when ATR > mult × SMA(ATR,20)
HTF_EMA_ALIGN:         bool  = os.getenv("HTF_EMA_ALIGN", "false").lower() == "true"  # L2: require HTF EMA agreement for trend entry
REGIME_VOTE_MODE:      bool  = os.getenv("REGIME_VOTE_MODE", "false").lower() == "true"  # L3: 2-of-3 vote for bear halt
GRID_SLEEP_ATR_THRESH: float = float(os.getenv("GRID_SLEEP_ATR_THRESH", "0.0"))   # L4: pause grid when ATR/price < threshold

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("log", exist_ok=True)

_script_name = os.path.splitext(os.path.basename(__file__))[0]
_handlers: list = [logging.StreamHandler()]
try:
    _fh = logging.FileHandler(f"log/{_script_name}.log")
    _handlers.append(_fh)
    print(f"Logging to: log/{_script_name}.log")
except Exception as exc:
    print(f"Warning: cannot create log file ({exc}); console only.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=_handlers,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def validate_config() -> None:
    """Raise ``ValueError`` if any required env var is missing or invalid."""
    global ENABLE_NOTIFICATIONS
    if not API_KEY or not API_SECRET:
        raise ValueError("API_KEY and API_SECRET must be set.")
    if not (0 < GRID_SPACING < 1):
        raise ValueError("GRID_SPACING must be in (0, 1).")
    if INITIAL_QUANTITY <= 0:
        raise ValueError("INITIAL_QUANTITY must be > 0.")
    if not (0 < LEVERAGE <= 100):
        raise ValueError("LEVERAGE must be in [1, 100].")
    if ENABLE_NOTIFICATIONS and (not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID):
        logger.warning(
            "Telegram tokens missing – disabling notifications."
        )
        ENABLE_NOTIFICATIONS = False
    logger.info(
        "Config OK – coin=%s grid_spacing=%s bull=%s bear=%s "
        "regime_ema=%d hyst=%.2f%% initial_qty=%d leverage=%dx",
        COIN_NAME, GRID_SPACING, BULL_SPACING, BEAR_SPACING,
        REGIME_EMA_PERIOD, REGIME_HYSTERESIS_PCT * 100,
        INITIAL_QUANTITY, LEVERAGE,
    )


# ---------------------------------------------------------------------------
# Grid trading bot
# ---------------------------------------------------------------------------

class GridTradingBot(BitunixExchange):
    """
    Bitunix perpetual-futures grid-trading bot.

    Inherits ``BitunixExchange`` for all REST calls and WebSocket auth
    helpers.  Adds grid logic, WS event loop, and Telegram notifications
    following the same patterns as ``gate_bot.GridTradingBot``.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        coin_name: str,
        grid_spacing: float,
        initial_quantity: int,
        leverage: int,
    ) -> None:
        super().__init__(api_key=api_key, secret_key=api_secret)

        self.coin_name = coin_name
        self.grid_spacing = grid_spacing          # base spacing (fallback pre-regime warm-up)
        self.bull_spacing: float = BULL_SPACING   # BTBW: spacing in bull regime (price ≥ regime_ema)
        self.bear_spacing: float = BEAR_SPACING   # BTBW: spacing in bear regime (price < regime_ema)
        self.regime_hysteresis_pct: float = REGIME_HYSTERESIS_PCT  # h0=0.00 — no band
        self.atr_parabolic_mult: float = ATR_PARABOLIC_MULT        # L1: 0=off
        self.htf_ema_align: bool = HTF_EMA_ALIGN                   # L2: False=off
        self.regime_vote_mode: bool = REGIME_VOTE_MODE             # L3: False=off
        self.grid_sleep_atr_thresh: float = GRID_SLEEP_ATR_THRESH  # L4: 0=off
        self.initial_quantity = initial_quantity
        self.leverage = leverage

        # Symbol used for all REST calls and WS subscriptions
        self.symbol: str = f"{coin_name}USDT"

        # Price precision (decimal places) – fetched from market info on setup
        self.price_precision: int = 4

        # Position state – kept live by WS + periodic REST reconciliation
        self.long_position: float = 0.0
        self.short_position: float = 0.0

        # Per-side order quantities used when placing grid legs
        self.long_initial_quantity: int = initial_quantity
        self.short_initial_quantity: int = initial_quantity

        # Open-order counts updated by WS order channel + check_orders_status()
        self.buy_long_orders: float = 0.0    # long entry pending qty
        self.sell_long_orders: float = 0.0   # long TP pending qty
        self.sell_short_orders: float = 0.0  # short entry pending qty
        self.buy_short_orders: float = 0.0   # short TP pending qty

        # Timestamps for debouncing order placement
        self.last_long_order_time: float = 0.0
        self.last_short_order_time: float = 0.0

        # Live price feed (populated by WS ticker channel)
        self.latest_price: float = 0.0
        self.best_bid_price: Optional[float] = None
        self.best_ask_price: Optional[float] = None

        # Balance cache updated from WS balance channel
        self.balance: Dict[str, Any] = {}

        # Grid boundary prices (recalculated on each ticker update)
        self.mid_price_long: float = 0.0
        self.lower_price_long: float = 0.0   # long entry
        self.upper_price_long: float = 0.0   # long take-profit
        self.mid_price_short: float = 0.0
        self.upper_price_short: float = 0.0  # short entry
        self.lower_price_short: float = 0.0  # short take-profit

        # Telegram / notification state
        self.last_summary_time: float = 0.0
        self.startup_notified: bool = False

        # Threshold alert flags – prevent repeated messages for the same event
        self.long_threshold_alerted: bool = False
        self.short_threshold_alerted: bool = False
        self.risk_reduction_alerted: bool = False

        # ── Candle buffer + indicator engine ──────────────────────────────
        # Seeded from REST on startup; kept live via WS kline channel.
        # regime_ema_period=175 matches the winning backtest config (h0, BTBW).
        # maxlen=210 ensures the 175-period EMA has a comfortable warm-up margin.
        self.candle_buffer: CandleBuffer = CandleBuffer(
            maxlen=210,
            interval="15min",
            adx_period=14,
            atr_period=14,
            rsi_period=14,
            bb_period=20,
            bb_mult=2.0,
            ema_fast=9,
            ema_slow=21,
            regime_ema_period=REGIME_EMA_PERIOD,
            vol_period=20,
            ms_lookback=20,
        )
        # Latest computed indicator snapshot (refreshed on each candle close).
        self.latest_signals: Optional[Signals] = None

        # ── Trend-capture runtime state ───────────────────────────────────
        # Mirrors the backtest GridSearch fields of the same names.
        self.trend_mode: Optional[str] = None           # "up" | "down" | None
        self.trend_pending_dir: Optional[str] = None    # direction accumulating confirms
        self.trend_confirm_counter: int = 0             # consecutive above-threshold candles
        self.trend_cooldown_counter: int = 0            # quiet candles since last trend ended
        self.trend_position: Optional[Dict[str, Any]] = None
        # trend_position keys: side, entry, qty, peak

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """
        Prepare the exchange connection: set leverage and position mode.

        Called once at startup before entering the WebSocket loop.
        Bitunix rejects set_position_mode if there are open positions, so we
        check first and skip if already in HEDGE mode.
        """
        logger.info("Setting leverage %dx on %s…", self.leverage, self.symbol)
        await self.set_leverage(self.symbol, self.leverage)
        logger.info("Setting HEDGE position mode…")
        await self.set_position_mode(hedge_mode=True)
        logger.info("Seeding candle buffer for %s…", self.symbol)
        await self.candle_buffer.seed(self, self.symbol)
        logger.info("Candle buffer ready – %d closed candles loaded", len(self.candle_buffer._closed))

    # ------------------------------------------------------------------
    # Trend-capture helpers
    # ------------------------------------------------------------------

    def _equity(self) -> float:
        """Return a best-effort equity estimate (available + margin USDT)."""
        usdt = self.balance.get("USDT", {})
        return float(usdt.get("available", 0.0)) + float(usdt.get("margin", 0.0))

    # ------------------------------------------------------------------
    # Layer 1–4 gate helpers (mirror backtest simulation engine)
    # ------------------------------------------------------------------

    def _parabolic_gate(self) -> bool:
        """Layer 1: True when ATR > mult × SMA(ATR,20) — blocks trend entries."""
        if self.atr_parabolic_mult <= 0 or self.latest_signals is None:
            return False
        atr = self.latest_signals.atr
        atr_sma = self.latest_signals.atr_sma
        if atr_sma <= 0:
            return False
        return atr > self.atr_parabolic_mult * atr_sma

    def _htf_bull(self) -> bool:
        """Layer 2: True when HTF fast EMA > HTF slow EMA (bullish alignment)."""
        if self.latest_signals is None:
            return False
        return self.latest_signals.htf_ema_fast > self.latest_signals.htf_ema_slow

    def _htf_bear(self) -> bool:
        """Layer 2: True when HTF fast EMA < HTF slow EMA (bearish alignment)."""
        if self.latest_signals is None:
            return False
        return self.latest_signals.htf_ema_fast < self.latest_signals.htf_ema_slow

    def _regime_vote_halt_longs(self) -> bool:
        """Layer 3: True when 2-of-3 EMAs vote bear — blocks long grid entries."""
        if not self.regime_vote_mode or self.latest_signals is None:
            return False
        sig = self.latest_signals
        price = sig.close
        hyst = self.regime_hysteresis_pct
        bear_votes = sum([
            price < sig.regime_ema * (1.0 - hyst) if sig.regime_ema > 0 else False,
            price < sig.regime_ema_87 * (1.0 - hyst) if sig.regime_ema_87 > 0 else False,
            price < sig.regime_ema_42 * (1.0 - hyst) if sig.regime_ema_42 > 0 else False,
        ])
        return bear_votes >= 2

    def _grid_sleep(self) -> bool:
        """Layer 4: True when ATR/price < threshold — pauses ALL grid entries."""
        if self.grid_sleep_atr_thresh <= 0 or self.latest_signals is None:
            return False
        price = self.latest_signals.close
        if price <= 0:
            return False
        return (self.latest_signals.atr / price) < self.grid_sleep_atr_thresh

    async def _open_trend_trade(self, side: str, price: float) -> None:
        """
        Open a directional trend-capture position.

        Parameters
        ----------
        side:
            ``"long"`` to buy, ``"short"`` to sell.
        price:
            Current close price (used for size calculation only — order is MARKET).
        """
        equity = self._equity()
        if equity <= 0 or price <= 0:
            logger.warning("_open_trend_trade: equity=%.2f price=%.4f — skipping", equity, price)
            return

        cap_margin = min(equity * TREND_SIZE_PCT, equity * 0.90)
        cap_qty = (cap_margin * self.leverage) / price
        qty_contracts = int(cap_qty)
        if qty_contracts < 1:
            logger.warning("_open_trend_trade: calculated qty < 1 — skipping (equity=%.2f)", equity)
            return

        order_side = "buy" if side == "long" else "sell"
        order = await self.place_market_order(
            symbol=self.symbol,
            side=order_side,
            quantity=qty_contracts,
            reduce_only=False,
            position_side=side,
        )
        if order:
            self.trend_position = {
                "side": side,
                "entry": price,
                "qty": qty_contracts,
                "peak": price,
            }
            logger.info(
                "🎯 Trend %s opened at %.4f  qty=%d  margin≈$%.0f",
                side.upper(), price, qty_contracts, cap_margin,
            )
            await self.send_telegram_message(
                f"🎯 *Trend {side.upper()} opened*\n"
                f"Price: `{price:.4f}`  Qty: `{qty_contracts}`  Margin≈`${cap_margin:.0f}`"
            )
        else:
            logger.error("_open_trend_trade: market order failed for %s", side)

    async def _close_trend_trade(self, price: float, reason: str = "trail") -> None:
        """
        Close an open trend-capture position with a reduce-only market order.

        Parameters
        ----------
        price:
            Current price (logged only).
        reason:
            Human-readable close reason ("trail" | "reversal").
        """
        tp = self.trend_position
        if tp is None:
            return

        close_side = "sell" if tp["side"] == "long" else "buy"
        order = await self.place_market_order(
            symbol=self.symbol,
            side=close_side,
            quantity=tp["qty"],
            reduce_only=True,
            position_side=tp["side"],
        )
        if order:
            pnl_est = (
                (price - tp["entry"]) * tp["qty"]
                if tp["side"] == "long"
                else (tp["entry"] - price) * tp["qty"]
            )
            logger.info(
                "🎯 Trend %s closed (%s) entry=%.4f exit=%.4f est_pnl=%+.2f",
                tp["side"].upper(), reason, tp["entry"], price, pnl_est,
            )
            await self.send_telegram_message(
                f"🎯 *Trend {tp['side'].upper()} closed* ({reason})\n"
                f"Entry: `{tp['entry']:.4f}`  Exit: `{price:.4f}`  Est PnL: `{pnl_est:+.2f}`"
            )
            self.trend_position = None
            if TREND_REENTRY_FAST:
                # Skip the 30-candle cooldown — reset state so confirmation
                # counter can re-fire within 3 candles on the next velocity signal.
                self.trend_mode = None
                self.trend_confirm_counter = 0
                self.trend_pending_dir = None
        else:
            logger.error("_close_trend_trade: market close order failed for %s", tp["side"])

    async def _evaluate_trend(self, price: float) -> None:
        """
        Called on every closed 15-min candle.  Mirrors the trend-detection and
        trend-capture logic from ``backtest_grid_bitunix.GridSearch`` exactly:

        1. Compute price velocity over ``TREND_LOOKBACK_CANDLES``.
        2. Manage existing trend position (trailing-stop check).
        3. Accumulate confirmation counter.
        4. On confirmation:
           a. Optionally force-close the opposing grid side.
           b. Open a directional capture position if velocity ≥ TREND_CAP_VEL_PCT.
        5. Count down cooldown once trend fades and no position is open.
        """
        # ── 1. Velocity ────────────────────────────────────────────────
        closed = list(self.candle_buffer._closed)
        if len(closed) < TREND_LOOKBACK_CANDLES:
            return  # not enough history yet

        past_price = closed[-TREND_LOOKBACK_CANDLES].close
        if past_price <= 0:
            return
        velocity = (price - past_price) / past_price
        trending_up   = velocity >  TREND_VELOCITY_PCT
        trending_down = velocity < -TREND_VELOCITY_PCT

        # ── 2. Manage existing position (trailing stop) ─────────────────
        if self.trend_position is not None:
            tp = self.trend_position
            if tp["side"] == "long":
                if price > tp["peak"]:
                    tp["peak"] = price
                trail_stop = tp["peak"] * (1.0 - TREND_TRAIL_PCT)
                if price <= trail_stop or trending_down:
                    await self._close_trend_trade(
                        price, reason="trail" if price <= trail_stop else "reversal"
                    )
            else:  # short
                if price < tp["peak"]:
                    tp["peak"] = price
                trail_stop = tp["peak"] * (1.0 + TREND_TRAIL_PCT)
                if price >= trail_stop or trending_up:
                    await self._close_trend_trade(
                        price, reason="trail" if price >= trail_stop else "reversal"
                    )

        # ── 3. Confirmation counter ─────────────────────────────────────
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
            self.trend_confirm_counter = 0
            self.trend_pending_dir = None

        confirmed_up   = trending_up   and self.trend_confirm_counter >= TREND_CONFIRM_CANDLES
        confirmed_down = trending_down and self.trend_confirm_counter >= TREND_CONFIRM_CANDLES

        # ── 4. Act on confirmed trend ───────────────────────────────────
        if confirmed_up and self.trend_mode != "up":
            self.trend_mode = "up"
            self.trend_cooldown_counter = 0
            logger.info(
                "📈 Trend UP confirmed (vel=%.2f%%  confirms=%d)",
                velocity * 100, self.trend_confirm_counter,
            )
            # Force-close opposing (short) grid
            if TREND_FORCE_CLOSE_GRID and self.short_position > 0:
                logger.info("Trend UP: force-closing short grid (pos=%.4f)", self.short_position)
                await self.cancel_orders_for_side(self.symbol, "short")
                await self.place_market_order(
                    symbol=self.symbol,
                    side="buy",
                    quantity=int(self.short_position),
                    reduce_only=True,
                    position_side="short",
                )
            # Open long capture if velocity strong enough, no position yet,
            # and ADX confirms a real trend (not just noise).
            adx_now = self.latest_signals.adx if self.latest_signals else 0.0
            if (
                self.trend_position is None
                and velocity >= TREND_CAP_VEL_PCT
                and adx_now >= ADX_MIN_TREND
                and not self._parabolic_gate()     # Layer 1
                and (not self.htf_ema_align or self._htf_bull())  # Layer 2
            ):
                await self._open_trend_trade("long", price)

        elif confirmed_down and self.trend_mode != "down":
            self.trend_mode = "down"
            self.trend_cooldown_counter = 0
            logger.info(
                "📉 Trend DOWN confirmed (vel=%.2f%%  confirms=%d)",
                velocity * 100, self.trend_confirm_counter,
            )
            # Force-close opposing (long) grid
            if TREND_FORCE_CLOSE_GRID and self.long_position > 0:
                logger.info("Trend DOWN: force-closing long grid (pos=%.4f)", self.long_position)
                await self.cancel_orders_for_side(self.symbol, "long")
                await self.place_market_order(
                    symbol=self.symbol,
                    side="sell",
                    quantity=int(self.long_position),
                    reduce_only=True,
                    position_side="long",
                )
            # Open short capture if velocity strong enough, no position yet,
            # and ADX confirms a real trend (not just noise).
            adx_now = self.latest_signals.adx if self.latest_signals else 0.0
            if (
                self.trend_position is None
                and abs(velocity) >= TREND_CAP_VEL_PCT
                and adx_now >= ADX_MIN_TREND
                and not self._parabolic_gate()      # Layer 1
                and (not self.htf_ema_align or self._htf_bear())  # Layer 2
            ):
                await self._open_trend_trade("short", price)

        # ── 5. Cooldown when trend fades (no open position) ─────────────
        elif self.trend_mode is not None and self.trend_position is None:
            if abs(velocity) < TREND_VELOCITY_PCT * 0.5:
                self.trend_cooldown_counter += 1
                if self.trend_cooldown_counter >= TREND_COOLDOWN_CANDLES:
                    logger.info(
                        "🔄 Trend %s ended — resuming hedge mode (cooldown complete)",
                        self.trend_mode.upper(),
                    )
                    self.trend_mode = None
                    self.trend_cooldown_counter = 0
                    self.trend_pending_dir = None
                    self.trend_confirm_counter = 0
            else:
                self.trend_cooldown_counter = 0

    # ------------------------------------------------------------------
    # Grid price helpers
    # ------------------------------------------------------------------

    def update_mid_price(self, side: str, latest_price: float) -> None:
        """
        Recalculate grid boundary prices around *latest_price*.

        BTBW regime-aware spacing
        -------------------------
        When the regime EMA (175-period) has warmed up the active spacing is
        selected from ``bull_spacing`` / ``bear_spacing`` based on whether the
        current price sits above or below ``regime_ema × (1 − hysteresis)``.
        Before the EMA warms up (< 180 closed candles) ``grid_spacing`` is
        used as a safe fallback.

        Long side:
          lower_price_long = latest_price × (1 − spacing)  ← entry
          upper_price_long = latest_price × (1 + spacing)  ← take-profit

        Short side:
          upper_price_short = latest_price × (1 + spacing) ← entry
          lower_price_short = latest_price × (1 − spacing) ← take-profit

        All prices are rounded to ``self.price_precision`` decimal places.
        """
        def _round(p: float) -> float:
            fmt = Decimal("0." + "0" * self.price_precision)
            return float(Decimal(str(p)).quantize(fmt, rounding=ROUND_HALF_UP))

        # ── BTBW: resolve effective spacing from regime signal ─────────────
        sig = self.latest_signals
        if sig is not None and sig.regime_ema > 0:
            regime_threshold = sig.regime_ema * (1.0 - self.regime_hysteresis_pct)
            in_bull = latest_price >= regime_threshold
            effective_spacing = self.bull_spacing if in_bull else self.bear_spacing
        else:
            effective_spacing = self.grid_spacing   # fallback: EMA not yet warmed up

        if side == "long":
            self.mid_price_long = _round(latest_price)
            self.lower_price_long = _round(latest_price * (1 - effective_spacing))
            self.upper_price_long = _round(latest_price * (1 + effective_spacing))
        elif side == "short":
            self.mid_price_short = _round(latest_price)
            self.upper_price_short = _round(latest_price * (1 + effective_spacing))
            self.lower_price_short = _round(latest_price * (1 - effective_spacing))

    def get_take_profit_quantity(self, position: float, side: str) -> int:
        """
        Determine the number of contracts for the take-profit order.

        Returns ``self.initial_quantity`` when position ≤ POSITION_LIMIT,
        otherwise scales up proportionally (mirrors gate_bot logic).
        Updates ``self.long_initial_quantity`` / ``self.short_initial_quantity``.
        """
        if position <= POSITION_LIMIT:
            qty = self.initial_quantity
        else:
            qty = max(self.initial_quantity, int(position / POSITION_LIMIT) * self.initial_quantity)

        if side == "long":
            self.long_initial_quantity = qty
        else:
            self.short_initial_quantity = qty
        return qty

    # ------------------------------------------------------------------
    # REST position / order state
    # ------------------------------------------------------------------

    async def get_position(self) -> Tuple[float, float]:
        """Fetch current position via REST; returns (long_qty, short_qty)."""
        return await self.get_positions(self.symbol)

    async def check_orders_status(self) -> Tuple[float, float, float, float]:
        """
        Fetch all open orders and classify into four buckets.

        Returns (buy_long_qty, sell_long_qty, sell_short_qty, buy_short_qty).

        Bitunix hedge-mode classification:
          side=BUY  + reduceOnly=False → buy_long  (open long entry)
          side=SELL + reduceOnly=True  → sell_long (close long / long TP)
          side=SELL + reduceOnly=False → sell_short (open short entry)
          side=BUY  + reduceOnly=True  → buy_short (close short / short TP)
        """
        orders = await self.get_open_orders(self.symbol)
        buy_long = sell_long = sell_short = buy_short = 0.0

        for o in orders:
            remaining = float(o.get("remaining", 0))
            side = o.get("side", "").upper()
            reduce = bool(o.get("reduceOnly", False))

            if side == "BUY" and not reduce:
                buy_long += remaining
            elif side == "SELL" and reduce:
                sell_long += remaining
            elif side == "SELL" and not reduce:
                sell_short += remaining
            elif side == "BUY" and reduce:
                buy_short += remaining

        return buy_long, sell_long, sell_short, buy_short

    # ------------------------------------------------------------------
    # Order placement wrappers (sync-style API matching gate_bot.place_order)
    # ------------------------------------------------------------------

    async def place_order(
        self,
        side: str,              # "buy" or "sell"
        price: float,
        quantity: int,
        is_reduce_only: bool = False,
        position_side: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Place a limit order (entry or take-profit leg).

        Thin wrapper around ``BitunixExchange.place_limit_order`` that
        normalises arguments and swallows exceptions so the grid loop
        continues on transient errors.
        """
        return await self.place_limit_order(
            symbol=self.symbol,
            side=side,
            quantity=quantity,
            price=price,
            reduce_only=is_reduce_only,
            position_side=position_side,
        )

    async def place_take_profit_order(
        self, side: str, price: float, quantity: int
    ) -> None:
        """
        Place a reduce-only limit take-profit order.

        side='long'  → SELL + reduceOnly=True to close a long position
        side='short' → BUY  + reduceOnly=True to close a short position
        """
        if side == "long":
            order = await self.place_limit_order(
                self.symbol, "sell", quantity, price, reduce_only=True
            )
            if order:
                logger.info(
                    "Long TP placed: SELL %d %s @ %s", quantity, self.symbol, price
                )
        elif side == "short":
            order = await self.place_limit_order(
                self.symbol, "buy", quantity, price, reduce_only=True
            )
            if order:
                logger.info(
                    "Short TP placed: BUY %d %s @ %s", quantity, self.symbol, price
                )

    # ------------------------------------------------------------------
    # Grid order management (mirrors gate_bot.place_long_orders / place_short_orders)
    # ------------------------------------------------------------------

    async def place_long_orders(self, latest_price: float) -> None:
        """Manage the long-side grid: cancel stale orders then place fresh TP + entry."""
        try:
            # ADX grid pause — don't add new legs during strong trending conditions
            if (
                self.latest_signals is not None
                and self.latest_signals.adx >= ADX_GRID_PAUSE
            ):
                logger.info(
                    "⏸️ ADX grid pause long (adx=%.1f ≥ %.1f) — skipping grid refresh",
                    self.latest_signals.adx, ADX_GRID_PAUSE,
                )
                return

            # Regime filter — halt new long legs when price is in bear regime
            # (price < regime_ema × (1 − hysteresis)).  Mirrors backtest halt_grid_longs.
            if self.regime_vote_mode:
                # Layer 3: 2-of-3 regime EMA vote replaces single-EMA check
                if self._regime_vote_halt_longs():
                    logger.info(
                        "🗳️ Regime vote: halting long grid (2-of-3 EMAs vote bear)",
                    )
                    return
            elif self.latest_signals is not None and self.latest_signals.regime_ema > 0:
                regime_threshold = self.latest_signals.regime_ema * (1.0 - self.regime_hysteresis_pct)
                if self.latest_price < regime_threshold:
                    logger.info(
                        "🐻 Regime filter: halting long grid (price=%.4f < ema%d=%.4f)",
                        self.latest_price, REGIME_EMA_PERIOD, self.latest_signals.regime_ema,
                    )
                    return

            # Layer 4: Grid sleep — pause ALL grid entries when ATR/price too low
            if self._grid_sleep():
                logger.info(
                    "💤 Grid sleep: long side paused (ATR/price < %.4f)",
                    self.grid_sleep_atr_thresh,
                )
                return

            self.get_take_profit_quantity(self.long_position, "long")
            if self.long_position <= 0:
                return

            if self.long_position > POSITION_THRESHOLD:
                logger.info(
                    "Long position %s exceeds threshold %s – lockdown mode",
                    self.long_position, POSITION_THRESHOLD,
                )
                await self.check_and_notify_position_threshold("long", self.long_position)
                if self.sell_long_orders <= 0:
                    # Place a lockdown TP above current price
                    ratio = float((int(self.long_position / max(self.short_position, 1)) / 100) + 1)
                    await self.place_take_profit_order(
                        "long",
                        self.latest_price * ratio,
                        self.long_initial_quantity,
                    )
            else:
                await self.check_and_notify_position_threshold("long", self.long_position)
                self.update_mid_price("long", latest_price)
                await self.cancel_orders_for_side(self.symbol, "long")
                await self.place_take_profit_order("long", self.upper_price_long, self.long_initial_quantity)
                await self.place_order("buy", self.lower_price_long, self.long_initial_quantity, False, "long")
                logger.info(
                    "Long grid placed – TP @ %s, entry @ %s",
                    self.upper_price_long, self.lower_price_long,
                )
        except Exception as exc:
            logger.error("place_long_orders failed: %s", exc)

    async def place_short_orders(self, latest_price: float) -> None:
        """Manage the short-side grid: cancel stale orders then place fresh TP + entry."""
        try:
            # ADX grid pause — don't add new legs during strong trending conditions
            if (
                self.latest_signals is not None
                and self.latest_signals.adx >= ADX_GRID_PAUSE
            ):
                logger.info(
                    "⏸️ ADX grid pause short (adx=%.1f ≥ %.1f) — skipping grid refresh",
                    self.latest_signals.adx, ADX_GRID_PAUSE,
                )
                return
            # Layer 4: Grid sleep — pause ALL grid entries when ATR/price too low
            if self._grid_sleep():
                logger.info(
                    "💤 Grid sleep: short side paused (ATR/price < %.4f)",
                    self.grid_sleep_atr_thresh,
                )
                return
            self.get_take_profit_quantity(self.short_position, "short")
            if self.short_position <= 0:
                return

            if self.short_position > POSITION_THRESHOLD:
                logger.info(
                    "Short position %s exceeds threshold %s – lockdown mode",
                    self.short_position, POSITION_THRESHOLD,
                )
                await self.check_and_notify_position_threshold("short", self.short_position)
                if self.buy_short_orders <= 0:
                    ratio = float((int(self.short_position / max(self.long_position, 1)) / 100) + 1)
                    await self.place_take_profit_order(
                        "short",
                        self.latest_price / ratio,
                        self.short_initial_quantity,
                    )
            else:
                await self.check_and_notify_position_threshold("short", self.short_position)
                self.update_mid_price("short", latest_price)
                await self.cancel_orders_for_side(self.symbol, "short")
                await self.place_take_profit_order("short", self.lower_price_short, self.short_initial_quantity)
                await self.place_order("sell", self.upper_price_short, self.short_initial_quantity, False, "short")
                logger.info(
                    "Short grid placed – TP @ %s, entry @ %s",
                    self.lower_price_short, self.upper_price_short,
                )
        except Exception as exc:
            logger.error("place_short_orders failed: %s", exc)

    # ------------------------------------------------------------------
    # Risk / threshold notifications (mirrors gate_bot.py)
    # ------------------------------------------------------------------

    async def check_and_notify_position_threshold(
        self, side: str, position: float
    ) -> None:
        """Send Telegram alert when position crosses POSITION_THRESHOLD (both ways)."""
        over = position > POSITION_THRESHOLD
        alerted_attr = f"{side}_threshold_alerted"
        already = getattr(self, alerted_attr, False)

        if over and not already:
            await self.send_telegram_message(
                f"⚠️ *{side.upper()} position risk*\n"
                f"Position {position:.2f} exceeds threshold {POSITION_THRESHOLD:.2f}\n"
                f"Latest price: {self.latest_price:.8f}",
                urgent=True,
            )
            setattr(self, alerted_attr, True)
        elif not over and already:
            await self.send_telegram_message(
                f"✅ *{side.upper()} position recovered*\n"
                f"Position {position:.2f} below threshold {POSITION_THRESHOLD:.2f}",
            )
            setattr(self, alerted_attr, False)

    async def check_and_notify_risk_reduction(self) -> None:
        """Alert when both sides simultaneously exceed 80 % of POSITION_THRESHOLD."""
        limit = POSITION_THRESHOLD * 0.8
        both_over = self.long_position >= limit and self.short_position >= limit

        if both_over and not self.risk_reduction_alerted:
            await self.send_telegram_message(
                f"📉 *Dual position risk*\n"
                f"Long: {self.long_position}, Short: {self.short_position}\n"
                f"Both exceed {limit:.2f}",
            )
            self.risk_reduction_alerted = True
        elif not both_over and self.risk_reduction_alerted:
            await self.send_telegram_message("✅ *Dual position risk resolved*")
            self.risk_reduction_alerted = False

    # ------------------------------------------------------------------
    # Stale-order watchdog
    # ------------------------------------------------------------------

    async def monitor_orders(self) -> None:
        """
        Background task: cancel orders pending more than 300 s without fill.
        Runs every 60 s. Mirrors gate_bot.monitor_orders().
        """
        while True:
            try:
                await asyncio.sleep(60)
                current_ms = int(time.time() * 1000)
                orders = await self.get_open_orders(self.symbol)
                for o in orders:
                    ctime = o.get("ctime", 0)
                    if ctime and (current_ms - ctime) > 300_000:
                        logger.info(
                            "Order %s stale (>300 s) – cancelling", o["orderId"]
                        )
                        await self.cancel_order(o["orderId"], self.symbol)
            except Exception as exc:
                logger.error("monitor_orders error: %s", exc)

    # ------------------------------------------------------------------
    # REST reconciliation
    # ------------------------------------------------------------------

    async def reconcile_state(self) -> None:
        """
        Fetch position + order state via REST and update cached attributes.
        Called periodically to recover from missed WebSocket events.
        """
        try:
            new_long, new_short = await self.get_position()
            if new_long != self.long_position or new_short != self.short_position:
                logger.info(
                    "reconcile positions: long %s→%s  short %s→%s",
                    self.long_position, new_long, self.short_position, new_short,
                )
            self.long_position = new_long
            self.short_position = new_short

            bl, sl, ss, bs = await self.check_orders_status()
            self.buy_long_orders = bl
            self.sell_long_orders = sl
            self.sell_short_orders = ss
            self.buy_short_orders = bs
        except Exception as exc:
            logger.error("reconcile_state error: %s", exc)

    # ------------------------------------------------------------------
    # WebSocket event loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Main entry point: initialise, seed state, then enter reconnect loop.

        Bitunix uses *separate* WebSocket endpoints for public and private
        channels, so we run them as two concurrent tasks:
          - public_ws_loop  → ticker
          - private_ws_loop → position, order, balance
        We also schedule the stale-order watchdog as a background task.
        """
        await self.setup()

        self.long_position, self.short_position = await self.get_position()
        logger.info(
            "Initial positions: long=%s short=%s",
            self.long_position, self.short_position,
        )
        (
            self.buy_long_orders,
            self.sell_long_orders,
            self.sell_short_orders,
            self.buy_short_orders,
        ) = await self.check_orders_status()
        logger.info(
            "Initial orders: buy_long=%s sell_long=%s sell_short=%s buy_short=%s",
            self.buy_long_orders, self.sell_long_orders,
            self.sell_short_orders, self.buy_short_orders,
        )

        await self.send_startup_notification()

        asyncio.create_task(self.monitor_orders())

        while True:
            try:
                await asyncio.gather(
                    self._public_ws_loop(),
                    self._private_ws_loop(),
                )
            except Exception as exc:
                logger.error("WebSocket top-level error: %s – reconnecting in 5 s", exc)
                await self.send_error_notification(str(exc), "WebSocket error")
                await asyncio.sleep(5)

    async def _public_ws_loop(self) -> None:
        """Connect to the public WS and subscribe to the ticker + kline channels."""
        while True:
            try:
                async with websockets.connect(WS_PUBLIC) as ws:
                    await ws.send(
                        json.dumps(
                            self.build_subscribe_payload("ticker", self.symbol)
                        )
                    )
                    await ws.send(
                        json.dumps(
                            self.build_subscribe_payload(
                                "market_kline_15min", self.symbol
                            )
                        )
                    )
                    logger.info(
                        "Public WS: subscribed ticker + market_kline_15min for %s",
                        self.symbol,
                    )
                    await self._recv_loop(ws, private=False)
            except Exception as exc:
                logger.error("Public WS error: %s – retry in 5 s", exc)
                await asyncio.sleep(5)

    async def _private_ws_loop(self) -> None:
        """Connect to the private WS, login, then subscribe to position/order/balance."""
        while True:
            try:
                async with websockets.connect(WS_PRIVATE) as ws:
                    # Authenticate
                    login_payload = self.build_ws_login_payload()
                    await ws.send(json.dumps(login_payload))
                    login_resp = json.loads(await ws.recv())
                    logger.info("Private WS login response: %s", login_resp)

                    # Subscribe to private channels (no symbol filter needed)
                    for ch in ("position", "order", "balance"):
                        await ws.send(
                            json.dumps(
                                self.build_subscribe_payload(ch, self.symbol)
                            )
                        )
                    logger.info("Private WS: subscribed position/order/balance")
                    await self._recv_loop(ws, private=True)
            except Exception as exc:
                logger.error("Private WS error: %s – retry in 5 s", exc)
                await asyncio.sleep(5)

    async def _recv_loop(self, ws: Any, *, private: bool) -> None:
        """
        Read messages from *ws* and dispatch to the appropriate handler.

        Sends periodic pings every 20 s to keep the connection alive.
        Runs a REST reconciliation every SYNC_TIME seconds.
        """
        last_ping = time.time()
        last_sync = time.time()

        while True:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=25)
                data = json.loads(message)
                ch = data.get("ch", "")

                if ch == "ticker":
                    await self.handle_ticker_update(data)
                elif ch == "market_kline_15min":
                    await self.handle_kline_update(data)
                elif ch == "position":
                    await self.handle_position_update(data)
                elif ch == "order":
                    await self.handle_order_update(data)
                elif ch == "balance":
                    await self.handle_balance_update(data)
                elif data.get("op") in ("ping", "pong"):
                    pass  # keepalive – no action needed
                else:
                    logger.debug("Unhandled WS message: %s", data)

                now = time.time()
                if now - last_ping > 20:
                    await ws.send(json.dumps(self.build_ping_payload()))
                    last_ping = now
                if now - last_sync > SYNC_TIME:
                    await self.reconcile_state()
                    await self.send_summary_notification()
                    last_sync = now

            except asyncio.TimeoutError:
                # No message for 25 s – send ping and continue
                await ws.send(json.dumps(self.build_ping_payload()))
                last_ping = time.time()
            except Exception as exc:
                logger.error("_recv_loop error: %s", exc)
                break

    # ------------------------------------------------------------------
    # WebSocket message handlers
    # ------------------------------------------------------------------

    async def handle_ticker_update(self, data: Dict[str, Any]) -> None:
        """
        Handle a ticker push.

        Bitunix ticker push fields (ch="ticker"):
          data.la  = last price
          data.q   = 24 h quote volume
          data.r   = 24 h change rate

        Updates ``self.latest_price`` then refreshes both grid sides.
        """
        try:
            payload = data.get("data", {})
            last_price = float(payload.get("la", 0))
            if last_price <= 0:
                return

            self.latest_price = last_price

            now = time.time()
            if now - self.last_long_order_time >= ORDER_FIRST_TIME:
                await self.place_long_orders(last_price)
                self.last_long_order_time = now
            if now - self.last_short_order_time >= ORDER_FIRST_TIME:
                await self.place_short_orders(last_price)
                self.last_short_order_time = now

            await self.check_and_notify_risk_reduction()
        except Exception as exc:
            logger.error("handle_ticker_update error: %s", exc)

    async def handle_kline_update(self, data: Dict[str, Any]) -> None:
        """
        Handle a ``market_kline_15min`` WS push (~every 500 ms).

        Bitunix kline WS payload::

            {
              "ch": "market_kline_15min",
              "symbol": "XRPUSDT",
              "ts": 1740000000000,        ← candle open-time (ms)
              "data": {
                "o": "1.25", "h": "1.27", "l": "1.23", "c": "1.26",
                "b": "2500000",           ← coin count (quoteVol)
                "q": "3150000"            ← USDT notional (== REST baseVol)
              }
            }

        We store USDT notional (``q``) as volume to match the parquet cache
        field (confirmed: ``q / b`` ≈ close price).

        A candle closes when the ``ts`` timestamp changes.  On close,
        ``self.latest_signals`` is refreshed and filter logic can be applied.

        Note
        ----
        Trend-capture entry logic is intentionally **not** implemented here
        yet.  This handler is the scaffolding stub — filter gates will be
        added once backtest results confirm the winning configuration.
        """
        try:
            kline_data = data.get("data", {})
            ts_ms = int(data.get("ts", 0))
            if not ts_ms or not kline_data:
                return

            candle_closed = self.candle_buffer.update(
                o=float(kline_data.get("o", 0)),
                h=float(kline_data.get("h", 0)),
                l=float(kline_data.get("l", 0)),
                c=float(kline_data.get("c", 0)),
                volume=float(kline_data.get("q", 0)),   # USDT notional
                ts_ms=ts_ms,
            )

            if candle_closed:
                self.latest_signals = self.candle_buffer.signals()
                if self.latest_signals:
                    regime_status = "–"
                    if self.latest_signals.regime_ema > 0:
                        regime_status = (
                            "bull" if self.latest_signals.close >= self.latest_signals.regime_ema
                            else "bear"
                        )
                    logger.debug(
                        "Candle closed | close=%.4f  adx=%.1f  rsi=%.1f  "
                        "bb_w=%.4f  vol_ratio=%.2f  ema_bias=%s  regime=%s(ema175=%.4f)",
                        self.latest_signals.close,
                        self.latest_signals.adx,
                        self.latest_signals.rsi,
                        self.latest_signals.bb_width,
                        self.latest_signals.vol_ratio,
                        "long" if self.latest_signals.ema_bias_long else
                        "short" if self.latest_signals.ema_bias_short else "flat",
                        regime_status,
                        self.latest_signals.regime_ema,
                    )
                    await self._evaluate_trend(self.latest_signals.close)

        except Exception as exc:
            logger.error("handle_kline_update error: %s", exc)

    async def handle_position_update(self, data: Dict[str, Any]) -> None:
        """
        Handle a position channel push.

        Bitunix position push fields (ch="position"):
          data.event  = OPEN / UPDATE / CLOSE
          data.side   = LONG / SHORT
          data.qty    = current position size (0 on CLOSE)
          data.symbol = trading pair
        """
        try:
            payload = data.get("data", {})
            if payload.get("symbol") != self.symbol:
                return
            side = payload.get("side", "").upper()
            qty = float(payload.get("qty", 0))
            event = payload.get("event", "")

            if side == "LONG":
                self.long_position = qty if event != "CLOSE" else 0.0
            elif side == "SHORT":
                self.short_position = abs(qty) if event != "CLOSE" else 0.0

            logger.info(
                "Position update [%s]: long=%.4f short=%.4f",
                event, self.long_position, self.short_position,
            )
            await self.check_and_notify_position_threshold("long", self.long_position)
            await self.check_and_notify_position_threshold("short", self.short_position)
            await self.check_and_notify_risk_reduction()
        except Exception as exc:
            logger.error("handle_position_update error: %s", exc)

    async def handle_order_update(self, data: Dict[str, Any]) -> None:
        """
        Handle an order channel push.

        Bitunix order push fields (ch="order"):
          data.event       = CREATE / UPDATE / CLOSE
          data.orderStatus = NEW / PART_FILLED / FILLED / CANCELED / …
          data.side        = BUY / SELL
          data.reduceOnly  = (not in WS docs, inferred from tradeSide CLOSE)
          data.qty         = original qty
          data.dealAmount  = filled qty

        On FILLED we refresh grid orders for the affected side.
        """
        try:
            payload = data.get("data", {})
            if payload.get("symbol", self.symbol) != self.symbol:
                return

            status = payload.get("orderStatus", "")
            side = payload.get("side", "").upper()
            event = payload.get("event", "")

            logger.info(
                "Order update: event=%s status=%s side=%s orderId=%s",
                event, status, side, payload.get("orderId"),
            )

            # On a fill, refresh the affected side's grid
            if status == "FILLED":
                # Determine whether this was a long or short leg via tradeSide
                # tradeSide=OPEN BUY → long entry filled → rebalance long
                # tradeSide=OPEN SELL → short entry filled → rebalance short
                # tradeSide=CLOSE → TP filled → rebalance the opposite (entry) side
                # We use a REST reconcile to be safe
                await self.reconcile_state()
                await self.place_long_orders(self.latest_price)
                await self.place_short_orders(self.latest_price)
        except Exception as exc:
            logger.error("handle_order_update error: %s", exc)

    async def handle_balance_update(self, data: Dict[str, Any]) -> None:
        """
        Handle a balance channel push.

        Bitunix balance push fields (ch="balance"):
          data.coin      = coin name (e.g. "USDT")
          data.available = free balance
          data.frozen    = locked by orders
          data.margin    = locked by positions
        """
        try:
            payload = data.get("data", {})
            coin = payload.get("coin", "UNKNOWN").upper()
            self.balance[coin] = {
                "available": float(payload.get("available", 0)),
                "frozen": float(payload.get("frozen", 0)),
                "margin": float(payload.get("margin", 0)),
            }
            logger.info(
                "Balance update: %s available=%.4f frozen=%.4f margin=%.4f",
                coin,
                self.balance[coin]["available"],
                self.balance[coin]["frozen"],
                self.balance[coin]["margin"],
            )
        except Exception as exc:
            logger.error("handle_balance_update error: %s", exc)

    # ------------------------------------------------------------------
    # Telegram helpers (mirrors gate_bot.py)
    # ------------------------------------------------------------------

    async def send_telegram_message(
        self, message: str, urgent: bool = False, silent: bool = False
    ) -> None:
        """Send a Telegram Bot API message. Swallows all exceptions."""
        if not ENABLE_NOTIFICATIONS or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            body = f"🤖 *{self.coin_name} grid bot* | {ts}\n\n{message}"
            if urgent:
                body = f"🚨 *Urgent* 🚨\n\n{body}"
            elif silent:
                body = f"🔇 *Summary* 🔇\n\n{body}"

            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": body,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
                "disable_notification": silent,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        logger.warning("Telegram send failed: HTTP %d", resp.status)
        except Exception as exc:
            logger.warning("send_telegram_message error: %s", exc)

    async def send_startup_notification(self) -> None:
        """Send a one-time startup summary (called once at bot start)."""
        if self.startup_notified:
            return
        btbw_str = (
            f"bull={self.bull_spacing:.2%} / bear={self.bear_spacing:.2%}"
            if self.bull_spacing != self.bear_spacing
            else f"{self.grid_spacing:.2%} (no BTBW)"
        )
        msg = (
            f"🚀 *Bot started*\n\n"
            f"• Coin: {COIN_NAME}\n"
            f"• Spacing (BTBW): {btbw_str}\n"
            f"• Regime EMA: {REGIME_EMA_PERIOD}-period  hysteresis={self.regime_hysteresis_pct:.2%}\n"
            f"• Initial qty: {INITIAL_QUANTITY} contracts\n"
            f"• Leverage: {LEVERAGE}x\n"
            f"• Position threshold: {POSITION_THRESHOLD:.2f}\n"
            f"• Position limit: {POSITION_LIMIT:.2f}"
        )
        await self.send_telegram_message(msg)
        self.startup_notified = True

    async def send_summary_notification(self) -> None:
        """Send a silent periodic summary if NOTIFICATION_INTERVAL has elapsed."""
        now = time.time()
        if now - self.last_summary_time < NOTIFICATION_INTERVAL:
            return

        usdt = self.balance.get("USDT", {})
        balance_str = (
            f"• USDT available: {usdt.get('available', 0):.2f}"
            if usdt
            else "• USDT balance: fetching…"
        )
        msg = (
            f"📊 *Status summary*\n\n"
            f"💰 {balance_str}\n\n"
            f"📈 *Positions*\n"
            f"• Long: {self.long_position}\n"
            f"• Short: {self.short_position}\n\n"
            f"📋 *Orders*\n"
            f"• Long entry: {self.buy_long_orders}\n"
            f"• Long TP: {self.sell_long_orders}\n"
            f"• Short entry: {self.sell_short_orders}\n"
            f"• Short TP: {self.buy_short_orders}\n\n"
            f"💹 Latest price: {self.latest_price:.8f}"
        )
        await self.send_telegram_message(msg, silent=True)
        self.last_summary_time = now

    async def send_error_notification(
        self, error_msg: str, error_type: str = "Runtime error"
    ) -> None:
        """Send an urgent Telegram error notification."""
        msg = (
            f"❌ *{error_type}*\n\n"
            f"{error_msg}\n\n"
            f"⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        await self.send_telegram_message(msg, urgent=True)

    async def get_balance_info(self) -> str:
        """Return a formatted USDT balance string (WS cache → REST fallback)."""
        usdt = self.balance.get("USDT", {})
        if usdt:
            return f"• USDT available: {usdt.get('available', 0):.2f}"
        try:
            data = await self.get_balance("USDT")
            return f"• USDT available: {data['free']:.2f} (REST)"
        except Exception:
            return "• USDT balance: unavailable"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Bootstrap the Bitunix grid bot from environment variables."""
    try:
        validate_config()
        bot = GridTradingBot(
            api_key=API_KEY,
            api_secret=API_SECRET,
            coin_name=COIN_NAME,
            grid_spacing=GRID_SPACING,
            initial_quantity=INITIAL_QUANTITY,
            leverage=LEVERAGE,
        )
        logger.info("Starting Bitunix grid bot for %s…", COIN_NAME)
        await bot.run()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Received stop signal – shutting down.")
    except Exception as exc:
        logger.exception("Unhandled error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

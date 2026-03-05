"""
src/single_bot/indicators.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Rolling candle buffer and technical indicator engine for the live Bitunix bot.

Mirrors the indicator logic from ``asBack/backtest_grid_bitunix.py`` so that
live signals are computed identically to the backtested versions.

Design
------
- ``CandleBuffer`` holds a fixed-length deque of *closed* candles plus one
  *in-progress* candle that is updated on every 500 ms kline WS push.
- When a new candle's timestamp arrives the previous candle is finalised and
  appended to the deque (capped at ``maxlen``).
- ``signals()`` converts the deque to numpy arrays and recomputes all
  indicators from scratch.  At 200 candles and 15-minute bars this is
  negligible CPU work (~sub-millisecond).

Seeding
-------
On startup, call ``await buf.seed(client, symbol)`` to prefill the buffer
from the REST kline endpoint so indicators are immediately valid.

Usage example
-------------
    from src.single_bot.indicators import CandleBuffer, Signals

    buf = CandleBuffer(maxlen=200, interval="15min")
    await buf.seed(exchange_client, "XRPUSDT")

    # Inside handle_kline_update:
    finished = buf.update(o, h, l, c, volume, ts_ms)
    if finished:                  # a candle just closed
        sig = buf.signals()
        print(sig.adx, sig.rsi, sig.bb_width)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Candle DTO
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    ts: int          # open-time in milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float    # baseVol (USDT notional) from Bitunix


# ---------------------------------------------------------------------------
# Signals snapshot
# ---------------------------------------------------------------------------

@dataclass
class Signals:
    """Computed indicator values for the most-recent *closed* candle."""

    # ── Trend / momentum ──────────────────────────────────────────────────
    adx: float = 0.0          # Wilder's ADX (0–100); > 25 = trending
    plus_di: float = 0.0      # +DI component (not used in filters yet)
    minus_di: float = 0.0     # -DI component (not used in filters yet)

    # ── Volatility / squeeze ──────────────────────────────────────────────
    bb_width: float = 0.0     # (upper-lower)/mid — low = squeeze
    atr: float = 0.0          # Wilder ATR in price units

    # ── Momentum ──────────────────────────────────────────────────────────
    rsi: float = 50.0         # Wilder RSI (0–100)

    # ── Trend direction ───────────────────────────────────────────────────
    ema_fast: float = 0.0     # fast EMA (default 9)
    ema_slow: float = 0.0     # slow EMA (default 21)
    ema_bias_long: bool = False   # fast > slow
    ema_bias_short: bool = False  # fast < slow

    # ── Regime filter (mirrors backtest regime_ema / halt_grid_longs) ─────
    regime_ema: float = 0.0   # slow EMA used for bull/bear regime detection (default 175)

    # ── Layer 1: ATR parabolic gate ───────────────────────────────────────
    atr_sma: float = 0.0     # SMA(ATR, 20) — baseline for parabolic detection
    atr_prev: float = 0.0    # ATR value N candles ago (acceleration filter)

    # ── Layer 2: HTF EMA alignment ────────────────────────────────────────
    htf_ema_fast: float = 0.0   # 1hr-equiv fast EMA (default 36 on 15m = 9hr)
    htf_ema_slow: float = 0.0   # 1hr-equiv slow EMA (default 84 on 15m = 21hr)

    # ── Layer 3: Regime vote mode ─────────────────────────────────────────
    regime_ema_87: float = 0.0  # secondary regime EMA (span 87)
    regime_ema_42: float = 0.0  # tertiary  regime EMA (span 42)

    # ── Volume ────────────────────────────────────────────────────────────
    volume: float = 0.0       # current candle volume (baseVol)
    vol_avg: float = 0.0      # rolling SMA of volume
    vol_ratio: float = 0.0    # volume / vol_avg

    # ── Market structure ──────────────────────────────────────────────────
    swing_high: float = 0.0   # highest high in preceding N candles
    swing_low: float = 0.0    # lowest low  in preceding N candles

    # ── Current price ─────────────────────────────────────────────────────
    close: float = 0.0


# ---------------------------------------------------------------------------
# CandleBuffer
# ---------------------------------------------------------------------------

class CandleBuffer:
    """
    Rolling fixed-length buffer of OHLCV candles for live indicator computation.

    Parameters
    ----------
    maxlen : int
        Maximum number of *closed* candles to retain (default 200).
        ADX/RSI need ~100 to warm up; 200 gives comfortable headroom.
    interval : str
        Bitunix kline interval string, e.g. ``"15min"`` or ``"1h"``.
        Used only when seeding from the REST API.
    adx_period : int
        Wilder ADX period (default 14).
    atr_period : int
        Wilder ATR period (default 14).
    rsi_period : int
        Wilder RSI period (default 14).
    bb_period : int
        Bollinger Band SMA period (default 20).
    bb_mult : float
        Bollinger Band standard-deviation multiplier (default 2.0).
    ema_fast : int
        Fast EMA span (default 9).
    ema_slow : int
        Slow EMA span (default 21).
    regime_ema_period : int
        Regime-filter EMA period (default 175).  Mirrors the ``ema=175``
        parameter in the winning backtest configs.  The live price is
        compared against this EMA each candle to decide whether to halt
        long grid legs (bear regime) or switch to ``bull_spacing`` (BTBW).
    vol_period : int
        Rolling volume SMA period (default 20).
    ms_lookback : int
        Market-structure swing-high/low lookback window (default 20).
    """

    def __init__(
        self,
        maxlen: int = 200,
        interval: str = "15min",
        adx_period: int = 14,
        atr_period: int = 14,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_mult: float = 2.0,
        ema_fast: int = 9,
        ema_slow: int = 21,
        regime_ema_period: int = 175,
        vol_period: int = 20,
        ms_lookback: int = 20,
        htf_ema_fast_period: int = 36,
        htf_ema_slow_period: int = 84,
        regime_ema_87_period: int = 87,
        regime_ema_42_period: int = 42,
        atr_sma_period: int = 20,
        atr_accel_lookback: int = 10,
    ) -> None:
        self.maxlen = maxlen
        self.interval = interval
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_mult = bb_mult
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.regime_ema_period = regime_ema_period
        self.vol_period = vol_period
        self.ms_lookback = ms_lookback
        self.htf_ema_fast_period = htf_ema_fast_period
        self.htf_ema_slow_period = htf_ema_slow_period
        self.regime_ema_87_period = regime_ema_87_period
        self.regime_ema_42_period = regime_ema_42_period
        self.atr_sma_period = atr_sma_period
        self.atr_accel_lookback = atr_accel_lookback

        self._closed: Deque[Candle] = deque(maxlen=maxlen)
        self._live: Optional[Candle] = None   # current in-progress candle

    # ------------------------------------------------------------------
    # Seeding from REST
    # ------------------------------------------------------------------

    async def seed(self, client: object, symbol: str) -> None:
        """
        Prefill the buffer with historical candles from the REST API.

        ``client`` is a ``BitunixExchange`` instance.  Fetches ``maxlen``
        candles (capped at 200 per request — the Bitunix API hard limit).
        Makes multiple requests if ``maxlen > 200``.
        """
        limit = min(self.maxlen, 200)
        candles_raw = await client.get_klines(symbol, self.interval, limit=limit)
        for c in candles_raw:
            self._closed.append(
                Candle(
                    ts=int(c["open_time"]),
                    open=float(c["open"]),
                    high=float(c["high"]),
                    low=float(c["low"]),
                    close=float(c["close"]),
                    volume=float(c["volume"]),
                )
            )

    # ------------------------------------------------------------------
    # Live updates (called on every WS kline push ~500 ms)
    # ------------------------------------------------------------------

    def update(
        self,
        o: float,
        h: float,
        l: float,
        c: float,
        volume: float,
        ts_ms: int,
    ) -> bool:
        """
        Ingest a kline WS update.

        Bitunix pushes the *current in-progress* candle every 500 ms.
        When the timestamp changes, the previous candle is finalised.

        Parameters
        ----------
        o, h, l, c : float
            OHLC from the WS payload (fields ``o``, ``h``, ``l``, ``c``).
        volume : float
            ``q`` field (USDT notional == baseVol equivalent over WS).
        ts_ms : int
            ``ts`` field from the WS message (candle open-time in ms).

        Returns
        -------
        bool
            ``True`` if a candle just closed (new timestamp arrived) —
            a good time to recompute signals and evaluate filter logic.
        """
        candle_closed = False

        if self._live is None:
            # First update ever
            self._live = Candle(ts=ts_ms, open=o, high=h, low=l, close=c, volume=volume)
        elif ts_ms != self._live.ts:
            # New candle started — commit the previous one
            self._closed.append(self._live)
            self._live = Candle(ts=ts_ms, open=o, high=h, low=l, close=c, volume=volume)
            candle_closed = True
        else:
            # Same candle, update running OHLCV
            self._live.high = max(self._live.high, h)
            self._live.low = min(self._live.low, l)
            self._live.close = c
            self._live.volume = volume

        return candle_closed

    # ------------------------------------------------------------------
    # Indicator computation
    # ------------------------------------------------------------------

    def signals(self) -> Optional[Signals]:
        """
        Compute and return a ``Signals`` snapshot from the closed candles.

        Returns ``None`` if fewer than ``max(adx_period, rsi_period, bb_period)``
        candles are buffered (warm-up period incomplete).
        """
        candles = list(self._closed)
        n = len(candles)
        min_required = max(self.adx_period, self.rsi_period, self.bb_period, self.ema_slow, self.regime_ema_period) + 5
        if n < min_required:
            return None

        # Unpack arrays
        highs   = np.array([c.high   for c in candles], dtype=float)
        lows    = np.array([c.low    for c in candles], dtype=float)
        closes  = np.array([c.close  for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)

        adx_val, plus_di, minus_di = self._adx(highs, lows, closes, self.adx_period)
        atr_val, atr_prev_val      = self._atr(highs, lows, closes, self.atr_period,
                                                prev_lookback=self.atr_accel_lookback)
        rsi_val                    = self._rsi(closes, self.rsi_period)
        bb_w                       = self._bb_width(closes, self.bb_period, self.bb_mult)
        ema_f                      = self._ema(closes, self.ema_fast)
        ema_s                      = self._ema(closes, self.ema_slow)
        regime_ema_val             = self._ema(closes, self.regime_ema_period)
        htf_ema_f                  = self._ema(closes, self.htf_ema_fast_period)
        htf_ema_s                  = self._ema(closes, self.htf_ema_slow_period)
        regime_ema_87_val          = self._ema(closes, self.regime_ema_87_period)
        regime_ema_42_val          = self._ema(closes, self.regime_ema_42_period)
        atr_sma_val                = self._atr_sma(highs, lows, closes, self.atr_period, self.atr_sma_period)
        vol_avg                    = self._vol_sma(volumes, self.vol_period)
        swing_hi                   = self._swing_high(highs, self.ms_lookback)
        swing_lo                   = self._swing_low(lows, self.ms_lookback)

        last_vol = volumes[-1]
        return Signals(
            adx=adx_val,
            plus_di=plus_di,
            minus_di=minus_di,
            bb_width=bb_w,
            atr=atr_val,
            atr_prev=atr_prev_val,
            rsi=rsi_val,
            ema_fast=ema_f,
            ema_slow=ema_s,
            ema_bias_long=ema_f > ema_s,
            ema_bias_short=ema_f < ema_s,
            regime_ema=regime_ema_val,
            atr_sma=atr_sma_val,
            htf_ema_fast=htf_ema_f,
            htf_ema_slow=htf_ema_s,
            regime_ema_87=regime_ema_87_val,
            regime_ema_42=regime_ema_42_val,
            volume=last_vol,
            vol_avg=vol_avg,
            vol_ratio=last_vol / vol_avg if vol_avg > 0 else 0.0,
            swing_high=swing_hi,
            swing_low=swing_lo,
            close=closes[-1],
        )

    # ------------------------------------------------------------------
    # Static indicator implementations (mirror backtest_grid_bitunix.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _wilder_ewm(arr: np.ndarray, period: int) -> np.ndarray:
        """Wilder smoothing: equivalent to EWM with alpha=1/period, adjust=False."""
        alpha = 1.0 / period
        out = np.empty_like(arr, dtype=float)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = out[i - 1] * (1.0 - alpha) + arr[i] * alpha
        return out

    @staticmethod
    def _ema_arr(arr: np.ndarray, span: int) -> np.ndarray:
        """Standard EMA: alpha = 2/(span+1), adjust=False."""
        alpha = 2.0 / (span + 1)
        out = np.empty_like(arr, dtype=float)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = out[i - 1] * (1.0 - alpha) + arr[i] * alpha
        return out

    @classmethod
    def _adx(
        cls, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int
    ) -> Tuple[float, float, float]:
        """
        Return (adx, +DI, -DI) for the most-recent candle.
        Mirrors ``_compute_adx`` in the backtest (Wilder EWM, alpha=1/period).
        """
        n = len(closes)
        prev_closes = np.empty(n)
        prev_closes[0] = closes[0]
        prev_closes[1:] = closes[:-1]

        tr = np.maximum.reduce(
            [highs - lows, np.abs(highs - prev_closes), np.abs(lows - prev_closes)]
        )

        up_move   = np.diff(highs, prepend=highs[0])
        down_move = -np.diff(lows, prepend=lows[0])
        plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_s    = cls._wilder_ewm(tr,       period)
        pdi_s   = cls._wilder_ewm(plus_dm,  period)
        mdi_s   = cls._wilder_ewm(minus_dm, period)

        eps = 1e-10
        plus_di  = 100.0 * pdi_s / (tr_s + eps)
        minus_di = 100.0 * mdi_s / (tr_s + eps)
        di_sum   = plus_di + minus_di
        dx       = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / di_sum, 0.0)
        adx      = cls._wilder_ewm(dx, period)

        return float(adx[-1]), float(plus_di[-1]), float(minus_di[-1])

    @classmethod
    def _atr(
        cls, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int,
        prev_lookback: int = 0,
    ) -> "float | tuple[float, float]":
        """Return ATR (price units) for the most-recent candle (Wilder smoothing).

        If *prev_lookback* > 0, returns ``(atr_current, atr_prev)`` where
        ``atr_prev`` is the ATR value *prev_lookback* candles before current.
        """
        n = len(closes)
        prev_closes = np.empty(n)
        prev_closes[0] = closes[0]
        prev_closes[1:] = closes[:-1]

        tr = np.maximum.reduce(
            [highs - lows, np.abs(highs - prev_closes), np.abs(lows - prev_closes)]
        )
        atr_arr = cls._wilder_ewm(tr, period)
        current = float(atr_arr[-1])
        if prev_lookback <= 0:
            return current
        idx = max(0, len(atr_arr) - 1 - prev_lookback)
        return current, float(atr_arr[idx])

    @classmethod
    def _atr_sma(
        cls,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        atr_period: int,
        sma_period: int,
    ) -> float:
        """Return SMA of the ATR series over the last *sma_period* candles."""
        n = len(closes)
        prev_closes = np.empty(n)
        prev_closes[0] = closes[0]
        prev_closes[1:] = closes[:-1]
        tr = np.maximum.reduce(
            [highs - lows, np.abs(highs - prev_closes), np.abs(lows - prev_closes)]
        )
        atr_arr = cls._wilder_ewm(tr, atr_period)
        w = min(sma_period, len(atr_arr))
        return float(np.mean(atr_arr[-w:]))

    @classmethod
    def _rsi(cls, closes: np.ndarray, period: int) -> float:
        """Return Wilder RSI (0–100) for the most-recent candle."""
        delta = np.diff(closes, prepend=closes[0])
        gain  = np.clip(delta,  0, None)
        loss  = np.clip(-delta, 0, None)
        avg_gain = cls._wilder_ewm(gain, period)
        avg_loss = cls._wilder_ewm(loss, period)
        last_loss = avg_loss[-1]
        if last_loss < 1e-10:
            return 100.0
        rs = avg_gain[-1] / last_loss
        return float(100.0 - 100.0 / (1.0 + rs))

    @staticmethod
    def _bb_width(closes: np.ndarray, period: int, mult: float) -> float:
        """Return (upper - lower) / mid for the most-recent candle."""
        if len(closes) < period:
            return 0.0
        window = closes[-period:]
        mid = float(np.mean(window))
        if mid < 1e-10:
            return 0.0
        std = float(np.std(window, ddof=0))
        return (2.0 * mult * std) / mid

    @classmethod
    def _ema(cls, closes: np.ndarray, span: int) -> float:
        """Return EMA (span) for the most-recent candle."""
        return float(cls._ema_arr(closes, span)[-1])

    @staticmethod
    def _vol_sma(volumes: np.ndarray, period: int) -> float:
        """Return rolling SMA of volume over the past `period` candles."""
        n = len(volumes)
        w = min(period, n)
        return float(np.mean(volumes[-w:]))

    @staticmethod
    def _swing_high(highs: np.ndarray, lookback: int) -> float:
        """Max high over the preceding `lookback` candles (excludes the most recent)."""
        n = len(highs)
        if n < 2:
            return float(highs[-1]) if n else 0.0
        window = highs[max(0, n - 1 - lookback): n - 1]
        return float(np.max(window)) if len(window) else float(highs[-1])

    @staticmethod
    def _swing_low(lows: np.ndarray, lookback: int) -> float:
        """Min low over the preceding `lookback` candles (excludes the most recent)."""
        n = len(lows)
        if n < 2:
            return float(lows[-1]) if n else 0.0
        window = lows[max(0, n - 1 - lookback): n - 1]
        return float(np.min(window)) if len(window) else float(lows[-1])

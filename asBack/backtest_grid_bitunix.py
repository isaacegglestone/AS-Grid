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

        self.orders: Dict[str, List] = {"long": [], "short": []}
        self.last_refresh_time = None
        self.last_long_price: Optional[float] = None
        self.last_short_price: Optional[float] = None

        self._init_orders(self.df["close"].iloc[0])

    # ------------------------------------------------------------------
    # Order placement helpers
    # ------------------------------------------------------------------

    def _init_orders(self, price: float) -> None:
        if self.direction in ["long", "both"]:
            self._place_long_orders(price)
        if self.direction in ["short", "both"]:
            self._place_short_orders(price)

    def _place_long_orders(self, current_price: float) -> None:
        """Long grid: take-profit above, re-entry below."""
        self.orders["long"] = [
            (current_price * (1 - self.long_settings["down_spacing"]), "BUY"),
            (current_price * (1 + self.long_settings["up_spacing"]), "SELL"),
        ]
        self.last_long_price = current_price

    def _place_short_orders(self, current_price: float) -> None:
        """Short grid: re-entry above, take-profit below."""
        self.orders["short"] = [
            (current_price * (1 + self.short_settings["up_spacing"]), "SELL_SHORT"),
            (current_price * (1 - self.short_settings["down_spacing"]), "COVER_SHORT"),
        ]
        self.last_short_price = current_price

    def _update_orders_after_trade(self, side: str, fill_price: float) -> None:
        if side == "long":
            self._place_long_orders(fill_price)
        elif side == "short":
            self._place_short_orders(fill_price)

    def _refresh_orders_if_needed(self, price: float, current_time: datetime) -> None:
        """Periodically re-anchor grid to current price."""
        if self.last_refresh_time is None or (
            current_time - self.last_refresh_time
            >= timedelta(minutes=self.config["grid_refresh_interval"])
        ):
            if self.direction in ["long", "both"]:
                self._place_long_orders(price)
            if self.direction in ["short", "both"]:
                self._place_short_orders(price)
            self.last_refresh_time = current_time

    def _calculate_unrealized_pnl(self, price: float) -> float:
        long_pnl = sum((price - ep) * qty for ep, qty, _ in self.long_positions)
        short_pnl = sum((ep - price) * qty for ep, qty, _ in self.short_positions)
        return long_pnl + short_pnl

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        for _, row in self.df.iterrows():
            if (
                len(self.short_positions) + len(self.long_positions)
                >= self.config["max_positions"]
            ):
                print("⚠️ Max positions reached")
                break

            price: float = row["close"]
            timestamp = row["open_time"]
            effective_order_value = self.config["order_value"] * self.leverage

            self._refresh_orders_if_needed(price, timestamp)

            used_margin = sum(pos[2] for pos in self.long_positions + self.short_positions)
            available_margin = self.balance - used_margin

            # LONG SIDE
            if self.direction in ["long", "both"]:
                for order_price, action in self.orders["long"]:
                    if action == "BUY" and price <= order_price:
                        qty = effective_order_value / price
                        margin_required = qty * price / self.leverage
                        fee_cost = qty * price * (self.fee / 2)
                        if (margin_required + fee_cost) > available_margin:
                            continue
                        self.balance -= margin_required + fee_cost
                        self.long_positions.append((price, qty, margin_required))
                        unrealized_pnl = self._calculate_unrealized_pnl(price)
                        self.trade_history.append((
                            timestamp, "BUY", price, qty, "LONG",
                            0.0, fee_cost, 0.0, unrealized_pnl,
                            self.balance + unrealized_pnl,
                        ))
                        self._update_orders_after_trade("long", price)
                        break

                    elif action == "SELL" and self.long_positions and price >= order_price:
                        entry_price, qty, margin_required = self.long_positions.pop(0)
                        fee_cost = qty * price * (self.fee / 2)
                        gross_pnl = (price - entry_price) * qty
                        net_pnl = gross_pnl - fee_cost
                        self.balance += margin_required + net_pnl
                        unrealized_pnl = self._calculate_unrealized_pnl(price)
                        self.trade_history.append((
                            timestamp, "SELL", price, qty, "LONG",
                            net_pnl, fee_cost, gross_pnl, unrealized_pnl,
                            self.balance + unrealized_pnl,
                        ))
                        self._update_orders_after_trade("long", price)
                        break

            # SHORT SIDE
            if self.direction in ["short", "both"]:
                for order_price, action in self.orders["short"]:
                    if action == "SELL_SHORT" and price >= order_price:
                        qty = effective_order_value / price
                        margin_required = qty * price / self.leverage
                        fee_cost = qty * price * (self.fee / 2)
                        if (margin_required + fee_cost) > available_margin:
                            continue
                        self.balance -= margin_required + fee_cost
                        self.short_positions.append((price, qty, margin_required))
                        unrealized_pnl = self._calculate_unrealized_pnl(price)
                        self.trade_history.append((
                            timestamp, "SELL_SHORT", price, qty, "SHORT",
                            0.0, fee_cost, 0.0, unrealized_pnl,
                            self.balance + unrealized_pnl,
                        ))
                        self._update_orders_after_trade("short", price)
                        break

                    elif action == "COVER_SHORT" and self.short_positions and price <= order_price:
                        entry_price, qty, margin_required = self.short_positions.pop(0)
                        fee_cost = qty * price * (self.fee / 2)
                        gross_pnl = (entry_price - price) * qty
                        net_pnl = gross_pnl - fee_cost
                        self.balance += margin_required + net_pnl
                        unrealized_pnl = self._calculate_unrealized_pnl(price)
                        self.trade_history.append((
                            timestamp, "COVER_SHORT", price, qty, "SHORT",
                            net_pnl, fee_cost, gross_pnl, unrealized_pnl,
                            self.balance + unrealized_pnl,
                        ))
                        self._update_orders_after_trade("short", price)
                        break

            # Equity tracking
            long_pnl = sum((price - ep) * qty for ep, qty, _ in self.long_positions)
            short_pnl = sum((ep - price) * qty for ep, qty, _ in self.short_positions)
            unrealized_pnl = long_pnl + short_pnl
            equity = self.balance + unrealized_pnl
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
        long_pnl = sum((final_price - ep) * qty for ep, qty, _ in self.long_positions)
        short_pnl = sum((ep - final_price) * qty for ep, qty, _ in self.short_positions)
        unrealized_pnl = long_pnl + short_pnl
        realized_pnl = sum(t[5] for t in self.trade_history if t[5] != 0.0)
        final_equity = self.balance + unrealized_pnl
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

    print(f"Fetching {symbol} {interval} klines from {start_dt.date()} to {end_dt.date()} …")

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
    print(f"  → {len(df):,} candles loaded")
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
        print(
            f"  → return: {result['return_pct'] * 100:.2f}%  "
            f"trades: {result['trades']}  "
            f"max_dd: {result['max_drawdown'] * 100:.2f}%"
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
    "order_value": 10,           # USD per grid order
    "max_drawdown": 0.9,
    "max_positions": 20,
    "fee_pct": 0.0006,           # Bitunix taker fee 0.06 %
    "leverage": 1,
    # "both" runs symmetric long+short grids — market-neutral, profits from
    # oscillation in either direction.  "long" only profits when price rises.
    "direction": "both",         # "long" | "short" | "both"
    "grid_refresh_interval": 10, # minutes between grid re-anchoring

    # ── Parameter sets to grid-search
    # Grid spacing = distance between order levels as a fraction of price.
    # Tighter spacing → more frequent fills but smaller profit per trade.
    # Wider spacing  → fewer fills but larger profit per trade.
    "param_sets": [
        {
            "name": "tight_0.2pct",
            "long_settings":  {"up_spacing": 0.002, "down_spacing": 0.002},
            "short_settings": {"up_spacing": 0.002, "down_spacing": 0.002},
        },
        {
            "name": "medium_0.3pct",
            "long_settings":  {"up_spacing": 0.003, "down_spacing": 0.003},
            "short_settings": {"up_spacing": 0.003, "down_spacing": 0.003},
        },
        {
            "name": "medium_0.5pct",
            "long_settings":  {"up_spacing": 0.005, "down_spacing": 0.005},
            "short_settings": {"up_spacing": 0.005, "down_spacing": 0.005},
        },
        {
            "name": "wide_0.8pct",
            "long_settings":  {"up_spacing": 0.008, "down_spacing": 0.008},
            "short_settings": {"up_spacing": 0.008, "down_spacing": 0.008},
        },
    ],
}

# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    grid_search_backtest(CONFIG)

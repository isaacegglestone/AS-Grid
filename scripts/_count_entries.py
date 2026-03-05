"""Count actual trend captures and PM condition firings in the 6m window."""
import asyncio, sys, os, copy
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from asBack.backtest_grid_bitunix import (
    fetch_klines_as_df, GridOrderBacktester, XRP_PM_CONFIG
)

async def main():
    cfg = copy.deepcopy(XRP_PM_CONFIG)
    cfg["start_date"] = datetime(2025, 8, 1)
    cfg["end_date"]   = datetime(2026, 2, 1)

    df = await fetch_klines_as_df(
        symbol=cfg["symbol"],
        interval=cfg["interval"],      # "1min" — must match CI exactly
        start_dt=cfg["start_date"],
        end_dt=cfg["end_date"],
    )
    print(f"Loaded {len(df)} candles  ({cfg['interval']} data)")

    # Build temp_config exactly as grid_search_backtest_async does
    params = cfg["param_sets"][0]  # pm_baseline
    temp_config = {k: v for k, v in cfg.items() if k not in ("param_sets",)}
    temp_config.update({
        "long_settings": params["long_settings"],
        "short_settings": params["short_settings"],
        "use_sl": params.get("use_sl", True),
        **({k: params[k] for k in (
            "trend_detection", "trend_capture",
            "trend_force_close_grid", "trend_confirm_candles",
            "trend_capture_size_pct", "trend_trailing_stop_pct",
            "trend_capture_velocity_pct", "trend_velocity_pct",
            "trend_lookback_candles",
            "adx_filter", "adx_period", "adx_min_trend", "adx_grid_pause",
            "atr_trail", "atr_period", "atr_trail_multiplier",
            "bb_squeeze_gate", "bb_squeeze_boost", "bb_period", "bb_mult",
            "bb_squeeze_threshold", "bb_squeeze_boost_mult",
            "rsi_filter", "rsi_period", "rsi_overbought", "rsi_oversold", "rsi_momentum",
            "vol_filter", "vol_period", "vol_multiplier",
            "ms_filter", "ms_lookback",
            "trend_reentry_fast",
            "adx_wide_trail_threshold", "adx_wide_trail_pct",
            "rsi_tight_trail", "rsi_tight_trail_ob", "rsi_tight_trail_os", "rsi_tight_trail_pct",
            "vol_reentry_scale", "vol_reentry_high_mult", "vol_reentry_low_pct",
            "crash_cb", "crash_cb_drop_pct", "crash_cb_lookback_candles", "crash_cb_halt_candles",
            "dd_halt", "max_drawdown", "dd_halt_candles",
            "grid_notional_cap_pct",
        ) if k in params}),
    })

    bt = GridOrderBacktester(df.copy(), None, temp_config)
    result = bt.run()

    from collections import Counter
    trade_types = Counter(t[4] for t in bt.trade_history)
    trend_trades = sum(v for k, v in trade_types.items() if "TREND" in k)
    grid_trades  = sum(v for k, v in trade_types.items() if "TREND" not in k)

    print(f"\n{'='*55}")
    print(f"Total trades:  {len(bt.trade_history)}")
    print(f"Grid trades:   {grid_trades}")
    print(f"Trend trades:  {trend_trades}")
    print(f"Return:        {result['return_pct']*100:.2f}%")
    print(f"\nTrade type breakdown:")
    for k, v in sorted(trade_types.items()):
        print(f"  {k}: {v}")
    print(f"\nVerdict: PM mechanics only affect trend positions.")
    if trend_trades <= 20:
        print(f"  ⚠  Only {trend_trades} trend trades — very few windows for PM to act.")
        print(f"     Identical PM variant results are expected, not a bug.")

asyncio.run(main())

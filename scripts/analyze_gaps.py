#!/usr/bin/env python3
"""
Gap Analysis — identify every day where the bot earned $0 or lost money.

Usage:
    python scripts/analyze_gaps.py [trades_csv] [equity_csv]

Defaults:
    trades_csv  = bitunix_best_grid_trades.csv
    equity_csv  = bitunix_best_equity_curve.csv

Output:
    Console table summary
    gap_analysis_daily.csv         — full daily P&L breakdown
    gap_analysis_report.html       — visual report with heatmap + charts
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Colours for console output ──────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# =========================================================================
# Data loading
# =========================================================================

def load_trades(path: str) -> pd.DataFrame:
    """Load the trades CSV exported by the backtester."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], format="ISO8601")
    df["date"] = df["time"].dt.date
    return df


def load_equity_curve(path: str) -> pd.DataFrame:
    """Load the equity curve CSV exported by the backtester."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], format="ISO8601")
    return df


# =========================================================================
# Daily P&L computation
# =========================================================================

def compute_daily_pnl(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    initial_balance: float = 1000.0,
) -> pd.DataFrame:
    """Build a day-by-day P&L table from trades and equity curve.

    Returns a DataFrame with columns:
        date, equity_open, equity_close, daily_pnl, daily_pct,
        cum_pnl, cum_pct, n_trades, realized_pnl, is_gap
    """
    # Equity end-of-day
    eq = equity_df.set_index("time")["equity"].resample("D").last().dropna()

    # Trades per day
    trades_grouped = trades_df.groupby("date")
    trade_counts   = trades_grouped.size().rename("n_trades")
    realized_daily = trades_grouped["pnl"].sum().rename("realized_pnl")

    # Build daily table
    rows = []
    prev_eq = initial_balance
    for dt, equity_close in eq.items():
        day = dt.date() if hasattr(dt, "date") else dt
        daily_pnl = equity_close - prev_eq
        daily_pct = (daily_pnl / prev_eq * 100) if prev_eq else 0.0
        cum_pnl   = equity_close - initial_balance
        cum_pct   = (cum_pnl / initial_balance * 100)
        n_trades  = int(trade_counts.get(day, 0))
        real_pnl  = float(realized_daily.get(day, 0.0))

        # GAP: any day with $0 or negative earnings
        is_gap = daily_pnl <= 0

        rows.append({
            "date":          day,
            "equity_open":   round(prev_eq, 2),
            "equity_close":  round(equity_close, 2),
            "daily_pnl":     round(daily_pnl, 2),
            "daily_pct":     round(daily_pct, 4),
            "cum_pnl":       round(cum_pnl, 2),
            "cum_pct":       round(cum_pct, 2),
            "n_trades":      n_trades,
            "realized_pnl":  round(real_pnl, 2),
            "is_gap":        is_gap,
        })
        prev_eq = equity_close

    return pd.DataFrame(rows)


# =========================================================================
# Gap identification & metrics
# =========================================================================

def identify_gap_streaks(daily: pd.DataFrame) -> pd.DataFrame:
    """Find consecutive runs of gap days (daily_pnl <= 0).

    Returns a DataFrame with columns:
        start_date, end_date, days, total_loss, worst_day, worst_day_pnl
    """
    gaps = daily[daily["is_gap"]].copy()
    if gaps.empty:
        return pd.DataFrame(columns=[
            "start_date", "end_date", "days", "total_loss",
            "worst_day", "worst_day_pnl",
        ])

    # Assign streak IDs by detecting breaks in consecutive gap days
    gaps = gaps.reset_index(drop=True)
    date_series = pd.to_datetime(gaps["date"])
    breaks = (date_series.diff().dt.days > 1).cumsum()
    gaps["streak_id"] = breaks

    streaks = []
    for _, grp in gaps.groupby("streak_id"):
        worst_idx = grp["daily_pnl"].idxmin()
        streaks.append({
            "start_date":   grp["date"].iloc[0],
            "end_date":     grp["date"].iloc[-1],
            "days":         len(grp),
            "total_loss":   round(grp["daily_pnl"].sum(), 2),
            "worst_day":    grp.loc[worst_idx, "date"],
            "worst_day_pnl": round(grp.loc[worst_idx, "daily_pnl"], 2),
        })

    return pd.DataFrame(streaks).sort_values("total_loss").reset_index(drop=True)


def compute_drawdown_periods(daily: pd.DataFrame) -> pd.DataFrame:
    """Find peak-to-trough drawdown periods.

    Returns a DataFrame with:
        peak_date, trough_date, recovery_date, peak_eq, trough_eq,
        drawdown_pct, days_to_trough, days_to_recovery
    """
    eq = daily.set_index("date")["equity_close"]
    running_max = eq.cummax()
    drawdown = (eq - running_max) / running_max

    # Identify drawdown periods (contiguous blocks where drawdown < 0)
    in_dd = drawdown < 0
    periods = []
    start = None
    peak_date = None
    for dt, is_dd in in_dd.items():
        if is_dd and start is None:
            start = dt
            # Peak is the last date where equity == running_max before this DD
            mask = eq.loc[:dt].index
            peak_candidates = mask[eq.loc[mask] == running_max.loc[mask]]
            peak_date = peak_candidates[-1] if len(peak_candidates) > 0 else dt
        elif not is_dd and start is not None:
            trough_idx = drawdown.loc[start:dt].idxmin()
            periods.append({
                "peak_date":         peak_date,
                "trough_date":       trough_idx,
                "recovery_date":     dt,
                "peak_eq":           round(float(running_max.loc[trough_idx]), 2),
                "trough_eq":         round(float(eq.loc[trough_idx]), 2),
                "drawdown_pct":      round(float(drawdown.loc[trough_idx]) * 100, 2),
                "days_to_trough":    (trough_idx - peak_date).days if hasattr(trough_idx, '__sub__') else 0,
                "days_to_recovery":  (dt - trough_idx).days if hasattr(dt, '__sub__') else 0,
            })
            start = None
            peak_date = None

    # Handle ongoing drawdown at end of data
    if start is not None:
        trough_idx = drawdown.loc[start:].idxmin()
        periods.append({
            "peak_date":         peak_date,
            "trough_date":       trough_idx,
            "recovery_date":     None,
            "peak_eq":           round(float(running_max.loc[trough_idx]), 2),
            "trough_eq":         round(float(eq.loc[trough_idx]), 2),
            "drawdown_pct":      round(float(drawdown.loc[trough_idx]) * 100, 2),
            "days_to_trough":    (trough_idx - peak_date).days if hasattr(trough_idx, '__sub__') else 0,
            "days_to_recovery":  None,
        })

    df = pd.DataFrame(periods)
    if not df.empty:
        df = df.sort_values("drawdown_pct").reset_index(drop=True)
    return df


def compute_monthly_heatmap(daily: pd.DataFrame) -> pd.DataFrame:
    """Pivot daily P&L into a year × month heatmap.

    Returns a DataFrame with years as index, months 1-12 as columns,
    values = monthly P&L %.
    """
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"]  = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    monthly = daily.groupby(["year", "month"])["daily_pct"].sum()
    return monthly.unstack(level="month").fillna(0).round(2)


# =========================================================================
# Console output
# =========================================================================

def print_summary(daily: pd.DataFrame, streaks: pd.DataFrame, drawdowns: pd.DataFrame) -> None:
    """Print the gap analysis summary to console."""
    total_days = len(daily)
    gap_days   = daily["is_gap"].sum()
    profit_days = total_days - gap_days
    loss_days   = len(daily[daily["daily_pnl"] < 0])
    zero_days   = len(daily[daily["daily_pnl"] == 0])
    idle_days   = len(daily[daily["n_trades"] == 0])

    print(f"\n{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}  GAP ANALYSIS REPORT{RESET}")
    print(f"{BOLD}{'=' * 80}{RESET}")

    # ── Overview ────────────────────────────────────────────────
    print(f"\n{BOLD}  Overview{RESET}")
    print(f"  {'─' * 50}")
    print(f"  Total days analysed:     {total_days:>6,}")
    print(f"  {GREEN}Profitable days (> $0):  {profit_days:>6,}  ({profit_days/total_days*100:.1f}%){RESET}")
    print(f"  {RED}Loss days (< $0):        {loss_days:>6,}  ({loss_days/total_days*100:.1f}%){RESET}")
    print(f"  {YELLOW}Zero-change days ($0):   {zero_days:>6,}  ({zero_days/total_days*100:.1f}%){RESET}")
    print(f"  Idle days (0 trades):    {idle_days:>6,}  ({idle_days/total_days*100:.1f}%)")
    print(f"  {RED}Total gap days (≤ $0):   {gap_days:>6,}  ({gap_days/total_days*100:.1f}%){RESET}")

    # ── Best & worst days ───────────────────────────────────────
    best_idx  = daily["daily_pnl"].idxmax()
    worst_idx = daily["daily_pnl"].idxmin()
    print(f"\n  {GREEN}Best day:   {daily.loc[best_idx, 'date']}  ${daily.loc[best_idx, 'daily_pnl']:>+10.2f}  ({daily.loc[best_idx, 'daily_pct']:>+.2f}%){RESET}")
    print(f"  {RED}Worst day:  {daily.loc[worst_idx, 'date']}  ${daily.loc[worst_idx, 'daily_pnl']:>+10.2f}  ({daily.loc[worst_idx, 'daily_pct']:>+.2f}%){RESET}")

    # ── All gap days table ──────────────────────────────────────
    gaps = daily[daily["is_gap"]].copy()
    if not gaps.empty:
        print(f"\n{BOLD}  All Gap Days (daily P&L ≤ $0) — {len(gaps)} days{RESET}")
        print(f"  {'─' * 76}")
        print(f"  {'Date':>12s}  {'Equity $':>11s}  {'Daily P&L':>10s}  {'Daily %':>8s}  {'Cum %':>8s}  {'Trades':>6s}  {'Real P&L':>9s}")
        print(f"  {'─' * 76}")
        for _, row in gaps.iterrows():
            colour = RED if row["daily_pnl"] < 0 else YELLOW
            print(
                f"  {colour}{str(row['date']):>12s}  "
                f"${row['equity_close']:>10,.2f}  "
                f"${row['daily_pnl']:>+9.2f}  "
                f"{row['daily_pct']:>+7.2f}%  "
                f"{row['cum_pct']:>+7.1f}%  "
                f"{row['n_trades']:>5d}   "
                f"${row['realized_pnl']:>+8.2f}{RESET}"
            )
        print(f"  {'─' * 76}")

    # ── Worst gap streaks ───────────────────────────────────────
    if not streaks.empty:
        top_streaks = streaks.head(10)
        print(f"\n{BOLD}  Top 10 Worst Gap Streaks (consecutive ≤$0 days){RESET}")
        print(f"  {'─' * 72}")
        print(f"  {'Start':>12s}  {'End':>12s}  {'Days':>5s}  {'Total Loss':>11s}  {'Worst Day':>12s}  {'Worst P&L':>10s}")
        print(f"  {'─' * 72}")
        for _, s in top_streaks.iterrows():
            print(
                f"  {RED}{str(s['start_date']):>12s}  "
                f"{str(s['end_date']):>12s}  "
                f"{s['days']:>5d}  "
                f"${s['total_loss']:>+10.2f}  "
                f"{str(s['worst_day']):>12s}  "
                f"${s['worst_day_pnl']:>+9.2f}{RESET}"
            )
        print(f"  {'─' * 72}")

    # ── Worst drawdowns ─────────────────────────────────────────
    if not drawdowns.empty:
        top_dd = drawdowns.head(10)
        print(f"\n{BOLD}  Top 10 Drawdown Periods{RESET}")
        print(f"  {'─' * 80}")
        print(f"  {'Peak Date':>12s}  {'Trough Date':>12s}  {'Recovery':>12s}  {'Peak $':>9s}  {'Trough $':>9s}  {'DD %':>7s}  {'Days':>5s}")
        print(f"  {'─' * 80}")
        for _, d in top_dd.iterrows():
            rec = str(d["recovery_date"]) if d["recovery_date"] else "ONGOING"
            days_str = str(d["days_to_trough"]) if d["days_to_trough"] else "?"
            print(
                f"  {RED}{str(d['peak_date']):>12s}  "
                f"{str(d['trough_date']):>12s}  "
                f"{rec:>12s}  "
                f"${d['peak_eq']:>8,.2f}  "
                f"${d['trough_eq']:>8,.2f}  "
                f"{d['drawdown_pct']:>+6.2f}%  "
                f"{days_str:>5s}{RESET}"
            )
        print(f"  {'─' * 80}")

    # ── Monthly heatmap (text) ──────────────────────────────────
    heatmap = compute_monthly_heatmap(daily)
    if not heatmap.empty:
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        print(f"\n{BOLD}  Monthly P&L % Heatmap{RESET}")
        print(f"  {'─' * 76}")
        header = f"  {'Year':>6s}  " + "  ".join(f"{m:>5s}" for m in month_names) + f"  {'Total':>7s}"
        print(header)
        print(f"  {'─' * 76}")
        for year, row in heatmap.iterrows():
            parts = []
            year_total = 0.0
            for m in range(1, 13):
                val = row.get(m, 0.0)
                year_total += val
                if val > 0:
                    parts.append(f"{GREEN}{val:>+5.1f}{RESET}")
                elif val < 0:
                    parts.append(f"{RED}{val:>+5.1f}{RESET}")
                else:
                    parts.append(f"{'0.0':>5s}")
            total_colour = GREEN if year_total > 0 else RED if year_total < 0 else ""
            reset_t = RESET if total_colour else ""
            print(f"  {year:>6d}  " + "  ".join(parts) + f"  {total_colour}{year_total:>+7.1f}{reset_t}")
        print(f"  {'─' * 76}")

    print(f"\n{BOLD}{'=' * 80}{RESET}\n")


# =========================================================================
# CSV export
# =========================================================================

def export_csv(daily: pd.DataFrame, out_path: str = "gap_analysis_daily.csv") -> None:
    """Export daily P&L to CSV."""
    daily.to_csv(out_path, index=False)
    print(f"  CSV saved → {out_path}")


# =========================================================================
# HTML report
# =========================================================================

def generate_html_report(
    daily: pd.DataFrame,
    streaks: pd.DataFrame,
    drawdowns: pd.DataFrame,
    out_path: str = "gap_analysis_report.html",
) -> None:
    """Generate a self-contained HTML gap analysis report."""

    total_days  = len(daily)
    gap_days    = daily["is_gap"].sum()
    profit_days = total_days - gap_days
    loss_days   = len(daily[daily["daily_pnl"] < 0])
    zero_days   = len(daily[daily["daily_pnl"] == 0])
    idle_days   = len(daily[daily["n_trades"] == 0])

    best_idx  = daily["daily_pnl"].idxmax()
    worst_idx = daily["daily_pnl"].idxmin()

    # Monthly heatmap data
    heatmap = compute_monthly_heatmap(daily)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Heatmap table rows
    heatmap_rows = ""
    for year, row in heatmap.iterrows():
        cells = ""
        year_total = 0.0
        for m in range(1, 13):
            val = row.get(m, 0.0)
            year_total += val
            if val > 2:
                cls = "strong-up"
            elif val > 0:
                cls = "up"
            elif val < -2:
                cls = "strong-down"
            elif val < 0:
                cls = "down"
            else:
                cls = "neutral"
            cells += f'<td class="{cls}">{val:+.1f}%</td>'
        total_cls = "up" if year_total > 0 else "down" if year_total < 0 else "neutral"
        cells += f'<td class="{total_cls}" style="font-weight:bold">{year_total:+.1f}%</td>'
        heatmap_rows += f"<tr><td><b>{year}</b></td>{cells}</tr>\n"

    # Gap days table rows
    gap_rows = ""
    gaps = daily[daily["is_gap"]]
    for _, r in gaps.iterrows():
        cls = "loss-row" if r["daily_pnl"] < 0 else "zero-row"
        gap_rows += f"""<tr class="{cls}">
            <td>{r['date']}</td>
            <td>${r['equity_close']:,.2f}</td>
            <td>${r['daily_pnl']:+.2f}</td>
            <td>{r['daily_pct']:+.2f}%</td>
            <td>{r['cum_pct']:+.1f}%</td>
            <td>{r['n_trades']}</td>
            <td>${r['realized_pnl']:+.2f}</td>
        </tr>\n"""

    # Streak rows
    streak_rows = ""
    for _, s in streaks.head(20).iterrows():
        streak_rows += f"""<tr>
            <td>{s['start_date']}</td>
            <td>{s['end_date']}</td>
            <td>{s['days']}</td>
            <td>${s['total_loss']:+.2f}</td>
            <td>{s['worst_day']}</td>
            <td>${s['worst_day_pnl']:+.2f}</td>
        </tr>\n"""

    # Drawdown rows
    dd_rows = ""
    for _, d in drawdowns.head(20).iterrows():
        rec = str(d["recovery_date"]) if d["recovery_date"] else "ONGOING"
        dd_rows += f"""<tr>
            <td>{d['peak_date']}</td>
            <td>{d['trough_date']}</td>
            <td>{rec}</td>
            <td>${d['peak_eq']:,.2f}</td>
            <td>${d['trough_eq']:,.2f}</td>
            <td>{d['drawdown_pct']:+.2f}%</td>
            <td>{d['days_to_trough']}</td>
        </tr>\n"""

    # Equity chart data (JSON arrays for inline Chart.js)
    dates_json   = daily["date"].astype(str).tolist()
    equity_json  = daily["equity_close"].tolist()
    pnl_json     = daily["daily_pnl"].tolist()
    pnl_colours  = ["#dc3545" if v <= 0 else "#28a745" for v in pnl_json]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gap Analysis Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1400px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #e0e0e0; }}
  h1 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }}
  h2 {{ color: #7dd3fc; margin-top: 30px; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
  .stat-card {{ background: #16213e; border-radius: 8px; padding: 20px; text-align: center; }}
  .stat-card .value {{ font-size: 2em; font-weight: bold; }}
  .stat-card .label {{ color: #7dd3fc; font-size: 0.9em; margin-top: 5px; }}
  .green {{ color: #28a745; }}
  .red {{ color: #dc3545; }}
  .yellow {{ color: #ffc107; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.85em; }}
  th {{ background: #16213e; color: #7dd3fc; padding: 8px 12px; text-align: right; position: sticky; top: 0; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #2a2a4a; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  .loss-row {{ background: rgba(220, 53, 69, 0.15); }}
  .zero-row {{ background: rgba(255, 193, 7, 0.10); }}
  .strong-up {{ background: #155724; color: #28a745; font-weight: bold; }}
  .up {{ background: rgba(40, 167, 69, 0.2); color: #28a745; }}
  .strong-down {{ background: #721c24; color: #dc3545; font-weight: bold; }}
  .down {{ background: rgba(220, 53, 69, 0.2); color: #dc3545; }}
  .neutral {{ color: #6c757d; }}
  canvas {{ background: #16213e; border-radius: 8px; padding: 10px; margin: 10px 0; }}
  .table-container {{ max-height: 500px; overflow-y: auto; border-radius: 8px; }}
  .generated {{ color: #6c757d; font-size: 0.8em; text-align: center; margin-top: 40px; }}
</style>
</head>
<body>

<h1>Gap Analysis Report</h1>
<p>Every day the bot earned $0 or lost money is a gap that needs investigation.</p>

<div class="stats">
  <div class="stat-card">
    <div class="value">{total_days:,}</div>
    <div class="label">Total Days</div>
  </div>
  <div class="stat-card">
    <div class="value green">{profit_days:,}</div>
    <div class="label">Profitable Days ({profit_days/total_days*100:.1f}%)</div>
  </div>
  <div class="stat-card">
    <div class="value red">{loss_days:,}</div>
    <div class="label">Loss Days ({loss_days/total_days*100:.1f}%)</div>
  </div>
  <div class="stat-card">
    <div class="value yellow">{zero_days:,}</div>
    <div class="label">Zero-Change Days ({zero_days/total_days*100:.1f}%)</div>
  </div>
  <div class="stat-card">
    <div class="value">{idle_days:,}</div>
    <div class="label">Idle Days ({idle_days/total_days*100:.1f}%)</div>
  </div>
  <div class="stat-card">
    <div class="value red">{gap_days:,}</div>
    <div class="label">Total Gap Days ({gap_days/total_days*100:.1f}%)</div>
  </div>
</div>

<div class="stats">
  <div class="stat-card">
    <div class="value green">{str(daily.loc[best_idx, 'date'])}</div>
    <div class="label">Best Day: ${daily.loc[best_idx, 'daily_pnl']:+.2f} ({daily.loc[best_idx, 'daily_pct']:+.2f}%)</div>
  </div>
  <div class="stat-card">
    <div class="value red">{str(daily.loc[worst_idx, 'date'])}</div>
    <div class="label">Worst Day: ${daily.loc[worst_idx, 'daily_pnl']:+.2f} ({daily.loc[worst_idx, 'daily_pct']:+.2f}%)</div>
  </div>
</div>

<h2>Equity Curve</h2>
<canvas id="equityChart" height="80"></canvas>

<h2>Daily P&amp;L</h2>
<canvas id="pnlChart" height="80"></canvas>

<h2>Monthly P&amp;L % Heatmap</h2>
<table>
  <thead>
    <tr><th>Year</th>{"".join(f"<th>{m}</th>" for m in month_names)}<th>Total</th></tr>
  </thead>
  <tbody>
    {heatmap_rows}
  </tbody>
</table>

<h2>All Gap Days (P&amp;L &le; $0) &mdash; {len(gaps)} days</h2>
<div class="table-container">
<table>
  <thead>
    <tr><th>Date</th><th>Equity</th><th>Daily P&amp;L</th><th>Daily %</th><th>Cum %</th><th>Trades</th><th>Real P&amp;L</th></tr>
  </thead>
  <tbody>
    {gap_rows}
  </tbody>
</table>
</div>

<h2>Top 20 Worst Gap Streaks</h2>
<table>
  <thead>
    <tr><th>Start</th><th>End</th><th>Days</th><th>Total Loss</th><th>Worst Day</th><th>Worst P&amp;L</th></tr>
  </thead>
  <tbody>
    {streak_rows}
  </tbody>
</table>

<h2>Top 20 Drawdown Periods</h2>
<table>
  <thead>
    <tr><th>Peak</th><th>Trough</th><th>Recovery</th><th>Peak $</th><th>Trough $</th><th>DD %</th><th>Days</th></tr>
  </thead>
  <tbody>
    {dd_rows}
  </tbody>
</table>

<script>
const dates = {dates_json};
const equity = {equity_json};
const pnl = {pnl_json};
const pnlColours = {pnl_colours};

new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{
    labels: dates,
    datasets: [{{
      label: 'Equity ($)',
      data: equity,
      borderColor: '#00d4ff',
      backgroundColor: 'rgba(0,212,255,0.1)',
      fill: true,
      pointRadius: 0,
      borderWidth: 1.5,
    }}]
  }},
  options: {{
    responsive: true,
    scales: {{
      x: {{ display: true, ticks: {{ maxTicksLimit: 20, color: '#6c757d' }}, grid: {{ color: '#2a2a4a' }} }},
      y: {{ display: true, ticks: {{ color: '#6c757d' }}, grid: {{ color: '#2a2a4a' }} }}
    }},
    plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }}
  }}
}});

new Chart(document.getElementById('pnlChart'), {{
  type: 'bar',
  data: {{
    labels: dates,
    datasets: [{{
      label: 'Daily P&L ($)',
      data: pnl,
      backgroundColor: pnlColours,
      borderWidth: 0,
    }}]
  }},
  options: {{
    responsive: true,
    scales: {{
      x: {{ display: true, ticks: {{ maxTicksLimit: 20, color: '#6c757d' }}, grid: {{ display: false }} }},
      y: {{ display: true, ticks: {{ color: '#6c757d' }}, grid: {{ color: '#2a2a4a' }} }}
    }},
    plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }}
  }}
}});
</script>

<p class="generated">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by AS-Grid gap analysis</p>
</body>
</html>"""

    Path(out_path).write_text(html)
    print(f"  HTML report saved → {out_path}")


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    trades_path = sys.argv[1] if len(sys.argv) > 1 else "bitunix_best_grid_trades.csv"
    equity_path = sys.argv[2] if len(sys.argv) > 2 else "bitunix_best_equity_curve.csv"

    if not os.path.exists(trades_path):
        print(f"ERROR: trades file not found: {trades_path}")
        print("Usage: python scripts/analyze_gaps.py [trades_csv] [equity_csv]")
        sys.exit(1)
    if not os.path.exists(equity_path):
        print(f"ERROR: equity curve file not found: {equity_path}")
        sys.exit(1)

    print(f"  Loading trades from:  {trades_path}")
    trades_df = load_trades(trades_path)
    print(f"  Loading equity from:  {equity_path}")
    equity_df = load_equity_curve(equity_path)

    print(f"  Computing daily P&L...")
    daily = compute_daily_pnl(trades_df, equity_df)

    print(f"  Identifying gap streaks...")
    streaks = identify_gap_streaks(daily)

    print(f"  Computing drawdown periods...")
    drawdowns = compute_drawdown_periods(daily)

    # ── Console summary ─────────────────────────────────────────
    print_summary(daily, streaks, drawdowns)

    # ── CSV ──────────────────────────────────────────────────────
    export_csv(daily)

    # ── HTML ─────────────────────────────────────────────────────
    generate_html_report(daily, streaks, drawdowns)


if __name__ == "__main__":
    main()

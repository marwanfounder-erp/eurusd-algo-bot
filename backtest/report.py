"""Generate an interactive dark-theme HTML backtest report using Plotly."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger


# ──────────────────────────────────────────────────────────────────────
# Dark-theme palette
# ──────────────────────────────────────────────────────────────────────
_BG      = "#0d1117"
_PANEL   = "#161b22"
_BORDER  = "#30363d"
_TEXT    = "#c9d1d9"
_GREEN   = "#3fb950"
_RED     = "#f85149"
_BLUE    = "#58a6ff"
_PURPLE  = "#bc8cff"
_YELLOW  = "#e3b341"


def _metric_cards_html(results: dict[str, Any]) -> str:
    """Return the HTML for the top-of-page KPI metric cards."""
    tr    = results.get("total_return", 0.0)
    sh    = results.get("sharpe", 0.0)
    dd    = results.get("max_drawdown", 0.0)
    wr    = results.get("win_rate", 0.0)
    pf    = results.get("profit_factor", 0.0)
    nt    = results.get("total_trades", 0)
    final = results.get("final_balance", results.get("initial_balance", 10_000))

    def _card(label: str, value: str, colour: str) -> str:
        return (
            f'<div style="background:{_PANEL};border:1px solid {_BORDER};'
            f'border-radius:8px;padding:18px 24px;min-width:140px;text-align:center;">'
            f'<div style="font-size:12px;color:#8b949e;letter-spacing:1px;'
            f'text-transform:uppercase;margin-bottom:6px;">{label}</div>'
            f'<div style="font-size:26px;font-weight:700;color:{colour};">{value}</div>'
            f'</div>'
        )

    tr_col  = _GREEN if tr >= 0 else _RED
    sh_col  = _GREEN if sh >= 1 else (_YELLOW if sh >= 0.5 else _RED)
    dd_col  = _GREEN if dd <= 0.05 else (_YELLOW if dd <= 0.10 else _RED)
    wr_col  = _GREEN if wr >= 0.52 else (_YELLOW if wr >= 0.48 else _RED)
    pf_col  = _GREEN if pf >= 1.5 else (_YELLOW if pf >= 1.0 else _RED)

    cards = "".join([
        _card("Total Return",   f"{tr:+.1%}",        tr_col),
        _card("Sharpe Ratio",   f"{sh:.2f}",         sh_col),
        _card("Max Drawdown",   f"{dd:.1%}",         dd_col),
        _card("Win Rate",       f"{wr:.1%}",         wr_col),
        _card("Profit Factor",  f"{pf:.2f}",         pf_col),
        _card("Total Trades",   f"{nt}",             _BLUE),
        _card("Final Balance",  f"${final:,.0f}",    _BLUE),
    ])

    return (
        f'<div style="display:flex;flex-wrap:wrap;gap:16px;'
        f'margin:24px 0 32px 0;">{cards}</div>'
    )


def generate_report(
    results: dict[str, Any],
    output_path: str = "logs/backtest_report.html",
    title: str = "EUR/USD London Breakout — Backtest Report",
) -> None:
    """Build and save a self-contained dark-theme HTML report.

    Sections
    --------
    1. KPI metric cards (top of page)
    2. Equity curve
    3. Monthly P&L bar chart
    4. Drawdown area chart
    5. Win/loss PnL histogram
    6. Win vs loss pie
    7. Stats table
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    equity_curve: list[float] = results.get("equity_curve", [])
    trades: list[dict[str, Any]] = results.get("trades", [])
    monthly_pnl: dict[str, float] = results.get("monthly_pnl", {})

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Equity Curve",   "Stats Table",
            "Monthly P&L",    "Drawdown (%)",
            "P&L Distribution", "Win vs Loss",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "table"}],
            [{"type": "bar"},     {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "pie"}],
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    # ── 1. Equity Curve ──────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            y=equity_curve, mode="lines", name="Equity",
            line={"color": _BLUE, "width": 2},
            fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
        ),
        row=1, col=1,
    )

    # ── 2. Stats Table ───────────────────────────────────────────────
    stat_labels = [
        "Total Trades", "Win Rate", "Total Return", "Sharpe Ratio",
        "Max Drawdown", "Profit Factor", "Initial Balance", "Final Balance",
        "Gross Profit", "Gross Loss", "Avg Win (pips)", "Avg Loss (pips)",
    ]
    def _fmt(k: str, default: Any = "N/A") -> str:
        v = results.get(k, default)
        if isinstance(v, float) and "pct" in k or "return" in k or "rate" in k or "drawdown" in k:
            return f"{v:.2%}"
        return str(v)

    stat_values = [
        str(results.get("total_trades", 0)),
        f"{results.get('win_rate', 0):.1%}",
        f"{results.get('total_return', 0):+.2%}",
        f"{results.get('sharpe', 0):.2f}",
        f"{results.get('max_drawdown', 0):.2%}",
        str(results.get("profit_factor", 0)),
        f"${results.get('initial_balance', 0):,.2f}",
        f"${results.get('final_balance', 0):,.2f}",
        f"${results.get('gross_profit', 0):,.2f}",
        f"${results.get('gross_loss', 0):,.2f}",
        f"{results.get('avg_win_pips', 0):.1f}",
        f"{results.get('avg_loss_pips', 0):.1f}",
    ]
    fig.add_trace(
        go.Table(
            header={
                "values": ["<b>Metric</b>", "<b>Value</b>"],
                "fill_color": "#1f6feb",
                "font": {"color": "white", "size": 12},
                "align": "left",
                "line_color": _BORDER,
            },
            cells={
                "values": [stat_labels, stat_values],
                "fill_color": [[_PANEL, "#0d1117"] * 6],
                "font": {"color": _TEXT, "size": 11},
                "align": "left",
                "line_color": _BORDER,
            },
        ),
        row=1, col=2,
    )

    # ── 3. Monthly P&L ───────────────────────────────────────────────
    if monthly_pnl:
        months = sorted(monthly_pnl.keys())
        pvals  = [monthly_pnl[m] for m in months]
        colors = [_GREEN if v >= 0 else _RED for v in pvals]
        fig.add_trace(
            go.Bar(x=months, y=pvals, name="Monthly P&L",
                   marker_color=colors, showlegend=False),
            row=2, col=1,
        )

    # ── 4. Drawdown ───────────────────────────────────────────────────
    if equity_curve:
        eq   = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        dd   = (peak - eq) / np.where(peak > 0, peak, 1) * 100
        fig.add_trace(
            go.Scatter(
                y=-dd, mode="lines", name="Drawdown",
                line={"color": _RED, "width": 1.5},
                fill="tozeroy", fillcolor="rgba(248,81,73,0.12)",
                showlegend=False,
            ),
            row=2, col=2,
        )

    # ── 5. P&L Distribution ──────────────────────────────────────────
    if trades:
        pnls = [t["pnl"] for t in trades]
        fig.add_trace(
            go.Histogram(x=pnls, nbinsx=30, name="Trade P&L",
                         marker_color=_PURPLE, showlegend=False),
            row=3, col=1,
        )

    # ── 6. Win vs Loss Pie ────────────────────────────────────────────
    wins   = results.get("wins", 0)
    losses = results.get("losses", 0)
    if wins + losses > 0:
        fig.add_trace(
            go.Pie(
                labels=["Wins", "Losses"], values=[wins, losses],
                marker_colors=[_GREEN, _RED], hole=0.42,
                textfont={"color": "white"}, showlegend=True,
            ),
            row=3, col=2,
        )

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG, plot_bgcolor=_PANEL,
        height=1300,
        title={"text": title, "x": 0.5, "font": {"size": 18, "color": _TEXT}},
        font={"color": _TEXT},
        legend={"orientation": "h", "y": -0.02},
    )
    fig.update_xaxes(gridcolor=_BORDER, zerolinecolor=_BORDER)
    fig.update_yaxes(gridcolor=_BORDER, zerolinecolor=_BORDER)
    fig.update_xaxes(title_text="Bar #",   row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_xaxes(title_text="Month",   row=2, col=1, tickangle=-45)
    fig.update_yaxes(title_text="P&L ($)", row=2, col=1)
    fig.update_xaxes(title_text="Bar #",   row=2, col=2)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=2)
    fig.update_xaxes(title_text="P&L ($)", row=3, col=1)
    fig.update_yaxes(title_text="Trades",  row=3, col=1)

    # ── Build full HTML with metric cards ─────────────────────────────
    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    cards_html = _metric_cards_html(results)
    ts_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: {_BG};
      color: {_TEXT};
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      padding: 32px 40px 60px;
    }}
    h1 {{
      font-size: 22px;
      font-weight: 600;
      margin-bottom: 4px;
      color: #e6edf3;
    }}
    .subtitle {{
      font-size: 13px;
      color: #8b949e;
      margin-bottom: 28px;
    }}
    hr {{
      border: none;
      border-top: 1px solid {_BORDER};
      margin: 28px 0;
    }}
  </style>
</head>
<body>
  <h1>📊 {title}</h1>
  <div class="subtitle">Generated: {ts_str} &nbsp;|&nbsp; Strategy: London Session Breakout &nbsp;|&nbsp; Symbol: EURUSD H1</div>
  <hr>
  {cards_html}
  {plot_html}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    logger.info("Report saved → {}", output_path)
    _print_text_summary(results)


def _print_text_summary(results: dict[str, Any]) -> None:
    print(
        f"\n  Final Balance  : ${results.get('final_balance', 0):,.2f}"
        f"  |  Total Return: {results.get('total_return', 0):+.2%}"
        f"  |  Sharpe: {results.get('sharpe', 0):.2f}"
    )

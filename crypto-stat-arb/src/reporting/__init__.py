"""Reporting module for crypto statistical arbitrage research."""

from src.reporting.metrics import (
    avg_holding_period,
    avg_trade,
    calmar_ratio,
    compute_drawdown,
    compute_equity_curve,
    exposure,
    max_drawdown,
    net_vs_gross,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    turnover,
    win_rate,
)
from src.reporting.plots import (
    plot_drawdown,
    plot_equity_curve,
    plot_monthly_returns,
    plot_rolling_beta,
    plot_rolling_correlation,
    plot_rolling_sharpe,
    plot_spread,
    plot_trade_distribution,
    plot_zscore,
)
from src.reporting.report_builder import generate_pair_report, generate_summary_report

__all__ = [
    # Metrics
    "compute_equity_curve",
    "compute_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "win_rate",
    "profit_factor",
    "avg_trade",
    "turnover",
    "avg_holding_period",
    "exposure",
    "net_vs_gross",
    # Plots
    "plot_equity_curve",
    "plot_drawdown",
    "plot_zscore",
    "plot_spread",
    "plot_rolling_beta",
    "plot_rolling_correlation",
    "plot_rolling_sharpe",
    "plot_trade_distribution",
    "plot_monthly_returns",
    # Report builder
    "generate_pair_report",
    "generate_summary_report",
]

"""Chart generation with matplotlib for backtest results.

This module provides plotting functions for visualizing backtest results.
All charts save to the reports/ directory as PNG files.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Set default style
plt.style.use('seaborn-v0_8-darkgrid')


def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Equity Curve",
    save_path: Optional[str] = None,
) -> str:
    """Plot equity curve with optional benchmark.

    Args:
        equity_curve: Series of equity values.
        benchmark: Optional benchmark series for comparison.
        title: Chart title.
        save_path: Path to save the chart. If None, saves to reports/equity_curve.png.

    Returns:
        Path to saved chart.
    """
    if save_path is None:
        save_path = Path("reports") / "equity_curve.png"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(equity_curve.index, equity_curve.values, label='Strategy', linewidth=1.5)

    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values, label='Benchmark', linewidth=1, alpha=0.7)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved equity curve to {save_path}")
    return str(save_path)


def plot_drawdown(
    drawdown_series: pd.DataFrame,
    title: str = "Drawdown",
    save_path: Optional[str] = None,
) -> str:
    """Plot drawdown chart.

    Args:
        drawdown_series: DataFrame with 'drawdown_pct' column.
        title: Chart title.
        save_path: Path to save the chart.

    Returns:
        Path to saved chart.
    """
    if save_path is None:
        save_path = Path("reports") / "drawdown.png"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    if 'drawdown_pct' in drawdown_series.columns:
        ax.fill_between(
            drawdown_series.index,
            drawdown_series['drawdown_pct'].values * 100,
            0,
            alpha=0.3,
            color='red',
            label='Drawdown'
        )
        ax.plot(drawdown_series.index, drawdown_series['drawdown_pct'].values * 100, color='red', linewidth=1)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved drawdown chart to {save_path}")
    return str(save_path)


def plot_zscore(
    zscore_series: pd.Series,
    entry_thresholds: List[float] = None,
    title: str = "Z-Score",
    save_path: Optional[str] = None,
) -> str:
    """Plot z-score chart with entry/exit bands.

    Args:
        zscore_series: Series of z-score values.
        entry_thresholds: List of entry threshold values (e.g., [2.0, -2.0]).
        title: Chart title.
        save_path: Path to save the chart.

    Returns:
        Path to saved chart.
    """
    if save_path is None:
        save_path = Path("reports") / "zscore.png"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(zscore_series.index, zscore_series.values, label='Z-Score', linewidth=1)

    # Plot threshold bands
    if entry_thresholds is None:
        entry_thresholds = [2.0, -2.0]

    for threshold in entry_thresholds:
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=-threshold, color='green', linestyle='--', alpha=0.5)

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Z-Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved z-score chart to {save_path}")
    return str(save_path)


def plot_spread(
    spread_series: pd.Series,
    mean: float = 0,
    title: str = "Spread",
    save_path: Optional[str] = None,
) -> str:
    """Plot spread chart.

    Args:
        spread_series: Series of spread values.
        mean: Mean spread value to plot as reference line.
        title: Chart title.
        save_path: Path to save the chart.

    Returns:
        Path to saved chart.
    """
    if save_path is None:
        save_path = Path("reports") / "spread.png"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(spread_series.index, spread_series.values, label='Spread', linewidth=1)
    ax.axhline(y=mean, color='red', linestyle='--', label=f'Mean ({mean:.4f})', alpha=0.7)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Spread')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved spread chart to {save_path}")
    return str(save_path)


def plot_rolling_beta(
    beta_series: pd.Series,
    title: str = "Rolling Beta",
    save_path: Optional[str] = None,
) -> str:
    """Plot rolling beta chart.

    Args:
        beta_series: Series of rolling beta values.
        title: Chart title.
        save_path: Path to save the chart.

    Returns:
        Path to saved chart.
    """
    if save_path is None:
        save_path = Path("reports") / "rolling_beta.png"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(beta_series.index, beta_series.values, label='Rolling Beta', linewidth=1)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Beta = 1')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Beta')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved rolling beta chart to {save_path}")
    return str(save_path)


def plot_rolling_correlation(
    corr_series: pd.Series,
    title: str = "Rolling Correlation",
    save_path: Optional[str] = None,
) -> str:
    """Plot rolling correlation chart.

    Args:
        corr_series: Series of rolling correlation values.
        title: Chart title.
        save_path: Path to save the chart.

    Returns:
        Path to saved chart.
    """
    if save_path is None:
        save_path = Path("reports") / "rolling_correlation.png"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(corr_series.index, corr_series.values, label='Rolling Correlation', linewidth=1)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Correlation')
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved rolling correlation chart to {save_path}")
    return str(save_path)


def plot_rolling_sharpe(
    sharpe_series: pd.Series,
    title: str = "Rolling Sharpe Ratio",
    save_path: Optional[str] = None,
) -> str:
    """Plot rolling Sharpe ratio chart.

    Args:
        sharpe_series: Series of rolling Sharpe ratio values.
        title: Chart title.
        save_path: Path to save the chart.

    Returns:
        Path to saved chart.
    """
    if save_path is None:
        save_path = Path("reports") / "rolling_sharpe.png"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(sharpe_series.index, sharpe_series.values, label='Rolling Sharpe', linewidth=1, color='blue')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved rolling Sharpe chart to {save_path}")
    return str(save_path)


def plot_trade_distribution(
    trades: List[dict[str, Any]],
    title: str = "Trade P&L Distribution",
    save_path: Optional[str] = None,
) -> str:
    """Plot trade P&L distribution histogram.

    Args:
        trades: List of trade dicts with 'net_pnl' key.
        title: Chart title.
        save_path: Path to save the chart.

    Returns:
        Path to saved chart.
    """
    if save_path is None:
        save_path = Path("reports") / "trade_distribution.png"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    pnls = [t.get('net_pnl', 0) for t in trades]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(pnls, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Breakeven')
    ax.axvline(x=np.mean(pnls), color='green', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(pnls):.2f}')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Net P&L ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved trade distribution chart to {save_path}")
    return str(save_path)


def plot_monthly_returns(
    returns_by_month: pd.Series,
    title: str = "Monthly Returns",
    save_path: Optional[str] = None,
) -> str:
    """Plot monthly returns as a bar chart.

    Args:
        returns_by_month: Series of monthly returns.
        title: Chart title.
        save_path: Path to save the chart.

    Returns:
        Path to saved chart.
    """
    if save_path is None:
        save_path = Path("reports") / "monthly_returns.png"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['green' if r >= 0 else 'red' for r in returns_by_month.values]
    ax.bar(returns_by_month.index, returns_by_month.values * 100, color=colors, alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Return (%)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved monthly returns chart to {save_path}")
    return str(save_path)

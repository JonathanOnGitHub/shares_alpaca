"""Performance metrics calculator for backtest results.

This module provides all the performance metrics required for evaluating
pair trading strategy results.
"""

import logging
from typing import Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_equity_curve(
    trades: List[dict[str, Any]],
    initial_capital: float,
) -> pd.Series:
    """Compute equity curve from trade list.

    Args:
        trades: List of trade dicts with 'net_pnl' and 'exit_time' keys.
        initial_capital: Starting capital.

    Returns:
        Series of equity values indexed by timestamp.
    """
    if not trades:
        return pd.Series([], dtype=float)

    equity = [initial_capital]
    timestamps = [pd.Timestamp(trades[0]['exit_time']) if trades else pd.Timestamp.now()]

    for trade in trades:
        new_equity = equity[-1] + trade.get('net_pnl', 0)
        equity.append(new_equity)
        timestamps.append(pd.Timestamp(trade['exit_time']))

    return pd.Series(equity, index=timestamps)


def compute_drawdown(
    equity_curve: pd.Series,
) -> pd.DataFrame:
    """Compute drawdown series from equity curve.

    Args:
        equity_curve: Series of equity values.

    Returns:
        DataFrame with columns:
        - drawdown: Absolute drawdown in dollars
        - drawdown_pct: Percentage drawdown
        - drawdown_peak: Peak equity at each point
    """
    if len(equity_curve) == 0:
        return pd.DataFrame(columns=['drawdown', 'drawdown_pct', 'drawdown_peak'])

    running_max = equity_curve.expanding().max()
    drawdown = equity_curve - running_max
    drawdown_pct = drawdown / running_max
    drawdown_peak = running_max

    return pd.DataFrame({
        'drawdown': drawdown,
        'drawdown_pct': drawdown_pct,
        'drawdown_peak': drawdown_peak,
    }, index=equity_curve.index)


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sharpe ratio (annualized).

    Args:
        returns: Series of returns.
        periods_per_year: Number of periods per year for annualization.

    Returns:
        Sharpe ratio.
    """
    clean_returns = returns.dropna()

    if len(clean_returns) < 2:
        return 0.0

    mean_return = clean_returns.mean()
    std_return = clean_returns.std()

    if std_return == 0 or np.isnan(std_return):
        return 0.0

    sharpe = mean_return / std_return * np.sqrt(periods_per_year)

    return float(sharpe)


def sortino_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sortino ratio (annualized).

    Uses downside deviation (std of negative returns only).

    Args:
        returns: Series of returns.
        periods_per_year: Number of periods per year for annualization.

    Returns:
        Sortino ratio.
    """
    clean_returns = returns.dropna()

    if len(clean_returns) < 2:
        return 0.0

    mean_return = clean_returns.mean()

    # Downside deviation - only negative returns
    downside_returns = clean_returns[clean_returns < 0]

    if len(downside_returns) == 0:
        return 0.0

    downside_std = downside_returns.std()

    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    sortino = mean_return / downside_std * np.sqrt(periods_per_year)

    return float(sortino)


def max_drawdown(
    drawdown_series: pd.DataFrame,
) -> float:
    """Get maximum drawdown percentage.

    Args:
        drawdown_series: DataFrame with 'drawdown_pct' column.

    Returns:
        Maximum drawdown as a fraction (e.g., 0.20 for 20%).
    """
    if len(drawdown_series) == 0:
        return 0.0

    if 'drawdown_pct' in drawdown_series.columns:
        return float(abs(drawdown_series['drawdown_pct'].min()))

    return 0.0


def calmar_ratio(
    equity_curve: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        equity_curve: Series of equity values.
        periods_per_year: Number of periods per year.

    Returns:
        Calmar ratio.
    """
    if len(equity_curve) < 2:
        return 0.0

    # Total return
    total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]

    # Max drawdown
    drawdown_df = compute_drawdown(equity_curve)
    max_dd = max_drawdown(drawdown_df)

    if max_dd == 0:
        return 0.0

    # Annualize return (assuming daily periods)
    n_periods = len(equity_curve)
    years = n_periods / periods_per_year
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    calmar = annualized_return / max_dd

    return float(calmar)


def win_rate(
    trades: List[dict[str, Any]],
) -> float:
    """Calculate win rate (percentage of profitable trades).

    Args:
        trades: List of trade dicts with 'net_pnl' key.

    Returns:
        Win rate as a fraction (e.g., 0.60 for 60%).
    """
    if not trades:
        return 0.0

    winning_trades = sum(1 for t in trades if t.get('net_pnl', 0) > 0)
    return winning_trades / len(trades)


def profit_factor(
    trades: List[dict[str, Any]],
) -> float:
    """Calculate profit factor (gross profits / gross losses).

    Args:
        trades: List of trade dicts with 'net_pnl' key.

    Returns:
        Profit factor (e.g., 1.5 means $1.50 profit for every $1 loss).
    """
    if not trades:
        return 0.0

    gross_profits = sum(t.get('net_pnl', 0) for t in trades if t.get('net_pnl', 0) > 0)
    gross_losses = abs(sum(t.get('net_pnl', 0) for t in trades if t.get('net_pnl', 0) < 0))

    if gross_losses == 0:
        return float('inf') if gross_profits > 0 else 0.0

    return gross_profits / gross_losses


def avg_trade(
    trades: List[dict[str, Any]],
) -> float:
    """Calculate average trade P&L.

    Args:
        trades: List of trade dicts with 'net_pnl' key.

    Returns:
        Average net P&L per trade.
    """
    if not trades:
        return 0.0

    return float(np.mean([t.get('net_pnl', 0) for t in trades]))


def turnover(
    trades: List[dict[str, Any]],
    equity_curve: pd.Series,
) -> float:
    """Calculate annualized turnover.

    Turnover = sum of absolute position changes / portfolio value.

    Args:
        trades: List of trade dicts.
        equity_curve: Series of equity values.

    Returns:
        Annualized turnover rate.
    """
    if not trades or len(equity_curve) == 0:
        return 0.0

    # Sum of notional values traded
    total_notional = sum(t.get('notional', 0) for t in trades)

    # Average equity
    avg_equity = equity_curve.mean()

    if avg_equity == 0:
        return 0.0

    # Annualize (assuming daily bars = 252 trading days)
    periods_per_year = 252
    n_periods = len(equity_curve)

    # Turnover = total notional / avg equity
    raw_turnover = total_notional / avg_equity

    # Annualize
    annual_turnover = raw_turnover * (periods_per_year / n_periods) if n_periods > 0 else 0

    return float(annual_turnover)


def avg_holding_period(
    trades: List[dict[str, Any]],
) -> float:
    """Calculate average holding period in bars.

    Args:
        trades: List of trade dicts with 'holding_period_bars' key.

    Returns:
        Average holding period in bars.
    """
    if not trades:
        return 0.0

    holding_periods = [t.get('holding_period_bars', 0) for t in trades]

    if not holding_periods:
        return 0.0

    return float(np.mean(holding_periods))


def exposure(
    trades: List[dict[str, Any]],
    equity_curve: pd.Series,
) -> float:
    """Calculate percentage of time in market.

    Args:
        trades: List of trade dicts.
        equity_curve: Series of equity values.

    Returns:
        Fraction of time in market (e.g., 0.40 for 40%).
    """
    if not trades or len(equity_curve) == 0:
        return 0.0

    # Sum of holding periods
    total_bars_in_market = sum(t.get('holding_period_bars', 0) for t in trades)
    total_bars = len(equity_curve)

    if total_bars == 0:
        return 0.0

    return float(min(1.0, total_bars_in_market / total_bars))


def net_vs_gross(
    gross_return: float,
    total_costs: float,
) -> dict[str, float]:
    """Calculate net vs gross performance breakdown.

    Args:
        gross_return: Gross return in dollars or fraction.
        total_costs: Total costs in dollars.

    Returns:
        Dict with gross_return, total_costs, net_return, cost_ratio.
    """
    net_return = gross_return - total_costs

    return {
        'gross_return': float(gross_return),
        'total_costs': float(total_costs),
        'net_return': float(net_return),
        'cost_ratio': float(abs(total_costs / gross_return)) if gross_return != 0 else 0.0,
    }


def compute_all_metrics(
    trades: List[dict[str, Any]],
    equity_curve: pd.Series,
    initial_capital: float,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Compute all performance metrics.

    Args:
        trades: List of trade dicts.
        equity_curve: Series of equity values.
        initial_capital: Starting capital.
        periods_per_year: Periods per year for annualization.

    Returns:
        Dictionary of all metrics.
    """
    # Compute returns
    equity_clean = equity_curve.dropna()
    if len(equity_clean) > 1:
        returns = equity_clean.pct_change().dropna()
    else:
        returns = pd.Series([], dtype=float)

    # Compute drawdown
    drawdown_df = compute_drawdown(equity_clean)

    # Get gross and net totals
    gross_return = sum(t.get('gross_pnl', 0) for t in trades)
    net_return = sum(t.get('net_pnl', 0) for t in trades)
    total_costs = sum(
        t.get('fee', 0) + t.get('slippage', 0) + t.get('funding', 0) + t.get('short_cost', 0)
        for t in trades
    )

    metrics = {
        'n_trades': len(trades),
        'win_rate': win_rate(trades),
        'avg_trade': avg_trade(trades),
        'profit_factor': profit_factor(trades),
        'gross_return': gross_return / initial_capital if initial_capital > 0 else 0,
        'net_return': net_return / initial_capital if initial_capital > 0 else 0,
        'sharpe_ratio': sharpe_ratio(returns, periods_per_year),
        'sortino_ratio': sortino_ratio(returns, periods_per_year),
        'max_drawdown': max_drawdown(drawdown_df),
        'calmar_ratio': calmar_ratio(equity_clean, periods_per_year),
        'turnover': turnover(trades, equity_curve),
        'avg_holding_period': avg_holding_period(trades),
        'exposure': exposure(trades, equity_curve),
        'total_costs': total_costs,
        'total_fees': sum(t.get('fee', 0) for t in trades),
        'total_slippage': sum(t.get('slippage', 0) for t in trades),
        'total_funding': sum(t.get('funding', 0) for t in trades),
        'total_short_cost': sum(t.get('short_cost', 0) for t in trades),
    }

    return metrics

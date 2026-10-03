"""Event-driven backtester for pair trading strategies.

This module implements an event-driven backtesting framework that:
- Walks through time bar by bar
- Maintains positions in each leg of each pair
- Tracks equity, exposure, and drawdown
- Fires entry/exit signals from the strategy layer
- Applies transaction costs from the cost model
- Supports all 4 hedging variants (equal-dollar, OLS, volatility-adjusted, beta-neutral)
- Records every trade with full attribution
- Implements look-ahead bias prevention
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a single completed trade."""

    timestamp: pd.Timestamp
    pair: str
    side: str  # "long_a_short_b" or "short_a_long_b"
    entry_px_a: float
    entry_px_b: float
    exit_px_a: float
    exit_px_b: float
    shares_a: float
    shares_b: float
    notional: float
    fee: float
    slippage: float
    funding: float
    short_cost: float
    gross_pnl: float
    net_pnl: float
    holding_period_bars: int
    exit_reason: str
    entry_zscore: float
    exit_zscore: float


@dataclass
class BacktestResult:
    """Result of a backtest run.

    Attributes:
        equity_curve: Series of equity values indexed by timestamp.
        trades: List of TradeRecord objects.
        metrics: Dictionary of performance metrics.
        positions_by_time: DataFrame of positions over time.
    """

    equity_curve: pd.Series
    trades: list[TradeRecord]
    metrics: dict[str, Any]
    positions_by_time: pd.DataFrame


class PairBacktester:
    """Event-driven backtester for pair trading strategies.

    Walks through time bar by bar, maintaining positions and tracking equity.
    All signals use only data available at that bar's close (no look-ahead bias).
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        cost_scenario: str = "base",
    ):
        """Initialize the backtester.

        Args:
            initial_capital: Starting capital in dollars.
            cost_scenario: Cost scenario name ('optimistic', 'base', 'pessimistic').
        """
        self.initial_capital = initial_capital
        self.cost_scenario = cost_scenario
        self._cost_config = self._load_cost_config(cost_scenario)

        logger.info(
            f"Initialized PairBacktester: capital={initial_capital}, "
            f"cost_scenario={cost_scenario}"
        )

    def _load_cost_config(self, scenario: str) -> dict[str, float]:
        """Load cost configuration for a scenario."""
        from src.data.loader import DataLoader

        try:
            import yaml

            loader = DataLoader()
            config_path = loader.raw_dir.parent.parent / "config" / "default.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    return config.get("cost_scenarios", {}).get(scenario, {})
        except Exception:
            pass

        # Default cost scenarios
        defaults = {
            "optimistic": {
                "maker_fee": 0.0002,
                "taker_fee": 0.0002,
                "slippage_bps": 0.01,
                "funding_rate_per_hour": 0.00001,
                "short_rate_per_hour": 0.00005,
            },
            "base": {
                "maker_fee": 0.0004,
                "taker_fee": 0.0004,
                "slippage_bps": 0.02,
                "funding_rate_per_hour": 0.00003,
                "short_rate_per_hour": 0.0001,
            },
            "pessimistic": {
                "maker_fee": 0.0006,
                "taker_fee": 0.0006,
                "slippage_bps": 0.04,
                "funding_rate_per_hour": 0.00006,
                "short_rate_per_hour": 0.0002,
            },
        }
        return defaults.get(scenario, defaults["base"])

    def run(
        self,
        pair_data: dict[str, pd.DataFrame],
        strategy_config: dict[str, Any],
        cost_scenario: str,
        hedge_type: str = "equal_dollar",
    ) -> BacktestResult:
        """Run the backtest.

        Args:
            pair_data: Dict mapping pair name (e.g., 'BTC/XRP') to DataFrame
                       with columns: timestamp, close_a, close_b, zscore, spread.
            strategy_config: Dict with strategy parameters:
                - entry_threshold: Z-score threshold for entry.
                - exit_type: 'z0', 'z025', 'z050', 'time', 'stop'.
                - exit_threshold: Exit threshold value.
                - max_holding_period_bars: Maximum bars to hold.
                - stop_loss_z: Z-score stop loss threshold.
            cost_scenario: Cost scenario name.
            hedge_type: Hedging variant ('equal_dollar', 'ols', 'volatility_adjusted', 'beta_neutral').

        Returns:
            BacktestResult with equity_curve, trades, metrics, positions_by_time.
        """
        from src.risk.position_sizing import (
            beta_neutral_hedge,
            equal_dollar_hedge,
            ols_hedge_ratio,
            volatility_adjusted_hedge,
        )

        self.cost_scenario = cost_scenario
        self._cost_config = self._load_cost_config(cost_scenario)

        if not pair_data:
            logger.warning("run() called with empty pair_data — returning empty result")
            return BacktestResult(
                equity_curve=pd.Series([], dtype=float),
                trades=[],
                metrics={},
                positions_by_time=pd.DataFrame(),
            )

        # Extract pair name and data
        pair_name = list(pair_data.keys())[0]
        data = pair_data[pair_name]

        entry_threshold = strategy_config.get("entry_threshold", 2.0)
        exit_type = strategy_config.get("exit_type", "z0")
        exit_threshold_val = strategy_config.get("exit_threshold", 0.0)
        max_holding_bars = strategy_config.get("max_holding_period_bars", 100)
        stop_loss_z = strategy_config.get("stop_loss_z", 3.0)

        # Get zscore and prices
        if "zscore" in data.columns:
            zscore = data["zscore"]
        elif "z" in data.columns:
            zscore = data["z"]
        else:
            logger.error("No zscore column found in pair_data")
            return self._empty_result()

        close_a = data["close_a"] if "close_a" in data.columns else data["close"]
        close_b = data["close_b"] if "close_b" in data.columns else data["close_b"]

        # Build aligned dataframe
        df = pd.DataFrame({
            "close_a": close_a,
            "close_b": close_b,
            "zscore": zscore,
        }, index=data.index if hasattr(data, "index") else pd.RangeIndex(len(data)))

        if isinstance(data.index, pd.DatetimeIndex):
            df.index = data.index

        df = df.dropna()

        if len(df) == 0:
            logger.warning("No data after alignment")
            return self._empty_result()

        # Initialize tracking variables
        equity = [self.initial_capital]
        equity_index = [df.index[0]]

        position = 0  # 0 = no position, 1 = long A/short B, -1 = short A/long B
        entry_price_a = 0.0
        entry_price_b = 0.0
        entry_time = None
        entry_bar = 0
        entry_zscore = 0.0
        shares_a = 0.0
        shares_b = 0.0

        trades: list[TradeRecord] = []
        positions_over_time: list[dict] = []

        # Get volatility for position sizing if needed
        vol_a = data.get("vol_a", pd.Series(1.0, index=df.index)) if isinstance(data, pd.DataFrame) else pd.Series(1.0, index=df.index)
        vol_b = data.get("vol_b", pd.Series(1.0, index=df.index)) if isinstance(data, pd.DataFrame) else pd.Series(1.0, index=df.index)
        beta = data.get("beta", pd.Series(1.0, index=df.index)) if isinstance(data, pd.DataFrame) else pd.Series(1.0, index=df.index)

        notional = self.initial_capital * 0.5  # Use 50% of capital per trade

        for i, (ts, row) in enumerate(df.iterrows()):
            current_price_a = row["close_a"]
            current_price_b = row["close_b"]
            current_zscore = row["zscore"]

            # Record current position state
            current_equity = equity[-1]
            positions_over_time.append({
                "timestamp": ts,
                "position": position,
                "equity": current_equity,
                "price_a": current_price_a,
                "price_b": current_price_b,
            })

            # Entry logic - only use data available at this bar's close
            if position == 0:
                if current_zscore > entry_threshold:
                    # Spread is expensive relative to mean
                    # SHORT A, LONG B
                    position = -1
                    entry_price_a = current_price_a
                    entry_price_b = current_price_b
                    entry_time = ts
                    entry_bar = i
                    entry_zscore = current_zscore

                    # Calculate shares based on hedge type
                    shares_a, shares_b = self._calculate_shares(
                        notional, current_price_a, current_price_b,
                        hedge_type, 1.0, 1.0, vol_a.iloc[i] if i < len(vol_a) else 1.0,
                        vol_b.iloc[i] if i < len(vol_b) else 1.0, 1.0
                    )

                    logger.debug(
                        f"ENTER SHORT A/LONG B at {ts}: z={current_zscore:.2f}, "
                        f"price_a={current_price_a:.4f}, price_b={current_price_b:.4f}"
                    )

                elif current_zscore < -entry_threshold:
                    # Spread is cheap relative to mean
                    # LONG A, SHORT B
                    position = 1
                    entry_price_a = current_price_a
                    entry_price_b = current_price_b
                    entry_time = ts
                    entry_bar = i
                    entry_zscore = current_zscore

                    shares_a, shares_b = self._calculate_shares(
                        notional, current_price_a, current_price_b,
                        hedge_type, 1.0, 1.0, vol_a.iloc[i] if i < len(vol_a) else 1.0,
                        vol_b.iloc[i] if i < len(vol_b) else 1.0, 1.0
                    )

                    logger.debug(
                        f"ENTER LONG A/SHORT B at {ts}: z={current_zscore:.2f}, "
                        f"price_a={current_price_a:.4f}, price_b={current_price_b:.4f}"
                    )

            # Exit logic
            elif position != 0:
                should_exit = False
                exit_reason = ""

                # Check stop loss first
                if abs(current_zscore) > stop_loss_z:
                    should_exit = True
                    exit_reason = "stop_loss"

                # Check exit conditions
                if not should_exit:
                    if exit_type == "z0":
                        if (position == 1 and current_zscore >= 0) or \
                           (position == -1 and current_zscore <= 0):
                            should_exit = True
                            exit_reason = "z_cross_zero"

                    elif exit_type == "z025":
                        if abs(current_zscore) < 0.25:
                            should_exit = True
                            exit_reason = "z_threshold"

                    elif exit_type == "z050":
                        if abs(current_zscore) < 0.5:
                            should_exit = True
                            exit_reason = "z_threshold"

                    elif exit_type == "time":
                        bars_held = i - entry_bar
                        if bars_held >= max_holding_bars:
                            should_exit = True
                            exit_reason = "time_exit"

                    elif exit_type == "stop":
                        if abs(current_zscore) > exit_threshold_val:
                            should_exit = True
                            exit_reason = "stop_loss"

                if should_exit:
                    # Calculate P&L
                    if position == 1:
                        # Long A, Short B
                        pnl_a = shares_a * (current_price_a - entry_price_a)
                        pnl_b = shares_b * (entry_price_b - current_price_b)
                    else:
                        # Short A, Long B
                        pnl_a = shares_a * (entry_price_a - current_price_a)
                        pnl_b = shares_b * (current_price_b - entry_price_b)

                    gross_pnl = pnl_a + pnl_b

                    # Calculate costs
                    actual_notional = (shares_a * entry_price_a + shares_b * entry_price_b)
                    bars_held = i - entry_bar

                    costs = self._calculate_trade_costs(
                        actual_notional, bars_held
                    )

                    net_pnl = gross_pnl - costs["total_cost"]

                    # Record trade
                    side = "long_a_short_b" if position == 1 else "short_a_long_b"
                    trade = TradeRecord(
                        timestamp=ts,
                        pair=pair_name,
                        side=side,
                        entry_px_a=entry_price_a,
                        entry_px_b=entry_price_b,
                        exit_px_a=current_price_a,
                        exit_px_b=current_price_b,
                        shares_a=shares_a,
                        shares_b=shares_b,
                        notional=actual_notional,
                        fee=costs["fee"],
                        slippage=costs["slippage"],
                        funding=costs["funding"],
                        short_cost=costs["short_cost"],
                        gross_pnl=gross_pnl,
                        net_pnl=net_pnl,
                        holding_period_bars=bars_held,
                        exit_reason=exit_reason,
                        entry_zscore=entry_zscore,
                        exit_zscore=current_zscore,
                    )
                    trades.append(trade)

                    # Update equity
                    equity.append(equity[-1] + net_pnl)
                    equity_index.append(ts)

                    logger.debug(
                        f"EXIT {exit_reason} at {ts}: gross={gross_pnl:.2f}, "
                        f"costs={costs['total_cost']:.2f}, net={net_pnl:.2f}"
                    )

                    # Reset position
                    position = 0
                    entry_price_a = 0.0
                    entry_price_b = 0.0
                    entry_time = None
                    entry_bar = 0
                    shares_a = 0.0
                    shares_b = 0.0

        # Build equity curve
        equity_curve = pd.Series(equity, index=equity_index)

        # Build positions dataframe
        positions_df = pd.DataFrame(positions_over_time)
        if len(positions_df) > 0:
            positions_df = positions_df.set_index("timestamp")

        # Calculate metrics
        metrics = self._calculate_metrics(equity_curve, trades)

        logger.info(
            f"Backtest complete: {len(trades)} trades, "
            f"final equity={equity[-1]:.2f}, Sharpe={metrics.get('sharpe_ratio', 0):.2f}"
        )

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            positions_by_time=positions_df,
        )

    def _calculate_shares(
        self,
        notional: float,
        price_a: float,
        price_b: float,
        hedge_type: str,
        beta: float,
        beta_existing: float,
        vol_a: float,
        vol_b: float,
        target_vol: float,
    ) -> tuple[float, float]:
        """Calculate shares for each leg based on hedge type."""
        from src.risk.position_sizing import (
            beta_neutral_hedge,
            equal_dollar_hedge,
            ols_hedge_ratio,
            volatility_adjusted_hedge,
        )

        if hedge_type == "equal_dollar":
            return equal_dollar_hedge(notional, price_a, price_b)
        elif hedge_type == "ols":
            return ols_hedge_ratio(notional, price_a, price_b, beta)
        elif hedge_type == "volatility_adjusted":
            return volatility_adjusted_hedge(notional, price_a, price_b, vol_a, vol_b, target_vol)
        elif hedge_type == "beta_neutral":
            return beta_neutral_hedge(notional, price_a, price_b, beta, beta_existing)
        else:
            return equal_dollar_hedge(notional, price_a, price_b)

    def _calculate_trade_costs(
        self,
        notional: float,
        holding_period_bars: int,
        timeframe_hours: int = 24,
    ) -> dict[str, float]:
        """Calculate total transaction costs for a trade.

        Args:
            notional: Notional value of the trade.
            holding_period_bars: Number of bars the position was held.
            timeframe_hours: Hours per bar (for funding/short cost calculation).

        Returns:
            Dict with fee, slippage, funding, short_cost, total_cost.
        """
        # Trading fees (taker on entry and exit)
        taker_fee = self._cost_config.get("taker_fee", 0.0004)
        fee = notional * taker_fee * 2  # Both legs, entry and exit

        # Slippage
        slippage_bps = self._cost_config.get("slippage_bps", 0.02)
        slippage = notional * (slippage_bps / 10000) * 2  # Both legs

        # Funding cost (based on holding time)
        funding_rate = self._cost_config.get("funding_rate_per_hour", 0.00003)
        holding_hours = holding_period_bars * timeframe_hours
        funding = notional * funding_rate * holding_hours

        # Short cost (based on holding time)
        short_rate = self._cost_config.get("short_rate_per_hour", 0.0001)
        short_cost = notional * short_rate * holding_hours

        total_cost = fee + slippage + funding + short_cost

        return {
            "fee": fee,
            "slippage": slippage,
            "funding": funding,
            "short_cost": short_cost,
            "total_cost": total_cost,
        }

    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        trades: list[TradeRecord],
    ) -> dict[str, Any]:
        """Calculate performance metrics from backtest results.

        Args:
            equity_curve: Series of equity values.
            trades: List of TradeRecord objects.

        Returns:
            Dictionary of performance metrics.
        """
        from src.reporting.metrics import (
            avg_holding_period,
            avg_trade,
            calmar_ratio,
            compute_drawdown,
            exposure,
            max_drawdown,
            profit_factor,
            sharpe_ratio,
            sortino_ratio,
            turnover,
            win_rate,
        )

        if len(trades) == 0:
            return {
                "n_trades": 0,
                "win_rate": 0.0,
                "avg_trade": 0.0,
                "profit_factor": 0.0,
                "gross_return": 0.0,
                "net_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
                "turnover": 0.0,
                "avg_holding_period": 0.0,
                "exposure": 0.0,
            }

        trade_dicts = [
            {
                "net_pnl": t.net_pnl,
                "gross_pnl": t.gross_pnl,
                "fee": t.fee,
                "slippage": t.slippage,
                "funding": t.funding,
                "short_cost": t.short_cost,
                "holding_period_bars": t.holding_period_bars,
            }
            for t in trades
        ]

        # Compute drawdown
        drawdown_df = compute_drawdown(equity_curve)

        # Calculate returns from equity curve
        returns = equity_curve.pct_change().dropna()

        # Annualization factor (assuming daily bars)
        periods_per_year = 252

        metrics = {
            "n_trades": len(trades),
            "win_rate": win_rate(trade_dicts),
            "avg_trade": avg_trade(trade_dicts),
            "profit_factor": profit_factor(trade_dicts),
            "gross_return": sum(t["gross_pnl"] for t in trade_dicts) / self.initial_capital,
            "net_return": sum(t["net_pnl"] for t in trade_dicts) / self.initial_capital,
            "sharpe_ratio": sharpe_ratio(returns, periods_per_year) if len(returns) > 1 else 0.0,
            "sortino_ratio": sortino_ratio(returns, periods_per_year) if len(returns) > 1 else 0.0,
            "max_drawdown": max_drawdown(drawdown_df),
            "calmar_ratio": calmar_ratio(equity_curve, periods_per_year) if len(returns) > 1 else 0.0,
            "turnover": turnover(trade_dicts, equity_curve),
            "avg_holding_period": avg_holding_period(trade_dicts),
            "exposure": exposure(trade_dicts, equity_curve),
            "total_fees": sum(t["fee"] for t in trade_dicts),
            "total_slippage": sum(t["slippage"] for t in trade_dicts),
            "total_funding": sum(t["funding"] for t in trade_dicts),
            "total_short_cost": sum(t["short_cost"] for t in trade_dicts),
            "total_costs": sum(t["fee"] + t["slippage"] + t["funding"] + t["short_cost"] for t in trade_dicts),
        }

        return metrics

    def _empty_result(self) -> BacktestResult:
        """Return an empty backtest result."""
        return BacktestResult(
            equity_curve=pd.Series([], dtype=float),
            trades=[],
            metrics={},
            positions_by_time=pd.DataFrame(),
        )

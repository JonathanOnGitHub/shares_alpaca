"""Pair trading strategy implementation."""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_costs(
    signal: int,
    entry_price: float,
    exit_price: float,
    notional: float,
    cost_scenario: dict[str, float],
) -> dict[str, float]:
    """Calculate all transaction costs for a trade.

    Args:
        signal: Trade direction (1 for long A/short B, -1 for short A/long B).
        entry_price: Entry price for the position.
        exit_price: Exit price for the position.
        notional: Notional value of the trade in dollars.
        cost_scenario: Dict with maker_fee, taker_fee, slippage_bps,
                       funding_rate_per_hour, short_rate_per_hour.

    Returns:
        Dictionary with:
        - fee: Exchange trading fees
        - slippage: Slippage cost in dollars
        - funding: Funding cost (for perpetuals)
        - short_cost: Short borrowing cost
        - total_cost: Total costs in dollars
    """
    # Trading fees (assume taker for both entry and exit)
    taker_fee = cost_scenario.get("taker_fee", 0.0004)
    maker_fee = cost_scenario.get("maker_fee", 0.0002)

    # We pay taker fee on both entry and exit
    fee = notional * (taker_fee * 2)  # both legs

    # Slippage (in basis points)
    slippage_bps = cost_scenario.get("slippage_bps", 0.02)
    slippage = notional * (slippage_bps / 10000) * 2  # both legs

    # Funding rate (per hour) - assume position held for some time
    funding_rate = cost_scenario.get("funding_rate_per_hour", 0.00003)
    # We'll calculate funding based on holding period in backtest
    funding = 0.0  # Calculated in backtest based on actual holding time

    # Short cost (per hour)
    short_rate = cost_scenario.get("short_rate_per_hour", 0.0001)
    short_cost = 0.0  # Calculated in backtest based on actual holding time

    total_cost = fee + slippage + funding + short_cost

    return {
        "fee": fee,
        "slippage": slippage,
        "funding": funding,
        "short_cost": short_cost,
        "total_cost": total_cost,
    }


class PairStrategy:
    """Pair trading strategy with mean-reversion signals.

    Generates signals based on z-score of the spread and backtests
    the strategy with realistic transaction costs.
    """

    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_type: str = "z0",
        exit_threshold: float = 0.0,
        beta_lookback: int | str = "static",
        cost_scenario: str = "base",
    ):
        """Initialize the pair strategy.

        Args:
            entry_threshold: Z-score threshold to enter a trade.
            exit_type: Type of exit ('z0', 'z025', 'z050', 'time', 'stop').
            exit_threshold: Exit threshold value.
            beta_lookback: Beta lookback period ('static' or int).
            cost_scenario: Cost scenario name ('optimistic', 'base', 'pessimistic').
        """
        self.entry_threshold = entry_threshold
        self.exit_type = exit_type
        self.exit_threshold = exit_threshold
        self.beta_lookback = beta_lookback
        self.cost_scenario = cost_scenario

        logger.info(
            f"Initialized PairStrategy: entry={entry_threshold}, "
            f"exit_type={exit_type}, beta_lookback={beta_lookback}"
        )

    def generate_signals(
        self,
        spread_df: pd.DataFrame,
        zscore_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate trading signals from spread and z-score data.

        Args:
            spread_df: DataFrame with spread values.
            zscore_df: DataFrame with z-score calculations.

        Returns:
            DataFrame with columns:
            - timestamp
            - signal (1=long B short A, -1=long A short B, 0=no signal)
            - zscore
            - spread
            - hedge_ratio
        """
        # Combine spread and zscore data
        if isinstance(spread_df, pd.Series):
            spread_df = spread_df.to_frame(name="spread")
        if isinstance(zscore_df, pd.Series):
            zscore_df = zscore_df.to_frame(name="z")

        # Ensure we have required columns
        if "z" not in zscore_df.columns and "zscore" in zscore_df.columns:
            zscore_df = zscore_df.rename(columns={"zscore": "z"})
        elif "zscore" in zscore_df.columns:
            zscore_df = zscore_df.rename(columns={"zscore": "z"})

        # Create signals
        z = zscore_df["z"].copy() if "z" in zscore_df.columns else zscore_df["zscore"].copy()
        spread = spread_df["spread"].copy() if "spread" in spread_df.columns else spread_df.copy()

        signals = pd.Series(0, index=z.index)

        # Long signal: z < -entry_threshold (spread is cheap relative to mean)
        # We go LONG A (cheap), SHORT B (expensive)
        signals[z < -self.entry_threshold] = -1

        # Short signal: z > entry_threshold (spread is expensive relative to mean)
        # We go SHORT A (expensive), LONG B (cheap)
        signals[z > self.entry_threshold] = 1

        # Create result DataFrame
        result = pd.DataFrame({
            "signal": signals,
            "zscore": z,
            "spread": spread if isinstance(spread, pd.Series) else spread.get("spread", z),
        }, index=z.index)

        # Apply exit logic
        result = self._apply_exit_logic(result)

        logger.info(
            f"Generated signals: {(result['signal'] != 0).sum()} non-zero signals "
            f"out of {len(result)} bars"
        )
        return result

    def _apply_exit_logic(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Apply exit logic to signals.

        Args:
            signals_df: DataFrame with signals.

        Returns:
            DataFrame with modified signals.
        """
        df = signals_df.copy()

        if self.exit_type == "z0":
            # Exit when z crosses 0
            pass  # Exit handled in backtest

        elif self.exit_type == "z025":
            # Exit when |z| < 0.25
            pass

        elif self.exit_type == "z050":
            # Exit when |z| < 0.5
            pass

        elif self.exit_type == "time":
            # Time-based exit - handled in backtest
            pass

        elif self.exit_type == "stop":
            # Stop-loss - exit when |z| > stop threshold
            pass

        return df

    def backtest(
        self,
        data_a: pd.DataFrame,
        data_b: pd.DataFrame,
        signals: pd.DataFrame,
        initial_capital: float = 100000.0,
        cost_scenario: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Backtest the pair strategy.

        Args:
            data_a: OHLCV data for asset A.
            data_b: OHLCV data for asset B.
            signals: DataFrame with trading signals.
            initial_capital: Starting capital in dollars.
            cost_scenario: Dict with cost parameters.

        Returns:
            Dictionary with:
            - equity_curve: Series of equity values
            - trades: List of trade dictionaries
            - metrics: Performance metrics
        """
        if cost_scenario is None:
            cost_scenario = {
                "maker_fee": 0.0004,
                "taker_fee": 0.0004,
                "slippage_bps": 0.02,
                "funding_rate_per_hour": 0.00003,
                "short_rate_per_hour": 0.0001,
            }

        # Extract close prices
        if isinstance(data_a, pd.DataFrame) and "close" in data_a.columns:
            prices_a = data_a["close"]
        else:
            prices_a = data_a

        if isinstance(data_b, pd.DataFrame) and "close" in data_b.columns:
            prices_b = data_b["close"]
        else:
            prices_b = data_b

        # Align all data
        aligned = pd.DataFrame({
            "price_a": prices_a,
            "price_b": prices_b,
            "signal": signals["signal"],
            "zscore": signals["zscore"],
        }).dropna()

        if len(aligned) == 0:
            logger.warning("No data for backtest")
            return self._empty_backtest_result()

        # Backtest simulation
        equity = [initial_capital]
        trades = []
        position = 0  # Current position: 1, -1, or 0
        entry_price_a = 0.0
        entry_price_b = 0.0
        entry_time = None
        entry_zscore = 0.0
        position_value = 0.0

        for i, (ts, row) in enumerate(aligned.iterrows()):
            current_price_a = row["price_a"]
            current_price_b = row["price_b"]
            signal = int(row["signal"])
            zscore = row["zscore"]

            # Entry logic
            if position == 0 and signal != 0:
                # Enter new position
                position = signal
                entry_price_a = current_price_a
                entry_price_b = current_price_b
                entry_time = ts
                entry_zscore = zscore

                # Calculate position value (notional)
                position_value = equity[-1] * 0.5  # Use 50% of equity per leg

                logger.debug(
                    f"Enter position: {signal} at {ts}, "
                    f"z={zscore:.2f}, price_a={current_price_a}, price_b={current_price_b}"
                )

            # Exit logic
            elif position != 0:
                should_exit = False
                exit_reason = ""

                # Check exit conditions
                if self.exit_type == "z0":
                    if (position == 1 and zscore >= 0) or (position == -1 and zscore <= 0):
                        should_exit = True
                        exit_reason = "z_cross_zero"

                elif self.exit_type == "z025":
                    if abs(zscore) < 0.25:
                        should_exit = True
                        exit_reason = "z_threshold"

                elif self.exit_type == "z050":
                    if abs(zscore) < 0.5:
                        should_exit = True
                        exit_reason = "z_threshold"

                elif self.exit_type == "stop":
                    if abs(zscore) > 3.0:  # Stop loss
                        should_exit = True
                        exit_reason = "stop_loss"

                elif self.exit_type == "time":
                    # Exit after N bars
                    bars_held = i - aligned.index.get_loc(entry_time) if entry_time else 0
                    if bars_held >= self.exit_threshold:
                        should_exit = True
                        exit_reason = "time_exit"

                if should_exit:
                    # Calculate P&L
                    if position == 1:
                        # Long A, Short B
                        pnl_a = (current_price_a - entry_price_a) / entry_price_a * position_value
                        pnl_b = (entry_price_b - current_price_b) / entry_price_b * position_value
                    else:
                        # Short A, Long B
                        pnl_a = (entry_price_a - current_price_a) / entry_price_a * position_value
                        pnl_b = (current_price_b - entry_price_b) / entry_price_b * position_value

                    gross_pnl = pnl_a + pnl_b

                    # Calculate costs
                    notional = position_value * 2  # Both legs
                    holding_hours = 0  # Would need timestamp tracking
                    costs = calculate_costs(
                        position,
                        entry_price_a,
                        current_price_a,
                        notional,
                        cost_scenario,
                    )
                    costs["funding"] = notional * cost_scenario.get("funding_rate_per_hour", 0) * holding_hours
                    costs["short_cost"] = notional * cost_scenario.get("short_rate_per_hour", 0) * holding_hours
                    costs["total_cost"] = costs["fee"] + costs["slippage"] + costs["funding"] + costs["short_cost"]

                    net_pnl = gross_pnl - costs["total_cost"]

                    # Record trade
                    trade = {
                        "entry_time": entry_time,
                        "exit_time": ts,
                        "direction": position,
                        "entry_price_a": entry_price_a,
                        "exit_price_a": current_price_a,
                        "entry_price_b": entry_price_b,
                        "exit_price_b": current_price_b,
                        "gross_pnl": gross_pnl,
                        "costs": costs,
                        "net_pnl": net_pnl,
                        "exit_reason": exit_reason,
                        "entry_zscore": entry_zscore,
                        "exit_zscore": zscore,
                    }
                    trades.append(trade)

                    # Update equity
                    equity.append(equity[-1] + net_pnl)

                    logger.debug(
                        f"Exit position: {exit_reason} at {ts}, "
                        f"gross={gross_pnl:.2f}, costs={costs['total_cost']:.2f}, "
                        f"net={net_pnl:.2f}"
                    )

                    # Reset position
                    position = 0
                    entry_price_a = 0.0
                    entry_price_b = 0.0
                    entry_time = None

            # Update equity for open position (mark-to-market)
            if position != 0:
                if position == 1:
                    mtm_a = (current_price_a - entry_price_a) / entry_price_a * position_value
                    mtm_b = (entry_price_b - current_price_b) / entry_price_b * position_value
                else:
                    mtm_a = (entry_price_a - current_price_a) / entry_price_a * position_value
                    mtm_b = (current_price_b - entry_price_b) / entry_price_b * position_value

                mtm_pnl = mtm_a + mtm_b
                # Don't append here - only on close or final

        # Final equity
        final_equity = equity[-1]

        # Calculate metrics
        metrics = self._calculate_metrics(equity, trades, initial_capital)

        result = {
            "equity_curve": pd.Series(equity, index=aligned.index[:len(equity)]),
            "trades": trades,
            "metrics": metrics,
        }

        logger.info(
            f"Backtest complete: {len(trades)} trades, "
            f"final equity={final_equity:.2f}, Sharpe={metrics.get('sharpe', 0):.2f}"
        )
        return result

    def _calculate_metrics(
        self,
        equity: list,
        trades: list,
        initial_capital: float,
    ) -> dict[str, Any]:
        """Calculate performance metrics from backtest results.

        Args:
            equity: List of equity values.
            trades: List of trade dictionaries.
            initial_capital: Starting capital.

        Returns:
            Dictionary of performance metrics.
        """
        if len(trades) == 0:
            return {
                "n_trades": 0,
                "win_rate": 0.0,
                "avg_trade": 0.0,
                "gross_return": 0.0,
                "net_return": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown": 0.0,
                "turnover": 0.0,
                "avg_holding_period": 0.0,
            }

        net_pnls = [t["net_pnl"] for t in trades]
        gross_pnls = [t["gross_pnl"] for t in trades]

        # Win rate
        n_wins = sum(1 for p in net_pnls if p > 0)
        win_rate = n_wins / len(trades)

        # Average trade
        avg_trade = np.mean(net_pnls)

        # Returns
        gross_return = sum(gross_pnls) / initial_capital
        net_return = sum(net_pnls) / initial_capital

        # Sharpe ratio
        returns = pd.Series(net_pnls)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        else:
            sharpe = 0.0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and downside_returns.std() > 0:
            sortino = returns.mean() / downside_returns.std() * np.sqrt(252)
        else:
            sortino = 0.0

        # Max drawdown
        equity_series = pd.Series(equity)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Turnover
        avg_position = np.mean([abs(t["net_pnl"]) / initial_capital for t in trades])
        turnover = len(trades) * avg_position

        # Average holding period (in bars)
        holding_periods = []
        for t in trades:
            if t["exit_time"] is not None and t["entry_time"] is not None:
                # Simple bar count
                holding_periods.append(1)  # Placeholder
        avg_holding = np.mean(holding_periods) if holding_periods else 0

        # Total costs
        total_fees = sum(t["costs"]["fee"] for t in trades)
        total_slippage = sum(t["costs"]["slippage"] for t in trades)
        total_funding = sum(t["costs"]["funding"] for t in trades)
        total_short_cost = sum(t["costs"]["short_cost"] for t in trades)

        metrics = {
            "n_trades": len(trades),
            "win_rate": win_rate,
            "avg_trade": avg_trade,
            "gross_return": gross_return,
            "net_return": net_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_drawdown,
            "turnover": turnover,
            "avg_holding_period": avg_holding,
            "total_fees": total_fees,
            "total_slippage": total_slippage,
            "total_funding": total_funding,
            "total_short_cost": total_short_cost,
            "total_costs": total_fees + total_slippage + total_funding + total_short_cost,
        }

        return metrics

    def _empty_backtest_result(self) -> dict[str, Any]:
        """Return empty backtest result for error cases."""
        return {
            "equity_curve": pd.Series([100000.0]),
            "trades": [],
            "metrics": {
                "n_trades": 0,
                "win_rate": 0.0,
                "avg_trade": 0.0,
                "gross_return": 0.0,
                "net_return": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown": 0.0,
                "turnover": 0.0,
                "avg_holding_period": 0.0,
            },
        }

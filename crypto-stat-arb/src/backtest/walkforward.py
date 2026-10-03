"""Walk-forward validation framework for pair trading strategies.

This module implements walk-forward analysis that:
- Splits data into train/test windows per config
- Estimates parameters (hedge ratios, means, stds, cointegration) in train window only
- Applies train-window parameters to test window
- Rolls forward by roll_days days
- Aggregates results across all test windows
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    """Result of a single walk-forward window."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: dict[str, Any]
    backtest_result: Any  # BacktestResult
    metrics: dict[str, Any]
    train_cointegration_pvalue: Optional[float] = None
    train_halflife: Optional[float] = None


@dataclass
class WalkForwardResult:
    """Aggregated results from walk-forward validation.

    Attributes:
        window_results: List of WindowResult for each train/test window.
        aggregate_metrics: Aggregated metrics across all windows.
        experiment_metadata: Metadata about the experiment configuration.
    """

    window_results: list[WindowResult]
    aggregate_metrics: dict[str, Any]
    experiment_metadata: dict[str, Any]


class WalkForwardValidator:
    """Walk-forward validation framework for pair trading strategies.

    Splits data into train/test windows and validates strategy parameters
    on out-of-sample test periods.
    """

    def __init__(
        self,
        train_days: int = 180,
        test_days: int = 30,
        roll_days: int = 30,
        min_train_days: int = 60,
        min_test_days: int = 10,
    ):
        """Initialize the walk-forward validator.

        Args:
            train_days: Number of days in each training window.
            test_days: Number of days in each test window.
            roll_days: Number of days to roll forward between windows.
            min_train_days: Minimum training window size.
            min_test_days: Minimum test window size.
        """
        self.train_days = train_days
        self.test_days = test_days
        self.roll_days = roll_days
        self.min_train_days = min_train_days
        self.min_test_days = min_test_days

        logger.info(
            f"Initialized WalkForwardValidator: train={train_days}d, "
            f"test={test_days}d, roll={roll_days}d"
        )

    def run(
        self,
        full_data: dict[str, pd.DataFrame],
        strategy_params: dict[str, Any],
    ) -> WalkForwardResult:
        """Run walk-forward validation.

        Args:
            full_data: Dict mapping pair name to DataFrame with OHLCV data.
            strategy_params: Dict with strategy parameters:
                - entry_thresholds: List of entry threshold values to test.
                - exit_types: List of exit type values to test.
                - beta_lookbacks: List of beta lookback values.
                - hedge_type: Hedging variant.
                - cost_scenario: Cost scenario name.
                - zscore_lookback: Lookback for z-score calculation.

        Returns:
            WalkForwardResult with per-window results and aggregate stats.
        """
        from src.backtest.engine import PairBacktester
        from src.statistics.cointegration import engle_granger_test
        from src.statistics.halflife import estimate_halflife
        from src.statistics.pairwise import (
            compute_hedge_ratio_ols,
            compute_log_prices,
            compute_spread,
            static_beta,
        )
        from src.statistics.zscore import compute_zscore

        pair_name = list(full_data.keys())[0]
        data = full_data[pair_name]

        if not isinstance(data.index, pd.DatetimeIndex):
            logger.error("Data must have DatetimeIndex for walk-forward validation")
            return WalkForwardResult(
                window_results=[],
                aggregate_metrics={},
                experiment_metadata={},
            )

        # Sort by index
        data = data.sort_index()

        # Strategy parameters
        entry_thresholds = strategy_params.get("entry_thresholds", [2.0])
        exit_types = strategy_params.get("exit_types", ["z0"])
        beta_lookbacks = strategy_params.get("beta_lookbacks", ["static"])
        hedge_type = strategy_params.get("hedge_type", "equal_dollar")
        cost_scenario = strategy_params.get("cost_scenario", "base")
        zscore_lookback = strategy_params.get("zscore_lookback", 30)

        # Generate walk-forward windows
        windows = self._generate_windows(data)

        if len(windows) == 0:
            logger.warning("No valid walk-forward windows generated")
            return WalkForwardResult(
                window_results=[],
                aggregate_metrics={},
                experiment_metadata={},
            )

        logger.info(f"Generated {len(windows)} walk-forward windows")

        window_results: list[WindowResult] = []
        all_metrics: list[dict[str, Any]] = []

        for window_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(
                f"Window {window_idx + 1}/{len(windows)}: "
                f"train={train_start.date()} to {train_end.date()}, "
                f"test={test_start.date()} to {test_end.date()}"
            )

            # Split data
            train_data = data[(data.index >= train_start) & (data.index <= train_end)].copy()
            test_data = data[(data.index >= test_start) & (data.index <= test_end)].copy()

            if len(train_data) < self.min_train_days or len(test_data) < self.min_test_days:
                logger.warning(f"Window {window_idx + 1}: insufficient data, skipping")
                continue

            # Estimate parameters on training window only
            # Calculate cointegration on train
            log_prices_train = compute_log_prices(train_data, train_data)
            if len(log_prices_train) > 20:
                log_a = log_prices_train["log_price_a"]
                log_b = log_prices_train["log_price_b"]
                eg_result = engle_granger_test(log_a, log_b)
                cointegration_pvalue = eg_result.get("p_value")
                train_halflife = None

                # Calculate half-life on train spread
                hedge_result = compute_hedge_ratio_ols(log_a, log_b)
                beta = hedge_result.get("beta", 1.0)
                spread = compute_spread(log_a, log_b, beta)
                spread_returns = spread.diff().dropna()
                if len(spread_returns) > 10:
                    train_halflife = estimate_halflife(spread_returns)
            else:
                cointegration_pvalue = None
                train_halflife = None
                beta = 1.0

            # Select best parameters based on training window
            best_params = self._select_best_params(
                train_data, entry_thresholds, exit_types, beta_lookbacks, zscore_lookback
            )

            # Prepare test data with train-estimated parameters
            test_pair_data = self._prepare_test_data(
                train_data, test_data, best_params, zscore_lookback
            )

            # Run backtest on test window
            backtester = PairBacktester(
                initial_capital=strategy_params.get("initial_capital", 100000.0),
                cost_scenario=cost_scenario,
            )

            strategy_config = {
                "entry_threshold": best_params["entry_threshold"],
                "exit_type": best_params["exit_type"],
                "exit_threshold": best_params.get("exit_threshold", 0.0),
                "max_holding_period_bars": strategy_params.get("max_holding_period_bars", 100),
                "stop_loss_z": strategy_params.get("stop_loss_z", 3.0),
            }

            backtest_result = backtester.run(
                test_pair_data,
                strategy_config,
                cost_scenario,
                hedge_type,
            )

            # Record window result
            window_result = WindowResult(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                params=best_params,
                backtest_result=backtest_result,
                metrics=backtest_result.metrics,
                train_cointegration_pvalue=cointegration_pvalue,
                train_halflife=train_halflife,
            )
            window_results.append(window_result)
            all_metrics.append(backtest_result.metrics)

        # Aggregate results
        aggregate_metrics = self._aggregate_metrics(all_metrics)

        # Build experiment metadata
        experiment_metadata = {
            "train_days": self.train_days,
            "test_days": self.test_days,
            "roll_days": self.roll_days,
            "n_windows": len(window_results),
            "strategy_params": strategy_params,
            "pair": pair_name,
            "data_start": str(data.index.min()),
            "data_end": str(data.index.max()),
            "random_seed": strategy_params.get("random_seed", 42),
        }

        # Try to add git commit
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )
            if result.returncode == 0:
                experiment_metadata["git_commit"] = result.stdout.strip()
        except Exception:
            pass

        logger.info(
            f"Walk-forward complete: {len(window_results)} windows, "
            f"avg Sharpe={aggregate_metrics.get('avg_sharpe_ratio', 0):.2f}"
        )

        return WalkForwardResult(
            window_results=window_results,
            aggregate_metrics=aggregate_metrics,
            experiment_metadata=experiment_metadata,
        )

    def _generate_windows(
        self,
        data: pd.DataFrame,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate train/test window boundaries.

        Args:
            data: Full DataFrame with DatetimeIndex.

        Returns:
            List of (train_start, train_end, test_start, test_end) tuples.
        """
        windows = []
        current_date = data.index.min()

        # Ensure we have enough data for first window
        min_start_date = current_date
        max_date = data.index.max()

        # Calculate minimum span needed
        min_span_days = self.train_days + self.test_days

        # Find first valid window start
        date_range = data.index

        # Get date indices
        date_indices = pd.DatetimeIndex(data.index)

        # First window: train starts at min date, test starts at train + train_days
        while True:
            train_start = date_indices.min()
            train_end = train_start + pd.Timedelta(days=self.train_days)

            # Check if we have enough data for train window
            train_data = data[(data.index >= train_start) & (data.index <= train_end)]
            if len(train_data) < self.min_train_days:
                # Move to next available date
                next_date = date_indices[date_indices > train_start][0]
                date_indices = date_indices[date_indices >= next_date]
                if len(date_indices) == 0:
                    break
                continue

            test_start = train_end
            test_end = test_start + pd.Timedelta(days=self.test_days)

            # Check if test end is within data range
            if test_end > max_date:
                break

            windows.append((train_start, train_end, test_start, test_end))

            # Roll forward
            roll_start = train_start + pd.Timedelta(days=self.roll_days)
            if roll_start >= max_date:
                break

            # Filter to dates >= roll_start
            date_indices = date_indices[date_indices >= roll_start]
            if len(date_indices) == 0:
                break

        return windows

    def _select_best_params(
        self,
        train_data: pd.DataFrame,
        entry_thresholds: list[float],
        exit_types: list[str],
        beta_lookbacks: list[Any],
        zscore_lookback: int,
    ) -> dict[str, Any]:
        """Select best parameters using only training data.

        This is the key method that implements parameter selection
        exclusively from the training window.

        Args:
            train_data: Training period DataFrame.
            entry_thresholds: List of entry thresholds to test.
            exit_types: List of exit types to test.
            beta_lookbacks: List of beta lookback values.
            zscore_lookback: Z-score lookback period.

        Returns:
            Dict with best parameters.
        """
        # Use default values if training parameter selection not implemented
        # In a full implementation, this would iterate through all combinations
        # and select the best based on training performance

        return {
            "entry_threshold": entry_thresholds[0] if entry_thresholds else 2.0,
            "exit_type": exit_types[0] if exit_types else "z0",
            "exit_threshold": 0.0,
            "beta_lookback": beta_lookbacks[0] if beta_lookbacks else "static",
            "zscore_lookback": zscore_lookback,
        }

    def _prepare_test_data(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        params: dict[str, Any],
        zscore_lookback: int,
    ) -> dict[str, pd.DataFrame]:
        """Prepare test data using parameters estimated from training.

        Args:
            train_data: Training period DataFrame.
            test_data: Test period DataFrame.
            params: Parameters from training.
            zscore_lookback: Z-score lookback.

        Returns:
            Dict with pair data prepared for backtesting.
        """
        from src.statistics.pairwise import compute_log_prices, compute_spread, static_beta
        from src.statistics.zscore import compute_zscore

        # Combine train and test for continuous calculation
        combined = pd.concat([train_data, test_data])

        # Calculate log prices
        log_prices = compute_log_prices(combined, combined)

        # Calculate static beta from training period only
        train_log_prices = compute_log_prices(train_data, train_data)
        beta = static_beta(train_log_prices["log_price_a"], train_log_prices["log_price_b"])

        # Calculate spread with train-estimated beta
        log_a = log_prices["log_price_a"]
        log_b = log_prices["log_price_b"]
        spread = compute_spread(log_a, log_b, beta)

        # Calculate z-score
        zscore_result = compute_zscore(spread, zscore_lookback)

        # Build result dataframe aligned with test period
        result_df = pd.DataFrame({
            "close_a": combined["close"] if "close" in combined.columns else combined["close_a"],
            "close_b": combined["close_b"] if "close_b" in combined.columns else combined["close"],
            "zscore": zscore_result["z"],
            "spread": spread,
            "beta": beta,
        }, index=combined.index)

        return {f"{train_data.name}/{test_data.name}" if hasattr(train_data, 'name') else "PAIR": result_df}

    def save_results(
        self,
        result: WalkForwardResult,
        output_path: str | Path,
    ) -> None:
        """Save walk-forward results to JSON.

        Args:
            result: WalkForwardResult to save.
            output_path: Path to save JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize window results
        windows_serializable = []
        for w in result.window_results:
            windows_serializable.append({
                "train_start": str(w.train_start),
                "train_end": str(w.train_end),
                "test_start": str(w.test_start),
                "test_end": str(w.test_end),
                "params": w.params,
                "metrics": w.metrics,
                "train_cointegration_pvalue": w.train_cointegration_pvalue,
                "train_halflife": w.train_halflife,
            })

        output = {
            "experiment_metadata": result.experiment_metadata,
            "aggregate_metrics": result.aggregate_metrics,
            "window_results": windows_serializable,
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Saved walk-forward results to {output_path}")

    def _aggregate_metrics(
        self,
        all_metrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate metrics across all walk-forward windows.

        Args:
            all_metrics: List of metrics dicts from each window.

        Returns:
            Dict of aggregated metrics.
        """
        if not all_metrics:
            return {}

        n = len(all_metrics)

        # Aggregate numeric metrics
        aggregations = {}

        for key in ["n_trades", "win_rate", "avg_trade", "profit_factor",
                    "gross_return", "net_return", "sharpe_ratio", "sortino_ratio",
                    "max_drawdown", "calmar_ratio", "turnover", "avg_holding_period",
                    "exposure", "total_costs"]:
            values = [m.get(key, 0) for m in all_metrics if isinstance(m.get(key), (int, float))]
            if values:
                aggregations[f"avg_{key}"] = np.mean(values)
                aggregations[f"std_{key}"] = np.std(values)
                aggregations[f"min_{key}"] = np.min(values)
                aggregations[f"max_{key}"] = np.max(values)

        # Calculate consistency (percentage of windows with positive return)
        net_returns = [m.get("net_return", 0) for m in all_metrics]
        positive_count = sum(1 for r in net_returns if r > 0)
        aggregations["pct_windows_profitable"] = positive_count / n if n > 0 else 0

        # Average Sharpe ratio
        sharpes = [m.get("sharpe_ratio", 0) for m in all_metrics if isinstance(m.get("sharpe_ratio"), (int, float))]
        aggregations["avg_sharpe_ratio"] = np.mean(sharpes) if sharpes else 0
        aggregations["sharpe_consistency"] = sum(1 for s in sharpes if s > 0) / len(sharpes) if sharpes else 0

        return aggregations

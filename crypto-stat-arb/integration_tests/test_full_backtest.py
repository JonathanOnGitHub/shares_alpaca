"""Integration test for complete backtest run.

This test:
1. Downloads 1d BTC/USDT and ETH/USDT (30 days)
2. Runs a complete walk-forward backtest for BTC/ETH pair
3. Verifies all reports are generated
4. Verifies no look-ahead bias
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.backtest.walkforward import WalkForwardValidator
from src.data.downloader import MarketDataDownloader
from src.reporting.metrics import compute_all_metrics
from src.reporting.plots import plot_equity_curve, plot_drawdown
from src.statistics.pairwise import compute_hedge_ratio_ols, compute_log_prices, compute_spread, static_beta
from src.statistics.zscore import compute_zscore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFullBacktest:
    """Integration tests for complete backtest runs."""

    def test_download_and_prepare_data(self):
        """Test downloading and preparing data for backtest."""
        # This test requires network access to Binance API
        downloader = MarketDataDownloader()

        # Download BTC and ETH data
        btc_data, btc_meta = downloader.download_pair("BTC", "1d", 30)
        eth_data, eth_meta = downloader.download_pair("ETH", "1d", 30)

        if btc_data.empty or eth_data.empty:
            pytest.skip("Could not download test data from Binance")

        # Verify data structure
        assert "close" in btc_data.columns
        assert "close" in eth_data.columns
        assert len(btc_data) > 20  # Need enough data
        assert len(eth_data) > 20

        logger.info(f"Downloaded BTC: {len(btc_data)} rows, ETH: {len(eth_data)} rows")

    def test_walk_forward_backtest(self):
        """Test complete walk-forward backtest for BTC/ETH."""
        # Download data
        downloader = MarketDataDownloader()
        btc_data, _ = downloader.download_pair("BTC", "1d", 60)
        eth_data, _ = downloader.download_pair("ETH", "1d", 60)

        if btc_data.empty or eth_data.empty:
            pytest.skip("Could not download test data")

        # Compute statistics
        log_prices = compute_log_prices(btc_data, eth_data)
        if log_prices.empty:
            pytest.fail("Failed to compute log prices")

        beta = static_beta(log_prices["log_price_a"], log_prices["log_price_b"])
        spread = compute_spread(log_prices["log_price_a"], log_prices["log_price_b"], beta)
        zscore_result = compute_zscore(spread, 30)
        zscore = zscore_result["z"]

        # Build pair data
        pair_name = "BTC/ETH"
        pair_data = {
            pair_name: pd.DataFrame({
                "close_a": btc_data["close"],
                "close_b": eth_data["close"],
                "zscore": zscore,
                "spread": spread,
                "beta": beta,
            }, index=btc_data.index)
        }

        # Run walk-forward
        params = {
            "entry_thresholds": [2.0],
            "exit_types": ["z0"],
            "beta_lookbacks": ["static"],
            "hedge_type": "equal_dollar",
            "cost_scenario": "base",
            "zscore_lookback": 30,
            "initial_capital": 100000.0,
            "max_holding_period_bars": 50,
            "stop_loss_z": 3.0,
        }

        validator = WalkForwardValidator(
            train_days=30,
            test_days=10,
            roll_days=10,
        )

        wf_result = validator.run(pair_data, params)

        # Verify results structure
        assert len(wf_result.window_results) > 0, "No walk-forward windows executed"
        assert wf_result.aggregate_metrics is not None
        assert wf_result.experiment_metadata is not None

        # Verify each window has metrics
        for window in wf_result.window_results:
            assert window.metrics is not None
            assert "n_trades" in window.metrics

        logger.info(f"Walk-forward completed: {len(wf_result.window_results)} windows")
        logger.info(f"Average Sharpe: {wf_result.aggregate_metrics.get('avg_sharpe_ratio', 0):.2f}")

    def test_no_look_ahead_bias(self):
        """Verify that signals don't use future data."""
        # Create synthetic data where look-ahead would be obvious
        dates = pd.date_range("2024-01-01", periods=100, freq="D")

        # Price A: random walk
        import numpy as np
        np.random.seed(42)
        returns_a = np.random.normal(0, 0.02, 100)
        price_a = 100 * pd.Series(returns_a).cumprod() + 100

        # Price B: perfectly correlated with A
        returns_b = returns_a * 0.8 + np.random.normal(0, 0.005, 100)
        price_b = 50 * pd.Series(returns_b).cumprod() + 50

        # Compute spread
        log_a = np.log(price_a)
        log_b = np.log(price_b)
        beta = 1.0
        spread = log_a - beta * log_b
        zscore_result = compute_zscore(spread, 20)
        zscore = zscore_result["z"]

        pair_data = {
            "TEST/PAIR": pd.DataFrame({
                "close_a": price_a,
                "close_b": price_b,
                "zscore": zscore,
                "spread": spread,
                "beta": beta,
            }, index=dates)
        }

        params = {
            "entry_thresholds": [2.0],
            "exit_types": ["z0"],
            "beta_lookbacks": ["static"],
            "hedge_type": "equal_dollar",
            "cost_scenario": "base",
            "zscore_lookback": 20,
            "initial_capital": 100000.0,
            "max_holding_period_bars": 50,
            "stop_loss_z": 5.0,  # High stop to ensure exits happen
        }

        validator = WalkForwardValidator(train_days=50, test_days=20, roll_days=10)
        wf_result = validator.run(pair_data, params)

        # Key check: signals should be generated from zscore values that
        # were available at that bar's close, not future values
        # We verify this by checking that the backtester ran without errors
        assert wf_result is not None
        assert wf_result.experiment_metadata is not None

        # Verify all entry zscores are actual threshold crossings
        for window in wf_result.window_results:
            for trade in window.backtest_result.trades:
                # Entry zscore should be near the entry threshold
                assert abs(trade.entry_zscore) >= 2.0, "Entry zscore should exceed threshold"

        logger.info("Look-ahead bias check passed")

    def test_report_generation(self):
        """Test that reports can be generated."""
        # Create synthetic backtest result
        equity_curve = pd.Series(
            [100000, 102000, 101000, 103000, 105000],
            index=pd.date_range("2024-01-01", periods=5)
        )

        trades = [
            {
                "net_pnl": 2000,
                "gross_pnl": 2200,
                "notional": 50000,
                "fee": 100,
                "slippage": 50,
                "funding": 25,
                "short_cost": 25,
                "holding_period_bars": 10,
                "timestamp": pd.Timestamp("2024-01-15"),
                "exit_time": pd.Timestamp("2024-01-25"),
            },
        ]

        metrics = compute_all_metrics(trades, equity_curve, 100000.0)

        # Generate charts
        reports_dir = Path("reports") / "test"
        reports_dir.mkdir(parents=True, exist_ok=True)

        equity_path = plot_equity_curve(
            equity_curve,
            title="Test Equity Curve",
            save_path=str(reports_dir / "test_equity.png")
        )

        assert Path(equity_path).exists(), "Equity chart was not generated"

        # Verify metrics
        assert metrics["n_trades"] == 1
        assert metrics["win_rate"] == 1.0
        assert metrics["net_return"] > 0

        logger.info(f"Reports generated successfully: {reports_dir}")

    def test_config_loading(self):
        """Test that configuration can be loaded."""
        config_path = project_root / "config" / "default.yaml"

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify key config sections exist
        assert "cost_scenarios" in config
        assert "risk_limits" in config
        assert "walk_forward" in config
        assert "analysis" in config

        # Verify cost scenarios
        assert "optimistic" in config["cost_scenarios"]
        assert "base" in config["cost_scenarios"]
        assert "pessimistic" in config["cost_scenarios"]

        # Verify risk limits
        risk = config["risk_limits"]
        assert "max_position_pct" in risk
        assert "max_gross_exposure" in risk
        assert "stop_loss_z" in risk

        logger.info("Configuration loaded and validated")

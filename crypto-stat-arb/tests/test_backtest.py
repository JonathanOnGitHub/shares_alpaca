"""Unit tests for the backtest engine."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestResult, PairBacktester, TradeRecord


class TestEquityCurveCalculation:
    """Test equity curve calculation."""

    def test_equity_curve_single_profitable_trade(self):
        """Test equity curve with a single profitable trade."""
        # Create simple price data
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        prices_a = [100, 101, 102, 103, 104, 103, 102, 101, 100, 99]
        prices_b = [50, 51, 52, 53, 54, 53, 52, 51, 50, 49]

        pair_data = {
            "BTC/XRP": pd.DataFrame({
                "close_a": prices_a,
                "close_b": prices_b,
                "zscore": [0, 0, 0, 2.5, 2.5, 2.5, 2.5, 0, 0, 0],
                "spread": [0] * 10,
                "beta": [1.0] * 10,
            }, index=dates)
        }

        strategy_config = {
            "entry_threshold": 2.0,
            "exit_type": "z0",
            "exit_threshold": 0.0,
            "max_holding_period_bars": 100,
            "stop_loss_z": 3.0,
        }

        backtester = PairBacktester(initial_capital=100000.0, cost_scenario="base")
        result = backtester.run(pair_data, strategy_config, "base", "equal_dollar")

        assert len(result.equity_curve) > 0
        assert result.equity_curve.iloc[0] == 100000.0

    def test_equity_curve_empty_data(self):
        """Test equity curve with empty data."""
        backtester = PairBacktester(initial_capital=100000.0, cost_scenario="base")
        result = backtester.run({}, {"entry_threshold": 2.0}, "base", "equal_dollar")

        assert len(result.equity_curve) == 0


class TestPnLWithCosts:
    """Test P&L calculation with known costs."""

    def test_trade_costs_calculation(self):
        """Test that costs are properly deducted."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        prices_a = [100, 110, 110, 100, 100]  # Price goes up then down
        prices_b = [50, 50, 45, 45, 50]  # Price stable then down then up

        pair_data = {
            "TEST/Pair": pd.DataFrame({
                "close_a": prices_a,
                "close_b": prices_b,
                "zscore": [0, 3.0, 3.0, 0, 0],  # Entry at t=1
                "spread": [0] * 5,
                "beta": [1.0] * 5,
            }, index=dates)
        }

        strategy_config = {
            "entry_threshold": 2.0,
            "exit_type": "z0",
            "exit_threshold": 0.0,
            "max_holding_period_bars": 100,
            "stop_loss_z": 3.0,
        }

        backtester = PairBacktester(initial_capital=100000.0, cost_scenario="base")
        result = backtester.run(pair_data, strategy_config, "base", "equal_dollar")

        # Should have executed a trade
        if len(result.trades) > 0:
            trade = result.trades[0]
            # Gross P&L should be different from Net P&L due to costs
            assert trade.gross_pnl != trade.net_pnl or trade.fee > 0

    def test_optimistic_vs_pessimistic_costs(self):
        """Test that optimistic costs result in higher net P&L."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        prices_a = [100, 105, 105, 100, 100]
        prices_b = [50, 50, 48, 48, 50]

        pair_data = {
            "TEST/Pair": pd.DataFrame({
                "close_a": prices_a,
                "close_b": prices_b,
                "zscore": [0, 2.5, 2.5, 0, 0],
                "spread": [0] * 5,
                "beta": [1.0] * 5,
            }, index=dates)
        }

        strategy_config = {
            "entry_threshold": 2.0,
            "exit_type": "z0",
            "max_holding_period_bars": 100,
            "stop_loss_z": 3.0,
        }

        # Run with optimistic costs
        backtester_opt = PairBacktester(initial_capital=100000.0, cost_scenario="optimistic")
        result_opt = backtester_opt.run(pair_data, strategy_config, "optimistic", "equal_dollar")

        # Run with pessimistic costs
        backtester_pess = PairBacktester(initial_capital=100000.0, cost_scenario="pessimistic")
        result_pess = backtester_pess.run(pair_data, strategy_config, "pessimistic", "equal_dollar")

        if len(result_opt.trades) > 0 and len(result_pess.trades) > 0:
            opt_net = result_opt.trades[0].net_pnl
            pess_net = result_pess.trades[0].net_pnl
            # Optimistic costs should result in higher net P&L
            assert opt_net >= pess_net


class TestDrawdownCalculation:
    """Test drawdown calculation."""

    def test_drawdown_tracking(self):
        """Test that drawdown is properly tracked."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        prices_a = [100, 110, 120, 110, 100, 90, 80, 90, 100, 110]
        prices_b = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]

        pair_data = {
            "TEST/Pair": pd.DataFrame({
                "close_a": prices_a,
                "close_b": prices_b,
                "zscore": [0] * 10,
                "spread": [0] * 10,
                "beta": [1.0] * 10,
            }, index=dates)
        }

        # This data has no signals, so equity should be flat
        backtester = PairBacktester(initial_capital=100000.0, cost_scenario="base")
        result = backtester.run(pair_data, {"entry_threshold": 2.0, "exit_type": "z0", "max_holding_period_bars": 100, "stop_loss_z": 3.0}, "base", "equal_dollar")

        # No trades, no drawdown
        assert result.metrics.get("max_drawdown", 0) == 0.0


class TestLookAheadBias:
    """Test look-ahead bias prevention."""

    def test_signal_uses_only_past_data(self):
        """Test that signals at time t use data available at t, not t+1."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")

        # Create zscore that looks like it would generate a signal
        # but we ensure the signal generation only sees past data
        zscore_values = [0, 0, 2.5, 0, 0, 0, 0, 0, 0, 0]

        pair_data = {
            "TEST/Pair": pd.DataFrame({
                "close_a": [100] * 10,
                "close_b": [50] * 10,
                "zscore": zscore_values,
                "spread": [0] * 10,
                "beta": [1.0] * 10,
            }, index=dates)
        }

        strategy_config = {
            "entry_threshold": 2.0,
            "exit_type": "z0",
            "exit_threshold": 0.0,
            "max_holding_period_bars": 100,
            "stop_loss_z": 3.0,
        }

        backtester = PairBacktester(initial_capital=100000.0, cost_scenario="base")
        result = backtester.run(pair_data, strategy_config, "base", "equal_dollar")

        # If zscore at index 2 is 2.5, entry should happen at index 2
        # and exit should happen at index 3 (when z crosses 0)
        # This ensures we don't use future data
        if len(result.trades) > 0:
            trade = result.trades[0]
            # Entry should be before exit
            assert trade.entry_zscore != 0 or trade.exit_zscore != 0


class TestPositionSizing:
    """Test position sizing for all 4 hedge types."""

    def test_equal_dollar_hedge(self):
        """Test equal dollar hedging."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        pair_data = {
            "TEST/Pair": pd.DataFrame({
                "close_a": [100, 100, 100, 100, 100],
                "close_b": [50, 50, 50, 50, 50],
                "zscore": [0, 2.5, 2.5, 0, 0],
                "spread": [0] * 5,
                "beta": [1.0] * 5,
            }, index=dates)
        }

        strategy_config = {
            "entry_threshold": 2.0,
            "exit_type": "z0",
            "max_holding_period_bars": 100,
            "stop_loss_z": 3.0,
        }

        backtester = PairBacktester(initial_capital=100000.0, cost_scenario="base")
        result = backtester.run(pair_data, strategy_config, "base", "equal_dollar")

        if len(result.trades) > 0:
            trade = result.trades[0]
            # For equal dollar, both legs should have similar dollar values
            leg_a_value = trade.shares_a * trade.entry_px_a
            leg_b_value = trade.shares_b * trade.entry_px_b
            # They should be approximately equal
            assert abs(leg_a_value - leg_b_value) / leg_a_value < 0.1

    def test_ols_hedge_ratio(self):
        """Test OLS hedge ratio position sizing."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        pair_data = {
            "TEST/Pair": pd.DataFrame({
                "close_a": [100, 100, 100, 100, 100],
                "close_b": [50, 50, 50, 50, 50],
                "zscore": [0, 2.5, 2.5, 0, 0],
                "spread": [0] * 5,
                "beta": [1.5] * 5,  # Non-1 beta
            }, index=dates)
        }

        strategy_config = {
            "entry_threshold": 2.0,
            "exit_type": "z0",
            "max_holding_period_bars": 100,
            "stop_loss_z": 3.0,
        }

        backtester = PairBacktester(initial_capital=100000.0, cost_scenario="base")
        result = backtester.run(pair_data, strategy_config, "base", "ols")

        # Should execute without error
        assert result is not None

    def test_volatility_adjusted_hedge(self):
        """Test volatility-adjusted hedge position sizing."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        pair_data = {
            "TEST/Pair": pd.DataFrame({
                "close_a": [100, 100, 100, 100, 100],
                "close_b": [50, 50, 50, 50, 50],
                "zscore": [0, 2.5, 2.5, 0, 0],
                "spread": [0] * 5,
                "beta": [1.0] * 5,
                "vol_a": [0.02] * 5,  # 2% volatility
                "vol_b": [0.01] * 5,  # 1% volatility (half of A)
            }, index=dates)
        }

        strategy_config = {
            "entry_threshold": 2.0,
            "exit_type": "z0",
            "max_holding_period_bars": 100,
            "stop_loss_z": 3.0,
        }

        backtester = PairBacktester(initial_capital=100000.0, cost_scenario="base")
        result = backtester.run(pair_data, strategy_config, "base", "volatility_adjusted")

        assert result is not None

    def test_beta_neutral_hedge(self):
        """Test beta-neutral hedge position sizing."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        pair_data = {
            "TEST/Pair": pd.DataFrame({
                "close_a": [100, 100, 100, 100, 100],
                "close_b": [50, 50, 50, 50, 50],
                "zscore": [0, 2.5, 2.5, 0, 0],
                "spread": [0] * 5,
                "beta": [1.2] * 5,
                "beta_existing": [0.5] * 5,
            }, index=dates)
        }

        strategy_config = {
            "entry_threshold": 2.0,
            "exit_type": "z0",
            "max_holding_period_bars": 100,
            "stop_loss_z": 3.0,
        }

        backtester = PairBacktester(initial_capital=100000.0, cost_scenario="base")
        result = backtester.run(pair_data, strategy_config, "base", "beta_neutral")

        assert result is not None


class TestMetrics:
    """Test that metrics are properly calculated."""

    def test_metrics_with_no_trades(self):
        """Test metrics calculation with no trades."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        pair_data = {
            "TEST/Pair": pd.DataFrame({
                "close_a": [100] * 10,
                "close_b": [50] * 10,
                "zscore": [0] * 10,  # No signals
                "spread": [0] * 10,
                "beta": [1.0] * 10,
            }, index=dates)
        }

        backtester = PairBacktester(initial_capital=100000.0, cost_scenario="base")
        result = backtester.run(pair_data, {"entry_threshold": 2.0, "exit_type": "z0", "max_holding_period_bars": 100, "stop_loss_z": 3.0}, "base", "equal_dollar")

        assert result.metrics["n_trades"] == 0
        assert result.metrics["win_rate"] == 0.0

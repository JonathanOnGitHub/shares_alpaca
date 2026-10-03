"""Unit tests for reporting metrics calculations."""

import numpy as np
import pandas as pd
import pytest

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


class TestComputeEquityCurve:
    """Tests for equity curve computation."""

    def test_equity_curve_single_trade(self):
        """Test equity curve with a single profitable trade."""
        trades = [
            {"net_pnl": 1000.0, "exit_time": pd.Timestamp("2024-01-15")},
        ]

        equity = compute_equity_curve(trades, initial_capital=100000.0)

        assert len(equity) == 2
        assert equity.iloc[0] == 100000.0
        assert equity.iloc[-1] == 101000.0

    def test_equity_curve_multiple_trades(self):
        """Test equity curve with multiple trades."""
        trades = [
            {"net_pnl": 500.0, "exit_time": pd.Timestamp("2024-01-10")},
            {"net_pnl": -200.0, "exit_time": pd.Timestamp("2024-01-20")},
            {"net_pnl": 800.0, "exit_time": pd.Timestamp("2024-01-30")},
        ]

        equity = compute_equity_curve(trades, initial_capital=100000.0)

        assert len(equity) == 4
        assert equity.iloc[0] == 100000.0
        assert equity.iloc[1] == 100500.0
        assert equity.iloc[2] == 100300.0
        assert equity.iloc[3] == 101100.0

    def test_equity_curve_empty_trades(self):
        """Test equity curve with no trades."""
        equity = compute_equity_curve([], initial_capital=100000.0)

        assert len(equity) == 0


class TestComputeDrawdown:
    """Tests for drawdown computation."""

    def test_drawdown_basic(self):
        """Test basic drawdown calculation."""
        equity = pd.Series([100000, 110000, 105000, 115000, 100000])
        equity.index = pd.date_range("2024-01-01", periods=5)

        dd = compute_drawdown(equity)

        assert "drawdown" in dd.columns
        assert "drawdown_pct" in dd.columns
        assert "drawdown_peak" in dd.columns

        # Max drawdown is from the all-time peak of 115000 (2024-01-04):
        # (100000 - 115000) / 115000 = -13.04%
        max_dd = dd["drawdown_pct"].min()
        assert abs(max_dd - (-0.1304)) < 0.01

    def test_drawdown_no_drawdown(self):
        """Test drawdown when equity is always increasing."""
        equity = pd.Series([100000, 110000, 120000, 130000])

        dd = compute_drawdown(equity)

        # Should have no drawdown
        assert dd["drawdown_pct"].min() == 0.0


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_sharpe_ratio_basic(self):
        """Test basic Sharpe ratio calculation."""
        returns = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01])

        sharpe = sharpe_ratio(returns, periods_per_year=252)

        # Sharpe should be calculated without error
        assert isinstance(sharpe, float)

    def test_sharpe_ratio_zero_std(self):
        """Test Sharpe with zero standard deviation."""
        returns = pd.Series([0.01, 0.01, 0.01, 0.01])

        sharpe = sharpe_ratio(returns, periods_per_year=252)

        # Should return 0 due to zero std
        assert sharpe == 0.0

    def test_sharpe_ratio_insufficient_data(self):
        """Test Sharpe with insufficient data."""
        returns = pd.Series([0.01])

        sharpe = sharpe_ratio(returns, periods_per_year=252)

        assert sharpe == 0.0


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_sortino_ratio_basic(self):
        """Test basic Sortino ratio calculation."""
        returns = pd.Series([0.01, -0.02, 0.015, 0.01, -0.01])

        sortino = sortino_ratio(returns, periods_per_year=252)

        assert isinstance(sortino, float)

    def test_sortino_ratio_all_positive(self):
        """Test Sortino with all positive returns."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.01])

        sortino = sortino_ratio(returns, periods_per_year=252)

        # Should return 0 since no downside returns
        assert sortino == 0.0


class TestMaxDrawdown:
    """Tests for maximum drawdown calculation."""

    def test_max_drawdown_basic(self):
        """Test basic max drawdown."""
        dd = pd.DataFrame({
            "drawdown_pct": pd.Series([0.0, -0.05, -0.10, -0.05, 0.0]),
        })

        max_dd = max_drawdown(dd)

        assert max_dd == 0.10

    def test_max_drawdown_no_drawdown(self):
        """Test max drawdown with no drawdown."""
        dd = pd.DataFrame({
            "drawdown_pct": pd.Series([0.0, 0.0, 0.0]),
        })

        max_dd = max_drawdown(dd)

        assert max_dd == 0.0


class TestCalmarRatio:
    """Tests for Calmar ratio calculation."""

    def test_calmar_ratio_basic(self):
        """Test basic Calmar ratio calculation."""
        equity = pd.Series([100000, 110000, 105000, 115000, 120000])

        calmar = calmar_ratio(equity, periods_per_year=252)

        assert isinstance(calmar, float)

    def test_calmar_ratio_zero_drawdown(self):
        """Test Calmar with zero drawdown."""
        equity = pd.Series([100000, 110000, 120000, 130000])

        calmar = calmar_ratio(equity, periods_per_year=252)

        # Should return 0 due to zero drawdown
        assert calmar == 0.0


class TestWinRate:
    """Tests for win rate calculation."""

    def test_win_rate_basic(self):
        """Test basic win rate calculation."""
        trades = [
            {"net_pnl": 100.0},
            {"net_pnl": -50.0},
            {"net_pnl": 200.0},
            {"net_pnl": -30.0},
            {"net_pnl": 150.0},
        ]

        wr = win_rate(trades)

        # 3 wins out of 5
        assert wr == 0.6

    def test_win_rate_all_wins(self):
        """Test win rate with all winning trades."""
        trades = [
            {"net_pnl": 100.0},
            {"net_pnl": 200.0},
            {"net_pnl": 150.0},
        ]

        wr = win_rate(trades)

        assert wr == 1.0

    def test_win_rate_all_losses(self):
        """Test win rate with all losing trades."""
        trades = [
            {"net_pnl": -100.0},
            {"net_pnl": -200.0},
            {"net_pnl": -150.0},
        ]

        wr = win_rate(trades)

        assert wr == 0.0

    def test_win_rate_empty(self):
        """Test win rate with no trades."""
        wr = win_rate([])

        assert wr == 0.0


class TestProfitFactor:
    """Tests for profit factor calculation."""

    def test_profit_factor_basic(self):
        """Test basic profit factor calculation."""
        trades = [
            {"net_pnl": 100.0},
            {"net_pnl": -50.0},
            {"net_pnl": 200.0},
            {"net_pnl": -25.0},
        ]

        pf = profit_factor(trades)

        # 300 / 75 = 4.0
        assert pf == 4.0

    def test_profit_factor_no_losses(self):
        """Test profit factor with no losses."""
        trades = [
            {"net_pnl": 100.0},
            {"net_pnl": 200.0},
        ]

        pf = profit_factor(trades)

        assert pf == float("inf")

    def test_profit_factor_no_trades(self):
        """Test profit factor with no trades."""
        pf = profit_factor([])

        assert pf == 0.0


class TestAvgTrade:
    """Tests for average trade calculation."""

    def test_avg_trade_basic(self):
        """Test basic average trade calculation."""
        trades = [
            {"net_pnl": 100.0},
            {"net_pnl": -50.0},
            {"net_pnl": 200.0},
        ]

        avg = avg_trade(trades)

        assert avg == (100.0 - 50.0 + 200.0) / 3

    def test_avg_trade_empty(self):
        """Test average trade with no trades."""
        avg = avg_trade([])

        assert avg == 0.0


class TestTurnover:
    """Tests for turnover calculation."""

    def test_turnover_basic(self):
        """Test basic turnover calculation."""
        trades = [
            {"notional": 50000.0, "net_pnl": 100.0},
            {"notional": 50000.0, "net_pnl": -50.0},
            {"notional": 50000.0, "net_pnl": 200.0},
        ]

        equity = pd.Series([100000, 100050, 100000, 100200])
        equity.index = pd.date_range("2024-01-01", periods=4)

        turnover_rate = turnover(trades, equity)

        assert isinstance(turnover_rate, float)
        assert turnover_rate > 0

    def test_turnover_empty(self):
        """Test turnover with no trades."""
        turnover_rate = turnover([], pd.Series([100000, 110000]))

        assert turnover_rate == 0.0


class TestAvgHoldingPeriod:
    """Tests for average holding period calculation."""

    def test_avg_holding_period_basic(self):
        """Test basic average holding period."""
        trades = [
            {"holding_period_bars": 10},
            {"holding_period_bars": 20},
            {"holding_period_bars": 15},
        ]

        avg_holding = avg_holding_period(trades)

        assert avg_holding == 15.0

    def test_avg_holding_period_empty(self):
        """Test average holding period with no trades."""
        avg_holding = avg_holding_period([])

        assert avg_holding == 0.0


class TestExposure:
    """Tests for exposure calculation."""

    def test_exposure_basic(self):
        """Test basic exposure calculation."""
        trades = [
            {"holding_period_bars": 10},
            {"holding_period_bars": 20},
        ]

        equity = pd.Series([100000] * 100)

        exp = exposure(trades, equity)

        # 30 bars out of 100
        assert exp == 0.30

    def test_exposure_empty(self):
        """Test exposure with no trades."""
        exp = exposure([], pd.Series([100000] * 100))

        assert exp == 0.0


class TestNetVsGross:
    """Tests for net vs gross breakdown."""

    def test_net_vs_gross_basic(self):
        """Test basic net vs gross calculation."""
        result = net_vs_gross(gross_return=10000.0, total_costs=2000.0)

        assert result["gross_return"] == 10000.0
        assert result["total_costs"] == 2000.0
        assert result["net_return"] == 8000.0
        assert result["cost_ratio"] == 0.20

    def test_net_vs_gross_zero_gross(self):
        """Test net vs gross with zero gross return."""
        result = net_vs_gross(gross_return=0.0, total_costs=1000.0)

        assert result["net_return"] == -1000.0
        assert result["cost_ratio"] == 0.0

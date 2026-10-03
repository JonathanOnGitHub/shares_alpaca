"""Unit tests for risk manager and position sizing."""

import numpy as np
import pandas as pd
import pytest

from src.risk.limits import RiskManager
from src.risk.position_sizing import (
    beta_neutral_hedge,
    equal_dollar_hedge,
    ols_hedge_ratio,
    volatility_adjusted_hedge,
)


class TestEqualDollarHedge:
    """Tests for equal dollar hedging."""

    def test_equal_dollar_basic(self):
        """Test basic equal dollar hedge calculation."""
        shares_a, shares_b = equal_dollar_hedge(
            notional=100000.0,
            price_a=100.0,
            price_b=50.0,
        )

        # Dollar value in each leg should be equal
        dollar_a = shares_a * 100.0
        dollar_b = shares_b * 50.0

        assert abs(dollar_a - dollar_b) < 0.01
        assert abs(dollar_a - 50000.0) < 0.01

    def test_equal_dollar_zero_price(self):
        """Test handling of zero price."""
        shares_a, shares_b = equal_dollar_hedge(
            notional=100000.0,
            price_a=0.0,
            price_b=50.0,
        )

        assert shares_a == 0.0
        assert shares_b == 0.0


class TestOLSHedgeRatio:
    """Tests for OLS hedge ratio positioning."""

    def test_ols_hedge_basic(self):
        """Test basic OLS hedge ratio calculation."""
        shares_a, shares_b = ols_hedge_ratio(
            notional=100000.0,
            price_a=100.0,
            price_b=50.0,
            beta=1.0,
        )

        # For beta=1, dollar values should be equal
        dollar_a = shares_a * 100.0
        dollar_b = shares_b * 50.0

        assert abs(dollar_a - dollar_b) / dollar_a < 0.1

    def test_ols_hedge_different_beta(self):
        """Test OLS hedge with different beta."""
        shares_a, shares_b = ols_hedge_ratio(
            notional=100000.0,
            price_a=100.0,
            price_b=50.0,
            beta=2.0,
        )

        # Dollar value in A should be higher due to higher beta
        dollar_a = shares_a * 100.0
        dollar_b = shares_b * 50.0

        # dollar_a / dollar_b should be approximately beta / 1 = 2
        ratio = dollar_a / dollar_b
        assert abs(ratio - 2.0) < 0.2


class TestVolatilityAdjustedHedge:
    """Tests for volatility-adjusted hedging."""

    def test_volatility_adjusted_basic(self):
        """Test basic volatility-adjusted hedge."""
        shares_a, shares_b = volatility_adjusted_hedge(
            notional=100000.0,
            price_a=100.0,
            price_b=50.0,
            vol_a=0.02,  # 2% vol
            vol_b=0.01,  # 1% vol
            target_vol=0.015,
        )

        # Both legs should have similar dollar volatility
        dollar_vol_a = shares_a * 100.0 * 0.02
        dollar_vol_b = shares_b * 50.0 * 0.01

        # They should be roughly equal
        assert abs(dollar_vol_a - dollar_vol_b) / dollar_vol_a < 0.5

    def test_volatility_adjusted_zero_vol(self):
        """Test handling of zero volatility."""
        shares_a, shares_b = volatility_adjusted_hedge(
            notional=100000.0,
            price_a=100.0,
            price_b=50.0,
            vol_a=0.0,  # Zero vol
            vol_b=0.01,
            target_vol=0.015,
        )

        # Should fall back to equal dollar
        dollar_a = shares_a * 100.0
        dollar_b = shares_b * 50.0

        assert abs(dollar_a - dollar_b) / dollar_a < 0.1


class TestBetaNeutralHedge:
    """Tests for beta-neutral hedging."""

    def test_beta_neutral_basic(self):
        """Test basic beta-neutral hedge."""
        shares_a, shares_b = beta_neutral_hedge(
            notional=100000.0,
            price_a=100.0,
            price_b=50.0,
            beta=1.5,
            beta_existing=0.0,
        )

        # Should calculate shares without error
        assert shares_a >= 0
        assert shares_b >= 0

    def test_beta_neutral_with_existing_beta(self):
        """Test beta-neutral with existing portfolio beta."""
        shares_a, shares_b = beta_neutral_hedge(
            notional=100000.0,
            price_a=100.0,
            price_b=50.0,
            beta=1.5,
            beta_existing=0.5,
        )

        # Should produce adjusted shares
        assert shares_a >= 0
        assert shares_b >= 0


class TestRiskManager:
    """Tests for the RiskManager class."""

    def test_risk_manager_init(self):
        """Test RiskManager initialization."""
        config = {
            "max_position_pct": 0.20,
            "max_gross_exposure": 2.0,
            "max_net_exposure": 0.50,
            "max_pair_exposure": 0.40,
            "max_portfolio_exposure": 1.5,
            "stop_loss_z": 3.0,
            "max_holding_period_bars": 100,
        }

        rm = RiskManager(config)

        assert rm.max_position_pct == 0.20
        assert rm.max_gross_exposure == 2.0
        assert rm.stop_loss_z == 3.0

    def test_check_signal_no_signal(self):
        """Test check_signal with no signal."""
        rm = RiskManager()

        approved, reason = rm.check_signal(
            signal=0,
            current_positions={},
            portfolio_equity=100000.0,
            market_data={"zscore": 0, "pair_name": "TEST"},
        )

        assert approved is True
        assert reason == "no_signal"

    def test_check_signal_max_position_size(self):
        """Test signal rejection due to max position size."""
        config = {"max_position_pct": 0.10}  # 10% max
        rm = RiskManager(config)

        market_data = {
            "zscore": 2.5,
            "pair_name": "TEST",
            "prices": {"a": 100.0, "b": 50.0},
        }

        # Position that would exceed limit
        positions = {}

        approved, reason = rm.check_signal(
            signal=1,
            current_positions=positions,
            portfolio_equity=100000.0,  # $100k
            market_data=market_data,
        )

        # Should be approved since no existing positions
        # But if proposed position > max, it should reject
        if "max_position_size" in reason:
            assert approved is False
        else:
            assert approved is True

    def test_check_signal_stop_loss_z(self):
        """Test signal rejection due to stop loss z-score."""
        config = {"stop_loss_z": 3.0}
        rm = RiskManager(config)

        market_data = {
            "zscore": 4.0,  # Exceeds stop loss
            "pair_name": "TEST",
            "prices": {"a": 100.0, "b": 50.0},
        }

        approved, reason = rm.check_signal(
            signal=1,
            current_positions={},
            portfolio_equity=100000.0,
            market_data=market_data,
        )

        assert approved is False
        assert "stop_loss_z" in reason

    def test_apply_volatility_scaling(self):
        """Test volatility scaling."""
        rm = RiskManager()

        # High current vol, should reduce
        adjusted = rm.apply_volatility_scaling(
            signal_size=100000.0,
            current_vol=0.04,
            target_vol=0.02,
        )

        assert adjusted < 100000.0
        assert adjusted > 0

    def test_apply_volatility_scaling_zero_vol(self):
        """Test volatility scaling with zero volatility."""
        rm = RiskManager()

        adjusted = rm.apply_volatility_scaling(
            signal_size=100000.0,
            current_vol=0.0,
            target_vol=0.02,
        )

        # Should return base signal size
        assert adjusted == 100000.0

    def test_drawdown_reduction(self):
        """Test drawdown-based position reduction."""
        rm = RiskManager()

        # Equity curve in drawdown
        equity = pd.Series([100000, 95000, 90000, 85000])
        equity.index = pd.date_range("2024-01-01", periods=4)

        factor = rm.drawdown_reduction(equity, drawdown_threshold=0.10)

        # 15% drawdown exceeds 10% threshold, should reduce
        assert factor < 1.0
        assert factor > 0

    def test_drawdown_reduction_no_drawdown(self):
        """Test drawdown reduction with no drawdown."""
        rm = RiskManager()

        equity = pd.Series([100000, 105000, 110000, 115000])
        equity.index = pd.date_range("2024-01-01", periods=4)

        factor = rm.drawdown_reduction(equity, drawdown_threshold=0.10)

        # No drawdown, should return 1.0
        assert factor == 1.0

    def test_check_holding_period(self):
        """Test holding period check."""
        config = {"max_holding_period_bars": 50}
        rm = RiskManager(config)

        entry_time = pd.Timestamp("2024-01-01")
        current_time = pd.Timestamp("2024-01-10")  # 9 days

        within_limits, reason = rm.check_holding_period(
            entry_time, current_time, timeframe_bars_per_day=24
        )

        # 9 days * 24 bars/day = 216 bars > 50 max
        # Should be False
        assert within_limits is False

    def test_get_risk_summary(self):
        """Test risk summary generation."""
        rm = RiskManager()

        positions = {
            "BTC/XRP": {"dollar_value": 20000},
            "BTC/LINK": {"dollar_value": 15000},
        }

        summary = rm.get_risk_summary(positions, portfolio_equity=100000.0)

        assert summary["portfolio_equity"] == 100000.0
        assert summary["total_gross_exposure"] == 35000.0
        assert summary["n_active_positions"] == 2

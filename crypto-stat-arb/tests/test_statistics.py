"""Unit tests for statistics module."""

import numpy as np
import pandas as pd
import pytest

from src.statistics.pairwise import (
    compute_log_prices,
    compute_spread,
    rolling_beta,
    static_beta,
    compute_hedge_ratio_ols,
)
from src.statistics.zscore import (
    compute_zscore,
    rolling_zscore,
    historical_max_zscore,
    mean_reversion_excursions,
)
from src.statistics.halflife import estimate_halflife
from src.statistics.moments import (
    rolling_correlation,
    spearman_correlation,
    pearson_correlation,
)


class TestPairwise:
    """Tests for pairwise.py functions."""

    def test_compute_log_prices(self, synthetic_prices):
        """Test log price computation."""
        df_a, df_b = synthetic_prices
        result = compute_log_prices(df_a, df_b)

        assert "log_price_a" in result.columns
        assert "log_price_b" in result.columns
        assert len(result) > 0
        # Log prices should be real numbers (can be negative when price < 1)
        assert result["log_price_a"].notna().all()
        assert result["log_price_b"].notna().all()

    def test_compute_spread(self, synthetic_prices):
        """Test spread computation with known beta."""
        df_a, df_b = synthetic_prices
        log_prices = compute_log_prices(df_a, df_b)

        # Use beta = 1.5 (as set in synthetic data)
        spread = compute_spread(
            log_prices["log_price_a"],
            log_prices["log_price_b"],
            beta=1.5,
        )

        assert len(spread) == len(log_prices)
        assert not spread.isna().all()

    def test_static_beta_known_coefficients(self):
        """Test static beta with known coefficients."""
        np.random.seed(42)
        n = 200

        # y = 2*x + noise
        x = pd.Series(np.random.randn(n) * 10 + 100)
        y = 2 * x + np.random.randn(n) * 0.1

        beta = static_beta(y, x)
        assert 1.9 < beta < 2.1  # Should be close to 2.0

    def test_rolling_beta(self, synthetic_prices):
        """Test rolling beta computation."""
        df_a, df_b = synthetic_prices
        log_prices = compute_log_prices(df_a, df_b)

        window = 50
        betas = rolling_beta(
            log_prices["log_price_a"],
            log_prices["log_price_b"],
            window=window,
        )

        assert len(betas) == len(log_prices)
        # First 'window' values should be NaN
        assert betas.iloc[:window].isna().all()
        # After window should have values
        assert betas.iloc[window:].notna().any()

    def test_compute_hedge_ratio_ols(self, synthetic_prices):
        """Test OLS hedge ratio computation."""
        df_a, df_b = synthetic_prices
        log_prices = compute_log_prices(df_a, df_b)

        result = compute_hedge_ratio_ols(
            log_prices["log_price_a"],
            log_prices["log_price_b"],
        )

        assert "beta" in result
        assert "intercept" in result
        assert "r_squared" in result
        assert "residuals" in result
        assert 0 <= result["r_squared"] <= 1


class TestZscore:
    """Tests for zscore.py functions."""

    def test_compute_zscore(self, synthetic_spread):
        """Test z-score computation."""
        zscore_df = compute_zscore(synthetic_spread, lookback=30)

        assert "z" in zscore_df.columns
        assert "spread_mean" in zscore_df.columns
        assert "spread_std" in zscore_df.columns
        # First 29 values should be NaN (lookback)
        assert zscore_df["z"].iloc[:29].isna().all()

    def test_rolling_zscore(self, synthetic_spread):
        """Test rolling z-score function."""
        z = rolling_zscore(synthetic_spread, lookback=30)
        assert isinstance(z, pd.Series)
        assert len(z) == len(synthetic_spread)

    def test_historical_max_zscore(self, synthetic_zscore):
        """Test historical max z-score."""
        max_z = historical_max_zscore(synthetic_zscore)
        assert max_z > 0
        # Should find at least the excursions we created
        assert max_z > 2.0

    def test_historical_max_zscore_with_window(self, synthetic_zscore):
        """Test max z-score with window restriction."""
        max_z = historical_max_zscore(synthetic_zscore, window=50)
        assert max_z >= 0

    def test_mean_reversion_excursions(self, synthetic_zscore):
        """Test excursion analysis."""
        result = mean_reversion_excursions(synthetic_zscore, threshold=2.0)

        assert "pct_excursions_that_reverted" in result
        assert "avg_time_to_reversion_bars" in result
        # We created an excursion at index 50:60
        assert result["pct_excursions_that_reverted"] >= 0


class TestHalflife:
    """Tests for halflife.py functions."""

    def test_estimate_halflife(self, synthetic_returns):
        """Test half-life estimation."""
        halflife = estimate_halflife(synthetic_returns)
        assert halflife > 0
        assert halflife < 1000  # Reasonable upper bound

    def test_estimate_halflife_insufficient_data(self):
        """Test half-life with insufficient data."""
        short_returns = pd.Series([0.1, 0.2, 0.1])
        halflife = estimate_halflife(short_returns)
        assert np.isnan(halflife)

    def test_estimate_halflife_known_process(self):
        """Test half-life estimation against a known OU process.

        The function estimates halflife from spread LEVELS (not returns) by running
        AR(1) on spread levels: spread[t] = c + lam * spread[t-1] + eps

        For an OU process with mean-reversion speed lambda (0 < lam < 1):
            spread[t] = (1-lam)*mu + lam*spread[t-1] + sigma*eps

        Half-life = -log(2) / log(lam)
        For lam=0.90: halflife ≈ 6.6 bars
        """
        np.random.seed(42)
        n = 1000
        lam = 0.90  # OU autoregressive coefficient (spread[t] = lam * spread[t-1] + eps)
        sigma = 0.01

        # Simulate OU spread levels directly (not returns)
        spread = np.zeros(n)
        for t in range(1, n):
            spread[t] = lam * spread[t - 1] + sigma * np.random.randn()

        spread_series = pd.Series(spread)
        # PASS SPREAD LEVELS (not diff) — estimate_halflife runs AR(1) internally
        halflife = estimate_halflife(spread_series)

        # Expected halflife for lam=0.90: -log(2)/log(0.90) ≈ 6.6
        assert 4 < halflife < 12, f"Expected halflife ~6.6, got {halflife}"


class TestMoments:
    """Tests for moments.py functions."""

    def test_rolling_correlation(self, synthetic_prices):
        """Test rolling correlation."""
        df_a, df_b = synthetic_prices

        corr = rolling_correlation(df_a["close"], df_b["close"], window=50)

        assert isinstance(corr, pd.Series)
        assert len(corr) == len(df_a)
        # First 49 values should be NaN
        assert corr.iloc[:49].isna().all()
        # Rest should have values between -1 and 1
        valid_corr = corr.iloc[50:].dropna()
        assert ((valid_corr >= -1) & (valid_corr <= 1)).all()

    def test_spearman_correlation(self, synthetic_prices):
        """Test Spearman correlation."""
        df_a, df_b = synthetic_prices

        corr, p_value = spearman_correlation(
            df_a["close"],
            df_b["close"],
        )

        assert isinstance(corr, float)
        assert isinstance(p_value, float)
        assert -1 <= corr <= 1
        assert 0 <= p_value <= 1

    def test_pearson_correlation(self, synthetic_prices):
        """Test Pearson correlation."""
        df_a, df_b = synthetic_prices

        corr, p_value = pearson_correlation(
            df_a["close"],
            df_b["close"],
        )

        assert isinstance(corr, float)
        assert isinstance(p_value, float)
        assert -1 <= corr <= 1
        assert 0 <= p_value <= 1

    def test_correlation_with_identical_series(self):
        """Test correlation of identical series = 1."""
        s = pd.Series([1, 2, 3, 4, 5])
        corr, p = pearson_correlation(s, s)
        assert abs(corr - 1.0) < 0.001

        corr, p = spearman_correlation(s, s)
        assert abs(corr - 1.0) < 0.001

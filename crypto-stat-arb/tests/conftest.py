"""Pytest fixtures for synthetic test data."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices():
    """Generate synthetic price series for testing.

    Creates two cointegrated price series where log(P_A) ≈ beta * log(P_B) + spread.
    """
    np.random.seed(42)
    n = 500

    # Generate base price B (random walk)
    log_price_b = np.cumsum(np.random.randn(n) * 0.01)
    log_price_b += np.linspace(0, 2, n)  # Trend

    # Generate log price A with beta = 1.5 and some noise
    beta = 1.5
    spread_mean = 0.0
    spread_std = 0.5
    spread_series = pd.Series(np.random.randn(n))
    spread = spread_series.rolling(10).mean()
    spread = spread.fillna(0).values

    log_price_a = beta * log_price_b + spread_mean + spread

    # Create DataFrames
    index = pd.date_range(start="2023-01-01", periods=n, freq="1h")

    df_a = pd.DataFrame({
        "timestamp": index,
        "open": np.exp(log_price_a) * (1 + np.random.randn(n) * 0.001),
        "high": np.exp(log_price_a) * (1 + np.random.randn(n) * 0.002),
        "low": np.exp(log_price_a) * (1 - np.random.randn(n) * 0.002),
        "close": np.exp(log_price_a),
        "volume": np.random.randint(1000, 10000, n),
    }).set_index("timestamp")

    df_b = pd.DataFrame({
        "timestamp": index,
        "open": np.exp(log_price_b) * (1 + np.random.randn(n) * 0.001),
        "high": np.exp(log_price_b) * (1 + np.random.randn(n) * 0.002),
        "low": np.exp(log_price_b) * (1 - np.random.randn(n) * 0.002),
        "close": np.exp(log_price_b),
        "volume": np.random.randint(1000, 10000, n),
    }).set_index("timestamp")

    return df_a, df_b


@pytest.fixture
def synthetic_spread():
    """Generate a synthetic mean-reverting spread for testing."""
    np.random.seed(42)
    n = 500

    # Generate spread as AR(1) process (mean-reverting)
    spread = np.zeros(n)
    lam = -0.05  # Speed of mean reversion
    spread[0] = 0.0

    for t in range(1, n):
        spread[t] = lam * spread[t - 1] + np.random.randn() * 0.1

    index = pd.date_range(start="2023-01-01", periods=n, freq="1h")
    return pd.Series(spread, index=index, name="spread")


@pytest.fixture
def synthetic_zscore():
    """Generate synthetic z-score series for testing."""
    np.random.seed(42)
    n = 500

    # Generate z-scores with known properties
    z = np.random.randn(n) * 0.5

    # Add some excursions
    z[50:60] = 2.5  # High z
    z[100:105] = -2.8  # Low z

    index = pd.date_range(start="2023-01-01", periods=n, freq="1h")
    return pd.Series(z, index=index, name="zscore")


@pytest.fixture
def synthetic_returns():
    """Generate synthetic returns for half-life testing."""
    np.random.seed(42)
    n = 500

    # AR(1) returns with known lambda
    returns = np.zeros(n)
    lam = 0.95  # Lambda close to 1 = slow mean reversion = long half-life

    for t in range(1, n):
        returns[t] = lam * returns[t - 1] + np.random.randn() * 0.01

    index = pd.date_range(start="2023-01-01", periods=n, freq="1h")
    return pd.Series(returns, index=index, name="returns")


@pytest.fixture
def synthetic_ohlcv():
    """Generate synthetic OHLCV data for validator testing."""
    np.random.seed(42)
    n = 100

    base_price = 50000
    returns = np.random.randn(n) * 0.01
    close = base_price * np.exp(np.cumsum(returns))

    index = pd.date_range(start="2023-01-01", periods=n, freq="1h")

    df = pd.DataFrame({
        "timestamp": index,
        "open": close * (1 + np.random.randn(n) * 0.001),
        "high": close * (1 + np.abs(np.random.randn(n)) * 0.005),
        "low": close * (1 - np.abs(np.random.randn(n)) * 0.005),
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })

    return df


@pytest.fixture
def cost_scenario_base():
    """Base cost scenario for testing."""
    return {
        "maker_fee": 0.0004,
        "taker_fee": 0.0004,
        "slippage_bps": 0.02,
        "funding_rate_per_hour": 0.00003,
        "short_rate_per_hour": 0.0001,
    }

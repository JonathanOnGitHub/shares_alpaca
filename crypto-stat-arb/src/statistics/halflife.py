"""Half-life estimation for mean-reverting spread."""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def estimate_halflife(spread_returns: pd.Series) -> float:
    """Estimate the half-life of mean reversion using Ornstein-Uhlenbeck formula.

    The OU model: dX_t = lambda * (mu - X_t) * dt + dW_t
    Where lambda < 0 for mean-reversion.

    The half-life is: -log(2) / log(1 - lambda)
    Where lambda is the AR(1) coefficient of spread returns.

    Args:
        spread_returns: Series of spread returns (first difference of spread).

    Returns:
        Estimated half-life in bars (time periods).
    """
    if len(spread_returns) < 10:
        logger.warning("Insufficient data for half-life estimation (need >= 10 obs)")
        return np.nan

    # Remove NaN values
    returns_clean = spread_returns.dropna()

    if len(returns_clean) < 10:
        return np.nan

    # AR(1) regression: r_t = lambda * r_{t-1} + epsilon
    y = returns_clean.values[1:]
    x = returns_clean.values[:-1]

    # OLS: r_t = a + b * r_{t-1}
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    lam = slope  # lambda coefficient

    # Half-life formula: -log(2) / log(1 - lambda)
    # lambda must be < 1 for stable half-life
    if lam >= 1 or lam <= -1:
        logger.warning(
            f"AR(1) coefficient {lam:.4f} out of stable range [-1, 1], "
            "cannot estimate half-life"
        )
        return np.nan

    halflife = -np.log(2) / np.log(lam)

    # Sanity check: halflife should be positive and reasonable
    if halflife < 0 or halflife > 10000:
        logger.warning(
            f"Computed half-life {halflife:.1f} is out of reasonable range"
        )
        return np.nan

    logger.info(
        f"Half-life estimation: lambda={lam:.4f}, half-life={halflife:.1f} bars"
    )
    return float(halflife)

"""Half-life estimation for mean-reverting spread."""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def estimate_halflife(spread_returns: pd.Series) -> float:
    """Estimate the half-life of mean reversion using Ornstein-Uhlenbeck formula.

    OU process: r_t = λ · r_{t-1} + const + ε_t  (mean-reverting when −1 < λ < 1)

    Half-life formula: HL = −log(2) / log(|λ|)

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
    # Guard against zero variance (flat spread returns — spread didn't move)
    if np.std(np.asarray(x), ddof=1) < 1e-10:
        logger.warning("Spread returns have zero variance — spread is flat; returning NaN")
        return np.nan

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    lam = slope  # lambda coefficient

    # Guard: lam must be strictly between 0 and 1 for a valid mean-reverting half-life.
    # Use |lam| in the formula so it works for both +ve and -ve AR(1) coefficients.
    if lam <= 0 or lam >= 1:
        logger.warning(
            f"AR(1) coefficient {lam:.4f} not in (0, 1) — "
            "spread is not mean-reverting; returning NaN"
        )
        return np.nan

    halflife = -np.log(2) / np.log(abs(lam))

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

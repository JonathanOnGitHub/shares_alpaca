"""Pairwise statistics: log prices, spread, beta, hedge ratio."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_log_prices(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute aligned log prices for two assets.

    Args:
        df_a: DataFrame with OHLCV data for asset A.
        df_b: DataFrame with OHLCV data for asset B.
        price_col: Column name for price (default: 'close'). Also accepts
            'close_a'/'close_b' when df is a pre-processed pair DataFrame.

    Returns:
        DataFrame with aligned log_prices_A and log_prices_B columns.
    """
    # Extract close prices — handle both raw OHLCV ('close') and
    # pre-processed pair data ('close_a' / 'close_b')
    def _price_from_df(df: pd.DataFrame, col: str) -> pd.Series:
        """Extract price series from a DataFrame, trying multiple column names."""
        if isinstance(df, pd.DataFrame):
            if col in df.columns:
                return df[col]
            # Pre-processed pair DataFrame: use close_a / close_b
            if "close_a" in df.columns and col == "close":
                return df["close_a"]
        # df is already a Series, or col not found — return as-is
        return df  # type: ignore[return-value]

    prices_a: pd.Series = _price_from_df(df_a, price_col)
    prices_b: pd.Series = _price_from_df(df_b, price_col)

    # Align on index
    aligned = pd.DataFrame({
        "price_a": prices_a,
        "price_b": prices_b,
    }).dropna()

    if len(aligned) == 0:
        logger.warning("No overlapping data found between pairs")
        return pd.DataFrame()

    result = pd.DataFrame({
        "log_price_a": np.log(aligned["price_a"]),
        "log_price_b": np.log(aligned["price_b"]),
    }, index=aligned.index)

    logger.debug(f"Computed log prices for {len(result)} aligned observations")
    return result


def compute_spread(
    log_a: pd.Series,
    log_b: pd.Series,
    beta: float,
) -> pd.Series:
    """Compute the spread between two log prices.

    spread = log(P_A) - beta * log(P_B)

    Args:
        log_a: Log price of asset A.
        log_b: Log price of asset B.
        beta: Hedge ratio (beta coefficient).

    Returns:
        Spread series.
    """
    spread = log_a - beta * log_b
    logger.debug(f"Computed spread with beta={beta:.4f}, mean={spread.mean():.6f}")
    return spread


def rolling_beta(
    log_a: pd.Series,
    log_b: pd.Series,
    window: int,
) -> pd.Series:
    """Compute rolling OLS beta between two log price series.

    Uses linear regression: log_a = alpha + beta * log_b + epsilon

    Args:
        log_a: Log price of asset A.
        log_b: Log price of asset B.
        window: Rolling window size in periods.

    Returns:
        Series of rolling beta values.
    """
    betas = []
    for i in range(len(log_a)):
        if i < window:
            betas.append(np.nan)
        else:
            window_a = log_a.iloc[i - window:i].values
            window_b = log_b.iloc[i - window:i].values

            # OLS: beta = cov(a,b) / var(b)
            cov = np.cov(window_a, window_b)[0, 1]
            var = np.var(window_b, ddof=1)

            if var > 0:
                beta = cov / var
            else:
                beta = np.nan
            betas.append(beta)

    result = pd.Series(betas, index=log_a.index, name="rolling_beta")
    logger.debug(f"Computed rolling beta with window={window}")
    return result


def static_beta(
    log_a: pd.Series,
    log_b: pd.Series,
) -> float:
    """Compute static (full-history) OLS beta between two log price series.

    Uses linear regression: log_a = alpha + beta * log_b + epsilon

    Args:
        log_a: Log price of asset A.
        log_b: Log price of asset B.

    Returns:
        Static beta value.
    """
    # Remove any NaN values
    mask = ~(log_a.isna() | log_b.isna())
    x = log_b.loc[mask].values
    y = log_a.loc[mask].values

    if len(x) < 3:
        logger.warning("Insufficient data for static beta calculation")
        return 1.0  # Default to 1.0

    # OLS: beta = cov(a,b) / var(b)
    cov = np.cov(x, y)[0, 1]
    var = np.var(x, ddof=1)

    if var > 0:
        beta = cov / var
    else:
        beta = 1.0

    logger.debug(f"Static beta: {beta:.6f}")
    return beta


def compute_hedge_ratio_ols(
    log_a: pd.Series,
    log_b: pd.Series,
) -> dict:
    """Compute hedge ratio using OLS regression.

    Regression: log(P_A) = intercept + beta * log(P_B) + residual

    Args:
        log_a: Log price of asset A.
        log_b: Log price of asset B.

    Returns:
        Dictionary with:
        - beta: Hedge ratio coefficient
        - intercept: Regression intercept
        - r_squared: R-squared of the regression
        - residuals: Series of regression residuals
    """
    # Remove any NaN values
    mask = ~(log_a.isna() | log_b.isna())
    x = log_b.loc[mask].values
    y = log_a.loc[mask].values

    if len(x) < 3:
        logger.warning("Insufficient data for OLS hedge ratio")
        return {
            "beta": 1.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "residuals": pd.Series(dtype=float),
        }

    # OLS using scipy
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Compute residuals
    y_pred = intercept + slope * x
    residuals = y - y_pred

    result = {
        "beta": slope,
        "intercept": intercept,
        "r_squared": r_value ** 2,
        "residuals": pd.Series(residuals, index=log_b.loc[mask].index),
    }

    logger.debug(
        f"Hedge ratio OLS: beta={slope:.4f}, "
        f"intercept={intercept:.6f}, R²={result['r_squared']:.4f}"
    )
    return result


# All 6 pair combinations
PAIR_COMBINATIONS = [
    ("BTC", "XRP"),
    ("BTC", "LINK"),
    ("BTC", "SOL"),
    ("XRP", "LINK"),
    ("XRP", "SOL"),
    ("LINK", "SOL"),
]


def get_pair_name(pair_a: str, pair_b: str) -> str:
    """Get standardized pair name.

    Args:
        pair_a: First asset.
        pair_b: Second asset.

    Returns:
        Pair name in format 'A/B'.
    """
    return f"{pair_a}/{pair_b}"

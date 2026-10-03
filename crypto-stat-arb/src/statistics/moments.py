"""Correlation and moment calculations for spread analysis."""

import logging
from typing import Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int,
) -> pd.Series:
    """Compute rolling Pearson correlation between two series.

    Args:
        series_a: First series.
        series_b: Second series.
        window: Rolling window size.

    Returns:
        Series of rolling correlations.
    """
    if len(series_a) < window or len(series_b) < window:
        logger.warning(
            f"Series length ({len(series_a)}, {len(series_b)}) < window {window}"
        )
        return pd.Series(dtype=float)

    # Use pandas rolling corr
    corr = series_a.rolling(window=window).corr(series_b)
    corr.name = f"rolling_corr_{window}"
    logger.debug(f"Computed {window}-period rolling correlation")
    return corr


def spearman_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
) -> Tuple[float, float]:
    """Compute Spearman rank correlation between two series.

    Args:
        series_a: First series.
        series_b: Second series.

    Returns:
        Tuple of (correlation coefficient, p-value).
    """
    # Align and drop NA
    aligned = pd.DataFrame({"a": series_a, "b": series_b}).dropna()

    if len(aligned) < 5:
        logger.warning("Insufficient paired observations for Spearman correlation")
        return np.nan, np.nan

    corr, p_value = stats.spearmanr(aligned["a"], aligned["b"])

    logger.debug(
        f"Spearman correlation: {corr:.4f}, p-value={p_value:.4f}"
    )
    return float(corr), float(p_value)


def pearson_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
) -> Tuple[float, float]:
    """Compute Pearson correlation between two series.

    Args:
        series_a: First series.
        series_b: Second series.

    Returns:
        Tuple of (correlation coefficient, p-value).
    """
    # Align and drop NA
    aligned = pd.DataFrame({"a": series_a, "b": series_b}).dropna()

    if len(aligned) < 5:
        logger.warning("Insufficient paired observations for Pearson correlation")
        return np.nan, np.nan

    corr, p_value = stats.pearsonr(aligned["a"], aligned["b"])

    logger.debug(
        f"Pearson correlation: {corr:.4f}, p-value={p_value:.4f}"
    )
    return float(corr), float(p_value)

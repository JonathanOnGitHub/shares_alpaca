"""Z-score calculations for spread analysis."""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_zscore(
    spread: pd.Series,
    lookback: int = 30,
) -> pd.DataFrame:
    """Compute z-score of the spread using rolling mean and std.

    z = (spread - rolling_mean) / rolling_std

    Args:
        spread: Spread series.
        lookback: Rolling window size for mean and std calculation.

    Returns:
        DataFrame with columns:
        - z: Z-score series
        - spread_mean: Rolling mean
        - spread_std: Rolling standard deviation
    """
    if len(spread) < lookback:
        logger.warning(
            f"Spread length {len(spread)} < lookback {lookback}, "
            "returning NaN z-scores"
        )
        return pd.DataFrame({
            "z": [np.nan] * len(spread),
            "spread_mean": spread,
            "spread_std": [np.nan] * len(spread),
        }, index=spread.index)

    spread_mean = spread.rolling(window=lookback, min_periods=lookback).mean()
    spread_std = spread.rolling(window=lookback, min_periods=lookback).std()

    # Avoid division by zero
    spread_std = spread_std.replace(0, np.nan)

    z = (spread - spread_mean) / spread_std

    result = pd.DataFrame({
        "z": z,
        "spread_mean": spread_mean,
        "spread_std": spread_std,
    }, index=spread.index)

    logger.debug(f"Computed z-score with lookback={lookback}")
    return result


def rolling_zscore(
    spread: pd.Series,
    lookback: int = 30,
) -> pd.Series:
    """Compute rolling z-score of the spread.

    This is a convenience function that returns just the z-score series.

    Args:
        spread: Spread series.
        lookback: Rolling window size.

    Returns:
        Series of z-scores.
    """
    zscore_df = compute_zscore(spread, lookback)
    return zscore_df["z"]


def historical_max_zscore(
    zscore_series: pd.Series,
    window: int | None = None,
) -> float:
    """Find the historical maximum absolute z-score.

    Args:
        zscore_series: Series of z-scores.
        window: Optional window to limit to recent observations.

    Returns:
        Maximum absolute z-score value.
    """
    if window is not None:
        zscore_series = zscore_series.iloc[-window:]

    zscore_clean = zscore_series.dropna()

    if len(zscore_clean) == 0:
        logger.warning("Empty z-score series, returning 0")
        return 0.0

    max_abs_z = float(zscore_clean.abs().max())

    logger.debug(f"Historical max |z-score|: {max_abs_z:.4f}")
    return max_abs_z


def mean_reversion_excursions(
    zscore_series: pd.Series,
    threshold: float = 2.0,
) -> dict[str, Any]:
    """Calculate mean reversion statistics for z-score excursions.

    An excursion occurs when |z| > threshold.
    This function tracks what happens after an excursion.

    Args:
        zscore_series: Series of z-scores.
        threshold: Z-score threshold for excursion detection.

    Returns:
        Dictionary with:
        - pct_excursions_that_reverted: % of excursions that mean-reverted
        - avg_time_to_reversion_bars: Average bars to mean reversion
    """
    zscore_clean = zscore_series.dropna()

    if len(zscore_clean) < 3:
        logger.warning("Insufficient data for excursion analysis")
        return {
            "pct_excursions_that_reverted": np.nan,
            "avg_time_to_reversion_bars": np.nan,
        }

    # Find excursion points (|z| exceeds threshold)
    abs_z = abs(zscore_clean)
    in_excursion = abs_z > threshold

    # Find where excursions start
    excursion_starts = []
    prev_in_excursion = False
    for i, (idx, in_exc) in enumerate(in_excursion.items()):
        if in_exc and not prev_in_excursion:
            excursion_starts.append((idx, i))
        prev_in_excursion = in_exc

    if len(excursion_starts) == 0:
        logger.info(f"No excursions found with threshold={threshold}")
        return {
            "pct_excursions_that_reverted": 0.0,
            "avg_time_to_reversion_bars": np.nan,
        }

    # For each excursion, check if it reverted to |z| < threshold
    reversion_times = []
    reverted_count = 0

    for start_idx, start_i in excursion_starts:
        # Look forward from excursion start
        for j in range(start_i + 1, len(zscore_clean)):
            if abs(zscore_clean.iloc[j]) < threshold:
                # Mean reversion occurred
                reversion_times.append(j - start_i)
                reverted_count += 1
                break

    if len(reversion_times) > 0:
        pct_reverted = 100.0 * reverted_count / len(excursion_starts)
        avg_time = np.mean(reversion_times)
    else:
        pct_reverted = 0.0
        avg_time = np.nan

    result = {
        "pct_excursions_that_reverted": pct_reverted,
        "avg_time_to_reversion_bars": avg_time,
    }

    logger.info(
        f"Excursion analysis (threshold={threshold}): "
        f"{reverted_count}/{len(excursion_starts)} reverted, "
        f"avg_time={avg_time:.1f} bars"
    )
    return result

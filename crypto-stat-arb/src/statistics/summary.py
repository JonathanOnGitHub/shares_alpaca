"""Summary statistics for pair analysis."""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def pair_summary(
    zscore: pd.DataFrame,
    cointegration_result: dict[str, Any],
    halflife: float,
    hedge_ratio: dict[str, Any],
) -> dict[str, Any]:
    """Generate comprehensive summary statistics for a trading pair.

    Args:
        zscore: DataFrame with z-score calculations (from compute_zscore).
        cointegration_result: Dict from engle_granger_test.
        halflife: Estimated half-life in bars.
        hedge_ratio: Dict from compute_hedge_ratio_ols.

    Returns:
        Dictionary with all required statistics.
    """
    summary = {}

    # Z-score statistics
    if zscore is not None and not zscore.empty:
        z_series = zscore["z"].dropna()
        if len(z_series) > 0:
            summary["zscore_mean"] = float(z_series.mean())
            summary["zscore_std"] = float(z_series.std())
            summary["zscore_min"] = float(z_series.min())
            summary["zscore_max"] = float(z_series.max())
            summary["zscore_abs_max"] = float(z_series.abs().max())

    # Cointegration results
    if cointegration_result is not None:
        summary["cointegration_eg_stat"] = cointegration_result.get(
            "test_statistic", np.nan
        )
        summary["cointegration_eg_pvalue"] = cointegration_result.get(
            "p_value", np.nan
        )
        summary["cointegration_conclusion"] = cointegration_result.get(
            "conclusion", "unknown"
        )

    # Half-life
    summary["halflife_bars"] = (
        float(halflife) if halflife is not None and not np.isnan(halflife)
        else np.nan
    )

    # Hedge ratio
    if hedge_ratio is not None:
        summary["hedge_ratio_beta"] = hedge_ratio.get("beta", np.nan)
        summary["hedge_ratio_intercept"] = hedge_ratio.get("intercept", np.nan)
        summary["hedge_ratio_r_squared"] = hedge_ratio.get("r_squared", np.nan)

    # Spread statistics
    if zscore is not None and "spread_mean" in zscore.columns:
        spread_mean = zscore["spread_mean"].dropna()
        if len(spread_mean) > 0:
            summary["spread_mean"] = float(spread_mean.mean())
            summary["spread_std"] = float(spread_mean.std())

    logger.info(f"Generated pair summary: {summary}")
    return summary

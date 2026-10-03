"""Cointegration tests: Engle-Granger, ADF, and rolling cointegration."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


def engle_granger_test(
    series_a: pd.Series,
    series_b: pd.Series,
) -> dict[str, Any]:
    """Engle-Granger two-step cointegration test.

    Step 1: Regress series_a on series_b to get residuals (spread)
    Step 2: ADF test on residuals to check for stationarity

    Args:
        series_a: First time series (e.g., log price of asset A).
        series_b: Second time series (e.g., log price of asset B).

    Returns:
        Dictionary with:
        - test_statistic: Engle-Granger test statistic
        - p_value: p-value from MacKinnon table
        - critical_values: dict with 1pct, 5pct, 10pct values
        - null_hypothesis: Description of H0
        - conclusion: 'cointegrated' or 'not_cointegrated'
    """
    # Align series
    aligned = pd.DataFrame({"a": series_a, "b": series_b}).dropna()

    if len(aligned) < 20:
        logger.warning("Insufficient data for Engle-Granger test (need >= 20 obs)")
        return _empty_cointegration_result()

    y = aligned["a"].values
    x = aligned["b"].values

    # Step 1: OLS regression
    x_with_const = add_constant(x)
    model = OLS(y, x_with_const).fit()

    # Get residuals (spread)
    residuals = model.resid

    # Step 2: ADF test on residuals
    adf_result = adfuller(
        residuals,
        maxlag=12,
        regression="c",  # constant only
        autolag="AIC",
    )

    test_stat = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]

    # Critical values from MacKinnon's table (approximate)
    macinnon_critical = {
        "1pct": critical_values.get("1%", -3.96),
        "5pct": critical_values.get("5%", -3.41),
        "10pct": critical_values.get("10%", -3.12),
    }

    # Conclusion: reject H0 (no cointegration) if test stat < critical value
    # ADF test H0: series has a unit root (non-stationary)
    # If we reject H0, the spread is stationary = cointegrated
    is_cointegrated = p_value < 0.05

    result = {
        "test_statistic": test_stat,
        "p_value": p_value,
        "critical_values": macinnon_critical,
        "null_hypothesis": "No cointegration (residuals have unit root)",
        "conclusion": "cointegrated" if is_cointegrated else "not_cointegrated",
        "residuals": residuals,
        "n_obs": len(aligned),
        "beta": model.params[1] if len(model.params) > 1 else 1.0,
        "intercept": model.params[0] if len(model.params) > 1 else 0.0,
    }

    logger.info(
        f"Engle-Granger test: stat={test_stat:.4f}, p={p_value:.4f}, "
        f"conclusion={result['conclusion']}"
    )
    return result


def adf_test(series: pd.Series) -> dict[str, Any]:
    """Augmented Dickey-Fuller test for stationarity.

    Tests H0: series has a unit root (non-stationary)
    If we reject H0, the series is stationary.

    Args:
        series: Time series to test.

    Returns:
        Dictionary with:
        - test_statistic: ADF test statistic
        - p_value: p-value
        - critical_values: dict with 1%, 5%, 10% critical values
        - lags: Number of lags used
        - n_obs: Number of observations
        - conclusion: 'stationary' or 'non_stationary'
    """
    series_clean = series.dropna()

    if len(series_clean) < 20:
        logger.warning("Insufficient data for ADF test (need >= 20 obs)")
        return _empty_adf_result()

    result = adfuller(
        series_clean,
        maxlag=12,
        regression="c",
        autolag="AIC",
    )

    test_stat = result[0]
    p_value = result[1]
    lags = result[2]
    n_obs = result[3]
    critical_values = result[4]

    # Conclusion
    is_stationary = p_value < 0.05

    adf_result = {
        "test_statistic": test_stat,
        "p_value": p_value,
        "critical_values": {
            "1pct": critical_values.get("1%", -3.96),
            "5pct": critical_values.get("5%", -3.41),
            "10pct": critical_values.get("10%", -3.12),
        },
        "lags": lags,
        "n_obs": n_obs,
        "conclusion": "stationary" if is_stationary else "non_stationary",
    }

    logger.info(
        f"ADF test: stat={test_stat:.4f}, p={p_value:.4f}, "
        f"lags={lags}, conclusion={adf_result['conclusion']}"
    )
    return adf_result


def rolling_cointegration(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    window: int = 120,
) -> pd.DataFrame:
    """Rolling cointegration test on two price series.

    Computes Engle-Granger and ADF statistics on rolling windows.

    Args:
        df_a: DataFrame with price data for asset A.
        df_b: DataFrame with price data for asset B.
        window: Rolling window size in periods.

    Returns:
        DataFrame with columns:
        - timestamp: Window end timestamp
        - eg_stat: Engle-Granger test statistic
        - eg_pvalue: Engle-Granger p-value
        - adf_stat: ADF test statistic on spread
        - adf_pvalue: ADF p-value
        - is_cointegrated: bool (True if p < 0.05)
    """
    # Extract close prices
    if isinstance(df_a, pd.DataFrame) and "close" in df_a.columns:
        price_a = df_a["close"]
    else:
        price_a = df_a

    if isinstance(df_b, pd.DataFrame) and "close" in df_b.columns:
        price_b = df_b["close"]
    else:
        price_b = df_b

    log_a = np.log(price_a)
    log_b = np.log(price_b)

    # Align
    aligned = pd.DataFrame({"log_a": log_a, "log_b": log_b}).dropna()

    results = []

    for i in range(window, len(aligned)):
        window_data = aligned.iloc[i - window:i]

        # Engle-Granger test
        eg_result = engle_granger_test(
            window_data["log_a"],
            window_data["log_b"],
        )

        # ADF test on spread
        if "residuals" in eg_result and len(eg_result["residuals"]) > 0:
            residuals = eg_result["residuals"]
            adf_result = adf_test(pd.Series(residuals))
        else:
            adf_result = _empty_adf_result()

        results.append({
            "timestamp": aligned.index[i],
            "eg_stat": eg_result.get("test_statistic", np.nan),
            "eg_pvalue": eg_result.get("p_value", np.nan),
            "adf_stat": adf_result.get("test_statistic", np.nan),
            "adf_pvalue": adf_result.get("p_value", np.nan),
            "is_cointegrated": bool(eg_result.get("p_value", 1.0) < 0.05),
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.set_index("timestamp")

    logger.info(
        f"Rolling cointegration ({window} bar window): "
        f"{result_df['is_cointegrated'].sum()}/{len(result_df)} windows cointegrated"
    )
    return result_df


def _empty_cointegration_result() -> dict[str, Any]:
    """Return empty cointegration result for error cases."""
    return {
        "test_statistic": np.nan,
        "p_value": np.nan,
        "critical_values": {"1pct": np.nan, "5pct": np.nan, "10pct": np.nan},
        "null_hypothesis": "No cointegration",
        "conclusion": "insufficient_data",
        "residuals": np.array([]),
        "n_obs": 0,
        "beta": np.nan,
        "intercept": np.nan,
    }


def _empty_adf_result() -> dict[str, Any]:
    """Return empty ADF result for error cases."""
    return {
        "test_statistic": np.nan,
        "p_value": np.nan,
        "critical_values": {"1pct": np.nan, "5pct": np.nan, "10pct": np.nan},
        "lags": 0,
        "n_obs": 0,
        "conclusion": "insufficient_data",
    }

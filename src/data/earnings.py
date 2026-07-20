import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class EarningsData:
    def __init__(self):
        self.cache: dict[str, pd.DataFrame] = {}

    def fetch_earnings(
        self, symbols: list[str], lookback_days: int = 800
    ) -> dict[str, pd.DataFrame]:
        end = pd.Timestamp.now(tz=None)
        start = end - pd.Timedelta(days=lookback_days)

        result = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                earnings = ticker.earnings_dates
                if earnings is None or earnings.empty:
                    logger.warning("No earnings data for %s", symbol)
                    continue
                earnings = earnings.sort_index()
                if earnings.index.tz is not None:
                    idx = earnings.index.tz_localize(None)
                else:
                    idx = earnings.index
                mask = (idx >= start) & (idx <= end)
                earnings = earnings[mask].copy()
                if earnings.empty:
                    logger.warning("No earnings in range for %s", symbol)
                    continue
                earnings = earnings.reset_index()
                date_col = earnings.columns[0]
                if earnings[date_col].dtype == "object":
                    earnings["date"] = pd.to_datetime(earnings[date_col], utc=True).dt.tz_localize(None)
                else:
                    earnings["date"] = pd.to_datetime(earnings[date_col]).dt.tz_localize(None)
                earnings["date"] = earnings["date"].dt.normalize()
                earnings["date"] = earnings["date"].apply(lambda x: x.to_datetime64() if hasattr(x, 'to_datetime64') else x)
                result[symbol] = earnings
                logger.info(
                    "%s: %d earnings dates found", symbol, len(earnings)
                )
            except Exception as e:
                logger.warning("Failed to fetch earnings for %s: %s", symbol, e)

        for symbol in list(result.keys()):
            result[symbol] = self.compute_surprise(result[symbol])
        self.cache = result
        return result

    @staticmethod
    def compute_surprise(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        est_col = None
        actual_col = None
        for c in df.columns:
            cl = c.lower()
            if "estimate" in cl:
                est_col = c
            if "reported" in cl or cl == "eps":
                actual_col = c

        if est_col is None or actual_col is None:
            logger.warning("EPS columns not found in %s", list(df.columns))
            return df

        eps_estimate = df[est_col]
        eps_actual = df[actual_col]
        surprise = eps_actual - eps_estimate
        estimate_abs = np.abs(eps_estimate).replace(0, np.nan)
        df["earnings_surprise_pct"] = (surprise / estimate_abs) * 100
        df["earnings_surprise_abs"] = surprise
        df["earnings_beat"] = (surprise > 0).astype(float)
        return df

    def get_features_for_symbol(
        self, symbol: str, price_index: "pd.DatetimeIndex"
    ) -> pd.DataFrame:
        if symbol not in self.cache or self.cache[symbol].empty:
            return pd.DataFrame(index=price_index)

        earnings = self.compute_surprise(self.cache[symbol])
        features = pd.DataFrame(index=price_index)
        features["days_since_earnings"] = np.nan
        features["earnings_surprise_pct"] = 0.0
        features["earnings_beat"] = 0.0
        features["in_pead_window"] = 0.0

        price_dates = features.index.date if hasattr(features.index, 'date') else features.index
        for _, row in earnings.iterrows():
            edate = pd.Timestamp(row["date"]).date()
            surprise = row.get("earnings_surprise_pct", 0)
            beat = row.get("earnings_beat", 0)
            if np.isnan(surprise):
                continue
            mask = np.array([d >= edate for d in price_dates])
            features.loc[mask, "days_since_earnings"] = (
                np.array([(d - edate).days for d in price_dates])[mask]
            )
            features.loc[mask, "earnings_surprise_pct"] = surprise
            features.loc[mask, "earnings_beat"] = beat

        features["in_pead_window"] = (
            (features["days_since_earnings"] >= 0)
            & (features["days_since_earnings"] <= 20)
        ).astype(float)

        features["days_since_earnings"] = features["days_since_earnings"].fillna(99)
        return features

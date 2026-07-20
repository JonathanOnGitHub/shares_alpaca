import numpy as np
import pandas as pd


class PEADOverlay:
    def __init__(self, config: dict):
        self.drift_window = config.get("drift_window", 20)
        self.min_surprise_pct = config.get("min_surprise_pct", 2.0)
        self.max_signal = config.get("max_signal", 0.02)

    def apply(
        self,
        predictions: dict[str, np.ndarray],
        data: dict[str, pd.DataFrame],
        earnings_cache: dict[str, pd.DataFrame],
    ) -> dict[str, np.ndarray]:
        result = {}
        for symbol in predictions:
            preds = predictions[symbol].copy()
            df = data.get(symbol)
            edf = earnings_cache.get(symbol) if earnings_cache else None

            if df is None or edf is None or edf.empty:
                result[symbol] = preds
                continue

            aligned_dates = df.index[:len(preds)]
            for i, date in enumerate(aligned_dates):
                pead_signal = self._get_signal(date, edf)
                if pead_signal is not None:
                    preds[i] = pead_signal

            result[symbol] = preds
        return result

    def _get_signal(self, date: pd.Timestamp, edf: pd.DataFrame) -> float | None:
        d = date.to_datetime64() if hasattr(date, 'to_datetime64') else np.datetime64(date)

        for _, row in edf.iterrows():
            edate = row.get("date")
            if edate is None:
                continue
            e = np.datetime64(edate) if not isinstance(edate, np.datetime64) else edate

            days_since = (d - e) / np.timedelta64(1, 'D')
            days_since = int(days_since)

            if 0 <= days_since <= self.drift_window:
                surprise = row.get("earnings_surprise_pct", 0)
                if np.isnan(surprise):
                    continue
                if abs(surprise) < self.min_surprise_pct:
                    continue

                decay = 1.0 - (days_since / self.drift_window)
                signal_strength = min(abs(surprise) / 100, self.max_signal)
                direction = 1.0 if surprise > 0 else -1.0
                return direction * signal_strength * decay

        return None

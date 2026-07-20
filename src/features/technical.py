import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self, config: dict):
        self.config = config

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        indicators = self.config.get("technical_indicators", {})
        multi_param = {"macd", "bollinger"}

        for name, params in indicators.items():
            method = getattr(self, f"add_{name}", None)
            if method is None:
                continue
            if name in multi_param:
                df = method(df, params)
            elif isinstance(params, list):
                for p in params:
                    df = method(df, p)
            else:
                df = method(df, params)

        price_config = self.config.get("price_features", [])
        for feature in price_config:
            if feature == "returns":
                df["returns"] = df["close"].pct_change()
            elif feature == "log_returns":
                df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
            elif feature == "high_low_pct":
                df["high_low_pct"] = (df["high"] - df["low"]) / df["low"]
            elif feature == "close_open_pct":
                df["close_open_pct"] = (df["close"] - df["open"]) / df["open"]

        return df

    @staticmethod
    def add_sma(df: pd.DataFrame, period: int) -> pd.DataFrame:
        df[f"sma_{period}"] = df["close"].rolling(period).mean()
        df[f"sma_{period}_ratio"] = df["close"] / df[f"sma_{period}"]
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
        df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
        df[f"ema_{period}_ratio"] = df["close"] / df[f"ema_{period}"]
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int) -> pd.DataFrame:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_macd(df: pd.DataFrame, params: list | int) -> pd.DataFrame:
        if isinstance(params, int):
            fast, slow, signal = 12, 26, 9
        else:
            fast, slow, signal = params
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_histogram"] = macd_line - signal_line
        return df

    @staticmethod
    def add_bollinger(df: pd.DataFrame, params: list | int) -> pd.DataFrame:
        if isinstance(params, int):
            period, std = 20, 2
        else:
            period, std = params
        sma = df["close"].rolling(period).mean()
        rolling_std = df["close"].rolling(period).std()
        df["bb_upper"] = sma + std * rolling_std
        df["bb_lower"] = sma - std * rolling_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int) -> pd.DataFrame:
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f"atr_{period}"] = tr.rolling(period).mean()
        df[f"atr_{period}_pct"] = df[f"atr_{period}"] / df["close"]
        return df

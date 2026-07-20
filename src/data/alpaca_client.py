import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from dotenv import load_dotenv

load_dotenv()


class AlpacaClient:
    def __init__(self, paper: bool = True):
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env file"
            )

        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)

    def get_bars(
        self,
        symbols: list[str],
        timeframe: str = "Day",
        lookback_days: int = 365,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> dict[str, pd.DataFrame]:
        tf_map = {
            "Day": TimeFrame(1, TimeFrameUnit.Day),
            "Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "Minute": TimeFrame(1, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        }

        if start is None:
            start = datetime.now() - timedelta(days=lookback_days)
        if end is None:
            end = datetime.now()

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Day)),
            start=start,
            end=end,
            adjustment="split",
            feed="iex",
        )

        bars = self.data_client.get_stock_bars(request)
        result = {}
        for symbol in symbols:
            if symbol in bars.data:
                df = pd.DataFrame([bar.__dict__ for bar in bars.data[symbol]])
                if not df.empty:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
                    df.sort_index(inplace=True)
                result[symbol] = df
        return result

    def get_tradable_assets(self) -> pd.DataFrame:
        assets = self.trading_client.get_all_positions()
        # Use GetAssetsRequest instead
        req = GetAssetsRequest(status="active")
        assets = self.trading_client.get_all_assets(req)
        df = pd.DataFrame([a.__dict__ for a in assets])
        if not df.empty:
            df = df[df["tradable"] == True]
        return df

    def get_account(self):
        return self.trading_client.get_account()

    def get_positions(self):
        return self.trading_client.get_all_positions()

    @staticmethod
    def bars_to_features(bars_df: pd.DataFrame) -> pd.DataFrame:
        df = bars_df.copy()
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["high_low_pct"] = (df["high"] - df["low"]) / df["low"]
        df["close_open_pct"] = (df["close"] - df["open"]) / df["open"]
        df["volume_ma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]
        return df

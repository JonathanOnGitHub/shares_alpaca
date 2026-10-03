"""Market data downloader for Binance public API."""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)

# API configuration
BINANCE_KLINE_URL = "https://api.binance.com/api/v3/klines"

# Default lookback days per interval
DEFAULT_LOOKBACK_DAYS = {
    "5m": 30,
    "15m": 60,
    "1h": 720,
    "4h": 720,
    "1d": 1825,
}

# Interval to Binance format mapping
INTERVAL_MAP = {
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


class MarketDataDownloader:
    """Download OHLCV data from Binance public API.

    Downloads historical klines for cryptocurrency pairs and saves them
    locally in parquet format with metadata for reproducibility.
    """

    def __init__(
        self,
        base_url: str = BINANCE_KLINE_URL,
        rate_limit_delay_ms: int = 500,
        max_retries: int = 3,
        timeout_seconds: int = 30,
        lookback_days: Optional[dict] = None,
    ):
        """Initialize the downloader.

        Args:
            base_url: Binance API endpoint for klines.
            rate_limit_delay_ms: Delay between requests in milliseconds.
            max_retries: Maximum number of retry attempts on failure.
            timeout_seconds: Request timeout in seconds.
            lookback_days: Dict mapping interval to lookback days.
        """
        self.base_url = base_url
        self.rate_limit_delay = rate_limit_delay_ms / 1000.0
        self.max_retries = max_retries
        self.timeout = timeout_seconds
        self.lookback_days = lookback_days or DEFAULT_LOOKBACK_DAYS

    def _get_params(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> dict:
        """Build request parameters for Binance API.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Candle interval (e.g., '1h').
            start_time: Start time in milliseconds.
            end_time: End time in milliseconds.
            limit: Maximum number of candles per request.

        Returns:
            Dictionary of request parameters.
        """
        params = {
            "symbol": symbol.upper(),
            "interval": INTERVAL_MAP.get(interval, interval),
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        return params

    def _fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> list:
        """Fetch klines from Binance API with retry logic.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Candle interval.
            start_time: Start time in milliseconds.
            end_time: End time in milliseconds.

        Returns:
            List of kline data from API.

        Raises:
            requests.HTTPError: If API request fails after retries.
        """
        params = self._get_params(symbol, interval, start_time, end_time)

        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                time.sleep(self.rate_limit_delay)
                return response.json()
            except requests.RequestException as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed for "
                    f"{symbol} {interval}: {e}"
                )
                if attempt == self.max_retries - 1:
                    raise
                # Exponential backoff
                time.sleep(2 ** attempt)

        return []

    def _parse_klines(self, klines: list) -> pd.DataFrame:
        """Parse raw kline data into DataFrame.

        Binance kline format:
        [
            [
                open_time,
                open, high, low, close, volume,
                close_time, quote_volume, trades,
                taker_buy_base, taker_buy_quote, ignore
            ]
        ]

        Args:
            klines: Raw kline data from Binance API.

        Returns:
            DataFrame with OHLCV columns and UTC timestamps.
        """
        if not klines:
            return pd.DataFrame()

        columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "num_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]

        df = pd.DataFrame(klines, columns=columns)

        # Convert numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert timestamp to UTC datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        # Keep only required columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Set timestamp as DatetimeIndex for downstream time-series operations
        df = df.set_index("timestamp")
        df.index = df.index.tz_convert(None)  # Drop UTC tz

        return df

    def download_pair(
        self,
        pair: str,
        interval: str,
        lookback_days: Optional[int] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Download historical data for a single pair.

        Args:
            pair: Base currency (e.g., 'BTC').
            interval: Candle interval (e.g., '1h', '1d').
            lookback_days: Number of days to look back. Uses default if None.

        Returns:
            Tuple of (DataFrame with OHLCV data, metadata dict).
        """
        if lookback_days is None:
            lookback_days = self.lookback_days.get(interval, 720)

        symbol = f"{pair.upper()}USDT"
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=lookback_days)

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        logger.info(f"Downloading {symbol} {interval} from {start_time.date()} to {end_time.date()}")

        all_klines = []
        current_start = start_ms

        # Binance returns max 1000 candles per request, so paginate
        while current_start < end_ms:
            klines = self._fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_ms,
            )
            if not klines:
                break

            all_klines.extend(klines)
            # Move start to last candle time + 1ms to avoid duplicates
            last_open_time = int(klines[-1][0])
            current_start = last_open_time + 1

            logger.debug(f"Fetched {len(klines)} candles, total: {len(all_klines)}")

        if not all_klines:
            logger.warning(f"No data returned for {symbol} {interval}")
            return pd.DataFrame(), {}

        df = self._parse_klines(all_klines)

        # Build metadata
        metadata = {
            "exchange": "binance",
            "pair": pair.upper(),
            "quote_currency": "USDT",
            "timezone": "UTC",
            "interval": interval,
            "data_start": str(df.index.min()),
            "data_end": str(df.index.max()),
            "missing_obs": 0,  # Would need gap detection logic
            "api_limitations": "Free tier: 1200 requests/min, max 1000 candles per query",
            "n_candles": len(df),
        }

        logger.info(f"Downloaded {len(df)} candles for {symbol} {interval}")

        return df, metadata

    def download_all_pairs(
        self,
        pairs: list[str],
        interval: str,
    ) -> dict[str, pd.DataFrame]:
        """Download data for multiple pairs.

        Args:
            pairs: List of base currencies.
            interval: Candle interval.

        Returns:
            Dictionary mapping pair name to DataFrame.
        """
        results = {}

        for pair in pairs:
            try:
                df, metadata = self.download_pair(pair, interval)
                if not df.empty:
                    results[pair] = df
                    # Save to disk
                    self._save_data(pair, interval, df, metadata)
                else:
                    logger.warning(f"No data for {pair}, skipping")
            except Exception as e:
                logger.error(f"Failed to download {pair}: {e}")
                continue

        return results

    def _save_data(
        self,
        pair: str,
        interval: str,
        df: pd.DataFrame,
        metadata: dict,
    ) -> None:
        """Save DataFrame and metadata to disk.

        Args:
            pair: Trading pair name.
            interval: Candle interval.
            df: OHLCV DataFrame.
            metadata: Metadata dictionary.
        """
        # Create directories
        raw_dir = Path("data/raw") / pair.lower()
        meta_dir = Path("data/metadata")
        raw_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        # Save parquet
        parquet_path = raw_dir / f"{interval}_{pair.lower()}.parquet"
        df.to_parquet(parquet_path, index=True)
        logger.info(f"Saved raw data to {parquet_path}")

        # Save metadata
        meta_path = meta_dir / f"{pair.lower()}_{interval}_meta.yaml"
        with open(meta_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)
        logger.info(f"Saved metadata to {meta_path}")


def download_pair(
    pair: str,
    interval: str,
    lookback_days: Optional[int] = None,
) -> tuple[pd.DataFrame, dict]:
    """Convenience function to download data for a single pair.

    Args:
        pair: Base currency (e.g., 'BTC').
        interval: Candle interval (e.g., '1h', '1d').
        lookback_days: Number of days to look back.

    Returns:
        Tuple of (DataFrame with OHLCV data, metadata dict).
    """
    downloader = MarketDataDownloader()
    return downloader.download_pair(pair, interval, lookback_days)


def download_all_pairs(
    pairs: list[str],
    interval: str,
) -> dict[str, pd.DataFrame]:
    """Convenience function to download data for multiple pairs.

    Args:
        pairs: List of base currencies.
        interval: Candle interval.

    Returns:
        Dictionary mapping pair name to DataFrame.
    """
    downloader = MarketDataDownloader()
    return downloader.download_all_pairs(pairs, interval)

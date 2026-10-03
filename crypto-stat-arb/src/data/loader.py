"""Data loader for reading local parquet files."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Load OHLCV data from local parquet files.

    Provides methods to load single pairs or multiple pairs aligned
    on timestamp index for statistical analysis.
    """

    def __init__(self, raw_dir: str | Path = "data/raw"):
        """Initialize the data loader.

        Args:
            raw_dir: Path to the data/raw/ directory.
        """
        self.raw_dir = Path(raw_dir)

    def load_pair(
        self,
        pair: str,
        interval: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Load data for a single pair.

        Args:
            pair: Base currency (e.g., 'BTC').
            interval: Candle interval (e.g., '1h', '1d').
            start: Optional start timestamp for filtering.
            end: Optional end timestamp for filtering.

        Returns:
            DataFrame with OHLCV data indexed by timestamp.
        """
        parquet_path = self.raw_dir / pair.lower() / f"{interval}_{pair.lower()}.parquet"

        if not parquet_path.exists():
            logger.error(f"File not found: {parquet_path}")
            return pd.DataFrame()

        df = pd.read_parquet(parquet_path)

        # Ensure timestamp is datetime and UTC
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        elif isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)

        # Filter by date range if specified
        if start is not None:
            start = pd.to_datetime(start, utc=True)
            df = df[df.index >= start]
        if end is not None:
            end = pd.to_datetime(end, utc=True)
            df = df[df.index <= end]

        logger.info(
            f"Loaded {len(df)} rows for {pair} {interval} "
            f"({df.index.min()} to {df.index.max()})"
        )

        return df

    def load_pairs(
        self,
        pairs: list[str],
        interval: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> dict[str, pd.DataFrame]:
        """Load data for multiple pairs, aligned on timestamp.

        Args:
            pairs: List of base currencies.
            interval: Candle interval.
            start: Optional start timestamp.
            end: Optional end timestamp.

        Returns:
            Dictionary mapping pair name to DataFrame, all indexed by timestamp.
        """
        result = {}
        min_start = None
        max_end = None

        for pair in pairs:
            df = self.load_pair(pair, interval, start, end)
            if not df.empty:
                result[pair] = df

                # Track overall date range
                if min_start is None or df.index.min() < min_start:
                    min_start = df.index.min()
                if max_end is None or df.index.max() > max_end:
                    max_end = df.index.max()

        logger.info(
            f"Loaded {len(result)} pairs for interval {interval}: "
            f"{list(result.keys())}"
        )

        if min_start is not None and max_end is not None:
            logger.info(f"Combined date range: {min_start} to {max_end}")

        return result

    def list_available_pairs(self) -> list[str]:
        """List all pairs available in the raw data directory.

        Returns:
            List of pair names (directory names).
        """
        if not self.raw_dir.exists():
            logger.warning(f"Raw data directory not found: {self.raw_dir}")
            return []

        pairs = [
            d.name for d in self.raw_dir.iterdir()
            if d.is_dir() and (d / "*.parquet").exists()
        ]
        logger.info(f"Available pairs: {pairs}")
        return pairs

    def list_available_intervals(self, pair: str) -> list[str]:
        """List all intervals available for a pair.

        Args:
            pair: Base currency name.

        Returns:
            List of interval strings.
        """
        pair_dir = self.raw_dir / pair.lower()
        if not pair_dir.exists():
            logger.warning(f"Pair directory not found: {pair_dir}")
            return []

        intervals = []
        for f in pair_dir.glob("*.parquet"):
            # Filename format: {interval}_{pair}.parquet
            interval = f.stem.replace(f"_{pair.lower()}", "")
            intervals.append(interval)

        intervals.sort()
        logger.info(f"Available intervals for {pair}: {intervals}")
        return intervals

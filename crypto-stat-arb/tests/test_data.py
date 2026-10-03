"""Unit tests for data module (downloader, validator, loader)."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os

from src.data.validator import validate_ohlcv, check_data_integrity
from src.data.loader import DataLoader


class TestValidator:
    """Tests for validator.py functions."""

    def test_validate_ohlcv_valid(self, synthetic_ohlcv):
        """Test validation of valid OHLCV data."""
        result = validate_ohlcv(synthetic_ohlcv)

        assert result["has_required_columns"] == True
        assert result["no_negative_prices"] == True
        assert result["no_null_ohlc"] == True
        assert result["monotonic_timestamps"] == True
        assert result["duplicate_timestamps"] == False

    def test_validate_ohlcv_missing_columns(self):
        """Test validation fails for missing columns."""
        df = pd.DataFrame({"open": [100], "high": [105]})
        result = validate_ohlcv(df)

        assert result["has_required_columns"] == False

    def test_validate_ohlcv_negative_prices(self, synthetic_ohlcv):
        """Test validation detects negative prices."""
        df = synthetic_ohlcv.copy()
        df.loc[0, "close"] = -100
        result = validate_ohlcv(df)

        assert result["no_negative_prices"] == False

    def test_validate_ohlcv_null_values(self, synthetic_ohlcv):
        """Test validation detects null OHLC values."""
        df = synthetic_ohlcv.copy()
        df.loc[0, "close"] = np.nan
        result = validate_ohlcv(df)

        assert result["no_null_ohlc"] == False

    def test_validate_ohlcv_empty(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()
        result = validate_ohlcv(df)

        assert result["has_required_columns"] == False

    def test_check_data_integrity(self, tmp_path):
        """Test integrity check on temporary directory."""
        # Create test data directory structure
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        # Create a test parquet file
        pair_dir = raw_dir / "btc"
        pair_dir.mkdir()
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="1h"),
            "open": np.random.randn(100) * 100 + 50000,
            "high": np.random.randn(100) * 100 + 51000,
            "low": np.random.randn(100) * 100 + 49000,
            "close": np.random.randn(100) * 100 + 50000,
            "volume": np.random.randint(1000, 10000, 100),
        })
        df.to_parquet(pair_dir / "1h_btc.parquet", index=False)

        result = check_data_integrity(raw_dir)

        assert result["total_files"] == 1
        assert result["valid_files"] == 1
        assert result["invalid_files"] == 0


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_load_pair_file_not_found(self, tmp_path):
        """Test loading non-existent file returns empty DataFrame."""
        loader = DataLoader(raw_dir=tmp_path)
        result = loader.load_pair("BTC", "1h")

        assert result.empty

    def test_load_pair_success(self, tmp_path):
        """Test successful loading of pair data."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)

        pair_dir = raw_dir / "btc"
        pair_dir.mkdir()

        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="1h"),
            "open": np.random.randn(100) * 100 + 50000,
            "high": np.random.randn(100) * 100 + 51000,
            "low": np.random.randn(100) * 100 + 49000,
            "close": np.random.randn(100) * 100 + 50000,
            "volume": np.random.randint(1000, 10000, 100),
        })
        df.to_parquet(pair_dir / "1h_btc.parquet", index=False)

        loader = DataLoader(raw_dir=raw_dir)
        result = loader.load_pair("BTC", "1h")

        assert len(result) == 100
        assert "close" in result.columns

    def test_load_pairs_multiple(self, tmp_path):
        """Test loading multiple pairs."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)

        for pair in ["btc", "eth"]:
            pair_dir = raw_dir / pair
            pair_dir.mkdir()

            df = pd.DataFrame({
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="1h"),
                "open": np.random.randn(100) * 100 + 50000,
                "high": np.random.randn(100) * 100 + 51000,
                "low": np.random.randn(100) * 100 + 49000,
                "close": np.random.randn(100) * 100 + 50000,
                "volume": np.random.randint(1000, 10000, 100),
            })
            df.to_parquet(pair_dir / "1h_btc.parquet", index=False)

        loader = DataLoader(raw_dir=raw_dir)
        result = loader.load_pairs(["btc", "eth"], "1h")

        assert len(result) == 2
        assert "btc" in result
        assert "eth" in result

    def test_load_pair_with_date_filter(self, tmp_path):
        """Test loading with date range filter."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)

        pair_dir = raw_dir / "btc"
        pair_dir.mkdir()

        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="1h"),
            "open": np.random.randn(100) * 100 + 50000,
            "high": np.random.randn(100) * 100 + 51000,
            "low": np.random.randn(100) * 100 + 49000,
            "close": np.random.randn(100) * 100 + 50000,
            "volume": np.random.randint(1000, 10000, 100),
        })
        df.to_parquet(pair_dir / "1h_btc.parquet", index=False)

        loader = DataLoader(raw_dir=raw_dir)
        result = loader.load_pair(
            "BTC", "1h",
            start=pd.Timestamp("2023-01-10"),
            end=pd.Timestamp("2023-01-20"),
        )

        assert len(result) < 100


@pytest.mark.network
class TestDownloader:
    """Tests for MarketDataDownloader class."""

    @patch("src.data.downloader.requests.get")
    def test_download_pair_success(self, mock_get):
        """Test successful data download."""
        # Create mock response
        mock_response = MagicMock()
        mock_response.json.return_value = [
            [
                1672531200000,  # open time
                "50000.0", "51000.0", "49000.0", "50500.0", "1000.0",
                1672534799999, "50000000.0", 100, "500.0", "25000.0", "0"
            ]
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        from src.data.downloader import MarketDataDownloader

        downloader = MarketDataDownloader(rate_limit_delay_ms=0)
        df, metadata = downloader.download_pair("BTC", "1h", lookback_days=1)

        assert len(df) == 1
        assert "close" in df.columns
        assert metadata["pair"] == "BTC"

    @patch("src.data.downloader.requests.get")
    def test_download_pair_api_error(self, mock_get):
        """Test download handles API errors gracefully."""
        import requests
        mock_get.side_effect = requests.RequestException("API Error")

        from src.data.downloader import MarketDataDownloader

        downloader = MarketDataDownloader(rate_limit_delay_ms=0, max_retries=1)
        df, metadata = downloader.download_pair("BTC", "1h", lookback_days=1)

        assert df.empty

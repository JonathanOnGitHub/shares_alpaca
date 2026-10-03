"""Data validation utilities for OHLCV data."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def validate_ohlcv(df: pd.DataFrame) -> dict[str, Any]:
    """Validate OHLCV DataFrame for data quality issues.

    Args:
        df: DataFrame with at least timestamp, open, high, low, close columns.

    Returns:
        Dictionary with validation results:
        - has_required_columns: bool
        - no_negative_prices: bool
        - no_null_ohlc: bool
        - monotonic_timestamps: bool
        - duplicate_timestamps: bool
        - gap_count: int
        - gap_periods: list of dict
    """
    result = {
        "has_required_columns": False,
        "no_negative_prices": False,
        "no_null_ohlc": False,
        "monotonic_timestamps": False,
        "duplicate_timestamps": False,
        "gap_count": 0,
        "gap_periods": [],
    }

    required_cols = {"timestamp", "open", "high", "low", "close"}
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided to validate_ohlcv")
        return result

    # Check required columns
    if required_cols.issubset(df.columns):
        result["has_required_columns"] = True

    # Check negative prices
    price_cols = ["open", "high", "low", "close"]
    if all(col in df.columns for col in price_cols):
        has_negative = (df[price_cols] <= 0).any().any()
        result["no_negative_prices"] = not has_negative

    # Check null values in OHLC
    if all(col in df.columns for col in price_cols):
        null_count = df[price_cols].isnull().sum().sum()
        result["no_null_ohlc"] = null_count == 0

    # Check timestamps
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        result["monotonic_timestamps"] = ts.is_monotonic_increasing
        result["duplicate_timestamps"] = ts.duplicated().any()

        # Detect gaps
        if len(ts) > 1:
            diffs = ts.diff()
            # Expected interval - assume 1h candles for detection
            # This is a simplified gap detector
            median_diff = diffs.median()
            large_gaps = diffs[diffs > median_diff * 5]
            result["gap_count"] = len(large_gaps)
            if len(large_gaps) > 0:
                gap_indices = large_gaps.index[1:]  # skip first NaT
                for idx in gap_indices:
                    result["gap_periods"].append({
                        "start": str(ts[idx - 1]),
                        "end": str(ts[idx]),
                        "gap_hours": float(diffs[idx].total_seconds() / 3600),
                    })

    logger.info(
        f"Validation for {len(df)} rows: "
        f"cols={result['has_required_columns']}, "
        f"prices={result['no_negative_prices']}, "
        f"nulls={result['no_null_ohlc']}, "
        f"gaps={result['gap_count']}"
    )

    return result


def check_data_integrity(raw_dir: str | Path) -> dict[str, Any]:
    """Scan all saved parquet files and report integrity status.

    Args:
        raw_dir: Path to the data/raw/ directory.

    Returns:
        Dictionary with overall integrity status and per-file details.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        logger.error(f"Raw data directory does not exist: {raw_path}")
        return {"error": f"Directory not found: {raw_path}"}

    report = {
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "files": [],
    }

    parquet_files = list(raw_path.rglob("*.parquet"))
    report["total_files"] = len(parquet_files)

    for pq_file in parquet_files:
        try:
            df = pd.read_parquet(pq_file)
            validation = validate_ohlcv(df)

            file_report = {
                "file": str(pq_file),
                "n_rows": len(df),
                "validation": validation,
                "status": "valid" if all([
                    validation["has_required_columns"],
                    validation["no_negative_prices"],
                    validation["no_null_ohlc"],
                ]) else "invalid",
            }
            report["files"].append(file_report)

            if file_report["status"] == "valid":
                report["valid_files"] += 1
            else:
                report["invalid_files"] += 1

        except Exception as e:
            logger.error(f"Failed to read {pq_file}: {e}")
            report["files"].append({
                "file": str(pq_file),
                "status": "error",
                "error": str(e),
            })
            report["invalid_files"] += 1

    # Also check metadata files
    meta_files = list(Path("data/metadata").rglob("*.yaml"))
    report["metadata_files"] = len(meta_files)

    logger.info(
        f"Integrity check complete: {report['valid_files']}/{report['total_files']} valid files"
    )

    return report

#!/usr/bin/env python3
"""Current market analysis script.

Outputs current state of all 6 pairs:
- z-score
- half-life
- cointegration status
- checks all risk limits

This is RESEARCH MODE ONLY - no live execution.

Usage:
    python scripts/current_market_analysis.py
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.downloader import MarketDataDownloader
from src.data.loader import DataLoader
from src.risk.limits import RiskManager
from src.statistics.cointegration import engle_granger_test
from src.statistics.halflife import estimate_halflife
from src.statistics.moments import pearson_correlation, spearman_correlation
from src.statistics.pairwise import compute_hedge_ratio_ols, compute_log_prices, compute_spread, static_beta
from src.statistics.zscore import compute_zscore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# All 6 pair combinations
PAIR_COMBINATIONS = [
    ("BTC", "XRP"),
    ("BTC", "LINK"),
    ("BTC", "SOL"),
    ("XRP", "LINK"),
    ("XRP", "SOL"),
    ("LINK", "SOL"),
]


def load_config() -> dict:
    """Load default configuration."""
    config_path = project_root / "config" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def analyze_pair(asset_a: str, asset_b: str, config: dict) -> dict:
    """Analyze a single pair.

    Args:
        asset_a: First asset.
        asset_b: Second asset.
        config: Configuration dict.

    Returns:
        Dict with analysis results.
    """
    pair_name = f"{asset_a}/{asset_b}"
    logger.info(f"Analyzing {pair_name}...")

    # Download latest data
    downloader = MarketDataDownloader()
    lookback_days = config.get("lookback_days", {}).get("1d", 90)

    df_a, meta_a = downloader.download_pair(asset_a, "1d", lookback_days)
    df_b, meta_b = downloader.download_pair(asset_b, "1d", lookback_days)

    if df_a.empty or df_b.empty:
        logger.warning(f"No data available for {pair_name}")
        return {"pair": pair_name, "status": "NO_DATA"}

    # Compute log prices
    log_prices = compute_log_prices(df_a, df_b)
    if log_prices.empty:
        return {"pair": pair_name, "status": "INSUFFICIENT_DATA"}

    log_a = log_prices["log_price_a"]
    log_b = log_prices["log_price_b"]

    # Compute beta
    beta = static_beta(log_a, log_b)

    # Compute spread
    spread = compute_spread(log_a, log_b, beta)

    # Compute z-score
    zscore_lookback = config.get("analysis", {}).get("zscore_lookback", 30)
    zscore_result = compute_zscore(spread, zscore_lookback)
    zscore = zscore_result["z"].iloc[-1] if len(zscore_result) > 0 else 0.0

    # Compute correlations
    pearson_corr, _ = pearson_correlation(log_a, log_b)
    spearman_corr, _ = spearman_correlation(log_a, log_b)

    # Cointegration test
    eg_result = engle_granger_test(log_a, log_b)
    cointegration_pvalue = eg_result.get("p_value")
    is_cointegrated = cointegration_pvalue < 0.05 if cointegration_pvalue else False

    # Half-life
    spread_returns = spread.diff().dropna()
    halflife = estimate_halflife(spread_returns)

    # Rolling cointegration stability
    rolling_coint_stable = False
    if len(spread) > 120:
        from src.statistics.cointegration import rolling_cointegration
        rolling_coint = rolling_cointegration(df_a, df_b, window=120)
        if len(rolling_coint) > 0:
            stable_ratio = rolling_coint["is_cointegrated"].sum() / len(rolling_coint)
            rolling_coint_stable = stable_ratio > 0.7  # 70% of windows show cointegration

    # Current prices
    current_price_a = df_a["close"].iloc[-1]
    current_price_b = df_b["close"].iloc[-1]

    # Determine signal
    entry_threshold = config.get("entry_thresholds", [2.0])[0] if isinstance(config.get("entry_thresholds"), list) else 2.0

    if zscore > entry_threshold:
        signal = f"SHORT {asset_a} / LONG {asset_b}"
    elif zscore < -entry_threshold:
        signal = f"LONG {asset_a} / SHORT {asset_b}"
    else:
        signal = "NO SIGNAL"

    # Confidence level
    if abs(zscore) > 3.0:
        confidence = "HIGH"
    elif abs(zscore) > 2.0:
        confidence = "MEDIUM"
    elif abs(zscore) > 1.5:
        confidence = "LOW"
    else:
        confidence = "NONE"

    result = {
        "pair": pair_name,
        "status": "ANALYZED",
        "zscore": float(zscore) if not pd.isna(zscore) else None,
        "halflife": float(halflife) if halflife and not pd.isna(halflife) else None,
        "cointegration": "PASS" if is_cointegrated else "FAIL",
        "rolling_stability": "PASS" if rolling_coint_stable else "FAIL",
        "signal": signal,
        "confidence": confidence,
        "pearson_correlation": float(pearson_corr) if pearson_corr and not pd.isna(pearson_corr) else None,
        "spearman_correlation": float(spearman_corr) if spearman_corr and not pd.isna(spearman_corr) else None,
        "beta": float(beta) if beta and not pd.isna(beta) else None,
        "current_price_a": float(current_price_a),
        "current_price_b": float(current_price_b),
        "data_end": str(df_a.index[-1]) if len(df_a) > 0 else None,
        "n_observations": len(df_a),
    }

    return result


def check_risk_limits(pair_result: dict, config: dict) -> dict:
    """Check risk limits for current state.

    Args:
        pair_result: Analysis result dict.
        config: Configuration dict.

    Returns:
        Dict with risk check results.
    """
    risk_manager = RiskManager(config.get("risk_limits", {}))

    # Build market data
    market_data = {
        "zscore": pair_result.get("zscore", 0),
        "pair_name": pair_result["pair"],
        "prices": {
            "a": pair_result.get("current_price_a", 0),
            "b": pair_result.get("current_price_b", 0),
        },
    }

    # Check risk limits
    approved, reason = risk_manager.check_signal(
        signal=1 if "LONG" in pair_result.get("signal", "") else -1 if "SHORT" in pair_result.get("signal", "") else 0,
        current_positions={},
        portfolio_equity=config.get("backtest", {}).get("initial_capital", 100000.0),
        market_data=market_data,
    )

    return {
        "approved": approved,
        "reason": reason,
        "limits": risk_manager.get_risk_summary(
            {},
            config.get("backtest", {}).get("initial_capital", 100000.0)
        ),
    }


def format_report(all_results: list, config: dict) -> str:
    """Format analysis results as a readable report.

    Args:
        all_results: List of analysis result dicts.
        config: Configuration dict.

    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 70,
        "CRYPTO STATISTICAL ARBITRAGE - CURRENT MARKET ANALYSIS",
        "=" * 70,
        "",
        "MODE: RESEARCH ONLY - NO LIVE EXECUTION",
        "",
        "=" * 70,
    ]

    for result in all_results:
        if result.get("status") == "NO_DATA":
            lines.append(f"\n{result['pair']}: NO DATA AVAILABLE")
            continue
        if result.get("status") == "INSUFFICIENT_DATA":
            lines.append(f"\n{result['pair']}: INSUFFICIENT DATA")
            continue

        lines.extend([
            "",
            f"{result['pair']}",
            "-" * 40,
            f"  z-score:           {result.get('zscore', 'N/A'):.2f}" if result.get('zscore') else "  z-score:           N/A",
            f"  half-life:         {result.get('halflife', 'N/A'):.1f} bars" if result.get('halflife') else "  half-life:         N/A",
            f"  cointegration:      {result.get('cointegration', 'N/A')}",
            f"  rolling stability:  {result.get('rolling_stability', 'N/A')}",
            f"  signal:            {result.get('signal', 'N/A')}",
            f"  confidence:        {result.get('confidence', 'N/A')}",
            "",
            f"  Pearson corr:      {result.get('pearson_correlation', 0):.4f}" if result.get('pearson_correlation') else "  Pearson corr:      N/A",
            f"  Spearman corr:     {result.get('spearman_correlation', 0):.4f}" if result.get('spearman_correlation') else "  Spearman corr:     N/A",
            f"  Beta:              {result.get('beta', 0):.4f}" if result.get('beta') else "  Beta:              N/A",
            "",
            f"  Price A:           ${result.get('current_price_a', 0):.4f}",
            f"  Price B:           ${result.get('current_price_b', 0):.4f}",
            f"  Data through:      {result.get('data_end', 'N/A')[:10]}" if result.get('data_end') else "  Data through:      N/A",
            f"  Observations:      {result.get('n_observations', 0)}",
        ])

        # Risk check
        risk = check_risk_limits(result, config)
        lines.extend([
            "",
            f"  Risk check:        {'APPROVED' if risk['approved'] else 'REJECTED'} - {risk['reason']}",
        ])

    lines.extend([
        "",
        "=" * 70,
        "INTERPRETATION GUIDE",
        "=" * 70,
        "",
        "z-score: Measures how far the spread is from its mean.",
        "  |z| > 2.0: Potential entry signal",
        "  |z| > 3.0: Strong signal",
        "",
        "half-life: Estimated time for spread to revert halfway to mean.",
        "  Lower values suggest faster mean reversion",
        "",
        "cointegration: Whether the spread is stationary.",
        "  PASS: Spread is mean-reverting",
        "  FAIL: Spread may not be tradeable",
        "",
        "rolling stability: Consistency of cointegration over time.",
        "  PASS: Cointegration stable across rolling windows",
        "",
        "signal: Current trading signal based on z-score.",
        "  Only act on signals when cointegration and stability both PASS",
        "",
        "confidence: Strength of the statistical signal.",
        "  HIGH: |z| > 3.0",
        "  MEDIUM: |z| > 2.0",
        "  LOW: |z| > 1.5",
        "  NONE: |z| < 1.5",
        "",
        "=" * 70,
        "DISCLAIMER",
        "=" * 70,
        "This analysis is for RESEARCH purposes only.",
        "No live trades should be executed based on this output.",
        "Past statistical relationships do not guarantee future results.",
        "=" * 70,
    ])

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Current market analysis for crypto pairs")
    parser.add_argument("--pairs", nargs="+", help="Specific pairs to analyze (e.g., BTC XRP)")
    parser.add_argument("--output", type=str, help="Output file path")

    args = parser.parse_args()

    # Load config
    config = load_config()

    # Determine pairs to analyze
    if args.pairs and len(args.pairs) >= 2:
        # Specific pairs provided
        pairs_to_analyze = [(args.pairs[0], args.pairs[1])]
    else:
        pairs_to_analyze = PAIR_COMBINATIONS

    logger.info(f"Analyzing {len(pairs_to_analyze)} pairs...")

    # Analyze each pair
    all_results = []
    for asset_a, asset_b in pairs_to_analyze:
        try:
            result = analyze_pair(asset_a, asset_b, config)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error analyzing {asset_a}/{asset_b}: {e}", exc_info=True)
            all_results.append({
                "pair": f"{asset_a}/{asset_b}",
                "status": "ERROR",
                "error": str(e),
            })

    # Format and print report
    report = format_report(all_results, config)
    print(report)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {output_path}")

    # Also save JSON for programmatic use
    json_path = Path("reports") / f"current_market_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"JSON results saved to {json_path}")


if __name__ == "__main__":
    main()

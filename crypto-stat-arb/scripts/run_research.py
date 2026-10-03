#!/usr/bin/env python3
"""Main research entry point for crypto statistical arbitrage research.

This script runs the full walk-forward validation for pair trading strategies.

Usage:
    python scripts/run_research.py --pair BTC/XRP --timeframe 1d --cost-scenario base
    python scripts/run_research.py --all-pairs --all-timeframes --all-cost-scenarios
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtest.walkforward import WalkForwardValidator
from src.data.downloader import MarketDataDownloader
from src.data.loader import DataLoader
from src.reporting.metrics import compute_all_metrics
from src.reporting.plots import (
    plot_drawdown,
    plot_equity_curve,
    plot_monthly_returns,
    plot_trade_distribution,
    plot_zscore,
)
from src.reporting.report_builder import classify_pair, generate_pair_report, generate_summary_report, save_experiment_json
from src.risk.position_sizing import (
    beta_neutral_hedge,
    equal_dollar_hedge,
    ols_hedge_ratio,
    volatility_adjusted_hedge,
)
from src.statistics.cointegration import engle_granger_test, rolling_cointegration
from src.statistics.halflife import estimate_halflife
from src.statistics.moments import pearson_correlation, rolling_correlation, spearman_correlation
from src.statistics.pairwise import (
    compute_hedge_ratio_ols,
    compute_log_prices,
    compute_spread,
    static_beta,
)
from src.statistics.zscore import compute_zscore, historical_max_zscore, mean_reversion_excursions

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

# Timeframe intervals
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

# Cost scenarios
COST_SCENARIOS = ["optimistic", "base", "pessimistic"]

# Beta lookbacks
BETA_LOOKBACKS = ["static", 30, 60, 90, 180]

# Entry thresholds
ENTRY_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]


def load_config() -> dict:
    """Load default configuration."""
    config_path = project_root / "config" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_data(pair: tuple, timeframe: str, lookback_days: int = None) -> dict:
    """Download data for a pair."""
    downloader = MarketDataDownloader()
    asset_a, asset_b = pair

    data_a, meta_a = downloader.download_pair(asset_a, timeframe, lookback_days)
    data_b, meta_b = downloader.download_pair(asset_b, timeframe, lookback_days)

    if data_a.empty or data_b.empty:
        logger.error(f"No data downloaded for {asset_a}/{asset_b}")
        return {}

    # Create aligned dataset
    data = {
        "a": data_a,
        "b": data_b,
        "pair_name": f"{asset_a}/{asset_b}",
    }

    return data


def run_pair_analysis(
    pair: tuple,
    timeframe: str,
    cost_scenario: str,
    config: dict,
) -> dict:
    """Run complete analysis for a single pair.

    Args:
        pair: Tuple of (asset_a, asset_b).
        timeframe: Data timeframe.
        cost_scenario: Cost scenario to use.
        config: Configuration dict.

    Returns:
        Dict with analysis results.
    """
    asset_a, asset_b = pair
    pair_name = f"{asset_a}/{asset_b}"

    logger.info(f"Running analysis for {pair_name} ({timeframe}) with {cost_scenario} costs")

    # Determine lookback days from config
    lookback_days = config.get("lookback_days", {}).get(timeframe, 365)

    # Download data
    data = download_data(pair, timeframe, lookback_days)
    if not data:
        return {}

    df_a = data["a"]
    df_b = data["b"]

    # Compute log prices
    log_prices = compute_log_prices(df_a, df_b)

    # Compute static beta
    beta = static_beta(log_prices["log_price_a"], log_prices["log_price_b"])

    # Compute spread
    spread = compute_spread(log_prices["log_price_a"], log_prices["log_price_b"], beta)

    # Compute z-score
    zscore_lookback = config.get("analysis", {}).get("zscore_lookback", 30)
    zscore_result = compute_zscore(spread, zscore_lookback)
    zscore = zscore_result["z"]

    # Compute correlation
    pearson_corr, pearson_pval = pearson_correlation(
        log_prices["log_price_a"], log_prices["log_price_b"]
    )
    spearman_corr, spearman_pval = spearman_correlation(
        log_prices["log_price_a"], log_prices["log_price_b"]
    )

    # Rolling correlations
    corr_windows = config.get("analysis", {}).get("correlation_windows", [30, 60, 90])
    rolling_corrs = {}
    for window in corr_windows:
        rolling_corrs[window] = rolling_correlation(
            log_prices["log_price_a"], log_prices["log_price_b"], window
        )

    # Cointegration test
    eg_result = engle_granger_test(
        log_prices["log_price_a"], log_prices["log_price_b"]
    )

    # Rolling cointegration
    cointegration_window = config.get("analysis", {}).get("cointegration_window", 120)
    rolling_coint = rolling_cointegration(df_a, df_b, cointegration_window)

    # Half-life estimation
    spread_returns = spread.diff().dropna()
    halflife = estimate_halflife(spread_returns)

    # Historical max z-score
    max_zscore = historical_max_zscore(zscore)

    # Mean reversion excursions
    excursion_stats = mean_reversion_excursions(zscore, threshold=2.0)

    # Build pair data for backtest
    pair_data = {
        pair_name: pd.DataFrame({
            "close_a": df_a["close"],
            "close_b": df_b["close"],
            "zscore": zscore,
            "spread": spread,
            "beta": beta,
        }, index=df_a.index)
    }

    # Walk-forward validation parameters
    wf_config = config.get("walk_forward", {})
    walk_forward_params = {
        "entry_thresholds": ENTRY_THRESHOLDS,
        "exit_types": ["z0", "z025", "z050"],
        "beta_lookbacks": BETA_LOOKBACKS,
        "hedge_type": config.get("backtest", {}).get("position_sizing", "equal_dollar"),
        "cost_scenario": cost_scenario,
        "zscore_lookback": zscore_lookback,
        "initial_capital": config.get("backtest", {}).get("initial_capital", 100000.0),
        "max_holding_period_bars": config.get("risk_limits", {}).get("max_holding_period_bars", 100),
        "stop_loss_z": config.get("risk_limits", {}).get("stop_loss_z", 3.0),
        "train_days": wf_config.get("train_days", 180),
        "test_days": wf_config.get("test_days", 30),
        "roll_days": wf_config.get("roll_days", 30),
    }

    # Run walk-forward validation
    validator = WalkForwardValidator(
        train_days=walk_forward_params["train_days"],
        test_days=walk_forward_params["test_days"],
        roll_days=walk_forward_params["roll_days"],
    )

    wf_result = validator.run(pair_data, walk_forward_params)

    # Generate charts
    charts_dir = Path("reports") / pair_name.replace("/", "_")
    charts_dir.mkdir(parents=True, exist_ok=True)

    if wf_result.window_results:
        # Use first window's backtest for charts
        first_result = wf_result.window_results[0]
        equity_curve = first_result.backtest_result.equity_curve
        trades = first_result.backtest_result.trades

        if len(equity_curve) > 0:
            plot_equity_curve(
                equity_curve,
                title=f"{pair_name} Equity Curve ({cost_scenario})",
                save_path=str(charts_dir / f"equity_curve_{cost_scenario}.png")
            )

            if len(trades) > 0:
                trade_dicts = [
                    {
                        "net_pnl": t.net_pnl,
                        "gross_pnl": t.gross_pnl,
                        "notional": t.notional,
                        "holding_period_bars": t.holding_period_bars,
                    }
                    for t in trades
                ]

                plot_trade_distribution(
                    trade_dicts,
                    title=f"{pair_name} Trade Distribution ({cost_scenario})",
                    save_path=str(charts_dir / f"trade_distribution_{cost_scenario}.png")
                )

    # Compile results
    result = {
        "pair": pair_name,
        "timeframe": timeframe,
        "cost_scenario": cost_scenario,
        "analysis": {
            "pearson_correlation": pearson_corr,
            "pearson_pvalue": pearson_pval,
            "spearman_correlation": spearman_corr,
            "spearman_pvalue": spearman_pval,
            "beta": beta,
            "halflife": halflife,
            "cointegration_pvalue": eg_result.get("p_value"),
            "cointegration_conclusion": eg_result.get("conclusion"),
            "max_zscore": max_zscore,
            "excursion_stats": excursion_stats,
        },
        "walk_forward": {
            "aggregate_metrics": wf_result.aggregate_metrics,
            "experiment_metadata": wf_result.experiment_metadata,
        },
        "metrics": wf_result.aggregate_metrics,
    }

    # Add classification
    sharpe = wf_result.aggregate_metrics.get("avg_sharpe_ratio", 0)
    net_return = wf_result.aggregate_metrics.get("avg_net_return", 0)
    max_dd = wf_result.aggregate_metrics.get("max_drawdown", 0)
    classification_thresholds = config.get("classification", {})

    result["classification"] = classify_pair(
        sharpe, net_return, max_dd, 0.5,  # win_rate not available at aggregate level
        robust_sharpe_min=classification_thresholds.get("robust_sharpe_min", 1.0),
        robust_pvalue_max=classification_thresholds.get("robust_pvalue_max", 0.10),
    )

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run crypto stat-arb research")
    parser.add_argument("--pair", type=str, help="Pair to analyze (e.g., BTC/XRP)")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe (e.g., 1d, 1h)")
    parser.add_argument("--cost-scenario", type=str, default="base", help="Cost scenario")
    parser.add_argument("--all-pairs", action="store_true", help="Run all pairs")
    parser.add_argument("--all-timeframes", action="store_true", help="Run all timeframes")
    parser.add_argument("--all-cost-scenarios", action="store_true", help="Run all cost scenarios")
    parser.add_argument("--output", type=str, help="Output file path")

    args = parser.parse_args()

    # Load config
    config = load_config()

    # Determine what to run
    pairs_to_run = []
    timeframes_to_run = []
    cost_scenarios_to_run = []

    if args.all_pairs:
        pairs_to_run = PAIR_COMBINATIONS
    elif args.pair:
        # Parse pair like "BTC/XRP"
        if "/" in args.pair:
            asset_a, asset_b = args.pair.split("/")
            pairs_to_run = [(asset_a, asset_b)]
        else:
            pairs_to_run = [(args.pair, "USDT")]
    else:
        pairs_to_run = [PAIR_COMBINATIONS[0]]  # Default to first pair

    if args.all_timeframes:
        timeframes_to_run = TIMEFRAMES
    else:
        timeframes_to_run = [args.timeframe]

    if args.all_cost_scenarios:
        cost_scenarios_to_run = COST_SCENARIOS
    else:
        cost_scenarios_to_run = [args.cost_scenario]

    logger.info(f"Running research: {len(pairs_to_run)} pairs x {len(timeframes_to_run)} timeframes x {len(cost_scenarios_to_run)} cost scenarios")

    # Run analysis
    all_results = []

    for pair in pairs_to_run:
        for timeframe in timeframes_to_run:
            for cost_scenario in cost_scenarios_to_run:
                try:
                    result = run_pair_analysis(pair, timeframe, cost_scenario, config)
                    if result:
                        all_results.append(result)

                        # Generate pair report
                        pair_name = result["pair"]
                        charts_dir = f"reports/{pair_name.replace('/', '_')}"

                        generate_pair_report(result, config, charts_dir)

                        logger.info(f"Completed {pair_name} {timeframe} {cost_scenario}: Sharpe={result.get('metrics', {}).get('avg_sharpe_ratio', 0):.2f}, Class={result.get('classification', 'N/A')}")

                except Exception as e:
                    logger.error(f"Error analyzing {pair} {timeframe} {cost_scenario}: {e}", exc_info=True)
                    continue

    # Generate summary report
    if all_results:
        summary_path = generate_summary_report(all_results, config)
        logger.info(f"Generated summary report: {summary_path}")

        # Save experiment JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if args.output:
            exp_path = args.output
        else:
            pair_str = "_".join([p[0] + p[1] for p in pairs_to_run]) if len(pairs_to_run) <= 2 else f"{len(pairs_to_run)}pairs"
            exp_path = f"reports/{pair_str}_{timeframes_to_run[0]}_{timestamp}.json"

        save_experiment_json({
            "results": all_results,
            "config": config,
            "timestamp": timestamp,
        }, exp_path)

        logger.info(f"Saved experiment to {exp_path}")

    else:
        logger.warning("No results generated")
        sys.exit(1)

    logger.info("Research complete!")


if __name__ == "__main__":
    main()

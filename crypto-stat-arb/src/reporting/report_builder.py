"""Report generation for pair trading research results.

This module generates markdown and JSON reports for backtest results.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def generate_pair_report(
    pair_result: Dict[str, Any],
    config: Dict[str, Any],
    charts_dir: str = "reports",
) -> str:
    """Generate a detailed markdown report for a single pair.

    Args:
        pair_result: Dict containing pair backtest results with keys:
            - pair: Pair name (e.g., 'BTC/XRP')
            - timeframe: Timeframe used
            - metrics: Performance metrics dict
            - trades: List of trade dicts
            - equity_curve: Series of equity values
            - window_results: Optional list of walk-forward window results
        config: Configuration dict used for the experiment.
        charts_dir: Directory where charts are saved.

    Returns:
        Path to the generated markdown report.
    """
    pair_name = pair_result.get("pair", "UNKNOWN")
    timeframe = pair_result.get("timeframe", "1d")
    metrics = pair_result.get("metrics", {})

    # Build report content
    report_lines = [
        f"# Pair Trading Research Report: {pair_name}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Timeframe:** {timeframe}",
        "",
        "## Performance Summary",
        "",
    ]

    # Key metrics table
    report_lines.extend([
        "| Metric | Value |",
        "|--------|-------|",
        f"| Number of Trades | {metrics.get('n_trades', 0)} |",
        f"| Win Rate | {metrics.get('win_rate', 0):.2%} |",
        f"| Average Trade | ${metrics.get('avg_trade', 0):.2f} |",
        f"| Profit Factor | {metrics.get('profit_factor', 0):.2f} |",
        f"| Gross Return | {metrics.get('gross_return', 0):.2%} |",
        f"| Net Return | {metrics.get('net_return', 0):.2%} |",
        f"| Sharpe Ratio | {metrics.get('sharpe_ratio', 0):.2f} |",
        f"| Sortino Ratio | {metrics.get('sortino_ratio', 0):.2f} |",
        f"| Maximum Drawdown | {metrics.get('max_drawdown', 0):.2%} |",
        f"| Calmar Ratio | {metrics.get('calmar_ratio', 0):.2f} |",
        f"| Annualized Turnover | {metrics.get('turnover', 0):.2f}x |",
        f"| Average Holding Period | {metrics.get('avg_holding_period', 0):.1f} bars |",
        f"| Time in Market | {metrics.get('exposure', 0):.2%} |",
        "",
    ])

    # Cost breakdown
    report_lines.extend([
        "## Cost Breakdown",
        "",
        "| Cost Component | Amount |",
        "|----------------|--------|",
        f"| Total Fees | ${metrics.get('total_fees', 0):.2f} |",
        f"| Total Slippage | ${metrics.get('total_slippage', 0):.2f} |",
        f"| Total Funding | ${metrics.get('total_funding', 0):.2f} |",
        f"| Total Short Cost | ${metrics.get('total_short_cost', 0):.2f} |",
        f"| **Total Costs** | **${metrics.get('total_costs', 0):.2f}** |",
        "",
    ])

    # Walk-forward results if available
    window_results = pair_result.get("window_results", [])
    if window_results:
        report_lines.extend([
            "## Walk-Forward Analysis",
            "",
            f"**Number of Windows:** {len(window_results)}",
            "",
            "| Window | Train Period | Test Period | Sharpe | Net Return |",
            "|--------|-------------|-------------|--------|------------|",
        ])

        for i, w in enumerate(window_results):
            train_period = f"{w.get('train_start', 'N/A')[:10]} to {w.get('train_end', 'N/A')[:10]}"
            test_period = f"{w.get('test_start', 'N/A')[:10]} to {w.get('test_end', 'N/A')[:10]}"
            w_metrics = w.get('metrics', {})
            report_lines.append(
                f"| {i+1} | {train_period} | {test_period} | "
                f"{w_metrics.get('sharpe_ratio', 0):.2f} | {w_metrics.get('net_return', 0):.2%} |"
            )

        report_lines.append("")

    # Charts
    report_lines.extend([
        "## Charts",
        "",
        f"![Equity Curve]({charts_dir}/equity_curve_{pair_name.replace('/', '_')}.png)",
        "",
        f"![Drawdown]({charts_dir}/drawdown_{pair_name.replace('/', '_')}.png)",
        "",
        f"![Z-Score]({charts_dir}/zscore_{pair_name.replace('/', '_')}.png)",
        "",
    ])

    # Trade log
    trades = pair_result.get("trades", [])
    if trades:
        report_lines.extend([
            "## Trade Log",
            "",
            "| Entry Time | Exit Time | Side | Entry A | Exit A | Entry B | Exit B | Net P&L |",
            "|------------|----------|------|---------|--------|---------|--------|---------|",
        ])

        for t in trades[:20]:  # Show first 20 trades
            entry_time = str(t.get('timestamp', ''))[:19]
            exit_time = str(t.get('exit_time', ''))[:19] if 'exit_time' in t else 'Open'
            side = t.get('side', 'UNKNOWN')
            entry_a = t.get('entry_px_a', 0)
            exit_a = t.get('exit_px_a', 0)
            entry_b = t.get('entry_px_b', 0)
            exit_b = t.get('exit_px_b', 0)
            net_pnl = t.get('net_pnl', 0)
            report_lines.append(
                f"| {entry_time} | {exit_time} | {side} | {entry_a:.4f} | {exit_a:.4f} | "
                f"{entry_b:.4f} | {exit_b:.4f} | ${net_pnl:.2f} |"
            )

        if len(trades) > 20:
            report_lines.append(f"\n*... and {len(trades) - 20} more trades*\n")

        report_lines.append("")

    # Save report
    pair_name_safe = pair_name.replace('/', '_')
    output_path = Path("reports") / f"pair_report_{pair_name_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Generated pair report: {output_path}")

    # Also save JSON for machine reading
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(pair_result, f, indent=2, default=str)

    logger.info(f"Saved JSON results: {json_path}")

    return str(output_path)


def generate_summary_report(
    all_pair_results: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> str:
    """Generate a summary report across all pairs.

    Args:
        all_pair_results: List of pair result dicts.
        config: Configuration dict.

    Returns:
        Path to the generated markdown summary report.
    """
    # Classification thresholds
    robust_sharpe_min = config.get("classification", {}).get("robust_sharpe_min", 1.0)
    robust_pvalue_max = config.get("classification", {}).get("robust_pvalue_max", 0.10)
    fragile_pvalue_min = config.get("classification", {}).get("fragile_pvalue_min", 0.10)
    no_edge_pvalue_min = config.get("classification", {}).get("no_edge_pvalue_min", 0.20)

    report_lines = [
        "# Crypto Statistical Arbitrage Research Summary",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Pairs Analyzed:** {len(all_pair_results)}",
        "",
        "## Classification Criteria",
        "",
        f"- **ROBUST**: Sharpe > {robust_sharpe_min}, p-value < {robust_pvalue_max}",
        f"- **PROMISING**: Directionally correct but fragile",
        f"- **FRAGILE**: Profitable only under optimistic costs, p > {fragile_pvalue_min}",
        f"- **NO EDGE**: Not profitable net of base costs, p > {no_edge_pvalue_min}",
        "",
        "## Pair Classification",
        "",
        "| Pair | Sharpe | Net Return | Max DD | Win Rate | Classification |",
        "|------|--------|-----------|--------|----------|---------------|",
    ]

    for result in all_pair_results:
        pair = result.get("pair", "UNKNOWN")
        metrics = result.get("metrics", {})
        sharpe = metrics.get("sharpe_ratio", 0)
        net_return = metrics.get("net_return", 0)
        max_dd = metrics.get("max_drawdown", 0)
        win_rate = metrics.get("win_rate", 0)

        # Classify based on criteria
        classification = classify_pair(
            sharpe, net_return, max_dd, win_rate,
            robust_sharpe_min, robust_pvalue_max
        )

        report_lines.append(
            f"| {pair} | {sharpe:.2f} | {net_return:.2%} | {max_dd:.2%} | "
            f"{win_rate:.2%} | **{classification}** |"
        )

    report_lines.extend([
        "",
        "## Performance Comparison",
        "",
        "| Pair | N Trades | Avg Trade | Profit Factor | Turnover |",
        "|------|---------|-----------|---------------|----------|",
    ])

    for result in all_pair_results:
        pair = result.get("pair", "UNKNOWN")
        metrics = result.get("metrics", {})
        n_trades = metrics.get("n_trades", 0)
        avg_trade = metrics.get("avg_trade", 0)
        pf = metrics.get("profit_factor", 0)
        turnover = metrics.get("turnover", 0)

        report_lines.append(
            f"| {pair} | {n_trades} | ${avg_trade:.2f} | {pf:.2f} | {turnover:.2f}x |"
        )

    # Summary statistics
    if all_pair_results:
        avg_sharpe = sum(r.get('metrics', {}).get('sharpe_ratio', 0) for r in all_pair_results) / len(all_pair_results)
        avg_net_return = sum(r.get('metrics', {}).get('net_return', 0) for r in all_pair_results) / len(all_pair_results)

        report_lines.extend([
            "",
            "## Aggregate Statistics",
            "",
            f"- **Average Sharpe Ratio:** {avg_sharpe:.2f}",
            f"- **Average Net Return:** {avg_net_return:.2%}",
            f"- **Pairs with Positive Sharpe:** {sum(1 for r in all_pair_results if r.get('metrics', {}).get('sharpe_ratio', 0) > 0)}/{len(all_pair_results)}",
            "",
        ])

    # Configuration used
    report_lines.extend([
        "## Configuration",
        "",
        f"- **Cost Scenario:** {config.get('cost_scenario', 'base')}",
        f"- **Entry Threshold:** {config.get('entry_threshold', 2.0)}",
        f"- **Exit Type:** {config.get('exit_type', 'z0')}",
        f"- **Hedge Type:** {config.get('hedge_type', 'equal_dollar')}",
        "",
    ])

    # Save report
    output_path = Path("reports") / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Generated summary report: {output_path}")

    # Save JSON summary
    json_output = {
        "generated_at": datetime.now().isoformat(),
        "n_pairs": len(all_pair_results),
        "pair_results": all_pair_results,
        "config": config,
        "aggregate_stats": {
            "avg_sharpe": avg_sharpe if all_pair_results else 0,
            "avg_net_return": avg_net_return if all_pair_results else 0,
            "pairs_positive_sharpe": sum(1 for r in all_pair_results if r.get('metrics', {}).get('sharpe_ratio', 0) > 0),
        }
    }

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2, default=str)

    logger.info(f"Saved JSON summary: {json_path}")

    return str(output_path)


def classify_pair(
    sharpe: float,
    net_return: float,
    max_drawdown: float,
    win_rate: float,
    robust_sharpe_min: float = 1.0,
    robust_pvalue_max: float = 0.10,
) -> str:
    """Classify a pair based on performance metrics.

    Args:
        sharpe: Sharpe ratio.
        net_return: Net return fraction.
        max_drawdown: Maximum drawdown fraction.
        win_rate: Win rate fraction.
        robust_sharpe_min: Minimum Sharpe for ROBUST classification.
        robust_pvalue_max: Maximum p-value for ROBUST classification.

    Returns:
        Classification string: 'ROBUST', 'PROMISING', 'FRAGILE', or 'NO EDGE'.
    """
    # ROBUST: Sharpe > 1.0, positive return, reasonable drawdown
    if sharpe >= robust_sharpe_min and net_return > 0 and max_drawdown < 0.3:
        return "ROBUST"

    # NO EDGE: Negative Sharpe or negative net return
    if sharpe < 0 or net_return < 0:
        return "NO EDGE"

    # FRAGILE: Sharpe < 0.5 or low win rate
    if sharpe < 0.5 or win_rate < 0.45:
        return "FRAGILE"

    # PROMISING: Everything else
    return "PROMISING"


def save_experiment_json(
    experiment_data: Dict[str, Any],
    output_path: str | Path,
) -> str:
    """Save experiment results to JSON.

    Args:
        experiment_data: Experiment data dict.
        output_path: Path to save JSON file.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(experiment_data, f, indent=2, default=str)

    logger.info(f"Saved experiment JSON: {output_path}")
    return str(output_path)

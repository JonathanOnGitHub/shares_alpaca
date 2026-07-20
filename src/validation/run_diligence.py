"""
Run the diligence suite against a strategy backtest and save a persistent
report to reports/, so results live in the repo instead of a notebook or
someone's memory.

Usage:
    .venv/bin/python -m src.validation.run_diligence --strategy momentum_smallcap
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.data.alpaca_client import AlpacaClient
from src.trading.momentum import CrossSectionalMomentum
from src.validation.diligence import DiligenceSuite

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = ROOT / "reports"

# Registry of strategy configs to validate. Add entries here as new
# strategies/parameter sets are proposed, so every candidate goes through
# the same gate before being considered for paper or live trading.
STRATEGY_CONFIGS = {
    "momentum_smallcap": {
        "symbols_config": "config/config_smallcap.yaml",
        "momentum_params": {
            "lookback_months": 12,
            "skip_months": 1,
            "holding_months": 1,
            "top_n": 5,
            "bottom_n": 0,
        },
    },
}


def load_symbols(config_path: str) -> list[str]:
    with open(ROOT / config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["data"]["symbols"]


def run(strategy_name: str) -> dict:
    if strategy_name not in STRATEGY_CONFIGS:
        raise ValueError(f"Unknown strategy '{strategy_name}'. Options: {list(STRATEGY_CONFIGS)}")

    spec = STRATEGY_CONFIGS[strategy_name]
    symbols = load_symbols(spec["symbols_config"])

    client = AlpacaClient()
    data = client.get_bars(symbols, timeframe="Day", lookback_days=1000)
    data = {s: df for s, df in data.items() if not df.empty and len(df) > 300}

    strategy = CrossSectionalMomentum(spec["momentum_params"])
    result = strategy.backtest(data)

    if "equity_curve" not in result or len(result["equity_curve"]) == 0:
        raise RuntimeError("Backtest produced no equity curve — check data/config")

    suite = DiligenceSuite(
        equity_curve=result["equity_curve"],
        trades=result.get("trade_log", []),
        prices=data,
        config=spec["momentum_params"],
        strategy_name=strategy_name,
    )
    report = suite.run_all()
    print(report.summary())

    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = REPORTS_DIR / f"{strategy_name}_{stamp}.json"
    payload = report.to_dict()
    payload["timestamp_utc"] = stamp
    payload["params"] = spec["momentum_params"]
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved report: {out_path.relative_to(ROOT)}")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="momentum_smallcap", choices=list(STRATEGY_CONFIGS))
    args = parser.parse_args()
    run(args.strategy)

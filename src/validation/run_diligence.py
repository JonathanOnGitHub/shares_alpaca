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

import numpy as np
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
        "type": "momentum",
        "symbols_config": "config/config_smallcap.yaml",
        "params": {
            "lookback_months": 12,
            "skip_months": 1,
            "holding_months": 1,
            "top_n": 5,
            "bottom_n": 0,
        },
    },
    "momentum_smallcap_ls": {
        "type": "momentum",
        "symbols_config": "config/config_smallcap.yaml",
        "params": {
            "lookback_months": 12,
            "skip_months": 1,
            "holding_months": 1,
            "top_n": 5,
            "bottom_n": 5,
        },
    },
    "ensemble_mega": {
        "type": "ensemble",
        "symbols_config": "config/config.yaml",
        "params": {},
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

    if spec["type"] == "momentum":
        data = client.get_bars(symbols, timeframe="Day", lookback_days=1000)
        data = {s: df for s, df in data.items() if not df.empty and len(df) > 300}
        strategy = CrossSectionalMomentum(spec["params"])
        result = strategy.backtest(data)
        if "equity_curve" not in result or len(result["equity_curve"]) == 0:
            raise RuntimeError("Backtest produced no equity curve — check data/config")
        suite = DiligenceSuite(
            equity_curve=result["equity_curve"],
            trades=result.get("trade_log", []),
            prices=data,
            config=spec["params"],
            strategy_name=strategy_name,
        )

    elif spec["type"] == "ensemble":
        from src.backtest.engine import BacktestEngine
        from src.features.technical import FeatureEngineer
        from src.models.ensemble import EnsembleModel

        engineer = FeatureEngineer({
            "technical_indicators": {"sma": [5,20], "rsi": [14], "macd": [12,26,9],
                                     "bollinger": [20,2], "atr": [14]},
            "price_features": ["returns"],
        })
        bars = client.get_bars(symbols, timeframe="Day", lookback_days=800)
        prices = {}
        for s in symbols:
            if s not in bars or bars[s].empty or len(bars[s]) < 200:
                continue
            df = engineer.compute_all(bars[s])
            df.dropna(inplace=True)
            if len(df) < 200:
                continue
            prices[s] = df

        engine = BacktestEngine(1_000_000)  # use 1M for readability
        test_bars = {}
        predictions = {}
        for s, df in prices.items():
            feat_cols = [c for c in df.columns if c not in (
                "symbol","open","high","low","close","volume","trade_count","vwap")]
            X = df[feat_cols].values.astype(np.float32)
            y = (df["close"].shift(-1) / df["close"] - 1).values[:-1]
            X = X[:-1]

            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train = y[:split]

            ensemble = EnsembleModel({
                "lstm": {"enabled":True,"sequence_length":20,"hidden_units":[32,16],
                         "dropout":0.2,"epochs":5,"batch_size":16,"learning_rate":0.001},
                "xgboost": {"enabled":True,"n_estimators":50,"max_depth":4,
                            "learning_rate":0.1,"subsample":0.8,"colsample_bytree":0.8},
                "linear": {"enabled":True},
                "weights": {"lstm":0.4,"xgboost":0.4,"linear":0.2},
            })
            ensemble.train(X_train, y_train)
            preds = ensemble.predict(X_test)
            if len(preds) < 5:
                continue
            predictions[s] = preds
            test_bars[s] = df.iloc[split:-1].iloc[-len(preds):]

        if not test_bars:
            raise RuntimeError("Ensemble produced no test bars")

        result = engine.run(test_bars, predictions, {
            "max_position_pct": 0.1, "max_open_positions": 5,
            "slippage_pct": 0.001, "commission_pct": 0, "min_signal_threshold": 0,
        })

        suite = DiligenceSuite(
            equity_curve=result.equity_curve,
            trades=result.trades,
            prices={s: bars[s] for s in predictions},
            strategy_name=strategy_name,
        )

    else:
        raise ValueError(f"Unknown strategy type: {spec['type']}")

    report = suite.run_all()
    print(report.summary())

    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = REPORTS_DIR / f"{strategy_name}_{stamp}.json"
    payload = report.to_dict()
    payload["timestamp_utc"] = stamp
    payload["params"] = spec.get("params", {})
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved report: {out_path.relative_to(ROOT)}")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="momentum_smallcap", choices=list(STRATEGY_CONFIGS))
    args = parser.parse_args()
    run(args.strategy)

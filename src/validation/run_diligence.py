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
import yfinance as yf

from src.data.alpaca_client import AlpacaClient
from src.data.deal_tracker import DealTracker
from src.data.edgar import EDGARClient
from src.trading.momentum import CrossSectionalMomentum
from src.trading.swing import SwingTrading
from src.trading.trend import TrendFollowing
from src.trading.merger_arb import MergerArbBacktest
from src.trading.ma_timing import MATiming
from src.trading.low_vol import LowVol
from src.trading.mean_reversion import MeanReversion
from src.trading.valuation_timing import ValuationTiming
from src.trading.covered_calls import CoveredCalls
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
    "jt_smallcap_decile": {
        "type": "jt_momentum",
        "symbols_config": "config/config_smallcap.yaml",
        "params": {
            "J": 6,
            "K": 6,
            "skip_months": 1,
            "percentile": "decile",
            "min_price": 0.0,
        },
    },
    "jt_smallcap_quintile": {
        "type": "jt_momentum",
        "symbols_config": "config/config_smallcap.yaml",
        "params": {
            "J": 9,
            "K": 3,
            "skip_months": 1,
            "percentile": "quintile",
            "min_price": 0.0,
        },
    },
    "ensemble_mega": {
        "type": "ensemble",
        "symbols_config": "config/config.yaml",
        "params": {},
    },
    "swing_mega": {
        "type": "swing",
        "symbols_config": "config/config.yaml",
        "params": {
            "rsi_period": 14,
            "rsi_oversold": 35,
            "rsi_overbought": 65,
            "sma_trend": 20,
            "sma_volume": 20,
            "volume_avg_multiplier": 1.0,
            "atr_period": 14,
            "atr_stop_mult": 2.0,
            "atr_target_mult": 3.0,
            "max_holding_days": 10,
            "min_holding_days": 2,
            "max_position_pct": 0.15,
            "max_open_positions": 8,
            "initial_capital": 100_000.0,
            "slippage_pct": 0.001,
            "commission_pct": 0.0,
        },
    },
    "swing_smallcap": {
        "type": "swing",
        "symbols_config": "config/config_smallcap.yaml",
        "params": {
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "sma_trend": 50,
            "sma_volume": 20,
            "volume_avg_multiplier": 1.2,
            "atr_period": 14,
            "atr_stop_mult": 2.0,
            "atr_target_mult": 3.0,
            "max_holding_days": 15,
            "min_holding_days": 2,
            "max_position_pct": 0.2,
            "max_open_positions": 5,
            "initial_capital": 100_000.0,
            "slippage_pct": 0.001,
            "commission_pct": 0.0,
        },
    },
    "trend_multiasset": {
        "type": "trend",
        "params": {
            "lookback_days": [126, 252],
            "vol_target": 0.15,
            "vol_lookback": 63,
            "smoothing": "average",
            "max_position_pct": 0.2,
        },
    },
    "crossasset_rot_long": {
        "type": "cross_asset_momentum",
        "params": {
            "lookback_days": [21, 63, 126],
            "vol_target": 0.15,
            "vol_lookback": 63,
            "max_position_pct": 0.25,
            "n_long": 4,
            "n_short": 0,
            "rebalance_frequency": "monthly",
        },
    },
    "crossasset_rot_ls": {
        "type": "cross_asset_momentum",
        "params": {
            "lookback_days": [21, 63, 126],
            "vol_target": 0.15,
            "vol_lookback": 63,
            "max_position_pct": 0.20,
            "n_long": 3,
            "n_short": 2,
            "rebalance_frequency": "monthly",
        },
    },
    "crossasset_rot_best": {
        "type": "cross_asset_momentum",
        "params": {
            "lookback_days": [63, 126, 252],
            "vol_target": 0.15,
            "vol_lookback": 63,
            "max_position_pct": 0.25,
            "n_long": 3,
            "n_short": 0,
            "rebalance_frequency": "monthly",
        },
    },
    "merger_arb": {
        "type": "merger_arb",
        "params": {},
    },
    "ma_timing_spy": {
        "type": "ma_timing",
        "params": {
            "ma_window": 200,
            "slippage_pct": 0.0005,
            "rf_rate": 0.05 / 252,
            "initial_capital": 100_000.0,
        },
        "benchmark": "SPY",
        "symbols_config": None,
    },
    "ma_timing_smallcap": {
        "type": "ma_timing",
        "params": {
            "ma_window": 200,
            "slippage_pct": 0.0005,
            "rf_rate": 0.05 / 252,
            "initial_capital": 100_000.0,
        },
        "benchmark": None,
        "symbols_config": "config/config_smallcap.yaml",
    },
    "ma_timing_mega": {
        "type": "ma_timing",
        "params": {
            "ma_window": 200,
            "slippage_pct": 0.0005,
            "rf_rate": 0.05 / 252,
            "initial_capital": 100_000.0,
        },
        "benchmark": None,
        "symbols_config": "config/config.yaml",
    },
    "ma_timing_spy_100d": {
        "type": "ma_timing",
        "params": {
            "ma_window": 100,
            "slippage_pct": 0.0005,
            "rf_rate": 0.05 / 252,
            "initial_capital": 100_000.0,
        },
        "benchmark": "SPY",
        "symbols_config": None,
    },
    "ma_timing_spy_50d": {
        "type": "ma_timing",
        "params": {
            "ma_window": 50,
            "slippage_pct": 0.0005,
            "rf_rate": 0.05 / 252,
            "initial_capital": 100_000.0,
        },
        "benchmark": "SPY",
        "symbols_config": None,
    },
    "lowvol_mega_inverse": {
        "type": "low_vol",
        "symbols_config": "config/config.yaml",
        "params": {
            "vol_window": 63,
            "lookback_days": 21,
            "n_stocks": 10,
            "weighting": "inverse_vol",
            "rebalance_frequency": "monthly",
            "initial_capital": 100_000.0,
            "slippage_pct": 0.001,
        },
    },
    "lowvol_mega_minvar": {
        "type": "low_vol",
        "symbols_config": "config/config.yaml",
        "params": {
            "vol_window": 63,
            "lookback_days": 21,
            "n_stocks": 10,
            "weighting": "min_variance",
            "rebalance_frequency": "monthly",
            "initial_capital": 100_000.0,
            "slippage_pct": 0.001,
        },
    },
    "lowvol_small_inverse": {
        "type": "low_vol",
        "symbols_config": "config/config_smallcap.yaml",
        "params": {
            "vol_window": 63,
            "lookback_days": 21,
            "n_stocks": 10,
            "weighting": "inverse_vol",
            "rebalance_frequency": "monthly",
            "initial_capital": 100_000.0,
            "slippage_pct": 0.001,
        },
    },
    "lowvol_small_quality": {
        "type": "low_vol",
        "params": {
            "vol_window": 63,
            "lookback_days": 21,
            "n_stocks": 10,
            "weighting": "inverse_vol",
            "rebalance_frequency": "monthly",
            "initial_capital": 100_000.0,
            "slippage_pct": 0.001,
            "min_quality_stocks": 5,
            "quality_filter": {
                "min_roe": 0.05,
                "max_debt_equity": 99999.0,
                "min_profit_margin": 0.0,
            },
        },
    },
    "lowvol_mega_quality": {
        "type": "low_vol",
        "params": {
            "vol_window": 63,
            "lookback_days": 21,
            "n_stocks": 10,
            "weighting": "inverse_vol",
            "rebalance_frequency": "monthly",
            "initial_capital": 100_000.0,
            "slippage_pct": 0.001,
            "min_quality_stocks": 5,
            "quality_filter": {
                "min_roe": 0.10,
                "max_debt_equity": 200.0,
                "min_profit_margin": 0.05,
            },
        },
    },
    "meanrev_mega": {
        "type": "mean_reversion",
        "params": {
            "rsi_period": 14,
            "rsi_oversold": 25,
            "rsi_overbought": 75,
            "rsi_exit_long": 55,
            "rsi_exit_short": 45,
            "atr_period": 14,
            "atr_target_mult": 1.5,
            "atr_stop_mult": 1.0,
            "max_holding_days": 3,
            "min_holding_days": 1,
            "max_position_pct": 0.10,
            "max_open_positions": 8,
            "initial_capital": 100_000.0,
            "slippage_pct": 0.001,
            "commission_pct": 0.0,
        },
    },
    "meanrev_small": {
        "type": "mean_reversion",
        "params": {
            "rsi_period": 14,
            "rsi_oversold": 20,
            "rsi_overbought": 80,
            "rsi_exit_long": 55,
            "rsi_exit_short": 45,
            "atr_period": 14,
            "atr_target_mult": 1.5,
            "atr_stop_mult": 1.0,
            "max_holding_days": 2,
            "min_holding_days": 1,
            "max_position_pct": 0.10,
            "max_open_positions": 8,
            "initial_capital": 100_000.0,
            "slippage_pct": 0.001,
            "commission_pct": 0.0,
        },
    },
    "val_timing_mega": {
        "type": "valuation_timing",
        "params": {
            "pe_ma_window": 60,
            "rebal_days": 21,
            "max_position_pct": 0.10,
            "max_open_positions": 8,
            "initial_capital": 100_000.0,
            "slippage_pct": 0.001,
            "commission_pct": 0.0,
        },
    },
    "val_timing_small": {
        "type": "valuation_timing",
        "params": {
            "pe_ma_window": 60,
            "rebal_days": 21,
            "max_position_pct": 0.10,
            "max_open_positions": 8,
            "initial_capital": 100_000.0,
            "slippage_pct": 0.001,
            "commission_pct": 0.0,
        },
    },
    "covered_calls_mega": {
        "type": "covered_calls",
        "params": {
            "strike_otm_pct": 0.05,
            "vol_window": 20,
            "days_to_expiry": 30,
            "rebal_days": 21,
            "max_position_pct": 0.10,
            "max_open_positions": 8,
            "initial_capital": 100_000.0,
            "slippage_pct": 0.001,
            "commission_pct": 0.0,
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
    client = AlpacaClient()

    if spec["type"] == "momentum":
        symbols = load_symbols(spec["symbols_config"])
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
        symbols = load_symbols(spec["symbols_config"])
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

    elif spec["type"] == "trend":
        assets = {
            'SPY': 'Equities', 'QQQ': 'Equities', 'IWM': 'Equities',
            'TLT': 'Bonds', 'IEF': 'Bonds', 'SHY': 'Bonds',
            'LQD': 'Bonds', 'HYG': 'Bonds',
            'GLD': 'Commodities', 'SLV': 'Commodities',
            'USO': 'Commodities', 'DBC': 'Commodities',
            'UUP': 'Currencies', 'FXE': 'Currencies',
            'VNQ': 'Real Estate',
            'XLU': 'Sectors', 'XLV': 'Sectors', 'XLK': 'Sectors',
        }
        prices = {}
        for sym in assets:
            h = yf.Ticker(sym).history(period='max')
            if h is not None and len(h) > 2000:
                prices[sym] = h['Close']
        pf = pd.DataFrame(prices)
        if hasattr(pf.index, 'tz') and pf.index.tz is not None:
            pf.index = pf.index.tz_localize(None)
        pf = pf.dropna(how='all')

        strategy = TrendFollowing(spec["params"])
        result = strategy.backtest(pf)

        prices_map = {s: pd.DataFrame({'close': pf[s]}, index=pf.index) for s in pf.columns}
        suite = DiligenceSuite(
            equity_curve=result["equity_curve"],
            trades=result.get("trade_log", []),
            prices=prices_map,
            config=spec["params"],
            strategy_name=strategy_name,
        )

    elif spec["type"] == "swing":
        symbols = load_symbols(spec["symbols_config"])
        data = client.get_bars(symbols, timeframe="Day", lookback_days=1500)
        data = {s: df for s, df in data.items() if not df.empty and len(df) > 300}
        strategy = SwingTrading(spec["params"])
        result = strategy.backtest(data)
        if "equity_curve" not in result or len(result["equity_curve"]) == 0:
            raise RuntimeError("Swing backtest produced no equity curve")
        suite = DiligenceSuite(
            equity_curve=result["equity_curve"],
            trades=result.get("trade_log", []),
            prices=data,
            config=spec["params"],
            strategy_name=strategy_name,
        )

    elif spec["type"] == "ma_timing":
        if spec.get("symbols_config"):
            symbols = load_symbols(spec["symbols_config"])
            data = client.get_bars(symbols, timeframe="Day", lookback_days=1000)
            data = {s: df for s, df in data.items() if not df.empty and len(df) > 300}
        else:
            data = {}

        strategy = MATiming(spec["params"])
        result = strategy.backtest(data, benchmark=spec.get("benchmark"))

        if "equity_curve" not in result or len(result["equity_curve"]) == 0:
            raise RuntimeError("MA timing backtest produced no equity curve")

        prices_for_suite = result.get("prices", {})
        if not prices_for_suite and spec.get("benchmark"):
            import yfinance as yf
            h = yf.Ticker(spec["benchmark"]).history(period="max")
            if h is not None and len(h) > 100:
                close_col = h["Close"]
                if hasattr(close_col.index, "tz") and close_col.index.tz is not None:
                    close_col.index = close_col.index.tz_localize(None)
                prices_for_suite = {spec["benchmark"]: pd.DataFrame({"close": close_col})}

        suite = DiligenceSuite(
            equity_curve=result["equity_curve"],
            trades=result.get("trade_log", []),
            prices=prices_for_suite,
            config=spec["params"],
            strategy_name=strategy_name,
        )

    elif spec["type"] == "low_vol":
        mega_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT',
            'JNJ', 'PG', 'XOM', 'BAC', 'DIS', 'HD', 'CVX', 'UNH', 'MA', 'COST',
            'NFLX', 'ADBE', 'CRM', 'AMD', 'CSCO', 'PFE', 'ABBV', 'MRK', 'TMO', 'AVGO',
            'ACN', 'LIN', 'TXN', 'QCOM', 'AMGN', 'NEE', 'PM', 'ORCL', 'HON', 'RTX',
            'LOW', 'IBM', 'CAT', 'GE', 'MCD', 'BKNG', 'T', 'SPGI', 'DE', 'SYK',
        ]
        smallcap_symbols = [
            'AEO', 'ANF', 'BOOT', 'CROX', 'DKS', 'FIVE', 'OLLI', 'CHWY', 'CVNA', 'COIN',
            'HOOD', 'AFRM', 'UPST', 'UAA', 'BBWI', 'ALGM', 'CRSP', 'BEAM', 'FIVN', 'RNG',
            'QTWO', 'TOST', 'MNDY', 'GTLB', 'SMAR', 'WIX', 'PATH', 'CALM', 'EXAS', 'GH',
        ]

        if strategy_name.startswith("lowvol_mega"):
            symbols = mega_symbols
        else:
            symbols = smallcap_symbols

        data = client.get_bars(symbols, timeframe="Day", lookback_days=1000)
        data = {s: df for s, df in data.items() if not df.empty and len(df) > 300}
        strategy = LowVol(spec["params"])
        result = strategy.backtest(data)
        if "equity_curve" not in result or len(result["equity_curve"]) == 0:
            raise RuntimeError("Low-vol backtest produced no equity curve")
        suite = DiligenceSuite(
            equity_curve=result["equity_curve"],
            trades=result.get("trade_log", []),
            prices=result.get("prices", data),
            config=spec["params"],
            strategy_name=strategy_name,
        )

    elif spec["type"] == "cross_asset_momentum":
        from src.trading.cross_asset_momentum import CrossAssetMomentum
        client = AlpacaClient()
        mega_symbols = [
            'SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'LQD', 'HYG',
            'GLD', 'DBC', 'VNQ', 'XLU', 'XLE', 'XLV', 'XLK',
            'VWO', 'EFA', 'EEM', 'TIP', 'IAU',
        ]
        data = client.get_bars(mega_symbols, timeframe="Day", lookback_days=1000)
        data = {s: df for s, df in data.items() if not df.empty and len(df) > 300}
        if len(data) < 5:
            raise RuntimeError(f"Only {len(data)} symbols with enough data")
        strategy = CrossAssetMomentum(spec["params"])
        result = strategy.backtest(symbols=mega_symbols)
        if "equity_curve" not in result or len(result["equity_curve"]) == 0:
            raise RuntimeError("Cross-asset momentum backtest produced no equity curve")
        suite = DiligenceSuite(
            equity_curve=result["equity_curve"],
            trades=result.get("trade_log", []),
            prices=result.get("prices", data),
            config=spec["params"],
            strategy_name=strategy_name,
        )

    elif spec["type"] == "mean_reversion":
        client = AlpacaClient()
        mega_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT',
            'JNJ', 'PG', 'XOM', 'BAC', 'DIS', 'HD', 'CVX', 'UNH', 'MA', 'COST',
            'NFLX', 'ADBE', 'CRM', 'AMD', 'CSCO', 'PFE', 'ABBV', 'MRK', 'TMO', 'AVGO',
        ]
        smallcap_symbols = [
            'AEO', 'ANF', 'BOOT', 'CROX', 'DKS', 'FIVE', 'OLLI', 'CHWY', 'CVNA', 'COIN',
            'HOOD', 'AFRM', 'UPST', 'UAA', 'BBWI', 'ALGM', 'CRSP', 'BEAM', 'FIVN', 'RNG',
            'QTWO', 'TOST', 'MNDY', 'GTLB', 'SMAR', 'WIX', 'PATH', 'CALM', 'EXAS', 'GH',
        ]
        symbols = mega_symbols if strategy_name == "meanrev_mega" else smallcap_symbols
        data = client.get_bars(symbols, timeframe="Day", lookback_days=1500)
        data = {s: df for s, df in data.items() if not df.empty and len(df) > 200}
        if len(data) < 5:
            raise RuntimeError(f"Only {len(data)} symbols with enough data")
        strategy = MeanReversion(spec["params"])
        result = strategy.backtest(data)
        if "equity_curve" not in result or len(result["equity_curve"]) == 0:
            raise RuntimeError("Mean reversion backtest produced no equity curve")
        suite = DiligenceSuite(
            equity_curve=result["equity_curve"],
            trades=result.get("trade_log", []),
            prices=data,
            config=spec["params"],
            strategy_name=strategy_name,
        )

    elif spec["type"] == "valuation_timing":
        client = AlpacaClient()
        mega_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT',
            'JNJ', 'PG', 'XOM', 'BAC', 'DIS', 'HD', 'CVX', 'UNH', 'MA', 'COST',
            'NFLX', 'ADBE', 'CRM', 'AMD', 'CSCO', 'PFE', 'ABBV', 'MRK', 'TMO', 'AVGO',
        ]
        smallcap_symbols = [
            'AEO', 'ANF', 'BOOT', 'CROX', 'DKS', 'FIVE', 'OLLI', 'CHWY', 'CVNA', 'COIN',
            'HOOD', 'AFRM', 'UPST', 'UAA', 'BBWI', 'ALGM', 'CRSP', 'BEAM', 'FIVN', 'RNG',
            'QTWO', 'TOST', 'MNDY', 'GTLB', 'SMAR', 'WIX', 'PATH', 'CALM', 'EXAS', 'GH',
        ]
        symbols = mega_symbols if strategy_name == "val_timing_mega" else smallcap_symbols
        data = client.get_bars(symbols, timeframe="Day", lookback_days=1500)
        data = {s: df for s, df in data.items() if not df.empty and len(df) > 200}
        if len(data) < 5:
            raise RuntimeError(f"Only {len(data)} symbols with enough data")
        strategy = ValuationTiming(spec["params"])
        result = strategy.backtest(data)
        if "equity_curve" not in result or len(result["equity_curve"]) == 0:
            raise RuntimeError("Valuation timing backtest produced no equity curve")
        suite = DiligenceSuite(
            equity_curve=result["equity_curve"],
            trades=result.get("trade_log", []),
            prices=data,
            config=spec["params"],
            strategy_name=strategy_name,
        )

    elif spec["type"] == "covered_calls":
        client = AlpacaClient()
        mega_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT',
            'JNJ', 'PG', 'XOM', 'BAC', 'DIS', 'HD', 'CVX', 'UNH', 'MA', 'COST',
            'NFLX', 'ADBE', 'CRM', 'AMD', 'CSCO', 'PFE', 'ABBV', 'MRK', 'TMO', 'AVGO',
        ]
        data = client.get_bars(mega_symbols, timeframe="Day", lookback_days=1500)
        data = {s: df for s, df in data.items() if not df.empty and len(df) > 200}
        if len(data) < 5:
            raise RuntimeError(f"Only {len(data)} symbols with enough data")
        strategy = CoveredCalls(spec["params"])
        result = strategy.backtest(data)
        if "equity_curve" not in result or len(result["equity_curve"]) == 0:
            raise RuntimeError("Covered calls backtest produced no equity curve")
        suite = DiligenceSuite(
            equity_curve=result["equity_curve"],
            trades=result.get("trade_log", []),
            prices=data,
            config=spec["params"],
            strategy_name=strategy_name,
        )

    elif spec["type"] == "merger_arb":
        edgar = EDGARClient()
        tracker = DealTracker(edgar)
        resolved = [d for d in tracker.deals if d.status in ("completed", "terminated") and d.offer_price]

        if len(resolved) < 5:
            logger.warning("Only %d resolved deals — results will be noisy", len(resolved))

        bt = MergerArbBacktest(client)
        result = bt.run(resolved)

        ticker_set = {d.ticker for d in resolved if d.ticker}
        bars = client.get_bars(list(ticker_set), timeframe="Day", lookback_days=1500)
        prices = {s: df for s, df in bars.items() if df is not None and not df.empty}
        suite = DiligenceSuite(
            equity_curve=result.equity_curve,
            trades=result.trades,
            prices=prices,
            strategy_name=strategy_name,
        )

    elif spec["type"] == "jt_momentum":
        import json
        import yfinance as yf
        import warnings
        from src.trading.jt_momentum import JTMomentum
        import yaml

        warnings.filterwarnings("ignore", category=UserWarning)

        symbols_cfg_path = ROOT / spec["symbols_config"]
        if symbols_cfg_path.exists():
            with open(symbols_cfg_path) as f:
                syms_cfg = yaml.safe_load(f)
                symbols = syms_cfg.get("symbols", [])
        else:
            symbols = []

        if not symbols:
            universe_path = Path("/home/burley/Personal/Trading/JT_momentum/smallcap_filtered.json")
            if universe_path.exists():
                with open(universe_path) as f:
                    symbols = list(json.load(f).keys())
            else:
                raise ValueError("No symbols for JT momentum")

        data = {}
        for sym in symbols[:500]:
            try:
                ticker = yf.Ticker(sym, session=False)
                df = ticker.history(period="max", auto_adjust=True)
                if df is not None and len(df) > 200 and "Close" in df.columns:
                    data[sym] = pd.DataFrame({"close": df["Close"]})
            except Exception:
                continue

        if len(data) < 20:
            raise RuntimeError(f"Only {len(data)} symbols with sufficient data")

        strategy = JTMomentum(
            J=spec["params"]["J"],
            K=spec["params"]["K"],
            skip_months=spec["params"].get("skip_months", 1),
            percentile=spec["params"].get("percentile", "decile"),
            min_price=spec["params"].get("min_price", 0.0),
        )
        result = strategy.backtest(data)

        if "equity_curve" not in result or len(result["equity_curve"]) == 0:
            raise RuntimeError("JT momentum backtest produced no equity curve")

        suite = DiligenceSuite(
            equity_curve=result["equity_curve"],
            trades=result.get("trade_log", []),
            prices=result.get("prices", data),
            config=spec["params"],
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
    parser.add_argument("--strategy", default="swing_mega", choices=list(STRATEGY_CONFIGS))
    args = parser.parse_args()
    run(args.strategy)

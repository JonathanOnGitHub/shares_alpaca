import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.alpaca_client import AlpacaClient
from src.data.earnings import EarningsData
from src.features.technical import FeatureEngineer
from src.models.ensemble import EnsembleModel
from src.backtest.engine import BacktestEngine, BacktestResult
from src.trading.executor import TradingExecutor
from src.trading.pead_overlay import PEADOverlay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str = "config/config.yaml") -> dict:
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def prepare_data_full(
    client: AlpacaClient, config: dict
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]], dict[str, pd.DataFrame]]:
    symbols = config["data"]["symbols"]
    timeframe = config["data"]["timeframe"]
    lookback = config["data"]["lookback_days"]
    engineer = FeatureEngineer(config["features"])
    mode = config.get("models", {}).get("mode", "regression")

    result = {}
    bars = client.get_bars(symbols, timeframe, lookback)

    features_by_symbol = {}
    for symbol, df in bars.items():
        if df.empty or len(df) < 100:
            logger.warning("Insufficient data for %s, skipping", symbol)
            continue
        df = engineer.compute_all(df)
        df.dropna(inplace=True)
        if len(df) < 100:
            logger.warning("Too few samples for %s after features", symbol)
            continue
        features_by_symbol[symbol] = df

    if "SPY" not in features_by_symbol and len(features_by_symbol) > 0:
        spy_bars = client.get_bars(["SPY"], timeframe, lookback)
        if "SPY" in spy_bars and not spy_bars["SPY"].empty:
            spy_df = engineer.compute_all(spy_bars["SPY"])
            spy_df.dropna(inplace=True)
            if len(spy_df) >= 100:
                features_by_symbol["SPY"] = spy_df

    spy_df = features_by_symbol.get("SPY")
    use_pead = config.get("features", {}).get("pead", {}).get("enabled", False)
    use_momentum = config.get("features", {}).get("momentum", {}).get("enabled", False)

    if use_pead:
        logger.info("Fetching earnings data for PEAD features...")
        earnings = EarningsData()
        earnings.fetch_earnings(list(features_by_symbol.keys()), lookback)

    price_df = pd.DataFrame()
    mom_ranks: dict[str, pd.DataFrame] = {}
    if use_momentum:
        mom_cfg = config["features"]["momentum"]
        rank_windows = mom_cfg.get("rank_windows", [21, 63, 126, 252])
        price_df = pd.DataFrame({sym: features_by_symbol[sym]["close"]
                                  for sym in features_by_symbol if sym != "SPY"})
        for window in rank_windows:
            col = f"mom_rank_{window}d"
            mom_ranks[col] = price_df.pct_change(window).rank(axis=1, pct=True)

    for symbol, df in features_by_symbol.items():
        if symbol == "SPY":
            continue
        df = df.copy()
        if spy_df is not None:
            spy_close = spy_df["close"].reindex(df.index).ffill()
            rolling_corr = df["close"].rolling(20).corr(spy_close)
            spy_std = spy_close.rolling(20).std().replace(0, np.nan)
            sym_std = df["close"].rolling(20).std()
            df["spy_beta_20d"] = rolling_corr * (sym_std / spy_std)
            df["relative_strength_5d"] = (
                df["close"].pct_change(5) - spy_close.pct_change(5)
            )

        if use_pead and symbol in earnings.cache:
            pead = earnings.get_features_for_symbol(symbol, df.index)
            for col in ["days_since_earnings", "earnings_surprise_pct",
                         "earnings_beat", "in_pead_window"]:
                if col in pead.columns:
                    df[col] = pead[col].values

        if use_momentum and symbol in price_df.columns:
            for window in rank_windows:
                col = f"mom_rank_{window}d"
                if col in mom_ranks and symbol in mom_ranks[col].columns:
                    rank_series = mom_ranks[col][symbol].reindex(df.index)
                    df[col] = rank_series.values
                    df[col] = df[col].fillna(0.5)

        df.dropna(inplace=True)
        if len(df) < 100:
            logger.warning("Too few samples for %s after features", symbol)
            continue

        feature_cols = [c for c in df.columns if c not in (
            "symbol", "open", "high", "low", "close", "volume", "trade_count", "vwap"
        )]
        X = df[feature_cols].values
        if mode == "classification":
            target = (df["close"].shift(-1) > df["close"]).astype(float)
        else:
            target = df["close"].shift(-1) / df["close"] - 1
        y = target.values[:-1]
        X = X[:-1]
        df_aligned = df.iloc[:-1]
        result[symbol] = (X, y, df_aligned)
        logger.info("%s: %d samples, %d features",
                     symbol, len(X), X.shape[1])

    earnings_cache = earnings.cache if use_pead else {}
    return result, earnings_cache


def run_walk_forward(
    full_data: dict[str, tuple],
    config: dict,
    earnings_cache: dict[str, pd.DataFrame] | None = None,
) -> BacktestResult | None:
    wf = config["backtest"]["walk_forward"]
    init_size = wf["initial_train_size"]
    test_size = wf["test_size"]
    step_size = wf["step_size"]
    model_config = config["models"]
    trading_config = config["trading"]

    engine = BacktestEngine(trading_config["initial_capital"])
    all_trades = []
    all_equity_curves = []
    window_results = []

    symbols = list(full_data.keys())
    max_len = min(len(full_data[s][0]) for s in symbols) if symbols else 0
    if max_len < init_size + test_size:
        logger.error(
            "Not enough data: need %d samples, have %d",
            init_size + test_size, max_len,
        )
        return None

    n_windows = 0
    while True:
        train_end = init_size + n_windows * step_size
        test_start = train_end
        test_end = test_start + test_size
        if test_end > max_len:
            break

        logger.info(
            "Window %d: train %d:%d, test %d:%d",
            n_windows + 1, n_windows * step_size, train_end,
            test_start, test_end,
        )

        models = {}
        for symbol in symbols:
            X, y, df_aligned = full_data[symbol]
            X_train = X[n_windows * step_size:train_end]
            y_train = y[n_windows * step_size:train_end]
            if len(X_train) < 50:
                logger.warning("  %s: insufficient train data, skipping", symbol)
                continue
            ensemble = EnsembleModel(model_config)
            ensemble.train(X_train, y_train)
            models[symbol] = ensemble

        test_bars = {}
        predictions = {}
        for symbol in symbols:
            if symbol not in models:
                continue
            X, _, df_aligned = full_data[symbol]
            X_test = X[test_start:test_end]
            if len(X_test) == 0:
                continue
            preds = models[symbol].predict(X_test)
            if len(preds) < 2:
                continue
            predictions[symbol] = preds
            test_bars[symbol] = df_aligned.iloc[test_start:test_start + len(preds)]

        if test_bars and earnings_cache:
            try:
                pead_overlay = PEADOverlay(config.get("features", {}).get("pead", {}))
                predictions = pead_overlay.apply(predictions, test_bars, earnings_cache)
            except Exception as e:
                logger.warning("PEAD overlay failed: %s, continuing without it", e)

        if test_bars:
            window_engine = BacktestEngine(trading_config["initial_capital"])
            result = window_engine.run(test_bars, predictions, trading_config)
            all_trades.extend(result.trades)
            all_equity_curves.append(result.equity_curve)
            window_results.append(result)
            logger.info(
                "  Window return: %.2f%%, trades: %d",
                result.total_return * 100, result.total_trades,
            )

        n_windows += 1

    if not all_trades:
        logger.warning("No trades across any window.")
        return None

    combined = _combine_walk_forward(all_equity_curves, all_trades, engine.initial_capital)
    logger.info("")
    logger.info("=" * 50)
    logger.info("WALK-FORWARD BACKTEST RESULTS")
    logger.info("=" * 50)
    logger.info("Windows:          %d", n_windows)
    logger.info("Total Return:     %.2f%%", combined.total_return * 100)
    logger.info("Annual Return:    %.2f%%", combined.annualized_return * 100)
    logger.info("Sharpe Ratio:     %.2f", combined.sharpe_ratio)
    logger.info("Max Drawdown:     %.2f%%", combined.max_drawdown * 100)
    logger.info("Win Rate:         %.2f%%", combined.win_rate * 100)
    logger.info("Total Trades:     %d", combined.total_trades)
    logger.info("Avg Monthly Ret:  %.2f%%", combined.monthly_returns.mean() * 100 if len(combined.monthly_returns) > 0 else 0)
    logger.info("Monthly Vol:      %.2f%%", combined.monthly_returns.std() * 100 if len(combined.monthly_returns) > 0 else 0)
    logger.info("=" * 50)

    return combined


def _combine_walk_forward(
    equity_curves: list[pd.Series],
    trades: list,
    initial_capital: float,
) -> BacktestResult:
    values = [initial_capital]
    index = [equity_curves[0].index[0] - pd.Timedelta(days=1)] if equity_curves else []
    for curve in equity_curves:
        values.extend(curve.iloc[1:].tolist())
        index.extend(curve.index[1:].tolist())
    combined = pd.Series(values, index=pd.DatetimeIndex(index))
    return BacktestEngine._compute_results(combined, trades)


def main():
    config = load_config()
    client = AlpacaClient(paper=True)

    logger.info("Fetching data and engineering features...")
    full_data, earnings_cache = prepare_data_full(client, config)

    if not full_data:
        logger.error("No data available. Check API keys and symbols.")
        return

    logger.info("Running walk-forward backtest...")
    result = run_walk_forward(full_data, config, earnings_cache)

    if result:
        logger.info("Walk-forward validation complete.")
    else:
        logger.warning("Walk-forward produced no results.")


if __name__ == "__main__":
    main()

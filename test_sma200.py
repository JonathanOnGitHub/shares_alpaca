#!/usr/bin/env python3
"""
Test SMA strategy: Buy when price > SMA(N), sell when price < SMA(N).
Compare SMA100 vs SMA200 vs Buy & Hold.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.backtest.engine import BacktestEngine, Trade


class SMAStrategy:
    """Long-only strategy: buy when close > SMA(N), sell when close < SMA(N)."""

    def __init__(self, sma_period: int = 200):
        self.sma_period = sma_period

    def compute_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
        """Return signal arrays: 1.0 = buy, -1.0 = sell, 0.0 = hold."""
        signals = {}
        for symbol, df in data.items():
            if df.empty or "close" not in df.columns:
                continue
            close = df["close"]
            sma = close.rolling(self.sma_period).mean()
            signal = np.where(close > sma, 1.0, -1.0)
            signals[symbol] = signal
        return signals


def fetch_data(symbols: list[str], start: str = "2015-01-01", end: str = "2025-01-01") -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data using yfinance."""
    data = {}
    for symbol in symbols:
        try:
            df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
            if df is not None and len(df) > 250:
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.columns = [c.lower() for c in df.columns]
                df.index = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in df.index])
                data[symbol] = df
        except Exception as e:
            print(f"  {symbol}: failed - {e}")
    return data


def compute_buy_and_hold(data: dict[str, pd.DataFrame], initial_capital: float) -> dict:
    """Compute equal-weight buy-and-hold benchmark."""
    prices = pd.DataFrame({sym: df["close"] for sym, df in data.items()})
    rets = prices.pct_change().dropna()
    ew_rets = rets.mean(axis=1)
    equity = (1 + ew_rets).cumprod() * initial_capital
    returns = equity.pct_change().dropna()

    n_years = max(len(returns) / 252, 1)
    total_return = (equity.iloc[-1] / initial_capital) - 1
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    dd = (equity / equity.cummax() - 1).min()

    return {
        "total_return": total_return,
        "annual_return": ann_return,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "equity": equity,
    }


def run_strategy(sma_period: int, data: dict, initial_capital: float) -> tuple:
    """Run a single SMA strategy and return results."""
    strategy = SMAStrategy(sma_period=sma_period)
    signals = strategy.compute_signals(data)

    engine = BacktestEngine(initial_capital=initial_capital)
    config = {
        "max_position_pct": 0.1,
        "max_open_positions": 10,
        "slippage_pct": 0.001,
        "commission_pct": 0.0,
        "min_signal_threshold": 0.0,
        "max_trades_per_day": 0,
    }

    result = engine.run(data, signals, config)
    return result


def run_diligence_checks(result, data, strategy_name="SMA"):
    """Run the diligence suite on the backtest results."""
    from src.validation.diligence import DiligenceSuite

    trades = result.trades
    equity = result.equity_curve

    prices = {sym: df[["close"]] for sym, df in data.items()}

    suite = DiligenceSuite(
        equity_curve=equity,
        trades=trades,
        prices=prices,
        strategy_name=strategy_name,
    )
    report = suite.run_all()
    print("\n" + report.summary())
    return report.to_dict()


def print_comparison(results: dict, bh: dict):
    """Print a comparison table of all strategies."""
    print("\n" + "=" * 70)
    print("COMPARISON: SMA100 vs SMA200 vs Buy & Hold")
    print("=" * 70)
    print(f"{'Strategy':<15} {'Total Ret':>12} {'Ann Ret':>10} {'Sharpe':>8} {'Max DD':>10} {'Win Rate':>10} {'Trades':>8}")
    print("-" * 70)
    
    print(f"{'SMA100':<15} {results['sma100'].total_return*100:>11.1f}% {results['sma100'].annualized_return*100:>9.1f}% {results['sma100'].sharpe_ratio:>8.2f} {results['sma100'].max_drawdown*100:>9.1f}% {results['sma100'].win_rate*100:>9.1f}% {results['sma100'].total_trades:>8}")
    print(f"{'SMA200':<15} {results['sma200'].total_return*100:>11.1f}% {results['sma200'].annualized_return*100:>9.1f}% {results['sma200'].sharpe_ratio:>8.2f} {results['sma200'].max_drawdown*100:>9.1f}% {results['sma200'].win_rate*100:>9.1f}% {results['sma200'].total_trades:>8}")
    print(f"{'Buy & Hold':<15} {bh['total_return']*100:>11.1f}% {bh['annual_return']*100:>9.1f}% {bh['sharpe']*100:>8.1f} {bh['max_drawdown']*100:>9.1f}% {'N/A':>10} {'N/A':>8}")
    print("=" * 70)


def main():
    symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
    start = "2015-01-01"
    end = "2025-01-01"
    initial_capital = 100000

    print(f"Fetching data for {len(symbols)} symbols...")
    data = fetch_data(symbols, start, end)
    for sym in data:
        print(f"  {sym}: {len(data[sym])} bars")

    if not data:
        print("No data fetched!")
        return

    print("\nRunning strategies...")

    results = {
        "sma100": run_strategy(100, data, initial_capital),
        "sma200": run_strategy(200, data, initial_capital),
    }
    bh = compute_buy_and_hold(data, initial_capital)

    print_comparison(results, bh)

    print("\n" + "=" * 50)
    print("SMA100 DILIGENCE CHECKS")
    print("=" * 50)
    sma100_report = run_diligence_checks(results["sma100"], data, "SMA100")

    print("\n" + "=" * 50)
    print("SMA200 DILIGENCE CHECKS")
    print("=" * 50)
    sma200_report = run_diligence_checks(results["sma200"], data, "SMA200")

    print("\n" + "=" * 50)
    print("DILIGENCE SUMMARY")
    print("=" * 50)
    print(f"SMA100: {sma100_report['passed']}/{sma100_report['total']} passed")
    print(f"SMA200: {sma200_report['passed']}/{sma200_report['total']} passed")

    print("\n--- Diligence Check Details ---")
    checks_order = ["vs Equal-Weight B&H", "Best Month Exclusion", "Trade Win Rate",
                    "Max Drawdown Analysis", "Monthly Consistency", "Trade Concentration",
                    "Sub-Period Consistency", "Sharpe Significance", "Permutation Test"]
    
    print(f"\n{'Check':<30} {'SMA100':>10} {'SMA200':>10}")
    print("-" * 52)
    for check in checks_order:
        s100 = "PASS" if sma100_report['checks'].get(check, {}).get('passed') else "FAIL"
        s200 = "PASS" if sma200_report['checks'].get(check, {}).get('passed') else "FAIL"
        print(f"{check:<30} {s100:>10} {s200:>10}")

    return results, bh, sma100_report, sma200_report


if __name__ == "__main__":
    main()
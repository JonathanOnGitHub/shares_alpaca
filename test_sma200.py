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
    print("\n" + "=" * 85)
    print("COMPARISON: SMA100 vs SMA200 vs SMA250 vs SMA300 vs Buy & Hold")
    print("=" * 85)
    print(f"{'Strategy':<12} {'Total Ret':>12} {'Ann Ret':>10} {'Sharpe':>8} {'Max DD':>10} {'Win Rate':>10} {'Trades':>8}")
    print("-" * 85)
    
    for sma in ["sma100", "sma200", "sma250", "sma300"]:
        r = results[sma]
        print(f"{sma.upper():<12} {r.total_return*100:>11.1f}% {r.annualized_return*100:>9.1f}% {r.sharpe_ratio:>8.2f} {r.max_drawdown*100:>9.1f}% {r.win_rate*100:>9.1f}% {r.total_trades:>8}")
    
    print(f"{'Buy & Hold':<12} {bh['total_return']*100:>11.1f}% {bh['annual_return']*100:>9.1f}% {bh['sharpe']:>8.2f} {bh['max_drawdown']*100:>9.1f}% {'N/A':>10} {'N/A':>8}")
    print("=" * 85)


def main():
    symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
    start = "2015-01-01"
    end = "2025-01-01"
    initial_capital = 100000
    sma_periods = [100, 200, 250, 300]

    print(f"Fetching data for {len(symbols)} symbols...")
    data = fetch_data(symbols, start, end)
    for sym in data:
        print(f"  {sym}: {len(data[sym])} bars")

    if not data:
        print("No data fetched!")
        return

    print("\nRunning strategies...")

    results = {f"sma{p}": run_strategy(p, data, initial_capital) for p in sma_periods}
    bh = compute_buy_and_hold(data, initial_capital)

    print_comparison(results, bh)

    reports = {}
    for p in sma_periods:
        print(f"\n" + "=" * 50)
        print(f"SMA{p} DILIGENCE CHECKS")
        print("=" * 50)
        reports[p] = run_diligence_checks(results[f"sma{p}"], data, f"SMA{p}")

    print("\n" + "=" * 50)
    print("DILIGENCE SUMMARY")
    print("=" * 50)
    for p in sma_periods:
        print(f"SMA{p}: {reports[p]['passed']}/{reports[p]['total']} passed")

    print("\n--- Diligence Check Details ---")
    checks_order = ["vs Equal-Weight B&H", "Best Month Exclusion", "Trade Win Rate",
                    "Max Drawdown Analysis", "Monthly Consistency", "Trade Concentration",
                    "Sub-Period Consistency", "Sharpe Significance", "Permutation Test"]
    
    header = "".join([f"{'SMA'+str(p):>10}" for p in sma_periods])
    print(f"\n{'Check':<30}{header}")
    print("-" * (30 + 10 * len(sma_periods)))
    for check in checks_order:
        row = "".join([f"{'PASS' if reports[p]['checks'].get(check, {}).get('passed') else 'FAIL':>10}" for p in sma_periods])
        print(f"{check:<30}{row}")

    return results, bh, reports


if __name__ == "__main__":
    main()
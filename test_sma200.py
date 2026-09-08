#!/usr/bin/env python3
"""
Test 200-day SMA strategy: Buy when price > SMA200, sell when price < SMA200.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.backtest.engine import BacktestEngine, Trade


class SMA200Strategy:
    """Long-only strategy: buy when close > SMA200, sell when close < SMA200."""

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
                print(f"  {symbol}: {len(df)} bars")
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


def run_sma200_backtest(
    symbols: list[str] = None,
    start: str = "2015-01-01",
    end: str = "2025-01-01",
    initial_capital: float = 100000,
) -> dict:
    if symbols is None:
        symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

    print(f"Fetching data for {len(symbols)} symbols...")
    data = fetch_data(symbols, start, end)

    if not data:
        print("No data fetched!")
        return {}

    print("\nRunning SMA200 strategy...")
    strategy = SMA200Strategy(sma_period=200)
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

    # Buy and hold benchmark
    bh = compute_buy_and_hold(data, initial_capital)

    print("\n" + "=" * 50)
    print("SMA200 BACKTEST RESULTS")
    print("=" * 50)
    print(f"Total Return:     {result.total_return*100:.2f}%")
    print(f"Annual Return:    {result.annualized_return*100:.2f}%")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown:     {result.max_drawdown*100:.2f}%")
    print(f"Win Rate:         {result.win_rate*100:.2f}%")
    print(f"Total Trades:     {result.total_trades}")

    print("\n--- Buy & Hold (Equal Weight) ---")
    print(f"Total Return:     {bh['total_return']*100:.2f}%")
    print(f"Annual Return:    {bh['annual_return']*100:.2f}%")
    print(f"Sharpe Ratio:     {bh['sharpe']:.2f}")
    print(f"Max Drawdown:     {bh['max_drawdown']*100:.2f}%")

    print(f"\nExcess Return (vs B&H): {(result.total_return - bh['total_return'])*100:.2f}%")

    print("\n--- Trade Log (first 10) ---")
    for t in result.trades[:10]:
        print(f"  {t.date.date()} | {t.side:4s} | {t.symbol:6s} | ${t.price:.2f} | {t.shares:.0f} shrs | PnL: ${t.pnl:.2f}")

    if len(result.trades) > 10:
        print(f"  ... and {len(result.trades) - 10} more trades")

    return {
        "result": result,
        "benchmark": bh,
        "data": data,
        "signals": signals,
    }


def run_diligence_checks(result, data, strategy_name="SMA200"):
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


if __name__ == "__main__":
    results = run_sma200_backtest()
    
    if results:
        print("\n" + "=" * 50)
        print("RUNNING DILIGENCE CHECKS")
        print("=" * 50)
        run_diligence_checks(results["result"], results["data"])
"""200-day moving average timing strategy.

Burns & Holland (2003): timing the market with the 200-day MA produced
~192% excess return over buy-and-hold with max drawdown of ~17% vs ~55%
for B&H over a 16-year test period.

Logic:
  - Price > 200-day MA  → in the market (long)
  - Price < 200-day MA  → switch to cash (no position)
  - Cash earns risk-free rate while out of market

Can be run on a single benchmark (e.g. SPY) or on individual stocks.
"""
import numpy as np
import pandas as pd

from src.backtest.engine import Trade


class MATiming:
    def __init__(self, config: dict):
        self.ma_window = config.get("ma_window", 200)
        self.slippage_pct = config.get("slippage_pct", 0.0005)
        self.rf_rate = config.get("rf_rate", 0.05 / 252)
        self.initial_capital = config.get("initial_capital", 100_000.0)
        self.signal_ma_dist = config.get("signal_ma_dist", 0.0)

    def compute_signals(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        signals = pd.DataFrame(index=data[list(data.keys())[0]].index, columns=data.keys(), dtype=float)
        for symbol, df in data.items():
            close = df["close"]
            ma = close.rolling(self.ma_window).mean()
            sig = (close > ma).astype(float)
            signals[symbol] = sig
        return signals

    def backtest(
        self,
        data: dict[str, pd.DataFrame],
        benchmark: str | None = None,
    ) -> dict:
        if benchmark is not None:
            import yfinance as yf
            if benchmark in data:
                close = data[benchmark]["close"]
            else:
                h = yf.Ticker(benchmark).history(period="max")
                if h is None or len(h) < self.ma_window + 100:
                    raise ValueError(f"Could not fetch {benchmark} data for MA timing backtest")
                close = h["Close"]
                if hasattr(close.index, "tz") and close.index.tz is not None:
                    close.index = close.index.tz_localize(None)
            prices = pd.DataFrame({"_benchmark": close})
            ma = close.rolling(self.ma_window).mean()
            signals = pd.DataFrame(index=close.index, columns=["_benchmark"], dtype=float)
            signals["_benchmark"] = (close > ma).astype(float)
        else:
            if not data:
                raise ValueError("No data provided and no benchmark specified")
            first_key = list(data.keys())[0]
            prices = pd.DataFrame({s: data[s]["close"] for s in data})
            close = prices
            ma = close.rolling(self.ma_window).mean()
            signals = (close > ma).astype(float)

        daily_returns = prices.pct_change().fillna(0)

        position = signals.shift(1).fillna(0)
        position = position.clip(0, 1)

        strat_rets = (position.shift(1) * daily_returns).sum(axis=1)

        cash_returns = pd.Series(self.rf_rate, index=strat_rets.index)
        no_position = (1 - position.shift(1).fillna(0).max(axis=1))
        strat_rets += no_position * cash_returns

        equity = (1 + strat_rets).cumprod() * self.initial_capital

        turnover = position.diff().abs().sum(axis=1)
        tc_cost = turnover * self.slippage_pct * 0.5
        net_rets = strat_rets - tc_cost

        net_equity = (1 + net_rets).cumprod() * self.initial_capital

        bh_prices = prices.iloc[:, 0] if prices.shape[1] == 1 else prices.mean(axis=1)
        bh_equity = (1 + bh_prices.pct_change().fillna(0)).cumprod() * self.initial_capital

        total_ret = net_equity.iloc[-1] / net_equity.iloc[0] - 1
        n_years = max(len(net_rets) / 252, 0.1)
        ann_ret = (1 + total_ret) ** (1 / n_years) - 1
        sharpe = net_rets.mean() / net_rets.std() * np.sqrt(252) if net_rets.std() > 0 else 0
        dd = net_equity / net_equity.cummax() - 1
        max_dd = float(dd.min())

        trades = []
        for i in range(1, len(signals)):
            prev_pos = position.iloc[i - 1]
            curr_pos = position.iloc[i]
            changed = curr_pos[prev_pos != curr_pos]
            date = signals.index[i]
            for sym in changed.index:
                if changed[sym] == 1:
                    price = prices[sym].iloc[i]
                    if pd.isna(price) or price <= 0:
                        continue
                    shares = self.initial_capital * 0.95 / price
                    slip = price * self.slippage_pct
                    trades.append(Trade(
                        date=date, symbol=sym, side="buy",
                        price=price + slip, shares=shares,
                        value=shares * (price + slip),
                    ))
                elif changed[sym] == 0:
                    price = prices[sym].iloc[i]
                    if pd.isna(price) or price <= 0:
                        continue
                    shares = self.initial_capital * 0.95 / price
                    slip = price * self.slippage_pct
                    trades.append(Trade(
                        date=date, symbol=sym, side="sell",
                        price=price - slip, shares=shares,
                        value=shares * (price - slip),
                    ))

        sell_trades = [t for t in trades if t.side == "sell"]
        wins = sum(1 for t in sell_trades if t.pnl and t.pnl > 0)
        win_rate = wins / max(len(sell_trades), 1)

        return {
            "total_return": total_ret,
            "annual_return": ann_ret,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "num_trades": len(trades),
            "equity_curve": net_equity,
            "benchmark_equity": bh_equity,
            "trade_log": trades,
            "prices": {s: data[s] for s in data},
        }

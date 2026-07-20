import numpy as np
import pandas as pd

from src.backtest.engine import Trade


class CrossSectionalMomentum:
    def __init__(self, config: dict):
        self.lookback_months = config.get("lookback_months", 12)
        self.skip_months = config.get("skip_months", 1)
        self.holding_months = config.get("holding_months", 1)
        self.top_n = config.get("top_n", 1)
        self.bottom_n = config.get("bottom_n", 1)

    def compute_signals(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = {}
        for symbol, df in data.items():
            prices[symbol] = df["close"]
        price_df = pd.DataFrame(prices)

        momentum_lookback = self.lookback_months * 21
        momentum_skip = self.skip_months * 21

        monthly_dates = price_df.resample("ME").last().index
        rebalance_dates = monthly_dates[monthly_dates > price_df.index[0] + pd.Timedelta(days=momentum_lookback * 2)]

        signals = pd.DataFrame(index=rebalance_dates, columns=price_df.columns, dtype=float)
        for rdate in rebalance_dates:
            past_returns = price_df.loc[:rdate].iloc[-momentum_lookback:-momentum_skip] if momentum_skip > 0 else price_df.loc[:rdate].iloc[-momentum_lookback:]
            if len(past_returns) < 2:
                continue
            total_return = (past_returns.iloc[-1] / past_returns.iloc[0] - 1)
            ranked = total_return.rank(ascending=False)
            n = max(self.top_n, self.bottom_n)
            signals.loc[rdate] = 0.0
            for sym in ranked.nsmallest(self.bottom_n).index:
                signals.loc[rdate, sym] = -1.0 / self.bottom_n
            for sym in ranked.nlargest(self.top_n).index:
                signals.loc[rdate, sym] = 1.0 / self.top_n

        return signals.dropna(how="all")

    def backtest(self, data: dict[str, pd.DataFrame]) -> dict:
        signals = self.compute_signals(data)
        prices = pd.DataFrame({s: data[s]["close"] for s in data if s in signals.columns})
        daily_returns = prices.pct_change()

        holding_days = self.holding_months * 21
        portfolio_returns = []
        dates_run = []
        trades: list[Trade] = []
        capital = 100_000.0

        for i in range(len(signals) - 1):
            entry_date = signals.index[i]
            weights = signals.iloc[i]
            active_symbols = weights[weights != 0].index
            if len(active_symbols) == 0:
                continue
            exit_date = signals.index[i + 1]
            window = daily_returns.loc[entry_date:exit_date].iloc[1:]
            for date, row in window.iterrows():
                ret = (weights[active_symbols] * row[active_symbols]).sum()
                portfolio_returns.append(ret)
                dates_run.append(date)

            # Build one round-trip Trade per active symbol per rebalance
            # period so the diligence suite can assess win rate and
            # concentration, not just the aggregate equity curve.
            for sym in active_symbols:
                entry_price = prices[sym].asof(entry_date)
                exit_price = prices[sym].asof(exit_date)
                if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
                    continue
                w = weights[sym]
                position_value = capital * abs(w)
                shares = position_value / entry_price
                direction = 1 if w > 0 else -1
                pnl = direction * shares * (exit_price - entry_price)
                trades.append(Trade(
                    date=entry_date, symbol=sym, side="buy",
                    price=entry_price, shares=shares, value=position_value,
                ))
                trades.append(Trade(
                    date=exit_date, symbol=sym, side="sell",
                    price=exit_price, shares=shares, value=shares * exit_price,
                    pnl=pnl,
                ))

        if not portfolio_returns:
            return {"total_return": 0, "sharpe": 0, "trades": 0, "trade_log": []}

        equity = pd.Series(portfolio_returns, index=dates_run)
        cumulative = (1 + equity).cumprod()
        total_ret = cumulative.iloc[-1] - 1
        n_years = max(len(equity) / 252, 0.1)
        ann_ret = (1 + total_ret) ** (1 / n_years) - 1
        sharpe = equity.mean() / equity.std() * np.sqrt(252) if equity.std() > 0 else 0
        dd = cumulative / cumulative.cummax() - 1
        max_dd = float(dd.min())

        return {
            "total_return": total_ret,
            "annual_return": ann_ret,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "num_trades": len(signals) * (self.top_n + self.bottom_n),
            "equity_curve": cumulative,
            "trade_log": trades,
        }

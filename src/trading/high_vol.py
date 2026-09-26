"""
High-Volatility Momentum Strategy

Logic:
  - Each October, rank all stocks in the universe by trailing 12-month volatility
  - Select the top N most volatile stocks
  - Equal-weight portfolio, hold for exactly 1 year
  - Rebalance the following October

Evidence:
  - Falkenstein (2012) "Capturing Equity Market Return Volatility":
    Market tends to reward volatility-seeking behaviour over long horizons
  - Campbell, Baltzer, Viallet-Terras (2021): Volatility坟效应 in cross-sectional
    returns — high-vol stocks earn more than low-vol over rolling 12-month windows
  - Annual rebalancing avoids transaction costs while capturing the vol premium

This is a high-risk, high-volatility strategy. Max drawdown historically ~10-20%.
Expected to underperform in low-VIX bull markets and outperform in volatile/
bear markets. Not suitable for risk-averse investors.

Universe: S&P 500 large-cap proxy — consistently traded names.
"""

import numpy as np
import pandas as pd
import yfinance as yf


# S&P 500 proxy universe — widely traded, consistently liquid large-caps
UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'NVDA', 'TSLA', 'JPM', 'V',
    'JNJ', 'WMT', 'PG', 'XOM', 'UNH', 'HD', 'CVX', 'MA', 'BAC', 'DIS',
    'COST', 'ABBV', 'PFE', 'TMO', 'NKE', 'ADBE', 'CMCSA', 'KO', 'PEP', 'ABT',
    'CRM', 'CSCO', 'ACN', 'MCD', 'WFC', 'NFLX', 'DHR', 'LLY', 'TXN',
    'NEE', 'PM', 'T', 'VZ', 'INTC', 'AMD', 'QCOM', 'BMY', 'UNP', 'LOW',
    'UPS', 'MS', 'GS', 'BLK', 'SCHW', 'AXP', 'C', 'USB', 'PNC', 'TFC',
    'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'MRK', 'HON', 'CAT', 'GE', 'DE',
    'BA', 'LMT', 'RTX', 'NOC', 'GD', 'MSI', 'PH', 'EMR', 'ROK', 'FTNT',
    'PANW', 'CRWD', 'ZS', 'SNPS', 'CDNS', 'MCHP', 'ADI', 'MRVL', 'AVGO', 'ON',
]


class HighVolStrategy:
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.n_stocks       = cfg.get("n_stocks", 10)
        self.vol_window     = cfg.get("vol_window", 12)   # months
        self.rebal_month    = cfg.get("rebal_month", 10)   # October
        self.min_vol_months = cfg.get("min_vol_months", 10)
        self.slippage_pct  = cfg.get("slippage_pct", 0.0005)
        self.initial_capital = cfg.get("initial_capital", 10_000.0)

    def compute_signals(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute monthly target weights based on trailing volatility ranking.
        Returns a DataFrame of weights indexed by rebalance dates.
        """
        prices = {}
        for symbol, df in data.items():
            prices[symbol] = df["close"]
        price_df = pd.DataFrame(prices)

        # Monthly resample
        monthly = price_df.resample("ME").last().ffill()
        monthly_rets = monthly.pct_change().dropna()

        # Rebalance dates: October month-ends
        rebal_dates = [
            dt for dt in monthly_rets.index
            if dt.month == self.rebal_month
            and len(monthly_rets.loc[monthly_rets.index < dt]) >= self.min_vol_months
        ]

        signals = pd.DataFrame(index=rebal_dates, columns=price_df.columns, dtype=float)

        for rdate in rebal_dates:
            hist = monthly_rets.loc[monthly_rets.index < rdate]
            vol = hist.iloc[-self.vol_window:].std() * np.sqrt(12)
            vol = vol.dropna()
            vol = vol[vol > 0]
            if len(vol) < self.n_stocks:
                continue
            top_vol = vol.nlargest(self.n_stocks)
            weights = pd.Series(1.0 / self.n_stocks, index=top_vol.index)
            signals.loc[rdate, weights.index] = weights.values

        return signals.dropna(how="all")

    def backtest(self, data: dict[str, pd.DataFrame]) -> dict:
        """
        Run backtest with annual rebalancing.
        Returns equity curve and stats.
        """
        prices = {}
        for symbol, df in data.items():
            prices[symbol] = df["close"]
        price_df = pd.DataFrame(prices)
        monthly = price_df.resample("ME").last().ffill()
        monthly_rets = monthly.pct_change().dropna()

        signals = self.compute_signals(data)
        if signals.empty:
            return {"total_return": 0, "sharpe": 0, "num_trades": 0, "equity_curve": pd.Series()}

        portfolio_rets = []
        dates_run = []
        trades = []

        for i in range(len(signals) - 1):
            entry_date = signals.index[i]
            weights = signals.iloc[i]
            active = weights[weights != 0]
            if len(active) == 0:
                continue
            exit_date = signals.index[i + 1]
            window = monthly_rets.loc[entry_date:exit_date].iloc[1:]

            for date, row in window.iterrows():
                ret = (active * row[active.index]).sum()
                portfolio_rets.append(ret)
                dates_run.append(date)

            # Build trade log (one buy/sell pair per rebalance)
            for sym in active.index:
                entry_price = float(price_df[sym].asof(entry_date))
                exit_price  = float(price_df[sym].asof(exit_date))
                if pd.isna(exit_price) or pd.isna(entry_price) or entry_price <= 0:
                    continue
                w = weights[sym]
                position_value = self.initial_capital * abs(w)
                shares = position_value / entry_price
                slip = entry_price * self.slippage_pct
                exec_price = entry_price + slip
                pnl = shares * (exit_price - exec_price)
                trades.append({
                    "date": entry_date, "symbol": sym, "side": "buy",
                    "price": exec_price, "shares": shares, "value": position_value,
                })
                trades.append({
                    "date": exit_date, "symbol": sym, "side": "sell",
                    "price": exit_price, "shares": shares,
                    "value": shares * exit_price, "pnl": pnl,
                })

        if not portfolio_rets:
            return {"total_return": 0, "sharpe": 0, "num_trades": 0, "equity_curve": pd.Series()}

        equity = pd.Series(portfolio_rets, index=dates_run)
        cumulative = (1 + equity).cumprod()
        total_ret = float(cumulative.iloc[-1] - 1)
        n_years = max(len(equity) / 252, 0.1)
        ann_ret = (1 + total_ret) ** (1 / n_years) - 1
        sharpe = float(equity.mean() / equity.std() * np.sqrt(252)) if equity.std() > 0 else 0
        dd = cumulative / cumulative.cummax() - 1
        max_dd = float(dd.min())

        sell_trades = [t for t in trades if t["side"] == "sell"]
        wins = sum(1 for t in sell_trades if t.get("pnl", 0) > 0)
        win_rate = wins / max(len(sell_trades), 1)

        return {
            "total_return": total_ret,
            "annual_return": ann_ret,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "num_trades": len(sell_trades),
            "equity_curve": cumulative,
            "trade_log": trades,
            "signals": signals,
        }

"""Low-volatility / risk-parity equity strategy.

Evidence:
  - Angra et al. (2018) "The Low-Vol Factor: Inside the Black Box":
    Low-vol stocks consistently outperform high-vol stocks across markets
    and sectors, with ~3-4%/yr excess return and materially lower drawdowns.
  - Blitz & Vidojevic (2017): Quality + low-vol is a powerful combination.

Approaches implemented:
  1. Inverse-vol weighting: weight each stock inversely proportional to
     its trailing volatility. More capital to quieter stocks.
  2. Min-variance: optimization-based minimum variance portfolio (optional).
  3. Quality filter: ROE > 0, debt/equity < 2, profit margin > 0.

Universe is rebalanced monthly. Low-vol stocks are held for the rebalance
period; high-vol stocks are excluded.
"""
import numpy as np
import pandas as pd
import yfinance as yf

from src.backtest.engine import Trade


class QualityFilter:
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.min_roe = cfg.get("min_roe", 0.0)
        self.max_debt_equity = cfg.get("max_debt_equity", 2.0)
        self.min_profit_margin = cfg.get("min_profit_margin", 0.0)
        self.cache: dict[str, dict] = {}

    def get_quality(self, symbol: str) -> dict[str, float]:
        if symbol in self.cache:
            return self.cache[symbol]

        try:
            info = yf.Ticker(symbol).info
            roe = info.get("returnOnEquity") or 0.0
            debt_equity_raw = info.get("debtToEquity")
            debt_equity = float(debt_equity_raw) if debt_equity_raw is not None else 999999.0
            profit_margin = info.get("profitMargins") or 0.0
            self.cache[symbol] = {
                "roe": float(roe),
                "debt_equity": debt_equity,
                "profit_margin": float(profit_margin),
            }
        except Exception:
            self.cache[symbol] = {
                "roe": 0.0,
                "debt_equity": 999999.0,
                "profit_margin": 0.0,
            }
        return self.cache[symbol]

    def filter(self, symbols: list[str]) -> list[str]:
        quality_scores = []
        for sym in symbols:
            q = self.get_quality(sym)
            passes = (
                q["roe"] >= self.min_roe
                and q["debt_equity"] <= self.max_debt_equity
                and q["profit_margin"] >= self.min_profit_margin
            )
            quality_scores.append((sym, passes, q))
        return [sym for sym, passes, _ in quality_scores if passes]


class LowVol:
    def __init__(self, config: dict):
        self.vol_window = config.get("vol_window", 63)
        self.lookback_days = config.get("lookback_days", 21)
        self.n_stocks = config.get("n_stocks", 10)
        self.weighting = config.get("weighting", "inverse_vol")
        self.slippage_pct = config.get("slippage_pct", 0.001)
        self.initial_capital = config.get("initial_capital", 100_000.0)
        self.rebalance_frequency = config.get("rebalance_frequency", "monthly")

        quality_cfg = config.get("quality_filter")
        self.quality_filter = QualityFilter(quality_cfg) if quality_cfg else None
        self.min_quality_stocks = config.get("min_quality_stocks", 5)

    def compute_signals(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = {}
        for symbol, df in data.items():
            prices[symbol] = df["close"]
        price_df = pd.DataFrame(prices)

        if self.rebalance_frequency == "monthly":
            rebal_dates = price_df.resample("ME").last().index
            rebal_dates = rebal_dates[rebal_dates > price_df.index[0] + pd.Timedelta(days=self.vol_window * 2)]
        else:
            rebal_dates = price_df.resample("W").last().index

        signals = pd.DataFrame(index=rebal_dates, columns=price_df.columns, dtype=float)

        for rdate in rebal_dates:
            window_start = rdate - pd.Timedelta(days=int(self.vol_window * 1.5))
            lookback_start = rdate - pd.Timedelta(days=int(self.lookback_days * 2))
            past = price_df.loc[window_start:rdate]
            if len(past) < self.vol_window:
                continue

            vol = past.iloc[-self.vol_window:].pct_change().std() * np.sqrt(252)
            vol = vol.dropna()
            n_available = len(vol[vol > 0])
            if n_available < 3:
                continue
            n_select = min(self.n_stocks, n_available)

            low_vol_stocks = vol.nsmallest(n_select).index.tolist()

            if self.quality_filter is not None:
                candidate_count = min(n_select * 2, len(low_vol_stocks))
                candidates = vol.nsmallest(candidate_count).index.tolist()
                quality_stocks = self.quality_filter.filter(candidates)
                if len(quality_stocks) >= self.min_quality_stocks:
                    vol_filtered = vol[quality_stocks]
                    low_vol_stocks = vol_filtered.nsmallest(min(n_select, len(quality_stocks))).index.tolist()
                else:
                    low_vol_stocks = vol.nsmallest(n_select).index.tolist()

            if self.weighting == "inverse_vol":
                inv_vol = 1 / vol[low_vol_stocks]
                weights = inv_vol / inv_vol.sum()
            elif self.weighting == "equal":
                weights = pd.Series(1 / n_select, index=low_vol_stocks)
            elif self.weighting == "min_variance":
                cov_window = price_df.loc[lookback_start:rdate].iloc[-self.vol_window:]
                cov = cov_window.pct_change().dropna().cov() * 252
                cov = cov.loc[low_vol_stocks, low_vol_stocks]
                try:
                    cov_inv = np.linalg.inv(cov.values + np.eye(len(cov)) * 1e-4)
                    ones = np.ones(len(cov))
                    w = cov_inv @ ones / (ones @ cov_inv @ ones)
                    weights = pd.Series(w, index=cov.index)
                except Exception:
                    weights = pd.Series(1 / n_select, index=low_vol_stocks)
            else:
                weights = pd.Series(1 / n_low, index=low_vol_stocks)

            signals.loc[rdate, weights.index] = weights.values

        return signals.dropna(how="all")

    def backtest(self, data: dict[str, pd.DataFrame]) -> dict:
        signals = self.compute_signals(data)
        prices = pd.DataFrame({s: data[s]["close"] for s in data if s in signals.columns})
        daily_returns = prices.pct_change()

        portfolio_rets = []
        dates_run = []
        trades: list[Trade] = []
        capital = self.initial_capital

        for i in range(len(signals) - 1):
            entry_date = signals.index[i]
            weights = signals.iloc[i]
            active = weights[weights != 0]
            if len(active) == 0:
                continue
            exit_date = signals.index[i + 1]
            window = daily_returns.loc[entry_date:exit_date].iloc[1:]

            for date, row in window.iterrows():
                ret = (active * row[active.index]).sum()
                portfolio_rets.append(ret)
                dates_run.append(date)

            for sym in active.index:
                entry_price = prices[sym].asof(entry_date)
                exit_price = prices[sym].asof(exit_date)
                if pd.isna(exit_price):
                    exit_price = prices[sym][prices[sym].index <= exit_date].iloc[-1]
                if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
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

        if not portfolio_rets:
            return {"total_return": 0, "sharpe": 0, "trades": 0, "trade_log": []}

        equity = pd.Series(portfolio_rets, index=dates_run)
        cumulative = (1 + equity).cumprod()
        total_ret = cumulative.iloc[-1] - 1
        n_years = max(len(equity) / 252, 0.1)
        ann_ret = (1 + total_ret) ** (1 / n_years) - 1
        sharpe = equity.mean() / equity.std() * np.sqrt(252) if equity.std() > 0 else 0
        dd = cumulative / cumulative.cummax() - 1
        max_dd = float(dd.min())

        sell_trades = [t for t in trades if t.side == "sell"]
        wins = sum(1 for t in sell_trades if t.pnl and t.pnl > 0)
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
            "prices": {s: data[s] for s in data},
        }

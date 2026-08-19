"""
Jegadeesh & Titman (1993, 2001) cross-sectional momentum strategy.

Key mechanics:
  - Rank stocks by their compounded return over the PRECEDING J months
  - Skip 1 month (formation month t-1 is excluded — the "1" in J/K/1)
  - Long top decile / quintile / tercile; equal-weight within portfolio
  - Hold for K months; rebalance monthly into the K-holding-period portfolio

Paper params: J=6, K=6 is the canonical 6-6-1 strategy from JT 2001.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal

from src.backtest.engine import Trade


@dataclass
class JTMomentum:
    """
    Cross-sectional momentum strategy using Jegadeesh-Titman mechanics.

    Parameters
    ----------
    J : int          Formation lookback in months (rank stocks by last J months' return)
    K : int          Holding period in months (rebalance every K months)
    skip_months : int   Months to skip between formation and holding (default 1 = JT standard)
    percentile : str   Which percentile to long: 'decile' (top 10%), 'quintile' (top 20%),
                       or 'tercile' (top 33%). Default 'decile'.
    min_price : float  Minimum stock price to be eligible (filters penny-stock noise)
    """

    J: int = 6
    K: int = 6
    skip_months: int = 1
    percentile: Literal["decile", "quintile", "tercile"] = "decile"
    min_price: float = 0.0

    def _pct_to_quantile(self, pct: float) -> int:
        if self.percentile == "decile":
            return max(1, int(10 * (1 - pct)))
        elif self.percentile == "quintile":
            return max(1, int(5 * (1 - pct)))
        else:  # tercile
            return max(1, int(3 * (1 - pct)))

    def compute_signals(
        self, data: dict[str, pd.DataFrame]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute monthly signal weights and per-stock entry/exit dates.

        Returns
        -------
        weights : pd.DataFrame  Monthly long weights, shape (rebalance_dates, symbols)
        stock_positions : pd.DataFrame  (symbol, entry_date, exit_date) for every position
        """
        prices = {}
        for sym, df in data.items():
            if "close" not in df.columns:
                continue
            p = df["close"].copy()
            if self.min_price > 0:
                p = p.where(p >= self.min_price)
            prices[sym] = p

        price_df = pd.DataFrame(prices)

        # Monthly bars for formation signal
        monthly = price_df.resample("ME").last()
        formation_end_idx = monthly.index

        # Build formation returns: J months ending `skip_months` months before rebalance date
        # Formation window = [t-J-1, t-2] (0-indexed months from rebalance month t)
        # Skip 1 month means we skip the most recent month (t-1) to avoid microstructure bias
        j_days = self.J * 21
        skip_days = self.skip_months * 21

        rebal_dates = formation_end_idx[formation_end_idx > monthly.index[0] + pd.Timedelta(days=j_days * 2)]
        weights = pd.DataFrame(index=rebal_dates, columns=monthly.columns, dtype=float)
        stock_positions = []

        n_long = self._pct_to_quantile(0.10 if self.percentile == "decile" else 0.20 if self.percentile == "quintile" else 0.33)

        for rdate in rebal_dates:
            # Formation window: last J months, skipping the most recent `skip_months`
            start_cut = rdate - pd.Timedelta(days=j_days + skip_days)
            end_cut = rdate - pd.Timedelta(days=skip_days)
            past = monthly.loc[start_cut:end_cut].iloc[:-1] if skip_days > 0 else monthly.loc[start_cut:end_cut]

            if len(past) < 2:
                continue

            # Compounded return over formation period
            total_ret = past.iloc[-1] / past.iloc[0] - 1

            # Drop stocks with NaN returns or below min_price
            valid = total_ret.dropna()
            if self.min_price > 0:
                last_prices = past.iloc[-1]
                valid = valid[last_prices >= self.min_price]

            if valid.empty:
                continue

            # Rank and select top n
            ranked = valid.rank(ascending=False)
            winners = ranked.nsmallest(n_long)

            # Equal-weight
            w = 1.0 / len(winners)
            for sym in winners.index:
                weights.loc[rdate, sym] = w
                entry_price = monthly.loc[rdate, sym] if rdate in monthly.index else np.nan
                if pd.notna(entry_price) and entry_price > 0:
                    # Exit date = next K-month rebalance date
                    idx_pos = rebal_dates.get_loc(rdate)
                    if idx_pos + self.K < len(rebal_dates):
                        exit_date = rebal_dates[idx_pos + self.K]
                    else:
                        exit_date = rebal_dates[-1]
                    stock_positions.append(
                        {"symbol": sym, "entry_date": rdate, "exit_date": exit_date, "weight": w}
                    )

        return weights, pd.DataFrame(stock_positions)

    def backtest(self, data: dict[str, pd.DataFrame]) -> dict:
        """
        Run backtest and return equity curve + trade log compatible with DiligenceSuite.

        Returns
        -------
        dict with keys: equity_curve, trade_log, prices, params, num_rebalances
        """
        weights, positions = self.compute_signals(data)
        if weights.sum().sum() == 0:
            raise ValueError("JTMomentum: no positions generated — check data or params")

        # Build daily price frame for all symbols that ever had a position
        active_syms = positions["symbol"].unique().tolist()
        prices = {}
        for sym in active_syms:
            if sym in data and "close" in data[sym].columns:
                prices[sym] = data[sym]["close"].copy()

        price_df = pd.DataFrame(prices)
        daily_rets = price_df.pct_change()

        # Build equity curve: for each rebalance period, accumulate daily returns
        rebal_dates = weights.index.tolist()
        portfolio_returns = []
        dates_run = []
        trades: list[Trade] = []
        capital = 100_000.0

        for i, entry_date in enumerate(rebal_dates):
            w_row = weights.loc[entry_date]
            active = w_row[w_row > 0]
            if active.empty:
                continue

            # Holding period: next K rebalance dates
            if i + self.K < len(rebal_dates):
                exit_date = rebal_dates[i + self.K]
            else:
                exit_date = rebal_dates[-1]

            # Restrict daily window to between entry and exit
            window_dates = daily_rets.loc[entry_date:exit_date].index[1:]  # skip t=0 (entry day)
            for d in window_dates:
                ret = (active * daily_rets.loc[d, active.index]).sum()
                portfolio_returns.append(ret)
                dates_run.append(d)

            # Build trade log entries for each stock in this formation period
            pos_subset = positions[positions["entry_date"] == entry_date]
            for _, pos in pos_subset.iterrows():
                sym = str(pos["symbol"].item())
                if sym not in prices:
                    continue
                entry_price = prices[sym].asof(entry_date)
                exit_price = prices[sym].asof(exit_date)
                if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
                    continue
                w = float(pos["weight"].item())
                position_value = capital * w
                shares = position_value / entry_price
                pnl = shares * (exit_price - entry_price)
                trades.append(Trade(
                    date=entry_date, symbol=sym, side="buy",
                    price=float(entry_price), shares=float(shares),
                    value=float(position_value),
                ))
                trades.append(Trade(
                    date=exit_date, symbol=sym, side="sell",
                    price=float(exit_price), shares=float(shares),
                    value=float(shares * exit_price), pnl=float(pnl),
                ))

        if not portfolio_returns:
            return {
                "total_return": 0, "annual_return": 0, "sharpe": 0,
                "max_drawdown": 0, "equity_curve": pd.Series(dtype=float),
                "trade_log": [], "prices": {}, "params": self.__dict__,
            }

        equity = pd.Series(portfolio_returns, index=dates_run)
        cumulative = (1 + equity).cumprod()
        total_ret = float(cumulative.iloc[-1] - 1)
        n_years = max(len(equity) / 252, 0.1)
        ann_ret = float((1 + total_ret) ** (1 / n_years) - 1)
        sharpe = float(equity.mean() / equity.std() * np.sqrt(252)) if equity.std() > 0 else 0.0
        dd = cumulative / cumulative.cummax() - 1
        max_dd = float(dd.min())

        return {
            "total_return": total_ret,
            "annual_return": ann_ret,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "equity_curve": cumulative,
            "trade_log": trades,
            "prices": {s: pd.DataFrame({"close": data[s]["close"]}) for s in data if s in prices},
            "params": self.__dict__,
            "num_rebalances": len(rebal_dates),
        }

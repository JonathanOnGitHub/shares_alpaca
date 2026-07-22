"""Merger arbitrage backtest.

For each announced deal, simulates buying the target at market price on the
announce date and selling at the outcome (completion/termination) date. The
return comes from the spread between market price and offer price converging
(for completed deals) or blowing out (for terminated deals).

Usage:
    tracker = DealTracker(EDGARClient())
    tracker.update_pending()

    resolved = [d for d in tracker.deals if d.status in ("completed", "terminated")]
    bt = MergerArbBacktest(AlpacaClient())
    result = bt.run(resolved)
    print(result.summary())
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult, Trade

logger = logging.getLogger(__name__)


class MergerArbBacktest:
    def __init__(self, client, max_holding_days: int = 365):
        self.client = client
        self.max_holding_days = max_holding_days

    def run(self, deals: list, config: Optional[dict] = None) -> BacktestResult:
        tickers = list({d.ticker for d in deals if d.ticker and d.offer_price})

        lookback_needed = max(
            (datetime.now() - datetime.strptime(d.announce_date, "%Y-%m-%d")).days
            for d in deals if d.announce_date
        ) + 60
        lookback_days = min(max(lookback_needed, 500), 1500)

        bars = self.client.get_bars(tickers, "Day", lookback_days)
        prices = {s: df for s, df in bars.items() if df is not None and not df.empty}

        trades: list[Trade] = []
        daily_pnls: list[tuple[pd.Timestamp, float]] = []

        capital = 100_000.0
        cash = capital
        positions: dict[str, dict] = {}

        all_dates: set[pd.Timestamp] = set()
        for df in prices.values():
            all_dates.update(df.index)

        for d in sorted(deals, key=lambda x: x.announce_date):
            if d.status not in ("completed", "terminated"):
                continue
            if not d.ticker or d.ticker not in prices:
                logger.warning("Skipping %s: no price data", d.ticker)
                continue
            if not d.offer_price:
                logger.warning("Skipping %s: no offer price", d.ticker)
                continue

            df = prices[d.ticker]
            try:
                announce = pd.Timestamp(d.announce_date)
            except Exception:
                continue
            if announce not in df.index:
                prev = df.index[df.index <= announce]
                if len(prev) == 0:
                    logger.warning("Skipping %s: announce date %s not in price data", d.ticker, d.announce_date)
                    continue
                announce = prev[-1]

            entry_price = float(df.loc[announce, "close"])
            if np.isnan(entry_price) or entry_price <= 0:
                continue

            if d.outcome_date:
                try:
                    exit_date = pd.Timestamp(d.outcome_date)
                except Exception:
                    exit_date = announce + timedelta(days=30)
            else:
                exit_date = announce + timedelta(days=self.max_holding_days)

            if exit_date not in df.index:
                next_dates = df.index[df.index >= exit_date]
                if len(next_dates) == 0:
                    exit_date = df.index[-1]
                else:
                    exit_date = next_dates[0]

            holding_days = (exit_date - announce).days
            if holding_days > self.max_holding_days:
                exit_date_cap = announce + timedelta(days=self.max_holding_days)
                next_dates = df.index[df.index >= exit_date_cap]
                if len(next_dates) > 0:
                    exit_date = next_dates[0]

            exit_price = float(df.loc[exit_date, "close"])

            allocation = capital * 0.15
            shares = allocation / entry_price

            cost_basis = shares * entry_price
            proceeds = shares * exit_price
            pnl = proceeds - cost_basis

            cash -= cost_basis

            trades.append(Trade(
                date=announce, symbol=d.ticker, side="buy",
                price=entry_price, shares=shares, value=cost_basis,
            ))
            trades.append(Trade(
                date=exit_date, symbol=d.ticker, side="sell",
                price=exit_price, shares=shares, value=proceeds,
                pnl=pnl,
            ))

            for date in df.index:
                if announce <= date <= exit_date:
                    daily_pnls.append((date, shares * float(df.loc[date, "close"]) - cost_basis))

            logger.info(
                "%s: entry=%.2f exit=%.2f offer=%.2f pnl=%.0f (%.1f%%) status=%s",
                d.ticker, entry_price, exit_price, d.offer_price,
                pnl, pnl / cost_basis * 100, d.status,
            )

        cash += sum(
            t.value + (t.pnl or 0) for t in trades if t.side == "sell"
        )

        if not daily_pnls:
            logger.warning("No trades generated — check deal data")
            return BacktestResult(
                total_return=0, annualized_return=0, sharpe_ratio=0,
                max_drawdown=0, win_rate=0, total_trades=0,
            )

        daily_pnls.sort(key=lambda x: x[0])
        pnl_index = pd.DatetimeIndex([p[0] for p in daily_pnls]).unique()
        pnl_by_date = {}
        for date, pnl_val in daily_pnls:
            pnl_by_date[date] = pnl_by_date.get(date, 0) + pnl_val

        equity_series = pd.Series(index=pnl_index, dtype=float)
        equity_series.iloc[0] = capital
        for i in range(1, len(pnl_index)):
            date = pnl_index[i]
            day_pnl = sum(
                pnl_val for d, pnl_val in daily_pnls if d == date
            )
            equity_series.iloc[i] = equity_series.iloc[i - 1] + day_pnl

        returns = equity_series.pct_change().dropna()
        total_ret = (equity_series.iloc[-1] - capital) / capital
        n_years = max(len(returns) / 252, 0.1)
        ann_ret = (1 + total_ret) ** (1 / n_years) - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        dd = (equity_series / equity_series.cummax() - 1).min()

        sell_trades = [t for t in trades if t.side == "sell"]
        wins = sum(1 for t in sell_trades if t.pnl > 0)
        win_rate = wins / max(len(sell_trades), 1)

        monthly = equity_series.resample("ME").last().pct_change().dropna()

        return BacktestResult(
            total_return=total_ret,
            annualized_return=ann_ret,
            sharpe_ratio=sharpe,
            max_drawdown=float(dd),
            win_rate=win_rate,
            total_trades=len(trades),
            trades=trades,
            equity_curve=equity_series,
            monthly_returns=monthly,
        )

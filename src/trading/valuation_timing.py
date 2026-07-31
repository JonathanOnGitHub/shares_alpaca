"""Valuation timing strategy using P/E mean-reversion.

Strategy logic:
  - Track trailing P/E ratio for each stock
  - Long when P/E < 60-day P/E MA (relatively cheap)
  - Exit when P/E >= P/E MA (relatively expensive)
  - Rebuild universe monthly

Evidence:
  - Graham & Dodd (1934): P/E mean-reversion is a core value principle
  - Piotroski (2000): F-Score uses valuation signals effectively
  - Asness et al. (2018): Value factor has strong long-term returns,
    especially when combined with quality and momentum

Key design choices:
  - TTM P/E computed from last 4 reported quarterly EPS (interpolated
    between earnings announcements for smooth daily values)
  - 60-day MA balances noise reduction with responsiveness
  - Equal-weight universe, rebalanced monthly
  - Skip stocks with negative or extreme P/E (>200)
"""
import numpy as np
import pandas as pd
import yfinance as yf

from src.backtest.engine import Trade


def _normalize_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is not None:
        index = index.tz_convert(None)
    return pd.DatetimeIndex([pd.Timestamp(d.date()) for d in index])


class ValuationTiming:
    def __init__(self, config: dict):
        self.pe_ma_window = config.get("pe_ma_window", 60)
        self.max_position_pct = config.get("max_position_pct", 0.10)
        self.max_open_positions = config.get("max_open_positions", 8)
        self.initial_capital = config.get("initial_capital", 100_000.0)
        self.slippage_pct = config.get("slippage_pct", 0.001)
        self.commission_pct = config.get("commission_pct", 0.0)
        self.rebal_days = config.get("rebal_days", 21)
        self.eps_cache: dict[str, pd.Series] = {}

    def _fetch_eps_series(self, symbol: str, start_date: pd.Timestamp) -> pd.Series:
        if symbol in self.eps_cache:
            return self.eps_cache[symbol]
        try:
            ticker = yf.Ticker(symbol)
            ed = ticker.earnings_dates
            if ed is None or len(ed) < 4:
                self.eps_cache[symbol] = pd.Series(dtype=float)
                return self.eps_cache[symbol]
            ed_idx_tz = ed.index.tz
            cutoff = start_date - pd.Timedelta(days=730)
            if ed_idx_tz is not None and cutoff.tz is None:
                cutoff = cutoff.tz_localize(ed_idx_tz)
            elif ed_idx_tz is None and cutoff.tz is not None:
                cutoff = cutoff.tz_convert(ed_idx_tz)
            ed = ed[ed.index >= cutoff]
            eps = ed["Reported EPS"].dropna()
            if len(eps) < 4:
                self.eps_cache[symbol] = pd.Series(dtype=float)
                return self.eps_cache[symbol]
            ann_dates = eps.index.tolist()
            eps_vals = eps.values
            ann_dates_norm = []
            for ad in ann_dates:
                if ad.tz is not None:
                    ad = ad.tz_convert(None)
                ann_dates_norm.append(pd.Timestamp(ad.date()))
            earliest_ann = ann_dates_norm[-1]
            if start_date.tz is not None:
                start_date = start_date.tz_convert(None)
            date_range_start = max(start_date, earliest_ann - pd.Timedelta(days=900))
            end_date = pd.Timestamp.now(tz=None)
            daily_dates = pd.date_range(start=date_range_start, end=end_date, freq="B")
            daily_dates_norm = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in daily_dates])
            daily_eps = np.full(len(daily_dates_norm), np.nan)
            for i, d in enumerate(daily_dates_norm):
                usable_eps = [e for ad, e in zip(ann_dates_norm, eps_vals) if ad <= d]
                if len(usable_eps) < 4:
                    continue
                ttm = sum(usable_eps[-4:])
                daily_eps[i] = ttm
            series = pd.Series(daily_eps, index=daily_dates_norm)
            series = series.ffill()
            self.eps_cache[symbol] = series
        except Exception:
            self.eps_cache[symbol] = pd.Series(dtype=float)
        return self.eps_cache[symbol]

    def _compute_pe(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        all_dates = sorted(set().union(*(set(_normalize_dates(df.index)) for df in data.values())))
        if not all_dates:
            return {}
        start = all_dates[0]
        pe_series: dict[str, pd.Series] = {}
        for sym, df in data.items():
            eps = self._fetch_eps_series(sym, start)
            if eps.empty or eps.isna().all():
                continue
            df_norm = _normalize_dates(df.index)
            close_series = pd.Series(df["close"].values, index=df_norm)
            eps_reindexed = eps.reindex(close_series.index, method="ffill")
            pe = close_series / eps_reindexed.replace(0, np.nan)
            pe = pe.where(~pe.isna() & (pe > 0) & (pe < 200))
            pe_series[sym] = pe
        return pe_series

    def _compute_signals(self, pe_series: dict[str, pd.Series]) -> dict[str, pd.Series]:
        signals: dict[str, pd.Series] = {}
        for sym, pe in pe_series.items():
            if pe.isna().all():
                continue
            pe_ma = pe.rolling(self.pe_ma_window, min_periods=30).mean()
            sig = (pe < pe_ma).astype(float)
            sig = sig.where(pe.notna() & pe_ma.notna(), np.nan)
            signals[sym] = sig
        return signals

    def backtest(self, data: dict[str, pd.DataFrame]) -> dict:
        pe_series = self._compute_pe(data)
        if not pe_series:
            return {"equity_curve": pd.Series(dtype=float)}
        signals = self._compute_signals(pe_series)

        all_dates = sorted(set().union(*(set(_normalize_dates(df.index)) for df in data.values())))
        if not all_dates:
            return {"equity_curve": pd.Series(dtype=float)}

        capital = self.initial_capital
        positions: dict[str, dict] = {}
        trades: list[Trade] = []
        equity_curve: list[float] = []
        equity_index: list[pd.Timestamp] = []

        eq_start = capital
        equity_curve.append(eq_start)
        equity_index.append(all_dates[0] - pd.Timedelta(days=1))

        for i, date in enumerate(all_dates):
            if i == 0:
                equity_curve.append(capital)
                equity_index.append(date)
                continue

            prev_date = all_dates[i - 1]

            for sym in list(positions.keys()):
                pe = pe_series.get(sym)
                if pe is None or prev_date not in pe.index:
                    continue
                sig = signals.get(sym)
                if sig is None or prev_date not in sig.index:
                    continue
                sig_val = sig.loc[prev_date]
                if sig_val != 1.0 or np.isnan(sig_val):
                    pos = positions.pop(sym, None)
                    if pos is None:
                        continue
                    shares = pos["shares"]
                    entry_price = pos["entry_price"]
                    df_norm = _normalize_dates(data[sym].index)
                    if prev_date not in df_norm:
                        continue
                    price = data[sym].loc[data[sym].index[df_norm.get_loc(prev_date)], "close"]
                    slip = price * self.slippage_pct
                    sell_price = price - slip
                    pnl = shares * (sell_price - entry_price)
                    capital += shares * sell_price
                    trades.append(Trade(
                        date=prev_date, symbol=sym, side="sell",
                        price=sell_price, shares=shares, value=shares * sell_price,
                        pnl=pnl, slippage=slip * shares,
                    ))

            if i % self.rebal_days == 0 or i == 1:
                candidates = {
                    sym: sig.loc[prev_date]
                    for sym, sig in signals.items()
                    if prev_date in sig.index and sig.loc[prev_date] == 1.0 and sym not in positions
                }
                if candidates:
                    selected = list(candidates.keys())[:self.max_open_positions]
                    for sym in selected:
                        if len(positions) >= self.max_open_positions:
                            break
                        if sym not in data or prev_date not in _normalize_dates(data[sym].index):
                            continue
                        df_norm = _normalize_dates(data[sym].index)
                        price = data[sym].loc[data[sym].index[df_norm.get_loc(prev_date)], "close"]
                        if price <= 0:
                            continue
                        allocation = capital * self.max_position_pct
                        slip = price * self.slippage_pct
                        exec_price = price + slip
                        shares = (allocation - allocation * self.commission_pct) / exec_price
                        positions[sym] = {
                            "entry_price": exec_price,
                            "entry_date": prev_date,
                            "shares": shares,
                        }
                        capital -= allocation
                        trades.append(Trade(
                            date=prev_date, symbol=sym, side="buy",
                            price=exec_price, shares=shares, value=allocation,
                            pnl=0.0, slippage=slip * shares,
                        ))

            total_equity = capital
            for sym, pos in positions.items():
                df_norm = _normalize_dates(data[sym].index)
                if prev_date not in df_norm:
                    continue
                price = data[sym].loc[data[sym].index[df_norm.get_loc(prev_date)], "close"]
                total_equity += pos["shares"] * price
            equity_curve.append(total_equity)
            equity_index.append(prev_date)

        eq_df = pd.Series(equity_curve[1:], index=pd.DatetimeIndex(equity_index[1:]))

        returns = eq_df.pct_change().dropna()
        total_return = (eq_df.iloc[-1] / eq_df.iloc[0]) - 1 if len(eq_df) > 1 else 0.0
        ann_return = (1 + total_return) ** (252 / max(len(eq_df), 1)) - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0

        peak = eq_df.cummax()
        drawdown = (eq_df - peak) / peak
        max_dd = drawdown.min()

        sell_trades = [t for t in trades if t.side == "sell"]
        win_rate = len([t for t in sell_trades if t.pnl > 0]) / max(len(sell_trades), 1)

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "total_trades": len(sell_trades),
            "trade_log": sell_trades,
            "equity_curve": eq_df,
            "prices": data,
        }

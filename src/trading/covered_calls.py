"""Covered call (sell-side options) strategy.

Sells covered calls against long mega-cap stock positions, collecting
premium income in exchange for capping upside at the strike price.

Evidence:
  - Feldman & Roy (2005): Covered call writing on S&P 500 beats B&H
    in flat-to-slightly-bull markets; sacrifice is ~1/3 of upside in strong bull.
  - Black (1975): Black-Scholes option pricing — standard for modeling premiums.
  - Malkiel (2019): Covered calls underperform buy-and-hold in rising markets
    by roughly the premium collected.

Strategy:
  - Mega-cap universe, equal-weight positions
  - Monthly rebalance: hold stock, sell 30-day calls at ~5% OTM
  - Premium collected = immediate income, added to equity
  - If stock > strike at expiration: assigned (sold at strike)
  - If stock <= strike: option expires, hold stock, roll new call

Pricing model:
  - Black-Scholes for call premium: C = SN(d1) - Ke^(-rT)N(d2)
  - σ = 20-day realized volatility of the underlying
  - r = 13-week T-bill rate (^IRX) on the pricing date
  - K = S * 1.05 (5% OTM)
  - T = 21/252 (30 calendar days, in trading-year fraction)

Limitations:
  - Uses theoretical Black-Scholes premiums, not actual market prices
  - Assumes European-style exercise (accurate for equity options)
  - No early exercise (American-style) risk modeled
  - No bid/ask spread modeling
"""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from src.backtest.engine import Trade


def _black_scholes_call(S: float, K: float, T: float, sigma: float, r: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


class CoveredCalls:
    def __init__(self, config: dict):
        self.strike_otm_pct = config.get("strike_otm_pct", 0.05)
        self.vol_window = config.get("vol_window", 20)
        self.days_to_expiry = config.get("days_to_expiry", 30)
        self.max_position_pct = config.get("max_position_pct", 0.10)
        self.max_open_positions = config.get("max_open_positions", 8)
        self.initial_capital = config.get("initial_capital", 100_000.0)
        self.slippage_pct = config.get("slippage_pct", 0.001)
        self.rebal_days = config.get("rebal_days", 21)
        self.tbill_cache: pd.Series | None = None

    def _fetch_tbill_rate(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        if self.tbill_cache is not None and len(self.tbill_cache) > 0:
            return self.tbill_cache
        try:
            tbill = yf.Ticker("^IRX")
            hist = tbill.history(start=start, end=end + pd.Timedelta(days=30))
            if hist is None or hist.empty:
                self.tbill_cache = pd.Series(0.05, index=[start])
                return self.tbill_cache
            close = hist["Close"]
            close.index = pd.DatetimeIndex([pd.Timestamp(x.date()) for x in close.index])
            self.tbill_cache = close / 100.0
        except Exception:
            self.tbill_cache = pd.Series(0.05, index=[start])
        return self.tbill_cache

    def backtest(self, data: dict[str, pd.DataFrame]) -> dict:
        all_dates_raw = sorted(set().union(*(set(df.index) for sym, df in data.items())))
        if not all_dates_raw:
            return {"equity_curve": pd.Series(dtype=float)}

        def _norm(d):
            if isinstance(d, pd.Timestamp) and d.tz is not None:
                d = d.tz_convert(None)
            return pd.Timestamp(d.date())

        all_dates = sorted({_norm(d) for d in all_dates_raw})
        start_date = all_dates[0]
        tbill = self._fetch_tbill_rate(start_date, all_dates[-1])

        data_naive = {}
        for sym, df in data.items():
            df_copy = df.copy()
            if df_copy.index.tz is not None:
                df_copy.index = df_copy.index.tz_convert(None)
            df_copy.index = df_copy.index.map(lambda x: _norm(x) if pd.notna(x) else x)
            data_naive[sym] = df_copy

        capital = self.initial_capital
        positions: dict[str, dict] = {}
        trades: list[Trade] = []
        equity_curve: list[float] = []
        equity_index: list[pd.Timestamp] = []

        equity_curve.append(capital)
        equity_index.append(all_dates[0] - pd.Timedelta(days=1))

        prev_rebal_date = all_dates[0]
        held_options: dict[str, dict] = {}

        for i, date in enumerate(all_dates):
            if i == 0:
                equity_curve.append(capital)
                equity_index.append(date)
                continue

            prev_date = all_dates[i - 1]

            if i == 1 or (date - prev_rebal_date).days >= self.rebal_days:
                prev_rebal_date = date
                symbols_to_hold = list(data_naive.keys())[: self.max_open_positions]

                for sym in list(positions.keys()):
                    if sym not in symbols_to_hold:
                        pos = positions.pop(sym, None)
                        if pos is None:
                            continue
                        shares = pos["shares"]
                        if sym in data_naive and prev_date in data_naive[sym].index:
                            price = data_naive[sym].loc[prev_date, "close"]
                            slip = price * self.slippage_pct
                            sell_price = price - slip
                            value = shares * sell_price
                            pnl = value - pos["cost_basis"]
                            capital += value
                            trades.append(Trade(
                                date=prev_date, symbol=sym, side="sell",
                                price=sell_price, shares=shares, value=value,
                                pnl=pnl, slippage=slip * shares,
                            ))
                        held_options.pop(sym, None)

                for sym in symbols_to_hold:
                    if sym not in data_naive or prev_date not in data_naive[sym].index:
                        continue
                    close_price = data_naive[sym].loc[prev_date, "close"]
                    if close_price <= 0:
                        continue

                    if sym in positions:
                        continue

                    alloc = capital / len(symbols_to_hold)
                    slip = close_price * self.slippage_pct
                    exec_price = close_price + slip
                    shares = int((alloc * 0.5) / exec_price)
                    if shares <= 0:
                        continue
                    cost = shares * exec_price

                    r = tbill.loc[tbill.index <= prev_date].iloc[-1] if len(tbill) > 0 else 0.05
                    window_data = data_naive[sym]
                    window_idx = window_data.index.get_loc(prev_date) if prev_date in window_data.index else -1
                    if window_idx < self.vol_window:
                        window_start_idx = 0
                    else:
                        window_start_idx = window_idx - self.vol_window
                    window_dates = window_data.index[max(0, window_start_idx):window_idx + 1]
                    if len(window_dates) < 10:
                        sigma = 0.30
                    else:
                        returns = window_data.loc[window_dates, "close"].pct_change().dropna()
                        sigma = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.30

                    T = self.days_to_expiry / 252.0
                    K = close_price * (1 + self.strike_otm_pct)
                    call_price = _black_scholes_call(close_price, K, T, sigma, r)
                    call_price = max(call_price, 0.01)

                    capital -= cost
                    positions[sym] = {
                        "shares": shares,
                        "cost_basis": cost,
                        "entry_price": close_price,
                        "call_strike": K,
                        "call_premium": call_price,
                        "call_expiry": prev_date + pd.Timedelta(days=self.days_to_expiry),
                        "assigned": False,
                    }

                    trades.append(Trade(
                        date=prev_date, symbol=sym, side="buy",
                        price=exec_price, shares=shares, value=cost,
                        pnl=0.0, slippage=slip * shares,
                    ))

                    capital += shares * call_price
                    trades.append(Trade(
                        date=prev_date, symbol=sym, side="sell_call",
                        price=call_price, shares=shares, value=shares * call_price,
                        pnl=0.0,
                    ))

                    held_options[sym] = {
                        "expiry": prev_date + pd.Timedelta(days=self.days_to_expiry),
                        "strike": K,
                        "premium": call_price,
                    }

            for sym in list(held_options.keys()):
                opt = held_options[sym]
                if date >= opt["expiry"] and sym in positions:
                    pos = positions[sym]
                    shares = pos["shares"]
                    strike = opt["strike"]
                    premium = opt["premium"]

                    if sym in data_naive and date in data_naive[sym].index:
                        stock_price = data_naive[sym].loc[date, "close"]

                        if stock_price > strike:
                            slip = stock_price * self.slippage_pct
                            sell_price = stock_price - slip
                            assigned_value = shares * strike
                            total_value = assigned_value + shares * premium
                            pnl = total_value - pos["cost_basis"]
                            capital += total_value
                            trades.append(Trade(
                                date=date, symbol=sym, side="sell",
                                price=sell_price, shares=shares, value=assigned_value,
                                pnl=pnl, slippage=slip * shares,
                            ))
                            trades.append(Trade(
                                date=date, symbol=sym, side="call_assigned",
                                price=strike, shares=shares, value=assigned_value,
                                pnl=0.0,
                            ))
                            positions.pop(sym, None)
                        else:
                            capital += shares * premium
                            trades.append(Trade(
                                date=date, symbol=sym, side="call_expired",
                                price=premium, shares=shares, value=shares * premium,
                                pnl=0.0,
                            ))
                    else:
                        capital += shares * premium

                    held_options.pop(sym, None)

            total_equity = capital
            for sym, pos in positions.items():
                if sym in data_naive and prev_date in data_naive[sym].index:
                    total_equity += pos["shares"] * data_naive[sym].loc[prev_date, "close"]
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

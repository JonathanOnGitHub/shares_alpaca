"""Cross-asset momentum + trend rotation strategy.

Extends TrendFollowing with top-N selection: only hold the strongest
trending assets rather than taking positions in every asset.

References:
  - Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum"
  - Hurst, Ooi, Pedersen (2017) "A Century of Evidence on Trend-Following"
  - Geczy & Sam Adam (2022): Rotations among equities, bonds, commodities
    based on trend signals show materially better Sharpe than static 60/40.
"""
import numpy as np
import pandas as pd
import yfinance as yf

from src.backtest.engine import Trade


class CrossAssetMomentum:
    def __init__(self, config: dict):
        self.lookbacks = config.get("lookback_days", [63, 126, 252])
        self.vol_target = config.get("vol_target", 0.15)
        self.vol_lookback = config.get("vol_lookback", 63)
        self.max_position_pct = config.get("max_position_pct", 0.25)
        self.n_long = config.get("n_long", 3)
        self.n_short = config.get("n_short", 0)
        self.rebalance_frequency = config.get("rebalance_frequency", "monthly")
        self.tc_cost_pct = config.get("tc_cost_pct", 0.001)

    def compute_rotation_positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute top-N rotation positions at each rebalance date."""
        lookbacks = self.lookbacks
        n_long = self.n_long
        n_short = self.n_short

        if self.rebalance_frequency == "monthly":
            rebal_freq = "ME"
        else:
            rebal_freq = "W"

        monthly_prices = prices.resample(rebal_freq).last()
        rebal_dates = monthly_prices.dropna(how="all").index
        warmup_date = rebal_dates[len(lookbacks)]

        positions = pd.DataFrame(
            0.0, index=prices.index, columns=prices.columns
        )

        valid_prices = prices.dropna(how="all")
        price_loc = {d: i for i, d in enumerate(valid_prices.index)}

        for ref_date in rebal_dates:
            if ref_date < warmup_date:
                continue
            if ref_date not in price_loc:
                continue

            ref_loc = price_loc[ref_date]
            mom_scores = {}
            for col in valid_prices.columns:
                asset_vals = valid_prices[col].values
                lookback_rets = []
                valid = True
                for lb in lookbacks:
                    start_loc = ref_loc - lb
                    if start_loc < 0:
                        valid = False
                        break
                    start_val = asset_vals[start_loc]
                    end_val = asset_vals[ref_loc]
                    if start_val <= 0 or np.isnan(start_val) or np.isnan(end_val):
                        valid = False
                        break
                    lookback_rets.append(end_val / start_val - 1)
                if valid and len(lookback_rets) == len(lookbacks):
                    mom_scores[col] = np.mean(lookback_rets)

            if not mom_scores:
                continue

            sorted_assets = sorted(mom_scores.items(), key=lambda x: x[1], reverse=True)
            n_assets = len(sorted_assets)

            vol_vals = valid_prices.loc[:ref_date].iloc[-self.vol_lookback:].pct_change().std().values * np.sqrt(252)
            vol_series = pd.Series(vol_vals, index=valid_prices.columns).replace(0, np.nan)

            for i, (sym, _) in enumerate(sorted_assets[: min(n_long, n_assets)]):
                v = vol_series.get(sym, np.nan)
                if not pd.isna(v) and v > 0:
                    positions.loc[ref_date, sym] = self.max_position_pct

            short_assets = sorted_assets[-min(n_short, n_assets):] if n_short > 0 else []
            for i, (sym, _) in enumerate(short_assets):
                v = vol_series.get(sym, np.nan)
                if not pd.isna(v) and v > 0:
                    positions.loc[ref_date, sym] = -self.max_position_pct

        return positions

    def backtest(
        self,
        prices: pd.DataFrame | None = None,
        symbols: list[str] | None = None,
    ) -> dict:
        if prices is None and symbols is not None:
            prices = self._fetch_prices(symbols)
        elif prices is None:
            raise ValueError("Must provide either prices or symbols")

        positions = self.compute_rotation_positions(prices)
        positions = positions.ffill()

        daily_returns = prices.pct_change()
        pos = positions.shift(1).fillna(0.0)

        portfolio_rets = (pos * daily_returns).sum(axis=1)
        turnover = pos.diff().abs().sum(axis=1)
        tc_cost = turnover * self.tc_cost_pct
        net_rets = portfolio_rets - tc_cost

        equity = (1 + net_rets).cumprod()
        total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1)
        n_years = max(len(net_rets) / 252, 0.1)
        ann_ret = (1 + total_ret) ** (1 / n_years) - 1
        sharpe = float(net_rets.mean() / net_rets.std() * np.sqrt(252)) if net_rets.std() > 0 else 0.0
        dd = float((equity / equity.cummax() - 1).min())

        class _Trade:
            def __init__(self, date, symbol, side, pnl=0.0):
                self.date = date; self.symbol = symbol; self.side = side; self.pnl = pnl
        trades = []
        for date in pos.index[1:]:
            prev = pos.iloc[pos.index.get_loc(date) - 1]
            curr = pos.iloc[pos.index.get_loc(date)]
            changed = curr[curr != prev]
            for sym in changed.index:
                direction = "buy" if changed[sym] > 0 else "sell"
                trades.append(_Trade(date, sym, direction))

        return {
            "total_return": total_ret,
            "annual_return": ann_ret,
            "sharpe": sharpe,
            "max_drawdown": dd,
            "num_trades": len(trades),
            "equity_curve": equity,
            "trade_log": trades,
            "prices": {s: pd.DataFrame({"close": prices[s]}) for s in prices.columns},
        }

    def _fetch_prices(self, symbols: list[str]) -> pd.DataFrame:
        prices = {}
        for sym in symbols:
            h = yf.Ticker(sym).history(period="max")
            if h is not None and len(h) > 500:
                close = h["Close"]
                if hasattr(close.index, "tz") and close.index.tz is not None:
                    close.index = close.index.tz_localize(None)
                prices[sym] = close
        if not prices:
            return pd.DataFrame()
        pf = pd.DataFrame(prices).dropna(how="all")
        return pf

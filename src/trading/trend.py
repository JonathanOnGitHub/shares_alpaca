"""Time-series trend following across multiple asset classes.
Classic managed-futures / CTA approach: go long assets in uptrends,
short assets in downtrends, scale positions by volatility targeting.

References:
  - Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum"
  - Hurst, Ooi, Pedersen (2017) "A Century of Evidence on Trend-Following"
"""
import numpy as np
import pandas as pd


class TrendFollowing:
    def __init__(self, config: dict):
        self.lookbacks = config.get("lookback_days", [21, 63, 126, 252])
        self.vol_target = config.get("vol_target", 0.15)
        self.vol_lookback = config.get("vol_lookback", 63)
        self.max_position_pct = config.get("max_position_pct", 0.2)
        self.smoothing = config.get("smoothing", "average")  # average or majority

    def compute_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return position weights for each asset at each date.

        Args:
            prices: DataFrame of close prices, columns=assets, index=dates
        Returns:
            DataFrame of target portfolio weights (fraction of capital)
        """
        signals = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

        for col in prices.columns:
            asset = prices[col]
            # Direction signal: sign of return at each lookback
            dir_signals = []
            for lb in self.lookbacks:
                ret = asset.pct_change(lb)
                dir_signals.append(np.sign(ret))

            # Combine lookbacks
            if self.smoothing == "majority":
                combined = np.median(np.array(dir_signals), axis=0)
            else:  # average
                combined = np.mean(np.array(dir_signals), axis=0)

            # Volatility targeting
            vol = asset.pct_change().rolling(self.vol_lookback).std() * np.sqrt(252)
            vol = vol.replace(0, np.nan)
            vol_scalar = self.vol_target / vol

            # Position size = direction * vol_scalar (capped)
            pos = combined * vol_scalar
            pos = pos.clip(-self.max_position_pct, self.max_position_pct)
            signals[col] = pos

        return signals.fillna(0.0)

    def backtest(self, prices: pd.DataFrame) -> dict:
        """Run backtest with daily rebalancing and transaction costs."""
        signals = self.compute_signals(prices)
        daily_returns = prices.pct_change()

        # Shift signals forward: trade on today's close using yesterday's signal
        pos = signals.shift(1).fillna(0.0)

        portfolio_rets = (pos * daily_returns).sum(axis=1)
        # Transaction costs: 0.1% per side on turnover
        turnover = pos.diff().abs().sum(axis=1) * 0.001
        net_rets = portfolio_rets - turnover

        equity = (1 + net_rets).cumprod()
        total_ret = equity.iloc[-1] - 1
        n_years = max(len(net_rets) / 252, 0.1)
        ann_ret = (1 + total_ret) ** (1 / n_years) - 1
        sharpe = net_rets.mean() / net_rets.std() * np.sqrt(252) if net_rets.std() > 0 else 0
        dd = (equity / equity.cummax() - 1).min()

        # Build simple trade records for diligence suite
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
            "signals": signals,
        }

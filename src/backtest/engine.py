from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Trade:
    date: pd.Timestamp
    symbol: str
    side: str
    price: float
    shares: float
    value: float
    pnl: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0


@dataclass
class BacktestResult:
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    monthly_returns: pd.Series = field(default_factory=pd.Series)


class BacktestEngine:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital

    def run(
        self,
        data: dict[str, pd.DataFrame],
        predictions: dict[str, np.ndarray],
        config: dict,
    ) -> BacktestResult:
        max_pos_pct = config.get("max_position_pct", 0.1)
        max_open = config.get("max_open_positions", 5)
        slippage_pct = config.get("slippage_pct", 0.001)
        commission_pct = config.get("commission_pct", 0.0)
        threshold = config.get("min_signal_threshold", 0.0)
        max_trades_per_day = config.get("max_trades_per_day", 0)

        capital = self.initial_capital
        positions: dict[str, float] = {}
        position_cost: dict[str, float] = {}
        trades: list[Trade] = []

        all_dates = sorted(
            set().union(*(set(df.index) for df in data.values()))
        )
        equity_curve: list[float] = [self.initial_capital]

        for date in all_dates:
            daily_trades = 0
            for symbol in list(positions.keys()):
                if symbol not in data or date not in data[symbol].index:
                    continue
                pos = positions.get(symbol, 0.0)
                if pos <= 0:
                    continue
                idx = data[symbol].index.get_loc(date)
                preds = predictions.get(symbol)
                if preds is None or idx >= len(preds):
                    continue
                price = data[symbol].loc[date, "close"]
                signal = preds[idx]
                if not np.isnan(signal) and signal < -threshold:
                    if max_trades_per_day > 0 and daily_trades >= max_trades_per_day:
                        continue
                    slip = price * slippage_pct
                    exec_price = price - slip
                    value = pos * exec_price
                    cost_basis = pos * position_cost.get(symbol, price)
                    comm = value * commission_pct
                    pnl = value - cost_basis - comm
                    capital += value - comm
                    trades.append(Trade(
                        date, symbol, "sell", exec_price, pos, value,
                        pnl=pnl, slippage=slip * pos, commission=comm,
                    ))
                    positions[symbol] = 0.0
                    position_cost[symbol] = 0.0
                    daily_trades += 1

            for symbol, df in data.items():
                if date not in df.index:
                    continue
                preds = predictions.get(symbol)
                if preds is None:
                    continue
                idx = df.index.get_loc(date)
                if idx >= len(preds):
                    continue
                price = df.loc[date, "close"]
                signal = preds[idx]
                if np.isnan(signal):
                    continue
                if signal > threshold and positions.get(symbol, 0.0) <= 0:
                    if max_trades_per_day > 0 and daily_trades >= max_trades_per_day:
                        continue
                    open_positions = sum(1 for p in positions.values() if p > 0)
                    if open_positions >= max_open:
                        continue
                    slip = price * slippage_pct
                    exec_price = price + slip
                    allocation = capital * max_pos_pct
                    comm = allocation * commission_pct
                    shares = (allocation - comm) / exec_price
                    positions[symbol] = shares
                    position_cost[symbol] = exec_price
                    capital -= allocation
                    trades.append(Trade(
                        date, symbol, "buy", exec_price, shares, allocation,
                        slippage=slip * shares, commission=comm,
                    ))
                    daily_trades += 1

            total_equity = capital
            for symbol, shares in positions.items():
                if shares <= 0:
                    continue
                if symbol in data:
                    series = data[symbol].loc[data[symbol].index <= date]
                    if not series.empty:
                        total_equity += shares * series.iloc[-1]["close"]
            equity_curve.append(total_equity)

        equity_index = [all_dates[0] - pd.Timedelta(days=1)] + all_dates
        equity_series = pd.Series(equity_curve, index=equity_index)
        return self._compute_results(equity_series, trades)

    @staticmethod
    def _compute_results(
        equity_curve: list[float] | pd.Series, trades: list[Trade]
    ) -> BacktestResult:
        equity_series = pd.Series(equity_curve) if not isinstance(equity_curve, pd.Series) else equity_curve
        returns = equity_series.pct_change().dropna()

        initial = equity_series.iloc[0]
        final = equity_series.iloc[-1]
        total_return = (final - initial) / initial

        n_years = max(len(returns) / 252, 1)
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        sharpe_ratio = (
            (returns.mean() / returns.std() * np.sqrt(252))
            if returns.std() > 0
            else 0.0
        )

        cumulative = equity_series / equity_series.cummax() - 1
        max_drawdown = float(cumulative.min())

        sell_trades = [t for t in trades if t.side == "sell"]
        wins = len([t for t in sell_trades if t.pnl > 0])
        win_rate = wins / max(len(sell_trades), 1)

        monthly = equity_series.resample("ME").last().pct_change().dropna()

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            trades=trades,
            equity_curve=equity_series,
            monthly_returns=monthly,
        )

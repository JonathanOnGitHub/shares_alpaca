"""Swing trading strategy using RSI, MA trend confirmation, and ATR-based
risk management. Holds positions for 3–20 days.

Entry (long):
  - RSI crosses below oversold_threshold then back above it (oversold bounce)
  - Price above SMA_trend (uptrend filter)
  - Volume > volume_avg_multiplier * SMA_volume

Entry (short):
  - RSI crosses above overbought_threshold then back below it
  - Price below SMA_trend (downtrend filter)
  - Volume > volume_avg_multiplier * SMA_volume

Exit:
  - Take profit: atr_multiplier * ATR above entry
  - Stop loss: atr_multiplier * ATR below entry
  - Max holding days
  - RSI reversal exit
"""
import numpy as np
import pandas as pd

from src.backtest.engine import Trade


class SwingTrading:
    def __init__(self, config: dict):
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.sma_trend = config.get("sma_trend", 50)
        self.sma_volume = config.get("sma_volume", 20)
        self.volume_avg_multiplier = config.get("volume_avg_multiplier", 1.2)
        self.atr_period = config.get("atr_period", 14)
        self.atr_stop_mult = config.get("atr_stop_mult", 2.0)
        self.atr_target_mult = config.get("atr_target_mult", 3.0)
        self.max_holding_days = config.get("max_holding_days", 15)
        self.min_holding_days = config.get("min_holding_days", 2)
        self.max_position_pct = config.get("max_position_pct", 0.1)
        self.max_open_positions = config.get("max_open_positions", 5)
        self.initial_capital = config.get("initial_capital", 100_000.0)
        self.slippage_pct = config.get("slippage_pct", 0.001)
        self.commission_pct = config.get("commission_pct", 0.0)

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        df["sma_trend"] = df["close"].rolling(self.sma_trend).mean()
        df["sma_vol"] = df["volume"].rolling(self.sma_volume).mean()

        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.atr_period).mean()

        return df

    def _generate_entry_signals(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0.0, index=df.index)
        if len(df) < self.sma_trend + self.rsi_period + 5:
            return sig

        rsi = df["rsi"]
        prev_rsi = rsi.shift(1)
        price = df["close"]
        sma = df["sma_trend"]
        vol_ok = df["volume"] > df["sma_vol"] * self.volume_avg_multiplier

        long_entry = (
            (prev_rsi <= self.rsi_oversold) & (rsi > self.rsi_oversold)
            & (price > sma)
            & vol_ok
        )
        short_entry = (
            (prev_rsi >= self.rsi_overbought) & (rsi < self.rsi_overbought)
            & (price < sma)
            & vol_ok
        )

        sig[long_entry] = 1.0
        sig[short_entry] = -1.0
        return sig

    def compute_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        signals = {}
        for symbol, df in data.items():
            ind = self._compute_indicators(df)
            sig = self._generate_entry_signals(ind)
            signals[symbol] = sig
        return signals

    def backtest(self, data: dict[str, pd.DataFrame]) -> dict:
        indicators = {
            sym: self._compute_indicators(df) for sym, df in data.items()
        }
        entry_sigs = {
            sym: self._generate_entry_signals(ind)
            for sym, ind in indicators.items()
        }
        all_dates = sorted(set().union(
            *(set(ind.index) for ind in indicators.values())
        ))

        capital = self.initial_capital
        positions: dict[str, dict] = {}
        pending_exits: dict[str, str] = {}
        pending_entries: list[tuple[str, float]] = []
        trades: list[Trade] = []
        equity_curve: list[float] = [capital]
        equity_index: list = [all_dates[0] - pd.Timedelta(days=1)]

        for i, date in enumerate(all_dates):
            if i == 0:
                total_equity = capital
                for sym, pos in positions.items():
                    ind = indicators.get(sym)
                    if ind is None:
                        continue
                    series_close = ind.loc[ind.index <= date, "close"]
                    if not series_close.empty:
                        total_equity += pos["shares"] * series_close.iloc[-1]
                equity_curve.append(total_equity)
                equity_index.append(date)
                continue

            prev_date = all_dates[i - 1]

            # === Execute pending exits (triggered by prev_date's close, fill at date's open) ===
            for sym in list(pending_exits.keys()):
                pos = positions.pop(sym, None)
                if pos is None:
                    continue
                ind = indicators.get(sym)
                if ind is None or date not in ind.index:
                    continue
                entry_price = pos["entry_price"]
                direction = pos["direction"]
                shares = pos["shares"]
                fill_price = ind.loc[date, "open"]
                if direction == 1:
                    slip = fill_price * self.slippage_pct
                    sell_price = fill_price - slip
                else:
                    slip = fill_price * self.slippage_pct
                    sell_price = fill_price + slip
                comm = pos["value"] * self.commission_pct
                pnl = direction * shares * (sell_price - entry_price) - comm
                capital += shares * sell_price - comm
                trades.append(Trade(
                    date=date, symbol=sym, side="sell",
                    price=sell_price, shares=shares,
                    value=shares * sell_price,
                    pnl=pnl, slippage=slip * shares, commission=comm,
                ))
            pending_exits.clear()

            # === Execute pending entries (triggered by prev_date's signal, fill at date's open) ===
            for sym, direction in pending_entries:
                if sym in positions:
                    continue
                if len(positions) >= self.max_open_positions:
                    break
                ind = indicators.get(sym)
                if ind is None or date not in ind.index:
                    continue
                row = ind.loc[date]
                price = row["open"]
                atr = row["atr"]
                if np.isnan(atr) or atr <= 0 or price <= 0:
                    continue
                allocation = capital * self.max_position_pct
                comm = allocation * self.commission_pct
                slip = price * self.slippage_pct
                exec_price = price + slip if direction == 1 else price - slip
                shares = (allocation - comm) / exec_price
                pos_value = allocation
                positions[sym] = {
                    "entry_price": exec_price,
                    "entry_date": date,
                    "entry_atr": atr,
                    "direction": direction,
                    "shares": shares,
                    "value": pos_value,
                }
                capital -= pos_value
                trades.append(Trade(
                    date=date, symbol=sym, side="buy",
                    price=exec_price, shares=shares,
                    value=pos_value,
                    slippage=slip * shares, commission=comm,
                ))
            pending_entries.clear()

            # === Detect new exit conditions using prev_date's close ===
            for sym, pos in list(positions.items()):
                ind = indicators.get(sym)
                if ind is None or prev_date not in ind.index:
                    continue
                row = ind.loc[prev_date]
                entry_price = pos["entry_price"]
                direction = pos["direction"]
                holding_days = (date - pos["entry_date"]).days
                atr = pos["entry_atr"]

                reason = None
                if direction == 1:
                    target = entry_price + self.atr_target_mult * atr
                    stop = entry_price - self.atr_stop_mult * atr
                    if row["close"] >= target:
                        reason = "take_profit"
                    elif row["close"] <= stop:
                        reason = "stop_loss"
                    elif holding_days >= self.max_holding_days:
                        reason = "max_holding"
                    elif not pd.isna(row.get("rsi")) and row["rsi"] > self.rsi_overbought:
                        reason = "rsi_reversal"
                else:
                    target = entry_price - self.atr_target_mult * atr
                    stop = entry_price + self.atr_stop_mult * atr
                    if row["close"] <= target:
                        reason = "take_profit"
                    elif row["close"] >= stop:
                        reason = "stop_loss"
                    elif holding_days >= self.max_holding_days:
                        reason = "max_holding"
                    elif not pd.isna(row.get("rsi")) and row["rsi"] < self.rsi_oversold:
                        reason = "rsi_reversal"

                if reason is not None and holding_days >= self.min_holding_days:
                    pending_exits[sym] = reason

            # === Detect new entry signals using prev_date's close ===
            for sym, sig_series in entry_sigs.items():
                if sym in positions:
                    continue
                if prev_date not in sig_series.index:
                    continue
                sig = sig_series.loc[prev_date]
                if sig == 0:
                    continue
                if len(positions) + len(pending_entries) >= self.max_open_positions:
                    break
                pending_entries.append((sym, sig))

            total_equity = capital
            for sym, pos in positions.items():
                ind = indicators.get(sym)
                if ind is None:
                    continue
                series_close = ind.loc[ind.index <= date, "close"]
                if not series_close.empty:
                    total_equity += pos["shares"] * series_close.iloc[-1]
            equity_curve.append(total_equity)
            equity_index.append(date)

        equity = pd.Series(equity_curve, index=equity_index)
        rets = equity.pct_change().dropna()
        total_ret = equity.iloc[-1] / equity.iloc[0] - 1
        n_years = max(len(rets) / 252, 0.1)
        ann_ret = (1 + total_ret) ** (1 / n_years) - 1
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        dd = equity / equity.cummax() - 1
        max_dd = float(dd.min())

        sell_trades = [t for t in trades if t.side == "sell"]
        wins = sum(1 for t in sell_trades if t.pnl > 0)
        win_rate = wins / max(len(sell_trades), 1)

        return {
            "total_return": total_ret,
            "annual_return": ann_ret,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "num_trades": len(sell_trades),
            "equity_curve": equity,
            "trade_log": trades,
        }

"""
Opening Range Breakout (ORB) strategy for daily bars.

Key features:
  - Range defined over N consecutive trading days
  - Breakout confirmed when price closes beyond the range high/low
  - ATR-width filter: range must be wider than atr_width_min × daily ATR
    (rejects consolidation days where stops get crushed)
  - ATR-based stops and targets (stop = entry ± atr_mult × ATR;
    target = entry ± (atr_mult × ATR) × risk_mult)

Usage:
    from src.strategies.orb import ORBStrategy, compute_atr
    strategy = ORBStrategy(window_bars=4, atr_mult=2.0, risk_mult=2.0,
                           range_width_pct=0.0, atr_width_min=1.5)
    signals = strategy.generate(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Standard 20-period ATR."""
    tr1 = df["high"] - df["low"]
    tr2 = np.abs(df["high"] - df["close"].shift())
    tr3 = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


class ORBStrategy:
    def __init__(
        self,
        window_bars: int = 4,
        atr_mult: float = 2.0,
        risk_mult: float = 2.0,
        range_width_pct: float = 0.0,
        atr_width_min: float = 0.0,
    ):
        """
        Parameters
        ----------
        window_bars     : number of prior bars defining the opening range
        atr_mult        : stop distance = atr_mult × ATR
        risk_mult       : target distance = risk_mult × (atr_mult × ATR)
                         (i.e. target = entry ± risk_mult × stop_distance)
        range_width_pct : min range width as % of price (0 = disabled)
        atr_width_min   : min range width as multiple of daily ATR (0 = disabled)
                          e.g. 1.5 means range_abs must exceed 1.5 × ATR
        """
        self.window_bars = window_bars
        self.atr_mult = atr_mult
        self.risk_mult = risk_mult
        self.range_width_pct = range_width_pct
        self.atr_width_min = atr_width_min

    def __repr__(self):
        return (
            f"ORB(wb={self.window_bars}, atr×{self.atr_mult}, "
            f"R:R={self.risk_mult}, rng%={self.range_width_pct}, "
            f"atr_w={self.atr_width_min})"
        )

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate daily ORB signals for a single ticker.

        Adds columns to df (does not mutate original):
          signal    : 1 (long breakout), -1 (short breakout), NaN (no signal)
          entry_px  : entry price
          stop_px   : stop-loss price
          target_px : profit target price

        Logic
        -----
        For each bar i >= window_bars:
          1. Look back window_bars bars to compute range high / low
          2. Reject if range_pct < range_width_pct
          3. Reject if range_abs < atr_width_min × ATR (ATR-width filter)
          4. Breakout = close beyond range high/low on the signal bar
          5. Entry = close of signal bar; stop = entry ± atr_mult × ATR;
             target = entry ± risk_mult × atr_mult × ATR
        """
        df = df.copy()
        atr = compute_atr(df)

        for col in ["signal", "entry_px", "stop_px", "target_px"]:
            df[col] = np.nan

        dates = df.index.normalize().unique()

        for i, day_ts in enumerate(dates):
            if i < self.window_bars:
                continue

            day_df = df[df.index.normalize() == day_ts]
            if len(day_df) < 1:
                continue

            # Build range from the prior `window_bars` trading days
            window_dates = [dates[i - j] for j in range(self.window_bars, 0, -1)]
            range_df = df[df.index.normalize().isin(window_dates)]
            if len(range_df) < self.window_bars:
                continue

            rh = range_df["high"].max()
            rl = range_df["low"].min()
            rng_abs = rh - rl
            rng_pct = rng_abs / rl

            # Filter: range width as % of price
            if self.range_width_pct > 0 and rng_pct < self.range_width_pct:
                continue

            # Filter: range width must exceed N × ATR
            if self.atr_width_min > 0:
                a0 = atr.iloc[i - 1] if i > 0 else np.nan
                if pd.isna(a0) or a0 == 0:
                    continue
                if rng_abs < self.atr_width_min * a0:
                    continue

            bl = rh * 1.0001   # breakout level (long)
            bs = rl * 0.9999   # breakout level (short)

            # Signal bar: first bar of the current day
            first_bar = day_df.iloc[0]
            p = first_bar["close"]
            a = atr.iloc[i] if i < len(atr) else np.nan
            if pd.isna(a) or a == 0:
                continue

            ci = df.columns.get_loc
            if p >= bl:
                stop    = p - self.atr_mult * a
                target  = p + (self.atr_mult * a) * self.risk_mult
                df.iloc[i, ci("signal")]    = 1
                df.iloc[i, ci("entry_px")]  = p
                df.iloc[i, ci("stop_px")]   = stop
                df.iloc[i, ci("target_px")] = target
            elif p <= bs:
                stop    = p + self.atr_mult * a
                target  = p - (self.atr_mult * a) * self.risk_mult
                df.iloc[i, ci("signal")]    = -1
                df.iloc[i, ci("entry_px")]  = p
                df.iloc[i, ci("stop_px")]   = stop
                df.iloc[i, ci("target_px")] = target

        return df


def backtest(
    data: dict[str, pd.DataFrame],
    signals: dict[str, pd.DataFrame],
    capital: float = 100_000,
    max_pos: int = 3,
    slippage: float = 0.0003,
) -> dict:
    """
    Simple daily-bar portfolio backtest.

    One active position at a time across all tickers.
    Tracks equity per trade; closes open positions at last bar close.

    Returns dict with:
      return, equity, n_trades, win_rate, avg_win, avg_loss, r_r,
      trades (list of dicts), per_ticker (dict)
    """
    equity  = capital
    pos     = 0
    entry   = 0.0
    stop    = 0.0
    target  = 0.0
    shares  = 0
    side    = 0
    trades  = []

    for sym, df in data.items():
        sig = signals.get(sym)
        if sig is None:
            continue
        for ts, row in df.iterrows():
            if ts not in sig.index:
                continue
            p = row["close"]
            s = sig.at[ts, "signal"]
            if pos == 0 and not pd.isna(s) and s != 0:
                side   = int(s)
                entry  = p * (1 + slippage if side == 1 else 1 - slippage)
                stop   = sig.at[ts, "stop_px"]
                target = sig.at[ts, "target_px"]
                shares = int((equity / max_pos) / entry)
                pos    = side
            elif pos != 0:
                exit_long  = pos == 1  and (p <= stop or p >= target)
                exit_short = pos == -1 and (p >= stop or p <= target)
                if exit_long or exit_short:
                    ep   = p * (1 - slippage if pos == 1 else 1 + slippage)
                    pnl  = (ep - entry) * shares if pos == 1 else (entry - ep) * shares
                    equity += pnl
                    trades.append({
                        "symbol": sym, "side": pos,
                        "entry": entry, "exit": ep,
                        "pnl": pnl, "close": p,
                    })
                    pos = 0; shares = 0; side = 0

        # Close open positions at last bar close
        if pos != 0:
            last = df["close"].iloc[-1]
            ep   = last * (1 - slippage if pos == 1 else 1 + slippage)
            pnl  = (ep - entry) * shares if pos == 1 else (entry - ep) * shares
            equity += pnl
            trades.append({
                "symbol": sym, "side": pos,
                "entry": entry, "exit": ep,
                "pnl": pnl, "close": last,
            })
            pos = 0; shares = 0; side = 0

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    avg_win  = np.mean([t["pnl"] for t in wins])   if wins   else 0.0
    avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 0.0
    r_r      = avg_win / avg_loss if avg_loss > 0 else 0.0

    per_ticker = {}
    for sym in signals:
        sym_trades = [t for t in trades if t["symbol"] == sym]
        if sym_trades:
            w = [t for t in sym_trades if t["pnl"] > 0]
            l = [t for t in sym_trades if t["pnl"] <= 0]
            per_ticker[sym] = {
                "return":   sum(t["pnl"] for t in sym_trades) / capital,
                "n":        len(sym_trades),
                "win_rate": len(w) / max(len(sym_trades), 1),
                "avg_win":  np.mean([t["pnl"] for t in w]) if w else 0.0,
                "avg_loss": abs(np.mean([t["pnl"] for t in l])) if l else 0.0,
            }

    return {
        "return":     (equity - capital) / capital,
        "equity":     equity,
        "n_trades":   len(trades),
        "win_rate":   len(wins) / max(len(trades), 1),
        "avg_win":    avg_win,
        "avg_loss":   avg_loss,
        "r_r":        r_r,
        "trades":     trades,
        "per_ticker": per_ticker,
    }


def permutation_test(
    data: dict[str, pd.DataFrame],
    signals: dict[str, pd.DataFrame],
    n_perms: int = 200,
    seed: int = 99,
) -> dict:
    """
    Randomise signal sides (preserve entries, stops, targets) to build
    a null distribution of returns under side-neutral ORB.
    Returns dict with perm_returns array, p_value, z_score.
    """
    rng = np.random.default_rng(seed)
    perm_returns = []

    for _ in range(n_perms):
        ps = {}
        for sym in signals:
            sig = signals[sym].copy()
            mask = sig["signal"].notna()
            if mask.sum() == 0:
                continue
            sig.loc[mask, "signal"] = rng.choice([-1, 1], size=int(mask.sum()))
            ps[sym] = sig
        bt = backtest(data, ps)
        perm_returns.append(bt["return"])

    perm_arr = np.array(perm_returns)
    real_ret = backtest(data, signals)["return"]
    p_value  = (perm_arr >= real_ret).mean()
    z_score  = (real_ret - perm_arr.mean()) / max(perm_arr.std(), 1e-9)

    return {
        "perm_returns": perm_returns,
        "real_return": real_ret,
        "null_mean":   perm_arr.mean(),
        "null_std":    perm_arr.std(),
        "p_value":    p_value,
        "z_score":    z_score,
    }

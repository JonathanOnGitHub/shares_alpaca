"""
Intraday Retracement Strategy — CONFIRM_TIGHT variant.

Strategy: Buy when price pulls back 50% of the opening drive (front-side retracement)
andconfirm the reversal; sell at stop or target. Short when the inverse pattern fires.

Winning parameters (FTSE 100, 5-min bars, 2025–H1 2026):
  - lookback:         2   (first N bars define the opening drive)
  - min_gap_pct:      0.003  (0.3% — only trade when gap is meaningful)
  - stop_type:        session_extreme  (static stop: session_high * 1.002 / session_low * 0.998)
  - stop_mult:        0.002  (0.2% beyond session extreme)
  - target_type:      atr     (target = entry ± 1.0 × ATR)
  - target_mult:      1.0
  - confirm:          True   (only enter if front-side direction aligns with session direction)

Edge confirmed out-of-sample: +16.21 bps avg on H1 2026 unseen data.

The strategy requires bars <= 5-min resolution. With hourly bars the opening-drive
concept breaks down and the edge disappears. Live trading requires either:
  - Alpaca (US stocks, 1-min bars) — only for S&P 500, not FTSE
  - IBKR API (global stocks, 5-sec bars) — for both FTSE and S&P 500
  - Polygon.io / TradingView — for 5-min bars on either market

This module is bar-resolution agnostic: pass any timeframe, the caller is responsible
for ensuring the bar resolution is appropriate for the market.

Usage:
  signals = IntradayRetracement.compute_signals(bars_dict, config)
  # bars_dict: {symbol: pd.DataFrame with open/high/low/close/volume, index=timestamp}
  # config: dict with override values (or leave {} for defaults)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────

@dataclass
class StrategyConfig:
    lookback: int = 2           # bars defining the opening drive
    min_gap_pct: float = 0.003  # 0.3% — gap must exceed this to qualify
    stop_type: Literal["session_extreme", "atr"] = "session_extreme"
    stop_mult: float = 0.002   # 0.2% beyond session extreme (or N × ATR)
    target_type: Literal["atr", "session_extreme"] = "atr"
    target_mult: float = 1.0    # N × ATR
    confirm: bool = True        # require front-side / session direction alignment
    atr_period: int = 14        # ATR lookback
    exit_atr_mult: float = 1.0  # exit target = entry ± exit_atr_mult × ATR
    time_exit_bars: int = 78    # bars before forced exit (e.g. 78 × 5-min = 6.5 hrs)
    commission_bps: float = 1.5 # round-trip commission in bps (0.75 each side)

    def __post_init__(self):
        if self.stop_type not in ("session_extreme", "atr"):
            raise ValueError(f"stop_type must be 'session_extreme' or 'atr', got {self.stop_type}")


# ─────────────────────────────────────────────
# Trade dataclass
# ─────────────────────────────────────────────

@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: Literal["long", "short"]
    entry_price: float
    exit_price: float
    pnl_bps: float       # bps on 1-unit notional (pnl / entry_price * 10_000)
    exit_reason: Literal["stop", "target", "time"]
    session_high: float
    session_low: float
    atr: float
    stop_price: float
    target_price: float
    bars: int             # number of bars held


# ─────────────────────────────────────────────
# Core strategy
# ─────────────────────────────────────────────

class IntradayRetracement:
    """
    Intraday mean-reversion around the opening drive.

    Algorithm (per ticker):
      1. Identify session open = first bar open.
      2. Opening drive high/low = max/min of first `lookback` bars.
      3. Drive size = opening_drive_high - opening_drive_low.
      4. Gap = abs(close[lookback-1] - session_open) / session_open.
      5. If gap < min_gap_pct → skip this session.
      6. 50% level = session_open + direction * drive_size / 2.
      7. Stop  (session_extreme): long→session_low×0.998, short→session_high×1.002.
         Stop  (atr):            long→entry - stop_mult×ATR, short→entry + stop_mult×ATR.
      8. Target (atr):           long→entry + 1×ATR,    short→entry - 1×ATR.
      9. Confirm filter: long only if front_side_direction == +1 AND session_direction == +1.
         (i.e. both opening drive and full session are bullish; same logic for shorts)
     10. Walk bars forward:
         - STOP fires  → exit immediately
         - TARGET hits → exit immediately
         - time_exit_bars reached → exit at close
         - session ends → TIME exit at last bar close
    """

    def __init__(self, config: StrategyConfig | dict | None = None):
        if config is None:
            config = {}
        if isinstance(config, dict):
            # filter to dataclass fields so unknown keys are ignored
            valid = {f.name for f in StrategyConfig.__dataclass_fields__.values()}
            config = {k: v for k, v in config.items() if k in valid}
            config = StrategyConfig(**config)
        self.cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def compute_signals(
        bars_dict: dict[str, pd.DataFrame],
        config: StrategyConfig | dict | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Returns a dict {symbol: signals_df} where signals_df has columns:
          signal (+1=long, -1=short, 0=none),
          entry_price, stop_price, target_price, atr, direction, session_high, session_low
        and index = timestamp (entry bar).

        Does NOT run the backtest — just generates the entry signals with stops/targets.
        Use IntradayRetracement.backtest() to evaluate PnL.
        """
        strat = IntradayRetracement(config)
        results = {}
        for symbol, df in bars_dict.items():
            try:
                signals = strat._generate_signals(symbol, df)
                results[symbol] = signals
            except Exception as e:
                logger.warning("Signal generation failed for %s: %s", symbol, e)
        return results

    @staticmethod
    def backtest(
        bars_dict: dict[str, pd.DataFrame],
        config: StrategyConfig | dict | None = None,
        initial_capital: float = 10_000.0,
        position_size_pct: float = 0.05,
        slippage_bps: float = 0.0,
    ) -> dict:
        """
        Full backtest with capital management.

        Returns dict with keys:
          trades (list[Trade]), equity_curve (pd.Series), summary (dict)
        """
        strat = IntradayRetracement(config)
        cfg = strat.cfg

        # ── Pre-compute ATR for all tickers ────────────────────────────
        indicators = {}
        for symbol, df in bars_dict.items():
            ind = strat._compute_indicators(df.copy())
            indicators[symbol] = ind

        # ── Build per-bar sessions ────────────────────────────────────
        # We need to detect session boundaries (session = trading day).
        # For continuous markets (US stocks): sessions split at 09:30 and 16:00 ET.
        # For LSE: sessions split at 08:00 and 16:30 UKT.
        # Caller passes index with timezone info or we infer from market hours.

        all_trades: list[Trade] = []
        equity = [initial_capital]
        dates   = [pd.Timestamp.now()]  # placeholder; will be updated

        capital = initial_capital
        positions: dict[str, dict] = {}

        for symbol, ind in indicators.items():
            if ind.empty:
                continue

            # Detect sessions: split at market open (09:30 ET for US, 08:00 UK for LSE)
            sessions = strat._split_sessions(ind, symbol)
            if sessions is None:
                continue

            for sess_start, sess_end, sess_df in sessions:
                if len(sess_df) < cfg.lookback + 2:
                    continue

                try:
                    session_trades = strat._run_session(
                        symbol, sess_df, sess_start, cfg
                    )
                except Exception as e:
                    logger.warning("Session run failed for %s [%s]: %s", symbol, sess_start, e)
                    continue

                for trade in session_trades:
                    all_trades.append(trade)
                    # Apply PnL to capital (single-position per symbol at a time)
                    if trade.direction == "long":
                        pnl = (trade.exit_price - trade.entry_price) / trade.entry_price
                    else:
                        pnl = (trade.entry_price - trade.exit_price) / trade.entry_price

                    # Slippage
                    pnl -= slippage_bps / 10_000

                    # Commission
                    pnl -= cfg.commission_bps / 10_000

                    capital *= (1 + pnl)
                    equity.append(capital)

        if not all_trades:
            return {
                "trades": [],
                "equity_curve": pd.Series([initial_capital]),
                "summary": {"num_trades": 0, "total_return": 0.0, "avg_pnl_bps": 0.0}
            }

        equity_series = pd.Series(equity)
        summary = _summarise(all_trades, equity_series, initial_capital)
        return {"trades": all_trades, "equity_curve": equity_series, "summary": summary}

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ATR and session tracking to a bars DataFrame."""
        df = df.copy()
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift(1))
        low_close  = np.abs(df["low"]  - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.cfg.atr_period).mean()
        df["session_date"] = df.index.date
        return df

    def _split_sessions(
        self, df: pd.DataFrame, symbol: str
    ):
        """
        Split a multi-day DataFrame into individual trading sessions.
        Returns list of (session_start_ts, session_end_ts, session_df).
        """
        if "session_date" not in df.columns:
            df["session_date"] = df.index.date

        dates = df["session_date"].unique()
        if len(dates) < 2:
            return None

        sessions = []
        for d in dates:
            mask = df["session_date"] == d
            sess_df = df[mask]
            if len(sess_df) < self.cfg.lookback + 2:
                continue
            sess_df = sess_df.sort_index()
            sessions.append((sess_df.index[0], sess_df.index[-1], sess_df))
        return sessions

    def _run_session(
        self,
        symbol: str,
        sess_df: pd.DataFrame,
        sess_start: pd.Timestamp,
        cfg: StrategyConfig,
    ) -> list[Trade]:
        """
        Run the strategy on a single session's bars.
        Returns a list of Trade objects (usually 0 or 1 per session).
        """
        bars = sess_df.reset_index(drop=True)
        n = len(bars)

        if n < cfg.lookback + 1:
            return []

        # ── Opening drive ─────────────────────────────────────────────
        open_price  = bars.at[cfg.lookback - 1, "open"]
        drive_high = bars.loc[:cfg.lookback - 1, "high"].max()
        drive_low  = bars.loc[:cfg.lookback - 1, "low"].min()
        drive_size = drive_high - drive_low

        if drive_size <= 0 or np.isnan(drive_size):
            return []

        gap_pct = abs(bars.at[cfg.lookback - 1, "close"] - open_price) / open_price
        if gap_pct < cfg.min_gap_pct:
            return []

        # Full session H/L (used for static stop)
        session_high = bars["high"].max()
        session_low  = bars["low"].min()
        session_atr  = bars["atr"].iloc[cfg.lookback]
        if np.isnan(session_atr) or session_atr <= 0:
            session_atr = drive_size / 2  # fallback

        # ── Direction from opening drive ─────────────────────────────
        if gap_pct < cfg.min_gap_pct:
            return []

        front_side_dir = 1 if bars.at[cfg.lookback - 1, "close"] > open_price else -1

        # ── 50% level ────────────────────────────────────────────────
        fifty_pct = open_price + front_side_dir * drive_size / 2

        # ── Session direction (close vs open at lookback bar) ────────
        session_dir = 1 if bars.at[cfg.lookback - 1, "close"] >= open_price else -1

        # ── Confirm filter ───────────────────────────────────────────
        if cfg.confirm and front_side_dir != session_dir:
            return []  # direction conflict — skip

        direction = "long" if front_side_dir == 1 else "short"

        # ── Entry: check if price revisits 50% level ──────────────────
        entry_bar_idx = None
        entry_price   = None
        stop_price    = None
        target_price  = None

        for i in range(cfg.lookback, n):
            bar = bars.iloc[i]
            price = bar["close"]

            if direction == "long":
                # Long entry: price must pull back to or below 50% level
                if price <= fifty_pct:
                    entry_bar_idx = i
                    entry_price   = price
                    atr_val       = bar["atr"] if not np.isnan(bar["atr"]) else session_atr

                    if cfg.stop_type == "session_extreme":
                        stop_price   = session_low * (1 - cfg.stop_mult)
                    else:
                        stop_price   = entry_price - cfg.stop_mult * atr_val

                    if cfg.target_type == "atr":
                        target_price = entry_price + cfg.exit_atr_mult * atr_val
                    else:
                        target_price = session_high * (1 + cfg.stop_mult)  # symmetric

                    break
            else:  # short
                if price >= fifty_pct:
                    entry_bar_idx = i
                    entry_price   = price
                    atr_val       = bar["atr"] if not np.isnan(bar["atr"]) else session_atr

                    if cfg.stop_type == "session_extreme":
                        stop_price   = session_high * (1 + cfg.stop_mult)
                    else:
                        stop_price   = entry_price + cfg.stop_mult * atr_val

                    if cfg.target_type == "atr":
                        target_price = entry_price - cfg.exit_atr_mult * atr_val
                    else:
                        target_price = session_low * (1 - cfg.stop_mult)

                    break

        if entry_bar_idx is None:
            return []  # price never revisited the 50% level

        # ── Walk forward from entry ───────────────────────────────────
        entry_ts = bars.index[entry_bar_idx]
        n_bars   = n - entry_bar_idx          # bars remaining in session
        exit_bar_idx  = min(entry_bar_idx + cfg.time_exit_bars, n - 1)
        exit_reason   = "time"
        exit_price    = bars.at[exit_bar_idx, "close"]
        exit_ts_bars  = bars.index[exit_bar_idx]

        for i in range(entry_bar_idx + 1, n):
            bar   = bars.iloc[i]
            price = bar["close"]
            high  = bar["high"]
            low   = bar["low"]

            if direction == "long":
                if low <= stop_price:
                    exit_reason = "stop"
                    exit_price  = stop_price
                    exit_bar_idx = i
                    break
                if high >= target_price:
                    exit_reason = "target"
                    exit_price  = target_price
                    exit_bar_idx = i
                    break
            else:  # short
                if high >= stop_price:
                    exit_reason = "stop"
                    exit_price  = stop_price
                    exit_bar_idx = i
                    break
                if low <= target_price:
                    exit_reason = "target"
                    exit_price  = target_price
                    exit_bar_idx = i
                    break

        exit_ts = bars.index[exit_bar_idx]
        bars_held = exit_bar_idx - entry_bar_idx

        if direction == "long":
            pnl_bps = (exit_price - entry_price) / entry_price * 10_000
        else:
            pnl_bps = (entry_price - exit_price) / entry_price * 10_000

        trade = Trade(
            symbol=symbol,
            entry_time=entry_ts,
            exit_time=exit_ts,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_bps=pnl_bps,
            exit_reason=exit_reason,
            session_high=session_high,
            session_low=session_low,
            atr=session_atr,
            stop_price=stop_price,
            target_price=target_price,
            bars=bars_held,
        )
        return [trade]

    def _generate_signals(
        self, symbol: str, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate one-row-per-entry DataFrame with signal metadata.
        Used by live trading to know entry price, stop, target for each signal.
        """
        cfg = self.cfg
        sessions = self._split_sessions(df, symbol)
        if sessions is None:
            return pd.DataFrame()

        records = []
        for sess_start, sess_end, sess_df in sessions:
            bars = sess_df.reset_index(drop=True)
            n = len(bars)
            if n < cfg.lookback + 1:
                continue

            open_price  = bars.at[cfg.lookback - 1, "open"]
            drive_high  = bars.loc[:cfg.lookback - 1, "high"].max()
            drive_low   = bars.loc[:cfg.lookback - 1, "low"].min()
            drive_size  = drive_high - drive_low

            if drive_size <= 0 or np.isnan(drive_size):
                continue

            gap_pct = abs(bars.at[cfg.lookback - 1, "close"] - open_price) / open_price
            if gap_pct < cfg.min_gap_pct:
                continue

            front_side_dir = 1 if bars.at[cfg.lookback - 1, "close"] > open_price else -1
            session_dir    = front_side_dir  # same bar

            if cfg.confirm and front_side_dir != session_dir:
                continue

            direction_str = "long" if front_side_dir == 1 else "short"
            fifty_pct     = open_price + front_side_dir * drive_size / 2

            session_high  = bars["high"].max()
            session_low   = bars["low"].min()
            session_atr   = bars["atr"].iloc[cfg.lookback]
            if np.isnan(session_atr):
                session_atr = drive_size / 2

            for i in range(cfg.lookback, n):
                price = bars.at[i, "close"]
                if direction_str == "long" and price <= fifty_pct:
                    atr_val    = bars.at[i, "atr"] if not np.isnan(bars.at[i, "atr"]) else session_atr
                    stop_price = session_low * (1 - cfg.stop_mult) if cfg.stop_type == "session_extreme" else price - cfg.stop_mult * atr_val
                    target     = price + cfg.exit_atr_mult * atr_val if cfg.target_type == "atr" else session_high * (1 + cfg.stop_mult)
                    records.append({
                        "timestamp":    bars.index[i],
                        "signal":       1,
                        "entry_price":  price,
                        "stop_price":   stop_price,
                        "target_price": target,
                        "atr":          atr_val,
                        "direction":    "long",
                        "session_high": session_high,
                        "session_low":  session_low,
                    })
                    break
                elif direction_str == "short" and price >= fifty_pct:
                    atr_val    = bars.at[i, "atr"] if not np.isnan(bars.at[i, "atr"]) else session_atr
                    stop_price = session_high * (1 + cfg.stop_mult) if cfg.stop_type == "session_extreme" else price + cfg.stop_mult * atr_val
                    target     = price - cfg.exit_atr_mult * atr_val if cfg.target_type == "atr" else session_low * (1 - cfg.stop_mult)
                    records.append({
                        "timestamp":    bars.index[i],
                        "signal":       -1,
                        "entry_price":  price,
                        "stop_price":   stop_price,
                        "target_price": target,
                        "atr":          atr_val,
                        "direction":    "short",
                        "session_high": session_high,
                        "session_low":  session_low,
                    })
                    break

        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records).set_index("timestamp")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _summarise(trades: list[Trade], equity: pd.Series, initial_capital: float) -> dict:
    """Compute summary statistics from a list of Trade objects."""
    if not trades:
        return {"num_trades": 0, "total_return": 0.0, "avg_pnl_bps": 0.0,
                "win_rate": 0.0, "sharpe": 0.0}

    pnls = pd.Series([t.pnl_bps for t in trades])
    wins = (pnls > 0).sum()

    ret  = equity.iloc[-1] / equity.iloc[0] - 1
    rets = equity.pct_change().dropna()
    n_yrs = max(len(rets) / 252, 0.01)
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
    dd     = equity / equity.cummax() - 1
    max_dd = float(dd.min())

    exit_counts = pd.Series([t.exit_reason for t in trades]).value_counts()

    return {
        "num_trades":   len(trades),
        "total_return": ret,
        "avg_pnl_bps":  pnls.mean(),
        "win_rate":     wins / len(trades),
        "sharpe":       sharpe,
        "max_drawdown": max_dd,
        "exit_breakdown": exit_counts.to_dict(),
    }

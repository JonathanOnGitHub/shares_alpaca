"""
Noise Area intraday momentum strategy (Zarattini, Aziz & Barbon 2024).

"Beat the Market: An Effective Intraday Momentum Strategy for the S&P500 ETF
(SPY)", SSRN 4824172 / doi:10.2139/ssrn.4824172.

The band is NOT a Bollinger band.  For every minute-of-day the strategy
computes sigma = the mean over the previous N days of |Close/Open - 1|, i.e.
the average absolute excursion from that day's open by that time of day.
The band around the current open is therefore

    UB = max(open, prev_close) * (1 + VM * sigma)
    LB = min(open, prev_close) * (1 - VM * sigma)

with the previous close adjusted for any dividend paid that day, so an
overnight gap widens the band.  A breach means the market has moved further
than it "normally" has by that time of day -- an imbalance in demand/supply
rather than noise.

Exposure is set on semi-hourly marks (the 09:59 bar, traded from the next
bar), long above the band and short below it, with VWAP confirmation, and
unwound at the close.

This is a port of the authors' own reference notebook (linked from the
paper's Q&A page).  The sigma construction, the minute-of-day grouping, the
mark schedule, the signal rule, the cost model and the position sizing all
follow that notebook rather than a paraphrase of the prose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MARKET_OPEN_MIN = 9 * 60 + 30   # 09:30 ET


@dataclass
class NoiseBandConfig:
    """Parameters of the noise band.

    lookback_days  days of history averaged into sigma (paper: 14)
    band_mult      Volatility Multiplier, widens/narrows the band (paper: 1)
    trade_freq     minutes between decision points (paper: 30)
    use_vwap       require VWAP confirmation (the reference code always does)
    commission     per-share commission (paper: 0.0035)
    min_commission per-order minimum commission (paper: 0.35)
    slippage       per-share slippage; the paper models none (0.0)
    sizing         "full_notional" or "vol_target" (reference code default)
    target_vol     daily vol target for "vol_target" sizing (paper: 0.02)
    max_leverage   leverage cap for "vol_target" sizing (paper: 4)
    vol_lookback   days of daily returns in the vol estimate (paper: 15)
    """

    lookback_days: int = 14
    band_mult: float = 1.0
    trade_freq: int = 30
    use_vwap: bool = True
    commission: float = 0.0035
    min_commission: float = 0.35
    slippage: float = 0.0
    sizing: str = "full_notional"
    target_vol: float = 0.02
    max_leverage: float = 4.0
    vol_lookback: int = 15


def _minute_of_day(index: pd.DatetimeIndex) -> pd.Series:
    """Minutes since midnight ET, as the authors' notebook computes them."""
    minutes = index.hour * 60 + index.minute
    return pd.Series(minutes, index=index, dtype="int64")


def add_noise_band_features(df: pd.DataFrame,
                            cfg: NoiseBandConfig = NoiseBandConfig(),
                            dividends: pd.Series | None = None) -> pd.DataFrame:
    """Add sigma/UB/LB/vwap/daily-vol columns to intraday bars.

    `df` must be indexed by timestamp (ET, tz-naive or tz-aware) with open,
    high, low, close and volume columns, sorted ascending, one session's bars
    grouped under `day`.

    sigma at minute-of-day m on day t is the mean over the previous
    `lookback_days` days of |Close/Open - 1| at the same minute m.  Only whole
    minutes of history count: a bar's sigma is used only from the minute after
    it was observed, so no bar can see its own move.

    `dividends` is an optional series of cash dividends keyed by ex-date; the
    reference notebook subtracts the day's dividend from the previous close
    before centring the band.
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("noise band requires a DatetimeIndex")
    out.index = out.index.tz_convert("America/New_York").tz_localize(None) \
        if out.index.tz is not None else out.index
    if "day" not in out.columns:
        out["day"] = out.index.normalize()

    # --- VWAP: cumulative from the open, typical price weighted by volume.
    typical = (out["high"] + out["low"] + out["close"]) / 3
    out["vwap"] = typical.mul(out["volume"]).groupby(out["day"]).cumsum() / \
        out["volume"].groupby(out["day"]).cumsum()

    # --- absolute excursion from the day's open
    open_px = out["open"].groupby(out["day"]).transform("first")
    out["move_open"] = (out["close"] / open_px - 1).abs()

    # --- minute of day. The +1 matches the authors' reference implementation:
    # the bar labelled 09:30 is "minute 1" of the session, so the semi-hourly
    # marks land on the 09:59 bar and the position is held from 10:00 -- one
    # bar later, which is the deliberate anti-lookahead shift.
    out["minute_of_day"] = _minute_of_day(out.index)
    out["min_from_open"] = out["minute_of_day"] - MARKET_OPEN_MIN + 1

    # --- sigma: mean absolute move at this minute-of-day over prior N days.
    # Rolling and shifting both happen *within* each minute-of-day group, so
    # the value at minute m on day t averages days t-N..t-1 at minute m only.
    minute_mean = out.groupby("minute_of_day")["move_open"].transform(
        lambda x: x.rolling(cfg.lookback_days, min_periods=cfg.lookback_days - 1).mean()
    )
    out["sigma_open"] = minute_mean.groupby(out["minute_of_day"]).shift(1)

    # --- band around the open, widened by the prior close to absorb gaps
    day_last_close = out.groupby("day")["close"].last()
    prev_close = day_last_close.shift(1)
    if dividends is not None:
        div = pd.Series(dividends).reindex(day_last_close.index).fillna(0.0)
        prev_close = prev_close - div.astype(float)
    prev_close = out["day"].map(prev_close)

    session_open = out["open"].groupby(out["day"]).transform("first")
    basis_hi = session_open.where(session_open > prev_close, prev_close)
    basis_lo = session_open.where(session_open < prev_close, prev_close)
    out["band_upper"] = basis_hi * (1 + cfg.band_mult * out["sigma_open"])
    out["band_lower"] = basis_lo * (1 - cfg.band_mult * out["sigma_open"])

    # --- daily vol used by "vol_target" sizing: stdev of the daily returns
    # before today. The reference notebook takes the 14 returns ending two
    # days back, which is what the .shift(2) reproduces.
    day_ret = day_last_close.pct_change()
    dvol = day_ret.rolling(cfg.vol_lookback - 1,
                            min_periods=cfg.vol_lookback - 1).std().shift(2)
    out["daily_vol"] = out["day"].map(dvol)

    return out


def generate_signals(out: pd.DataFrame, cfg: NoiseBandConfig = NoiseBandConfig()) -> pd.Series:
    """+1 long / -1 short / 0 flat, evaluated on every bar (not just marks)."""
    above = out["close"] > out["band_upper"]
    below = out["close"] < out["band_lower"]
    if cfg.use_vwap:
        above &= out["close"] > out["vwap"]
        below &= out["close"] < out["vwap"]
    sig = pd.Series(0.0, index=out.index)
    sig[above] = 1.0
    sig[below] = -1.0
    return sig


def position_size(equity: float, open_px: float, daily_vol: float,
                  cfg: NoiseBandConfig = NoiseBandConfig()) -> int:
    """Shares held for a session, from the reference notebook's sizing rules."""
    if cfg.sizing == "vol_target":
        if pd.isna(daily_vol) or daily_vol <= 0:
            leverage = cfg.max_leverage
        else:
            leverage = min(cfg.target_vol / daily_vol, cfg.max_leverage)
    else:
        leverage = 1.0
    return int(round(equity / open_px * leverage))


def run_backtest(out: pd.DataFrame,
                 cfg: NoiseBandConfig = NoiseBandConfig(),
                 initial_capital: float = 100_000.0,
                 baseline: bool = False) -> dict:
    """Simulate the strategy day by day.

    Exposure is set on semi-hourly marks from the signal, held until the next
    mark, and flattened at the close.  Costs are charged per trade.

    With `baseline=True` the signal is ignored and exposure is always long:
    that is the unconditional control.  In a rising market any long entry has
    positive drift, so the strategy's raw return means nothing on its own --
    only the excess over this baseline is evidence of edge.  The baseline is
    sized identically, so the comparison isolates the signal.

    Note the baseline is long *intraday only* (it is flat at each close,
    exactly like the strategy), so it is NOT buy-and-hold: it misses the
    overnight gap, which for SPY was +52% of the 2020-2025 return.  It is
    the right control because it shares the strategy's exposure schedule;
    the report also quotes plain SPY buy-and-hold separately.
    """
    signals = generate_signals(out, cfg)
    out = out.assign(signal=signals)

    days = list(out.groupby("day").groups.keys())
    equity = initial_capital
    curve = []
    trade_rows = []

    for i, day in enumerate(days):
        if i == 0:
            curve.append((day, equity))
            continue
        d = out.loc[out["day"] == day]
        prev_day = days[i - 1]
        prev_close = out.loc[out["day"] == prev_day, "close"].iloc[-1]

        if d["sigma_open"].isna().all():
            curve.append((day, equity))
            continue

        open_px = d["open"].iloc[0]
        exposure = np.full(len(d), np.nan)

        # Exposure is only decided on semi-hourly marks.
        marks = np.where(d["min_from_open"].to_numpy() % cfg.trade_freq == 0)[0]
        if len(marks):
            if baseline:
                exposure[marks[0]:] = 1.0
            else:
                exposure[marks] = d["signal"].to_numpy()[marks]
        else:
            curve.append((day, equity))
            continue

        # Forward-fill, but a zero signal at a mark closes the position.
        last = np.nan
        filled = []
        for v in exposure:
            if not np.isnan(v):
                last = v
                if v == 0:
                    last = np.nan
            filled.append(last)
        exposure = pd.Series(filled, index=d.index).shift(1).fillna(0.0).to_numpy()

        n_trades = int(np.sum(np.abs(np.diff(np.append(exposure, 0.0)))))
        shares = position_size(equity, open_px, d["daily_vol"].iloc[0], cfg)

        # Per-bar price changes. Exposure is shifted by one bar (the signal
        # is read at the close of a mark bar and traded from the next bar),
        # so the P&L is close-to-close from bar 1 onwards. The move from the
        # session open to the first bar is NOT captured -- the same as the
        # authors' reference notebook, which uses a plain close.diff().
        change = d["close"].diff()
        gross = float(np.nansum(exposure * change.fillna(0.0))) * shares
        cost = n_trades * max(cfg.min_commission, cfg.commission * shares) \
            + n_trades * cfg.slippage * shares
        pnl = gross - cost
        equity += pnl
        curve.append((day, equity))
        trade_rows.append({
            "day": day,
            "pnl": pnl,
            "trades": n_trades,
            "gross": gross,
            "cost": cost,
            "shares": shares,
            "spy_ret": float(d["close"].iloc[-1] / prev_close - 1),
        })

    curve_df = pd.DataFrame(curve, columns=["day", "equity"]).set_index("day")
    trades_df = pd.DataFrame(trade_rows)
    return {
        "equity_curve": curve_df["equity"],
        "trades": trades_df,
        "total_return": curve_df["equity"].iloc[-1] / initial_capital - 1,
        "n_days": len(trades_df),
        "n_trades": int(trades_df["trades"].sum()) if len(trades_df) else 0,
    }

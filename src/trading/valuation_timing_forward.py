"""Forward paper-trading for valuation timing strategy.
Run daily via cron.

Strategy:
  - Track trailing P/E ratio for each stock using last 4 reported quarterly EPS
  - Long when P/E < 60-day P/E MA (relatively cheap vs recent history)
  - Exit when P/E >= P/E MA (no longer cheap)
  - Rebalance monthly

Usage:
  python -m src.trading.valuation_timing_forward
  python -m src.trading.valuation_timing_forward --universe mega
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src.data.alpaca_client import AlpacaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

POSITIONS_FILE = Path(__file__).resolve().parent / "valtiming_positions.json"

MEGA_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT',
    'JNJ', 'PG', 'XOM', 'BAC', 'DIS', 'HD', 'CVX', 'UNH', 'MA', 'COST',
    'NFLX', 'ADBE', 'CRM', 'AMD', 'CSCO', 'PFE', 'ABBV', 'MRK', 'TMO', 'AVGO',
]

SMALL_SYMBOLS = [
    'AEO', 'ANF', 'BOOT', 'CROX', 'DKS', 'FIVE', 'OLLI', 'CHWY', 'CVNA', 'COIN',
    'HOOD', 'AFRM', 'UPST', 'UAA', 'BBWI', 'ALGM', 'CRSP', 'BEAM', 'FIVN', 'RNG',
    'QTWO', 'TOST', 'MNDY', 'GTLB', 'SMAR', 'WIX', 'PATH', 'CALM', 'EXAS', 'GH',
]

PE_MA_WINDOW = 60
MAX_OPEN_POSITIONS = 8
MAX_POSITION_PCT = 0.10


def _fetch_eps_today(symbol: str, as_of_date: pd.Timestamp) -> float | None:
    try:
        ticker = yf.Ticker(symbol)
        ed = ticker.earnings_dates
        if ed is None or len(ed) < 4:
            return None
        cutoff = as_of_date - pd.Timedelta(days=730)
        if ed.index.tz is not None and cutoff.tz is None:
            cutoff = cutoff.tz_localize(ed.index.tz)
        ed = ed[ed.index >= cutoff]
        eps = ed["Reported EPS"].dropna()
        if len(eps) < 4:
            return None
        ann_dates = eps.index.tolist()
        eps_vals = eps.values
        ann_dates_norm = []
        for ad in ann_dates:
            if ad.tz is not None:
                ad = ad.tz_convert(None)
            ann_dates_norm.append(pd.Timestamp(ad.date()))
        as_of_norm = pd.Timestamp(as_of_date.date())
        usable_eps = [e for ad, e in zip(ann_dates_norm, eps_vals) if ad <= as_of_norm]
        if len(usable_eps) < 4:
            return None
        return sum(usable_eps[-4:])
    except Exception as e:
        logger.warning("Failed to fetch EPS for %s: %s", symbol, e)
        return None


def _load_positions() -> dict:
    if POSITIONS_FILE.exists():
        return json.loads(POSITIONS_FILE.read_text())
    return {}


def _save_positions(positions: dict):
    POSITIONS_FILE.write_text(json.dumps(positions, indent=2))


def get_signals(client: AlpacaClient, symbols: list[str]) -> dict[str, float]:
    today = datetime.now(timezone.utc)
    today_ts = pd.Timestamp(today)

    lookback = max(PE_MA_WINDOW + 60, 150)
    bars = client.get_bars(symbols, timeframe="Day", lookback_days=lookback)
    data = {s: df for s, df in bars.items() if not df.empty and len(df) > PE_MA_WINDOW}
    if len(data) < 5:
        logger.warning("Only %d symbols with enough data", len(data))
        return {}

    ttm_eps: dict[str, float] = {}
    for sym in data:
        eps = _fetch_eps_today(sym, today_ts)
        if eps is not None and eps > 0:
            ttm_eps[sym] = eps

    pe_series: dict[str, pd.Series] = {}
    for sym, df in data.items():
        if sym not in ttm_eps:
            continue
        eps_ttm = ttm_eps[sym]
        close_series = df["close"]
        pe_vals = close_series / eps_ttm
        pe_vals = pe_vals.where((pe_vals > 0) & (pe_vals < 200))
        pe_series[sym] = pe_vals

    signals = {}
    for sym, pe in pe_series.items():
        if len(pe) < PE_MA_WINDOW:
            continue
        pe_ma = pe.rolling(PE_MA_WINDOW, min_periods=30).mean()
        current_pe = pe.iloc[-1]
        current_pe_ma = pe_ma.iloc[-1]
        if pd.isna(current_pe) or pd.isna(current_pe_ma):
            continue
        if current_pe < current_pe_ma:
            signals[sym] = 1.0
        else:
            signals[sym] = 0.0

    return signals


def get_exits(client: AlpacaClient, current_signals: dict[str, float]) -> dict[str, str]:
    positions = client.trading_client.get_all_positions()
    if not positions:
        return {}

    stored = _load_positions()
    symbols = [p.symbol for p in positions]
    today = datetime.now(timezone.utc).date()
    to_close = {}

    for pos in positions:
        sym = pos.symbol
        direction = stored.get(sym, {}).get("direction", 1)

        if direction == 1 and current_signals.get(sym, 0.0) != 1.0:
            to_close[sym] = "pe_exit"
            logger.info("Closing %s: PE signal no longer long (direction=long)", sym)
        elif direction == -1 and current_signals.get(sym, 0.0) != -1.0:
            to_close[sym] = "pe_exit"
            logger.info("Closing %s: PE signal no longer short (direction=short)", sym)

    return to_close


def execute_trades(
    client: AlpacaClient,
    to_close: dict[str, str],
    current_signals: dict[str, float],
    max_pos: int,
):
    trading_client = client.trading_client
    account = trading_client.get_account()
    equity = float(account.equity)
    existing = {p.symbol: abs(float(p.qty)) for p in trading_client.get_all_positions()}
    stored = _load_positions()
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    orders = []

    for sym in to_close:
        qty = abs(int(existing.get(sym, 0)))
        if qty == 0:
            continue
        try:
            trading_client.submit_order(MarketOrderRequest(
                symbol=sym, qty=qty, side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            ))
            orders.append(f"Closed {qty} {sym} ({to_close[sym]})")
            logger.info("Closed %s %d %s", sym, qty, to_close[sym])
            stored.pop(sym, None)
        except Exception as e:
            logger.warning("Failed to close %s: %s", sym, e)

    current_count = sum(1 for s in existing if s not in to_close)
    remaining = max_pos - current_count

    if remaining > 0:
        long_candidates = {
            sym: sig for sym, sig in current_signals.items()
            if sig == 1.0 and sym not in to_close and sym not in existing
        }
        if long_candidates:
            selected = list(long_candidates.keys())[:remaining]
            alloc = equity * MAX_POSITION_PCT
            for sym in selected:
                bars = client.get_bars([sym], "Day", 5)
                if not bars or sym not in bars or bars[sym].empty:
                    continue
                price = float(bars[sym]["close"].iloc[-1])
                if price <= 0:
                    continue
                qty = max(1, int(alloc / price))
                try:
                    trading_client.submit_order(MarketOrderRequest(
                        symbol=sym, qty=qty, side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                    ))
                    orders.append(f"BUY {qty} {sym}")
                    logger.info("BUY %d %s @ $%.2f", qty, sym, price)
                    stored[sym] = {"entry_date": today_str, "direction": 1.0}
                except Exception as e:
                    logger.warning("Failed to order %s: %s", sym, e)

    _save_positions(stored)
    return orders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["mega", "small"], default="mega")
    args = parser.parse_args()

    symbols = MEGA_SYMBOLS if args.universe == "mega" else SMALL_SYMBOLS
    label = f"val-timing {'mega' if args.universe == 'mega' else 'small'}"
    client = AlpacaClient(paper=True)

    signals = get_signals(client, symbols)

    if not signals:
        logger.info("%s — no signals generated", label)
        return

    to_close = get_exits(client, signals)

    if not to_close and not any(v == 1.0 for v in signals.values()):
        logger.info("%s — no trades", label)
        return

    logger.info("%s — close %d, long signals: %d", label, len(to_close), sum(1 for v in signals.values() if v == 1.0))
    for sym, reason in to_close.items():
        logger.info("  CLOSE %s (%s)", sym, reason)
    for sym, sig in signals.items():
        if sig == 1.0:
            logger.info("  LONG %s", sym)

    orders = execute_trades(client, to_close, signals, MAX_OPEN_POSITIONS)
    logger.info("Orders placed: %d", len(orders))


if __name__ == "__main__":
    main()

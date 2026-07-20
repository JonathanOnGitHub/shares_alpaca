"""Forward paper-trading for cross-sectional momentum.
Run monthly via cron. Supports --universe small (best) or large.

Best configs from walk-forward validation:
  small-cap: 12m lookback, 1m skip, top 5/5  (+69%, 0.92 Sharpe)
  large-cap: 6m lookback,  no skip, top 5/5  (+12%, 0.39 Sharpe)

Usage:
  python -m src.trading.momentum_forward --universe small
  python -m src.trading.momentum_forward --universe large
"""
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from src.data.alpaca_client import AlpacaClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

CONFIGS = {
    "small": {
        "symbols": [
            'AEO','ANF','BOOT','CROX','DKS','FIVE','OLLI','CHWY','CVNA','COIN',
            'HOOD','AFRM','UPST','UAA','BBWI','ALGM','CRSP','BEAM','FIVN','RNG',
            'QTWO','TOST','MNDY','GTLB','SMAR','WIX','PATH','CALM','EXAS','GH',
            'NTLA','TXG','MARA','RIOT','SOFI','PLTR','NET','ZS','CFLT','DDOG',
            'MDB','SNOW','OKTA','MRNA','BIIB','ILMN',
        ],
        "lookback_days": 252,
        "skip_days": 21,
        "top_n": 5,
        "max_position_pct": 0.05,
    },
    "large": {
        "symbols": [
            'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','JPM','V','WMT',
            'JNJ','PG','XOM','BAC','DIS','HD','CVX','UNH','MA','COST',
            'NFLX','ADBE','CRM','AMD','CSCO','PFE','ABBV','MRK','TMO','AVGO',
            'ACN','LIN','TXN','QCOM','AMGN','NEE','PM','ORCL','HON','RTX',
            'LOW','IBM','CAT','GE','MCD','BKNG','T','SPGI','DE','SYK',
        ],
        "lookback_days": 126,
        "skip_days": 0,
        "top_n": 5,
        "max_position_pct": 0.04,
    },
}


def get_signals(client, cfg: dict) -> dict[str, float]:
    lookback = cfg["lookback_days"]
    skip = cfg["skip_days"]
    top_n = cfg["top_n"]
    symbols = cfg["symbols"]

    fetch_days = max(600, int((lookback + skip) * 1.4))
    bars = client.get_bars(symbols, 'Day', fetch_days)

    prices = {}
    for s in symbols:
        if s in bars and not bars[s].empty and len(bars[s]) >= lookback + skip + 10:
            close = bars[s]['close']
            if hasattr(close.index, 'tz') and close.index.tz is not None:
                close.index = close.index.tz_localize(None)
            prices[s] = close

    if len(prices) < top_n * 2 + 1:
        logger.warning("Only %d/%d symbols have enough data", len(prices), len(symbols))
        return {}

    pf = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in prices.items()]))
    if hasattr(pf.index, 'tz') and pf.index.tz is not None:
        pf.index = pf.index.tz_localize(None)
    pf = pf.ffill().dropna(how='all')

    if len(pf) < lookback + skip:
        return {}

    past = pf.iloc[-lookback:-skip] if skip > 0 else pf.iloc[-lookback:]
    returns = (past.iloc[-1] / past.iloc[0] - 1)
    ranked = returns.rank(ascending=False)

    signals = {}
    for sym in ranked.nlargest(top_n).index:
        signals[sym] = 1.0
    for sym in ranked.nsmallest(top_n).index:
        signals[sym] = -1.0

    return signals


def execute_signals(client, signals: dict[str, float], cfg: dict):
    trading_client = client.trading_client
    max_pos_pct = cfg["max_position_pct"]
    top_n = cfg["top_n"]

    existing = {p.symbol: float(p.qty) for p in trading_client.get_all_positions()}
    account = trading_client.get_account()
    equity = float(account.equity)

    target = {}
    for sym, direction in signals.items():
        if direction > 0:
            alloc = equity * max_pos_pct
            bars = client.get_bars([sym], 'Day', 5)
            if sym not in bars or bars[sym].empty:
                continue
            price = float(bars[sym]['close'].iloc[-1])
            if price > 0:
                target[sym] = max(1, int(alloc / price))

    symbols_in_play = set(signals.keys()) | set(existing.keys())
    orders = []

    for sym in symbols_in_play:
        current = existing.get(sym, 0)
        desired = target.get(sym, 0)

        if current == desired:
            continue

        if desired == 0 and current != 0:
            side = OrderSide.SELL if current > 0 else OrderSide.BUY
            try:
                trading_client.submit_order(MarketOrderRequest(
                    symbol=sym, qty=abs(int(current)), side=side, time_in_force=TimeInForce.DAY,
                ))
                orders.append(f"Closed {sym} ({side} {int(abs(current))})")
                logger.info("Closed %s", sym)
            except Exception as e:
                logger.warning("Failed to close %s: %s", sym, e)
        elif desired > 0 and current != desired:
            diff = desired - current
            side = OrderSide.BUY if diff > 0 else OrderSide.SELL
            qty = abs(int(diff))
            try:
                trading_client.submit_order(MarketOrderRequest(
                    symbol=sym, qty=qty, side=side, time_in_force=TimeInForce.DAY,
                ))
                orders.append(f"{side} {qty} {sym}")
                logger.info("%s %d %s", side, qty, sym)
            except Exception as e:
                logger.warning("Failed to order %s: %s", sym, e)

    return orders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["small", "large"], default="small",
                        help="Stock universe (default: small — best walk-forward result)")
    args = parser.parse_args()

    cfg = CONFIGS[args.universe]
    label = f"{args.universe}-cap momentum"

    client = AlpacaClient(paper=True)
    signals = get_signals(client, cfg)

    if not signals:
        logger.warning("No signals generated for %s", label)
        return

    logger.info("%s — %d signals", label, len(signals))
    for sym, direction in sorted(signals.items(), key=lambda x: -x[1]):
        logger.info("  %s %s", "LONG " if direction > 0 else "SHORT", sym)

    orders = execute_signals(client, signals, cfg)
    logger.info("Orders placed: %d", len(orders))


if __name__ == "__main__":
    main()

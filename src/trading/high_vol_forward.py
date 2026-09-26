"""
Forward paper-trading for the High-Volatility strategy.
Rebalances every October, selects top N most volatile stocks from S&P 500 proxy,
equal weights, holds for 1 year.

Run monthly via cron (or daily — will no-op if not October).
The first rebalance date of the year is used as the annual rebalance.

Usage:
  python -m src.trading.high_vol_forward
  python -m src.trading.high_vol_forward --universe sp500
  python -m src.trading.high_vol_forward --force   # force rebalance regardless of month
"""
import argparse
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data.alpaca_client import AlpacaClient
from src.trading.high_vol import HighVolStrategy, UNIVERSE

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# Re-balance every October
REBAL_MONTH = 10


def is_rebalance_day(today: datetime) -> bool:
    """True if today is in October (rebalance month)."""
    return today.month == REBAL_MONTH


def get_target_positions(client: AlpacaClient, n_stocks: int = 10) -> dict[str, float]:
    """
    Compute equal-weight target positions for top N highest-volatility stocks.
    Uses trailing 12-month volatility from Alpaca price history.
    """
    lookback = 400  # ~14 months of daily data for monthly resampling
    bars = client.get_bars(UNIVERSE, "Day", lookback_days=lookback)

    data = {}
    for sym, df in bars.items():
        if df.empty or len(df) < 250:
            continue
        data[sym] = df

    if len(data) < n_stocks:
        logger.warning("Only %d/%d symbols have enough data", len(data), len(UNIVERSE))
        return {}

    # Build monthly prices from daily bars
    monthly_prices = {}
    for sym, df in data.items():
        s = df["close"].copy()
        s.index = pd.to_datetime(s.index)
        s.index = s.index.tz_localize(None) if s.index.tz else s.index
        monthly_prices[sym] = s.resample("ME").last().ffill()

    price_df = pd.DataFrame(monthly_prices)
    monthly_rets = price_df.pct_change().dropna()

    # Rebalance today: use previous month's end as formation date
    today = datetime.now(timezone.utc)
    formation_dt = pd.Timestamp(today.year, today.month, 1) - pd.DateOffset(days=1)
    # Actually use last month-end as the formation date
    formation_dt = formation_dt.replace(day=1) - pd.DateOffset(days=1)

    hist = monthly_rets.loc[monthly_rets.index < formation_dt]
    if len(hist) < 12:
        logger.warning("Not enough history for vol computation (%d months)", len(hist))
        return {}

    vol = hist.iloc[-12:].std() * np.sqrt(12)
    vol = vol.dropna()
    vol = vol[vol > 0]

    if len(vol) < n_stocks:
        logger.warning("Only %d stocks with valid vol estimate", len(vol))
        return {}

    top_vol = vol.nlargest(n_stocks)
    weights = {ticker: (1.0 / n_stocks) for ticker in top_vol.index}
    return weights


def execute_rebalance(
    client: AlpacaClient,
    targets: dict[str, float],
    max_pos_pct: float = 0.20,
) -> None:
    """
    Submit market orders to shift from current positions to target weights.
    Liquidates any existing positions not in the new target set.
    """
    trading_client = client.trading_client
    account = trading_client.get_account()
    equity = float(account.equity)

    # Current positions
    existing = {}
    for p in trading_client.get_all_positions():
        existing[p.symbol] = abs(float(p.qty))

    # Get current prices for targets
    bars = client.get_bars(list(targets.keys()), "Day", 5)
    target_qty = {}
    for sym, w in targets.items():
        if sym not in bars or bars[sym].empty:
            continue
        price = float(bars[sym]["close"].iloc[-1])
        if price <= 0:
            continue
        alloc = equity * w
        target_qty[sym] = max(1, int(alloc / price))

    all_syms = set(target_qty.keys()) | set(existing.keys())

    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    orders_placed = 0
    for sym in sorted(all_syms):
        current = existing.get(sym, 0)
        desired = target_qty.get(sym, 0)

        if current == desired:
            continue

        if desired == 0 and current > 0:
            # Liquidate
            try:
                trading_client.submit_order(MarketOrderRequest(
                    symbol=sym,
                    qty=int(current),
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                ))
                logger.info("SELL %d %s (liquidate)", int(current), sym)
                orders_placed += 1
            except Exception as e:
                logger.warning("Failed to sell %s: %s", sym, e)

        elif desired > 0:
            diff = desired - current
            if diff == 0:
                continue
            side = OrderSide.BUY if diff > 0 else OrderSide.SELL
            qty = abs(int(diff))
            try:
                trading_client.submit_order(MarketOrderRequest(
                    symbol=sym,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                ))
                logger.info("%s %d %s", side.value, qty, sym)
                orders_placed += 1
            except Exception as e:
                logger.warning("Failed to order %s: %s", sym, e)

    logger.info("Total orders placed: %d", orders_placed)


def main():
    parser = argparse.ArgumentParser(description="High-Vol paper trading — October rebalance")
    parser.add_argument("--n-stocks", type=int, default=10,
                        help="Number of top-vol stocks to hold (default: 10)")
    parser.add_argument("--force", action="store_true",
                        help="Force rebalance regardless of month")
    args = parser.parse_args()

    today = datetime.now(timezone.utc)
    logger.info("HighVol forward — checking rebalance for %s", today.strftime("%Y-%m-%d"))

    if not args.force and not is_rebalance_day(today):
        logger.info("Not October — skipping (use --force to override)")
        return

    client = AlpacaClient(paper=True)
    targets = get_target_positions(client, n_stocks=args.n_stocks)

    if not targets:
        logger.warning("No target positions generated — check data or try --force next month")
        return

    logger.info("HighVol — targeting %d positions (top %d by vol):",
                len(targets), args.n_stocks)
    for sym, w in sorted(targets.items(), key=lambda x: -x[1]):
        logger.info("  %s: %.1f%%", sym, w * 100)

    execute_rebalance(client, targets)


if __name__ == "__main__":
    main()

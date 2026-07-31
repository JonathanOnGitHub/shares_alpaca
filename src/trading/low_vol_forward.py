"""Forward paper-trading for low-volatility strategy.
Run monthly via cron (or daily — will no-op if not rebalance day).

Selects the lowest-volatility stocks from a mega-cap universe,
weights them by inverse volatility, and rebalances monthly.

Usage:
  python -m src.trading.low_vol_forward
  python -m src.trading.low_vol_forward --universe mega
"""
import argparse
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data.alpaca_client import AlpacaClient
from src.trading.low_vol import LowVol

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

MEGA_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT',
    'JNJ', 'PG', 'XOM', 'BAC', 'DIS', 'HD', 'CVX', 'UNH', 'MA', 'COST',
]

SMALL_UNIVERSE = [
    'AEO', 'ANF', 'BOOT', 'CROX', 'DKS', 'FIVE', 'OLLI', 'CHWY', 'CVNA', 'COIN',
    'HOOD', 'SOFI', 'PLTR', 'MRVI', 'NET', 'AFRM', 'UPST', 'UAA', 'BBWI', 'ALGM',
]

UNIVERSE_CONFIGS = {
    "mega": {"symbols": MEGA_UNIVERSE, "n_stocks": 8, "vol_window": 63, "lookback_days": 21},
    "small": {"symbols": SMALL_UNIVERSE, "n_stocks": 6, "vol_window": 63, "lookback_days": 21},
}


def is_rebalance_day(today: datetime) -> bool:
    return today.day <= 3


def get_target_positions(client: AlpacaClient, cfg: dict) -> dict[str, float]:
    symbols = cfg["symbols"]
    lookback = max(cfg["vol_window"] * 3, 300)
    bars = client.get_bars(symbols, "Day", lookback_days=lookback)
    data = {s: df for s, df in bars.items() if not df.empty and len(df) > cfg["vol_window"]}
    if len(data) < cfg["n_stocks"]:
        logger.warning("Only %d/%d symbols have enough data", len(data), len(symbols))
        return {}

    strategy = LowVol({
        "vol_window": cfg["vol_window"],
        "lookback_days": cfg["lookback_days"],
        "n_stocks": cfg["n_stocks"],
        "weighting": "inverse_vol",
        "rebalance_frequency": "monthly",
    })

    signals = strategy.compute_signals(data)
    if signals.empty:
        return {}

    latest = signals.iloc[-1]
    active = latest[latest != 0]
    return active.to_dict()


def execute_rebalance(
    client: AlpacaClient,
    targets: dict[str, float],
    max_pos_pct: float = 0.15,
):
    trading_client = client.trading_client
    account = trading_client.get_account()
    equity = float(account.equity)

    existing = {}
    for p in trading_client.get_all_positions():
        existing[p.symbol] = abs(float(p.qty))

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
    for sym in all_syms:
        current = existing.get(sym, 0)
        desired = target_qty.get(sym, 0)
        if current == desired:
            continue
        if desired == 0 and current > 0:
            try:
                trading_client.submit_order(MarketOrderRequest(
                    symbol=sym, qty=int(current), side=OrderSide.SELL,
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
                    symbol=sym, qty=qty, side=side, time_in_force=TimeInForce.DAY,
                ))
                logger.info("%s %d %s", side.value, qty, sym)
                orders_placed += 1
            except Exception as e:
                logger.warning("Failed to order %s: %s", sym, e)

    logger.info("Total orders placed: %d", orders_placed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["mega", "small"], default="mega")
    parser.add_argument("--force", action="store_true", help="Force rebalance regardless of day")
    args = parser.parse_args()

    cfg = UNIVERSE_CONFIGS[args.universe]
    today = datetime.now(timezone.utc)

    if not args.force and not is_rebalance_day(today):
        logger.info("Not a rebalance day (day %d). Skip.", today.day)
        return

    client = AlpacaClient(paper=True)
    targets = get_target_positions(client, cfg)

    if not targets:
        logger.warning("No target positions generated — check data")
        return

    logger.info("Low-vol %s — targeting %d positions:", args.universe, len(targets))
    for sym, w in sorted(targets.items(), key=lambda x: -x[1]):
        logger.info("  %s: %.1f%%", sym, w * 100)

    execute_rebalance(client, targets)


if __name__ == "__main__":
    main()

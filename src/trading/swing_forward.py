"""Forward paper-trading for swing strategy.
Run daily via cron.

Usage:
  python -m src.trading.swing_forward --universe small
  python -m src.trading.swing_forward --universe large
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src.data.alpaca_client import AlpacaClient
from src.trading.swing import SwingTrading

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

POSITIONS_FILE = Path(__file__).resolve().parent / "swing_positions.json"


def _load_positions() -> dict:
    if POSITIONS_FILE.exists():
        return json.loads(POSITIONS_FILE.read_text())
    return {}


def _save_positions(positions: dict):
    POSITIONS_FILE.write_text(json.dumps(positions, indent=2))

CONFIGS = {
    "small": {
        "symbols": [
            'AEO','ANF','BOOT','CROX','DKS','FIVE','OLLI','CHWY','CVNA','COIN',
            'HOOD','SOFI','PLTR','MRNA','NET',
        ],
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "sma_trend": 50,
        "volume_avg_multiplier": 1.2,
        "atr_stop_mult": 2.0,
        "atr_target_mult": 3.0,
        "max_holding_days": 15,
        "min_holding_days": 2,
        "max_position_pct": 0.2,
        "max_open_positions": 5,
    },
    "large": {
        "symbols": [
            'AAPL','MSFT','GOOGL','AMZN','SPY',
        ],
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        "sma_trend": 20,
        "volume_avg_multiplier": 1.0,
        "atr_stop_mult": 2.0,
        "atr_target_mult": 3.0,
        "max_holding_days": 10,
        "min_holding_days": 2,
        "max_position_pct": 0.15,
        "max_open_positions": 8,
    },
}


def get_positions_to_close(
    client: AlpacaClient, cfg: dict,
) -> dict[str, str]:
    """Check open positions for exit conditions and return {symbol: reason}."""
    positions = client.trading_client.get_all_positions()
    if not positions:
        return {}

    stored = _load_positions()
    symbols = [p.symbol for p in positions]
    bars = client.get_bars(symbols, timeframe="Day", lookback_days=max(cfg["max_holding_days"] + 20, 60))
    indicators = {}
    for s in symbols:
        if s not in bars or bars[s].empty:
            continue
        df = bars[s]
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        df["rsi"] = 100 - (100 / (1 + rs))
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        indicators[s] = df

    today = datetime.now(timezone.utc).date()
    to_close = {}
    for pos in positions:
        s = pos.symbol
        if s not in indicators:
            continue
        df = indicators[s]
        row = df.iloc[-1]
        if pd.isna(row.get("close")) or pd.isna(row.get("atr")) or row["atr"] <= 0:
            continue

        entry_price = float(pos.avg_entry_price)
        entry_str = stored.get(s, {}).get("entry_date")
        if entry_str:
            entry_date = datetime.fromisoformat(entry_str).date()
        else:
            entry_date = today
            logger.warning("No stored entry date for %s, using today", s)
        current_price = float(pos.current_price)
        direction = 1 if float(pos.qty) > 0 else -1
        holding_days = (today - entry_date).days

        atr = row["atr"]
        reason = None

        if direction == 1:
            target = entry_price + cfg["atr_target_mult"] * atr
            stop = entry_price - cfg["atr_stop_mult"] * atr
            if current_price >= target:
                reason = "take_profit"
            elif current_price <= stop:
                reason = "stop_loss"
            elif holding_days >= cfg["max_holding_days"]:
                reason = "max_holding"
            elif not pd.isna(row.get("rsi")) and row["rsi"] > cfg["rsi_overbought"]:
                reason = "rsi_reversal"
        else:
            target = entry_price - cfg["atr_target_mult"] * atr
            stop = entry_price + cfg["atr_stop_mult"] * atr
            if current_price <= target:
                reason = "take_profit"
            elif current_price >= stop:
                reason = "stop_loss"
            elif holding_days >= cfg["max_holding_days"]:
                reason = "max_holding"
            elif not pd.isna(row.get("rsi")) and row["rsi"] < cfg["rsi_oversold"]:
                reason = "rsi_reversal"

        if reason is not None and holding_days >= cfg["min_holding_days"]:
            to_close[s] = reason
            logger.info("Closing %s: %s (entry=%.2f curr=%.2f held=%dd)", s, reason, entry_price, current_price, holding_days)

    return to_close


def get_entry_signals(client: AlpacaClient, cfg: dict) -> dict[str, float]:
    """Return {symbol: direction} for new entry signals (1.0 long, -1.0 short)."""
    symbols = cfg["symbols"]
    lookback = max(cfg["sma_trend"] + cfg["max_holding_days"] + 50, 200)
    bars = client.get_bars(symbols, timeframe="Day", lookback_days=lookback)
    data = {s: df for s, df in bars.items() if not df.empty and len(df) > lookback // 2}

    strategy = SwingTrading({
        "rsi_period": 14,
        "rsi_oversold": cfg["rsi_oversold"],
        "rsi_overbought": cfg["rsi_overbought"],
        "sma_trend": cfg["sma_trend"],
        "sma_volume": 20,
        "volume_avg_multiplier": cfg["volume_avg_multiplier"],
        "atr_period": 14,
        "atr_stop_mult": cfg["atr_stop_mult"],
        "atr_target_mult": cfg["atr_target_mult"],
        "max_holding_days": cfg["max_holding_days"],
        "min_holding_days": cfg["min_holding_days"],
        "max_position_pct": cfg["max_position_pct"],
        "max_open_positions": cfg["max_open_positions"],
        "initial_capital": 100_000.0,
        "slippage_pct": 0.001,
        "commission_pct": 0.0,
    })

    signals = strategy.compute_signals(data)
    entries = {}
    for sym, sig_series in signals.items():
        if sig_series.empty:
            continue
        latest_sig = sig_series.iloc[-1]
        if latest_sig != 0:
            entries[sym] = latest_sig
    return entries


def execute_signals(client: AlpacaClient, to_close: dict[str, str], to_open: dict[str, float], cfg: dict):
    trading_client = client.trading_client
    max_pos_pct = cfg["max_position_pct"]
    max_open = cfg["max_open_positions"]

    account = trading_client.get_account()
    equity = float(account.equity)
    existing = {p.symbol: float(p.qty) for p in trading_client.get_all_positions()}
    stored = _load_positions()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    orders = []

    for sym in to_close:
        qty = abs(int(existing.get(sym, 0)))
        if qty == 0:
            continue
        try:
            trading_client.submit_order(MarketOrderRequest(
                symbol=sym, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
            ))
            orders.append(f"Closed {qty} {sym} ({to_close[sym]})")
            logger.info("Closed %d %s — %s", qty, sym, to_close[sym])
            stored.pop(sym, None)
        except Exception as e:
            logger.warning("Failed to close %s: %s", sym, e)

    current_count = sum(1 for s in existing if s not in to_close)
    remaining_slots = max_open - current_count

    if remaining_slots > 0:
        bars = client.get_bars(list(to_open.keys()), "Day", 5)
        for sym, direction in sorted(to_open.items(), key=lambda x: -x[1]):
            if remaining_slots <= 0:
                break
            if sym in to_close or sym in existing:
                continue
            if sym not in bars or bars[sym].empty:
                continue
            price = float(bars[sym]["close"].iloc[-1])
            if price <= 0:
                continue
            alloc = equity * max_pos_pct
            qty = max(1, int(alloc / price))
            side = OrderSide.BUY if direction > 0 else OrderSide.SELL
            try:
                trading_client.submit_order(MarketOrderRequest(
                    symbol=sym, qty=qty, side=side, time_in_force=TimeInForce.DAY,
                ))
                orders.append(f"{side} {qty} {sym}")
                logger.info("%s %d %s", side, qty, sym)
                stored[sym] = {"entry_date": today, "direction": direction}
                remaining_slots -= 1
            except Exception as e:
                logger.warning("Failed to order %s: %s", sym, e)

    _save_positions(stored)
    return orders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["small", "large"], default="small")
    args = parser.parse_args()

    cfg = CONFIGS[args.universe]
    label = f"{args.universe}-cap swing"
    client = AlpacaClient(paper=True)

    to_close = get_positions_to_close(client, cfg)
    to_open = get_entry_signals(client, cfg)

    if not to_close and not to_open:
        logger.info("%s — no trades", label)
        return

    logger.info("%s — close %d, open %d", label, len(to_close), len(to_open))
    for sym, reason in to_close.items():
        logger.info("  CLOSE %s (%s)", sym, reason)
    for sym, direction in to_open.items():
        logger.info("  OPEN %s %s", "LONG" if direction > 0 else "SHORT", sym)

    orders = execute_signals(client, to_close, to_open, cfg)
    logger.info("Orders placed: %d", len(orders))


if __name__ == "__main__":
    main()

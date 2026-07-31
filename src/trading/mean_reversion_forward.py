"""Forward paper-trading for mean-reversion strategy.
Run daily via cron.

Strategy:
  Long: RSI crosses below oversold (25) then back above (buy the bounce)
  Short: RSI crosses above overbought (75) then back below (short the drop)
  Exit: RSI mean-reverts (55 for longs, 45 for shorts), ATR target/stop hit,
        or max holding days reached.

Usage:
  python -m src.trading.mean_reversion_forward
  python -m src.trading.mean_reversion_forward --universe mega
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
from src.trading.mean_reversion import MeanReversion

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

POSITIONS_FILE = Path(__file__).resolve().parent / "meanrev_positions.json"

MEGA_CONFIG = {
    "symbols": [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT',
        'JNJ', 'PG', 'XOM', 'BAC', 'DIS', 'HD', 'CVX', 'UNH', 'MA', 'COST',
        'NFLX', 'ADBE', 'CRM', 'AMD', 'CSCO', 'PFE', 'ABBV', 'MRK', 'TMO', 'AVGO',
    ],
    "rsi_oversold": 25,
    "rsi_overbought": 75,
    "rsi_exit_long": 55,
    "rsi_exit_short": 45,
    "atr_target_mult": 1.5,
    "atr_stop_mult": 1.0,
    "max_holding_days": 3,
    "min_holding_days": 1,
    "max_position_pct": 0.10,
    "max_open_positions": 8,
}

SMALL_CONFIG = {
    "symbols": [
        'AEO', 'ANF', 'BOOT', 'CROX', 'DKS', 'FIVE', 'OLLI', 'CHWY', 'CVNA', 'COIN',
        'HOOD', 'AFRM', 'UPST', 'UAA', 'BBWI', 'ALGM', 'CRSP', 'BEAM', 'FIVN', 'RNG',
        'QTWO', 'TOST', 'MNDY', 'GTLB', 'SMAR', 'WIX', 'PATH', 'CALM', 'EXAS', 'GH',
    ],
    "rsi_oversold": 20,
    "rsi_overbought": 80,
    "rsi_exit_long": 55,
    "rsi_exit_short": 45,
    "atr_target_mult": 1.5,
    "atr_stop_mult": 1.0,
    "max_holding_days": 2,
    "min_holding_days": 1,
    "max_position_pct": 0.10,
    "max_open_positions": 8,
}


def _load_positions() -> dict:
    if POSITIONS_FILE.exists():
        return json.loads(POSITIONS_FILE.read_text())
    return {}


def _save_positions(positions: dict):
    POSITIONS_FILE.write_text(json.dumps(positions, indent=2))


def _compute_indicators(df: pd.DataFrame, rsi_period: int = 14, atr_period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = 100 - (100 / (1 + rs))
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(atr_period).mean()
    return df


def get_exits(client: AlpacaClient, cfg: dict) -> dict[str, str]:
    positions = client.trading_client.get_all_positions()
    if not positions:
        return {}

    stored = _load_positions()
    symbols = [p.symbol for p in positions]
    bars = client.get_bars(symbols, timeframe="Day", lookback_days=max(cfg["max_holding_days"] + 30, 60))
    indicators = {}
    for s in symbols:
        if s not in bars or bars[s].empty:
            continue
        indicators[s] = _compute_indicators(bars[s])

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

        direction = 1 if float(pos.qty) > 0 else -1
        holding_days = (today - entry_date).days
        atr = row["atr"]
        entry_atr = stored.get(s, {}).get("entry_atr", atr)
        current_price = float(pos.current_price)

        reason = None
        if direction == 1:
            target = entry_price + cfg["atr_target_mult"] * entry_atr
            stop = entry_price - cfg["atr_stop_mult"] * entry_atr
            if current_price >= target:
                reason = "take_profit"
            elif current_price <= stop:
                reason = "stop_loss"
            elif holding_days >= cfg["max_holding_days"]:
                reason = "max_holding"
            elif not pd.isna(row.get("rsi")) and row["rsi"] > cfg["rsi_exit_long"]:
                reason = "rsi_exit"
        else:
            target = entry_price - cfg["atr_target_mult"] * entry_atr
            stop = entry_price + cfg["atr_stop_mult"] * entry_atr
            if current_price <= target:
                reason = "take_profit"
            elif current_price >= stop:
                reason = "stop_loss"
            elif holding_days >= cfg["max_holding_days"]:
                reason = "max_holding"
            elif not pd.isna(row.get("rsi")) and row["rsi"] < cfg["rsi_exit_short"]:
                reason = "rsi_exit"

        if reason is not None and holding_days >= cfg["min_holding_days"]:
            to_close[s] = reason
            logger.info("Closing %s: %s (entry=%.2f curr=%.2f held=%dd atr=%.2f)",
                        s, reason, entry_price, current_price, holding_days, entry_atr)

    return to_close


def get_entries(client: AlpacaClient, cfg: dict) -> dict[str, float]:
    symbols = cfg["symbols"]
    lookback = max(cfg["max_holding_days"] + 60, 150)
    bars = client.get_bars(symbols, timeframe="Day", lookback_days=lookback)
    data = {s: df for s, df in bars.items() if not df.empty and len(df) > 60}
    if len(data) < 5:
        logger.warning("Only %d symbols with data", len(data))
        return {}

    indicators = {s: _compute_indicators(df) for s, df in data.items()}

    prev_rsi = {s: ind["rsi"].shift(1) for s, ind in indicators.items()}
    curr_rsi = {s: ind["rsi"] for s, ind in indicators.items()}

    entries = {}
    for s in data:
        if s not in prev_rsi or s not in curr_rsi:
            continue
        prev = prev_rsi[s]
        curr = curr_rsi[s]
        if prev.isna().all() or curr.isna().all():
            continue

        prev_val = prev.iloc[-1]
        curr_val = curr.iloc[-1]
        if pd.isna(prev_val) or pd.isna(curr_val):
            continue

        if (
            prev_val <= cfg["rsi_oversold"] and curr_val > cfg["rsi_oversold"]
        ):
            entries[s] = 1.0
        elif (
            prev_val >= cfg["rsi_overbought"] and curr_val < cfg["rsi_overbought"]
        ):
            entries[s] = -1.0

    return entries


def execute_trades(
    client: AlpacaClient,
    to_close: dict[str, str],
    to_open: dict[str, float],
    cfg: dict,
):
    trading_client = client.trading_client
    max_pos_pct = cfg["max_position_pct"]
    max_open = cfg["max_open_positions"]

    account = trading_client.get_account()
    equity = float(account.equity)
    existing = {p.symbol: abs(float(p.qty)) for p in trading_client.get_all_positions()}
    stored = _load_positions()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

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
    remaining = max_open - current_count

    if remaining > 0:
        bars = client.get_bars(list(to_open.keys()), "Day", 5)
        for sym, direction in sorted(to_open.items(), key=lambda x: -x[1]):
            if remaining <= 0:
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
            bars_sym = client.get_bars([sym], "Day", 14)
            atr = 0.0
            if bars_sym and sym in bars_sym and not bars_sym[sym].empty:
                high_low = bars_sym[sym]["high"] - bars_sym[sym]["low"]
                atr = float(high_low.iloc[-14:].mean())
            try:
                trading_client.submit_order(MarketOrderRequest(
                    symbol=sym, qty=qty, side=side, time_in_force=TimeInForce.DAY,
                ))
                orders.append(f"{side.value} {qty} {sym}")
                logger.info("%s %d %s", side.value, qty, sym)
                stored[sym] = {
                    "entry_date": today,
                    "direction": direction,
                    "entry_atr": atr,
                }
                remaining -= 1
            except Exception as e:
                logger.warning("Failed to order %s: %s", sym, e)

    _save_positions(stored)
    return orders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["mega", "small"], default="mega")
    args = parser.parse_args()

    cfg = MEGA_CONFIG if args.universe == "mega" else SMALL_CONFIG
    label = f"mean-rev {'mega' if args.universe == 'mega' else 'small'}"
    client = AlpacaClient(paper=True)

    to_close = get_exits(client, cfg)
    to_open = get_entries(client, cfg)

    if not to_close and not to_open:
        logger.info("%s — no trades", label)
        return

    logger.info("%s — close %d, open %d", label, len(to_close), len(to_open))
    for sym, reason in to_close.items():
        logger.info("  CLOSE %s (%s)", sym, reason)
    for sym, direction in to_open.items():
        logger.info("  OPEN %s %s", "LONG" if direction > 0 else "SHORT", sym)

    orders = execute_trades(client, to_close, to_open, cfg)
    logger.info("Orders placed: %d", len(orders))


if __name__ == "__main__":
    main()

"""Forward paper-trading for covered call strategy.
Run monthly via cron.

Strategy:
  - Hold equal-weight mega-cap long stock positions
  - Sell 30-day calls at ~5% OTM against each position
  - Collect premium immediately; hold stock until called away or rebalanced

Paper trading approach:
  - Tracks positions + open short calls in valcalls_positions.json
  - Uses Black-Scholes to model premium and check assignment
  - Executes stock trades via Alpaca market orders
  - "Assigns" covered calls when stock > strike at expiry:
    sells stock at strike price, collects premium

Usage:
  python -m src.trading.covered_calls_forward
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from scipy.stats import norm

from src.data.alpaca_client import AlpacaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

POSITIONS_FILE = Path(__file__).resolve().parent / "valcalls_positions.json"

MEGA_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT',
    'JNJ', 'PG', 'XOM', 'BAC', 'DIS', 'HD', 'CVX', 'UNH', 'MA', 'COST',
    'NFLX', 'ADBE', 'CRM', 'AMD', 'CSCO', 'PFE', 'ABBV', 'MRK', 'TMO', 'AVGO',
]

STRIKE_OTM_PCT = 0.05
VOL_WINDOW = 20
DAYS_TO_EXPIRY = 30
MAX_OPEN_POSITIONS = 8
MAX_POSITION_PCT = 0.10
REBAL_DAYS = 21


def _black_scholes_call(S: float, K: float, T: float, sigma: float, r: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return max(0.0, S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _fetch_rf_rate() -> float:
    try:
        tbill = yf.Ticker("^IRX")
        hist = tbill.history(period="3mo")
        if hist is None or hist.empty:
            return 0.05
        return float(hist["Close"].iloc[-1]) / 100.0
    except Exception:
        return 0.05


def _load_positions() -> dict:
    if POSITIONS_FILE.exists():
        return json.loads(POSITIONS_FILE.read_text())
    return {"stocks": {}, "calls": {}, "last_rebal": None}


def _save_positions(positions: dict):
    POSITIONS_FILE.write_text(json.dumps(positions, indent=2))


def _compute_vol(prices: pd.Series, window: int = 20) -> float:
    if len(prices) < window:
        return 0.30
    ret = prices.pct_change().dropna()
    if len(ret) < 5:
        return 0.30
    return float(ret.iloc[-window:].std() * np.sqrt(252))


def get_option_premium(symbol: str, S: float, r: float, days: int = 30) -> tuple[float, float]:
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            K = S * (1 + STRIKE_OTM_PCT)
            sigma = 0.30
            T = days / 252.0
            price = _black_scholes_call(S, K, T, sigma, r)
            return price, K

        nearest_exp = min(expirations, key=lambda x: abs((datetime.fromisoformat(x) - datetime.now()).days))
        chain = ticker.option_chain(nearest_exp)
        calls = chain.calls
        K_target = S * (1 + STRIKE_OTM_PCT)

        candidates = calls[calls["bid"] > 0].copy()
        if candidates.empty:
            K = S * (1 + STRIKE_OTM_PCT)
            sigma = 0.30
            T = days / 252.0
            price = _black_scholes_call(S, K, T, sigma, r)
            return price, K

        closest_idx = (candidates["strike"] - K_target).abs().idxmin()
        row = candidates.loc[closest_idx]
        strike = float(row["strike"])
        price = float(row["lastPrice"])
        if price <= 0 or np.isnan(price):
            iv = float(row["impliedVolatility"])
            if iv <= 0:
                iv = 0.30
            T = days / 252.0
            price = _black_scholes_call(S, strike, T, iv, r)
        return price, strike
    except Exception:
        K = S * (1 + STRIKE_OTM_PCT)
        sigma = 0.30
        T = days / 252.0
        price = _black_scholes_call(S, K, T, sigma, r)
        return price, K


def check_expirations(client: AlpacaClient, positions: dict, r: float) -> list[str]:
    today = datetime.now(timezone.utc).date()
    to_close_calls = []
    to_assign_stocks = []

    for sym, call_info in list(positions.get("calls", {}).items()):
        expiry_str = call_info.get("expiry")
        if expiry_str is None:
            continue
        expiry_date = datetime.fromisoformat(expiry_str).date()
        if today >= expiry_date:
            to_close_calls.append(sym)
            strike = call_info["strike"]
            stock_price = call_info.get("stock_price", 0)
            if stock_price > strike:
                to_assign_stocks.append(sym)

    return to_close_calls + to_assign_stocks


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    client = AlpacaClient(paper=True)
    trading_client = client.trading_client
    r = _fetch_rf_rate()

    stored = _load_positions()
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_ts = pd.Timestamp(datetime.now(timezone.utc))

    bar_lookback = max(VOL_WINDOW + 30, 150)
    bars = client.get_bars(MEGA_SYMBOLS, timeframe="Day", lookback_days=bar_lookback)
    data = {s: df for s, df in bars.items() if not df.empty and len(df) > VOL_WINDOW}

    positions = trading_client.get_all_positions()
    stock_holdings = {p.symbol: float(p.qty) for p in positions}

    rebal_due = False
    last_rebal = stored.get("last_rebal")
    if last_rebal is None:
        rebal_due = True
    else:
        last_rebal_date = datetime.fromisoformat(last_rebal).date()
        days_since = (today_ts.date() - last_rebal_date).days
        rebal_due = days_since >= REBAL_DAYS

    process_trades = check_expirations(client, stored, r)
    rebal_due = rebal_due or bool(process_trades)

    if rebal_due:
        logger.info("Rebalance due — processing...")

    actions_log = []

    for sym in process_trades:
        call_info = stored.get("calls", {}).get(sym, {})
        if not call_info:
            continue
        premium = call_info.get("premium", 0)
        strike = call_info["strike"]
        shares = stock_holdings.get(sym, 0)

        if sym in data and not data[sym].empty:
            current_price = float(data[sym]["close"].iloc[-1])
        else:
            current_price = strike

        stock_qty = int(abs(stock_holdings.get(sym, 0)))
        if stock_qty > 0:
            try:
                trading_client.submit_order(MarketOrderRequest(
                    symbol=sym, qty=stock_qty, side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                ))
                logger.info("SOLD %d %s @ $%.2f (call assigned at strike $%.2f, premium $%.2f)",
                            stock_qty, sym, strike, strike, premium)
                actions_log.append(f"ASSIGN: {sym} qty={stock_qty} strike={strike}")
                stock_holdings.pop(sym, None)
            except Exception as e:
                logger.warning("Failed to sell %s: %s", sym, e)

        if sym in stored.get("stocks", {}):
            stored["stocks"].pop(sym, None)
        stored["calls"].pop(sym, None)

    if rebal_due:
        account = trading_client.get_account()
        equity = float(account.equity)
        alloc = equity * MAX_POSITION_PCT

        open_symbols = list(stock_holdings.keys())
        available = [s for s in MEGA_SYMBOLS if s not in open_symbols and s in data and not data[s].empty]
        to_open = available[:MAX_OPEN_POSITIONS - len(open_symbols)]

        all_target = open_symbols + to_open
        logger.info("Rebalance: holding %d, opening %d new symbols", len(open_symbols), len(to_open))

        for sym in open_symbols:
            if sym not in data or data[sym].empty:
                continue
            prices = data[sym]["close"]
            sigma = _compute_vol(prices, VOL_WINDOW)
            S = float(prices.iloc[-1])
            premium, strike = get_option_premium(sym, S, r, DAYS_TO_EXPIRY)
            expiry_date = today_ts + pd.Timedelta(days=DAYS_TO_EXPIRY)

            stored["stocks"][sym] = {
                "entry_date": today_str,
                "shares": int(stock_holdings.get(sym, 0)),
                "entry_price": S,
            }
            stored["calls"][sym] = {
                "strike": strike,
                "premium": premium,
                "expiry": expiry_date.isoformat(),
                "stock_price": S,
            }
            actions_log.append(f"HOLD: {sym} call premium=${premium:.2f} strike=${strike:.2f} expiry={expiry_date.date()}")

        for sym in to_open:
            if sym not in data or data[sym].empty:
                continue
            prices = data[sym]["close"]
            sigma = _compute_vol(prices, VOL_WINDOW)
            S = float(prices.iloc[-1])
            if S <= 0:
                continue
            premium, strike = get_option_premium(sym, S, r, DAYS_TO_EXPIRY)
            qty = max(1, int(alloc / S))
            expiry_date = today_ts + pd.Timedelta(days=DAYS_TO_EXPIRY)

            try:
                trading_client.submit_order(MarketOrderRequest(
                    symbol=sym, qty=qty, side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                ))
                logger.info("BUY %d %s @ $%.2f, sold call premium $%.2f (strike $%.2f)",
                            qty, sym, S, premium, strike)
                actions_log.append(f"OPEN: {sym} qty={qty} @${S:.2f} call=${premium:.2f} strike=${strike:.2f}")
                stock_holdings[sym] = qty
                stored["stocks"][sym] = {
                    "entry_date": today_str,
                    "shares": qty,
                    "entry_price": S,
                }
                stored["calls"][sym] = {
                    "strike": strike,
                    "premium": premium,
                    "expiry": expiry_date.isoformat(),
                    "stock_price": S,
                }
            except Exception as e:
                logger.warning("Failed to buy %s: %s", sym, e)

        stored["last_rebal"] = today_str

    _save_positions(stored)
    logger.info("Done. Actions: %d", len(actions_log))
    for a in actions_log:
        logger.info("  %s", a)


if __name__ == "__main__":
    main()

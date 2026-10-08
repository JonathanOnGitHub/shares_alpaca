"""
executor_live.py
================
Paper trading executor for CONFIRM_TIGHT intraday signals.

Reads signals.json (written by signal_gen_service.py) and executes paper trades
via Alpaca (US / S&P 500) or IBKR (UK / FTSE 100).

Architecture
────────────
  signal_gen_service  →  signals.json  →  executor_live  →  broker API (paper mode)

Each signal carries: symbol, direction, entry_price, stop_price, target_price, atr.

Execution rules
───────────────
  • One active position per symbol at a time.
  • Enter at entry_price (or market order if stale > 3 bars).
  • Stop-loss and take-profit are monitor-only (no OCO — paper execution tracks them).
  • Time-exit: if neither stop nor target fires within time_exit_bars, close at bar close.
  • Commission: 0.75 bps per side (Alpaca), negotiable on IBKR.

Signal staleness
─────────────────
  If the signal bar is more than 3 bars old and we haven't entered, skip the signal.
  If the entry price is > 0.2% away from current price, use market order instead.

Broker support
──────────────
  US (Alpaca):
    - Paper trading via TradingClient(paper=True)
    - Commission: free (US equities)
    - Market hours: 09:30–16:00 ET

  UK (IBKR):
    - Paper trading via IBKR Python API (paper port 7497)
    - Commission: ~£3–5/trade
    - Market hours: 08:00–16:30 UKT
    - Requires ib_insync: pip install ib_insync
"""

from __future__ import annotations

import json
import logging
import signal as _signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event
from typing import Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Broker backends
# ─────────────────────────────────────────────────────────────────────────────

class BrokerBackend:
    """Abstract broker interface."""

    def submit_market_order(self, symbol: str, qty: float, side: Literal["buy", "sell"]) -> str:
        raise NotImplementedError

    def submit_limit_order(self, symbol: str, qty: float, limit_price: float, side: Literal["buy", "sell"]) -> str:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError

    def get_positions(self) -> dict[str, float]:
        raise NotImplementedError

    def close_position(self, symbol: str, qty: Optional[float] = None) -> None:
        raise NotImplementedError

    def get_cash(self) -> float:
        raise NotImplementedError

    def market_hours_open(self) -> bool:
        raise NotImplementedError


class AlpacaBackend(BrokerBackend):
    """Alpaca paper trading backend (US equities only)."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest

        self._TradingClient  = TradingClient
        self._OrderSide      = OrderSide
        self._TimeInForce    = TimeInForce
        self._MarketOrderRequest  = MarketOrderRequest
        self._LimitOrderRequest  = LimitOrderRequest

        self.client = TradingClient(api_key, secret_key, paper=paper)
        self._api_key    = api_key
        self._secret_key = secret_key

    def submit_market_order(self, symbol: str, qty: float, side: Literal["buy", "sell"]) -> str:
        req = self._MarketOrderRequest(
            symbol=symbol, qty=qty,
            side=self._OrderSide.BUY if side == "buy" else self._OrderSide.SELL,
            time_in_force=self._TimeInForce.DAY,
        )
        order = self.client.submit_order(req)
        return str(order.id)

    def submit_limit_order(self, symbol: str, qty: float, limit_price: float, side: Literal["buy", "sell"]) -> str:
        req = self._LimitOrderRequest(
            symbol=symbol, qty=qty, limit_price=limit_price,
            side=self._OrderSide.BUY if side == "buy" else self._OrderSide.SELL,
            time_in_force=self._TimeInForce.DAY,
        )
        order = self.client.submit_order(req)
        return str(order.id)

    def cancel_order(self, order_id: str) -> None:
        self.client.cancel_order_by_id(order_id)

    def get_positions(self) -> dict[str, float]:
        return {p.symbol: float(p.qty) for p in self.client.get_all_positions()}

    def close_position(self, symbol: str, qty: Optional[float] = None) -> None:
        self.client.close_position(symbol)

    def get_cash(self) -> float:
        return float(self.client.get_account().cash)

    def market_hours_open(self) -> bool:
        # US equity market: 09:30–16:00 ET
        now_et = pd.Timestamp.now("America/New_York")
        if now_et.hour < 9 or now_et.hour >= 16:
            return False
        if now_et.weekday() >= 5:
            return False
        return True


class IBKRBackend(BrokerBackend):
    """IBKR paper trading backend (global stocks including FTSE 100)."""

    def __init__(self, account: str = "paper"):
        self._ib = None
        self._account = account
        self._connected = False
        self._connect()

    def _connect(self):
        try:
            from ib_insync import IB
        except ImportError:
            logger.error("ib_insync not installed. Run: pip install ib_insync")
            raise

        self._ib = IB()
        # Paper trading: port 7497 (live: 7496)
        self._ib.connect("127.0.0.1", 7497, clientId=99)
        self._connected = True
        logger.info("[IBKR] Connected to paper trading port 7497")

    def submit_market_order(self, symbol: str, qty: float, side: Literal["buy", "sell"]) -> str:
        contract = self._make_contract(symbol)
        order = self._ib.marketOrder(
            "BUY" if side == "buy" else "SELL", abs(qty)
        )
        trade = self._ib.placeOrder(contract, order)
        return str(trade.order.orderId)

    def submit_limit_order(self, symbol: str, qty: float, limit_price: float, side: Literal["buy", "sell"]) -> str:
        contract = self._make_contract(symbol)
        order = self._ib.limitOrder(
            "BUY" if side == "buy" else "SELL", abs(qty), limit_price
        )
        trade = self._ib.placeOrder(contract, order)
        return str(trade.order.orderId)

    def cancel_order(self, order_id: str) -> None:
        self._ib.cancelOrder(self._ib.orderId(int(order_id)))

    def get_positions(self) -> dict[str, float]:
        positions = {}
        for pos in self._ib.positions():
            positions[pos.contract.symbol] = pos.position
        return positions

    def close_position(self, symbol: str, qty: Optional[float] = None) -> None:
        current = self.get_positions()
        if symbol in current:
            qty_to_close = qty if qty is not None else current[symbol]
            self.submit_market_order(symbol, abs(qty_to_close), "sell")

    def get_cash(self) -> float:
        # IBKR account value — approximate via account summary
        pass  # implement if needed

    def market_hours_open(self) -> bool:
        # UK market: 08:00–16:30 UKT
        now_uk = pd.Timestamp.now("Europe/London")
        if now_uk.hour < 8 or now_uk.hour >= 16 or (now_uk.hour == 16 and now_uk.minute > 30):
            return False
        if now_uk.weekday() >= 5:
            return False
        return True

    def _make_contract(self, symbol: str):
        """Build an IBKR contract. Assumes stock symbol; FTSE index uses different secType."""
        from ib_insync import Stock
        # Determine exchange: LSE for UK stocks, SMART for US
        exchange = "LSE" if symbol.isupper() and len(symbol) <= 3 else "SMART"
        return Stock(symbol, exchange=exchange)


# ─────────────────────────────────────────────────────────────────────────────
# Active position tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActivePosition:
    symbol:       str
    direction:    Literal["long", "short"]
    entry_price:  float
    stop_price:   float
    target_price: float
    atr:          float
    qty:          float
    signal_time:  pd.Timestamp
    entry_time:   Optional[pd.Timestamp]
    order_id:     Optional[str]
    status:       Literal["pending", "entered", "exited"] = "pending"
    exit_reason:  Optional[str] = None
    exit_price:   Optional[float] = None
    pnl_bps:      Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# PaperExecutor
# ─────────────────────────────────────────────────────────────────────────────

class PaperExecutor:
    """
    Reads signals.json and executes paper trades via the configured broker backend.

    Execution loop:
      1. Poll signals.json every `poll_seconds`.
      2. For each new signal not already tracked, submit a limit order at entry_price.
      3. Monitor active positions: check if stop / target / time-exit fires each bar.
      4. Close positions and log PnL.
      5. Write trade log to paper_trades.csv.
    """

    STALENESS_BARS    = 3       # skip signals older than this
    MARKET_ORDER_AGE  = 3       # bars before switching to market order
    PAPER_CAPITAL_US  = 10_000.0  # USD
    PAPER_CAPITAL_UK  = 10_000.0  # GBP

    def __init__(
        self,
        signals_file: str = "signals.json",
        broker: Literal["alpaca", "ibkr"] = "alpaca",
        market: Literal["US", "UK"] = "US",
        poll_seconds: int = 5,
        log_file: str = "paper_trades.csv",
    ):
        self.signals_file = Path(signals_file)
        self.market      = market.upper()
        self.poll_seconds = poll_seconds
        self.log_file    = Path(log_file)
        self._stop       = Event()

        # Initialise broker backend
        if broker == "alpaca":
            self._init_alpaca()
        elif broker == "ibkr":
            self._init_ibkr()
        else:
            raise ValueError(f"Unknown broker: {broker}")

        self.positions: dict[str, ActivePosition] = {}  # symbol → active position
        self.trade_log: list[dict] = []
        self._bar_count = 0   # simple counter for staleness

        logger.info(
            "[%s/%s] PaperExecutor initialised | broker=%s | signals=%s",
            market, broker, broker, signals_file,
        )

    def _init_alpaca(self):
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_key    = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        if not api_key or not secret_key:
            raise ValueError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env")
        self.broker = AlpacaBackend(api_key, secret_key, paper=True)
        self.capital = self.PAPER_CAPITAL_US
        logger.info("[Alpaca] Paper trading initialised (commission-free)")

    def _init_ibkr(self):
        self.broker = IBKRBackend(account="paper")
        self.capital = self.PAPER_CAPITAL_UK
        logger.info("[IBKR] Paper trading initialised")

    # ── public ─────────────────────────────────────────────────────────────

    def run(self):
        """Blocking execution loop."""
        self._stop.clear()
        logger.info("[%s] Executor running | capital=$%.2f", self.market, self.capital)

        while not self._stop.is_set():
            try:
                self._bar_count += 1
                self._process_signals()
                self._monitor_positions()
                self._check_time_exits()
            except Exception as e:
                logger.error("[%s] Executor error: %s", self.market, e, exc_info=True)

            if self._stop.wait(timeout=self.poll_seconds):
                break

        self._close_all()
        self._write_log()
        logger.info("[%s] Executor stopped", self.market)

    def stop(self):
        self._stop.set()

    # ── internal ───────────────────────────────────────────────────────────

    def _process_signals(self):
        """Read new signals and submit orders."""
        if not self.signals_file.exists():
            return

        try:
            data = json.loads(self.signals_file.read_text())
        except Exception as e:
            logger.warning("Failed to read signals.json: %s", e)
            return

        signals = data.get("signals", [])
        now = pd.Timestamp.now()

        for sig in signals:
            sym = sig.get("symbol")
            if not sym:
                continue
            if sym in self.positions and self.positions[sym].status != "exited":
                continue  # already have an active position

            sig_ts = pd.Timestamp(sig.get("timestamp"))
            # Staleness check
            bars_old = self._bar_count - (sig_ts.value // (self.poll_seconds * 10**9))
            if bars_old > self.STALENESS_BARS:
                logger.info("[%s] Skipping stale signal for %s (%d bars old)", self.market, sym, bars_old)
                continue

            self._open_position(sig, now)

    def _open_position(self, sig: dict, now: pd.Timestamp):
        sym         = sig["symbol"]
        direction   = sig["direction"]   # "long" or "short"
        entry_price = float(sig["entry_price"])
        stop_price  = float(sig["stop_price"])
        target      = float(sig["target_price"])
        atr         = float(sig["atr"])
        sig_ts      = pd.Timestamp(sig.get("timestamp"))

        # Position sizing: risk 1% of capital per trade
        risk_pct  = 0.01
        risk_amt  = self.capital * risk_pct
        stop_dist = abs(entry_price - stop_price)
        if stop_dist == 0:
            logger.warning("[%s] Zero stop distance for %s — skipping", self.market, sym)
            return
        qty = risk_amt / stop_dist

        # Submit limit order at entry_price
        side = "buy" if direction == "long" else "sell"
        try:
            if bars_old := self._bar_count - (sig_ts.value // (self.poll_seconds * 10**9)) < self.MARKET_ORDER_AGE:
                order_id = self.broker.submit_limit_order(sym, qty, entry_price, side)
            else:
                order_id = self.broker.submit_market_order(sym, qty, side)
                logger.info("[%s] %s: market order (signal stale)", self.market, sym)
        except Exception as e:
            logger.error("[%s] Order submission failed for %s: %s", self.market, sym, e)
            return

        self.positions[sym] = ActivePosition(
            symbol=sym,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target,
            atr=atr,
            qty=qty,
            signal_time=sig_ts,
            entry_time=None,
            order_id=order_id,
            status="pending",
        )
        logger.info(
            "[%s] OPEN %s %s qty=%.2f @ %.4f  stop=%.4f  target=%.4f",
            self.market, direction, sym, qty, entry_price, stop_price, target,
        )

    def _monitor_positions(self):
        """Check stop / target hits on all entered positions."""
        for sym, pos in list(self.positions.items()):
            if pos.status != "entered":
                continue

            # Get current market price (use latest bar from signal_gen_service's live.jsonl)
            try:
                current_price = self._get_current_price(sym)
            except Exception:
                continue

            should_exit = False
            exit_reason = None
            exit_price  = None

            if pos.direction == "long":
                if current_price <= pos.stop_price:
                    should_exit, exit_reason, exit_price = True, "stop", pos.stop_price
                elif current_price >= pos.target_price:
                    should_exit, exit_reason, exit_price = True, "target", pos.target_price
            else:  # short
                if current_price >= pos.stop_price:
                    should_exit, exit_reason, exit_price = True, "stop", pos.stop_price
                elif current_price <= pos.target_price:
                    should_exit, exit_reason, exit_price = True, "target", pos.target_price

            if should_exit:
                self._close_position(sym, exit_reason, exit_price)

    def _check_time_exits(self):
        """Force exit any position that has exceeded time_exit_bars."""
        # Not implemented without live bar timestamps — placeholder
        pass

    def _close_position(self, sym: str, reason: str, exit_price: float):
        pos = self.positions[sym]
        side = "sell" if pos.direction == "long" else "buy"

        try:
            self.broker.close_position(sym)
        except Exception as e:
            logger.warning("[%s] Broker close failed for %s: %s", self.market, sym, e)

        if pos.direction == "long":
            pnl_bps = (exit_price - pos.entry_price) / pos.entry_price * 10_000
        else:
            pnl_bps = (pos.entry_price - exit_price) / pos.entry_price * 10_000

        # Commission: 0.75 bps per side × 2
        pnl_bps -= 1.5
        self.capital *= (1 + pnl_bps / 10_000)

        pos.status     = "exited"
        pos.exit_reason = reason
        pos.exit_price  = exit_price
        pos.pnl_bps     = pnl_bps

        self.trade_log.append({
            "symbol":       sym,
            "direction":    pos.direction,
            "entry_time":   str(pos.entry_time or pos.signal_time),
            "exit_time":    str(pd.Timestamp.now()),
            "entry_price":  pos.entry_price,
            "exit_price":   exit_price,
            "qty":          pos.qty,
            "pnl_bps":      pnl_bps,
            "exit_reason":  reason,
            "atr":          pos.atr,
            "market":       self.market,
        })

        logger.info(
            "[%s] CLOSE %s %s @ %.4f  pnl=%.1f bps  reason=%s  capital=%.2f",
            self.market, sym, side, exit_price, pnl_bps, reason, self.capital,
        )

    def _close_all(self):
        for sym in list(self.positions):
            pos = self.positions[sym]
            if pos.status == "entered":
                self._close_position(sym, "force_close", pos.entry_price)

    def _get_current_price(self, symbol: str) -> float:
        """
        Get the current market price for a symbol.
        For Alpaca: use get_bars(symbol, '1Min', 1)
        For IBKR: query market data.
        """
        if isinstance(self.broker, AlpacaBackend):
            from src.data.alpaca_client import AlpacaClient
            client = AlpacaClient(paper=True)
            bars = client.get_bars([symbol], "Minute", lookback_days=1)
            if symbol in bars and not bars[symbol].empty:
                return float(bars[symbol]["close"].iloc[-1])
        # Fallback: read latest from live.jsonl
        live_path = Path("live.jsonl")
        if live_path.exists():
            lines = live_path.read_text().strip().split("\n")
            for line in reversed(lines):
                try:
                    bar = json.loads(line)
                    if bar.get("symbol") == symbol:
                        return float(bar["close"])
                except Exception:
                    continue
        raise ValueError(f"Cannot find current price for {symbol}")

    def _write_log(self):
        if not self.trade_log:
            return
        df = pd.DataFrame(self.trade_log)
        df.to_csv(self.log_file, index=False)
        logger.info("[%s] Trade log written to %s (%d trades)", self.market, self.log_file, len(df))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _main():
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="CONFIRM_TIGHT paper trading executor")
    parser.add_argument("--broker", default="alpaca", choices=["alpaca", "ibkr"])
    parser.add_argument("--market", default="US", choices=["US", "UK"])
    parser.add_argument("--signals", default="signals/signals_us.json")
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--log-file", default="paper_trades.csv")
    args = parser.parse_args()

    exec = PaperExecutor(
        signals_file=args.signals,
        broker=args.broker,
        market=args.market,
        poll_seconds=args.poll_seconds,
        log_file=args.log_file,
    )

    def _sig_handler(sig, frame):
        logger.info("Received signal %d", sig)
        exec.stop()
        sys.exit(0)

    _signal.signal(_signal.SIGINT,  _sig_handler)
    _signal.signal(_signal.SIGTERM, _sig_handler)

    exec.run()


if __name__ == "__main__":
    _main()

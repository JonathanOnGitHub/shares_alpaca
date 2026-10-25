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
import math
import signal as _signal
import sys
import time
from dataclasses import dataclass
import datetime as _datetime_mod
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

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Latest traded price for `symbol` from the paper account.

        The executor previously built its own AlpacaClient for this,
        which meant the injected broker was bypassed entirely and any
        test/mock broker was silently ignored. Keeping the lookup here
        means one broker, one price source.
        """
        try:
            from src.data.alpaca_client import AlpacaClient
            client = AlpacaClient(paper=True)
            bars = client.get_bars([symbol], "Minute", lookback_days=1)
            if symbol in bars and not bars[symbol].empty:
                return float(bars[symbol]["close"].iloc[-1])
        except Exception:
            return None
        return None

    def close_position(self, symbol: str, qty: Optional[float] = None) -> None:
        """Close `symbol`, optionally only `qty` shares.

        The Alpaca paper account is shared with other projects, so
        closing the WHOLE broker position for a symbol could flatten a
        position that belongs to a different strategy. When the caller
        knows how much it actually holds, close exactly that much.
        """
        if qty is None:
            self.client.close_position(symbol)
            return
        qty = abs(float(qty))
        if qty < 1e-9:
            return
        pos = {p.symbol: float(p.qty) for p in self.client.get_all_positions()}
        held = pos.get(symbol, 0.0)
        if held == 0:
            return
        # Never close more than is actually held.
        qty = min(qty, abs(held))
        side = self._OrderSide.SELL if held > 0 else self._OrderSide.BUY
        req = self._MarketOrderRequest(
            symbol=symbol, qty=qty, side=side,
            time_in_force=self._TimeInForce.DAY,
        )
        self.client.submit_order(req)

    def get_cash(self) -> float:
        return float(self.client.get_account().cash)

    def market_hours_open(self) -> bool:
        # US equity market: 09:30-16:00 ET.
        # Minute-precision: the old "hour < 9" test opened the gate an
        # hour early, which let orders be submitted pre-market.
        now_et = pd.Timestamp.now("America/New_York")
        if now_et.weekday() >= 5:
            return False
        minutes = now_et.hour * 60 + now_et.minute
        if minutes < 9 * 60 + 30 or minutes >= 16 * 60:
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
    # Overnight tracking
    bar_date:     Optional[str] = None   # date engulfing bar formed (for overnight)
    close_mode:   Literal["intraday", "overnight"] = "intraday"
    close_deadline_ts: Optional[pd.Timestamp] = None  # close by this time
    entry_filled_price: Optional[float] = None  # actual fill price for market orders


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
        strategy: str = "fvg",          # tag for portfolio tracking
        close_mode: Literal["intraday", "overnight"] = "intraday",
    ):
        self.signals_file = Path(signals_file)
        self.market       = market.upper()
        self.poll_seconds = poll_seconds
        self.log_file     = Path(log_file)
        self.strategy     = strategy        # "fvg" or "engulfing"
        self.close_mode   = close_mode      # "intraday" or "overnight"
        self._stop        = Event()

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
        # Signal keys already acted on, so the same setup is never
        # entered twice (see _process_signals). Persisted to disk because
        # the intraday scanner re-derives the same signal on every poll
        # and an executor restart mid-session would otherwise re-enter it.
        self._acted_signals: set[str] = set()
        self._acted_file = Path(f".acted_{self.strategy}.json")
        self._load_acted_signals()
        # Broker positions seen that this executor did not open (another
        # strategy sharing the same Alpaca account). Warned once each.
        self._foreign_positions_seen: set[str] = set()
        # Open positions are persisted so a daemon restart re-adopts the
        # positions it already opened. Without this, a restart left the
        # positions at the broker with no stop, no target and no
        # deadline — an unbounded risk.
        self._positions_file = Path(f".positions_{self.strategy}.json")
        self._load_positions()
        # A restored position whose deadline has already passed belongs to
        # a previous session. It must not be carried into today: close it
        # out now (recording it in the trade log) rather than leaving an
        # unbounded position that no session is watching.
        self._reconcile_stale_positions()

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
                self._reconcile_with_broker()
            except Exception as e:
                logger.error("[%s] Executor error: %s", self.market, e, exc_info=True)

            # Stop cleanly once the session is over and flat, so the next
            # trading day starts from a fresh process.
            if self._session_ended():
                logger.info("[%s] Session over and flat — executor exiting", self.market)
                break

            if self._stop.wait(timeout=self.poll_seconds):
                break

        self._close_all()
        self._write_log()
        logger.info("[%s] Executor stopped", self.market)

    def stop(self):
        self._stop.set()

    # ── internal ───────────────────────────────────────────────────────────
    # ── signal freshness / session helpers ──────────────────────────────────

    ET = "America/New_York"

    # Wall-clock override for tests. Set to a pd.Timestamp to freeze
    # "now"; None means use the real clock. Tests patch this rather than
    # monkeypatching pd.Timestamp, which is far too blunt an instrument.
    _now_override: Optional[pd.Timestamp] = None

    def _now(self):
        """Current time (overridable for tests)."""
        if self._now_override is not None:
            return self._now_override
        return pd.Timestamp.now()

    def _now_et(self):
        """Current time in US/Eastern (respecting the test clock override)."""
        if self._now_override is not None:
            ts = self._now_override
            if ts.tzinfo is None:
                ts = ts.tz_localize(self.ET)
            return ts.tz_convert(self.ET)
        return pd.Timestamp.now(self.ET)

    @staticmethod
    def _next_weekday(d):
        """Next weekday (Mon-Fri) strictly after date d. Returns a date."""
        nd = pd.Timestamp(d).normalize() + pd.Timedelta(days=1)
        while nd.weekday() >= 5:
            nd += pd.Timedelta(days=1)
        return nd.date()

    @staticmethod
    def _add_weekdays(d, n):
        """Add n weekdays to date d. Returns a date."""
        nd = pd.Timestamp(d).normalize()
        while n > 0:
            nd += pd.Timedelta(days=1)
            if nd.weekday() < 5:
                n -= 1
        return nd.date()

    @classmethod
    def _session_close_ts(cls, session_date, hour=16, minute=0):
        """Market close on session_date, as a naive host-local timestamp.

        The session close is a US/Eastern wall-clock event (16:00 ET), so it
        must be built in ET and then converted to host-local time. Building
        it directly in host-local time is wrong by an hour for half the year.

        Note: `tz_convert(None)` yields UTC-naive, NOT host-local naive, so
        the conversion has to go through the host's own timezone explicitly.
        Comparing a UTC-naive deadline against `pd.Timestamp.now()` (which is
        host-local) is off by the UTC offset.
        """
        ts = pd.Timestamp(session_date).tz_localize(cls.ET) + pd.Timedelta(hours=hour, minutes=minute)
        return ts.tz_convert(cls._host_tz()).tz_localize(None)

    @staticmethod
    def _host_tz():
        """The host machine's timezone as a tzinfo object.

        `pd.Timestamp.now()` is already naive host-local time, so its .tz
        attribute is None and cannot be passed to tz_convert(). Use
        astimezone() on an aware UTC timestamp to get the host zone
        reliably.
        """
        return _datetime_mod.datetime.now().astimezone().tzinfo

    def _signal_is_fresh(self, sig: dict) -> bool:
        """
        Session-aware staleness test.

        The previous check compared raw wall-clock seconds against a fixed
        window (16 h for overnight, 3 min for intraday).  That window is
        wrong for a daily-bar signal: the signal is generated once the bar
        has closed and is meant to be entered at the NEXT session's open,
        which can be ~90 h later (Friday signal -> Monday entry).  On top
        of that the executor is started by the same cron run that creates
        the signal, so the signal is minutes old at first poll and any
        wall-clock window measured from "now" expires long before the
        intended entry.

        Judge freshness by the signal's bar date and the session it is
        meant to trade in instead:

        * overnight — valid until the end of the next session's open
          (with one weekday of grace for exchange holidays).
        * intraday  — valid for the remainder of the signal bar's session.
        """
        bar_date = sig.get("bar_date")
        if not bar_date:
            ts = sig.get("timestamp")
            if not ts:
                return True
            try:
                bar_date = str(pd.Timestamp(ts).date())
            except Exception:
                return True
        try:
            bdate = pd.Timestamp(bar_date).date()
        except Exception:
            return True

        now_et = pd.Timestamp.now(self.ET).date()

        # ── Intraday signals ────────────────────────────────────────────────
        # An intraday signal is generated DURING the session from a
        # 5-minute bar and must be entered in that same session. It is
        # NOT a daily-bar signal that waits for the next open.
        #
        # The two intraday flavours differ:
        #   "intraday_fvg_5m_displacement" - 5-min bars, generated live
        #     during the session, entered immediately, flat by 14:00 ET.
        #   daily-bar intraday - generated from the previous session's
        #     close, entered at the next session's open.
        if self.close_mode == "intraday":
            if sig.get("bar_time_et"):
                # Live intraday signal: valid only during its own session.
                return bool(now_et == bdate)
            # Daily-bar intraday signal: entered at the next session's open.
            entry_session = self._next_weekday(bdate)
            return bool(now_et == entry_session)

        # ── Overnight (daily-bar) signals ───────────────────────────────────
        # Entry happens at the OPEN of the session after the signal bar:
        # the backtest enters at the open of bar i+1 and is walked forward
        # for stop/target across sessions.
        entry_session = self._next_weekday(bdate)

        # Overnight: valid from the entry session through one further
        # weekday (the grace window covers exchange holidays).
        grace_limit = self._next_weekday(entry_session)
        return bool(entry_session <= now_et <= grace_limit)

    def _close_deadline_for_entry(self, entry_date, flat_time_et=None):
        """Close deadline for a position entered during session `entry_date`.

        Both strategies enter at the OPEN of the session after the signal
        bar, so the exit is anchored to the session we actually entered in:

        * overnight - the backtest holds the position into the next session
          (exit at the open of the session after entry). The live
          equivalent closes at that session's open.
        * intraday  - the backtest exits at the close of the entry session.
          For the intraday FVG strategy the backtest is explicit: flat by
          14:00 ET, NOT the 16:00 close. `flat_time_et` carries that
          through from the signal.

        The session close/open is a US/Eastern wall-clock event, so the
        timestamp is built in ET and converted. Building it in host-local
        time is wrong by an hour for half the year.
        """
        if self.close_mode == "intraday":
            hour, minute = 16, 0
            if flat_time_et:
                try:
                    hh, mm = str(flat_time_et).split(":")
                    hour, minute = int(hh), int(mm)
                except Exception:
                    hour, minute = 16, 0
            return self._session_close_ts(entry_date, hour=hour, minute=minute)
        # Overnight: open of the next weekday after the entry session.
        next_session = self._next_weekday(entry_date)
        ts = pd.Timestamp(next_session).tz_localize(self.ET) + pd.Timedelta(hours=9, minutes=30)
        return ts.tz_convert(self._host_tz()).tz_localize(None)

    @staticmethod
    def _signal_key(sig: dict) -> str:
        """A stable identity for a signal setup.

        Two signals are the SAME setup when they refer to the same symbol
        and the same bar. Intraday scanners re-derive the day's signals
        on every poll, so the timestamp field (regenerated each scan) is
        NOT a usable identity — the bar time is.

        For daily-bar (overnight) strategies the bar date identifies the
        setup instead.
        """
        sym = sig.get("symbol", "")
        bar_time = sig.get("bar_time_et")
        if bar_time:
            return f"{sym}|{sig.get('bar_date')}|{bar_time}"
        return f"{sym}|{sig.get('bar_date')}|{sig.get('entry_price')}"

    def _load_positions(self):
        """Restore open positions from disk after a daemon restart.

        A restart previously left the broker positions orphaned: the new
        process had an empty `positions` dict, so it neither managed the
        stop/target nor knew the position existed. It then classified
        them as "foreign" positions belonging to another strategy.
        """
        try:
            if not self._positions_file.exists():
                return
            data = json.loads(self._positions_file.read_text())
            for sym, d in (data.get("positions") or {}).items():
                try:
                    d = dict(d)
                    d["signal_time"] = pd.Timestamp(d["signal_time"])
                    if d.get("entry_time"):
                        d["entry_time"] = pd.Timestamp(d["entry_time"])
                    if d.get("close_deadline_ts"):
                        d["close_deadline_ts"] = pd.Timestamp(d["close_deadline_ts"])
                    self.positions[sym] = ActivePosition(**d)
                except Exception as e:
                    logger.warning(
                        "[%s/%s] Could not restore position %s: %s",
                        self.market, self.strategy, sym, e,
                    )
            if self.positions:
                logger.info(
                    "[%s/%s] Restored %d open position(s) from %s: %s",
                    self.market, self.strategy, len(self.positions),
                    self._positions_file, ", ".join(self.positions),
                )
        except Exception as e:
            logger.warning(
                "[%s/%s] Could not read position file %s: %s",
                self.market, self.strategy, self._positions_file, e,
            )

    def _save_positions(self):
        """Persist the open positions to disk."""
        try:
            out = {}
            for sym, pos in self.positions.items():
                if pos.status == "exited":
                    continue
                d = {k: v for k, v in vars(pos).items()}
                for k in ("signal_time", "entry_time", "close_deadline_ts"):
                    if d.get(k) is not None:
                        d[k] = str(d[k])
                out[sym] = d
            self._positions_file.write_text(json.dumps(
                {"positions": out}, indent=2, default=str,
            ))
        except Exception as e:
            logger.warning(
                "[%s/%s] Could not write position file %s: %s",
                self.market, self.strategy, self._positions_file, e,
            )

    def _load_acted_signals(self):
        """Restore the acted-on signal keys from disk.

        The intraday scanner re-derives the same signal on every poll, so
        an executor that restarts mid-session would otherwise re-enter a
        setup it had already traded.
        """
        try:
            if self._acted_file.exists():
                data = json.loads(self._acted_file.read_text())
                self._acted_signals = set(data.get("acted", []))
                logger.info(
                    "[%s/%s] Restored %d acted-on signal key(s) from %s",
                    self.market, self.strategy, len(self._acted_signals),
                    self._acted_file,
                )
        except Exception as e:
            logger.warning(
                "[%s/%s] Could not read acted-signal file %s: %s",
                self.market, self.strategy, self._acted_file, e,
            )

    def _save_acted_signals(self):
        """Persist the acted-on signal keys to disk."""
        try:
            self._acted_file.write_text(
                json.dumps({"acted": sorted(self._acted_signals)}, indent=2)
            )
        except Exception as e:
            logger.warning(
                "[%s/%s] Could not write acted-signal file %s: %s",
                self.market, self.strategy, self._acted_file, e,
            )

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

        # Only trade while the market is open.  The cron wrapper that
        # creates the signal usually runs either before the open or after
        # the close, and Alpaca rejects market orders outside the session.
        # Deferring here is what makes "enter at the next open" work: the
        # signal stays queued and is acted on at the next open.
        if not self.broker.market_hours_open():
            return

        for sig in signals:
            sym = sig.get("symbol")
            if not sym:
                continue
            if sym in self.positions and self.positions[sym].status != "exited":
                continue  # already have an active position

            # A signal key identifies a specific setup: the symbol plus the
            # bar time it was generated from. The intraday FVG scanner
            # re-derives TODAY's signals on every poll, so without this
            # guard the SAME 09:35 signal would be re-entered every 5
            # minutes for the rest of the session — each time after the
            # previous position had been closed by target or stop.
            #
            # The key is derived from the signal's own bar time, so a
            # genuinely new setup (a later bar) is still tradable.
            sig_key = self._signal_key(sig)
            if sig_key in self._acted_signals:
                continue
            if sym in self.positions and self.positions[sym].status == "exited":
                # The position was closed in a previous cycle. Without
                # removing it from the dict, the check above never fires
                # again for this symbol and the symbol can never be
                # re-entered for the rest of the session.
                self.positions.pop(sym, None)

            sig_ts = pd.Timestamp(sig.get("timestamp"))
            if not self._signal_is_fresh(sig):
                logger.info("[%s] Skipping stale signal for %s (bar_date=%s)",
                            self.market, sym, sig.get("bar_date"))
                continue

            self._open_position(sig, now, sig_key)

    def _close_deadline_for(self, sig_ts: pd.Timestamp) -> pd.Timestamp:
        """Return the close deadline for an overnight position.

        Entry fires at the next open, so the deadline must be measured from
        the session we actually enter in — not from the signal date.  Using
        the signal date made the deadline already-in-the-past by the time
        the market order filled at the next open, so the position could be
        force-closed immediately (or never monitored correctly).
        """
        # Next trading day = sig_ts + 1 calendar day, skipping weekends
        next_day = sig_ts.normalize() + pd.Timedelta(days=1)
        while next_day.weekday() >= 5:   # skip Sat/Sun
            next_day += pd.Timedelta(days=1)
        # 16:00 ET = 20:00 UTC (winter) / 21:00 UTC (summer) — use 20:00 UTC as close-of-day proxy
        # Close-of-day is roughly 20:00-21:00 UTC depending on DST; use 20:00 UTC conservatively
        return next_day.replace(hour=20, minute=0, second=0, tzinfo=None)

    def _open_position(self, sig: dict, now: pd.Timestamp, sig_key: Optional[str] = None):
        sym         = sig["symbol"]
        direction   = sig["direction"]   # "long" or "short"
        entry_price = float(sig["entry_price"])
        stop_price  = float(sig["stop_price"])
        target      = float(sig["target_price"])
        atr         = float(sig["atr"])
        sig_ts      = pd.Timestamp(sig.get("timestamp"))

        # Guard: skip if already held at broker (handles re-run / duplicate signal races)
        broker_positions = {p.symbol for p in self.broker.client.get_all_positions()}
        if sym in broker_positions:
            logger.info("[%s] %s already open at broker — skipping", self.market, sym)
            return

        # Position sizing: risk 1% of capital per trade
        risk_pct  = 0.01
        risk_amt  = self.capital * risk_pct

        # The intraday FVG backtest sizes by a FIXED risk ($100/trade) and
        # caps the notional at 2x that risk. The fixed-risk sizing keeps
        # the live P&L per trade directly comparable to the backtest's
        # Sharpe/expectancy, and the notional cap stops a very tight stop
        # (a few cents on a $50 stock) from levering into a huge position.
        if sig.get("risk_dollars"):
            risk_amt = float(sig["risk_dollars"])
        max_risk_mult = float(sig.get("max_risk_mult") or 2.0)

        stop_dist = abs(entry_price - stop_price)
        if stop_dist == 0:
            logger.warning("[%s] Zero stop distance for %s — skipping", self.market, sym)
            return

        # Sanity check the level ordering. A long with stop ABOVE entry is
        # nonsense and would book a "stop" profit the instant it filled.
        if direction == "long" and stop_price >= entry_price:
            logger.warning("[%s] %s: stop %.2f is not below entry %.2f — skipping",
                           self.market, sym, stop_price, entry_price)
            return
        if direction == "short" and stop_price <= entry_price:
            logger.warning("[%s] %s: stop %.2f is not above entry %.2f — skipping",
                           self.market, sym, stop_price, entry_price)
            return

        # Skip if the market has already gapped through the stop since the
        # signal bar closed. Entering there means an instant full stop-out
        # (or, worse, a fictitious profit if the gap is the other way).
        try:
            cur = self._get_current_price(sym)
            if direction == "long" and cur <= stop_price:
                logger.info("[%s] %s: price %.2f already below stop %.2f — skipping entry",
                            self.market, sym, cur, stop_price)
                return
            if direction == "short" and cur >= stop_price:
                logger.info("[%s] %s: price %.2f already above stop %.2f — skipping entry",
                            self.market, sym, cur, stop_price)
                return
        except Exception:
            pass  # price unavailable — let the order attempt proceed

        qty = risk_amt / stop_dist

        # Notional cap: 2x the risk amount. A very tight stop (e.g. a few
        # cents on a $50 stock) would otherwise produce an enormous
        # position for a nominal risk.
        max_notional = risk_amt * max_risk_mult
        if entry_price * qty > max_notional:
            qty = max_notional / entry_price
            logger.info("[%s] %s: notional capped at %.2f (qty=%.2f)",
                        self.market, sym, max_notional, qty)

        # Alpaca's paper API rejects FRACTIONAL short orders outright
        # ("fractional orders cannot be sold short"). Short quantities
        # must therefore be whole numbers. Round UP so the intended risk
        # is not silently reduced, but never below 1 share.
        if direction == "short":
            qty = math.ceil(qty)
            if qty < 1:
                logger.warning("[%s] %s: short size rounds to 0 shares — skipping",
                               self.market, sym)
                return

        # Submit limit order at entry_price (rounded to 2dp for Alpaca penny-increment compliance)
        side = "buy" if direction == "long" else "sell"
        # For overnight engulfing: submit at next open using market order.
        # entry_price is a close-of-engulfing-bar proxy; actual next-open fill is
        # obtained via market order to avoid limit-order miss.
        entry_rounded = round(entry_price, 2)
        try:
            if self.close_mode == "overnight":
                # Market order: enter at actual next open. The signal's
                # entry_price is a close-of-bar proxy, so a limit order at
                # that price would miss when the open gaps away from it.
                order_id = self.broker.submit_market_order(sym, qty, side)
                logger.info("[%s] %s: overnight — market order at %.4f (proxy=%.4f)",
                            self.market, sym, entry_rounded, entry_price)
            else:
                # Intraday: the BACKTEST enters at the signal bar's CLOSE.
                # A limit order at that price only fills if the market
                # comes back to it — so the live book would contain only
                # the setups that happened to retrace, and every signal
                # that ran away without touching the entry price would
                # be silently dropped. That is a selection bias: the
                # live P&L would be measured on a different (and
                # systematically better-behaved) sample than the
                # backtest it is supposed to validate.
                #
                # So intraday entries go out as MARKET orders, which
                # reproduce the backtest's "enter at the signal bar's
                # close" assumption (the scanner polls every 5 minutes,
                # so the market order lands within a few seconds of the
                # bar close, at the close plus the usual slippage).
                #
                # The old code placed a LIMIT order for fresh signals
                # and only switched to a market order once the signal
                # was older than 3 x poll_seconds (180 s). With a
                # 300-second scanner poll, most signals landed in the
                # limit-order window — which is why XOM's order sat
                # unfilled until the end-of-day cleanup cancelled it.
                signal_age_s = (self._now() - sig_ts).total_seconds()
                order_id = self.broker.submit_market_order(sym, qty, side)
                logger.info("[%s] %s: market order (signal %.0f s old, entry proxy=%.4f)",
                            self.market, sym, signal_age_s, entry_rounded)
        except Exception as e:
            logger.error("[%s] Order submission failed for %s: %s", self.market, sym, e)
            return

        stop_rounded = round(stop_price, 2)
        target_rounded = round(target, 2)

        # Deadline is anchored to the session we actually enter in.
        #
        # For a live intraday signal (5-minute bar, bar_time_et present) the
        # entry happens DURING the signal's own session, so the deadline
        # belongs to that session. For a daily-bar signal the entry is at
        # the open of the session AFTER the signal bar.
        #
        # Using the signal date for a daily-bar signal made the deadline
        # already-in-the-past by the time the market order filled at the
        # next open, so the position was force-closed immediately on the
        # following poll.
        sig_bar_date = pd.Timestamp(sig.get("bar_date") or sig_ts).date()
        if sig.get("bar_time_et"):
            # Live intraday signal — entered in its own session.
            entry_session = sig_bar_date
        else:
            entry_session = self._next_weekday(sig_bar_date)
        close_deadline = self._close_deadline_for_entry(
            entry_session, flat_time_et=sig.get("flat_time_et"))

        self.positions[sym] = ActivePosition(
            symbol=sym,
            direction=direction,
            entry_price=entry_rounded,
            stop_price=stop_rounded,
            target_price=target_rounded,
            atr=atr,
            qty=qty,
            signal_time=sig_ts,
            entry_time=None,
            order_id=order_id,
            status="pending",
            bar_date=sig.get("bar_date"),
            close_mode=self.close_mode,
            close_deadline_ts=close_deadline,
        )
        logger.info(
            "[%s] OPEN %s %s qty=%.2f @ %.4f  stop=%.4f  target=%.4f",
            self.market, direction, sym, qty, entry_price, stop_price, target,
        )

        # Mark the setup as acted on ONLY now that an order actually
        # exists. Skips (stale, invalid geometry, price already past the
        # stop) return before this point, so a skipped signal may be
        # reconsidered on a later poll — but a placed order must never
        # be duplicated.
        if sig_key:
            self._acted_signals.add(sig_key)
            self._save_acted_signals()
        self._save_positions()

    def _monitor_positions(self):
        """Check stop / target hits on all entered positions."""
        for sym, pos in list(self.positions.items()):
            # Transition pending → entered once broker confirms fill; record actual price
            if pos.status == "pending":
                try:
                    if self._is_filled(sym, pos.order_id):
                        pos.status = "entered"
                        pos.entry_time = self._now()
                        # For market orders, fetch the actual fill price from broker
                        if pos.entry_filled_price is None:
                            try:
                                fill_price = self._get_fill_price(sym, pos.order_id)
                                if fill_price:
                                    pos.entry_filled_price = round(fill_price, 4)
                            except Exception:
                                pass
                        # Use the ACTUAL filled quantity for a partial fill.
                        #
                        # The order was submitted for `pos.qty` shares, but
                        # a partially filled order leaves fewer shares
                        # actually short. Tracking the requested qty would
                        # mis-state the position size and, on a close,
                        # close more than was actually held.
                        filled_qty = self._get_filled_qty(pos.order_id)
                        if filled_qty and filled_qty > 0:
                            pos.qty = filled_qty
                        logger.info("[%s] %s FILLED qty=%.2f at %.4f (entry_price proxy was %.4f)",
                                    self.market, sym, pos.qty,
                                    pos.entry_filled_price or pos.entry_price, pos.entry_price)
                        self._save_positions()
                    else:
                        # The order is DEAD at the broker (cancelled,
                        # expired, rejected) — it will never fill. The
                        # position must be retired, otherwise it sits in
                        # "pending" forever, blocks the "session over and
                        # flat" exit condition, and wedges the daemon
                        # across the close into the next session.
                        self._retire_dead_pending(sym, pos)
                    continue   # not yet filled — skip stop/target checks
                except Exception:
                    continue

            if pos.status != "entered":
                continue

            # Re-sync the position size while the entry order is still
            # working. A partially filled order can fill further on a
            # later poll, which increases the size actually held; the
            # tracked qty must follow, or the close would move more
            # shares than were ever short.
            try:
                o = self._get_order(pos.order_id)
                if o is not None and self._order_status(o) in (
                        "new", "partially_filled", "pending_new", "accepted"):
                    fq = self._get_filled_qty(pos.order_id)
                    if fq and fq != pos.qty:
                        logger.info("[%s] %s: filled qty changed %.2f -> %.2f",
                                    self.market, sym, pos.qty, fq)
                        pos.qty = fq
            except Exception:
                pass

            # Get current market price
            try:
                current_price = self._get_current_price(sym)
            except Exception:
                continue

            should_exit = False
            exit_reason = None
            exit_price  = None

            # Use actual fill price for P&L if available; proxy entry_price otherwise
            entry_px = pos.entry_filled_price or pos.entry_price
            if pos.direction == "long":
                if current_price <= pos.stop_price:
                    should_exit, exit_reason = True, "stop"
                    # If the market gapped THROUGH the stop, the fill happens
                    # at the gapped price, not at the stop level. Using
                    # pos.stop_price in that case books a fictitious profit
                    # (e.g. bought at 658 with a stop at 690 -> "profit").
                    exit_price = min(pos.stop_price, current_price)
                elif current_price >= pos.target_price:
                    should_exit, exit_reason = True, "target"
                    # Same logic for a gap through the target.
                    exit_price = max(pos.target_price, current_price)
            else:  # short
                if current_price >= pos.stop_price:
                    should_exit, exit_reason = True, "stop"
                    exit_price = max(pos.stop_price, current_price)
                elif current_price <= pos.target_price:
                    should_exit, exit_reason = True, "target"
                    exit_price = min(pos.target_price, current_price)

            if should_exit:
                self._close_position(sym, exit_reason, exit_price)

    def _check_time_exits(self):
        """Force exit overnight positions by their close deadline, or intraday by 4 PM ET."""
        now = pd.Timestamp.now()
        for sym, pos in list(self.positions.items()):
            # A pending position whose deadline has passed must be
            # retired: the order is either dead at the broker or would
            # be filled with no stop attached. Cancel it first, then
            # retire it.
            if pos.status == "pending":
                if pos.close_deadline_ts and now >= pos.close_deadline_ts:
                    self._cancel_open_order(pos.order_id)
                    self._retire_dead_pending(sym, pos)
                continue

            if pos.status != "entered":
                continue

            # Intraday: close at the entry session's close (16:00 ET).
            # The deadline is stored per position, so use it rather than a
            # hardcoded 21:00 host-local hour — the latter is off by an hour
            # for half the year and closes the position before the session
            # actually ends.
            if self.close_mode == "intraday":
                if pos.close_deadline_ts and now >= pos.close_deadline_ts:
                    self._close_position(sym, "intraday_exit", None)
                continue

            # Overnight: close by configured deadline
            if pos.close_deadline_ts and now >= pos.close_deadline_ts:
                self._close_position(sym, "overnight_exit", None)

    def _session_ended(self) -> bool:
        """True once the trading session is over.

        The intraday executor must not linger past the close. A daemon
        left running overnight keeps yesterday's `_acted_signals` set and
        yesterday's `positions` dict, so the next morning's cron run sees
        "executor already running" and reuses a process whose in-memory
        state belongs to the previous session. Exiting at the close makes
        the next session start from a clean process.

        The exit is unconditional once the session is over — it does NOT
        require the position book to be empty. The previous version
        waited for "no position left open", so a single pending position
        whose order had been cancelled wedged the daemon indefinitely:
        it never exited, the next day's wrapper saw a live PID and reused
        the stale process, and the strategy silently stopped trading.

        Any position still open at that point is force-closed by
        `_close_all()` in the exit path, which records it in the trade
        log.
        """
        if self.close_mode != "intraday":
            return False
        now_et = self._now_et()
        if now_et.hour * 60 + now_et.minute < 16 * 60 + 5:
            return False
        return True

    def _reconcile_with_broker(self):
        """Detect broker positions this executor does not know about.

        The Alpaca paper account is shared by several projects (the
        Leveraged and OANDA repos hold the same API key), so a position
        can appear here that was opened by a completely different
        strategy. This executor has no stop, no target and no deadline
        for such a position, so it is an unbounded risk that must be
        flagged loudly.

        A position is "ours" if it is in self.positions. Anything else
        at the broker that is not ours is reported. Positions belonging
        to a different strategy are NOT closed automatically — that
        could break someone else's trade — but they are logged and
        surfaced by the end-of-day report.
        """
        try:
            broker_positions = {
                p.symbol: float(p.qty)
                for p in self.broker.client.get_all_positions()
            }
        except Exception as e:
            logger.warning("[%s] Broker reconciliation failed: %s", self.market, e)
            return

        mine = {s for s, p in self.positions.items() if p.status != "exited"}
        for sym, qty in broker_positions.items():
            if sym in mine:
                continue
            if abs(qty) < 1e-9:
                continue
            # Only warn once per symbol per session.
            if sym in self._foreign_positions_seen:
                continue
            self._foreign_positions_seen.add(sym)
            logger.warning(
                "[%s] UNMANAGED broker position %s qty=%.4f — not opened by "
                "this executor (another strategy sharing the account). "
                "No stop/target/deadline applies to it.",
                self.market, sym, qty,
            )

    def _close_position(self, sym: str, reason: str, exit_price: Optional[float]):
        pos = self.positions[sym]
        side = "sell" if pos.direction == "long" else "buy"

        # Cancel any still-resting entry order FIRST. If the position is
        # closed while its entry limit is still working, that order can
        # fill moments later and open a brand-new, completely unmanaged
        # position (no stop, no target, no deadline).
        self._cancel_open_order(pos.order_id)

        # Fetch market price for time-based exits where no price was specified
        if exit_price is None:
            try:
                exit_price = float(self._get_current_price(sym))
            except Exception:
                exit_price = pos.entry_price  # fallback

        try:
            # Close only what this executor actually holds. The account
            # is shared, so closing the whole broker position for a
            # symbol could flatten another strategy's trade.
            self.broker.close_position(sym, pos.qty)
        except Exception as e:
            logger.warning("[%s] Broker close failed for %s: %s", self.market, sym, e)

        # P&L must be computed from the ACTUAL fill price. The signal's
        # entry_price is a proxy (close of the signal bar); the market
        # order fills at the next open, which can be several percent
        # away. Using the proxy made every overnight trade look like it
        # entered at the signal close rather than the real fill.
        entry_px = pos.entry_filled_price or pos.entry_price
        if pos.direction == "long":
            pnl_bps = (exit_price - entry_px) / entry_px * 10_000
        else:
            pnl_bps = (entry_px - exit_price) / entry_px * 10_000

        # Commission: 0.75 bps per side × 2
        pnl_bps -= 1.5

        # The capital update must use the ACTUAL dollar P&L of the
        # position, not the price move in basis points applied to the
        # whole account.
        #
        # The old code did `capital *= (1 + pnl_bps / 10_000)`, i.e. it
        # treated a 19.3 bps price move as a 19.3 bps move of the entire
        # $10,000 account. A $200 notional position that lost 19.3 bps
        # actually lost $0.39, but the account was marked down by $19.30
        # — a 50x overstatement of the loss. The backtest's Sharpe is
        # computed on RISK-UNIT returns (pnl_dollars / risk_dollars),
        # so the live equity curve must move by dollars to be
        # comparable.
        entry_px_cap = pos.entry_filled_price or pos.entry_price
        notional = abs(float(pos.qty) * float(entry_px_cap))
        pnl_dollars = notional * pnl_bps / 10_000.0
        self.capital += pnl_dollars

        pos.status     = "exited"
        pos.exit_reason = reason
        pos.exit_price  = exit_price
        pos.pnl_bps     = pnl_bps

        # The log row must carry everything the P&L report needs, in the
        # schema the report actually reads:
        #   status    — "closed" for a real fill, "unfilled" for an order
        #               that expired without filling
        #   pnl       — DOLLARS (the report sums this column)
        #   pnl_bps   — basis points (for Sharpe/expectancy comparison
        #               with the backtest)
        #   exit_date — the ET calendar date of the exit, so "daily P&L"
        #               is bucketed by the trading session, not by the
        #               host's local midnight.
        entry_px_log = pos.entry_filled_price or pos.entry_price
        exit_ts = self._now()
        self.trade_log.append({
            "symbol":       sym,
            "strategy":     self.strategy,
            "direction":    pos.direction,
            "entry_time":   str(pos.entry_time or pos.signal_time),
            "exit_time":    str(exit_ts),
            "exit_date":    str(pd.Timestamp.now(self.ET).date()),
            "entry_price":  entry_px_log,
            "exit_price":   exit_price,
            "qty":          pos.qty,
            "pnl":          round(pnl_dollars, 4),
            "pnl_bps":      round(pnl_bps, 4),
            "exit_reason":  reason,
            "status":       "closed",
            "atr":          pos.atr,
            "market":       self.market,
            "close_mode":   pos.close_mode,
        })

        logger.info(
            "[%s] CLOSE %s %s @ %.4f  pnl=%.1f bps ($%+.2f)  reason=%s  capital=%.2f",
            self.market, sym, side, exit_price, pnl_bps, pnl_dollars, reason, self.capital,
        )
        self._save_positions()
        # Write the trade log IMMEDIATELY, not only on process exit. The
        # log used to be written once, when the daemon stopped — so a
        # daemon that was killed (or wedged) at the end of the session
        # left every trade of that session unrecorded, and the daily
        # report showed "0 trades / $0.00" for a day in which trades
        # were actually taken and closed.
        self._write_log()

    def _reconcile_stale_positions(self):
        """Close out positions restored from disk whose session is over.

        On a restart the executor re-adopts the positions it had open.
        If the session that owned them has already ended (the deadline has
        passed), those positions must be closed immediately — they carry a
        stop, target and deadline that no live session is watching, and
        leaving them open is unbounded risk.

        A pending position whose deadline has passed is retired rather
        than closed, because nothing was ever filled at the broker.
        """
        now = pd.Timestamp.now()
        for sym, pos in list(self.positions.items()):
            deadline = pos.close_deadline_ts
            if deadline is None or now < deadline:
                continue
            try:
                if pos.status == "entered":
                    logger.info(
                        "[%s/%s] Restored position %s is past its deadline — closing it out",
                        self.market, self.strategy, sym,
                    )
                    self._close_position(sym, "stale_session_exit", None)
                else:
                    self._retire_dead_pending(sym, pos)
            except Exception as e:
                logger.warning(
                    "[%s/%s] Could not reconcile stale position %s: %s",
                    self.market, self.strategy, sym, e,
                )

    def _retire_dead_pending(self, sym: str, pos: ActivePosition) -> bool:
        """Retire a pending position whose entry order can never fill.

        A pending position whose order was cancelled, expired or rejected
        is not a position at all — the broker holds nothing. Left as
        "pending" it would:

          * block `_session_ended()` forever, so the daemon never exits
            and the next session inherits yesterday's state;
          * sit in `.positions_fvg.json` forever, so a restart re-adopts
            a phantom position with a long-dead order id.

        The row is still written to the trade log, with status "unfilled"
        and zero P&L, so the session's record is complete and auditable
        (a signal that was taken but never filled is a real fact about
        the strategy, not a trade).

        Returns True if the position was retired.
        """
        order = self._get_order(pos.order_id)
        status = self._order_status(order) if order is not None else None

        DEAD_STATUSES = {
            "cancelled", "canceled", "expired", "rejected",
            "replaced", "pending_cancel", "pending_replace",
        }

        dead = status in DEAD_STATUSES

        # The order object is gone entirely (cancelled before the executor
        # ever saw it, or the broker purged it). Treat as dead too, but
        # only once the position's own deadline has passed — otherwise a
        # transient lookup failure would retire a live order.
        if order is None and not dead:
            deadline = pos.close_deadline_ts
            if deadline is not None and self._now() >= deadline:
                dead = True
                status = "unknown_past_deadline"

        if not dead:
            return False

        logger.info(
            "[%s] %s entry order never filled (status=%s) — retiring pending position",
            self.market, sym, status or "unknown",
        )

        pos.status = "exited"
        pos.exit_reason = "unfilled"
        pos.exit_price = None
        pos.pnl_bps = 0.0

        # An unfilled signal belongs to the session that PRODUCED it, not
        # to the session in which the dead order happened to be noticed.
        # A signal from Monday that is retired on Monday's EOD cleanup
        # must be dated Monday. Using "now" would mis-file every
        # back-filled or late-retired signal into the wrong trading day,
        # corrupting the daily P&L bucket.
        session_date = pos.bar_date
        if not session_date:
            try:
                session_date = str(pd.Timestamp(pos.signal_time).date())
            except Exception:
                session_date = str(self._now_et().date())

        self.trade_log.append({
            "symbol":       sym,
            "strategy":     self.strategy,
            "direction":    pos.direction,
            "entry_time":   str(pos.entry_time or pos.signal_time),
            "exit_time":    str(self._now()),
            "exit_date":    str(session_date),
            "entry_price":  pos.entry_filled_price or pos.entry_price,
            "exit_price":   None,
            "qty":          pos.qty,
            "pnl":          0.0,
            "pnl_bps":      0.0,
            "exit_reason":  "unfilled",
            "status":       "unfilled",
            "atr":          pos.atr,
            "market":       self.market,
            "close_mode":   pos.close_mode,
        })
        self._save_positions()
        self._write_log()
        return True

    def _close_all(self):
        """Flatten everything this executor owns, at exit.

        Entered positions are closed at the market; pending positions
        have their entry orders CANCELLED (nothing was filled, so there
        is nothing to close) and are then retired so they do not linger
        in the position file.
        """
        for sym in list(self.positions):
            pos = self.positions[sym]
            if pos.status == "entered":
                self._close_position(sym, "force_close", pos.entry_price)
            elif pos.status == "pending":
                self._cancel_open_order(pos.order_id)
                self._retire_dead_pending(sym, pos)

    def _get_current_price(self, symbol: str) -> float:
        """
        Get the current market price for a symbol.
        For Alpaca: use get_bars(symbol, '1Min', 1)
        For IBKR: query market data.
        """
        if isinstance(self.broker, AlpacaBackend):
            # Prefer the broker's own price method so that the executor,
            # the broker and the tests all agree on one price source.
            # Falling back to a fresh AlpacaClient here would bypass the
            # injected broker entirely.
            getter = getattr(self.broker, "get_current_price", None)
            if callable(getter):
                try:
                    px = getter(symbol)
                    if px:
                        return float(px)
                except Exception:
                    pass
        else:
            # Non-Alpaca (or mock/test) brokers expose their own price
            # method — use it rather than falling through to a live
            # Alpaca lookup.
            getter = getattr(self.broker, "get_current_price", None)
            if callable(getter):
                try:
                    px = getter(symbol)
                    if px:
                        return float(px)
                except Exception:
                    pass
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

    def _get_order(self, order_id: Optional[str]):
        """Return the broker order object for `order_id`, or None.

        Uses the injected broker's own client so that mock/test brokers
        are honoured. The previous implementation built a fresh
        AlpacaClient here, which bypassed the injected broker entirely
        and made the fill-detection path untestable.

        NOTE: this alpaca-py version's `get_orders()` takes NO keyword
        arguments — calling it as `get_orders(status="all", limit=50)`
        raises TypeError, which the old `except Exception: return None`
        swallowed. Every fill check therefore silently returned False,
        so a filled entry was never transitioned to "entered" and the
        position sat unmanaged with no stop and no exit.
        """
        if not order_id:
            return None
        client = getattr(self.broker, "client", None)
        if client is None:
            return None
        try:
            # Try the modern request-object API first, then fall back to
            # the bare call for older SDKs.
            try:
                from alpaca.trading.requests import GetOrdersRequest
                orders = client.get_orders(GetOrdersRequest(status="all", limit=100))
            except ImportError:
                orders = client.get_orders()
        except Exception as e:
            logger.warning("[%s] Could not fetch orders: %s", self.market, e)
            return None
        for o in orders or []:
            if str(getattr(o, "id", "")) == str(order_id):
                return o
        return None

    @staticmethod
    def _order_status(o) -> str:
        """Normalise an order's status to a lowercase string.

        alpaca-py returns an `OrderStatus` enum. `str(enum)` yields
        "OrderStatus.FILLED", NOT "filled", so a naive
        `str(o.status) == "filled"` check never matches. The enum's
        `.value` attribute holds the wire value.
        """
        st = getattr(o, "status", "")
        val = getattr(st, "value", None)
        if val is not None:
            return str(val).lower()
        return str(st).lower()

    def _is_filled(self, symbol: str, order_id: Optional[str]) -> bool:
        """Return True if the order has been filled at the broker.

        A PARTIALLY filled order counts as filled: the broker holds a
        real position for the filled quantity, so the executor must
        start monitoring it for stop/target. The previous version
        required status == "filled" exactly, so a partially filled
        order was never transitioned to "entered" and the position
        sat in "pending" with no stop, no target and no exit — an
        unbounded risk.
        """
        o = self._get_order(order_id)
        if o is None:
            return False
        return self._order_status(o) in ("filled", "partially_filled", "filled_partially")

    def _get_fill_price(self, symbol: str, order_id: Optional[str]) -> float:
        """Return the average fill price of a filled order.

        Falls back to the average of the filled quantity for a partial
        fill, and to 0.0 when the broker reports no fill price.
        """
        o = self._get_order(order_id)
        if o is None:
            return 0.0
        price = getattr(o, "filled_avg_price", None)
        try:
            if price:
                return float(price)
        except Exception:
            pass
        return 0.0

    def _get_filled_qty(self, order_id: Optional[str]) -> float:
        """Return the filled quantity of an order (0 if unknown)."""
        o = self._get_order(order_id)
        if o is None:
            return 0.0
        try:
            return float(getattr(o, "filled_qty", 0) or 0)
        except Exception:
            return 0.0

    def _cancel_open_order(self, order_id: Optional[str]) -> None:
        """Cancel a resting entry order, if it is still open.

        If a position is closed (stop / target / time exit) while its
        entry limit order is still resting, that order can fill later
        and open a NEW, unmanaged position with no stop. This cancels
        any still-open order for the symbol when a position closes.
        """
        if not order_id:
            return
        try:
            o = self._get_order(order_id)
            if o is None:
                return
            status = str(getattr(o, "status", ""))
            if status in ("new", "partially_filled", "pending_new", "accepted"):
                self.broker.cancel_order(order_id)
                logger.info("[%s] Cancelled resting entry order for %s (status=%s)",
                            self.market, getattr(o, "symbol", "?"), status)
        except Exception as e:
            logger.warning("[%s] Failed to cancel resting order %s: %s",
                           self.market, order_id, e)

    def _write_log(self):
        """Persist the trade log to disk, MERGING with any existing rows.

        Two problems this fixes:

        1. The log was only written on process exit. A daemon that was
           killed (or wedged) at the end of the session left every trade
           of that session unrecorded.

        2. The log was OVERWRITTEN with the in-memory list only. After a
           restart the in-memory list is empty, so a restart mid-session
           would wipe the trades already recorded in the file.

        The merge is keyed on (symbol, entry_time, exit_time) so it is
        idempotent: writing the same trade twice is a no-op.
        """
        if not self.trade_log:
            return
        try:
            df_new = pd.DataFrame(self.trade_log)
            if self.log_file.exists() and self.log_file.stat().st_size > 0:
                try:
                    df_old = pd.read_csv(self.log_file)
                except Exception:
                    df_old = pd.DataFrame()
                if not df_old.empty:
                    key_cols = [c for c in ("symbol", "entry_time", "exit_time")
                                if c in df_new.columns and c in df_old.columns]
                    if key_cols:
                        df_old = df_old[~df_old.set_index(key_cols).index.isin(
                            df_new.set_index(key_cols).index)]
                    else:
                        df_old = df_old.iloc[0:0]
                    df_new = pd.concat([df_old, df_new], ignore_index=True)

            df_new = df_new.sort_values(by="exit_time").reset_index(drop=True)
            df_new.to_csv(self.log_file, index=False)
            logger.info("[%s] Trade log written to %s (%d trades)",
                        self.market, self.log_file, len(df_new))
        except Exception as e:
            logger.warning("[%s] Failed to write trade log: %s", self.market, e)


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

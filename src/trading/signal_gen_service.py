"""
signal_gen_service.py
=====================
Daemon that reads live bar data and generates CONFIRM_TIGHT entry signals.

Architecture
────────────
  Data feeder (external)  →  live.jsonl  →  signal_gen_service  →  signals.json  →  executor

live.jsonl format (one JSON object per line):
    {
      "symbol":    "AAPL",
      "timestamp": "2026-10-08 09:35:00",   # bar close time
      "open": 234.5, "high": 234.8, "low": 234.2, "close": 234.6,
      "volume": 10000
    }

signals.json format (written by this service):
    {
      "generated_at": "2026-10-08 09:42:00",
      "signals": [
        {
          "symbol": "AAPL", "timestamp": "2026-10-08 09:42:00",
          "direction": "long",          # "long" or "short"
          "entry_price": 234.55,
          "stop_price":  234.10,         # session_low × 0.998 (for longs)
          "target_price": 235.05,        # entry + 1 × ATR
          "atr": 0.45,
          "session_high": 235.20,
          "session_low":  234.05,
          "market": "US"                # "US" or "UK"
        }
      ]
    }

Usage
─────
    python -m src.trading.signal_gen_service --market US --poll-seconds 5

The service initialises its bar buffer from live.jsonl (full session written at
session open by the data feeder). New bars are appended every poll cycle.

CONFIRM_TIGHT strategy parameters (FTSE 100, 5-min bars):
  lookback:        2 bars (opening drive)
  min_gap_pct:     0.3%
  stop_type:       session_extreme × 0.2%
  target_type:     1 × ATR
  confirm:         True  (front-side must align with session direction)

For S&P 500 the edge is NOT confirmed — use confirm=False and accept
that the strategy has no positive-expectancy evidence at this stage.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from threading import Event, Thread
from typing import Optional

import pandas as pd

from src.trading.intraday_retracement import IntradayRetracement, StrategyConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Market configuration
# ─────────────────────────────────────────────────────────────────────────────

MARKET_CONFIG = {
    "US": {
        "signals_file": "signals_us.json",
        "lookback": 2,
        "confirm": False,          # NOT proven on US — unconfirmed signal
        "min_gap_pct": 0.003,
        "stop_mult": 0.002,
        "exit_atr_mult": 1.0,
        "time_exit_bars": 78,      # 78 × 5-min = 6.5 hrs → exit at 16:00 ET
    },
    "UK": {
        "signals_file": "signals_uk.json",
        "lookback": 2,
        "confirm": True,           # CONFIRM_TIGHT — proven on FTSE 100
        "min_gap_pct": 0.003,
        "stop_mult": 0.002,
        "exit_atr_mult": 1.0,
        "time_exit_bars": 78,      # 78 × 5-min = 6.5 hrs → exit at 16:30 UKT
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# SignalGenerator
# ─────────────────────────────────────────────────────────────────────────────

class SignalGenerator:
    """
    Maintains a rolling bar buffer per symbol and emits CONFIRM_TIGHT signals.

    Bar buffer strategy:
      - live.jsonl is written by an external data feeder.
      - At startup we read ALL existing lines to build the full session bar buffer.
      - Each poll cycle reads new lines since the last read cursor.
      - Per symbol we keep the last N_bars (default 500) to cover a full session.
      - When a new bar triggers an entry signal we write it to signals.json.
    """

    N_BARS_BUFFER = 500   # enough for a full trading session + lookback
    POLL_DEFAULT  = 5     # seconds

    def __init__(
        self,
        live_file: str = "live.jsonl",
        signals_dir: str = "signals",
        market: str = "US",
        poll_seconds: int = POLL_DEFAULT,
        confirm: Optional[bool] = None,
    ):
        self.live_file    = Path(live_file)
        self.signals_dir  = Path(signals_dir)
        self.market       = market.upper()
        self.poll_seconds = poll_seconds

        if self.market not in MARKET_CONFIG:
            raise ValueError(f"Unknown market: {market}. Choose: {list(MARKET_CONFIG)}")

        mcfg = MARKET_CONFIG[self.market]
        self.confirm = confirm if confirm is not None else mcfg["confirm"]

        # Build strategy config
        self.strat_cfg = StrategyConfig(
            lookback=mcfg["lookback"],
            min_gap_pct=mcfg["min_gap_pct"],
            stop_type="session_extreme",
            stop_mult=mcfg["stop_mult"],
            target_type="atr",
            exit_atr_mult=mcfg["exit_atr_mult"],
            confirm=self.confirm,
            time_exit_bars=mcfg["time_exit_bars"],
        )
        self.strategy = IntradayRetracement(self.strat_cfg)

        self.signals_path = self.signals_dir / mcfg["signals_file"]
        self.bars: dict[str, pd.DataFrame] = {}   # symbol → rolling bar buffer
        self.file_cursors: dict[str, int] = {}    # symbol → byte offset
        self._stop = Event()

        self.signals_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "[%s] SignalGenerator initialised | confirm=%s | live=%s | out=%s",
            self.market, self.confirm, self.live_file, self.signals_path,
        )

    # ── public ─────────────────────────────────────────────────────────────

    def start(self):
        """Run the signal generation loop. Blocks until stop() is called."""
        self._stop.clear()
        self._load_initial_bars()

        poll = Thread(target=self._poll_loop, daemon=True, name=f"poll-{self.market}")
        poll.start()

        logger.info("[%s] Signal generation started (pid=%d)", self.market, os.getpid())
        self._stop.wait()
        poll.join(timeout=10)
        logger.info("[%s] Signal generation stopped", self.market)

    def stop(self):
        self._stop.set()

    # ── internal ────────────────────────────────────────────────────────────

    def _load_initial_bars(self):
        """Read every line in live.jsonl to build the full session bar buffer."""
        if not self.live_file.exists():
            logger.warning("[%s] live.jsonl not found — waiting for data feeder", self.market)
            return

        logger.info("[%s] Loading initial bars from %s", self.market, self.live_file)
        n_lines = 0
        with open(self.live_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                self._process_bar_line(line)
                n_lines += 1
        logger.info("[%s] Loaded %d initial bars", self.market, n_lines)

    def _poll_loop(self):
        while not self._stop.is_set():
            try:
                self._read_new_lines()
                self._generate_signals()
            except Exception as e:
                logger.error("[%s] Poll error: %s", self.market, e, exc_info=True)

            if self._stop.wait(timeout=self.poll_seconds):
                break   # stop flag was set

    def _read_new_lines(self):
        """Append any new lines since last read."""
        if not self.live_file.exists():
            return
        with open(self.live_file) as fh:
            fh.seek(self.file_cursors.get("global", 0))
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                self._process_bar_line(line)
            self.file_cursors["global"] = fh.tell()

    def _process_bar_line(self, line: str):
        """Parse a JSON bar line and append to the per-symbol bar buffer."""
        try:
            bar = json.loads(line)
        except json.JSONDecodeError:
            return

        sym = bar.get("symbol")
        if not sym:
            return

        # Parse timestamp → index
        ts_str = bar.get("timestamp", "")
        try:
            ts = pd.Timestamp(ts_str)
        except Exception:
            return

        row = {
            "open":   float(bar["open"]),
            "high":   float(bar["high"]),
            "low":    float(bar["low"]),
            "close":  float(bar["close"]),
            "volume": float(bar.get("volume", 0)),
        }

        if sym not in self.bars:
            self.bars[sym] = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = self.bars[sym]
        df.loc[ts] = row
        # Keep rolling window
        if len(df) > self.N_BARS_BUFFER:
            df = df.iloc[-self.N_BARS_BUFFER:]
        self.bars[sym] = df

    def _generate_signals(self):
        """
        For each symbol with a complete session bar buffer, run the strategy
        and write any new signals to signals.json.

        Signal deduplication: we only emit a signal for a symbol if we don't
        already have an active signal for the current session date.
        """
        new_signals: list[dict] = []
        now = pd.Timestamp.now()

        for sym, bars_df in list(self.bars.items()):
            if bars_df.empty or len(bars_df) < self.strat_cfg.lookback + 2:
                continue

            try:
                sigs = IntradayRetracement.compute_signals(
                    {sym: bars_df}, self.strat_cfg
                )
            except Exception as e:
                logger.warning("[%s] Signal gen failed for %s: %s", self.market, sym, e)
                continue

            if sym not in sigs or sigs[sym].empty:
                continue

            sig_df = sigs[sym]
            # Deduplicate: only the latest signal per session date
            if not sig_df.index.inferred_type == "datetime64":
                sig_df.index = pd.to_datetime(sig_df.index)

            sig_df = sig_df.sort_index()
            latest = sig_df.iloc[-1]

            # Skip if we already wrote a signal for this session today
            sig_date = latest.name.date() if hasattr(latest.name, "date") else None
            if sig_date and self._already_signalled_today(sym, sig_date):
                continue

            # Skip stale signals (bar is > 10 bars old)
            bar_age = len(bars_df) - (bars_df.index.get_loc(latest.name) if latest.name in bars_df.index else 0) - 1
            if bar_age > 10:
                continue

            signal_dict = {
                "symbol":        sym,
                "timestamp":     str(latest.name),
                "direction":     latest["direction"],
                "entry_price":   float(latest["entry_price"]),
                "stop_price":   float(latest["stop_price"]),
                "target_price": float(latest["target_price"]),
                "atr":          float(latest["atr"]),
                "session_high": float(latest["session_high"]),
                "session_low":  float(latest["session_low"]),
                "market":       self.market,
                "generated_at": str(now),
            }
            new_signals.append(signal_dict)
            logger.info(
                "[%s] SIGNAL %s %s @ %.4f  stop=%.4f  target=%.4f  atr=%.4f",
                self.market, sym, latest["direction"],
                latest["entry_price"], latest["stop_price"],
                latest["target_price"], latest["atr"],
            )

        if new_signals:
            self._write_signals(new_signals)

    def _already_signalled_today(self, sym: str, sig_date) -> bool:
        """Return True if signals.json already has a signal for this sym+date."""
        if not self.signals_path.exists():
            return False
        try:
            data = json.loads(self.signals_path.read_text())
        except Exception:
            return False
        for s in data.get("signals", []):
            if s.get("symbol") == sym:
                s_ts = pd.Timestamp(s.get("timestamp"))
                if s_ts.date() == sig_date:
                    return True
        return False

    def _write_signals(self, new_signals: list[dict]):
        """Append new signals to signals.json."""
        if self.signals_path.exists():
            try:
                data = json.loads(self.signals_path.read_text())
            except Exception:
                data = {"generated_at": "", "signals": []}
        else:
            data = {"generated_at": "", "signals": []}

        data["generated_at"] = str(pd.Timestamp.now())
        data["signals"].extend(new_signals)

        # Write atomically
        tmp = self.signals_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.signals_path)
        logger.info("[%s] Wrote %d signal(s) → %s", self.market, len(new_signals), self.signals_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(description="CONFIRM_TIGHT signal generation service")
    parser.add_argument(
        "--market", default="US",
        choices=["US", "UK"],
        help="'US' = S&P 500 (unconfirmed), 'UK' = FTSE 100 (CONFIRM_TIGHT proven)",
    )
    parser.add_argument(
        "--live-file", default="live.jsonl",
        help="Path to incoming live bar data (default: live.jsonl)",
    )
    parser.add_argument(
        "--signals-dir", default="signals",
        help="Directory for output signals.json files (default: signals/)",
    )
    parser.add_argument(
        "--poll-seconds", type=int, default=SignalGenerator.POLL_DEFAULT,
        help="Poll interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--confirm", type=lambda x: x.lower() in ("1", "true", "yes"),
        default=None,
        help="Override confirm filter. Default: US=False, UK=True.",
    )
    args = parser.parse_args()

    gen = SignalGenerator(
        live_file=args.live_file,
        signals_dir=args.signals_dir,
        market=args.market,
        poll_seconds=args.poll_seconds,
        confirm=args.confirm,
    )

    def _sig_handler(sig, frame):
        logger.info("Received signal %d — shutting down", sig)
        gen.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    gen.start()


if __name__ == "__main__":
    _main()

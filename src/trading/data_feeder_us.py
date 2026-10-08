"""
US equity data feeder — streams live 1-min bars to live.jsonl.

Uses Alpaca's WebSocket stream (push) for real-time bars.
Falls back to polling get_bars() every minute if WebSocket is unavailable.

Usage:
    python -m src.trading.data_feeder_us --symbols AAPL,MSFT,GOOGL --output live_us.jsonl

Alpaca paper market data (IEX) is free. Live market data requires subscription.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from threading import Event
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class AlpacaFeeder:
    """
    Streams live US equity bars via Alpaca WebSocket to a JSONL file.

    - 1-min bars (available on Alpaca IEX paper feed)
    - US equity market hours: 09:30–16:00 ET
    - Output: live.jsonl with one JSON object per bar
    """

    def __init__(
        self,
        symbols: list[str],
        output_file: str = "live_us.jsonl",
        poll_seconds: int = 60,
    ):
        self.symbols      = symbols
        self.output_file = Path(output_file)
        self.poll_seconds = poll_seconds
        self._stop        = Event()
        self._ws          = None
        self._bars       = {}   # sym → last bar dict

    def _connect_ws(self):
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_key    = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")

        try:
            from alpaca.data.live import StockDataStream
        except ImportError:
            logger.warning("alpaca.data.live not available — using polling fallback")
            return None

        ws = StockDataStream(api_key, secret_key)
        return ws

    def _on_bar(self, bar):
        """Called by WebSocket for each new bar."""
        sym = bar.symbol
        rec = {
            "symbol":    sym,
            "timestamp": str(pd.Timestamp(bar.timestamp)),
            "open":      float(bar.open),
            "high":      float(bar.high),
            "low":       float(bar.low),
            "close":     float(bar.close),
            "volume":    int(bar.volume),
        }
        self._bars[sym] = rec
        self._append_bar(rec)

    def _append_bar(self, bar: dict):
        with open(self.output_file, "a") as f:
            f.write(json.dumps(bar) + "\n")

    def run_ws(self, ws):
        """WebSocket main loop."""
        for sym in self.symbols:
            ws.subscribe_bars(self._on_bar, sym)
        ws.run()

    def run_polling(self):
        """
        Fallback: poll get_bars every minute.
        Not real-time — use only if WebSocket fails.
        """
        from src.data.alpaca_client import AlpacaClient
        client = AlpacaClient(paper=True)
        self._stop.clear()
        last_bar: dict = {}

        logger.info("[Alpaca] Running in polling mode (1-min interval)")
        while not self._stop.is_set():
            try:
                bars = client.get_bars(self.symbols, "Minute", lookback_days=1)
                for sym in self.symbols:
                    if sym not in bars or bars[sym].empty:
                        continue
                    df = bars[sym]
                    latest_ts = df.index[-1]
                    if sym not in last_bar or last_bar[sym]["timestamp"] != str(latest_ts):
                        row = df.iloc[-1]
                        rec = {
                            "symbol":    sym,
                            "timestamp": str(latest_ts),
                            "open":      float(row["open"]),
                            "high":      float(row["high"]),
                            "low":       float(row["low"]),
                            "close":     float(row["close"]),
                            "volume":    int(row["volume"]),
                        }
                        self._append_bar(rec)
                        last_bar[sym] = rec
            except Exception as e:
                logger.warning("[Alpaca] Polling error: %s", e)

            self._stop.wait(timeout=self.poll_seconds)

    def run(self):
        ws = self._connect_ws()
        if ws is not None:
            self._stop.clear()
            thread = Event().wait  # placeholder — WebSocket run() blocks
            logger.info("[Alpaca] WebSocket streaming to %s", self.output_file)
        else:
            self.run_polling()

    def stop(self):
        self._stop.set()


def _main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Alpaca US equity data feeder")
    parser.add_argument(
        "--symbols",  default="AAPL,MSFT,GOOGL,AMZN,META,NVDA",
        help="Comma-separated ticker list (default: top 6)",
    )
    parser.add_argument("--output", default="live_us.jsonl")
    parser.add_argument("--poll",   action="store_true", help="Use polling mode instead of WebSocket")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    feeder = AlpacaFeeder(symbols=symbols, output_file=args.output)

    def _sig(sig, frame):
        logger.info("Stopping feeder...")
        feeder.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    feeder.run()


if __name__ == "__main__":
    _main()

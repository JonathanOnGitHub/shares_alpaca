"""
Data feeder for FTSE 100 intraday trading.

Connects to IBKR via ib_insync and streams 5-min bars to live.jsonl.
Bars are written in real-time: one JSON line per bar, closed bars only.

Usage:
    python -m src.trading.data_feeder_uk --universe ftse100 --output live_ftse100.jsonl

IBKR paper trading: port 7497, clientId=1
IBKR live trading:  port 7496, clientId=1

Requirements:
    pip install ib_insync alpaca-py python-dotenv pandas
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FTSE 100 components
# ─────────────────────────────────────────────────────────────────────────────

FTSE100_SYMBOLS = [
    "AAL", "ABF", "ADM", "AHT", "AML", "ANG", "ANTO", "AAPL", "AV.", "AZN",
    "BA.", "BARC", "BATS", "BDEV", "BEZ", "BKG", "BLND", "BHP", "BWY", "BYG",
    "CAP", "CCL", "CEY", "CNA", "COL", "CONT", "CRH", "CRPR", "CWK", "DCC",
    "DGE", "DLG", "DPH", "EMG", "ENOG", "EVR", "EXPN", "EZJ", "FERG", "FLTR",
    "FRAS", "FRES", "GLEN", "GRG", "HL", "HIK", "HMRC", "HSBA", "HST", "HTG",
    "HUM", "ICP", "IHG", "III", "IMB", "IMS", "INCH", "INF", "INTU", "ISF",
    "ITRK", "JD", "JG", "JMAT", "KGF", "LAND", "LGEN", "LLOY", "LSEG", "MAB",
    "MKS", "MOON", "MRN", "MTH", "NAC", "NEX", "NWG", "NXT", "ODP", "OMU",
    "PAYC", "PBH", "PRU", "PSN", "PSON", "RDSA", "RDSB", "REDD", "REL", "RKT",
    "RIO", "RLE", "RMV", "RR.", "RTO", "SBRY", "SDR", "SGE", "SGRO", "SHEL",
    "SL", "SMIN", "SPX", "SSE", "SSON", "STAN", "SVT", "TQH", "TSCO", "TW.",
    "ULVR", "UNIQ", "UTG", "UU.", "VOD", "WEIR", "WKP", "WPP", "XPP",
]

# Fallback to major liquid names if IBKR contract resolution fails
UK100_MAJORS = [
    "HSBA.L", "SHEL.L", "AZN.L", "ULVR.L", "RIO.L", "BP.L", "GSK.L",
    "REL.L", "DGE.L", "NG.L", "LSEG.L", "BATS.L", "ABF.L", "PRU.L",
    "VOD.L", "BT-A.L", "EXPN.L", "AAL.L", "LLOY.L", "BARC.L",
]


class IBKRFeeder:
    """
    Connects to IBKR and streams 5-min bars for UK stocks to a JSONL file.

    Market hours: 08:00–16:30 UKT (LSE).
    Bar frequency: 5 minutes → 102 bars per session.
    """

    BAR_FREQUENCY  = "5 mins"
    MARKET_OPEN    = "08:00"
    MARKET_CLOSE   = "16:30"
    PAPER_PORT     = 7497
    LIVE_PORT      = 7496

    def __init__(
        self,
        output_file: str = "live_ftse100.jsonl",
        universe: str = "majors",
        client_id: int = 1,
        paper: bool = True,
    ):
        self.output_file = Path(output_file)
        self.universe    = universe
        self.client_id   = client_id
        self.paper       = paper
        self._stop       = Event()
        self._bars       = {}   # symbol → DataFrame of today's bars

        self._ib = None

    def _connect(self):
        from ib_insync import IB
        self._ib = IB()
        port = self.PAPER_PORT if self.paper else self.LIVE_PORT
        self._ib.connect("127.0.0.1", port, clientId=self.client_id)
        logger.info("[IBKR] Connected to %s port %d", "paper" if self.paper else "live", port)

    def _get_symbols(self):
        if self.universe == "ftse100":
            return FTSE100_SYMBOLS
        elif self.universe == "majors":
            return UK100_MAJORS
        return [self.universe]

    def _make_contract(self, symbol: str):
        from ib_insync import Stock
        # Strip .L suffix if present
        sym = symbol.replace(".L", "")
        return Stock(sym, exchange="LSE", currency="GBP")

    def _wait_for_market_open(self):
        """Sleep until 08:00 UKT."""
        now = pd.Timestamp.now("Europe/London")
        market_open = now.normalize() + pd.Timedelta(hours=8)
        if now >= market_open:
            # Already past open — stream immediately
            return
        delay = (market_open - now).total_seconds()
        logger.info("[UK] Market opens in %.0f minutes — waiting", delay / 60)
        time.sleep(max(0, delay))

    def _stream_bars(self):
        """
        Request 5-min bars for each symbol from market open.
        Each completed bar is appended to live.jsonl.
        """
        symbols = self._get_symbols()
        contracts = {}
        bars_collected = 0

        for sym in symbols:
            try:
                contract = self._make_contract(sym)
                self._ib.qualifyContract(contract)
                contracts[sym] = contract
            except Exception as e:
                logger.warning("[IBKR] Could not qualify %s: %s", sym, e)

        if not contracts:
            logger.error("[IBKR] No contracts qualified — aborting")
            return

        logger.info("[IBKR] Streaming 5-min bars for %d symbols", len(contracts))

        # Request live bars
        def on_bar_update(bars, has_new_bar):
            if not has_new_bar:
                return
            bar = bars[-1]  # most recent closed bar
            sym = bar.contract.symbol
            rec = {
                "symbol":    sym,
                "timestamp": str(pd.Timestamp(bar.date)),
                "open":      float(bar.open),
                "high":      float(bar.high),
                "low":       float(bar.low),
                "close":     float(bar.close),
                "volume":    int(bar.volume),
            }
            self._append_bar(rec)
            logger.debug("[BAR] %s %s O=%.4f H=%.4f L=%.4f C=%.4f",
                         rec["symbol"], rec["timestamp"],
                         rec["open"], rec["high"], rec["low"], rec["close"])

        for sym, contract in contracts.items():
            self._ib.reqRealTimeBars(
                contract, barType=5,  # 5 = 5-min bars
                whatToShow="TRADES", useRTH=True,
                callback=on_bar_update,
            )

        # Keep thread alive until stopped
        self._stop.wait()
        for contract in contracts.values():
            self._ib.cancelRealTimeBars(contract)

    def _append_bar(self, bar: dict):
        with open(self.output_file, "a") as f:
            f.write(json.dumps(bar) + "\n")

    def run(self):
        self._connect()
        self._stop.clear()
        self._wait_for_market_open()
        self._stream_bars()

    def stop(self):
        self._stop.set()


def _main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="IBKR FTSE 100 data feeder")
    parser.add_argument("--output",    default="live_ftse100.jsonl")
    parser.add_argument("--universe",  default="majors", choices=["ftse100", "majors"])
    parser.add_argument("--client-id",  type=int, default=1)
    parser.add_argument("--live",       action="store_true", help="Use live trading port (default: paper)")
    args = parser.parse_args()

    feeder = IBKRFeeder(
        output_file=args.output,
        universe=args.universe,
        client_id=args.client_id,
        paper=not args.live,
    )

    def _sig(sig, frame):
        logger.info("Stopping feeder...")
        feeder.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    feeder.run()


if __name__ == "__main__":
    _main()

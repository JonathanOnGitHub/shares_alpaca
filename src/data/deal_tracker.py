"""
Merger deal outcome tracker.

Merger arb's risk isn't generic market risk — it's deal-break risk: most
of the time the spread between market price and offer price closes
quietly as the deal completes, but occasionally a deal breaks and the
target's price craters back toward its pre-announcement level. You can't
backtest or size this strategy honestly without knowing, for each
announced deal, whether it actually closed or fell apart, and how long
that took.

This module:
  1. Records deals detected by EDGARClient's merger-detection (see edgar.py)
  2. Periodically rescans each target's subsequent 8-Ks for completion or
     termination language
  3. Persists everything to a JSON ledger so expensive/rate-limited EDGAR
     scanning doesn't have to be redone, and so the backtest has a stable
     dataset to run against.

Usage:
    client = EDGARClient()
    tracker = DealTracker(client)

    # After discovering a merger filing via client.get_8k_filings(...):
    tracker.add_deal(cik=cik, ticker=ticker, filing=filing_result, announce_date=date)

    # Periodically (e.g. daily cron), update outcomes for open deals:
    tracker.update_pending()

    # For backtesting:
    completed = [d for d in tracker.deals if d.status == "completed"]
    terminated = [d for d in tracker.deals if d.status == "terminated"]
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.data.edgar import EDGARClient

# High-precision phrases only, same philosophy as merger detection in
# edgar.py — generic words like "completed" or "terminated" alone are far
# too common in unrelated filings to use as signals.
COMPLETION_PHRASES = [
    "COMPLETION OF ACQUISITION OR DISPOSITION OF ASSETS",  # standard Item 2.01 caption
    "MERGER HAS BEEN COMPLETED",
    "MERGER WAS COMPLETED",
    "CONSUMMATED THE MERGER",
    "CONSUMMATION OF THE MERGER",
    "EFFECTIVE TIME OF THE MERGER",
    "MERGER BECAME EFFECTIVE",
    "MERGER HAS BECOME EFFECTIVE",
]

TERMINATION_PHRASES = [
    "TERMINATED THE MERGER AGREEMENT",
    "TERMINATION OF THE MERGER AGREEMENT",
    "MUTUALLY AGREED TO TERMINATE",
    "MERGER AGREEMENT HAS BEEN TERMINATED",
    "WILL NOT PROCEED WITH THE PROPOSED MERGER",
    "WILL NOT PROCEED WITH THE MERGER",
    "ABANDONED THE PROPOSED MERGER",
]


@dataclass
class Deal:
    cik: str
    ticker: str
    announce_date: str
    source_url: str
    target_hint: Optional[str] = None
    offer_price: Optional[float] = None
    confidence: str = "low"
    status: str = "pending"  # pending | completed | terminated | stale
    outcome_date: Optional[str] = None
    outcome_url: Optional[str] = None
    outcome_phrase: Optional[str] = None
    last_checked: Optional[str] = None

    def days_to_outcome(self) -> Optional[int]:
        if not self.outcome_date:
            return None
        a = datetime.strptime(self.announce_date, "%Y-%m-%d")
        o = datetime.strptime(self.outcome_date, "%Y-%m-%d")
        return (o - a).days


class DealTracker:
    def __init__(self, client: Optional[EDGARClient] = None, store_path: Optional[str] = None):
        self.client = client or EDGARClient()
        self.store_path = Path(store_path) if store_path else (
            Path(__file__).resolve().parent / "deals_ledger.json"
        )
        self.deals: list[Deal] = []
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            raw = json.loads(self.store_path.read_text())
            self.deals = [Deal(**d) for d in raw]

    def _save(self) -> None:
        self.store_path.write_text(json.dumps([asdict(d) for d in self.deals], indent=2))

    def add_deal(self, cik: str, ticker: str, filing: dict, announce_date: str) -> Optional[Deal]:
        """Add a newly-detected merger filing as a tracked deal, avoiding
        duplicates (same CIK announced within 30 days is treated as the
        same deal being re-filed/amended, not a new one)."""
        for d in self.deals:
            if d.cik == cik:
                existing = datetime.strptime(d.announce_date, "%Y-%m-%d")
                new = datetime.strptime(announce_date, "%Y-%m-%d")
                if abs((new - existing).days) <= 30:
                    return None  # duplicate/amendment of an existing tracked deal

        deal = Deal(
            cik=cik,
            ticker=ticker,
            announce_date=announce_date,
            source_url=filing.get("url", ""),
            target_hint=filing.get("target_hint"),
            offer_price=filing.get("offer_price"),
            confidence=filing.get("confidence", "low"),
        )
        self.deals.append(deal)
        self._save()
        return deal

    def check_outcome(self, deal: Deal, max_lookforward_days: int = 270) -> Deal:
        """Scan 8-Ks filed by this CIK after the announce date for
        completion/termination language. Marks 'stale' if no outcome is
        found within max_lookforward_days, so it stops being rechecked
        forever but is still visible as an unresolved case."""
        if deal.status != "pending":
            return deal

        quarters_needed = max_lookforward_days // 90 + 2
        rows = self.client.list_8k_filings_by_cik(
            deal.cik, start_date=deal.announce_date, quarters_back=quarters_needed
        )

        for row in rows:
            if row.get("date", "") <= deal.announce_date:
                continue  # skip the announcement filing itself
            html = self.client._get(row["href"])
            if not html:
                continue
            text = self.client._html_to_text(html).upper()

            hit = next((p for p in COMPLETION_PHRASES if p in text), None)
            if hit:
                deal.status = "completed"
                deal.outcome_date = row.get("date")
                deal.outcome_url = row["href"]
                deal.outcome_phrase = hit
                break

            hit = next((p for p in TERMINATION_PHRASES if p in text), None)
            if hit:
                deal.status = "terminated"
                deal.outcome_date = row.get("date")
                deal.outcome_url = row["href"]
                deal.outcome_phrase = hit
                break

        deal.last_checked = datetime.now().strftime("%Y-%m-%d")

        if deal.status == "pending":
            announce = datetime.strptime(deal.announce_date, "%Y-%m-%d")
            if datetime.now() - announce > timedelta(days=max_lookforward_days):
                deal.status = "stale"

        return deal

    def update_pending(self, max_lookforward_days: int = 270) -> dict:
        """Re-check every pending deal. Returns a summary of status changes
        so this can be run as a daily/weekly job and logged."""
        before = {d.cik: d.status for d in self.deals}
        for deal in self.deals:
            if deal.status == "pending":
                self.check_outcome(deal, max_lookforward_days)
        self._save()

        changed = [
            (d.cik, d.ticker, before[d.cik], d.status)
            for d in self.deals if before.get(d.cik) != d.status
        ]
        return {"checked": sum(1 for d in self.deals if d.cik in before), "changed": changed}

    def summary(self) -> dict:
        statuses = {}
        for d in self.deals:
            statuses[d.status] = statuses.get(d.status, 0) + 1
        completed = [d for d in self.deals if d.status == "completed" and d.days_to_outcome()]
        avg_days = (
            sum(d.days_to_outcome() for d in completed) / len(completed) if completed else None
        )
        return {
            "total_deals": len(self.deals),
            "by_status": statuses,
            "completion_rate": (
                statuses.get("completed", 0) /
                max(statuses.get("completed", 0) + statuses.get("terminated", 0), 1)
            ),
            "avg_days_to_completion": avg_days,
        }


if __name__ == "__main__":
    # Smoke test with synthetic filing text, no network required.
    class _FakeClient(EDGARClient):
        def __init__(self, filings_by_cik):
            super().__init__()
            self._filings_by_cik = filings_by_cik

        def list_8k_filings_by_cik(self, cik, start_date, quarters_back=8):
            return self._filings_by_cik.get(cik, [])

        def _get(self, url):
            return self._html_by_url.get(url)

    completion_html = """<html><body><p>Item 2.01 Completion of Acquisition
    or Disposition of Assets. On July 15, 2026, the Merger became effective
    and the merger was completed.</p></body></html>"""
    termination_html = """<html><body><p>On July 15, 2026, the parties
    mutually agreed to terminate the Merger Agreement, dated March 1, 2026,
    following failure to obtain regulatory approval.</p></body></html>"""

    fake = _FakeClient({
        "0000000001": [{"date": "2026-07-15", "href": "http://fake/complete.htm"}],
        "0000000002": [{"date": "2026-07-15", "href": "http://fake/terminate.htm"}],
    })
    fake._html_by_url = {
        "http://fake/complete.htm": completion_html,
        "http://fake/terminate.htm": termination_html,
    }

    tracker = DealTracker(client=fake, store_path="/tmp/test_deals_ledger.json")
    tracker.deals = []  # reset for repeatable smoke test
    tracker.add_deal("0000000001", "TGTA", {"url": "x", "offer_price": 42.5, "confidence": "high"}, "2026-03-01")
    tracker.add_deal("0000000002", "TGTB", {"url": "y", "offer_price": 15.0, "confidence": "high"}, "2026-03-01")

    tracker.update_pending()
    for d in tracker.deals:
        print(d.ticker, d.status, d.outcome_date, "|", d.outcome_phrase, "| days:", d.days_to_outcome())
    print(tracker.summary())

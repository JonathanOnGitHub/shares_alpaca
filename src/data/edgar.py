"""SEC EDGAR 8-K filing fetcher for merger announcements.
Rate-limited to comply with SEC requirements (10 req/s max).
"""
import hashlib
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)

SEC_HEADERS = {
    "User-Agent": "SharesAlpaca/1.0 (research project; jonathan.c.burley@gmail.com)",
    "Accept-Encoding": "gzip, deflate",
}
SEC_BASE = "https://www.sec.gov"
SEC_ARCHIVE = f"{SEC_BASE}/Archives"
INDEX_URL = f"{SEC_ARCHIVE}/edgar/full-index"


class EDGARClient:
    def __init__(self):
        self._last_request = 0.0
        self.cache_dir = Path(__file__).resolve().parent / ".edgar_cache"
        self.cache_dir.mkdir(exist_ok=True)

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < 0.15:  # ~7 req/s, under the 10 req/s limit
            time.sleep(0.15 - elapsed)
        self._last_request = time.time()

    def _get(self, url: str, max_age_seconds: Optional[int] = None) -> Optional[str]:
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_path = self.cache_dir / cache_key
        if cache_path.is_file():
            if max_age_seconds is None or (time.time() - cache_path.stat().st_mtime) < max_age_seconds:
                return cache_path.read_text(encoding="utf-8")

        self._rate_limit()
        try:
            r = requests.get(url, headers=SEC_HEADERS, timeout=30)
            if r.status_code == 200:
                cache_path.write_text(r.text, encoding="utf-8")
                return r.text
            logger.warning("EDGAR HTTP %d for %s", r.status_code, url)
            return None
        except Exception as e:
            logger.warning("EDGAR request failed: %s", e)
            return None

    def get_8k_filings(self, ticker: str, days: int = 365, quarters_back: int = 8) -> list[dict]:
        """Get parsed 8-K filings for a ticker over the given lookback period."""
        cik = self._ticker_to_cik(ticker)
        if not cik:
            return []

        now = datetime.now()
        start = now - timedelta(days=days)
        rows = self.list_8k_filings_by_cik(cik, start.strftime("%Y-%m-%d"), quarters_back)

        filings = []
        for row in rows:
            filing = self._parse_filing(row["href"])
            if filing:
                filing["date"] = row.get("date", "")
                filings.append(filing)
        return filings

    def list_8k_filings_by_cik(self, cik: str, start_date: str, quarters_back: int = 8) -> list[dict]:
        """Return raw 8-K filing rows (date, href) for a CIK since start_date,
        without running merger-detection parsing on each — used both by
        get_8k_filings (discovery) and by outcome tracking (which applies
        its own completion/termination phrase scan instead)."""
        match_cik = cik.lstrip("0")
        now = datetime.now()
        rows_out = []

        for q in range(quarters_back):
            total_months_back = q * 3
            target_month = now.month - total_months_back
            yr = now.year
            while target_month < 1:
                target_month += 12
                yr -= 1
            qtr_num = (target_month - 1) // 3 + 1
            if yr < 2020 or yr > now.year:
                break
            idx = self._get_quarterly_index(yr, qtr_num)
            if not idx:
                continue
            for row in idx:
                row_cik = row.get("cik", "").strip().lstrip("0")
                if (row.get("form") in ("8-K", "8-K/A")
                        and match_cik == row_cik
                        and row.get("date", "") >= start_date):
                    rows_out.append(row)

        return sorted(rows_out, key=lambda r: r.get("date", ""))

    def _load_cik_map(self) -> dict[str, str]:
        """Return {ticker: CIK} map, cached locally from SEC company_tickers.json."""
        cache_file = Path(__file__).resolve().parent / "cik_lookup.json"
        if cache_file.exists():
            import json
            return json.loads(cache_file.read_text())

        url = f"{SEC_BASE}/files/company_tickers.json"
        data = self._get(url)
        if data:
            import json
            lookup = {}
            for entry in json.loads(data).values():
                lookup[entry["ticker"]] = str(entry["cik_str"]).zfill(10)
            cache_file.write_text(json.dumps(lookup))
            return lookup
        return {}

    def _ticker_to_cik(self, ticker: str) -> Optional[str]:
        return self._load_cik_map().get(ticker.upper())

    def get_8k_filings_bulk(
        self, tickers: list[str], days: int = 365, quarters_back: int = 8
    ) -> dict[str, list[dict]]:
        """Scan 8-K filings for many tickers at once, loading each quarterly
        index only once instead of once per ticker.

        This is the method merger arb should use — define a broad universe
        of potential targets and scan them all efficiently, rather than
        guessing which single ticker will announce a deal.

        Returns {ticker: [parsed_filing, ...]} for tickers that had any
        8-Ks in the period (not just merger-related ones — each filing's
        ``is_merger`` field tells you whether deal language was detected).
        """
        lookup = self._load_cik_map()
        cik_to_ticker: dict[str, str] = {}
        ticker_to_cik: dict[str, str] = {}
        for t in tickers:
            cik = lookup.get(t.upper())
            if cik:
                stripped = cik.lstrip("0")
                cik_to_ticker[stripped] = t.upper()
                ticker_to_cik[t.upper()] = stripped

        if not ticker_to_cik:
            return {}

        now = datetime.now()
        start = now - timedelta(days=days)
        start_date = start.strftime("%Y-%m-%d")
        target_ciks = set(cik_to_ticker.keys())

        rows_by_ticker: dict[str, list[dict]] = {}
        for q in range(quarters_back):
            total_months_back = q * 3
            target_month = now.month - total_months_back
            yr = now.year
            while target_month < 1:
                target_month += 12
                yr -= 1
            qtr_num = (target_month - 1) // 3 + 1
            if yr < 2020 or yr > now.year:
                break
            idx = self._get_quarterly_index(yr, qtr_num)
            if not idx:
                continue
            for row in idx:
                row_cik = row.get("cik", "").strip().lstrip("0")
                if (row.get("form") not in ("8-K", "8-K/A")
                        or row_cik not in target_ciks
                        or row.get("date", "") < start_date):
                    continue
                ticker = cik_to_ticker[row_cik]
                rows_by_ticker.setdefault(ticker, []).append(row)

        for ticker in rows_by_ticker:
            rows_by_ticker[ticker].sort(key=lambda r: r.get("date", ""))

        result: dict[str, list[dict]] = {}
        for ticker, rows in rows_by_ticker.items():
            filings = []
            for row in rows:
                filing = self._parse_filing(row["href"])
                if filing:
                    filing["date"] = row.get("date", "")
                    filings.append(filing)
            if filings:
                result[ticker] = filings

        return result

    def _load_cik_reverse_map(self) -> dict[str, str]:
        """Return {stripped_CIK: ticker} map for resolving tickers from
        index rows during full-text discovery."""
        lookup = self._load_cik_map()
        return {cik.lstrip("0"): ticker for ticker, cik in lookup.items()}

    def discover_merger_deals(
        self, days: int = 365 * 2, quarters_back: int = 8,
    ) -> list[dict]:
        """Scan every 8-K in the EDGAR index for merger language, without
        filtering by ticker — catches targets a watchlist would miss.

        Expensive on first run (rate-limited parsing of thousands of
        filings per quarter), but the disk cache makes repeat runs cheap.
        After the scan, feed results into ``DealTracker.add_deal()``.

        Returns list of merger filings, each enriched with ``ticker``,
        ``cik``, and ``date`` alongside the standard ``_parse_filing`` keys.
        """
        cik_to_ticker = self._load_cik_reverse_map()

        now = datetime.now()
        start = now - timedelta(days=days)
        start_date = start.strftime("%Y-%m-%d")

        found: list[dict] = []
        for q in range(quarters_back):
            total_months_back = q * 3
            target_month = now.month - total_months_back
            yr = now.year
            while target_month < 1:
                target_month += 12
                yr -= 1
            qtr_num = (target_month - 1) // 3 + 1
            if yr < 2020 or yr > now.year:
                break

            idx = self._get_quarterly_index(yr, qtr_num)
            if not idx:
                continue

            qtr_label = f"{yr} Q{qtr_num}"
            eight_ks = 0
            for row in idx:
                if row.get("form") not in ("8-K", "8-K/A"):
                    continue
                if row.get("date", "") < start_date:
                    continue
                eight_ks += 1
                filing = self._parse_filing(row["href"])
                if filing and filing.get("is_merger"):
                    row_cik = row.get("cik", "").strip().lstrip("0")
                    filing["ticker"] = cik_to_ticker.get(row_cik)
                    filing["cik"] = row_cik
                    filing["date"] = row.get("date", "")
                    found.append(filing)

            logger.info(
                "discover_merger_deals %s: scanned %d 8-Ks, %d merger hits (%d total)",
                qtr_label, eight_ks, sum(1 for f in found if f.get("date", "").startswith(f"{yr}-")),
                len(found),
            )

        return found

    def _get_quarterly_index(self, year: int, qtr: int) -> list[dict]:
        url = f"{INDEX_URL}/{year}/QTR{qtr}/form.idx"

        now = datetime.now()
        current_qtr = (now.month - 1) // 3 + 1
        is_current_quarter = (year == now.year and qtr == current_qtr)
        # Closed quarters are immutable — cache forever. The in-progress
        # quarter gains new filings daily, so cap its cache age; otherwise
        # deal outcome tracking would silently miss anything filed after
        # the first time this quarter was fetched.
        max_age = 6 * 3600 if is_current_quarter else None

        text = self._get(url, max_age_seconds=max_age)
        if not text:
            return []

        rows = []
        in_data = False
        for line in text.split("\n"):
            if "--------" in line:
                in_data = True
                continue
            if not in_data or not line.strip():
                continue
            # Fixed-width: form(0-12) name(12-79) cik(79-91) date(91-101) filename(101+)
            form = line[0:12].strip() if len(line) > 12 else ""
            name = line[12:79].strip() if len(line) > 79 else ""
            cik = line[79:91].strip() if len(line) > 91 else ""
            date = line[91:101].strip() if len(line) > 101 else ""
            fname = line[101:].strip() if len(line) > 101 else ""
            if form:
                rows.append({
                    "cik": cik,
                    "name": name,
                    "form": form,
                    "date": date,
                    "href": f"{SEC_ARCHIVE}/{fname}" if fname else "",
                })
        return rows

    def _parse_filing(self, url: str) -> Optional[dict]:
        html = self._get(url)
        if not html:
            return None

        text = self._html_to_text(html)
        upper = text.upper()

        # High-precision phrases only. Bare "MERGER", "TENDER OFFER", or
        # "DEFINITIVE AGREEMENT" are too generic — they show up in routine
        # risk-factor boilerplate, unrelated debt/buyback filings, and
        # legal disclaimers on nearly every 8-K. Multi-word deal-specific
        # phrases are far less prone to false positives.
        strong_phrases = [
            "AGREEMENT AND PLAN OF MERGER",
            "DEFINITIVE MERGER AGREEMENT",
            "DEFINITIVE AGREEMENT TO ACQUIRE",
            "DEFINITIVE AGREEMENT TO BE ACQUIRED",
            "AGREEMENT AND PLAN OF ACQUISITION",
            "AGREED TO BE ACQUIRED BY",
            "TENDER OFFER TO PURCHASE ALL",
            "MERGER SUB",  # near-unambiguous: entity type only created for M&A
        ]
        matched = [p for p in strong_phrases if p in upper]

        # Require either 2+ distinct strong phrases, or 1 phrase repeated
        # multiple times (press-release-style filings restate deal terms
        # several times; a single incidental mention usually isn't a deal).
        match_counts = {p: upper.count(p) for p in matched}
        total_hits = sum(match_counts.values())
        is_merger = len(matched) >= 2 or total_hits >= 3

        result = {
            "url": url,
            "is_merger": is_merger,
            "confidence": "high" if len(matched) >= 2 else ("medium" if is_merger else "low"),
            "matched_phrases": matched,
            "text_snippet": text[:5000] if is_merger else "",
        }

        if is_merger:
            result["target_hint"] = self._extract_target_hint(upper)
            price, price_context = self._extract_offer_price(upper, matched)
            if price is not None:
                result["offer_price"] = price
                result["offer_price_context"] = price_context

        return result

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Strip HTML/XBRL markup so keyword and regex matching runs
        against readable prose, not tags/attributes (which can both hide
        real matches split across tags and create false positives inside
        boilerplate metadata)."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style"]):
                tag.decompose()
            return soup.get_text(separator=" ")
        except Exception:
            # Fall back to a crude tag strip if parsing fails
            return re.sub(r"<[^>]+>", " ", html)

    @staticmethod
    def _extract_target_hint(upper: str) -> Optional[str]:
        for pattern in [
            r"AGREEMENT AND PLAN OF MERGER[,\s]+DATED[^,]*,\s+(?:BY AND )?AMONG\s+([A-Z][A-Z\s.,&]+)",
            r"AGREED TO ACQUIRE\s+([A-Z][A-Z\s.,&]+?)(?:\s+FOR|\s+IN\s+A)",
            r"TO\s+(?:BE\s+)?ACQUIRE(?:D BY)?\s+([A-Z][A-Z\s.,&]+?)(?:\s+FOR|\.|,)",
        ]:
            m = re.search(pattern, upper)
            if m:
                return m.group(0)[:200]
        return None

    @staticmethod
    def _extract_offer_price(upper: str, matched_phrases: list[str]) -> tuple[Optional[float], Optional[str]]:
        """Only search for a per-share price within a window around a
        matched deal phrase, and require nearby deal-consideration context
        (cash/consideration/purchase price language), rather than taking
        the first '$X per share' anywhere in the filing — which could be
        an unrelated dividend, option strike, or fee."""
        window = 400
        candidates: dict[float, int] = {}
        context_by_price: dict[float, str] = {}

        price_patterns = [
            r"\$\s?(\d+\.\d{2})\s+(?:IN\s+CASH\s+)?PER\s+SHARE",
            r"\$\s?(\d+\.\d{2})\s+PER\s+SHARE\s+IN\s+CASH",
            r"PURCHASE\s+PRICE\s+OF\s+\$\s?(\d+\.\d{2})",
        ]

        anchors = [m.start() for phrase in matched_phrases for m in re.finditer(re.escape(phrase), upper)]
        for anchor in anchors:
            snippet = upper[max(0, anchor - window):anchor + window]
            for pat in price_patterns:
                for m in re.finditer(pat, snippet):
                    price = float(m.group(1))
                    candidates[price] = candidates.get(price, 0) + 1
                    context_by_price.setdefault(price, m.group(0))

        if not candidates:
            return None, None

        # Most frequently repeated value near deal language wins — deal
        # press releases typically restate the offer price consistently.
        best_price = max(candidates, key=candidates.get)
        return best_price, context_by_price[best_price]

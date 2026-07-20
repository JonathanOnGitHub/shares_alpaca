"""SEC EDGAR 8-K filing fetcher for merger announcements.
Rate-limited to comply with SEC requirements (10 req/s max).
"""
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
    "User-Agent": "SharesAlpaca/1.0 (research project; contact@example.com)",
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

    def _get(self, url: str) -> Optional[str]:
        self._rate_limit()
        try:
            r = requests.get(url, headers=SEC_HEADERS, timeout=30)
            if r.status_code == 200:
                return r.text
            logger.warning("EDGAR HTTP %d for %s", r.status_code, url)
            return None
        except Exception as e:
            logger.warning("EDGAR request failed: %s", e)
            return None

    def get_8k_filings(self, ticker: str, days: int = 365, quarters_back: int = 8) -> list[dict]:
        """Get 8-K filing URLs for a ticker over the given lookback period."""
        cik = self._ticker_to_cik(ticker)
        if not cik:
            return []

        filings = []
        match_cik = cik.lstrip("0")
        now = datetime.now()
        start = now - timedelta(days=days)

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
                        and row.get("date", "") >= start.strftime("%Y-%m-%d")):
                    filing = self._parse_filing(row["href"])
                    if filing:
                        filing["date"] = row.get("date", "")
                        filings.append(filing)

        return filings

    def _ticker_to_cik(self, ticker: str) -> Optional[str]:
        cache_file = Path(__file__).resolve().parent / "cik_lookup.json"
        if cache_file.exists():
            import json
            lookup = json.loads(cache_file.read_text())
            if ticker.upper() in lookup:
                return lookup[ticker.upper()]

        # Fetch CIK from SEC ticker map
        url = f"{SEC_BASE}/files/company_tickers.json"
        data = self._get(url)
        if data:
            import json
            lookup = {}
            for entry in json.loads(data).values():
                lookup[entry["ticker"]] = str(entry["cik_str"]).zfill(10)
            cache_file.write_text(json.dumps(lookup))
            return lookup.get(ticker.upper())
        return None

    def _get_quarterly_index(self, year: int, qtr: int) -> list[dict]:
        url = f"{INDEX_URL}/{year}/QTR{qtr}/form.idx"
        text = self._get(url)
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

        text = html.upper()

        # Detect merger-related content
        merger_keywords = [
            "MERGER AGREEMENT", "AGREEMENT AND PLAN OF MERGER",
            "ACQUISITION AGREEMENT", "DEFINITIVE AGREEMENT",
            "MERGER", "TENDER OFFER",
        ]
        is_merger = any(kw in text for kw in merger_keywords)

        result = {
            "url": url,
            "is_merger": is_merger,
            "text_snippet": html[:5000] if is_merger else "",
        }

        if is_merger:
            # Try to extract target company name
            for pattern in [
                r"PURSUANT TO THE MERGER AGREEMENT[^.]*",
                r"AGREEMENT AND PLAN OF MERGER[^.]*",
                r"ACQUIRE\s+([A-Z][A-Z\s.,]+)",
                r"MERGER\s+(?:WITH|OF)\s+([A-Z][A-Z\s.,]+)",
            ]:
                m = re.search(pattern, text)
                if m:
                    result["target_hint"] = m.group(0)[:200]
                    break

            # Try to extract per-share price
            price_patterns = [
                r"\$(\d+\.?\d*)\s+PER\s+SHARE",
                r"(\d+\.?\d*)\s+DOLLARS\s+PER\s+SHARE",
                r"PURCHASE\s+PRICE[^$]*\$(\d+\.?\d*)",
            ]
            for pat in price_patterns:
                m = re.search(pat, text)
                if m:
                    result["offer_price"] = float(m.group(1))
                    break

        return result

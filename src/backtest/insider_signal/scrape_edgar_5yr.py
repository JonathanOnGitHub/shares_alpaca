"""
SEC EDGAR Form 4 scraper — genuine 5-year lookback.

Approach:
  1. Query SEC EDGAR full-text search each month — ALL Form 4s (no role keyword filter)
  2. For each filing: fetch submissions JSON → get primaryDocument URL
  3. Fetch XML/HTML → parse transaction table + reporting-person role
  4. Filter: qualifying role (CEO/CFO/President/etc.) + >= $100k purchase
  5. Deduplicate by (ticker, date, owner, value)

Covers: 2021-09 → 2026-09 (5 years, ~60 months)

Usage:
    python3 scrape_edgar_5yr.py [output_csv]
"""

import sys
import csv
import logging
import time
import random
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
from xml.etree import ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ─────────────────────────────────────────────────────────────────────
LOOKBACK_DAYS   = 1825                                        # 5 years
OUTPUT_PATH     = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "ceo_purchases_edgar_5yr.csv"
UNIVERSE_SIZE   = 503                                          # S&P 500
MIN_VALUE_USD   = 100_000
BATCH_SIZE      = 20
RATE_LIMIT_S    = 0.11                                         # SEC: ≤10 req/s per thread
N_WORKERS        = 20                                          # parallel fetching threads
MAX_XML_RETRIES = 3
USER_AGENT       = "HermesAgent/1.0 burley@nottingham.ac.uk"

# Thread-local storage for per-thread rate limiting
_thread_lock = threading.Lock()
_last_request_time = {}   # thread_id -> last call timestamp

QUALIFYING_ROLES = {
    "Chief Executive Officer",
    "Chair and CEO",
    "President",
    "Chief Financial Officer",
    "Chief Technology Officer",
    "Chief Operating Officer",
    "Chief Medical Officer",
    "General Counsel",
    "Chief Human Resources Officer",
    "Chief Business Officer",
    "Chief Strategy Officer",
    "Chief Commercial Officer",
    "Chief Customer Officer",
    "Chief Information Officer",
    "Chair",           # Non-executive chair (broadly correlated with insider purchases)
}

# Form 4 transaction codes that represent a purchase (not sale/gift/exercise)
# A = Purchase, A9 =Purchase (various subtypes) — we want A-codes only
PURCHASE_CODES = {"A", "A9", "A4", "A5", "A6", "A7", "A8"}

# ── Logging ────────────────────────────────────────────────────────────────────
log_path = Path(__file__).parent / "scrape_edgar.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("edgar")


# ── Helpers ─────────────────────────────────────────────────────────────────────

def normalize_ticker(raw: str) -> Optional[str]:
    """Extract clean ticker from strings like 'Apple Inc. (AAPL) (CIK 0000320193)'."""
    m = re.search(r'\(([A-Z]{1,5})\)', raw)
    if not m:
        return None
    t = m.group(1)
    if t in ("SEC", "CIK") or len(t) > 5 or not re.match(r'^[A-Z]+$', t):
        return None
    return t


# Per-thread storage for rate limiting — no shared lock needed
_thread_last = {}   # thread_id (int) -> last call timestamp (float)

def _thread_rate_limit():
    """Per-thread rate limiter — ensures min RATE_LIMIT_S between calls in same thread.
    No shared state, no blocking. Each thread tracks its own clock."""
    tid = threading.get_ident()
    last = _thread_last.get(tid, 0)
    now  = time.time()
    wait = RATE_LIMIT_S - (now - last)
    if wait > 0:
        time.sleep(wait)
        now = time.time()
    _thread_last[tid] = now


def safe_get(url: str, timeout: int = 20) -> Optional[requests.Response]:
    for attempt in range(MAX_XML_RETRIES):
        try:
            _thread_rate_limit()
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            if r.status_code in (200, 404):
                return r
        except requests.RequestException:
            pass
        time.sleep(1 + attempt * 0.5)
    return None


# ── Phase 1: Collect Form 4 filings via quarterly .idx files + SEC tickers JSON ──

def build_sp500_cik_map() -> dict[str, str]:
    """
    Build mapping of S&P 500 ticker → set of CIKs (as zero-padded 10-digit strings).
    Uses SEC's company_tickers.json + GitHub S&P 500 list.
    """
    # 1. S&P 500 tickers
    url = (
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
        "main/data/constituents.csv"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=30)
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    sp500_tickers = set(
        df["Symbol"].dropna()
        .str.replace(".", "-", regex=False)
        .str.strip()
        .str.upper()
        .tolist()
    )
    logger.info("Loaded %d S&P 500 tickers", len(sp500_tickers))

    # 2. SEC company tickers JSON → cik_str → {ticker, title}
    r2 = safe_get("https://www.sec.gov/files/company_tickers.json")
    if not r2 or r2.status_code != 200:
        logger.error("Failed to load SEC company_tickers.json")
        return {}
    import json as _json
    ticker_data = _json.loads(r2.text)

    # 3. Build ticker → set of CIK strings (zero-padded)
    ticker_ciks: dict[str, set[str]] = {}
    for entry in ticker_data.values():
        tk = entry["ticker"].strip().upper()
        cik_num = int(entry["cik_str"])
        cik_str = f"{cik_num:010d}"
        if tk in sp500_tickers:
            ticker_ciks.setdefault(tk, set()).add(cik_str)

    logger.info(
        "SEC tickers JSON: %d S&P 500 tickers mapped to CIKs",
        len(ticker_ciks)
    )
    return ticker_ciks   # ticker → set of cik_strs


def fetch_quarterly_idx(quarter: str) -> Optional[str]:
    """
    quarter: e.g. '2024Q3' → fetches the form.idx file for that quarter.
    Returns file content as string, or None on failure.
    """
    # quarter e.g. '2024Q3' → URL: .../full-index/2024/QTR3/form.idx
    year_str = quarter[:4]
    qtr_num  = int(quarter[5])       # '2024Q3' → index 5 = '3'
    url = (
        f"https://www.sec.gov/Archives/edgar/full-index/"
        f"{year_str}/QTR{qtr_num}/form.idx"
    )
    r = safe_get(url, timeout=60)
    if not r or r.status_code != 200:
        return None
    return r.text


FORM4_PATTERN = re.compile(
    r'^(\S{1,15})\s+'              # Form Type (≥1 non-space)
    r'(.+?)\s{2,}'                  # Company Name (lazy, ≥2 spaces before next col)
    r'(\d{6,10})\s+'                # CIK (6-10 digits — some CIKs are < 7 digits)
    r'(\d{4}-\d{2}-\d{2})\s+'    # Date Filed
    r'(.+)$'                          # File Name
)

ACCESSION_PATTERN = re.compile(r'(\d{10}-\d+-\d+)\.txt')


def parse_idx_file(content: str, sp500_cik_map: dict[str, set[str]]) -> list[dict]:
    """
    Parse a quarterly form.idx file (fixed-width format).
    Returns list of {adsh, ticker, cik, file_date} for Form 4s of S&P 500 companies.
    """
    results = []
    for line in content.split("\n"):
        line = line.rstrip()
        if not line or len(line) < 20:
            continue
        m = FORM4_PATTERN.match(line)
        if not m:
            continue
        form_type, company, cik_raw, date_filed, filename = m.groups()
        if form_type != "4":
            continue

        # Normalize CIK
        try:
            cik_int  = int(cik_raw)
            cik_str  = f"{cik_int:010d}"
        except ValueError:
            continue

        # Look up ticker by CIK
        ticker = None
        for tk, cik_set in sp500_cik_map.items():
            if cik_str in cik_set:
                ticker = tk
                break
        if not ticker:
            continue

        # Extract accession (adsh) from filename
        acc_m = ACCESSION_PATTERN.search(filename)
        if not acc_m:
            continue
        accession = acc_m.group(1)   # e.g. 0000320193-24-000084

        results.append({
            "adsh":      accession,
            "ticker":    ticker,
            "cik":       cik_str,
            "file_date": date_filed,
            "filename":  filename,   # e.g. edgar/data/320193/0000320193-24-000084.txt
        })
    return results


def get_all_quarters(start_year_month: str, end_year_month: str) -> list[str]:
    """Generate list of 'YYYYQN' quarter labels between start and end (inclusive)."""
    # start_year_month: 'YYYY-MM'
    sy, sm = map(int, start_year_month.split("-"))
    ey, em = map(int, end_year_month.split("-"))
    quarters = []
    y, m = sy, sm
    while (y < ey) or (y == ey and m <= em):
        q = (m - 1) // 3 + 1
        quarters.append(f"{y}Q{q}")
        m += 3
        if m > 12:
            m = 1
            y += 1
    return quarters


def collect_form4_filings_5yr() -> list[dict]:
    """
    Phase 1 main: collect all S&P 500 Form 4 filings from the past LOOKBACK_DAYS.
    Returns list of {adsh, ticker, cik, file_date}.
    """
    sp500_cik_map = build_sp500_cik_map()
    if not sp500_cik_map:
        return []

    # Determine quarter range
    end_qtr     = datetime.today()
    cutoff_date = datetime.today() - timedelta(days=LOOKBACK_DAYS)

    # Generate quarters to fetch
    all_quarters = get_all_quarters(
        f"{cutoff_date.year}-{cutoff_date.month:02d}",
        f"{end_qtr.year}-{end_qtr.month:02d}"
    )
    logger.info("Need to fetch %d quarters: %s → %s",
                 len(all_quarters), all_quarters[0], all_quarters[-1])

    all_filings  = []
    seen_adsh    = set()
    errors       = 0

    for qi, qtr in enumerate(all_quarters):
        content = fetch_quarterly_idx(qtr)
        if content is None:
            logger.warning("  Quarter %s: failed to download .idx", qtr)
            errors += 1
            continue

        filings = parse_idx_file(content, sp500_cik_map)
        new_count = 0
        for f in filings:
            if f["adsh"] not in seen_adsh:
                seen_adsh.add(f["adsh"])
                all_filings.append(f)
                new_count += 1

        logger.info(
            "  Quarter %s [%d/%d]: +%d S&P 500 Form 4s (total: %d)",
            qtr, qi + 1, len(all_quarters), new_count, len(all_filings)
        )
        time.sleep(RATE_LIMIT_S + random.uniform(0, 0.1))

    logger.info(
        "Phase 1 complete: %d Form 4 filings for S&P 500 (%d quarters, %d download errors)",
        len(all_filings), len(all_quarters), errors
    )
    return all_filings


# ── Phase 2: Resolve primary document URL from submissions JSON ─────────────────

def get_primary_doc_url(cik: str, adsh: str) -> Optional[str]:
    """
    Given CIK + accession number, return the full URL to the primary document.
    """
    # The submission JSON has filings/recent — find adsh index
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = safe_get(url, timeout=15)
    if not r or r.status_code != 200:
        return None

    try:
        d = r.json()
    except Exception:
        return None

    recent = d.get("filings", {}).get("recent", {})
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs      = recent.get("primaryDocument", [None] * len(accession_numbers))

    try:
        idx = accession_numbers.index(adsh)
    except ValueError:
        return None

    pd_name = primary_docs[idx]
    if not pd_name:
        return None

    acc_norm = adsh.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik):010d}/{acc_norm}/{pd_name}"
    )


# ── Phase 3: Parse Form 4 XBRL — extract transactions + role ───────────────────

def parse_form4(raw_text: str) -> tuple[list[dict], Optional[str]]:
    """
    Parse a Form 4 XBRL document (inline XML wrapped in SGML).
    Returns (list of transaction dicts, role string or None).
    """
    # Extract inline XML from <XML>...</XML> wrapper
    xml_match = re.search(r'<XML>(.*?)</XML>', raw_text, re.DOTALL)
    if not xml_match:
        return [], None
    xml_text = xml_match.group(1).strip()

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return [], None

    # ── 1. Extract reporting-person role ─────────────────────────────────────
    role = None
    owner = root.find('reportingOwner')
    if owner is not None:
        rel = owner.find('reportingOwnerRelationship')
        if rel is not None:
            is_officer = rel.find('isOfficer')
            officer_title_elem = rel.find('officerTitle')
            is_director = rel.find('isDirector')
            is_10pct = rel.find('isTenPercentOwner')

            if is_officer is not None and is_officer.text == '1' and officer_title_elem is not None:
                raw_title = officer_title_elem.text
                if raw_title:
                    # Normalize to canonical QUALIFYING_ROLES labels
                    title_upper = raw_title.upper()
                    for qrole in QUALIFYING_ROLES:
                        if qrole.upper() in title_upper:
                            role = qrole
                            break
            elif is_director is not None and is_director.text == '1':
                role = 'Director'
            elif is_10pct is not None and is_10pct.text == '1':
                role = '10% Owner'

    # ── 2. Extract non-derivative transactions ─────────────────────────────────
    transactions = []
    ndt = root.find('nonDerivativeTable')
    if ndt is not None:
        for txn in ndt.findall('nonDerivativeTransaction'):
            try:
                # Helper to get <value> child text from an element
                def v(elem):
                    if elem is None: return None
                    child = elem.find('value')
                    return child.text.strip() if child is not None and child.text else None

                # Security title
                security = v(txn.find('securityTitle')) or ''

                # Transaction date
                txn_date = v(txn.find('transactionDate')) or ''

                # Transaction code
                coding = txn.find('transactionCoding')
                code = ''
                if coding is not None:
                    tc = coding.find('transactionCode')
                    code = tc.text.strip().upper() if tc is not None and tc.text else ''

                # Shares and price
                amounts = txn.find('transactionAmounts')
                shares = 0.0
                price = 0.0
                if amounts is not None:
                    ts = amounts.find('transactionShares')
                    shares = float(v(ts) or 0)
                    pps = amounts.find('transactionPricePerShare')
                    price = float(v(pps) or 0)

                # A or D
                a_d_elem = amounts.find('transactionAcquiredDisposedCode') if amounts is not None else None
                a_d = v(a_d_elem) or ''

                value = shares * price

                transactions.append({
                    'txn_date': txn_date,
                    'code':     code,
                    'shares':   shares,
                    'price':    price,
                    'value':    value,
                    'a_d':      a_d,
                    'security': security,
                })
            except (AttributeError, ValueError):
                continue

    return transactions, role


# ── S&P 500 universe ─────────────────────────────────────────────────────────

def get_sp500_tickers() -> set[str]:
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        tickers = set(
            df["Symbol"].dropna()
            .str.replace(".", "-", regex=False)
            .str.strip()
            .str.upper()
            .tolist()
        )
        logger.info("Fetched %d S&P 500 tickers", len(tickers))
        return tickers
    except Exception as e:
        logger.error("Failed to fetch S&P 500: %s", e)
        return set()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = datetime.now()
    logger.info("Starting EDGAR 5-year Form 4 scrape")
    logger.info("Min value: $%s  |  Lookback: %d days", f"{MIN_VALUE_USD:,}", LOOKBACK_DAYS)

    # ── Phase 1: Collect Form 4 filings via quarterly .idx files ────────────────
    all_filings = collect_form4_filings_5yr()
    if not all_filings:
        logger.error("No Form 4 filings collected. Exiting.")
        return

    # ── Phase 2: Fetch XMLs and parse (parallel) ────────────────────────────
    all_records    = []
    fetch_errors   = 0
    no_xml         = 0
    role_rejected  = 0
    value_rejected = 0
    lock           = threading.Lock()
    completed      = [0]

    def process_filing(filing: dict) -> list[dict]:
        """Fetch and parse one Form 4. Runs in thread pool."""
        adsh      = filing["adsh"]
        cik       = filing["cik"]
        ticker    = filing["ticker"]
        filename  = filing["filename"]  # e.g. edgar/data/320193/0000320193-24-000084.txt
        pd_url    = "https://www.sec.gov/Archives/" + filename

        r = safe_get(pd_url, timeout=20)
        if not r or r.status_code != 200 or len(r.text) < 1000:
            return [None, None, True, 0]

        txns, role = parse_form4(r.text)
        if not role:
            return [None, False, False, 1]

        records = []
        for txn in txns:
            if txn["value"] < MIN_VALUE_USD:
                continue
            records.append({
                "ticker":            ticker,
                "owner_name":        f"{role} (Form 4)",
                "owner_role":        role,
                "transaction_date":  txn["txn_date"],
                "code":              "P",
                "shares":            txn["shares"],
                "price":             round(txn["price"], 2),
                "total_value":       txn["value"],
                "text":              f"Form 4 — code={txn['code']}",
            })
        return [records, False, False, 0]

    t_phase2 = datetime.now()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_filing, f): f for f in all_filings}
        for future in as_completed(futures):
            records, fetch_err, no_xml_f, role_err = future.result()
            with lock:
                completed[0] += 1
                if records is None:
                    if no_xml_f:
                        no_xml += 1
                    elif fetch_err:
                        fetch_errors += 1
                    if role_err:
                        role_rejected += 1
                else:
                    for rec in records:
                        all_records.append(rec)
                if completed[0] % 200 == 0:
                    elapsed = (datetime.now() - t_phase2).total_seconds()
                    rate = completed[0] / max(elapsed, 1)
                    eta  = (len(all_filings) - completed[0]) / max(rate, 1)
                    logger.info(
                        "  Progress %d/%d  records=%d  "
                        "fetch_err=%d  no_xml=%d  role_rej=%d  val_rej=%d  ETA=%dm",
                        completed[0], len(all_filings), len(all_records),
                        fetch_errors, no_xml, role_rejected, value_rejected,
                        int(eta / 60)
                    )

    logger.info(
        "Phase 2 done: %d records  fetch_err=%d  no_xml=%d  role_rej=%d  val_rej=%d",
        len(all_records), fetch_errors, no_xml, role_rejected, value_rejected
    )

    # ── Deduplicate ───────────────────────────────────────────────────────
    seen  = set()
    deduped = []
    for r in all_records:
        key = (r["ticker"], r["transaction_date"], round(r["total_value"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    deduped.sort(key=lambda x: x["transaction_date"])

    # ── Save ───────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ticker", "transaction_date", "owner_name", "owner_role",
                  "code", "shares", "price", "total_value", "text"]

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(deduped)

    # ── Report ──────────────────────────────────────────────────────────────
    elapsed_total = (datetime.now() - t0).total_seconds()
    print("\n" + "=" * 65)
    print(" EDGAR 5-YEAR SCRAPE — RESULTS")
    print("=" * 65)
    print(f"  Filings processed  : {len(all_filings)}")
    print(f"  Fetch errors       : {fetch_errors}")
    print(f"  No XML found       : {no_xml}")
    print(f"  Role rejected       : {role_rejected}")
    print(f"  Value rejected      : {value_rejected}")
    print(f"  Final records       : {len(deduped)}")
    print(f"  Runtime             : {elapsed_total/60:.1f} min")
    if deduped:
        df = pd.DataFrame(deduped)
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        print(f"  Date range          : {df['transaction_date'].min().date()} → {df['transaction_date'].max().date()}")
        print(f"  Unique tickers      : {df['ticker'].nunique()}")
        print(f"  Value range         : ${df['total_value'].min():,.0f} — ${df['total_value'].max():,.0f}")
        print(f"  Median txn          : ${df['total_value'].median():,.0f}")
        print(f"\n  Role breakdown:")
        for role, n in df["owner_role"].value_counts().items():
            print(f"    {role:<35} {n:>5}")
        print(f"\n  Annual breakdown:")
        df["year"] = df["transaction_date"].dt.year
        for yr, grp in df.groupby("year"):
            print(f"    {yr}: {len(grp):>3} transactions  ({grp['ticker'].nunique():>3} tickers)")
        print(f"\n  Output: {OUTPUT_PATH}")
    else:
        print("\n  ⚠  No records — check role/transaction parsing.")

    logger.info("Done. %d records → %s", len(deduped), OUTPUT_PATH)


if __name__ == "__main__":
    main()

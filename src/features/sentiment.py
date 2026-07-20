"""Loughran-McDonald financial sentiment analyzer.
Used to classify the tone of SEC filings and press releases.
"""
import csv
import re
from pathlib import Path


class LMSentiment:
    def __init__(self, dictionary_path: str | None = None):
        if dictionary_path is None:
            dictionary_path = str(Path(__file__).resolve().parent.parent / "data" / "lm_dictionary.csv")
        self.positive: set[str] = set()
        self.negative: set[str] = set()
        self.uncertainty: set[str] = set()
        self.litigious: set[str] = set()
        self._load(dictionary_path)

    def _load(self, path: str):
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row.get("Word", "").strip().upper()
                if row.get("Positive") == "2009":
                    self.positive.add(word)
                if row.get("Negative") == "2009":
                    self.negative.add(word)
                if row.get("Uncertainty") == "2009":
                    self.uncertainty.add(word)
                if row.get("Litigious") == "2009":
                    self.litigious.add(word)

    def analyze(self, text: str) -> dict:
        """Analyze text sentiment using the LM dictionary.

        Returns:
            dict with word counts, proportions, and net sentiment score.
        """
        words = re.findall(r"[A-Za-z]+", text.upper())
        total = len(words)
        if total == 0:
            return {"positive": 0, "negative": 0, "net": 0.0, "uncertainty": 0, "litigious": 0}

        pos_count = sum(1 for w in words if w in self.positive)
        neg_count = sum(1 for w in words if w in self.negative)
        unc_count = sum(1 for w in words if w in self.uncertainty)
        lit_count = sum(1 for w in words if w in self.litigious)

        return {
            "positive": pos_count,
            "negative": neg_count,
            "net": (pos_count - neg_count) / total,
            "uncertainty": unc_count,
            "litigious": lit_count,
            "total_words": total,
        }

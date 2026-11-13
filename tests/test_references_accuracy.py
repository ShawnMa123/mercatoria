"""Reference-resolution accuracy on hand-checked articles (P1-5, ≥30 samples).

The annotations below were checked against the article text: each expected
target is the article the sentence actually points at. Samples that resolve to
nothing are negative controls (no reference in the text) or cross-part
references that must stay ``unresolved``.

This is an integration test: it needs the fetched corpus and a built index
(both are excluded from git), so it skips on a bare clone.
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (article_id, expected resolved article_ids)
SAMPLES: list[tuple[str, list[str]]] = [
    ("cn-customs-guarantee-2018:14", ["cn-customs-guarantee-2018:7"]),
    ("cn-rcep-origin-2021:4", ["cn-rcep-origin-2021:3"]),
    ("cn-dual-use-general-license-2009:23", ["cn-dual-use-general-license-2009:14"]),
    ("cn-goods-import-export-2024:68", ["cn-goods-import-export-2024:51"]),
    ("cn-nuclear-export-control-2006:17", ["cn-nuclear-export-control-2006:5"]),
    ("cn-dual-use-general-license-2009:9", ["cn-dual-use-general-license-2009:8"]),
    ("cn-independent-guarantee-2020:20", ["cn-independent-guarantee-2020:14"]),
    ("cn-independent-guarantee-2020:18", ["cn-independent-guarantee-2020:12"]),
    ("cn-customs-audit-2022:25", ["cn-customs-audit-2022:16"]),
    ("cn-customs-valuation-measures-2013:48", ["cn-customs-valuation-measures-2013:6"]),
    ("cn-tariff-law-2024:28", ["cn-tariff-law-2024:27"]),
    ("cn-import-export-tariff-2017:53", ["cn-import-export-tariff-2017:50", "cn-import-export-tariff-2017:52"]),
    ("cn-tariff-law-2024:14", ["cn-tariff-law-2024:12", "cn-tariff-law-2024:13"]),
    ("cn-customs-law-2021:58", ["cn-customs-law-2021:56", "cn-customs-law-2021:57"]),
    ("cn-customs-law-2021:83", ["cn-customs-law-2021:82"]),
    ("cn-customs-law-2021:51", ["cn-customs-law-2021:30"]),
    ("au-autonomous-sanctions-act-2011:sec-4", ["au-autonomous-sanctions-act-2011:sec-5"]),
    ("us-cfr-19-122:122.25", ["us-cfr-19-122:122.23", "us-cfr-19-122:122.24"]),
    ("us-cfr-19-151:151.74", ["us-cfr-19-151:151.71", "us-cfr-19-151:151.73"]),
    ("us-cfr-19-10:10.932", ["us-cfr-19-10:10.909"]),
    ("us-cfr-15-740:740.12", ["us-cfr-15-740:740.19"]),
    (
        "us-cfr-15-744:744.11",
        [
            "us-cfr-15-744:744.16",
            "us-cfr-15-744:744.8",
            "us-cfr-15-744:744.23",
            "us-cfr-15-744:744.15",
        ],
    ),
    ("us-cfr-19-182:182.45", ["us-cfr-19-182:182.44"]),
    ("eu-ucc-2013:art-52", ["eu-ucc-2013:art-33", "eu-ucc-2013:art-14"]),
    ("eu-ucc-2013:art-250", ["eu-ucc-2013:art-223"]),
    ("eu-ucc-2013:art-193", ["eu-ucc-2013:art-285"]),
    ("eu-dual-use-2021:art-16", ["eu-dual-use-2021:art-7", "eu-dual-use-2021:art-13", "eu-dual-use-2021:art-23"]),
    ("wto-dsu:art-17", ["wto-dsu:art-10", "wto-dsu:art-4"]),
    ("wto-tbt:art-15", ["wto-tbt:art-12"]),
    ("us-cfr-31-560:560.411", []),
    ("us-cfr-19-10:10.3027", []),
    ("us-cfr-31-560:560.106", []),
]


class ReferenceAccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from mercatoria.config import REPO_ROOT, load_config
        from mercatoria.store import open_current

        index_dir = load_config("config.example.json").resolved_index_dir()
        current = index_dir / "current.txt"
        if not current.is_file():
            raise unittest.SkipTest("no built index in data/index")
        cls.index = open_current(index_dir)

    def test_expected_targets_and_no_unexpected_resolution(self) -> None:
        wrong: list[str] = []
        for article_id, expected in SAMPLES:
            refs = self.index.article_references(article_id)
            got = [ref["article_id"] for ref in refs["resolved"]]
            if sorted(got) != sorted(expected):
                wrong.append(f"{article_id}: got {got}, expected {expected}")
        accuracy = 1 - len(wrong) / len(SAMPLES)
        self.assertGreaterEqual(
            accuracy, 0.9, "accuracy below 90%:\n" + "\n".join(wrong)
        )

    def test_references_are_deterministic(self) -> None:
        first = self.index.article_references("cn-customs-law-2021:58")
        self.index._references.pop("cn-customs-law-2021:58", None)
        second = self.index.article_references("cn-customs-law-2021:58")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

"""Converters for the EU/UK corpus fetchers.

The fetchers talk to the network; these tests pin the parsing contract with
synthetic inputs so a markup change fails here instead of silently producing a
document with zero articles.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "corpus"))
sys.path.insert(0, str(ROOT / "tools"))

import fetch_eu_uk  # noqa: E402
import check_corpus  # noqa: E402

EU_HTML = """<html><body>
<p class="oj-ti-section-1">CHAPTER 1</p>
<p class="oj-sti-art">Scope of the customs legislation</p>
<p class="oj-ti-art">Article 1</p>
<p class="oj-normal">1. This Regulation establishes the Union Customs Code.</p>
<p class="oj-ti-art">Article 2</p>
<p class="oj-normal">Delegation of <span class="oj-italic">power</span>.</p>
<p class="oj-note">(1) OJ L 269, 10.10.2013, p. 1.</p>
</body></html>"""

UK_XML = b"""<Legislation><Body><Part><P1group>
<Title>General</Title>
<Section><Number>1</Number><P1para><Text>Short title</Text></P1para></Section>
<Section><Number>2</Number><Title>Commencement</Title>
<P1para><Text>This Act comes into force on exit day.</Text></P1para></Section>
</P1group></Part></Body></Legislation>"""


class EUConverterTests(unittest.TestCase):
    def test_articles_and_chapters_become_headings(self) -> None:
        md, count = fetch_eu_uk.eu_to_markdown(EU_HTML, "Test Regulation")
        self.assertEqual(count, 2)
        self.assertIn("## Article 1", md)
        self.assertIn("### CHAPTER 1", md)
        self.assertIn("Scope of the customs legislation", md)
        self.assertIn("Delegation of power.", md)
        self.assertIn("<!-- document: Test Regulation -->", md)

    def test_notes_are_dropped(self) -> None:
        md, _ = fetch_eu_uk.eu_to_markdown(EU_HTML, "Test Regulation")
        self.assertNotIn("OJ L 269", md)

    def test_no_articles_detected(self) -> None:
        _, count = fetch_eu_uk.eu_to_markdown("<html><body><p>x</p></body></html>", "T")
        self.assertEqual(count, 0)


class UKConverterTests(unittest.TestCase):
    def test_sections_become_headings(self) -> None:
        md, count = fetch_eu_uk.uk_to_markdown(UK_XML, "Test Act")
        self.assertEqual(count, 2)
        self.assertIn("## Section 1", md)
        self.assertIn("## Section 2 — Commencement", md)
        self.assertIn("This Act comes into force on exit day.", md)
        self.assertNotIn("<Text>", md)


class CorpusCheckTests(unittest.TestCase):
    """`scripts/setup-corpus.sh --check` delegates here (P0-4)."""

    def _make(self, root: Path, skip: set[str] = frozenset()) -> None:
        for relative, want, _optional in check_corpus.SPECS:
            if relative in skip:
                continue
            directory = root / relative
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(want):
                (directory / f"doc-{index}.md").write_text("x", encoding="utf-8")

    def test_uk_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make(root, skip={"uk"})
            ok, lines = check_corpus.check(root)
            self.assertTrue(ok, lines)
            self.assertTrue(any("optional" in line for line in lines))
            self.assertEqual(check_corpus.main(["--corpus", tmp]), 0)

    def test_missing_required_source_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make(root, skip={"cn"})
            ok, lines = check_corpus.check(root)
            self.assertFalse(ok)
            self.assertTrue(any("cn" in line and "缺" in line for line in lines))
            self.assertEqual(check_corpus.main(["--corpus", tmp]), 1)


if __name__ == "__main__":
    unittest.main()

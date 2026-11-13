from __future__ import annotations

import unittest

from mercatoria.glossary import expand_query


class GlossaryTests(unittest.TestCase):
    def test_chinese_legal_terms_get_english(self) -> None:
        out = expand_query("GATT 国民待遇原则对国内税有什么要求？")
        self.assertIn("national treatment", out)
        self.assertIn("internal taxes", out)

    def test_english_query_is_unchanged(self) -> None:
        query = "What is the role of the Appellate Body?"
        self.assertEqual(expand_query(query), query)

    def test_unknown_chinese_query_is_unchanged(self) -> None:
        query = "今天天气怎么样？"
        self.assertEqual(expand_query(query), query)


if __name__ == "__main__":
    unittest.main()

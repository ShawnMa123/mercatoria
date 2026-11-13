from __future__ import annotations

import unittest

from mercatoria.tokenizer import STOPWORDS, query_terms


class QueryTermTests(unittest.TestCase):
    def test_regulatory_boilerplate_is_dropped(self) -> None:
        terms = query_terms("本法规定应当如何办理报关手续")
        self.assertNotIn("本法", terms)
        self.assertNotIn("应当", terms)
        self.assertIn("报关", terms)
        self.assertIn("手续", terms)

    def test_english_function_words_are_dropped(self) -> None:
        terms = query_terms("What shall the exporter declare under this regime?")
        self.assertNotIn("shall", terms)
        self.assertNotIn("under", terms)
        self.assertNotIn("this", terms)
        self.assertIn("exporter", terms)

    def test_all_stopword_query_falls_back_to_raw_tokens(self) -> None:
        terms = query_terms("应当 可以")
        self.assertTrue(terms)
        self.assertIn("应当", terms)

    def test_terms_are_deduplicated_in_order(self) -> None:
        terms = query_terms("customs valuation customs")
        self.assertEqual(terms.count("customs"), 1)
        self.assertEqual(terms[0], "customs")
        self.assertEqual(terms[1], "valuation")

    def test_stopword_set_matches_filtering(self) -> None:
        self.assertIn("根据", STOPWORDS)
        self.assertIn("such", STOPWORDS)
        self.assertNotIn("关税", STOPWORDS)


if __name__ == "__main__":
    unittest.main()

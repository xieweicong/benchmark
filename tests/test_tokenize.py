import unittest

from pii_benchmark.tokenize import build_sized_text, token_count, truncate_to_tokens


class TokenizeTests(unittest.TestCase):
    def test_token_count_handles_cjk_and_words(self):
        self.assertEqual(token_count("田中 test@example.com hello"), 4)

    def test_truncate_to_tokens(self):
        self.assertEqual(truncate_to_tokens("Alice Bob Carol", 2), "Alice Bob")

    def test_build_sized_text_reaches_target(self):
        text = build_sized_text("Alice alice@example.com", 5)
        self.assertEqual(token_count(text), 5)

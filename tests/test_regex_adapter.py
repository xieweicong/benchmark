import unittest

from pii_benchmark.models import RegexAdapter


class RegexAdapterTests(unittest.TestCase):
    def test_regex_adapter_redacts_common_pii(self):
        adapter = RegexAdapter({"name": "regex", "type": "regex"})
        output = adapter.redact(
            "Email jane@example.com, phone +1 (415) 555-0199, key sk-proj-abcdef123456."
        )
        self.assertNotIn("jane@example.com", output.text)
        self.assertNotIn("+1 (415) 555-0199", output.text)
        self.assertNotIn("sk-proj-abcdef123456", output.text)
        self.assertIn("[EMAIL_1]", output.text)

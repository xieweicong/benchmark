import unittest

from pii_benchmark.models import _apply_spans, _fallback_spans_from_decoded_tokens, _spans_from_bioes_offsets


class FakeTokenizer:
    def decode(self, token_ids):
        mapping = {
            1: "Alice",
            2: " Smith",
            3: " alice@example.com",
        }
        return "".join(mapping[token_id] for token_id in token_ids)


class MLXAdapterHelpersTests(unittest.TestCase):
    def test_spans_from_bioes_offsets_merges_bioes_tokens(self):
        offsets = [(0, 5), (6, 11), (16, 33)]
        labels = ["B-private_person", "E-private_person", "S-private_email"]

        spans = _spans_from_bioes_offsets(offsets, labels)

        self.assertEqual(
            spans,
            [
                type(spans[0])(label="private_person", start=0, end=11),
                type(spans[0])(label="private_email", start=16, end=33),
            ],
        )

    def test_fallback_spans_decodes_grouped_tokens(self):
        text = "Alice Smith <alice@example.com>"
        token_ids = [1, 2, 3]
        labels = ["B-private_person", "E-private_person", "S-private_email"]

        spans = _fallback_spans_from_decoded_tokens(text, token_ids, labels, tokenizer=FakeTokenizer())

        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0].label, "private_person")
        self.assertEqual(text[spans[0].start:spans[0].end], "Alice Smith")
        self.assertEqual(spans[1].label, "private_email")
        self.assertEqual(text[spans[1].start:spans[1].end], "alice@example.com")

    def test_apply_spans_uses_consistent_placeholders(self):
        text = "Alice Smith <alice@example.com>"
        spans = _spans_from_bioes_offsets(
            [(0, 5), (6, 11), (13, 30)],
            ["B-private_person", "E-private_person", "S-private_email"],
        )

        redacted = _apply_spans(text, spans)

        self.assertEqual(redacted, "[PII_PRIVATE_PERSON_1] <[PII_PRIVATE_EMAIL_1]>")

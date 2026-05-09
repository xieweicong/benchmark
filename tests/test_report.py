import tempfile
import unittest
from pathlib import Path

from pii_benchmark.report import load_rows, write_markdown


class ReportTests(unittest.TestCase):
    def test_markdown_report_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "run.jsonl"
            input_path.write_text(
                '{"kind":"run_metadata","run_id":"abc","created_at":"now","hardware":{"git":{}}}\n'
                '{"kind":"warmup","run_id":"abc","model":"regex","model_type":"regex",'
                '"repeats":1,"load":{"load_s":0.01},'
                '"measurements":[{"latency_s":0.3,"opf_model_forward_s":0.2,'
                '"opf_windows":1}],'
                '"aggregate":{"latency_s_p50":0.3,"input_tps_mean":500,'
                '"opf_model_forward_s_mean":0.2,"opf_windows_mean":1}}\n'
                '{"kind":"speed","run_id":"abc","model":"regex","model_type":"regex",'
                '"bucket_tokens":128,"successful_repeats":1,"repeats":1,'
                '"measurements":[{"latency_s":0.1,"opf_model_forward_s":0.04,'
                '"opf_tokenize_s":0.01,"opf_windows":1}],'
                '"aggregate":{"latency_s_p50":0.1,"input_tps_mean":1000}}\n'
                '{"kind":"speed","run_id":"abc","model":"regex","model_type":"regex",'
                '"bucket_tokens":256,"successful_repeats":1,"repeats":1,'
                '"measurements":[{"latency_s":0.2,"opf_model_forward_s":0.08,'
                '"opf_tokenize_s":0.02,"opf_windows":1}],'
                '"aggregate":{"latency_s_p50":0.2,"input_tps_mean":1000}}\n'
                '{"kind":"quality","run_id":"abc","model":"regex","model_type":"regex",'
                '"sample_count":1,"successful_samples":1,"recall":0.5,"pii_hit":1,"pii_total":2,'
                '"anchor_keep":0,"anchor_total":1,"anchor_keep_rate":0,"latency_s":0.1,'
                '"errors":[],"sample_scores":[{"sample_id":"s1","missed_pii":["x"],'
                '"changed_anchors":["y"]}]}\n',
                encoding="utf-8",
            )
            rows = load_rows([input_path])
            out = tmp_path / "report.md"
            write_markdown(rows, out)
            text = out.read_text(encoding="utf-8")
            self.assertIn("PII Redaction Benchmark Report", text)
            self.assertIn("Warmup", text)
            self.assertIn("Stage Breakdown", text)
            self.assertIn("Scaling Estimate", text)
            self.assertIn("Quality Details", text)

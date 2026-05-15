"""Model adapters for PII redaction benchmark tasks."""

from __future__ import annotations

import gc
import math
import os
import platform
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tokenize import token_count


PROMPT_TEMPLATE = (
    "You are a privacy filter. Replace every PII occurrence in the text below "
    "with a labelled placeholder, keep everything else unchanged. Use:\n"
    "- person name -> [PERSON]\n"
    "- email -> [EMAIL]\n"
    "- phone -> [PHONE]\n"
    "- address -> [ADDRESS]\n"
    "- account/card/iban/id-number -> [ACCOUNT]\n"
    "- api key/token/password/secret -> [SECRET]\n"
    "- date -> [DATE]\n"
    "- url -> [URL]\n"
    "Do NOT paraphrase. Output ONLY the rewritten text.\n\n"
    "TEXT:\n{text}\n\nREWRITTEN:\n"
)

PII_CATEGORIES = {
    "secret",
    "account_number",
    "private_email",
    "private_phone",
    "private_person",
    "private_address",
    "private_date",
    "private_url",
}


@dataclass
class RedactionOutput:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpanMatch:
    label: str
    start: int
    end: int


class ModelAdapter:
    def __init__(self, config: dict[str, Any], device: str = "auto") -> None:
        self.config = config
        self.name = str(config.get("name") or config.get("model_id") or config.get("type"))
        self.type = str(config.get("type"))
        self.device = device

    def load(self) -> dict[str, Any]:
        return {"load_s": 0.0}

    def close(self) -> None:
        gc.collect()

    def redact(self, text: str) -> RedactionOutput:
        raise NotImplementedError

    def speed_once(self, text: str, decode_new_tokens: int) -> dict[str, Any]:
        started = time.perf_counter()
        output = self.redact(text)
        latency = time.perf_counter() - started
        input_tokens = output.input_tokens or token_count(text)
        return {
            "latency_s": latency,
            "input_tokens": input_tokens,
            "input_tps": input_tokens / latency if latency > 0 else None,
            "output_tokens": output.output_tokens,
            **output.metrics,
        }


class RegexAdapter(ModelAdapter):
    PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        (
            "SECRET",
            re.compile(
                r"\b(?:sk-[A-Za-z0-9_-]{8,}|sk-proj-[A-Za-z0-9_-]{8,}|(?:AKIA|ASIA)[A-Z0-9]{16})\b"
            ),
        ),
        (
            "SECRET",
            re.compile(
                r"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\b\s*[:=]\s*[^\s\"']{4,}",
                re.IGNORECASE,
            ),
        ),
        ("EMAIL", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
        ("URL", re.compile(r"\bhttps?://[^\s<>\"]+", re.IGNORECASE)),
        (
            "PHONE",
            re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,5}\d{2,4}"),
        ),
        ("ACCOUNT", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
        ("ACCOUNT", re.compile(r"\b\d[\d -]{11,}\d\b")),
        ("DATE", re.compile(r"\b\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?\b")),
        ("ADDRESS", re.compile(r"\d{1,4}(?:丁目|番地|番|号)")),
        (
            "ADDRESS",
            re.compile(
                r"\b\d{1,6}\s+[A-Za-z0-9 .'-]+(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|boulevard|blvd|way|court|ct)\b",
                re.IGNORECASE,
            ),
        ),
    )

    def redact(self, text: str) -> RedactionOutput:
        redacted = text
        counts: dict[str, int] = {}
        for label, pattern in self.PATTERNS:
            def replace(match: re.Match[str]) -> str:
                counts[label] = counts.get(label, 0) + 1
                return f"[{label}_{counts[label]}]"

            redacted = pattern.sub(replace, redacted)
        return RedactionOutput(
            text=redacted,
            input_tokens=token_count(text),
            output_tokens=token_count(redacted),
            metrics={"matches": sum(counts.values())},
        )


class OPFAdapter(ModelAdapter):
    def __init__(self, config: dict[str, Any], device: str = "auto") -> None:
        super().__init__(config, device)
        self._opf: Any = None
        self._torch: Any = None
        self.resolved_device = "cpu"
        self.categories = set(config.get("categories") or PII_CATEGORIES)

    def _pick_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch  # type: ignore
        except Exception:
            return "cpu"
        self._torch = torch
        if torch.cuda.is_available():
            _raise_if_cuda_unsupported(torch)
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            from opf._api import OPF  # type: ignore
        except Exception as exc:
            raise RuntimeError("OPF adapter requires `pip install .[opf]`.") from exc

        device = self._pick_device()
        self.resolved_device = device
        if device == "mps":
            os.environ.setdefault("OPF_MOE_TRITON", "0")
        checkpoint = self.config.get("checkpoint")
        if checkpoint:
            os.environ["OPF_CHECKPOINT"] = str(checkpoint)
        elif os.environ.get("PII_BENCH_OPF_AUTO_DOWNLOAD", "1").lower() not in {"0", "false"}:
            checkpoint = _ensure_opf_checkpoint()
            os.environ["OPF_CHECKPOINT"] = str(checkpoint)
        self._opf = OPF(
            device=device,
            output_mode="typed",
            decode_mode=str(self.config.get("decode_mode", "viterbi")),
            trim_whitespace=True,
        )
        return {"load_s": time.perf_counter() - started, "device": device}

    def redact(self, text: str) -> RedactionOutput:
        if self._opf is None:
            self.load()
        try:
            return self._redact_timed(text)
        except (ImportError, AttributeError):
            return self._redact_public(text)

    def _redact_public(self, text: str) -> RedactionOutput:
        started = time.perf_counter()
        result = self._opf.redact(text)
        if isinstance(result, str):
            redacted = result
            spans = []
        else:
            spans = [
                span for span in getattr(result, "detected_spans", [])
                if getattr(span, "label", None) in self.categories
            ]
            redacted = _apply_spans(text, spans)
        latency = time.perf_counter() - started
        return RedactionOutput(
            text=redacted,
            input_tokens=token_count(text),
            output_tokens=token_count(redacted),
            metrics={"latency_s": latency, "spans": len(spans)},
        )

    def _sync(self) -> None:
        if self._torch is None:
            return
        if self.resolved_device == "cuda" and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
        elif (
            self.resolved_device == "mps"
            and getattr(self._torch, "mps", None) is not None
        ):
            self._torch.mps.synchronize()

    def _redact_timed(self, text: str) -> RedactionOutput:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
        from opf._core import runtime as opf_runtime  # type: ignore

        if self._torch is None:
            self._torch = torch

        metrics: dict[str, Any] = {
            "opf_component_s": 0.0,
            "opf_tokenize_s": 0.0,
            "opf_window_prepare_s": 0.0,
            "opf_model_forward_s": 0.0,
            "opf_logprob_transfer_s": 0.0,
            "opf_aggregation_s": 0.0,
            "opf_decode_s": 0.0,
            "opf_span_postprocess_s": 0.0,
            "opf_redaction_s": 0.0,
            "opf_windows": 0,
            "opf_window_tokens": 0,
        }

        with torch.inference_mode():
            self._sync()
            total_started = time.perf_counter()

            stage_started = time.perf_counter()
            runtime, decoder = self._opf.get_prediction_components()
            self._sync()
            metrics["opf_component_s"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            token_ids = tuple(
                int(tok) for tok in runtime.encoding.encode(text, allowed_special="all")
            )
            metrics["opf_tokenize_s"] = time.perf_counter() - stage_started

            if not token_ids:
                metrics["latency_s"] = time.perf_counter() - total_started
                metrics["spans"] = 0
                return RedactionOutput(text=text, input_tokens=0, output_tokens=0, metrics=metrics)

            example_id = "benchmark-example"
            background = int(runtime.label_info.background_token_label)
            example = opf_runtime.TokenizedExample(
                tokens=token_ids,
                labels=tuple(background for _ in token_ids),
                example_id=example_id,
                text=text,
            )
            aggregation = opf_runtime.ExampleAggregation(
                logprob_logsumexp=[],
                counts=[],
                labels=[],
                token_ids=[],
            )

            for window in opf_runtime.example_to_windows(example, runtime.n_ctx):
                if not window.tokens:
                    continue
                metrics["opf_windows"] += 1
                metrics["opf_window_tokens"] += len(window.tokens)

                stage_started = time.perf_counter()
                window_tokens = torch.tensor(
                    [list(window.tokens)],
                    device=runtime.device,
                    dtype=torch.int32,
                )
                attention_mask = torch.ones_like(window_tokens, dtype=torch.bool)
                self._sync()
                metrics["opf_window_prepare_s"] += time.perf_counter() - stage_started

                self._sync()
                stage_started = time.perf_counter()
                logits = runtime.model(window_tokens, attention_mask=attention_mask)
                self._sync()
                metrics["opf_model_forward_s"] += time.perf_counter() - stage_started

                stage_started = time.perf_counter()
                log_probs = F.log_softmax(logits.float(), dim=-1)[0].cpu()
                self._sync()
                metrics["opf_logprob_transfer_s"] += time.perf_counter() - stage_started
                if log_probs.shape[0] != len(window.tokens):
                    raise ValueError("Logprob output length does not match window length")

                stage_started = time.perf_counter()
                for token_pos, is_valid in enumerate(window.mask):
                    if not bool(is_valid):
                        continue
                    token_idx = int(window.offsets[token_pos])
                    if token_idx < 0:
                        continue
                    aggregation.ensure_capacity(token_idx)
                    score_vec = log_probs[token_pos]
                    existing = aggregation.logprob_logsumexp[token_idx]
                    if existing is None:
                        aggregation.logprob_logsumexp[token_idx] = score_vec.clone()
                    else:
                        aggregation.logprob_logsumexp[token_idx] = torch.logaddexp(
                            existing,
                            score_vec,
                        )
                    aggregation.counts[token_idx] += 1
                    aggregation.record_token_id(
                        token_idx,
                        int(window.tokens[token_pos]),
                        example_id,
                    )
                    aggregation.length = max(aggregation.length, token_idx + 1)
                metrics["opf_aggregation_s"] += time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            token_positions: list[int] = []
            token_score_vectors: list[Any] = []
            for token_idx in range(aggregation.length):
                if token_idx >= len(aggregation.logprob_logsumexp):
                    continue
                score_sum = aggregation.logprob_logsumexp[token_idx]
                count = aggregation.counts[token_idx]
                if score_sum is None or count <= 0:
                    continue
                avg_logprob = score_sum - math.log(float(count))
                token_positions.append(token_idx)
                token_score_vectors.append(avg_logprob)

            if not token_score_vectors:
                metrics["opf_decode_s"] = time.perf_counter() - stage_started
                metrics["latency_s"] = time.perf_counter() - total_started
                metrics["spans"] = 0
                return RedactionOutput(
                    text=text,
                    input_tokens=len(token_ids),
                    output_tokens=token_count(text),
                    metrics=metrics,
                )

            stacked_scores = torch.stack(token_score_vectors, dim=0)
            if decoder is not None:
                decoded_labels = decoder.decode(stacked_scores)
                if len(decoded_labels) != len(token_positions):
                    decoded_labels = stacked_scores.argmax(dim=1).tolist()
            else:
                decoded_labels = stacked_scores.argmax(dim=1).tolist()
            predicted_labels_by_index = {
                token_idx: int(label)
                for token_idx, label in zip(token_positions, decoded_labels)
            }
            predicted_token_spans = opf_runtime.labels_to_spans(
                predicted_labels_by_index,
                runtime.label_info,
            )
            metrics["opf_decode_s"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            decoded_text, char_starts, char_ends = opf_runtime.decode_text_with_offsets(
                token_ids,
                runtime.encoding,
            )
            decoded_mismatch = decoded_text != text
            source_text = decoded_text if decoded_mismatch else text
            predicted_char_spans = opf_runtime.token_spans_to_char_spans(
                predicted_token_spans,
                char_starts,
                char_ends,
            )
            if runtime.trim_span_whitespace:
                predicted_char_spans = opf_runtime.trim_char_spans_whitespace(
                    predicted_char_spans,
                    source_text,
                )
            if runtime.discard_overlapping_predicted_spans:
                predicted_char_spans = opf_runtime.discard_overlapping_spans_by_label(
                    predicted_char_spans,
                )

            detected: list[Any] = []
            for label_idx, start, end in predicted_char_spans:
                if not (0 <= start < end <= len(source_text)):
                    continue
                label = (
                    str(runtime.label_info.span_class_names[label_idx])
                    if 0 <= int(label_idx) < len(runtime.label_info.span_class_names)
                    else f"label_{label_idx}"
                )
                detected.append(
                    opf_runtime.DetectedSpan(
                        label=label,
                        start=int(start),
                        end=int(end),
                        text=source_text[start:end],
                        placeholder=opf_runtime._label_placeholder(label),
                    )
                )
            display_spans = opf_runtime._apply_output_mode_to_detected_spans(
                opf_runtime._select_non_overlapping_spans(detected),
                output_mode=runtime.output_mode,
            )
            filtered_spans = [
                span for span in display_spans
                if getattr(span, "label", None) in self.categories
            ]
            metrics["opf_span_postprocess_s"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            redacted = _apply_spans(source_text, filtered_spans)
            metrics["opf_redaction_s"] = time.perf_counter() - stage_started
            metrics["latency_s"] = time.perf_counter() - total_started
            metrics["spans"] = len(filtered_spans)
            metrics["opf_decoded_mismatch"] = decoded_mismatch

        return RedactionOutput(
            text=redacted,
            input_tokens=len(token_ids),
            output_tokens=token_count(redacted),
            metrics=metrics,
        )


class HFCausalLMAdapter(ModelAdapter):
    def __init__(self, config: dict[str, Any], device: str = "auto") -> None:
        super().__init__(config, device)
        self.model_id = str(config["model_id"])
        self.prompt_template = str(config.get("prompt_template") or PROMPT_TEMPLATE)
        self.tok: Any = None
        self.model: Any = None
        self.torch: Any = None
        self.resolved_device = "cpu"

    def _pick_device(self) -> str:
        if self.device != "auto":
            return self.device
        if self.torch.cuda.is_available():
            return "cuda"
        if (
            getattr(self.torch.backends, "mps", None) is not None
            and self.torch.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"

    def _dtype(self) -> Any:
        dtype = str(self.config.get("torch_dtype", "auto"))
        if dtype == "auto":
            return self.torch.float16 if self.resolved_device in {"cuda", "mps"} else self.torch.float32
        return getattr(self.torch, dtype)

    def load(self) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:
            raise RuntimeError("HF causal LM adapter requires `pip install .[hf]`.") from exc

        self.torch = torch
        self.resolved_device = self._pick_device()
        if self.resolved_device == "cuda":
            _raise_if_cuda_unsupported(self.torch)
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        kwargs: dict[str, Any] = {
            "torch_dtype": self._dtype(),
            "low_cpu_mem_usage": True,
        }
        device_map = self.config.get("device_map")
        if device_map:
            kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if not device_map:
            self.model = self.model.to(self.resolved_device)
        self.model.eval()
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token_id = self.tok.eos_token_id
        return {"load_s": time.perf_counter() - started, "device": self.resolved_device}

    def close(self) -> None:
        self.model = None
        self.tok = None
        gc.collect()
        if self.torch is not None and self.resolved_device == "cuda" and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        if (
            self.torch is not None
            and self.resolved_device == "mps"
            and getattr(self.torch, "mps", None) is not None
        ):
            self.torch.mps.empty_cache()

    def _sync(self) -> None:
        if self.resolved_device == "cuda":
            self.torch.cuda.synchronize()
        elif self.resolved_device == "mps":
            self.torch.mps.synchronize()

    def _encode_prompt(self, text: str) -> Any:
        prompt = self.prompt_template.format(text=text)
        encoded = self.tok(prompt, return_tensors="pt")
        if not self.config.get("device_map"):
            encoded = encoded.to(self.resolved_device)
        return encoded

    def redact(self, text: str) -> RedactionOutput:
        if self.model is None:
            self.load()
        encoded = self._encode_prompt(text)
        input_tokens = int(encoded.input_ids.shape[1])
        max_new = int(self.config.get("quality_max_new_tokens") or max(256, input_tokens * 1.3))
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
        self._sync()
        latency = time.perf_counter() - started
        generated = self.tok.decode(output[0][input_tokens:], skip_special_tokens=True)
        if "</think>" in generated:
            generated = generated.split("</think>", 1)[1]
        generated = generated.strip()
        return RedactionOutput(
            text=generated,
            input_tokens=input_tokens,
            output_tokens=int(output.shape[1]) - input_tokens,
            metrics={"latency_s": latency},
        )

    def speed_once(self, text: str, decode_new_tokens: int) -> dict[str, Any]:
        if self.model is None:
            self.load()
        encoded = self._encode_prompt(text)
        input_tokens = int(encoded.input_ids.shape[1])

        if self.resolved_device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()

        with self.torch.inference_mode():
            self._sync()
            started = time.perf_counter()
            self.model.generate(
                **encoded,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
            self._sync()
            ttft = time.perf_counter() - started

            started = time.perf_counter()
            output = self.model.generate(
                **encoded,
                max_new_tokens=decode_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
            self._sync()
            total = time.perf_counter() - started

        new_tokens = int(output.shape[1]) - input_tokens
        decode_time = max(total - ttft, 1e-9)
        peak_gpu_mb = None
        if self.resolved_device == "cuda":
            peak_gpu_mb = self.torch.cuda.max_memory_allocated() / (1024 * 1024)
        return {
            "latency_s": total,
            "input_tokens": input_tokens,
            "generated_tokens": new_tokens,
            "ttft_s": ttft,
            "prefill_tps": input_tokens / ttft if ttft > 0 else None,
            "decode_tps": max(new_tokens - 1, 0) / decode_time,
            "short_e2e_tps": (input_tokens + new_tokens) / total if total > 0 else None,
            "peak_gpu_mem_mb": peak_gpu_mb,
        }


class MLXTokenClassificationAdapter(ModelAdapter):
    def __init__(self, config: dict[str, Any], device: str = "auto") -> None:
        super().__init__(config, device)
        self.model_id = str(config.get("model_id") or "mlx-community/openai-privacy-filter-bf16")
        self.model: Any = None
        self.tokenizer: Any = None
        self.mx: Any = None
        self.model_path: Any = None
        self.id2label: dict[Any, Any] = {}
        self.resolved_device = "mps"
        self.categories = set(config.get("categories") or PII_CATEGORIES)
        self.decode_mode = str(config.get("decode_mode") or "viterbi")
        self.trim_whitespace = bool(config.get("trim_whitespace", True))
        self.discard_overlapping_predicted_spans = bool(
            config.get("discard_overlapping_predicted_spans", False)
        )
        self._torch: Any = None
        self._decoder: Any = None
        self._label_info: Any = None
        self._opf_labels_to_spans: Any = None
        self._opf_token_spans_to_char_spans: Any = None
        self._opf_trim_char_spans_whitespace: Any = None
        self._opf_discard_overlapping_spans_by_label: Any = None

    def load(self) -> dict[str, Any]:
        started = time.perf_counter()
        if self.device not in {"auto", "mps"}:
            raise ValueError("MLX adapter supports only `auto` or `mps` devices.")
        if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
            raise RuntimeError("MLX adapter requires macOS on Apple Silicon.")

        try:
            import mlx.core as mx  # type: ignore
            from mlx_embeddings.utils import get_model_path, load as load_mlx  # type: ignore
        except Exception as exc:
            raise RuntimeError("MLX adapter requires `pip install .[mlx]` on Apple Silicon.") from exc

        self.mx = mx
        self.model, self.tokenizer = load_mlx(self.model_id)
        self.model_path = get_model_path(self.model_id)
        config = getattr(self.model, "config", None)
        self.id2label = dict(getattr(config, "id2label", {}) or {})
        self._load_decoder_support()
        self.resolved_device = "mps"
        return {
            "load_s": time.perf_counter() - started,
            "device": self.resolved_device,
            "model_id": self.model_id,
            "decode_mode": self.decode_mode,
        }

    def _load_decoder_support(self) -> None:
        if not self.id2label:
            return
        class_names = [_lookup_label(self.id2label, index) for index in range(len(self.id2label))]
        if self.decode_mode == "argmax":
            return
        try:
            import torch  # type: ignore
            from opf._core.decoding import build_sequence_decoder  # type: ignore
            from opf._core.sequence_labeling import build_label_info  # type: ignore
            from opf._core.spans import (  # type: ignore
                discard_overlapping_spans_by_label,
                labels_to_spans,
                token_spans_to_char_spans,
                trim_char_spans_whitespace,
            )
        except Exception as exc:
            raise RuntimeError(
                "MLX Viterbi decode requires OPF helpers; install `pip install .[mlx]`."
            ) from exc

        self._torch = torch
        self._label_info = build_label_info(class_names)
        self._decoder, _ = build_sequence_decoder(
            decode_mode=self.decode_mode,
            label_info=self._label_info,
            viterbi_calibration_path=self.config.get("viterbi_calibration_path"),
            checkpoint_dir=str(self.model_path) if self.model_path else None,
        )
        self._opf_labels_to_spans = labels_to_spans
        self._opf_token_spans_to_char_spans = token_spans_to_char_spans
        self._opf_trim_char_spans_whitespace = trim_char_spans_whitespace
        self._opf_discard_overlapping_spans_by_label = discard_overlapping_spans_by_label

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.mx is not None:
            clear_cache = getattr(self.mx, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()
                return
            metal = getattr(self.mx, "metal", None)
            clear_cache = getattr(metal, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()

    def redact(self, text: str) -> RedactionOutput:
        if self.model is None or self.tokenizer is None:
            self.load()

        metrics: dict[str, Any] = {
            "mlx_tokenize_s": 0.0,
            "mlx_model_forward_s": 0.0,
            "mlx_decode_s": 0.0,
            "mlx_offsets_s": 0.0,
            "mlx_span_postprocess_s": 0.0,
            "mlx_redaction_s": 0.0,
        }

        total_started = time.perf_counter()

        stage_started = time.perf_counter()
        offset_mapping = None
        try:
            inputs = self.tokenizer(text, return_tensors="mlx", return_offsets_mapping=True)
            offset_mapping = _coerce_offset_mapping(inputs.get("offset_mapping"))
        except Exception:
            inputs = self.tokenizer(text, return_tensors="mlx")
        metrics["mlx_tokenize_s"] = time.perf_counter() - stage_started

        token_ids = _coerce_token_ids(inputs.get("input_ids"))
        attention_mask = inputs.get("attention_mask")
        input_tokens = _attention_token_count(attention_mask, fallback=len(token_ids))

        stage_started = time.perf_counter()
        outputs = self.model(inputs["input_ids"], attention_mask=attention_mask)
        logits = outputs.logits
        self.mx.eval(logits)
        metrics["mlx_model_forward_s"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        pred_ids = self._decode_predictions(logits)
        metrics["mlx_decode_s"] = time.perf_counter() - stage_started

        if offset_mapping is None:
            stage_started = time.perf_counter()
            offset_inputs = self.tokenizer(text, return_offsets_mapping=True)
            offset_mapping = _coerce_offset_mapping(offset_inputs.get("offset_mapping"))
            metrics["mlx_offsets_s"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        spans = self._spans_from_predictions(text, offset_mapping, token_ids, pred_ids)
        if not spans:
            labels = [_lookup_label(self.id2label, pred_id) for pred_id in pred_ids]
            spans = _fallback_spans_from_decoded_tokens(
                text,
                token_ids,
                labels,
                tokenizer=self.tokenizer,
                allowed_labels=self.categories,
            )
        metrics["mlx_span_postprocess_s"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        redacted = _apply_spans(text, spans)
        metrics["mlx_redaction_s"] = time.perf_counter() - stage_started
        metrics["latency_s"] = time.perf_counter() - total_started
        metrics["spans"] = len(spans)

        return RedactionOutput(
            text=redacted,
            input_tokens=input_tokens,
            output_tokens=token_count(redacted),
            metrics=metrics,
        )

    def _decode_predictions(self, logits: Any) -> list[int]:
        sequence_logits = logits[0].astype(self.mx.float32)
        if self._decoder is None or self._torch is None:
            preds = self.mx.argmax(sequence_logits, axis=-1)
            self.mx.eval(preds)
            return _coerce_token_ids(preds)

        try:
            import numpy as np  # type: ignore
        except Exception as exc:
            raise RuntimeError("MLX Viterbi decode requires numpy.") from exc

        log_probs = sequence_logits - self.mx.logsumexp(sequence_logits, axis=-1, keepdims=True)
        self.mx.eval(log_probs)
        torch_scores = self._torch.from_numpy(np.array(log_probs, copy=False)).to(
            dtype=self._torch.float32
        )
        decoded = self._decoder.decode(torch_scores)
        return [int(label) for label in decoded]

    def _spans_from_predictions(
        self,
        text: str,
        offset_mapping: list[tuple[int, int]],
        token_ids: list[int],
        pred_ids: list[int],
    ) -> list[SpanMatch]:
        if (
            self._label_info is None
            or self._opf_labels_to_spans is None
            or self._opf_token_spans_to_char_spans is None
        ):
            labels = [_lookup_label(self.id2label, pred_id) for pred_id in pred_ids]
            return _spans_from_bioes_offsets(offset_mapping, labels, allowed_labels=self.categories)

        labels_by_index = {index: int(label) for index, label in enumerate(pred_ids)}
        token_spans = self._opf_labels_to_spans(labels_by_index, self._label_info)
        char_starts = [int(start) for start, _end in offset_mapping[: len(token_ids)]]
        char_ends = [int(end) for _start, end in offset_mapping[: len(token_ids)]]
        char_spans = self._opf_token_spans_to_char_spans(token_spans, char_starts, char_ends)
        if self.trim_whitespace and self._opf_trim_char_spans_whitespace is not None:
            char_spans = self._opf_trim_char_spans_whitespace(char_spans, text)
        if (
            self.discard_overlapping_predicted_spans
            and self._opf_discard_overlapping_spans_by_label is not None
        ):
            char_spans = self._opf_discard_overlapping_spans_by_label(char_spans)

        spans: list[SpanMatch] = []
        for label_idx, start, end in char_spans:
            if not (0 <= int(start) < int(end) <= len(text)):
                continue
            label = (
                str(self._label_info.span_class_names[label_idx])
                if 0 <= int(label_idx) < len(self._label_info.span_class_names)
                else f"label_{label_idx}"
            )
            if label not in self.categories:
                continue
            spans.append(SpanMatch(label=label, start=int(start), end=int(end)))
        return spans


def _lookup_label(id2label: dict[Any, Any], prediction: int) -> str:
    label = id2label.get(prediction)
    if label is None:
        label = id2label.get(str(prediction))
    return str(label or "O")


def _entity_name(label: str) -> str | None:
    _tag, entity = _split_bioes_label(label)
    return entity


def _split_bioes_label(label: str) -> tuple[str | None, str | None]:
    if label == "O":
        return None, None
    prefix, sep, entity = label.partition("-")
    prefix = prefix.upper()
    if sep and prefix in {"B", "I", "E", "S"}:
        return prefix, entity
    return "S", label


def _coerce_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not value:
        return []
    if isinstance(value[0], list):
        value = value[0]
    return [int(token_id) for token_id in value]


def _coerce_offset_mapping(value: Any) -> list[tuple[int, int]]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not value:
        return []
    if isinstance(value[0], list) and value[0] and isinstance(value[0][0], list | tuple):
        value = value[0]
    offsets: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, list | tuple) or len(item) < 2:
            continue
        offsets.append((int(item[0]), int(item[1])))
    return offsets


def _attention_token_count(attention_mask: Any, fallback: int) -> int:
    if attention_mask is None:
        return fallback
    if hasattr(attention_mask, "tolist"):
        attention_mask = attention_mask.tolist()
    if not attention_mask:
        return fallback
    if isinstance(attention_mask[0], list):
        attention_mask = attention_mask[0]
    return sum(int(value) for value in attention_mask)


def _spans_from_bioes_offsets(
    offsets: list[tuple[int, int]],
    labels: list[str],
    allowed_labels: set[str] | None = None,
) -> list[SpanMatch]:
    spans: list[SpanMatch] = []
    current_label: str | None = None
    current_start: int | None = None
    current_end: int | None = None

    def flush() -> None:
        nonlocal current_label, current_start, current_end
        if (
            current_label is not None
            and current_start is not None
            and current_end is not None
            and current_start < current_end
            and (allowed_labels is None or current_label in allowed_labels)
        ):
            spans.append(SpanMatch(label=current_label, start=current_start, end=current_end))
        current_label = None
        current_start = None
        current_end = None

    for offset, label in zip(offsets, labels):
        start, end = int(offset[0]), int(offset[1])
        if start >= end:
            continue

        tag, entity = _split_bioes_label(label)
        if entity is None:
            flush()
            continue

        if tag == "S":
            flush()
            if allowed_labels is None or entity in allowed_labels:
                spans.append(SpanMatch(label=entity, start=start, end=end))
            continue

        if tag == "B":
            flush()
            current_label = entity
            current_start = start
            current_end = end
            continue

        if tag == "I":
            if current_label == entity and current_start is not None:
                current_end = max(int(current_end or end), end)
            else:
                flush()
                current_label = entity
                current_start = start
                current_end = end
            continue

        if tag == "E":
            if current_label == entity and current_start is not None:
                current_end = max(int(current_end or end), end)
                flush()
            elif allowed_labels is None or entity in allowed_labels:
                spans.append(SpanMatch(label=entity, start=start, end=end))
            continue

        flush()

    flush()
    return spans


def _fallback_spans_from_decoded_tokens(
    text: str,
    token_ids: list[int],
    labels: list[str],
    tokenizer: Any,
    allowed_labels: set[str] | None = None,
) -> list[SpanMatch]:
    spans: list[SpanMatch] = []
    search_from = 0
    group_label: str | None = None
    group_tokens: list[int] = []

    def flush() -> None:
        nonlocal group_label, group_tokens, search_from
        if group_label is None or not group_tokens:
            group_label = None
            group_tokens = []
            return

        label = group_label
        decoded = str(tokenizer.decode(group_tokens)).strip()
        group_label = None
        group_tokens = []
        if not decoded:
            return

        start = text.find(decoded, search_from)
        if start < 0:
            start = text.find(decoded)
        if start < 0:
            return

        end = start + len(decoded)
        search_from = end
        spans.append(SpanMatch(label=label, start=start, end=end))

    for token_id, label in zip(token_ids, labels):
        entity = _entity_name(label)
        if entity is None or (allowed_labels is not None and entity not in allowed_labels):
            flush()
            continue
        if group_label == entity:
            group_tokens.append(token_id)
            continue
        flush()
        group_label = entity
        group_tokens = [token_id]

    flush()
    return spans


def _apply_spans(text: str, spans: list[Any]) -> str:
    counters: dict[str, int] = {}
    redacted = text
    for span in sorted(spans, key=lambda item: int(getattr(item, "start")), reverse=True):
        label = str(getattr(span, "label"))
        counters[label] = counters.get(label, 0) + 1
        start = int(getattr(span, "start"))
        end = int(getattr(span, "end"))
        placeholder = f"[PII_{label.upper()}_{counters[label]}]"
        redacted = redacted[:start] + placeholder + redacted[end:]
    return redacted


def _valid_opf_checkpoint(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and any(path.glob("*.safetensors"))
    )


def _promote_opf_original_subtree(path: Path) -> None:
    original = path / "original"
    if not original.is_dir():
        return
    for child in original.iterdir():
        destination = path / child.name
        if destination.exists():
            continue
        shutil.move(str(child), str(destination))
    try:
        original.rmdir()
    except OSError:
        pass


def _ensure_opf_checkpoint() -> Path:
    env_path = os.environ.get("OPF_CHECKPOINT")
    if env_path:
        path = Path(env_path).expanduser()
        if _valid_opf_checkpoint(path):
            return path

    cache_dir = os.environ.get("PII_BENCH_OPF_CACHE_DIR")
    default_path = Path.home() / ".opf/privacy_filter"
    _promote_opf_original_subtree(default_path)
    if _valid_opf_checkpoint(default_path):
        return default_path

    path = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache/pii-benchmark/openai-privacy-filter"
    _promote_opf_original_subtree(path)
    if _valid_opf_checkpoint(path):
        return path

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        raise RuntimeError("OPF checkpoint auto-download requires huggingface_hub.") from exc

    print(f"[opf] downloading checkpoint to {path}...", flush=True)
    snapshot_download(
        repo_id="openai/privacy-filter",
        local_dir=str(path),
        allow_patterns=["original/*"],
    )
    _promote_opf_original_subtree(path)
    if not _valid_opf_checkpoint(path):
        raise RuntimeError(f"Downloaded OPF checkpoint is incomplete: {path}")
    return path


def _raise_if_cuda_unsupported(torch: Any) -> None:
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    supported = set(getattr(torch.cuda, "get_arch_list", lambda: [])())
    if not supported or arch in supported or f"compute_{major}{minor}" in supported:
        return

    name = torch.cuda.get_device_name(0)
    supported_text = ", ".join(sorted(supported)) or "unknown"
    raise RuntimeError(
        f"CUDA device {name} has capability {arch}, but this PyTorch build only supports "
        f"{supported_text}. This usually means the GPU is too old for the installed "
        "PyTorch/CUDA wheel. Use T4 or newer, install a PyTorch build compiled for "
        f"{arch}, "
        "or set DEVICE=cpu to benchmark CPU instead."
    )


def build_adapter(config: dict[str, Any], device: str = "auto") -> ModelAdapter:
    adapter_type = str(config.get("type"))
    if adapter_type == "regex":
        return RegexAdapter(config, device=device)
    if adapter_type == "opf":
        return OPFAdapter(config, device=device)
    if adapter_type == "hf_causal_lm":
        return HFCausalLMAdapter(config, device=device)
    if adapter_type == "mlx_token_classifier":
        return MLXTokenClassificationAdapter(config, device=device)
    raise ValueError(f"Unsupported model adapter type: {adapter_type}")

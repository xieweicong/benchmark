"""Model adapters for PII redaction benchmark tasks."""

from __future__ import annotations

import gc
import os
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
        self.categories = set(config.get("categories") or PII_CATEGORIES)

    def _pick_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch  # type: ignore
        except Exception:
            return "cpu"
        if torch.cuda.is_available():
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


def build_adapter(config: dict[str, Any], device: str = "auto") -> ModelAdapter:
    adapter_type = str(config.get("type"))
    if adapter_type == "regex":
        return RegexAdapter(config, device=device)
    if adapter_type == "opf":
        return OPFAdapter(config, device=device)
    if adapter_type == "hf_causal_lm":
        return HFCausalLMAdapter(config, device=device)
    raise ValueError(f"Unsupported model adapter type: {adapter_type}")

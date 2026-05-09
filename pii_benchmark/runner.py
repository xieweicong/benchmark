"""Benchmark execution engine."""

from __future__ import annotations

import json
import math
import signal
import statistics
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .dataset import Sample, load_jsonl, seed_text
from .hardware import capture_metadata
from .models import build_adapter
from .tokenize import build_sized_text, token_count


SCHEMA_VERSION = 1


class BenchmarkTimeout(TimeoutError):
    pass


@contextmanager
def time_limit(seconds: float | None) -> Iterator[None]:
    if (
        not seconds
        or seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def _raise_timeout(signum: int, frame: Any) -> None:
        raise BenchmarkTimeout(f"Operation exceeded {seconds:.1f}s timeout.")

    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    rank = (len(values) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (rank - low)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _numeric_values(measurements: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for measurement in measurements:
        value = measurement.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _aggregate(measurements: list[dict[str, Any]], power_watts: float | None = None) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for key in (
        "latency_s",
        "input_tps",
        "ttft_s",
        "prefill_tps",
        "decode_tps",
        "short_e2e_tps",
        "peak_gpu_mem_mb",
    ):
        values = _numeric_values(measurements, key)
        if not values:
            continue
        aggregate[f"{key}_mean"] = _mean(values)
        aggregate[f"{key}_p50"] = _percentile(values, 0.50)
        aggregate[f"{key}_p95"] = _percentile(values, 0.95)
    if power_watts and power_watts > 0:
        for source_key, target_key in (
            ("input_tps_mean", "input_tps_per_watt"),
            ("prefill_tps_mean", "prefill_tps_per_watt"),
            ("decode_tps_mean", "decode_tps_per_watt"),
        ):
            value = aggregate.get(source_key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                aggregate[target_key] = float(value) / power_watts
    return aggregate


def _score_sample(sample: Sample, output: str) -> dict[str, Any]:
    pii_total = len(sample.pii)
    missed_pii = [value for value in sample.pii if value in output]
    pii_hit = pii_total - len(missed_pii)
    anchor_total = len(sample.anchors)
    changed_anchors = [value for value in sample.anchors if value not in output]
    anchor_keep = anchor_total - len(changed_anchors)
    return {
        "sample_id": sample.id,
        "pii_total": pii_total,
        "pii_hit": pii_hit,
        "missed_pii": missed_pii,
        "anchor_total": anchor_total,
        "anchor_keep": anchor_keep,
        "changed_anchors": changed_anchors,
        "over_redaction": pii_total == 0 and anchor_keep < anchor_total,
    }


class BenchmarkRunner:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        models: list[dict[str, Any]],
        out_path: str | Path,
        project_root: str | Path,
        device: str = "auto",
        speed_sizes: list[int] | None = None,
        repeats: int | None = None,
        quality_limit: int | None = None,
        run_quality: bool = True,
    ) -> None:
        self.config = config
        self.models = models
        self.out_path = Path(out_path)
        self.project_root = Path(project_root)
        self.device = device
        self.speed_sizes = speed_sizes or list(config.get("speed_sizes", [256, 1024, 4096]))
        self.repeats = repeats or int(config.get("repeats", 3))
        self.quality_limit = quality_limit
        self.run_quality = run_quality
        self.run_id = uuid.uuid4().hex[:12]

    def run(self) -> Path:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.out_path.exists():
            self.out_path.unlink()

        dataset_path = self._resolve_path(str(self.config["dataset"]))
        samples = load_jsonl(dataset_path)
        metadata = capture_metadata(self.project_root)
        power_watts = _power_budget_watts(metadata, self.config)
        _write_jsonl(self.out_path, {
            "schema_version": SCHEMA_VERSION,
            "kind": "run_metadata",
            "run_id": self.run_id,
            "suite": self.config.get("suite", "pii-redaction"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "speed_sizes": self.speed_sizes,
                "repeats": self.repeats,
                "dataset": str(dataset_path),
                "device": self.device,
                "quality_limit": self.quality_limit,
                "run_quality": self.run_quality,
                "power_watts": power_watts,
            },
            "hardware": metadata,
        })

        seed = seed_text(samples)
        for model_config in self.models:
            self._run_model(model_config, samples, seed, power_watts)
        return self.out_path

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.project_root / path

    def _run_model(
        self,
        model_config: dict[str, Any],
        samples: list[Sample],
        seed: str,
        power_watts: float | None,
    ) -> None:
        adapter = build_adapter(model_config, device=str(model_config.get("device") or self.device))
        load_info: dict[str, Any] = {}
        try:
            with time_limit(_timeout_for(model_config, self.config)):
                load_info = adapter.load()
            self._warmup(adapter, model_config)
            for bucket in self.speed_sizes:
                self._run_speed_bucket(adapter, model_config, load_info, seed, bucket, power_watts)
            if self.run_quality:
                self._run_quality(adapter, model_config, load_info, samples)
        except Exception as exc:
            _write_jsonl(self.out_path, {
                "schema_version": SCHEMA_VERSION,
                "kind": "model_error",
                "run_id": self.run_id,
                "model": adapter.name,
                "model_type": adapter.type,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "load": load_info,
            })
        finally:
            adapter.close()

    def _warmup(self, adapter: Any, model_config: dict[str, Any]) -> None:
        warmup_text = str(self.config.get("warmup_text", "warmup: jane@example.com +1 415 555 0199"))
        warmup_repeats = int(model_config.get("warmup_repeats", self.config.get("warmup_repeats", 1)))
        for _ in range(max(warmup_repeats, 0)):
            with time_limit(_timeout_for(model_config, self.config)):
                adapter.speed_once(warmup_text, int(self.config.get("decode_new_tokens", 32)))

    def _run_speed_bucket(
        self,
        adapter: Any,
        model_config: dict[str, Any],
        load_info: dict[str, Any],
        seed: str,
        bucket: int,
        power_watts: float | None,
    ) -> None:
        text = build_sized_text(seed, bucket)
        measurements: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for repeat_index in range(self.repeats):
            try:
                with time_limit(_timeout_for(model_config, self.config)):
                    measurement = adapter.speed_once(
                        text,
                        int(model_config.get("decode_new_tokens", self.config.get("decode_new_tokens", 64))),
                    )
                measurement["repeat_index"] = repeat_index
                measurements.append(measurement)
            except Exception as exc:
                errors.append({"type": type(exc).__name__, "message": str(exc)})
                if bool(model_config.get("stop_on_error", False)):
                    break

        aggregate = _aggregate(measurements, power_watts)
        cost_per_hour = model_config.get("cost_per_hour_usd", self.config.get("cost_per_hour_usd"))
        if cost_per_hour and aggregate.get("input_tps_mean"):
            aggregate["usd_per_1m_input_tokens"] = (
                float(cost_per_hour) / 3600.0 / float(aggregate["input_tps_mean"]) * 1_000_000
            )

        _write_jsonl(self.out_path, {
            "schema_version": SCHEMA_VERSION,
            "kind": "speed",
            "run_id": self.run_id,
            "model": adapter.name,
            "model_type": adapter.type,
            "bucket_tokens": bucket,
            "actual_tokens": token_count(text),
            "chars": len(text),
            "power_watts": power_watts,
            "repeats": self.repeats,
            "successful_repeats": len(measurements),
            "load": load_info,
            "measurements": measurements,
            "aggregate": aggregate,
            "errors": errors,
        })

    def _run_quality(
        self,
        adapter: Any,
        model_config: dict[str, Any],
        load_info: dict[str, Any],
        samples: list[Sample],
    ) -> None:
        selected = samples[: self.quality_limit] if self.quality_limit else samples
        pii_total = 0
        pii_hit = 0
        anchor_total = 0
        anchor_keep = 0
        sample_scores: list[dict[str, Any]] = []
        latency_s = 0.0
        errors: list[dict[str, str]] = []

        for sample in selected:
            try:
                with time_limit(_timeout_for(model_config, self.config)):
                    started = time.perf_counter()
                    output = adapter.redact(sample.text)
                    latency_s += time.perf_counter() - started
                score = _score_sample(sample, output.text)
                sample_scores.append(score)
                pii_total += score["pii_total"]
                pii_hit += score["pii_hit"]
                anchor_total += score["anchor_total"]
                anchor_keep += score["anchor_keep"]
            except Exception as exc:
                errors.append({"sample_id": sample.id, "type": type(exc).__name__, "message": str(exc)})

        _write_jsonl(self.out_path, {
            "schema_version": SCHEMA_VERSION,
            "kind": "quality",
            "run_id": self.run_id,
            "model": adapter.name,
            "model_type": adapter.type,
            "sample_count": len(selected),
            "successful_samples": len(sample_scores),
            "pii_total": pii_total,
            "pii_hit": pii_hit,
            "recall": pii_hit / pii_total if pii_total else 1.0,
            "anchor_total": anchor_total,
            "anchor_keep": anchor_keep,
            "anchor_keep_rate": anchor_keep / anchor_total if anchor_total else 1.0,
            "latency_s": latency_s,
            "sample_scores": sample_scores,
            "load": load_info,
            "errors": errors,
        })


def _timeout_for(model_config: dict[str, Any], config: dict[str, Any]) -> float | None:
    value = model_config.get("timeout_s", config.get("timeout_s"))
    return float(value) if value else None


def _power_budget_watts(metadata: dict[str, Any], config: dict[str, Any]) -> float | None:
    value = config.get("power_watts")
    if value:
        try:
            watts = float(value)
        except (TypeError, ValueError):
            watts = 0
        if watts > 0:
            return watts

    value = metadata.get("power", {}).get("budget_watts")
    if value:
        try:
            watts = float(value)
        except (TypeError, ValueError):
            watts = 0
        if watts > 0:
            return watts
    return None

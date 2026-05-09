"""Report generation from benchmark JSONL files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .hardware import hardware_label


OPF_STAGE_FIELDS: tuple[tuple[str, str], ...] = (
    ("opf_component_s", "Components ms"),
    ("opf_tokenize_s", "Tokenize ms"),
    ("opf_window_prepare_s", "Tensor ms"),
    ("opf_model_forward_s", "Forward ms"),
    ("opf_logprob_transfer_s", "Logprob/copy ms"),
    ("opf_aggregation_s", "Aggregate ms"),
    ("opf_decode_s", "Decode ms"),
    ("opf_span_postprocess_s", "Span ms"),
    ("opf_redaction_s", "Redact ms"),
)


def load_rows(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for file in files:
            with file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        row["_source"] = str(file)
                        rows.append(row)
    return rows


def write_markdown(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = [row for row in rows if row.get("kind") == "run_metadata"]
    warmup_rows = [row for row in rows if row.get("kind") == "warmup"]
    speed_rows = [row for row in rows if row.get("kind") == "speed"]
    batch_rows = [row for row in rows if row.get("kind") == "batch_compare"]
    quality_rows = [row for row in rows if row.get("kind") == "quality"]
    errors = [row for row in rows if row.get("kind") == "model_error"]

    lines: list[str] = []
    lines.append("# PII Redaction Benchmark Report")
    lines.append("")
    if metadata:
        lines.append("## Runs")
        lines.append("")
        lines.append("| Run | Created | Hardware | Power W | Git | Source |")
        lines.append("|---|---|---|---:|---|---|")
        for row in metadata:
            hw = hardware_label(row.get("hardware", {}))
            power_watts = row.get("hardware", {}).get("power", {}).get("budget_watts")
            git = row.get("hardware", {}).get("git", {})
            commit = (git.get("commit") or "")[:8]
            dirty = " dirty" if git.get("dirty") else ""
            lines.append(
                f"| `{row.get('run_id')}` | {row.get('created_at', '')} | "
                f"{hw} | {_fmt(power_watts)} | `{commit}{dirty}` | `{row.get('_source')}` |"
            )
        lines.append("")

    if warmup_rows:
        lines.append("## Warmup")
        lines.append("")
        lines.append(
            "First measured call after adapter load. Useful for lazy model load and kernel/cache setup."
        )
        lines.append("")
        lines.append(
            "| Model | Type | Repeats | Load s | Warmup p50 s | Input tok/s | Forward ms | Windows |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in warmup_rows:
            agg = row.get("aggregate", {})
            load = row.get("load", {})
            lines.append(
                f"| {row.get('model')} | {row.get('model_type')} | {row.get('repeats')} | "
                f"{_fmt(load.get('load_s'))} | {_fmt(agg.get('latency_s_p50'))} | "
                f"{_fmt(agg.get('input_tps_mean'))} | "
                f"{_fmt_ms(agg.get('opf_model_forward_s_mean'))} | "
                f"{_fmt(agg.get('opf_windows_mean'))} |"
            )
        lines.append("")

    if speed_rows:
        lines.append("## Speed")
        lines.append("")
        lines.append(
            "| Model | Type | Bucket tok | Success | Latency p50 s | Input tok/s | "
            "Power W | Input tok/s/W | Prefill tok/s | Decode tok/s |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in speed_rows:
            agg = row.get("aggregate", {})
            lines.append(
                f"| {row.get('model')} | {row.get('model_type')} | {row.get('bucket_tokens')} | "
                f"{row.get('successful_repeats')}/{row.get('repeats')} | "
                f"{_fmt(agg.get('latency_s_p50'))} | {_fmt(agg.get('input_tps_mean'))} | "
                f"{_fmt(row.get('power_watts'))} | {_fmt(agg.get('input_tps_per_watt'))} | "
                f"{_fmt(agg.get('prefill_tps_mean'))} | {_fmt(agg.get('decode_tps_mean'))} |"
            )
        lines.append("")

        stage_rows = _stage_breakdowns(speed_rows)
        if stage_rows:
            lines.append("## Stage Breakdown")
            lines.append("")
            lines.append(
                "Measured inside the OPF call with GPU/MPS synchronization around model work."
            )
            lines.append("")
            lines.append(
                "| Model | Bucket tok | Latency s | Components ms | Tokenize ms | Tensor ms | "
                "Forward ms | Logprob/copy ms | Aggregate ms | Decode ms | Span ms | "
                "Redact ms | Other ms | Windows |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for item in stage_rows:
                lines.append(
                    f"| {item.get('model')} | {item.get('bucket_tokens')} | "
                    f"{_fmt(item.get('latency_s'))} | "
                    f"{_fmt_ms(item.get('opf_component_s'))} | "
                    f"{_fmt_ms(item.get('opf_tokenize_s'))} | "
                    f"{_fmt_ms(item.get('opf_window_prepare_s'))} | "
                    f"{_fmt_ms(item.get('opf_model_forward_s'))} | "
                    f"{_fmt_ms(item.get('opf_logprob_transfer_s'))} | "
                    f"{_fmt_ms(item.get('opf_aggregation_s'))} | "
                    f"{_fmt_ms(item.get('opf_decode_s'))} | "
                    f"{_fmt_ms(item.get('opf_span_postprocess_s'))} | "
                    f"{_fmt_ms(item.get('opf_redaction_s'))} | "
                    f"{_fmt_ms(item.get('other_s'))} | {_fmt(item.get('opf_windows'))} |"
                )
            lines.append("")

        scaling = _scaling_estimates(speed_rows)
        if scaling:
            lines.append("## Scaling Estimate")
            lines.append("")
            lines.append(
                "Fits `latency = fixed_overhead + per_token_time * tokens` from speed buckets. "
                "Use this as a rough diagnostic, not a hardware spec."
            )
            lines.append("")
            lines.append(
                "| Run | Model | Type | Points | Fixed overhead s | Per-token ms | "
                "Asymptotic tok/s | R^2 |"
            )
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
            for item in scaling:
                lines.append(
                    f"| `{item['run_id']}` | {item['model']} | {item['model_type']} | "
                    f"{item['points']} | {_fmt(item['fixed_overhead_s'])} | "
                    f"{_fmt(item['per_token_ms'])} | {_fmt(item['asymptotic_tps'])} | "
                    f"{_fmt(item['r2'])} |"
                )
            lines.append("")

    if batch_rows:
        lines.append("## Batch Compare")
        lines.append("")
        lines.append(
            "Compares many small requests against one concatenated long request with the same total token budget."
        )
        lines.append("")
        for row in batch_rows:
            serial_count = row.get("serial_count")
            serial_chunk_tokens = row.get("serial_chunk_tokens")
            serial_total_tokens = row.get("serial_total_tokens")
            concat_tokens = row.get("concat_tokens")
            serial_total_latency = row.get("serial_total_latency_s")
            concat_latency = row.get("concat_latency_s")
            lines.append(f"- Model `{row.get('model')}`")
            lines.append(
                f"  - Serial: `{serial_count} x {serial_chunk_tokens}` calls, "
                f"about `{_fmt(serial_total_tokens)}` tokens total, "
                f"{_fmt(serial_total_latency)} s, `{_fmt(row.get('serial_effective_tps'))}` tok/s"
            )
            lines.append(
                f"  - Concat: `1 x {concat_tokens}` call, "
                f"{_fmt(concat_latency)} s, `{_fmt(row.get('concat_input_tps'))}` tok/s"
            )
            lines.append(
                f"  - Speedup from concatenation: `{_fmt(row.get('speedup_vs_serial'))}x`"
            )
        lines.append("")

    if quality_rows:
        lines.append("## Quality")
        lines.append("")
        lines.append("| Model | Type | Samples | Recall | PII hit | Anchor keep | Latency s | Errors |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in quality_rows:
            lines.append(
                f"| {row.get('model')} | {row.get('model_type')} | "
                f"{row.get('successful_samples')}/{row.get('sample_count')} | "
                f"{_pct(row.get('recall'))} | {row.get('pii_hit')}/{row.get('pii_total')} | "
                f"{row.get('anchor_keep')}/{row.get('anchor_total')} ({_pct(row.get('anchor_keep_rate'))}) | "
                f"{_fmt(row.get('latency_s'))} | {len(row.get('errors', []))} |"
            )
        lines.append("")

        detail_rows: list[tuple[str, dict[str, Any]]] = []
        for row in quality_rows:
            for sample in row.get("sample_scores", []):
                if sample.get("missed_pii") or sample.get("changed_anchors"):
                    detail_rows.append((str(row.get("model")), sample))
        if detail_rows:
            lines.append("## Quality Details")
            lines.append("")
            lines.append("| Model | Sample | Missed PII | Changed anchors |")
            lines.append("|---|---|---|---|")
            for model, sample in detail_rows:
                lines.append(
                    f"| {model} | {sample.get('sample_id')} | "
                    f"{_list(sample.get('missed_pii'))} | {_list(sample.get('changed_anchors'))} |"
                )
            lines.append("")

    if errors:
        lines.append("## Model Errors")
        lines.append("")
        lines.append("| Model | Type | Error |")
        lines.append("|---|---|---|")
        for row in errors:
            lines.append(
                f"| {row.get('model')} | {row.get('model_type')} | "
                f"{row.get('error_type')}: {str(row.get('error', '')).replace('|', '/')[:180]} |"
            )
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        kind = row.get("kind")
        if kind in {"speed", "warmup", "batch_compare"}:
            agg = row.get("aggregate", {})
            flat_rows.append({
                "kind": kind,
                "source": row.get("_source"),
                "run_id": row.get("run_id"),
                "model": row.get("model"),
                "model_type": row.get("model_type"),
                "bucket_tokens": row.get("bucket_tokens"),
                "successful_repeats": row.get("successful_repeats"),
                "repeats": row.get("repeats"),
                "load_s": row.get("load", {}).get("load_s"),
                "serial_count": row.get("serial_count"),
                "serial_chunk_tokens": row.get("serial_chunk_tokens"),
                "serial_total_tokens": row.get("serial_total_tokens"),
                "serial_total_latency_s": row.get("serial_total_latency_s"),
                "serial_effective_tps": row.get("serial_effective_tps"),
                "concat_tokens": row.get("concat_tokens"),
                "concat_latency_s": row.get("concat_latency_s"),
                "concat_input_tps": row.get("concat_input_tps"),
                "speedup_vs_serial": row.get("speedup_vs_serial"),
                "latency_s_p50": agg.get("latency_s_p50"),
                "latency_s_p95": agg.get("latency_s_p95"),
                "input_tps_mean": agg.get("input_tps_mean"),
                "power_watts": row.get("power_watts"),
                "input_tps_per_watt": agg.get("input_tps_per_watt"),
                "prefill_tps_mean": agg.get("prefill_tps_mean"),
                "prefill_tps_per_watt": agg.get("prefill_tps_per_watt"),
                "decode_tps_mean": agg.get("decode_tps_mean"),
                "decode_tps_per_watt": agg.get("decode_tps_per_watt"),
            })
            _add_stage_columns(flat_rows[-1], row)
        elif kind == "quality":
            flat_rows.append({
                "kind": kind,
                "source": row.get("_source"),
                "run_id": row.get("run_id"),
                "model": row.get("model"),
                "model_type": row.get("model_type"),
                "sample_count": row.get("sample_count"),
                "successful_samples": row.get("successful_samples"),
                "recall": row.get("recall"),
                "pii_hit": row.get("pii_hit"),
                "pii_total": row.get("pii_total"),
                "anchor_keep_rate": row.get("anchor_keep_rate"),
                "latency_s": row.get("latency_s"),
            })
    for item in _scaling_estimates([row for row in rows if row.get("kind") == "speed"]):
        flat_rows.append({
            "kind": "scaling_estimate",
            "source": item.get("source"),
            "run_id": item.get("run_id"),
            "model": item.get("model"),
            "model_type": item.get("model_type"),
            "points": item.get("points"),
            "fixed_overhead_s": item.get("fixed_overhead_s"),
            "per_token_ms": item.get("per_token_ms"),
            "asymptotic_tps": item.get("asymptotic_tps"),
            "r2": item.get("r2"),
        })
    fieldnames = sorted({key for row in flat_rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _list(value: Any) -> str:
    if not value:
        return "-"
    if isinstance(value, list):
        return "<br>".join(str(item).replace("|", "/") for item in value)
    return str(value).replace("|", "/")


def _fmt_ms(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 1000.0:.1f}"
    except (TypeError, ValueError):
        return str(value)


def _measurement_values(row: dict[str, Any], key: str) -> list[float]:
    values: list[float] = []
    for measurement in row.get("measurements", []):
        value = measurement.get(key)
        if _is_number(value):
            values.append(float(value))
    return values


def _measurement_mean(row: dict[str, Any], key: str) -> float | None:
    values = _measurement_values(row, key)
    return sum(values) / len(values) if values else None


def _add_stage_columns(target: dict[str, Any], row: dict[str, Any]) -> None:
    for key, _label in OPF_STAGE_FIELDS:
        target[f"{key}_mean"] = _measurement_mean(row, key)
    target["opf_windows_mean"] = _measurement_mean(row, "opf_windows")
    target["opf_window_tokens_mean"] = _measurement_mean(row, "opf_window_tokens")


def _stage_breakdowns(speed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in speed_rows:
        if not any(_measurement_values(row, key) for key, _label in OPF_STAGE_FIELDS):
            continue
        item: dict[str, Any] = {
            "run_id": row.get("run_id"),
            "model": row.get("model"),
            "model_type": row.get("model_type"),
            "bucket_tokens": row.get("bucket_tokens"),
            "latency_s": _measurement_mean(row, "latency_s"),
            "opf_windows": _measurement_mean(row, "opf_windows"),
        }
        accounted = 0.0
        for key, _label in OPF_STAGE_FIELDS:
            value = _measurement_mean(row, key)
            item[key] = value
            if value is not None:
                accounted += value
        latency = item.get("latency_s")
        item["other_s"] = max(float(latency) - accounted, 0.0) if _is_number(latency) else None
        rows.append(item)
    return rows


def _scaling_estimates(speed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[tuple[float, float]]] = {}
    meta: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in speed_rows:
        agg = row.get("aggregate", {})
        latency = agg.get("latency_s_p50")
        tokens = row.get("bucket_tokens")
        if not _is_number(latency) or not _is_number(tokens):
            continue
        if row.get("successful_repeats", 0) <= 0:
            continue
        key = (
            str(row.get("_source") or ""),
            str(row.get("run_id") or ""),
            str(row.get("model") or ""),
            str(row.get("model_type") or ""),
        )
        groups.setdefault(key, []).append((float(tokens), float(latency)))
        meta[key] = {
            "source": row.get("_source"),
            "run_id": row.get("run_id"),
            "model": row.get("model"),
            "model_type": row.get("model_type"),
        }

    estimates: list[dict[str, Any]] = []
    for key, points in groups.items():
        unique = sorted(set(points))
        if len(unique) < 2:
            continue
        xs = [point[0] for point in unique]
        ys = [point[1] for point in unique]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denominator = sum((x - x_mean) ** 2 for x in xs)
        if denominator <= 0:
            continue
        slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
        intercept = y_mean - slope * x_mean
        if slope <= 0:
            continue

        predicted = [intercept + slope * x for x in xs]
        ss_res = sum((y - y_hat) ** 2 for y, y_hat in zip(ys, predicted))
        ss_tot = sum((y - y_mean) ** 2 for y in ys)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else None

        estimates.append({
            **meta[key],
            "points": len(unique),
            "fixed_overhead_s": max(intercept, 0.0),
            "raw_intercept_s": intercept,
            "per_token_ms": slope * 1000.0,
            "asymptotic_tps": 1.0 / slope,
            "r2": r2,
        })
    return estimates


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)

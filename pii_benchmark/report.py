"""Report generation from benchmark JSONL files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .hardware import hardware_label


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
    speed_rows = [row for row in rows if row.get("kind") == "speed"]
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
        if kind == "speed":
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

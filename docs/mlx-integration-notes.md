# MLX Privacy Filter Integration Notes

This document records the MLX integration work for `mlx-community/openai-privacy-filter-bf16`, the issues found during benchmarking, the final optimization applied, and the current behavior of the benchmark harness.

## Goal

Add an MLX-based benchmark path alongside the existing Torch/OPF path so the repository can compare:

- `opf` on the current Torch runtime
- `mlx-opf-bf16` via `mlx-embeddings` on Apple Silicon

The target model is:

- `mlx-community/openai-privacy-filter-bf16`

The benchmark path is exposed as:

- `./run.sh mlx`

## What Was Added

The integration adds a new adapter and supporting configuration:

- `MLXTokenClassificationAdapter` in `pii_benchmark/models.py`
- model config entry `mlx-opf-bf16` in `configs/pii-redaction.json`
- `mlx` optional dependency group in `pyproject.toml`
- local runner support for `./run.sh mlx`
- package metadata capture for `mlx` and `mlx-embeddings`
- helper tests for span reconstruction and placeholder rendering in `tests/test_mlx_adapter.py`

## Initial Implementation

The first MLX implementation followed the straightforward token-classification path exposed by `mlx-embeddings`:

1. tokenize text
2. run the MLX model
3. take token-wise `argmax`
4. reconstruct BIOES spans
5. apply benchmark placeholders with `_apply_spans(...)`

This version produced very strong speed numbers on Apple M4 Pro.

## First Benchmark Result

Compared against the existing OPF result in `results/opf-20260513T022245Z.md`, the first MLX run showed much higher steady-state throughput.

Approximate throughput comparison:

| Bucket | OPF tok/s | Initial MLX tok/s | Speedup |
|---|---:|---:|---:|
| 256 | 150.36 | 9459.74 | 62.9x |
| 1024 | 154.68 | 20195.95 | 130.6x |
| 4096 | 154.77 | 15931.27 | 102.9x |

However, quality inspection showed that the initial MLX path was not equivalent to OPF text output.

## Problem Found During Quality Comparison

The initial MLX path matched OPF on entity recall for the local synthetic dataset, but it differed in text rendering.

Observed problems included:

- missing spaces around placeholders
- span boundaries shifted left or right by one token fragment
- occasional anchor damage near Japanese text boundaries
- sample outputs that were semantically close but not text-identical to OPF

Before the final fix:

- OPF and initial MLX had the same recall on the dataset
- initial MLX had worse anchor preservation
- exact output match rate was only `1/8`

This showed that the issue was not primarily model accuracy. It was a decoding and post-processing mismatch.

## Root Cause

The initial MLX adapter used simple per-token `argmax` decoding.

The OPF runtime does more than that. Its inference path uses:

1. log-prob aggregation
2. Viterbi sequence decoding
3. token-span reconstruction with OPF label metadata
4. token-span to char-span conversion
5. whitespace trimming
6. overlap handling

So the main issue was:

- the MLX adapter did not yet reproduce OPF's decode pipeline

This is not an unavoidable MLX conversion artifact.

## Final Optimization Applied

The MLX adapter was updated to reuse OPF-compatible decoding helpers during post-processing:

- `build_label_info(...)`
- `build_sequence_decoder(...)`
- `labels_to_spans(...)`
- `token_spans_to_char_spans(...)`
- `trim_char_spans_whitespace(...)`
- optional overlap filtering

The MLX adapter now defaults to:

- `decode_mode = "viterbi"`
- `trim_whitespace = true`

This means the MLX path still runs the model in MLX, but it uses OPF's decoding logic to reconstruct spans in the same way as the Torch/OPF path.

## Dependency Note

This final MLX implementation depends on two external packages:

- `mlx-embeddings` from GitHub, not the older PyPI-only release
- `opf`, because the MLX adapter now uses OPF's decoding helpers for Viterbi and span post-processing

As a result, the `mlx` extra includes both:

- `mlx-embeddings`
- `opf`

## Final Quality Result

After the Viterbi/post-processing alignment, MLX matches OPF exactly on the local dataset.

Dataset summary:

| Model | PII Hit | PII Total | Recall | Anchor Keep | Anchor Total | Anchor Keep Rate |
|---|---:|---:|---:|---:|---:|---:|
| `opf` | 24 | 26 | 92.3% | 18 | 21 | 85.7% |
| `mlx-opf-bf16` | 24 | 26 | 92.3% | 18 | 21 | 85.7% |

Exact output match after the final fix:

- `8/8` samples matched OPF text output exactly

This is the strongest indication that the earlier quality differences were caused by incomplete decoding logic, not by the MLX model conversion itself.

## Final Speed Result

After adding Viterbi decoding, MLX remains much faster than OPF, although slower than the initial argmax-only MLX prototype.

Measured with the final MLX implementation on Apple M4 Pro:

| Bucket | OPF tok/s | Final MLX tok/s | Speedup |
|---|---:|---:|---:|
| 256 | 150.36 | 6593.41 | 43.8x |
| 1024 | 154.68 | 11645.94 | 75.3x |
| 4096 | 154.77 | 9362.80 | 60.5x |

Additional notes:

- the final MLX implementation is slower than the first argmax-only MLX version
- this is expected because Viterbi decoding adds real decode work
- despite that, steady-state MLX throughput remains dramatically higher than OPF on this machine

## Cold Start Behavior

Steady-state speed and cold-start behavior are different.

MLX characteristics:

- higher initial load time than OPF
- much faster warm and repeated inference once loaded

So MLX is the clear winner for repeated local inference, while OPF may still have a smaller startup footprint in some cases.

## Current User Workflow

Run the optimized MLX path locally on Apple Silicon:

```bash
./run.sh mlx
```

Direct CLI form:

```bash
python -m pii_benchmark.cli run \
  --models mlx-opf-bf16 \
  --sizes 256,1024,4096 \
  --repeats 3 \
  --device mps
```

## Current Limitations

- MLX is intended only for macOS Apple Silicon
- Docker CUDA, Jetson, and Kaggle flows do not support this MLX path
- the benchmark currently reports MLX timing fields as `mlx_*` metrics, separate from `opf_*`
- the benchmark quality comparison described here was run directly against the local dataset and not yet written into a persistent `results/*.jsonl` quality row for MLX

## Summary

The important conclusion from this work is:

- MLX conversion is not inherently less accurate for this model
- the initial quality gap came from decode and span post-processing differences
- after aligning MLX with OPF's Viterbi-based decode path, MLX matched OPF output exactly on the local dataset while still keeping a large speed advantage

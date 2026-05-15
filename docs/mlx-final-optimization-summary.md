# MLX Final Optimization Summary

This note summarizes the final MLX implementation used in the benchmark after quality and speed validation.

## Final State

The benchmark now supports an Apple Silicon MLX path for the privacy filter model:

- model: `mlx-community/openai-privacy-filter-bf16`
- entrypoint: `./run.sh mlx`
- adapter: `MLXTokenClassificationAdapter`

## What Changed From The First MLX Version

The first MLX version used simple token-wise `argmax` decoding.

The final MLX version adds OPF-compatible post-processing:

- Viterbi sequence decoding
- label-space-aware span reconstruction
- token-to-char span conversion
- whitespace trimming
- optional overlap filtering

## Why This Optimization Matters

The initial MLX version was already very fast, but its redacted text output differed from OPF because its decode path was too simple.

The final optimization fixes that by aligning the MLX output path with OPF's decode logic while keeping the model inference itself on MLX.

## Final Accuracy Outcome

On the local synthetic dataset:

- OPF recall: `24/26 = 92.3%`
- MLX recall after optimization: `24/26 = 92.3%`
- OPF anchor keep: `18/21 = 85.7%`
- MLX anchor keep after optimization: `18/21 = 85.7%`
- exact text match after optimization: `8/8` samples

This means the final MLX implementation matches OPF exactly on the local benchmark dataset.

## Final Speed Outcome

On Apple M4 Pro, the final optimized MLX path remains much faster than OPF:

| Bucket | OPF tok/s | Final MLX tok/s | Speedup |
|---|---:|---:|---:|
| 256 | 150.36 | 6593.41 | 43.8x |
| 1024 | 154.68 | 11645.94 | 75.3x |
| 4096 | 154.77 | 9362.80 | 60.5x |

The final MLX path is slower than the earlier argmax-only MLX prototype because Viterbi decoding adds extra work, but it still keeps a very large speed advantage over OPF.

## Dependency Requirements

The final MLX path depends on:

- `mlx`
- `mlx-embeddings` from GitHub
- `opf`, for decoder and span post-processing helpers

These are included through the repository's `mlx` extra.

## Recommended Way To Run

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

## Practical Recommendation

For repeated local inference on Apple Silicon, the final optimized MLX path is the preferred option because it combines:

- OPF-equivalent output quality on the benchmark dataset
- significantly higher steady-state throughput

If cold-start latency is the only concern, OPF may still have a smaller startup footprint, but for real repeated benchmark runs the MLX path is the better tradeoff.

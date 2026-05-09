# PII Local Benchmark

Portable benchmark harness for local PII redaction models. It is designed to run on a laptop, an on-prem box, or a rented cloud GPU instance, then emit portable JSONL results that can be merged into one report.

## What It Measures

- Hardware and software metadata: OS, Python, CPU, RAM, GPU, CUDA/MPS, package versions, git commit.
- Jetson metadata when present: L4T release, device-tree model, `nvpmodel`, and a `tegrastats` snapshot.
- Speed: load time, warm latency, p50/p95 latency, input tokens/sec, and HF prefill/decode tokens/sec when available.
- Efficiency: optional power budget and tokens/sec/watt for local deployment comparisons.
- Quality: recall against synthetic multilingual PII samples plus anchor preservation to catch over-redaction or paraphrasing.

## Quick Start

The easiest local path uses `uv` and the wrapper script:

```bash
cd pii-benchmark
./run.sh smoke
```

`./run.sh` auto-selects a runner:

- Jetson/L4T device -> `scripts/run_jetson.sh`
- Linux NVIDIA GPU with Docker -> `scripts/run_docker_cuda.sh`
- everything else -> `scripts/run_local_uv.sh`

Force a runner when needed:

```bash
PII_BENCH_RUNNER=local ./run.sh opf
PII_BENCH_RUNNER=system ./run.sh opf
PII_BENCH_RUNNER=docker-cuda ./run.sh opf
PII_BENCH_RUNNER=jetson ./run.sh opf
```

The first `uv` run creates `.venv/` locally and uses `uv.lock` when possible.

Run OPF locally:

```bash
./run.sh opf
```

Run a Hugging Face causal LM locally:

```bash
./run.sh hf qwen3-0.8b
```

By default, `./run.sh opf` automatically uses the PII Shield checkpoint at:

```text
~/Library/Application Support/PII Shield/model/privacy_filter
```

Override defaults with env vars:

```bash
SIZES=256,1024 REPEATS=5 DEVICE=mps ./run.sh opf
PII_BENCH_OPF_CHECKPOINT=/path/to/privacy_filter ./run.sh opf
MODELS=qwen3-1.7b DEVICE=cuda ./run.sh hf
```

Add a power budget to get `tok/s/W`:

```bash
PII_BENCH_POWER_WATTS=30 PII_BENCH_POWER_NOTE="M1 Pro measured package power" ./run.sh opf
```

Every run writes three files:

```text
results/<name>.jsonl   # source of truth
results/<name>.md      # readable report
results/<name>.csv     # spreadsheet-friendly summary
```

Generate a merged report from multiple machines:

```bash
python -m pii_benchmark.cli report \
  --input results/mac-mps.jsonl results/l4.jsonl results/a100.jsonl \
  --markdown results/summary.md \
  --csv results/summary.csv
```

## Cloud GPU Workflow

On an NVIDIA GPU machine with Docker + NVIDIA Container Toolkit:

```bash
cd pii-benchmark
./scripts/run_docker_cuda.sh opf
```

Run a Hugging Face baseline:

```bash
./scripts/run_docker_cuda.sh hf qwen3-0.8b
```

Useful overrides:

```bash
SIZES=256,1024,4096 REPEATS=5 ./scripts/run_docker_cuda.sh opf
MODELS=qwen3-1.7b ./scripts/run_docker_cuda.sh hf
BUILD=1 ./scripts/run_docker_cuda.sh smoke
```

The Docker script:

- builds `pii-benchmark:cuda` if it does not exist
- mounts `results/` so outputs stay on the host
- mounts the Hugging Face cache to avoid re-downloading models
- passes `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` into the container when set

Typical cloud flow:

1. Rent a GPU instance.
2. Clone this project or copy the `pii-benchmark/` directory.
3. Run `./scripts/run_docker_cuda.sh opf`.
4. Pull back the `results/*.jsonl` files.
5. Use `python -m pii_benchmark.cli report` locally to merge them.

The first version intentionally avoids cloud-provider orchestration. That keeps the run format stable before adding SSH runners, Docker images, or platform-specific launchers.

## Kaggle / Notebook Workflow

Kaggle usually has no Docker daemon and already has a Python/CUDA stack. In that environment, `./run.sh` auto-selects the system Python runner instead of creating a uv virtualenv.

Smoke test:

```bash
./run.sh smoke
```

Run OPF. On Kaggle, the system runner auto-installs missing OPF dependencies into the notebook Python:

```bash
./run.sh opf
```

Run a Hugging Face baseline. Missing `torch`/`transformers` dependencies are auto-installed on Kaggle when needed:

```bash
./run.sh hf qwen3-0.8b
```

Disable auto-install if you want to manage the environment yourself:

```bash
PII_BENCH_AUTO_INSTALL=0 ./run.sh opf
```

If you ever see noisy notebook `sitecustomize` warnings from uv, force the system runner:

```bash
PII_BENCH_RUNNER=system ./run.sh smoke
```

## Jetson Workflow

On real Jetson hardware, use the auto runner or call the Jetson script directly:

```bash
./run.sh opf
./scripts/run_jetson.sh opf
```

Jetson-specific notes:

- Use the JetPack/L4T Python and PyTorch environment already installed on the device, or run inside an NVIDIA L4T container.
- `scripts/run_jetson.sh` defaults to `DEVICE=cuda` and `OPF_MOE_TRITON=0`.
- Set `PII_BENCH_POWER_WATTS` to the configured or measured power budget for `tok/s/W`.
- The metadata collector records `/etc/nv_tegra_release`, device-tree model, `nvpmodel -q`, and one `tegrastats` line when available.

Examples:

```bash
PII_BENCH_POWER_WATTS=15 PII_BENCH_POWER_NOTE="Orin Nano 15W mode" ./scripts/run_jetson.sh opf
PII_BENCH_POWER_WATTS=25 SIZES=256,1024,4096 REPEATS=5 ./scripts/run_jetson.sh opf
```

## Direct CLI

The scripts call this Python entrypoint underneath. Use it directly when you need full control:

```bash
python -m pii_benchmark.cli run \
  --models regex \
  --sizes 128,512 \
  --repeats 2 \
  --power-watts 30 \
  --power-note "manual power budget" \
  --quality-limit 3 \
  --markdown-report results/smoke.md \
  --csv results/smoke.csv
```

After `pip install -e .`, the shorter console command is available:

```bash
pii-bench run --models regex --sizes 128,512
```

## Result Format

Each JSONL file contains:

- one `run_metadata` row
- one `speed` row per model and bucket
- one `quality` row per model
- optional `model_error` rows when a dependency, model download, OOM, or timeout fails

JSONL is the source of truth. Markdown and CSV are generated views.

## Config

Default config lives at `configs/pii-redaction.json`.

Models are selected by `name`, `type`, or HF `model_id`:

```bash
python -m pii_benchmark.cli run --models regex,opf
```

Disabled models in the config are skipped by default, but can still be selected explicitly:

```bash
python -m pii_benchmark.cli run --models qwen3-1.7b
```

## Notes

- Dataset samples are synthetic and local.
- The dependency-free token counter is a stable benchmark bucket, not a model tokenizer clone.
- HF adapters also report model tokenizer input length for speed rows.
- Use Docker for Linux/NVIDIA cloud GPUs.
- Use `uv` for local Mac/Apple Silicon because Docker does not give a realistic MPS benchmark.
- Python 3.12 is recommended for local OPF runs. Override the uv interpreter with `UV_PYTHON=3.11` or `UV_PYTHON=3.12.11`.
- OPF on Apple Silicon sets `OPF_MOE_TRITON=0` automatically when using MPS.

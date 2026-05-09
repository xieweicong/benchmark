#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-opf}"
if [[ $# -gt 0 ]]; then
  shift
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed or not on PATH." >&2
  exit 127
fi

IMAGE="${IMAGE:-pii-benchmark:cuda}"
BUILD="${BUILD:-auto}"
DEVICE="${DEVICE:-cuda}"
DOCKER_GPU_ARGS="${DOCKER_GPU_ARGS:---gpus all}"

case "$MODE" in
  smoke)
    MODELS="${MODELS:-regex}"
    SIZES="${SIZES:-128,512}"
    REPEATS="${REPEATS:-2}"
    QUALITY_LIMIT="${QUALITY_LIMIT:-3}"
    ;;
  opf)
    MODELS="${MODELS:-opf}"
    SIZES="${SIZES:-256,1024,4096}"
    REPEATS="${REPEATS:-3}"
    ;;
  hf)
    MODELS="${MODELS:-${1:-qwen3-0.8b}}"
    SIZES="${SIZES:-256,1024}"
    REPEATS="${REPEATS:-3}"
    QUALITY_LIMIT="${QUALITY_LIMIT:-5}"
    ;;
  all)
    MODELS="${MODELS:-regex,opf}"
    SIZES="${SIZES:-256,1024,4096}"
    REPEATS="${REPEATS:-3}"
    ;;
  *)
    echo "Usage: scripts/run_docker_cuda.sh [smoke|opf|hf|all]" >&2
    echo "Override with env vars: MODELS, SIZES, REPEATS, QUALITY_LIMIT, DEVICE, IMAGE, BUILD, DOCKER_GPU_ARGS" >&2
    exit 2
    ;;
esac

if [[ "$BUILD" == "1" || "$BUILD" == "true" ]]; then
  docker build -f Dockerfile.cuda -t "$IMAGE" .
elif [[ "$BUILD" == "auto" ]] && ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -f Dockerfile.cuda -t "$IMAGE" .
fi

mkdir -p results
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_BASE="results/${MODE}-cuda-${STAMP}"
QUALITY_ARGS=()
if [[ -n "${QUALITY_LIMIT:-}" ]]; then
  QUALITY_ARGS=(--quality-limit "$QUALITY_LIMIT")
fi

OPF_ARGS=()
OPF_MOUNT=()
if [[ -n "${PII_BENCH_OPF_CHECKPOINT:-}" ]]; then
  OPF_MOUNT=(-v "${PII_BENCH_OPF_CHECKPOINT}:/models/opf:ro")
  OPF_ARGS=(--opf-checkpoint /models/opf)
fi

echo "Running Docker CUDA mode=$MODE models=$MODELS sizes=$SIZES repeats=$REPEATS device=$DEVICE image=$IMAGE"
if [[ -n "${PII_BENCH_POWER_WATTS:-}" ]]; then
  echo "Power budget: ${PII_BENCH_POWER_WATTS} W"
fi
read -r -a GPU_ARGS <<< "$DOCKER_GPU_ARGS"
CMD=(docker run --rm)
if ((${#GPU_ARGS[@]})); then
  CMD+=("${GPU_ARGS[@]}")
fi
CMD+=(-e "HF_TOKEN=${HF_TOKEN:-}")
CMD+=(-e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}")
CMD+=(-e TOKENIZERS_PARALLELISM=false)
CMD+=(-e "PII_BENCH_POWER_WATTS=${PII_BENCH_POWER_WATTS:-}")
CMD+=(-e "PII_BENCH_POWER_NOTE=${PII_BENCH_POWER_NOTE:-}")
CMD+=(-v "$ROOT_DIR/results:/app/results")
CMD+=(-v "$HF_CACHE:/root/.cache/huggingface")
if ((${#OPF_MOUNT[@]})); then
  CMD+=("${OPF_MOUNT[@]}")
fi
CMD+=("$IMAGE")
CMD+=(python -m pii_benchmark.cli run)
CMD+=(--models "$MODELS")
CMD+=(--sizes "$SIZES")
CMD+=(--repeats "$REPEATS")
CMD+=(--device "$DEVICE")
if ((${#QUALITY_ARGS[@]})); then
  CMD+=("${QUALITY_ARGS[@]}")
fi
if ((${#OPF_ARGS[@]})); then
  CMD+=("${OPF_ARGS[@]}")
fi
CMD+=(--out "${OUT_BASE}.jsonl")
CMD+=(--markdown-report "${OUT_BASE}.md")
CMD+=(--csv "${OUT_BASE}.csv")

"${CMD[@]}"

echo
echo "Done:"
echo "  ${OUT_BASE}.jsonl"
echo "  ${OUT_BASE}.md"
echo "  ${OUT_BASE}.csv"

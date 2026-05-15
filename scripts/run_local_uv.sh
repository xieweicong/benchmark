#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

is_valid_opf_checkpoint() {
  local path="$1"
  [[ -f "$path/config.json" ]]
}

MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 127
fi

mkdir -p results
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEVICE="${DEVICE:-auto}"
UV_PYTHON="${UV_PYTHON:-3.12}"
IS_APPLE_SILICON=0
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  IS_APPLE_SILICON=1
fi
QUALITY_ARGS=()
EXTRAS=()
RUN_ARGS=()

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
    EXTRAS=(--extra opf --extra system)
    DEFAULT_OPF_CHECKPOINT="$HOME/Library/Application Support/PII Shield/model/privacy_filter"
    OPF_CHECKPOINT="${PII_BENCH_OPF_CHECKPOINT:-}"
    if [[ -z "$OPF_CHECKPOINT" ]] && is_valid_opf_checkpoint "$DEFAULT_OPF_CHECKPOINT"; then
      OPF_CHECKPOINT="$DEFAULT_OPF_CHECKPOINT"
    fi
    if [[ -n "$OPF_CHECKPOINT" ]]; then
      RUN_ARGS+=(--opf-checkpoint "$OPF_CHECKPOINT")
    fi
    ;;
  hf)
    MODELS="${MODELS:-${1:-qwen3-0.8b}}"
    SIZES="${SIZES:-256,1024}"
    REPEATS="${REPEATS:-3}"
    QUALITY_LIMIT="${QUALITY_LIMIT:-5}"
    EXTRAS=(--extra hf --extra system)
    ;;
  mlx)
    MODELS="${MODELS:-${1:-mlx-opf-bf16}}"
    SIZES="${SIZES:-256,1024,4096}"
    REPEATS="${REPEATS:-3}"
    QUALITY_LIMIT="${QUALITY_LIMIT:-5}"
    DEVICE="mps"
    EXTRAS=(--extra mlx --extra system)
    ;;
  all)
    if [[ "$IS_APPLE_SILICON" == "1" ]]; then
      MODELS="${MODELS:-regex,opf,mlx-opf-bf16}"
      EXTRAS=(--extra opf --extra hf --extra mlx --extra system)
    else
      MODELS="${MODELS:-regex,opf}"
      EXTRAS=(--extra opf --extra hf --extra system)
    fi
    SIZES="${SIZES:-256,1024,4096}"
    REPEATS="${REPEATS:-3}"
    DEFAULT_OPF_CHECKPOINT="$HOME/Library/Application Support/PII Shield/model/privacy_filter"
    OPF_CHECKPOINT="${PII_BENCH_OPF_CHECKPOINT:-}"
    if [[ -z "$OPF_CHECKPOINT" ]] && is_valid_opf_checkpoint "$DEFAULT_OPF_CHECKPOINT"; then
      OPF_CHECKPOINT="$DEFAULT_OPF_CHECKPOINT"
    fi
    if [[ -n "$OPF_CHECKPOINT" ]]; then
      RUN_ARGS+=(--opf-checkpoint "$OPF_CHECKPOINT")
    fi
    ;;
  *)
    echo "Usage: ./run.sh [smoke|opf|hf|mlx|all]" >&2
    echo "Override with env vars: MODELS, SIZES, REPEATS, QUALITY_LIMIT, DEVICE, PII_BENCH_OPF_CHECKPOINT" >&2
    exit 2
    ;;
esac

if [[ -n "${QUALITY_LIMIT:-}" ]]; then
  QUALITY_ARGS=(--quality-limit "$QUALITY_LIMIT")
fi

OUT_BASE="results/${MODE}-${STAMP}"

echo "Running mode=$MODE models=$MODELS sizes=$SIZES repeats=$REPEATS device=$DEVICE python=$UV_PYTHON"
if [[ -n "${PII_BENCH_POWER_WATTS:-}" ]]; then
  echo "Power budget: ${PII_BENCH_POWER_WATTS} W"
fi
CMD=(uv run --python "$UV_PYTHON")
if ((${#EXTRAS[@]})); then
  CMD+=("${EXTRAS[@]}")
fi
CMD+=(python -m pii_benchmark.cli run)
CMD+=(--models "$MODELS")
CMD+=(--sizes "$SIZES")
CMD+=(--repeats "$REPEATS")
CMD+=(--device "$DEVICE")
if ((${#QUALITY_ARGS[@]})); then
  CMD+=("${QUALITY_ARGS[@]}")
fi
if ((${#RUN_ARGS[@]})); then
  CMD+=("${RUN_ARGS[@]}")
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

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-opf}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ ! -f /etc/nv_tegra_release ]] && ! command -v tegrastats >/dev/null 2>&1; then
  echo "Warning: this does not look like a Jetson/L4T device. Continuing anyway." >&2
fi

mkdir -p results

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
QUALITY_ARGS=()
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
    export OPF_MOE_TRITON="${OPF_MOE_TRITON:-0}"
    if [[ -n "${PII_BENCH_OPF_CHECKPOINT:-}" ]]; then
      RUN_ARGS+=(--opf-checkpoint "$PII_BENCH_OPF_CHECKPOINT")
    fi
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
    export OPF_MOE_TRITON="${OPF_MOE_TRITON:-0}"
    if [[ -n "${PII_BENCH_OPF_CHECKPOINT:-}" ]]; then
      RUN_ARGS+=(--opf-checkpoint "$PII_BENCH_OPF_CHECKPOINT")
    fi
    ;;
  *)
    echo "Usage: scripts/run_jetson.sh [smoke|opf|hf|all]" >&2
    echo "Override with env vars: MODELS, SIZES, REPEATS, QUALITY_LIMIT, DEVICE, PYTHON_BIN, PII_BENCH_POWER_WATTS" >&2
    exit 2
    ;;
esac

if [[ -n "${QUALITY_LIMIT:-}" ]]; then
  QUALITY_ARGS=(--quality-limit "$QUALITY_LIMIT")
fi

OUT_BASE="results/${MODE}-jetson-${STAMP}"

echo "Running Jetson mode=$MODE models=$MODELS sizes=$SIZES repeats=$REPEATS device=$DEVICE python=$PYTHON_BIN"
echo "Tip: set PII_BENCH_POWER_WATTS to the measured or configured power budget for tok/s/W."

CMD=("$PYTHON_BIN" -m pii_benchmark.cli run)
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


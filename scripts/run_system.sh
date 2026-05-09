#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi

mkdir -p results

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEVICE="${DEVICE:-auto}"
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
    DEFAULT_OPF_CHECKPOINT="$HOME/Library/Application Support/PII Shield/model/privacy_filter"
    OPF_CHECKPOINT="${PII_BENCH_OPF_CHECKPOINT:-}"
    if [[ -z "$OPF_CHECKPOINT" && -d "$DEFAULT_OPF_CHECKPOINT" ]]; then
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
    ;;
  all)
    MODELS="${MODELS:-regex,opf}"
    SIZES="${SIZES:-256,1024,4096}"
    REPEATS="${REPEATS:-3}"
    export OPF_MOE_TRITON="${OPF_MOE_TRITON:-0}"
    DEFAULT_OPF_CHECKPOINT="$HOME/Library/Application Support/PII Shield/model/privacy_filter"
    OPF_CHECKPOINT="${PII_BENCH_OPF_CHECKPOINT:-}"
    if [[ -z "$OPF_CHECKPOINT" && -d "$DEFAULT_OPF_CHECKPOINT" ]]; then
      OPF_CHECKPOINT="$DEFAULT_OPF_CHECKPOINT"
    fi
    if [[ -n "$OPF_CHECKPOINT" ]]; then
      RUN_ARGS+=(--opf-checkpoint "$OPF_CHECKPOINT")
    fi
    ;;
  *)
    echo "Usage: scripts/run_system.sh [smoke|opf|hf|all]" >&2
    echo "Override with env vars: MODELS, SIZES, REPEATS, QUALITY_LIMIT, DEVICE, PYTHON_BIN, PII_BENCH_OPF_CHECKPOINT" >&2
    exit 2
    ;;
esac

if [[ -n "${QUALITY_LIMIT:-}" ]]; then
  QUALITY_ARGS=(--quality-limit "$QUALITY_LIMIT")
fi

OUT_BASE="results/${MODE}-system-${STAMP}"

echo "Running system mode=$MODE models=$MODELS sizes=$SIZES repeats=$REPEATS device=$DEVICE python=$PYTHON_BIN"
if [[ -n "${PII_BENCH_POWER_WATTS:-}" ]]; then
  echo "Power budget: ${PII_BENCH_POWER_WATTS} W"
fi

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


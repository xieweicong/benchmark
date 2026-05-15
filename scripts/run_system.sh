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

mkdir -p results

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEVICE="${DEVICE:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
IS_APPLE_SILICON=0
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  IS_APPLE_SILICON=1
fi
QUALITY_ARGS=()
RUN_ARGS=()
INSTALL_EXTRAS=""

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
    INSTALL_EXTRAS="opf,system"
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
    INSTALL_EXTRAS="hf,system"
    ;;
  mlx)
    MODELS="${MODELS:-${1:-mlx-opf-bf16}}"
    SIZES="${SIZES:-256,1024,4096}"
    REPEATS="${REPEATS:-3}"
    QUALITY_LIMIT="${QUALITY_LIMIT:-5}"
    DEVICE="mps"
    INSTALL_EXTRAS="mlx,system"
    ;;
  all)
    if [[ "$IS_APPLE_SILICON" == "1" ]]; then
      MODELS="${MODELS:-regex,opf,mlx-opf-bf16}"
      INSTALL_EXTRAS="opf,hf,mlx,system"
    else
      MODELS="${MODELS:-regex,opf}"
      INSTALL_EXTRAS="opf,hf,system"
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
    echo "Usage: scripts/run_system.sh [smoke|opf|hf|mlx|all]" >&2
    echo "Override with env vars: MODELS, SIZES, REPEATS, QUALITY_LIMIT, DEVICE, PYTHON_BIN, PII_BENCH_OPF_CHECKPOINT" >&2
    exit 2
    ;;
esac

if [[ -n "${QUALITY_LIMIT:-}" ]]; then
  QUALITY_ARGS=(--quality-limit "$QUALITY_LIMIT")
fi

OUT_BASE="results/${MODE}-system-${STAMP}"

is_kaggle() {
  [[ -d /kaggle || -n "${KAGGLE_URL_BASE:-}" || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]]
}

module_available() {
  "$PYTHON_BIN" - "$1" <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PY
}

need_install() {
  case "$MODE" in
    opf)
      module_available opf || return 0
      ;;
    hf)
      module_available torch || return 0
      module_available transformers || return 0
      ;;
    mlx)
      module_available mlx || return 0
      module_available mlx_embeddings || return 0
      ;;
    all)
      module_available opf || return 0
      module_available torch || return 0
      module_available transformers || return 0
      module_available mlx || return 0
      module_available mlx_embeddings || return 0
      ;;
  esac
  return 1
}

AUTO_INSTALL="${PII_BENCH_AUTO_INSTALL:-}"
if [[ -z "$AUTO_INSTALL" ]]; then
  if is_kaggle; then
    AUTO_INSTALL=1
  else
    AUTO_INSTALL=0
  fi
fi

if [[ "$AUTO_INSTALL" == "1" || "$AUTO_INSTALL" == "true" ]]; then
  if [[ -n "$INSTALL_EXTRAS" ]] && need_install; then
    echo "Installing missing dependencies into system Python: .[$INSTALL_EXTRAS]"
    "$PYTHON_BIN" -m pip install -e ".[$INSTALL_EXTRAS]"
  fi
fi

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

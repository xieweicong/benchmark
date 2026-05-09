#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUNNER="${PII_BENCH_RUNNER:-auto}"

is_jetson() {
  if [[ -f /etc/nv_tegra_release ]]; then
    return 0
  fi
  if [[ -r /proc/device-tree/model ]] && tr -d '\0' </proc/device-tree/model | grep -qiE 'jetson|tegra|orin'; then
    return 0
  fi
  if command -v tegrastats >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

has_cuda_server_stack() {
  [[ "$(uname -s)" == "Linux" ]] || return 1
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  command -v docker >/dev/null 2>&1 || return 1
  docker info >/dev/null 2>&1 || return 1
  return 0
}

is_kaggle() {
  if [[ -d /kaggle ]]; then
    return 0
  fi
  if [[ -n "${KAGGLE_URL_BASE:-}" || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]]; then
    return 0
  fi
  return 1
}

case "$RUNNER" in
  auto)
    if is_jetson; then
      echo "Auto runner: Jetson/L4T detected."
      exec "$ROOT_DIR/scripts/run_jetson.sh" "$@"
    fi
    if is_kaggle; then
      echo "Auto runner: Kaggle/notebook environment detected; using system Python runner."
      exec "$ROOT_DIR/scripts/run_system.sh" "$@"
    fi
    if has_cuda_server_stack; then
      echo "Auto runner: Linux NVIDIA GPU + Docker detected."
      exec "$ROOT_DIR/scripts/run_docker_cuda.sh" "$@"
    fi
    echo "Auto runner: using local uv runner."
    exec "$ROOT_DIR/scripts/run_local_uv.sh" "$@"
    ;;
  local|uv)
    exec "$ROOT_DIR/scripts/run_local_uv.sh" "$@"
    ;;
  system|notebook|kaggle)
    exec "$ROOT_DIR/scripts/run_system.sh" "$@"
    ;;
  docker|docker-cuda|cuda-docker)
    exec "$ROOT_DIR/scripts/run_docker_cuda.sh" "$@"
    ;;
  jetson)
    exec "$ROOT_DIR/scripts/run_jetson.sh" "$@"
    ;;
  *)
    echo "Unknown PII_BENCH_RUNNER=$RUNNER" >&2
    echo "Use one of: auto, local, system, docker-cuda, jetson" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

# Rebuild unified TRT engines (feature_runner + post_gwc_runner) from ONNX.
# Default behavior: build only missing .engine files.
# Use --force to rebuild all.
#
# Usage:
#   bash scripts/rebuild_unified_engines.sh
#   bash scripts/rebuild_unified_engines.sh --force
#   bash scripts/rebuild_unified_engines.sh engines/480x320_8iter_unified engines/640x448_8iter_unified

FORCE=0
USE_CUDAGRAPH=1
declare -a TARGETS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --no-cudagraph)
      USE_CUDAGRAPH=0
      shift
      ;;
    -*)
      echo "Unknown option: $1"
      exit 1
      ;;
    *)
      TARGETS+=("$1")
      shift
      ;;
  esac
done

if ! command -v trtexec >/dev/null 2>&1; then
  echo "ERROR: trtexec not found in PATH."
  echo "Install TensorRT tools or source your Jetson environment first."
  exit 1
fi

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  while IFS= read -r d; do
    TARGETS+=("$d")
  done < <(find engines -mindepth 1 -maxdepth 1 -type d | sort)
fi

CUDA_GRAPH_FLAG=""
if [[ "$USE_CUDAGRAPH" -eq 1 ]]; then
  CUDA_GRAPH_FLAG="--useCudaGraph"
fi

build_one() {
  local dir="$1"
  local feat_onnx="$dir/feature_runner.onnx"
  local post_onnx="$dir/post_gwc_runner.onnx"
  local feat_engine="$dir/feature_runner.engine"
  local post_engine="$dir/post_gwc_runner.engine"

  if [[ ! -f "$feat_onnx" || ! -f "$post_onnx" ]]; then
    return 0
  fi

  echo
  echo "=== $dir ==="

  if [[ "$FORCE" -eq 1 || ! -f "$feat_engine" ]]; then
    echo "[build] feature_runner.engine"
    trtexec --onnx="$feat_onnx" --saveEngine="$feat_engine" --fp16 $CUDA_GRAPH_FLAG
  else
    echo "[skip ] feature_runner.engine exists"
  fi

  if [[ "$FORCE" -eq 1 || ! -f "$post_engine" ]]; then
    echo "[build] post_gwc_runner.engine"
    trtexec --onnx="$post_onnx" --saveEngine="$post_engine" --fp16 $CUDA_GRAPH_FLAG
  else
    echo "[skip ] post_gwc_runner.engine exists"
  fi
}

for d in "${TARGETS[@]}"; do
  if [[ -d "$d" ]]; then
    build_one "$d"
  fi
done

echo
echo "Done."

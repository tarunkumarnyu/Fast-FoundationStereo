#!/bin/bash
# Benchmark all single-engine configs against recorded bags.
# Usage: bash scripts/benchmark_bag.sh [bag_path]
#
# Uses Python mcap reader to decompress and publish (bypasses ros2 bag play yaml-cpp bug).

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENGINE_BASE="$REPO_DIR/engines"
BAG_PATH="${1:-/home/race6/falcon/experiments/yang_bags/race6_bag_20251024_174028.mcap}"
RESULTS_DIR="$REPO_DIR/benchmark_results"
NS="/race6/cam1"

# Source ROS2
source /opt/ros/humble/setup.bash
source "$REPO_DIR/cpp/install/setup.bash" 2>/dev/null || {
  echo "ERROR: C++ install not found. Rebuild first."
  exit 1
}

mkdir -p "$RESULTS_DIR"

# All single-engine configs
CONFIGS=(
  "448x256_md128_single"
  "448x256_md96_single"
  "448x256_md64_single"
  "320x192_md128_single"
  "320x192_md96_single"
  "320x192_md64_single"
  "288x160_md96_single"
  "288x160_md64_single"
  "224x128_md64_single"
  "224x128_md32_single"
)

BAG_NAME="$(basename "$BAG_PATH" .mcap)"
echo "=== FFS Benchmark: $(date) ==="
echo "Bag: $BAG_PATH ($BAG_NAME)"
echo "Results: $RESULTS_DIR"
echo ""

# Summary file
SUMMARY="$RESULTS_DIR/summary_${BAG_NAME}.csv"
echo "config,resolution,max_disp,avg_fps,avg_gpu_ms,frames_processed" > "$SUMMARY"

# Cleanup
PIDS_TO_KILL=()
cleanup() {
  for p in "${PIDS_TO_KILL[@]}"; do
    kill "$p" 2>/dev/null
    wait "$p" 2>/dev/null
  done
  PIDS_TO_KILL=()
}
trap cleanup EXIT

for cfg in "${CONFIGS[@]}"; do
  ENGINE_DIR="$ENGINE_BASE/$cfg"
  ENGINE_FILE="$ENGINE_DIR/foundation_stereo.engine"

  if [ ! -f "$ENGINE_FILE" ]; then
    echo "SKIP $cfg — no engine file"
    continue
  fi

  echo "========================================"
  echo "Benchmarking: $cfg"
  echo "========================================"

  LOG_FILE="$RESULTS_DIR/${cfg}_${BAG_NAME}.log"
  rm -f "$LOG_FILE"

  # Start C++ FFS node (run binary directly to avoid ros2 run wrapper PID issues)
  FFS_BIN="$REPO_DIR/cpp/install/fast_ffs/lib/fast_ffs/live_ffs_node"
  "$FFS_BIN" --ros-args \
    -p engine_dir:="$ENGINE_DIR" \
    -p ns:="$NS" \
    > "$LOG_FILE" 2>&1 &
  FFS_PID=$!
  PIDS_TO_KILL=($FFS_PID)
  echo "  FFS node PID: $FFS_PID"

  # Wait for engine load + warmup
  sleep 10

  # Check if FFS node is still alive
  if ! kill -0 $FFS_PID 2>/dev/null; then
    echo "  ERROR: FFS node died during startup. Log:"
    tail -5 "$LOG_FILE"
    echo "$cfg,error,error,0,0,0" >> "$SUMMARY"
    PIDS_TO_KILL=()
    echo ""
    continue
  fi

  # Play bag with Python mcap reader (decompresses and publishes raw images)
  echo "  Playing bag (Python mcap reader)..."
  python3 "$REPO_DIR/scripts/play_bag_decompressed.py" \
    --bag "$BAG_PATH" --ns "$NS" --rate 1.0 2>&1 | tee "$RESULTS_DIR/${cfg}_play.log"

  echo "  Bag playback done. Draining pipeline..."
  sleep 5

  # Stop FFS node
  kill $FFS_PID 2>/dev/null; wait $FFS_PID 2>/dev/null
  PIDS_TO_KILL=()

  # Extract stats from log
  LAST_FPS=$(grep -oP '[\d.]+Hz' "$LOG_FILE" | tail -1 | sed 's/Hz//')
  AVG_GPU=$(grep -oP 'TOTAL=[\d.]+' "$LOG_FILE" | tail -1 | sed 's/TOTAL=//')
  LAST_FRAME=$(grep -oP 'Frame \d+' "$LOG_FILE" | tail -1 | grep -oP '\d+')

  # Parse resolution and max_disp from config name
  RES=$(echo "$cfg" | sed 's/_md.*//;s/_single//')
  MD=$(echo "$cfg" | grep -oP 'md\d+' | sed 's/md//')

  echo "  Result: ${LAST_FPS:-?}Hz, GPU=${AVG_GPU:-?}ms, frames=${LAST_FRAME:-?}"
  echo "$cfg,$RES,$MD,${LAST_FPS:-0},${AVG_GPU:-0},${LAST_FRAME:-0}" >> "$SUMMARY"

  echo ""
  sleep 2
done

echo "=== Benchmark complete: $(date) ==="
echo ""
echo "Summary:"
column -t -s',' "$SUMMARY"
echo ""
echo "Full results in: $RESULTS_DIR"

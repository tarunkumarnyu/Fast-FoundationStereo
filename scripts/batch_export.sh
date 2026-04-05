#!/bin/bash
# Batch export: ONNX + TRT engines for all resolution/max_disp configs.
# Run from repo root: bash scripts/batch_export.sh
# This exports single ONNX (full model) and builds TRT engines for each config.
# Walk away — takes several hours on Jetson Orin NX.

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHTS="$REPO_DIR/weights/20-30-48/model_best_bp2_serialize.pth"
ENGINE_BASE="$REPO_DIR/engines"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
ITERS=8

# All configs: NAME HEIGHT WIDTH MAX_DISP
CONFIGS=(
  "448x256_md128  256 448 128"
  "448x256_md96   256 448  96"
  "448x256_md64   256 448  64"
  "320x192_md128  192 320 128"
  "320x192_md96   192 320  96"
  "320x192_md64   192 320  64"
  "288x160_md96   160 288  96"
  "288x160_md64   160 288  64"
  "224x128_md64   128 224  64"
  "224x128_md32   128 224  32"
)

LOGFILE="$ENGINE_BASE/batch_export.log"
mkdir -p "$ENGINE_BASE"
echo "=== Batch export started: $(date) ===" | tee "$LOGFILE"
echo "Weights: $WEIGHTS" | tee -a "$LOGFILE"
echo "Configs: ${#CONFIGS[@]}" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

TOTAL=${#CONFIGS[@]}
IDX=0

for cfg in "${CONFIGS[@]}"; do
  read -r NAME H W MD <<< "$cfg"
  IDX=$((IDX + 1))
  DIR="$ENGINE_BASE/${NAME}_single"
  ONNX_FILE="$DIR/foundation_stereo.onnx"
  ENGINE_FILE="$DIR/foundation_stereo.engine"

  echo "[$IDX/$TOTAL] $NAME: ${W}x${H}, max_disp=$MD" | tee -a "$LOGFILE"

  # Skip if engine already exists
  if [ -f "$ENGINE_FILE" ]; then
    echo "  Engine exists, skipping." | tee -a "$LOGFILE"
    continue
  fi

  # Step 1: Export ONNX (also exports feature/post for fallback)
  if [ ! -f "$ONNX_FILE" ]; then
    echo "  Exporting ONNX..." | tee -a "$LOGFILE"
    cd "$REPO_DIR"
    python3 scripts/make_onnx.py \
      --model_dir "$WEIGHTS" \
      --save_path "$DIR" \
      --height "$H" --width "$W" \
      --valid_iters "$ITERS" \
      --max_disp "$MD" \
      --single_onnx 2>&1 | tee -a "$LOGFILE"
    echo "  ONNX done." | tee -a "$LOGFILE"
  else
    echo "  ONNX exists, skipping export." | tee -a "$LOGFILE"
  fi

  # Step 2: Build TRT engine (fp16)
  echo "  Building TRT engine (fp16)..." | tee -a "$LOGFILE"
  $TRTEXEC \
    --onnx="$ONNX_FILE" \
    --saveEngine="$ENGINE_FILE" \
    --fp16 2>&1 | tee -a "$LOGFILE"
  echo "  Engine built: $ENGINE_FILE" | tee -a "$LOGFILE"

  echo "" | tee -a "$LOGFILE"
done

echo "=== Batch export finished: $(date) ===" | tee -a "$LOGFILE"
echo ""
echo "All engines saved under: $ENGINE_BASE/*_single/"
echo "Log: $LOGFILE"

# Fast-FoundationStereo — ARPL Live Depth Setup

Real-time stereo depth from RealSense D455 IR stereo using Fast-FoundationStereo TRT on Jetson Orin NX.

## Prerequisites

- Jetson Orin NX (16GB) with JetPack 6.x (TensorRT 10, CUDA 12)
- ROS2 Humble
- RealSense D455 with `realsense2_camera` ROS2 package
- Python packages: `torch`, `omegaconf`, `pyyaml`, `opencv-contrib-python-headless`, `triton`

```bash
pip install opencv-contrib-python-headless cupy-cuda12x
```

## Setup

### 1. Clone this repo (outside falcon workspace)

```bash
cd ~/falcon/src
git clone git@github.com:tarunkumarnyu/Fast-FoundationStereo.git
cd Fast-FoundationStereo
```

### 2. Download model weights

Download from [Google Drive](https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link) and place under `weights/`:

```
weights/20-30-48/model_best_bp2_serialize.pth
```

### 3. Export ONNX models

Choose your resolution/iteration config. Higher res + more iters = better quality but slower.

```bash
# Recommended: 480x320, 4 iterations (~13Hz on Orin NX standard, ~20Hz on Super Mode)
python3 scripts/make_onnx.py \
  --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
  --height 320 --width 480 --valid_iters 4 --max_disp 128 --low_memory 0 \
  --save_path engines/480x320_4iter/

# Fast: 320x224, 3 iterations (~29Hz)
python3 scripts/make_onnx.py \
  --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
  --height 224 --width 320 --valid_iters 3 --max_disp 96 --low_memory 0 \
  --save_path engines/320x224_3iter/

# Max quality: 640x448, 8 iterations (~7Hz)
python3 scripts/make_onnx.py \
  --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
  --height 448 --width 640 --valid_iters 8 --max_disp 128 --low_memory 0 \
  --save_path engines/640x448_8iter/
```

**IMPORTANT:** After export, add `image_size` and `cv_group` to each `engines/*/onnx.yaml`:
```yaml
cv_group: 8
image_size:
- <height>
- <width>
```

### 4. Build TRT engines (hardware-specific, must be done on each device)

```bash
# Feature engine
trtexec --onnx=engines/480x320_4iter/feature_runner.onnx \
  --saveEngine=engines/480x320_4iter/feature_runner.engine \
  --fp16 --useCudaGraph

# Post engine (takes 3-10 min depending on size)
trtexec --onnx=engines/480x320_4iter/post_runner.onnx \
  --saveEngine=engines/480x320_4iter/post_runner.engine \
  --fp16 --useCudaGraph
```

Repeat for each resolution config you want.

### 5. NVIDIA Super Mode (Super Hadron carrier board only)

If using Connecttech Super Hadron-DM carrier board, enable Super Mode for ~50% more GPU performance (25W → 40W, 100 TOPS → 157 TOPS):

```bash
sudo nvpmodel -m 0   # Max performance mode
sudo jetson_clocks    # Lock clocks to max
```

This can push 480x320 4-iter from ~13Hz to ~20Hz.

## Running

### Start RealSense (IR stereo only)

```bash
ros2 launch realsense2_camera rs_launch.py \
  camera_name:=cam1 camera_namespace:=<DRONE_NAME> \
  enable_infra1:=true enable_infra2:=true \
  enable_color:=false enable_depth:=false \
  infra_width:=848 infra_height:=480 infra_fps:=30
```

### Start live depth node

```bash
cd ~/falcon/src/Fast-FoundationStereo
python3 scripts/live_ffs.py --ros-args \
  -p engine_dir:=$(pwd)/engines/480x320_4iter \
  -p ns:=/<DRONE_NAME>/cam1
```

Replace `<DRONE_NAME>` with your drone's namespace (e.g., `race16`, `quadrotor`).

### Published topics

| Topic | Type | Description |
|-------|------|-------------|
| `/ffs/disp_viz/compressed` | CompressedImage | Turbo colormap depth visualization |
| `/ffs/disp_gray/compressed` | CompressedImage | Grayscale depth (white=close, black=far) |

### Visualize

```bash
# rqt (needs display or X forwarding)
rqt_image_view /ffs/disp_viz/compressed

# Foxglove (remote, install foxglove-bridge first)
sudo apt install ros-humble-foxglove-bridge
ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765
# Then connect Foxglove Studio to ws://<DRONE_IP>:8765
```

## Available engine configs

| Config | Resolution | Iters | max_disp | FPS (Orin NX) | FPS (Super Mode est.) | Quality |
|--------|-----------|-------|----------|--------------|----------------------|---------|
| 320x224_3iter | 320×224 | 3 | 96 | ~29Hz | ~42Hz | Good |
| 480x320_2iter | 480×320 | 2 | 128 | ~14Hz | ~21Hz | Better |
| 480x320_4iter | 480×320 | 4 | 128 | ~13Hz | ~20Hz | Great |
| 640x448_8iter | 640×448 | 8 | 128 | ~7Hz | ~10Hz | Best |

## Architecture

```
RealSense D455 IR stereo (848x480 @ 30Hz)
    ↓ resize
TRT Feature Engine (stream_feat) → GWC Volume (Triton kernel)
    ↓ pipelined
TRT Post Engine (stream_post) → disparity map
    ↓
Guided upsampling (cv2.ximgproc.guidedFilter with IR edge guide)
    ↓
Percentile-based TURBO colormap → CompressedImage publish
```

### Key optimizations

1. **Pipelined TRT inference** — feature extraction and post processing overlap on separate CUDA streams
2. **CompressedImage publish** — JPEG (~8KB) instead of raw Image (~460KB), saves 100ms+
3. **Independent subscribers** — no message_filters sync (avoids 10-20% frame drops)
4. **GC disabled** — eliminates 100-200ms Python GC pauses
5. **CUDA stream-aware H2D** — CPU→GPU transfers on correct stream, avoids implicit sync
6. **Pre-allocated GPU buffers** — no per-frame CUDA malloc

### Key files

- `scripts/live_ffs.py` — Main live depth ROS2 node
- `core/pipelined_runner.py` — Pipelined TRT inference engine
- `scripts/make_onnx.py` — ONNX export script
- `scripts/benchmark_trt.py` — TRT benchmarking
- `engines/*/onnx.yaml` — Per-config metadata

# Installation Guide

This guide walks through setting up the starVLA development environment from scratch.

## Prerequisites

- Linux (tested on Ubuntu 20.04+)
- NVIDIA GPU with CUDA 12.x driver (tested with CUDA 12.8)
- Conda (Miniconda or Anaconda)
- `git`, `cmake`, `make` (for building decord from source)
- `ffmpeg` available on `PATH` (used by dataset/video preprocessing scripts)

Recommended Ubuntu system packages for the full deploy stack:

```bash
sudo apt update
sudo apt install -y \
    git cmake make ffmpeg \
    libgl1 libglib2.0-0 \
    libusb-1.0-0 libusb-1.0-0-dev
```

These packages cover the most common runtime issues for OpenCV / video tooling and RealSense-related USB access. Hardware-specific drivers, udev rules, and vendor runtime services may still be required.

Check your CUDA driver:
```bash
nvidia-smi    # should show driver version >= 535
nvcc -V       # CUDA toolkit version
```

## Step 1: Create Conda Environment

```bash
conda create -n starVLA python=3.10 -y
conda activate starVLA
```

## Step 2: Install PyTorch

Install PyTorch 2.7.0 with CUDA 12.8 support:
```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

## Step 3: Install FlashAttention

```bash
pip install flash-attn --no-build-isolation --no-cache-dir
```

> **Troubleshooting:** If you see `Invalid cross-device link` errors, add `--no-cache-dir`. flash-attn must match your CUDA toolkit and torch versions.

## Step 4: Install Python Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

Install external editable source dependencies from the repository parent directory, not inside this repository. `pip install -e .` above installs starVLA itself from the current checkout; the editable installs below clone their own repositories under `../`.

`requirements.txt` now includes both the common deploy-time Python packages and the full deploy / hardware extras used by the current deployment stack:

- `opencv-python`
- `draccus`
- `pynput`
- `pyserial`
- `imageio-ffmpeg`
- `lerobot==0.3.4` (overridden by the editable source install in Step 6)
- `viser==1.0.24`
- `pyrealsense2==2.57.7.10387`
- `ur-rtde==1.6.3`
- `dynamixel-sdk==4.0.3`

This makes `pip install -r requirements.txt` target a full deploy environment by default. Hardware-specific packages may still require matching drivers, USB permissions, and vendor runtime support on the host machine.

If you are setting up a training-only environment, this single-file install will still pull the deploy extras. Keep that in mind when installing on minimal servers.

## Step 5: Hardware Notes

- `pyrealsense2`
  Required when using RealSense cameras. You may still need Intel RealSense system drivers / udev rules.
- `ur-rtde`
  Provides `rtde_control` and `rtde_receive` for UR robot deployment.
- `dynamixel-sdk`
  Required for Gello teleoperation.
- `lerobot`
  Used by `deployment/collect.py`, `deployment/inference.py`, `deployment/teleoperate.py`, and replay utilities. Install it from the pinned source commit in Step 6.
- `viser`
  Used by VR / visualization demo scripts.

## Step 6: Install Editable Source Dependencies

External repositories that are installed with `pip install -e .` should be cloned next to this repository under `../`, not inside the starVLA checkout. Run these commands from the starVLA repository root:

```bash
REPO_ROOT=$(pwd)
cd "$REPO_ROOT/.."

git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout d602e8169cbad9e93a4a3b3ee1dd8b332af7ebf8
pip install -e .

cd "$REPO_ROOT/.."
```

starVLA uses [decord](https://github.com/dmlc/decord) for fast video decoding. It also needs to be built from source:

```bash
cd "$REPO_ROOT/.."
git clone --recursive https://github.com/dmlc/decord.git
cd decord
mkdir -p build && cd build
cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../python
pip install -e .

cd "$REPO_ROOT"
```

## Verification

```bash
conda activate starVLA

# Check all declared Python dependencies via spec probing
python scripts/test/test_install_dependencies.py

# Fail if any optional hardware / acceleration dependency is still missing
python scripts/test/test_install_dependencies.py --strict-optional
```

If you are on a headless server, keep the default `spec` probe mode. It checks whether a package is installed without eagerly importing modules such as `pynput` that may require an X display. To force real imports instead:

```bash
python scripts/test/test_install_dependencies.py --mode import
```

Expected results:

- `python scripts/test/test_install_dependencies.py`
  Should pass on any correctly installed environment, including headless servers.
- `python scripts/test/test_install_dependencies.py --strict-optional`
  Should pass only when the full deploy / hardware stack is installed and importable.

## Key Package Versions (Tested)

| Package | Version |
|---------|---------|
| Python | 3.10 |
| torch | 2.7.0+cu128 |
| torchvision | 0.22.0+cu128 |
| flash-attn | 2.8.3 |
| transformers | 4.57.0 |
| accelerate | 1.12.0 |
| deepspeed | 0.16.9 |
| numpy | 1.26.4 |
| imageio-ffmpeg | 0.6.0 |
| draccus | 0.10.0 |
| pynput | 1.8.1 |
| pyserial | 3.5 |
| decord | 0.6.0 (source) |
| torchcodec | 0.4.0 |
| datasets | 4.7.0 |
| pin (pinocchio) | 3.9.0 |
| lerobot | d602e8169cbad9e93a4a3b3ee1dd8b332af7ebf8 (source) |
| viser | 1.0.24 |
| pyrealsense2 | 2.57.7.10387 |
| ur-rtde | 1.6.3 |
| dynamixel-sdk | 4.0.3 |

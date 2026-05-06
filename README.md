# VGGT + Factor Graph Refinement

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Video → 3D in one command. No COLMAP. Outputs COLMAP/nerfstudio format directly.**

VGGT gives you instant poses but OOMs past 50 frames. VGGT-SLAM 2.0 fixes that but requires 4 repos, conda, a missing checkpoint, manual ffmpeg, and a 24GB GPU — and outputs nothing you can actually use downstream.

This project is the practical middle ground:

```bash
python run.py --video my_phone_video.mp4 --output scene/ --export all
```

```
scene/
├── sparse/0/          # COLMAP format → feed into ANY 3DGS pipeline
├── transforms.json    # nerfstudio format → splatfacto/nerfacto
├── scene.ply          # Colored point cloud
├── scene.splat        # Web viewer (.splat format)
├── poses_c2w.npy      # Raw poses
└── summary.json       # Timing + metrics
```

## Why not VGGT-SLAM 2.0?

| | VGGT-SLAM 2.0 (MIT SPARK) | This project |
|---|---|---|
| **Install** | conda + 4 git clones + hunt for SALAD checkpoint | `pip install gtsam` + clone this repo |
| **Input** | Pre-extracted frames (manual ffmpeg) | Direct video file |
| **GPU** | 24GB minimum (crashes on 12GB) | 8GB+ (auto chunk sizing) |
| **Output** | Viser visualization only | COLMAP, nerfstudio, PLY, .splat |
| **Metric scale** | No | Optional (MoGe-2 alignment) |
| **COLMAP export** | No ([issue #10](https://github.com/MIT-SPARK/VGGT-SLAM/issues/10) — unanswered) | Yes |
| **Point cloud save** | No ([issue #24](https://github.com/MIT-SPARK/VGGT-SLAM/issues/24)) | Yes |
| **Gaussian splatting** | No | Built-in gsplat training |
| **Mesh export** | No | TSDF fusion |
| **Stability** | SL(4) singularity crashes ([issue #5](https://github.com/MIT-SPARK/VGGT-SLAM/issues/5)) | SE(3) + robust kernels (no crashes) |

This is not a research SLAM system. It's a tool for getting usable 3D output from video.

## Demo

![3D visualization](assets/orbit.gif)

## Results

### Pose Accuracy (TUM-RGBD, 80 frames, chunk_size=8)

| Sequence | Naive Stitch ATE | Factor Graph ATE | Improvement |
|----------|-----------------|------------------|-------------|
| fr1/desk | 0.187 m | **0.031 m** | **83.5%** |
| fr1/xyz | 0.176 m | **0.060 m** | **66.1%** |
| fr1/room | 0.134 m | **0.085 m** | **36.4%** |
| fr2/desk | 0.127 m | **0.021 m** | **83.6%** |
| fr3/office | 0.105 m | **0.049 m** | **53.7%** |

### Replica Dataset (80 frames, chunk_size=8)

| Sequence | Naive Stitch ATE | Factor Graph ATE | Improvement |
|----------|-----------------|------------------|-------------|
| office0 | 0.410 m | **0.104 m** | **74.6%** |
| office1 | 0.208 m | **0.068 m** | **67.3%** |
| room0 | 0.511 m | **0.082 m** | **84.0%** |
| room1 | 0.463 m | **0.078 m** | **83.1%** |

Average improvement: **70.3%** across 9 sequences on 2 datasets.

### Scaling

| Frames | VGGT Single-Shot | Naive Stitch | Factor Graph (ours) |
|--------|-----------------|-------------|-------------------|
| 10 | 0.002 m | 0.004 m | 0.003 m |
| 30 | 0.004 m | 0.016 m | 0.004 m |
| 50 | 0.005 m | 0.033 m | 0.005 m |
| 80 | OOM | 0.042 m | **0.015 m** |
| 200 | OOM | 0.132 m | **0.043 m** |
| 300 | OOM | 0.190 m | **0.056 m** |

![Scaling](assets/scaling.png)

### Gaussian Splatting Render Quality

| Metric | Naive Poses | Factor Graph Poses | Improvement |
|--------|-----------|-------------------|-------------|
| Mean PSNR | 8.16 dB | **13.28 dB** | **+5.12 dB** |
| Training loss | ~0.50 (stuck) | ~0.16 (converged) | 3x lower |

![render comparison](assets/render_comparison.png)

## Quick Start

```bash
# Install
pip install gtsam torch torchvision scipy opencv-python-headless tqdm
git clone https://github.com/facebookresearch/vggt && cd vggt && pip install -e . && cd ..
git clone https://github.com/jashshah999/vggt-factor-refinement && cd vggt-factor-refinement

# Run on your video (outputs COLMAP + nerfstudio + PLY + .splat)
python run.py --video my_video.mp4 --output scene/ --export all

# Run on image directory
python run.py --images path/to/frames/ --output scene/

# Also train Gaussian Splatting
python run.py --video my_video.mp4 --output scene/ --train-gaussians --train-iters 3000

# Benchmark on TUM-RGBD
python benchmark_chunked.py --seq fr1/desk --chunk-size 8 --overlap 2
```

## Export Formats

| Format | Flag | Use case |
|--------|------|----------|
| COLMAP sparse | `--export colmap` | Feed into gaussian-splatting, nerfstudio, 3DGS |
| nerfstudio | `--export nerfstudio` | Direct use with splatfacto/nerfacto |
| PLY | `--export ply` | View in MeshLab, CloudCompare, Blender |
| .splat | `--export splat` | Web-based 3DGS viewers (antimatter15/splat) |
| All | `--export all` | Everything above |

## How It Works

```
Video / Image directory
    |
    v
[Frame extraction + keyframe selection]
    |
    v
[VGGT per chunk (auto-sized for your GPU)]
    |
    v
[Sim(3) overlap stitching (confidence-weighted)]
    |
    v
[iSAM2 factor graph]
  - Within-chunk odometry (confidence-weighted noise)
  - Cross-chunk overlap constraints (Cauchy robust kernel)
  - DINOv2 appearance loop closure + ORB geometric verification
  - Covisibility graph loop closure (3D voxel overlap)
    |
    v
[Optional: Sparse point BA (200 landmark joint optimization)]
    |
    v
[Export to COLMAP / nerfstudio / PLY / .splat]
    |
    v
[Optional: Train Gaussian Splatting with gsplat]
```

## Architecture

```
src/
├── chunked_pipeline.py       # Main orchestrator
├── factor_graph.py           # Batch LM optimization
├── isam2_backend.py          # iSAM2 incremental solver
├── covisibility.py           # 3D covisibility graph
├── point_ba.py               # Joint point + pose BA
├── multi_backend.py          # VGGT + MASt3R ensemble
├── keyframe_selection.py     # Smart frame selection
├── depth_fusion.py           # Multi-view depth consistency
├── trajectory_smoothing.py   # SE(3) temporal smoothing
├── uncertainty.py            # Calibrated pose uncertainty
├── loop_closure.py           # DINOv2 appearance matching
├── cross_chunk_align.py      # 3D point RANSAC alignment
├── sl4_graph.py              # SL(4) for uncalibrated cameras
├── vggt_wrapper.py           # VGGT model interface
├── metrics.py                # ATE, RPE evaluation
├── data_loaders.py           # TUM, Replica loaders
├── gaussian_render.py        # gsplat training
└── exporters/
    ├── colmap_export.py      # COLMAP sparse model (text + binary)
    ├── nerfstudio_export.py  # transforms.json
    ├── ply_export.py         # Colored point cloud
    └── splat_export.py       # .splat format for web viewers
```

## Key Features

| Feature | Description |
|---------|-------------|
| **One-command pipeline** | Video → 3D in a single command |
| **COLMAP replacement** | Direct COLMAP-format output for any 3DGS pipeline |
| **8GB GPU support** | Auto-detects VRAM and reduces chunk size |
| **iSAM2 Backend** | O(log n) incremental optimization |
| **Covisibility Graph** | Finds loop closures via shared 3D geometry |
| **Point BA** | Joint pose + landmark optimization |
| **Multi-backend** | Ensemble VGGT + MASt3R for better coverage |
| **Robust Kernels** | Cauchy/Huber M-estimators for outlier rejection |
| **Depth Fusion** | Multi-view consistency filtering |
| **Trajectory Smoothing** | Spline/Savitzky-Golay/bilateral on SE(3) |
| **Uncertainty Estimation** | Calibrated 6-DOF pose uncertainty |

## Limitations

- Not real-time (batch offline processing)
- Accuracy below dedicated SLAM systems (ORB-SLAM3, DROID-SLAM) on well-supported sequences
- Loop closure relative poses derived from stitched trajectory (circular when drift is large)
- DINOv2 may match repetitive textures incorrectly (ORB verification catches most)

## Requirements

- CUDA GPU (8GB+ with auto chunk reduction, 24GB for default settings)
- Python 3.10+
- PyTorch, GTSAM, VGGT

## License

MIT

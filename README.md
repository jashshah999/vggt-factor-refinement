# VGGT + Factor Graph Refinement

**Foundation models give you a great initialization. Factor graphs give you global consistency.**

VGGT predicts camera poses from images in a single forward pass, but it can only process ~15-30 frames at once (24GB GPU). For longer videos, you must chunk and stitch. Naive stitching accumulates drift at chunk boundaries.

This project adds a GTSAM factor graph on top of VGGT to enforce global consistency across chunks:
- Within-chunk odometry factors (tight, since VGGT is accurate locally)
- Cross-chunk overlap constraints
- Visual loop closure via feature matching
- LM optimization produces globally consistent trajectory

## Results on TUM-RGBD

**fr1/desk** (80 frames, 13 chunks of 8): **67.8% ATE reduction**

![fr1/desk](assets/tum_fr1_desk.png)

| Sequence | Naive Stitch ATE | Factor Graph ATE | Improvement |
|----------|-----------------|------------------|-------------|
| fr1/desk | 0.2905 m | **0.0934 m** | **67.8%** |
| fr1/room | 0.1820 m | **0.1691 m** | **7.1%** |
| fr1/xyz  | 0.1416 m | 0.1415 m | 0.1% |

The factor graph helps most when chunk stitching introduces drift (fr1/desk has many chunk boundary misalignments). On sequences where naive stitching already works well (fr1/xyz has smooth linear motion), the factor graph preserves accuracy without hurting.

## How It Works

```
Long video (100+ frames)
    |
    v
[Split into chunks of 8-15 frames]
    |
    v
[VGGT per chunk] --> local poses + depth + 3D point maps
    |
    v
[Naive stitch via overlap alignment] --> initial global trajectory
    |
    v
[Build GTSAM factor graph]
  - Within-chunk odometry (tight noise, VGGT is good locally)
  - Cross-chunk overlap constraints
  - Loop closures (ORB feature matching for verification)
    |
    v
[Levenberg-Marquardt optimization] --> refined global trajectory
```

## Quick Start

```bash
# Install
pip install gtsam gsplat torch
pip install -e .

# Run on TUM-RGBD
python benchmark_chunked.py --seq fr1/desk --chunk-size 8 --overlap 2

# Run on your own video
python run.py --video my_video.mp4 --output output/
```

## Why Factor Graphs?

VGGT is a feed-forward model. It processes each chunk independently with no mechanism to enforce that:
1. Overlapping frames from different chunks should agree on their 3D positions
2. The camera trajectory should form a consistent loop when revisiting locations
3. Chunk boundaries should be smooth (no jumps)

A factor graph provides all three. It takes VGGT's output as initial estimates and soft constraints, then optimizes for global consistency. The factor graph adds ~0.1s of compute on top of VGGT's inference time.

## Requirements

- CUDA GPU with 24GB+ VRAM (for VGGT)
- Python 3.10+
- GTSAM, gsplat, PyTorch, VGGT

## License

MIT

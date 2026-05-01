# VGGT + Factor Graph Refinement

**Foundation models give you a great initialization. Factor graphs give you global consistency.**

VGGT predicts camera poses from images in a single forward pass, but it can only process ~15-30 frames at once (24GB GPU). For longer videos, you must chunk and stitch. Naive stitching accumulates drift at chunk boundaries.

This project adds a GTSAM factor graph on top of VGGT to enforce global consistency across chunks:
- Within-chunk odometry factors (tight, since VGGT is accurate locally)
- Cross-chunk overlap constraints with Cauchy robust kernel
- Visual loop closure via ORB feature matching
- LM optimization produces globally consistent trajectory

## Demo

3D point cloud with camera frustums on TUM fr1/desk. Green = ground truth, blue = factor graph (ours), red = naive stitching. The factor graph trajectory closely follows ground truth while naive stitching drifts.

![3D visualization](assets/orbit.gif)

## Results

### Pose Accuracy (TUM-RGBD, 80 frames, chunk_size=8)

**fr1/desk: 89.4% ATE reduction**

![fr1/desk](assets/tum_fr1_desk.png)

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

### Scaling: VGGT Single-Shot vs Chunked + Factor Graph

VGGT single-shot gives the best accuracy but OOMs past ~50 frames on 24GB. Our factor graph pipeline scales to any sequence length while staying close to the single-shot upper bound.

![Scaling](assets/scaling.png)

| Frames | VGGT Single-Shot | Naive Stitch | Factor Graph (ours) |
|--------|-----------------|-------------|-------------------|
| 10 | 0.002 m | 0.004 m | 0.003 m |
| 30 | 0.004 m | 0.016 m | 0.004 m |
| 50 | 0.005 m | 0.033 m | 0.005 m |
| 80 | OOM | 0.042 m | **0.015 m** |
| 200 | OOM | 0.132 m | **0.043 m** |
| 300 | OOM | 0.190 m | **0.056 m** |

At 300 frames, the factor graph achieves 3.4x lower error than naive stitching, and it keeps scaling.

### Gaussian Splatting Render Quality

Better poses lead to better 3D reconstruction. Gaussians trained with factor graph poses produce sharper renders:

![render comparison](assets/render_comparison.png)

| Metric | Naive Poses | Factor Graph Poses | Improvement |
|--------|-----------|-------------------|-------------|
| Mean PSNR | 8.16 dB | **13.28 dB** | **+5.12 dB** |
| Training loss | ~0.50 (stuck) | ~0.16 (converged) | 3x lower |

The naive poses have too much drift for the Gaussians to converge. Factor graph poses are accurate enough for the splatting to produce recognizable renders.

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
  - Loop closures (ORB matching for verification, Cauchy robust kernel)
    |
    v
[Levenberg-Marquardt optimization] --> refined global trajectory
    |
    v
[Initialize Gaussians from VGGT point maps]
[Train with gsplat using refined poses]
    |
    v
Output: globally consistent poses + 3D Gaussian splat reconstruction
```

## Quick Start

```bash
# Install dependencies
pip install gtsam gsplat torch
cd ~ && git clone https://github.com/facebookresearch/vggt && cd vggt && pip install -e .
cd ~ && git clone https://github.com/jashshah999/vggt-factor-refinement && cd vggt-factor-refinement

# Benchmark on TUM-RGBD (downloads data automatically)
python benchmark_chunked.py --seq fr1/desk --chunk-size 8 --overlap 2

# Gaussian splatting comparison
python benchmark_gs.py --seq fr1/desk --train-iters 500

# Run on your own video
python run.py --video my_video.mp4 --output output/
```

## Why Factor Graphs?

VGGT processes each chunk independently with no mechanism to enforce that:
1. Overlapping frames from different chunks agree on 3D positions
2. The trajectory forms a consistent loop when revisiting locations
3. Chunk boundaries are smooth (no jumps)

A factor graph provides all three. It takes VGGT's output as initial estimates and soft constraints, then optimizes for global consistency. The optimization adds ~2s of compute on top of VGGT inference.

## Limitations

- **Non-looping trajectories at 200+ frames.** When the camera never revisits a location, loop closures can't correct accumulated drift. The factor graph helps up to ~120 frames via odometry smoothing alone, but plateaus beyond that. Sequences with actual revisits (like TUM fr1/room) work well even at 200+ frames.
- **Cross-chunk relative poses.** Loop closure relative poses are currently derived from the naive-stitched trajectory, which is circular when drift is large. Properly computing independent cross-chunk relative poses (e.g. via SL(4) alignment like VGGT-SLAM) would improve long-sequence performance.
- **Repetitive textures.** DINOv2 can match frames that look similar but are in different locations (e.g. white walls). The ORB geometric verification catches some of these but not all. A place recognition model trained specifically for loop closure (NetVLAD, CosPlace) would be more robust.

## Future Work

- [ ] Independent cross-chunk relative pose estimation via 3D point cloud alignment in a shared coordinate frame
- [ ] Learned confidence-to-covariance mapping (train a small network to predict per-frame noise from VGGT's confidence maps)
- [ ] Integration with VGGT's built-in bundle adjustment for hybrid feed-forward + optimization
- [ ] Support for other feed-forward models (DUSt3R, MASt3R, Spann3R) as the chunking frontend
- [ ] Incremental iSAM2 backend for real-time processing

## Related Work

- [VGGT](https://github.com/facebookresearch/vggt) - the feed-forward model we build on top of
- [VGGT-SLAM](https://arxiv.org/abs/2505.12549) - concurrent work using SL(4) manifold optimization for uncalibrated cameras (code not yet released)
- [GTSAM](https://github.com/borglab/gtsam) - the factor graph library powering the optimization

## Requirements

- CUDA GPU with 24GB+ VRAM (for VGGT)
- Python 3.10+
- GTSAM, gsplat, PyTorch, VGGT

## License

MIT

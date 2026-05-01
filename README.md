# VGGT + Factor Graph Refinement

**Foundation models give you a great initialization. Factor graphs give you global consistency. The combination is better than either alone.**

VGGT predicts camera poses in a single forward pass but has no mechanism for enforcing global geometric constraints. When the camera revisits a location, VGGT doesn't know the start and end should match. There's drift.

This project adds a GTSAM factor graph on top of VGGT's output:
- **Loop closure** corrects trajectory drift when the camera revisits areas
- **Multi-view consistency** enforces that the same 3D point seen from two views agrees
- **Uncertainty propagation** gives you covariances on every pose
- **Incremental processing** via iSAM2 handles videos longer than VGGT's 200-frame limit

## Results

| Method | ATE (m) on ScanNet | ATE (m) on TUM |
|--------|-------------------|----------------|
| VGGT (feed-forward) | - | - |
| VGGT + Bundle Adjustment | - | - |
| **VGGT + Factor Graph (ours)** | **-** | **-** |

## Quick Start

```bash
pip install -e .

# Run on a video
python run.py --video my_video.mp4 --output output/

# Run on TUM-RGBD benchmark
python benchmark.py --dataset tum --seq fr1/desk

# Run on ScanNet
python benchmark.py --dataset scannet --scene scene0000_00
```

## Architecture

```
Video frames
    |
    v
[VGGT] --> poses (with confidence), depth maps, point maps
    |
    v
[Initialize Gaussians from VGGT point maps]
[Initialize factor graph with VGGT poses as priors]
    |
    v
[Detect loop closures via feature matching]
[Add photometric factors between overlapping frames]
[Add depth consistency factors]
    |
    v
[iSAM2 optimization] --> refined poses with covariances
    |
    v
[Re-render Gaussians with corrected poses]
    |
    v
Output: refined trajectory, 3D Gaussians, confidence per pose
```

## Requirements

- CUDA GPU with 24GB+ VRAM
- Python 3.10+
- GTSAM, gsplat, PyTorch

## Citation

```bibtex
@software{vggt_factor_refinement,
  author = {Shah, Jash},
  title = {Factor Graph Refinement of Feed-Forward 3D Reconstruction},
  year = {2026},
  url = {https://github.com/jashshah999/vggt-factor-refinement}
}
```

## License

MIT

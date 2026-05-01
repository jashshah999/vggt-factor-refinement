"""Initialize Gaussian splats from VGGT point maps."""

import numpy as np
import torch


def init_gaussians_from_vggt(
    points: np.ndarray,
    images: np.ndarray,
    point_conf: np.ndarray,
    poses_c2w: np.ndarray,
    conf_threshold: float = 0.5,
    stride: int = 4,
    max_gaussians: int = 300000,
    device: str = "cuda",
) -> dict:
    """Create Gaussian parameters from VGGT point maps.

    Args:
        points: (N, H, W, 3) world-space point maps from VGGT
        images: (N, H, W, 3) RGB images in [0, 1]
        point_conf: (N, H, W) confidence maps
        poses_c2w: (N, 4, 4) camera-to-world poses
        conf_threshold: minimum confidence to include a point
        stride: spatial stride for subsampling points
        max_gaussians: maximum number of Gaussians

    Returns:
        dict with means, colors, scales, opacities, quats as torch Parameters
    """
    import cv2

    all_means = []
    all_colors = []

    N, H_pts, W_pts, _ = points.shape

    for i in range(N):
        pts = points[i, ::stride, ::stride]  # (H', W', 3)
        conf = point_conf[i, ::stride, ::stride]  # (H', W')

        # Resize image to match point map resolution before striding
        img_resized = cv2.resize(images[i], (W_pts, H_pts))
        cols = img_resized[::stride, ::stride]

        mask = conf > conf_threshold
        mask &= np.isfinite(pts).all(axis=-1)
        mask &= np.linalg.norm(pts, axis=-1) < 50.0  # filter distant points

        pts_valid = pts[mask]
        cols_valid = cols[mask]

        if len(pts_valid) > 0:
            all_means.append(pts_valid)
            all_colors.append(cols_valid)

    if not all_means:
        raise ValueError("No valid points found. Check confidence threshold.")

    means = np.concatenate(all_means, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    # Subsample if too many
    if len(means) > max_gaussians:
        idx = np.random.choice(len(means), max_gaussians, replace=False)
        means = means[idx]
        colors = colors[idx]

    n = len(means)

    # Estimate scale from point cloud density (median nearest neighbor distance)
    from scipy.spatial import cKDTree
    if n > 10000:
        sample_idx = np.random.choice(n, 10000, replace=False)
        tree = cKDTree(means[sample_idx])
        dists, _ = tree.query(means[sample_idx], k=2)
        nn_dist = np.median(dists[:, 1])
    else:
        tree = cKDTree(means)
        dists, _ = tree.query(means, k=2)
        nn_dist = np.median(dists[:, 1])

    init_scale = float(np.clip(nn_dist * 2, 0.005, 0.2))

    params = {
        "means": torch.nn.Parameter(torch.tensor(means, dtype=torch.float32, device=device)),
        "colors": torch.nn.Parameter(torch.tensor(colors, dtype=torch.float32, device=device)),
        "scales": torch.nn.Parameter(
            torch.full((n, 3), init_scale, dtype=torch.float32, device=device)
        ),
        "opacities": torch.nn.Parameter(
            torch.full((n,), 2.0, dtype=torch.float32, device=device)
        ),
        "quats": torch.nn.Parameter(
            torch.nn.functional.normalize(
                torch.randn(n, 4, dtype=torch.float32, device=device), dim=-1
            )
        ),
    }

    return params

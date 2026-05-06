"""Export to .splat format for web-based Gaussian Splatting viewers.

The .splat format is used by antimatter15/splat and other web viewers.
It's a compact binary format for viewing Gaussian splats in the browser.
"""

import numpy as np
import struct
from pathlib import Path


def export_splat(
    means: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    output_path: str,
) -> str:
    """Export Gaussian splat to .splat format.

    Args:
        means: (N, 3) Gaussian centers
        colors: (N, 3) RGB colors [0, 1]
        opacities: (N,) opacity values [0, 1]
        scales: (N, 3) scale per axis
        rotations: (N, 4) quaternions [w, x, y, z]
        output_path: Output .splat file path

    Returns:
        Path to the .splat file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    N = len(means)

    with open(output_path, "wb") as f:
        for i in range(N):
            # Position (3 floats)
            f.write(struct.pack("<3f", *means[i]))
            # Scale (3 floats, log scale)
            f.write(struct.pack("<3f", *np.log(scales[i] + 1e-8)))
            # Color (4 uint8: RGBA)
            r, g, b = (np.clip(colors[i], 0, 1) * 255).astype(np.uint8)
            a = int(np.clip(opacities[i], 0, 1) * 255)
            f.write(struct.pack("<4B", r, g, b, a))
            # Rotation (4 uint8: quaternion normalized to [0, 255])
            q = rotations[i]  # [w, x, y, z]
            q_normalized = ((q / (np.linalg.norm(q) + 1e-8) + 1) * 127.5).astype(np.uint8)
            f.write(struct.pack("<4B", *q_normalized))

    print(f"  Splat exported: {output_path} ({N} gaussians)")
    return str(output_path)


def pointcloud_to_splat(
    points: np.ndarray,
    colors: np.ndarray,
    output_path: str,
    default_scale: float = 0.01,
    default_opacity: float = 0.9,
) -> str:
    """Convert a point cloud to a viewable .splat file.

    Each point becomes a small isotropic Gaussian. Useful for quick
    visualization of VGGT point clouds without training 3DGS.
    """
    N = len(points)
    means = points.astype(np.float32)
    scales = np.full((N, 3), default_scale, dtype=np.float32)
    opacities = np.full(N, default_opacity, dtype=np.float32)
    rotations = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (N, 1))  # identity

    return export_splat(means, colors, opacities, scales, rotations, output_path)

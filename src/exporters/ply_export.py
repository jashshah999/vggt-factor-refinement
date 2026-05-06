"""Export colored point cloud to PLY format."""

import numpy as np
from pathlib import Path


def export_ply(
    points: np.ndarray,
    colors: np.ndarray,
    output_path: str,
    normals: np.ndarray = None,
) -> str:
    """Export a colored point cloud to PLY.

    Args:
        points: (M, 3) point positions
        colors: (M, 3) RGB colors in [0, 1]
        output_path: Output .ply file path
        normals: Optional (M, 3) normals

    Returns:
        Path to the PLY file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    M = len(points)
    colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    has_normals = normals is not None

    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {M}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_normals:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(M):
            x, y, z = points[i]
            r, g, b = colors_uint8[i]
            if has_normals:
                nx, ny, nz = normals[i]
                f.write(f"{x} {y} {z} {nx} {ny} {nz} {r} {g} {b}\n")
            else:
                f.write(f"{x} {y} {z} {r} {g} {b}\n")

    print(f"  PLY exported: {output_path} ({M} points)")
    return str(output_path)

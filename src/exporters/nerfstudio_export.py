"""Export to nerfstudio transforms.json format.

nerfstudio is the most popular NeRF/3DGS framework. Their format is
a JSON file with camera poses and intrinsics. This allows direct use
with nerfstudio's splatfacto, nerfacto, etc.
"""

import json
import os
import numpy as np
from pathlib import Path


def export_nerfstudio(
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_paths: list[str],
    output_dir: str,
    image_sizes: list[tuple[int, int]] = None,
    aabb_scale: int = 16,
) -> str:
    """Export to nerfstudio transforms.json format.

    Args:
        poses_c2w: (N, 4, 4) camera-to-world poses
        intrinsics: (N, 3, 3) camera intrinsics
        image_paths: List of image file paths
        output_dir: Output directory
        image_sizes: List of (H, W) per image
        aabb_scale: Scene bounding box scale (nerfstudio param)

    Returns:
        Path to transforms.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    N = len(poses_c2w)
    K = intrinsics[0]  # Use first frame's intrinsics as reference
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    if image_sizes:
        h, w = image_sizes[0]
    else:
        w, h = int(cx * 2), int(cy * 2)

    frames = []
    for i in range(N):
        # nerfstudio uses OpenGL convention (camera looks down -Z)
        # VGGT/COLMAP uses OpenCV convention (camera looks down +Z)
        # Convert: flip Y and Z axes
        c2w = poses_c2w[i].copy()
        c2w[:3, 1:3] *= -1  # flip Y and Z columns

        frame = {
            "file_path": os.path.relpath(image_paths[i], output_dir) if os.path.isabs(image_paths[i]) else image_paths[i],
            "transform_matrix": c2w.tolist(),
        }

        # Per-frame intrinsics if they vary
        if not np.allclose(intrinsics[i], intrinsics[0], atol=1.0):
            frame["fl_x"] = float(intrinsics[i][0, 0])
            frame["fl_y"] = float(intrinsics[i][1, 1])
            frame["cx"] = float(intrinsics[i][0, 2])
            frame["cy"] = float(intrinsics[i][1, 2])

        frames.append(frame)

    transforms = {
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "aabb_scale": aabb_scale,
        "frames": frames,
    }

    out_path = output_dir / "transforms.json"
    with open(out_path, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"  nerfstudio transforms exported to {out_path}")
    return str(out_path)

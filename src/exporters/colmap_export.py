"""Export to COLMAP sparse model format.

This is the most requested feature — users want to feed VGGT poses into
3D Gaussian Splatting pipelines (nerfstudio, gsplat, 3DGS) which all
expect COLMAP sparse models as input.

Outputs:
  sparse/0/cameras.bin  (or .txt)
  sparse/0/images.bin   (or .txt)
  sparse/0/points3D.bin (or .txt)
"""

import os
import struct
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


def export_colmap(
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_paths: list[str],
    points3d: np.ndarray,
    point_colors: np.ndarray,
    output_dir: str,
    image_sizes: list[tuple[int, int]] = None,
    binary: bool = True,
) -> str:
    """Export reconstruction to COLMAP sparse model format.

    Args:
        poses_c2w: (N, 4, 4) camera-to-world poses
        intrinsics: (N, 3, 3) camera intrinsic matrices
        image_paths: List of image file paths
        points3d: (M, 3) 3D point positions
        point_colors: (M, 3) RGB colors [0, 1]
        output_dir: Output directory
        image_sizes: List of (H, W) per image (inferred from intrinsics if None)
        binary: Write binary format (faster) or text format (human-readable)

    Returns:
        Path to the sparse model directory.
    """
    sparse_dir = Path(output_dir) / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    N = len(poses_c2w)

    if binary:
        _write_cameras_bin(sparse_dir / "cameras.bin", intrinsics, image_sizes, N)
        _write_images_bin(sparse_dir / "images.bin", poses_c2w, image_paths)
        _write_points3d_bin(sparse_dir / "points3D.bin", points3d, point_colors)
    else:
        _write_cameras_txt(sparse_dir / "cameras.txt", intrinsics, image_sizes, N)
        _write_images_txt(sparse_dir / "images.txt", poses_c2w, image_paths)
        _write_points3d_txt(sparse_dir / "points3D.txt", points3d, point_colors)

    # Also create images/ symlinks or copy list
    images_txt = sparse_dir.parent.parent / "image_list.txt"
    with open(images_txt, "w") as f:
        for p in image_paths:
            f.write(f"{p}\n")

    print(f"  COLMAP sparse model exported to {sparse_dir}")
    return str(sparse_dir)


def _write_cameras_txt(path, intrinsics, image_sizes, N):
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {N}\n")
        for i in range(N):
            K = intrinsics[i]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            if image_sizes:
                h, w = image_sizes[i]
            else:
                w, h = int(cx * 2), int(cy * 2)
            # PINHOLE model: fx, fy, cx, cy
            f.write(f"{i + 1} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")


def _write_cameras_bin(path, intrinsics, image_sizes, N):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", N))
        for i in range(N):
            K = intrinsics[i]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            if image_sizes:
                h, w = image_sizes[i]
            else:
                w, h = int(cx * 2), int(cy * 2)
            camera_id = i + 1
            model_id = 1  # PINHOLE
            f.write(struct.pack("<IiQQ", camera_id, model_id, w, h))
            f.write(struct.pack("<4d", fx, fy, cx, cy))


def _write_images_txt(path, poses_c2w, image_paths):
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in range(len(poses_c2w)):
            w2c = np.linalg.inv(poses_c2w[i])
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            name = os.path.basename(image_paths[i])
            f.write(f"{i + 1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {i + 1} {name}\n")
            f.write("\n")  # empty POINTS2D line


def _write_images_bin(path, poses_c2w, image_paths):
    N = len(poses_c2w)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", N))
        for i in range(N):
            w2c = np.linalg.inv(poses_c2w[i])
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            quat = Rotation.from_matrix(R).as_quat()
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            name = os.path.basename(image_paths[i])
            image_id = i + 1
            camera_id = i + 1
            f.write(struct.pack("<I", image_id))
            f.write(struct.pack("<4d", qw, qx, qy, qz))
            f.write(struct.pack("<3d", t[0], t[1], t[2]))
            f.write(struct.pack("<I", camera_id))
            name_bytes = name.encode("utf-8") + b"\x00"
            f.write(name_bytes)
            f.write(struct.pack("<Q", 0))  # num_points2D = 0


def _write_points3d_txt(path, points3d, colors):
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        n = min(len(points3d), 100000)  # limit to 100k points for COLMAP compat
        indices = np.linspace(0, len(points3d) - 1, n, dtype=int) if len(points3d) > n else np.arange(len(points3d))
        for idx, i in enumerate(indices):
            x, y, z = points3d[i]
            r, g, b = (colors[i] * 255).astype(int)
            f.write(f"{idx + 1} {x} {y} {z} {r} {g} {b} 0.0\n")


def _write_points3d_bin(path, points3d, colors):
    n = min(len(points3d), 100000)
    indices = np.linspace(0, len(points3d) - 1, n, dtype=int) if len(points3d) > n else np.arange(len(points3d))
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(indices)))
        for idx, i in enumerate(indices):
            x, y, z = points3d[i].astype(np.float64)
            r, g, b = (colors[i] * 255).astype(np.uint8)
            point3d_id = idx + 1
            f.write(struct.pack("<Q", point3d_id))
            f.write(struct.pack("<3d", x, y, z))
            f.write(struct.pack("<3B", r, g, b))
            f.write(struct.pack("<d", 0.0))  # error
            f.write(struct.pack("<Q", 0))  # track_length = 0

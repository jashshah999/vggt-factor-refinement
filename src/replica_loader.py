"""Replica dataset loader.

Downloads and loads sequences from the Replica dataset (NICE-SLAM format).
Uses the preprocessed version from NICE-SLAM which has RGB images + GT poses.

See: https://github.com/cvg/nice-slam
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path


REPLICA_SCENES = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]

REPLICA_URL = "https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"


def load_replica_sequence(
    scene: str = "office0",
    data_root: str = "data",
    stride: int = 5,
    max_frames: int = 100,
) -> dict:
    """Load a Replica scene (NICE-SLAM format).

    Returns dict with: images (N,H,W,3), gt_poses (N,4,4), K (3,3), H, W
    """
    replica_dir = os.path.join(data_root, "Replica", scene, "results")

    if not os.path.isdir(replica_dir):
        _download_replica(data_root)

    if not os.path.isdir(replica_dir):
        raise FileNotFoundError(
            f"Replica scene {scene} not found at {replica_dir}. "
            f"Download manually or check the path."
        )

    # Load trajectory
    traj_path = os.path.join(data_root, "Replica", scene, "traj.txt")
    poses = _load_replica_trajectory(traj_path)

    # Find available frames
    frame_files = sorted(Path(replica_dir).glob("frame*.jpg"))
    if not frame_files:
        frame_files = sorted(Path(replica_dir).glob("frame*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frame images found in {replica_dir}")

    # Subsample
    indices = list(range(0, len(frame_files), stride))
    if max_frames > 0:
        indices = indices[:max_frames]

    images = []
    gt_poses = []
    for idx in indices:
        img_path = str(frame_files[idx])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        images.append(img)

        if idx < len(poses):
            gt_poses.append(poses[idx])
        else:
            gt_poses.append(np.eye(4))

    images = np.array(images)
    gt_poses = np.array(gt_poses)
    H, W = images.shape[1], images.shape[2]

    # Replica intrinsics (from NICE-SLAM config)
    fx, fy, cx, cy = 600.0, 600.0, 599.5, 339.5
    if H == 480 and W == 640:
        fx, fy, cx, cy = 320.0, 320.0, 319.5, 239.5

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    return {
        "images": images,
        "gt_poses": gt_poses,
        "K": K,
        "H": H,
        "W": W,
        "seq_name": f"replica/{scene}",
    }


def _load_replica_trajectory(traj_path: str) -> list:
    """Load Replica trajectory file (4x4 matrices, one per line group)."""
    poses = []
    with open(traj_path) as f:
        lines = f.readlines()

    # Format: each pose is 4 lines of 4 numbers
    i = 0
    while i + 3 < len(lines):
        try:
            row0 = [float(x) for x in lines[i].strip().split()]
            row1 = [float(x) for x in lines[i+1].strip().split()]
            row2 = [float(x) for x in lines[i+2].strip().split()]
            row3 = [float(x) for x in lines[i+3].strip().split()]
            if len(row0) == 4 and len(row1) == 4 and len(row2) == 4 and len(row3) == 4:
                T = np.array([row0, row1, row2, row3])
                poses.append(T)
                i += 4
                continue
        except ValueError:
            pass

        # Try single-line format: 16 floats per line
        try:
            vals = [float(x) for x in lines[i].strip().split()]
            if len(vals) == 16:
                T = np.array(vals).reshape(4, 4)
                poses.append(T)
            elif len(vals) == 12:
                T = np.eye(4)
                T[:3, :] = np.array(vals).reshape(3, 4)
                poses.append(T)
        except ValueError:
            pass
        i += 1

    return poses


def _download_replica(data_root: str):
    """Download Replica dataset (NICE-SLAM format)."""
    import urllib.request
    import zipfile

    os.makedirs(data_root, exist_ok=True)
    zip_path = os.path.join(data_root, "Replica.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading Replica dataset...")
        print(f"  URL: {REPLICA_URL}")
        print(f"  This is ~5GB, may take a while...")
        urllib.request.urlretrieve(REPLICA_URL, zip_path)

    print(f"Extracting to {data_root}/...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_root)

    print("Done.")

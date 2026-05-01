"""Dataset loaders for TUM-RGBD and ScanNet benchmarks."""

import os
import cv2
import numpy as np
import tarfile
import urllib.request
from pathlib import Path


TUM_BASE_URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg{fid}/rgbd_dataset_freiburg{fid}_{seq_name}.tgz"

TUM_SEQUENCES = {
    "fr1/desk": ("1", "desk"),
    "fr1/room": ("1", "room"),
    "fr1/xyz": ("1", "xyz"),
    "fr2/desk": ("2", "desk"),
    "fr3/office": ("3", "long_office_household"),
}

TUM_INTRINSICS = {
    "fr1": np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]], dtype=np.float64),
    "fr2": np.array([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]], dtype=np.float64),
    "fr3": np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]], dtype=np.float64),
}


def quaternion_to_matrix(qx, qy, qz, qw) -> np.ndarray:
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])
    return R


def load_tum_sequence(seq_name: str, data_root: str = "data", stride: int = 5,
                      max_frames: int = 100) -> dict:
    """Load a TUM-RGBD sequence.

    Returns dict with: images (N,H,W,3), depths (N,H,W), gt_poses (N,4,4),
    K (3,3), H, W, timestamps
    """
    assert seq_name in TUM_SEQUENCES, f"Unknown: {seq_name}. Options: {list(TUM_SEQUENCES.keys())}"
    fid, sname = TUM_SEQUENCES[seq_name]
    data_dir = os.path.join(data_root, f"rgbd_dataset_freiburg{fid}_{sname}")

    if not os.path.isdir(data_dir):
        _download_tum(data_root, fid, sname)

    K = TUM_INTRINSICS[f"fr{fid}"]

    # Read ground truth
    gt_path = os.path.join(data_dir, "groundtruth.txt")
    gt_poses = {}
    with open(gt_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            T = np.eye(4)
            T[:3, :3] = quaternion_to_matrix(qx, qy, qz, qw)
            T[:3, 3] = [tx, ty, tz]
            gt_poses[ts] = T

    # Read or generate associations
    assoc_path = os.path.join(data_dir, "associations.txt")
    if not os.path.exists(assoc_path):
        assoc_path = _generate_associations(data_dir)

    frames = _read_associations(assoc_path, gt_poses, data_dir)
    frames = frames[::stride]
    if max_frames > 0:
        frames = frames[:max_frames]

    images, depths, poses, timestamps = [], [], [], []
    for f in frames:
        img = cv2.imread(f["rgb_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        depth = cv2.imread(f["depth_path"], cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000.0
        images.append(img)
        depths.append(depth)
        poses.append(f["pose"])
        timestamps.append(f["timestamp"])

    return {
        "images": np.array(images),
        "depths": np.array(depths),
        "gt_poses": np.array(poses),
        "K": K,
        "H": 480,
        "W": 640,
        "timestamps": timestamps,
        "seq_name": seq_name,
    }


def _download_tum(data_root, fid, sname):
    os.makedirs(data_root, exist_ok=True)
    url = TUM_BASE_URL.format(fid=fid, seq_name=sname)
    tgz_path = os.path.join(data_root, f"freiburg{fid}_{sname}.tgz")
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, tgz_path)
    print(f"Extracting to {data_root}/...")
    with tarfile.open(tgz_path) as tar:
        tar.extractall(data_root)
    os.remove(tgz_path)


def _read_associations(assoc_path, gt_poses, data_dir):
    gt_times = sorted(gt_poses.keys())
    frames = []
    with open(assoc_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            ts_rgb = float(parts[0])
            rgb_path = os.path.join(data_dir, parts[1])
            depth_path = os.path.join(data_dir, parts[3])

            # Find closest GT pose
            idx = np.searchsorted(gt_times, ts_rgb)
            best_dt, best_ts = float("inf"), None
            for k in [max(0, idx - 1), min(len(gt_times) - 1, idx)]:
                dt = abs(gt_times[k] - ts_rgb)
                if dt < best_dt:
                    best_dt, best_ts = dt, gt_times[k]
            if best_dt < 0.02:
                frames.append({
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                    "pose": gt_poses[best_ts],
                    "timestamp": ts_rgb,
                })
    return frames


def _generate_associations(data_dir):
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")

    rgb_ts = {}
    for f in sorted(os.listdir(rgb_dir)):
        if f.endswith(".png"):
            ts = float(f.replace(".png", ""))
            rgb_ts[ts] = os.path.join("rgb", f)

    depth_ts = {}
    for f in sorted(os.listdir(depth_dir)):
        if f.endswith(".png"):
            ts = float(f.replace(".png", ""))
            depth_ts[ts] = os.path.join("depth", f)

    assoc_path = os.path.join(data_dir, "associations.txt")
    depth_keys = sorted(depth_ts.keys())

    with open(assoc_path, "w") as out:
        for rt in sorted(rgb_ts.keys()):
            idx = np.searchsorted(depth_keys, rt)
            best_dt, best_key = float("inf"), None
            for k in [max(0, idx - 1), min(len(depth_keys) - 1, idx)]:
                dt = abs(depth_keys[k] - rt)
                if dt < best_dt:
                    best_dt, best_key = dt, depth_keys[k]
            if best_dt < 0.02:
                out.write(f"{rt:.6f} {rgb_ts[rt]} {best_key:.6f} {depth_ts[best_key]}\n")

    return assoc_path

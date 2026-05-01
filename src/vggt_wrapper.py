"""Wrapper around VGGT for extracting poses, depth, and point maps."""

import torch
import numpy as np
from pathlib import Path


def load_vggt(device="cuda"):
    """Load VGGT model from torch hub."""
    model = torch.hub.load("facebookresearch/vggt", "vggt", pretrained=True)
    model = model.to(device).eval()
    return model


def run_vggt(model, images: torch.Tensor, device="cuda"):
    """Run VGGT on a batch of images.

    Args:
        model: VGGT model
        images: (N, 3, H, W) tensor, values in [0, 1]

    Returns:
        dict with keys: extrinsics (N, 3, 4), depth (N, H, W),
        points (N, H, W, 3), depth_conf (N, H, W), pose_conf (N,)
    """
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        images = images.to(device)
        predictions = model(images.unsqueeze(0))

    extrinsics = predictions["extrinsic"][0].float().cpu().numpy()
    depth = predictions["depth"][0].float().cpu().numpy()
    depth_conf = predictions["depth_confidence"][0].float().cpu().numpy()
    points = predictions["world_points"][0].float().cpu().numpy()
    point_conf = predictions["world_points_confidence"][0].float().cpu().numpy()

    # Convert extrinsics (3x4) to 4x4
    N = extrinsics.shape[0]
    poses = np.zeros((N, 4, 4), dtype=np.float64)
    poses[:, :3, :] = extrinsics
    poses[:, 3, 3] = 1.0

    # Invert to get camera-to-world (VGGT gives world-to-camera)
    c2w = np.linalg.inv(poses)

    # Aggregate confidence per frame
    pose_conf = np.median(depth_conf, axis=(1, 2))

    return {
        "poses_c2w": c2w,
        "poses_w2c": poses,
        "depth": depth,
        "depth_conf": depth_conf,
        "points": points,
        "point_conf": point_conf,
        "pose_conf": pose_conf,
    }


def vggt_conf_to_covariance(pose_conf: np.ndarray, base_sigma: float = 0.1) -> np.ndarray:
    """Convert VGGT confidence scores to 6-DOF pose covariance (diagonal).

    Higher confidence -> smaller covariance (more certain).
    Returns array of shape (N, 6) with diagonal covariance entries.
    """
    # Clamp confidence to avoid division by zero
    conf = np.clip(pose_conf, 0.01, 1.0)
    # Inverse relationship: low confidence = high uncertainty
    sigma = base_sigma / conf
    # 6-DOF: (rx, ry, rz, tx, ty, tz)
    return np.stack([sigma] * 6, axis=-1)

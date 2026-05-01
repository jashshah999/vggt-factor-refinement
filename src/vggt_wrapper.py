"""Wrapper around VGGT for extracting poses, depth, and point maps."""

import torch
import numpy as np


def load_vggt(device="cuda"):
    """Load VGGT model from HuggingFace."""
    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to(device).eval()
    return model


def run_vggt_on_images(model, images_np: np.ndarray, device="cuda", max_batch: int = 15):
    """Run VGGT on numpy images.

    Args:
        model: VGGT model
        images_np: (N, H, W, 3) numpy array, values in [0, 1]
        device: cuda or cpu
        max_batch: max frames per VGGT forward pass (memory limited)

    Returns:
        dict with poses_c2w (N,4,4), depth, depth_conf, points, point_conf, pose_conf
    """
    import cv2

    N, H, W, _ = images_np.shape

    # VGGT expects (B, N, 3, H, W) with images resized to fit model
    # Resize to 336 on the shorter side, keeping aspect ratio
    target_size = 336
    if H < W:
        new_H = target_size
        new_W = int(W * target_size / H)
    else:
        new_W = target_size
        new_H = int(H * target_size / W)
    # Make divisible by 14 (ViT patch size)
    new_H = (new_H // 14) * 14
    new_W = (new_W // 14) * 14

    resized = np.stack([cv2.resize(img, (new_W, new_H)) for img in images_np])
    images_torch = torch.tensor(resized, dtype=torch.float32).permute(0, 3, 1, 2)

    if N <= max_batch:
        return _run_vggt_batch(model, images_torch, device)

    # Process in overlapping chunks
    overlap = 3
    all_poses = [None] * N
    all_depth = [None] * N
    all_depth_conf = [None] * N
    all_points = [None] * N
    all_point_conf = [None] * N

    for start in range(0, N, max_batch - overlap):
        end = min(start + max_batch, N)
        chunk = images_torch[start:end]
        out = _run_vggt_batch(model, chunk, device)

        for local_i, global_i in enumerate(range(start, end)):
            if all_poses[global_i] is None:
                all_poses[global_i] = out["poses_c2w"][local_i]
                all_depth[global_i] = out["depth"][local_i]
                all_depth_conf[global_i] = out["depth_conf"][local_i]
                all_points[global_i] = out["points"][local_i]
                all_point_conf[global_i] = out["point_conf"][local_i]

        if end >= N:
            break

    pose_conf = np.array([np.median(dc) for dc in all_depth_conf])

    return {
        "poses_c2w": np.array(all_poses),
        "depth": np.array(all_depth),
        "depth_conf": np.array(all_depth_conf),
        "points": np.array(all_points),
        "point_conf": np.array(all_point_conf),
        "pose_conf": pose_conf,
    }


def _run_vggt_batch(model, images_torch, device):
    """Run VGGT on a single batch of images."""
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        images_batch = images_torch.unsqueeze(0).to(device)
        predictions = model(images_batch)

    # Extract predictions
    N = images_torch.shape[0]

    # VGGT outputs extrinsics as world-to-camera
    # We also need the pose_enc for camera intrinsics
    # Get extrinsics from pose_enc
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    pose_enc = predictions["pose_enc"]
    img_h, img_w = images_torch.shape[2], images_torch.shape[3]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=(img_h, img_w)
    )
    extrinsics = extrinsics[0].float().cpu().numpy()  # (N, 3, 4)

    depth = predictions["depth"][0].float().cpu().numpy()  # (N, H, W)
    depth_conf = predictions["depth_conf"][0].float().cpu().numpy()
    points = predictions["world_points"][0].float().cpu().numpy()  # (N, H, W, 3)
    point_conf = predictions["world_points_conf"][0].float().cpu().numpy()

    # Convert 3x4 extrinsics to 4x4 and invert to get camera-to-world
    poses_w2c = np.zeros((N, 4, 4), dtype=np.float64)
    poses_w2c[:, :3, :] = extrinsics
    poses_w2c[:, 3, 3] = 1.0

    poses_c2w = np.linalg.inv(poses_w2c)
    pose_conf = np.median(depth_conf, axis=(1, 2))

    return {
        "poses_c2w": poses_c2w,
        "poses_w2c": poses_w2c,
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
    conf = np.clip(pose_conf, 0.01, 1.0)
    sigma = base_sigma / conf
    return np.stack([sigma] * 6, axis=-1)

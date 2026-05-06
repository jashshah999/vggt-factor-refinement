"""Multi-view depth fusion for consistent dense geometry.

VGGT produces per-frame depth maps that may disagree across views.
This module fuses multiple depth observations into a single consistent
depth map per frame via confidence-weighted averaging and outlier filtering.
"""

import numpy as np
from typing import Optional


def fuse_depth_maps(
    depth_maps: list[np.ndarray],
    depth_confs: list[np.ndarray],
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    method: str = "confidence_weighted",
    consistency_threshold: float = 0.05,
) -> list[np.ndarray]:
    """Fuse depth maps from multiple views for consistency.

    For each pixel in each frame, check if the depth agrees with
    reprojection from neighboring frames. Filter inconsistent depths.

    Args:
        depth_maps: list of (H, W) depth arrays
        depth_confs: list of (H, W) confidence arrays
        poses_c2w: (N, 4, 4) camera-to-world poses
        intrinsics: (N, 3, 3) camera intrinsics
        method: "confidence_weighted" or "geometric_consistency"
        consistency_threshold: relative depth difference threshold

    Returns:
        List of fused depth maps with inconsistent regions zeroed out.
    """
    N = len(depth_maps)
    fused = [d.copy() for d in depth_maps]

    if method == "geometric_consistency":
        for i in range(N):
            # Check against neighboring frames
            neighbors = _get_neighbor_frames(i, N, window=3)
            consistency_mask = _check_depth_consistency(
                i, neighbors, depth_maps, poses_c2w, intrinsics, consistency_threshold
            )
            fused[i] = depth_maps[i] * consistency_mask
    elif method == "confidence_weighted":
        for i in range(N):
            neighbors = _get_neighbor_frames(i, N, window=2)
            fused[i] = _fuse_with_neighbors(
                i, neighbors, depth_maps, depth_confs, poses_c2w, intrinsics
            )

    return fused


def _get_neighbor_frames(idx: int, N: int, window: int = 3) -> list[int]:
    """Get neighboring frame indices."""
    neighbors = []
    for offset in range(-window, window + 1):
        j = idx + offset
        if j != idx and 0 <= j < N:
            neighbors.append(j)
    return neighbors


def _check_depth_consistency(
    ref_idx: int,
    neighbors: list[int],
    depth_maps: list[np.ndarray],
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Check geometric consistency of depth map against neighbors.

    For each pixel in ref frame, project to 3D, then reproject to neighbor.
    If depth in neighbor agrees (within threshold), mark as consistent.
    """
    ref_depth = depth_maps[ref_idx]
    H, W = ref_depth.shape
    mask = np.zeros((H, W), dtype=np.float32)
    n_checks = 0

    K_ref = intrinsics[ref_idx]
    T_ref = poses_c2w[ref_idx]

    # Create pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3).astype(np.float64)

    # Unproject ref depth to 3D
    z = ref_depth.reshape(-1)
    valid = z > 0
    K_inv = np.linalg.inv(K_ref)
    pts_cam = (K_inv @ pixels.T).T * z[:, None]  # (H*W, 3)
    pts_world = (T_ref[:3, :3] @ pts_cam.T).T + T_ref[:3, 3]  # (H*W, 3)

    for j in neighbors:
        T_j = poses_c2w[j]
        K_j = intrinsics[j]
        depth_j = depth_maps[j]
        H_j, W_j = depth_j.shape

        # Project world points to neighbor camera
        T_j_inv = np.linalg.inv(T_j)
        pts_j_cam = (T_j_inv[:3, :3] @ pts_world.T).T + T_j_inv[:3, 3]
        pts_j_img = (K_j @ pts_j_cam.T).T

        # Normalize
        z_j = pts_j_cam[:, 2]
        u_j = (pts_j_img[:, 0] / (z_j + 1e-8)).astype(int)
        v_j = (pts_j_img[:, 1] / (z_j + 1e-8)).astype(int)

        # Check in-bounds
        in_bounds = (u_j >= 0) & (u_j < W_j) & (v_j >= 0) & (v_j < H_j) & (z_j > 0) & valid

        if in_bounds.sum() == 0:
            continue

        # Check depth consistency
        depth_at_proj = depth_j[v_j[in_bounds], u_j[in_bounds]]
        projected_z = z_j[in_bounds]

        relative_diff = np.abs(depth_at_proj - projected_z) / (projected_z + 1e-8)
        consistent = relative_diff < threshold

        # Update mask
        consistent_full = np.zeros(H * W, dtype=bool)
        in_bounds_idx = np.where(in_bounds)[0]
        consistent_full[in_bounds_idx[consistent]] = True
        mask += consistent_full.reshape(H, W).astype(np.float32)
        n_checks += 1

    if n_checks > 0:
        mask = mask / n_checks
        return (mask > 0.5).astype(np.float32)

    return np.ones((H, W), dtype=np.float32)


def _fuse_with_neighbors(
    ref_idx: int,
    neighbors: list[int],
    depth_maps: list[np.ndarray],
    depth_confs: list[np.ndarray],
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Fuse depth with weighted average from neighbors."""
    ref_depth = depth_maps[ref_idx].copy()
    ref_conf = depth_confs[ref_idx].copy()
    H, W = ref_depth.shape

    # Weight by confidence
    weighted_sum = ref_depth * ref_conf
    weight_sum = ref_conf.copy()

    for j in neighbors:
        # Warp neighbor depth to ref frame
        warped_depth = _warp_depth(
            depth_maps[j], poses_c2w[j], poses_c2w[ref_idx],
            intrinsics[j], intrinsics[ref_idx], (H, W)
        )
        if warped_depth is None:
            continue

        conf_j = depth_confs[j]
        # Resize conf if needed
        if conf_j.shape != (H, W):
            import cv2
            conf_j = cv2.resize(conf_j, (W, H))

        valid = warped_depth > 0
        weighted_sum[valid] += warped_depth[valid] * conf_j.reshape(H, W)[valid] * 0.5
        weight_sum[valid] += conf_j.reshape(H, W)[valid] * 0.5

    result = np.zeros_like(ref_depth)
    valid = weight_sum > 0
    result[valid] = weighted_sum[valid] / weight_sum[valid]
    return result


def _warp_depth(
    src_depth: np.ndarray,
    src_c2w: np.ndarray,
    dst_c2w: np.ndarray,
    src_K: np.ndarray,
    dst_K: np.ndarray,
    dst_size: tuple[int, int],
) -> Optional[np.ndarray]:
    """Warp source depth map to destination view."""
    H_s, W_s = src_depth.shape
    H_d, W_d = dst_size

    u, v = np.meshgrid(np.arange(W_s), np.arange(H_s))
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3).astype(np.float64)
    z = src_depth.reshape(-1)
    valid = z > 0

    if valid.sum() < 100:
        return None

    K_inv = np.linalg.inv(src_K)
    pts_cam = (K_inv @ pixels[valid].T).T * z[valid, None]
    pts_world = (src_c2w[:3, :3] @ pts_cam.T).T + src_c2w[:3, 3]

    # Project to dst
    dst_w2c = np.linalg.inv(dst_c2w)
    pts_dst_cam = (dst_w2c[:3, :3] @ pts_world.T).T + dst_w2c[:3, 3]
    pts_dst_img = (dst_K @ pts_dst_cam.T).T

    z_dst = pts_dst_cam[:, 2]
    u_dst = (pts_dst_img[:, 0] / (z_dst + 1e-8)).astype(int)
    v_dst = (pts_dst_img[:, 1] / (z_dst + 1e-8)).astype(int)

    in_bounds = (u_dst >= 0) & (u_dst < W_d) & (v_dst >= 0) & (v_dst < H_d) & (z_dst > 0)

    warped = np.zeros((H_d, W_d), dtype=np.float32)
    warped[v_dst[in_bounds], u_dst[in_bounds]] = z_dst[in_bounds]

    return warped

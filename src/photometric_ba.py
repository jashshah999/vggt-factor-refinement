"""Photometric bundle adjustment — refine poses using image reprojection loss.

Standard BA uses point correspondences. Photometric BA directly optimizes
camera poses to maximize image alignment via differentiable rendering.
This is more robust for textureless regions where feature points are sparse.

Uses a coarse multi-scale approach:
1. Render depth from current poses
2. Warp neighboring frames to reference
3. Compute photometric loss (SSIM + L1)
4. Backprop to refine poses via Lie algebra parameterization
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


def photometric_refine(
    poses_c2w: np.ndarray,
    depth_maps: list[np.ndarray],
    images: np.ndarray,
    intrinsics: np.ndarray,
    n_iterations: int = 50,
    lr: float = 1e-3,
    window_size: int = 3,
    scales: list[float] = [0.25, 0.5, 1.0],
    device: str = "cuda",
) -> np.ndarray:
    """Refine poses via photometric consistency.

    For each frame, warps neighboring frames using current depth + poses,
    then optimizes the pose (as se(3) delta) to minimize photometric error.

    Args:
        poses_c2w: (N, 4, 4) initial poses
        depth_maps: List of (H, W) depth maps
        images: (N, H, W, 3) images in [0, 1]
        intrinsics: (N, 3, 3) camera matrices
        n_iterations: Optimization iterations per scale
        lr: Learning rate for pose refinement
        window_size: Number of neighboring frames to use
        scales: Multi-scale pyramid (coarse to fine)
        device: CUDA device

    Returns:
        Refined poses (N, 4, 4)
    """
    N = len(poses_c2w)
    refined = poses_c2w.copy()

    # Convert to torch
    images_t = torch.tensor(images, device=device, dtype=torch.float32)
    K = torch.tensor(intrinsics[0], device=device, dtype=torch.float32)

    # Parameterize pose corrections as se(3) vectors (6-DOF per frame)
    # First frame is fixed (anchor)
    deltas = torch.zeros(N, 6, device=device, requires_grad=True)

    for scale in scales:
        H_s = int(images.shape[1] * scale)
        W_s = int(images.shape[2] * scale)
        K_s = K.clone()
        K_s[0] *= scale
        K_s[1] *= scale

        # Downsample images
        imgs_s = F.interpolate(
            images_t.permute(0, 3, 1, 2), size=(H_s, W_s), mode="bilinear"
        ).permute(0, 2, 3, 1)

        optimizer = torch.optim.Adam([deltas], lr=lr)

        for iteration in range(n_iterations):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            n_pairs = 0

            for i in range(1, N):  # skip first (anchor)
                # Current pose with delta applied
                pose_i = _apply_delta(
                    torch.tensor(refined[i], device=device, dtype=torch.float32),
                    deltas[i]
                )

                # Get depth for this frame
                depth_i = depth_maps[i]
                if depth_i is None or depth_i.size < 10:
                    continue
                depth_t = torch.tensor(
                    _resize_depth(depth_i, H_s, W_s),
                    device=device, dtype=torch.float32
                )

                # Warp neighbors to this frame
                neighbors = _get_neighbors(i, N, window_size)
                for j in neighbors:
                    pose_j = _apply_delta(
                        torch.tensor(refined[j], device=device, dtype=torch.float32),
                        deltas[j] if j > 0 else torch.zeros(6, device=device)
                    )

                    warped = _warp_frame(imgs_s[j], depth_t, pose_i, pose_j, K_s, H_s, W_s)
                    if warped is None:
                        continue

                    # Photometric loss (L1 + SSIM)
                    valid_mask = (warped.sum(dim=-1) > 0).float()
                    if valid_mask.sum() < 100:
                        continue

                    l1_loss = (torch.abs(warped - imgs_s[i]) * valid_mask.unsqueeze(-1)).mean()
                    ssim_loss = 1.0 - _compute_ssim(
                        warped.permute(2, 0, 1).unsqueeze(0),
                        imgs_s[i].permute(2, 0, 1).unsqueeze(0)
                    )

                    total_loss = total_loss + 0.85 * ssim_loss + 0.15 * l1_loss
                    n_pairs += 1

            if n_pairs > 0:
                loss = total_loss / n_pairs
                loss.backward()
                optimizer.step()

        # Apply converged deltas to poses
        with torch.no_grad():
            for i in range(1, N):
                refined[i] = _apply_delta(
                    torch.tensor(refined[i], device=device, dtype=torch.float32),
                    deltas[i]
                ).cpu().numpy()

        # Reset deltas for next scale
        deltas = torch.zeros(N, 6, device=device, requires_grad=True)

    return refined


def _apply_delta(pose: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Apply se(3) delta to a pose."""
    # delta = [rx, ry, rz, tx, ty, tz]
    dR = _so3_exp(delta[:3])
    dt = delta[3:]

    result = pose.clone()
    result[:3, :3] = dR @ pose[:3, :3]
    result[:3, 3] = pose[:3, 3] + dt
    return result


def _so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Exponential map from so(3) to SO(3) via Rodrigues."""
    theta = torch.norm(omega)
    if theta < 1e-6:
        return torch.eye(3, device=omega.device, dtype=omega.dtype)

    k = omega / theta
    K = torch.tensor([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ], device=omega.device, dtype=omega.dtype)

    return torch.eye(3, device=omega.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K


def _warp_frame(
    src_img: torch.Tensor, ref_depth: torch.Tensor,
    ref_pose: torch.Tensor, src_pose: torch.Tensor,
    K: torch.Tensor, H: int, W: int,
) -> Optional[torch.Tensor]:
    """Warp source image to reference view using depth."""
    # Create pixel grid
    v, u = torch.meshgrid(torch.arange(H, device=K.device, dtype=torch.float32),
                          torch.arange(W, device=K.device, dtype=torch.float32), indexing="ij")
    ones = torch.ones_like(u)
    pixels = torch.stack([u, v, ones], dim=-1).reshape(-1, 3)

    # Unproject reference pixels to 3D
    z = ref_depth.reshape(-1)
    valid = z > 0
    if valid.sum() < 100:
        return None

    K_inv = torch.inverse(K)
    pts_cam = (K_inv @ pixels.T).T * z.unsqueeze(-1)

    # Transform to world then to source camera
    ref_w2c = torch.inverse(ref_pose)
    src_w2c = torch.inverse(src_pose)
    rel = src_w2c @ ref_pose

    pts_src = (rel[:3, :3] @ pts_cam.T).T + rel[:3, 3]

    # Project to source image
    pts_proj = (K @ pts_src.T).T
    z_src = pts_proj[:, 2:3]
    uv_src = pts_proj[:, :2] / (z_src + 1e-8)

    # Normalize to [-1, 1] for grid_sample
    uv_norm = torch.zeros_like(uv_src)
    uv_norm[:, 0] = 2.0 * uv_src[:, 0] / W - 1.0
    uv_norm[:, 1] = 2.0 * uv_src[:, 1] / H - 1.0

    grid = uv_norm.reshape(1, H, W, 2)
    src_img_4d = src_img.permute(2, 0, 1).unsqueeze(0)

    warped = F.grid_sample(src_img_4d, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return warped.squeeze(0).permute(1, 2, 0)


def _compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Compute mean SSIM between two images."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2

    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.mean()


def _get_neighbors(idx: int, N: int, window: int) -> list[int]:
    neighbors = []
    for offset in range(-window, window + 1):
        j = idx + offset
        if j != idx and 0 <= j < N:
            neighbors.append(j)
    return neighbors


def _resize_depth(depth: np.ndarray, H: int, W: int) -> np.ndarray:
    import cv2
    return cv2.resize(depth.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)

"""Dynamic object masking for robust reconstruction.

Real-world videos contain moving objects (people, cars, animals) that
violate the static scene assumption. This module detects and masks
dynamic regions so they don't corrupt the pose estimation or 3D model.

Uses a combination of:
1. Optical flow inconsistency (forward-backward check)
2. Epipolar geometry violation (points that don't fit the fundamental matrix)
3. Optional: SAM2 segmentation for semantic filtering
"""

import numpy as np
import cv2
from typing import Optional


def compute_dynamic_masks(
    image_paths: list[str],
    poses_c2w: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    method: str = "flow_consistency",
    threshold: float = 2.0,
) -> list[np.ndarray]:
    """Compute binary masks where 1 = static, 0 = dynamic.

    Args:
        image_paths: Frame image paths
        poses_c2w: Optional poses (enables epipolar check)
        intrinsics: Optional camera matrices
        method: "flow_consistency", "epipolar", or "both"
        threshold: Sensitivity (lower = more aggressive masking)

    Returns:
        List of (H, W) binary masks per frame
    """
    N = len(image_paths)
    masks = []

    for i in range(N):
        img = cv2.imread(image_paths[i])
        H, W = img.shape[:2]
        mask = np.ones((H, W), dtype=np.float32)

        if method in ("flow_consistency", "both") and i > 0:
            prev_img = cv2.imread(image_paths[i - 1])
            flow_mask = _flow_consistency_mask(prev_img, img, threshold)
            mask *= flow_mask

        if method in ("epipolar", "both") and poses_c2w is not None and i > 0:
            prev_img = cv2.imread(image_paths[i - 1])
            epi_mask = _epipolar_mask(prev_img, img, poses_c2w[i-1], poses_c2w[i], intrinsics[i], threshold)
            mask *= epi_mask

        masks.append(mask)

    return masks


def _flow_consistency_mask(
    img1: np.ndarray, img2: np.ndarray, threshold: float
) -> np.ndarray:
    """Forward-backward optical flow consistency check.

    Static points should have consistent flow: flow_fwd(x) + flow_bwd(x + flow_fwd(x)) ≈ 0
    Dynamic points violate this.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute forward and backward flow
    flow_fwd = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_bwd = cv2.calcOpticalFlowFarneback(gray2, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    H, W = gray1.shape

    # Warp backward flow to forward frame
    coords = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).astype(np.float32)
    warped_coords = coords + flow_fwd
    warped_coords[:, :, 0] = np.clip(warped_coords[:, :, 0], 0, W - 1)
    warped_coords[:, :, 1] = np.clip(warped_coords[:, :, 1], 0, H - 1)

    # Sample backward flow at warped positions
    flow_bwd_warped = cv2.remap(flow_bwd, warped_coords[:, :, 0], warped_coords[:, :, 1], cv2.INTER_LINEAR)

    # Consistency error
    error = np.sqrt(np.sum((flow_fwd + flow_bwd_warped) ** 2, axis=-1))

    # Threshold: static pixels have low error
    mask = (error < threshold).astype(np.float32)

    # Dilate mask slightly to be conservative
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)

    return mask


def _epipolar_mask(
    img1: np.ndarray, img2: np.ndarray,
    pose1: np.ndarray, pose2: np.ndarray,
    K: np.ndarray, threshold: float,
) -> np.ndarray:
    """Epipolar geometry violation mask.

    Points that don't satisfy the epipolar constraint (given known poses)
    are likely dynamic objects.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    H, W = gray1.shape

    # Detect features
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 20:
        return np.ones((H, W), dtype=np.float32)

    # Match
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 10:
        return np.ones((H, W), dtype=np.float32)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Compute fundamental matrix from known poses
    R_rel = np.linalg.inv(pose2[:3, :3]) @ pose1[:3, :3]
    t_rel = np.linalg.inv(pose2[:3, :3]) @ (pose1[:3, 3] - pose2[:3, 3])
    t_x = np.array([[0, -t_rel[2], t_rel[1]],
                    [t_rel[2], 0, -t_rel[0]],
                    [-t_rel[1], t_rel[0], 0]])
    E = t_x @ R_rel
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv

    # Compute epipolar distance for each match
    mask = np.ones((H, W), dtype=np.float32)

    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])

    # Epipolar line in image 2: l = F @ x1
    lines2 = (F @ pts1_h.T).T
    # Distance from point to line
    dists = np.abs(np.sum(lines2 * pts2_h, axis=1)) / (np.sqrt(lines2[:, 0]**2 + lines2[:, 1]**2) + 1e-8)

    # Mark dynamic point neighborhoods
    dynamic_pts = pts1[dists > threshold * 3]
    for pt in dynamic_pts:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(mask, (x, y), 30, 0.0, -1)

    return mask


def filter_points_by_mask(
    points: np.ndarray,
    point_colors: np.ndarray,
    masks: list[np.ndarray],
    depth_maps: list[np.ndarray],
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove 3D points that project into dynamic regions.

    For each 3D point, check if it projects into a masked (dynamic) region
    in any observing frame. If so, remove it from the reconstruction.
    """
    N = len(masks)
    M = len(points)

    # For efficiency, subsample points
    if M > 50000:
        indices = np.random.choice(M, 50000, replace=False)
        points_sub = points[indices]
        colors_sub = point_colors[indices]
    else:
        points_sub = points
        colors_sub = point_colors
        indices = np.arange(M)

    keep = np.ones(len(points_sub), dtype=bool)

    for i in range(min(N, 10)):  # check against 10 frames for speed
        mask = masks[i]
        if mask is None:
            continue
        H, W = mask.shape
        K = intrinsics[i]
        w2c = np.linalg.inv(poses_c2w[i])

        # Project points to this frame
        pts_cam = (w2c[:3, :3] @ points_sub.T).T + w2c[:3, 3]
        pts_img = (K @ pts_cam.T).T
        z = pts_cam[:, 2]

        u = (pts_img[:, 0] / (z + 1e-8)).astype(int)
        v = (pts_img[:, 1] / (z + 1e-8)).astype(int)

        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)

        for idx in np.where(in_bounds)[0]:
            if mask[v[idx], u[idx]] < 0.5:  # dynamic region
                keep[idx] = False

    return points_sub[keep], colors_sub[keep]

"""Cross-chunk relative pose estimation via 3D point cloud matching.

When two frames from different VGGT chunks see the same scene, we estimate
their relative pose by matching features in 2D, looking up corresponding
3D points from each chunk's point map, and running Procrustes alignment.

This gives an independent relative pose that doesn't depend on the
(potentially drifted) naive-stitched trajectory.
"""

import cv2
import numpy as np


def estimate_cross_chunk_relative_pose(
    img_i: np.ndarray,
    img_j: np.ndarray,
    pts_i: np.ndarray,
    pts_j: np.ndarray,
    conf_i: np.ndarray,
    conf_j: np.ndarray,
    pose_i_in_chunk: np.ndarray,
    pose_j_in_chunk: np.ndarray,
    min_matches: int = 20,
) -> tuple:
    """Estimate relative pose between two frames from different chunks.

    Args:
        img_i, img_j: (H, W, 3) images in [0, 1]
        pts_i, pts_j: (H_vggt, W_vggt, 3) point maps from each chunk
        conf_i, conf_j: (H_vggt, W_vggt) confidence maps
        pose_i_in_chunk, pose_j_in_chunk: 4x4 poses in their respective chunk coords
        min_matches: minimum number of 3D-3D correspondences

    Returns:
        (relative_pose_4x4, n_inliers) or (None, 0) if failed
    """
    # Match features in 2D
    pts2d_i, pts2d_j, n_inliers = _match_frames(img_i, img_j)
    if n_inliers < min_matches:
        return None, 0

    # Look up 3D points from each chunk's point map
    H_pts, W_pts = pts_i.shape[:2]
    H_img, W_img = img_i.shape[:2]

    # Scale 2D matches to point map resolution
    scale_x = W_pts / W_img
    scale_y = H_pts / H_img

    pts3d_i = []
    pts3d_j = []
    for (xi, yi), (xj, yj) in zip(pts2d_i, pts2d_j):
        # Map to point map coordinates
        pi_x, pi_y = int(xi * scale_x), int(yi * scale_y)
        pj_x, pj_y = int(xj * scale_x), int(yj * scale_y)

        # Bounds check
        pi_x = np.clip(pi_x, 0, W_pts - 1)
        pi_y = np.clip(pi_y, 0, H_pts - 1)
        pj_x = np.clip(pj_x, 0, W_pts - 1)
        pj_y = np.clip(pj_y, 0, H_pts - 1)

        # Check confidence
        if conf_i[pi_y, pi_x] < 0.3 or conf_j[pj_y, pj_x] < 0.3:
            continue

        p3d_i = pts_i[pi_y, pi_x]
        p3d_j = pts_j[pj_y, pj_x]

        if not (np.isfinite(p3d_i).all() and np.isfinite(p3d_j).all()):
            continue

        pts3d_i.append(p3d_i)
        pts3d_j.append(p3d_j)

    if len(pts3d_i) < min_matches:
        return None, 0

    pts3d_i = np.array(pts3d_i)
    pts3d_j = np.array(pts3d_j)

    # RANSAC Procrustes: find the best Sim(3) alignment
    best_T, best_scale, best_inliers = _ransac_procrustes(pts3d_i, pts3d_j)
    if best_inliers < min_matches:
        return None, 0

    # The Procrustes gives us: p_j = scale * R @ p_i + t
    # Both point maps are in world coordinates of their respective chunks.
    # The relative pose between the two camera frames is:
    # T_j_from_i = T_j_in_chunkB^{-1} @ Sim3 @ T_i_in_chunkA
    # But since we're working with world points (not camera-frame points),
    # the Sim3 transform IS the chunk-to-chunk alignment.
    # So the relative pose between frames is:
    # T_rel = inv(pose_j) @ Sim3_transform @ pose_i (in respective chunk coords)

    # Build the Sim3 as a 4x4 (applying scale to translation)
    sim3 = np.eye(4)
    sim3[:3, :3] = best_scale * best_T[:3, :3]
    sim3[:3, 3] = best_T[:3, 3]

    rel = np.linalg.inv(pose_j_in_chunk) @ sim3 @ pose_i_in_chunk
    return rel, best_inliers


def _match_frames(img_i, img_j):
    """Match ORB features between two frames. Returns matched 2D points and inlier count."""
    gray_i = (img_i * 255).astype(np.uint8)
    gray_j = (img_j * 255).astype(np.uint8)
    if gray_i.ndim == 3:
        gray_i = cv2.cvtColor(gray_i, cv2.COLOR_RGB2GRAY)
        gray_j = cv2.cvtColor(gray_j, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(2000)
    kp_i, des_i = orb.detectAndCompute(gray_i, None)
    kp_j, des_j = orb.detectAndCompute(gray_j, None)

    if des_i is None or des_j is None or len(kp_i) < 10 or len(kp_j) < 10:
        return np.array([]), np.array([]), 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_i, des_j)

    if len(matches) < 10:
        return np.array([]), np.array([]), 0

    pts_i = np.float32([kp_i[m.queryIdx].pt for m in matches])
    pts_j = np.float32([kp_j[m.trainIdx].pt for m in matches])

    # Fundamental matrix RANSAC to filter outliers
    _, mask = cv2.findFundamentalMat(pts_i, pts_j, cv2.FM_RANSAC, 3.0)
    if mask is None:
        return np.array([]), np.array([]), 0

    mask = mask.ravel().astype(bool)
    return pts_i[mask], pts_j[mask], int(mask.sum())


def _ransac_procrustes(src, dst, n_iters=200, inlier_thresh=0.1):
    """RANSAC Sim(3) alignment from src to dst point sets.

    Returns (T_4x4, scale, n_inliers).
    """
    N = len(src)
    best_inliers = 0
    best_T = np.eye(4)
    best_scale = 1.0

    for _ in range(n_iters):
        # Sample 4 correspondences
        idx = np.random.choice(N, min(4, N), replace=False)
        s = src[idx]
        d = dst[idx]

        T, scale = _procrustes_sim3(s, d)
        if T is None:
            continue

        # Count inliers
        transformed = scale * (T[:3, :3] @ src.T).T + T[:3, 3]
        errors = np.linalg.norm(transformed - dst, axis=1)
        inliers = (errors < inlier_thresh).sum()

        if inliers > best_inliers:
            best_inliers = inliers
            best_T = T
            best_scale = scale

    # Refine on all inliers
    if best_inliers >= 4:
        transformed = best_scale * (best_T[:3, :3] @ src.T).T + best_T[:3, 3]
        errors = np.linalg.norm(transformed - dst, axis=1)
        inlier_mask = errors < inlier_thresh
        if inlier_mask.sum() >= 4:
            T_refined, scale_refined = _procrustes_sim3(src[inlier_mask], dst[inlier_mask])
            if T_refined is not None:
                best_T = T_refined
                best_scale = scale_refined
                transformed = best_scale * (best_T[:3, :3] @ src.T).T + best_T[:3, 3]
                errors = np.linalg.norm(transformed - dst, axis=1)
                best_inliers = (errors < inlier_thresh).sum()

    return best_T, best_scale, best_inliers


def _procrustes_sim3(src, dst):
    """Sim(3) Procrustes alignment."""
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    src_centered = src - src_c
    dst_centered = dst - dst_c
    src_var = np.mean(np.sum(src_centered ** 2, axis=1))
    if src_var < 1e-10:
        return None, 1.0

    H = src_centered.T @ dst_centered / len(src)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    scale = np.sum(S * np.diag(D)) / src_var
    t = dst_c - scale * R @ src_c

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, scale

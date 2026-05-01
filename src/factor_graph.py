"""Factor graph construction and optimization using GTSAM."""

import numpy as np
import gtsam
from typing import Optional


def matrix_to_pose3(T: np.ndarray) -> gtsam.Pose3:
    """Convert 4x4 matrix to GTSAM Pose3."""
    R = gtsam.Rot3(T[:3, :3])
    t = gtsam.Point3(T[:3, 3])
    return gtsam.Pose3(R, t)


def pose3_to_matrix(pose: gtsam.Pose3) -> np.ndarray:
    """Convert GTSAM Pose3 to 4x4 matrix."""
    T = np.eye(4)
    T[:3, :3] = pose.rotation().matrix()
    T[:3, 3] = pose.translation()
    return T


def build_factor_graph(
    poses_c2w: np.ndarray,
    pose_sigmas: np.ndarray,
    loop_closures: Optional[list] = None,
    odom_sigma: float = 0.05,
) -> tuple:
    """Build a GTSAM factor graph from VGGT poses.

    Args:
        poses_c2w: (N, 4, 4) camera-to-world poses from VGGT
        pose_sigmas: (N, 6) diagonal sigma for each pose prior
        loop_closures: list of (i, j, relative_pose_matrix) tuples
        odom_sigma: sigma for odometry (between) factors

    Returns:
        (optimized_poses, covariances) after iSAM2 optimization
    """
    N = len(poses_c2w)

    isam = gtsam.ISAM2(gtsam.ISAM2Params())

    # Add all poses incrementally
    for i in range(N):
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        key = gtsam.symbol("x", i)
        pose_i = matrix_to_pose3(poses_c2w[i])

        if i == 0:
            # Strong prior on first pose
            prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.001)
            graph.addPriorPose3(key, pose_i, prior_noise)
        else:
            # Odometry factor from consecutive VGGT poses
            prev_key = gtsam.symbol("x", i - 1)
            prev_pose = matrix_to_pose3(poses_c2w[i - 1])
            odom = prev_pose.between(pose_i)
            odom_noise = gtsam.noiseModel.Isotropic.Sigma(6, odom_sigma)
            graph.add(gtsam.BetweenFactorPose3(prev_key, key, odom, odom_noise))

            # VGGT pose prior (weighted by confidence)
            sigma = pose_sigmas[i]
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(sigma)
            graph.addPriorPose3(key, pose_i, prior_noise)

        values.insert(key, pose_i)
        isam.update(graph, values)

    # Add loop closure factors
    if loop_closures:
        lc_graph = gtsam.NonlinearFactorGraph()
        for i, j, rel_pose_matrix in loop_closures:
            key_i = gtsam.symbol("x", i)
            key_j = gtsam.symbol("x", j)
            rel_pose = matrix_to_pose3(rel_pose_matrix)
            lc_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.02)
            lc_graph.add(gtsam.BetweenFactorPose3(key_i, key_j, rel_pose, lc_noise))

        isam.update(lc_graph, gtsam.Values())
        # Extra iterations for convergence after loop closure
        for _ in range(5):
            isam.update()

    # Extract results
    estimate = isam.calculateEstimate()
    optimized = np.zeros_like(poses_c2w)
    for i in range(N):
        key = gtsam.symbol("x", i)
        optimized[i] = pose3_to_matrix(estimate.atPose3(key))

    # Extract marginal covariances
    marginals = gtsam.Marginals(isam.getFactorsUnsafe(), estimate)
    covariances = []
    for i in range(N):
        try:
            cov = marginals.marginalCovariance(gtsam.symbol("x", i))
            covariances.append(np.diag(cov))
        except Exception:
            covariances.append(np.ones(6) * 999.0)

    return optimized, np.array(covariances)


def detect_loop_closures(
    poses_c2w: np.ndarray,
    images: np.ndarray = None,
    points: np.ndarray = None,
    point_conf: np.ndarray = None,
    distance_threshold: float = 1.0,
    min_frame_gap: int = 20,
) -> list:
    """Detect loop closures via spatial proximity + visual verification.

    When VGGT point maps are available, estimates the loop closure relative
    pose by aligning overlapping 3D points (Procrustes). This is independent
    of the VGGT pose estimates, breaking the circularity problem.

    Returns list of (i, j, relative_pose_matrix) tuples.
    """
    N = len(poses_c2w)
    positions = poses_c2w[:, :3, 3]
    candidates = []

    for i in range(N):
        for j in range(i + min_frame_gap, N):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < distance_threshold:
                candidates.append((i, j))

    if not candidates:
        return []

    if len(candidates) > 100:
        indices = np.linspace(0, len(candidates) - 1, 100, dtype=int)
        candidates = [candidates[k] for k in indices]

    closures = []
    for i, j in candidates:
        if images is not None:
            n_inliers = _count_match_inliers(images[i], images[j])
            if n_inliers < 30:
                continue

        if points is not None and point_conf is not None:
            # Align 3D point clouds from the two frames (Procrustes)
            rel = _align_pointmaps(points[i], points[j], point_conf[i], point_conf[j])
            if rel is not None:
                closures.append((i, j, rel))
                continue

        # Fallback: use VGGT relative pose
        rel = np.linalg.inv(poses_c2w[i]) @ poses_c2w[j]
        closures.append((i, j, rel))

    if len(closures) > 50:
        indices = np.linspace(0, len(closures) - 1, 50, dtype=int)
        closures = [closures[k] for k in indices]

    return closures


def _align_pointmaps(
    pts1: np.ndarray, pts2: np.ndarray,
    conf1: np.ndarray, conf2: np.ndarray,
    conf_thresh: float = 0.3, stride: int = 8,
) -> np.ndarray:
    """Align two VGGT point maps via Procrustes on co-visible 3D points.

    Both point maps are in world coordinates. Find nearby point pairs and
    estimate the rigid transform from frame 1's coordinate system to frame 2's.
    """
    from scipy.spatial import cKDTree

    # Subsample and filter by confidence
    p1 = pts1[::stride, ::stride].reshape(-1, 3)
    p2 = pts2[::stride, ::stride].reshape(-1, 3)
    c1 = conf1[::stride, ::stride].ravel()
    c2 = conf2[::stride, ::stride].ravel()

    mask1 = (c1 > conf_thresh) & np.isfinite(p1).all(axis=1)
    mask2 = (c2 > conf_thresh) & np.isfinite(p2).all(axis=1)
    p1, p2 = p1[mask1], p2[mask2]

    if len(p1) < 50 or len(p2) < 50:
        return None

    # Find mutual nearest neighbors
    tree2 = cKDTree(p2)
    dists, indices = tree2.query(p1, k=1)
    close = dists < 0.1  # 10cm threshold for correspondence

    if close.sum() < 30:
        return None

    src = p1[close]
    dst = p2[indices[close]]

    # Procrustes alignment (no scale)
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = dst_c - R @ src_c

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _count_match_inliers(img1: np.ndarray, img2: np.ndarray) -> int:
    """Count feature match inliers between two images to verify visual overlap."""
    import cv2

    gray1 = (img1 * 255).astype(np.uint8)
    gray2 = (img2 * 255).astype(np.uint8)
    if gray1.ndim == 3:
        gray1 = cv2.cvtColor(gray1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(gray2, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 10:
        return 0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)
    if mask is None:
        return 0

    return int(mask.ravel().sum())

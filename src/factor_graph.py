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
    distance_threshold: float = 1.0,
    min_frame_gap: int = 20,
) -> list:
    """Detect loop closures based on spatial proximity and visual verification.

    If images are provided, uses feature matching to compute more accurate
    relative poses between loop closure candidates (instead of using the
    potentially drifted VGGT poses).

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

    # Limit candidates
    if len(candidates) > 100:
        indices = np.linspace(0, len(candidates) - 1, 100, dtype=int)
        candidates = [candidates[k] for k in indices]

    closures = []
    for i, j in candidates:
        if images is not None:
            # Compute relative pose from feature matching
            rel = _match_relative_pose(images[i], images[j])
            if rel is not None:
                closures.append((i, j, rel))
        else:
            # Fallback to VGGT poses
            rel = np.linalg.inv(poses_c2w[i]) @ poses_c2w[j]
            closures.append((i, j, rel))

    # Keep at most 50
    if len(closures) > 50:
        indices = np.linspace(0, len(closures) - 1, 50, dtype=int)
        closures = [closures[k] for k in indices]

    return closures


def _match_relative_pose(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Estimate relative pose between two images using ORB + essential matrix.

    Returns 4x4 relative pose matrix, or None if matching fails.
    """
    import cv2

    # Convert to uint8 gray
    gray1 = (img1 * 255).astype(np.uint8)
    gray2 = (img2 * 255).astype(np.uint8)
    if gray1.ndim == 3:
        gray1 = cv2.cvtColor(gray1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(gray2, cv2.COLOR_RGB2GRAY)

    # ORB features
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None

    # BF matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    if len(matches) < 20:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Estimate essential matrix (assume rough intrinsics)
    h, w = gray1.shape
    focal = max(h, w)
    pp = (w / 2, h / 2)
    E, mask = cv2.findEssentialMat(pts1, pts2, focal, pp, cv2.RANSAC, 0.999, 1.0)
    if E is None:
        return None

    inliers = mask.ravel().sum()
    if inliers < 15:
        return None

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, focal=focal, pp=pp, mask=mask)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T

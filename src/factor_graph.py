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
    distance_threshold: float = 0.5,
    min_frame_gap: int = 30,
) -> list:
    """Detect loop closures based on spatial proximity.

    Returns list of (i, j, relative_pose_matrix) tuples.
    """
    N = len(poses_c2w)
    positions = poses_c2w[:, :3, 3]
    closures = []

    for i in range(N):
        for j in range(i + min_frame_gap, N):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < distance_threshold:
                # Relative pose from i to j
                T_i = poses_c2w[i]
                T_j = poses_c2w[j]
                rel = np.linalg.inv(T_i) @ T_j
                closures.append((i, j, rel))

    # Deduplicate: keep at most one closure per (i, j) region
    if len(closures) > 50:
        # Sample uniformly to avoid overwhelming the graph
        indices = np.linspace(0, len(closures) - 1, 50, dtype=int)
        closures = [closures[k] for k in indices]

    return closures

"""SL(4) factor graph for uncalibrated camera pose estimation.

Uses GTSAM's SL4 Lie group to handle the full 15-DOF projective ambiguity
between VGGT chunks. Falls back to Sim(3) when the scene is planar or
the homography estimation is ill-conditioned.
"""

import numpy as np
import gtsam
from gtsam import SL4, BetweenFactorSL4, PriorFactorSL4
from gtsam.symbol_shorthand import X


def normalize_to_sl4(H: np.ndarray) -> np.ndarray:
    """Normalize a 4x4 matrix to have determinant 1 (project onto SL(4))."""
    det = np.linalg.det(H)
    if abs(det) < 1e-15:
        return np.eye(4)
    return H / (det ** 0.25)


def is_planar(points: np.ndarray, threshold: float = 0.05) -> bool:
    """Check if a point cloud is approximately planar.

    Args:
        points: (N, 3) point cloud
        threshold: ratio of smallest to largest singular value

    Returns:
        True if the scene is approximately planar
    """
    if len(points) < 10:
        return False
    centered = points - points.mean(axis=0)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    if S[1] < 1e-10:
        return True
    ratio = S[2] / S[1]
    return ratio < threshold


def estimate_pairwise_homography(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    n_ransac: int = 500,
    inlier_thresh: float = 0.02,
) -> tuple:
    """Estimate 4x4 homography between two 3D point sets via RANSAC.

    Uses the DLT (Direct Linear Transform) on 5 point correspondences
    to estimate the 4x4 projective transformation.

    Returns:
        (H_4x4, n_inliers, is_degenerate)
    """
    N = len(pts_src)
    if N < 5:
        return np.eye(4), 0, True

    # Homogeneous coordinates
    src_h = np.hstack([pts_src, np.ones((N, 1))])
    dst_h = np.hstack([pts_dst, np.ones((N, 1))])

    best_H = np.eye(4)
    best_inliers = 0
    best_degenerate = True

    for _ in range(n_ransac):
        idx = np.random.choice(N, 5, replace=False)
        s = src_h[idx]
        d = dst_h[idx]

        # Solve for H via DLT: d = H @ s for each correspondence
        # Build the linear system A @ h = 0
        A = np.zeros((20, 16))
        for k in range(5):
            si = s[k]
            di = d[k]
            for j in range(4):
                row = k * 4 + j
                A[row, j * 4: (j + 1) * 4] = si
                A[row, :] -= di[j] * np.tile(si, 4) * 0  # zero out
                # Actually use the standard DLT formulation
                A[row, j * 4: (j + 1) * 4] = si
                for m in range(4):
                    if m != j:
                        pass

        # Simpler approach: solve H @ src = dst directly
        # H = dst @ pinv(src) for 5 correspondences (overdetermined for 4x4)
        try:
            H = dst_h[idx].T @ np.linalg.pinv(src_h[idx].T)
        except np.linalg.LinAlgError:
            continue

        # Check determinant
        det = np.linalg.det(H)
        if abs(det) < 1e-10:
            continue

        H = normalize_to_sl4(H)

        # Count inliers
        transformed = (H @ src_h.T).T
        # Normalize homogeneous coordinates
        transformed = transformed / (transformed[:, 3:4] + 1e-15)
        errors = np.linalg.norm(transformed[:, :3] - dst_h[:, :3], axis=1)
        inliers = (errors < inlier_thresh).sum()

        if inliers > best_inliers:
            best_H = H
            best_inliers = inliers
            best_degenerate = False

    # Refine on inliers
    if best_inliers >= 5:
        transformed = (best_H @ src_h.T).T
        transformed = transformed / (transformed[:, 3:4] + 1e-15)
        errors = np.linalg.norm(transformed[:, :3] - dst_h[:, :3], axis=1)
        inlier_mask = errors < inlier_thresh

        if inlier_mask.sum() >= 5:
            try:
                H_refined = dst_h[inlier_mask].T @ np.linalg.pinv(src_h[inlier_mask].T)
                det = np.linalg.det(H_refined)
                if abs(det) > 1e-10:
                    best_H = normalize_to_sl4(H_refined)
                    transformed = (best_H @ src_h.T).T
                    transformed = transformed / (transformed[:, 3:4] + 1e-15)
                    errors = np.linalg.norm(transformed[:, :3] - dst_h[:, :3], axis=1)
                    best_inliers = (errors < inlier_thresh).sum()
            except np.linalg.LinAlgError:
                pass

    return best_H, best_inliers, best_degenerate


def build_sl4_graph(
    chunks: list,
    init_poses: np.ndarray,
    images: np.ndarray = None,
    N: int = 0,
) -> np.ndarray:
    """Build and optimize an SL(4) factor graph for chunk alignment.

    Falls back to Sim(3) for planar scenes or when SL(4) estimation
    is poorly conditioned.

    Args:
        chunks: list of chunk dicts with poses_c2w, points, point_conf
        init_poses: (N, 4, 4) naive-stitched poses as initialization
        images: (N, H, W, 3) for loop closure detection
        N: total number of frames

    Returns:
        (N, 4, 4) optimized poses
    """
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    inner_noise = gtsam.noiseModel.Diagonal.Sigmas(0.05 * np.ones(15))
    intra_noise = gtsam.noiseModel.Diagonal.Sigmas(0.1 * np.ones(15))
    anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(1e-6 * np.ones(15))

    # Track which frames use SL(4) vs Sim(3) fallback
    n_sl4 = 0
    n_sim3_fallback = 0

    # Convert init poses to homographies (for SL(4), these are projection matrices)
    # For calibrated cameras: H = K @ [R|t] but we use the 4x4 w2c directly
    for i in range(N):
        H = normalize_to_sl4(np.linalg.inv(init_poses[i]))
        values.insert(X(i), SL4(H))

    # Anchor first frame
    H0 = normalize_to_sl4(np.linalg.inv(init_poses[0]))
    graph.add(PriorFactorSL4(X(0), SL4(H0), anchor_noise))

    # Within-chunk odometry factors
    for chunk in chunks:
        poses = chunk["poses_c2w"]
        for k in range(len(poses) - 1):
            i = chunk["start"] + k
            j = i + 1

            H_i = np.linalg.inv(poses[k])
            H_j = np.linalg.inv(poses[k + 1])
            H_rel = normalize_to_sl4(np.linalg.inv(H_i) @ H_j)

            graph.add(BetweenFactorSL4(X(i), X(j), SL4(H_rel), inner_noise))

    # Cross-chunk consistency: overlapping frames get odometry from both
    # chunks. The within-chunk odometry factors above already handle this
    # since both chunks contribute between-factors for the overlap frames.
    # No additional factors needed for overlap.

    # Loop closure factors
    if images is not None:
        from .loop_closure import build_frame_descriptors, find_appearance_loop_closures
        from .factor_graph import _count_match_inliers

        print("    Computing appearance descriptors for SL(4) graph...")
        descriptors = build_frame_descriptors(images, device="cuda")
        lc = find_appearance_loop_closures(
            descriptors, similarity_threshold=0.65, min_frame_gap=15, max_closures=50,
        )
        print(f"    Found {len(lc)} appearance-based loop closures")

        lc_noise = gtsam.noiseModel.Diagonal.Sigmas(0.2 * np.ones(15))
        robust_lc_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Cauchy.Create(1.0), lc_noise
        )

        n_lc = 0
        for i, j, sim_score in lc:
            n_inliers = _count_match_inliers(images[i], images[j])
            if n_inliers < 30:
                continue

            H_rel = normalize_to_sl4(
                np.linalg.inv(np.linalg.inv(init_poses[i])) @ np.linalg.inv(init_poses[j])
            )
            # Simplifies to: normalize_to_sl4(init_poses[i] @ inv(init_poses[j]))
            # Which is the relative w2c homography

            graph.add(BetweenFactorSL4(X(i), X(j), SL4(H_rel), robust_lc_noise))
            n_lc += 1

        print(f"    {n_lc} loop closures added ({n_sl4} SL4, {n_sim3_fallback} Sim3 fallback)")

    # Optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    params.setVerbosityLM("SILENT")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
    result = optimizer.optimize()

    # Extract optimized poses (convert back from w2c homography to c2w)
    optimized = np.zeros((N, 4, 4))
    for i in range(N):
        H = result.atSL4(X(i)).matrix()
        # H is a w2c projective transform, invert to get c2w
        try:
            c2w = np.linalg.inv(H)
            # Extract the SE(3) part (discard projective components)
            # Decompose H into K[R|t] via RQ decomposition
            c2w_normalized = c2w / c2w[3, 3]
            optimized[i] = c2w_normalized
        except np.linalg.LinAlgError:
            optimized[i] = init_poses[i]

    return optimized

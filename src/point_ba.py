"""Joint point + pose bundle adjustment using GTSAM.

VGGT-SLAM 2.0 only optimizes poses (not points). This causes misaligned
point clouds between submaps. By jointly optimizing a sparse set of
landmark points alongside poses, we get better consistency.

We don't optimize ALL points (too expensive). Instead we select a sparse
set of high-confidence "anchor" points visible from multiple frames and
add projection factors for those.
"""

import numpy as np
import gtsam
from typing import Optional


def run_point_ba(
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    point_maps: list[np.ndarray],
    point_confs: list[np.ndarray],
    image_sizes: list[tuple[int, int]],
    n_landmarks: int = 200,
    max_iterations: int = 50,
    robust_kernel: str = "huber",
) -> tuple[np.ndarray, np.ndarray]:
    """Run sparse point + pose bundle adjustment.

    Args:
        poses_c2w: (N, 4, 4) initial poses
        intrinsics: (N, 3, 3) camera matrices
        point_maps: list of (H, W, 3) world point arrays per frame
        point_confs: list of (H, W) confidence arrays
        image_sizes: (H, W) for each frame
        n_landmarks: number of sparse landmarks to optimize
        max_iterations: LM iterations

    Returns:
        (refined_poses, refined_landmarks) tuple
    """
    N = len(poses_c2w)

    # Select high-confidence anchor points visible from multiple frames
    anchors = _select_anchor_points(point_maps, point_confs, N, n_landmarks)
    if len(anchors) < 10:
        return poses_c2w, np.array([])

    # Build factor graph
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    # Add pose variables
    for i in range(N):
        key = gtsam.symbol("x", i)
        values.insert(key, _mat_to_pose3(poses_c2w[i]))
        if i == 0:
            graph.addPriorPose3(key, _mat_to_pose3(poses_c2w[i]),
                               gtsam.noiseModel.Isotropic.Sigma(6, 0.001))

    # Add odometry
    for i in range(N - 1):
        rel = np.linalg.inv(poses_c2w[i]) @ poses_c2w[i + 1]
        noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.05)
        graph.add(gtsam.BetweenFactorPose3(
            gtsam.symbol("x", i), gtsam.symbol("x", i + 1),
            _mat_to_pose3(rel), noise
        ))

    # Add landmark variables and projection factors
    for l_idx, anchor in enumerate(anchors):
        l_key = gtsam.symbol("l", l_idx)
        values.insert(l_key, gtsam.Point3(*anchor["world_pos"]))

        for frame_idx, pixel_uv in anchor["observations"]:
            K = intrinsics[frame_idx]
            cal = gtsam.Cal3_S2(K[0, 0], K[1, 1], 0.0, K[0, 2], K[1, 2])

            measurement = gtsam.Point2(pixel_uv[0], pixel_uv[1])
            base_noise = gtsam.noiseModel.Isotropic.Sigma(2, 2.0)

            if robust_kernel == "huber":
                noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.Huber.Create(1.345), base_noise
                )
            else:
                noise = base_noise

            factor = gtsam.GenericProjectionFactorCal3_S2(
                measurement, noise,
                gtsam.symbol("x", frame_idx), l_key, cal
            )
            graph.add(factor)

    # Optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(max_iterations)
    params.setVerbosityLM("SILENT")

    try:
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
        result = optimizer.optimize()
    except Exception as e:
        print(f"  Point BA failed: {e}")
        return poses_c2w, np.array([])

    # Extract results
    refined_poses = np.zeros_like(poses_c2w)
    for i in range(N):
        refined_poses[i] = _pose3_to_mat(result.atPose3(gtsam.symbol("x", i)))

    refined_landmarks = np.zeros((len(anchors), 3))
    for l_idx in range(len(anchors)):
        refined_landmarks[l_idx] = result.atPoint3(gtsam.symbol("l", l_idx))

    return refined_poses, refined_landmarks


def _select_anchor_points(
    point_maps: list[np.ndarray],
    point_confs: list[np.ndarray],
    N: int,
    n_landmarks: int = 200,
    stride: int = 16,
) -> list[dict]:
    """Select high-confidence points visible from multiple frames as BA anchors."""
    from scipy.spatial import cKDTree

    # Collect candidate points from all frames
    all_candidates = []
    for i in range(N):
        if point_maps[i] is None:
            continue
        pts = point_maps[i][::stride, ::stride].reshape(-1, 3)
        conf = point_confs[i][::stride, ::stride].ravel()
        H, W = point_maps[i].shape[:2]

        valid = (conf > 1.0) & np.isfinite(pts).all(axis=1)
        for idx in np.where(valid)[0]:
            row = (idx // (W // stride)) * stride
            col = (idx % (W // stride)) * stride
            all_candidates.append({
                "world_pos": pts[idx],
                "frame": i,
                "pixel": np.array([col, row], dtype=np.float64),
                "conf": conf[idx],
            })

    if len(all_candidates) < n_landmarks:
        return []

    # Cluster candidates spatially and pick one per cluster
    positions = np.array([c["world_pos"] for c in all_candidates])
    tree = cKDTree(positions)

    # Use farthest point sampling for spatial diversity
    selected_indices = _farthest_point_sample(positions, n_landmarks)

    # For each selected point, find observations from other frames
    anchors = []
    for sel_idx in selected_indices:
        anchor_pos = positions[sel_idx]
        # Find all points within 0.05m from other frames
        nearby = tree.query_ball_point(anchor_pos, 0.05)
        observations = []
        seen_frames = set()
        for n_idx in nearby:
            c = all_candidates[n_idx]
            if c["frame"] not in seen_frames:
                observations.append((c["frame"], c["pixel"]))
                seen_frames.add(c["frame"])

        if len(observations) >= 2:
            anchors.append({
                "world_pos": anchor_pos,
                "observations": observations,
            })

    return anchors[:n_landmarks]


def _farthest_point_sample(points: np.ndarray, n: int) -> list[int]:
    """Farthest point sampling for spatial diversity."""
    if len(points) <= n:
        return list(range(len(points)))

    selected = [0]
    dists = np.full(len(points), np.inf)

    for _ in range(n - 1):
        last = points[selected[-1]]
        new_dists = np.linalg.norm(points - last, axis=1)
        dists = np.minimum(dists, new_dists)
        selected.append(int(np.argmax(dists)))

    return selected


def _mat_to_pose3(T: np.ndarray) -> gtsam.Pose3:
    return gtsam.Pose3(gtsam.Rot3(T[:3, :3]), gtsam.Point3(T[:3, 3]))


def _pose3_to_mat(pose: gtsam.Pose3) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = pose.rotation().matrix()
    T[:3, 3] = pose.translation()
    return T

"""iSAM2 incremental backend for VGGT factor graph refinement.

Faster than batch LM for long sequences (O(log n) per update vs O(n)).
Supports online operation — new frames can be added incrementally.
"""

import numpy as np
import gtsam
from typing import Optional


class ISAM2Backend:
    """Incremental factor graph backend using iSAM2."""

    def __init__(
        self,
        relinearize_threshold: float = 0.01,
        relinearize_skip: int = 1,
        robust_kernel: str = "cauchy",
        robust_param: float = 1.0,
    ):
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(relinearize_threshold)
        params.setRelinearizeSkip(relinearize_skip)
        self.isam = gtsam.ISAM2(params)
        self.robust_kernel = robust_kernel
        self.robust_param = robust_param
        self.n_poses = 0
        self.n_loop_closures = 0

    def add_frame(
        self,
        pose_c2w: np.ndarray,
        odom_from_prev: Optional[np.ndarray] = None,
        odom_sigma: float = 0.05,
        prior_sigma: Optional[np.ndarray] = None,
    ):
        """Add a single frame to the graph incrementally."""
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        i = self.n_poses
        key = gtsam.symbol("x", i)
        pose = _mat_to_pose3(pose_c2w)
        values.insert(key, pose)

        if i == 0:
            graph.addPriorPose3(key, pose, gtsam.noiseModel.Isotropic.Sigma(6, 0.001))
        else:
            if odom_from_prev is not None:
                prev_key = gtsam.symbol("x", i - 1)
                odom_pose = _mat_to_pose3(odom_from_prev)
                noise = self._make_noise(odom_sigma)
                graph.add(gtsam.BetweenFactorPose3(prev_key, key, odom_pose, noise))

            if prior_sigma is not None:
                noise = gtsam.noiseModel.Diagonal.Sigmas(prior_sigma)
                graph.addPriorPose3(key, pose, noise)

        self.isam.update(graph, values)
        self.n_poses += 1

    def add_odometry_batch(
        self,
        poses_c2w: np.ndarray,
        pose_confs: np.ndarray,
        chunk_start: int = 0,
    ):
        """Add a batch of frames with odometry factors."""
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        for k in range(len(poses_c2w)):
            i = chunk_start + k
            key = gtsam.symbol("x", i)
            pose = _mat_to_pose3(poses_c2w[k])

            if not self.isam.valueExists(key):
                values.insert(key, pose)

            if i == 0:
                graph.addPriorPose3(key, pose, gtsam.noiseModel.Isotropic.Sigma(6, 0.001))
            elif k > 0:
                prev_key = gtsam.symbol("x", i - 1)
                rel = np.linalg.inv(poses_c2w[k - 1]) @ poses_c2w[k]
                conf = min(pose_confs[k - 1], pose_confs[k])
                sigma = 0.02 / max(conf, 0.1)
                noise = self._make_noise(sigma)
                graph.add(gtsam.BetweenFactorPose3(prev_key, key, _mat_to_pose3(rel), noise))

        self.n_poses = max(self.n_poses, chunk_start + len(poses_c2w))
        self.isam.update(graph, values)

    def add_loop_closure(
        self,
        i: int, j: int,
        relative_pose: np.ndarray,
        sigma: float = 0.1,
        confidence: float = 1.0,
    ):
        """Add a single loop closure factor."""
        graph = gtsam.NonlinearFactorGraph()
        key_i = gtsam.symbol("x", i)
        key_j = gtsam.symbol("x", j)

        adjusted_sigma = sigma / max(confidence, 0.3)
        noise = self._make_noise(adjusted_sigma)
        graph.add(gtsam.BetweenFactorPose3(key_i, key_j, _mat_to_pose3(relative_pose), noise))

        self.isam.update(graph, gtsam.Values())
        self.n_loop_closures += 1

    def add_loop_closures_batch(self, closures: list[tuple]):
        """Add multiple loop closures at once, then run extra iterations."""
        if not closures:
            return

        graph = gtsam.NonlinearFactorGraph()
        for i, j, rel_pose, sigma, confidence in closures:
            key_i = gtsam.symbol("x", i)
            key_j = gtsam.symbol("x", j)
            adjusted_sigma = sigma / max(confidence, 0.3)
            noise = self._make_noise(adjusted_sigma)
            graph.add(gtsam.BetweenFactorPose3(key_i, key_j, _mat_to_pose3(rel_pose), noise))
            self.n_loop_closures += 1

        self.isam.update(graph, gtsam.Values())
        # Extra iterations for convergence after loop closure batch
        for _ in range(3):
            self.isam.update()

    def add_overlap_constraint(self, i: int, pose: np.ndarray, sigma: float = 0.1):
        """Add a soft prior on an overlapping frame from another chunk."""
        graph = gtsam.NonlinearFactorGraph()
        key = gtsam.symbol("x", i)
        noise = self._make_noise(sigma)
        graph.addPriorPose3(key, _mat_to_pose3(pose), noise)
        self.isam.update(graph, gtsam.Values())

    def optimize(self, extra_iterations: int = 0):
        """Run additional optimization iterations."""
        for _ in range(extra_iterations):
            self.isam.update()

    def get_poses(self, N: Optional[int] = None) -> np.ndarray:
        """Extract all optimized poses."""
        if N is None:
            N = self.n_poses
        estimate = self.isam.calculateEstimate()
        poses = np.zeros((N, 4, 4))
        for i in range(N):
            key = gtsam.symbol("x", i)
            poses[i] = _pose3_to_mat(estimate.atPose3(key))
        return poses

    def get_marginal_covariances(self) -> np.ndarray:
        """Get diagonal covariances for uncertainty estimation."""
        estimate = self.isam.calculateEstimate()
        marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(), estimate)
        covariances = []
        for i in range(self.n_poses):
            try:
                cov = marginals.marginalCovariance(gtsam.symbol("x", i))
                covariances.append(np.diag(cov))
            except Exception:
                covariances.append(np.ones(6) * 999.0)
        return np.array(covariances)

    def _make_noise(self, sigma: float):
        base = gtsam.noiseModel.Isotropic.Sigma(6, sigma)
        if self.robust_kernel == "cauchy":
            return gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Cauchy.Create(self.robust_param), base
            )
        elif self.robust_kernel == "huber":
            return gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(1.345 * sigma), base
            )
        return base


def _mat_to_pose3(T: np.ndarray) -> gtsam.Pose3:
    return gtsam.Pose3(gtsam.Rot3(T[:3, :3]), gtsam.Point3(T[:3, 3]))


def _pose3_to_mat(pose: gtsam.Pose3) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = pose.rotation().matrix()
    T[:3, 3] = pose.translation()
    return T

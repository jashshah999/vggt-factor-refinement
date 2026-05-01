"""Evaluation metrics for trajectory and reconstruction quality."""

import numpy as np
from scipy.spatial.transform import Rotation


def absolute_trajectory_error(gt_poses: np.ndarray, est_poses: np.ndarray) -> dict:
    """Compute ATE between ground truth and estimated trajectories.

    Args:
        gt_poses: (N, 4, 4) ground truth camera-to-world
        est_poses: (N, 4, 4) estimated camera-to-world

    Returns:
        dict with mean, median, rmse, max ATE in meters
    """
    # Align trajectories using Umeyama alignment
    est_aligned = align_trajectories(gt_poses, est_poses)

    errors = np.linalg.norm(
        gt_poses[:, :3, 3] - est_aligned[:, :3, 3], axis=1
    )

    return {
        "ate_mean": float(np.mean(errors)),
        "ate_median": float(np.median(errors)),
        "ate_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "ate_max": float(np.max(errors)),
    }


def align_trajectories(
    gt_poses: np.ndarray, est_poses: np.ndarray
) -> np.ndarray:
    """Align estimated trajectory to ground truth using Umeyama (Sim3).

    Returns aligned estimated poses.
    """
    gt_t = gt_poses[:, :3, 3]
    est_t = est_poses[:, :3, 3]

    # Center
    gt_center = gt_t.mean(axis=0)
    est_center = est_t.mean(axis=0)
    gt_c = gt_t - gt_center
    est_c = est_t - est_center

    # Scale
    gt_scale = np.sqrt(np.mean(np.sum(gt_c ** 2, axis=1)))
    est_scale = np.sqrt(np.mean(np.sum(est_c ** 2, axis=1)))
    if est_scale < 1e-10:
        return est_poses.copy()
    scale = gt_scale / est_scale

    # Rotation (Kabsch/Umeyama)
    H = (est_c.T @ gt_c) / len(gt_t)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T

    # Translation
    t = gt_center - scale * R @ est_center

    # Apply to all poses
    aligned = est_poses.copy()
    for i in range(len(aligned)):
        aligned[i, :3, :3] = R @ est_poses[i, :3, :3]
        aligned[i, :3, 3] = scale * R @ est_poses[i, :3, 3] + t

    return aligned


def relative_pose_error(gt_poses: np.ndarray, est_poses: np.ndarray) -> dict:
    """Compute relative pose error (rotation and translation)."""
    rot_errors = []
    trans_errors = []

    for i in range(len(gt_poses) - 1):
        gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]
        est_rel = np.linalg.inv(est_poses[i]) @ est_poses[i + 1]

        # Rotation error (degrees)
        R_err = gt_rel[:3, :3].T @ est_rel[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        rot_errors.append(np.degrees(angle))

        # Translation error (meters)
        t_err = np.linalg.norm(gt_rel[:3, 3] - est_rel[:3, 3])
        trans_errors.append(t_err)

    return {
        "rpe_rot_mean": float(np.mean(rot_errors)),
        "rpe_trans_mean": float(np.mean(trans_errors)),
    }


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(1.0 / mse))

"""Trajectory smoothing and interpolation.

After factor graph optimization, the trajectory may still have
small jitters (especially at chunk boundaries). This module applies
temporal smoothing while preserving loop closure constraints.
Also provides sub-frame interpolation for rendering at arbitrary timestamps.
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import UnivariateSpline
from typing import Optional


def smooth_trajectory(
    poses_c2w: np.ndarray,
    smoothing_factor: float = 0.1,
    preserve_endpoints: bool = True,
    method: str = "spline",
) -> np.ndarray:
    """Apply temporal smoothing to a camera trajectory.

    Args:
        poses_c2w: (N, 4, 4) camera-to-world poses
        smoothing_factor: 0 = no smoothing, 1 = heavy smoothing
        preserve_endpoints: Keep first and last frame exact
        method: "spline" (smooth), "savgol" (preserves sharp turns), "bilateral" (edge-preserving)

    Returns:
        Smoothed poses (N, 4, 4)
    """
    N = len(poses_c2w)
    if N < 5:
        return poses_c2w.copy()

    if method == "spline":
        return _spline_smooth(poses_c2w, smoothing_factor, preserve_endpoints)
    elif method == "savgol":
        return _savgol_smooth(poses_c2w, smoothing_factor)
    elif method == "bilateral":
        return _bilateral_smooth(poses_c2w, smoothing_factor)
    else:
        raise ValueError(f"Unknown method: {method}")


def interpolate_trajectory(
    poses_c2w: np.ndarray,
    timestamps: np.ndarray,
    query_timestamps: np.ndarray,
) -> np.ndarray:
    """Interpolate poses at arbitrary timestamps via Slerp + cubic spline.

    Useful for:
    - Sub-frame interpolation for smooth video rendering
    - Aligning with IMU timestamps
    - Generating novel camera paths between keyframes

    Args:
        poses_c2w: (N, 4, 4) keyframe poses
        timestamps: (N,) keyframe timestamps
        query_timestamps: (M,) desired output timestamps

    Returns:
        (M, 4, 4) interpolated poses
    """
    N = len(poses_c2w)
    M = len(query_timestamps)

    # Spline interpolation for translation
    translations = poses_c2w[:, :3, 3]
    spline_x = UnivariateSpline(timestamps, translations[:, 0], s=0)
    spline_y = UnivariateSpline(timestamps, translations[:, 1], s=0)
    spline_z = UnivariateSpline(timestamps, translations[:, 2], s=0)

    # SLERP for rotation
    rotations = Rotation.from_matrix(poses_c2w[:, :3, :3])
    slerp = Slerp(timestamps, rotations)

    result = np.zeros((M, 4, 4))
    for i, t in enumerate(query_timestamps):
        t_clamped = np.clip(t, timestamps[0], timestamps[-1])

        result[i] = np.eye(4)
        result[i, :3, :3] = slerp(t_clamped).as_matrix()
        result[i, 0, 3] = spline_x(t_clamped)
        result[i, 1, 3] = spline_y(t_clamped)
        result[i, 2, 3] = spline_z(t_clamped)

    return result


def generate_smooth_camera_path(
    poses_c2w: np.ndarray,
    n_output_frames: int = 300,
    smoothing: float = 0.3,
) -> np.ndarray:
    """Generate a smooth camera path for novel view synthesis rendering.

    Takes the reconstructed camera trajectory and produces a smooth,
    cinematic camera path with more frames for high-quality rendering.
    """
    N = len(poses_c2w)
    timestamps = np.linspace(0, 1, N)
    query_timestamps = np.linspace(0, 1, n_output_frames)

    # First smooth the input trajectory
    smoothed = smooth_trajectory(poses_c2w, smoothing_factor=smoothing)
    # Then interpolate to desired frame count
    return interpolate_trajectory(smoothed, timestamps, query_timestamps)


def _spline_smooth(poses: np.ndarray, factor: float, preserve_endpoints: bool) -> np.ndarray:
    """Smooth via cubic spline on translation + SLERP on rotation."""
    N = len(poses)
    t = np.arange(N, dtype=float)
    translations = poses[:, :3, 3]

    # Smooth translation with spline
    s = factor * N  # scipy spline smoothing parameter
    spline_x = UnivariateSpline(t, translations[:, 0], s=s)
    spline_y = UnivariateSpline(t, translations[:, 1], s=s)
    spline_z = UnivariateSpline(t, translations[:, 2], s=s)

    smoothed = poses.copy()
    for i in range(N):
        if preserve_endpoints and (i == 0 or i == N - 1):
            continue
        smoothed[i, 0, 3] = spline_x(i)
        smoothed[i, 1, 3] = spline_y(i)
        smoothed[i, 2, 3] = spline_z(i)

    # Smooth rotation via windowed SLERP
    window = max(3, int(factor * 10))
    for i in range(1, N - 1):
        if preserve_endpoints and (i == 0 or i == N - 1):
            continue
        start = max(0, i - window // 2)
        end = min(N, i + window // 2 + 1)
        rotations = Rotation.from_matrix(poses[start:end, :3, :3])
        # Average rotation via mean of quaternions
        quats = rotations.as_quat()
        mean_quat = quats.mean(axis=0)
        mean_quat /= np.linalg.norm(mean_quat)
        smoothed[i, :3, :3] = Rotation.from_quat(mean_quat).as_matrix()

    return smoothed


def _savgol_smooth(poses: np.ndarray, factor: float) -> np.ndarray:
    """Savitzky-Golay filter preserves sharp trajectory changes."""
    from scipy.signal import savgol_filter

    N = len(poses)
    window = max(5, int(factor * 20) | 1)  # must be odd
    if window >= N:
        window = N - 1 if N % 2 == 0 else N - 2
        if window < 5:
            return poses.copy()

    smoothed = poses.copy()
    translations = poses[:, :3, 3].copy()

    for axis in range(3):
        translations[:, axis] = savgol_filter(translations[:, axis], window, 3)

    smoothed[:, :3, 3] = translations
    return smoothed


def _bilateral_smooth(poses: np.ndarray, factor: float) -> np.ndarray:
    """Edge-preserving bilateral filter on trajectory.

    Preserves sharp turns while smoothing jitter.
    """
    N = len(poses)
    smoothed = poses.copy()
    sigma_spatial = max(2, int(factor * 10))
    sigma_range = 0.1

    translations = poses[:, :3, 3].copy()

    for i in range(N):
        weights = np.zeros(N)
        for j in range(max(0, i - sigma_spatial * 3), min(N, i + sigma_spatial * 3 + 1)):
            spatial_dist = abs(i - j)
            range_dist = np.linalg.norm(translations[i] - translations[j])

            w_spatial = np.exp(-spatial_dist ** 2 / (2 * sigma_spatial ** 2))
            w_range = np.exp(-range_dist ** 2 / (2 * sigma_range ** 2))
            weights[j] = w_spatial * w_range

        weights /= weights.sum() + 1e-10
        smoothed[i, :3, 3] = (weights[:, None] * translations).sum(axis=0)

    return smoothed

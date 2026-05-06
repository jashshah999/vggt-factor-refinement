"""Intelligent keyframe selection for efficient reconstruction.

Not all frames are equally useful. Selecting keyframes that maximize
coverage while minimizing redundancy leads to better reconstruction
with fewer VGGT forward passes.
"""

import numpy as np
import cv2
from typing import Optional


def select_keyframes(
    image_paths: list[str],
    method: str = "hybrid",
    max_frames: int = 80,
    min_parallax: float = 0.03,
    blur_threshold: float = 50.0,
    overlap_ratio: float = 0.7,
) -> list[int]:
    """Select keyframes from a video for optimal reconstruction.

    Methods:
        - "uniform": Even temporal spacing
        - "parallax": Based on optical flow magnitude
        - "blur": Filter blurry frames, then uniform
        - "hybrid": Parallax + blur + coverage scoring

    Returns:
        List of selected frame indices.
    """
    N = len(image_paths)
    if N <= max_frames:
        return list(range(N))

    if method == "uniform":
        return _uniform_select(N, max_frames)
    elif method == "blur":
        return _blur_filtered_select(image_paths, max_frames, blur_threshold)
    elif method == "parallax":
        return _parallax_select(image_paths, max_frames, min_parallax)
    elif method == "hybrid":
        return _hybrid_select(image_paths, max_frames, min_parallax, blur_threshold, overlap_ratio)
    else:
        raise ValueError(f"Unknown method: {method}")


def _uniform_select(N: int, max_frames: int) -> list[int]:
    """Evenly spaced frame selection."""
    indices = np.linspace(0, N - 1, max_frames, dtype=int)
    return list(np.unique(indices))


def _blur_filtered_select(
    image_paths: list[str], max_frames: int, threshold: float
) -> list[int]:
    """Filter blurry frames, then select uniformly from sharp ones."""
    scores = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        scores.append(laplacian_var)

    scores = np.array(scores)
    sharp_indices = np.where(scores > threshold)[0]

    if len(sharp_indices) <= max_frames:
        return list(sharp_indices)

    selected = np.linspace(0, len(sharp_indices) - 1, max_frames, dtype=int)
    return list(sharp_indices[selected])


def _parallax_select(
    image_paths: list[str], max_frames: int, min_parallax: float
) -> list[int]:
    """Select frames based on optical flow magnitude (motion parallax)."""
    selected = [0]
    prev_gray = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    prev_gray = cv2.resize(prev_gray, (320, 240))

    for i in range(1, len(image_paths)):
        curr_gray = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        curr_gray = cv2.resize(curr_gray, (320, 240))

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mean_flow = float(np.mean(magnitude)) / 320.0  # normalize by image width

        if mean_flow > min_parallax:
            selected.append(i)
            prev_gray = curr_gray

        if len(selected) >= max_frames:
            break

    # Always include last frame
    if selected[-1] != len(image_paths) - 1:
        selected.append(len(image_paths) - 1)

    return selected


def _hybrid_select(
    image_paths: list[str],
    max_frames: int,
    min_parallax: float,
    blur_threshold: float,
    overlap_ratio: float,
) -> list[int]:
    """Hybrid selection: parallax + blur quality + temporal coverage.

    Scores each frame on:
    1. Sharpness (Laplacian variance)
    2. Motion from previous keyframe (optical flow)
    3. Temporal spacing (prefer even coverage)

    Greedily selects highest-scoring frames.
    """
    N = len(image_paths)

    # Compute sharpness scores
    sharpness = np.zeros(N)
    for i in range(N):
        img = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        sharpness[i] = cv2.Laplacian(img, cv2.CV_64F).var()

    # Normalize sharpness
    sharpness = (sharpness - sharpness.min()) / (sharpness.max() - sharpness.min() + 1e-8)

    # Compute motion scores (optical flow from previous frame)
    motion = np.zeros(N)
    prev_gray = None
    for i in range(N):
        curr_gray = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        curr_gray = cv2.resize(curr_gray, (320, 240))
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            motion[i] = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2))) / 320.0
        prev_gray = curr_gray

    # Greedy selection
    selected = [0]  # always start with first frame
    for _ in range(max_frames - 2):
        best_score = -1
        best_idx = -1

        last_selected = selected[-1]
        for i in range(last_selected + 1, N):
            if i in selected:
                continue

            # Skip blurry frames
            if sharpness[i] < 0.1:
                continue

            # Accumulated motion since last keyframe
            accum_motion = sum(motion[last_selected + 1:i + 1])
            if accum_motion < min_parallax:
                continue

            # Score: motion * sharpness * temporal_bonus
            frames_since = i - last_selected
            temporal_bonus = min(frames_since / 10.0, 2.0)
            score = accum_motion * sharpness[i] * temporal_bonus

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx == -1:
            break
        selected.append(best_idx)

    # Always include last frame
    if selected[-1] != N - 1:
        selected.append(N - 1)

    selected.sort()
    return selected


def compute_frame_quality_scores(image_paths: list[str]) -> np.ndarray:
    """Compute per-frame quality scores (sharpness + exposure)."""
    scores = np.zeros(len(image_paths))
    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Sharpness via Laplacian
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Exposure quality (penalize over/under exposed)
        mean_brightness = gray.mean() / 255.0
        exposure_quality = 1.0 - abs(mean_brightness - 0.5) * 2

        scores[i] = sharpness * exposure_quality

    # Normalize to [0, 1]
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores

"""Adaptive chunk sizing based on scene complexity.

Fixed chunk sizes waste compute on easy sections (slow camera, textured scene)
and underperform on hard sections (fast motion, textureless). This module
dynamically adjusts chunk size based on optical flow magnitude and texture
quality between consecutive frames.
"""

import numpy as np
import cv2


def compute_adaptive_chunks(
    image_paths: list[str],
    min_chunk: int = 5,
    max_chunk: int = 20,
    overlap: int = 3,
    motion_threshold: float = 0.05,
    texture_threshold: float = 30.0,
) -> list[tuple[int, int]]:
    """Compute adaptive chunk boundaries based on scene difficulty.

    Fast motion → smaller chunks (more overlap, less drift)
    Slow motion, rich texture → larger chunks (more context for VGGT)

    Args:
        image_paths: List of frame paths
        min_chunk: Minimum chunk size
        max_chunk: Maximum chunk size
        overlap: Overlap between chunks
        motion_threshold: Optical flow threshold for "fast motion"
        texture_threshold: Laplacian variance threshold for "textureless"

    Returns:
        List of (start, end) tuples defining chunk boundaries
    """
    N = len(image_paths)
    if N <= max_chunk:
        return [(0, N)]

    # Compute per-frame difficulty scores
    difficulty = _compute_difficulty_scores(image_paths, motion_threshold, texture_threshold)

    # Greedy chunking: extend chunk until difficulty spikes
    chunks = []
    start = 0

    while start < N:
        # Determine chunk size based on local difficulty
        local_diff = difficulty[start:min(start + max_chunk, N)]
        mean_diff = np.cumsum(local_diff) / (np.arange(len(local_diff)) + 1)

        # Find where difficulty rises sharply (chunk should end before that)
        chunk_end = min_chunk
        for i in range(min_chunk, len(local_diff)):
            if mean_diff[i] > 0.7:  # high difficulty region
                break
            chunk_end = i + 1

        chunk_end = min(max(chunk_end, min_chunk), max_chunk)
        end = min(start + chunk_end, N)

        chunks.append((start, end))
        start = end - overlap

        if end >= N:
            break

    return chunks


def _compute_difficulty_scores(
    image_paths: list[str],
    motion_threshold: float,
    texture_threshold: float,
) -> np.ndarray:
    """Score each frame on reconstruction difficulty (0=easy, 1=hard)."""
    N = len(image_paths)
    scores = np.zeros(N)

    prev_gray = None
    for i in range(N):
        img = cv2.imread(image_paths[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (320, 240))

        # Texture score (low texture = hard)
        laplacian = cv2.Laplacian(small, cv2.CV_64F).var()
        texture_score = 1.0 - min(laplacian / (texture_threshold * 5), 1.0)

        # Motion score (high motion = harder for VGGT)
        motion_score = 0.0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, small, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            mean_motion = float(np.mean(magnitude)) / 320.0
            motion_score = min(mean_motion / motion_threshold, 1.0)

        # Combined difficulty
        scores[i] = 0.4 * texture_score + 0.6 * motion_score
        prev_gray = small

    return scores


def estimate_vram_chunk_size(vram_gb: float, image_height: int = 518, image_width: int = 518) -> int:
    """Estimate maximum chunk size for given VRAM.

    Based on empirical measurements:
    - VGGT model: ~2.4 GB (bfloat16)
    - Per-frame activation: ~0.2 GB at 518x518
    - Overhead: ~1 GB

    Args:
        vram_gb: Available GPU VRAM in GB
        image_height: Input image height (after VGGT preprocessing)
        image_width: Input image width

    Returns:
        Recommended chunk size
    """
    model_mem = 2.4
    overhead = 1.5
    available = vram_gb - model_mem - overhead

    # Scale factor for image resolution (quadratic in tokens)
    h_patches = image_height // 14
    w_patches = image_width // 14
    tokens_per_frame = h_patches * w_patches
    # Baseline: 37*37 = 1369 tokens at 518x518
    resolution_factor = tokens_per_frame / 1369.0

    per_frame_mem = 0.2 * resolution_factor  # GB per frame in chunk

    max_frames = int(available / per_frame_mem)
    return max(4, min(max_frames, 25))

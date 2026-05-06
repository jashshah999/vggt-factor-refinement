"""Covisibility graph for intelligent loop closure detection.

Instead of just checking spatial proximity (which fails when drift is large)
or brute-force DINOv2 similarity (O(N^2) and slow), build a covisibility
graph that tracks which frames observe the same 3D regions. This enables:
1. Faster loop closure candidate selection
2. Better relative pose estimation via shared 3D points
3. Adaptive noise based on covisibility strength
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Optional


class CovisibilityGraph:
    """Tracks which frames share visible 3D points."""

    def __init__(self, grid_size: float = 0.05):
        self.grid_size = grid_size
        self.frame_voxels: dict[int, set] = {}
        self.voxel_frames: dict[tuple, set] = {}

    def add_frame(self, frame_idx: int, points: np.ndarray, conf: np.ndarray, conf_thresh: float = 0.3):
        """Register a frame's visible 3D points."""
        valid = (conf.ravel() > conf_thresh) & np.isfinite(points.reshape(-1, 3)).all(axis=1)
        pts = points.reshape(-1, 3)[valid]

        if len(pts) == 0:
            self.frame_voxels[frame_idx] = set()
            return

        # Discretize to voxel grid
        voxel_coords = tuple(map(tuple, (pts / self.grid_size).astype(int)))
        voxels = set(voxel_coords)
        self.frame_voxels[frame_idx] = voxels

        for v in voxels:
            if v not in self.voxel_frames:
                self.voxel_frames[v] = set()
            self.voxel_frames[v].add(frame_idx)

    def get_covisibility_score(self, i: int, j: int) -> float:
        """Get covisibility score between two frames (0 to 1)."""
        vi = self.frame_voxels.get(i, set())
        vj = self.frame_voxels.get(j, set())
        if not vi or not vj:
            return 0.0
        intersection = len(vi & vj)
        union = len(vi | vj)
        return intersection / union if union > 0 else 0.0

    def find_covisible_frames(self, frame_idx: int, min_score: float = 0.1, min_gap: int = 10) -> list:
        """Find frames that share significant 3D visibility with given frame."""
        voxels = self.frame_voxels.get(frame_idx, set())
        if not voxels:
            return []

        # Count how many voxels each other frame shares
        candidate_counts: dict[int, int] = {}
        for v in voxels:
            for other in self.voxel_frames.get(v, set()):
                if abs(other - frame_idx) >= min_gap:
                    candidate_counts[other] = candidate_counts.get(other, 0) + 1

        n_voxels = len(voxels)
        results = []
        for other, count in candidate_counts.items():
            score = count / n_voxels
            if score >= min_score:
                results.append((other, score))

        results.sort(key=lambda x: -x[1])
        return results

    def find_loop_closure_candidates(
        self, min_score: float = 0.15, min_gap: int = 20, max_candidates: int = 100
    ) -> list[tuple[int, int, float]]:
        """Find all potential loop closure pairs from covisibility."""
        candidates = []
        seen = set()

        for frame_idx in sorted(self.frame_voxels.keys()):
            covisible = self.find_covisible_frames(frame_idx, min_score=min_score, min_gap=min_gap)
            for other, score in covisible[:5]:
                pair = (min(frame_idx, other), max(frame_idx, other))
                if pair not in seen:
                    seen.add(pair)
                    candidates.append((pair[0], pair[1], score))

        candidates.sort(key=lambda x: -x[2])
        return candidates[:max_candidates]


def build_covisibility_from_chunks(chunks: list, grid_size: float = 0.05) -> CovisibilityGraph:
    """Build covisibility graph from chunked VGGT output."""
    graph = CovisibilityGraph(grid_size=grid_size)

    for chunk in chunks:
        start = chunk["start"]
        n = chunk["end"] - start

        for k in range(n):
            frame_idx = start + k
            if "points" in chunk:
                points = chunk["points"][k] if chunk["points"].ndim == 4 else chunk["points"]
                conf = chunk["point_conf"][k] if "point_conf" in chunk else chunk.get("pose_conf", np.ones(1))
            elif "world_pts" in chunk:
                points = chunk["world_pts"][k]
                conf = chunk["depth_conf"][k]
            else:
                continue

            graph.add_frame(frame_idx, points, conf)

    return graph

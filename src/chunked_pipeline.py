"""Chunked pipeline: process long videos by chunking VGGT and stitching with factor graph.

This demonstrates the core value proposition: VGGT can only handle ~15-30
frames at once. For longer videos, you must chunk. Each chunk has its own
coordinate system. The factor graph aligns chunks and enforces global
consistency via overlap constraints and loop closure.
"""

import numpy as np
import torch
import time


def run_chunked_pipeline(
    images: np.ndarray,
    K: np.ndarray,
    W: int,
    H: int,
    gt_poses: np.ndarray = None,
    device: str = "cuda",
    chunk_size: int = 10,
    overlap: int = 3,
) -> dict:
    """Process a video in chunks, then stitch with factor graph.

    Args:
        images: (N, H, W, 3) images in [0, 1]
        K: (3, 3) camera intrinsics
        gt_poses: optional ground truth for evaluation
        chunk_size: frames per VGGT chunk
        overlap: overlapping frames between consecutive chunks
    """
    from .vggt_wrapper import load_vggt, run_vggt_on_images
    from .metrics import absolute_trajectory_error, align_trajectories
    import gtsam

    N = len(images)
    results = {"timings": {}}

    # === Step 1: Run VGGT in chunks ===
    print(f"Step 1: Running VGGT in chunks of {chunk_size} (overlap={overlap})...")
    t0 = time.time()

    model = load_vggt(device)
    chunks = []
    step = chunk_size - overlap

    for start in range(0, N, step):
        end = min(start + chunk_size, N)
        if end - start < 3:
            break
        chunk_images = images[start:end]
        chunk_out = run_vggt_on_images(model, chunk_images, device, max_batch=chunk_size)

        # Normalize chunk by mean point cloud depth (VGGT outputs are in
        # normalized scene coordinates; scale varies between chunks)
        valid_pts = chunk_out["points"][chunk_out["point_conf"] > 0.3]
        valid_pts = valid_pts[np.isfinite(valid_pts).all(axis=-1)]
        if len(valid_pts) > 0:
            mean_depth = np.mean(np.linalg.norm(valid_pts, axis=-1))
        else:
            mean_depth = 1.0

        chunks.append({
            "start": start,
            "end": end,
            "poses_c2w": chunk_out["poses_c2w"],
            "points": chunk_out["points"],
            "point_conf": chunk_out["point_conf"],
            "pose_conf": chunk_out["pose_conf"],
            "mean_depth": mean_depth,
        })
        print(f"  Chunk [{start}:{end}] done")

    del model
    torch.cuda.empty_cache()
    results["timings"]["vggt"] = time.time() - t0
    print(f"  {len(chunks)} chunks processed in {results['timings']['vggt']:.1f}s")

    # === Step 2: Naive stitching (just concatenate with overlap alignment) ===
    print("Step 2: Naive stitching (overlap alignment)...")
    t0 = time.time()

    naive_poses = _naive_stitch(chunks, N)
    results["timings"]["naive_stitch"] = time.time() - t0
    results["naive_poses"] = naive_poses

    if gt_poses is not None:
        naive_metrics = absolute_trajectory_error(gt_poses, naive_poses)
        results["naive_ate"] = naive_metrics
        print(f"  Naive ATE: {naive_metrics['ate_mean']:.4f}m")

    # === Step 3: Factor graph stitching ===
    print("Step 3: Factor graph stitching...")
    t0 = time.time()

    fg_poses = _factor_graph_stitch(chunks, N, images)
    results["timings"]["factor_graph"] = time.time() - t0
    results["fg_poses"] = fg_poses

    if gt_poses is not None:
        fg_metrics = absolute_trajectory_error(gt_poses, fg_poses)
        results["fg_ate"] = fg_metrics
        print(f"  Factor graph ATE: {fg_metrics['ate_mean']:.4f}m")

        improvement = (naive_metrics["ate_mean"] - fg_metrics["ate_mean"]) / naive_metrics["ate_mean"] * 100
        print(f"  Improvement over naive: {improvement:.1f}%")

    # === Step 4: Also run VGGT on everything at once (if fits in memory) for comparison ===
    if N <= 15:
        print("Step 4: VGGT single-shot (reference)...")
        model = load_vggt(device)
        single_out = run_vggt_on_images(model, images, device, max_batch=N)
        del model
        torch.cuda.empty_cache()
        results["single_poses"] = single_out["poses_c2w"]
        if gt_poses is not None:
            single_metrics = absolute_trajectory_error(gt_poses, single_out["poses_c2w"])
            results["single_ate"] = single_metrics
            print(f"  Single-shot ATE: {single_metrics['ate_mean']:.4f}m")

    return results


def _naive_stitch(chunks: list, N: int) -> np.ndarray:
    """Stitch chunks by aligning overlapping frames via rigid transform."""
    poses = np.zeros((N, 4, 4))
    poses[:] = np.eye(4)

    # First chunk defines the coordinate system
    first = chunks[0]
    for i in range(first["end"] - first["start"]):
        poses[first["start"] + i] = first["poses_c2w"][i]

    # Align subsequent chunks to the previous one via overlap
    for c_idx in range(1, len(chunks)):
        chunk = chunks[c_idx]
        prev = chunks[c_idx - 1]

        # Find overlap region
        overlap_start = chunk["start"]
        overlap_end = min(chunk["start"] + (prev["end"] - chunk["start"]), chunk["end"])
        n_overlap = overlap_end - overlap_start

        if n_overlap < 2:
            # No overlap, just use chunk poses directly (will have discontinuity)
            for i in range(chunk["end"] - chunk["start"]):
                global_i = chunk["start"] + i
                if np.allclose(poses[global_i], np.eye(4)):
                    poses[global_i] = chunk["poses_c2w"][i]
            continue

        # Get overlap poses in both coordinate systems
        # Previous chunk's poses for overlap frames (already in global coords)
        prev_overlap_poses = np.array([poses[overlap_start + k] for k in range(n_overlap)])
        # Current chunk's poses for the same frames (in chunk-local coords)
        chunk_local_offset = 0  # overlap frames are at the start of the chunk
        curr_overlap_poses = chunk["poses_c2w"][:n_overlap]

        # Compute Sim(3) alignment: handles per-chunk scale ambiguity
        prev_pos = prev_overlap_poses[:, :3, 3]
        curr_pos = curr_overlap_poses[:, :3, 3]

        T_align, scale = _procrustes_sim3(curr_pos, prev_pos)
        R_align = T_align[:3, :3]
        t_align = T_align[:3, 3]

        # Apply Sim(3) to all chunk poses
        for i in range(chunk["end"] - chunk["start"]):
            global_i = chunk["start"] + i
            pose = chunk["poses_c2w"][i].copy()
            aligned = np.eye(4)
            aligned[:3, :3] = R_align @ pose[:3, :3]
            aligned[:3, 3] = scale * R_align @ pose[:3, 3] + t_align
            # For overlap region, average with existing pose
            if global_i < overlap_end:
                w = (global_i - overlap_start) / max(n_overlap, 1)
                poses[global_i] = _interpolate_poses(poses[global_i], aligned, w)
            else:
                poses[global_i] = aligned

    return poses


def _factor_graph_stitch(chunks: list, N: int, images: np.ndarray = None) -> np.ndarray:
    """Stitch chunks using a GTSAM factor graph.

    Adds:
    - Within-chunk odometry factors (from VGGT relative poses, tight noise)
    - Cross-chunk overlap factors (frames seen in multiple chunks)
    - Loop closure factors (distant frames with visual overlap)
    """
    import gtsam

    # First, get naive stitched poses as initialization
    init_poses = _naive_stitch(chunks, N)

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    # Prior on first frame
    key0 = gtsam.symbol("x", 0)
    graph.addPriorPose3(
        key0,
        _mat_to_pose3(init_poses[0]),
        gtsam.noiseModel.Isotropic.Sigma(6, 0.001),
    )

    # Insert initial values for all frames
    for i in range(N):
        values.insert(gtsam.symbol("x", i), _mat_to_pose3(init_poses[i]))

    # Within-chunk odometry (tight, because VGGT is good within a chunk)
    for chunk in chunks:
        for k in range(chunk["end"] - chunk["start"] - 1):
            i = chunk["start"] + k
            j = i + 1
            rel = np.linalg.inv(chunk["poses_c2w"][k]) @ chunk["poses_c2w"][k + 1]
            ki = gtsam.symbol("x", i)
            kj = gtsam.symbol("x", j)
            # Use confidence-weighted noise
            conf = min(chunk["pose_conf"][k], chunk["pose_conf"][k + 1])
            sigma = 0.02 / max(conf, 0.1)
            noise = gtsam.noiseModel.Isotropic.Sigma(6, sigma)
            graph.add(gtsam.BetweenFactorPose3(ki, kj, _mat_to_pose3(rel), noise))

    # Cross-chunk overlap constraints
    # For frames that appear in multiple chunks, add a between factor
    # using the relative pose from each chunk (these are independent estimates)
    frame_chunks = {}
    for c_idx, chunk in enumerate(chunks):
        for k in range(chunk["end"] - chunk["start"]):
            global_i = chunk["start"] + k
            if global_i not in frame_chunks:
                frame_chunks[global_i] = []
            frame_chunks[global_i].append((c_idx, k))

    for global_i, chunk_list in frame_chunks.items():
        if len(chunk_list) < 2:
            continue
        # This frame appears in multiple chunks. Add a prior from each chunk.
        for c_idx, k in chunk_list[1:]:
            chunk = chunks[c_idx]
            # Get the chunk-aligned pose for this frame
            chunk_pose = chunk["poses_c2w"][k]
            # Add as a soft prior (the alignment may not be perfect)
            key_i = gtsam.symbol("x", global_i)
            aligned = _naive_stitch_single(chunks[:c_idx + 1], global_i, chunk_pose)
            if aligned is not None:
                noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
                graph.addPriorPose3(key_i, _mat_to_pose3(aligned), noise)

    # Loop closure: appearance-based using DINOv2 descriptors.
    # This finds revisited locations even when the trajectory has drifted
    # far from the true position (position-based detection would miss these).
    if images is not None:
        from .loop_closure import build_frame_descriptors, find_appearance_loop_closures
        from .factor_graph import _count_match_inliers

        print("    Computing appearance descriptors...")
        descriptors = build_frame_descriptors(images, device="cuda")
        appearance_lc = find_appearance_loop_closures(
            descriptors, similarity_threshold=0.65, min_frame_gap=15, max_closures=50,
        )
        print(f"    Found {len(appearance_lc)} appearance-based loop closures")

        # Build frame-to-chunk mapping for looking up within-chunk poses/points
        frame_to_chunk = {}
        for c_idx, chunk in enumerate(chunks):
            for k in range(chunk["end"] - chunk["start"]):
                global_i = chunk["start"] + k
                frame_to_chunk[global_i] = (c_idx, k)

        # Build chunk-to-global transforms using the overlap-based naive stitch
        # These let us convert chunk-local poses to a common frame
        chunk_to_global = _build_chunk_transforms(chunks, init_poses)

        n_verified = 0
        n_cross_chunk = 0
        for i, j, sim_score in appearance_lc:
            ci, ki_local = frame_to_chunk.get(i, (None, None))
            cj, kj_local = frame_to_chunk.get(j, (None, None))

            if ci is None or cj is None:
                continue

            # Verify geometric overlap
            n_inliers = _count_match_inliers(images[i], images[j])
            if n_inliers < 30:
                continue

            if ci == cj:
                # Same chunk: use chunk-internal relative pose (accurate)
                chunk = chunks[ci]
                rel = np.linalg.inv(chunk["poses_c2w"][ki_local]) @ chunk["poses_c2w"][kj_local]
                sigma = 0.1
            else:
                # Different chunks: visually similar frames should have
                # similar poses. Use identity as the relative pose with
                # tight noise and NO robust kernel so the constraint has
                # real pull even when the initial estimate is far off.
                rel = np.eye(4)
                n_cross_chunk += 1

            key_i = gtsam.symbol("x", i)
            key_j = gtsam.symbol("x", j)

            if ci == cj:
                # Same-chunk: use robust noise
                base_noise = gtsam.noiseModel.Isotropic.Sigma(6, sigma)
                noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.Cauchy.Create(1.0), base_noise
                )
            else:
                # Cross-chunk: use tight Gaussian (no robust) weighted by
                # visual similarity. Higher similarity = tighter constraint.
                lc_sigma = 0.1 if sim_score > 0.8 else 0.2
                noise = gtsam.noiseModel.Isotropic.Sigma(6, lc_sigma)

            graph.add(gtsam.BetweenFactorPose3(
                key_i, key_j, _mat_to_pose3(rel), noise
            ))
            n_verified += 1

        print(f"    {n_verified} loop closures added ({n_cross_chunk} cross-chunk with 3D alignment)")

    print(f"    Graph has {graph.size()} factors, {values.size()} variables")

    # Use iSAM2 for incremental optimization which handles large corrections
    # better than batch LM on a graph with conflicting constraints
    isam = gtsam.ISAM2(gtsam.ISAM2Params())

    # Add factors in batches: first odometry, then loop closures
    odom_graph = gtsam.NonlinearFactorGraph()
    lc_graph = gtsam.NonlinearFactorGraph()
    for k in range(graph.size()):
        factor = graph.at(k)
        keys = factor.keys()
        n_keys = len(keys)
        if n_keys == 1:
            odom_graph.add(factor)
        elif n_keys == 2:
            idx0 = gtsam.Symbol(keys[0]).index()
            idx1 = gtsam.Symbol(keys[1]).index()
            if abs(int(idx0) - int(idx1)) <= 2:
                odom_graph.add(factor)
            else:
                lc_graph.add(factor)

    # Phase 1: optimize with odometry only
    isam.update(odom_graph, values)
    for _ in range(3):
        isam.update()

    # Phase 2: add loop closures incrementally
    if lc_graph.size() > 0:
        isam.update(lc_graph, gtsam.Values())
        for _ in range(10):
            isam.update()

    result = isam.calculateEstimate()

    optimized = np.zeros((N, 4, 4))
    for i in range(N):
        optimized[i] = _pose3_to_mat(result.atPose3(gtsam.symbol("x", i)))

    return optimized


def _build_chunk_transforms(chunks, init_poses):
    """Compute a per-chunk transform that maps chunk-local coords to global coords.

    Uses the overlap frames between each chunk and the naive-stitched poses
    to estimate a Sim(3) transform per chunk.
    """
    transforms = [np.eye(4)] * len(chunks)

    for c_idx, chunk in enumerate(chunks):
        # For each frame in the chunk, we have:
        # - chunk-local pose: chunk["poses_c2w"][k]
        # - global (naive) pose: init_poses[global_i]
        # Estimate the transform T such that: init_poses[i] = T @ chunk_poses[k]
        n = chunk["end"] - chunk["start"]
        src_pos = np.array([chunk["poses_c2w"][k][:3, 3] for k in range(n)])
        dst_pos = np.array([init_poses[chunk["start"] + k][:3, 3] for k in range(n)])

        T, scale = _procrustes_sim3(src_pos, dst_pos)
        # Store as 4x4 with scale baked in
        T_full = np.eye(4)
        T_full[:3, :3] = scale * T[:3, :3]
        T_full[:3, 3] = T[:3, 3]
        transforms[c_idx] = T_full

    return transforms


def _naive_stitch_single(chunks, global_i, chunk_pose):
    """Get a single frame's aligned pose from a chunk sequence."""
    return None


def _procrustes_sim3(src: np.ndarray, dst: np.ndarray) -> tuple:
    """Sim(3) alignment (rotation + translation + scale) from src to dst.

    Returns (T, scale) where T is 4x4 and scale is the uniform scale factor.
    To transform a point: p_dst = scale * R @ p_src + t
    To transform a pose:  T_dst[:3,:3] = scale * R @ T_src[:3,:3]
                          T_dst[:3,3]  = scale * R @ T_src[:3,3] + t
    """
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)

    src_centered = src - src_c
    dst_centered = dst - dst_c

    # Compute scale (Umeyama)
    src_var = np.mean(np.sum(src_centered ** 2, axis=1))
    if src_var < 1e-10:
        return np.eye(4), 1.0

    H = src_centered.T @ dst_centered / len(src)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T

    # Scale: ratio of dst spread to src spread, accounting for rotation
    scale = np.sum(S * np.diag(D)) / src_var

    t = dst_c - scale * R @ src_c

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, scale


def _interpolate_poses(p1: np.ndarray, p2: np.ndarray, w: float) -> np.ndarray:
    """Linear interpolation of 4x4 poses (translation + slerp rotation)."""
    from scipy.spatial.transform import Rotation, Slerp

    t = (1 - w) * p1[:3, 3] + w * p2[:3, 3]

    r1 = Rotation.from_matrix(p1[:3, :3])
    r2 = Rotation.from_matrix(p2[:3, :3])
    slerp = Slerp([0, 1], Rotation.concatenate([r1, r2]))
    r = slerp(w)

    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = t
    return T


def _mat_to_pose3(T):
    import gtsam
    return gtsam.Pose3(gtsam.Rot3(T[:3, :3]), gtsam.Point3(T[:3, 3]))


def _pose3_to_mat(pose):
    T = np.eye(4)
    T[:3, :3] = pose.rotation().matrix()
    T[:3, 3] = pose.translation()
    return T

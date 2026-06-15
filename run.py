"""Run VGGT + Factor Graph on a video or image directory.

One command to go from video → globally consistent poses + multiple export formats.

Usage:
    python run.py --video my_video.mp4 --output scene/
    python run.py --images path/to/frames/ --output scene/ --export colmap,nerfstudio,ply
    python run.py --video my_video.mp4 --output scene/ --export all --train-gaussians
"""

import argparse
import os
import sys
import json
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))


def load_video_frames(video_path, max_frames=100, target_size=None):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n = min(total, max_frames)
    indices = np.linspace(0, total - 1, n, dtype=int)

    frames = []
    frame_paths = []
    os.makedirs("/tmp/vggt_frames", exist_ok=True)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # Save frame for downstream use
        frame_path = f"/tmp/vggt_frames/frame_{len(frames):04d}.jpg"
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if target_size:
            frame = cv2.resize(frame, target_size)
        frames.append(frame)
    cap.release()

    H, W = frames[0].shape[:2]
    fx = W / (2 * np.tan(np.radians(30)))
    K = np.array([[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]], dtype=np.float64)

    return np.array(frames), K, W, H, frame_paths


def load_image_dir(image_dir, max_frames=100, target_size=None):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    from pathlib import Path
    paths = sorted([p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts])
    if not paths:
        raise ValueError(f"No images in {image_dir}")

    n = min(len(paths), max_frames)
    indices = np.linspace(0, len(paths) - 1, n, dtype=int)
    paths = [paths[i] for i in indices]

    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if target_size:
            img = cv2.resize(img, target_size)
        frames.append(img)

    H, W = frames[0].shape[:2]
    fx = W / (2 * np.tan(np.radians(30)))
    K = np.array([[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]], dtype=np.float64)

    return np.array(frames), K, W, H, [str(p) for p in paths]


def main():
    parser = argparse.ArgumentParser(description="VGGT + Factor Graph: Video → 3D")
    parser.add_argument("--video", help="Path to input video")
    parser.add_argument("--images", help="Path to image directory")
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--output", "-o", default="output/scene")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--no-factor-graph", action="store_true")
    parser.add_argument("--export", default="all",
                       help="Export formats: colmap,nerfstudio,ply,splat,all (comma-separated)")
    parser.add_argument("--train-gaussians", action="store_true",
                       help="Train Gaussian Splatting on the refined poses")
    parser.add_argument("--train-iters", type=int, default=3000)
    args = parser.parse_args()

    t_start = time.time()

    if args.video:
        print(f"Loading video: {args.video}")
        images, K, W, H, image_paths = load_video_frames(args.video, args.max_frames)
    elif args.images:
        print(f"Loading images: {args.images}")
        images, K, W, H, image_paths = load_image_dir(args.images, args.max_frames)
    else:
        parser.error("Provide --video or --images")

    N = len(images)
    print(f"  {N} frames, {W}x{H}")

    # Run pipeline
    from src.chunked_pipeline import run_chunked_pipeline
    results = run_chunked_pipeline(
        images=images, K=K, W=W, H=H,
        device=args.device,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    # Use factor graph poses (best available)
    if "fg_poses" in results and not args.no_factor_graph:
        poses = results["fg_poses"]
        method = "factor_graph"
    elif "isam_poses" in results:
        poses = results["isam_poses"]
        method = "isam2"
    else:
        poses = results["naive_poses"]
        method = "naive"

    print(f"\nUsing {method} poses for export")

    # Build intrinsics array (same K for all frames in this case)
    intrinsics = np.tile(K, (N, 1, 1))
    image_sizes = [(H, W)] * N

    # Collect point cloud from chunks
    points, colors = _collect_points(results, images, poses)

    # Export
    os.makedirs(args.output, exist_ok=True)
    export_formats = args.export.split(",") if args.export != "all" else ["colmap", "nerfstudio", "ply", "splat"]

    if "colmap" in export_formats:
        from src.exporters.colmap_export import export_colmap
        export_colmap(poses, intrinsics, image_paths, points, colors, args.output, image_sizes)

    if "nerfstudio" in export_formats:
        from src.exporters.nerfstudio_export import export_nerfstudio
        export_nerfstudio(poses, intrinsics, image_paths, args.output, image_sizes)

    if "ply" in export_formats:
        from src.exporters.ply_export import export_ply
        export_ply(points, colors, os.path.join(args.output, "scene.ply"))

    if "splat" in export_formats:
        from src.exporters.splat_export import pointcloud_to_splat
        pointcloud_to_splat(points, colors, os.path.join(args.output, "scene.splat"))

    # Train Gaussians if requested
    if args.train_gaussians:
        print("\nTraining Gaussian Splatting...")
        try:
            from src.gaussian_render import train_gaussians_simple
            train_gaussians_simple(
                images=images,
                poses_c2w=poses,
                K=K, W=W, H=H,
                points_init=points,
                colors_init=colors,
                n_iters=args.train_iters,
                output_dir=args.output,
            )
        except Exception as e:
            print(f"  Gaussian training failed: {e}")

    # Save poses as numpy
    np.save(os.path.join(args.output, "poses_c2w.npy"), poses)
    np.save(os.path.join(args.output, "intrinsics.npy"), intrinsics)

    # Save summary
    t_total = time.time() - t_start
    summary = {
        "n_frames": N,
        "method": method,
        "total_time_s": round(t_total, 1),
        "timings": {k: round(v, 2) for k, v in results.get("timings", {}).items()},
        "exports": export_formats,
        "n_points": len(points),
    }
    if "fg_ate" in results:
        summary["fg_ate"] = results["fg_ate"]
    if "naive_ate" in results:
        summary["naive_ate"] = results["naive_ate"]

    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Done in {t_total:.1f}s")
    print(f"{'='*50}")
    print(f"Outputs ({args.output}):")
    if "colmap" in export_formats:
        print(f"  COLMAP:      {args.output}/sparse/0/")
    if "nerfstudio" in export_formats:
        print(f"  nerfstudio:  {args.output}/transforms.json")
    if "ply" in export_formats:
        print(f"  Point cloud: {args.output}/scene.ply")
    if "splat" in export_formats:
        print(f"  Splat:       {args.output}/scene.splat")
    print(f"  Poses:       {args.output}/poses_c2w.npy")
    print(f"  Summary:     {args.output}/summary.json")


def _collect_points(results, images, poses):
    """Collect a globally-aligned, colored point cloud from pipeline chunks.

    VGGT emits per-chunk world points in each chunk's own coordinate frame
    (independent origin and scale). We align every chunk to the refined global
    poses with a Sim(3) fit on camera centers before merging, and sample colors
    at the point-map resolution so colors correspond 1:1 with the points.
    """
    from src.chunked_pipeline import _procrustes_sim3

    if "chunks" not in results:
        return np.zeros((0, 3)), np.zeros((0, 3))

    all_pts = []
    all_colors = []
    for chunk in results["chunks"]:
        pts_grid = chunk.get("points")
        if pts_grid is None or np.ndim(pts_grid) != 4:
            continue
        n, Hv, Wv = pts_grid.shape[:3]
        conf = chunk.get("point_conf", np.ones((n, Hv, Wv)))

        # Sim(3) mapping chunk-local world coords -> refined global coords,
        # estimated from camera-center correspondences.
        gi0 = chunk["start"]
        local_c = np.asarray(chunk["poses_c2w"])[:, :3, 3]
        global_c = poses[gi0:gi0 + n, :3, 3]
        m = min(len(local_c), len(global_c))
        if m >= 3:
            T_align, scale = _procrustes_sim3(local_c[:m], global_c[:m])
            R_align, t_align = T_align[:3, :3], T_align[:3, 3]
        else:
            R_align, t_align, scale = np.eye(3), np.zeros(3), 1.0

        for k in range(n):
            gi = gi0 + k
            if gi >= len(images):
                continue
            pts_k = pts_grid[k].reshape(-1, 3)
            conf_k = np.asarray(conf[k]).reshape(-1)
            valid = (conf_k > 0.5) & np.isfinite(pts_k).all(axis=1)
            if not valid.any():
                continue
            pts_v = pts_k[valid]
            pts_g = (scale * (R_align @ pts_v.T)).T + t_align

            frame = images[gi]
            if frame.shape[:2] != (Hv, Wv):
                frame = cv2.resize(frame, (Wv, Hv))
            colors_v = frame.reshape(-1, 3)[valid]

            all_pts.append(pts_g)
            all_colors.append(colors_v)

    if all_pts:
        return np.concatenate(all_pts), np.concatenate(all_colors)

    # Fallback: no points available
    return np.zeros((0, 3)), np.zeros((0, 3))


if __name__ == "__main__":
    main()

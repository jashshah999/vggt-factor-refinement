"""Run VGGT + Factor Graph on a video or image directory.

Usage:
    python run.py --video my_video.mp4 --output output/my_video
    python run.py --images path/to/frames/ --output output/my_scene
"""

import argparse
import os
import sys
import json

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))


def load_video_frames(video_path, max_frames=100, target_size=None):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sample frames uniformly
    n = min(total, max_frames)
    indices = np.linspace(0, total - 1, n, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if target_size:
            frame = cv2.resize(frame, target_size)
        frames.append(frame)
    cap.release()

    H, W = frames[0].shape[:2]
    # Estimate intrinsics (assume standard FOV ~60 degrees)
    fx = W / (2 * np.tan(np.radians(30)))
    K = np.array([[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]], dtype=np.float64)

    return np.array(frames), K, W, H


def load_image_dir(image_dir, max_frames=100, target_size=None):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    from pathlib import Path
    paths = sorted([p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts])
    if not paths:
        raise ValueError(f"No images in {image_dir}")

    # Subsample
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

    return np.array(frames), K, W, H


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Path to input video")
    parser.add_argument("--images", help="Path to image directory")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--output", default="output/run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-factor-graph", action="store_true")
    parser.add_argument("--no-gaussians", action="store_true")
    parser.add_argument("--train-iters", type=int, default=500)
    args = parser.parse_args()

    if args.video:
        print(f"Loading video: {args.video}")
        images, K, W, H = load_video_frames(args.video, args.max_frames)
    elif args.images:
        print(f"Loading images: {args.images}")
        images, K, W, H = load_image_dir(args.images, args.max_frames)
    else:
        parser.error("Provide --video or --images")

    print(f"  {len(images)} frames, {W}x{H}")

    from src.pipeline import run_pipeline
    results = run_pipeline(
        images=images, K=K, W=W, H=H,
        device=args.device,
        use_factor_graph=not args.no_factor_graph,
        train_gaussians=not args.no_gaussians,
        n_train_iters=args.train_iters,
    )

    os.makedirs(args.output, exist_ok=True)

    # Save poses
    np.save(os.path.join(args.output, "vggt_poses.npy"), results["vggt_poses"])
    if "refined_poses" in results:
        np.save(os.path.join(args.output, "refined_poses.npy"), results["refined_poses"])
    if "covariances" in results:
        np.save(os.path.join(args.output, "covariances.npy"), results["covariances"])

    # Save renders
    if "vggt_render" in results:
        cv2.imwrite(
            os.path.join(args.output, "render_vggt.png"),
            cv2.cvtColor((results["vggt_render"] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
    if "fg_render" in results:
        cv2.imwrite(
            os.path.join(args.output, "render_refined.png"),
            cv2.cvtColor((results["fg_render"] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )

    # Save summary
    summary = {"timings": results["timings"]}
    if "vggt_render_psnr" in results:
        summary["vggt_render_psnr"] = results["vggt_render_psnr"]
    if "fg_render_psnr" in results:
        summary["fg_render_psnr"] = results["fg_render_psnr"]

    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs saved to {args.output}/")


if __name__ == "__main__":
    main()

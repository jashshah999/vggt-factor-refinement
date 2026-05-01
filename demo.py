"""End-to-end demo: video in, trajectory + renders out.

Processes a video (or the bundled TUM room walkthrough), shows the
difference between naive chunk stitching and factor graph refinement.

Usage:
    python demo.py                           # use bundled TUM room demo
    python demo.py --video my_video.mp4      # use your own video
"""

import argparse
import json
import os
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.vggt_wrapper import load_vggt, run_vggt_on_images
from src.chunked_pipeline import run_chunked_pipeline, _naive_stitch
from src.gaussian_init import init_gaussians_from_vggt
from src.gaussian_render import train_gaussians, render_gaussians
from src.metrics import psnr


def load_video(path, max_frames=100, target_h=480):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Subsample to max_frames
    n = min(total, max_frames)
    indices = np.linspace(0, total - 1, n, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Resize if needed
        if frame.shape[0] != target_h:
            scale = target_h / frame.shape[0]
            new_w = int(frame.shape[1] * scale)
            frame = cv2.resize(frame, (new_w, target_h))
        frames.append(frame)
    cap.release()

    images = np.array(frames)
    H, W = images.shape[1], images.shape[2]
    fx = W / (2 * np.tan(np.radians(30)))
    K = np.array([[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]], dtype=np.float64)
    return images, K, W, H, fps


def make_trajectory_gif(naive_poses, fg_poses, save_path, n_frames_anim=60):
    """Create an animated trajectory comparison GIF."""
    N = len(naive_poses)

    naive_t = np.array([p[:3, 3] for p in naive_poses])
    fg_t = np.array([p[:3, 3] for p in fg_poses])

    frames = []
    for k in range(1, min(N, n_frames_anim) + 1):
        fig, ax = plt.subplots(figsize=(8, 6))
        i = int(k * N / n_frames_anim)
        i = min(i, N)

        ax.plot(naive_t[:i, 0], naive_t[:i, 2], "r-", label="Naive Stitch", lw=2, alpha=0.7)
        ax.plot(fg_t[:i, 0], fg_t[:i, 2], "b-", label="Factor Graph", lw=2)

        # Current position markers
        if i > 0:
            ax.plot(naive_t[i-1, 0], naive_t[i-1, 2], "ro", ms=8)
            ax.plot(fg_t[i-1, 0], fg_t[i-1, 2], "bs", ms=8)

        # Fixed axis limits
        all_pts = np.concatenate([naive_t, fg_t])
        margin = 0.3
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"Camera Trajectory (frame {i}/{N})")
        ax.legend(loc="upper right")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        frames.append(img)
        plt.close()

    import imageio
    imageio.mimsave(save_path, frames, duration=0.15, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--train-iters", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="output/demo")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load video
    if args.video:
        print(f"Loading video: {args.video}")
        images, K, W, H, fps = load_video(args.video, args.max_frames)
    else:
        # Use TUM room data
        print("Loading TUM fr1/room demo...")
        from src.data_loaders import load_tum_sequence
        data = load_tum_sequence("fr1/room", stride=2, max_frames=args.max_frames)
        images, K, W, H = data["images"], data["K"], data["W"], data["H"]
        fps = 15

    N = len(images)
    print(f"  {N} frames, {W}x{H}")

    # Run VGGT
    print("\nRunning VGGT...")
    t0 = time.time()
    model = load_vggt(args.device)
    vggt_out = run_vggt_on_images(model, images, args.device, max_batch=15)
    del model
    torch.cuda.empty_cache()
    vggt_time = time.time() - t0
    print(f"  Done in {vggt_time:.1f}s")

    # Chunked pipeline
    print("\nRunning chunked pipeline...")
    gt_poses = data.get("gt_poses") if not args.video else None
    results = run_chunked_pipeline(
        images=images, K=K, W=W, H=H, gt_poses=gt_poses,
        device=args.device, chunk_size=args.chunk_size, overlap=args.overlap,
    )

    naive_poses = results["naive_poses"]
    fg_poses = results["fg_poses"]

    # Trajectory plot (static)
    print("\nGenerating trajectory plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    naive_t = np.array([p[:3, 3] for p in naive_poses])
    fg_t = np.array([p[:3, 3] for p in fg_poses])

    ax.plot(naive_t[:, 0], naive_t[:, 2], "r--", label="Naive Stitch", lw=2, alpha=0.7)
    ax.plot(fg_t[:, 0], fg_t[:, 2], "b-", label="Factor Graph", lw=2.5)

    if gt_poses is not None:
        from src.metrics import align_trajectories
        gt_t = gt_poses[:, :3, 3]
        ax.plot(gt_t[:, 0], gt_t[:, 2], "g-", label="Ground Truth", lw=2)

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Z (m)", fontsize=12)
    ax.set_title("Camera Trajectory: Naive Stitch vs Factor Graph", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "trajectory.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Trajectory GIF
    print("Generating trajectory GIF...")
    try:
        make_trajectory_gif(
            naive_poses, fg_poses,
            os.path.join(args.output, "trajectory.gif"),
        )
        print(f"  Saved trajectory.gif")
    except ImportError:
        print("  imageio not installed, skipping GIF")

    # Gaussian splatting renders
    print("\nTraining Gaussians...")
    gs_params = init_gaussians_from_vggt(
        vggt_out["points"], images, vggt_out["point_conf"],
        fg_poses, conf_threshold=0.3, stride=4, device=args.device,
    )
    print(f"  {len(gs_params['means'])} Gaussians")

    pts_H, pts_W = vggt_out["depth"].shape[1], vggt_out["depth"].shape[2]
    render_images = np.stack([cv2.resize(img, (pts_W, pts_H)) for img in images])
    K_render = K.copy()
    K_render[0, :] *= pts_W / W
    K_render[1, :] *= pts_H / H

    # Train with naive poses
    print("  Training with naive poses...")
    gs_naive = {k: torch.nn.Parameter(v.data.clone()) for k, v in gs_params.items()}
    train_gaussians(gs_naive, naive_poses, render_images, K_render, pts_W, pts_H,
                    n_iters=args.train_iters, lr=0.005, device=args.device)

    # Train with factor graph poses
    print("  Training with factor graph poses...")
    gs_fg = {k: torch.nn.Parameter(v.data.clone()) for k, v in gs_params.items()}
    train_gaussians(gs_fg, fg_poses, render_images, K_render, pts_W, pts_H,
                    n_iters=args.train_iters, lr=0.005, device=args.device)

    # Render comparison at multiple viewpoints
    print("\nRendering comparisons...")
    test_indices = np.linspace(0, N - 1, 5, dtype=int)

    for idx in test_indices:
        gt_img = render_images[idx]
        naive_render = render_gaussians(gs_naive, naive_poses[idx], K_render, pts_W, pts_H, args.device)
        fg_render = render_gaussians(gs_fg, fg_poses[idx], K_render, pts_W, pts_H, args.device)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(gt_img)
        axes[0].set_title("Input Frame")
        axes[0].axis("off")
        axes[1].imshow(naive_render)
        axes[1].set_title(f"Naive Poses ({psnr(gt_img, naive_render):.1f} dB)")
        axes[1].axis("off")
        axes[2].imshow(fg_render)
        axes[2].set_title(f"Factor Graph ({psnr(gt_img, fg_render):.1f} dB)")
        axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, f"render_{idx:03d}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # Summary
    print(f"\n{'='*50}")
    print("  DEMO COMPLETE")
    print(f"{'='*50}")
    if "naive_ate" in results:
        print(f"  Naive ATE:       {results['naive_ate']['ate_mean']:.4f} m")
        print(f"  Factor Graph ATE: {results['fg_ate']['ate_mean']:.4f} m")
        imp = (results["naive_ate"]["ate_mean"] - results["fg_ate"]["ate_mean"]) / results["naive_ate"]["ate_mean"] * 100
        print(f"  Improvement:     {imp:.1f}%")
    print(f"  Outputs: {args.output}/")


if __name__ == "__main__":
    main()

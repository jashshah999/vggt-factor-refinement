"""Visualize camera poses as frustums in 3D point cloud.

Renders a birds-eye orbiting view of the reconstruction with camera
frustums colored by method (red=naive, blue=factor graph, green=GT).

Usage:
    python visualize_3d.py --seq fr1/desk --output output/viz
"""

import argparse
import os
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.vggt_wrapper import load_vggt, run_vggt_on_images
from src.chunked_pipeline import run_chunked_pipeline
from src.data_loaders import load_tum_sequence
from src.metrics import align_trajectories


def draw_frustum(ax, pose, color, size=0.08, alpha=0.3):
    """Draw a camera frustum at the given pose."""
    # Camera frustum corners in camera frame
    hw = size * 0.6
    hh = size * 0.4
    d = size
    corners_cam = np.array([
        [0, 0, 0],
        [-hw, -hh, d],
        [hw, -hh, d],
        [hw, hh, d],
        [-hw, hh, d],
    ])

    # Transform to world frame
    R = pose[:3, :3]
    t = pose[:3, 3]
    corners_world = (R @ corners_cam.T).T + t

    # Draw edges from apex to corners
    apex = corners_world[0]
    for i in range(1, 5):
        ax.plot3D(*zip(apex, corners_world[i]), color=color, alpha=alpha, lw=0.8)

    # Draw rectangle at the front
    rect = corners_world[1:5]
    rect_closed = np.vstack([rect, rect[0:1]])
    ax.plot3D(rect_closed[:, 0], rect_closed[:, 1], rect_closed[:, 2],
              color=color, alpha=alpha, lw=0.8)


def render_scene(gt_poses, naive_poses, fg_poses, points, point_conf,
                 elev=30, azim=45, title="", save_path="scene.png"):
    """Render 3D scene with point cloud and camera frustums."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Subsample and filter point cloud
    pts = points.reshape(-1, 3)
    conf = point_conf.ravel()
    mask = (conf > 0.3) & np.isfinite(pts).all(axis=1)
    pts = pts[mask]

    if len(pts) > 50000:
        idx = np.random.choice(len(pts), 50000, replace=False)
        pts = pts[idx]

    # Filter outliers
    center = np.median(pts, axis=0)
    dists = np.linalg.norm(pts - center, axis=1)
    pts = pts[dists < np.percentile(dists, 95)]

    # Plot point cloud
    ax.scatter(pts[::5, 0], pts[::5, 1], pts[::5, 2],
               s=0.1, c="gray", alpha=0.15)

    # Align trajectories for fair comparison
    if gt_poses is not None:
        naive_aligned = align_trajectories(gt_poses, naive_poses)
        fg_aligned = align_trajectories(gt_poses, fg_poses)
    else:
        naive_aligned = naive_poses
        fg_aligned = fg_poses

    # Draw trajectories as lines
    if gt_poses is not None:
        gt_t = gt_poses[:, :3, 3]
        ax.plot3D(gt_t[:, 0], gt_t[:, 1], gt_t[:, 2],
                  "g-", lw=2, label="Ground Truth", alpha=0.8)

    naive_t = naive_aligned[:, :3, 3]
    fg_t = fg_aligned[:, :3, 3]
    ax.plot3D(naive_t[:, 0], naive_t[:, 1], naive_t[:, 2],
              "r--", lw=1.5, label="Naive Stitch", alpha=0.6)
    ax.plot3D(fg_t[:, 0], fg_t[:, 1], fg_t[:, 2],
              "b-", lw=2, label="Factor Graph", alpha=0.8)

    # Draw camera frustums (every 5th frame to avoid clutter)
    N = len(fg_aligned)
    for i in range(0, N, max(1, N // 15)):
        if gt_poses is not None:
            draw_frustum(ax, gt_poses[i], "green", size=0.06, alpha=0.4)
        draw_frustum(ax, naive_aligned[i], "red", size=0.06, alpha=0.3)
        draw_frustum(ax, fg_aligned[i], "blue", size=0.06, alpha=0.5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.view_init(elev=elev, azim=azim)

    # Equal aspect ratio
    all_pts = np.concatenate([naive_t, fg_t])
    if gt_poses is not None:
        all_pts = np.concatenate([all_pts, gt_t])
    mid = all_pts.mean(axis=0)
    span = max(all_pts.max(axis=0) - all_pts.min(axis=0)) / 2 * 1.2
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def render_orbit_gif(gt_poses, naive_poses, fg_poses, points, point_conf,
                     save_path="orbit.gif", n_frames=36):
    """Render an orbiting GIF around the scene."""
    import imageio

    frames = []
    for i in range(n_frames):
        azim = i * 360 / n_frames
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Subsample points
        pts = points.reshape(-1, 3)
        conf = point_conf.ravel()
        mask = (conf > 0.3) & np.isfinite(pts).all(axis=1)
        pts = pts[mask]
        if len(pts) > 30000:
            idx = np.random.choice(len(pts), 30000, replace=False)
            pts = pts[idx]
        center = np.median(pts, axis=0)
        dists = np.linalg.norm(pts - center, axis=1)
        pts = pts[dists < np.percentile(dists, 95)]

        ax.scatter(pts[::5, 0], pts[::5, 1], pts[::5, 2],
                   s=0.1, c="gray", alpha=0.1)

        if gt_poses is not None:
            naive_aligned = align_trajectories(gt_poses, naive_poses)
            fg_aligned = align_trajectories(gt_poses, fg_poses)
            gt_t = gt_poses[:, :3, 3]
            ax.plot3D(gt_t[:, 0], gt_t[:, 1], gt_t[:, 2], "g-", lw=2, alpha=0.8)
        else:
            naive_aligned = naive_poses
            fg_aligned = fg_poses

        naive_t = naive_aligned[:, :3, 3]
        fg_t = fg_aligned[:, :3, 3]
        ax.plot3D(naive_t[:, 0], naive_t[:, 1], naive_t[:, 2], "r--", lw=1.5, alpha=0.6)
        ax.plot3D(fg_t[:, 0], fg_t[:, 1], fg_t[:, 2], "b-", lw=2, alpha=0.8)

        N = len(fg_aligned)
        for j in range(0, N, max(1, N // 12)):
            draw_frustum(ax, naive_aligned[j], "red", size=0.05, alpha=0.3)
            draw_frustum(ax, fg_aligned[j], "blue", size=0.05, alpha=0.5)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        all_pts = np.concatenate([naive_t, fg_t])
        if gt_poses is not None:
            all_pts = np.concatenate([all_pts, gt_t])
        mid = all_pts.mean(axis=0)
        span = max(all_pts.max(axis=0) - all_pts.min(axis=0)) / 2 * 1.2
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)

        ax.view_init(elev=25, azim=azim)
        ax.set_title("Red: Naive Stitch | Blue: Factor Graph | Green: GT", fontsize=11)

        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(img)
        plt.close()

    imageio.mimsave(save_path, frames, duration=0.2, loop=0)
    print(f"  Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="fr1/desk")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="output/viz")
    parser.add_argument("--gif", action="store_true", help="Generate orbit GIF")
    args = parser.parse_args()

    out_dir = os.path.join(args.output, args.seq.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {args.seq}...")
    data = load_tum_sequence(args.seq, stride=args.stride, max_frames=args.max_frames)
    N = len(data["images"])
    print(f"  {N} frames")

    print("Running VGGT...")
    model = load_vggt(args.device)
    vggt_out = run_vggt_on_images(model, data["images"], args.device, max_batch=15)
    del model
    torch.cuda.empty_cache()

    print("Running chunked pipeline...")
    results = run_chunked_pipeline(
        images=data["images"], K=data["K"], W=data["W"], H=data["H"],
        gt_poses=data["gt_poses"], device=args.device,
        chunk_size=args.chunk_size, overlap=args.overlap,
    )

    print("\nRendering 3D visualization...")

    # Static views from multiple angles
    for elev, azim, name in [(30, 45, "view1"), (60, 0, "topdown"), (15, 90, "side")]:
        render_scene(
            data["gt_poses"], results["naive_poses"], results["fg_poses"],
            vggt_out["points"], vggt_out["point_conf"],
            elev=elev, azim=azim,
            title=f"TUM {args.seq} (ATE: naive={results['naive_ate']['ate_mean']:.3f}m, FG={results['fg_ate']['ate_mean']:.3f}m)",
            save_path=os.path.join(out_dir, f"{name}.png"),
        )

    if args.gif:
        print("\nRendering orbit GIF...")
        render_orbit_gif(
            data["gt_poses"], results["naive_poses"], results["fg_poses"],
            vggt_out["points"], vggt_out["point_conf"],
            save_path=os.path.join(out_dir, "orbit.gif"),
        )

    print(f"\nOutputs saved to {out_dir}/")


if __name__ == "__main__":
    main()

"""Benchmark Gaussian splatting quality: naive poses vs factor graph poses.

Shows that better poses from the factor graph lead to better 3D renders.
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
from src.chunked_pipeline import run_chunked_pipeline
from src.data_loaders import load_tum_sequence
from src.gaussian_init import init_gaussians_from_vggt
from src.gaussian_render import train_gaussians, render_gaussians
from src.metrics import psnr, align_trajectories
from src.vggt_wrapper import load_vggt, run_vggt_on_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="fr1/desk")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--train-iters", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="output/gs_compare")
    args = parser.parse_args()

    out_dir = os.path.join(args.output, args.seq.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {args.seq}...")
    data = load_tum_sequence(args.seq, stride=args.stride, max_frames=args.max_frames)
    N = len(data["images"])
    print(f"  {N} frames")

    # Run VGGT to get point maps (needed for Gaussian init)
    print("Running VGGT for point maps...")
    model = load_vggt(args.device)
    vggt_out = run_vggt_on_images(model, data["images"], args.device, max_batch=15)
    del model
    torch.cuda.empty_cache()

    # Run chunked pipeline to get naive and factor graph poses
    print("Running chunked pipeline for poses...")
    results = run_chunked_pipeline(
        images=data["images"], K=data["K"], W=data["W"], H=data["H"],
        gt_poses=data["gt_poses"], device=args.device,
        chunk_size=args.chunk_size, overlap=args.overlap,
    )

    naive_poses = results["naive_poses"]
    fg_poses = results["fg_poses"]

    # Initialize Gaussians from VGGT point maps
    print("Initializing Gaussians...")
    gs_params = init_gaussians_from_vggt(
        vggt_out["points"], data["images"], vggt_out["point_conf"],
        naive_poses, conf_threshold=0.3, stride=4, device=args.device,
    )
    n_gs = len(gs_params["means"])
    print(f"  {n_gs} Gaussians")

    # Prepare render-sized images and intrinsics
    pts_H, pts_W = vggt_out["depth"].shape[1], vggt_out["depth"].shape[2]
    render_images = np.stack([cv2.resize(img, (pts_W, pts_H)) for img in data["images"]])
    K_render = data["K"].copy()
    K_render[0, :] *= pts_W / data["W"]
    K_render[1, :] *= pts_H / data["H"]

    # Train with naive poses
    print(f"Training Gaussians with naive poses ({args.train_iters} iters)...")
    gs_naive = {k: torch.nn.Parameter(v.data.clone()) for k, v in gs_params.items()}
    train_gaussians(gs_naive, naive_poses, render_images, K_render,
                    pts_W, pts_H, n_iters=args.train_iters, device=args.device)

    # Train with factor graph poses
    print(f"Training Gaussians with factor graph poses ({args.train_iters} iters)...")
    gs_fg = {k: torch.nn.Parameter(v.data.clone()) for k, v in gs_params.items()}
    train_gaussians(gs_fg, fg_poses, render_images, K_render,
                    pts_W, pts_H, n_iters=args.train_iters, device=args.device)

    # Render and compute PSNR for multiple frames
    print("Rendering comparisons...")
    psnrs_naive = []
    psnrs_fg = []
    test_indices = np.linspace(0, N - 1, min(N, 8), dtype=int)

    for idx in test_indices:
        render_n = render_gaussians(gs_naive, naive_poses[idx], K_render, pts_W, pts_H, args.device)
        render_f = render_gaussians(gs_fg, fg_poses[idx], K_render, pts_W, pts_H, args.device)
        gt = render_images[idx]

        pn = psnr(gt, render_n)
        pf = psnr(gt, render_f)
        psnrs_naive.append(pn)
        psnrs_fg.append(pf)

    mean_psnr_naive = np.mean(psnrs_naive)
    mean_psnr_fg = np.mean(psnrs_fg)

    print(f"\n  Mean PSNR (naive poses):  {mean_psnr_naive:.2f} dB")
    print(f"  Mean PSNR (FG poses):    {mean_psnr_fg:.2f} dB")
    print(f"  Improvement:             {mean_psnr_fg - mean_psnr_naive:+.2f} dB")

    # Save side-by-side renders for the middle frame
    mid = N // 2
    gt_img = render_images[mid]
    naive_render = render_gaussians(gs_naive, naive_poses[mid], K_render, pts_W, pts_H, args.device)
    fg_render = render_gaussians(gs_fg, fg_poses[mid], K_render, pts_W, pts_H, args.device)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(gt_img)
    axes[0].set_title(f"Ground Truth (frame {mid})")
    axes[0].axis("off")
    axes[1].imshow(naive_render)
    axes[1].set_title(f"Naive Poses ({mean_psnr_naive:.1f} dB)")
    axes[1].axis("off")
    axes[2].imshow(fg_render)
    axes[2].set_title(f"Factor Graph ({mean_psnr_fg:.1f} dB)")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "render_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # PSNR per frame chart
    fig, ax = plt.subplots(figsize=(8, 4))
    x = test_indices
    ax.bar(x - 0.3, psnrs_naive, 0.6, label="Naive Poses", color="salmon", alpha=0.8)
    ax.bar(x + 0.3, psnrs_fg, 0.6, label="Factor Graph", color="steelblue", alpha=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"Render Quality: {args.seq}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "psnr_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    summary = {
        "sequence": args.seq,
        "n_frames": N,
        "n_gaussians": n_gs,
        "mean_psnr_naive": float(mean_psnr_naive),
        "mean_psnr_fg": float(mean_psnr_fg),
        "psnr_improvement": float(mean_psnr_fg - mean_psnr_naive),
        "naive_ate": results["naive_ate"],
        "fg_ate": results["fg_ate"],
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs saved to {out_dir}/")


if __name__ == "__main__":
    main()

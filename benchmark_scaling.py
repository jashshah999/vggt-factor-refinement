"""Scaling experiment: how does ATE change as sequence length grows?

Shows that:
1. VGGT+BA works great but OOMs past ~30 frames on 24GB
2. Naive stitching degrades linearly with sequence length
3. Factor graph stitching stays reasonable at any length

Usage:
    python benchmark_scaling.py --seq fr1/room --device cuda
    python benchmark_scaling.py --dataset replica --seq office0 --device cuda
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.vggt_wrapper import load_vggt, run_vggt_on_images, vggt_conf_to_covariance
from src.chunked_pipeline import run_chunked_pipeline
from src.metrics import absolute_trajectory_error


def run_vggt_single_shot(images, K, gt_poses, device, model):
    """Run VGGT on all frames at once (will OOM for large N)."""
    try:
        vggt_out = run_vggt_on_images(model, images, device, max_batch=len(images))
        metrics = absolute_trajectory_error(gt_poses, vggt_out["poses_c2w"])
        return metrics["ate_mean"]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="room0")
    parser.add_argument("--dataset", default="replica", choices=["tum", "replica"])
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="output/scaling")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load a long sequence
    if args.dataset == "replica":
        from src.replica_loader import load_replica_sequence
        data = load_replica_sequence(
            args.seq, data_root=args.data_root, stride=1, max_frames=500,
        )
    else:
        from src.data_loaders import load_tum_sequence
        data = load_tum_sequence(
            args.seq, data_root=args.data_root, stride=1, max_frames=500,
        )

    total_frames = len(data["images"])
    print(f"Loaded {total_frames} frames from {args.dataset}/{args.seq}")

    # Test at different sequence lengths
    frame_counts = [10, 15, 20, 30, 50, 80, 120, 200, 300]
    frame_counts = [n for n in frame_counts if n <= total_frames]

    results = {
        "frame_counts": [],
        "vggt_single_ate": [],
        "naive_ate": [],
        "fg_ate": [],
    }

    model = load_vggt(args.device)

    for n_frames in frame_counts:
        print(f"\n{'='*50}")
        print(f"  N = {n_frames} frames")
        print(f"{'='*50}")

        images = data["images"][:n_frames]
        gt_poses = data["gt_poses"][:n_frames]

        # 1. VGGT single-shot (will OOM at some point)
        print("  VGGT single-shot...", end=" ")
        single_ate = run_vggt_single_shot(images, data["K"], gt_poses, args.device, model)
        if single_ate is not None:
            print(f"ATE = {single_ate:.4f}m")
        else:
            print("OOM")
        results["vggt_single_ate"].append(single_ate)

        # 2. Chunked pipeline (naive + factor graph)
        # Need to reload model if it was cleared
        try:
            _ = model.device
        except Exception:
            model = load_vggt(args.device)

        print("  Chunked pipeline...")
        chunk_results = run_chunked_pipeline(
            images=images, K=data["K"], W=data["W"], H=data["H"],
            gt_poses=gt_poses, device=args.device,
            chunk_size=args.chunk_size, overlap=args.overlap,
        )
        results["naive_ate"].append(chunk_results["naive_ate"]["ate_mean"])
        results["fg_ate"].append(chunk_results["fg_ate"]["ate_mean"])
        results["frame_counts"].append(n_frames)

        print(f"  Naive: {chunk_results['naive_ate']['ate_mean']:.4f}m")
        print(f"  FG:    {chunk_results['fg_ate']['ate_mean']:.4f}m")

    del model
    torch.cuda.empty_cache()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = results["frame_counts"]

    ax.plot(x, results["naive_ate"], "r-o", label="Naive Stitch", lw=2, ms=6)
    ax.plot(x, results["fg_ate"], "b-s", label="Factor Graph (ours)", lw=2, ms=6)

    # Plot VGGT single-shot where it didn't OOM
    vggt_x = [x[i] for i in range(len(x)) if results["vggt_single_ate"][i] is not None]
    vggt_y = [y for y in results["vggt_single_ate"] if y is not None]
    if vggt_x:
        ax.plot(vggt_x, vggt_y, "g-^", label="VGGT Single-Shot", lw=2, ms=8)
        # Mark OOM point
        oom_idx = next((i for i, v in enumerate(results["vggt_single_ate"]) if v is None), None)
        if oom_idx is not None:
            ax.axvline(x=x[oom_idx], color="green", linestyle=":", alpha=0.5)
            ax.annotate("VGGT OOM", xy=(x[oom_idx], ax.get_ylim()[1] * 0.9),
                        fontsize=10, color="green", ha="center")

    ax.set_xlabel("Number of Frames", fontsize=12)
    ax.set_ylabel("ATE (m)", fontsize=12)
    ax.set_title(f"Scaling: {args.dataset}/{args.seq} (chunk={args.chunk_size})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, max(x) * 1.05)
    ax.set_ylim(0, None)

    plt.tight_layout()
    save_path = os.path.join(args.output, f"scaling_{args.dataset}_{args.seq.replace('/', '_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved scaling plot to {save_path}")

    # Save raw results
    with open(os.path.join(args.output, f"scaling_{args.dataset}_{args.seq.replace('/', '_')}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'Frames':>8} {'VGGT':>10} {'Naive':>10} {'FG':>10} {'FG Improve':>12}")
    print("-" * 55)
    for i, n in enumerate(results["frame_counts"]):
        vggt = results["vggt_single_ate"][i]
        naive = results["naive_ate"][i]
        fg = results["fg_ate"][i]
        imp = (naive - fg) / naive * 100
        vggt_str = f"{vggt:.4f}" if vggt is not None else "OOM"
        print(f"{n:>8} {vggt_str:>10} {naive:>10.4f} {fg:>10.4f} {imp:>11.1f}%")


if __name__ == "__main__":
    main()

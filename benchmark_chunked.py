"""Benchmark: chunked VGGT with factor graph stitching vs naive stitching.

This demonstrates the core value: for long videos that don't fit in VGGT's
memory, the factor graph provides global consistency that naive overlap
stitching cannot achieve.

Usage:
    python benchmark_chunked.py --seq fr1/room --chunk-size 10 --overlap 3
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.chunked_pipeline import run_chunked_pipeline
from src.data_loaders import load_tum_sequence, TUM_SEQUENCES
from src.metrics import align_trajectories


def plot_comparison(gt, naive, fg, single=None, title="", save_path="trajectory.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    gt_t = gt[:, :3, 3]
    naive_a = align_trajectories(gt, naive)[:, :3, 3]
    fg_a = align_trajectories(gt, fg)[:, :3, 3]

    ax = axes[0]
    ax.plot(gt_t[:, 0], gt_t[:, 2], "g-", label="Ground Truth", lw=2)
    ax.plot(naive_a[:, 0], naive_a[:, 2], "r--", label="Naive Stitch", lw=1.5, alpha=0.7)
    ax.plot(fg_a[:, 0], fg_a[:, 2], "b-", label="Factor Graph", lw=1.5)
    if single is not None:
        single_a = align_trajectories(gt, single)[:, :3, 3]
        ax.plot(single_a[:, 0], single_a[:, 2], "m:", label="VGGT Single-Shot", lw=1.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    ax = axes[1]
    naive_errs = np.linalg.norm(gt[:, :3, 3] - align_trajectories(gt, naive)[:, :3, 3], axis=1)
    fg_errs = np.linalg.norm(gt[:, :3, 3] - align_trajectories(gt, fg)[:, :3, 3], axis=1)
    x = np.arange(len(gt))
    ax.bar(x - 0.2, naive_errs, 0.4, label="Naive Stitch", color="salmon", alpha=0.8)
    ax.bar(x + 0.2, fg_errs, 0.4, label="Factor Graph", color="steelblue", alpha=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Error (m)")
    ax.set_title("Per-frame ATE")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="fr1/room")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="output/chunked")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Chunked Benchmark: TUM-RGBD {args.seq}")
    print(f"  Chunk size: {args.chunk_size}, Overlap: {args.overlap}")
    print(f"{'='*60}\n")

    data = load_tum_sequence(
        args.seq, data_root=args.data_root,
        stride=args.stride, max_frames=args.max_frames,
    )
    print(f"  {len(data['images'])} frames, {data['W']}x{data['H']}")

    results = run_chunked_pipeline(
        images=data["images"],
        K=data["K"],
        W=data["W"],
        H=data["H"],
        gt_poses=data["gt_poses"],
        device=args.device,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    out_dir = os.path.join(args.output, args.seq.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    plot_comparison(
        data["gt_poses"],
        results["naive_poses"],
        results["fg_poses"],
        results.get("single_poses"),
        title=f"TUM {args.seq} (chunk={args.chunk_size}, overlap={args.overlap})",
        save_path=os.path.join(out_dir, "trajectory.png"),
    )

    # Print summary
    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    print(f"  Naive stitch ATE:   {results['naive_ate']['ate_mean']:.4f}m")
    print(f"  Factor graph ATE:   {results['fg_ate']['ate_mean']:.4f}m")
    if "single_ate" in results:
        print(f"  Single-shot ATE:    {results['single_ate']['ate_mean']:.4f}m")
    imp = (results["naive_ate"]["ate_mean"] - results["fg_ate"]["ate_mean"]) / results["naive_ate"]["ate_mean"] * 100
    print(f"  Improvement:        {imp:.1f}%")

    summary = {
        "sequence": args.seq,
        "n_frames": len(data["images"]),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "naive_ate": results["naive_ate"],
        "fg_ate": results["fg_ate"],
        "timings": results["timings"],
    }
    if "single_ate" in results:
        summary["single_ate"] = results["single_ate"]

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs saved to {out_dir}/")


if __name__ == "__main__":
    main()

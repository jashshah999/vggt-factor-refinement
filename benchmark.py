"""Benchmark: compare VGGT vs VGGT + Factor Graph on TUM-RGBD.

Usage:
    python benchmark.py --seq fr1/desk --stride 5 --max-frames 60
    python benchmark.py --all
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

sys.path.insert(0, os.path.dirname(__file__))
from src.pipeline import run_pipeline
from src.data_loaders import load_tum_sequence


def plot_trajectories(gt, vggt, refined, title, save_path):
    """Plot XZ trajectory comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    gt_t = np.array([p[:3, 3] for p in gt])
    vggt_t = np.array([p[:3, 3] for p in vggt])
    ref_t = np.array([p[:3, 3] for p in refined])

    # Trajectory plot
    ax = axes[0]
    ax.plot(gt_t[:, 0], gt_t[:, 2], "g-", label="Ground Truth", lw=2)
    ax.plot(vggt_t[:, 0], vggt_t[:, 2], "r--", label="VGGT", lw=1.5, alpha=0.7)
    ax.plot(ref_t[:, 0], ref_t[:, 2], "b-", label="VGGT + Factor Graph", lw=1.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # Per-frame error plot
    ax = axes[1]
    from src.metrics import align_trajectories
    vggt_aligned = align_trajectories(gt, vggt)
    ref_aligned = align_trajectories(gt, refined)

    vggt_errs = np.linalg.norm(gt[:, :3, 3] - vggt_aligned[:, :3, 3], axis=1)
    ref_errs = np.linalg.norm(gt[:, :3, 3] - ref_aligned[:, :3, 3], axis=1)

    x = np.arange(len(gt))
    ax.bar(x - 0.2, vggt_errs, 0.4, label="VGGT", color="salmon", alpha=0.8)
    ax.bar(x + 0.2, ref_errs, 0.4, label="VGGT + FG", color="steelblue", alpha=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Error (m)")
    ax.set_title("Per-frame ATE")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved trajectory plot to {save_path}")


def plot_renders(gt_img, vggt_render, fg_render, save_path):
    """Side-by-side render comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(gt_img)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(vggt_render)
    axes[1].set_title("VGGT Poses")
    axes[1].axis("off")

    axes[2].imshow(fg_render)
    axes[2].set_title("VGGT + Factor Graph")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved render comparison to {save_path}")


def run_benchmark(seq_name, args):
    """Run benchmark on a single sequence."""
    print(f"\n{'='*60}")
    print(f"  Benchmarking: TUM-RGBD {seq_name}")
    print(f"{'='*60}\n")

    data = load_tum_sequence(seq_name, data_root=args.data_root,
                             stride=args.stride, max_frames=args.max_frames)

    print(f"  {len(data['images'])} frames, {data['W']}x{data['H']}")

    results = run_pipeline(
        images=data["images"],
        K=data["K"],
        W=data["W"],
        H=data["H"],
        gt_poses=data["gt_poses"],
        device=args.device,
        use_factor_graph=True,
        train_gaussians=args.train_gaussians,
        n_train_iters=args.train_iters,
    )

    # Save outputs
    out_dir = os.path.join(args.output, seq_name.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    # Trajectory plot
    plot_trajectories(
        data["gt_poses"], results["vggt_poses"], results["refined_poses"],
        f"TUM-RGBD {seq_name}", os.path.join(out_dir, "trajectory.png"),
    )

    # Render comparison
    if args.train_gaussians and "fg_render" in results:
        mid = len(data["images"]) // 2
        plot_renders(
            data["images"][mid], results["vggt_render"], results["fg_render"],
            os.path.join(out_dir, "render_comparison.png"),
        )

    # Summary
    summary = {
        "sequence": seq_name,
        "n_frames": len(data["images"]),
        "vggt_ate": results.get("vggt_ate"),
        "fg_ate": results.get("fg_ate"),
        "timings": results["timings"],
    }
    if args.train_gaussians:
        summary["vggt_render_psnr"] = results.get("vggt_render_psnr")
        summary["fg_render_psnr"] = results.get("fg_render_psnr")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark VGGT + Factor Graph refinement")
    parser.add_argument("--seq", default="fr1/desk", help="TUM sequence name")
    parser.add_argument("--all", action="store_true", help="Run all sequences")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--train-gaussians", action="store_true")
    parser.add_argument("--train-iters", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    sequences = list(TUM_SEQUENCES.keys()) if args.all else [args.seq]

    from src.data_loaders import TUM_SEQUENCES

    all_results = []
    for seq in sequences:
        summary = run_benchmark(seq, args)
        all_results.append(summary)

    # Print final comparison table
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Sequence':<15} {'VGGT ATE':>10} {'FG ATE':>10} {'Improve':>10}")
    print("-" * 50)
    for r in all_results:
        vggt = r["vggt_ate"]["ate_mean"] if r.get("vggt_ate") else float("nan")
        fg = r["fg_ate"]["ate_mean"] if r.get("fg_ate") else float("nan")
        imp = (vggt - fg) / vggt * 100 if vggt > 0 else 0
        print(f"{r['sequence']:<15} {vggt:>10.4f} {fg:>10.4f} {imp:>9.1f}%")

    # Save all results
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()

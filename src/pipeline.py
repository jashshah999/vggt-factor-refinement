"""Full pipeline: VGGT -> Factor Graph -> Gaussians."""

import numpy as np
import torch
import time


def run_pipeline(
    images: np.ndarray,
    K: np.ndarray,
    W: int,
    H: int,
    gt_poses: np.ndarray = None,
    device: str = "cuda",
    use_factor_graph: bool = True,
    train_gaussians: bool = True,
    n_train_iters: int = 500,
) -> dict:
    """Run the full reconstruction pipeline.

    Args:
        images: (N, H, W, 3) images in [0, 1]
        K: (3, 3) camera intrinsics
        W, H: image dimensions
        gt_poses: (N, 4, 4) optional ground truth for evaluation
        device: cuda or cpu
        use_factor_graph: whether to apply factor graph refinement
        train_gaussians: whether to train Gaussians
        n_train_iters: Gaussian training iterations

    Returns:
        dict with poses, metrics, and timing
    """
    from .vggt_wrapper import load_vggt, run_vggt_on_images, vggt_conf_to_covariance
    from .factor_graph import build_factor_graph, detect_loop_closures
    from .gaussian_init import init_gaussians_from_vggt
    from .gaussian_render import train_gaussians as train_gs, render_gaussians
    from .metrics import absolute_trajectory_error, relative_pose_error, psnr

    results = {"timings": {}}
    N = len(images)

    # === Step 1: VGGT inference ===
    print("Step 1: Running VGGT...")
    t0 = time.time()

    model = load_vggt(device)
    vggt_out = run_vggt_on_images(model, images, device, max_batch=15)
    del model
    torch.cuda.empty_cache()

    results["timings"]["vggt"] = time.time() - t0
    results["vggt_poses"] = vggt_out["poses_c2w"].copy()
    print(f"  VGGT done in {results['timings']['vggt']:.1f}s, {N} frames")

    # Evaluate VGGT poses
    if gt_poses is not None:
        vggt_metrics = absolute_trajectory_error(gt_poses, vggt_out["poses_c2w"])
        results["vggt_ate"] = vggt_metrics
        print(f"  VGGT ATE: {vggt_metrics['ate_mean']:.4f}m")

    # === Step 2: Factor graph refinement ===
    if use_factor_graph:
        print("Step 2: Factor graph refinement...")
        t0 = time.time()

        pose_sigmas = vggt_conf_to_covariance(vggt_out["pose_conf"])
        loop_closures = detect_loop_closures(
            vggt_out["poses_c2w"], images=images,
            points=vggt_out["points"], point_conf=vggt_out["point_conf"],
            distance_threshold=1.0, min_frame_gap=20,
        )
        print(f"  Detected {len(loop_closures)} loop closures")

        refined_poses, covariances = build_factor_graph(
            vggt_out["poses_c2w"], pose_sigmas, loop_closures
        )

        results["timings"]["factor_graph"] = time.time() - t0
        results["refined_poses"] = refined_poses.copy()
        results["covariances"] = covariances
        print(f"  Factor graph done in {results['timings']['factor_graph']:.1f}s")

        if gt_poses is not None:
            fg_metrics = absolute_trajectory_error(gt_poses, refined_poses)
            results["fg_ate"] = fg_metrics
            print(f"  Refined ATE: {fg_metrics['ate_mean']:.4f}m")
            improvement = (
                (vggt_metrics["ate_mean"] - fg_metrics["ate_mean"])
                / vggt_metrics["ate_mean"] * 100
            )
            print(f"  Improvement: {improvement:.1f}%")

    # === Step 3: Gaussian initialization ===
    if train_gaussians:
        print("Step 3: Initializing Gaussians from VGGT point maps...")
        t0 = time.time()

        gs_params = init_gaussians_from_vggt(
            vggt_out["points"], images, vggt_out["point_conf"],
            vggt_out["poses_c2w"], conf_threshold=0.3, stride=4,
            device=device,
        )
        n_gs = len(gs_params["means"])
        results["timings"]["gaussian_init"] = time.time() - t0
        print(f"  Initialized {n_gs} Gaussians in {results['timings']['gaussian_init']:.1f}s")

        # === Step 4: Train Gaussians with VGGT poses ===
        print("Step 4a: Training Gaussians with VGGT poses...")
        t0 = time.time()
        gs_params_vggt = {k: torch.nn.Parameter(v.data.clone()) for k, v in gs_params.items()}

        # Resize images for rendering if needed
        import cv2
        render_H, render_W = H, W
        render_images = images

        vggt_loss = train_gs(
            gs_params_vggt, vggt_out["poses_c2w"], render_images,
            K, render_W, render_H, n_iters=n_train_iters, device=device,
        )
        results["timings"]["gaussian_train_vggt"] = time.time() - t0
        results["vggt_render_loss"] = vggt_loss

        # Render a sample for comparison
        mid = N // 2
        vggt_render = render_gaussians(
            gs_params_vggt, vggt_out["poses_c2w"][mid], K, render_W, render_H, device
        )
        results["vggt_render"] = vggt_render
        results["vggt_render_psnr"] = psnr(images[mid], vggt_render)
        print(f"  VGGT render PSNR: {results['vggt_render_psnr']:.2f} dB")

        # === Step 4b: Train Gaussians with refined poses ===
        if use_factor_graph:
            print("Step 4b: Training Gaussians with refined poses...")
            t0 = time.time()
            gs_params_fg = {k: torch.nn.Parameter(v.data.clone()) for k, v in gs_params.items()}

            fg_loss = train_gs(
                gs_params_fg, refined_poses, render_images,
                K, render_W, render_H, n_iters=n_train_iters, device=device,
            )
            results["timings"]["gaussian_train_fg"] = time.time() - t0
            results["fg_render_loss"] = fg_loss

            fg_render = render_gaussians(
                gs_params_fg, refined_poses[mid], K, render_W, render_H, device
            )
            results["fg_render"] = fg_render
            results["fg_render_psnr"] = psnr(images[mid], fg_render)
            print(f"  Refined render PSNR: {results['fg_render_psnr']:.2f} dB")

    return results

"""Multi-model backend: ensemble VGGT + MASt3R for better poses.

VGGT is fast but struggles on textureless regions. MASt3R is slower but
handles those cases better. By running both and fusing their predictions
via confidence weighting in the factor graph, we get the best of both.
"""

import numpy as np
import torch
from typing import Optional


def run_multi_backend(
    images: np.ndarray,
    device: str = "cuda",
    chunk_size: int = 10,
    overlap: int = 3,
    backends: list[str] = ["vggt", "mast3r"],
) -> dict:
    """Run multiple pose estimation backends and return their predictions.

    Each backend provides independent pose estimates. The factor graph
    fuses them using per-backend confidence as noise scaling.
    """
    results = {}

    if "vggt" in backends:
        from .vggt_wrapper import load_vggt, run_vggt_on_images
        print("  Running VGGT backend...")
        model = load_vggt(device)
        vggt_out = run_vggt_on_images(model, images, device, max_batch=chunk_size)
        del model
        torch.cuda.empty_cache()
        results["vggt"] = vggt_out

    if "mast3r" in backends:
        print("  Running MASt3R backend...")
        try:
            mast3r_out = _run_mast3r(images, device)
            results["mast3r"] = mast3r_out
        except ImportError:
            print("    MASt3R not installed, skipping")
        except Exception as e:
            print(f"    MASt3R failed: {e}")

    return results


def fuse_multi_backend_poses(
    backend_results: dict,
    method: str = "confidence_weighted",
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse poses from multiple backends.

    Args:
        backend_results: dict mapping backend name -> output dict
        method: "confidence_weighted" or "best_per_frame"

    Returns:
        (fused_poses_c2w, fused_confidence) arrays
    """
    backends = list(backend_results.keys())
    if len(backends) == 1:
        out = backend_results[backends[0]]
        return out["poses_c2w"], out["pose_conf"]

    N = backend_results[backends[0]]["poses_c2w"].shape[0]
    all_poses = []
    all_confs = []

    for name in backends:
        out = backend_results[name]
        all_poses.append(out["poses_c2w"])
        all_confs.append(out["pose_conf"])

    all_poses = np.array(all_poses)  # (B, N, 4, 4)
    all_confs = np.array(all_confs)  # (B, N)

    if method == "best_per_frame":
        best_idx = np.argmax(all_confs, axis=0)
        fused = np.array([all_poses[best_idx[i], i] for i in range(N)])
        fused_conf = np.max(all_confs, axis=0)
    else:
        # Confidence-weighted average of translations, SLERP of rotations
        from scipy.spatial.transform import Rotation, Slerp

        weights = all_confs / all_confs.sum(axis=0, keepdims=True)  # (B, N)
        fused = np.zeros((N, 4, 4))

        for i in range(N):
            # Weighted average translation
            t = sum(weights[b, i] * all_poses[b, i, :3, 3] for b in range(len(backends)))

            # SLERP between rotations (use highest confidence as base)
            rotations = [Rotation.from_matrix(all_poses[b, i, :3, :3]) for b in range(len(backends))]
            w = weights[:, i]
            # Simple: interpolate between the two
            if len(rotations) == 2:
                slerp = Slerp([0, 1], Rotation.concatenate(rotations))
                r = slerp(w[1])
            else:
                r = rotations[np.argmax(w)]

            fused[i] = np.eye(4)
            fused[i, :3, :3] = r.as_matrix()
            fused[i, :3, 3] = t

        fused_conf = np.max(all_confs, axis=0)

    return fused, fused_conf


def _run_mast3r(images: np.ndarray, device: str) -> dict:
    """Run MASt3R sparse global alignment on images."""
    from mast3r.model import AsymmetricMASt3R
    from mast3r.image_pairs import make_pairs
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    import mast3r.utils.path_to_dust3r  # noqa
    from dust3r.utils.image import load_images as dust3r_load
    import tempfile
    import cv2

    # Save images temporarily for dust3r loader
    tmp_dir = tempfile.mkdtemp()
    paths = []
    for i, img in enumerate(images):
        path = f"{tmp_dir}/frame_{i:04d}.png"
        cv2.imwrite(path, (img * 255).astype(np.uint8)[:, :, ::-1])
        paths.append(path)

    model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    ).to(device)

    loaded = dust3r_load(paths, size=512)
    pairs = make_pairs(loaded, scene_graph="swin-5", prefilter=None, symmetrize=True)

    scene = sparse_global_alignment(
        paths, pairs, tmp_dir, model,
        lr1=0.07, niter1=200, lr2=0.01, niter2=200,
        device=device, opt_depth=True, shared_intrinsics=True,
    )

    poses_c2w = scene.get_im_poses().detach().cpu().numpy()
    pose_conf = np.ones(len(images))  # MASt3R doesn't give per-frame conf

    del model, scene
    torch.cuda.empty_cache()

    return {
        "poses_c2w": poses_c2w,
        "pose_conf": pose_conf,
    }

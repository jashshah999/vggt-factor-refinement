"""Gaussian splatting rendering using gsplat."""

import torch
import numpy as np
from gsplat import rasterization


def render_gaussians(
    params: dict,
    pose_c2w: np.ndarray,
    K: np.ndarray,
    W: int,
    H: int,
    device: str = "cuda",
) -> np.ndarray:
    """Render Gaussians from a given camera pose.

    Args:
        params: dict with means, colors, scales, opacities, quats
        pose_c2w: (4, 4) camera-to-world pose
        K: (3, 3) camera intrinsics
        W, H: image dimensions

    Returns:
        (H, W, 3) rendered image in [0, 1]
    """
    viewmat = torch.tensor(
        np.linalg.inv(pose_c2w), dtype=torch.float32, device=device
    )
    # gsplat Y-flip convention
    viewmat[1] = -viewmat[1]

    K_torch = torch.tensor(K, dtype=torch.float32, device=device)

    with torch.no_grad():
        rendered, _, _ = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"],
            opacities=params["opacities"].sigmoid(),
            colors=params["colors"],
            viewmats=viewmat[None],
            Ks=K_torch[None],
            width=W,
            height=H,
            packed=False,
        )

    return np.clip(rendered[0].cpu().numpy(), 0, 1)


def train_gaussians(
    params: dict,
    poses_c2w: np.ndarray,
    images: np.ndarray,
    K: np.ndarray,
    W: int,
    H: int,
    n_iters: int = 500,
    lr: float = 0.005,
    device: str = "cuda",
) -> float:
    """Train Gaussians on a set of posed images.

    Returns final average loss.
    """
    K_torch = torch.tensor(K, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam([
        {"params": [params["means"]], "lr": lr * 0.1},
        {"params": [params["colors"]], "lr": lr},
        {"params": [params["scales"]], "lr": lr * 0.5},
        {"params": [params["opacities"]], "lr": lr},
        {"params": [params["quats"]], "lr": lr * 0.1},
    ])

    N = len(poses_c2w)
    final_loss = 0.0

    for step in range(n_iters):
        # Pick a random frame
        idx = np.random.randint(N)
        target = torch.tensor(images[idx], dtype=torch.float32, device=device)

        viewmat = torch.tensor(
            np.linalg.inv(poses_c2w[idx]), dtype=torch.float32, device=device
        )
        viewmat[1] = -viewmat[1]

        # Clamp scales and normalize quats
        with torch.no_grad():
            params["scales"].data.clamp_(0.001, 2.0)
            params["quats"].data = torch.nn.functional.normalize(
                params["quats"].data, dim=-1
            )

        optimizer.zero_grad()

        rendered, _, _ = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"],
            opacities=params["opacities"].sigmoid(),
            colors=params["colors"],
            viewmats=viewmat[None],
            Ks=K_torch[None],
            width=W,
            height=H,
            packed=False,
        )

        loss = torch.nn.functional.l1_loss(rendered[0], target)
        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        if step % 100 == 0:
            print(f"  Gaussian training step {step}/{n_iters}: loss={final_loss:.4f}")

    return final_loss

"""Appearance-based loop closure detection.

Uses DINOv2 global features to find visually similar frames regardless
of how far apart they are in the estimated trajectory. This fixes the
failure mode where position-based loop closure misses revisits because
the naive trajectory has drifted too far.
"""

import numpy as np
import torch
import torch.nn.functional as F


def build_frame_descriptors(images: np.ndarray, device: str = "cuda", batch_size: int = 16) -> np.ndarray:
    """Extract global DINOv2 descriptors for each frame.

    Args:
        images: (N, H, W, 3) in [0, 1]

    Returns:
        (N, D) L2-normalized descriptor array
    """
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    model = model.to(device).eval()

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    N = len(images)
    all_descs = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = images[start:end]

        # Resize to 224x224 for DINOv2
        import cv2
        resized = np.stack([cv2.resize(img, (224, 224)) for img in batch])
        tensors = torch.tensor(resized, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        tensors = transform(tensors)

        with torch.no_grad():
            features = model(tensors)

        # L2 normalize
        features = F.normalize(features, dim=-1)
        all_descs.append(features.cpu().numpy())

    del model
    torch.cuda.empty_cache()

    return np.concatenate(all_descs, axis=0)


def find_appearance_loop_closures(
    descriptors: np.ndarray,
    similarity_threshold: float = 0.7,
    min_frame_gap: int = 20,
    max_closures: int = 50,
) -> list:
    """Find loop closure pairs based on visual similarity.

    Returns list of (i, j, similarity_score) tuples.
    """
    N = len(descriptors)
    # Compute pairwise similarity matrix
    sim = descriptors @ descriptors.T

    candidates = []
    for i in range(N):
        for j in range(i + min_frame_gap, N):
            if sim[i, j] > similarity_threshold:
                candidates.append((i, j, float(sim[i, j])))

    # Sort by similarity (best first)
    candidates.sort(key=lambda x: -x[2])

    # Deduplicate: don't add closures too close to existing ones
    selected = []
    used_pairs = set()
    for i, j, score in candidates:
        # Check no nearby pair already selected
        too_close = False
        for si, sj, _ in selected:
            if abs(i - si) < 5 and abs(j - sj) < 5:
                too_close = True
                break
        if not too_close:
            selected.append((i, j, score))
            if len(selected) >= max_closures:
                break

    return selected

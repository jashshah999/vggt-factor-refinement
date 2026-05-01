"""Debug Replica scale and alignment issues."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from src.replica_loader import load_replica_sequence
from src.vggt_wrapper import load_vggt, run_vggt_on_images
from src.cross_chunk_align import estimate_cross_chunk_relative_pose

data = load_replica_sequence("room0", stride=1, max_frames=20)
print(f"Image shape: {data['images'][0].shape}")

model = load_vggt("cuda")

# Run two chunks
chunk1 = run_vggt_on_images(model, data["images"][:8], "cuda", max_batch=8)
chunk2 = run_vggt_on_images(model, data["images"][6:14], "cuda", max_batch=8)

print(f"Points shape per frame: {chunk1['points'][0].shape}")

# Check point cloud scales
for name, chunk in [("chunk1", chunk1), ("chunk2", chunk2)]:
    valid = chunk["points"][chunk["point_conf"] > 0.3]
    valid = valid[np.isfinite(valid).all(axis=-1)]
    print(f"{name}: mean_dist={np.mean(np.linalg.norm(valid, axis=-1)):.3f}, "
          f"range=[{valid.min():.2f}, {valid.max():.2f}]")

# Test cross-chunk alignment between overlapping frames
# Frame 7 in chunk1 (local index 7) = Frame 7 in chunk2 (local index 1)
rel, n_inliers = estimate_cross_chunk_relative_pose(
    data["images"][7], data["images"][7],  # same frame!
    chunk1["points"][7], chunk2["points"][1],
    chunk1["point_conf"][7], chunk2["point_conf"][1],
    chunk1["poses_c2w"][7], chunk2["poses_c2w"][1],
    min_matches=10,
)
print(f"\nSame-frame cross-chunk alignment: inliers={n_inliers}")
if rel is not None:
    print(f"Relative pose (should be near identity):")
    print(f"  translation norm: {np.linalg.norm(rel[:3, 3]):.6f}")
    print(f"  rotation angle: {np.degrees(np.arccos(np.clip((np.trace(rel[:3,:3]) - 1) / 2, -1, 1))):.2f} deg")
else:
    print("  FAILED to align")

# Test with different frames
rel2, n_inliers2 = estimate_cross_chunk_relative_pose(
    data["images"][3], data["images"][10],
    chunk1["points"][3], chunk2["points"][4],
    chunk1["point_conf"][3], chunk2["point_conf"][4],
    chunk1["poses_c2w"][3], chunk2["poses_c2w"][4],
    min_matches=10,
)
print(f"\nDifferent-frame cross-chunk alignment: inliers={n_inliers2}")
if rel2 is not None:
    print(f"  translation norm: {np.linalg.norm(rel2[:3, 3]):.4f}")
    print(f"  rotation angle: {np.degrees(np.arccos(np.clip((np.trace(rel2[:3,:3]) - 1) / 2, -1, 1))):.2f} deg")
    # Compare with GT relative pose
    gt_rel = np.linalg.inv(data["gt_poses"][3]) @ data["gt_poses"][10]
    print(f"  GT translation norm: {np.linalg.norm(gt_rel[:3, 3]):.4f}")
    print(f"  GT rotation angle: {np.degrees(np.arccos(np.clip((np.trace(gt_rel[:3,:3]) - 1) / 2, -1, 1))):.2f} deg")

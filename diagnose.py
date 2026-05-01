"""Quick diagnostic for factor graph regression."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from src.data_loaders import load_tum_sequence

for seq in ["fr1/desk", "fr1/room", "fr1/xyz", "fr3/office"]:
    try:
        data = load_tum_sequence(seq, stride=3, max_frames=80)
    except Exception as e:
        print(f"{seq}: {e}")
        continue

    gt = data["gt_poses"]
    N = len(gt)
    positions = gt[:, :3, 3]

    traj_len = sum(np.linalg.norm(positions[i+1] - positions[i]) for i in range(N-1))
    max_disp = max(np.linalg.norm(positions[i] - positions[0]) for i in range(N))

    # Count close revisits (loop closure candidates)
    lc_count = 0
    for i in range(N):
        for j in range(i + 15, N):
            if np.linalg.norm(positions[i] - positions[j]) < 1.5:
                lc_count += 1

    print(f"{seq}: {N} frames, traj={traj_len:.1f}m, max_disp={max_disp:.1f}m, lc_candidates={lc_count}")

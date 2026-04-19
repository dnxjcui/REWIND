"""Load DynHaMR smooth_fit NPZ and run MANO forward pass for the right hand."""

import numpy as np

# chumpy (used inside MANO_RIGHT.pkl) references removed numpy aliases.
# Patch them before importing smplx.
np.bool    = bool    # type: ignore[attr-defined]
np.int     = int     # type: ignore[attr-defined]
np.float   = float   # type: ignore[attr-defined]
np.complex = complex # type: ignore[attr-defined]
np.object  = object  # type: ignore[attr-defined]
np.str     = str     # type: ignore[attr-defined]
np.unicode = str     # type: ignore[attr-defined]

import torch
import smplx
from pathlib import Path


def load_mano_trajectory(npz_path: str | Path, mano_dir: str | Path):
    """Return per-frame MANO joints and pose parameters for the right-hand track.

    Returns
    -------
    dict with keys:
        joints       : (T, 21, 3) float32, world frame (Y-down optimization frame)
        root_orient  : (T, 3)     float32, axis-angle
        trans        : (T, 3)     float32
        betas        : (10,)      float32
        is_right_mean: float      sanity value (should be ~1.0)
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    data = np.load(str(npz_path), allow_pickle=False)

    # Select the right-hand track (glove is right-hand only)
    # is_right shape: (B, T); pick track with highest mean is_right value
    is_right = data["is_right"]          # (B, T)
    track = int(np.argmax(is_right.mean(axis=1)))
    is_right_mean = float(is_right[track].mean())

    if is_right_mean < 0.5:
        raise RuntimeError(
            f"No right-hand track found (best track is_right mean = {is_right_mean:.3f}). "
            "Check the NPZ file or track index."
        )
    if is_right_mean < 0.95:
        print(
            f"[WARNING] Right-hand track {track} has is_right mean = {is_right_mean:.3f} "
            "(< 0.95). Some frames may be left-hand detections."
        )

    # Extract single-track arrays: (T, ...)
    pose_body   = data["pose_body"][track]    # (T, 15, 3) axis-angle per joint
    root_orient = data["root_orient"][track]  # (T, 3)
    trans       = data["trans"][track]        # (T, 3)
    betas       = data["betas"][track]        # (10,)
    T = pose_body.shape[0]

    # Flatten pose_body: (T, 15, 3) → (T, 45)
    pose_body_flat = pose_body.reshape(T, 45)

    # MANO forward pass — always use right-hand model; no x-flip needed (is_right ≈ 1)
    mano_dir = Path(mano_dir)
    model = smplx.create(
        str(mano_dir),
        model_type="mano",
        is_rhand=True,
        use_pca=False,
        num_betas=10,
        batch_size=T,
    )
    model.eval()

    with torch.no_grad():
        out = model(
            hand_pose=torch.tensor(pose_body_flat, dtype=torch.float32),
            betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0).expand(T, -1),
            global_orient=torch.tensor(root_orient, dtype=torch.float32),
            transl=torch.tensor(trans, dtype=torch.float32),
        )

    # joints: (T, 16, 3) — 16 MANO pose joints (DynHaMR pkl has no extra fingertip joints)
    # vertices: (T, 778, 3) — full MANO mesh vertices; use for fingertip positions
    joints   = out.joints.numpy().astype(np.float32)
    vertices = out.vertices.numpy().astype(np.float32)

    return {
        "joints":        joints,    # (T, 16, 3)
        "vertices":      vertices,  # (T, 778, 3)
        "root_orient":   root_orient.astype(np.float32),
        "trans":         trans.astype(np.float32),
        "betas":         betas.astype(np.float32),
        "is_right_mean": is_right_mean,
        "T":             T,
    }

"""Load MANO data from DynHaMR NPZ for a single frame."""

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

# Patch deprecated numpy aliases before importing smplx/chumpy
# chumpy does `from numpy import ..., unicode` which was removed in NumPy 1.24+
import numpy as _np
if not hasattr(_np, "bool"):    _np.bool    = bool
if not hasattr(_np, "int"):     _np.int     = int
if not hasattr(_np, "float"):   _np.float   = float
if not hasattr(_np, "complex"): _np.complex = complex
if not hasattr(_np, "object"):  _np.object  = object
if not hasattr(_np, "str"):     _np.str     = str
if not hasattr(_np, "unicode"): _np.unicode = str  # chumpy specifically needs this

import torch
import smplx


_model_cache: dict = {}


def load_frame(npz_path, mano_dir, frame_idx: int) -> dict:
    """Load MANO data for one frame.

    Returns dict with:
        T_wrist   : (4, 4) float64 — wrist world transform, Y-down frame
        thumb_tip : (3,)   float64 — world position of vertex 745
        index_tip : (3,)   float64 — world position of vertex 317
    """
    data = np.load(str(npz_path), allow_pickle=False)

    track = int(np.argmax(data["is_right"].mean(axis=1)))

    root_orient = data["root_orient"][track, frame_idx].astype(np.float64)  # (3,)
    trans       = data["trans"][track, frame_idx].astype(np.float64)        # (3,)
    # pose_body shape is (15, 3) — reshape to (45,) for MANO hand_pose
    pose_body   = data["pose_body"][track, frame_idx].astype(np.float32).reshape(-1)  # (45,)
    betas       = data["betas"][track].astype(np.float32)                   # (10,)

    cache_key = str(mano_dir)
    if cache_key not in _model_cache:
        m = smplx.create(
            str(mano_dir), model_type="mano", is_rhand=True,
            use_pca=False, num_betas=10, batch_size=1,
        )
        m.eval()
        _model_cache[cache_key] = m
    model = _model_cache[cache_key]
    with torch.no_grad():
        out = model(
            hand_pose=torch.tensor(pose_body.reshape(1, 45), dtype=torch.float32),
            betas=torch.tensor(betas[None], dtype=torch.float32),
            global_orient=torch.tensor(root_orient[None].astype(np.float32), dtype=torch.float32),
            transl=torch.tensor(trans[None].astype(np.float32), dtype=torch.float32),
        )

    vertices = out.vertices[0].numpy().astype(np.float64)  # (778, 3)

    T_wrist = np.eye(4, dtype=np.float64)
    T_wrist[:3, :3] = Rotation.from_rotvec(root_orient).as_matrix()
    T_wrist[:3,  3] = trans

    return {
        "T_wrist":   T_wrist,
        "thumb_tip": vertices[745].copy(),
        "index_tip": vertices[317].copy(),
    }

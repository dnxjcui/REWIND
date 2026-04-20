from pathlib import Path
import json
import numpy as np
import numpy as np

_GLOVE_SIM_ROOT = Path(__file__).parent
_PROJECT_ROOT   = _GLOVE_SIM_ROOT.parent

DYNHAMR_ROOT = _PROJECT_ROOT / "Dyn-HaMR"
URDF_PATH    = _PROJECT_ROOT / "rewind_glove_assembly/urdf/rewind_glove_for_urdfpy.urdf"
MJCF_PATH    = _PROJECT_ROOT / "rewind_glove_assembly/urdf/rewind_glove_mujoco.xml"
MESH_DIR     = _PROJECT_ROOT / "rewind_glove_assembly/meshes"
MANO_DIR     = DYNHAMR_ROOT / "_DATA/data"

NPZ_PATH = (
    DYNHAMR_ROOT
    / "outputs/logs/video-custom/2026-03-25"
    / "knot_one_handed-all-shot-0-0--1/smooth_fit"
    / "knot_one_handed_000300_world_results.npz"
)

GLB_DIR = (
    DYNHAMR_ROOT
    / "outputs/logs/video-custom/2026-03-25"
    / "knot_one_handed-all-shot-0-0--1/unity_export/frames"
)

ALIGNED_DIR = _GLOVE_SIM_ROOT / "outputs/aligned"

IK_TOL                = 1e-6   # scipy ftol
IK_MAX_NFEV           = 200
IK_RESIDUAL_THRESHOLD = 0.005  # 5 mm

# T_WRIST_TO_HM is loaded automatically from plane_annotation.json.
# Re-run annotate_planes.py to update it; no manual edits needed.
_ANNOTATION_PATH = ALIGNED_DIR / "plane_annotation.json"
if _ANNOTATION_PATH.exists():
    T_WRIST_TO_HM = np.array(
        json.loads(_ANNOTATION_PATH.read_text())["T_WRIST_TO_HM"],
        dtype=np.float64,
    )
else:
    T_WRIST_TO_HM = np.eye(4, dtype=np.float64)  # identity until first annotation

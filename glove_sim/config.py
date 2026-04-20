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

# Derive the video name from the NPZ path segment, e.g. "knot_one_handed".
# Annotation JSONs stay at ALIGNED_DIR root (shared across runs of the same video).
# Per-frame outputs live under ALIGNED_DIR / VIDEO_NAME / {glb_frames,urdf_frames}.
VIDEO_NAME = NPZ_PATH.parent.parent.name.split('-all-shot')[0]
GLB_FRAMES_DIR  = ALIGNED_DIR / VIDEO_NAME / "glb_frames"
URDF_FRAMES_DIR = ALIGNED_DIR / VIDEO_NAME / "urdf_frames"

SEQUENCE_START = 18  # first usable frame (frames 0-17 are pre-task junk)

IK_TOL                = 1e-6   # scipy ftol
IK_MAX_NFEV           = 200
IK_RESIDUAL_THRESHOLD = 0.005  # 5 mm

# Calibration loaded automatically from plane_annotation.json.
# Re-run annotate_planes.py to update; no manual edits needed.
_ANNOTATION_PATH = ALIGNED_DIR / "plane_annotation.json"
if _ANNOTATION_PATH.exists():
    _ann = json.loads(_ANNOTATION_PATH.read_text())
    T_WRIST_TO_HM = np.array(_ann["T_WRIST_TO_HM"], dtype=np.float64)

    # Dynamic per-frame Kabsch: anchor MANO vertex IDs and matching glove points
    # in hand_mount local frame (MANO convention). Both are None if the annotation
    # pre-dates this feature (re-run annotate_planes.py to populate them).
    ANCHOR_VERTEX_IDS  = _ann.get("anchor_vertex_ids")          # list[int] or None
    GLOVE_ANCHOR_PTS_HM = (
        np.array(_ann["glove_anchor_pts_hm"], dtype=np.float64)
        if "glove_anchor_pts_hm" in _ann else None
    )

    # Hand dorsal plane for IK collision penalty (in GLB space).
    _hd = _ann.get("hand_dorsal", {})
    HAND_DORSAL_CENTROID_GLB = np.array(_hd.get("centroid", [0, 0, 0]), dtype=np.float64)
    HAND_DORSAL_NORMAL_GLB   = np.array(_hd.get("normal",   [0, 1, 0]), dtype=np.float64)
else:
    T_WRIST_TO_HM            = np.eye(4, dtype=np.float64)
    ANCHOR_VERTEX_IDS        = None
    GLOVE_ANCHOR_PTS_HM      = None
    HAND_DORSAL_CENTROID_GLB = np.zeros(3, dtype=np.float64)
    HAND_DORSAL_NORMAL_GLB   = np.array([0.0, 1.0, 0.0], dtype=np.float64)

_FT_ANNOTATION_PATH = ALIGNED_DIR / "fingertip_annotation.json"
if _FT_ANNOTATION_PATH.exists():
    _ft = json.loads(_FT_ANNOTATION_PATH.read_text())
    THUMB_TIP_OFFSET = np.array(_ft["thumb_tip_offset_wrist"], dtype=np.float64)
    INDEX_TIP_OFFSET = np.array(_ft["index_tip_offset_wrist"], dtype=np.float64)
else:
    THUMB_TIP_OFFSET = np.zeros(3, dtype=np.float64)
    INDEX_TIP_OFFSET = np.zeros(3, dtype=np.float64)

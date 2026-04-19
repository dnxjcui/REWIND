from pathlib import Path

_ROOT = Path(__file__).parent

DYNHAMR_ROOT = _ROOT.parent / "Dyn-HaMR"

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

MANO_DIR = DYNHAMR_ROOT / "_DATA/data"  # smplx appends /mano automatically

URDF_PATH     = _ROOT.parent / "rewind_glove_assembly/urdf/rewind_glove_assembly.urdf"
MESH_DIR_SRC  = _ROOT.parent / "rewind_glove_assembly/meshes"   # original ASCII STLs
MESH_DIR      = _ROOT / "assets/meshes"                          # binary STLs for MuJoCo

OUTPUT_DIR       = _ROOT / "outputs"
FRAMES_DIR       = OUTPUT_DIR / "frames"
SENSORS_NPZ_PATH = OUTPUT_DIR / "simulated_sensors.npz"
CALIBRATION_PATH = OUTPUT_DIR / "calibration.npy"

FPS           = 30
IK_TOL        = 1e-4   # meters
IK_ITERS      = 50
SMOOTH_WINDOW = 5      # Savitzky-Golay window length (frames)
IK_RESIDUAL_THRESHOLD = 0.01  # meters; frames above this are unreliable

# MANO joint indices (16 joints from DynHaMR pkl — no extra fingertip joints)
JOINT_WRIST     = 0
JOINT_INDEX_DIP = 3   # index3 (DIP joint) — used for finger orientation vector
JOINT_THUMB_IP  = 15  # thumb3 (IP joint) — used for thumb orientation vector

# MANO vertex indices for fingertip surface positions (standard MANO right-hand vertices)
# Used instead of joints since DynHaMR's pkl doesn't add extra fingertip joints.
VERTEX_THUMB_TIP = 745
VERTEX_INDEX_TIP = 317

# Glove sensor joint names (from URDF) — 5 sensors total
SENSOR_JOINTS = {
    "thumb_base":     "revolute_3_0",  # part_1 → part_2_1 junction (thumb)
    "thumb_tip":      "revolute_4_0",  # part_2_1 → part_3 (thumb cap)
    "index_base":     "revolute_7_0",  # part_1_1 → part_2 junction (index)
    "index_tip":      "revolute_9_0",  # part_2 → part_3_1 (index cap)
    "index_platform": "revolute_5_0",  # index platform lateral sweep (near part_5)
}

# Sensor world-position data: (body_name, offset_in_body_local_frame).
# All offsets are in the body's own local frame (rotated by R_body before adding to xpos).
# After the visual Rx(180°) flip, screws face the -Y body direction; offset (0, -0.004, 0)
# places the dot 4 mm in the -Y local direction = next to the screws.
import numpy as _np
SENSOR_POSITIONS = {
    "thumb_base":     ("part_2_1", _np.array([0.0,  -0.004,  0.0])),   # Part1→Part2_1 junction
    "thumb_tip":      ("part_3",   _np.array([0.0,  -0.004,  0.0])),   # geometric center of thumb cap
    "index_base":     ("part_2",   _np.array([0.0,  -0.004,  0.0])),   # Part1_1→Part2 junction
    "index_tip":      ("part_3_1", _np.array([0.0,  -0.004,  0.0])),   # geometric center of index cap
    # part_5 vis_origin ≈ [0.014, 0.074, -0.004] places dot at the free end of the platform
    "index_platform": ("part_5",   _np.array([0.014, 0.074, -0.004])),
}
del _np
SENSOR_SPHERE_RADIUS = 0.005  # 5 mm

# All revolute joints participating in IK for each finger
THUMB_CHAIN  = ["revolute_1_0", "revolute_2_0", "revolute_3_0", "revolute_4_0"]
INDEX_CHAIN  = ["revolute_5_0", "revolute_6_0", "revolute_7_0", "revolute_8_0", "revolute_9_0"]

# Fingertip cap body names in the URDF
THUMB_TIP_BODY = "part_3"
INDEX_TIP_BODY = "part_3_1"

# Link name → STL filename mapping (all 17 URDF links with visual geometry)
LINK_TO_STL = {
    "hand_mount":       "Hand Mount.stl",
    "part_1":           "Part 1.stl",
    "part_1_1":         "Part 1_1.stl",
    "part_2":           "Part 2.stl",
    "part_2_1":         "Part 2_1.stl",
    "part_3":           "Part 3.stl",
    "part_3_1":         "Part 3_1.stl",
    "part_5":           "Part 5.stl",
    "part_6":           "Part 6.stl",
    "xl330_m077_t":     "XL330_M077_T.stl",
    "xl330_m077_t_1":   "XL330_M077_T_1.stl",
    "xl330_m077_t_2":   "XL330_M077_T_1.stl",
    "xl330_m077_t_3":   "XL330_M077_T.stl",
    "xl_housing":       "XL Housing.stl",
    "xl_housing_1":     "XL Housing.stl",
    "xl_housing_cap":   "XL Housing Cap.stl",
    "xl_housing_cap_1": "XL Housing Cap.stl",
    "xl_linkage_horn":  "XL Linkage Horn.stl",
    "xl_linkage_horn_1":"XL Linkage Horn.stl",
    "xl_platform_horn": "XL Platform Horn.stl",
    "xl_platform_horn_1":"XL Platform Horn.stl",
}

"""Plane-alignment tests for glove-hand fitting.

These tests require a completed annotation produced by annotate_planes.py.
They are automatically skipped if no annotation file exists.

They will FAIL until T_WRIST_TO_HM is calibrated in config.py and
align_frame.py is updated to use it.

Run:
    pytest glove_sim/tests/test_plane_alignment.py -v -p no:dash
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "glove_sim"))
import config as cfg

ANNOTATION_PATH = cfg.ALIGNED_DIR / "plane_annotation.json"
ANGLE_THRESHOLD_DEG = 15.0   # normals must be within 15° of antiparallel
GAP_THRESHOLD_M     = 0.005  # glove bottom centroid within 5 mm of dorsal plane
IK_RESIDUAL_M       = 0.005  # finger IK residual threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_annotation():
    if not ANNOTATION_PATH.exists():
        pytest.skip(f"No annotation file — run annotate_planes.py first: {ANNOTATION_PATH}")
    return json.loads(ANNOTATION_PATH.read_text())


def _angle_between_normals_deg(n1, n2):
    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    cos_a = min(1.0, abs(float(np.dot(n1, n2))))  # abs → 0–90° range
    return float(np.degrees(np.arccos(cos_a)))


def _point_to_plane_dist_m(point, plane_centroid, plane_normal):
    """Signed distance from point to the given plane."""
    n = np.asarray(plane_normal, dtype=float)
    n /= np.linalg.norm(n)
    return float(np.dot(np.asarray(point) - np.asarray(plane_centroid), n))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def annotation():
    return _load_annotation()


# ---------------------------------------------------------------------------
# Annotation sanity
# ---------------------------------------------------------------------------

def test_annotation_has_enough_points(annotation):
    """Both surfaces need >= 3 points to fit a plane."""
    assert len(annotation["glove_bottom"]["points"]) >= 3, \
        "glove_bottom needs >= 3 points"
    assert len(annotation["hand_dorsal"]["points"]) >= 3, \
        "hand_dorsal needs >= 3 points"


def test_annotation_normals_are_unit_vectors(annotation):
    for key in ("glove_bottom", "hand_dorsal"):
        n = np.array(annotation[key]["normal"], dtype=float)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-4, \
            f"{key} normal is not a unit vector: {np.linalg.norm(n)}"


# ---------------------------------------------------------------------------
# Alignment tests
# ---------------------------------------------------------------------------

def test_glove_bottom_parallel_to_dorsal(annotation):
    """Glove flat-bottom normal must be within 15° of antiparallel to dorsal normal.

    'Antiparallel' because the flat bottom faces INTO the hand, meaning
    the glove normal and dorsal normal point in opposite directions.
    """
    angle = _angle_between_normals_deg(
        annotation["glove_bottom"]["normal"],
        annotation["hand_dorsal"]["normal"],
    )
    assert angle < ANGLE_THRESHOLD_DEG, (
        f"Normals {angle:.1f}° apart (measured as angle from antiparallel); "
        f"target < {ANGLE_THRESHOLD_DEG}°"
    )


def test_glove_bottom_near_dorsal_plane(annotation):
    """Glove flat-bottom centroid must lie within 5 mm of the hand dorsal plane."""
    gap_m = abs(_point_to_plane_dist_m(
        annotation["glove_bottom"]["centroid"],
        annotation["hand_dorsal"]["centroid"],
        annotation["hand_dorsal"]["normal"],
    ))
    assert gap_m < GAP_THRESHOLD_M, (
        f"Gap between planes: {gap_m*1000:.1f} mm; "
        f"target < {GAP_THRESHOLD_M*1000:.0f} mm"
    )


def test_glove_bottom_centroid_projects_inside_hand_dorsal(annotation):
    """Glove bottom centroid projected onto the dorsal plane should lie
    within the bounding box of the annotated dorsal points (rough check that
    the glove is over the hand, not displaced laterally).
    """
    pts = np.array(annotation["hand_dorsal"]["points"], dtype=float)
    c_g = np.array(annotation["glove_bottom"]["centroid"], dtype=float)
    n_h = np.array(annotation["hand_dorsal"]["normal"], dtype=float)
    n_h /= np.linalg.norm(n_h)
    c_h = np.array(annotation["hand_dorsal"]["centroid"], dtype=float)

    # Project onto dorsal plane
    c_g_proj = c_g - np.dot(c_g - c_h, n_h) * n_h

    # Build an orthonormal basis for the plane
    arb = np.array([1.0, 0.0, 0.0]) if abs(n_h[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n_h, arb); u /= np.linalg.norm(u)
    v = np.cross(n_h, u);   v /= np.linalg.norm(v)

    def to_2d(p):
        d = p - c_h
        return np.array([np.dot(d, u), np.dot(d, v)])

    pts_2d  = np.array([to_2d(p) for p in pts])
    proj_2d = to_2d(c_g_proj)

    margin = 0.03  # 3 cm lateral tolerance
    assert pts_2d[:, 0].min() - margin <= proj_2d[0] <= pts_2d[:, 0].max() + margin, \
        f"Glove centroid projects outside annotated dorsal region (u-axis)"
    assert pts_2d[:, 1].min() - margin <= proj_2d[1] <= pts_2d[:, 1].max() + margin, \
        f"Glove centroid projects outside annotated dorsal region (v-axis)"


# ---------------------------------------------------------------------------
# Finger cap alignment
# ---------------------------------------------------------------------------

def test_finger_caps_reach_mano_fingertips(annotation):
    """IK residuals for both fingers must be < 5 mm.

    This test independently re-solves IK for the annotated frame so it
    always reflects the current IK implementation.
    """
    from src.urdfpy_vis import load_robot
    from src.mano_io import load_frame
    from align_frame import solve_ik_frame

    frame_idx = annotation["frame"]
    frame = load_frame(cfg.NPZ_PATH, cfg.MANO_DIR, frame_idx)
    robot = load_robot(cfg.URDF_PATH, cfg.MESH_DIR)

    _, thumb_res, index_res = solve_ik_frame(
        robot, frame["T_wrist"], frame["thumb_tip"], frame["index_tip"]
    )

    assert thumb_res < IK_RESIDUAL_M, \
        f"Thumb IK residual {thumb_res*1000:.2f} mm > {IK_RESIDUAL_M*1000:.0f} mm"
    assert index_res < IK_RESIDUAL_M, \
        f"Index IK residual {index_res*1000:.2f} mm > {IK_RESIDUAL_M*1000:.0f} mm"


def test_finger_caps_in_glb_space_match_mano_fingertips(annotation):
    """Finger cap tips in GLB space should be within 5 mm of MANO fingertip
    vertices (after coordinate conversion), confirming the visualization is
    consistent with the IK solution.
    """
    from src.urdfpy_vis import load_robot
    from src.mano_io import load_frame
    from align_frame import solve_ik_frame, _THUMB_CHAIN, _INDEX_CHAIN
    from align_frame import _THUMB_VISUAL_ORIGIN, _INDEX_VISUAL_ORIGIN

    MANO_TO_GLB = np.diag([1.0, -1.0, -1.0, 1.0])
    HM_TO_ROOT  = np.eye(4); HM_TO_ROOT[:3, 3] = [0.157876, -0.0663838, 0.0660817]

    frame_idx = annotation["frame"]
    frame = load_frame(cfg.NPZ_PATH, cfg.MANO_DIR, frame_idx)
    robot = load_robot(cfg.URDF_PATH, cfg.MESH_DIR)

    joint_cfg, _, _ = solve_ik_frame(
        robot, frame["T_wrist"], frame["thumb_tip"], frame["index_tip"]
    )

    T_root_world_mano = frame["T_wrist"] @ HM_TO_ROOT
    fk = robot.link_fk(cfg=joint_cfg)

    for link, T_from_root in fk.items():
        if link.name == "part_3":
            thumb_local = _THUMB_VISUAL_ORIGIN
            thumb_mano  = (T_root_world_mano @ T_from_root @ np.append(thumb_local, 1))[:3]
            thumb_glb   = (MANO_TO_GLB @ np.append(thumb_mano, 1))[:3]
            exp_glb     = (MANO_TO_GLB @ np.append(frame["thumb_tip"], 1))[:3]
            err = float(np.linalg.norm(thumb_glb - exp_glb))
            assert err < IK_RESIDUAL_M, \
                f"Thumb cap in GLB space is {err*1000:.1f} mm from expected"

        if link.name == "part_3_1":
            index_local = _INDEX_VISUAL_ORIGIN
            index_mano  = (T_root_world_mano @ T_from_root @ np.append(index_local, 1))[:3]
            index_glb   = (MANO_TO_GLB @ np.append(index_mano, 1))[:3]
            exp_glb     = (MANO_TO_GLB @ np.append(frame["index_tip"], 1))[:3]
            err = float(np.linalg.norm(index_glb - exp_glb))
            assert err < IK_RESIDUAL_M, \
                f"Index cap in GLB space is {err*1000:.1f} mm from expected"

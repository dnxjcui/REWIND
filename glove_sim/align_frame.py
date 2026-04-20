"""Glove alignment: urdfpy FK + scipy TRF IK with joint limits, collision penalty,
and optional dynamic per-frame Kabsch base tracking."""

import sys
import numpy as np
import scipy.optimize
import trimesh
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from src.urdfpy_vis import ROOT_TO_HANDMOUNT_XYZ

_THUMB_CHAIN = ['revolute_1_0', 'revolute_2_0', 'revolute_3_0', 'revolute_4_0']
_INDEX_CHAIN = ['revolute_5_0', 'revolute_6_0', 'revolute_7_0', 'revolute_8_0', 'revolute_9_0']

# Dome-tip centroid in link-local frame, computed from STL vertex data.
_THUMB_VISUAL_ORIGIN = np.array([-0.00047,  0.0, -0.03095])
_INDEX_VISUAL_ORIGIN = np.array([-0.00259,  0.0, -0.03085])

# hand_mount → urdf root (pure translation, no rotation)
_HM_TO_ROOT = np.eye(4, dtype=np.float64)
_HM_TO_ROOT[:3, 3] = -ROOT_TO_HANDMOUNT_XYZ

_MANO_TO_GLB = np.diag([1.0, -1.0, -1.0, 1.0])

# ---------------------------------------------------------------------------
# Joint bounds [lower, upper] in radians.
# All URDF joints are 'continuous' with no declared limits.  We use ±π as
# hard bounds: wide enough not to restrict the solver, but preventing the
# unbounded drift that a purely unconstrained optimisation can produce.
# Directional clipping is handled by the soft collision penalty, not here.
# ---------------------------------------------------------------------------
_JOINT_BOUNDS = {j: (-np.pi, np.pi) for j in
                 ['revolute_1_0', 'revolute_2_0', 'revolute_3_0', 'revolute_4_0',
                  'revolute_5_0', 'revolute_6_0', 'revolute_7_0', 'revolute_8_0',
                  'revolute_9_0']}

# Chain links to check for hand-plane collision (excludes the tip link, which
# the IK target already controls).
_THUMB_COLLISION_LINKS = ['xl330_m077_t_1', 'xl_linkage_horn', 'part_2_1']
_INDEX_COLLISION_LINKS = ['part_6', 'xl_housing_1', 'part_2', 'xl_linkage_horn_1']

_COLLISION_MARGIN = 0.003   # penalise links within 3 mm of the dorsal plane
_COLLISION_WEIGHT = 8.0     # scale of soft-constraint residual term
# Weight rationale: IK residual is ~0.001–0.005 m.  A 3 mm penetration produces
# a penalty of 8 * 0.003 = 0.024 m — roughly 5–24× the IK error, enough to
# deter clipping without dominating the optimisation and breaking convergence.

# Multi-start candidates for the first joint of each chain.
# At revolute_5_0 = 0 (default), xl_platform_horn_1 sits 1.9 mm inside the
# dorsal plane.  At −π/2 it rises to +46 mm.  Sweeping avoids the local
# minimum that the zero warm-start traps the solver in.
_INDEX_Q0_CANDIDATES = [np.array([a, 0.0, 0.0, 0.0, 0.0])
                        for a in (-np.pi/2, -np.pi/4, 0.0, np.pi/4, np.pi/2)]
_THUMB_Q0_CANDIDATES = [np.array([a, 0.0, 0.0, 0.0])
                        for a in (-np.pi/2, -np.pi/4, 0.0, np.pi/4, np.pi/2)]


# ---------------------------------------------------------------------------
# Kabsch algorithm (SVD rigid registration, det = +1)
# ---------------------------------------------------------------------------

def _kabsch(P: np.ndarray, Q: np.ndarray):
    """Optimal rotation + translation mapping P → Q (both (N,3)).

    Returns R (3,3), t (3,) such that Q ≈ (R @ P.T).T + t.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    p_c = P.mean(axis=0)
    q_c = Q.mean(axis=0)
    H = (P - p_c).T @ (Q - q_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = q_c - R @ p_c
    return R, t


# ---------------------------------------------------------------------------
# Per-frame root-world transform (dynamic Kabsch or static fallback)
# ---------------------------------------------------------------------------

def _compute_T_root_world(T_wrist: np.ndarray, vertices: "np.ndarray | None") -> np.ndarray:
    """Return the 4×4 world transform of the URDF root link in MANO world space.

    If anchor vertex IDs and glove anchor points are calibrated, runs a fresh
    Kabsch solve every frame so the glove base tracks deforming skin vertices.
    Falls back to the static T_WRIST_TO_HM when the new annotation fields are
    absent (backwards compatibility).
    """
    if (vertices is not None
            and cfg.ANCHOR_VERTEX_IDS is not None
            and cfg.GLOVE_ANCHOR_PTS_HM is not None):
        skin_pts = vertices[cfg.ANCHOR_VERTEX_IDS]          # (3,3) MANO world
        R, t = _kabsch(cfg.GLOVE_ANCHOR_PTS_HM, skin_pts)
        T_hm_world = np.eye(4, dtype=np.float64)
        T_hm_world[:3, :3] = R
        T_hm_world[:3,  3] = t
        return T_hm_world @ _HM_TO_ROOT
    return T_wrist @ cfg.T_WRIST_TO_HM @ _HM_TO_ROOT


# ---------------------------------------------------------------------------
# IK solver (one finger chain)
# ---------------------------------------------------------------------------

def _ik_finger(robot, chain_joints, tip_link_name, visual_origin,
               target_in_root, collision_links,
               plane_normal_root, plane_point_root, q0=None,
               q0_candidates=None):
    """Solve IK for one finger chain.

    Hard constraints: physical joint bounds (TRF bounds parameter).
    Soft constraints: penalty when chain links approach the dorsal hand plane.

    When q0 is None and q0_candidates is provided, the solver is run from each
    candidate start and the result with the lowest IK residual is returned.
    When q0 is not None (warm-start from previous frame), q0_candidates is
    ignored and a single solve is performed from q0.

    Returns (joint_cfg_dict, residual_metres).
    """
    lower = np.array([_JOINT_BOUNDS[j][0] for j in chain_joints])
    upper = np.array([_JOINT_BOUNDS[j][1] for j in chain_joints])

    n_penalty = len(collision_links)

    def residual(q_vec):
        fk = robot.link_fk(cfg=dict(zip(chain_joints, q_vec)))

        ik_err = np.full(3, 1e6)
        link_pos = {}
        for link, T in fk.items():
            if link.name == tip_link_name:
                ik_err = T[:3, :3] @ visual_origin + T[:3, 3] - target_in_root
            elif link.name in collision_links:
                link_pos[link.name] = T[:3, 3]

        # Soft collision penalty: one term per tracked link
        penalty = np.zeros(n_penalty)
        if plane_normal_root is not None:
            for i, lname in enumerate(collision_links):
                pos = link_pos.get(lname)
                if pos is None:
                    continue
                dist = float(np.dot(pos - plane_point_root, plane_normal_root))
                if dist < _COLLISION_MARGIN:
                    penalty[i] = _COLLISION_WEIGHT * (_COLLISION_MARGIN - dist)

        return np.concatenate([ik_err, penalty])

    def _run_once(start):
        start = np.clip(start, lower, upper)
        res = scipy.optimize.least_squares(
            residual, start,
            bounds=(lower, upper),
            method='trf',
            ftol=cfg.IK_TOL,
            max_nfev=cfg.IK_MAX_NFEV,
        )
        ik_residual = float(np.linalg.norm(res.fun[:3]))
        return res.x, ik_residual

    if q0 is not None:
        # Warm-start: single solve from previous-frame solution
        best_x, best_res = _run_once(q0)
    else:
        # No warm-start: try each candidate, keep best IK residual
        starts = q0_candidates if q0_candidates else [np.zeros(len(chain_joints))]
        best_x, best_res = None, np.inf
        for start in starts:
            x, r = _run_once(start)
            if r < best_res:
                best_x, best_res = x, r

    return dict(zip(chain_joints, best_x)), best_res


# ---------------------------------------------------------------------------
# Per-frame solve
# ---------------------------------------------------------------------------

def solve_ik_frame(robot, T_wrist, thumb_tip_world, index_tip_world,
                   prev_q_thumb=None, prev_q_index=None, vertices=None):
    """Solve IK for both fingers in one frame.

    Parameters
    ----------
    robot           : urdfpy.URDF loaded via load_robot()
    T_wrist         : (4,4) wrist world transform, Y-down frame
    thumb_tip_world : (3,) thumb fingertip world position, Y-down frame
    index_tip_world : (3,) index fingertip world position, Y-down frame
    prev_q_thumb    : (4,) warm-start angles for thumb chain, or None
    prev_q_index    : (5,) warm-start angles for index chain, or None
    vertices        : (778,3) MANO mesh vertices for this frame, or None.
                      When provided and anchor calibration exists, the glove
                      base is placed via dynamic per-frame Kabsch instead of
                      the static T_WRIST_TO_HM.

    Returns
    -------
    joint_cfg      : dict[str, float] — all 9 joint angles in radians
    thumb_residual : float — Euclidean distance in metres
    index_residual : float — Euclidean distance in metres
    """
    # Apply calibrated fingertip offsets (wrist-local → world)
    R_w = T_wrist[:3, :3]
    thumb_tip_world = thumb_tip_world + R_w @ cfg.THUMB_TIP_OFFSET
    index_tip_world = index_tip_world + R_w @ cfg.INDEX_TIP_OFFSET

    # Glove root world pose (dynamic Kabsch if available, else static offset)
    T_root_world = _compute_T_root_world(T_wrist, vertices)
    inv_T = np.linalg.inv(T_root_world)

    # Transform dorsal plane from GLB → MANO world → root-local
    # (used for the soft collision penalty inside _ik_finger)
    centroid_mano = (_MANO_TO_GLB @ np.append(cfg.HAND_DORSAL_CENTROID_GLB, 1.0))[:3]
    normal_mano   = _MANO_TO_GLB[:3, :3] @ cfg.HAND_DORSAL_NORMAL_GLB
    plane_centroid_root = (inv_T @ np.append(centroid_mano, 1.0))[:3]
    plane_normal_root   = inv_T[:3, :3] @ normal_mano
    # Ensure normal points away from the hand (toward the glove side).
    # The URDF root is above the dorsal plane, so its signed distance must be positive.
    root_dist = float(np.dot(-plane_centroid_root, plane_normal_root))  # root is at origin
    if root_dist < 0:
        plane_normal_root = -plane_normal_root

    # Note: MANO thumb (v745) → index chain (part_3_1); MANO index (v317) → thumb chain (part_3).
    # The glove's physical thumb cap is on the part_3_1/INDEX_CHAIN side and vice versa.
    thumb_target = (inv_T @ np.append(thumb_tip_world, 1.0))[:3]
    index_target = (inv_T @ np.append(index_tip_world, 1.0))[:3]

    thumb_cfg, thumb_res = _ik_finger(
        robot, _THUMB_CHAIN, 'part_3',   _THUMB_VISUAL_ORIGIN, index_target,
        _THUMB_COLLISION_LINKS, plane_normal_root, plane_centroid_root,
        q0=prev_q_thumb, q0_candidates=_THUMB_Q0_CANDIDATES)
    index_cfg, index_res = _ik_finger(
        robot, _INDEX_CHAIN, 'part_3_1', _INDEX_VISUAL_ORIGIN, thumb_target,
        _INDEX_COLLISION_LINKS, plane_normal_root, plane_centroid_root,
        q0=prev_q_index, q0_candidates=_INDEX_Q0_CANDIDATES)

    return {**thumb_cfg, **index_cfg}, thumb_res, index_res


def _assert_frame(joint_cfg, thumb_res, index_res, frame_idx):
    assert thumb_res < cfg.IK_RESIDUAL_THRESHOLD, (
        f"Frame {frame_idx} thumb IK: {thumb_res*1000:.1f}mm "
        f"> {cfg.IK_RESIDUAL_THRESHOLD*1000:.0f}mm tolerance"
    )
    assert index_res < cfg.IK_RESIDUAL_THRESHOLD, (
        f"Frame {frame_idx} index IK: {index_res*1000:.1f}mm "
        f"> {cfg.IK_RESIDUAL_THRESHOLD*1000:.0f}mm tolerance"
    )
    for name, angle in joint_cfg.items():
        assert -np.pi <= angle <= np.pi, (
            f"Frame {frame_idx} joint {name}: {np.degrees(angle):.1f}° "
            f"out of [-180°, 180°]"
        )


def _export_frame(robot, T_wrist, joint_cfg, frame_idx, vertices=None):
    """Export combined hand+glove GLB for one frame."""
    from src.urdfpy_vis import get_glove_scene

    # Use the same Kabsch-or-static logic as solve_ik_frame so the GLB matches.
    T_root_world = _compute_T_root_world(T_wrist, vertices)
    T_root_to_hm = np.eye(4)
    T_root_to_hm[:3, 3] = ROOT_TO_HANDMOUNT_XYZ
    T_hand_mount = T_root_world @ T_root_to_hm
    glove_scene = get_glove_scene(robot, joint_cfg, T_hand_mount)
    meshes = list(glove_scene.geometry.values())

    hand_glb = cfg.GLB_DIR / f"{frame_idx:06d}_hands.glb"
    if hand_glb.exists():
        hand_scene = trimesh.load(str(hand_glb), force="scene")
        meshes.extend(hand_scene.geometry.values())
    else:
        print(f"[WARNING] Hand GLB not found: {hand_glb}")

    out_path = cfg.ALIGNED_DIR / f"{frame_idx:06d}_aligned.glb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Scene(meshes).export(str(out_path))
    return out_path


def _total_frames() -> int:
    data = np.load(str(cfg.NPZ_PATH), allow_pickle=False)
    return int(data["trans"].shape[1])


def main():
    import argparse
    from src.urdfpy_vis import load_robot
    from src.mano_io import load_frame

    parser = argparse.ArgumentParser(
        description="Align glove URDF onto DynHaMR hand frames and export GLBs."
    )
    parser.add_argument(
        "--frames", nargs="+", type=int, default=None,
        help="Explicit list of frame indices to process. Mutually exclusive with --frames-test.",
    )
    parser.add_argument(
        "--frames-test", action="store_true",
        help="Quick test: run 10 evenly-spaced frames with strict assertions (no skipping).",
    )
    args = parser.parse_args()

    if args.frames is not None and args.frames_test:
        parser.error("--frames and --frames-test are mutually exclusive.")

    n_total = _total_frames()

    if args.frames_test:
        indices = [int(round(i * (n_total - 1) / 9)) for i in range(10)]
        strict  = True
        print(f"Test mode: {len(indices)} evenly-spaced frames out of {n_total} total.")
    elif args.frames is not None:
        indices = args.frames
        strict  = True
        print(f"Explicit frames: {indices}")
    else:
        indices = list(range(n_total))
        strict  = False
        print(f"Full sequence: {n_total} frames — failures will be skipped.")

    robot = load_robot(cfg.URDF_PATH, cfg.MESH_DIR)
    print(f"URDF loaded: {len(robot.links)} links, {len(robot.actuated_joints)} actuated joints")

    prev_q_thumb = None
    prev_q_index = None
    out_paths    = []
    failed       = []

    for frame_idx in indices:
        try:
            frame = load_frame(cfg.NPZ_PATH, cfg.MANO_DIR, frame_idx)

            joint_cfg, thumb_res, index_res = solve_ik_frame(
                robot, frame["T_wrist"], frame["thumb_tip"], frame["index_tip"],
                prev_q_thumb=prev_q_thumb, prev_q_index=prev_q_index,
                vertices=frame.get("vertices"),
            )

            if strict:
                _assert_frame(joint_cfg, thumb_res, index_res, frame_idx)
                print(f"  [{frame_idx:05d}] thumb {thumb_res*1000:.1f}mm  "
                      f"index {index_res*1000:.1f}mm  OK")
            else:
                _assert_frame(joint_cfg, thumb_res, index_res, frame_idx)
                if frame_idx % 100 == 0:
                    print(f"  [{frame_idx:05d}] thumb {thumb_res*1000:.1f}mm  "
                          f"index {index_res*1000:.1f}mm")

            out_path = _export_frame(robot, frame["T_wrist"], joint_cfg, frame_idx,
                                     vertices=frame.get("vertices"))
            out_paths.append(out_path)

            prev_q_thumb = np.array([joint_cfg[j] for j in _THUMB_CHAIN])
            prev_q_index = np.array([joint_cfg[j] for j in _INDEX_CHAIN])

        except Exception as exc:
            if strict:
                raise
            print(f"  [SKIP] Frame {frame_idx}: {exc}")
            failed.append(frame_idx)
            prev_q_thumb = None
            prev_q_index = None

    print(f"\n{'='*60}")
    print(f"Done. {len(out_paths)} frames exported, {len(failed)} skipped.")
    if failed:
        print(f"Failed frames: {failed}")
    if out_paths:
        print(f"\nFirst GLB: {out_paths[0]}")
        if strict:
            print(f"\n*** STOP. Review the GLBs above visually before processing the full sequence. ***")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

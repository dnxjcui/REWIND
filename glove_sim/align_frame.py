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

# ---------------------------------------------------------------------------
# Virtual proxy points: the bottommost vertex of each tracked link's mesh
# in the link's own local coordinate frame (found by minimising world-space Y
# at the zero configuration, where Y-down MANO world means min Y = closest to
# the dorsal surface).  Using the mesh hull rather than the link origin avoids
# the "impossible penalty" problem where the joint axis sits structurally below
# the dorsal plane regardless of joint angle.
# ---------------------------------------------------------------------------
_COLLISION_PROXIES = {
    'xl330_m077_t_1':    np.array([ 0.00900, -0.02450, -0.00320]),
    'xl_linkage_horn':   np.array([ 0.00300, -0.00297,  0.00716]),
    'part_2_1':          np.array([-0.00496,  0.00250,  0.00061]),
    'part_6':            np.array([-0.00832, -0.00090, -0.05431]),
    'xl_housing_1':      np.array([ 0.01215, -0.02665,  0.00000]),
    'part_2':            np.array([-0.00435,  0.00250,  0.00247]),
    'xl_linkage_horn_1': np.array([ 0.01550, -0.00625,  0.00434]),
}

_COLLISION_MARGIN = 0.003   # penalise proxy points within 3 mm of the dorsal plane
_COLLISION_WEIGHT = 2000.0  # quadratic scale factor
# Quadratic penalty: WEIGHT * (MARGIN - dist)^2
# At the boundary (dist = MARGIN): penalty = 0
# At contact (dist = 0):          penalty = 2000 * (0.003)^2 = 0.018 m  ≈ IK residual
# At 3 mm deep (dist = -0.003):   penalty = 2000 * (0.006)^2 = 0.072 m  ≈ 14× IK residual
# This creates a "brick-wall" effect: barely touching costs roughly the same as
# the IK error; deep clipping costs an order of magnitude more.

_COLLISION_LATERAL_CUTOFF = 0.040  # 40 mm from plane centroid — suppress penalty for
# links that have wrapped around the sides of the hand (off the dorsal surface).

# ---------------------------------------------------------------------------
# Per-joint dorsal-direction map (measured empirically: apply +0.8 rad to each
# joint individually and observe whether the linkage moves toward or away from
# the dorsal hand plane).
#
#   "away"    → positive angle moves the linkage AWAY from the hand (safe)
#   "towards" → positive angle moves the linkage INTO the hand (clips); to route
#               safely, use a NEGATIVE angle for this joint
#   "na"      → twist axis — neither toward nor away; swept separately at ±π/2
# ---------------------------------------------------------------------------
_JOINT_DORSAL_DIRECTION = {
    'revolute_1_0': 'na',
    'revolute_2_0': 'towards',
    'revolute_3_0': 'away',
    'revolute_4_0': 'away',
    'revolute_5_0': 'away',
    'revolute_6_0': 'towards',
    'revolute_7_0': 'away',
    'revolute_8_0': 'towards',
    'revolute_9_0': 'away',
}


def _build_candidates(chain_joints):
    """Generate sign-aware multi-start seeds for one finger chain.

    For each "pre-bent" candidate the joints are set so every active joint
    moves in its measured safe direction (+angle if "away", -angle if "towards").
    This ensures seeds route the linkage over the dorsal surface rather than
    into it regardless of individual joint axis orientation.
    """
    n = len(chain_joints)
    # Safe-direction sign for each joint in this chain
    signs = np.array([
        +1.0 if _JOINT_DORSAL_DIRECTION[j] == 'away' else
        -1.0 if _JOINT_DORSAL_DIRECTION[j] == 'towards' else
         0.0
        for j in chain_joints
    ])
    active = [i for i, s in enumerate(signs) if s != 0.0]
    twist  = [i for i, s in enumerate(signs) if s == 0.0]

    Q, H = np.pi / 4, np.pi / 2
    cands = []

    # Progressive pre-bending at π/4 — add one joint at a time in safe direction.
    # These are the primary "route-over-the-hand" seeds.
    for k in range(1, len(active) + 1):
        q = np.zeros(n)
        for i in active[:k]:
            q[i] = signs[i] * Q
        cands.append(q.copy())

    # Same pattern but at π/2 — stronger deflection for difficult frames.
    for k in range(1, len(active) + 1):
        q = np.zeros(n)
        for i in active[:k]:
            q[i] = signs[i] * H
        cands.append(q.copy())

    # Sweep twist (na) joints at ±π/2 while active joints are at their π/4 safe value.
    q_base = signs * Q  # active joints at Q, na joints at 0
    for i in twist:
        for tv in (H, -H):
            q = q_base.copy()
            q[i] = tv
            cands.append(q)

    # Zero start — safe fallback when the target is close to rest configuration.
    cands.append(np.zeros(n))

    # Opposite-direction sweep — cover frames where the physically correct
    # solution happens to require joints in their "towards" direction.
    cands.append(-signs * Q)
    cands.append(-signs * H)

    return cands


_INDEX_Q0_CANDIDATES = _build_candidates(_INDEX_CHAIN)
_THUMB_Q0_CANDIDATES = _build_candidates(_THUMB_CHAIN)

# Hover offset: push the IK target outward from the fingertip vertex along the
# skin normal so the cap dome sits on the skin instead of sinking into it.
#
# IMPORTANT: this offset interacts with the calibrated THUMB_TIP_OFFSET /
# INDEX_TIP_OFFSET in config.py.  Those offsets were measured by annotate_fingertips.py
# with _HOVER_DIST = 0 (target = exact MANO vertex).  Setting a non-zero hover
# without re-running the fingertip annotation will over-shoot the target and
# inflate IK residuals — breaking frames that otherwise converge cleanly.
#
# Workflow to use hover correctly:
#   1. Set _HOVER_DIST to the desired value.
#   2. Re-run: python glove_sim/tools/annotate_fingertips.py --frame 300
#   3. The new THUMB/INDEX_TIP_OFFSET will be calibrated relative to the hovered target.
_HOVER_DIST = 0.0  # disabled until re-calibrated — set to 0.010 after re-annotating


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
        link_T = {}
        for link, T in fk.items():
            if link.name == tip_link_name:
                ik_err = T[:3, :3] @ visual_origin + T[:3, 3] - target_in_root
            elif link.name in collision_links:
                link_T[link.name] = T

        # Soft quadratic collision penalty using mesh proxy points.
        # Each proxy offset (link-local) is rotated/translated to root frame via FK,
        # giving the physical mesh bottom rather than the abstract joint origin.
        # Penalty is suppressed for links that have moved laterally off the dorsal
        # surface (more than LATERAL_CUTOFF from the plane centroid).
        penalty = np.zeros(n_penalty)
        if plane_normal_root is not None:
            for i, lname in enumerate(collision_links):
                T = link_T.get(lname)
                if T is None:
                    continue
                proxy_local = _COLLISION_PROXIES.get(lname, np.zeros(3))
                proxy_pos = T[:3, :3] @ proxy_local + T[:3, 3]
                v = proxy_pos - plane_point_root
                dist = float(np.dot(v, plane_normal_root))
                if dist >= _COLLISION_MARGIN:
                    continue
                # Lateral distance from centroid (in the plane) — suppress if off the
                # dorsal surface so we don't penalise linkages wrapping the hand sides.
                lateral = float(np.linalg.norm(v - dist * plane_normal_root))
                if lateral > _COLLISION_LATERAL_CUTOFF:
                    continue
                penetration = _COLLISION_MARGIN - dist
                penalty[i] = _COLLISION_WEIGHT * (penetration ** 2)

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

    def _is_clipping(q_vec):
        """Return True if any proxy point penetrates the hand plane (dist < 0)."""
        if plane_normal_root is None:
            return False
        fk = robot.link_fk(cfg=dict(zip(chain_joints, q_vec)))
        for link, T in fk.items():
            if link.name not in collision_links:
                continue
            proxy_local = _COLLISION_PROXIES.get(link.name, np.zeros(3))
            pos = T[:3, :3] @ proxy_local + T[:3, 3]
            v = pos - plane_point_root
            dist = float(np.dot(v, plane_normal_root))
            if dist < 0:
                lateral = float(np.linalg.norm(v - dist * plane_normal_root))
                if lateral <= _COLLISION_LATERAL_CUTOFF:
                    return True
        return False

    def _best_of(results):
        """Given [(x, residual), ...], return the best.

        Priority: non-clipping with lowest IK residual first; if all clip,
        return the lowest-residual result regardless.
        """
        non_clip = [(x, r) for x, r in results if not _is_clipping(x)]
        pool = non_clip if non_clip else results
        return min(pool, key=lambda t: t[1])

    starts = list(q0_candidates) if q0_candidates else [np.zeros(len(chain_joints))]

    if q0 is not None:
        # Fast path: warm-start from previous frame.
        best_x, best_res = _run_once(q0)
        if not _is_clipping(best_x):
            # Non-clipping warm-start — accept without trying candidates.
            return dict(zip(chain_joints, best_x)), best_res
        # Warm-start clips: run all candidates and take the best non-clipping
        # result (or lowest residual if everything clips).
        candidates_results = [(best_x, best_res)]
        for start in starts:
            candidates_results.append(_run_once(start))
        best_x, best_res = _best_of(candidates_results)
    else:
        # No warm-start: try every candidate.
        candidates_results = [_run_once(s) for s in starts]
        best_x, best_res = _best_of(candidates_results)

    return dict(zip(chain_joints, best_x)), best_res


# ---------------------------------------------------------------------------
# Per-frame solve
# ---------------------------------------------------------------------------

def solve_ik_frame(robot, T_wrist, thumb_tip_world, index_tip_world,
                   prev_q_thumb=None, prev_q_index=None, vertices=None,
                   vertex_normals=None):
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
    vertex_normals  : (778,3) outward unit normals at each MANO vertex, or None.
                      When provided, the IK target is pushed _HOVER_DIST (10 mm)
                      along the fingertip normal so the cap dome sits on the skin
                      rather than sinking into it.

    Returns
    -------
    joint_cfg      : dict[str, float] — all 9 joint angles in radians
    thumb_residual : float — Euclidean distance in metres
    index_residual : float — Euclidean distance in metres
    """
    # Hover offset: push each fingertip target outward along the skin normal
    # before applying the calibrated correction, so the cap dome centre floats
    # _HOVER_DIST above the skin rather than coinciding with the vertex.
    if vertex_normals is not None:
        thumb_tip_world = thumb_tip_world + _HOVER_DIST * vertex_normals[745]
        index_tip_world = index_tip_world + _HOVER_DIST * vertex_normals[317]

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

    out_path = cfg.GLB_FRAMES_DIR / f"{frame_idx:06d}_aligned.glb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Scene(meshes).export(str(out_path))
    return out_path


def export_frame_urdf(robot, joint_cfg, T_root_world, frame_idx) -> Path:
    """Write a snapshot URDF for a single solved frame.

    The output URDF is a self-contained copy of the original with:
      - All revolute joints converted to fixed joints at their solved angles.
        The new joint origin encodes the relative FK transform at that angle,
        computed as inv(T_parent) @ T_child from urdfpy's link_fk result.
      - A 'world' root link and fixed 'world_to_glove' joint that embeds
        T_root_world (MANO world space) as the absolute pose.
      - Mesh file paths rewritten to absolute paths so the file is portable.

    Written to cfg.URDF_FRAMES_DIR / rewind_glove_{frame_idx:06d}.urdf.
    """
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation as Rot

    # FK at solved config: {link: T_from_root}
    fk = robot.link_fk(cfg=joint_cfg)
    link_T = {link.name: T for link, T in fk.items()}

    # Joint → (parent_link_name, child_link_name) mapping
    joint_parents = {j.name: (j.parent, j.child) for j in robot.joints}

    def mat_to_xyzrpy(T):
        xyz = T[:3, 3]
        rpy = Rot.from_matrix(T[:3, :3]).as_euler('xyz')
        return (f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}",
                f"{rpy[0]:.8f} {rpy[1]:.8f} {rpy[2]:.8f}")

    tree = ET.parse(str(cfg.URDF_PATH))
    root_elem = tree.getroot()
    root_elem.set('name', f'rewind_glove_frame_{frame_idx:06d}')

    # Rewrite mesh paths to absolute so the file is usable from any location
    mesh_dir_abs = str(cfg.MESH_DIR.resolve())
    for mesh_elem in root_elem.iter('mesh'):
        fname = mesh_elem.get('filename', '')
        if fname.startswith('../meshes/'):
            abs_fname = mesh_dir_abs + '/' + fname[len('../meshes/'):]
            mesh_elem.set('filename', abs_fname)

    # Replace all revolute/continuous joints with fixed joints at solved angles
    for joint_elem in root_elem.findall('joint'):
        jtype = joint_elem.get('type')
        jname = joint_elem.get('name')
        if jtype not in ('revolute', 'continuous'):
            continue
        parent_name, child_name = joint_parents[jname]
        T_parent = link_T.get(parent_name, np.eye(4))
        T_child  = link_T.get(child_name,  np.eye(4))
        T_rel    = np.linalg.inv(T_parent) @ T_child
        xyz_str, rpy_str = mat_to_xyzrpy(T_rel)

        joint_elem.set('type', 'fixed')
        origin_elem = joint_elem.find('origin')
        if origin_elem is None:
            origin_elem = ET.SubElement(joint_elem, 'origin')
        origin_elem.set('xyz', xyz_str)
        origin_elem.set('rpy', rpy_str)
        for tag in ('axis', 'limit', 'dynamics', 'safety_controller'):
            for sub in joint_elem.findall(tag):
                joint_elem.remove(sub)

    # Prepend a world link and fixed joint anchoring the glove in MANO world space
    base_link_name = robot.base_link.name
    world_link_elem = ET.Element('link')
    world_link_elem.set('name', 'world')
    root_elem.insert(0, world_link_elem)

    xyz_str, rpy_str = mat_to_xyzrpy(T_root_world)
    world_joint_elem = ET.Element('joint')
    world_joint_elem.set('name', 'world_to_glove')
    world_joint_elem.set('type', 'fixed')
    ET.SubElement(world_joint_elem, 'parent').set('link', 'world')
    ET.SubElement(world_joint_elem, 'child').set('link', base_link_name)
    origin_elem = ET.SubElement(world_joint_elem, 'origin')
    origin_elem.set('xyz', xyz_str)
    origin_elem.set('rpy', rpy_str)
    root_elem.append(world_joint_elem)

    out_dir = cfg.URDF_FRAMES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'rewind_glove_{frame_idx:06d}.urdf'
    tree.write(str(out_path), encoding='unicode', xml_declaration=True)
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
        help="Quick test: 10 evenly-spaced frames (from SEQUENCE_START) with strict assertions.",
    )
    parser.add_argument(
        "--urdf", action="store_true",
        help="Also export a per-frame snapshot URDF to outputs/aligned/urdf_frames/.",
    )
    args = parser.parse_args()

    if args.frames is not None and args.frames_test:
        parser.error("--frames and --frames-test are mutually exclusive.")

    n_total = _total_frames()
    start   = cfg.SEQUENCE_START  # default: skip pre-task frames 0-17

    if args.frames_test:
        span = n_total - 1 - start
        indices = [start + int(round(i * span / 9)) for i in range(10)]
        strict  = True
        print(f"Test mode: {len(indices)} evenly-spaced frames [{indices[0]}…{indices[-1]}].")
    elif args.frames is not None:
        indices = args.frames
        strict  = True
        print(f"Explicit frames: {indices}")
    else:
        indices = list(range(start, n_total))
        strict  = False
        print(f"Full sequence: frames {start}–{n_total-1} ({len(indices)} frames) — failures skipped.")

    robot = load_robot(cfg.URDF_PATH, cfg.MESH_DIR)
    print(f"URDF loaded: {len(robot.links)} links, {len(robot.actuated_joints)} actuated joints")

    prev_q_thumb = None
    prev_q_index = None
    out_paths    = []
    urdf_paths   = []
    failed       = []

    for frame_idx in indices:
        try:
            frame = load_frame(cfg.NPZ_PATH, cfg.MANO_DIR, frame_idx)

            joint_cfg, thumb_res, index_res = solve_ik_frame(
                robot, frame["T_wrist"], frame["thumb_tip"], frame["index_tip"],
                prev_q_thumb=prev_q_thumb, prev_q_index=prev_q_index,
                vertices=frame.get("vertices"),
                vertex_normals=frame.get("vertex_normals"),
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

            T_root_world = _compute_T_root_world(frame["T_wrist"], frame.get("vertices"))

            out_path = _export_frame(robot, frame["T_wrist"], joint_cfg, frame_idx,
                                     vertices=frame.get("vertices"))
            out_paths.append(out_path)

            if args.urdf:
                urdf_path = export_frame_urdf(robot, joint_cfg, T_root_world, frame_idx)
                urdf_paths.append(urdf_path)

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
    print(f"Done. {len(out_paths)} GLBs exported to {cfg.GLB_FRAMES_DIR}")
    if args.urdf:
        print(f"      {len(urdf_paths)} URDFs exported to {cfg.URDF_FRAMES_DIR}")
    print(f"      {len(failed)} frames skipped.")
    if failed:
        print(f"Failed frames: {failed}")
    if out_paths and strict:
        print(f"\n*** STOP. Review GLBs visually before processing the full sequence. ***")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

"""Collision diagnostic for glove_sim.

Runs IK at a single frame over a sweep of collision weights, then:
  1. Reports IK residuals and per-link signed distances to the dorsal plane.
  2. Exports a temporary GLB and tests actual mesh-on-mesh intersections using
     trimesh's CollisionManager.

Usage:
    python glove_sim/tools/diagnose_collision.py --frame 300
    python glove_sim/tools/diagnose_collision.py --frame 300 --weights 0 1 5 8 20 50
"""

import sys
import argparse
import tempfile
import numpy as np
import trimesh
import scipy.optimize
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.urdfpy_vis import load_robot, get_glove_scene, ROOT_TO_HANDMOUNT_XYZ
from src.mano_io import load_frame

# Import align_frame items without triggering main()
from align_frame import (
    _THUMB_CHAIN, _INDEX_CHAIN,
    _THUMB_VISUAL_ORIGIN, _INDEX_VISUAL_ORIGIN,
    _THUMB_COLLISION_LINKS, _INDEX_COLLISION_LINKS,
    _THUMB_Q0_CANDIDATES, _INDEX_Q0_CANDIDATES,
    _JOINT_BOUNDS, _HM_TO_ROOT, _MANO_TO_GLB,
    _COLLISION_MARGIN, _COLLISION_WEIGHT,
    _compute_T_root_world,
)


# ---------------------------------------------------------------------------
# Per-link signed-distance reporter
# ---------------------------------------------------------------------------

def _link_distances(robot, chain_joints, joint_cfg, collision_links,
                    plane_normal_root, plane_centroid_root):
    """Return {link_name: signed_dist_mm} for tracked links."""
    fk = robot.link_fk(cfg=joint_cfg)
    dists = {}
    for link, T in fk.items():
        if link.name in collision_links:
            pos = T[:3, 3]
            d = float(np.dot(pos - plane_centroid_root, plane_normal_root)) * 1000
            dists[link.name] = d
    return dists


# ---------------------------------------------------------------------------
# Custom single-weight IK (bypasses the module-level constant)
# ---------------------------------------------------------------------------

def _run_ik(robot, chain_joints, tip_link_name, visual_origin,
            target_in_root, collision_links,
            plane_normal_root, plane_centroid_root,
            collision_weight, q0_candidates):
    lower = np.array([_JOINT_BOUNDS[j][0] for j in chain_joints])
    upper = np.array([_JOINT_BOUNDS[j][1] for j in chain_joints])
    n_pen = len(collision_links)

    def residual(q_vec):
        fk = robot.link_fk(cfg=dict(zip(chain_joints, q_vec)))
        ik_err = np.full(3, 1e6)
        link_pos = {}
        for link, T in fk.items():
            if link.name == tip_link_name:
                ik_err = T[:3, :3] @ visual_origin + T[:3, 3] - target_in_root
            elif link.name in collision_links:
                link_pos[link.name] = T[:3, 3]
        penalty = np.zeros(n_pen)
        if plane_normal_root is not None and collision_weight > 0:
            for i, lname in enumerate(collision_links):
                pos = link_pos.get(lname)
                if pos is None:
                    continue
                d = float(np.dot(pos - plane_centroid_root, plane_normal_root))
                if d < _COLLISION_MARGIN:
                    penalty[i] = collision_weight * (_COLLISION_MARGIN - d)
        return np.concatenate([ik_err, penalty])

    best_x, best_res = None, np.inf
    for start in q0_candidates:
        start = np.clip(start, lower, upper)
        r = scipy.optimize.least_squares(
            residual, start, bounds=(lower, upper), method='trf',
            ftol=cfg.IK_TOL, max_nfev=cfg.IK_MAX_NFEV)
        ik_r = float(np.linalg.norm(r.fun[:3]))
        if ik_r < best_res:
            best_x, best_res = r.x, ik_r
    return dict(zip(chain_joints, best_x)), best_res


# ---------------------------------------------------------------------------
# Mesh-level collision check via trimesh CollisionManager
# ---------------------------------------------------------------------------

def _check_mesh_collisions(robot, joint_cfg, T_root_world, hand_glb_path):
    """Return list of (glove_mesh_name, collides_with_hand: bool)."""
    from src.urdfpy_vis import get_glove_scene

    T_root_to_hm = np.eye(4)
    T_root_to_hm[:3, 3] = ROOT_TO_HANDMOUNT_XYZ
    T_hand_mount = T_root_world @ T_root_to_hm
    glove_scene = get_glove_scene(robot, joint_cfg, T_hand_mount)

    hand_scene = trimesh.load(str(hand_glb_path), force="scene")
    # Select right-hand sub-mesh (largest by vertex count among 778-vertex meshes)
    hand_meshes = [m for m in hand_scene.geometry.values()
                   if hasattr(m, "vertices") and len(m.vertices) == 778]
    if not hand_meshes:
        hand_meshes = list(hand_scene.geometry.values())
    # Pick the one whose centroid is closest to the wrist (already in GLB coords)
    hand_mesh = hand_meshes[0] if len(hand_meshes) == 1 else max(
        hand_meshes, key=lambda m: float(np.linalg.norm(m.centroid)))

    manager = trimesh.collision.CollisionManager()
    manager.add_object("hand", hand_mesh)

    results = []
    for name, mesh in glove_scene.geometry.items():
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        # Glove scene is in MANO world; hand GLB is in GLB (Y-up) space.
        # Apply MANO→GLB to bring glove into the same space as the hand mesh.
        glove_glb = mesh.copy()
        glove_glb.apply_transform(_MANO_TO_GLB)
        collides, _ = manager.in_collision_single(glove_glb)
        results.append((name, collides))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Glove collision diagnostic")
    parser.add_argument("--frame", type=int, default=300,
                        help="Frame index to test (default: 300)")
    parser.add_argument("--weights", nargs="+", type=float,
                        default=[0.0, 1.0, 5.0, 8.0, 20.0, 50.0],
                        help="Collision weights to sweep")
    args = parser.parse_args()

    frame_idx = args.frame
    print(f"Loading frame {frame_idx} ...")
    frame = load_frame(cfg.NPZ_PATH, cfg.MANO_DIR, frame_idx)
    robot = load_robot(cfg.URDF_PATH, cfg.MESH_DIR)

    T_wrist   = frame["T_wrist"]
    vertices  = frame.get("vertices")
    T_root_world = _compute_T_root_world(T_wrist, vertices)
    inv_T = np.linalg.inv(T_root_world)

    # Fingertip offsets
    R_w = T_wrist[:3, :3]
    thumb_tip = frame["thumb_tip"] + R_w @ cfg.THUMB_TIP_OFFSET
    index_tip = frame["index_tip"] + R_w @ cfg.INDEX_TIP_OFFSET

    # Plane in root-local frame
    centroid_mano = (_MANO_TO_GLB @ np.append(cfg.HAND_DORSAL_CENTROID_GLB, 1.0))[:3]
    normal_mano   = _MANO_TO_GLB[:3, :3] @ cfg.HAND_DORSAL_NORMAL_GLB
    plane_centroid_root = (inv_T @ np.append(centroid_mano, 1.0))[:3]
    plane_normal_root   = inv_T[:3, :3] @ normal_mano
    if float(np.dot(-plane_centroid_root, plane_normal_root)) < 0:
        plane_normal_root = -plane_normal_root

    # Targets (swapped: MANO thumb → index chain, MANO index → thumb chain)
    thumb_target = (inv_T @ np.append(thumb_tip, 1.0))[:3]
    index_target = (inv_T @ np.append(index_tip, 1.0))[:3]

    hand_glb = cfg.GLB_DIR / f"{frame_idx:06d}_hands.glb"

    col_w = 18
    print(f"\n{'Weight':>8}  {'Thumb res':>10}  {'Index res':>10}  "
          f"{'Part6 dist':>10}  {'XL horn dist':>12}  {'Mesh collide':>12}")
    print("-" * 80)

    for w in args.weights:
        thumb_cfg, thumb_res = _run_ik(
            robot, _THUMB_CHAIN, 'part_3', _THUMB_VISUAL_ORIGIN, index_target,
            _THUMB_COLLISION_LINKS, plane_normal_root, plane_centroid_root,
            w, _THUMB_Q0_CANDIDATES)
        index_cfg, index_res = _run_ik(
            robot, _INDEX_CHAIN, 'part_3_1', _INDEX_VISUAL_ORIGIN, thumb_target,
            _INDEX_COLLISION_LINKS, plane_normal_root, plane_centroid_root,
            w, _INDEX_Q0_CANDIDATES)

        joint_cfg = {**thumb_cfg, **index_cfg}

        # Per-link distances
        t_dists = _link_distances(robot, _THUMB_CHAIN, joint_cfg,
                                  _THUMB_COLLISION_LINKS,
                                  plane_normal_root, plane_centroid_root)
        i_dists = _link_distances(robot, _INDEX_CHAIN, joint_cfg,
                                  _INDEX_COLLISION_LINKS,
                                  plane_normal_root, plane_centroid_root)

        part6_dist   = i_dists.get('part_6', float('nan'))
        xlhorn_dist  = i_dists.get('xl_platform_horn_1', float('nan'))

        # Mesh-level collision (only if hand GLB exists)
        mesh_col_str = "n/a"
        if hand_glb.exists():
            col_results = _check_mesh_collisions(
                robot, joint_cfg, T_root_world, hand_glb)
            n_colliding = sum(1 for _, c in col_results if c)
            mesh_col_str = f"{n_colliding}/{len(col_results)} meshes"

        print(f"{w:>8.1f}  {thumb_res*1000:>9.2f}mm  {index_res*1000:>9.2f}mm  "
              f"{part6_dist:>9.1f}mm  {xlhorn_dist:>11.1f}mm  {mesh_col_str:>12}")

    # Detailed per-link distances for the default weight
    default_w = _COLLISION_WEIGHT
    print(f"\nDetailed link distances (weight={default_w}, frame {frame_idx}):")
    thumb_cfg, _ = _run_ik(
        robot, _THUMB_CHAIN, 'part_3', _THUMB_VISUAL_ORIGIN, index_target,
        _THUMB_COLLISION_LINKS, plane_normal_root, plane_centroid_root,
        default_w, _THUMB_Q0_CANDIDATES)
    index_cfg, _ = _run_ik(
        robot, _INDEX_CHAIN, 'part_3_1', _INDEX_VISUAL_ORIGIN, thumb_target,
        _INDEX_COLLISION_LINKS, plane_normal_root, plane_centroid_root,
        default_w, _INDEX_Q0_CANDIDATES)
    joint_cfg = {**thumb_cfg, **index_cfg}

    print("  Thumb chain:")
    for name, d in _link_distances(robot, _THUMB_CHAIN, joint_cfg,
                                   _THUMB_COLLISION_LINKS,
                                   plane_normal_root, plane_centroid_root).items():
        flag = " *** INSIDE HAND" if d < 0 else (" (within margin)" if d < 3 else "")
        print(f"    {name:<30} {d:+.1f} mm{flag}")
    print("  Index chain:")
    for name, d in _link_distances(robot, _INDEX_CHAIN, joint_cfg,
                                   _INDEX_COLLISION_LINKS,
                                   plane_normal_root, plane_centroid_root).items():
        flag = " *** INSIDE HAND" if d < 0 else (" (within margin)" if d < 3 else "")
        print(f"    {name:<30} {d:+.1f} mm{flag}")

    if hand_glb.exists():
        print("\nMesh-level collision details (default weight):")
        for name, col in _check_mesh_collisions(robot, joint_cfg, T_root_world, hand_glb):
            status = "COLLIDES" if col else "clear"
            print(f"    {name:<40} {status}")
    else:
        print(f"\n[WARNING] Hand GLB not found: {hand_glb} — skipping mesh collision check")


if __name__ == "__main__":
    main()

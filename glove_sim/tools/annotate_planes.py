#!/usr/bin/env python3
"""
glove_sim/tools/annotate_planes.py

Two-phase interactive annotation tool for glove-hand alignment using
the Kabsch algorithm (SVD-based rigid registration).

Phase 1: Pick 3 landmarks ON THE GLOVE, in this exact order:
  1. Wrist-side centre of the Hand Mount base
  2. Index-finger knuckle area of the Hand Mount
  3. Pinky-finger knuckle area of the Hand Mount

Phase 2: Pick the matching 3 anatomical landmarks ON THE HAND, in the same order:
  1. Centre of the wrist
  2. Index-finger knuckle
  3. Pinky-finger knuckle

The Kabsch algorithm computes the optimal rotation (det = +1, no reflections)
and translation that maps the glove landmarks onto the hand landmarks.

Controls (both phases):
  Right-click (no drag) — add a point on the mesh surface
  Left-drag             — rotate camera
  SHIFT+Left-drag       — pan camera
  Scroll / Right-drag   — zoom
  U                     — undo last point
  Enter                 — confirm (need exactly 3 points)
  Q                     — quit / abort

Usage:
  python glove_sim/tools/annotate_planes.py [--frame 300]
  python glove_sim/tools/annotate_planes.py --recompute   # recompute from saved annotation
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "glove_sim"))
import config as cfg


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def fit_plane(points):
    """Fit a plane to N×3 points via SVD.  Returns (centroid, unit_normal)."""
    pts = np.asarray(points, dtype=float)
    c = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - c)
    n = Vt[-1]
    return c, n / np.linalg.norm(n)


def kabsch(P, Q):
    """Kabsch algorithm: find R (det=+1) and t that minimise ||Q - (R@P.T).T - t||².

    Parameters
    ----------
    P : (N, 3) source points
    Q : (N, 3) target points  (same order as P)

    Returns
    -------
    R : (3, 3) rotation matrix, det == +1
    t : (3,)  translation vector  — satisfies  Q_i ≈ R @ P_i + t
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    p_c = P.mean(axis=0)
    q_c = Q.mean(axis=0)
    H   = (P - p_c).T @ (Q - q_c)
    U, _, Vt = np.linalg.svd(H)
    # Force det = +1 to prevent reflections
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = q_c - R @ p_c
    return R, t


def _get_glove_bottom_geometry_local():
    """Extract flat-bottom centroid and outward normal from the Hand Mount STL.

    Returns (g_c_local, g_n_local) in the hand_mount link's own rest frame.
    g_n_local points TOWARD the linkage/body side (away from hand).
    """
    _ROOT_IN_HM = np.array([0.157876, -0.0663838, 0.0660817])
    root_dir    = _ROOT_IN_HM / np.linalg.norm(_ROOT_IN_HM)

    mesh = trimesh.load(str(cfg.MESH_DIR / "Hand Mount.stl"), force='mesh')
    dots = mesh.face_normals @ root_dir
    for thresh in (-0.7, -0.5, -0.3, 0.0):
        mask = dots < thresh
        if mask.sum() >= 3:
            break
    if mask.sum() < 3:
        raise RuntimeError("Could not find flat-bottom faces in Hand Mount.stl")

    verts    = mesh.vertices[mesh.faces[mask].ravel()]
    g_c_local = verts.mean(axis=0)
    g_n_raw   = mesh.face_normals[mask].mean(axis=0)
    g_n_raw  /= np.linalg.norm(g_n_raw)
    if np.dot(g_n_raw, root_dir) < 0:
        g_n_raw = -g_n_raw          # point toward linkage side (away from hand)
    return g_c_local, g_n_raw


def compute_T_from_landmarks(T_wrist, T_wrist_to_hm_prev, glove_pts, hand_pts):
    """Compute T_WRIST_TO_HM from 3 corresponding landmarks using Kabsch SVD.

    Parameters
    ----------
    T_wrist            : (4,4) wrist world transform (MANO / Y-down)
    T_wrist_to_hm_prev : (4,4) T_WRIST_TO_HM active when annotation was done
                         (used only to determine the glove's current GLB pose)
    glove_pts          : (3,3) landmark positions on the glove in GLB space
    hand_pts           : (3,3) matching landmark positions on the hand in GLB space

    Returns
    -------
    T_offset    : (4,4) new T_WRIST_TO_HM (wrist-local)
    T_hm_new_glb: (4,4) new hand_mount-to-GLB transform (for downstream use)
    """
    MANO_TO_GLB = np.diag([1.0, 1.0, -1.0, 1.0])

    # Current HM-to-GLB transform (where the glove was when the user annotated it)
    T_hm_curr_glb = MANO_TO_GLB @ T_wrist @ T_wrist_to_hm_prev

    # Kabsch: correction in GLB space (maps glove landmarks → hand landmarks)
    R_corr, t_corr = kabsch(glove_pts, hand_pts)
    T_corr = np.eye(4, dtype=float)
    T_corr[:3, :3] = R_corr
    T_corr[:3,  3] = t_corr

    # Apply correction to the current HM transform
    T_hm_new_glb = T_corr @ T_hm_curr_glb

    # Convert back to wrist-local offset
    T_hm_new_mano = MANO_TO_GLB @ T_hm_new_glb   # MANO_TO_GLB is self-inverse
    T_offset = np.linalg.inv(T_wrist) @ T_hm_new_mano
    return T_offset, T_hm_new_glb


# ---------------------------------------------------------------------------
# Viewer with right-click picking (pyrender / pyglet 2.x compatible)
# ---------------------------------------------------------------------------

class PickingViewer:
    """Interactive mesh viewer that records right-click ray-mesh hit points.

    Opens a window and BLOCKS until closed (Enter = confirm, Q = abort).
    """

    _N_REQUIRED = 3

    def __init__(self, trimesh_mesh, phase_label):
        import pyrender

        self._selected = []
        self._aborted  = False
        self._target   = trimesh_mesh
        self._intersector = trimesh.ray.ray_triangle.RayMeshIntersector(trimesh_mesh)

        print(f"\n{'='*60}")
        print(f"  {phase_label}")
        print(f"  Left-drag           : rotate camera")
        print(f"  SHIFT+Left-drag     : pan camera")
        print(f"  Scroll / Right-drag : zoom")
        print(f"  Right-click (no drag): ADD POINT on surface")
        print(f"  U                   : undo last point")
        print(f"  Enter               : confirm (need exactly {self._N_REQUIRED} points)")
        print(f"  Q                   : abort")
        print(f"{'='*60}\n")

        pr_scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1, 1.0],
                                  ambient_light=[0.3, 0.3, 0.3])
        pr_scene.add(pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=False))

        picking_self = self

        class _PV(pyrender.Viewer):
            _rpress_pos = None

            def on_mouse_press(self_, x, y, buttons, modifiers):
                import pyglet.window.mouse as _mouse
                if buttons == _mouse.RIGHT:
                    self_._rpress_pos = (x, y)
                super().on_mouse_press(x, y, buttons, modifiers)

            def on_mouse_release(self_, x, y, button, modifiers):
                import pyglet.window.mouse as _mouse
                if button == _mouse.RIGHT and self_._rpress_pos is not None:
                    dx = abs(x - self_._rpress_pos[0])
                    dy = abs(y - self_._rpress_pos[1])
                    self_._rpress_pos = None
                    if dx <= 4 and dy <= 4:
                        picking_self._do_pick(self_, x, y)
                        return
                super().on_mouse_release(x, y, button, modifiers)

            def on_key_press(self_, symbol, modifiers):
                import pyglet.window.key as _key
                if symbol in (_key.ENTER, _key.RETURN, _key.NUM_ENTER):
                    n = len(picking_self._selected)
                    if n == picking_self._N_REQUIRED:
                        print(f"  Confirmed {n} points.")
                        self_.on_close()
                    else:
                        print(f"  Need exactly {picking_self._N_REQUIRED} points "
                              f"(have {n}).")
                elif symbol == _key.U:
                    picking_self._cb_undo()
                elif symbol == _key.Q:
                    picking_self._cb_abort()
                    self_.on_close()
                else:
                    super().on_key_press(symbol, modifiers)

        self._viewer = _PV(
            pr_scene,
            use_raymond_lighting=True,
            viewer_flags={'window_title': phase_label},
        )

    @property
    def selected_points(self):
        return list(self._selected)

    @property
    def aborted(self):
        return self._aborted

    def _cb_undo(self):
        if self._selected:
            self._selected.pop()
            print(f"  Undone. {len(self._selected)} point(s) remain.")
        else:
            print("  Nothing to undo.")

    def _cb_abort(self):
        print("  Aborting.")
        self._aborted = True
        self._selected.clear()

    def _do_pick(self, viewer, px, py):
        try:
            origin, direction = self._pixel_to_ray(viewer, px, py)
            locs, _, _ = self._intersector.intersects_location(
                ray_origins=origin[None],
                ray_directions=direction[None],
                multiple_hits=False,
            )
            if len(locs):
                pt = locs[0].tolist()
                idx = len(self._selected) + 1
                labels = ["wrist-centre", "index-knuckle", "pinky-knuckle"]
                label  = labels[idx - 1] if idx <= len(labels) else f"pt{idx}"
                self._selected.append(pt)
                print(f"  [{idx}] {label}: "
                      f"({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})")
                self._add_marker(viewer, np.array(pt))
                if len(self._selected) == self._N_REQUIRED:
                    print(f"  All {self._N_REQUIRED} points collected — press Enter to confirm.")
            else:
                print("  (no hit — click directly on the visible mesh surface)")
        except Exception as exc:
            print(f"  (pick error: {exc})")

    def _pixel_to_ray(self, viewer, px, py):
        import pyrender
        T    = viewer._camera_node.matrix
        w, h = viewer.viewport_size
        cam  = viewer._camera_node.camera
        yfov   = float(cam.yfov)
        aspect = w / h
        half_h = np.tan(yfov / 2.0)
        half_w = half_h * aspect
        ndx = (2.0 * px / w) - 1.0
        ndy = (2.0 * py / h) - 1.0
        d_cam = np.array([ndx * half_w, ndy * half_h, -1.0])
        d_cam /= np.linalg.norm(d_cam)
        origin    = T[:3, 3].copy()
        direction = T[:3, :3] @ d_cam
        direction /= np.linalg.norm(direction)
        return origin, direction

    def _add_marker(self, viewer, pt):
        try:
            import pyrender
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.005)
            sphere.visual.face_colors = np.array([255, 60, 60, 220], dtype=np.uint8)
            T = np.eye(4); T[:3, 3] = pt
            sphere.apply_transform(T)
            viewer.scene.add(pyrender.Mesh.from_trimesh(sphere, smooth=False))
        except Exception:
            pass


def pick_surface(trimesh_mesh, phase_label):
    """Open interactive viewer; return list of 3 selected points or None if aborted."""
    v = PickingViewer(trimesh_mesh, phase_label)
    return None if v.aborted else v.selected_points


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Landmark-based glove-hand alignment calibration (Kabsch/SVD)."
    )
    parser.add_argument("--frame", type=int, default=300,
                        help="Frame index to use for annotation (default: 300)")
    parser.add_argument("--out", type=Path,
                        default=cfg.ALIGNED_DIR / "plane_annotation.json")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute T_WRIST_TO_HM from existing annotation without "
                             "re-annotating (useful after fixing the algorithm)")
    args = parser.parse_args()

    if args.recompute:
        if not args.out.exists():
            print(f"ERROR: no annotation at {args.out}")
            sys.exit(1)
        print(f"Loading existing annotation from {args.out} …")
        existing    = json.loads(args.out.read_text())
        T_wrist     = np.array(existing["T_wrist"], dtype=float)
        glove_pts   = np.array(existing["glove_landmarks"], dtype=float)
        hand_pts    = np.array(existing["hand_landmarks"],  dtype=float)
        frame_idx   = existing["frame"]
    else:
        # ---- Load meshes ----
        print("Loading MANO frame data and computing IK …")
        from src.urdfpy_vis import load_robot, get_glove_scene
        from src.mano_io    import load_frame
        from align_frame    import solve_ik_frame

        frame_data = load_frame(cfg.NPZ_PATH, cfg.MANO_DIR, args.frame)
        robot      = load_robot(cfg.URDF_PATH, cfg.MESH_DIR)
        joint_cfg, _, _ = solve_ik_frame(
            robot,
            frame_data["T_wrist"],
            frame_data["thumb_tip"],
            frame_data["index_tip"],
        )

        T_hand_mount = frame_data["T_wrist"] @ cfg.T_WRIST_TO_HM
        glove_scene  = get_glove_scene(robot, joint_cfg, T_hand_mount)
        glove_mesh   = trimesh.util.concatenate(list(glove_scene.geometry.values()))

        hand_glb = cfg.GLB_DIR / f"{args.frame:06d}_hands.glb"
        if not hand_glb.exists():
            print(f"ERROR: hand GLB not found: {hand_glb}")
            sys.exit(1)
        hand_scene = trimesh.load(str(hand_glb), force="scene")
        hand_mesh  = trimesh.util.concatenate(list(hand_scene.geometry.values()))

        print(f"Glove mesh: {len(glove_mesh.faces)} faces")
        print(f"Hand  mesh: {len(hand_mesh.faces)} faces")

        def _require(pts, phase):
            if not pts or len(pts) != 3:
                print(f"Annotation aborted or wrong point count in {phase}.  Exiting.")
                sys.exit(1)
            return np.array(pts, dtype=float)

        # ---- Phase 1: 3 glove landmarks ----
        print("\n" + "="*60)
        print("PHASE 1 — GLOVE LANDMARKS")
        print("Pick exactly 3 points IN THIS ORDER:")
        print("  1. Wrist-side centre of the Hand Mount base plate")
        print("  2. Index-finger knuckle corner of the Hand Mount")
        print("  3. Pinky-finger knuckle corner of the Hand Mount")
        glove_pts = _require(
            pick_surface(glove_mesh,
                         "Phase 1/2 — GLOVE: wrist-centre, index-knuckle, pinky-knuckle"),
            "Phase 1")

        # ---- Phase 2: 3 matching hand landmarks ----
        print("\n" + "="*60)
        print("PHASE 2 — HAND LANDMARKS (same order as Phase 1)")
        print("Pick exactly 3 points IN THIS ORDER:")
        print("  1. Centre of the wrist (same anatomical point as glove pt 1)")
        print("  2. Index-finger knuckle (metacarpal head)")
        print("  3. Pinky-finger knuckle (metacarpal head)")
        hand_pts = _require(
            pick_surface(hand_mesh,
                         "Phase 2/2 — HAND: wrist-centre, index-knuckle, pinky-knuckle"),
            "Phase 2")

        T_wrist   = np.array(frame_data["T_wrist"], dtype=float)
        frame_idx = args.frame

    # ---- Compute T_WRIST_TO_HM via Kabsch ----
    T_offset, T_hm_new_glb = compute_T_from_landmarks(
        T_wrist, cfg.T_WRIST_TO_HM, glove_pts, hand_pts
    )

    # Verify det(R) == +1
    det = np.linalg.det(T_hm_new_glb[:3, :3])
    assert abs(det - 1.0) < 1e-6, f"det(R) = {det:.6f} — reflection detected!"

    # ---- Residual (should be ~0 for 3-point exact fit) ----
    R, t = T_hm_new_glb[:3, :3], T_hm_new_glb[:3, 3]
    MANO_TO_GLB = np.diag([1.0, 1.0, -1.0, 1.0])
    T_hm_curr   = MANO_TO_GLB @ T_wrist @ cfg.T_WRIST_TO_HM
    glove_pts   = np.asarray(glove_pts, dtype=float)
    hand_pts    = np.asarray(hand_pts,  dtype=float)
    R_c, t_c    = T_hm_curr[:3, :3], T_hm_curr[:3, 3]
    # Corrected glove landmark positions
    R_corr      = T_hm_new_glb[:3, :3] @ np.linalg.inv(T_hm_curr[:3, :3])
    t_corr      = T_hm_new_glb[:3, 3]  - R_corr @ T_hm_curr[:3, 3]
    corrected   = (R_corr @ glove_pts.T).T + t_corr
    rmsd_mm     = float(np.sqrt(((corrected - hand_pts)**2).sum(axis=1).mean())) * 1000

    # ---- Glove bottom and hand dorsal for plane-alignment tests ----
    g_c_local, g_n_local = _get_glove_bottom_geometry_local()
    g_c_glb = T_hm_new_glb[:3, :3] @ g_c_local + T_hm_new_glb[:3, 3]
    g_n_glb = T_hm_new_glb[:3, :3] @ g_n_local

    h_c, h_n = fit_plane(hand_pts)
    if np.dot(h_n, g_c_glb - h_c) < 0:
        h_n = -h_n

    cos_a = abs(float(np.dot(g_n_glb / np.linalg.norm(g_n_glb), h_n)))
    angle_deg = float(np.degrees(np.arccos(min(1.0, cos_a))))
    gap_mm    = abs(float(np.dot(g_c_glb - h_c, h_n))) * 1000

    print(f"\n{'='*60}")
    print(f"Kabsch RMSD            : {rmsd_mm:.3f} mm  (0 = perfect 3-point fit)")
    print(f"det(R)                 : {det:.6f}  (must be +1)")
    print(f"Glove bottom → dorsal angle : {angle_deg:.1f}°  (target < 15°)")
    print(f"Glove bottom → dorsal gap   : {gap_mm:.1f} mm  (target < 5 mm)")
    print(f"{'='*60}")

    # ---- Save ----
    annotation = {
        "frame":            frame_idx,
        "T_wrist":          T_wrist.tolist(),
        "glove_landmarks":  glove_pts.tolist(),
        "hand_landmarks":   hand_pts.tolist(),
        # Fields consumed by test_plane_alignment.py
        "glove_bottom": {
            "points":   glove_pts.tolist(),
            "centroid": g_c_glb.tolist(),
            "normal":   (g_n_glb / np.linalg.norm(g_n_glb)).tolist(),
        },
        "hand_dorsal": {
            "points":   hand_pts.tolist(),
            "centroid": h_c.tolist(),
            "normal":   h_n.tolist(),
        },
        "metrics": {
            "kabsch_rmsd_mm":           rmsd_mm,
            "angle_between_normals_deg": angle_deg,
            "gap_between_planes_m":      gap_mm / 1000.0,
        },
        "T_WRIST_TO_HM": T_offset.tolist(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(annotation, indent=2))
    print(f"\nAnnotation saved → {args.out}")
    print(f"config.py will auto-load the new T_WRIST_TO_HM on next import.\n")
    print(f"Next: run  python glove_sim/align_frame.py --frames 300  and verify visually.")


if __name__ == "__main__":
    main()

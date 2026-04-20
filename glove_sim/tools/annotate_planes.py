#!/usr/bin/env python3
"""
glove_sim/tools/annotate_planes.py

Two-phase interactive annotation tool for glove-hand plane alignment.

Phase 1: shows the GLOVE mesh alone — click the flat underside of the Hand Mount.
Phase 2: shows the HAND mesh alone — click the dorsal (back) surface above the knuckles.

After both phases the script fits planes, computes the T_WRIST_TO_HM offset
matrix, and writes everything to plane_annotation.json.

Controls (same for both phases):
  Left-click  — add a point on the mesh surface
  U           — undo last point
  S / Enter   — confirm selection (need >= 3 points)
  Q           — quit / abort

Usage:
  python glove_sim/tools/annotate_planes.py [--frame 300]
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


def _rotation_from_to(v_from, v_to):
    """Minimum rotation matrix that maps unit vector v_from → unit vector v_to."""
    v_from = np.asarray(v_from, dtype=float)
    v_to   = np.asarray(v_to,   dtype=float)
    v_from /= np.linalg.norm(v_from)
    v_to   /= np.linalg.norm(v_to)
    axis = np.cross(v_from, v_to)
    sin_a = np.linalg.norm(axis)
    cos_a = float(np.dot(v_from, v_to))
    if sin_a < 1e-9:
        # Parallel or antiparallel
        if cos_a > 0:
            return np.eye(3)
        # 180° rotation about a perpendicular axis
        perp = (np.array([1.0, 0.0, 0.0]) if abs(v_from[0]) < 0.9
                else np.array([0.0, 1.0, 0.0]))
        axis = np.cross(v_from, perp)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    axis /= sin_a
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)


def compute_T_wrist_to_hm(T_wrist, glove_centroid, glove_normal,
                           hand_centroid, hand_normal):
    """Compute the T_WRIST_TO_HM offset matrix.

    Finds the rigid transform (in the wrist local frame) such that when
    applied to the wrist pose, the glove bottom plane aligns flush with the
    hand dorsal plane.

    Parameters
    ----------
    T_wrist        : (4,4) wrist world transform (MANO world / Y-down)
    glove_centroid : (3,)  centre of glove bottom plane in GLB space
    glove_normal   : (3,)  unit normal of glove bottom plane in GLB space
    hand_centroid  : (3,)  centre of hand dorsal plane in GLB space
    hand_normal    : (3,)  unit normal of hand dorsal plane in GLB space

    Returns
    -------
    T_offset : (4,4) transform in the wrist LOCAL frame
    """
    MANO_TO_GLB = np.diag([1.0, 1.0, -1.0, 1.0])

    # Current T_hand_mount in GLB space (T_offset = identity)
    T_hm_glb     = MANO_TO_GLB @ T_wrist
    T_hm_glb_inv = np.linalg.inv(T_hm_glb)

    # Glove bottom plane in hand_mount LOCAL frame
    g_n  = np.asarray(glove_normal,   dtype=float); g_n /= np.linalg.norm(g_n)
    g_c  = np.asarray(glove_centroid, dtype=float)
    h_n  = np.asarray(hand_normal,    dtype=float); h_n /= np.linalg.norm(h_n)
    h_c  = np.asarray(hand_centroid,  dtype=float)

    g_n_local = T_hm_glb_inv[:3, :3] @ g_n
    g_c_local = (T_hm_glb_inv @ np.append(g_c, 1.0))[:3]

    # Current glove bottom normal expressed in GLB (should ≈ g_n already)
    g_n_glb_current = T_hm_glb[:3, :3] @ g_n_local

    # We want the glove bottom to face INTO the hand → antiparallel to h_n
    target_g_n_glb = -h_n

    # Rotation correction in GLB space
    R_corr = _rotation_from_to(g_n_glb_current, target_g_n_glb)

    # New T_hand_mount rotation in GLB space
    R_new_glb = R_corr @ T_hm_glb[:3, :3]

    # New translation so glove centroid sits on the dorsal surface
    P_new_glb = h_c - R_new_glb @ g_c_local

    T_hm_new_glb = np.eye(4, dtype=float)
    T_hm_new_glb[:3, :3] = R_new_glb
    T_hm_new_glb[:3,  3] = P_new_glb

    # Convert back to MANO world (MANO_TO_GLB is its own inverse)
    T_hm_new_mano = MANO_TO_GLB @ T_hm_new_glb

    # Express as offset in wrist LOCAL frame
    T_offset = np.linalg.inv(T_wrist) @ T_hm_new_mano
    return T_offset


# ---------------------------------------------------------------------------
# Viewer with picking
# ---------------------------------------------------------------------------

class PickingViewer(trimesh.viewer.SceneViewer):
    """SceneViewer subclass that records left-click ray-mesh hit points."""

    _MIN_POINTS = 3

    def __init__(self, scene, target_mesh, phase_label, **kwargs):
        super().__init__(scene, **kwargs)
        self._target      = target_mesh
        self._intersector = trimesh.ray.ray_triangle.RayMeshIntersector(target_mesh)
        self._selected    = []   # list of [x, y, z]
        self._aborted     = False
        print(f"\n{'='*60}")
        print(f"  {phase_label}")
        print(f"  Left-click  : add point on surface")
        print(f"  U           : undo last point")
        print(f"  S / Enter   : confirm (need >= {self._MIN_POINTS} points)")
        print(f"  Q           : abort")
        print(f"{'='*60}\n")

    # ---- public ----

    @property
    def selected_points(self):
        return list(self._selected)

    # ---- event handlers ----

    def on_mouse_press(self, x, y, buttons, modifiers):
        try:
            import pyglet.window.mouse as mouse
        except ImportError:
            super().on_mouse_press(x, y, buttons, modifiers)
            return

        if buttons & mouse.LEFT:
            try:
                origin, direction = self._pixel_to_ray(x, y)
                locs, _, _ = self._intersector.intersects_location(
                    ray_origins=origin[None],
                    ray_directions=direction[None],
                    multiple_hits=False,
                )
                if len(locs):
                    pt = locs[0].tolist()
                    self._selected.append(pt)
                    print(f"  [{len(self._selected)}] ({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})")
                    self._try_add_marker(np.array(pt))
                else:
                    print("  (no hit — click directly on the visible mesh surface)")
            except Exception as exc:
                print(f"  (pick error: {exc})")
        else:
            super().on_mouse_press(x, y, buttons, modifiers)

    def on_key_press(self, symbol, modifiers):
        try:
            import pyglet.window.key as key
        except ImportError:
            super().on_key_press(symbol, modifiers)
            return

        if symbol in (key.ENTER, key.RETURN, key.S):
            n = len(self._selected)
            if n >= self._MIN_POINTS:
                print(f"  Confirmed {n} points.")
                self.close()
            else:
                print(f"  Need at least {self._MIN_POINTS} points (have {n}).")
        elif symbol == key.U:
            if self._selected:
                self._selected.pop()
                print(f"  Undone. {len(self._selected)} point(s) remain.")
            else:
                print("  Nothing to undo.")
        elif symbol == key.Q:
            print("  Aborting.")
            self._aborted = True
            self._selected.clear()
            self.close()
        else:
            super().on_key_press(symbol, modifiers)

    # ---- helpers ----

    def _pixel_to_ray(self, px, py):
        """Convert pyglet window pixel to world-space ray (origin, unit direction).

        pyglet origin is bottom-left; y increases upward (same as OpenGL NDC).
        trimesh camera looks down its local -Z axis.
        """
        T   = self.scene.camera_transform          # camera → world, (4, 4)
        w, h = self.get_size()
        cam  = self.scene.camera

        # Field of view
        fov = cam.fov
        if hasattr(fov, "__len__") and len(fov) >= 2:
            fov_y_rad = np.radians(float(fov[1]))
        else:
            fov_y_rad = np.radians(float(fov))

        aspect   = w / h
        half_h   = np.tan(fov_y_rad / 2.0)
        half_w   = half_h * aspect

        ndx = (2.0 * px / w) - 1.0   # [-1, 1]
        ndy = (2.0 * py / h) - 1.0   # [-1, 1]

        # Camera-space ray (camera looks down -Z)
        d_cam = np.array([ndx * half_w, ndy * half_h, -1.0])
        d_cam /= np.linalg.norm(d_cam)

        origin    = T[:3, 3]
        direction = T[:3, :3] @ d_cam
        direction /= np.linalg.norm(direction)
        return origin, direction

    def _try_add_marker(self, pt):
        """Add a small red sphere at pt; silently fail if the API is unavailable."""
        try:
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.005)
            sphere.visual.face_colors = np.array([255, 60, 60, 220], dtype=np.uint8)
            T = np.eye(4); T[:3, 3] = pt
            sphere.apply_transform(T)
            name = f"_pick_{len(self._selected)}"
            self.scene.add_geometry(sphere, geom_name=name)
            # Ask the viewer to rebuild its vertex buffers
            if hasattr(self, "_update_vertex_list"):
                self._update_vertex_list()
        except Exception:
            pass   # visual marker is cosmetic — don't crash if it fails


# ---------------------------------------------------------------------------
# Per-phase helper
# ---------------------------------------------------------------------------

def pick_surface(mesh, phase_label):
    """Open a viewer for one mesh, return list of selected points."""
    import pyglet

    scene  = trimesh.Scene([mesh])
    viewer = PickingViewer(scene, mesh, phase_label)
    pyglet.app.run()          # blocks until the window is closed

    if viewer._aborted:
        return None
    return viewer.selected_points


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Annotate glove-bottom and hand-dorsal planes for alignment calibration."
    )
    parser.add_argument("--frame", type=int, default=300,
                        help="Frame to use for annotation (default: 300)")
    parser.add_argument("--out",   type=Path,
                        default=cfg.ALIGNED_DIR / "plane_annotation.json")
    args = parser.parse_args()

    # ---- Prepare meshes ----
    print("Loading MANO frame data and computing IK…")
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

    # Glove: concatenate all glove link meshes into one pickable mesh
    glove_scene  = get_glove_scene(robot, joint_cfg, frame_data["T_wrist"])
    glove_mesh   = trimesh.util.concatenate(list(glove_scene.geometry.values()))

    # Hand: load directly from the per-frame GLB
    hand_glb = cfg.GLB_DIR / f"{args.frame:06d}_hands.glb"
    if not hand_glb.exists():
        print(f"ERROR: hand GLB not found: {hand_glb}")
        print("Run align_frame.py first to generate aligned outputs.")
        sys.exit(1)
    hand_scene = trimesh.load(str(hand_glb), force="scene")
    hand_mesh  = trimesh.util.concatenate(list(hand_scene.geometry.values()))

    print(f"Glove mesh: {len(glove_mesh.faces)} faces")
    print(f"Hand  mesh: {len(hand_mesh.faces)} faces")

    # ---- Phase 1: glove bottom ----
    print("\nPhase 1 of 2: GLOVE BOTTOM")
    print("Click the flat underside of the Hand Mount (opposite the linkages/sensors).")
    glove_pts = pick_surface(
        glove_mesh,
        "Phase 1 — GLOVE BOTTOM: click the flat underside of the Hand Mount",
    )
    if not glove_pts or len(glove_pts) < 3:
        print("Annotation aborted or too few points.  Exiting.")
        sys.exit(1)

    # ---- Phase 2: hand dorsal ----
    print("\nPhase 2 of 2: HAND DORSAL SURFACE")
    print("Click the back of the hand above the knuckle area (dorsal metacarpal surface).")
    hand_pts = pick_surface(
        hand_mesh,
        "Phase 2 — HAND DORSAL: click the back of the hand above the knuckles",
    )
    if not hand_pts or len(hand_pts) < 3:
        print("Annotation aborted or too few points.  Exiting.")
        sys.exit(1)

    # ---- Fit planes ----
    g_pts = np.array(glove_pts, dtype=float)
    h_pts = np.array(hand_pts,  dtype=float)

    g_centroid, g_normal = fit_plane(g_pts)
    h_centroid, h_normal = fit_plane(h_pts)

    cos_a   = abs(float(np.dot(g_normal, h_normal)))
    angle   = float(np.degrees(np.arccos(min(1.0, cos_a))))
    gap_m   = abs(float(np.dot(g_centroid - h_centroid, h_normal)))

    print(f"\n{'='*60}")
    print(f"Glove bottom centroid : {g_centroid}")
    print(f"Glove bottom normal   : {g_normal}")
    print(f"Hand dorsal  centroid : {h_centroid}")
    print(f"Hand dorsal  normal   : {h_normal}")
    print(f"Angle between normals : {angle:.1f}°  (target < 15°)")
    print(f"Gap between planes    : {gap_m*1000:.1f} mm  (target < 5 mm)")

    # ---- Compute T_WRIST_TO_HM ----
    T_wrist  = np.array(frame_data["T_wrist"], dtype=float)
    T_offset = compute_T_wrist_to_hm(T_wrist, g_centroid, g_normal,
                                     h_centroid, h_normal)

    print(f"\nComputed T_WRIST_TO_HM:")
    print(repr(T_offset.tolist()))
    print("\nPaste into glove_sim/config.py as:")
    print("  T_WRIST_TO_HM = np.array(")
    for row in T_offset:
        print(f"      {row.tolist()},")
    print("  )")
    print(f"{'='*60}")

    # ---- Save ----
    annotation = {
        "frame": args.frame,
        "T_wrist": T_wrist.tolist(),
        "glove_bottom": {
            "points":   glove_pts,
            "centroid": g_centroid.tolist(),
            "normal":   g_normal.tolist(),
        },
        "hand_dorsal": {
            "points":   hand_pts,
            "centroid": h_centroid.tolist(),
            "normal":   h_normal.tolist(),
        },
        "metrics": {
            "angle_between_normals_deg": angle,
            "gap_between_planes_m":      gap_m,
        },
        "T_WRIST_TO_HM": T_offset.tolist(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(annotation, indent=2))
    print(f"\nAnnotation saved → {args.out}")
    print("\nNext steps:")
    print("  1. Add T_WRIST_TO_HM to glove_sim/config.py")
    print("  2. Update align_frame.py to use T_WRIST_TO_HM")
    print("  3. Re-run:  python glove_sim/align_frame.py --frames 300")
    print("  4. Check:   pytest glove_sim/tests/test_plane_alignment.py -v -p no:dash")


if __name__ == "__main__":
    main()

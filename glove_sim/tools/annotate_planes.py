#!/usr/bin/env python3
"""
glove_sim/tools/annotate_planes.py

Five-phase interactive annotation tool for glove-hand plane alignment.

Phase 1: GLOVE FRONT — flat bottom of Hand Mount, knuckle-side edge
Phase 2: GLOVE BACK  — flat bottom of Hand Mount, wrist-side edge
Phase 3: GLOVE TOP   — top of Hand Mount (linkage/sensor side, faces AWAY from hand)
Phase 4: HAND FRONT  — dorsal hand surface above the knuckles
Phase 5: HAND BACK   — dorsal hand surface near the wrist

The glove TOP annotation resolves the normal-direction ambiguity: the vector
from bottom-centroid to top-centroid unambiguously points away from the hand,
so the flat bottom can be flipped correctly without going through the hand.
Two sub-regions per surface pin the remaining in-plane rotation (6 DOF total).

After all five phases the script fits planes, computes T_WRIST_TO_HM,
and writes everything to plane_annotation.json.

Controls (same for all phases):
  Right-click (no drag) — add a point on the mesh surface
  Left-drag             — rotate camera
  SHIFT+Left-drag       — pan camera
  Scroll / Right-drag   — zoom
  U                     — undo last point
  Enter                 — confirm selection (need >= 3 points)
  Q                     — quit / abort

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
        if cos_a > 0:
            return np.eye(3)
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


def _ortho_frame(z_vec, x_hint):
    """Build a 3×3 orthonormal matrix whose columns are [x, y, z].

    z is normalised z_vec; x is x_hint projected onto the plane ⟂ z; y = z×x.
    """
    z = np.asarray(z_vec,  dtype=float); z /= np.linalg.norm(z)
    x = np.asarray(x_hint, dtype=float)
    x = x - np.dot(x, z) * z              # project out z component
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])      # columns are basis vectors


def _get_glove_bottom_geometry_local():
    """Extract flat-bottom centroid, normal, and in-plane PCA axis from Hand Mount STL.

    All results are in the hand_mount link's local frame (= HM rest frame).
    g_n_local is oriented to point TOWARD the linkage/body side (away from hand).
    g_axis_local_raw is the long PCA axis of the bottom face vertices (sign ambiguous).
    """
    _ROOT_IN_HM = np.array([0.157876, -0.0663838, 0.0660817])
    root_dir    = _ROOT_IN_HM / np.linalg.norm(_ROOT_IN_HM)

    mesh_path = cfg.MESH_DIR / "Hand Mount.stl"
    mesh = trimesh.load(str(mesh_path), force='mesh')

    # Bottom faces: face normals pointing AWAY from the linkage/root direction.
    dots = mesh.face_normals @ root_dir
    for thresh in (-0.7, -0.5, -0.3, 0.0):
        bottom_mask = dots < thresh
        if bottom_mask.sum() >= 3:
            break
    if bottom_mask.sum() < 3:
        raise RuntimeError(
            f"Could not find flat-bottom faces in {mesh_path}. "
            "Check MESH_DIR and mesh orientation."
        )
    print(f"  [mesh] {bottom_mask.sum()} bottom faces found (dot threshold {thresh:.1f})")

    bottom_verts = mesh.vertices[mesh.faces[bottom_mask].ravel()]
    g_c_local = bottom_verts.mean(axis=0)

    # Normal: mean of bottom face normals, oriented toward linkage side.
    g_n_raw = mesh.face_normals[bottom_mask].mean(axis=0)
    g_n_raw /= np.linalg.norm(g_n_raw)
    if np.dot(g_n_raw, root_dir) < 0:
        g_n_raw = -g_n_raw          # ensure it points toward linkages (away from hand)
    g_n_local = g_n_raw

    # In-plane long axis via PCA of bottom vertices projected onto the bottom plane.
    vp = bottom_verts - g_c_local
    vp = vp - (vp @ g_n_local)[:, None] * g_n_local
    _, _, Vt = np.linalg.svd(vp, full_matrices=False)
    g_axis_local_raw = Vt[0]       # first PC = long axis (sign ambiguous)

    return g_c_local, g_n_local, g_axis_local_raw


def compute_T_wrist_to_hm(T_wrist, T_wrist_to_hm_prev,
                           glove_front_pts, glove_back_pts, glove_top_pts,
                           hand_front_pts,  hand_back_pts):
    """Compute the T_WRIST_TO_HM offset matrix from plane annotations.

    The glove bottom geometry (centroid + normal + in-plane axis) is extracted
    directly from the Hand Mount STL mesh in its own local frame, so the result
    is completely independent of T_wrist_to_hm_prev (which may be grossly wrong).

    Parameters
    ----------
    T_wrist           : (4,4) wrist world transform (MANO world / Y-down)
    T_wrist_to_hm_prev: unused — kept for API compatibility
    glove_front_pts   : (N,3) knuckle-side edge of glove flat bottom (GLB space)
    glove_back_pts    : (M,3) wrist-side edge of glove flat bottom (GLB space)
    glove_top_pts     : unused — kept for API compatibility
    hand_front_pts    : (N,3) knuckle area of hand dorsal (GLB space)
    hand_back_pts     : (M,3) wrist area of hand dorsal (GLB space)

    Returns
    -------
    T_offset : (4,4) transform in the wrist LOCAL frame  (= T_WRIST_TO_HM)
    g_c, g_n : glove bottom centroid and normal in GLB space (g_c == h_c by construction)
    h_c, h_n : hand dorsal centroid and (toward-glove) normal in GLB space
    """
    MANO_TO_GLB = np.diag([1.0, 1.0, -1.0, 1.0])

    # --- Hand dorsal plane from annotation (all in GLB space) ---
    all_hand  = np.vstack([hand_front_pts, hand_back_pts])
    h_c, h_n  = fit_plane(all_hand)

    h_front_c = np.asarray(hand_front_pts, dtype=float).mean(axis=0)
    h_back_c  = np.asarray(hand_back_pts,  dtype=float).mean(axis=0)

    def _proj(vec, normal):
        v = np.asarray(vec, dtype=float)
        n = np.asarray(normal, dtype=float); n /= np.linalg.norm(n)
        return v - np.dot(v, n) * n

    h_axis = _proj(h_back_c - h_front_c, h_n)

    # --- Glove bottom geometry from URDF mesh (T_prev-independent) ---
    g_c_local, g_n_local, g_axis_local_raw = _get_glove_bottom_geometry_local()

    # --- Orient g_axis_local to match the annotated glove front→back direction ---
    # We project g_axis_local to GLB space using only the normal-alignment rotation
    # (no translation, no T_prev) and compare with the annotation-based axis.
    all_glove    = np.vstack([glove_front_pts, glove_back_pts])
    g_c_annot, g_n_annot = fit_plane(all_glove)
    g_axis_annot = _proj(
        np.asarray(glove_back_pts,  dtype=float).mean(axis=0)
        - np.asarray(glove_front_pts, dtype=float).mean(axis=0),
        g_n_annot,
    )
    R_z = _rotation_from_to(g_n_local, h_n)   # rotation that aligns normals only
    if np.dot(R_z @ g_axis_local_raw, g_axis_annot) < 0:
        g_axis_local_raw = -g_axis_local_raw
    g_axis_local = g_axis_local_raw

    # --- Orient h_n toward the glove (away from hand body interior) ---
    if np.dot(h_n, g_c_annot - h_c) < 0:
        h_n = -h_n

    # --- Build orthonormal frames and rotation ---
    # Both z-axes point "away from hand surface" → they should align directly.
    F_src = _ortho_frame(g_n_local, g_axis_local)   # HM local frame
    F_tgt = _ortho_frame(h_n,       h_axis)          # hand GLB frame
    R_new = F_tgt @ F_src.T

    # --- T_hm_new_glb: maps HM-local → GLB, placing flat-bottom centroid at h_c ---
    T_hm_new_glb = np.eye(4, dtype=float)
    T_hm_new_glb[:3, :3] = R_new
    T_hm_new_glb[:3,  3] = h_c - R_new @ g_c_local

    # --- Convert to wrist-local offset ---
    T_hm_new_mano = MANO_TO_GLB @ T_hm_new_glb   # MANO_TO_GLB is self-inverse
    T_offset = np.linalg.inv(T_wrist) @ T_hm_new_mano

    # --- GLB-space glove bottom for annotation storage ---
    g_c_glb = R_new @ g_c_local + T_hm_new_glb[:3, 3]   # == h_c by construction
    g_n_glb = R_new @ g_n_local

    return T_offset, g_c_glb, g_n_glb, h_c, h_n


# ---------------------------------------------------------------------------
# Viewer with picking (pyrender-based, compatible with pyglet 2.x)
# ---------------------------------------------------------------------------

class PickingViewer:
    """Interactive mesh viewer that records left-click ray-mesh hit points.

    Instantiating this class opens a window and BLOCKS until the window is
    closed (confirm with Enter, abort with Q).  Results are available via
    ``selected_points`` and ``aborted`` after the constructor returns.
    """

    _MIN_POINTS = 3

    def __init__(self, trimesh_mesh, phase_label):
        import pyrender

        self._selected = []
        self._aborted  = False
        self._target   = trimesh_mesh
        self._intersector = trimesh.ray.ray_triangle.RayMeshIntersector(trimesh_mesh)

        print(f"\n{'='*60}")
        print(f"  {phase_label}")
        print(f"  Left-drag         : rotate camera")
        print(f"  SHIFT+Left-drag   : pan camera")
        print(f"  Scroll / Right-drag : zoom")
        print(f"  Right-click (no drag) : ADD POINT on surface")
        print(f"  U                 : undo last point")
        print(f"  Enter             : confirm (need >= {self._MIN_POINTS} points)")
        print(f"  Q                 : abort")
        print(f"{'='*60}\n")

        # Build pyrender scene from the trimesh mesh
        pr_scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1, 1.0],
                                  ambient_light=[0.3, 0.3, 0.3])
        pr_scene.add(pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=False))
        self._pr_scene = pr_scene

        # Subclass approach: we need on_mouse_press/release overrides for picking.
        # Build the viewer subclass inline so it can close over self.
        picking_self = self

        class _PV(pyrender.Viewer):
            # Track right-click press position to distinguish click vs drag
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
                    if dx <= 4 and dy <= 4:   # click, not a drag
                        picking_self._do_pick(self_, x, y)
                        return   # don't pass to super — skip zoom-state release
                super().on_mouse_release(x, y, button, modifiers)

            def on_key_press(self_, symbol, modifiers):
                import pyglet.window.key as _key
                if symbol in (_key.ENTER, _key.RETURN, _key.NUM_ENTER):
                    n = len(picking_self._selected)
                    if n >= picking_self._MIN_POINTS:
                        print(f"  Confirmed {n} points.")
                        self_.on_close()
                    else:
                        print(f"  Need at least {picking_self._MIN_POINTS} "
                              f"points (have {n}).")
                elif symbol == _key.U:
                    picking_self._cb_undo(self_)
                elif symbol == _key.Q:
                    picking_self._cb_abort(self_)
                    self_.on_close()
                else:
                    super().on_key_press(symbol, modifiers)

        # Instantiating _PV blocks until the window closes
        self._viewer = _PV(
            pr_scene,
            use_raymond_lighting=True,
            viewer_flags={'window_title': phase_label},
        )

    # ---- public ----

    @property
    def selected_points(self):
        return list(self._selected)

    @property
    def aborted(self):
        return self._aborted

    # ---- callbacks ----

    def _cb_undo(self, viewer=None):
        if self._selected:
            self._selected.pop()
            print(f"  Undone. {len(self._selected)} point(s) remain.")
        else:
            print("  Nothing to undo.")

    def _cb_abort(self, viewer=None):
        print("  Aborting.")
        self._aborted = True
        self._selected.clear()

    # ---- picking ----

    def _do_pick(self, viewer, px, py):
        """Cast a ray from pixel (px, py) and record the first mesh hit."""
        try:
            origin, direction = self._pixel_to_ray(viewer, px, py)
            locs, _, _ = self._intersector.intersects_location(
                ray_origins=origin[None],
                ray_directions=direction[None],
                multiple_hits=False,
            )
            if len(locs):
                pt = locs[0].tolist()
                self._selected.append(pt)
                print(f"  [{len(self._selected)}] "
                      f"({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})")
                self._add_marker(viewer, np.array(pt))
            else:
                print("  (no hit — click directly on the visible mesh surface)")
        except Exception as exc:
            print(f"  (pick error: {exc})")

    def _pixel_to_ray(self, viewer, px, py):
        """Convert pyrender viewport pixel to world-space (origin, direction).

        pyrender / pyglet: origin is bottom-left; y increases upward.
        Camera looks down its local -Z axis (OpenGL convention).
        """
        import pyrender
        T    = viewer._camera_node.matrix        # camera-to-world, (4, 4)
        w, h = viewer.viewport_size
        cam  = viewer._camera_node.camera

        yfov   = float(cam.yfov)                 # radians
        aspect = w / h
        half_h = np.tan(yfov / 2.0)
        half_w = half_h * aspect

        ndx = (2.0 * px / w) - 1.0              # [-1, 1]
        ndy = (2.0 * py / h) - 1.0

        d_cam = np.array([ndx * half_w, ndy * half_h, -1.0])
        d_cam /= np.linalg.norm(d_cam)

        origin    = T[:3, 3].copy()
        direction = T[:3, :3] @ d_cam
        direction /= np.linalg.norm(direction)
        return origin, direction

    def _add_marker(self, viewer, pt):
        """Add a small red sphere at pt in the pyrender scene."""
        try:
            import pyrender
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.005)
            sphere.visual.face_colors = np.array([255, 60, 60, 220], dtype=np.uint8)
            T = np.eye(4); T[:3, 3] = pt
            sphere.apply_transform(T)
            viewer.scene.add(pyrender.Mesh.from_trimesh(sphere, smooth=False))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Per-phase helper
# ---------------------------------------------------------------------------

def pick_surface(trimesh_mesh, phase_label):
    """Open an interactive viewer for one mesh; return list of selected points."""
    v = PickingViewer(trimesh_mesh, phase_label)
    if v.aborted:
        return None
    return v.selected_points


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
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute T_WRIST_TO_HM from existing annotation without re-annotating")
    args = parser.parse_args()

    if args.recompute:
        # ---- Recompute from saved annotation (no interactive session needed) ----
        if not args.out.exists():
            print(f"ERROR: no annotation found at {args.out}")
            print("Run without --recompute first to create an annotation.")
            sys.exit(1)
        print(f"Loading existing annotation from {args.out} …")
        existing    = json.loads(args.out.read_text())
        T_wrist     = np.array(existing["T_wrist"], dtype=float)
        g_front_pts = np.array(existing["glove_front"]["points"], dtype=float)
        g_back_pts  = np.array(existing["glove_back"]["points"],  dtype=float)
        g_top_pts   = np.array(
            existing.get("glove_top", existing["glove_front"])["points"], dtype=float
        )
        h_front_pts = np.array(existing["hand_front"]["points"], dtype=float)
        h_back_pts  = np.array(existing["hand_back"]["points"],  dtype=float)
        frame_idx   = existing["frame"]
    else:
        # ---- Interactive annotation ----
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

        T_hand_mount = frame_data["T_wrist"] @ cfg.T_WRIST_TO_HM
        glove_scene  = get_glove_scene(robot, joint_cfg, T_hand_mount)
        glove_mesh   = trimesh.util.concatenate(list(glove_scene.geometry.values()))

        hand_glb = cfg.GLB_DIR / f"{args.frame:06d}_hands.glb"
        if not hand_glb.exists():
            print(f"ERROR: hand GLB not found: {hand_glb}")
            print("Run align_frame.py first to generate the hand GLB.")
            sys.exit(1)
        hand_scene = trimesh.load(str(hand_glb), force="scene")
        hand_mesh  = trimesh.util.concatenate(list(hand_scene.geometry.values()))

        print(f"Glove mesh: {len(glove_mesh.faces)} faces")
        print(f"Hand  mesh: {len(hand_mesh.faces)} faces")

        def _require(pts, phase):
            if not pts or len(pts) < 3:
                print(f"Annotation aborted or too few points in {phase}.  Exiting.")
                sys.exit(1)
            return np.array(pts, dtype=float)

        # Phase 1: glove FRONT (knuckle-side edge of flat bottom)
        print("\nPhase 1 of 4: GLOVE FRONT EDGE")
        print("Click the knuckle-side edge of the flat underside of the Hand Mount.")
        g_front_pts = _require(
            pick_surface(glove_mesh,
                         "Phase 1/4 — GLOVE FRONT: knuckle-side edge of flat bottom"),
            "Phase 1")

        # Phase 2: glove BACK (wrist-side edge of flat bottom)
        print("\nPhase 2 of 4: GLOVE BACK EDGE")
        print("Click the wrist-side edge of the flat underside of the Hand Mount.")
        g_back_pts = _require(
            pick_surface(glove_mesh,
                         "Phase 2/4 — GLOVE BACK: wrist-side edge of flat bottom"),
            "Phase 2")

        # Phase 3: hand FRONT (dorsal above knuckles)
        print("\nPhase 3 of 4: HAND DORSAL FRONT")
        print("Click the dorsal hand surface just above the knuckles.")
        h_front_pts = _require(
            pick_surface(hand_mesh,
                         "Phase 3/4 — HAND FRONT: dorsal surface above knuckles"),
            "Phase 3")

        # Phase 4: hand BACK (dorsal near wrist)
        print("\nPhase 4 of 4: HAND DORSAL BACK")
        print("Click the dorsal hand surface near the wrist.")
        h_back_pts = _require(
            pick_surface(hand_mesh,
                         "Phase 4/4 — HAND BACK: dorsal surface near wrist"),
            "Phase 4")

        T_wrist   = np.array(frame_data["T_wrist"], dtype=float)
        g_top_pts = g_front_pts   # unused placeholder
        frame_idx = args.frame

    # ---- Compute T_WRIST_TO_HM ----
    T_offset, g_centroid, g_normal, h_centroid, h_normal = compute_T_wrist_to_hm(
        T_wrist, cfg.T_WRIST_TO_HM,
        g_front_pts, g_back_pts, g_top_pts, h_front_pts, h_back_pts)

    cos_a = abs(float(np.dot(g_normal, h_normal)))
    angle = float(np.degrees(np.arccos(min(1.0, cos_a))))
    gap_m = abs(float(np.dot(g_centroid - h_centroid, h_normal)))

    print(f"\n{'='*60}")
    print(f"Glove bottom centroid : {g_centroid}")
    print(f"Glove bottom normal   : {g_normal}")
    print(f"Hand dorsal  centroid : {h_centroid}")
    print(f"Hand dorsal  normal   : {h_normal}")
    print(f"Angle between normals : {angle:.1f}°  (target < 15°)")
    print(f"Gap between planes    : {gap_m*1000:.1f} mm  (target < 5 mm)")

    print(f"\nComputed T_WRIST_TO_HM:")
    print("\nPaste into glove_sim/config.py as:")
    print("  T_WRIST_TO_HM = np.array(")
    for row in T_offset:
        print(f"      {row.tolist()},")
    print("  )")
    print(f"{'='*60}")

    # ---- Save ----
    annotation = {
        "frame": frame_idx,
        "T_wrist": T_wrist.tolist(),
        "glove_front": {"points": g_front_pts.tolist()},
        "glove_back":  {"points": g_back_pts.tolist()},
        "hand_front":  {"points": h_front_pts.tolist()},
        "hand_back":   {"points": h_back_pts.tolist()},
        # Combined fits (used by test_plane_alignment.py)
        "glove_bottom": {
            "points":   np.vstack([g_front_pts, g_back_pts]).tolist(),
            "centroid": g_centroid.tolist(),
            "normal":   g_normal.tolist(),
        },
        "hand_dorsal": {
            "points":   np.vstack([h_front_pts, h_back_pts]).tolist(),
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

#!/usr/bin/env python3
"""
glove_sim/tools/annotate_fingertips.py

Two-phase annotation tool for fingertip IK calibration.

Phase 1: Pick the INDEX cap centre, then THUMB cap centre on the rendered glove.
Phase 2: Pick the INDEX fingertip,  then THUMB fingertip  on the hand mesh.
         (same order as Phase 1)

The MANO model's fingertip regressor can be slightly off relative to the actual
mesh vertices.  This tool measures that offset (in wrist-local frame) so that
align_frame.py can apply a per-finger correction to every IK target in the sequence.

Controls (both phases):
  Right-click (no drag) — add a point on the mesh surface
  Left-drag             — rotate camera
  SHIFT+Left-drag       — pan camera
  Scroll / Right-drag   — zoom
  U                     — undo last point
  Enter                 — confirm (need exactly 2 points)
  Q                     — abort

Usage:
  python glove_sim/tools/annotate_fingertips.py [--frame 300]
  python glove_sim/tools/annotate_fingertips.py --recompute
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

_MANO_TO_GLB = np.diag([1.0, -1.0, -1.0, 1.0])


def _load_right_hand_mesh(glb_path: Path, mano_centroid_world: np.ndarray) -> "trimesh.Trimesh":
    """Return only the right-hand sub-mesh from a DynHaMR unity_export GLB."""
    scene = trimesh.load(str(glb_path), force="scene")
    geoms = list(scene.geometry.values())
    if len(geoms) == 1:
        return geoms[0]
    expected = (_MANO_TO_GLB @ np.append(mano_centroid_world, 1.0))[:3]
    best_idx = min(range(len(geoms)),
                   key=lambda i: np.linalg.norm(geoms[i].vertices.mean(axis=0) - expected))
    return geoms[best_idx]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_fingertip_offsets(T_wrist,
                               index_tip_regressor, thumb_tip_regressor,
                               index_tip_glb, thumb_tip_glb):
    """Compute per-finger IK target offsets in wrist-local frame.

    The corrected IK target for a future frame is:
        adjusted_tip_world = frame_tip_world + T_wrist[:3,:3] @ offset_wrist_local

    Parameters
    ----------
    T_wrist               : (4,4) wrist world transform (MANO / Y-down)
    index_tip_regressor   : (3,) MANO regressor index tip — MANO world space
    thumb_tip_regressor   : (3,) MANO regressor thumb tip — MANO world space
    index_tip_glb         : (3,) annotated index tip on hand mesh — GLB space
    thumb_tip_glb         : (3,) annotated thumb tip on hand mesh — GLB space

    Returns
    -------
    index_offset_wrist : (3,) correction offset in wrist-local frame
    thumb_offset_wrist : (3,) correction offset in wrist-local frame
    """
    # GLB → MANO world (_MANO_TO_GLB is self-inverse: diag([1,1,-1,1]))
    index_tip_mano = (_MANO_TO_GLB @ np.append(index_tip_glb, 1.0))[:3]
    thumb_tip_mano = (_MANO_TO_GLB @ np.append(thumb_tip_glb,  1.0))[:3]

    # Delta in world space → rotate to wrist-local frame
    R_wrist_inv = T_wrist[:3, :3].T          # R.T == R^{-1} for rotation matrices
    index_offset_wrist = R_wrist_inv @ (index_tip_mano - index_tip_regressor)
    thumb_offset_wrist = R_wrist_inv @ (thumb_tip_mano  - thumb_tip_regressor)

    return index_offset_wrist, thumb_offset_wrist


# ---------------------------------------------------------------------------
# Viewer with right-click picking (pyrender / pyglet 2.x compatible)
# ---------------------------------------------------------------------------

class PickingViewer:
    """Interactive mesh viewer — collects exactly N_REQUIRED right-click hits.

    Blocks until the window is closed (Enter = confirm, Q = abort).
    """

    def __init__(self, trimesh_mesh, phase_label, n_required=2):
        import pyrender

        self._selected   = []
        self._aborted    = False
        self._n_required = n_required
        self._intersector = trimesh.ray.ray_triangle.RayMeshIntersector(trimesh_mesh)

        print(f"\n{'='*60}")
        print(f"  {phase_label}")
        print(f"  Left-drag             : rotate camera")
        print(f"  SHIFT+Left-drag       : pan camera")
        print(f"  Scroll / Right-drag   : zoom")
        print(f"  Right-click (no drag) : ADD POINT on surface")
        print(f"  U                     : undo last point")
        print(f"  Enter                 : confirm (need exactly {n_required} points)")
        print(f"  Q                     : abort")
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
                    if n == picking_self._n_required:
                        print(f"  Confirmed {n} points.")
                        self_.on_close()
                    else:
                        print(f"  Need exactly {picking_self._n_required} points "
                              f"(have {n}).")
                elif symbol == _key.U:
                    if picking_self._selected:
                        picking_self._selected.pop()
                        print(f"  Undone. {len(picking_self._selected)} point(s) remain.")
                    else:
                        print("  Nothing to undo.")
                elif symbol == _key.Q:
                    picking_self._aborted = True
                    picking_self._selected.clear()
                    self_.on_close()
                else:
                    super().on_key_press(symbol, modifiers)

        self._viewer = _PV(pr_scene, use_raymond_lighting=True,
                           viewer_flags={'window_title': phase_label})

    @property
    def selected_points(self):
        return list(self._selected)

    @property
    def aborted(self):
        return self._aborted

    def _do_pick(self, viewer, px, py):
        try:
            origin, direction = self._pixel_to_ray(viewer, px, py)
            locs, _, _ = self._intersector.intersects_location(
                ray_origins=origin[None],
                ray_directions=direction[None],
                multiple_hits=False,
            )
            if len(locs):
                pt  = locs[0].tolist()
                idx = len(self._selected) + 1
                labels = ["index-cap", "thumb-cap",
                          "index-tip", "thumb-tip"]
                label  = labels[idx - 1] if idx <= len(labels) else f"pt{idx}"
                self._selected.append(pt)
                print(f"  [{idx}] {label}: "
                      f"({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})")
                self._add_marker(viewer, np.array(pt))
                if len(self._selected) == self._n_required:
                    print(f"  All {self._n_required} points collected — "
                          f"press Enter to confirm.")
            else:
                print("  (no hit — click directly on the visible mesh surface)")
        except Exception as exc:
            print(f"  (pick error: {exc})")

    def _pixel_to_ray(self, viewer, px, py):
        T    = viewer._camera_node.matrix
        w, h = viewer.viewport_size
        cam  = viewer._camera_node.camera
        half_h = np.tan(float(cam.yfov) / 2.0)
        half_w = half_h * w / h
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


def pick_surface(mesh, label, n_required=2):
    """Open viewer; return list of n_required points, or None if aborted."""
    v = PickingViewer(mesh, label, n_required=n_required)
    return None if v.aborted else v.selected_points


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Annotate finger cap + fingertip correspondences for IK calibration."
    )
    parser.add_argument("--frame", type=int, default=300)
    parser.add_argument("--out",   type=Path,
                        default=cfg.ALIGNED_DIR / "fingertip_annotation.json")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute offsets from saved annotation (no re-annotation)")
    args = parser.parse_args()

    # ---- Always need frame data for the MANO regressor positions ----
    print("Loading MANO frame data …")
    from src.mano_io import load_frame
    frame_data = load_frame(cfg.NPZ_PATH, cfg.MANO_DIR,
                            args.frame if not args.recompute else
                            (json.loads(args.out.read_text())["frame"]
                             if args.out.exists() else args.frame))
    T_wrist             = np.array(frame_data["T_wrist"],   dtype=float)
    index_tip_regressor = np.array(frame_data["index_tip"], dtype=float)
    thumb_tip_regressor = np.array(frame_data["thumb_tip"], dtype=float)

    if args.recompute:
        if not args.out.exists():
            print(f"ERROR: no annotation at {args.out}")
            sys.exit(1)
        print(f"Loading existing annotation from {args.out} …")
        existing      = json.loads(args.out.read_text())
        index_cap_glb = np.array(existing["index_cap_glb"], dtype=float)
        thumb_cap_glb = np.array(existing["thumb_cap_glb"], dtype=float)
        index_tip_glb = np.array(existing["index_tip_glb"], dtype=float)
        thumb_tip_glb = np.array(existing["thumb_tip_glb"], dtype=float)
        frame_idx     = existing["frame"]
    else:
        # ---- Build meshes ----
        from src.urdfpy_vis import load_robot, get_glove_scene
        from align_frame    import solve_ik_frame

        robot     = load_robot(cfg.URDF_PATH, cfg.MESH_DIR)
        joint_cfg, _, _ = solve_ik_frame(
            robot, T_wrist,
            frame_data["thumb_tip"],
            frame_data["index_tip"],
        )
        T_hand_mount = T_wrist @ cfg.T_WRIST_TO_HM
        glove_scene  = get_glove_scene(robot, joint_cfg, T_hand_mount)
        glove_mesh   = trimesh.util.concatenate(list(glove_scene.geometry.values()))

        hand_glb = cfg.GLB_DIR / f"{args.frame:06d}_hands.glb"
        if not hand_glb.exists():
            print(f"ERROR: hand GLB not found: {hand_glb}")
            sys.exit(1)
        hand_mesh = _load_right_hand_mesh(hand_glb, mano_centroid_world=T_wrist[:3, 3])

        print(f"Glove mesh: {len(glove_mesh.faces)} faces")
        print(f"Hand  mesh: {len(hand_mesh.faces)} faces")

        def _require(pts, phase):
            if not pts or len(pts) != 2:
                print(f"Annotation aborted or wrong point count in {phase}.  Exiting.")
                sys.exit(1)
            return [np.array(p, dtype=float) for p in pts]

        # Phase 1 — glove caps
        print("\n" + "="*60)
        print("PHASE 1 — FINGER CAP CENTRES on the glove")
        print("Pick exactly 2 points IN THIS ORDER:")
        print("  1. Centre of the INDEX finger cap")
        print("  2. Centre of the THUMB  finger cap")
        caps = _require(
            pick_surface(glove_mesh,
                         "Phase 1/2 — GLOVE: index-cap centre, thumb-cap centre"),
            "Phase 1")
        index_cap_glb, thumb_cap_glb = caps[0], caps[1]

        # Phase 2 — hand tips
        print("\n" + "="*60)
        print("PHASE 2 — FINGERTIPS on the hand mesh (same order)")
        print("Pick exactly 2 points IN THIS ORDER:")
        print("  1. Tip of the INDEX finger")
        print("  2. Tip of the THUMB  finger")
        tips = _require(
            pick_surface(hand_mesh,
                         "Phase 2/2 — HAND: index fingertip, thumb fingertip"),
            "Phase 2")
        index_tip_glb, thumb_tip_glb = tips[0], tips[1]
        frame_idx = args.frame

    # ---- Compute offsets ----
    index_offset_wrist, thumb_offset_wrist = compute_fingertip_offsets(
        T_wrist,
        index_tip_regressor, thumb_tip_regressor,
        index_tip_glb, thumb_tip_glb,
    )

    # ---- Diagnostics ----
    index_corr_mano = (_MANO_TO_GLB @ np.append(index_tip_glb, 1.0))[:3]
    thumb_corr_mano = (_MANO_TO_GLB @ np.append(thumb_tip_glb,  1.0))[:3]

    print(f"\n{'='*60}")
    print(f"Index: MANO regressor  → {index_tip_regressor.round(4)}")
    print(f"       annotated tip   → {index_corr_mano.round(4)}")
    print(f"       offset (wrist)  → {index_offset_wrist.round(4)}")
    print(f"       |offset|        = {np.linalg.norm(index_offset_wrist)*1000:.1f} mm")
    print()
    print(f"Thumb:  MANO regressor  → {thumb_tip_regressor.round(4)}")
    print(f"        annotated tip   → {thumb_corr_mano.round(4)}")
    print(f"        offset (wrist)  → {thumb_offset_wrist.round(4)}")
    print(f"        |offset|        = {np.linalg.norm(thumb_offset_wrist)*1000:.1f} mm")
    print(f"{'='*60}")

    # ---- Save ----
    annotation = {
        "frame":                     frame_idx,
        "T_wrist":                   T_wrist.tolist(),
        "index_cap_glb":             index_cap_glb.tolist(),
        "thumb_cap_glb":             thumb_cap_glb.tolist(),
        "index_tip_glb":             index_tip_glb.tolist(),
        "thumb_tip_glb":             thumb_tip_glb.tolist(),
        "index_tip_offset_wrist":    index_offset_wrist.tolist(),
        "thumb_tip_offset_wrist":    thumb_offset_wrist.tolist(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(annotation, indent=2))
    print(f"\nAnnotation saved → {args.out}")
    print("config.py will auto-load the offsets on next import.")
    print("Re-run:  python glove_sim/align_frame.py --frames 300  to verify.")


if __name__ == "__main__":
    main()

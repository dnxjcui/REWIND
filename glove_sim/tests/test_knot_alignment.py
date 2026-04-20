import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestKnotAlignment(unittest.TestCase):
    def _scene_metrics(self, scene: trimesh.Scene) -> dict:
        geoms = []
        for _, geom in scene.geometry.items():
            geoms.append(
                {
                    "n_vertices": int(len(geom.vertices)),
                    "n_faces": int(len(geom.faces)),
                    "centroid": geom.centroid.tolist(),
                    "extents": geom.extents.tolist(),
                }
            )
        bounds = scene.bounds if scene.bounds is not None else np.zeros((2, 3))
        return {"n_geom": len(geoms), "bounds_min": bounds[0], "bounds_max": bounds[1], "geom": geoms}

    def test_compute_alignment_metrics_shapes(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "glove_sim"))
        from knot_alignment import _compute_frame_metrics

        hand_pos = np.array([0.0, 0.0, 0.0], dtype=float)
        hand_z = np.array([0.0, 0.0, 1.0], dtype=float)
        horn_pos = np.array([0.05, 0.0, 0.0], dtype=float)
        wrist = np.array([0.0, 0.0, 0.0], dtype=float)
        index_knuckle = np.array([0.06, 0.0, 0.0], dtype=float)
        pinky_knuckle = np.array([0.0, 0.04, 0.0], dtype=float)

        m = _compute_frame_metrics(
            hand_pos=hand_pos,
            hand_z_axis=hand_z,
            horn_pos=horn_pos,
            wrist=wrist,
            index_knuckle=index_knuckle,
            pinky_knuckle=pinky_knuckle,
        )
        self.assertIn("flush_distance_m", m)
        self.assertIn("orientation_error", m)
        self.assertIn("horn_to_knuckle_distance_m", m)
        self.assertTrue(np.isfinite(m["flush_distance_m"]))
        self.assertTrue(np.isfinite(m["orientation_error"]))
        self.assertTrue(np.isfinite(m["horn_to_knuckle_distance_m"]))

    def test_alignment_script_emits_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "align"
            cmd = [
                sys.executable,
                str(REPO_ROOT / "glove_sim" / "knot_alignment.py"),
                "--max-frames",
                "6",
                "--max-iters",
                "2000",
                "--out-dir",
                str(out_dir),
            ]
            # Preserve environment while allowing OpenMP runtime coexistence in test subprocess.
            env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
            res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
            self.assertEqual(0, res.returncode, msg=res.stderr or res.stdout)
            report = out_dir / "report.json"
            self.assertTrue(report.is_file(), "alignment report must be generated")
            data = json.loads(report.read_text(encoding="utf-8"))
            self.assertIn("optimized_t_wrist_to_base", data)
            self.assertIn("summary", data)
            self.assertIn("frames", data)
            self.assertEqual(6, len(data["frames"]), "flush-only scope uses exactly 6 sparse keyframes")
            self.assertLessEqual(
                float(data["summary"]["flush_max_m"]),
                0.005,
                "hand-mount must be flush (<= 5 mm) over sparse keyframes",
            )

    def test_overlay_export_glove_connectivity_matches_glove_only_export(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "glove_sim"))
        import config as cfg
        from src.glove_ik import GloveSimulator
        from src.visualize import export_frame_glb, export_glove_only_glb

        sim = GloveSimulator(cfg.URDF_PATH, cfg.MESH_DIR)
        # Use any valid base pose; internal linkage should stay consistent.
        sim.set_base_pose(np.array([-0.02, 0.06, -0.01], dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
        import mujoco
        mujoco.mj_forward(sim.model, sim.data)
        geom_poses = sim.get_geom_world_poses()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            overlay = tmp_path / "overlay.glb"
            glove_only = tmp_path / "glove_only.glb"
            # Point to empty hand-glb dir so export_frame_glb contains only glove.
            empty_glb_dir = tmp_path / "empty"
            empty_glb_dir.mkdir(parents=True, exist_ok=True)
            export_frame_glb(
                frame_idx=0,
                geom_poses=geom_poses,
                out_path=overlay,
                mesh_dir=cfg.MESH_DIR,
                glb_dir=empty_glb_dir,
                sensor_positions_ydown=None,
            )
            export_glove_only_glb(geom_poses=geom_poses, out_path=glove_only, mesh_dir=cfg.MESH_DIR)

            s_overlay = trimesh.load(str(overlay), force="scene")
            s_glove = trimesh.load(str(glove_only), force="scene")
            m_o = self._scene_metrics(s_overlay)
            m_g = self._scene_metrics(s_glove)
            # If transforms are coherent, these two exports should be near-identical.
            delta = max(
                float(np.max(np.abs(np.array(m_o["bounds_min"]) - np.array(m_g["bounds_min"])))),
                float(np.max(np.abs(np.array(m_o["bounds_max"]) - np.array(m_g["bounds_max"])))),
            )
            self.assertLessEqual(
                delta,
                1e-6,
                "overlay glove transforms diverge from glove-only export; indicates connectivity/frame mismatch",
            )

    def test_annotation_schema_validation(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "glove_sim"))
        from knot_alignment import load_annotations

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ann.json"
            valid = {
                "schema_version": 1,
                "reference_frame_idx": 119,
                "hand_points": [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
                "glove_points": [[0.0, 0.0, 0.005], [0.01, 0.0, 0.005], [0.0, 0.01, 0.005]],
            }
            p.write_text(json.dumps(valid), encoding="utf-8")
            out = load_annotations(p)
            self.assertEqual(3, len(out["hand_points"]))
            self.assertEqual(3, len(out["glove_points"]))

            invalid = {
                "schema_version": 1,
                "reference_frame_idx": 119,
                "hand_points": [[0.0, 0.0, 0.0]],
                "glove_points": [],
            }
            p.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_annotations(p)

    def test_connectivity_validation_rejects_missing_required_body(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "glove_sim"))
        from knot_alignment import validate_connectivity_graph

        # hand_mount root with one disconnected body should fail.
        parents = {
            "hand_mount": None,
            "part_1": "hand_mount",
            "xl_linkage_horn_1": "orphan",
        }
        with self.assertRaises(ValueError):
            validate_connectivity_graph(parents, required_bodies=["part_1", "xl_linkage_horn_1"])

    def test_runtime_connectivity_check_passes_for_model(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "glove_sim"))
        import config as cfg
        from knot_alignment import _enforce_model_connectivity
        from src.glove_ik import GloveSimulator

        sim = GloveSimulator(cfg.URDF_PATH, cfg.MESH_DIR)
        _enforce_model_connectivity(sim)


if __name__ == "__main__":
    unittest.main()

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from glove_sim.mujoco_device_setup import parse_vec3

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestMujocoDeviceSetup(unittest.TestCase):
    """Unit tests for CLI parsing utilities in MuJoCo bring-up tool."""

    def test_parse_vec3_valid(self) -> None:
        out = parse_vec3("1.0, 2.5, -3")
        self.assertTrue(np.allclose(out, np.array([1.0, 2.5, -3.0])))

    def test_parse_vec3_invalid_number_of_fields(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_vec3("1.0,2.0")

    def test_parse_vec3_non_numeric(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_vec3("a,b,c")

    def test_setup_script_writes_rest_parity_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_glb = Path(tmp) / "rest.glb"
            report = Path(tmp) / "report.json"
            cmd = [
                sys.executable,
                str(REPO_ROOT / "glove_sim" / "mujoco_device_setup.py"),
                "--export-glb",
                str(out_glb),
                "--report",
                str(report),
            ]
            res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
            self.assertEqual(0, res.returncode, msg=res.stderr or res.stdout)
            self.assertTrue(out_glb.is_file(), "MuJoCo rest GLB should be created.")
            self.assertTrue((out_glb.parent / "native_rest.glb").is_file(), "Native rest GLB should be created.")
            self.assertTrue(report.is_file(), "Report JSON should be created.")
            data = json.loads(report.read_text(encoding="utf-8"))
            self.assertIn("rest_parity", data)
            self.assertTrue(data["rest_parity"]["ok"], "MuJoCo rest export should match native URDFpy rest export.")

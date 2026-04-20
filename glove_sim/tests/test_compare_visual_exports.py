import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestCompareVisualExports(unittest.TestCase):
    def test_rest_first_and_small_sweep_pass(self) -> None:
        import sys

        sys.path.insert(0, str(REPO_ROOT / "glove_sim"))
        from compare_visual_exports import run_parity_export_compare

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "parity"
            report = run_parity_export_compare(
                out_dir=out_dir,
                rest_atol=1e-8,
                sweep_atol=1e-7,
                only_rest=False,
            )
            self.assertTrue(report["rest_ok"], "Rest pose must pass before sweep.")
            self.assertTrue(report["ok"], "Full small sweep must pass.")
            self.assertTrue((out_dir / "report.json").is_file(), "report.json should be written.")
            self.assertGreaterEqual(len(report["results"]), 4, "Expected rest + additional sweep configs.")

    def test_only_rest_mode(self) -> None:
        import sys

        sys.path.insert(0, str(REPO_ROOT / "glove_sim"))
        from compare_visual_exports import run_parity_export_compare

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "parity"
            report = run_parity_export_compare(
                out_dir=out_dir,
                rest_atol=1e-8,
                sweep_atol=1e-7,
                only_rest=True,
            )
            self.assertIsNotNone(report["rest_ok"])
            self.assertEqual(len(report["results"]), 1, "only-rest mode should evaluate a single config.")


if __name__ == "__main__":
    unittest.main()

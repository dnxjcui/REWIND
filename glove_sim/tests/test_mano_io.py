import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def test_load_frame_shapes():
    from src.mano_io import load_frame
    result = load_frame(cfg.NPZ_PATH, cfg.MANO_DIR, 0)
    assert result["T_wrist"].shape == (4, 4)
    assert result["thumb_tip"].shape == (3,)
    assert result["index_tip"].shape == (3,)


def test_load_frame_wrist_is_valid_rotation():
    from src.mano_io import load_frame
    R = load_frame(cfg.NPZ_PATH, cfg.MANO_DIR, 0)["T_wrist"][:3, :3]
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-6)
    assert abs(np.linalg.det(R) - 1.0) < 1e-6


def test_load_frame_fingertips_nonzero():
    from src.mano_io import load_frame
    result = load_frame(cfg.NPZ_PATH, cfg.MANO_DIR, 0)
    assert np.linalg.norm(result["thumb_tip"]) > 0.01
    assert np.linalg.norm(result["index_tip"]) > 0.01


def test_load_frame_300():
    from src.mano_io import load_frame
    result = load_frame(cfg.NPZ_PATH, cfg.MANO_DIR, 300)
    assert result["T_wrist"].shape == (4, 4)
    assert result["thumb_tip"].dtype == np.float64

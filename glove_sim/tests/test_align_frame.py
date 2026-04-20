import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.urdfpy_vis import load_robot
from src.mano_io import load_frame


@pytest.fixture(scope="module")
def robot():
    return load_robot(cfg.URDF_PATH, cfg.MESH_DIR)


@pytest.mark.parametrize("frame_idx", [0, 300])
def test_ik_residuals_below_5mm(robot, frame_idx):
    from align_frame import solve_ik_frame
    frame = load_frame(cfg.NPZ_PATH, cfg.MANO_DIR, frame_idx)
    joint_cfg, thumb_res, index_res = solve_ik_frame(
        robot, frame["T_wrist"], frame["thumb_tip"], frame["index_tip"]
    )
    assert thumb_res < 0.005, \
        f"Frame {frame_idx} thumb: {thumb_res*1000:.2f}mm > 5mm"
    assert index_res < 0.005, \
        f"Frame {frame_idx} index: {index_res*1000:.2f}mm > 5mm"


@pytest.mark.parametrize("frame_idx", [0, 300])
def test_joint_angles_in_bounds(robot, frame_idx):
    from align_frame import solve_ik_frame
    frame = load_frame(cfg.NPZ_PATH, cfg.MANO_DIR, frame_idx)
    joint_cfg, _, _ = solve_ik_frame(
        robot, frame["T_wrist"], frame["thumb_tip"], frame["index_tip"]
    )
    for name, angle in joint_cfg.items():
        assert -np.pi <= angle <= np.pi, \
            f"Frame {frame_idx} {name}: {np.degrees(angle):.1f}° out of bounds"


def test_warmstart_preserves_low_residual(robot):
    from align_frame import solve_ik_frame
    import numpy as np
    frame0 = load_frame(cfg.NPZ_PATH, cfg.MANO_DIR, 0)
    cfg0, tr0, ir0 = solve_ik_frame(
        robot, frame0["T_wrist"], frame0["thumb_tip"], frame0["index_tip"]
    )
    _THUMB_CHAIN = ['revolute_1_0', 'revolute_2_0', 'revolute_3_0', 'revolute_4_0']
    _INDEX_CHAIN = ['revolute_5_0', 'revolute_6_0', 'revolute_7_0', 'revolute_8_0', 'revolute_9_0']
    prev_q_thumb = np.array([cfg0[j] for j in _THUMB_CHAIN])
    prev_q_index = np.array([cfg0[j] for j in _INDEX_CHAIN])

    frame300 = load_frame(cfg.NPZ_PATH, cfg.MANO_DIR, 300)
    _, tr300, ir300 = solve_ik_frame(
        robot, frame300["T_wrist"], frame300["thumb_tip"], frame300["index_tip"],
        prev_q_thumb=prev_q_thumb, prev_q_index=prev_q_index,
    )
    assert tr300 < 0.005
    assert ir300 < 0.005

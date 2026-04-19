"""Post-process and save simulated sensor data."""

import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter

from config import FPS, SMOOTH_WINDOW, IK_RESIDUAL_THRESHOLD


def save_sensor_data(
    sensor_angles: np.ndarray,
    fingertip_positions: np.ndarray,
    ik_residuals: np.ndarray,
    calibration: np.ndarray,
    out_path: str | Path,
) -> None:
    """Smooth, differentiate, and save all sensor outputs.

    Parameters
    ----------
    sensor_angles      : (T, 4) raw joint angles [thumb_base, thumb_tip, index_base, index_tip]
    fingertip_positions: (T, 2, 3) world positions [thumb_tip, index_tip]
    ik_residuals       : (T,) IK position error in meters after convergence
    calibration        : (4, 4) T_wrist_to_base transform used
    out_path           : path for the output .npz file
    """
    T = sensor_angles.shape[0]
    dt = 1.0 / FPS

    # Smooth angles with Savitzky-Golay before differentiating
    wl = SMOOTH_WINDOW if T >= SMOOTH_WINDOW else (T if T % 2 == 1 else T - 1)
    if wl >= 3:
        angles_smooth = savgol_filter(sensor_angles, window_length=wl, polyorder=2, axis=0)
    else:
        angles_smooth = sensor_angles.copy()

    velocities     = np.gradient(angles_smooth, dt, axis=0)
    accelerations  = np.gradient(velocities,    dt, axis=0)

    tip_dist = np.linalg.norm(
        fingertip_positions[:, 0] - fingertip_positions[:, 1], axis=1
    )
    reliability_mask = ik_residuals < IK_RESIDUAL_THRESHOLD

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(out_path),
        timestamps=np.arange(T, dtype=np.float32) / FPS,
        sensor_angles=angles_smooth.astype(np.float32),
        sensor_velocities=velocities.astype(np.float32),
        sensor_accelerations=accelerations.astype(np.float32),
        fingertip_positions=fingertip_positions.astype(np.float32),
        fingertip_distance=tip_dist.astype(np.float32),
        ik_residuals=ik_residuals.astype(np.float32),
        reliability_mask=reliability_mask,
        sensor_labels=np.array(["thumb_base", "thumb_tip", "index_base", "index_tip"]),
        calibration=calibration.astype(np.float64),
        fps=np.float32(FPS),
    )
    print(f"Saved sensor data → {out_path}")
    _print_summary(angles_smooth, ik_residuals, reliability_mask)


def _print_summary(angles, residuals, mask):
    print(f"  Frames total      : {len(angles)}")
    print(f"  Reliable frames   : {mask.sum()} ({100*mask.mean():.1f}%)")
    print(f"  IK residual median: {np.median(residuals)*1000:.2f} mm")
    print(f"  IK residual max   : {np.max(residuals)*1000:.2f} mm")
    labels = ["thumb_base", "thumb_tip", "index_base", "index_tip"]
    for i, lbl in enumerate(labels):
        mn, mx = angles[:, i].min(), angles[:, i].max()
        print(f"  {lbl:12s}: [{np.degrees(mn):.1f}°, {np.degrees(mx):.1f}°]")

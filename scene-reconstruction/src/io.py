"""
I/O utilities for loading Dyn-HaMR output files.

Coordinate conventions (important):
  cameras.json rotation/translation = c2w in Y-up world frame.
    save_camera_json applies T=diag(1,-1,-1) to convert from the internal
    Y-down optimization frame before writing. Load with load_cameras().

  world_results.npz cam_R/cam_t = w2c in Y-down optimization frame.
    Use directly to project MANO vertices (which are also in Y-down frame).
    DO NOT mix with cameras.json without an axis flip.
"""

import glob
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np


# ── Public loaders ────────────────────────────────────────────────────────────

def load_cameras(cameras_json_path: str):
    """
    Load cameras.json produced by Dyn-HaMR's save_camera_json().

    Returns
    -------
    Rs : (N, 3, 3) float64  c2w rotation matrices in Y-up world frame
    ts : (N, 3)    float64  camera positions in Y-up world (c2w translation)
    Ks : (N, 4)    float64  [fx, fy, cx, cy] per frame
    """
    with open(cameras_json_path) as f:
        d = json.load(f)

    N = len(d["rotation"])
    Rs = np.array(d["rotation"], dtype=np.float64).reshape(N, 3, 3)
    ts = np.array(d["translation"], dtype=np.float64)          # (N, 3)
    Ks = np.array(d["intrinsics"], dtype=np.float64)           # (N, 4)
    return Rs, ts, Ks


def load_world_results(npz_or_dir: str) -> dict:
    """
    Load the latest *_world_results.npz from a file path or phase directory.

    Keys returned (all numpy arrays):
        betas       (B, 10)
        pose_body   (B, T, 45)    axis-angle, 15 joints × 3
        root_orient (B, T, 3)
        trans       (B, T, 3)
        is_right    (B, T)        float32: 0=left, 1=right
        cam_R       (B, T, 3, 3) w2c rotation in Y-down optimization frame
        cam_t       (B, T, 3)    w2c translation in Y-down optimization frame
        intrins     (4,)          [fx, fy, cx, cy] shared across all frames/tracks
        world_scale float         defaults to 1.0 if absent from file
    """
    path = npz_or_dir
    if os.path.isdir(npz_or_dir):
        path = find_latest_npz(npz_or_dir)

    raw = np.load(path, allow_pickle=False)
    out = {k: raw[k].copy() for k in raw.files}

    # world_scale is absent from most real .npz files — default to 1.0
    if "world_scale" not in out:
        out["world_scale"] = np.array([[1.0]], dtype=np.float32)

    print(f"Loaded world results: {path}")
    B, T = out["trans"].shape[:2]
    print(f"  {B} track(s), {T} frame(s)")
    return out


def find_latest_npz(phase_dir: str) -> str:
    """Return path to the last *_world_results.npz in phase_dir (sorted)."""
    files = sorted(glob.glob(os.path.join(phase_dir, "*_world_results.npz")))
    if not files:
        raise FileNotFoundError(
            f"No *_world_results.npz found in {phase_dir}"
        )
    return files[-1]


def extract_frames_from_video(
    video_path: str,
    out_dir: str,
    overwrite: bool = False,
) -> list:
    """
    Extract frames from video using ffmpeg, matching Dyn-HaMR's 1-indexed
    naming convention: 000001.jpg, 000002.jpg, ...

    Returns sorted list of extracted frame paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    existing = get_frame_paths(out_dir)
    if existing and not overwrite:
        print(f"Frames already extracted ({len(existing)} files) in {out_dir}")
        return existing

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-start_number", "1",
        "-q:v", "2",
        os.path.join(out_dir, "%06d.jpg"),
    ]
    print("Extracting frames:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return get_frame_paths(out_dir)


def get_frame_paths(frames_dir: str, ext: str = "jpg") -> list:
    """Return sorted list of all frame image paths in frames_dir."""
    paths = sorted(glob.glob(os.path.join(frames_dir, f"*.{ext}")))
    if not paths:
        # also try png
        paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    return paths


# ── Convenience container ────────────────────────────────────────────────────

class DynHaMRData:
    """
    Bundles cameras.json + world_results.npz fields for easy per-frame access.

    Attributes
    ----------
    Rs          (N, 3, 3) c2w rotations in Y-up frame (from cameras.json)
    ts          (N, 3)    c2w translations in Y-up frame (from cameras.json)
    Ks          (N, 4)    intrinsics [fx,fy,cx,cy] from cameras.json
    cam_R       (B, T, 3, 3) w2c rotations in Y-down frame (from .npz)
    cam_t       (B, T, 3)    w2c translations in Y-down frame (from .npz)
    intrins     (4,)      shared intrinsics from .npz
    world_scale float
    B, T, N     int       number of tracks, frames, camera frames
    """

    def __init__(self, cameras_json: str, npz_or_dir: str):
        self.Rs, self.ts, self.Ks = load_cameras(cameras_json)
        wr = load_world_results(npz_or_dir)

        self.cam_R       = wr["cam_R"]        # (B, T, 3, 3)
        self.cam_t       = wr["cam_t"]        # (B, T, 3)
        self.intrins     = wr["intrins"]      # (4,)
        self.world_scale = float(wr["world_scale"].flat[0])
        self.pose_body   = wr["pose_body"]    # (B, T, 45)
        self.root_orient = wr["root_orient"]  # (B, T, 3)
        self.trans       = wr["trans"]        # (B, T, 3)
        self.betas       = wr["betas"]        # (B, 10)
        self.is_right    = wr["is_right"]     # (B, T)

        self.B, self.T = self.cam_R.shape[:2]
        self.N = len(self.Rs)

    def c2w_matrix_4x4(self, frame_idx: int) -> np.ndarray:
        """Return (4,4) c2w matrix in Y-up frame for frame_idx."""
        m = np.eye(4)
        m[:3, :3] = self.Rs[frame_idx]
        m[:3,  3] = self.ts[frame_idx]
        return m

    def frame_intrinsic_matrix(self, frame_idx: int) -> np.ndarray:
        """Return (3,3) K matrix for frame_idx (from cameras.json per-frame Ks)."""
        fx, fy, cx, cy = self.Ks[frame_idx]
        K = np.eye(3)
        K[0, 0] = fx; K[1, 1] = fy
        K[0, 2] = cx; K[1, 2] = cy
        return K

    def npz_intrinsic_matrix(self) -> np.ndarray:
        """Return (3,3) K matrix from .npz shared intrinsics."""
        fx, fy, cx, cy = self.intrins
        K = np.eye(3)
        K[0, 0] = fx; K[1, 1] = fy
        K[0, 2] = cx; K[1, 2] = cy
        return K

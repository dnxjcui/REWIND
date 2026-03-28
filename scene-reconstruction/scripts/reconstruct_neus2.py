#!/usr/bin/env python3
"""
NeuS2 scene reconstruction from Dyn-HaMR output.

Best for static or near-static scenes. Produces clean SDF meshes but requires
CUDA-compiled NeuS2 (see README / third-party/neus2/) and longer training time.

Usage
-----
conda activate scene-recon
python scripts/reconstruct_neus2.py \\
    --log_dir  ../Dyn-HaMR/outputs/logs/video-custom/2026-03-24/demo1-all-shot-0-0--1 \\
    --video    ../Dyn-HaMR/test/videos/demo1.mp4 \\
    --config   configs/neus2.yaml \\
    --out_dir  outputs/demo1_neus2/

Outputs:
  <out_dir>/
    images/           symlinks to selected frames
    transforms.json   NeuS2-compatible camera file
    scene_mesh.glb    extracted mesh in Y-up world frame
"""

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import cv2
import yaml

from src.io import load_cameras, get_frame_paths, extract_frames_from_video
from src.neus2_bridge import NeuS2Bridge


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _merge_cli(cfg: dict, args: argparse.Namespace) -> dict:
    if args.log_dir:
        cfg["dynhamr"]["log_dir"] = args.log_dir
    if args.video:
        cfg["dynhamr"]["video_path"] = args.video
    if args.frames_dir:
        cfg["dynhamr"]["frames_dir"] = args.frames_dir
    if args.out_dir:
        cfg["output"]["dir"] = args.out_dir
    if args.phase:
        cfg["dynhamr"]["phase"] = args.phase
    if args.stride is not None:
        cfg["frames"]["stride"] = args.stride
    if args.n_steps is not None:
        cfg["neus2"]["n_steps"] = args.n_steps
    if args.neus2_repo:
        cfg["neus2"]["repo_path"] = args.neus2_repo
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="NeuS2 scene reconstruction from Dyn-HaMR output"
    )
    parser.add_argument("--log_dir",    help="Dyn-HaMR run_opt output directory")
    parser.add_argument("--video",      help="Source video .mp4")
    parser.add_argument("--config",     default=str(_REPO_ROOT / "configs" / "neus2.yaml"))
    parser.add_argument("--frames_dir", help="Pre-extracted frames directory")
    parser.add_argument("--out_dir",    help="Output directory")
    parser.add_argument("--phase",      default=None)
    parser.add_argument("--stride",     type=int, default=None)
    parser.add_argument("--n_steps",    type=int, default=None)
    parser.add_argument("--neus2_repo", default=None, help="Path to NeuS2 repo")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    cfg = _merge_cli(cfg, args)

    log_dir = cfg["dynhamr"]["log_dir"]
    if not log_dir:
        parser.error("--log_dir is required")

    video_path  = cfg["dynhamr"]["video_path"]
    frames_dir  = cfg["dynhamr"]["frames_dir"]
    stride      = cfg["frames"]["stride"]
    start       = cfg["frames"]["start"]
    end         = cfg["frames"]["end"]

    out_dir = cfg["output"]["dir"] or os.path.join(log_dir, "neus2_reconstruction")
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Load cameras ─────────────────────────────────────────────────
    cameras_json = os.path.join(log_dir, "cameras.json")
    print(f"\n[1/3] Loading cameras from {cameras_json}")
    Rs, ts, Ks = load_cameras(cameras_json)
    print(f"  {len(Rs)} camera frames")

    # ── Step 2: Locate frames ────────────────────────────────────────────────
    print(f"\n[2/3] Preparing frames")
    if frames_dir:
        frame_paths = get_frame_paths(frames_dir)
        print(f"  Using pre-extracted: {len(frame_paths)} frames")
    elif video_path:
        tmp_frames = os.path.join(out_dir, "_frames")
        frame_paths = extract_frames_from_video(video_path, tmp_frames)
        print(f"  Extracted {len(frame_paths)} frames")
    else:
        parser.error("Either --video or --frames_dir is required")

    # Read image dimensions
    img0 = cv2.imread(frame_paths[0])
    img_h, img_w = img0.shape[:2]
    print(f"  Image dimensions: {img_w}x{img_h}")

    # ── Step 3: Run NeuS2 pipeline ───────────────────────────────────────────
    print(f"\n[3/3] Running NeuS2 pipeline")
    n2_cfg = cfg["neus2"]
    bridge = NeuS2Bridge(
        neus2_repo_path=n2_cfg.get("repo_path"),
        n_steps=n2_cfg["n_steps"],
        near=n2_cfg["near"],
        far=n2_cfg["far"],
        aabb_min=n2_cfg["aabb_min"],
        aabb_max=n2_cfg["aabb_max"],
        mesh_resolution=n2_cfg["mesh_resolution"],
    )

    output_glb = bridge.run_full_pipeline(
        Rs, ts, Ks, frame_paths, out_dir, img_w, img_h,
        stride=stride, start=start, end=end,
        mesh_filename=cfg["output"]["mesh_filename"],
    )

    print(f"\nDone. Output: {output_glb}")


if __name__ == "__main__":
    main()

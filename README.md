# REWIND — Senior Design Project

**REWIND** reconstructs 4D hand-object interactions from monocular RGB video and exports them to Unity for real-time playback.

Given a handheld video clip, the pipeline produces:
- Per-frame MANO hand meshes (`.glb`) in a consistent world frame
- A co-registered dense 3D scene mesh (`.glb`) of the background environment
- All outputs aligned to a shared Y-up right-handed coordinate frame for direct import into Unity via glTFast

---

## Repository Structure

```
REWIND_SENIOR_DESIGN/
├── Dyn-HaMR/               # Fork of Dyn-HaMR (CVPR 2025) — 4D hand motion from video
│   ├── dyn-hamr/           # Core pipeline code
│   ├── third-party/        # DROID-SLAM, HaMeR, VIPE submodules
│   └── test/               # Input videos and images
│
├── scene-reconstruction/   # Dense scene reconstruction co-registered with Dyn-HaMR
│   ├── src/                # TSDF fusion, depth estimation, masking, NeuS2 bridge
│   ├── scripts/            # reconstruct_tsdf.py, reconstruct_neus2.py
│   ├── configs/            # tsdf.yaml, neus2.yaml
│   └── outputs/            # Generated scene meshes (gitignored)
│
└── Hand-BMC-pytorch/       # Bone-length Motion Constraint for hand pose refinement
```

---

## Modules

### Dyn-HaMR
Reconstructs 4D hand motion (position, shape, pose) from a single moving-camera video. Produces per-frame `.glb` hand meshes and `cameras.json` with per-frame camera intrinsics/extrinsics in Y-up world space.

See [`Dyn-HaMR/README.md`](Dyn-HaMR/README.md) and [`Dyn-HaMR/SETUP_ENV.md`](Dyn-HaMR/SETUP_ENV.md).

### Scene Reconstruction
Takes Dyn-HaMR outputs (camera poses + video) and produces a dense `scene_mesh.glb` of the background, with hands masked out. Two pipelines:

| Pipeline | Speed | Quality | Dynamic scenes |
|---|---|---|---|
| TSDF + Depth Anything V2 | ~5–15 min | Good | Yes |
| NeuS2 | 30 min – 2 hr | High (SDF) | Static only |

See [`scene-reconstruction/README.md`](scene-reconstruction/README.md) for setup and usage.

### Hand-BMC-pytorch
Applies Bone-length Motion Constraints to smooth and physically plausible hand pose sequences. See [`Hand-BMC-pytorch/README.md`](Hand-BMC-pytorch/README.md).

---

## Quick Start

```bash
# 1. Run Dyn-HaMR on a video
conda activate dynhamr
cd Dyn-HaMR
python dyn-hamr/main.py --video test/videos/demo1.mp4

# 2. Reconstruct the scene
conda activate scene-recon
cd scene-reconstruction
python scripts/reconstruct_tsdf.py \
    --log_dir ../Dyn-HaMR/outputs/logs/video-custom/<run-id> \
    --video   ../Dyn-HaMR/test/videos/demo1.mp4 \
    --config  configs/tsdf.yaml \
    --out_dir outputs/demo1_tsdf/

# 3. Import scene_mesh.glb + hand GLBs into Unity via glTFast — no manual alignment needed
```

---

## Coordinate Frame

All outputs use **right-handed Y-up** (glTF/OpenGL standard). Unity's glTFast automatically converts to left-handed on import.

---

## Dependencies

- Python 3.9+, PyTorch, CUDA
- [Dyn-HaMR](https://github.com/Nick0693/Dyn-HaMR) (submodule)
- [Depth Anything V2](https://github.com/LiheYoung/Depth-Anything) (submodule)
- Open3D, trimesh, pyrender
- See individual `environment.yml` / `requirements.txt` per module

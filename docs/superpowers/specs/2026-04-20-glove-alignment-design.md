# Glove Alignment & Visualization Design

**Date:** 2026-04-20  
**Scope:** Align the REWIND haptic glove URDF onto DynHaMR `knot_one_handed` hand meshes, verify on single frames, then export all 1196 frames.  
**Out of scope:** Sensor data extraction, DTW evaluation, any performance scoring.

---

## Goals

1. For each frame of `knot_one_handed`: place the glove's `hand_mount` at the MANO wrist, solve IK so the thumb and index fingertip caps reach the MANO fingertip vertex positions, and export a combined GLB (hand mesh + positioned glove).
2. Test on frames 0 and 300 (configurable) before any full-sequence run.
3. Require manual visual approval of the single-frame GLBs before proceeding to the full sequence — enforced by the scripts themselves, not just convention.

---

## Architecture

### New files

| File | Purpose |
|---|---|
| `glove_sim/src/mano_io.py` | Load DynHaMR NPZ + smplx forward pass; return wrist transform + fingertip positions for one frame |
| `glove_sim/align_frame.py` | Phase 1: urdfpy + scipy IK, two test frames, assertions, GLB export, then stop |
| `glove_sim/align_sequence.py` | Phase 2: MuJoCo URDF IK, single-frame verification gate, then all 1196 frames |
| `rewind_glove_assembly/urdf/rewind_glove_mujoco.xml` | One-time MuJoCo MJCF (generated once from URDF, with freejoint + sites added; checked in) |

### Preserved files (unchanged)

- `glove_sim/src/urdfpy_vis.py` — `load_robot()`, `get_glove_scene()` — FK visualization, already correct
- `glove_sim/compare_visual_exports.py` — parity testing

### Deleted / superseded

All other existing `glove_sim/` scripts and `src/` modules are superseded and should not be depended on.

### Config

`glove_sim/config.py` is rewritten to contain path constants only — no transformation matrices, no calibration values.

---

## Data sources

| Data | Path |
|---|---|
| DynHaMR NPZ | `Dyn-HaMR/outputs/logs/video-custom/2026-03-25/knot_one_handed-all-shot-0-0--1/smooth_fit/knot_one_handed_000300_world_results.npz` |
| Hand GLBs (per-frame) | `Dyn-HaMR/outputs/logs/video-custom/2026-03-25/knot_one_handed-all-shot-0-0--1/unity_export/frames/{frame:06d}_hands.glb` |
| URDF (urdfpy-compatible) | `rewind_glove_assembly/urdf/rewind_glove_for_urdfpy.urdf` |
| Glove meshes | `rewind_glove_assembly/meshes/` |
| MANO model | `Dyn-HaMR/_DATA/data/` |
| MuJoCo MJCF (Phase 2) | `rewind_glove_assembly/urdf/rewind_glove_mujoco.xml` |

---

## Section 1: MANO data loading (`src/mano_io.py`)

Single public function:

```python
def load_frame(npz_path, mano_dir, frame_idx: int) -> dict:
    """
    Returns:
        T_wrist   : (4, 4) float64 — wrist world transform, Y-down frame
        thumb_tip : (3,)   float64 — world position of MANO vertex 745
        index_tip : (3,)   float64 — world position of MANO vertex 317
    """
```

Implementation:
1. `np.load(npz_path)` — keys: `trans`, `root_orient`, `pose_body`, `betas`, `is_right`
2. Select right-hand track: `track = argmax(is_right.mean(axis=1))`
3. Run `smplx.create(mano_dir, model_type="mano", is_rhand=True, batch_size=1)` for just `frame_idx`
4. `T_wrist[:3, :3] = Rotation.from_rotvec(root_orient[track, frame_idx]).as_matrix()`, `T_wrist[:3, 3] = trans[track, frame_idx]`
5. Return `T_wrist`, `vertices[0, 745]`, `vertices[0, 317]`

No multi-frame batching, no extra outputs. The smplx numpy alias patches (`np.bool = bool`, etc.) are applied at module import.

---

## Section 2: Phase 1 — `align_frame.py`

### CLI

```
python glove_sim/align_frame.py [--frames N [M ...]]   # default: 0 300
```

### Coordinate frame note

The URDF has a virtual `root` link as its true root, connected to `hand_mount` by a fixed joint at `xyz=[-0.157876, 0.0663838, -0.0660817]`. `robot.link_fk(cfg)` returns transforms **from the `root` frame**, not from `hand_mount`. `get_glove_scene()` already accounts for this offset (it exposes `ROOT_TO_HANDMOUNT_XYZ`). IK targets must also be expressed in the `root` frame.

```python
from src.urdfpy_vis import ROOT_TO_HANDMOUNT_XYZ  # [-0.157876, 0.0663838, -0.0660817]
HM_TO_ROOT = np.eye(4); HM_TO_ROOT[:3, 3] = -ROOT_TO_HANDMOUNT_XYZ
T_root_world = T_wrist @ HM_TO_ROOT
```

### Per-frame steps

1. `load_frame(NPZ_PATH, MANO_DIR, t)` → `T_wrist`, `thumb_tip_world`, `index_tip_world`
2. `robot = load_robot(URDF_PATH, MESH_DIR)` (cached after first call)
3. Compute `T_root_world = T_wrist @ HM_TO_ROOT` (pure-translation fixed joint, computed once per frame)
4. **IK — thumb finger:**
   - `target_in_root = np.linalg.inv(T_root_world) @ [*thumb_tip_world, 1.0]`
   - Residual: `robot.link_fk(cfg)[part_3] @ [-0.125132, 0.004875, -0.0466837, 1.0] - target_in_root`
   - `scipy.optimize.least_squares(residual, x0=prev_q_thumb, method='lm', ftol=1e-6)`
   - Chain: `revolute_1_0`, `revolute_2_0`, `revolute_3_0`, `revolute_4_0`
5. **IK — index finger:**
   - `target_in_root = np.linalg.inv(T_root_world) @ [*index_tip_world, 1.0]`
   - Residual: `robot.link_fk(cfg)[part_3_1] @ [-0.0674761, 0.004875, -0.0250619, 1.0] - target_in_root`
   - `scipy.optimize.least_squares(residual, x0=prev_q_index, method='lm', ftol=1e-6)`
   - Chain: `revolute_5_0`, `revolute_6_0`, `revolute_7_0`, `revolute_8_0`, `revolute_9_0`
6. Warm-start: `prev_q_thumb`, `prev_q_index` carry over between test frames
7. Merge joint dicts: `joint_cfg = {**thumb_cfg, **index_cfg}`

### Tests (assertions, not pytest)

```python
assert thumb_residual_m < 0.005,  f"Thumb IK: {thumb_residual_m*1000:.1f}mm > 5mm"
assert index_residual_m < 0.005,  f"Index IK: {index_residual_m*1000:.1f}mm > 5mm"
for name, angle in joint_cfg.items():
    assert -np.pi <= angle <= np.pi, f"{name}: {np.degrees(angle):.1f}° out of [-180°, 180°]"
```

If any assertion fails, the script raises with a clear message and exits non-zero. No GLB is exported for a failing frame.

### Visualization

```python
glove_scene = get_glove_scene(robot, joint_cfg, T_wrist)
hand_scene  = trimesh.load(GLB_DIR / f"{t:06d}_hands.glb", force="scene")
combined    = trimesh.Scene([*hand_scene.geometry.values(),
                             *glove_scene.geometry.values()])
combined.export(ALIGNED_DIR / f"{t:06d}_aligned.glb")
```

### Manual gate

After both test frames pass and GLBs are written:

```
Tests passed for frames [0, 300].
GLBs written to:
  glove_sim/outputs/aligned/000000_aligned.glb
  glove_sim/outputs/aligned/000300_aligned.glb

Open with: f3d glove_sim/outputs/aligned/000000_aligned.glb

Review alignment visually. When satisfied, run:
  python glove_sim/align_sequence.py
```

Script exits 0. No further processing happens automatically.

---

## Section 3: Phase 2 — `align_sequence.py`

### One-time MJCF preparation (done manually once, result checked in)

1. Embed `<mujoco><compiler meshdir="../meshes/" discardvisual="false" balanceinertia="true"/></mujoco>` into `rewind_glove_for_urdfpy.urdf` (in-memory, not on disk)
2. Load with `mujoco.MjModel.from_xml_string()` — MuJoCo handles URDF natively, no custom parser
3. Print/save as MJCF to `rewind_glove_assembly/urdf/rewind_glove_mujoco.xml`
4. Manually add to that file:
   - `<freejoint name="base_free"/>` inside the `root` body (the URDF virtual root link)
   - `<site name="thumb_tip_site" pos="-0.125132 0.004875 -0.0466837" size="0.004"/>` inside `part_3`
   - `<site name="index_tip_site" pos="-0.0674761 0.004875 -0.0250619" size="0.004"/>` inside `part_3_1`
5. Check in `rewind_glove_mujoco.xml` — never regenerated at runtime

This step is a one-time setup task, not part of the main pipeline. A helper script `glove_sim/tools/generate_mjcf.py` automates steps 1–3.

### CLI

```
python glove_sim/align_sequence.py [--verify-frame N]   # default: 0
```

### Verification gate (same pattern as Phase 1)

1. Run IK + export GLB for `--verify-frame` only
2. Print path, then:
   ```python
   input("Review the GLB above, then press Enter to process all frames (Ctrl+C to abort): ")
   ```
3. Only after Enter: process all 1196 frames

### Per-frame IK (MuJoCo)

1. Load `rewind_glove_mujoco.xml` once → `mujoco.MjModel`, `mujoco.MjData`
2. Compute `T_root_world = T_wrist @ HM_TO_ROOT`; set freejoint qpos: `data.qpos[base_qadr:base_qadr+3] = T_root_world[:3, 3]`, quaternion from `Rotation.from_matrix(T_root_world[:3, :3]).as_quat()` reordered to wxyz
3. For each site (thumb, index): extract fingertip target from `load_frame()`; LM loop using `mujoco.mj_jacSite()` for analytical Jacobian; damped least-squares step `dq = (J.T @ J + λI)⁻¹ J.T @ err`
4. Warm-start: qpos carries over between frames
5. Visualization: same `get_glove_scene()` + hand GLB composite as Phase 1

### Output

- `glove_sim/outputs/aligned/{frame:06d}_aligned.glb` — one per frame, all 1196

---

## Test tolerances

| Metric | Threshold |
|---|---|
| IK residual per fingertip | < 5 mm |
| Joint angle range | within (−π, π) |

---

## Approval protocol

1. Run `python glove_sim/align_frame.py` — tests must pass, inspect both GLBs
2. Approve visually → run `python glove_sim/align_sequence.py` → verify single frame again → press Enter for full sequence
3. No code changes between Phase 1 and Phase 2 runs unless something looks wrong

---

## Non-goals

- DTW or any performance evaluation
- Sensor angle extraction
- Custom calibration offset matrices between wrist and glove base
- Runtime MJCF generation (the `.xml` is static, checked in)

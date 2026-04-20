# glove_sim

Projects the REWIND haptic glove (URDF) onto DynHaMR expert hand reconstructions
frame-by-frame using Forward Kinematics (urdfpy) and Inverse Kinematics (scipy).
Outputs combined glove + hand `.glb` files for visual verification.

---

## Quick start

```bash
conda activate glove_sim

# Test 10 evenly-spaced frames — strict assertions, review GLBs before committing
python glove_sim/align_frame.py --frames-test

# Align a specific frame
python glove_sim/align_frame.py --frames 300

# Align the entire sequence (skips and reports failures)
python glove_sim/align_frame.py
```

---

## Calibration (one-time setup)

Run these once per recording, in order. Both tools show the glove in resting
(zero-joint) position for clarity.

### 1. Base alignment — `annotate_planes.py`

Annotates 3 landmark correspondences (wrist centre, index knuckle, pinky
knuckle) on both the glove and the hand mesh. Solves the rigid transform via
Kabsch / SVD and writes `T_WRIST_TO_HM` to
`outputs/aligned/plane_annotation.json`.

```bash
python glove_sim/tools/annotate_planes.py --frame 300
```

Controls: right-click to place a point, U to undo, Enter to confirm, Q to abort.

Pick order (both phases, same order):
1. Wrist centre
2. Index-finger knuckle
3. Pinky-finger knuckle

### 2. Fingertip correction — `annotate_fingertips.py`

Annotates the index cap centre and thumb cap centre on the glove (Phase 1),
then the index and thumb fingertips on the hand mesh (Phase 2). Computes a
per-finger IK target correction and writes it to
`outputs/aligned/fingertip_annotation.json`.

```bash
python glove_sim/tools/annotate_fingertips.py --frame 300
```

Pick order — both phases use the same order:
1. Index finger (cap / tip)
2. Thumb (cap / tip)

To recompute offsets from a saved annotation without re-annotating:

```bash
python glove_sim/tools/annotate_fingertips.py --recompute
```

---

## Pipeline overview

```
DynHaMR NPZ
    │
    ▼
src/mano_io.py          Load T_wrist, thumb_tip (v745), index_tip (v317)
    │
    ▼
align_frame.py          Apply T_WRIST_TO_HM + fingertip offsets
  solve_ik_frame()      urdfpy FK + scipy TRF minimises distance from
                        physical cap dome to MANO fingertip
    │
    ▼
src/urdfpy_vis.py       get_glove_scene() — FK at solved joint angles,
                        converted MANO Y-down → GLB (Unity Y-up) coords
    │
    ▼
outputs/aligned/        NNNNNN_aligned.glb  (glove + hand combined scene)
```

### Coordinate systems

| Frame | Convention |
|---|---|
| MANO world | Right-handed, Y-down (+X fwd, +Y down, +Z left) |
| DynHaMR GLB | Unity left-handed Y-up: `GLB = diag([1,−1,−1]) @ MANO` |
| URDF root | Identical to MANO world; FK results promoted via `T_root_world` |

---

## File structure

```
glove_sim/
├── align_frame.py          Main entry point — IK + GLB export
├── config.py               Paths and calibration constants (auto-loaded from JSONs)
├── environment.yml         Conda environment spec
├── src/
│   ├── mano_io.py          Load MANO vertices and wrist transform from DynHaMR NPZ
│   └── urdfpy_vis.py       urdfpy robot loader and trimesh scene builder
├── tools/
│   ├── annotate_planes.py  Interactive 3-landmark Kabsch annotation (base alignment)
│   └── annotate_fingertips.py  Interactive fingertip IK correction annotation
├── tests/
│   ├── test_align_frame.py
│   ├── test_mano_io.py
│   └── test_plane_alignment.py
└── outputs/
    └── aligned/
        ├── plane_annotation.json       T_WRIST_TO_HM (from annotate_planes)
        ├── fingertip_annotation.json   Per-finger IK offsets (from annotate_fingertips)
        └── NNNNNN_aligned.glb          Per-frame combined scenes
```

---

## Running tests

```bash
pytest glove_sim/tests/ -v -p no:dash
```

Tests that depend on annotation files are skipped automatically if the files
do not exist yet.

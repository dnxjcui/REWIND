# Glove Sim Pipeline — Full Technical Description

## Overview

The pipeline places the CAD/URDF model of the haptic glove onto the expert hand
mesh produced by DynHaMR, frame by frame, and exports a combined `.glb` for
visual verification. There is **no MuJoCo involvement** in the current active
pipeline. Everything uses **urdfpy** (Forward Kinematics) + **scipy** (IK
optimisation) + **trimesh** (mesh I/O and GLB export).

---

## Step-by-step Pipeline (align_frame.py)

### Step 1 — Load MANO data for the frame

**File:** `glove_sim/src/mano_io.py` → `load_frame(npz_path, mano_dir, frame_idx)`

1. Open the DynHaMR `.npz` file (`knot_one_handed_000300_world_results.npz`).
2. Pick the right-hand track (`is_right` column).
3. Run the SMPL-X / MANO model (`smplx.create`) with the frame's
   `root_orient`, `trans`, `pose_body`, and `betas`.
4. Extract:
   - **`T_wrist`** — 4×4 rigid transform (rotation from `root_orient` axis-angle,
     translation from `trans`). This is in **MANO world space**, Y-axis pointing
     down (right-handed, +X forward, +Y down, +Z left).
   - **`thumb_tip`** — 3-D world position of MANO vertex **745** (thumb tip).
   - **`index_tip`** — 3-D world position of MANO vertex **317** (index tip).

### Step 2 — Apply calibrated spatial offsets (from config.py)

**File:** `glove_sim/config.py`

Two pre-computed offsets are loaded from JSON files at import time:

| Config var | Source file | Meaning |
|---|---|---|
| `T_WRIST_TO_HM` | `plane_annotation.json` | 4×4 rigid transform in wrist-local frame that places the glove `hand_mount` body flush against the dorsal hand surface. Produced by Kabsch / SVD 3-landmark annotation. |
| `THUMB_TIP_OFFSET` / `INDEX_TIP_OFFSET` | `fingertip_annotation.json` | 3-D correction vectors in wrist-local frame that shift the MANO regressor tip to the annotated actual mesh tip. |

### Step 3 — Compute the URDF root world pose

**File:** `align_frame.py` → `solve_ik_frame`

The URDF has a kinematic chain:

```
world → hand_mount → [fixed joint: ROOT_TO_HANDMOUNT_XYZ] → root_link → … → part_3 (thumb cap)
                                                                            → part_3_1 (index cap)
```

The glove's world pose is set by:

```python
T_hand_mount_world = T_wrist @ T_WRIST_TO_HM        # hand_mount in MANO world
T_root_world       = T_hand_mount_world @ T_hm_to_root  # URDF root in MANO world
```

where `T_hm_to_root` translates by `[0.157876, -0.0663838, 0.0660817]` (the
fixed offset from `hand_mount` to the URDF root).

### Step 4 — Express the IK target in URDF-root-local frame

The MANO fingertip (with offset correction applied) is transformed from MANO
world space into URDF root local space so the IK can operate entirely inside
the robot's own coordinate frame:

```python
tip_world    = mano_tip + R_wrist @ wrist_local_offset   # correct MANO regressor
target_root  = inv(T_root_world) @ [tip_world, 1]        # into URDF root frame
```

### Step 5 — Inverse Kinematics via urdfpy FK + scipy least_squares

**File:** `align_frame.py` → `_ik_finger`

**Library:** `urdfpy` for FK, `scipy.optimize.least_squares` (Trust-Region
Reflective) for optimisation.

**How it works:**

For each finger (thumb chain: 4 joints; index chain: 5 joints), a scalar
residual function is minimised:

```python
def residual(q_vec):                           # q_vec = joint angles (radians)
    fk = robot.link_fk(cfg=dict(zip(chain, q_vec)))
    T  = fk[tip_link]                          # 4×4 transform: part_3 → URDF root
    return T[:3,:3] @ visual_origin + T[:3,3] - target_root
```

`visual_origin` is a hardcoded 3-D point **in the tip link's local frame** that
is supposed to represent the physical cap tip. `T[:3,:3] @ visual_origin + T[:3,3]`
converts that point into URDF root frame; the residual is its distance from the
target. `scipy` minimises the Euclidean distance by adjusting the joint angles.

---

## Why the IK Fails — Root Cause

### `visual_origin` is NOT on the mesh surface

The values currently hardcoded are taken directly from the URDF
`<visual><origin xyz=…>` attribute:

```
_THUMB_VISUAL_ORIGIN = np.array([-0.125132,  0.004875, -0.0466837])   # part_3
_INDEX_VISUAL_ORIGIN = np.array([-0.0674761, 0.004875, -0.0250619])   # part_3_1
```

The URDF `<visual><origin>` specifies where the **STL mesh frame origin (0,0,0)**
is placed inside the link's local frame. It is **not** a point on the mesh surface.

Inspecting the STL files directly:

| | `Part 3.stl` (thumb) | `Part 3_1.stl` (index) |
|---|---|---|
| Vertices (link-local) X range | −0.012 … +0.011 | −0.012 … +0.011 |
| Vertices (link-local) Y range | −0.013 … +0.018 | −0.013 … +0.018 |
| Vertices (link-local) Z range | −0.031 … +0.015 | −0.031 … +0.015 |
| Mesh centroid (link-local)    | (−0.000, +0.002, −0.009) | (−0.001, +0.002, −0.009) |
| **Dome tip centroid** (link-local) | **(−0.0005, 0.0, −0.0310)** | **(−0.0026, 0.0, −0.0309)** |

**The hardcoded `visual_origin` values sit at X ≈ −0.125, which is ~12 cm outside
the mesh bounds (the mesh spans only ±0.012 in X from the link origin).**

This means the IK numerically converges — scipy finds joint angles that place
this phantom point at the MANO fingertip — but the physical cap dome ends up
displaced from the target by the full 12 cm offset, just in a different direction
depending on the solved joint angles.

---

## The Fix

Replace `_THUMB_VISUAL_ORIGIN` and `_INDEX_VISUAL_ORIGIN` in `align_frame.py`
with the **dome-tip centroid** computed from the STL vertex data (the centroid
of all vertices within 3 mm of the minimum-Z extreme, which is the closed dome
of the cap that physically contacts the fingertip):

```python
# OLD (STL frame origin — not on the mesh)
_THUMB_VISUAL_ORIGIN = np.array([-0.125132,  0.004875, -0.0466837])
_INDEX_VISUAL_ORIGIN = np.array([-0.0674761, 0.004875, -0.0250619])

# NEW (dome-tip centroid in link-local frame, computed from STL)
_THUMB_VISUAL_ORIGIN = np.array([-0.00047,  0.0, -0.03095])
_INDEX_VISUAL_ORIGIN = np.array([-0.00259,  0.0, -0.03085])
```

With this fix, `T[:3,:3] @ visual_origin + T[:3,3]` evaluates to the world
position of the physical dome tip of the cap, not a phantom point in space.
The IK will then correctly drive the physical cap tip to the MANO fingertip.

---

## IK Library / Method Assessment

The current approach (urdfpy FK + scipy TRF) is sound for a small chain
(4–5 DOF). The TRF method is gradient-based and will find a local minimum.
Alternatives:

| Approach | Pro | Con |
|---|---|---|
| **Current: urdfpy FK + scipy TRF** | No extra deps, uses full URDF | visual_origin bug just described |
| **ikpy** | Dedicated IK, DH chain, good for revolute chains | Requires building DH table from URDF manually |
| **roboticstoolbox-python** | Full analytical + numerical IK, URDF import | Heavier dependency, URDF import can be fragile |
| **MuJoCo damped LS (glove_ik.py)** | Analytical Jacobian, fast, numerically stable | More complex setup; site positions had same visual_origin bug |

**Recommendation:** Fix `visual_origin` first. If residuals remain > 5 mm after
that fix, switch to `ikpy` which handles joint-limit clamping and chain
parameterisation more robustly.

---

## Coordinate System Notes

| Frame | Convention |
|---|---|
| MANO world | Right-handed, Y-down (+X forward, +Y down, +Z left) |
| DynHaMR GLB | Unity left-handed Y-up: `GLB = diag([1, -1, -1]) @ MANO` |
| URDF root | Same as MANO world; FK outputs are in URDF-root-local then promoted to MANO world via `T_root_world` |

The transform between MANO world and GLB is `MANO_TO_GLB = diag([1, -1, -1, 1])`,
which is self-inverse. This was incorrectly set to `diag([1, 1, -1, 1])` in
earlier code — the fix is applied everywhere as of the current session.

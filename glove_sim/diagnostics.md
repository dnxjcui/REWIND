# Glove Sim — Collision Prevention Diagnostics

## Pipeline overview

```
MANO NPZ frame
    │
    ▼  mano_io.load_frame()
T_wrist, thumb_tip (v745), index_tip (v317),
vertices (778,3), vertex_normals (778,3)
    │
    ▼  _compute_T_root_world()
T_root_world  ←─ dynamic Kabsch on MANO skin verts  (or static T_WRIST_TO_HM fallback)
    │
    ▼  solve_ik_frame()
    ├─ [optional] hover offset along fingertip normal (currently disabled — see below)
    ├─ calibrated fingertip correction  (THUMB/INDEX_TIP_OFFSET, wrist-local)
    ├─ transform dorsal plane  GLB → MANO world → URDF root-local
    ├─ thumb chain → _ik_finger()  drives part_3   to MANO index tip (v317 + offsets)
    ├─ index chain → _ik_finger()  drives part_3_1 to MANO thumb tip (v745 + offsets)
    │       └─ scipy TRF: [ik_err(3), collision_penalty(n)], multi-start, non-clip prefer
    └─ joint_cfg (9 angles), thumb_residual, index_residual
    │
    ▼  _export_frame()
outputs/aligned/{video_name}/glb_frames/{frame:06d}_aligned.glb
```

**Cap swap note:** The physical glove has the thumb cap on the index-chain side. MANO thumb tip (v745) drives the index chain; MANO index tip (v317) drives the thumb chain.

---

## Coordinate frames

| Frame | Convention | Used for |
|---|---|---|
| MANO world | Right-handed, Y-down | NPZ / SMPLX output |
| GLB / Unity | Left-handed, Y-up: `diag([1,−1,−1]) @ MANO` | DynHaMR `.glb` files |
| URDF root | Same as MANO world; urdfpy `link_fk()` returns transforms in this frame | IK residual computation |

Dorsal plane centroid + normal are stored in GLB space in `plane_annotation.json`.  
At solve time: **GLB → MANO world → URDF root-local** via `inv(T_root_world)`.

---

## IK target pipeline

```
v745 / v317  ←  MANO vertex position (world, Y-down)
    + [_HOVER_DIST * vertex_normal]   ← disabled; see section below
    + R_wrist @ TIP_OFFSET            ← calibrated in annotate_fingertips.py
    → transform to URDF root-local
    → pass to scipy TRF as target
```

### Hover offset (`_HOVER_DIST`) — currently 0.0 (DISABLED)

The hover idea is sound: push the IK target 10 mm outward along the fingertip surface normal so the cap dome sits on the skin rather than sinking in. However, it **must be co-calibrated** with the fingertip annotation:

- `THUMB_TIP_OFFSET` / `INDEX_TIP_OFFSET` are measured by `annotate_fingertips.py` assuming the target is the raw MANO vertex (hover = 0).
- Adding a non-zero hover on top of a calibration measured at hover = 0 shifts the effective target by that distance, consistently inflating IK residuals and breaking frames that were previously within the 5 mm tolerance.

**To enable hover:**
1. Set `_HOVER_DIST = 0.010` in `align_frame.py`.
2. Re-run `python glove_sim/tools/annotate_fingertips.py --frame 300` to measure new offsets.
3. The solver now works relative to the hovered target.

---

## Collision system

### 1. Mesh proxy points

Link origins can sit inside the hand by construction (joint axes don't move). Each tracked link therefore has a **proxy offset** in link-local frame — the vertex of that link's mesh closest to the dorsal surface at the zero configuration. At solve time:

```
proxy_pos = T_link[:3,:3] @ proxy_local + T_link[:3,3]
```

| Link | Proxy local (m) | Chain |
|---|---|---|
| `xl330_m077_t_1` | [0.009, −0.025, −0.003] | Thumb |
| `xl_linkage_horn` | [0.003, −0.003, 0.007] | Thumb |
| `part_2_1` | [−0.005, 0.003, 0.001] | Thumb |
| `part_6` | [−0.008, −0.001, −0.054] | Index |
| `xl_housing_1` | [0.012, −0.027, 0.000] | Index |
| `part_2` | [−0.004, 0.003, 0.002] | Index |
| `xl_linkage_horn_1` | [0.016, −0.006, 0.004] | Index |

### 2. Quadratic penalty

```
penalty[i] = 2000 * max(0, MARGIN − dist)²     MARGIN = 3 mm
```

Behaviour: at contact (dist = 0) → penalty ≈ IK residual magnitude; at 3 mm penetration → ~14× IK residual. Deep clipping costs far more than shallow contact.

### 3. Lateral cutoff

The infinite dorsal plane would penalise linkages wrapping correctly around the finger sides. The penalty is suppressed when the proxy point is more than **40 mm** laterally from the plane centroid.

### 4. Multi-start with non-clipping preference

The solver priority is:
1. Use warm-start (previous frame) immediately if it does not clip — fastest path.
2. If warm-start clips, run all candidates and return the **lowest-residual non-clipping** result.
3. If every candidate clips, return the lowest-residual result regardless.

### 5. Sign-aware candidate generation

Each revolute joint was empirically tested (+0.8 rad individually) to determine whether positive rotation moves the linkage toward or away from the dorsal plane:

| Joint | Direction (+ rad) | Safe start sign |
|---|---|---|
| revolute_1_0 | na (twist) | 0 — swept at ±π/2 |
| revolute_2_0 | towards | **−** |
| revolute_3_0 | away | + |
| revolute_4_0 | away | + |
| revolute_5_0 | away | + |
| revolute_6_0 | towards | **−** |
| revolute_7_0 | away | + |
| revolute_8_0 | towards | **−** |
| revolute_9_0 | away | + |

Candidates are built by `_build_candidates()`: progressive pre-bending at π/4 then π/2, each joint set to its measured safe-direction sign.

---

## Known limitations and active issues

### Clipping persists in some frames

Despite the proxy penalty and multi-start, some frames still clip. Likely causes in priority order:

1. **All candidates start in or near a clipping basin.** The multi-start set covers the safe direction, but if the IK target for that frame requires a configuration not reachable without passing through the hand from any candidate, every solve converges to a clipping minimum. The solver picks the least-bad option.

2. **Dorsal plane is a flat infinite proxy.** The real hand surface is curved; the knuckles and finger sides are not explicitly protected. The lateral cutoff (40 mm) helps but is approximate.

3. **`part_6` structural clipping.** This link's joint-axis origin sits ~1 mm inside the hand regardless of joint angles. The proxy at `[−0.008, −0.001, −0.054]` tracks the housing face, but if the housing itself overlaps the hand, the penalty cannot fully prevent it without completely sacrificing IK accuracy.

4. **No temporal smoothness.** Large hand motion between frames can leave the warm-start on the wrong side of the collision surface. The clipping-detection fallback (run multi-start if warm-start clips) addresses this, but only if a non-clipping solution is reachable from one of the candidates.

### Hover offset currently disabled

See the IK target section above. Re-enable and re-calibrate before using.

---

## Output locations

```
glove_sim/outputs/aligned/
├── plane_annotation.json          one-time calibration (annotate_planes.py)
├── fingertip_annotation.json      one-time calibration (annotate_fingertips.py)
└── {VIDEO_NAME}/
    ├── glb_frames/
    │   └── {frame:06d}_aligned.glb
    └── urdf_frames/                    (requires --urdf flag)
        └── rewind_glove_{frame:06d}.urdf
```

---

## Diagnostic tool

```bash
python glove_sim/tools/diagnose_collision.py --frame 300
python glove_sim/tools/diagnose_collision.py --frame 50 --weights 0 100 500 2000 5000
```

Reports per collision weight: IK residuals, per-link proxy distances (mm), lateral suppression flags, and trimesh mesh-on-mesh intersection count.

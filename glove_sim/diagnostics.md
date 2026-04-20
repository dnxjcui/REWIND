# Glove Sim — Collision Prevention Diagnostics

## What the pipeline currently does

### 1. Soft collision penalty in the IK solver

`_ik_finger` in `align_frame.py` appends extra residual terms to the SciPy TRF objective whenever tracked chain links get too close to the hand's dorsal plane.

**Mechanism:**
- The dorsal plane (centroid + normal) is annotated once in `annotate_planes.py` and stored in `plane_annotation.json` in GLB space.
- At solve time the plane is converted from GLB space → MANO world → URDF root-local frame (same space as the FK output).
- For each link in `_THUMB_COLLISION_LINKS` / `_INDEX_COLLISION_LINKS`, the signed distance to the plane is computed: `dist = dot(link_origin − plane_centroid, plane_normal)`.
- If `dist < COLLISION_MARGIN` (3 mm), a penalty term `COLLISION_WEIGHT * (COLLISION_MARGIN − dist)` is added to the residual vector.

**Constants (current):**
```
_COLLISION_MARGIN = 0.003   # 3 mm
_COLLISION_WEIGHT = 8.0
```

A link 3 mm inside the plane produces a penalty of `8 × 0.006 = 0.048 m`, versus a typical IK residual of 0.001–0.005 m — about 10–48× the IK error. This is intentionally strong enough to deter clipping, but not so strong that it prevents convergence.

**Tracked links:**
| Chain | Links checked |
|-------|--------------|
| Thumb | `xl330_m077_t_1`, `xl_linkage_horn`, `part_2_1` |
| Index | `part_6`, `xl_housing_1`, `part_2`, `xl_linkage_horn_1` |

The tip links (`part_3`, `part_3_1`) are excluded because the IK target already pulls them to the fingertip; their position is fully controlled by the objective.

### 2. Hard joint bounds

All 9 revolute joints are declared `continuous` in the URDF (no URDF limits). We impose ±π bounds via SciPy's `bounds` parameter, which the TRF solver enforces hard at each step. This prevents unbounded drift but does not encode any directional collision constraint — that is left to the soft penalty.

---

## Known limitations

### `part_6` origin is always inside the hand

`part_6` is the child link of `revolute_5_0`. Its URDF origin coincides with the joint axis location, which sits roughly 1 mm below the dorsal plane at all revolute_5_0 angles. Because the link origin does not move with the joint angle, the soft penalty for `part_6` fires at every configuration, contributing a constant ~0.024 penalty term that the solver cannot eliminate. In practice the solver drives all other links clear and accepts the `part_6` violation.

**Implication:** `part_6`'s *mesh* may still clip the hand regardless of the joint solution because the mesh spans the joint and its physical extent is larger than the origin point.

### Zero warm-start places `xl_platform_horn_1` inside the hand

At `revolute_5_0 = 0°` (the default warm-start for the first frame), `xl_platform_horn_1` sits at −1.9 mm below the dorsal plane. A sweep over `revolute_5_0` shows:

```
revolute_5_0 = −90°  →  xl_platform_horn_1 = +46.1 mm  (well clear)
revolute_5_0 =   0°  →  xl_platform_horn_1 =  −1.9 mm  (inside hand)
revolute_5_0 = +90°  →  xl_platform_horn_1 = −39.5 mm  (deep inside)
```

When the solver starts at zero, it is already in a clipping configuration. The soft penalty pushes it toward −90°, but if the IK target is reachable from a local minimum that only partially reduces the penalty, the solver may converge there instead.

**Mitigation (implemented):** Multi-start IK (see below). When no warm-start is available, the solver is run from five candidate initial angles for the first joint of each chain and the solution with the lowest IK residual is kept.

### Plane proxy is approximate

The dorsal plane is a flat infinite half-space fit to three annotated landmarks. The real dorsal surface of the hand is curved and deforms frame-to-frame. Links near the wrist or sides of the hand may sit above or below the annotated plane even when the physical glove is clear of the skin.

### No mesh-level collision detection in the solver

The penalty operates on link *origins* (FK position vectors), not on actual mesh geometry. A link whose origin is 4 mm above the plane may still have mesh faces that intersect the hand if the mesh extends downward. Use `diagnose_collision.py` to check actual mesh intersections in the exported GLBs.

---

## Multi-start IK (as of this version)

To avoid the zero warm-start trap, `_ik_finger` now accepts a `q0_candidates` list. When called with `q0=None`, it runs the full TRF optimisation from each candidate, then returns the solution with the lowest IK residual (first 3 components only, excluding penalty terms).

**Default candidates:**
```
Index chain (revolute_5_0 first):  [−π/2, −π/4, 0, π/4, π/2]
Thumb chain (revolute_1_0 first):  [−π/2, −π/4, 0, π/4, π/2]
```

When a valid warm-start is available from the previous frame (`q0` is not None), multi-start is bypassed — the previous solution is used directly. Multi-start fires only on the first frame or after a failure that resets `prev_q` to None.

---

## How to run the collision diagnostic

```bash
python glove_sim/tools/diagnose_collision.py --frame 300
```

The script:
1. Runs IK at the specified frame for each collision weight in `[0, 1, 5, 8, 20, 50]`.
2. Reports IK residuals and per-link signed distances to the dorsal plane.
3. Exports a temporary GLB for each weight and uses `trimesh.collision.CollisionManager` to test whether any glove mesh intersects the hand mesh.

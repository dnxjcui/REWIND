# Glove Sim — Full Pipeline Diagnostics

---

## 1. End-to-end pipeline

```
DynHaMR NPZ
    │
    ▼  mano_io.load_frame(npz_path, mano_dir, frame_idx)
    │    smplx.MANO(global_orient, hand_pose, betas, transl)
    │    → vertices (778,3)  MANO world Y-down
    │    → vertex_normals (778,3)  outward unit normals (trimesh)
    │    → T_wrist (4,4)  wrist world transform
    │    → thumb_tip  = vertices[745]   (MANO index finger)
    │    → index_tip  = vertices[317]   (MANO thumb finger)
    │
    ▼  _compute_T_root_world(T_wrist, vertices)
    │    if anchor_vertex_ids calibrated → Kabsch per-frame rigid registration
    │    else                            → T_wrist @ T_WRIST_TO_HM @ _HM_TO_ROOT
    │    → T_root_world (4,4)  URDF root origin in MANO world space
    │
    ▼  solve_ik_frame(robot, T_wrist, thumb_tip, index_tip, prev_q_thumb, prev_q_index,
    │                 vertices, vertex_normals)
    │    1. hover offset  (currently disabled — see §5)
    │    2. calibrated fingertip correction  R_wrist @ TIP_OFFSET
    │    3. GLB dorsal plane → MANO world → URDF root-local
    │    4. thumb chain → _ik_finger() → drives part_3   to MANO index tip (v317 + offsets)
    │    5. index chain → _ik_finger() → drives part_3_1 to MANO thumb tip (v745 + offsets)
    │    → joint_cfg (9 angles), thumb_residual, index_residual
    │
    ▼  _export_frame(robot, T_wrist, joint_cfg, frame_idx, vertices)
    │    FK → glove meshes in MANO world space
    │    merge with DynHaMR {frame:06d}_hands.glb
    │    → outputs/aligned/{VIDEO_NAME}/glb_frames/{frame:06d}_aligned.glb
    │
    ▼  export_frame_urdf(robot, joint_cfg, T_root_world, frame_idx)   [--urdf flag]
         revolute joints → fixed at solved angles (FK baked)
         world link + world_to_glove fixed joint (embeds T_root_world)
         → outputs/aligned/{VIDEO_NAME}/urdf_frames/rewind_glove_{frame:06d}.urdf
```

**Cap swap:** The physical glove has the thumb cap on the index-chain side.
`MANO thumb tip (v745)` → drives `index chain (part_3_1)`.
`MANO index tip (v317)` → drives `thumb chain (part_3)`.

---

## 2. Coordinate frames

| Frame | Convention | Notes |
|---|---|---|
| MANO world | Right-handed, Y-down | NPZ / smplx output |
| GLB / Unity | Left-handed, Y-up: `diag([1,−1,−1]) @ MANO` | DynHaMR `.glb` files |
| URDF root | Same as MANO world; `urdfpy.link_fk()` returns transforms relative to root | IK residual computation |

`_MANO_TO_GLB = np.diag([1, -1, -1, 1])` is its own inverse (applies in both directions).

Dorsal plane centroid + normal are stored in GLB space inside `plane_annotation.json`.
At solve time: **GLB → MANO world → URDF root-local** via `inv(T_root_world)`.

---

## 3. Glove base placement

### 3a. Static fallback (most frames)

```python
def _compute_T_root_world(T_wrist, vertices):
    # Static: T_WRIST_TO_HM calibrated once by annotate_planes.py
    return T_wrist @ cfg.T_WRIST_TO_HM @ _HM_TO_ROOT
```

`T_WRIST_TO_HM` is a fixed 4×4 that translates from MANO wrist origin to the
`hand_mount` link. `_HM_TO_ROOT` is the inverse translation (hand_mount → URDF root).

### 3b. Dynamic Kabsch (when anchor calibration exists)

```python
if (vertices is not None
        and cfg.ANCHOR_VERTEX_IDS is not None
        and cfg.GLOVE_ANCHOR_PTS_HM is not None):
    skin_pts = vertices[cfg.ANCHOR_VERTEX_IDS]     # (N,3) MANO world
    R, t = _kabsch(cfg.GLOVE_ANCHOR_PTS_HM, skin_pts)
    T_hm_world[:3,:3] = R
    T_hm_world[:3, 3] = t
    return T_hm_world @ _HM_TO_ROOT
```

SVD Kabsch maps `N` fixed glove anchor points (in hand_mount frame) to their
corresponding live MANO skin vertices each frame, giving a per-frame rigid fit.
Requires re-running `annotate_planes.py` to populate `anchor_vertex_ids` and
`glove_anchor_pts_hm` in `plane_annotation.json`.

---

## 4. IK target pipeline

```
vertices[745 or 317]           ← raw MANO vertex (world, Y-down)
    + _HOVER_DIST * vertex_normals[v]   ← disabled; see §5
    + T_wrist[:3,:3] @ TIP_OFFSET       ← calibrated in annotate_fingertips.py
    → transform to URDF root-local:  (inv(T_root_world) @ [pt, 1])[:3]
    → passed as target to scipy TRF
```

`THUMB_TIP_OFFSET` / `INDEX_TIP_OFFSET` are calibrated by `annotate_fingertips.py`
and stored in `fingertip_annotation.json`. They express the vector from the raw
MANO vertex to the desired cap centre, in wrist-local coordinates.

---

## 5. Hover offset (`_HOVER_DIST`) — currently 0.0 (DISABLED)

```python
_HOVER_DIST = 0.0  # disabled — re-enable only after re-running annotate_fingertips.py
```

The hover idea: push the IK target 10 mm outward along the skin normal so the
cap dome sits on the skin rather than sinking in. **It must be co-calibrated:**

- `THUMB_TIP_OFFSET` / `INDEX_TIP_OFFSET` are measured assuming `_HOVER_DIST = 0`.
- Adding non-zero hover without re-calibrating overshoots the target, inflating
  IK residuals and breaking frames that otherwise converge cleanly.

**To enable hover correctly:**
1. Set `_HOVER_DIST = 0.010` in `align_frame.py`.
2. Re-run `python glove_sim/tools/annotate_fingertips.py --frame 300`.
3. New offsets are now measured relative to the hovered target.

---

## 6. The IK solver in full detail

### 6a. Residual vector

```python
def residual(q_vec):
    fk = robot.link_fk(cfg=dict(zip(chain_joints, q_vec)))

    ik_err = np.full(3, 1e6)
    link_T = {}
    for link, T in fk.items():
        if link.name == tip_link_name:
            ik_err = T[:3,:3] @ visual_origin + T[:3,3] - target_in_root
        elif link.name in collision_links:
            link_T[link.name] = T

    penalty = np.zeros(n_penalty)
    if plane_normal_root is not None:
        for i, lname in enumerate(collision_links):
            T = link_T.get(lname)
            proxy_local = _COLLISION_PROXIES.get(lname, np.zeros(3))
            pos = T[:3,:3] @ proxy_local + T[:3,3]
            v = pos - plane_centroid_root
            dist = np.dot(v, plane_normal_root)
            if dist >= _COLLISION_MARGIN: continue
            lateral = np.linalg.norm(v - dist * plane_normal_root)
            if lateral > _COLLISION_LATERAL_CUTOFF: continue
            penetration = _COLLISION_MARGIN - dist
            penalty[i] = _COLLISION_WEIGHT * penetration**2

    return np.concatenate([ik_err, penalty])
```

`robot.link_fk(cfg=...)` calls urdfpy forward kinematics, returning
`{Link: T_from_root (4×4)}` for every link.

`visual_origin` is the dome-tip centroid in the tip link's local frame
(computed from STL vertex data):
```python
_THUMB_VISUAL_ORIGIN = np.array([-0.00047,  0.0, -0.03095])
_INDEX_VISUAL_ORIGIN = np.array([-0.00259,  0.0, -0.03085])
```

The first 3 components are the 3D Euclidean IK error in metres.
Remaining `n_penalty` components are soft collision constraints.

### 6b. Bounds and optimizer

```python
lower = np.array([_JOINT_BOUNDS[j][0] for j in chain_joints])  # -π for all
upper = np.array([_JOINT_BOUNDS[j][1] for j in chain_joints])  # +π for all

scipy.optimize.least_squares(
    residual, start,
    bounds=(lower, upper),
    method='trf',       # Trust Region Reflective — handles box constraints
    ftol=cfg.IK_TOL,    # 1e-6
    max_nfev=cfg.IK_MAX_NFEV,  # 200 function evaluations
)
```

`ik_residual = np.linalg.norm(res.fun[:3])` — only the IK error components,
not the penalty, are used for the final residual metric.

---

## 7. Seeding strategy (golden seed + per-frame warm-start)

### 7a. Golden seed

On startup, `main()` reads `outputs/aligned/golden_seed.json` (if it exists):

```python
golden_q_thumb, golden_q_index = load_golden_seed()
prev_q_thumb = golden_q_thumb   # None if file absent
prev_q_index = golden_q_index
```

`golden_seed.json` stores the solved joint angles from a single reference frame
(currently frame 300, saved with `--save-golden`):

```json
{
  "frame_idx": 300,
  "joint_cfg": {
    "revolute_1_0": 0.267,
    "revolute_2_0": 0.850,
    ...
  }
}
```

**Critical separation:** The golden seed provides *joint angles only*. The glove
base position for each new frame is computed fresh from that frame's wrist data:

```
glove base = _compute_T_root_world(T_wrist_frame_N, vertices_frame_N)
```

So the base is correctly co-located with the hand dorsal surface in every frame,
while the joint angles start from the frame-300 solution rather than from a cold
multi-start. This eliminated the early-frame local-minimum problem that plagued
cold-start frame 18.

### 7b. Per-frame warm-start propagation

After each successfully solved frame, the solution is passed as `q0` to the next:

```python
# main() loop body
prev_q_thumb = np.array([joint_cfg[j] for j in _THUMB_CHAIN])
prev_q_index = np.array([joint_cfg[j] for j in _INDEX_CHAIN])

# On any exception (frame skip):
prev_q_thumb = None   # reset; next frame runs cold multi-start
prev_q_index = None
```

### 7c. Warm-start acceptance inside `_ik_finger()` (local optimizer)

```python
if q0 is not None:
    best_x, best_res = _run_local(q0)

    # Accept fast path ONLY if non-clipping AND within IK tolerance.
    # Previously only the clipping check was here; adding the residual
    # check prevents bad-basin propagation from high-residual solutions.
    if not _is_clipping(best_x) and best_res < cfg.IK_RESIDUAL_THRESHOLD:
        return dict(zip(chain_joints, best_x)), best_res

    # Warm-start clips or has high residual: run all candidates, pick best.
    candidates_results = [(best_x, best_res)]
    for start in starts:
        candidates_results.append(_run_local(start))
    best_x, best_res = _best_of(candidates_results)
else:
    # No warm-start: exhaustive multi-start over all sign-aware candidates.
    candidates_results = [_run_local(s) for s in starts]
    best_x, best_res = _best_of(candidates_results)
```

For `global_de` and `global_bh`, `q0` is injected differently:
- **DE**: `q0` (golden seed or previous frame) is placed as individual 0 of the
  initial population (`init_pop[0] = q0`). The rest of the population is
  Latin-hypercube sampled over the bounds. This biases DE toward the known-good
  region while still exploring globally.
- **BH**: `q0` is used directly as the starting point for the first hop.

---

## 8. Multi-start candidate generation

Candidates are built once at import time and shared across all frames.

```python
_JOINT_DORSAL_DIRECTION = {
    'revolute_1_0': 'na',       # twist axis — swept separately at ±π/2
    'revolute_2_0': 'towards',  # positive bends INTO the hand
    'revolute_3_0': 'away',
    'revolute_4_0': 'away',
    'revolute_5_0': 'away',
    'revolute_6_0': 'towards',
    'revolute_7_0': 'away',
    'revolute_8_0': 'towards',
    'revolute_9_0': 'away',
}

def _build_candidates(chain_joints):
    signs = np.array([+1 if dir=='away' else -1 if dir=='towards' else 0
                      for j in chain_joints
                      for dir in [_JOINT_DORSAL_DIRECTION[j]]])
    active = [i for i,s in enumerate(signs) if s != 0]
    twist  = [i for i,s in enumerate(signs) if s == 0]
    Q, H = np.pi/4, np.pi/2

    cands = []
    # Progressive π/4: add one joint at a time in safe direction
    for k in range(1, len(active)+1):
        q = np.zeros(n); [q.__setitem__(i, signs[i]*Q) for i in active[:k]]
        cands.append(q)
    # Progressive π/2: same but stronger deflection
    for k in range(1, len(active)+1):
        q = np.zeros(n); [q.__setitem__(i, signs[i]*H) for i in active[:k]]
        cands.append(q)
    # Twist sweeps: active joints at Q, twist joint at ±π/2
    q_base = signs * Q
    for i in twist:
        for tv in (H, -H):
            q = q_base.copy(); q[i] = tv; cands.append(q)
    # Fallbacks
    cands.append(np.zeros(n))      # rest pose
    cands.append(-signs * Q)       # opposite direction at π/4
    cands.append(-signs * H)       # opposite direction at π/2
    return cands
```

Thumb chain (4 joints): 11 candidates.
Index chain (5 joints): 13 candidates.

The direction map was determined **empirically**: each joint was set to +0.8 rad
individually and the visual result checked in urdfpy to see whether the linkage
moved toward or away from the dorsal hand surface.

---

## 9. Collision system

### 9a. Mesh proxy points

Link origins can sit inside the hand by construction (joint axes don't move to
the surface). Each tracked link therefore has a **proxy offset** in link-local
frame — the vertex of that link's mesh closest to the dorsal surface at the zero
configuration.

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

### 9b. Quadratic penalty

```python
penalty[i] = _COLLISION_WEIGHT * (penetration ** 2)
# _COLLISION_WEIGHT = 2000.0
# _COLLISION_MARGIN = 0.003  (3 mm)
# penetration = max(0, MARGIN - dist)
```

At `dist = MARGIN` (boundary): penalty = 0.
At `dist = 0` (contact): penalty = 2000 × (0.003)² = 0.018 m ≈ IK residual magnitude.
At `dist = −0.003` (3 mm deep): penalty = 2000 × (0.006)² = 0.072 m ≈ 14× IK residual.

### 9c. Lateral cutoff

```python
lateral = np.linalg.norm(v - dist * plane_normal_root)
if lateral > _COLLISION_LATERAL_CUTOFF:  # 40 mm
    continue
```

Suppresses penalty when the proxy point is >40 mm from the plane centroid —
prevents penalising linkages that have correctly wrapped around the finger sides.

### 9d. Clipping detection

```python
def _is_clipping(q_vec):
    for link, T in fk.items():
        if link.name not in collision_links: continue
        proxy_local = _COLLISION_PROXIES.get(link.name, np.zeros(3))
        pos = T[:3,:3] @ proxy_local + T[:3,3]
        v = pos - plane_centroid_root
        dist = np.dot(v, plane_normal_root)
        if dist < 0:
            lateral = np.linalg.norm(v - dist * plane_normal_root)
            if lateral <= _COLLISION_LATERAL_CUTOFF:
                return True
    return False
```

### 9e. Best-of selection

```python
def _best_of(results):
    non_clip = [(x, r) for x, r in results if not _is_clipping(x)]
    pool = non_clip if non_clip else results
    return min(pool, key=lambda t: t[1])
```

Priority: lowest-residual non-clipping result; if all clip, lowest-residual overall.

---

## 10. Loss tolerances

### 10a. IK Euclidean distance threshold

```python
IK_RESIDUAL_THRESHOLD = 0.005   # 5 mm  (config.py)
```

This is the maximum acceptable distance (in metres) between the finger cap's
dome-tip centroid (`visual_origin` in link-local frame) and the MANO fingertip
target (vertex + calibrated wrist-local offset). It is used in two places:

1. **Warm-start fast-path gate** — a warm-start result is only accepted without
   running multi-start if `ik_residual < IK_RESIDUAL_THRESHOLD` AND not clipping.
2. **Frame assertion** — `_assert_frame()` raises if either finger exceeds this
   threshold. In full-sequence mode failures are skipped; in `--frames` mode they
   are hard errors.

**Design intent:** The annotated dot was placed to represent the exact fingertip
contact point inside the cap. The IK target is that exact world-space position;
5 mm is the maximum we consider acceptable. Frames that converge to 0 mm are
placing the cap dome precisely at the annotated vertex, which is the goal.
Lowering the threshold (e.g. to 2 mm) would cause more frames to fail and trigger
multi-start or hybrid DE fallback, which is the right tradeoff if visual accuracy
matters more than processing speed.

### 10b. Collision (plane-proximity) tolerance

```python
_COLLISION_MARGIN = 0.003   # 3 mm  — penalty activates inside this distance
_COLLISION_WEIGHT = 2000.0  # quadratic scale
```

This is **not** an acceptance threshold — it is a soft penalty that biases the
optimizer away from configurations where proxy points are within 3 mm of the
dorsal plane. There is no hard cutoff for collision: a frame can be accepted even
if a proxy point is marginally inside the hand, as long as the IK residual is
within 5 mm.

---

## 11. Mesh intersection in the optimizer — current status

**Short answer: mesh-on-mesh intersection is NOT part of the optimization loss.**

What *is* in the loss:

```
residual = [ik_err(3), plane_penalty(n)]
```

The `plane_penalty` is a proxy: it measures how far each link's bottommost mesh
vertex is from a calibrated flat dorsal plane, and penalises penetration below
that plane. It is a *geometric approximation* of collision, not actual
mesh-on-mesh intersection testing.

What is **not** in the loss:
- No `trimesh.collision.CollisionManager` checks during optimization.
- No per-face intersection tests.
- No contact-normal forces.

**Why not?** Trimesh collision queries are 2–3 orders of magnitude slower than
the FK + dot-product math in the current penalty. Running a mesh intersection
test at every residual evaluation (200 per TRF call × 2 chains × 1100 frames)
would increase solve time from ~30 s to hours.

**What the diagnostic tool does (offline only):**

```python
manager = trimesh.collision.CollisionManager()
manager.add_object("hand", hand_mesh)
_r = manager.in_collision_single(glove_mesh_in_glb_space)
collides = _r[0] if isinstance(_r, tuple) else bool(_r)
```

This test is run *after* the solve in `diagnose_collision.py` to measure actual
mesh overlap, but the result does not feed back into the optimizer.

**Implication:** A frame can report 0 mm IK residual while still having minor
mesh penetration if the proxy-plane penalty is insufficient to fully exclude that
configuration. The plane is flat; the real hand surface is curved; and some links
(particularly `part_6`) have joint axes that structurally overlap the hand
regardless of joint angle.

---

## 12. Known issues and active problems

### Contorted configurations / local minima

Even with the golden seed and multi-start, the solver can find a solution that
satisfies the 5 mm IK constraint but uses an "inside-out" kinematic configuration
where intermediate links (Part 2, Part 3) are twisted into the thumb or finger.
This happens because:

1. The proxy-plane penalty only covers the dorsal surface — twisting into the
   thumb from the side is not penalised.
2. TRF is a local optimizer: it follows the gradient from the starting point and
   stops at the nearest minimum. If the nearest minimum to the golden seed is a
   contorted one, it finds that.

The recommended fix is `--optimizer global_de` or `--hybrid` for frames that
exhibit this behaviour.

### Clipping in some frames

Despite proxy penalty + multi-start, some frames still clip:

1. All candidates may converge to the same clipping basin if the IK target
   requires a configuration not reachable without passing through the hand.
2. Dorsal plane is flat; real hand surface is curved.
3. `part_6` joint origin sits ~1 mm inside the hand structurally.
4. No temporal smoothness term — large inter-frame motion can leave the warm-start
   on the wrong side of the collision surface.

---

## 13. Output locations

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

## 14. Diagnostic tool

```bash
python glove_sim/tools/diagnose_collision.py --frame 300
python glove_sim/tools/diagnose_collision.py --frame 50 --weights 0 100 500 2000 5000
```

Reports per collision weight: IK residuals, per-link proxy distances (mm),
lateral suppression flags, and trimesh mesh-on-mesh intersection count.

---

## 15. Suggested next diagnostic steps

To distinguish "wrong basin from warm-start" from "target unreachable":

```bash
# Run frame 19 in isolation (no warm-start) — if it passes, warm-start is the problem
python glove_sim/align_frame.py --frames 19

# Compare frame 19 with and without warm-start by checking residuals
# vs running all frames sequentially from 18
python glove_sim/align_frame.py --frames 18 19 20 21 22 23

# Run the full collision diagnostic on a failing early frame
python glove_sim/tools/diagnose_collision.py --frame 19 --weights 0 2000
```

If frame 19 in isolation converges to 0 mm but fails when seeded from frame 18,
the fix is to add a residual-threshold check to the warm-start fast path so that
high-residual (but non-clipping) warm-starts still trigger multi-start.

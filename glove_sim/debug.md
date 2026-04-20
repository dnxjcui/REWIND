# Glove Sim — Debug Reference

This document contains the exact code implementing collision enforcement and the
global optimizer backends, with an honest assessment of what is and is not working.

---

## 1. What "no clipping" actually means in this codebase

There are **two independent clipping checks**, and they test different things.

### 1a. Proxy-plane check (`_is_clipping`)

A fast approximation used inside the solver selection logic. Measures the signed
distance of each link's *proxy point* (the bottommost vertex of its mesh at zero
config, hardcoded as an offset in link-local frame) from a calibrated flat dorsal
plane. Returns True if any proxy point has `dist < 0` and is within 40 mm
laterally of the plane centroid.

```python
_COLLISION_PROXIES = {
    'xl330_m077_t_1':    np.array([ 0.009, -0.0245, -0.0032]),
    'xl_linkage_horn':   np.array([ 0.003, -0.003,   0.0072]),
    'part_2_1':          np.array([-0.005,  0.0025,  0.0006]),
    'part_6':            np.array([-0.008, -0.0009, -0.0543]),
    'xl_housing_1':      np.array([ 0.012, -0.0267,  0.000 ]),
    'part_2':            np.array([-0.004,  0.0025,  0.0025]),
    'xl_linkage_horn_1': np.array([ 0.016, -0.0063,  0.0043]),
}
_COLLISION_MARGIN         = 0.003   # 3 mm — penalty activates inside this
_COLLISION_LATERAL_CUTOFF = 0.200   # 200 mm — finger links extend up to ~120mm from centroid

def _is_clipping(q_vec) -> bool:
    fk = robot.link_fk(cfg=dict(zip(chain_joints, q_vec)))
    for link, T in fk.items():
        if link.name not in collision_links:
            continue
        proxy_local = _COLLISION_PROXIES.get(link.name, np.zeros(3))
        pos = T[:3, :3] @ proxy_local + T[:3, 3]
        v = pos - plane_point_root          # vector from plane centroid to proxy
        dist = float(np.dot(v, plane_normal_root))
        if dist < 0:
            lateral = float(np.linalg.norm(v - dist * plane_normal_root))
            if lateral <= _COLLISION_LATERAL_CUTOFF:
                return True
    return False
```

**What it misses:** The plane is flat and infinite (modulo lateral cutoff). It
does not protect the curved sides of the finger, the inter-finger gap, or any
region that is not dorsal. A linkage can twist into the thumb from the side and
`_is_clipping` will not fire.

The proxy-plane penalty is also added as a soft cost inside the TRF residual:

```python
# Inside residual(q_vec) — this runs at every TRF function evaluation:
penetration = _COLLISION_MARGIN - dist   # > 0 when inside the margin
penalty[i]  = _COLLISION_WEIGHT * (penetration ** 2)
# _COLLISION_WEIGHT = 2000.0
# At dist=0 (contact): 2000 * 0.003^2 = 0.018 m ≈ 18 mm  (large vs 5mm threshold)
# At dist=-3mm (deep): 2000 * 0.006^2 = 0.072 m ≈ 72 mm  (14× IK cost)
```

### 1b. Exact mesh check (`_exact_collision`)

Transforms each danger-link visual mesh into GLB space and queries a
`trimesh.collision.CollisionManager` loaded with that frame's MANO hand mesh.
Tests actual face-face intersection, not a proxy approximation.

```python
_DANGER_LINKS = frozenset({
    'part_2', 'part_3', 'xl_linkage_horn_1', 'part_6',  # index chain
    'part_2_1', 'part_3_1', 'xl_linkage_horn',          # thumb chain
    # Deliberately excludes: hand_mount, xl330_m077_t, xl_housing
    # (base links sit on dorsal surface, protected by proxy-plane)
})

def _exact_collision(q_vec) -> bool:
    if hand_manager is None or T_root_world is None:
        return False
    fk = robot.link_fk(cfg=dict(zip(chain_joints, q_vec)))
    T_to_glb = _MANO_TO_GLB @ T_root_world   # root-local → MANO world → GLB Y-up

    for link, T_link in fk.items():
        if link.name not in _DANGER_LINKS:
            continue
        for visual in link.visuals:
            mesh = _get_visual_mesh(visual)         # urdfpy visual → trimesh.Trimesh
            if mesh is None:
                continue
            origin = visual.origin if visual.origin is not None else np.eye(4)
            T_vis_glb = T_to_glb @ T_link @ origin  # mesh verts → GLB space
            try:
                _r = hand_manager.in_collision_single(mesh, transform=T_vis_glb)
                collides = _r[0] if isinstance(_r, tuple) else bool(_r)
                if collides:
                    return True
            except Exception:
                pass
    return False
```

**Per-frame setup in `main()`:**

```python
hand_glb     = cfg.GLB_DIR / f"{frame_idx:06d}_hands.glb"
hand_manager = _build_hand_manager(hand_glb)   # None if file absent
# Passed into solve_ik_frame → _ik_finger → _exact_collision closure
```

**When `hand_manager` is None** (GLB file missing for that frame), both
`_exact_collision` and `_best_of`'s exact tier silently degrade to proxy-only.
This is a real gap: if the hands GLB is unavailable, there is no exact check.

---

## 2. Where the clipping enforcement is applied

### 2a. Warm-start gate (Stratum 1)

After TRF refines the warm-start (golden seed or previous frame), the result must
pass **all three** gates before being accepted without trying any other candidate:

```python
if q0 is not None:
    best_x, best_res = _run_local(q0)

    warm_ok = (
        not _is_clipping(best_x)               # proxy-plane gate
        and best_res < cfg.IK_RESIDUAL_THRESHOLD  # 5 mm IK gate
        and not _exact_collision(best_x)        # exact mesh gate
    )
    if warm_ok:
        return dict(zip(chain_joints, best_x)), best_res

    # Any gate failed → fall through to multi-start
    candidates_results = [(best_x, best_res)]
    for start in starts:
        candidates_results.append(_run_local(start))
```

### 2b. Multi-start filter (Stratum 2)

All TRF candidate results are filtered. A result only qualifies if it passes
both the proxy-plane check and the exact mesh check:

```python
def _best_of(results):
    # Tier 1: proxy-OK AND exact-OK
    tier1 = [(x, r) for x, r in results
             if not _is_clipping(x) and not _exact_collision(x)]
    if tier1:
        return min(tier1, key=lambda t: t[1])   # lowest IK residual

    # Tier 2: no hand mesh loaded → proxy-only fallback
    if hand_manager is None:
        proxy_ok = [(x, r) for x, r in results if not _is_clipping(x)]
        pool = proxy_ok if proxy_ok else results
        return min(pool, key=lambda t: t[1])

    # All candidates intersect → return sentinel to trigger DE
    return None, None
```

### 2c. Basin-hopping accept gate (per-hop)

Each BH hop is rejected before it can propagate if the local TRF converged to an
intersecting configuration:

```python
def _hop_accept(f_new, x_new, f_old, x_old):
    # Called by scipy AFTER each local minimizer converges.
    # Returning False keeps x_old; BH tries a different random step.
    if _exact_collision(np.clip(x_new, lower_arr, upper_arr)):
        return False
    return True   # standard Metropolis acceptance criterion applies

res = scipy.optimize.basinhopping(
    cost, start, niter=10,
    minimizer_kwargs={"method": _local_trf},
    accept_test=_hop_accept,
    seed=0,
)
```

### 2d. Auto-DE fallback (Stratum 3)

If `_best_of` returns `(None, None)` (every TRF candidate intersected), DE is
triggered automatically for that finger on that frame:

```python
best_x, best_res = _best_of(candidates_results)
if best_x is None:
    best_x, best_res = _run_de(seed_x=q0, use_exact_collision=True)
```

---

## 3. Global optimizer implementations

### 3a. Differential evolution

```python
def _run_de(seed_x=None, use_exact_collision=False):
    n_pop    = max(7 * n, 15)      # e.g. 28 for thumb (4 joints), 35 for index (5)
    rng      = np.random.default_rng(0)
    init_pop = rng.uniform(lower_arr, upper_arr, (n_pop, n))
    if seed_x is not None:
        init_pop[0] = np.clip(seed_x, lower_arr, upper_arr)  # golden seed as ind. 0

    _ik_sq_threshold = (3.0 * cfg.IK_RESIDUAL_THRESHOLD) ** 2  # (15mm)^2

    def cost_fn(q_vec):
        q = np.clip(q_vec, lower_arr, upper_arr)
        r = residual(q)
        c = float(np.dot(r, r))   # sum of squares of [ik_err(3), proxy_penalty(n)]
        if use_exact_collision:
            # Exact check only for "promising" candidates — avoids budget on garbage
            if float(np.dot(r[:3], r[:3])) < _ik_sq_threshold:
                if _exact_collision(q):
                    c += 1.0   # 1.0 m² >> any realistic IK cost
        return c

    res = scipy.optimize.differential_evolution(
        cost_fn,
        list(zip(lower_arr.tolist(), upper_arr.tolist())),
        init=init_pop,
        updating='deferred',   # whole pop evaluated before any update (stable)
        maxiter=150,
        tol=1e-10,
        seed=0,
        polish=(not use_exact_collision),  # TRF polish disabled with discontinuous cost
    )
    x = np.clip(res.x, lower_arr, upper_arr)

    # Manual TRF polish when exact collision was in the objective
    if use_exact_collision:
        xp, _ = _run_local(x)
        if not _exact_collision(xp) and not _is_clipping(xp):
            x = xp

    return x, _ik_res_of(x)
```

### 3b. Basin hopping

```python
def _run_bh(start):
    start = np.clip(start, lower_arr, upper_arr)

    def _local_trf(fun, x0, args=(), **options):
        x0 = np.clip(x0, lower_arr, upper_arr)   # BH random step can exceed bounds
        r = scipy.optimize.least_squares(
            residual, x0, bounds=(lower_arr, upper_arr),
            method='trf', ftol=cfg.IK_TOL, max_nfev=cfg.IK_MAX_NFEV,
        )
        class _Result:
            x   = r.x
            fun = float(np.dot(r.fun, r.fun))
            success = True
        return _Result()

    def _hop_accept(f_new, x_new, f_old, x_old):
        if _exact_collision(np.clip(x_new, lower_arr, upper_arr)):
            return False   # reject hop; BH keeps x_old and tries again
        return True        # Metropolis criterion applies

    res = scipy.optimize.basinhopping(
        cost, start, niter=10,
        minimizer_kwargs={"method": _local_trf},
        accept_test=_hop_accept,
        seed=0,
    )
    x = np.clip(res.x, lower_arr, upper_arr)
    return x, _ik_res_of(x)
```

If BH's final result still intersects (all 10 hops rejected, so `x` is the
original `start`), the BH dispatch falls back to DE:

```python
if optimizer == 'global_bh':
    start = q0 if q0 is not None else np.zeros(n)
    best_x, best_res = _run_bh(start)
    if hand_manager is not None and _exact_collision(best_x):
        best_x, best_res = _run_de(seed_x=best_x, use_exact_collision=True)
    return dict(zip(chain_joints, best_x)), best_res
```

---

## 4. Why global optimizers are currently performing worse than local + golden seed

This is the honest assessment.

### 4a. The global optimizers don't know about temporal continuity

The golden seed (frame 300) provides a physically correct starting configuration.
Local TRF from that seed exploits the fact that **hand motion between adjacent
frames is small** — the correct solution for frame N is usually within a few
degrees of frame N-1. TRF follows the gradient downhill and arrives at the right
local minimum quickly.

Global optimizers (DE, BH) don't care about temporal proximity. DE samples the
entire `[-π, π]^n` space from scratch. BH takes random steps of 0.5 rad (~29°)
which is large relative to inter-frame joint change. Both explore configurations
that are kinematically valid but physically wrong (e.g., an inside-out wrist
rotation that happens to put the fingertip in roughly the right place).

### 4b. The scalar cost function for DE has no temporal term

DE minimises:

```
cost = ||ik_err||² + ||proxy_penalty||² + [1.0 if exact_collision]
```

There is no term penalising deviation from the golden seed or from the previous
frame's configuration. Two configurations with identical IK error and no
collision are treated as equally good even if one is wildly contorted relative
to frame N-1.

### 4c. BH's random step size is not tuned for this problem

The default `stepsize=0.5` rad is large. For the thumb chain (4 joints at ~0.3 rad
each in the golden seed), a step of 0.5 rad frequently lands in a basin on the
wrong side of the hand. With `_hop_accept` rejecting intersecting hops, BH may
never find a successful hop and stays at `start` for all 10 iterations — which
is then exact-checked again at dispatch, possibly triggering DE.

### 4d. What should actually be done

For most frames, **local TRF + golden seed is correct and should be the default**.
The global optimizers should be reserved as an *explicit fallback* for frames that
local TRF demonstrably fails (IK residual > 5 mm or exact collision detected).

The Stratum 3 auto-DE-fallback already does this for the case where every
multi-start candidate intersects. The `--hybrid` flag does it for the case of
high IK residual.

For frames that are still visually clipping despite 0 mm IK residual: the
clipping is most likely happening in links **not in `_DANGER_LINKS`**, or the
exact collision transform is wrong, or the hand GLB is absent for that frame.

---

## 5. Recommended run commands

```bash
# Sanity check — frame 300 should always be 0.0mm / 0.0mm
python glove_sim/align_frame.py --frames 300

# Early-frame spot check (historically hardest)
python glove_sim/align_frame.py --frames 18 19 20 30 50 100

# 10-frame evenly-spaced sweep across the full sequence
python glove_sim/align_frame.py --frames-test

# Full sequence — local TRF + golden seed (recommended, ~30s)
python glove_sim/align_frame.py

# Full sequence + auto-DE fallback on high-residual frames
python glove_sim/align_frame.py --hybrid

# Per-frame collision diagnostic on a specific bad frame
python glove_sim/tools/diagnose_collision.py --frame N

# Export URDFs for visual inspection alongside GLBs
python glove_sim/align_frame.py --frames 18 50 100 --urdf
```

**Do not use `--optimizer global_de` or `--optimizer global_bh` for the full
sequence.** They are 10–25× slower and perform worse due to the issues described
in §4. Use them only as targeted diagnostic tools on specific failing frames.

---

## 6. Diagnostic finding: lateral cutoff was too small (fixed)

Running `python glove_sim/tools/diagnose_collision.py --frame 18` produced:

```
part_6:            -3.4 mm  lateral=104.2mm  *** INSIDE HAND [SUPPRESSED - off-side]
xl_housing_1:      -6.8 mm  lateral=110.3mm  *** INSIDE HAND [SUPPRESSED - off-side]
part_2:           -44.5 mm  lateral= 45.7mm  *** INSIDE HAND [SUPPRESSED - off-side]
Mesh-level collision: 0/21 meshes
```

Every proxy point was suppressed because all finger links are 45–120mm from the
dorsal plane centroid — above the 40mm lateral cutoff.  This meant:

- `_is_clipping` always returned **False** for every frame (no proxy-plane gating at all).
- The soft proxy penalty inside TRF was **never evaluated** (no gradient guidance).

The exact mesh collision check (0/21) is authoritative and was working correctly:
**frame 18 has no actual mesh intersection.**  The proxy check was just silently
broken.

**Fix:** `_COLLISION_LATERAL_CUTOFF` raised from 40mm to 200mm.  Finger links
extend at most ~120mm from the dorsal centroid, so 200mm captures them all while
still suppressing links that have wrapped all the way to the palmar side.

**Why 0/21 mesh collisions despite proxy distances showing negative values:**
The dorsal plane is a flat infinite surface — a link can be on the "wrong side"
of it (negative signed distance) while still being geometrically clear of the
actual curved hand mesh.  The exact trimesh check is the ground truth; the proxy
is only a fast approximation.

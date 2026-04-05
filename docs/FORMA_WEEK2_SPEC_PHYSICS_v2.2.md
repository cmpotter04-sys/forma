# FORMA — Week 2 Spec: Physics Hardening & Edge Cases
**Version:** 2.2 (post-Week 1, self-reviewed, co-strategist reviewed)  
**Phase:** 1 / Week 2  
**Owner:** Founder  
**Executor:** Claude Code CLI  
**Prerequisite:** Week 1 complete (commit 4776e4c, 210/210 tests, adversarial audit closed)  
**Status:** READY TO BUILD

---

## What Changed From v1.0

Week 1 and the adversarial audit surfaced several issues that require spec
revisions:

1. **New AC-0: MakeHuman full-body mesh.** The Week 1 parametric torso cylinder
   has no arms, no shoulder slope, no realistic contour. Sleeve attachment,
   shoulder fit, and body collision all require a real body. This must come first
   because every other AC depends on it.

2. **Size S does not produce fit=False.** The GarmentCode size S T-shirt is 88cm
   circumference on a 95cm body — only a 7cm deficit, producing yellow (-13mm)
   not red (-25mm). The original spec expected XS and S to be fit=False. Revised:
   only XS must be fit=False (via a scaled-down pattern).

3. **Solver convergence is geometry-dominated.** The XPBD solver converges in 2-3
   steps because cylindrical pre-placement puts panels near equilibrium. Fabric
   sensitivity (AC-3) is only meaningful after body collision is enabled (AC-2),
   because collision gives fabric stiffness something to push against.

4. **Adversarial audit deferred items are now in-scope:**
   - No fit=False from real simulation → addressed in AC-1
   - Sleeve exclusion → addressed in AC-2
   - Tunnel-through metric meaningless without body collision → addressed in AC-2

5. **Co-strategist review (Gemini) identified four critical additions:**
   - Constraint strain ratio as secondary fit metric (post-collision, clearance
     alone cannot detect tightness)
   - Angle-sorted perimeter for circumference measurement (convex hull
     over-measures concave torso cross-sections)
   - Seam warm-up protocol for sleeve attachment stability
   - Hard-pivot rule: collision > sleeves if AC-2 stalls

---

## Objective

Upgrade the Week 1 proof-of-concept into a physically credible simulation.
By end of Week 2, the Geometer runs on a realistic body mesh with arms,
handles sleeve attachment, enforces body collision, differentiates fabrics,
and produces at least one genuine fit=False result from the full pipeline.

This is the second go/no-go gate.

---

## Acceptance Criteria

### AC-0: MakeHuman Full-Body Mesh (DO THIS FIRST)

Replace the parametric torso cylinder with a MakeHuman CC0 full-body mesh.

**Implementation:**
- Use the MakeHuman base.obj (21,833 verts, CC0) with morph targets from the
  makehuman pip package (already installed)
- Apply `caucasian-male-young` macro morph (gets to ~179.5cm)
- Iterate `measure-bust-circ`, `measure-waist-circ`, `measure-hips-circ` morphs
  to hit target measurements: chest=96cm, waist=80cm, hip=95cm
- Uniform scale to nail 180cm height exactly
- Export as PLY to `data/bodies/makehuman_male_M.ply`

**Circumference measurement procedure (angle-sorted perimeter):**
Convex hull over-measures because the human torso is concave at the spine. Use
angle-sorted vertex perimeter instead:
1. Identify the Y-height of maximum torso circumference within the chest band
   (Y = 1.12m to 1.38m, per region_map.py)
2. Take all vertices within ±1cm of that Y-height
3. Exclude vertices with |X| > 0.22m (arm vertices in T-pose extend beyond this)
4. Project remaining vertices into the XZ plane
5. Compute the centroid of the projected points
6. Sort vertices by their angle from the centroid (atan2(z - cz, x - cx))
7. Walk the sorted list, summing Euclidean distances between consecutive points
   (including the wrap-around distance from last point back to first)
8. That sum is the circumference

This follows the actual surface contour including spinal concavity, unlike
convex hull which bridges across it. Apply the same method at waist and hip
heights, adjusting the X-exclusion threshold as needed.

For overall height, use max(Y) - min(Y).

**Generate ALL THREE body sizes before moving to AC-1:**

| Body | Chest | Waist | Hip | Height | Filename |
|------|-------|-------|-----|--------|----------|
| S    | 88cm  | 72cm  | 87cm | 170cm | makehuman_male_S.ply |
| M    | 96cm  | 80cm  | 95cm | 180cm | makehuman_male_M.ply |
| XL   | 108cm | 96cm  | 105cm | 185cm | makehuman_male_XL.ply |

**Pass conditions:**
- Full body mesh with arms (at least to elbows), realistic shoulder slope,
  torso contour
- All three bodies within ±2cm of target circumferences and ±1cm of height
  (measured using the angle-sorted perimeter method above)
- Vertex count under 50,000 per mesh
- CC0 license confirmed
- body_profile.json updated with new mesh paths and measurements

**After M body is ready, immediately:**
1. Update `region_map.py` height bands if needed for the new body proportions
2. Re-run S/M/XL garment sweep on new M body, print clearance maps
3. Establish new baselines — Week 1 numbers are invalidated by the body swap
4. Update all test assertions that reference old clearance values
5. All existing tests must pass on the new body
6. **Update CLAUDE.md** — change body mesh path references from
   `mannequin_sizeM_180cm.ply` to `makehuman_male_M.ply` and note that the
   body is now a full MakeHuman mesh with arms, not a parametric torso cylinder

### AC-1: Extended Size Range + fit=False Test

**Depends on:** AC-0 complete (all three bodies generated).

Run 5 garment sizes on the M body: XS, S, M, L, XL.

**Pattern generation:** GarmentCodeData may not have all 5 sizes. If XS and L
patterns don't exist, generate them by scaling the M pattern:
- XS: scale all panel dimensions by 0.85 (target circumference ~85cm)
- L: scale all panel dimensions by 1.08 (target circumference ~108cm)
- Save to `data/patterns/tshirt_size_{XS,L}.json` with corresponding seam
  manifests

**Expected monotonic relationship:**
```
XS: most negative delta_mm (tightest) — fit=False (at least one red region)
S:  negative delta_mm — fit=True with yellow regions (per Week 1 findings)
M:  small positive delta_mm — fit=True, all green
L:  moderate positive delta_mm — fit=True, all green
XL: large positive delta_mm — fit=True, all green
```

**Pass conditions:**
- All 5 simulations converge without explosion
- chest_front delta_mm is strictly monotonically increasing: XS < S < M < L < XL
- **XS fit=False** — at least one red region (delta < -25mm). This is the
  end-to-end fit=False test the adversarial audit requested.
- S fit=True (yellow acceptable, per Week 1 revised AC)
- M, L, XL fit=True
- Save all 5 verdicts to `output/verdicts/extended/`

**Why XS not S for fit=False:** Size S is only 7cm smaller than the M body (88
vs 95cm). That produces -13mm (yellow). XS at ~85cm will be ~10cm smaller,
which should produce -20 to -30mm clearance — enough for red in free-space
draping mode.

**Cross-body test:** Also run size M garment on all 3 body sizes:
- M garment on S body → should show positive ease (body smaller than garment)
- M garment on M body → baseline fit
- M garment on XL body → should show negative clearance (body larger)
Save to `output/verdicts/crossbody/`

**Important — re-run after AC-2:** After body collision and strain ratio are
enabled in AC-2, re-run the full size sweep. See AC-2 Sub-Problem 2F for the
updated fit logic.

### AC-2: Sleeve Attachment + Body Collision + Strain Metric

This is the hardest AC in Week 2. It addresses three adversarial audit concerns
(sleeve exclusion, body collision, tunnel-through) plus a co-strategist
finding (strain-based fit detection post-collision).

**HARD-PIVOT RULE:** If sub-problems 2A-2C (sleeve attachment) are not working
after 3 calendar days, STOP sleeve work immediately. Hard-pivot to:
- Enable body collision for torso panels only (sub-problems 2D, 2E, 2F)
- Keep sleeves excluded (same as Week 1)
- Defer sleeve attachment to Week 3
- Mark AC-2 as PARTIAL and proceed with AC-3, AC-4, AC-5

**Collision is more important than sleeves.** A torso-only simulation with
correct collision and strain detection is a more valuable API than a full-body
simulation that constantly explodes. Protect the Week 4 API delivery date.

#### Sub-Problem 2A: Identify the Armhole Boundary

Before placing sleeves, we need to know where the armhole is on the assembled
torso.

1. After cylindrical wrapping, the torso panels form a tube. The "armhole" is
   the open gap at the top of each side where front panel and back panel side
   seams end, below the shoulder seam.
2. Identify armhole boundary vertices: these are the topmost vertices of the
   torso side seam edges (the edges that connect front-to-back panels on each
   side). They form a rough curve at each arm opening.
3. Compute the armhole centroid (mean position of armhole boundary vertices).
   This is the target position for the sleeve cap center.
4. Compute the armhole plane normal (roughly pointing laterally outward — the
   direction the arm extends in T-pose).

The key data needed: armhole centroid position (x, y, z) and armhole radius
(half the armhole edge arc length / π). These come from the assembled torso,
not from the body mesh.

#### Sub-Problem 2B: Sleeve Initial Placement

Position each sleeve panel so its cap aligns with the armhole:

1. Compute the sleeve cap curve — the edge of the sleeve panel that attaches
   to the armhole (typically the longest edge or the edge labeled "sleeve_cap"
   in the seam manifest).
2. Translate the entire sleeve panel so the sleeve cap centroid aligns with
   the armhole centroid from 2A.
3. Rotate the sleeve panel to lie roughly along the arm direction. For T-pose:
   left sleeve extends in +X, right sleeve extends in -X. The sleeve's long
   axis (cap to cuff) should align with the arm.
4. Optionally: wrap the sleeve around the arm cylinder (similar to how torso
   panels wrap around the body cylinder). This reduces initial seam gaps
   further.

Target: sleeve cap to armhole gap < 3cm after placement, before XPBD runs.

#### Sub-Problem 2C: Enable Sleeve Seam Constraints (with Warm-up)

In `_assemble_garment`, remove the filter that excludes sleeve-to-torso seams
(lines 536-556 of xpbd_simulate.py). With proper initial placement from 2B,
the seam gaps should be small enough for the solver to close without
destabilizing the torso.

**Seam warm-up protocol (critical for stability):**
During the first 20 XPBD steps, cap `max_correction_m` for sleeve-to-torso
seam constraints at **0.005m (5mm)**. This prevents the high-tension seam
constraints from snapping the sleeve through the arm mesh before the collision
solver can react. After step 20, relax to the normal cap.

Implementation:
```python
# In _run_xpbd, distinguish sleeve seams from torso seams:
for step in range(max_steps):
    # ... gravity ...
    sleeve_max_corr = 0.005 if step < 20 else 0.05   # warm-up
    _solve_distance_constraints_batch(..., max_correction_m=0.05)      # stretch
    _solve_distance_constraints_batch(..., max_correction_m=0.05)      # torso seams
    _solve_distance_constraints_batch(..., max_correction_m=sleeve_max_corr)  # sleeve seams
    _apply_body_collision(...)
```

If the solver still explodes after warm-up:
- Try soft compliance (alpha > 0) on sleeve seams instead of hard (alpha = 0)
- Reduce n_constraint_iters for the first 20 steps
- If nothing works, trigger the HARD-PIVOT RULE

#### Sub-Problem 2D: Body Collision During XPBD

Enable `_apply_body_collision` inside the solver loop:

```python
for _ in range(n_constraint_iters):
    _solve_distance_constraints_batch(...)   # stretch
    _solve_distance_constraints_batch(...)   # seams
_apply_body_collision(positions, body_tree, body_vertices, body_normals, margin_m=0.001)
```

Collision margin: 1mm. Any garment vertex closer than 1mm to the body surface
(on the inside) gets pushed out to 1mm outside.

Note: collision is applied ONCE per step AFTER all constraint iterations, not
inside the constraint iteration loop. Applying collision inside the inner loop
fights the constraint solver and can cause oscillation.

#### Sub-Problem 2E: Tunnel-Through Becomes Meaningful

With body collision active, any garment vertex inside the body surface is a
real tunneling error (the collision solver failed to push it out), not expected
free-space overlap.

The existing `detect_tunnel_through` function checks for vertices within 2mm
of the body surface on the inside. This should now correctly detect collision
failures. Re-enable the 2% hard limit.

#### Sub-Problem 2F: Constraint Strain Ratio — Secondary Fit Metric

**This is the key addition from co-strategist review.**

With body collision enabled, clearance alone cannot detect tightness. An
undersized garment compressed against the body shows clearance ≈ +1mm (the
collision margin), indistinguishable from a correctly-fitting garment. The
garment's internal strain reveals the difference.

**Strain ratio definition:**
For each stretch constraint (edge between adjacent garment vertices):
```
strain_ratio = current_length / rest_length
```
- strain_ratio = 1.0 → edge at rest length (no deformation)
- strain_ratio > 1.0 → edge stretched beyond rest length (tension)
- strain_ratio < 1.0 → edge compressed below rest length

**Per-region median strain ratio:**
After simulation, for each body region, compute the median strain_ratio of all
stretch constraint edges whose vertices belong to that region.

**Updated fit rule (post-collision):**
```
region_severity = "red"    if delta_mm < -25  OR  median_strain_ratio > 1.15
region_severity = "yellow" if delta_mm < -10  OR  median_strain_ratio > 1.08
region_severity = "green"  otherwise
fit = True if zero regions have severity "red"
```

The strain thresholds (1.15 for red, 1.08 for yellow) are starting values for
cotton jersey. They should eventually be fabric-dependent (denim ruptures at
lower strain, stretch fabrics tolerate more). For Week 2, a single set of
thresholds is acceptable.

**Implementation:**
1. After XPBD converges, compute strain_ratio for every stretch constraint
2. Assign each constraint to a body region (both endpoint vertices must be in
   the same region; skip constraints that span regions)
3. Compute median strain_ratio per region
4. Add `median_strain_ratio` to each region entry in the fit_verdict.json
   strain_map
5. Update `classify_severity` in clearance.py to accept strain_ratio as an
   optional parameter and apply the OR logic above

**Schema addition to fit_verdict.json strain_map entries:**
```json
{
  "region": "chest_front",
  "delta_mm": 1.2,
  "severity": "red",
  "median_strain_ratio": 1.18
}
```

**Critical: Do NOT remove the delta_mm-based severity.** Strain ratio is
additive — it catches tightness that delta_mm misses post-collision. Both
metrics contribute. The OR logic means either one can trigger yellow or red.

**Pass conditions for AC-2:**
- Sleeves visually attach to torso at armhole (generate 4-angle visualization
  for before/after comparison) — OR sleeves excluded if HARD-PIVOT triggered
- If sleeves attached: seam gaps < 2cm after simulation, warm-up protocol in
  place
- All 6 regions have garment vertex counts > 30 — OR if HARD-PIVOT: same
  regions as Week 1 (sleeves excluded from shoulder measurement)
- Body collision active during XPBD iteration
- Tunnel-through < 2% for M garment on M body
- **Strain ratio computed and reported** in fit_verdict.json for all regions
- Size M T-shirt on M body still fit=True after all changes
- Re-run full size sweep with updated fit logic (delta_mm OR strain_ratio)
  and print results. Bring to Strategist before adjusting any thresholds.

### AC-3: Fabric Parameter Sensitivity

**Depends on:** AC-2 complete (body collision enabled). Without collision,
fabric stiffness barely affects the solver and the test is trivially weak.

Run size M T-shirt on M body with 3 fabrics:
- `cotton_jersey_default` (baseline)
- `silk_charmeuse` (low stiffness, high drape)
- `denim_12oz` (high stiffness, low drape)

**Expected behavior with collision enabled:** Stiffer fabrics resist compression
against the body surface more, producing slightly larger positive clearance and
lower strain ratios. Drapey fabrics conform more closely to the body surface,
producing smaller clearance and potentially higher strain ratios where the
garment is pulled tight.

**Pass conditions:**
- All 3 simulations converge
- All 3 show fit=True
- At least 2 of the 3 fabrics produce different chest_front delta_mm values
  (difference > 0.5mm)
- No fabric flips fit verdict for same garment+body pair
- Save verdicts to `output/verdicts/fabric_test_{fabric_id}.json`

**If AC-2 is only PARTIAL (collision enabled but sleeves excluded):** AC-3 can
still run. The collision interaction with fabric stiffness applies to torso
panels regardless of sleeve state. Proceed and note the limitation.

### AC-4: Second Garment Type — Long Sleeve Shirt

**Depends on:** AC-0 (need arms on body). Ideally also AC-2 (sleeve
attachment), but can run with sleeves excluded if AC-2 is PARTIAL.

Load a long-sleeve shirt from GarmentCodeData (6+ panels).

**Pattern selection:** Search the GarmentCodeData repository for a shirt pattern
with long sleeves. Candidates: any pattern with panel names containing "sleeve"
and panel count ≥ 6. If no suitable pattern exists, use the T-shirt pattern as
a baseline and skip this AC (note as SKIPPED with reason).

**Pass conditions:**
- Pattern loads with valid seam_manifest (more seam pairs than T-shirt)
- Simulation converges on M body
- All 6 regions populated with garment vertices (if sleeves attached)
- Size M long-sleeve on M body → fit=True
- Save verdict to `output/verdicts/longsleeve_M_on_M.json`

**If AC-2 is PARTIAL:** Run with sleeves excluded. Shoulder regions will have
approximate clearance from torso panels only (same limitation as Week 1).
Mark verdict as `sleeve_mode: "excluded"`.

### AC-5: Pipeline Integration Function

Build `src/pipeline.py` with a clean programmatic interface:

```python
def run_fit_check(
    body_mesh_path: str,
    pattern_path: str,
    seam_manifest_path: str,
    fabric_id: str = "cotton_jersey_default",
) -> dict:
    """
    Full pipeline: body + pattern + fabric → fit_verdict dict (v1.2 schema)

    Raises:
        SeamValidationError — if seam manifest is invalid
        SimulationExplosionError — if XPBD solver diverges
        FileNotFoundError — if any input path doesn't exist
    """
```

**Pass conditions:**
- Runs end-to-end without shell subprocess calls
- Returns valid fit_verdict dict matching v1.2 schema (including
  median_strain_ratio per region if AC-2 complete)
- Raises typed exceptions (not generic RuntimeError)
- Calling 3 times with S/M/XL produces deterministic, consistent results
- Can be called in a loop for batch processing

---

## Execution Order

The ACs have strict dependencies:

```
AC-0  (MakeHuman body — all 3 sizes)
  ↓
AC-1  (extended garment sizes + cross-body test)
  ↓
AC-2  (sleeves + collision + strain metric)   ← hardest, HARD-PIVOT after 3 days
  ↓
AC-3  (fabric sensitivity)                    ← needs collision from AC-2
  ↓
AC-4  (long sleeve shirt)                     ← needs body from AC-0, ideally AC-2
  ↓
AC-5  (pipeline function)                     ← integration wrapper, do last
```

**Parallelization opportunities:**
- AC-5 can be started in parallel with AC-2 (wraps existing pipeline, doesn't
  depend on collision — update later to include strain_ratio)
- AC-4 can run in parallel with AC-3 after AC-2 completes (or after HARD-PIVOT)
- AC-1 cross-body tests can run as soon as AC-0 generates all 3 bodies

**After AC-2 completes (or HARD-PIVOT), re-run:**
- AC-1's full size sweep with updated fit logic (delta_mm OR strain_ratio)
- AC-1's XS test (clearance values will change with collision; strain_ratio
  should now detect tightness even if clearance is clamped)
- The Week 1 visualization (before/after comparison)

---

## Go / No-Go Decision Criteria

**GO if:**
- MakeHuman body meets measurement targets (all 3 sizes)
- Monotonic size relationship holds across 5 garment sizes
- At least one genuine fit=False from real simulation (XS)
- Body collision is enabled and working
- Strain ratio is computed and reported in verdicts
- Fabric sensitivity shows measurable differences (> 0.5mm)
- Pipeline function works programmatically
- All tests pass

**GO WITH CAVEATS if:**
- Sleeve attachment is partial or deferred (HARD-PIVOT triggered — collision
  on torso only, sleeves excluded, deferred to Week 3)
- Long-sleeve shirt pattern not found in GarmentCodeData (AC-4 SKIPPED)
- Fabric differences are measurable but small (0.5-2mm)
- Strain ratio thresholds are approximate (single set for all fabrics)

**NO-GO if:**
- Monotonic relationship breaks (clearance computation has systemic bug)
- MakeHuman body can't hit measurement targets within ±5cm
- Body collision causes solver explosion that can't be stabilized within 200
  steps, even in torso-only mode
- No garment size produces fit=False from real simulation (neither delta_mm
  nor strain_ratio triggers red)
- Pipeline function can't run end-to-end

If NO-GO: bring failing verdicts and solver logs to Strategist for diagnosis.

---

## What Comes Next

**Week 3:** Archivist (Supabase integration) — FORMA_WEEK3_SPEC_SUPABASE.md  
**Week 4:** REST API + developer docs — FORMA_WEEK4_SPEC.md

If HARD-PIVOT was triggered in AC-2, Week 3 scope expands to include sleeve
attachment as a parallel workstream alongside Supabase integration.

---

*FORMA Architecture v1.2.1 — Phase 1 Week 2 Spec v2.2*  
*Revised based on Week 1 findings, adversarial audit, self-review, and co-strategist review*  
*Do not modify acceptance criteria without founder sign-off*

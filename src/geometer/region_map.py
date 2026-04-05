"""
src/geometer/region_map.py

Body region segmentation for the parametric torso mesh.
No SMPL-X — uses vertex height and normal direction to assign each body
vertex to one of the Forma regions.

Supports two body coordinate systems automatically:
    MakeHuman  — Y-up  (col 1 is height), facing +Z (col 2 normal = front)
    Anny       — Z-up  (col 2 is height), facing +Y (col 1 normal = front)

The up axis is auto-detected as the column with the greatest vertex extent.
Height bands are expressed as proportions of total body height so they work
for any mesh scale or coordinate orientation.

Garment coverage per region set:

  TSHIRT_REGIONS   — torso only (chest, shoulders, upper back, waist)
  DRESS_REGIONS    — torso + hip + thigh  (adds hip, thigh_left, thigh_right)
  TROUSERS_REGIONS — lower body (adds seat, inseam, knee_left, knee_right,
                      thigh_left, thigh_right)

All region sets are supersets: each new garment type adds regions, never
removes or renames existing ones.  This means clearance.py can always find
the 6 base regions when processing a T-shirt.

New lower-body bands are marked # TODO: validate with real dress/trouser patterns
and must be confirmed against actual GarmentCode dress/trouser geometry before
promoting to a production release.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Region name constants — avoid scattered string literals in downstream code
# ---------------------------------------------------------------------------

# --- T-shirt / upper-body regions (Phase 1 baseline, do not rename) --------
REGION_CHEST_FRONT    = "chest_front"
REGION_CHEST_SIDE     = "chest_side"
REGION_SHOULDER_LEFT  = "shoulder_left"
REGION_SHOULDER_RIGHT = "shoulder_right"
REGION_UPPER_BACK     = "upper_back"
REGION_WAIST          = "waist"

# --- Dress additions ---------------------------------------------------------
REGION_HIP            = "hip"
REGION_THIGH_LEFT     = "thigh_left"
REGION_THIGH_RIGHT    = "thigh_right"

# --- Trousers additions (beyond dress) ---------------------------------------
REGION_SEAT           = "seat"
REGION_INSEAM         = "inseam"
REGION_KNEE_LEFT      = "knee_left"
REGION_KNEE_RIGHT     = "knee_right"

# ---------------------------------------------------------------------------
# Region sets per garment category
# ---------------------------------------------------------------------------

TSHIRT_REGIONS: list[str] = [
    REGION_CHEST_FRONT,
    REGION_CHEST_SIDE,
    REGION_SHOULDER_LEFT,
    REGION_SHOULDER_RIGHT,
    REGION_UPPER_BACK,
    REGION_WAIST,
]

DRESS_REGIONS: list[str] = TSHIRT_REGIONS + [
    REGION_HIP,
    REGION_THIGH_LEFT,
    REGION_THIGH_RIGHT,
]

TROUSERS_REGIONS: list[str] = TSHIRT_REGIONS + [
    REGION_HIP,
    REGION_SEAT,
    REGION_INSEAM,
    REGION_THIGH_LEFT,
    REGION_THIGH_RIGHT,
    REGION_KNEE_LEFT,
    REGION_KNEE_RIGHT,
]

# Legacy alias — kept so existing callers (clearance.py, tests) don't break.
# Points to TSHIRT_REGIONS; expand to a per-garment list as needed.
REQUIRED_REGIONS: list[str] = TSHIRT_REGIONS

# ---------------------------------------------------------------------------
# Region → garment-type mapping
# Use this to look up the correct region list for a given garment type string.
# ---------------------------------------------------------------------------
GARMENT_REGIONS: dict[str, list[str]] = {
    "tshirt": TSHIRT_REGIONS,
    "shirt":  TSHIRT_REGIONS,
    "hoody":  TSHIRT_REGIONS,
    "dress":  DRESS_REGIONS,
    "jumpsuit": TROUSERS_REGIONS,   # full body
    "trousers": TROUSERS_REGIONS,
}

# ---------------------------------------------------------------------------
# Proportional height bands (fraction of total body height, feet = 0, head = 1)
#
# Upper-body bands — Pheasant "Bodyspace" 2nd ed.; Winter "Biomechanics" 4th ed.
# Lower-body bands — approximated from the same anthropometric sources.
# ---------------------------------------------------------------------------

# Upper body (Phase 1 baseline — do not change without regression testing)
_SHOULDER_LO = 0.79   # shoulder / armscye base
_SHOULDER_HI = 0.87   # top of shoulder region / base of neck
_CHEST_LO    = 0.68   # bottom of chest / bust band
_CHEST_HI    = 0.84   # top of chest band (overlaps shoulder slightly)
_WAIST_LO    = 0.55   # bottom of natural-waist band
_WAIST_HI    = 0.68   # top of natural-waist band (= chest bottom)

# Lower body — TODO: validate with real dress/trouser patterns
_HIP_LO      = 0.44   # bottom of high-hip / gluteal band
_HIP_HI      = 0.55   # top of hip band (= waist bottom)
_SEAT_LO     = 0.44   # bottom of seat (rear projection; same Z-band as hip)
_SEAT_HI     = 0.55   # top of seat band
_THIGH_LO    = 0.30   # bottom of thigh band (mid-thigh)
_THIGH_HI    = 0.44   # top of thigh band (= hip bottom)
_INSEAM_LO   = 0.30   # inseam crotch region (medial face only)
_INSEAM_HI   = 0.46   # top of inseam band (covers crotch point)
_KNEE_LO     = 0.18   # bottom of knee band
_KNEE_HI     = 0.30   # top of knee band (= thigh bottom)


def classify_body_vertices(
    vertices: np.ndarray,
    normals: np.ndarray,
    garment_type: str = "tshirt",
) -> dict[str, list[int]]:
    """
    Assign each body vertex to exactly one Forma region.

    Works with both Y-up (MakeHuman) and Z-up (Anny) meshes by auto-detecting
    the up axis from vertex extents.  Height bands are proportional so they
    are independent of absolute mesh scale.

    Priority order (upper body): shoulders first (they overlap the chest height
    band), then front/back chest, then chest_side, then waist.
    Priority order (lower body): seat (rear) before hip (front/side), then
    inseam (medial), then thigh, then knee — all bilateral last.

    Parameters
    ----------
    vertices     : (N, 3) float array — body mesh vertices in meters
    normals      : (N, 3) float array — per-vertex outward normals (unit)
    garment_type : str — controls which region set is populated.
                   One of "tshirt", "shirt", "hoody", "dress", "trousers",
                   "jumpsuit".  Defaults to "tshirt" (Phase 1 behaviour).

    Returns
    -------
    dict mapping region_name → list of vertex indices.
    Only regions in GARMENT_REGIONS[garment_type] are present as keys.
    """
    if garment_type not in GARMENT_REGIONS:
        supported = ", ".join(sorted(GARMENT_REGIONS))
        raise ValueError(
            f"Unknown garment_type {garment_type!r}. Supported: {supported}"
        )

    active_regions = GARMENT_REGIONS[garment_type]
    n = len(vertices)

    # --- Auto-detect coordinate axes from vertex extents --------------------
    extents = vertices.max(axis=0) - vertices.min(axis=0)

    # The axis with the greatest span is "up" (height).
    up_axis = int(np.argmax(extents))

    # Among the two remaining axes, the wider one is lateral (left-right),
    # the narrower one is the depth / forward axis.
    non_up = [a for a in (0, 1, 2) if a != up_axis]
    lat_axis = non_up[int(np.argmax([extents[a] for a in non_up]))]
    fwd_axis = [a for a in non_up if a != lat_axis][0]

    h  = vertices[:, up_axis]    # height coordinate (feet = min, head = max)
    x  = vertices[:, lat_axis]   # lateral coordinate (+ = wearer's left)
    nf = normals[:, fwd_axis]    # forward normal component (+ = front-facing)
    nl = normals[:, lat_axis]    # lateral normal (+ = wearer's left side)

    H = h.max()   # total body height (used to compute proportional bands)

    assigned = np.zeros(n, dtype=bool)
    regions: dict[str, list[int]] = {r: [] for r in active_regions}

    def _assign(mask: np.ndarray, region: str) -> None:
        """Mark unassigned vertices matching mask as belonging to region."""
        if region not in regions:
            return  # region not active for this garment type
        idx = np.where(mask & ~assigned)[0]
        regions[region].extend(idx.tolist())
        assigned[idx] = True

    # -----------------------------------------------------------------------
    # Upper-body regions (Phase 1 baseline — priority order unchanged)
    # -----------------------------------------------------------------------

    # Shoulder zones (proportional height band), classified by lateral position
    _assign(
        (h >= H * _SHOULDER_LO) & (h <= H * _SHOULDER_HI) & (x < -0.05),
        REGION_SHOULDER_RIGHT,
    )
    _assign(
        (h >= H * _SHOULDER_LO) & (h <= H * _SHOULDER_HI) & (x >  0.05),
        REGION_SHOULDER_LEFT,
    )

    # Chest / upper-back zone, classified by forward normal direction
    chest_band = (h >= H * _CHEST_LO) & (h <= H * _CHEST_HI)
    _assign(chest_band & (nf >  0.25), REGION_CHEST_FRONT)
    _assign(chest_band & (nf < -0.25), REGION_UPPER_BACK)
    _assign(chest_band,                REGION_CHEST_SIDE)   # remainder in band

    # Waist zone
    _assign((h >= H * _WAIST_LO) & (h <= H * _WAIST_HI), REGION_WAIST)

    # -----------------------------------------------------------------------
    # Lower-body regions — dress, trousers, jumpsuit
    # TODO: validate Z-fraction bands with real dress/trouser patterns
    # -----------------------------------------------------------------------

    # Hip — front-facing and lateral vertices in the hip height band
    # (rear vertices in the same band are claimed by seat below)
    hip_band = (h >= H * _HIP_LO) & (h < H * _HIP_HI)
    _assign(hip_band & (nf >= 0.0), REGION_HIP)  # front half of hip band

    # Seat — posterior (rear-facing normal) vertices in the hip/seat height band
    # TODO: validate with real dress/trouser patterns
    seat_band = (h >= H * _SEAT_LO) & (h < H * _SEAT_HI)
    _assign(seat_band & (nf < 0.0), REGION_SEAT)  # rear half = glutes
    # Remaining hip-band vertices that weren't front-facing
    _assign(hip_band, REGION_HIP)

    # Inseam — medial (inner-facing lateral normal) vertices in the inseam band
    # TODO: validate with real dress/trouser patterns
    inseam_band = (h >= H * _INSEAM_LO) & (h < H * _INSEAM_HI)
    _assign(inseam_band & (np.abs(nl) < 0.3), REGION_INSEAM)  # near-medial

    # Thigh — bilateral split by lateral position
    # TODO: validate with real dress/trouser patterns
    thigh_band = (h >= H * _THIGH_LO) & (h < H * _THIGH_HI)
    _assign(thigh_band & (x < -0.03), REGION_THIGH_RIGHT)
    _assign(thigh_band & (x >  0.03), REGION_THIGH_LEFT)
    # Midline thigh (near x=0) goes to whichever side has its lateral normal
    _assign(thigh_band & (nl < 0.0), REGION_THIGH_RIGHT)
    _assign(thigh_band,               REGION_THIGH_LEFT)

    # Knee — bilateral split by lateral position
    # TODO: validate with real dress/trouser patterns
    knee_band = (h >= H * _KNEE_LO) & (h < H * _KNEE_HI)
    _assign(knee_band & (x < -0.03), REGION_KNEE_RIGHT)
    _assign(knee_band & (x >  0.03), REGION_KNEE_LEFT)
    _assign(knee_band & (nl < 0.0),  REGION_KNEE_RIGHT)
    _assign(knee_band,                REGION_KNEE_LEFT)

    return regions


def assign_garment_to_body_regions(
    garment_vertices: np.ndarray,
    body_vertices: np.ndarray,
    body_regions: dict[str, list[int]],
    max_distance_m: float = 0.10,
) -> dict[str, list[int]]:
    """
    For each garment vertex, find its nearest body vertex and inherit
    that body vertex's region assignment.

    Garment vertices further than max_distance_m from the body surface are
    excluded (typically sleeve panels placed far from the torso).  The default
    threshold of 10 cm accommodates very large garment sizes while excluding
    sleeve panels that produce large spurious clearance values.

    The output dict contains exactly the same region keys as body_regions,
    so callers can pass TSHIRT_REGIONS, DRESS_REGIONS, or TROUSERS_REGIONS
    bodies without any code change here.

    Parameters
    ----------
    garment_vertices : (M, 3) float array
    body_vertices    : (N, 3) float array
    body_regions     : output of classify_body_vertices()
    max_distance_m   : garment vertices farther than this are ignored

    Returns
    -------
    dict mapping region_name → list of garment vertex indices.
    Keys mirror body_regions (i.e. the active region set for the garment type).
    """
    tree = KDTree(body_vertices)
    dists, nearest_body_idx = tree.query(garment_vertices)

    # Build reverse map: body_vertex_idx → region name
    body_vertex_to_region: dict[int, str] = {}
    for region, vids in body_regions.items():
        for vid in vids:
            body_vertex_to_region[vid] = region

    # Mirror the exact keys present in body_regions (supports any garment type)
    garment_regions: dict[str, list[int]] = {r: [] for r in body_regions}
    for gv_idx, (bv_idx, dist) in enumerate(zip(nearest_body_idx, dists)):
        if dist > max_distance_m:
            continue   # sleeve or other far-out vertex — skip
        region = body_vertex_to_region.get(int(bv_idx))
        if region is not None:
            garment_regions[region].append(gv_idx)

    return garment_regions

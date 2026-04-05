# Forma kernel: contact_pressure.py
# Derived from:
#   - Macklin et al., "XPBD: Position-Based Simulation of Compliant
#     Constrained Dynamics" (MIG 2016) — constraint force recovery from
#     positional correction, F ≈ k * |Δx|
#   - Macklin & Müller, "Position Based Fluids" (SIGGRAPH 2013) — per-vertex
#     area lumping (1/3 of surrounding triangle areas)
#   - Forma Phase 1 XPBD solver (src/geometer/xpbd_simulate.py) — signed
#     clearance convention (negative = body larger than garment, i.e. contact)
#   - Baraff & Witkin, "Large Steps in Cloth Simulation" (SIGGRAPH 1998) —
#     contact force formulation for cloth-body collision
# No code from NvidiaWarp-GarmentCode was referenced.

"""
src/geometer/warp/contact_pressure.py

Per-region contact pressure extraction from Warp GPU simulation output.

After warp_simulate.py converges, garment particles have final positions.
Vertices with negative signed clearance (body surface closer than the vertex)
are in contact.  We recover an approximate contact force from the XPBD
compliance relationship and divide by per-vertex contact area to get pressure.

Physical model (XPBD approximation)
------------------------------------
In XPBD, a penetration constraint of depth d is resolved by a positional
correction Δx ≈ d.  The corresponding Lagrange multiplier (constraint force
magnitude) is:

    F ≈ |Δx| * ke            (ke = spring stiffness, N/m)

where ke for the contact spring is model.soft_contact_ke (1e4 N/m in Forma).
The depth d is the signed clearance converted to metres (d = -delta_m for
contact vertices, i.e. how far the vertex is inside the body surface).

For pressure we need area:

    A_vertex = (1/3) * sum(area of adjacent triangles)   [m²]

Then:

    P_vertex = F / A_vertex   [N/m²]

We convert to N/mm² by dividing by 1e6.

Per-region pressure is the **median** of contact-vertex pressures within the
region (robust to mesh-density variations at region boundaries).
Regions with no contact vertices return 0.0.

Units
-----
    Input positions : metres
    delta_m         : metres (negative = contact)
    F               : Newtons
    A               : m²
    P               : N/m² → converted to N/mm² (÷1e6)

Fallback
--------
If `warp` is not importable this module still imports cleanly and
compute_contact_pressure() returns all-zero dicts.  The caller is
run_simulation_warp() which already guards Warp availability.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Warp availability guard — this module must import cleanly even without warp.
# ---------------------------------------------------------------------------

try:
    import warp as _warp  # noqa: F401
    _HAS_WARP = True
except ImportError:
    _HAS_WARP = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_per_vertex_areas(
    garment_vertices: np.ndarray,
    garment_faces: np.ndarray,
) -> np.ndarray:
    """
    Compute per-vertex contact area by lumping 1/3 of each adjacent triangle.

    This is the standard FEM vertex-area lumping scheme:
        A_v = sum_{t: v ∈ t} (area_t / 3)

    Parameters
    ----------
    garment_vertices : (N, 3) float — garment vertex positions (metres)
    garment_faces    : (M, 3) int   — triangle face indices

    Returns
    -------
    (N,) float — per-vertex area in m².  Never zero (floor at 1e-8 m²).
    """
    N = len(garment_vertices)
    vertex_areas = np.zeros(N, dtype=np.float64)

    if len(garment_faces) == 0:
        # Degenerate — uniform fallback (1 cm² per vertex)
        return np.full(N, 1e-4, dtype=np.float64)

    v0 = garment_vertices[garment_faces[:, 0]]
    v1 = garment_vertices[garment_faces[:, 1]]
    v2 = garment_vertices[garment_faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_areas = np.linalg.norm(cross, axis=1) * 0.5   # (M,) in m²

    # Distribute 1/3 of each triangle's area to each of its three vertices
    third_areas = tri_areas / 3.0
    np.add.at(vertex_areas, garment_faces[:, 0], third_areas)
    np.add.at(vertex_areas, garment_faces[:, 1], third_areas)
    np.add.at(vertex_areas, garment_faces[:, 2], third_areas)

    # Floor: prevent division by zero for isolated/degenerate vertices
    return np.maximum(vertex_areas, 1e-8)


def _compute_signed_distances(
    garment_vertices: np.ndarray,
    body_vertices: np.ndarray,
    body_normals: np.ndarray,
) -> np.ndarray:
    """
    Compute per-garment-vertex signed distance to the body surface.

    Positive = garment vertex is outside body (ease/loose).
    Negative = garment vertex is inside body surface (contact/penetration).

    Uses the same KDTree nearest-neighbor projection as clearance.py.

    Parameters
    ----------
    garment_vertices : (N, 3) float — draped garment vertex positions (metres)
    body_vertices    : (B, 3) float — body mesh vertices (metres)
    body_normals     : (B, 3) float — per-vertex outward unit normals

    Returns
    -------
    (N,) float — signed distance in metres for each garment vertex.
    """
    tree = KDTree(body_vertices)
    _, nearest_idx = tree.query(garment_vertices)

    nearest_pts = body_vertices[nearest_idx]      # (N, 3)
    nearest_nrm = body_normals[nearest_idx]       # (N, 3)
    displacement = garment_vertices - nearest_pts  # (N, 3)

    # Project displacement onto outward normal — negative means inside body
    signed_dists = np.einsum("ij,ij->i", displacement, nearest_nrm)  # (N,)
    return signed_dists


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_contact_pressure(
    sim_result: dict,
    body_vertices: np.ndarray,
    garment_vertices: np.ndarray,
    region_labels: dict[str, list[int]],
    fabric_params: dict,
    dt: float = 1 / 60,
    garment_faces: np.ndarray | None = None,
    body_normals: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute per-region contact pressure in N/mm² from simulation output.

    Uses the XPBD force-recovery approximation:

        F_contact = penetration_depth_m * soft_contact_ke
        P_vertex  = F_contact / vertex_area_m2
        P_region  = median(P_vertex for contact vertices in region)

    where soft_contact_ke = 1e4 N/m (Forma Warp model default, matches the
    value set in warp_simulate._build_warp_model).

    The dt parameter is kept in the signature for API consistency with the
    XPBD impulse formulation (F = Δx / dt²) but the spring-stiffness path
    is preferred because it is independent of the (variable) convergence step.

    Parameters
    ----------
    sim_result       : dict returned by run_simulation_warp().  Not directly
                       used for positions (pass garment_vertices explicitly)
                       but may carry 'warp_stretch_ke' if available.
    body_vertices    : (B, 3) float — body mesh vertices in metres
    garment_vertices : (N, 3) float — final draped garment positions in metres
    region_labels    : dict mapping region_name → list of garment vertex indices
                       (output of assign_garment_to_body_regions)
    fabric_params    : dict from fabric_library.json — must contain
                       'density_kg_m2'.  Used to derive stiffness if not
                       available from sim_result.
    dt               : simulation timestep in seconds (default 1/60 s)
    garment_faces    : (M, 3) int — garment triangle faces for per-vertex area
                       computation.  When None, a uniform area approximation
                       is used (total_area / N, estimated from fabric density).
    body_normals     : (B, 3) float — body outward unit normals.  When None,
                       approximate normals are estimated from the body mesh
                       (less accurate but functional for coarse pressure).

    Returns
    -------
    dict mapping region_name → pressure_N_mm2 (float).
    Returns 0.0 for regions with no contact (positive clearance).
    Returns all-zero dict gracefully if warp is not installed.

    Notes
    -----
    - This is an approximation, not a FEM solver.  Accuracy is sufficient for
      fit severity classification (red/yellow/green) but not for biomechanical
      analysis.
    - Pressure sign convention: always non-negative.  Use clearance_map for
      directional (tight/loose) information.
    """
    # ---- Graceful fallback when Warp is unavailable -------------------------
    if not _HAS_WARP:
        return {region: 0.0 for region in region_labels}

    # ---- Guard: empty inputs ------------------------------------------------
    if len(garment_vertices) == 0 or len(body_vertices) == 0:
        return {region: 0.0 for region in region_labels}

    # ---- 1. Body normals ----------------------------------------------------
    # Prefer caller-supplied normals.  When absent, estimate from nearest-
    # neighbor averaging (cheap but adequate for pressure approximation).
    if body_normals is None:
        body_normals = _estimate_body_normals(body_vertices)

    # ---- 2. Per-vertex signed distance to body surface ----------------------
    signed_dists = _compute_signed_distances(
        garment_vertices, body_vertices, body_normals
    )   # (N,) in metres; negative = contact

    # ---- 3. Contact stiffness -----------------------------------------------
    # Use the soft_contact_ke from the Warp model (1e4 N/m).
    # If a custom value was stored in sim_result, prefer that.
    soft_contact_ke: float = float(sim_result.get("soft_contact_ke", 1e4))

    # ---- 4. Per-vertex contact area -----------------------------------------
    if garment_faces is not None and len(garment_faces) > 0:
        vertex_areas = _compute_per_vertex_areas(garment_vertices, garment_faces)
    else:
        # Fallback: estimate total garment area from fabric density assumption.
        # Forma garments are ~0.3–0.8 m² for a T-shirt.  Distribute uniformly.
        N = len(garment_vertices)
        # Try to get total area from fabric params or sim_result
        total_area_m2 = float(
            sim_result.get("total_area_m2")
            or fabric_params.get("total_area_m2")
            or 0.5   # 0.5 m² default (reasonable T-shirt area)
        )
        mean_area = total_area_m2 / max(N, 1)
        vertex_areas = np.full(N, mean_area, dtype=np.float64)

    # ---- 5. Per-vertex contact force and pressure ---------------------------
    # Contact vertices: signed_dist < 0 (garment is inside body surface).
    # Penetration depth (metres, positive): d = -signed_dist
    penetration_m = np.where(signed_dists < 0.0, -signed_dists, 0.0)  # (N,)

    # XPBD spring contact force: F = k * d  (N)
    # This mirrors the soft_contact_ke spring in warp_simulate._build_warp_model.
    contact_force = penetration_m * soft_contact_ke   # (N,) in Newtons

    # Pressure: P = F / A  (N/m²) → convert to N/mm² (÷ 1e6)
    pressure_N_m2 = contact_force / vertex_areas       # (N,) in N/m²
    pressure_N_mm2 = pressure_N_m2 / 1e6              # (N,) in N/mm²

    # ---- 6. Aggregate per region --------------------------------------------
    pressure_map: dict[str, float] = {}
    for region, vertex_ids in region_labels.items():
        if not vertex_ids:
            pressure_map[region] = 0.0
            continue

        ids = np.asarray(vertex_ids, dtype=np.int64)
        # Clamp to valid range (guard against stale region_labels)
        ids = ids[ids < len(garment_vertices)]
        if len(ids) == 0:
            pressure_map[region] = 0.0
            continue

        region_pressures = pressure_N_mm2[ids]           # (K,)
        region_contact = region_pressures[region_pressures > 0.0]

        if len(region_contact) == 0:
            pressure_map[region] = 0.0
        else:
            # Median is robust to mesh-density variation at region boundaries.
            pressure_map[region] = round(float(np.median(region_contact)), 6)

    return pressure_map


# ---------------------------------------------------------------------------
# Internal: fallback normal estimation (only used when body_normals is None)
# ---------------------------------------------------------------------------

def _estimate_body_normals(body_vertices: np.ndarray) -> np.ndarray:
    """
    Estimate per-vertex outward normals for a body mesh when normals are
    not supplied.

    Approximation: assume the body is roughly centred on the XZ plane at the
    centroid.  Normal = (vertex - centroid_xz) normalised.  This is only
    used as a fallback for pressure computation and is not called in the
    normal Warp simulation path (warp_simulate passes real normals).

    Parameters
    ----------
    body_vertices : (N, 3) float — body mesh vertices (metres)

    Returns
    -------
    (N, 3) float — estimated outward unit normals
    """
    centroid = body_vertices.mean(axis=0)
    # Only use XZ displacement for lateral normal (body is Y-up)
    radial = body_vertices - centroid
    radial[:, 1] = 0.0   # zero out Y component — normals point outward radially

    norms = np.linalg.norm(radial, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return radial / norms

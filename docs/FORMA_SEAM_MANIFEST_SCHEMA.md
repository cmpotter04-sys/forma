# FORMA — Seam Manifest Schema Spec
**Version:** 1.0  
**Phase:** 1 / Week 1  
**Depends on:** FORMA_WEEK1_SPEC.md (AC-2), FORMA_GEOMETER_SPEC.md  
**Status:** READY TO BUILD

---

## Purpose

The `seam_manifest.json` is the contract between the Tailor (schema converter)
and the Geometer (XPBD simulation). It describes which edges on which panels
must be stitched together, and in what order. Without this spec, Claude Code
cannot build the Tailor converter or the Geometer's seam constraint setup.

---

## Schema: seam_manifest.json

```json
{
  "garment_id": "tshirt_gc_v1_size_M",
  "panel_count": 4,
  "panels": [
    {
      "panel_id": "front_body",
      "vertex_count": 128,
      "edge_count": 4,
      "edges": [
        {
          "edge_id": "front_body_e0",
          "vertices": [0, 1, 2, 3, 4, 5],
          "arc_length_mm": 420.0,
          "label": "left_side_seam"
        },
        {
          "edge_id": "front_body_e1",
          "vertices": [5, 6, 7, 8, 9, 10],
          "arc_length_mm": 200.0,
          "label": "left_shoulder"
        }
      ]
    },
    {
      "panel_id": "back_body",
      "vertex_count": 130,
      "edge_count": 4,
      "edges": [
        {
          "edge_id": "back_body_e0",
          "vertices": [0, 1, 2, 3, 4, 5],
          "arc_length_mm": 421.2,
          "label": "left_side_seam"
        }
      ]
    }
  ],
  "seam_pairs": [
    {
      "seam_id": "seam_001",
      "edge_a": "front_body_e0",
      "edge_b": "back_body_e0",
      "arc_length_diff_mm": 1.2,
      "valid": true,
      "stitch_type": "standard"
    },
    {
      "seam_id": "seam_002",
      "edge_a": "front_body_e1",
      "edge_b": "left_sleeve_e0",
      "arc_length_diff_mm": 0.8,
      "valid": true,
      "stitch_type": "standard"
    }
  ],
  "validation": {
    "all_seams_valid": true,
    "max_arc_length_diff_mm": 1.2,
    "tolerance_mm": 2.0,
    "unmatched_edges": [],
    "total_seam_pairs": 6
  }
}
```

---

## Field Definitions

### Panel Object

| Field | Type | Description |
|-------|------|-------------|
| panel_id | string | Unique identifier from GarmentCode pattern (e.g. "front_body", "left_sleeve") |
| vertex_count | integer | Number of vertices in the panel outline after discretization |
| edge_count | integer | Number of distinct edges (boundary segments between corners) |
| edges | array | List of edge objects (see below) |

### Edge Object

| Field | Type | Description |
|-------|------|-------------|
| edge_id | string | Unique edge identifier: `{panel_id}_e{index}` |
| vertices | array[int] | Ordered list of vertex indices along this edge (panel-local indices) |
| arc_length_mm | float | Total length of this edge in millimeters |
| label | string | Semantic label (e.g. "left_side_seam", "neckline", "hem"). Used for human readability only — matching is by edge_id. |

### Seam Pair Object

| Field | Type | Description |
|-------|------|-------------|
| seam_id | string | `seam_{3-digit index}` |
| edge_a | string | edge_id of the first edge in the pair |
| edge_b | string | edge_id of the second edge in the pair |
| arc_length_diff_mm | float | Absolute difference in arc length between edge_a and edge_b. Must be < tolerance_mm for the seam to be valid. |
| valid | boolean | `true` if arc_length_diff_mm < tolerance_mm |
| stitch_type | string | enum: `"standard"` (default), `"gather"` (for sleeves/necklines where slight mismatch is intentional) |

### Validation Object

| Field | Type | Description |
|-------|------|-------------|
| all_seams_valid | boolean | `true` if every seam pair has valid=true |
| max_arc_length_diff_mm | float | Largest arc length difference across all seam pairs |
| tolerance_mm | float | Maximum allowed arc length difference (default: 2.0mm) |
| unmatched_edges | array[string] | List of edge_ids that have no seam partner (e.g. hem edges, neckline openings) |
| total_seam_pairs | integer | Number of seam pairs in the manifest |

---

## Validation Rules

The Tailor MUST enforce all of the following before passing to the Geometer:

1. **Arc length tolerance:** `arc_length_diff_mm < 2.0` for all standard seams.
   Gather seams may exceed 2.0mm but must not exceed 10.0mm.

2. **No orphan seam references:** Every `edge_a` and `edge_b` in seam_pairs
   must reference an edge_id that exists in the panels array.

3. **No duplicate seam assignments:** Each edge_id may appear in at most one
   seam pair. An edge cannot be stitched to two different edges.

4. **Expected unmatched edges:** Hem edges and neckline openings should appear
   in `unmatched_edges`. For a T-shirt, expect 3 unmatched edges: bottom hem,
   left sleeve hem, right sleeve hem (neckline is typically a closed seam).

5. **Vertex count compatibility:** For each seam pair, both edges must have
   the same number of vertices (resampled if needed). The Geometer creates
   one distance constraint per vertex pair.

---

## Tailor Conversion Pipeline

```
GarmentCode pattern.json
        ↓
  Parse panel outlines (2D polygons with curves)
        ↓
  Discretize curves into polylines (target segment length: 5mm)
        ↓
  Extract edges (boundary segments between corner vertices)
        ↓
  Read GarmentCode stitch annotations → identify seam pairs
        ↓
  Compute arc lengths per edge
        ↓
  Validate arc length differences within ±2mm
        ↓
  Resample edges so paired edges have equal vertex count
        ↓
  Write seam_manifest.json
```

---

## SeamValidationError

If any validation rule fails, the Tailor MUST raise `SeamValidationError`
and never silently proceed. This is a non-negotiable rule from CLAUDE.md.

```python
class SeamValidationError(Exception):
    """Raised when seam manifest validation fails."""
    def __init__(self, message, failing_seams=None):
        super().__init__(message)
        self.failing_seams = failing_seams or []
```

---

*FORMA Seam Manifest Schema v1.0*  
*Companion to FORMA_WEEK1_SPEC.md AC-2 and FORMA_GEOMETER_SPEC.md*

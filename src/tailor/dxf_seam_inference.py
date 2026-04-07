"""
src/tailor/dxf_seam_inference.py

Deterministic seam inference for DXF-derived T-shirt patterns.

This module intentionally infers only high-confidence seams. Ambiguous edges are
left unmatched rather than guessed. The result is a valid seam-manifest-shaped
object that can serve as the contract between DXF intake and downstream garment
assembly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tailor.panel_labeler import label_tshirt_panels


@dataclass(frozen=True)
class EdgeRole:
    panel_id: str
    edge_id: str
    role: str


class DXFSeamInferenceError(Exception):
    """Raised when a seam manifest cannot be inferred from labeled DXF panels."""


def _edge_points(panel: dict, edge: dict) -> np.ndarray:
    verts = panel["vertices"]
    start, end = edge["endpoints"]
    pts = edge["polyline"] + [verts[end]]
    return np.asarray(pts, dtype=float)


def _edge_features(panel: dict) -> list[dict]:
    feats = []
    polygon = np.asarray(panel["vertices"], dtype=float)
    centroid_x = float(np.mean(polygon[:, 0]))
    for edge in panel["edges"]:
        pts = _edge_points(panel, edge)
        midpoint = pts.mean(axis=0)
        delta = pts[-1] - pts[0]
        angle = float(abs(np.arctan2(delta[1], delta[0])))
        feats.append(
            {
                "edge_id": f"{panel['panel_id']}_e{edge['edge_index']}",
                "edge_index": edge["edge_index"],
                "mid_x": float(midpoint[0]),
                "mid_y": float(midpoint[1]),
                "length_mm": float(edge["arc_length_mm"]),
                "horizontal_score": abs(abs(angle) - 0.0),
                "centroid_x": centroid_x,
            }
        )
    return feats


def infer_tshirt_edge_roles(pattern: dict) -> list[EdgeRole]:
    labeled = pattern if all(
        "label" in panel for panel in pattern["panels"].values()
    ) else label_tshirt_panels(pattern)

    roles: list[EdgeRole] = []
    for panel_id, panel in labeled["panels"].items():
        panel = dict(panel)
        panel["panel_id"] = panel_id
        feats = _edge_features(panel)
        label = panel["label"]

        if label in {"front_body", "back_body"}:
            by_y = sorted(feats, key=lambda item: item["mid_y"])
            hem = by_y[0]
            roles.append(EdgeRole(panel_id, hem["edge_id"], "hem"))

            remaining = [item for item in feats if item["edge_id"] != hem["edge_id"]]

            left_side = min(remaining, key=lambda item: item["mid_x"])
            right_side = max(remaining, key=lambda item: item["mid_x"])
            roles.append(EdgeRole(panel_id, left_side["edge_id"], "left_side"))
            roles.append(EdgeRole(panel_id, right_side["edge_id"], "right_side"))

            upper = [
                item for item in remaining
                if item["edge_id"] not in {left_side["edge_id"], right_side["edge_id"]}
            ]
            if upper:
                neckline = min(
                    upper,
                    key=lambda item: (
                        abs(item["mid_x"] - item["centroid_x"]),
                        -item["mid_y"],
                        item["length_mm"],
                    ),
                )
                roles.append(EdgeRole(panel_id, neckline["edge_id"], "neckline"))

                shoulder_candidates = [
                    item for item in upper if item["edge_id"] != neckline["edge_id"]
                ]
                left_upper = [
                    item for item in shoulder_candidates
                    if item["mid_x"] <= item["centroid_x"]
                ]
                right_upper = [
                    item for item in shoulder_candidates
                    if item["mid_x"] > item["centroid_x"]
                ]

                if left_upper:
                    left_shoulder = max(left_upper, key=lambda item: item["mid_y"])
                    roles.append(EdgeRole(panel_id, left_shoulder["edge_id"], "left_shoulder"))
                    for candidate in left_upper:
                        if candidate["edge_id"] != left_shoulder["edge_id"]:
                            roles.append(
                                EdgeRole(panel_id, candidate["edge_id"], "left_armhole")
                            )

                if right_upper:
                    right_shoulder = max(right_upper, key=lambda item: item["mid_y"])
                    roles.append(
                        EdgeRole(panel_id, right_shoulder["edge_id"], "right_shoulder")
                    )
                    for candidate in right_upper:
                        if candidate["edge_id"] != right_shoulder["edge_id"]:
                            roles.append(
                                EdgeRole(panel_id, candidate["edge_id"], "right_armhole")
                            )

        elif label in {"left_sleeve", "right_sleeve"}:
            cuff = min(feats, key=lambda item: item["mid_y"])
            roles.append(EdgeRole(panel_id, cuff["edge_id"], "cuff"))

            remaining = [item for item in feats if item["edge_id"] != cuff["edge_id"]]
            if remaining:
                cap = max(remaining, key=lambda item: item["mid_y"])
                roles.append(EdgeRole(panel_id, cap["edge_id"], "sleeve_cap"))

                seam_candidates = [
                    item for item in remaining if item["edge_id"] != cap["edge_id"]
                ]
                if seam_candidates:
                    front_like = max(seam_candidates, key=lambda item: item["mid_x"])
                    back_like = min(seam_candidates, key=lambda item: item["mid_x"])
                    roles.append(
                        EdgeRole(panel_id, front_like["edge_id"], "front_underarm")
                    )
                    if back_like["edge_id"] != front_like["edge_id"]:
                        roles.append(
                            EdgeRole(panel_id, back_like["edge_id"], "back_underarm")
                        )

    return roles


def build_tshirt_dxf_seam_manifest(pattern: dict) -> dict:
    """
    Infer a seam-manifest-like structure from a labeled DXF pattern.

    Current scope:
      - side seams: front_body <-> back_body
      - leaves shoulders, sleeves, neckline, hem unmatched until a richer edge
        segmentation / sleeve-cap inference pass is implemented
    """
    labeled = pattern if all(
        "label" in panel for panel in pattern["panels"].values()
    ) else label_tshirt_panels(pattern)

    panels = labeled["panels"]
    by_label = {panel["label"]: panel_id for panel_id, panel in panels.items()}
    required = {"front_body", "back_body"}
    if not required.issubset(by_label):
        raise DXFSeamInferenceError(
            f"Need labeled front/back torso panels, got {sorted(by_label)}"
        )

    edge_roles = infer_tshirt_edge_roles(labeled)
    role_map = {(item.panel_id, item.role): item.edge_id for item in edge_roles}

    manifest_panels = []
    all_edge_ids = set()
    for panel_id, panel in panels.items():
        edges = []
        for edge in panel["edges"]:
            edge_id = f"{panel_id}_e{edge['edge_index']}"
            all_edge_ids.add(edge_id)
            role = next(
                (item.role for item in edge_roles if item.edge_id == edge_id),
                "unknown",
            )
            full_poly_count = len(edge["polyline"]) + 1
            edges.append(
                {
                    "edge_id": edge_id,
                    "vertices": list(range(full_poly_count)),
                    "arc_length_mm": round(float(edge["arc_length_mm"]), 3),
                    "label": role,
                }
            )
        manifest_panels.append(
            {
                "panel_id": panel_id,
                "vertex_count": len(panel["vertices"]),
                "edge_count": len(panel["edges"]),
                "edges": edges,
            }
        )

    seam_pairs = []
    seam_specs = [
        ("front_body", "left_side", "back_body", "left_side"),
        ("front_body", "right_side", "back_body", "right_side"),
        ("front_body", "left_shoulder", "back_body", "left_shoulder"),
        ("front_body", "right_shoulder", "back_body", "right_shoulder"),
    ]
    for idx, (a_panel_label, a_role, b_panel_label, b_role) in enumerate(seam_specs, start=1):
        panel_a = by_label[a_panel_label]
        panel_b = by_label[b_panel_label]
        edge_a = role_map.get((panel_a, a_role))
        edge_b = role_map.get((panel_b, b_role))
        if edge_a is None or edge_b is None:
            continue

        arc_a = next(
            edge["arc_length_mm"]
            for panel in manifest_panels
            if panel["panel_id"] == panel_a
            for edge in panel["edges"]
            if edge["edge_id"] == edge_a
        )
        arc_b = next(
            edge["arc_length_mm"]
            for panel in manifest_panels
            if panel["panel_id"] == panel_b
            for edge in panel["edges"]
            if edge["edge_id"] == edge_b
        )
        arc_diff = round(abs(arc_a - arc_b), 3)
        if arc_diff >= 2.0:
            continue

        seam_pairs.append(
            {
                "seam_id": f"seam_{idx:03d}",
                "edge_a": edge_a,
                "edge_b": edge_b,
                "arc_length_diff_mm": arc_diff,
                "valid": True,
                "stitch_type": "standard",
            }
        )

    used_edges = {sp["edge_a"] for sp in seam_pairs} | {sp["edge_b"] for sp in seam_pairs}
    unmatched = sorted(all_edge_ids - used_edges)
    max_diff = max((sp["arc_length_diff_mm"] for sp in seam_pairs), default=0.0)

    return {
        "garment_id": labeled["garment_id"],
        "panel_count": len(manifest_panels),
        "panels": manifest_panels,
        "seam_pairs": seam_pairs,
        "validation": {
            "all_seams_valid": all(sp["valid"] for sp in seam_pairs),
            "max_arc_length_diff_mm": round(max_diff, 3),
            "tolerance_mm": 2.0,
            "unmatched_edges": unmatched,
            "total_seam_pairs": len(seam_pairs),
        },
    }

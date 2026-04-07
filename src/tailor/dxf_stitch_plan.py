"""
src/tailor/dxf_stitch_plan.py

High-level stitch planning for DXF-derived T-shirt panels.

This module does not emit seam_manifest pairs directly for sleeve-cap
attachments because the current seam schema expects one edge to appear in at
most one seam pair. A raw sleeve-cap edge typically needs to be split into
front/back segments first. The stitch plan records those intended attachments
explicitly so the next conversion step can perform valid edge splitting.
"""

from __future__ import annotations

from tailor.dxf_seam_inference import infer_tshirt_edge_roles
from tailor.panel_labeler import label_tshirt_panels


class DXFStitchPlanError(Exception):
    """Raised when a sleeve/armhole stitch plan cannot be inferred."""


def build_tshirt_stitch_plan(pattern: dict) -> dict:
    labeled = pattern if all(
        "label" in panel for panel in pattern["panels"].values()
    ) else label_tshirt_panels(pattern)

    by_label = {panel["label"]: panel_id for panel_id, panel in labeled["panels"].items()}
    required = {"front_body", "back_body", "left_sleeve", "right_sleeve"}
    if not required.issubset(by_label):
        raise DXFStitchPlanError(
            f"Need labeled front/back/sleeves, got {sorted(by_label)}"
        )

    roles = infer_tshirt_edge_roles(labeled)
    role_map = {(item.panel_id, item.role): item.edge_id for item in roles}

    front = by_label["front_body"]
    back = by_label["back_body"]
    left_sleeve = by_label["left_sleeve"]
    right_sleeve = by_label["right_sleeve"]

    plan = {
        "garment_id": labeled["garment_id"],
        "panel_roles": by_label,
        "torso_pairs": [
            {
                "kind": "side_seam",
                "edge_a": role_map.get((front, "left_side")),
                "edge_b": role_map.get((back, "left_side")),
            },
            {
                "kind": "side_seam",
                "edge_a": role_map.get((front, "right_side")),
                "edge_b": role_map.get((back, "right_side")),
            },
            {
                "kind": "shoulder_seam",
                "edge_a": role_map.get((front, "left_shoulder")),
                "edge_b": role_map.get((back, "left_shoulder")),
            },
            {
                "kind": "shoulder_seam",
                "edge_a": role_map.get((front, "right_shoulder")),
                "edge_b": role_map.get((back, "right_shoulder")),
            },
        ],
        "sleeve_caps": [
            {
                "side": "left",
                "sleeve_panel": left_sleeve,
                "sleeve_cap_edge": role_map.get((left_sleeve, "sleeve_cap")),
                "attach_to": [
                    role_map.get((front, "left_armhole")),
                    role_map.get((back, "left_armhole")),
                ],
            },
            {
                "side": "right",
                "sleeve_panel": right_sleeve,
                "sleeve_cap_edge": role_map.get((right_sleeve, "sleeve_cap")),
                "attach_to": [
                    role_map.get((front, "right_armhole")),
                    role_map.get((back, "right_armhole")),
                ],
            },
        ],
        "underarm_pairs": [
            {
                "side": "left",
                "edge_a": role_map.get((left_sleeve, "front_underarm")),
                "edge_b": role_map.get((left_sleeve, "back_underarm")),
            },
            {
                "side": "right",
                "edge_a": role_map.get((right_sleeve, "front_underarm")),
                "edge_b": role_map.get((right_sleeve, "back_underarm")),
            },
        ],
    }

    return plan

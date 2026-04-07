"""
src/tailor/panel_labeler.py

Deterministic rule-based labeling for raw T-shirt panels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PanelLabel:
    raw_panel_id: str
    role: str
    confidence: float


class PanelLabelError(Exception):
    """Raised when coarse panel semantics cannot be assigned."""


_EXPLICIT_TSHIRT_IDS = {
    "front_body": "front_body",
    "back_body": "back_body",
    "left_sleeve": "left_sleeve",
    "right_sleeve": "right_sleeve",
}


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return abs(0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _panel_features(panel_id: str, outline: list[list[float]]) -> dict:
    pts = np.asarray(outline, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise PanelLabelError(f"Invalid outline for panel {panel_id!r}")

    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    width = float(xmax - xmin)
    height = float(ymax - ymin)
    area = _polygon_area(pts)
    centroid = pts.mean(axis=0)

    center_x = float((xmin + xmax) * 0.5)
    center_mask = np.abs(pts[:, 0] - center_x) <= max(width * 0.15, 1.0)
    top_y = float(ymax)
    top_center_y = float(np.max(pts[center_mask, 1])) if np.any(center_mask) else top_y
    neckline_depth = max(0.0, top_y - top_center_y)

    return {
        "panel_id": panel_id,
        "outline": pts,
        "area": area,
        "width": width,
        "height": height,
        "centroid_x": float(centroid[0]),
        "neckline_depth": neckline_depth,
    }


def _explicit_panel_labels(panel_ids: list[str]) -> list[PanelLabel] | None:
    normalized = {panel_id: panel_id.lower() for panel_id in panel_ids}
    roles = {
        panel_id: _EXPLICIT_TSHIRT_IDS[norm_id]
        for panel_id, norm_id in normalized.items()
        if norm_id in _EXPLICIT_TSHIRT_IDS
    }
    if not roles:
        return None
    if len(roles) != 4:
        raise PanelLabelError(
            "Unsupported T-shirt topology: expected exactly 4 named panels "
            "(front_body, back_body, left_sleeve, right_sleeve), "
            f"got {sorted(roles.values())}"
        )
    # Extra panel_ids beyond the 4 named ones are silently ignored here;
    # they will be dropped at the load_dxf boundary.
    return [PanelLabel(panel_id, role, 1.0) for panel_id, role in roles.items()]


def infer_tshirt_panel_labels(panel_map: dict[str, list[list[float]]]) -> list[PanelLabel]:
    explicit = _explicit_panel_labels(list(panel_map))
    if explicit is not None:
        return explicit

    if len(panel_map) < 4:
        raise PanelLabelError(f"Need at least 4 panels, got {len(panel_map)}")

    features = [
        _panel_features(panel_id, outline)
        for panel_id, outline in panel_map.items()
    ]
    by_area = sorted(features, key=lambda item: item["area"], reverse=True)
    # When more than 4 panels are present (e.g. grainline arrows, pocket pieces,
    # annotation rectangles), silently take the 4 largest — those will always be
    # the two torso panels and two sleeves for a standard T-shirt DXF.
    by_area = by_area[:4]
    torso = by_area[:2]
    sleeves = by_area[2:4]

    if len(torso) < 2 or len(sleeves) < 2:
        raise PanelLabelError("Unable to separate torso and sleeve panels")

    front = max(torso, key=lambda item: item["neckline_depth"])
    back = torso[0] if torso[1]["panel_id"] == front["panel_id"] else torso[1]

    left = max(sleeves, key=lambda item: item["centroid_x"])
    right = sleeves[0] if sleeves[1]["panel_id"] == left["panel_id"] else sleeves[1]

    return [
        PanelLabel(front["panel_id"], "front_body", 0.95),
        PanelLabel(back["panel_id"], "back_body", 0.95),
        PanelLabel(left["panel_id"], "left_sleeve", 0.90),
        PanelLabel(right["panel_id"], "right_sleeve", 0.90),
    ]


def _translation_for_role(role: str) -> list[float]:
    if role == "front_body":
        return [0.0, 0.0, 12.0]
    if role == "back_body":
        return [0.0, 0.0, -12.0]
    if role == "left_sleeve":
        return [35.0, 0.0, 12.0]
    if role == "right_sleeve":
        return [-35.0, 0.0, 12.0]
    return [0.0, 0.0, 0.0]


def label_tshirt_panels(panel_input: dict) -> dict:
    """
    Public API for T-shirt panel labeling.

    Accepts either:
      - raw mapping: {panel_id: outline}
      - canonical pattern dict with "panels"
    """
    if "panels" in panel_input:
        pattern = dict(panel_input)
        if all("label" in panel for panel in pattern["panels"].values()):
            return pattern
        raw_map = {
            panel_id: panel["vertices"]
            for panel_id, panel in pattern["panels"].items()
        }
        labels = {item.raw_panel_id: item for item in infer_tshirt_panel_labels(raw_map)}

        labeled_panels = {}
        for panel_id, panel in pattern["panels"].items():
            info = labels[panel_id]
            next_panel = dict(panel)
            next_panel["label"] = info.role
            next_panel["label_confidence"] = info.confidence
            next_panel["translation"] = _translation_for_role(info.role)
            next_panel["rotation"] = [0.0, 0.0, 0.0]
            labeled_panels[panel_id] = next_panel

        pattern["panels"] = labeled_panels
        return pattern

    return {
        item.raw_panel_id: item.role
        for item in infer_tshirt_panel_labels(panel_input)
    }

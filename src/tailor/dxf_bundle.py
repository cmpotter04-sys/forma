"""
src/tailor/dxf_bundle.py

Bridge utilities that turn a DXF-derived canonical pattern into pipeline-ready
JSON artifacts readable by the existing pattern and seam loaders.
"""

from __future__ import annotations

import json
from pathlib import Path

from pattern_maker.load_dxf import load_dxf_pattern
from tailor.dxf_stitch_plan import build_tshirt_stitch_plan
from tailor.panel_labeler import label_tshirt_panels


_PANEL_ID_MAP = {
    "front_body": "front_ftorso",
    "back_body": "back_btorso",
    "left_sleeve": "left_sleeve",
    "right_sleeve": "right_sleeve",
}


class DXFBundleError(Exception):
    """Raised when a DXF pattern cannot produce a complete pipeline bundle."""


def _rename_edge_id(edge_id: str, panel_name_map: dict[str, str]) -> str:
    panel_id, edge_suffix = edge_id.rsplit("_e", 1)
    renamed_panel = panel_name_map.get(panel_id, panel_id)
    return f"{renamed_panel}_e{edge_suffix}"


def _remap_edge_ids(
    edge_id: str,
    panel_name_map: dict[str, str],
    split_map: dict[str, dict[int, list[int]]],
) -> list[str]:
    panel_id, edge_suffix = edge_id.rsplit("_e", 1)
    edge_index = int(edge_suffix)
    remapped_indices = split_map.get(panel_id, {}).get(edge_index, [edge_index])
    renamed_panel = panel_name_map.get(panel_id, panel_id)
    return [f"{renamed_panel}_e{idx}" for idx in remapped_indices]


def _split_edge_vertices(vertices: list[list[float]], edge_index: int) -> tuple[list[list[float]], list[int]]:
    """
    Split a polygon edge by inserting a midpoint after the edge start vertex.

    Returns (new_vertices, new_edge_indices_for_original_edge).
    """
    n = len(vertices)
    i = edge_index % n
    j = (i + 1) % n
    p0 = vertices[i]
    p1 = vertices[j]
    midpoint = [
        round((p0[0] + p1[0]) * 0.5, 6),
        round((p0[1] + p1[1]) * 0.5, 6),
    ]

    new_vertices = list(vertices)
    insert_at = i + 1
    new_vertices.insert(insert_at, midpoint)
    return new_vertices, [edge_index, edge_index + 1]


def _build_raw_edges(vertices: list[list[float]]) -> list[dict]:
    edges = []
    n = len(vertices)
    for i in range(n):
        edges.append({"endpoints": [i, (i + 1) % n]})
    return edges


def _edge_length_mm(vertices: list[list[float]], edge_index: int) -> float:
    n = len(vertices)
    i = edge_index % n
    j = (i + 1) % n
    x0, y0 = vertices[i]
    x1, y1 = vertices[j]
    return (((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5) * 10.0


def canonicalize_dxf_pattern_for_pipeline(pattern: dict) -> tuple[dict, dict[str, str], dict[str, dict[int, list[int]]]]:
    """
    Convert a DXF-derived pattern dict into a GarmentCode-like raw JSON payload.
    """
    labeled = pattern if all(
        "label" in panel for panel in pattern["panels"].values()
    ) else label_tshirt_panels(pattern)
    stitch_plan = build_tshirt_stitch_plan(labeled)

    panel_name_map = {}
    split_map: dict[str, dict[int, list[int]]] = {}
    raw_panels = {}
    for panel_id, panel in labeled["panels"].items():
        label = panel.get("label", panel_id)
        out_panel_id = _PANEL_ID_MAP.get(label, panel_id)
        panel_name_map[panel_id] = out_panel_id
        vertices = list(panel["vertices"])
        original_edge_count = len(panel["edges"])
        per_panel_split_map: dict[int, list[int]] = {}

        sleeve_cap_specs = [
            item for item in stitch_plan["sleeve_caps"]
            if item["sleeve_panel"] == panel_id and item["sleeve_cap_edge"] is not None
        ]
        if sleeve_cap_specs:
            sleeve_cap_edge = sleeve_cap_specs[0]["sleeve_cap_edge"]
            old_edge_index = int(sleeve_cap_edge.rsplit("_e", 1)[1])
            vertices, new_edge_indices = _split_edge_vertices(vertices, old_edge_index)
            for idx in range(original_edge_count):
                if idx < old_edge_index:
                    per_panel_split_map[idx] = [idx]
                elif idx == old_edge_index:
                    per_panel_split_map[idx] = new_edge_indices
                else:
                    per_panel_split_map[idx] = [idx + 1]
        else:
            for idx in range(original_edge_count):
                per_panel_split_map[idx] = [idx]

        raw_edges = _build_raw_edges(vertices)
        split_map[panel_id] = per_panel_split_map

        raw_panels[out_panel_id] = {
            "vertices": vertices,
            "edges": raw_edges,
            "translation": panel.get("translation", [0.0, 0.0, 0.0]),
            "rotation": panel.get("rotation", [0.0, 0.0, 0.0]),
        }

    raw_pattern = {
        "_forma_metadata": {
            "garment_id": labeled["garment_id"],
            "fabric_id": labeled.get("fabric_id", "cotton_jersey_default"),
        },
        "properties": {
            "units_in_meter": labeled.get("units_in_meter", 100),
        },
        "pattern": {
            "panels": raw_panels,
            "stitches": [],
        },
    }
    return raw_pattern, panel_name_map, split_map


def canonicalize_dxf_manifest_for_pipeline(
    pattern: dict,
    panel_name_map: dict[str, str],
    split_map: dict[str, dict[int, list[int]]],
    raw_pattern: dict,
) -> dict:
    """
    Build a seam manifest from a DXF-derived pattern and rename panel/edge ids so
    they align with the pipeline-ready pattern JSON.
    """
    labeled = pattern if all(
        "label" in panel for panel in pattern["panels"].values()
    ) else label_tshirt_panels(pattern)
    stitch_plan = build_tshirt_stitch_plan(labeled)

    next_panels = []
    edge_lengths: dict[str, float] = {}
    all_edge_ids: set[str] = set()
    for panel_id, raw_panel in raw_pattern["pattern"]["panels"].items():
        edges = []
        for idx, _ in enumerate(raw_panel["edges"]):
            edge_id = f"{panel_id}_e{idx}"
            all_edge_ids.add(edge_id)
            edge_lengths[edge_id] = _edge_length_mm(raw_panel["vertices"], idx)
            edges.append(
                {
                    "edge_id": edge_id,
                    "vertices": [idx, (idx + 1) % len(raw_panel["vertices"])],
                    "arc_length_mm": round(edge_lengths[edge_id], 3),
                    "label": "unknown",
                }
            )
        next_panels.append(
            {
                "panel_id": panel_id,
                "vertex_count": len(raw_panel["vertices"]),
                "edge_count": len(raw_panel["edges"]),
                "edges": edges,
            }
        )

    edge_label_updates = {}
    for item in stitch_plan["torso_pairs"]:
        if item["edge_a"]:
            edge_label_updates[_remap_edge_ids(item["edge_a"], panel_name_map, split_map)[0]] = item["kind"]
        if item["edge_b"]:
            edge_label_updates[_remap_edge_ids(item["edge_b"], panel_name_map, split_map)[0]] = item["kind"]
    for item in stitch_plan["underarm_pairs"]:
        if item["edge_a"]:
            edge_label_updates[_remap_edge_ids(item["edge_a"], panel_name_map, split_map)[0]] = "underarm"
        if item["edge_b"]:
            edge_label_updates[_remap_edge_ids(item["edge_b"], panel_name_map, split_map)[0]] = "underarm"
    for item in stitch_plan["sleeve_caps"]:
        if item["sleeve_cap_edge"]:
            for edge_id in _remap_edge_ids(item["sleeve_cap_edge"], panel_name_map, split_map):
                edge_label_updates[edge_id] = "sleeve_cap"
        for attach in item["attach_to"]:
            if attach:
                edge_label_updates[_remap_edge_ids(attach, panel_name_map, split_map)[0]] = "armhole"

    for panel in next_panels:
        for edge in panel["edges"]:
            if edge["edge_id"] in edge_label_updates:
                edge["label"] = edge_label_updates[edge["edge_id"]]

    next_seams = []
    dropped_pairs = []

    def _record_pair(
        edge_a: str | None,
        edge_b: str | None,
        seam_id: str,
        stitch_type: str,
        seam_label: str,
    ) -> None:
        if not edge_a or not edge_b:
            dropped_pairs.append(
                {
                    "seam_id": seam_id,
                    "reason": "missing_edge_role",
                    "seam_label": seam_label,
                    "edge_a": edge_a,
                    "edge_b": edge_b,
                }
            )
            return
        tol = 10.0 if stitch_type == "gather" else 2.0
        arc_diff = round(abs(edge_lengths[edge_a] - edge_lengths[edge_b]), 3)
        if arc_diff >= tol:
            dropped_pairs.append(
                {
                    "seam_id": seam_id,
                    "reason": "arc_length_tolerance",
                    "seam_label": seam_label,
                    "edge_a": edge_a,
                    "edge_b": edge_b,
                    "arc_length_diff_mm": arc_diff,
                    "tolerance_mm": tol,
                }
            )
            return
        next_seams.append(
            {
                "seam_id": seam_id,
                "edge_a": edge_a,
                "edge_b": edge_b,
                "arc_length_diff_mm": arc_diff,
                "valid": True,
                "stitch_type": stitch_type,
            }
        )

    seam_counter = 1
    for idx, item in enumerate(stitch_plan["torso_pairs"], start=1):
        _record_pair(
            _remap_edge_ids(item["edge_a"], panel_name_map, split_map)[0] if item["edge_a"] else None,
            _remap_edge_ids(item["edge_b"], panel_name_map, split_map)[0] if item["edge_b"] else None,
            f"seam_{seam_counter:03d}",
            "standard",
            f"torso_pair_{idx}:{item['kind']}",
        )
        seam_counter += 1

    for item in stitch_plan["underarm_pairs"]:
        _record_pair(
            _remap_edge_ids(item["edge_a"], panel_name_map, split_map)[0] if item["edge_a"] else None,
            _remap_edge_ids(item["edge_b"], panel_name_map, split_map)[0] if item["edge_b"] else None,
            f"seam_{seam_counter:03d}",
            "gather",
            f"{item['side']}_underarm",
        )
        seam_counter += 1

    for item in stitch_plan["sleeve_caps"]:
        if not item["sleeve_cap_edge"]:
            dropped_pairs.append(
                {
                    "seam_id": f"seam_{seam_counter:03d}",
                    "reason": "missing_sleeve_cap",
                    "seam_label": f"{item['side']}_sleeve_cap",
                    "edge_a": None,
                    "edge_b": None,
                }
            )
            continue
        split_edge_ids = _remap_edge_ids(item["sleeve_cap_edge"], panel_name_map, split_map)
        attach_edges = [
            _remap_edge_ids(edge_id, panel_name_map, split_map)[0] if edge_id else None
            for edge_id in item["attach_to"]
        ]
        for split_edge, attach_edge in zip(split_edge_ids, attach_edges):
            _record_pair(
                split_edge,
                attach_edge,
                f"seam_{seam_counter:03d}",
                "gather",
                f"{item['side']}_sleeve_cap_segment_{seam_counter:03d}",
            )
            seam_counter += 1

    if not next_seams:
        raise DXFBundleError("DXF seam inference did not produce any valid seam pairs")

    used_edges = {sp["edge_a"] for sp in next_seams} | {sp["edge_b"] for sp in next_seams}
    next_validation = {
        "all_seams_valid": all(sp["valid"] for sp in next_seams),
        "max_arc_length_diff_mm": round(max((sp["arc_length_diff_mm"] for sp in next_seams), default=0.0), 3),
        "tolerance_mm": 2.0,
        "unmatched_edges": sorted(all_edge_ids - used_edges),
        "total_seam_pairs": len(next_seams),
        "inference_complete": len(dropped_pairs) == 0,
        "dropped_seam_candidates": dropped_pairs,
    }

    return {
        "garment_id": labeled["garment_id"],
        "panel_count": len(next_panels),
        "panels": next_panels,
        "seam_pairs": next_seams,
        "validation": next_validation,
    }


def write_dxf_pipeline_bundle(
    dxf_path: str | Path,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """
    Convert a DXF file into pipeline-readable `pattern.json` and
    `seam_manifest.json` files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = load_dxf_pattern(dxf_path)
    raw_pattern, panel_name_map, split_map = canonicalize_dxf_pattern_for_pipeline(pattern)
    manifest = canonicalize_dxf_manifest_for_pipeline(pattern, panel_name_map, split_map, raw_pattern)

    pattern_path = out_dir / "pattern.json"
    manifest_path = out_dir / "seam_manifest.json"

    with open(pattern_path, "w") as f:
        json.dump(raw_pattern, f, indent=2)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return pattern_path, manifest_path

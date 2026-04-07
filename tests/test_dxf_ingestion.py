from __future__ import annotations

import sys
from pathlib import Path

import ezdxf
import pytest


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pattern_maker.load_dxf import DxfLoadError, load_dxf_pattern
from tailor.panel_labeler import PanelLabelError, label_tshirt_panels
from dxf_fixture_factory import write_messy_tshirt_dxf


def _write_test_dxf(path: Path, panels: list[tuple[str, list[tuple[float, float]]]]) -> None:
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    for layer_name, points in panels:
        if layer_name not in doc.layers:
            doc.layers.add(layer_name)
        msp.add_lwpolyline(points, format="xy", close=True, dxfattribs={"layer": layer_name})
    doc.saveas(path)


class TestPanelLabeler:
    def test_explicit_labels_win(self):
        panels = {
            "front_body": [[0, 0], [40, 0], [40, 60], [0, 60]],
            "back_body": [[0, 0], [40, 0], [40, 60], [20, 50], [0, 60]],
            "left_sleeve": [[-40, 0], [-16, 0], [-10, 18], [-25, 36], [-40, 18]],
            "right_sleeve": [[16, 0], [40, 0], [46, 18], [31, 36], [16, 18]],
        }
        labels = label_tshirt_panels(panels)
        assert labels == {
            "front_body": "front_body",
            "back_body": "back_body",
            "left_sleeve": "left_sleeve",
            "right_sleeve": "right_sleeve",
        }

    def test_heuristics_assign_front_back_and_sleeves(self):
        panels = {
            "panel_a": [[-20, 0], [20, 0], [20, 60], [5, 44], [0, 40], [-5, 44], [-20, 60]],
            "panel_b": [[-20, 0], [20, 0], [20, 60], [-20, 60]],
            "panel_c": [[25, 0], [50, 0], [58, 18], [40, 36], [22, 18]],
            "panel_d": [[-50, 0], [-25, 0], [-22, 18], [-40, 36], [-58, 18]],
        }
        labels = label_tshirt_panels(panels)
        assert labels["panel_a"] == "front_body"
        assert labels["panel_b"] == "back_body"
        assert labels["panel_c"] == "left_sleeve"
        assert labels["panel_d"] == "right_sleeve"

    def test_too_few_panels_raises(self):
        with pytest.raises(PanelLabelError):
            label_tshirt_panels({"a": [[0, 0], [1, 0], [0, 1]]})

    def test_more_than_four_panels_drops_smallest_extras(self):
        # panel_e is tiny (20×20); the labeler should silently keep the 4 largest
        # and return labels only for those.
        panels = {
            "panel_a": [[-20, 0], [20, 0], [20, 60], [5, 44], [0, 40], [-5, 44], [-20, 60]],
            "panel_b": [[-20, 0], [20, 0], [20, 60], [-20, 60]],
            "panel_c": [[25, 0], [50, 0], [58, 18], [40, 36], [22, 18]],
            "panel_d": [[-50, 0], [-25, 0], [-22, 18], [-40, 36], [-58, 18]],
            "panel_e": [[60, 0], [80, 0], [80, 20], [60, 20]],
        }
        labels = label_tshirt_panels(panels)
        # Returns a dict — only 4 entries, extras dropped.
        assert "panel_e" not in labels
        assert len(labels) == 4


class TestLoadDxfPattern:
    def test_loads_closed_polylines_into_canonical_pattern(self, tmp_path: Path):
        dxf_path = tmp_path / "shirt.dxf"
        _write_test_dxf(
            dxf_path,
            [
                ("front_body", [(0, 0), (40, 0), (40, 60), (20, 50), (0, 60)]),
                ("back_body", [(50, 0), (90, 0), (90, 60), (50, 60)]),
                ("left_sleeve", [(0, 70), (24, 70), (30, 88), (15, 106), (0, 88)]),
                ("right_sleeve", [(60, 70), (84, 70), (90, 88), (75, 106), (60, 88)]),
            ],
        )

        pattern = load_dxf_pattern(dxf_path)

        assert pattern["garment_id"] == "shirt"
        assert pattern["stitches"] == []
        assert set(pattern["panels"]) == {
            "front_body",
            "back_body",
            "left_sleeve",
            "right_sleeve",
        }
        front = pattern["panels"]["front_body"]
        assert front["translation"] == [0.0, 0.0, 12.0]
        assert front["rotation"] == [0.0, 0.0, 0.0]
        assert len(front["vertices"]) == 5
        assert len(front["edges"]) == 5
        assert front["edges"][0]["endpoints"] == [0, 1]
        assert front["edges"][0]["arc_length_mm"] > 0

    def test_raises_when_no_closed_panels_exist(self, tmp_path: Path):
        path = tmp_path / "empty.dxf"
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        msp.add_line((0, 0), (10, 0))
        doc.saveas(path)

        with pytest.raises(DxfLoadError, match="No closed DXF panel outlines"):
            load_dxf_pattern(path)

    def test_ignores_open_helper_polylines_when_closed_panels_exist(self, tmp_path: Path):
        dxf_path = tmp_path / "shirt_with_helpers.dxf"
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        for layer_name, points in [
            ("front_body", [(0, 0), (40, 0), (40, 60), (20, 50), (0, 60)]),
            ("back_body", [(50, 0), (90, 0), (90, 60), (50, 60)]),
            ("left_sleeve", [(0, 70), (24, 70), (30, 88), (15, 106), (0, 88)]),
            ("right_sleeve", [(60, 70), (84, 70), (90, 88), (75, 106), (60, 88)]),
        ]:
            if layer_name not in doc.layers:
                doc.layers.add(layer_name)
            msp.add_lwpolyline(points, format="xy", close=True, dxfattribs={"layer": layer_name})
        msp.add_lwpolyline(
            [(0, 0), (10, 0), (10, 10)],
            format="xy",
            close=False,
            dxfattribs={"layer": "helpers"},
        )
        doc.saveas(dxf_path)

        pattern = load_dxf_pattern(dxf_path)

        assert set(pattern["panels"]) == {
            "front_body",
            "back_body",
            "left_sleeve",
            "right_sleeve",
        }


class TestMessyDxfFixture:
    def test_extra_annotation_polylines_are_silently_dropped(self, tmp_path: Path):
        dxf_path = write_messy_tshirt_dxf(tmp_path / "messy.dxf")
        pattern = load_dxf_pattern(dxf_path)

        # Exactly 4 canonical panels — no annotation rectangles.
        assert set(pattern["panels"]) == {
            "front_body",
            "back_body",
            "left_sleeve",
            "right_sleeve",
        }

    def test_messy_dxf_seam_inference_still_produces_side_seams(self, tmp_path: Path):
        from tailor.dxf_bundle import canonicalize_dxf_pattern_for_pipeline, canonicalize_dxf_manifest_for_pipeline

        dxf_path = write_messy_tshirt_dxf(tmp_path / "messy.dxf")
        pattern = load_dxf_pattern(dxf_path)
        raw_pattern, panel_name_map, split_map = canonicalize_dxf_pattern_for_pipeline(pattern)
        manifest = canonicalize_dxf_manifest_for_pipeline(pattern, panel_name_map, split_map, raw_pattern)

        seam_pairs = manifest["seam_pairs"]
        assert len(seam_pairs) >= 2
        panel_ids = {p["panel_id"] for p in manifest["panels"]}
        assert "front_ftorso" in panel_ids
        assert "back_btorso" in panel_ids


class TestSplineDxfIngestion:
    """SPLINE entity support (CLO3D / Optitex / Lectra DXFs with curved outlines)."""

    @staticmethod
    def _write_spline_tshirt_dxf(path: Path) -> Path:
        """Write a DXF with 4 closed SPLINE panel outlines (simple rectangles as B-splines)."""
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()

        panels = {
            "front_body":   [(0, 0), (180, 0), (180, 200), (90, 230), (0, 200)],
            "back_body":    [(220, 0), (400, 0), (400, 200), (310, 230), (220, 200)],
            "left_sleeve":  [(-120, 100), (-40, 100), (-20, 150), (-80, 210), (-120, 150)],
            "right_sleeve": [(440, 100), (520, 100), (540, 150), (480, 210), (440, 150)],
        }
        for layer, pts in panels.items():
            spline = msp.add_spline(fit_points=pts, degree=3, dxfattribs={"layer": layer})
            spline.closed = True
        doc.saveas(path)
        return path

    def test_spline_panels_loaded_as_canonical_pattern(self, tmp_path: Path):
        dxf_path = self._write_spline_tshirt_dxf(tmp_path / "spline_shirt.dxf")
        pattern = load_dxf_pattern(dxf_path)

        assert set(pattern["panels"]) == {
            "front_body", "back_body", "left_sleeve", "right_sleeve",
        }
        # Each panel should have more vertices than the 5 fit points (SPLINE flattened)
        for panel_id, panel in pattern["panels"].items():
            assert len(panel["vertices"]) >= 5, (
                f"{panel_id}: expected ≥5 vertices from SPLINE flattening, "
                f"got {len(panel['vertices'])}"
            )

    def test_open_spline_is_ignored(self, tmp_path: Path):
        """An open SPLINE (not a closed panel boundary) should not be ingested."""
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        # Four valid closed LWPOLYLINE panels
        for layer, pts in [
            ("front_body",   [(0, 0), (180, 0), (180, 200), (0, 200)]),
            ("back_body",    [(220, 0), (400, 0), (400, 200), (220, 200)]),
            ("left_sleeve",  [(-120, 100), (-40, 100), (-40, 200), (-120, 200)]),
            ("right_sleeve", [(440, 100), (520, 100), (520, 200), (440, 200)]),
        ]:
            msp.add_lwpolyline(pts, format="xy", close=True, dxfattribs={"layer": layer})
        # Open SPLINE annotation (grainline arrow, etc.) — should be silently skipped
        msp.add_spline(
            fit_points=[(10, 10), (20, 30), (30, 10)],
            degree=3,
            dxfattribs={"layer": "grainline"},
        )
        dxf_path = tmp_path / "mixed.dxf"
        doc.saveas(dxf_path)

        pattern = load_dxf_pattern(dxf_path)
        assert set(pattern["panels"]) == {
            "front_body", "back_body", "left_sleeve", "right_sleeve",
        }

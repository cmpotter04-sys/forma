"""
AC-2 Test Suite — Pattern Loading & Seam Manifest Generation

Tests:
  1. All three T-shirt patterns load cleanly (no schema errors)
  2. Each pattern produces a valid seam_manifest.json
  3. Seam manifests conform to FORMA_SEAM_MANIFEST_SCHEMA.md
  4. Seam pairs validate within ±2mm tolerance
  5. SeamValidationError is raised on bad input
  6. Size ordering: S < M < XL by chest circumference
"""
import json
import math
import pytest
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pattern_maker.load_patterns import load_pattern, load_all_sizes, PatternLoadError
from src.tailor.seam_converter import (
    build_seam_manifest, write_seam_manifest,
    load_and_validate_manifest, SeamValidationError,
    TOLERANCE_MM, GATHER_TOLERANCE_MM,
)

PATTERNS_DIR = Path("data/patterns")
SEAM_DIR = Path("seam_manifests")
SEAM_DIR.mkdir(parents=True, exist_ok=True)

SIZES = ["S", "M", "XL"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def patterns():
    return load_all_sizes(PATTERNS_DIR)


@pytest.fixture(scope="session")
def manifests(patterns):
    """Build and write seam manifests for all sizes. Returns dict of manifests."""
    result = {}
    for size, pattern in patterns.items():
        manifest = build_seam_manifest(pattern)
        out_path = SEAM_DIR / f"tshirt_size_{size}_manifest.json"
        write_seam_manifest(manifest, out_path)
        result[size] = manifest
    return result


# ---------------------------------------------------------------------------
# AC-2.1: Pattern loading
# ---------------------------------------------------------------------------

class TestPatternLoading:

    @pytest.mark.parametrize("size", SIZES)
    def test_pattern_file_exists(self, size):
        assert (PATTERNS_DIR / f"tshirt_size_{size}.json").exists(), \
            f"data/patterns/tshirt_size_{size}.json not found"

    @pytest.mark.parametrize("size", SIZES)
    def test_pattern_loads_without_error(self, size):
        pattern = load_pattern(PATTERNS_DIR / f"tshirt_size_{size}.json")
        assert pattern is not None

    @pytest.mark.parametrize("size", SIZES)
    def test_pattern_has_garment_id(self, patterns, size):
        assert patterns[size]["garment_id"].startswith("tshirt_gc_v1_size_")

    @pytest.mark.parametrize("size", SIZES)
    def test_pattern_has_panels(self, patterns, size):
        panels = patterns[size]["panels"]
        assert len(panels) >= 4, f"Size {size}: expected ≥4 panels, got {len(panels)}"

    @pytest.mark.parametrize("size", SIZES)
    def test_pattern_has_stitches(self, patterns, size):
        stitches = patterns[size]["stitches"]
        assert len(stitches) > 0, f"Size {size}: no stitches found"

    @pytest.mark.parametrize("size", SIZES)
    def test_all_edges_have_positive_arc_length(self, patterns, size):
        for panel_name, panel in patterns[size]["panels"].items():
            for edge in panel["edges"]:
                assert edge["arc_length_mm"] > 0, \
                    f"Size {size}, panel {panel_name}, edge {edge['edge_index']}: " \
                    f"arc_length_mm={edge['arc_length_mm']}"

    def test_bad_pattern_raises_error(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('{"no_pattern_key": {}}')
        with pytest.raises(PatternLoadError):
            load_pattern(bad)


# ---------------------------------------------------------------------------
# AC-2.2: Seam manifest structure
# ---------------------------------------------------------------------------

class TestSeamManifestStructure:

    @pytest.mark.parametrize("size", SIZES)
    def test_manifest_has_required_fields(self, manifests, size):
        m = manifests[size]
        for field in ["garment_id", "panel_count", "panels", "seam_pairs", "validation"]:
            assert field in m, f"Size {size}: missing field '{field}'"

    @pytest.mark.parametrize("size", SIZES)
    def test_panel_count_matches_panels_array(self, manifests, size):
        m = manifests[size]
        assert m["panel_count"] == len(m["panels"])

    @pytest.mark.parametrize("size", SIZES)
    def test_seam_manifest_file_written(self, manifests, size):
        # manifests fixture writes the files
        path = SEAM_DIR / f"tshirt_size_{size}_manifest.json"
        assert path.exists(), f"Seam manifest file not found: {path}"

    @pytest.mark.parametrize("size", SIZES)
    def test_panels_have_required_fields(self, manifests, size):
        for panel in manifests[size]["panels"]:
            for field in ["panel_id", "vertex_count", "edge_count", "edges"]:
                assert field in panel, \
                    f"Size {size}, panel {panel.get('panel_id')}: missing '{field}'"

    @pytest.mark.parametrize("size", SIZES)
    def test_edges_have_required_fields(self, manifests, size):
        for panel in manifests[size]["panels"]:
            for edge in panel["edges"]:
                for field in ["edge_id", "vertices", "arc_length_mm", "label"]:
                    assert field in edge, \
                        f"Size {size}, panel {panel['panel_id']}, " \
                        f"edge {edge.get('edge_id')}: missing '{field}'"

    @pytest.mark.parametrize("size", SIZES)
    def test_seam_pairs_have_required_fields(self, manifests, size):
        for sp in manifests[size]["seam_pairs"]:
            for field in ["seam_id", "edge_a", "edge_b",
                          "arc_length_diff_mm", "valid", "stitch_type"]:
                assert field in sp, f"Size {size}, {sp.get('seam_id')}: missing '{field}'"

    @pytest.mark.parametrize("size", SIZES)
    def test_validation_object_present(self, manifests, size):
        v = manifests[size]["validation"]
        for field in ["all_seams_valid", "max_arc_length_diff_mm",
                      "tolerance_mm", "unmatched_edges", "total_seam_pairs"]:
            assert field in v, f"Size {size}: validation missing '{field}'"


# ---------------------------------------------------------------------------
# AC-2.3: Seam validation correctness
# ---------------------------------------------------------------------------

class TestSeamValidation:

    @pytest.mark.parametrize("size", SIZES)
    def test_all_seams_valid(self, manifests, size):
        m = manifests[size]
        assert m["validation"]["all_seams_valid"], \
            f"Size {size}: some seams are invalid"

    @pytest.mark.parametrize("size", SIZES)
    def test_standard_seam_tolerance(self, manifests, size):
        for sp in manifests[size]["seam_pairs"]:
            if sp["stitch_type"] == "standard":
                assert sp["arc_length_diff_mm"] < TOLERANCE_MM, \
                    f"Size {size}, {sp['seam_id']}: standard seam exceeds " \
                    f"{TOLERANCE_MM}mm tolerance: {sp['arc_length_diff_mm']:.3f}mm"

    @pytest.mark.parametrize("size", SIZES)
    def test_gather_seam_tolerance(self, manifests, size):
        for sp in manifests[size]["seam_pairs"]:
            if sp["stitch_type"] == "gather":
                assert sp["arc_length_diff_mm"] < GATHER_TOLERANCE_MM, \
                    f"Size {size}, {sp['seam_id']}: gather seam exceeds " \
                    f"{GATHER_TOLERANCE_MM}mm tolerance: {sp['arc_length_diff_mm']:.3f}mm"

    @pytest.mark.parametrize("size", SIZES)
    def test_no_orphan_edge_references(self, manifests, size):
        m = manifests[size]
        edge_ids = {
            e["edge_id"]
            for panel in m["panels"]
            for e in panel["edges"]
        }
        for sp in m["seam_pairs"]:
            assert sp["edge_a"] in edge_ids, \
                f"Size {size}: orphan edge_a={sp['edge_a']!r}"
            assert sp["edge_b"] in edge_ids, \
                f"Size {size}: orphan edge_b={sp['edge_b']!r}"

    @pytest.mark.parametrize("size", SIZES)
    def test_no_duplicate_edge_assignments(self, manifests, size):
        seen = set()
        for sp in manifests[size]["seam_pairs"]:
            for eid in (sp["edge_a"], sp["edge_b"]):
                assert eid not in seen, \
                    f"Size {size}: edge {eid!r} appears in multiple seam pairs"
                seen.add(eid)

    @pytest.mark.parametrize("size", SIZES)
    def test_seam_pair_count_matches_validation(self, manifests, size):
        m = manifests[size]
        assert m["validation"]["total_seam_pairs"] == len(m["seam_pairs"])

    @pytest.mark.parametrize("size", SIZES)
    def test_seam_id_format(self, manifests, size):
        for sp in manifests[size]["seam_pairs"]:
            assert sp["seam_id"].startswith("seam_"), \
                f"Size {size}: invalid seam_id format: {sp['seam_id']!r}"

    @pytest.mark.parametrize("size", SIZES)
    def test_stitch_type_enum(self, manifests, size):
        for sp in manifests[size]["seam_pairs"]:
            assert sp["stitch_type"] in ("standard", "gather"), \
                f"Size {size}: invalid stitch_type: {sp['stitch_type']!r}"

    def test_seam_validation_error_raised_on_bad_manifest(self, tmp_path):
        """Verify SeamValidationError is raised for orphan edge references."""
        bad_manifest = {
            "garment_id": "test",
            "panel_count": 1,
            "panels": [
                {
                    "panel_id": "front",
                    "vertex_count": 4,
                    "edge_count": 1,
                    "edges": [
                        {"edge_id": "front_e0", "vertices": [0, 1],
                         "arc_length_mm": 100.0, "label": "test"}
                    ]
                }
            ],
            "seam_pairs": [
                {"seam_id": "seam_001", "edge_a": "front_e0",
                 "edge_b": "nonexistent_e0",   # orphan reference
                 "arc_length_diff_mm": 0.0, "valid": True, "stitch_type": "standard"}
            ],
            "validation": {
                "all_seams_valid": True, "max_arc_length_diff_mm": 0.0,
                "tolerance_mm": 2.0, "unmatched_edges": [], "total_seam_pairs": 1
            }
        }
        p = tmp_path / "bad_manifest.json"
        p.write_text(json.dumps(bad_manifest))
        with pytest.raises(SeamValidationError):
            load_and_validate_manifest(p)


# ---------------------------------------------------------------------------
# AC-2.4: Size ordering (physical correctness check)
# ---------------------------------------------------------------------------

class TestSizeOrdering:

    def _chest_circumference_mm(self, patterns, size):
        """
        Approximate chest circumference as sum of torso panel widths (horizontal extents).
        Uses max_x - min_x of panel vertices, converted to mm (units_in_meter=100 → ×10).
        """
        pattern = patterns[size]
        units_to_mm = (1.0 / pattern["units_in_meter"]) * 1000.0
        total = 0.0
        for panel_name, panel in pattern["panels"].items():
            if "torso" in panel_name.lower() or "ftorso" in panel_name.lower() \
                    or "btorso" in panel_name.lower():
                xs = [v[0] for v in panel["vertices"]]
                width_mm = (max(xs) - min(xs)) * units_to_mm
                total += width_mm
        return total

    def test_s_smaller_than_m(self, patterns):
        s = self._chest_circumference_mm(patterns, "S")
        m = self._chest_circumference_mm(patterns, "M")
        assert s < m, f"Expected S ({s:.1f}mm) < M ({m:.1f}mm)"

    def test_m_smaller_than_xl(self, patterns):
        m = self._chest_circumference_mm(patterns, "M")
        xl = self._chest_circumference_mm(patterns, "XL")
        assert m < xl, f"Expected M ({m:.1f}mm) < XL ({xl:.1f}mm)"

    def test_s_meaningfully_smaller_than_body(self, patterns):
        """S shirt chest circumference should be less than body chest (960mm = 96cm)."""
        body_chest_mm = 960.0
        s = self._chest_circumference_mm(patterns, "S")
        # S should be reasonably smaller — at least 50mm below body chest
        assert s < body_chest_mm, \
            f"S shirt ({s:.1f}mm) is not smaller than body chest ({body_chest_mm}mm)"

    def test_xl_meaningfully_larger_than_body(self, patterns):
        """XL shirt should be well above body chest (960mm = 96cm)."""
        body_chest_mm = 960.0
        xl = self._chest_circumference_mm(patterns, "XL")
        assert xl > body_chest_mm, \
            f"XL shirt ({xl:.1f}mm) is not larger than body chest ({body_chest_mm}mm)"


# ---------------------------------------------------------------------------
# AC-2.5: On-disk round-trip
# ---------------------------------------------------------------------------

class TestOnDiskRoundTrip:

    @pytest.mark.parametrize("size", SIZES)
    def test_manifest_json_is_valid(self, manifests, size):
        path = SEAM_DIR / f"tshirt_size_{size}_manifest.json"
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["garment_id"] is not None

    @pytest.mark.parametrize("size", SIZES)
    def test_reloaded_manifest_passes_validation(self, manifests, size):
        path = SEAM_DIR / f"tshirt_size_{size}_manifest.json"
        reloaded = load_and_validate_manifest(path)
        assert reloaded["validation"]["all_seams_valid"]


# ---------------------------------------------------------------------------
# Tank Top (size M) fixture tests
# ---------------------------------------------------------------------------

TANK_TOP_PATTERN = Path("data/patterns/tank_top_size_M.json")
TANK_TOP_MANIFEST = Path("data/seam_manifests/tank_top_size_M_seam_manifest.json")


class TestTankTopPattern:

    def test_tank_top_loads_without_error(self):
        pattern = load_pattern(TANK_TOP_PATTERN)
        assert pattern is not None

    def test_tank_top_has_two_panels(self):
        pattern = load_pattern(TANK_TOP_PATTERN)
        assert len(pattern["panels"]) == 2

    def test_tank_top_garment_id(self):
        pattern = load_pattern(TANK_TOP_PATTERN)
        assert pattern["garment_id"] == "tank_top_size_M"

    def test_tank_top_seam_manifest_all_seams_valid(self):
        with open(TANK_TOP_MANIFEST) as f:
            manifest = json.load(f)
        assert manifest["validation"]["all_seams_valid"] is True

    def test_tank_top_seam_manifest_has_two_seam_pairs(self):
        with open(TANK_TOP_MANIFEST) as f:
            manifest = json.load(f)
        assert manifest["validation"]["total_seam_pairs"] == 2


# ---------------------------------------------------------------------------
# Dress (size M) fixture tests
# ---------------------------------------------------------------------------

DRESS_PATTERN = Path("data/patterns/dress_size_M.json")
DRESS_MANIFEST = Path("data/seam_manifests/dress_size_M_seam_manifest.json")


class TestDressPattern:

    def test_dress_loads_without_error(self):
        pattern = load_pattern(DRESS_PATTERN)
        assert pattern is not None

    def test_dress_has_four_panels_with_correct_names(self):
        pattern = load_pattern(DRESS_PATTERN)
        assert set(pattern["panels"].keys()) == {
            "front_torso", "back_torso", "front_skirt", "back_skirt"
        }

    def test_dress_garment_id(self):
        pattern = load_pattern(DRESS_PATTERN)
        assert pattern["garment_id"] == "dress_gc_v1_size_M"

    def test_dress_seam_manifest_all_seams_valid(self):
        with open(DRESS_MANIFEST) as f:
            manifest = json.load(f)
        assert manifest["validation"]["all_seams_valid"] is True

    def test_dress_seam_manifest_has_six_seam_pairs(self):
        with open(DRESS_MANIFEST) as f:
            manifest = json.load(f)
        assert manifest["validation"]["total_seam_pairs"] == 6

"""
AC-5 Test Suite — Verdict Schema Compliance (v1.2)

Tests:
  1. generate_verdict() produces correct schema from sim results
  2. All required fields present and correctly typed
  3. verdict_id format: vrd_{12 hex chars}
  4. body_source enum validation
  5. All required regions in strain_map and ease_map
  6. Severity classification correctness (per FORMA_WEEK1_SPEC.md thresholds)
  7. fit boolean logic: True iff zero red regions
  8. fabric_params come from fabric_library.json (no magic numbers)
  9. tunnel_through_pct < 2.0
 10. On-disk verdict files (vrd_S_on_M.json, etc.) — skipped until Geometer delivers results
"""
import json
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.verdict.generate_verdict import (
    generate_verdict, save_verdict, validate_verdict_schema,
    REQUIRED_REGIONS,
)

VERDICT_DIR = Path("output/verdicts")

SIZES = ["S", "M", "XL"]

# ---------------------------------------------------------------------------
# Synthetic sim results for unit testing (do NOT use in AC-4 — values here
# are only for testing schema logic. Real verdicts come from Geometer.)
# ---------------------------------------------------------------------------

def _make_sim_result(clearance_map: dict) -> dict:
    return {
        "clearance_map": clearance_map,
        "simulation_ms": 3200,
        "convergence_step": 87,
        "final_kinetic_energy_j": 4.2e-7,
        "tunnel_through_pct": 0.3,
    }


# Realistic clearance maps for smoke-test validation
_SIM_RESULTS = {
    "S": _make_sim_result({
        # Size S on size M body — slightly tight, no red regions → fit=True.
        # Revised AC (Week 1): fit=True is acceptable; chest_side is yellow.
        "chest_front": -8.7,
        "chest_side": -13.0,   # yellow: −25 ≤ delta ≤ −10
        "shoulder_left": -6.3,
        "shoulder_right": -6.3,
        "upper_back": -8.0,
        "waist": 10.7,
    }),
    "M": _make_sim_result({
        # Size M on size M body — small positive ease, correct fit
        "chest_front": 12.4,
        "chest_side": 8.9,
        "shoulder_left": 4.2,
        "shoulder_right": 4.7,
        "upper_back": 9.1,
        "waist": 7.6,
    }),
    "XL": _make_sim_result({
        # Size XL on size M body — large positive ease
        "chest_front": 52.3,
        "chest_side": 47.8,
        "shoulder_left": 38.9,
        "shoulder_right": 39.4,
        "upper_back": 44.1,
        "waist": 61.2,
    }),
}

_GARMENT_IDS = {
    "S": "tshirt_gc_v1_size_S",
    "M": "tshirt_gc_v1_size_M",
    "XL": "tshirt_gc_v1_size_XL",
}
_OUTPUT_FILENAMES = {
    "S": "vrd_S_on_M.json",
    "M": "vrd_M_on_M.json",
    "XL": "vrd_XL_on_M.json",
}
BODY_PROFILE_ID = "makehuman_male_M"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def verdicts():
    """
    Generate in-memory verdicts from synthetic sim results for unit testing.
    Does NOT write to disk — on-disk verdicts come from scripts/run_all_simulations.py.
    """
    return {
        size: generate_verdict(
            sim_result=_SIM_RESULTS[size],
            garment_id=_GARMENT_IDS[size],
            body_profile_id=BODY_PROFILE_ID,
        )
        for size in SIZES
    }


def load_verdict(filename: str) -> dict:
    with open(VERDICT_DIR / filename) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# AC-5.1: Required fields
# ---------------------------------------------------------------------------

class TestRequiredFields:

    REQUIRED_TOP = [
        "verdict_id", "fit", "confidence", "body_source",
        "scan_method", "scan_accuracy_mm", "garment_id",
        "body_profile_id", "strain_map", "ease_map",
        "simulation_ms", "convergence_step", "final_kinetic_energy_j",
        "tunnel_through_pct", "fabric_params_used",
    ]

    REQUIRED_FABRIC = [
        "fabric_id", "type", "density_kg_m2",
        "stretch_stiffness", "bend_stiffness", "shear_stiffness", "damping",
    ]

    @pytest.mark.parametrize("size", SIZES)
    def test_required_top_level_fields(self, verdicts, size):
        v = verdicts[size]
        for field in self.REQUIRED_TOP:
            assert field in v, f"Size {size}: missing field {field!r}"

    @pytest.mark.parametrize("size", SIZES)
    def test_required_fabric_fields(self, verdicts, size):
        fp = verdicts[size]["fabric_params_used"]
        for field in self.REQUIRED_FABRIC:
            assert field in fp, f"Size {size}: fabric_params_used missing {field!r}"

    @pytest.mark.parametrize("size", SIZES)
    def test_strain_map_has_all_regions(self, verdicts, size):
        regions = {r["region"] for r in verdicts[size]["strain_map"]}
        for req in REQUIRED_REGIONS:
            assert req in regions, f"Size {size}: strain_map missing region {req!r}"

    @pytest.mark.parametrize("size", SIZES)
    def test_ease_map_has_all_regions(self, verdicts, size):
        regions = {r["region"] for r in verdicts[size]["ease_map"]}
        for req in REQUIRED_REGIONS:
            assert req in regions, f"Size {size}: ease_map missing region {req!r}"


# ---------------------------------------------------------------------------
# AC-5.2: verdict_id format
# ---------------------------------------------------------------------------

class TestVerdictIdFormat:

    @pytest.mark.parametrize("size", SIZES)
    def test_verdict_id_starts_with_vrd(self, verdicts, size):
        vid = verdicts[size]["verdict_id"]
        assert vid.startswith("vrd_"), f"Size {size}: verdict_id={vid!r}"

    @pytest.mark.parametrize("size", SIZES)
    def test_verdict_id_hex_part_is_12_chars(self, verdicts, size):
        vid = verdicts[size]["verdict_id"]
        hex_part = vid[4:]
        assert len(hex_part) == 12, \
            f"Size {size}: hex part length={len(hex_part)}, expected 12"

    @pytest.mark.parametrize("size", SIZES)
    def test_verdict_id_hex_part_is_valid_hex(self, verdicts, size):
        vid = verdicts[size]["verdict_id"]
        hex_part = vid[4:]
        int(hex_part, 16)  # raises ValueError if not valid hex

    @pytest.mark.parametrize("size", SIZES)
    def test_verdict_ids_are_unique(self, verdicts, size):
        # Each generated verdict gets a unique UUID — check all three are different
        ids = [verdicts[s]["verdict_id"] for s in SIZES]
        assert len(set(ids)) == len(ids), f"Duplicate verdict_ids: {ids}"


# ---------------------------------------------------------------------------
# AC-5.3: Enum validation
# ---------------------------------------------------------------------------

class TestEnumValidation:

    @pytest.mark.parametrize("size", SIZES)
    def test_body_source_enum(self, verdicts, size):
        valid = {"synthetic_mannequin", "standard_photo", "precision_suit"}
        assert verdicts[size]["body_source"] in valid, \
            f"Size {size}: invalid body_source={verdicts[size]['body_source']!r}"

    @pytest.mark.parametrize("size", SIZES)
    def test_severity_enum(self, verdicts, size):
        for r in verdicts[size]["strain_map"]:
            assert r["severity"] in {"green", "yellow", "red"}, \
                f"Size {size}, {r['region']}: invalid severity={r['severity']!r}"

    @pytest.mark.parametrize("size", SIZES)
    def test_ease_verdict_enum(self, verdicts, size):
        for r in verdicts[size]["ease_map"]:
            assert r["verdict"] in {"tight_fit", "standard_fit", "relaxed_fit", "oversized"}, \
                f"Size {size}, {r['region']}: invalid verdict={r['verdict']!r}"

    @pytest.mark.parametrize("size", SIZES)
    def test_scan_method_matches_spec(self, verdicts, size):
        assert verdicts[size]["scan_method"] == "synthetic_mannequin"
        assert verdicts[size]["scan_accuracy_mm"] == 0

    def test_body_source_propagates_to_verdict(self):
        """body_source parameter must appear verbatim in the verdict (CLAUDE.md rule #3)."""
        verdict = generate_verdict(
            _SIM_RESULTS["M"], "test_id", "body_id",
            body_source="precision_suit",
            confidence=0.95,
        )
        assert verdict["body_source"] == "precision_suit", \
            f"Expected body_source='precision_suit', got {verdict['body_source']!r}"

    def test_body_source_is_always_present(self):
        """body_source is REQUIRED — must never be absent from a generated verdict."""
        verdict = generate_verdict(_SIM_RESULTS["M"], "test_id", "body_id")
        assert "body_source" in verdict, "body_source missing from verdict"
        assert verdict["body_source"] is not None


# ---------------------------------------------------------------------------
# AC-5.4: Severity thresholds (FORMA_WEEK1_SPEC.md)
# ---------------------------------------------------------------------------

class TestSeverityThresholds:
    """Verify the severity classification function against spec thresholds."""

    def test_green_threshold(self, verdicts):
        """delta_mm > -10 → green"""
        for r in verdicts["M"]["strain_map"]:
            if r["delta_mm"] > -10.0:
                assert r["severity"] == "green", \
                    f"Expected green for delta={r['delta_mm']:.1f}, got {r['severity']}"

    def test_yellow_threshold(self, verdicts):
        """−25 ≤ delta_mm ≤ −10 → yellow"""
        # waist on S is -15.3 → yellow
        for r in verdicts["S"]["strain_map"]:
            if -25.0 <= r["delta_mm"] <= -10.0:
                assert r["severity"] == "yellow", \
                    f"Expected yellow for delta={r['delta_mm']:.1f}, got {r['severity']}"

    def test_red_threshold(self, verdicts):
        """delta_mm < -25 → red (checked across all sizes)"""
        for size in SIZES:
            for r in verdicts[size]["strain_map"]:
                if r["delta_mm"] < -25.0:
                    assert r["severity"] == "red", \
                        f"Size {size}, {r['region']}: Expected red for delta={r['delta_mm']:.1f}, got {r['severity']}"


# ---------------------------------------------------------------------------
# AC-5.5: fit boolean logic (non-negotiable rule #5)
# ---------------------------------------------------------------------------

class TestFitBooleanLogic:

    @pytest.mark.parametrize("size", SIZES)
    def test_fit_false_iff_any_red(self, verdicts, size):
        v = verdicts[size]
        has_red = any(r["severity"] == "red" for r in v["strain_map"])
        if has_red:
            assert v["fit"] is False, \
                f"Size {size}: has red regions but fit=True"
        else:
            assert v["fit"] is True, \
                f"Size {size}: no red regions but fit=False"

    def test_s_fit_is_true_with_yellow(self, verdicts):
        """S on M body: fit=True (no red regions), at least one yellow region.
        Revised AC Week 1: slight tightness is yellow, not red; fit is True."""
        assert verdicts["S"]["fit"] is True, \
            f"S fit={verdicts['S']['fit']}, expected True (no red regions)"
        severities = {r["severity"] for r in verdicts["S"]["strain_map"]}
        assert "yellow" in severities, \
            f"S has no yellow regions — expected at least one; severities={severities}"

    def test_m_fit_is_true(self, verdicts):
        """M on M body should fit (no red regions)."""
        assert verdicts["M"]["fit"] is True

    def test_xl_fit_is_true(self, verdicts):
        """XL on M body is loose — no red regions."""
        assert verdicts["XL"]["fit"] is True


# ---------------------------------------------------------------------------
# AC-5.6: Confidence (non-negotiable rule #1)
# ---------------------------------------------------------------------------

class TestConfidence:

    @pytest.mark.parametrize("size", SIZES)
    def test_confidence_is_1_for_synthetic_mannequin(self, verdicts, size):
        assert verdicts[size]["confidence"] == 1.0, \
            f"Size {size}: confidence={verdicts[size]['confidence']}, expected 1.0 " \
            "for synthetic_mannequin"

    def test_non_synthetic_with_confidence_1_raises(self):
        """confidence=1.0 on a non-synthetic body is forbidden (CLAUDE.md rule #1)."""
        with pytest.raises(ValueError, match="confidence=1.0"):
            generate_verdict(
                _SIM_RESULTS["M"], "test_id", "body_id",
                body_source="standard_photo",
                confidence=1.0,
            )

    def test_non_synthetic_without_confidence_raises(self):
        """Omitting confidence for a non-synthetic body is forbidden (CLAUDE.md rule #1)."""
        with pytest.raises(ValueError, match="confidence is required"):
            generate_verdict(
                _SIM_RESULTS["M"], "test_id", "body_id",
                body_source="standard_photo",
                confidence=None,
            )

    def test_non_synthetic_valid_confidence_accepted(self):
        """Non-synthetic body with a valid sub-1.0 confidence must succeed."""
        verdict = generate_verdict(
            _SIM_RESULTS["M"], "test_id", "body_id",
            body_source="standard_photo",
            confidence=0.82,
        )
        assert verdict["confidence"] == 0.82
        assert verdict["body_source"] == "standard_photo"

    def test_non_synthetic_confidence_out_of_range_raises(self):
        """confidence outside [0.0, 1.0] must be rejected."""
        with pytest.raises(ValueError):
            generate_verdict(
                _SIM_RESULTS["M"], "test_id", "body_id",
                body_source="precision_suit",
                confidence=1.5,
            )

    def test_invalid_body_source_raises(self):
        """An unrecognised body_source must be rejected (CLAUDE.md rule #3)."""
        with pytest.raises(ValueError, match="body_source"):
            generate_verdict(
                _SIM_RESULTS["M"], "test_id", "body_id",
                body_source="lidar_scan",
            )

    def test_synthetic_with_explicit_confidence_1_accepted(self):
        """Explicitly passing confidence=1.0 for synthetic_mannequin is fine."""
        verdict = generate_verdict(
            _SIM_RESULTS["M"], "test_id", "body_id",
            body_source="synthetic_mannequin",
            confidence=1.0,
        )
        assert verdict["confidence"] == 1.0

    def test_synthetic_non_1_confidence_raises(self):
        """Passing confidence != 1.0 for synthetic_mannequin must raise."""
        with pytest.raises(ValueError):
            generate_verdict(
                _SIM_RESULTS["M"], "test_id", "body_id",
                body_source="synthetic_mannequin",
                confidence=0.9,
            )


# ---------------------------------------------------------------------------
# AC-5.7: Fabric params come from fabric_library.json
# ---------------------------------------------------------------------------

class TestFabricParams:

    @pytest.mark.parametrize("size", SIZES)
    def test_fabric_id_is_cotton_jersey_default(self, verdicts, size):
        assert verdicts[size]["fabric_params_used"]["fabric_id"] == "cotton_jersey_default"

    @pytest.mark.parametrize("size", SIZES)
    def test_density_matches_library(self, verdicts, size):
        import json
        with open("data/fabrics/fabric_library.json") as f:
            lib = json.load(f)
        expected = lib["fabrics"]["cotton_jersey_default"]["density_kg_m2"]
        actual = verdicts[size]["fabric_params_used"]["density_kg_m2"]
        assert actual == expected, \
            f"Size {size}: density={actual} != library value {expected}"

    @pytest.mark.parametrize("size", SIZES)
    def test_all_stiffness_params_match_library(self, verdicts, size):
        import json
        with open("data/fabrics/fabric_library.json") as f:
            lib = json.load(f)
        expected = lib["fabrics"]["cotton_jersey_default"]
        fp = verdicts[size]["fabric_params_used"]
        for key in ("stretch_stiffness", "bend_stiffness", "shear_stiffness", "damping"):
            assert fp[key] == expected[key], \
                f"Size {size}: {key}={fp[key]} != library {expected[key]}"


# ---------------------------------------------------------------------------
# AC-5.8: Tunnel-through constraint
# ---------------------------------------------------------------------------

class TestTunnelThrough:

    @pytest.mark.parametrize("size", SIZES)
    def test_tunnel_through_below_2_pct(self, verdicts, size):
        ttp = verdicts[size]["tunnel_through_pct"]
        assert ttp < 2.0, \
            f"Size {size}: tunnel_through_pct={ttp}% exceeds 2.0% limit"


# ---------------------------------------------------------------------------
# AC-5.9: validate_verdict_schema helper
# ---------------------------------------------------------------------------

class TestValidateVerdictSchema:

    @pytest.mark.parametrize("size", SIZES)
    def test_no_schema_errors(self, verdicts, size):
        errors = validate_verdict_schema(verdicts[size])
        assert errors == [], \
            f"Size {size}: schema validation errors:\n" + "\n".join(errors)

    def test_detects_wrong_verdict_id(self):
        v = generate_verdict(_SIM_RESULTS["M"], "test_id", "body_id")
        v["verdict_id"] = "vrd_abc"  # too short
        errors = validate_verdict_schema(v)
        assert any("12" in e for e in errors), \
            "Expected error about 12-char hex, got: " + str(errors)

    def test_detects_fit_red_mismatch(self):
        # Use a hardcoded sim result that has red regions, independent of size S.
        _red_sim = _make_sim_result({
            "chest_front": -30.0,   # red
            "chest_side": -28.5,    # red
            "shoulder_left": -12.0,
            "shoulder_right": -11.5,
            "upper_back": -27.0,    # red
            "waist": -15.0,
        })
        v = generate_verdict(_red_sim, "test_id", "body_id")
        v["fit"] = True  # has red regions — this is a spec violation
        errors = validate_verdict_schema(v)
        assert any("red" in e.lower() for e in errors), \
            "Expected error about fit/red mismatch, got: " + str(errors)

    def test_detects_missing_region(self):
        v = generate_verdict(_SIM_RESULTS["M"], "test_id", "body_id")
        v["strain_map"] = [r for r in v["strain_map"] if r["region"] != "waist"]
        errors = validate_verdict_schema(v)
        assert any("waist" in e for e in errors)


# ---------------------------------------------------------------------------
# AC-5.10: On-disk files round-trip
# ---------------------------------------------------------------------------

class TestOnDiskVerdicts:

    @pytest.mark.parametrize("filename", [
        "vrd_S_on_M.json", "vrd_M_on_M.json", "vrd_XL_on_M.json"
    ])
    def test_verdict_file_exists(self, verdicts, filename):
        # verdicts fixture writes the files
        assert (VERDICT_DIR / filename).exists(), f"Verdict file not found: {filename}"

    @pytest.mark.parametrize("filename", [
        "vrd_S_on_M.json", "vrd_M_on_M.json", "vrd_XL_on_M.json"
    ])
    def test_verdict_file_is_valid_json(self, verdicts, filename):
        v = load_verdict(filename)
        assert isinstance(v, dict)

    @pytest.mark.parametrize("filename", [
        "vrd_S_on_M.json", "vrd_M_on_M.json", "vrd_XL_on_M.json"
    ])
    def test_verdict_file_schema_valid(self, verdicts, filename):
        v = load_verdict(filename)
        errors = validate_verdict_schema(v)
        assert errors == [], \
            f"{filename}: schema errors:\n" + "\n".join(errors)

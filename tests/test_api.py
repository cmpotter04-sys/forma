from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VALID_SIM_RESULT = {
    "clearance_map": {
        "chest_front": 5.0,
        "chest_side": 3.0,
        "shoulder_left": 8.0,
        "shoulder_right": 7.0,
        "upper_back": 6.0,
        "waist": 4.0,
    },
    "simulation_ms": 1200,
    "convergence_step": 45,
    "final_kinetic_energy_j": 0.002,
    "tunnel_through_pct": 0.1,
}

# Minimal valid body_mesh_path payload (no actual files needed — pipeline is mocked)
BASE_FIT_CHECK_PAYLOAD = {
    "body_mesh_path": "/fake/body.ply",
    "pattern_path": "/fake/pattern.json",
    "seam_manifest_path": "/fake/manifest.json",
}

MOCK_TARGET = "src.api.run_fit_check"


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_returns_200():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_body():
    resp = client.get("/health")
    data = resp.json()
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# GET /sizes
# ---------------------------------------------------------------------------

def test_sizes_returns_200():
    resp = client.get("/sizes")
    assert resp.status_code == 200


def test_sizes_has_male_and_female_keys():
    resp = client.get("/sizes")
    data = resp.json()
    assert "male" in data
    assert "female" in data


def test_sizes_male_contains_expected_sizes():
    resp = client.get("/sizes")
    male = resp.json()["male"]
    assert set(male.keys()) == {"S", "M", "XL"}


def test_sizes_female_contains_expected_sizes():
    resp = client.get("/sizes")
    female = resp.json()["female"]
    assert set(female.keys()) == {"XS", "S", "M", "L", "XL"}


# ---------------------------------------------------------------------------
# POST /fit-check — valid request (body_mesh_path)
# ---------------------------------------------------------------------------

def test_fit_check_valid_body_mesh_path_returns_200():
    with patch(MOCK_TARGET, return_value=VALID_SIM_RESULT):
        resp = client.post("/fit-check", json=BASE_FIT_CHECK_PAYLOAD)
    assert resp.status_code == 200


def test_fit_check_valid_body_mesh_path_returns_sim_result():
    with patch(MOCK_TARGET, return_value=VALID_SIM_RESULT):
        resp = client.post("/fit-check", json=BASE_FIT_CHECK_PAYLOAD)
    data = resp.json()
    assert "clearance_map" in data
    assert data["simulation_ms"] == 1200


def test_fit_check_passes_correct_args_to_pipeline():
    with patch(MOCK_TARGET, return_value=VALID_SIM_RESULT) as mock_run:
        client.post("/fit-check", json=BASE_FIT_CHECK_PAYLOAD)
    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["body_mesh_path"] == "/fake/body.ply"
    assert kwargs["pattern_path"] == "/fake/pattern.json"
    assert kwargs["seam_manifest_path"] == "/fake/manifest.json"
    assert kwargs["backend"] == "cpu"
    assert kwargs["fabric_id"] == "cotton_jersey_default"


# ---------------------------------------------------------------------------
# POST /fit-check — valid request (anny_measurements)
# ---------------------------------------------------------------------------

ANNY_PAYLOAD = {
    "anny_measurements": {
        "height_cm": 168.0,
        "chest_cm": 90.0,
        "waist_cm": 72.0,
        "hips_cm": 96.0,
        "inseam_cm": 76.0,
        "shoulder_width_cm": 38.0,
    },
    "pattern_path": "/fake/pattern.json",
    "seam_manifest_path": "/fake/manifest.json",
}


def test_fit_check_valid_anny_measurements_returns_200():
    with patch(MOCK_TARGET, return_value=VALID_SIM_RESULT):
        resp = client.post("/fit-check", json=ANNY_PAYLOAD)
    assert resp.status_code == 200


def test_fit_check_anny_passes_measurements_to_pipeline():
    with patch(MOCK_TARGET, return_value=VALID_SIM_RESULT) as mock_run:
        client.post("/fit-check", json=ANNY_PAYLOAD)
    _, kwargs = mock_run.call_args
    assert kwargs["body_mesh_path"] is None
    assert kwargs["anny_measurements"]["chest_cm"] == 90.0


# ---------------------------------------------------------------------------
# POST /fit-check — missing required fields → 422
# ---------------------------------------------------------------------------

def test_fit_check_missing_pattern_path_returns_422():
    payload = {
        "body_mesh_path": "/fake/body.ply",
        "seam_manifest_path": "/fake/manifest.json",
    }
    resp = client.post("/fit-check", json=payload)
    assert resp.status_code == 422


def test_fit_check_missing_seam_manifest_returns_422():
    payload = {
        "body_mesh_path": "/fake/body.ply",
        "pattern_path": "/fake/pattern.json",
    }
    resp = client.post("/fit-check", json=payload)
    assert resp.status_code == 422


def test_fit_check_missing_body_source_returns_422():
    payload = {
        "pattern_path": "/fake/pattern.json",
        "seam_manifest_path": "/fake/manifest.json",
    }
    resp = client.post("/fit-check", json=payload)
    assert resp.status_code == 422


def test_fit_check_empty_body_returns_422():
    resp = client.post("/fit-check", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /fit-check — mutual exclusion: both anny_measurements + body_mesh_path → 422
# ---------------------------------------------------------------------------

def test_fit_check_both_body_sources_returns_422():
    payload = {
        **BASE_FIT_CHECK_PAYLOAD,
        "anny_measurements": {
            "height_cm": 168.0,
            "chest_cm": 90.0,
            "waist_cm": 72.0,
            "hips_cm": 96.0,
            "inseam_cm": 76.0,
            "shoulder_width_cm": 38.0,
        },
    }
    resp = client.post("/fit-check", json=payload)
    assert resp.status_code == 422


def test_fit_check_both_body_sources_error_message():
    payload = {
        **BASE_FIT_CHECK_PAYLOAD,
        "anny_measurements": {
            "height_cm": 168.0,
            "chest_cm": 90.0,
            "waist_cm": 72.0,
            "hips_cm": 96.0,
            "inseam_cm": 76.0,
            "shoulder_width_cm": 38.0,
        },
    }
    resp = client.post("/fit-check", json=payload)
    body = resp.text
    assert "not both" in body.lower() or "422" in str(resp.status_code)


# ---------------------------------------------------------------------------
# POST /fit-check — pipeline errors surface as expected HTTP codes
# ---------------------------------------------------------------------------

def test_fit_check_file_not_found_returns_422():
    with patch(MOCK_TARGET, side_effect=FileNotFoundError("body not found")):
        resp = client.post("/fit-check", json=BASE_FIT_CHECK_PAYLOAD)
    assert resp.status_code == 422


def test_fit_check_value_error_returns_422():
    with patch(MOCK_TARGET, side_effect=ValueError("bad fabric_id")):
        resp = client.post("/fit-check", json=BASE_FIT_CHECK_PAYLOAD)
    assert resp.status_code == 422


def test_fit_check_import_error_returns_503():
    with patch(MOCK_TARGET, side_effect=ImportError("warp not installed")):
        resp = client.post("/fit-check", json=BASE_FIT_CHECK_PAYLOAD)
    assert resp.status_code == 503


def test_fit_check_runtime_error_returns_500():
    with patch(MOCK_TARGET, side_effect=RuntimeError("simulation exploded")):
        resp = client.post("/fit-check", json=BASE_FIT_CHECK_PAYLOAD)
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# POST /fit-check/quick — valid request
# ---------------------------------------------------------------------------

QUICK_PAYLOAD_MALE = {
    "size": "M",
    "gender": "male",
    "pattern_path": "/fake/pattern.json",
    "seam_manifest_path": "/fake/manifest.json",
}

QUICK_PAYLOAD_FEMALE = {
    "size": "S",
    "gender": "female",
    "pattern_path": "/fake/pattern.json",
    "seam_manifest_path": "/fake/manifest.json",
}


def test_quick_fit_check_male_returns_200():
    with patch(MOCK_TARGET, return_value=VALID_SIM_RESULT):
        resp = client.post("/fit-check/quick", json=QUICK_PAYLOAD_MALE)
    assert resp.status_code == 200


def test_quick_fit_check_female_returns_200():
    with patch(MOCK_TARGET, return_value=VALID_SIM_RESULT):
        resp = client.post("/fit-check/quick", json=QUICK_PAYLOAD_FEMALE)
    assert resp.status_code == 200


def test_quick_fit_check_returns_sim_result():
    with patch(MOCK_TARGET, return_value=VALID_SIM_RESULT):
        resp = client.post("/fit-check/quick", json=QUICK_PAYLOAD_MALE)
    assert resp.json()["simulation_ms"] == 1200


def test_quick_fit_check_passes_preset_measurements_to_pipeline():
    with patch(MOCK_TARGET, return_value=VALID_SIM_RESULT) as mock_run:
        client.post("/fit-check/quick", json=QUICK_PAYLOAD_MALE)
    _, kwargs = mock_run.call_args
    # Male M preset: height_cm=176, chest_cm=96 (from SIZE_PRESETS)
    assert kwargs["anny_measurements"]["height_cm"] == 176
    assert kwargs["anny_measurements"]["chest_cm"] == 96


def test_quick_fit_check_female_preset_resolved_correctly():
    with patch(MOCK_TARGET, return_value=VALID_SIM_RESULT) as mock_run:
        client.post("/fit-check/quick", json=QUICK_PAYLOAD_FEMALE)
    _, kwargs = mock_run.call_args
    # Female S preset: height_cm=163, chest_cm=84 (from SIZE_PRESETS_FEMALE)
    assert kwargs["anny_measurements"]["height_cm"] == 163
    assert kwargs["anny_measurements"]["chest_cm"] == 84


# ---------------------------------------------------------------------------
# POST /fit-check/quick — invalid size → 422
# ---------------------------------------------------------------------------

def test_quick_fit_check_unknown_male_size_returns_422():
    payload = {**QUICK_PAYLOAD_MALE, "size": "XXL"}
    resp = client.post("/fit-check/quick", json=payload)
    assert resp.status_code == 422


def test_quick_fit_check_unknown_female_size_returns_422():
    payload = {**QUICK_PAYLOAD_FEMALE, "size": "XXXS"}
    resp = client.post("/fit-check/quick", json=payload)
    assert resp.status_code == 422


def test_quick_fit_check_male_size_xs_invalid_returns_422():
    # XS exists for female but not male
    payload = {**QUICK_PAYLOAD_MALE, "size": "XS"}
    resp = client.post("/fit-check/quick", json=payload)
    assert resp.status_code == 422


def test_quick_fit_check_missing_gender_returns_422():
    payload = {
        "size": "M",
        "pattern_path": "/fake/pattern.json",
        "seam_manifest_path": "/fake/manifest.json",
    }
    resp = client.post("/fit-check/quick", json=payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /bodies
# ---------------------------------------------------------------------------

def test_bodies_returns_200():
    resp = client.get("/bodies")
    assert resp.status_code == 200


def test_bodies_has_count_and_list():
    resp = client.get("/bodies")
    data = resp.json()
    assert "bodies" in data
    assert "count" in data
    assert isinstance(data["bodies"], list)


def test_bodies_count_matches_list_length():
    resp = client.get("/bodies")
    data = resp.json()
    assert data["count"] == len(data["bodies"])


def test_bodies_has_six_mannequins():
    """Expect S/M/XL for both male and female = 6 profiles."""
    resp = client.get("/bodies")
    assert resp.json()["count"] == 6


def test_bodies_each_has_required_fields():
    resp = client.get("/bodies")
    required = {"body_profile_id", "mesh_path", "achieved_measurements", "vertex_count"}
    for body in resp.json()["bodies"]:
        missing = required - set(body.keys())
        assert not missing, f"Body {body.get('body_profile_id')} missing: {missing}"


# ---------------------------------------------------------------------------
# GET /garments
# ---------------------------------------------------------------------------

def test_garments_returns_200():
    resp = client.get("/garments")
    assert resp.status_code == 200


def test_garments_has_tshirt():
    data = client.get("/garments").json()
    assert "tshirt" in data["garments"]


def test_garments_tshirt_has_five_sizes():
    data = client.get("/garments").json()
    assert set(data["garments"]["tshirt"]["sizes"]) == {"XS", "S", "M", "L", "XL"}


def test_quick_fit_check_invalid_gender_returns_422():
    payload = {**QUICK_PAYLOAD_MALE, "gender": "nonbinary"}
    resp = client.post("/fit-check/quick", json=payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /fit-check/quick — pipeline errors
# ---------------------------------------------------------------------------

def test_quick_fit_check_file_not_found_returns_422():
    with patch(MOCK_TARGET, side_effect=FileNotFoundError("pattern missing")):
        resp = client.post("/fit-check/quick", json=QUICK_PAYLOAD_MALE)
    assert resp.status_code == 422


def test_quick_fit_check_import_error_returns_503():
    with patch(MOCK_TARGET, side_effect=ImportError("warp not available")):
        resp = client.post("/fit-check/quick", json=QUICK_PAYLOAD_MALE)
    assert resp.status_code == 503


def test_quick_fit_check_runtime_error_returns_500():
    with patch(MOCK_TARGET, side_effect=RuntimeError("sim diverged")):
        resp = client.post("/fit-check/quick", json=QUICK_PAYLOAD_MALE)
    assert resp.status_code == 500

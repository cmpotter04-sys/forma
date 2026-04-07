from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import generate_stage1_profile as profile_script


class TestStage1ProfileGeometryCounts:
    def test_baseline_counts_are_real_geometry(self):
        counts = profile_script._measure_geometry_counts(subdivide_target=0)

        assert counts["body_vertex_count"] == 21833
        assert counts["garment_vertex_count"] > 500

    def test_smoke_counts_respect_subdivision_target(self):
        counts = profile_script._measure_geometry_counts(
            subdivide_target=profile_script.SMOKE_SUBDIVIDE_TARGET
        )

        assert counts["body_vertex_count"] == 21833
        assert counts["garment_vertex_count"] >= profile_script.SMOKE_SUBDIVIDE_TARGET

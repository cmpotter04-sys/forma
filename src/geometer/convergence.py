"""
src/geometer/convergence.py

Energy-based XPBD convergence detection per FORMA_GEOMETER_SPEC.md.

Primary criterion : kinetic energy < threshold_j
Secondary check  : explosion detection (energy spike > 10×)
Tertiary fallback: vertex movement check (original spec criterion)
"""

from __future__ import annotations
import numpy as np


class SimulationExplosionError(Exception):
    """Raised when XPBD kinetic energy spikes unexpectedly (solver instability)."""


def check_convergence(
    velocities: np.ndarray,
    masses: np.ndarray,
    threshold_j: float = 1e-6,
    prev_energy: float | None = None,
) -> tuple[bool, float]:
    """
    Energy-based convergence check.

    Parameters
    ----------
    velocities  : (N, 3) float — per-vertex velocities (m/s)
    masses      : (N,)   float — per-vertex masses (kg)
    threshold_j : float        — convergence threshold in Joules
    prev_energy : float | None — previous step's kinetic energy

    Returns
    -------
    (converged: bool, kinetic_energy: float)

    Raises
    ------
    SimulationExplosionError if energy spikes more than 10× vs prev_energy
    """
    speed_sq = np.sum(velocities ** 2, axis=1)          # (N,)
    kinetic_energy = float(0.5 * np.dot(masses, speed_sq))

    if prev_energy is not None and kinetic_energy > prev_energy * 10.0:
        raise SimulationExplosionError(
            f"Energy spike: {prev_energy:.6f} → {kinetic_energy:.6f} J "
            f"(ratio {kinetic_energy/max(prev_energy, 1e-30):.1f}×)"
        )

    return kinetic_energy < threshold_j, kinetic_energy


def check_vertex_movement(
    positions: np.ndarray,
    prev_positions: np.ndarray,
    threshold_mm: float = 0.4,
    max_pct: float = 1.5,
) -> bool:
    """
    Secondary convergence check: original spec criterion.

    Returns True if fewer than max_pct% of vertices moved more than
    threshold_mm millimetres since the previous step.
    """
    movement_mm = np.linalg.norm(positions - prev_positions, axis=1) * 1000.0
    pct_moving = np.sum(movement_mm > threshold_mm) / max(len(movement_mm), 1) * 100.0
    return pct_moving < max_pct

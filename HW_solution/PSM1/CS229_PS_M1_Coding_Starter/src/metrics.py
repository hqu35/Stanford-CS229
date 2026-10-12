"""Provided evaluation metrics (do not modify for graded comparisons)."""

from __future__ import annotations

import numpy as np

from .data import component_means


def mode_metrics(samples: np.ndarray, radius: float = 0.35,
                 threshold: float = 0.01) -> dict:
    """Compute PS-M1 validity and eight mode-coverage values.

    p_k = fraction of ALL generated samples within radius of component k.
    A mode is covered if p_k >= threshold. 'validity' is fraction of samples
    near at least one component. Mode discs do not overlap for our defaults.
    """
    a = np.asarray(samples, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) == 0:
        raise ValueError("samples must be a nonempty (N,2) array")
    if not np.isfinite(a).all():
        raise ValueError("samples contain nonfinite entries")
    if radius <= 0 or not (0 < threshold <= 1):
        raise ValueError("invalid radius or threshold")
    means = component_means()
    dist = np.linalg.norm(a[:, None, :] - means[None, :, :], axis=2)
    fractions = (dist < radius).mean(axis=0)
    return {
        "p_k": [float(v) for v in fractions],
        "covered_modes": int(np.sum(fractions >= threshold)),
        "validity": float((dist.min(axis=1) < radius).mean()),
        "radius": float(radius),
        "coverage_threshold": float(threshold),
        "n_samples": int(len(a)),
    }

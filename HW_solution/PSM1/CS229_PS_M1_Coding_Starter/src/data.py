"""Provided loading and reference-distribution utilities (not graded)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def component_means(k: int = 8, radius: float = 2.0) -> np.ndarray:
    """Return the centers of equally spaced 2D Gaussian components."""
    angles = 2.0 * np.pi * np.arange(k, dtype=np.float64) / k
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=-1)


def read_split(name: str, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Read pre-generated observations and hidden component IDs (diagnostics only).

    Returns x:(N,2) float32, component:(N,) int64.
    Training must use only x, not component labels.
    """
    if name not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    with np.load(data_dir / f"{name}.npz", allow_pickle=False) as payload:
        x = payload["x"].copy()
        component = payload["component"].copy()
    if x.ndim != 2 or x.shape[1] != 2 or len(component) != len(x):
        raise ValueError(f"Malformed {name} split")
    return x, component

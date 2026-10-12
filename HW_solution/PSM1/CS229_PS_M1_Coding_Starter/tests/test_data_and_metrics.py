"""Supplied-infrastructure tests: should pass before student edits anything."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.data import DATA_DIR, component_means, read_split
from src.experiment_config import linear_betas, select_betas
from src.metrics import mode_metrics


def test_pre_generated_data_and_checksums():
    meta = json.loads((DATA_DIR / "manifest.json").read_text())
    assert meta["seed"] == 229
    for part, count in (("train", 50_000), ("val", 10_000)):
        x, y = read_split(part)
        assert x.shape == (count, 2)
        assert x.dtype == np.float32
        assert y.shape == (count,)
        assert np.isfinite(x).all()
        assert np.all((y >= 0) & (y <= 7))
        src = DATA_DIR / f"{part}.npz"
        assert hashlib.sha256(src.read_bytes()).hexdigest() == meta["files"][src.name]["sha256"]
    x_train, _ = read_split("train")
    x_val, _ = read_split("val")
    assert not np.array_equal(x_train[:100], x_val[:100])


def test_known_mixture_geometry():
    means = component_means()
    assert means.shape == (8, 2)
    assert np.allclose(np.linalg.norm(means, axis=1), 2.0)
    assert np.allclose(means[0], [2.0, 0.0])


def test_metrics_on_exact_centers():
    centers = component_means()
    m = mode_metrics(np.repeat(centers, 20, axis=0))
    assert m["covered_modes"] == 8
    assert m["validity"] == pytest.approx(1)
    assert m["p_k"] == pytest.approx([0.125] * 8)


def test_metrics_far_from_centers():
    m = mode_metrics(np.full((100, 2), 100.0))
    assert m["covered_modes"] == 0
    assert m["validity"] == 0


def test_linear_schedule_infrastructure():
    betas = linear_betas(200)
    assert betas.shape == (200,)
    assert betas[0] == pytest.approx(1e-4)
    assert betas[-1] == pytest.approx(0.05)
    assert np.all(np.diff(betas) > 0)
    assert np.allclose(select_betas("linear", 200), betas)

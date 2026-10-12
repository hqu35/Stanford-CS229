"""Public tests for the matched-noise ablation in Problem 5(d)."""

import numpy as np
import pytest

from src.experiment_config import linear_betas, matched_constant_betas, select_betas


def test_matched_constant_schedule_properties():
    linear = linear_betas()
    matched = matched_constant_betas(linear)
    assert isinstance(matched, np.ndarray)
    assert matched.shape == linear.shape
    assert np.isfinite(matched).all()
    assert np.all((matched > 0) & (matched < 1))
    assert np.max(matched) - np.min(matched) < 1e-12
    target = np.prod(1 - linear)
    actual = np.prod(1 - matched)
    assert actual == pytest.approx(target, rel=1e-10, abs=1e-12)
    np.testing.assert_allclose(matched, select_betas("constant-matched"), rtol=0, atol=0)

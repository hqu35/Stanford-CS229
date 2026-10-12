"""Public tests for Problems 4(a,b); no model implementation is supplied."""

import numpy as np
import pytest

from src.diffusion_math import build_schedule, posterior_parameters, sample_forward


def test_schedule_indexing_and_product():
    betas = np.array([0.1, 0.2, 0.3])
    original = betas.copy()
    sch = build_schedule(betas)
    assert len(sch.beta) == len(sch.alpha) == len(sch.alpha_bar) == 4
    np.testing.assert_allclose(sch.beta, [0, 0.1, 0.2, 0.3], rtol=0, atol=1e-12)
    np.testing.assert_allclose(sch.alpha, [1, 0.9, 0.8, 0.7], rtol=0, atol=1e-12)
    np.testing.assert_allclose(sch.alpha_bar, [1, 0.9, 0.72, 0.504], atol=1e-12)
    np.testing.assert_array_equal(betas, original)


@pytest.mark.parametrize("betas", [
    np.array([]), np.array([0.0, 0.1]), np.array([1.0]),
    np.array([-0.1]), np.array([np.nan]), np.array([np.inf]),
    np.array([[0.1, 0.2]]),
])
def test_schedule_rejects_invalid_values(betas):
    with pytest.raises(ValueError):
        build_schedule(betas)


def test_forward_batch_shape_and_reproducibility():
    sch = build_schedule(np.array([0.05, 0.1, 0.15, 0.2]))
    x0 = np.arange(15, dtype=np.float64).reshape(5, 3) / 10
    t = np.array([1, 2, 3, 4, 1], dtype=np.int64)
    xa, ea = sample_forward(x0, t, sch, np.random.default_rng(123))
    xb, eb = sample_forward(x0, t, sch, np.random.default_rng(123))
    assert xa.shape == ea.shape == x0.shape
    assert np.isfinite(xa).all()
    np.testing.assert_array_equal(xa, xb)
    np.testing.assert_array_equal(ea, eb)
    expect = np.sqrt(sch.alpha_bar[t])[:, None] * x0 + \
             np.sqrt(1 - sch.alpha_bar[t])[:, None] * ea
    np.testing.assert_allclose(xa, expect, rtol=1e-7, atol=1e-7)


def test_forward_monte_carlo_conditional_moments():
    sch = build_schedule(np.array([0.04, 0.07, 0.1, 0.12]))
    n = 20_000
    x0 = np.tile(np.array([0.8, -1.5]), (n, 1))
    x, _ = sample_forward(x0, np.full(n, 4), sch, np.random.default_rng(432))
    theory_mu = np.sqrt(sch.alpha_bar[4]) * x0[0]
    theory_var = 1 - sch.alpha_bar[4]
    np.testing.assert_allclose(x.mean(axis=0), theory_mu, atol=0.03)
    np.testing.assert_allclose(np.diag(np.cov(x.T)), theory_var, atol=0.04)
    assert abs(np.cov(x.T)[0, 1]) < 0.035


def test_posterior_against_independent_gaussian_precision_formula():
    sch = build_schedule(np.array([0.03, 0.1, 0.2, 0.25]))
    x0 = np.array([[0.2, 0.7], [-0.4, 1.2], [2.1, -2.0]])
    xt = np.array([[0.1, 0.8], [0.0, 0.5], [-1.0, 2.0]])
    t = np.array([2, 3, 4])
    mu, var = posterior_parameters(x0, xt, t, sch)
    assert mu.shape == (3, 2)
    assert var.shape == (3,)
    prior_var = 1 - sch.alpha_bar[t - 1]
    expected_var = 1 / (1 / prior_var + sch.alpha[t] / sch.beta[t])
    expected_mu = expected_var[:, None] * (
        np.sqrt(sch.alpha_bar[t - 1])[:, None] * x0 / prior_var[:, None]
        + np.sqrt(sch.alpha[t])[:, None] * xt / sch.beta[t, None]
    )
    np.testing.assert_allclose(var, expected_var, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(mu, expected_mu, rtol=1e-10, atol=1e-12)

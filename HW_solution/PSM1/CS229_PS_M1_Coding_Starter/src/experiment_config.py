"""Fixed protocol and schedule helpers.

The linear schedule is supplied infrastructure. The matched constant schedule
is a student exercise; derive and implement it using the constraint on alpha_bar_T.
"""

from __future__ import annotations

import numpy as np


def linear_betas(t_steps: int = 200, start: float = 1e-4,
                 end: float = 5e-2) -> np.ndarray:
    """Return beta_1,...,beta_T as a float64 array (without beta_0)."""
    if t_steps < 2 or not (0.0 < start <= end < 1.0):
        raise ValueError("Require T>=2 and 0 < start <= end < 1")
    return np.linspace(start, end, t_steps, dtype=np.float64)


def matched_constant_betas(baseline_betas: np.ndarray) -> np.ndarray:
    """Return constant betas matching baseline product prod(1-beta_t).

    Args:
        baseline_betas: shape (T,), values strictly between zero and one.
    Returns:
        shape (T,), identical entries, with matching terminal alpha_bar.

    Student task, Problem 5(d): derive the constant and implement it in a
    numerically stable manner. Prefer log-domain calculations when possible.
    """
    # *** START CODE HERE ***
    raise NotImplementedError("Problem 5(d): derive matched constant schedule")
    # *** END CODE HERE ***


def select_betas(kind: str, t_steps: int = 200, start: float = 1e-4,
                 end: float = 5e-2) -> np.ndarray:
    baseline = linear_betas(t_steps, start, end)
    if kind == "linear":
        return baseline
    if kind == "constant-matched":
        return matched_constant_betas(baseline)
    raise ValueError(f"Unknown schedule kind: {kind}")

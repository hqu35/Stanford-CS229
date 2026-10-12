"""Problem 4: forward diffusion and analytic Gaussian posterior (NumPy).

All schedule arrays use direct mathematical indexing:
  beta[0]=0, alpha[0]=1, alpha_bar[0]=1; beta[t] is beta_t for 1<=t<=T.

Student implementations belong in the marked regions. Do not change the
specified function signatures or returned shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DiffusionSchedule:
    beta: np.ndarray       # shape (T+1,), beta[0]=0
    alpha: np.ndarray      # shape (T+1,), alpha[0]=1
    alpha_bar: np.ndarray  # shape (T+1,), alpha_bar[0]=1


def build_schedule(betas: np.ndarray) -> DiffusionSchedule:
    """Construct a schedule from beta_1,...,beta_T.

    Args:
        betas: 1D array of length T with values strictly between 0 and 1.
    Returns:
        DiffusionSchedule with three float64 arrays of length T+1.

    Raise ValueError for non-1D, empty, nonfinite or out-of-range inputs.
    Do not mutate the input.
    """
    # *** START CODE HERE ***
    raise NotImplementedError("Problem 4(a): build_schedule")
    # *** END CODE HERE ***


def sample_forward(
    x0: np.ndarray,
    t: np.ndarray,
    schedule: DiffusionSchedule,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample x_t conditional on x_0 directly (without stepping the chain).

    Args:
        x0: (N,d) finite array, dtype float64/float32.
        t: (N,) integer array, each in [1,T], potentially heterogeneous.
        schedule: from build_schedule.
        rng: np.random.Generator passed explicitly for repeatability.
    Returns:
        xt: (N,d) sampled noisy observations.
        eps: (N,d) sampled N(0,I) noise used in the construction.

    Use direct sampling from the closed-form marginal, with independent noise
    for every observation and coordinate. No hidden RNG global state.
    """
    # *** START CODE HERE ***
    raise NotImplementedError("Problem 4(b): sample_forward")
    # *** END CODE HERE ***


def posterior_parameters(
    x0: np.ndarray,
    xt: np.ndarray,
    t: np.ndarray,
    schedule: DiffusionSchedule,
) -> tuple[np.ndarray, np.ndarray]:
    """Parameters of q(x_{t-1} | x_t, x_0), for t >= 2.

    Args:
        x0,xt: arrays of shape (N,d).
        t: (N,) integer array with each t in [2,T].
        schedule: from build_schedule.
    Returns:
        mean: (N,d) posterior means.
        variance: (N,) scalar posterior variances (covariance=var_i*I_d).

    Treat t=1 as invalid: the endpoint is a degenerate posterior.
    """
    # *** START CODE HERE ***
    raise NotImplementedError("Problem 4(b): posterior_parameters")
    # *** END CODE HERE ***

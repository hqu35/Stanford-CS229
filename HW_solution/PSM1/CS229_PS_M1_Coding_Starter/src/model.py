"""Problem 5(a): time-conditioned MLP predicting epsilon.

Use three hidden layers of width 128, with SiLU or GELU. A timestep embedding
is required. No ready-made diffusion model architectures are allowed.
"""

from __future__ import annotations

import torch
from torch import nn


class NoisePredictor(nn.Module):
    """A network (x_t, t) -> predicted epsilon.

    Args:
        hidden_dim: width of three hidden layers (baseline 128).
        time_dim: dimension of timestep features (baseline 32).

    forward:
        x: (B,2), float tensor.
        t: (B,), integer timesteps in [1,T] (or float convertible).
        output: (B,2), same device/dtype as x.

    Student task: choose/implement a timestep feature representation and
    the fully connected network. Document your choice in written report.
    """

    def __init__(self, hidden_dim: int = 128, time_dim: int = 32):
        super().__init__()
        # *** START CODE HERE ***
        raise NotImplementedError("Problem 5(a): construct time-conditioned MLP")
        # *** END CODE HERE ***

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # *** START CODE HERE ***
        raise NotImplementedError("Problem 5(a): NoisePredictor.forward")
        # *** END CODE HERE ***

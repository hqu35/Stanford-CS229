"""Public tests for Problems 5(a-c)."""

import torch
from torch import nn

from src.ddpm import ddpm_loss, reverse_mean, sample_ddpm
from src.model import NoisePredictor


def small_schedule():
    beta = torch.tensor([0.0, 0.03, 0.05, 0.08, 0.1], dtype=torch.float32)
    alpha = 1 - beta
    return beta, alpha, torch.cumprod(alpha, dim=0)


def test_noise_predictor_output_and_gradient():
    model = NoisePredictor(hidden_dim=16, time_dim=8)
    x = torch.randn(6, 2, requires_grad=True)
    t = torch.tensor([1, 2, 3, 4, 2, 1])
    y = model(x, t)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None
    assert any(p.grad is not None for p in model.parameters())


def test_ddpm_loss_scalar_and_autograd():
    _, _, alpha_bar = small_schedule()
    torch.manual_seed(77)
    class TimeIgnoringLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, x, t):
            return self.linear(x)

    model = TimeIgnoringLinear()
    x0 = torch.randn(64, 2)
    loss = ddpm_loss(model, x0, alpha_bar)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    assert model.linear.weight.grad is not None
    assert model.linear.weight.grad.shape == (2, 2)


def test_ddpm_loss_magnitude_for_zero_predictor():
    class ZeroModel(nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)
    _, _, alpha_bar = small_schedule()
    torch.manual_seed(7)
    # Expected E[||epsilon||_2^2] = 2, NOT 1 (coordinate-mean MSE).
    loss = ddpm_loss(ZeroModel(), torch.zeros(20_000, 2), alpha_bar)
    assert abs(float(loss) - 2) < 0.13


def test_reverse_mean_matches_gaussian_parameterization():
    beta, alpha, alpha_bar = small_schedule()
    xt = torch.tensor([[1.0, -2.0], [0.5, 0.75]])
    eps_hat = torch.tensor([[0.3, 0.7], [0.0, -0.2]])
    t = 3
    mu = reverse_mean(xt, t, eps_hat, beta, alpha, alpha_bar)
    expected = (xt - beta[t] / torch.sqrt(1 - alpha_bar[t]) * eps_hat) / \
               torch.sqrt(alpha[t])
    torch.testing.assert_close(mu, expected)


def test_sampler_returns_all_intermediate_states_in_order():
    class RecordingZeroModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []
        def forward(self, x, t):
            self.calls.append(t.detach().cpu().clone())
            return torch.zeros_like(x)

    beta, alpha, alpha_bar = small_schedule()
    model = RecordingZeroModel()
    model.train()
    torch.manual_seed(9)
    samples, states = sample_ddpm(
        model, 17, beta, alpha, alpha_bar,
        torch.device("cpu"), snapshot_steps=(4, 2, 0),
    )
    assert samples.shape == (17, 2)
    assert torch.isfinite(samples).all()
    assert not samples.requires_grad
    assert set(states) == {4, 2, 0}
    for s in states.values():
        assert s.shape == (17, 2) and torch.isfinite(s).all()
    assert len(model.calls) == 4
    for expected_t, call in zip((4, 3, 2, 1), model.calls):
        assert torch.all(call == expected_t)
    assert model.training  # Sampling must not permanently change mode.

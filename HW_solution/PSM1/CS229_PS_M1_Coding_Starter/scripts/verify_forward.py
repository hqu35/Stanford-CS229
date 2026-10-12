#!/usr/bin/env python3
"""Problem 4(c): empirical forward moments and independent Markov-chain check.

Runs after student implementation of src/diffusion_math.py.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.diffusion_math import build_schedule, sample_forward  # noqa: E402
from src.experiment_config import linear_betas  # noqa: E402


def sequential_sample(x0: np.ndarray, step: int,
                      beta: np.ndarray, seed: int) -> np.ndarray:
    """Independent reference SIMULATOR of the one-step forward definition.

    This verifies the student's direct sampling against an entirely different
    implementation path; it does not implement their target functions.
    """
    rng = np.random.default_rng(seed)
    x = np.repeat(x0[None, :], repeats=10_000, axis=0).astype(np.float64)
    for s in range(1, step + 1):
        x = np.sqrt(1 - beta[s]) * x + np.sqrt(beta[s]) * rng.normal(size=x.shape)
    return x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=50_000)
    p.add_argument("--output", type=Path, default=ROOT / "results" / "forward")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x0 = np.array([1.0, -1.0], dtype=np.float64)
    schedule = build_schedule(linear_betas())
    rows = []
    steps = [1, 50, 100, 200]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
    for ax, t in zip(axes, steps):
        n = args.num_samples
        xt, _ = sample_forward(
            np.repeat(x0[None, :], repeats=n, axis=0),
            np.full(n, t, dtype=np.int64), schedule,
            rng=np.random.default_rng(229 + t),
        )
        empirical_mean = xt.mean(axis=0)
        empirical_cov = np.cov(xt.T)
        theoretical_mean = np.sqrt(schedule.alpha_bar[t]) * x0
        theoretical_cov = (1 - schedule.alpha_bar[t]) * np.eye(2)
        rows.append({
            "t": t,
            "mean_theory_0": theoretical_mean[0],
            "mean_theory_1": theoretical_mean[1],
            "mean_empirical_0": empirical_mean[0],
            "mean_empirical_1": empirical_mean[1],
            "var_theory": theoretical_cov[0, 0],
            "var_empirical_0": empirical_cov[0, 0],
            "var_empirical_1": empirical_cov[1, 1],
            "cov_empirical_offdiag": empirical_cov[0, 1],
            "mean_error_l2": np.linalg.norm(empirical_mean - theoretical_mean),
            "cov_error_fro": np.linalg.norm(empirical_cov - theoretical_cov, ord="fro"),
        })
        ax.scatter(xt[:2000, 0], xt[:2000, 1], s=2, alpha=0.3)
        ax.scatter([theoretical_mean[0]], [theoretical_mean[1]], marker="x", s=70)
        ax.set_title(f"t={t}")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Direct forward diffusion at selected timesteps")
    fig.tight_layout()
    fig.savefig(args.output / "forward_scatter.png", dpi=140)
    plt.close(fig)
    with (args.output / "forward_moments.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    chosen_t = 100
    direct, _ = sample_forward(
        np.repeat(x0[None, :], repeats=10_000, axis=0),
        np.full(10_000, chosen_t), schedule,
        rng=np.random.default_rng(57),
    )
    chain = sequential_sample(x0, chosen_t, schedule.beta, seed=58)
    delta_mean = float(np.linalg.norm(direct.mean(0) - chain.mean(0)))
    delta_cov = float(np.linalg.norm(np.cov(direct.T) - np.cov(chain.T), ord="fro"))
    print(f"Saved: {args.output}")
    for r in rows:
        print(f"t={r['t']:>3}: mean L2 error={r['mean_error_l2']:.5f}, "
              f"cov Frobenius error={r['cov_error_fro']:.5f}")
    print(f"Direct vs sequential at t={chosen_t}: mean L2={delta_mean:.5f}, "
          f"cov Frobenius={delta_cov:.5f}")
    print("Discuss sampling uncertainty; agreement is empirical, not a proof.")


if __name__ == "__main__":
    main()

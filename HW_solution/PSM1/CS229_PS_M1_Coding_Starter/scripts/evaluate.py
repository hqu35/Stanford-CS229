#!/usr/bin/env python3
"""Evaluate a trained student DDPM using held-out data and fixed metrics.

This script supplies the evaluation protocol; it does not implement the
student's diffusion sampling or reverse mean formulas.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import read_split  # noqa: E402
from src.ddpm import sample_ddpm  # noqa: E402
from src.metrics import mode_metrics  # noqa: E402
from src.model import NoisePredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--n-samples", type=int, default=4_000)
    p.add_argument("--val-limit", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=2_048)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def denoising_error(model: NoisePredictor, val: torch.Tensor,
                    alpha_bar: torch.Tensor, timestep: int | None,
                    batch_size: int, seed: int) -> float:
    """Mean-over-data of SUM-over-coordinates squared prediction error.

    Use the same random-seed protocol across schedules; random noise is not
    shared between different evaluation timesteps, but seed is deterministic.
    """
    total = 0.0
    count = 0
    generator = torch.Generator(device=val.device).manual_seed(seed)
    T = len(alpha_bar) - 1
    with torch.no_grad():
        for x0 in torch.split(val, batch_size):
            if timestep is None:
                t = torch.randint(1, T + 1, (len(x0),),
                                  device=val.device, generator=generator)
            else:
                t = torch.full((len(x0),), timestep, dtype=torch.long,
                               device=val.device)
            eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype,
                              generator=generator)
            abar = alpha_bar[t].unsqueeze(1)
            xt = abar.sqrt() * x0 + (1 - abar).sqrt() * eps
            eps_hat = model(xt, t)
            if eps_hat.shape != eps.shape:
                raise ValueError("Noise predictor must return a (B,2) tensor")
            total += float(((eps - eps_hat) ** 2).sum(dim=1).sum().cpu())
            count += len(x0)
    return total / count


def scatter(ax, points: np.ndarray, title: str) -> None:
    ax.scatter(points[:, 0], points[:, 1], alpha=0.4, s=3)
    ax.set_title(title)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def main() -> None:
    args = parse_args()
    if args.n_samples < 1 or args.val_limit < 1 or args.batch_size < 1:
        raise ValueError("sample count, validation limit and batch size must be positive")
    torch.set_num_threads(args.threads)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA not available")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = NoisePredictor(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    beta = checkpoint["beta"].to(device=device, dtype=torch.float32)
    T = len(beta) - 1
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    assert abs(float(alpha_bar[0]) - 1.0) < 1e-6
    val_np, _ = read_split("val")
    val = torch.from_numpy(val_np[:args.val_limit]).to(device)
    steps = sorted(set([1, max(1, T // 4), max(1, T // 2), T]))
    results = {}
    for t in steps:
        results[str(t)] = denoising_error(model, val, alpha_bar, t,
                                           args.batch_size, seed=args.seed + t)
    uniform = denoising_error(model, val, alpha_bar, None,
                              args.batch_size, seed=args.seed + 999)
    print("Validation noise-prediction MSE (sum of coordinate errors):")
    for t in steps:
        print(f"  t={t:<4} {results[str(t)]:.6f}")
    print(f"  uniform: {uniform:.6f}")

    # Use one designated seed for all reverse sampling comparisons.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    snapshots_requested = sorted(set([T, int(0.75 * T), T // 2, T // 4, 0]))
    samples, snapshots = sample_ddpm(
        model, args.n_samples, beta, alpha, alpha_bar, device,
        snapshot_steps=snapshots_requested,
    )
    generated = samples.detach().cpu().numpy()
    snapshots_np = {int(t): arr.detach().cpu().numpy()
                    for t, arr in snapshots.items()}
    if set(snapshots_np) != set(snapshots_requested):
        raise ValueError("Sampler did not return all requested trajectory snapshots")
    metrics = mode_metrics(generated)
    out = args.output or args.checkpoint.parent
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "generated.npz", samples=generated,
                        **{f"snapshot_{t}": arr for t, arr in snapshots_np.items()})

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    scatter(axs[0], val_np[:min(args.n_samples, len(val_np))], "Held-out target samples")
    scatter(axs[1], generated, "DDPM generated samples")
    fig.tight_layout()
    fig.savefig(out / "generated_samples.png", dpi=145)
    plt.close(fig)

    fig, axs = plt.subplots(1, len(snapshots_requested),
                            figsize=(4 * len(snapshots_requested), 4))
    for ax, t in zip(axs, sorted(snapshots_requested, reverse=True)):
        scatter(ax, snapshots_np[t], f"Reverse process: t={t}")
    fig.tight_layout()
    fig.savefig(out / "reverse_trajectory.png", dpi=145)
    plt.close(fig)

    report = {
        "checkpoint": str(args.checkpoint),
        "metadata": checkpoint["metadata"],
        "beta_terminal": float(beta[-1].cpu()),
        "alpha_bar_terminal": float(alpha_bar[-1].cpu()),
        "validation_n": len(val),
        "validation_loss_at_t": results,
        "validation_loss_uniform": uniform,
        "sample_metrics": metrics,
        "sample_seed": args.seed,
        "sampling_convention": "posterior variance for t>=2; deterministic at t=1",
    }
    (out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "validation_by_t.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "denoising_error_sum_of_squared_coordinates"])
        writer.writerows([[t, results[str(t)]] for t in steps])
    print("Samples:", metrics)
    print(f"Saved evaluation plots + JSON + sample arrays in {out}")


if __name__ == "__main__":
    main()

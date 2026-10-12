#!/usr/bin/env python3
"""Supplied training driver for Problem 5 (student writes the model and DDPM).

Examples:
    python scripts/train.py --run-name linear --schedule linear
    python scripts/train.py --run-name matched --schedule constant-matched
    python scripts/train.py --run-name smoke --updates 2 --batch-size 16
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import read_split  # noqa: E402
from src.ddpm import ddpm_loss  # noqa: E402
from src.diffusion_math import build_schedule  # noqa: E402
from src.experiment_config import select_betas  # noqa: E402
from src.model import NoisePredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="linear")
    p.add_argument("--schedule", choices=["linear", "constant-matched"], default="linear")
    p.add_argument("--updates", type=int, default=12_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--time-dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=229)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--output", type=Path, default=ROOT / "results")
    return p.parse_args()


def select_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(arg)


def main() -> None:
    args = parse_args()
    if not args.run_name or not all(c.isalnum() or c in "_-" for c in args.run_name):
        raise ValueError("--run-name may contain only letters, numbers, '_' and '-'")
    if args.updates < 1 or args.batch_size < 1 or args.threads < 1:
        raise ValueError("updates, batch size and thread count must be positive")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = select_device(args.device)

    x_np, _ = read_split("train")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_np)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    if len(loader) == 0:
        raise ValueError("batch size larger than training dataset")
    beta_np = select_betas(args.schedule, args.T)
    sched = build_schedule(beta_np)
    alpha_bar = torch.tensor(sched.alpha_bar, dtype=torch.float32, device=device)
    model = NoisePredictor(hidden_dim=args.hidden_dim, time_dim=args.time_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    run_dir = args.output / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "loss.csv"
    print(f"Device: {device}; schedule: {args.schedule}; "
          f"terminal alpha_bar: {sched.alpha_bar[-1]:.8f}; "
          f"updates: {args.updates}")
    step = 0
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["update", "train_loss"])
        writer.writeheader()
        while step < args.updates:
            for (x0,) in loader:
                x0 = x0.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = ddpm_loss(model, x0, alpha_bar)
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise ValueError("ddpm_loss must return a finite scalar")
                loss.backward()
                optimizer.step()
                step += 1
                writer.writerow({"update": step, "train_loss": float(loss.detach().cpu())})
                if step == 1 or step % args.log_every == 0 or step == args.updates:
                    print(f"step {step:>6}/{args.updates}: loss={float(loss.detach()):.6f}")
                if step >= args.updates:
                    break
    ckpt = {
        "model_state_dict": model.cpu().state_dict(),
        "model_kwargs": {"hidden_dim": args.hidden_dim, "time_dim": args.time_dim},
        "beta": torch.from_numpy(sched.beta).float().cpu(),
        "metadata": {
            "run_name": args.run_name,
            "schedule": args.schedule,
            "updates": args.updates,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": args.seed,
            "T": args.T,
            "training_device": str(device),
            "terminal_alpha_bar": float(sched.alpha_bar[-1]),
        },
    }
    torch.save(ckpt, run_dir / "checkpoint.pt")
    (run_dir / "train_metadata.json").write_text(
        json.dumps(ckpt["metadata"], indent=2) + "\n")
    print(f"Saved checkpoint and loss log to {run_dir}")


if __name__ == "__main__":
    main()

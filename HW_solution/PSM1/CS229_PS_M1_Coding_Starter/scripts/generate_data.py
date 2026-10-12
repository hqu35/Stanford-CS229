#!/usr/bin/env python3
"""Regenerate the supplied eight-component mixture, deterministically.

From repository root: python scripts/generate_data.py
The RNGs for train and validation are independently spawned from seed 229.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import component_means  # noqa: E402


def generate(n: int, rng: np.random.Generator, means: np.ndarray,
             std: float) -> tuple[np.ndarray, np.ndarray]:
    comp = rng.integers(0, len(means), size=n, dtype=np.int64)
    x = means[comp] + rng.normal(loc=0.0, scale=std, size=(n, 2))
    return x.astype(np.float32), comp


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=229)
    p.add_argument("--train-size", type=int, default=50_000)
    p.add_argument("--val-size", type=int, default=10_000)
    p.add_argument("--output", type=Path, default=ROOT / "data")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.output / "train.npz", args.output / "val.npz"]
    if not args.overwrite and any(x.exists() for x in paths):
        raise SystemExit("Data exists; pass --overwrite to regenerate")
    if args.train_size < 1 or args.val_size < 1:
        raise SystemExit("Both splits must have positive sizes")
    ss_train, ss_val = np.random.SeedSequence(args.seed).spawn(2)
    means = component_means(k=8, radius=2.0)
    for name, n, seq, path in zip(
        ("train", "val"), (args.train_size, args.val_size),
        (ss_train, ss_val), paths,
    ):
        x, comp = generate(n, np.random.default_rng(seq), means, std=0.1)
        np.savez_compressed(path, x=x, component=comp)
        print(f"{name}: {x.shape}, saved to {path}")
    manifest = {
        "dataset_name": "CS229-PS-M1-8gaussians",
        "generator": "scripts/generate_data.py",
        "seed": args.seed,
        "split_rng": "np.random.SeedSequence(seed).spawn(2)",
        "train_size": args.train_size,
        "val_size": args.val_size,
        "num_components": 8,
        "component_probability": 0.125,
        "center_radius": 2.0,
        "component_covariance": [[0.01, 0.0], [0.0, 0.01]],
        "feature_dtype": "float32",
        "labels": "component IDs provided for diagnostics; prohibit supervised use",
        "files": {path.name: {"sha256": sha256(path),
                              "bytes": path.stat().st_size} for path in paths},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Wrote manifest.json with SHA-256 checksums")


if __name__ == "__main__":
    main()

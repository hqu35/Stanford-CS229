#!/usr/bin/env python3
"""Compare outputs of evaluate.py for linear vs terminal-matched schedules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--linear", type=Path, required=True,
                   help="linear/evaluation.json")
    p.add_argument("--constant", type=Path, required=True,
                   help="constant-matched/evaluation.json")
    p.add_argument("--output", type=Path, default=Path("results/comparison"))
    args = p.parse_args()
    reports = [json.loads(x.read_text()) for x in (args.linear, args.constant)]
    names = ["linear", "constant-matched"]
    end0, end1 = [x["alpha_bar_terminal"] for x in reports]
    if abs(end0 - end1) > 1e-5:
        raise ValueError(f"Schedules not terminal-matched: {end0} vs {end1}")
    if reports[0]["validation_n"] != reports[1]["validation_n"]:
        raise ValueError("Different validation sizes: not a fair comparison")
    for r in reports:
        if r["sample_metrics"]["n_samples"] != reports[0]["sample_metrics"]["n_samples"]:
            raise ValueError("Different sample counts: not a fair comparison")
    args.output.mkdir(parents=True, exist_ok=True)
    steps = sorted(set(reports[0]["validation_loss_at_t"]) &
                   set(reports[1]["validation_loss_at_t"]), key=int)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, r in zip(names, reports):
        ax.plot([int(s) for s in steps],
                [r["validation_loss_at_t"][s] for s in steps],
                marker="o", label=name)
    ax.set(xlabel="timestep", ylabel="held-out noise prediction SSE",
           title="Comparison of denoising error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "validation_comparison.png", dpi=145)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ticks = np.arange(8)
    for shift, name, r in zip((-0.2, 0.2), names, reports):
        ax.bar(ticks + shift, r["sample_metrics"]["p_k"], width=0.4, label=name)
    ax.set(xticks=ticks, xlabel="Gaussian component k", ylabel="fraction of all samples",
           title="Generated mode coverage: per-component p_k")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "mode_comparison.png", dpi=145)
    plt.close(fig)

    print(f"Terminal alpha_bar: linear={end0:.8f}, constant={end1:.8f}")
    for name, r in zip(names, reports):
        m = r["sample_metrics"]
        print(f"{name}: validation uniform SSE={r['validation_loss_uniform']:.5f}, "
              f"covered modes={m['covered_modes']}/8, validity={m['validity']:.4f}")
    summary = {
        "terminal_alpha_bar_linear": end0,
        "terminal_alpha_bar_constant": end1,
        "linear": reports[0], "constant_matched": reports[1],
    }
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()

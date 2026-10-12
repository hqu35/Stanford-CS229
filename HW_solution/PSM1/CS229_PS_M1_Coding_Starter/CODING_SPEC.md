# CS229 PS-M1 — Coding specifications and experimental protocol

This document completes the numerical and dataset details intentionally
omitted from the main LaTeX assignment. **The five-question mathematics
assignment remains authoritative** on derivations and point allocations.

## General conventions

`config/baseline.json` records the baseline in one place for reference.
The executable scripts use equivalent CLI defaults; editing that JSON alone
does **not** change a training run. Use the CLI flags and record any changes.

Let `T = 200`. Both NumPy and PyTorch schedules use **length T+1** arrays,
with indices equal to mathematical timesteps:

- `beta[0]=0`, `alpha[0]=1`, `alpha_bar[0]=1`.
- For `1 <= t <= T`, `beta[t]=beta_t`.
- `alpha[t]=1-beta[t]` and `alpha_bar[t]=prod_{s=1}^t alpha[s]`.
- Sampled per-example timesteps are integers in **[1,T] inclusive**.
- Inputs are row batches `(B, d)` and timestep vectors are `(B,)`.
- This assignment uses `d=2` for data/model experiments. P4 NumPy
  implementations should handle any positive feature dimension `d`.
- For P4, all analytical arrays should be returned as NumPy arrays;
  schedules should use float64 for reliable moment calculations.
- For P5, PyTorch tensors must remain on the device of the input batch.

## Problem 4 — Numerical verification (15 points)

### P4(a) [3]: noise-schedule construction

**Student file:** `src/diffusion_math.py` → `build_schedule`.

Input: 1D array `betas` of length T, containing beta_1...beta_T.
Output: `DiffusionSchedule(beta, alpha, alpha_bar)`, each length T+1.

Reject empty, multidimensional, nonfinite or out-of-range beta inputs with
`ValueError`; require each beta_t strictly between 0 and 1.
Do not modify the input array.

### P4(b) [7]: direct sampling and posterior parameters

**Student file:** `src/diffusion_math.py` → `sample_forward`,
`posterior_parameters`.

`sample_forward(x0, t, schedule, rng)`:

- `x0` has shape `(N,d)`; `t` has shape `(N,)` and may contain **different**
  values for each row.
- Draw independent noise for every coordinate and every observation using
  the explicitly passed NumPy `Generator`.
- Return `(xt, eps)`, each shape `(N,d)`.
- Use the **closed-form** marginal from P1(a), not a `T`-step loop.

`posterior_parameters(x0, xt, t, schedule)`:

- `x0,xt` are `(N,d)`, timesteps `(N,)` with `t >= 2`.
- Return the Gaussian posterior mean `(N,d)` and scalar variance `(N,)`.
  Each posterior covariance is this scalar times the identity matrix.
- `t=1` is not valid for the nondegenerate intermediate posterior API;
  do **not** silently insert an epsilon to avoid a zero variance.

### P4(c) [5]: empirical tests

Run:

```bash
python scripts/verify_forward.py
```

The provided driver uses a linear schedule, `T=200`, beta from `0.0001` to
`0.05` inclusive, `x0=(1,-1)`, `50,000` independent direct samples per t,
and timesteps `t = 1, 50, 100, 200`. It saves an empirical-vs-theoretical
mean/covariance CSV and a scatter plot.

The driver **also** independently simulates the original one-step Gaussian
Markov transitions to compare the direct and sequential marginal
approximations at `t=100` (10,000 samples in each method).

Required discussion: both diagonal covariance entries, the off-diagonal
covariance entry, errors in the mean (L2 norm) and covariance (Frobenius
norm), and sampling uncertainty. A numerical agreement is not itself a proof.

## Problem 5 — A DDPM from scratch (25 points)

### Dataset (all splits supplied)

The exact reference density is a mixture of eight equally weighted Gaussian
components in 2D:

    q_data(x) = (1/8) * sum_{k=0}^7 Normal(x; m_k, 0.01 * I_2)
    m_k = (2*cos(2*pi*k/8), 2*sin(2*pi*k/8))

`data/train.npz` contains `x` of shape `(50_000,2)` float32.
`data/val.npz` contains a separately sampled `x` of shape `(10_000,2)`.
Both also contain the integer array `component` for optional **diagnostics**.
Do not use component labels as training targets or model inputs.

Generation uses NumPy `default_rng` initialized through
`SeedSequence(229).spawn(2)` so the two splits have independently spawned RNG
streams. `data/manifest.json` records exact file hashes and distribution
parameters. To reconstruct them, run:

```bash
python scripts/generate_data.py --overwrite
```

**Do not train on validation examples.**

### P5(a) [5]: epsilon prediction network

**Student file:** `src/model.py` → `NoisePredictor`.

Create an MLP with **three hidden layers of width 128**, activation SiLU
(or GELU), and an explicit timestep feature representation (e.g., sinusoidal
embedding). Use embedding dimension 32 in the baseline.

Inputs: noisy `x_t:(B,2)` and timesteps `t:(B,)`. Output: predicted Gaussian
noise `(B,2)` with no final activation constraint. Describe how your time
features are calculated and why they are necessary.

`NoisePredictor(hidden_dim=128,time_dim=32)` must also accept smaller widths
for public tests. Design it to save/load with the supplied checkpoint format.

### P5(b) [5]: stochastic denoising objective

**Student file:** `src/ddpm.py` → `ddpm_loss`.

For each example independently:

1. Sample an integer `t` uniformly from **1 through T (inclusive)**.
2. Sample fresh epsilon from `Normal(0,I_2)`.
3. Construct `x_t` using the closed-form forward conditional.
4. Predict epsilon via `model(x_t,t)`.
5. Calculate `mean_B sum_{j=1}^2 (epsilon_ij - predicted_ij)^2`.

**Loss convention:** sum of squared errors over the 2 coordinates, then mean
across batch. In this convention, an all-zero noise predictor has expected
loss 2, not 1. Do not inadvertently average across BOTH batch and
coordinates without reporting the conversion factor.

Use the provided `scripts/train.py` driver (Adam, lr=0.001, batch=256,
12,000 optimizer updates, seed 229). The driver saves `loss.csv` and
`checkpoint.pt` under `results/<run-name>/`.

### P5(c) [7]: sampling from learned reverse transitions

**Student file:** `src/ddpm.py` → `reverse_mean` and `sample_ddpm`.

- `reverse_mean` computes the learned Gaussian mean from `x_t`,
  `epsilon_theta(x_t,t)`, and the schedule for each scalar timestep.
- `sample_ddpm` starts with standard Gaussian samples at time T. Work
  backward in descending integer timesteps T,...,1.
- For `t >= 2`, sample using the **analytical posterior variance**
  `tilde_beta_t` from P2(a), and the model-predicted reverse mean.
- **For `t=1`, apply the learned reverse mean without additional noise.**
  This is the practical deterministic endpoint convention (and distinct
  from sampling a positive-variance Gaussian decoder for an ELBO).
- `sample_ddpm` returns `(samples, snapshots)`: generated samples
  `(N,2)` at t=0 plus a dictionary mapping requested timestep integers to
  `(N,2)` **CPU** tensors. Include requested initial t=T and final t=0
  snapshots. Preserve `model.training` state and disable gradients.

Suggested trajectory snapshots: `t=200,150,100,50,0`.

Run:

```bash
python scripts/evaluate.py --checkpoint results/linear/checkpoint.pt
```

The supplied evaluator saves `generated.npz`, `generated_samples.png`,
`reverse_trajectory.png`, `evaluation.json`, and `validation_by_t.csv`.
Use common axes for target versus generated scatter plots.

### P5(d) [8]: matched-terminal noise-schedule ablation

**Student file:** `src/experiment_config.py` → `matched_constant_betas`.

**Baseline schedule:** `T=200`, linearly spaced
`beta_1=1e-4, beta_T=0.05`.

**Comparison schedule:** `beta_t=beta_const` for every `t>=1`, and require
EXACTLY THE SAME terminal `alpha_bar_T` as the linear baseline (up to
floating-point error). Derive beta_const algebraically in your written
solutions, then implement it. Use numerically stable arithmetic if possible.

Use identical data, architecture, optimizer, learning rate, batch size,
update count, and random seed for each run. The input `t` refers to the
**integer index**; equal indices generally have different noise strengths
across the schedules.

Required evaluations (supplied scripts calculate the metrics):

- **Held-out denoising error** at t = 1, 50, 100, 200, plus a uniform-t
  average on held-out data. All use the **sum-over-coordinates** SSE
  convention described above.
- **Mode fractions:** `p_k = (# generated samples within Euclidean distance
  0.35 of m_k) / N`, for all eight k.
- **Covered modes:** number of k with `p_k >= 0.01`.
- **Sample validity:** fraction of generated samples within 0.35 of any mean.
- **Generation size:** 4,000 new samples per model, using a shared sample seed.

The discs of radius 0.35 do not overlap for this mixture, so the sum of p_k
is equal to sample validity. These metrics are intentionally imperfect:
proximity to a mode does not establish that the full density is learned.

Run `scripts/compare_schedules.py` on the two `evaluation.json` files to
obtain comparisons of losses, mode fractions and terminal alpha_bar values.

**Interpretation:** discuss the relationship between denoising error and
generative quality; identify possible mode dropping and limitations of a
single-run ablation. Equal terminal noise distributions do not imply equal
intermediate noise distributions.

## Experiments / required deliverables

Submit:

- Student-owned source files: `diffusion_math.py`, `model.py`, `ddpm.py`,
  `experiment_config.py` (plus any helper files you wrote).
- `results/forward/forward_moments.csv` and its figure.
- `results/linear` and `results/constant` training logs, evaluation JSON,
  generated scatter plots, reverse-trajectory figures.
- `results/comparison/comparison.json` and comparison figures.
- Written answers for the mathematics problems and the coding-analysis
  questions, using `REPORT_TEMPLATE.md` if helpful.

You do not need to submit regenerated data files if they exactly match the
provided checksums. Do not submit third-party model weights or copied
reference solutions.

## Testing protocol

`python scripts/preflight.py` and `pytest -q tests/test_data_and_metrics.py`
should pass on extraction.

`python -m pytest -q` runs tests for student TODOs and **will fail until
those are implemented**. After completing your code, the entire test suite
should pass. It exercises indexing, shapes, posterior precision matching,
numerical conditional moments, autograd, reverse-mean semantics, snapshot
capture, and the matched-terminal constraint. The tests are public diagnostics
only, not a complete correctness guarantee.

## Suggested references

- [Stanford CS229 official notes (2026), Chapter 14](https://cs229.stanford.edu/main_notes.pdf)
- [Stanford CS229 Spring 2026 course](https://cs229.stanford.edu/index.html-spr26)
- [Ho et al., Denoising Diffusion Probabilistic Models (2020)](https://arxiv.org/abs/2006.11239)
- [Independent Spring 2026 notes](https://github.com/az9713/cs229-machine-learning-notes)

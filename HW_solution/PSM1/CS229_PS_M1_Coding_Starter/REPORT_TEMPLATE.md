# CS229 PS-M1 — Coding report (Problems 4 and 5)

**Student:**  
**Environment:** Python / NumPy / PyTorch / CPU or GPU  
**Random seed:** 229 for training; state any additional seeds.  
**Code revision:** 

Keep the report focused on quantitative evidence and mathematical
interpretation; short answers are welcome when justified.

## P4(a,b): Implementation and representations [10 pts]

- Describe the indexing convention for `beta`, `alpha`, and `alpha_bar`.
- Give function signatures and array shapes, including heterogeneous `t`.
- Briefly justify the direct sampler and posterior parameter formulas.
- Explain safeguards for invalid schedules, t=1, and floating-point behavior.

## P4(c): Numerical verification [5 pts]

Paste or attach `results/forward/forward_moments.csv` and the scatter plot.

| t | Theory mean (2D) | Empirical mean | Theory var | Empirical var1/var2 | Empirical cov12 | Mean L2 error | Cov Fro error |
|---|---|---|---|---|---|---|---|
| 1 | | | | | | | |
| 50 | | | | | | | |
| 100 | | | | | | | |
| 200 | | | | | | | |

- Compare direct vs sequential marginal at t=100.
- Explain the source of Monte Carlo uncertainty.
- Explain why experiments do not replace mathematical proof.

## P5(a): Network architecture [5 pts]

- Layer types, number of hidden layers, hidden width, activation.
- Exact timestep representation, including its dimension and normalization.
- Why is time conditioning necessary? What is the output shape?

## P5(b): Training objective [5 pts]

- Define the implemented loss and coordinate/batch reduction convention.
- Show how a batch and timestep are sampled.
- Explain why the objective is an unbiased Monte Carlo estimator under
  the specified sampling distribution.
- Include training loss plot or relevant excerpts from `loss.csv`.

## P5(c): Sampling and generative quality [7 pts]

- Explain initial prior, intermediate posterior variance and final-step rule.
- Include target/generated 2D scatter plots on the same axes.
- Include snapshots at t=200,150,100,50,0.
- Did the model generate all eight modes? Discuss density shape and failures.

## P5(d): Controlled schedule ablation [8 pts]

- Derive the constant beta needed to match the baseline terminal alpha_bar.
- Verify and report both terminal alpha_bar values.
- Show table of held-out denoising error at t=1,50,100,200 and uniform t.

| Schedule | t=1 | t=50 | t=100 | t=200 | Uniform t |
|---|---:|---:|---:|---:|---:|
| Linear | | | | | |
| Constant matched | | | | | |

- Report generated sample metrics for each schedule:

| Schedule | p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7 | Covered modes | Validity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear | | | | | | | | | | |
| Constant matched | | | | | | | | | | |

- Insert comparison plots from `scripts/compare_schedules.py`.
- Do lower denoising losses imply better samples in your experiment?
- State at least two limitations or confounding factors (e.g., one seed,
  finite training budget, non-equivalent intermediate noise distributions).

## Test results and reproducibility

- `python scripts/preflight.py` output summary.
- `python -m pytest -q` output summary.
- CLI commands used to train and evaluate both models.
- Changes to baseline parameters, if any, with reasons.
- Any unresolved failures or anomalous results.

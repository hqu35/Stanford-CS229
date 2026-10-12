# Stanford CS229 — Modern Extensions, PS-M1 Coding Supplement

**Diffusion Models: Problems 4–5 (40 points)**  
*Independent study supplement, Autumn 2026. Not an official Stanford assignment.*

This package contains **everything required to begin the programming half** of
PS-M1: fixed train/validation data, a deterministic generator, data manifest,
student starter code, fixed experimental protocol, training/evaluation drivers,
public tests, and a report template. The analytic diffusion and neural-network
solutions are **intentionally not included**.

## Quick start

From the extracted project directory, create an environment with Python 3.10+
(3.13 was tested here):

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/preflight.py
```

The preflight check verifies dependency imports, data shapes, and file hashes;
**it is expected to pass before any student code is written**.

To run the supplied infrastructure tests without hitting unfinished functions:

```bash
python -m pytest -q tests/test_data_and_metrics.py
```

To work on the assignment:

1. Read [`CODING_SPEC.md`](CODING_SPEC.md), which supplies all numerical
   hyperparameters omitted from the mathematical LaTeX problem set.
2. Complete the `# *** START CODE HERE ***` regions in
   `src/diffusion_math.py` (**P4(a,b)**).
3. Run `python -m pytest -q tests/test_diffusion_math.py`, then
   `python scripts/verify_forward.py` (**P4(c)**).
4. Complete `src/model.py`, `src/ddpm.py`, and
   `src/experiment_config.py` (**P5(a–d)**).
5. Run `python -m pytest -q` and the training/evaluation pipeline below.
6. Submit the source files, figures, JSON/CSV metrics, and a completed
   [`REPORT_TEMPLATE.md`](REPORT_TEMPLATE.md) as a PDF or Markdown report.

**Before completing the TODOs**, many public tests will fail with
`NotImplementedError`. These failures are expected and are not environment bugs.

## Baseline experiment

After your implementation passes its unit tests:

```bash
# Baseline: linearly increasing beta, T = 200.
python scripts/train.py --run-name linear --schedule linear
python scripts/evaluate.py --checkpoint results/linear/checkpoint.pt

# Ablation: constant beta with matching terminal alpha_bar.
python scripts/train.py --run-name constant --schedule constant-matched
python scripts/evaluate.py --checkpoint results/constant/checkpoint.pt

# Quantitative side-by-side report and comparison plots.
python scripts/compare_schedules.py \
  --linear results/linear/evaluation.json \
  --constant results/constant/evaluation.json
```

For a **minimal runtime smoke test** after writing your functions (not a
substitute for the full 12,000-update experiment):

```bash
python scripts/train.py --run-name smoke --updates 2 --batch-size 16 --device cpu
python scripts/evaluate.py --checkpoint results/smoke/checkpoint.pt \
  --n-samples 32 --val-limit 64
```

The scripts detect CUDA automatically; pass `--device cpu` to force CPU.
Install a CUDA-specific PyTorch wheel through the [official PyTorch installation
instructions](https://pytorch.org/get-started/locally/) if desired. Results may
vary between CPU and GPU kernels.

## What is in the package?

```text
CS229_PS_M1_Coding_Starter/
├── README.md                     # this file
├── CODING_SPEC.md                # full numerical and interface requirements
├── REPORT_TEMPLATE.md            # required plots, metrics and discussion
├── requirements.txt
├── pyproject.toml                # pytest configuration
├── Makefile
├── config/baseline.json          # human-readable experiment defaults
├── data/
│   ├── train.npz                 # 50,000 training points
│   ├── val.npz                   # 10,000 held-out points
│   └── manifest.json             # precise distribution and SHA-256 hashes
├── src/
│   ├── diffusion_math.py         # STUDENT: schedules, sampling, posterior
│   ├── model.py                  # STUDENT: time-conditioned MLP
│   ├── ddpm.py                   # STUDENT: denoising loss / sampler
│   ├── experiment_config.py      # STUDENT: matched constant schedule
│   ├── data.py                   # PROVIDED: loading and target centers
│   └── metrics.py                # PROVIDED: fixed comparison metrics
├── scripts/
│   ├── generate_data.py          # rebuild exact same data
│   ├── preflight.py              # imports, checksums, shape checks
│   ├── verify_forward.py         # P4 Monte Carlo & sequential comparison
│   ├── train.py                  # P5 training driver
│   ├── evaluate.py               # P5 validation + sample plots
│   └── compare_schedules.py      # P5 controlled ablation report
├── tests/
│   ├── test_data_and_metrics.py  # infrastructure (passes immediately)
│   ├── test_diffusion_math.py     # student P4
│   ├── test_ddpm.py              # student P5
│   └── test_schedule_matching.py # student P5(d)
├── figures/                      # optional student-created figures
└── results/                      # created by experiment scripts
```

## Non-goals and limitations

- These are **synthetic tabular coordinates**, not MNIST/images; the intent is
  to isolate the probability and numerical methods.
- The data files are **not** pretrained-model checkpoints. All model training
  is done by you.
- Public tests are sanity checks rather than a formal proof or exhaustive
  hidden grader. Your mathematical write-up remains central.
- No private solutions or grading keys are included.

## Sources

The problem set is independently developed with reference to
[CS229 lecture notes (2026), Chapter 14](https://cs229.stanford.edu/main_notes.pdf),
[Spring 2026 CS229](https://cs229.stanford.edu/index.html-spr26), and
[Ho et al. (2020), DDPM](https://arxiv.org/abs/2006.11239).
This project is not affiliated with or endorsed by Stanford University.

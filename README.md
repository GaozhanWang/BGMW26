# Dividend RL

Numerical experiments accompanying **Reinforcement Learning for Optimal
Dividend Problem under Diffusion Model**, by Lihua Bai, Thejani Gamage,
Jin Ma, and Gaozhan Wang.

[Paper](https://arxiv.org/abs/2309.10242)


## Experiments

| Script | Experiment | Main outputs |
| --- | --- | --- |
| `fixed_policy.py` | Independent value/derivative networks under a frozen Gibbs policy | Reference comparison, error curves, per-seed metrics |
| `run_sensitivity.py` | Truncation horizon, network width, and initialization | Two-panel sensitivity figure, tables, raw predictions |
| `policy_learning.py` | Policy iteration, fixed-policy consistency, policy densities, wealth quantiles | Learned-policy checkpoint and numerical diagnostics |
| `legacy/run_constant.py` | Original constant-drift policy-iteration run | Historical loss/value figures and checkpoint |
| `legacy/run_state_dependent.py` | Original state-dependent policy-iteration run | Historical loss/value figures and checkpoint |

The two legacy entry points share one implementation. The two newer
policy-learning variants share `policy_learning.py`, with explicit profiles.
All generated filenames use general experiment names.

## Installation

Use Python 3.10 or later. A virtual environment is recommended:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows Command Prompt,
`.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate`
on macOS/Linux. Then install the core dependencies:

```bash
python -m pip install -r requirements.txt
```

The fixed-policy and sensitivity experiments run on a CPU and do not require
PyTorch. For policy learning and the legacy runners:

```bash
python -m pip install -r requirements-torch.txt
```

`requirements-reproduction.txt` records the numerical package versions used
for the included fixed-policy results. Their environment used Python 3.12.

## Fixed-policy validation

```bash
python fixed_policy.py
```

Results are written to `results/fixed_policy/`. The default is five
initialization seeds, with three hidden layers of width 64. The value network
uses `V(x) = x Q(x)` to enforce its zero boundary value. Both networks use
deterministic representation targets and are trained independently without a
consistency penalty. The infinite-horizon reference is used for evaluation.

For a smoke run:

```bash
python fixed_policy.py --quick --output-dir smoke_output/fixed_policy
```

## Sensitivity study

The included results can be replotted immediately:

```bash
python run_sensitivity.py --plot-only --output-dir results/sensitivity
```

To produce a fresh full run:

```bash
python run_sensitivity.py --output-dir results/new_run
```

The study compares horizons `0.2, 0.5, 1, 2` and widths `32, 64, 128`, using
five initialization seeds per width. Horizon errors are evaluated directly
from the representation targets; network fits are performed at `T=2`.
Panel (a) displays value truncation error only. Panel (b) displays derivative
RMSE means with sample-standard-deviation error bars. The CSVs retain all
computed metrics, including derivative truncation errors.

The default experiment is the full profile. `--quick` is a smoke profile.
Completed fits are reused only when their stored source/configuration hashes
match. `--plot-only` never trains or changes numerical metadata. The bundled
sensitivity data retain the hashes of the source versions that generated
them; the cleaned scripts have a documented rename history in


## Policy learning and trajectory diagnostics

Install the PyTorch dependencies before these commands:

```bash
python policy_learning.py --profile action_sampled --case constant
python policy_learning.py --profile action_sampled --case state_dependent
```

The default profile uses paired common-random-number finite-difference
derivative targets and a consistency penalty of 5. Training uses the
policy-averaged drift. The wealth diagnostic samples actions at each time
step. Default parameters include `c=5`, `T=2`, `dt=0.02`, `lambda=1`, and
action interval `[0,1]`.

The alternative profile preserves the original consistency experiment:

```bash
python policy_learning.py --profile consistency --case constant
```

That profile defaults to the original **uncorrected stopped-BEL target**.
It is retained for comparison and requires a boundary correction before it
can be identified with the killed-process representation in `fixed_policy.py`.
Use `--derivative-target crn_fd` to select the finite-difference alternative;
use `--consistency-weight 0` to disable the penalty. These are explicit changes
of experimental method and should be stated when reporting results.

For a short execution check:

```bash
python policy_learning.py --quick --device cpu --output-dir smoke_output/policy_learning
```

## Citation and license


The code is available under the custom **Academic Research-Only License** in
[LICENSE](LICENSE). Noncommercial academic research is permitted under its
terms. Commercial use and other uses outside those terms require separate
written authorization. Licensing and collaboration contact:
[Gaozhan Wang](mailto:gaozhanw@usc.edu).

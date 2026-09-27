# Historical policy-iteration experiments

These files preserve the uploaded **uncorrected stopped-BEL** implementation for reproducibility. They do not implement the corrected killed-semigroup estimator. The numerical training and simulation functions are unchanged; the changes consolidate duplicate code, add command-line options, and save outputs in a chosen directory.

Run from the repository root:

```bash
python legacy/run_constant.py --help
python legacy/run_constant.py --output-dir results/legacy_constant
python legacy/run_state_dependent.py --output-dir results/legacy_state_dependent
```

The help commands do not require PyTorch. Training requires NumPy and PyTorch; plotting also requires Matplotlib. Both CPU and CUDA execution are supported through `--device`.

## Preserved configurations

| Setting | Constant entrypoint | State-dependent entrypoint |
| --- | --- | --- |
| Original input | `pia_dividend_ruin_bel_const.py` | `pia_dividend_ruin_bel_x2.py` |
| Drift | `2` | `2*x**2 + 3` |
| Drift derivative | `0` | `4*x` |
| Volatility | `1` | `1` |
| Discount | `5` | `10` |
| Entropy parameter | `1` | `1` |
| Action interval | `[0, 1]` | `[0, 1]` |
| Horizon / step | `1 / 0.04` | `1 / 0.04` |
| Ruin threshold | `0` | `0` |
| Training initial state | Always `x0=1` | Always `x0=1` |
| Validation domain | `[0, 3]` | `[0, 3]` |
| PI iterations | `20` | `20` |
| Maximum evaluation / policy-fit epochs | `200 / 200` | `200 / 200` |
| Training batch / paths per state | `256 / 128` | `256 / 128` |
| Learning rates | `0.0008` | `0.0008` |
| Early-stopping patience / minimum change | `12 / 0.0001` | `12 / 0.0001` |
| Acceptance paths / base seed | `6000 / 2025` | `6000 / 2025` |
| Acceptance multiplier | `0.5` | `0.3` |
| Training seed | `123` | `17` |
| Networks | Three hidden tanh layers, width `64` | Three hidden tanh layers, width `64` |

These are the executable configurations in the uploaded files, not the manuscript's stated configurations. In particular, the state-dependent executable used a **quadratic** drift, although its class defaults specified a linear drift. The shared `DividendRuinSpec` retains the constant file's class defaults; both entrypoints explicitly override the coefficient functions and all experiment settings.

CLI overrides are recorded in `run_configuration.json`. A small execution check can use reduced training counts; such a run is not a paper result:

```bash
python legacy/run_constant.py --device cpu --pia-iters 1 --pe-epochs 1 --pi-epochs 1 --pe-batch 2 --pi-batch 2 --mc-paths 2 --eval-paths 8 --no-plots --output-dir results/legacy_check
```

## Files and outputs

- `policy_iteration.py`: shared historical networks, distribution helpers, simulator, evaluation, and training functions.
- `_entrypoint.py`: CLI configuration and output handling, with no training on import.
- `plots.py`: shared diagnostic plotting functions, with generic output names.
- `run_constant.py` and `run_state_dependent.py`: lightweight entrypoints.

Each run writes `run_configuration.json`, `training_history.json`, and `models.pt`. The history includes epoch curves, Monte Carlo acceptance values, and elapsed time. Non-finite history metrics are explicitly flagged and represented by JSON `null`. The checkpoint contains policy, value, and derivative state dictionaries plus configuration; the returned value and derivative networks may precede the final accepted policy update.

The generic plots are `value_loss.png`, `derivative_loss.png`, `policy_fit_loss.png`, `policy_improvement.png`, `policy_parameter.png`, `mean_dividend.png`, and `value_network.png`. The constant run also retains the original final-policy density and wealth-quantile plots as `policy_density.png` and `wealth_quantiles.png`. Additional deterministic training plots are now exported for the constant run; this does not alter training or the original subsequent random path draws. `--no-plots` skips plotting.

## Known issues retained in the historical numerical code

1. The derivative target applies an ordinary full-space BEL weight to a process stopped at ruin. It does not include the derivative of the killing event. This is not generally a valid derivative representation for the killed value function.
2. The tangent flow omits the derivative of the state-dependent policy. With state-dependent volatility, its stochastic integral also uses the newly updated tangent with the same Brownian increment.
3. The truncated-exponential mean formula suffers cancellation near a zero policy parameter in float32. The original narrow limiting branch is unchanged.
4. The simulator evaluates the reward at the current state but discounts it using the next grid time; a horizon that is not a multiple of the step is rounded upward. Ruin is monitored only at grid endpoints.
5. Best-epoch state dictionaries are stored without cloning, so the historical restoration does not recover an independent best-epoch snapshot.
6. Training occurs only at `x0=1`. Values or policy densities elsewhere are extrapolations from that training, despite validation over a wider interval.
7. There is no final policy-evaluation pass after the final improvement, and the outer iteration count is fixed. The acceptance statistic is a Monte Carlo heuristic, not a guarantee of monotonicity in the exact objective.
8. The quadratic state-dependent drift is unbounded and can produce exploding trajectories, overflow, or non-finite targets.

These retained limitations should be checked before treating the historical output as validated paper evidence. This folder is kept distinct from the corrected diagnostic and sensitivity experiments.

## Modification record

The two uploaded numerical cores were deduplicated. The constant source's function and class definitions through `run_policy_iteration` were preserved without numerical changes. The two original `__main__` configurations were moved into explicit CLI settings; import now has no experiment side effects. Plot filenames and a misleading final-evaluation title were clarified, and configuration/history/checkpoint exports were added. No derivative-estimator correction, coefficient substitution, optimizer change, or checkpoint-cloning fix was applied here.

Copyright (c) 2026 Gaozhan Wang. See the repository's `LICENSE`.

#!/usr/bin/env python3
# Copyright (c) 2026 Gaozhan Wang. All rights reserved.
# See LICENSE for permissions and conditions.
"""BEL validation and two-network consistency for the dividend/ruin model.

This standalone fixed-policy experiment is deliberately narrower than the full
policy-iteration code.  It freezes a state-independent Gibbs dividend policy
in the constant-coefficient model and compares two independently fitted neural
networks:

    * V_psi is fitted to the fixed-policy value representation;
    * P_eta is fitted to a boundary-aware BEL representation of its derivative.

The policy is

    pi_beta(a) = exp(beta * a) / int_0^A exp(beta * u) du,

and the killed state process is

    dX_t = b_bar dt + sigma dW_t,   tau = inf{t >= 0 : X_t <= 0},
    b_bar = mu - E_pi[A].

For y > 0, the killed transition density can be written as

    p_D(t,x,y) = g(t,x,y) h(t,x,y),
    h(t,x,y) = 1 - exp(-2*x*y/(sigma^2*t)),

where g is the unrestricted Gaussian transition density.  Consequently the
killed-semigroup BEL (likelihood-ratio) weight is

    N_D(t,x,y) = d/dx log p_D(t,x,y)
               = (y-x-b_bar*t)/(sigma^2*t)
                 + 2*y*exp(-2*x*y/(sigma^2*t))
                   / (sigma^2*t*h(t,x,y)).

This boundary-aware weight is essential.  The ordinary full-space weight
W_t/t does not account for differentiating the first-exit event and is not a
valid substitute near the ruin boundary.

The representation targets are evaluated by high-order quadrature, not by the
BVP.  The fixed-policy BVP has a separate closed-form solution and is used only
for evaluation.  In particular, the derivative network never sees the BVP
reference during training.

The script produces a two-panel diagnostic figure:

    (a) P_eta (BEL target) versus the fixed-policy BVP derivative;
    (b) |P_eta - V_x^ref| and |V_x^psi - V_x^ref|.

Example
-------
    python3 fixed_policy.py

Fast smoke test
---------------
    python3 fixed_policy.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor


@dataclass(frozen=True)
class ProblemConfig:
    """Constant-coefficient fixed-policy diagnostic."""

    mu: float = 3.0
    sigma: float = 1.0
    discount: float = 10.0
    lambda_reg: float = 1.0
    action_max: float = 1.0
    fixed_beta: float = 0.75
    horizon: float = 2.0
    x_train_lo: float = 1.0e-3
    x_train_hi: float = 1.25
    x_plot_lo: float = 0.05
    x_plot_hi: float = 1.00

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if self.discount <= 0.0 or self.lambda_reg <= 0.0:
            raise ValueError("discount and lambda_reg must be positive")
        if self.action_max <= 0.0 or self.horizon <= 0.0:
            raise ValueError("action_max and horizon must be positive")
        if not 0.0 < self.x_train_lo < self.x_plot_lo:
            raise ValueError("Require 0 < x_train_lo < x_plot_lo")
        if not self.x_plot_hi <= self.x_train_hi:
            raise ValueError("Require x_plot_hi <= x_train_hi")


@dataclass
class TrainingConfig:
    hidden_width: int = 64
    hidden_layers: int = 3
    training_points: int = 650
    quadrature_nodes: int = 420
    max_iterations: int = 3500
    replicates: int = 5
    plot_points: int = 300
    alpha: float = 1.0e-10
    tolerance: float = 1.0e-10
    base_seed: int = 1729

    @classmethod
    def quick(cls) -> "TrainingConfig":
        return cls(
            hidden_width=32,
            hidden_layers=2,
            training_points=220,
            quadrature_nodes=160,
            max_iterations=700,
            replicates=2,
            plot_points=160,
            tolerance=1.0e-8,
        )


def truncated_exponential_statistics(
    beta: float,
    action_max: float,
    lambda_reg: float,
) -> Tuple[float, float, float]:
    """Return E[A], log normalizer, and E[A] + lambda*entropy."""

    z = beta * action_max
    if abs(z) < 1.0e-6:
        z2 = z * z
        mean = action_max * (
            0.5 + z / 12.0 - z * z2 / 720.0 + z * z2 * z2 / 30240.0
        )
        log_normalizer = math.log(action_max) + (
            0.5 * z + z2 / 24.0 - z2 * z2 / 2880.0
        )
    else:
        exp_z = math.exp(z)
        mean = action_max * (exp_z / math.expm1(z) - 1.0 / z)
        log_normalizer = math.log(action_max) + math.log(math.expm1(z) / z)

    expected_log_density = beta * mean - log_normalizer
    regularized_reward = mean - lambda_reg * expected_log_density
    return mean, log_normalizer, regularized_reward


def fixed_policy_constants(cfg: ProblemConfig) -> Dict[str, float]:
    mean_action, log_normalizer, reward = truncated_exponential_statistics(
        cfg.fixed_beta,
        cfg.action_max,
        cfg.lambda_reg,
    )
    drift = cfg.mu - mean_action
    kappa = (
        drift
        + math.sqrt(drift * drift + 2.0 * cfg.discount * cfg.sigma**2)
    ) / cfg.sigma**2
    return {
        "mean_action": mean_action,
        "log_normalizer": log_normalizer,
        "regularized_reward": reward,
        "fixed_policy_drift": drift,
        "kappa": kappa,
    }


def killed_survival_and_bel_gradient(
    x: np.ndarray,
    t: np.ndarray,
    cfg: ProblemConfig,
    constants: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return P_x(tau>t) and its killed-semigroup BEL derivative.

    The derivative is the expectation of the boundary-aware score

        E[g_reward(X_t) 1_{tau>t} N_D(t,x,X_t)]

    with the reward removed.  In the constant-coefficient half-line model this
    expectation reduces to the closed form below.  Using the closed form only
    evaluates the BEL expectation efficiently; it does not use the BVP.
    """

    x_col = np.asarray(x, dtype=float).reshape(-1, 1)
    t_row = np.asarray(t, dtype=float).reshape(1, -1)
    sigma = cfg.sigma
    sigma2 = sigma * sigma
    drift = constants["fixed_policy_drift"]
    scale = sigma * np.sqrt(t_row)

    upper = (x_col + drift * t_row) / scale
    reflected = (-x_col + drift * t_row) / scale
    boundary_factor = np.exp(-2.0 * drift * x_col / sigma2)

    phi_upper = np.exp(-0.5 * np.square(upper)) / math.sqrt(2.0 * math.pi)
    phi_reflected = (
        np.exp(-0.5 * np.square(reflected)) / math.sqrt(2.0 * math.pi)
    )

    survival = ndtr(upper) - boundary_factor * ndtr(reflected)
    survival_gradient = (
        phi_upper / scale
        + (2.0 * drift / sigma2) * boundary_factor * ndtr(reflected)
        + boundary_factor * phi_reflected / scale
    )
    return np.clip(survival, 0.0, 1.0), survival_gradient


def representation_targets(
    x: np.ndarray,
    cfg: ProblemConfig,
    train: TrainingConfig,
    constants: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the value and killed-BEL derivative representations.

    Gauss--Legendre quadrature is applied after t = T*u^2.  The change of
    variables resolves the short-time boundary layer without placing a node at
    the singular endpoint t=0.
    """

    nodes, weights = leggauss(train.quadrature_nodes)
    u = 0.5 * (nodes + 1.0)
    u_weights = 0.5 * weights
    times = cfg.horizon * np.square(u)
    time_weights = u_weights * (2.0 * cfg.horizon * u)

    survival, survival_gradient = killed_survival_and_bel_gradient(
        x,
        times,
        cfg,
        constants,
    )
    discounted_weights = time_weights * np.exp(-cfg.discount * times)
    reward = constants["regularized_reward"]
    value = reward * (survival @ discounted_weights)
    derivative = reward * (survival_gradient @ discounted_weights)
    return value, derivative


def bvp_reference(
    x: np.ndarray,
    cfg: ProblemConfig,
    constants: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Infinite-horizon fixed-policy BVP solution and its derivative."""

    x_array = np.asarray(x, dtype=float)
    reward = constants["regularized_reward"]
    kappa = constants["kappa"]
    value = (reward / cfg.discount) * (1.0 - np.exp(-kappa * x_array))
    derivative = (
        reward * kappa / cfg.discount * np.exp(-kappa * x_array)
    )
    return value, derivative


def make_training_grid(cfg: ProblemConfig, train: TrainingConfig) -> np.ndarray:
    """Boundary-refined deterministic grid for the two regressions."""

    u = np.linspace(0.0, 1.0, train.training_points)
    boundary_refined = cfg.x_train_lo + (
        cfg.x_train_hi - cfg.x_train_lo
    ) * np.square(u)
    linear = np.linspace(cfg.x_train_lo, cfg.x_train_hi, train.training_points)
    return np.unique(np.concatenate((boundary_refined, linear)))


def scale_x(x: np.ndarray, cfg: ProblemConfig) -> np.ndarray:
    scaled = 2.0 * (np.asarray(x) - cfg.x_train_lo) / (
        cfg.x_train_hi - cfg.x_train_lo
    ) - 1.0
    return scaled.reshape(-1, 1)


def fit_network(
    x: np.ndarray,
    target: np.ndarray,
    cfg: ProblemConfig,
    train: TrainingConfig,
    seed: int,
) -> MLPRegressor:
    """Fit one tanh MLP with a deterministic quasi-Newton optimizer."""

    network = MLPRegressor(
        hidden_layer_sizes=(train.hidden_width,) * train.hidden_layers,
        activation="tanh",
        solver="lbfgs",
        alpha=train.alpha,
        tol=train.tolerance,
        max_iter=train.max_iterations,
        max_fun=max(15_000, 30 * train.max_iterations),
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        network.fit(scale_x(x, cfg), np.asarray(target, dtype=float))
    return network


def network_output_and_derivative(
    network: MLPRegressor,
    x: np.ndarray,
    cfg: ProblemConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a fitted tanh MLP and its exact input derivative."""

    activation = scale_x(x, cfg)
    derivative = np.full_like(
        activation,
        2.0 / (cfg.x_train_hi - cfg.x_train_lo),
    )

    for weights, bias in zip(network.coefs_[:-1], network.intercepts_[:-1]):
        preactivation = activation @ weights + bias
        activation = np.tanh(preactivation)
        derivative = (derivative @ weights) * (1.0 - np.square(activation))

    output = activation @ network.coefs_[-1] + network.intercepts_[-1]
    output_derivative = derivative @ network.coefs_[-1]
    return output[:, 0], output_derivative[:, 0]


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def mean_and_ci(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(samples, axis=0)
    if samples.shape[0] == 1:
        return mean, np.zeros_like(mean)
    standard_error = np.std(samples, axis=0, ddof=1) / math.sqrt(samples.shape[0])
    return mean, 1.96 * standard_error


def bootstrap_mean_interval(
    samples: np.ndarray,
    *,
    seed: int,
    resamples: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointwise percentile interval for a nonnegative mean curve."""

    mean = np.mean(samples, axis=0)
    if samples.shape[0] == 1:
        return mean, mean.copy(), mean.copy()
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        samples.shape[0],
        size=(resamples, samples.shape[0]),
    )
    bootstrap_means = np.mean(samples[indices], axis=1)
    lower, upper = np.percentile(bootstrap_means, (2.5, 97.5), axis=0)
    return mean, lower, upper


def run_experiment(
    cfg: ProblemConfig,
    train: TrainingConfig,
) -> Tuple[
    np.ndarray,
    Dict[str, np.ndarray],
    List[Dict[str, float]],
    Dict[str, float],
]:
    constants = fixed_policy_constants(cfg)
    training_grid = make_training_grid(cfg, train)
    value_target, bel_target = representation_targets(
        training_grid,
        cfg,
        train,
        constants,
    )

    # V(x)=x*Q(x) enforces V(0)=0.  Q has a finite boundary limit V_x(0+).
    value_quotient_target = value_target / training_grid

    plot_grid = np.linspace(cfg.x_plot_lo, cfg.x_plot_hi, train.plot_points)
    reference_value, reference_derivative = bvp_reference(
        plot_grid,
        cfg,
        constants,
    )
    target_value_plot, target_bel_plot = representation_targets(
        plot_grid,
        cfg,
        train,
        constants,
    )

    value_predictions: List[np.ndarray] = []
    autodiff_predictions: List[np.ndarray] = []
    bel_predictions: List[np.ndarray] = []
    metrics: List[Dict[str, float]] = []

    for replicate in range(train.replicates):
        seed = train.base_seed + 997 * replicate
        q_network = fit_network(
            training_grid,
            value_quotient_target,
            cfg,
            train,
            seed,
        )
        p_network = fit_network(
            training_grid,
            bel_target,
            cfg,
            train,
            seed + 100_003,
        )

        q_value, q_derivative = network_output_and_derivative(
            q_network,
            plot_grid,
            cfg,
        )
        value_prediction = plot_grid * q_value
        autodiff_prediction = q_value + plot_grid * q_derivative
        bel_prediction, _ = network_output_and_derivative(
            p_network,
            plot_grid,
            cfg,
        )

        value_predictions.append(value_prediction)
        autodiff_predictions.append(autodiff_prediction)
        bel_predictions.append(bel_prediction)
        metrics.append(
            {
                "seed": float(seed),
                "rms_P_minus_reference": rms(
                    bel_prediction - reference_derivative
                ),
                "max_P_minus_reference": float(
                    np.max(np.abs(bel_prediction - reference_derivative))
                ),
                "rms_dV_minus_reference": rms(
                    autodiff_prediction - reference_derivative
                ),
                "max_dV_minus_reference": float(
                    np.max(
                        np.abs(
                            autodiff_prediction - reference_derivative
                        )
                    )
                ),
                "rms_P_minus_dV": rms(
                    bel_prediction - autodiff_prediction
                ),
                "max_P_minus_dV": float(
                    np.max(np.abs(bel_prediction - autodiff_prediction))
                ),
                "rms_V_minus_reference": rms(
                    value_prediction - reference_value
                ),
            }
        )

    arrays = {
        "grid": plot_grid,
        "reference_value": reference_value,
        "reference_derivative": reference_derivative,
        "representation_value_target": target_value_plot,
        "representation_bel_target": target_bel_plot,
        "value": np.stack(value_predictions),
        "dV": np.stack(autodiff_predictions),
        "P": np.stack(bel_predictions),
    }
    return plot_grid, arrays, metrics, constants


def save_metrics_and_curves(
    output_dir: Path,
    cfg: ProblemConfig,
    train: TrainingConfig,
    arrays: Dict[str, np.ndarray],
    metrics: Sequence[Dict[str, float]],
    constants: Dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_fields = list(metrics[0].keys())
    with (output_dir / "consistency_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metrics)

    p_mean, p_ci = mean_and_ci(arrays["P"])
    dv_mean, dv_ci = mean_and_ci(arrays["dV"])
    gap_mean, gap_lower, gap_upper = bootstrap_mean_interval(
        np.abs(arrays["P"] - arrays["dV"]),
        seed=train.base_seed + 700_001,
    )
    p_error_mean, p_error_lower, p_error_upper = bootstrap_mean_interval(
        np.abs(arrays["P"] - arrays["reference_derivative"][None, :]),
        seed=train.base_seed + 700_002,
    )
    dv_error_mean, dv_error_lower, dv_error_upper = (
        bootstrap_mean_interval(
            np.abs(
                arrays["dV"] - arrays["reference_derivative"][None, :]
            ),
            seed=train.base_seed + 700_003,
        )
    )

    curve_fields = [
        "x",
        "reference_derivative",
        "bel_network_mean",
        "bel_network_ci95_half_width",
        "autodiff_mean",
        "autodiff_ci95_half_width",
        "abs_P_minus_dV_mean",
        "abs_P_minus_dV_bootstrap_ci95_lower",
        "abs_P_minus_dV_bootstrap_ci95_upper",
        "abs_P_minus_reference_mean",
        "abs_P_minus_reference_bootstrap_ci95_lower",
        "abs_P_minus_reference_bootstrap_ci95_upper",
        "abs_dV_minus_reference_mean",
        "abs_dV_minus_reference_bootstrap_ci95_lower",
        "abs_dV_minus_reference_bootstrap_ci95_upper",
    ]
    with (output_dir / "consistency_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_fields)
        writer.writeheader()
        for index, x_value in enumerate(arrays["grid"]):
            writer.writerow(
                {
                    "x": float(x_value),
                    "reference_derivative": float(
                        arrays["reference_derivative"][index]
                    ),
                    "bel_network_mean": float(p_mean[index]),
                    "bel_network_ci95_half_width": float(p_ci[index]),
                    "autodiff_mean": float(dv_mean[index]),
                    "autodiff_ci95_half_width": float(dv_ci[index]),
                    "abs_P_minus_dV_mean": float(gap_mean[index]),
                    "abs_P_minus_dV_bootstrap_ci95_lower": float(
                        gap_lower[index]
                    ),
                    "abs_P_minus_dV_bootstrap_ci95_upper": float(
                        gap_upper[index]
                    ),
                    "abs_P_minus_reference_mean": float(p_error_mean[index]),
                    "abs_P_minus_reference_bootstrap_ci95_lower": float(
                        p_error_lower[index]
                    ),
                    "abs_P_minus_reference_bootstrap_ci95_upper": float(
                        p_error_upper[index]
                    ),
                    "abs_dV_minus_reference_mean": float(
                        dv_error_mean[index]
                    ),
                    "abs_dV_minus_reference_bootstrap_ci95_lower": float(
                        dv_error_lower[index]
                    ),
                    "abs_dV_minus_reference_bootstrap_ci95_upper": float(
                        dv_error_upper[index]
                    ),
                }
            )

    summary: Dict[str, Dict[str, float]] = {}
    for key in metric_fields:
        if key == "seed":
            continue
        values = np.asarray([row[key] for row in metrics], dtype=float)
        summary[key] = {
            "mean": float(np.mean(values)),
            "sample_sd": (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            ),
        }

    target_rms = rms(
        arrays["representation_bel_target"] - arrays["reference_derivative"]
    )
    payload = {
        "experiment": (
            "constant-coefficient fixed Gibbs policy; boundary-aware "
            "killed-semigroup BEL target"
        ),
        "problem": asdict(cfg),
        "training": asdict(train),
        "fixed_policy_constants": constants,
        "replicate_metrics": list(metrics),
        "summary": summary,
        "representation_target_rms_minus_bvp": target_rms,
        "important_note": (
            "The derivative target uses the killed-semigroup BEL score. "
            "The ordinary full-space W_t/t kernel omits the derivative of "
            "the ruin event and is not used."
        ),
    }
    with (output_dir / "consistency_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)


def plot_figure(
    output_dir: Path,
    cfg: ProblemConfig,
    train: TrainingConfig,
    arrays: Dict[str, np.ndarray],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = arrays["grid"]
    reference = arrays["reference_derivative"]
    p_mean, p_ci = mean_and_ci(arrays["P"])

    p_error = np.abs(arrays["P"] - reference[None, :])
    dv_error = np.abs(arrays["dV"] - reference[None, :])
    p_error_mean, p_error_lower, p_error_upper = (
        bootstrap_mean_interval(
            p_error,
            seed=train.base_seed + 700_002,
        )
    )
    dv_error_mean, dv_error_lower, dv_error_upper = (
        bootstrap_mean_interval(
            dv_error,
            seed=train.base_seed + 700_003,
        )
    )

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 11,
            "legend.fontsize": 9.2,
            "figure.dpi": 140,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    ax = axes[0]
    ax.plot(
        grid,
        reference,
        color="black",
        linewidth=2.25,
        label=r"Fixed-policy BVP derivative $V_x^{\rm ref}$",
    )
    ax.plot(
        grid,
        p_mean,
        color="#1f77b4",
        linewidth=2.0,
        label=r"Independent network $P^{\eta}$",
    )
    ax.fill_between(
        grid,
        p_mean - p_ci,
        p_mean + p_ci,
        color="#1f77b4",
        alpha=0.20,
        label="95% CI across seeds",
    )
    ax.set_xlabel(r"Wealth $x$")
    ax.set_ylabel("Derivative")
    ax.set_title("(a) BEL derivative versus BVP benchmark")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=True)

    ax = axes[1]
    floor = 1.0e-8
    ax.plot(
        grid,
        np.maximum(p_error_mean, floor),
        color="#1f77b4",
        linewidth=2.0,
        label=r"$|P^{\eta}-V_x^{\rm ref}|$",
    )
    ax.fill_between(
        grid,
        np.maximum(p_error_lower, floor),
        np.maximum(p_error_upper, floor),
        color="#1f77b4",
        alpha=0.12,
    )
    ax.plot(
        grid,
        np.maximum(dv_error_mean, floor),
        color="#d62728",
        linewidth=2.0,
        linestyle="--",
        label=r"$|\partial_xV^{\psi}-V_x^{\rm ref}|$",
    )
    ax.fill_between(
        grid,
        np.maximum(dv_error_lower, floor),
        np.maximum(dv_error_upper, floor),
        color="#d62728",
        alpha=0.10,
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"Wealth $x$")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("(b) Errors relative to BVP benchmark")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper right", frameon=True)

    fig.suptitle(
        "BEL derivative validation and two-network consistency",
        y=1.015,
        fontsize=12.5,
    )
    fig.text(
        0.5,
        0.963,
        (
            rf"constant coefficients; fixed Gibbs policy $\beta={cfg.fixed_beta:g}$; "
            f"means across {train.replicates} independent training seeds; "
            "shading shows pointwise 95% intervals"
        ),
        ha="center",
        va="top",
        fontsize=9.7,
        color="0.30",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    fig.savefig(
        output_dir / "fixed_policy_consistency.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "fixed_policy_consistency.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fixed_policy"),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fixed-beta", type=float, default=0.75)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProblemConfig(fixed_beta=args.fixed_beta)
    train = TrainingConfig.quick() if args.quick else TrainingConfig()
    if args.replicates is not None:
        train.replicates = args.replicates
    if args.width is not None:
        train.hidden_width = args.width
    if args.layers is not None:
        train.hidden_layers = args.layers
    if train.replicates < 1:
        raise ValueError("replicates must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Small CPU fits are faster without BLAS oversubscription.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        grid, arrays, metrics, constants = run_experiment(cfg, train)
    del grid
    save_metrics_and_curves(
        args.output_dir,
        cfg,
        train,
        arrays,
        metrics,
        constants,
    )
    plot_figure(args.output_dir, cfg, train, arrays)

    print("Fixed-policy constants")
    for key, value in constants.items():
        print(f"  {key}: {value:.10g}")
    print("Replicate metrics")
    for row in metrics:
        print(
            f"  seed={int(row['seed'])}: "
            f"RMS(P-ref)={row['rms_P_minus_reference']:.3e}, "
            f"RMS(dV-ref)={row['rms_dV_minus_reference']:.3e}"
        )
    target_rms = rms(
        arrays["representation_bel_target"] - arrays["reference_derivative"]
    )
    print(f"BEL quadrature target RMS versus BVP: {target_rms:.3e}")
    print(f"Saved {args.output_dir / 'fixed_policy_consistency.png'}")
    print(f"Saved {args.output_dir / 'fixed_policy_consistency.pdf'}")
    print(f"Saved {args.output_dir / 'consistency_metrics.csv'}")
    print(f"Saved {args.output_dir / 'consistency_curves.csv'}")
    print(f"Saved {args.output_dir / 'consistency_summary.json'}")


if __name__ == "__main__":
    main()

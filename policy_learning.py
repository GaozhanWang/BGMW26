#!/usr/bin/env python3
"""Learn dividend policies and evaluate two-network policy consistency.

Training and policy evaluation use relaxed dynamics with the policy-averaged
control in the drift. Optional wealth plots instead sample one action at each
Euler step. These are different experiments.

Profiles preserve the numerical defaults of the two source implementations:
  action_sampled: paired CRN finite-difference targets; policy/wealth plots.
  consistency:   uncorrected BEL targets; no policy/wealth plots by default.
Both use c=5, lambda=1, T=2, dt=0.02 and consistency weight 5. The --quick
configuration remains a smoke test, not a publication configuration.

The ``bel`` mode preserves a whole-space BEL weight applied to killed paths.
It is an uncorrected diagnostic, not the killed-process score estimator in the
separate fixed-policy experiment. ``crn_fd`` uses paired finite differences of
the discretized killed return. Neither target is an exact analytical label.

The fixed-policy BVP is used only for evaluation. Source behavior is retained:
training evaluates the policy network outside its training interval, whereas
BVP and wealth diagnostics use constant endpoint continuation. Thus their
out-of-domain policy extensions differ; numerical cleanup does not resolve it.
Final-evaluation replicates share one frozen learned policy and are not
independent policy-iteration runs.

Examples:
  python policy_learning.py --quick
  python policy_learning.py --profile consistency --consistency-weight 0
  python policy_learning.py --load-policy results/policy_learning/action_sampled/final_policy.pt --diagnostics-only

Outputs default to results/policy_learning/<profile>.
"""
# Modification: consolidate two entry points, add explicit profiles, make help
# dependency-free, clarify estimator labels, and use generic output filenames.
# Numerical algorithms and configuration values are intentionally preserved.
# Copyright (c) 2026 Gaozhan Wang. See LICENSE for permissions and conditions.

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI options without importing numerical or machine-learning packages."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--profile", choices=("action_sampled", "consistency"), default="action_sampled",
        help="Default estimator and diagnostics profile (default: action_sampled).",
    )
    parser.add_argument(
        "--case", choices=("constant", "state_dependent"), default="state_dependent",
        help="Drift example (default: state_dependent).",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Results directory (default: results/policy_learning/<profile>).",
    )
    parser.add_argument("--quick", action="store_true", help="Use the original smoke-test configuration.")
    parser.add_argument(
        "--consistency-weight", type=float, default=None,
        help="One-way consistency penalty; 0 disables it (default: 5).",
    )
    parser.add_argument(
        "--derivative-target", choices=("bel", "crn_fd"), default=None,
        help=("Override the profile target: bel is an uncorrected whole-space BEL audit; "
              "crn_fd is a paired finite-difference target. Defaults: consistency=bel, "
              "action_sampled=crn_fd."),
    )
    parser.add_argument(
        "--replicates", type=int, default=None,
        help="Number of final-evaluation seeds under one frozen policy (default: 5; quick: 2).",
    )
    parser.add_argument(
        "--load-policy", type=Path, default=None,
        help="Load a final_policy.pt checkpoint instead of rerunning policy iteration.",
    )
    parser.add_argument(
        "--diagnostics-only", action="store_true",
        help="Generate policy and action-sampled wealth plots, skipping final network evaluation.",
    )
    parser.add_argument(
        "--wealth-diagnostics", action=argparse.BooleanOptionalAction, default=None,
        help="Enable policy and sampled-action wealth plots (default: action_sampled profile only).",
    )
    parser.add_argument("--policy-wealth-levels", type=float, nargs="+", default=(0.1, 0.5, 1.0, 1.5))
    parser.add_argument("--trajectory-initial-wealths", type=float, nargs="+", default=(0.1, 0.5, 1.0))
    parser.add_argument(
        "--trajectory-paths", type=int, default=None,
        help="Paths per initial wealth (default: 20000; quick: 3000).",
    )
    parser.add_argument("--trajectory-horizon", type=float, default=None, help="Plot horizon (default: policy-evaluation horizon).")
    parser.add_argument("--trajectory-seed", type=int, default=91_337)
    args = parser.parse_args(argv)
    if args.derivative_target is None:
        args.derivative_target = "bel" if args.profile == "consistency" else "crn_fd"
    if args.output_dir is None:
        args.output_dir = Path("results") / "policy_learning" / args.profile
    if args.diagnostics_only and args.wealth_diagnostics is False:
        parser.error("--diagnostics-only requires wealth diagnostics")
    if args.wealth_diagnostics is None:
        args.wealth_diagnostics = args.profile == "action_sampled" or args.diagnostics_only
    return args


# --help exits here, before torch/scipy/numpy imports are attempted.
_CLI_ARGS = parse_args() if __name__ == "__main__" else None

import copy
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_bvp


# -----------------------------------------------------------------------------
# Reproducibility and configuration
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@dataclass
class ProblemConfig:
    coefficient_case: str = "state_dependent"
    a_max: float = 1.0
    lambda_reg: float = 1.0
    discount: float = 5.0
    horizon: float = 2.0
    dt: float = 0.02
    ruin: float = 0.0
    beta_clip: float = 10.0
    x_train_lo: float = 0.05
    x_train_hi: float = 3.0
    x_plot_lo: float = 0.10
    x_plot_hi: float = 1.00
    x_reference_hi: float = 5.0

    def mu_torch(self, x: torch.Tensor) -> torch.Tensor:
        if self.coefficient_case == "constant":
            return torch.full_like(x, 3.0)
        if self.coefficient_case == "state_dependent":
            return 2.0 * x + 3.0
        raise ValueError(f"Unknown coefficient_case={self.coefficient_case!r}")

    def mu_prime_torch(self, x: torch.Tensor) -> torch.Tensor:
        if self.coefficient_case == "constant":
            return torch.zeros_like(x)
        if self.coefficient_case == "state_dependent":
            return torch.full_like(x, 2.0)
        raise ValueError(f"Unknown coefficient_case={self.coefficient_case!r}")

    def sigma_torch(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)

    def sigma_prime_torch(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def mu_numpy(self, x: np.ndarray) -> np.ndarray:
        if self.coefficient_case == "constant":
            return np.full_like(x, 3.0, dtype=float)
        if self.coefficient_case == "state_dependent":
            return 2.0 * np.asarray(x, dtype=float) + 3.0
        raise ValueError(f"Unknown coefficient_case={self.coefficient_case!r}")

    def sigma_numpy(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(np.asarray(x, dtype=float))


@dataclass
class TrainingConfig:
    width: int = 64
    hidden_layers: int = 3
    pia_iterations: int = 4
    pe_epochs: int = 70
    pe_batch_states: int = 40
    pe_paths_per_state: int = 80
    pe_lr: float = 8e-4
    pe_patience: int = 18
    pe_min_delta: float = 1e-6
    derivative_target_method: str = "crn_fd"
    crn_fd_step: float = 0.02
    boundary_sampling_fraction: float = 0.75
    boundary_sampling_hi: float = 1.20
    consistency_weight: float = 5.00
    pi_epochs: int = 180
    pi_batch_states: int = 256
    pi_lr: float = 8e-4
    pi_patience: int = 25
    final_eval_epochs: int = 240
    final_eval_batch_states: int = 96
    final_eval_paths_per_state: int = 256
    final_eval_replicates: int = 5
    final_eval_patience: int = 35
    final_dataset_states: int = 192
    final_dataset_paths: int = 2048
    final_validation_states: int = 96
    final_validation_paths: int = 4096
    final_dataset_chunk_states: int = 16
    final_optimizer_steps: int = 4000
    final_optimizer_batch: int = 96
    final_validation_every: int = 25
    final_validation_checks_patience: int = 60
    acceptance_paths: int = 12000
    acceptance_z: float = 0.50
    x0_acceptance: float = 1.0
    validation_states: int = 96
    validation_paths_per_state: int = 384
    plot_points: int = 240
    base_seed: int = 123

    @classmethod
    def quick(cls) -> "TrainingConfig":
        return cls(
            width=40,
            hidden_layers=3,
            pia_iterations=2,
            pe_epochs=24,
            pe_batch_states=24,
            pe_paths_per_state=48,
            pe_patience=8,
            pi_epochs=60,
            pi_batch_states=96,
            pi_patience=10,
            final_eval_epochs=45,
            final_eval_batch_states=32,
            final_eval_paths_per_state=64,
            final_eval_replicates=2,
            final_eval_patience=12,
            final_dataset_states=96,
            final_dataset_paths=512,
            final_validation_states=48,
            final_validation_paths=1024,
            final_dataset_chunk_states=12,
            final_optimizer_steps=1000,
            final_optimizer_batch=64,
            final_validation_every=25,
            final_validation_checks_patience=20,
            acceptance_paths=2000,
            validation_states=40,
            validation_paths_per_state=96,
            plot_points=160,
        )


# -----------------------------------------------------------------------------
# Stable truncated-exponential policy moments
# -----------------------------------------------------------------------------


def truncated_exponential_stats_torch(
    beta: torch.Tensor, a_max: float, lambda_reg: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return E[a], Var(a), log Z(beta), and E[a] + lambda H(pi).

    The series branch prevents the catastrophic cancellation in the earlier
    implementation around beta=0.  Since z=beta*a_max, differentiating log Z
    gives the mean and variance directly.
    """

    z = beta * a_max
    small = z.abs() < 1.0e-3
    z_safe = torch.where(small, torch.ones_like(z), z)
    exp_z = torch.exp(z_safe)
    expm1_z = torch.expm1(z_safe)

    log_z_general = math.log(a_max) + torch.log(torch.abs(expm1_z / z_safe))
    mean_general = a_max * (exp_z / expm1_z - 1.0 / z_safe)
    var_general = (a_max**2) * (
        1.0 / (z_safe**2) - exp_z / (expm1_z**2)
    )

    z2 = z * z
    log_z_series = (
        math.log(a_max)
        + 0.5 * z
        + z2 / 24.0
        - z2 * z2 / 2880.0
        + z2 * z2 * z2 / 181440.0
    )
    mean_series = a_max * (
        0.5 + z / 12.0 - z * z2 / 720.0 + z * z2 * z2 / 30240.0
    )
    var_series = (a_max**2) * (
        1.0 / 12.0 - z2 / 240.0 + z2 * z2 / 6048.0
    )

    log_z = torch.where(small, log_z_series, log_z_general)
    mean = torch.where(small, mean_series, mean_general)
    variance = torch.where(small, var_series, var_general).clamp_min(0.0)
    expected_log_pi = beta * mean - log_z
    reward = mean - lambda_reg * expected_log_pi
    return mean, variance, log_z, reward


def truncated_exponential_stats_numpy(
    beta: np.ndarray, a_max: float, lambda_reg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beta = np.asarray(beta, dtype=float)
    z = beta * a_max
    small = np.abs(z) < 1.0e-5
    z_safe = np.where(small, 1.0, z)
    exp_z = np.exp(np.clip(z_safe, -50.0, 50.0))
    expm1_z = np.expm1(np.clip(z_safe, -50.0, 50.0))

    log_general = np.log(a_max) + np.log(np.abs(expm1_z / z_safe))
    mean_general = a_max * (exp_z / expm1_z - 1.0 / z_safe)
    var_general = (a_max**2) * (1.0 / z_safe**2 - exp_z / expm1_z**2)

    z2 = z * z
    log_series = (
        np.log(a_max)
        + 0.5 * z
        + z2 / 24.0
        - z2 * z2 / 2880.0
        + z2 * z2 * z2 / 181440.0
    )
    mean_series = a_max * (
        0.5 + z / 12.0 - z * z2 / 720.0 + z * z2 * z2 / 30240.0
    )
    var_series = (a_max**2) * (
        1.0 / 12.0 - z2 / 240.0 + z2 * z2 / 6048.0
    )

    log_z = np.where(small, log_series, log_general)
    mean = np.where(small, mean_series, mean_general)
    variance = np.maximum(np.where(small, var_series, var_general), 0.0)
    reward = mean - lambda_reg * (beta * mean - log_z)
    return mean, variance, log_z, reward


def sample_truncated_exponential_torch(
    beta: torch.Tensor,
    a_max: float,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample A with density proportional to exp(beta*A) on [0, a_max].

    For beta != 0, inverse-transform sampling gives

        A = log(1 + U (exp(beta*a_max) - 1)) / beta,

    with U uniform on [0, 1].  The beta=0 limit is the uniform distribution.
    The policy beta is already clipped by the caller, so the exponential is
    numerically safe for the configurations used in this script.
    """

    uniform = torch.rand(
        beta.shape,
        dtype=beta.dtype,
        device=beta.device,
        generator=generator,
    )
    small = beta.abs() < 1.0e-6
    beta_safe = torch.where(small, torch.ones_like(beta), beta)
    sampled_general = torch.log1p(
        uniform * torch.expm1(beta_safe * a_max)
    ) / beta_safe
    sampled_uniform = a_max * uniform
    return torch.where(small, sampled_uniform, sampled_general).clamp(0.0, a_max)


# -----------------------------------------------------------------------------
# Networks
# -----------------------------------------------------------------------------


class ScalarMLP(nn.Module):
    def __init__(
        self,
        x_domain: Tuple[float, float],
        width: int,
        hidden_layers: int,
        enforce_zero_at_ruin: bool = False,
    ) -> None:
        super().__init__()
        lo, hi = x_domain
        self.register_buffer("x_mid", torch.tensor(0.5 * (lo + hi)))
        self.register_buffer("x_half", torch.tensor(0.5 * (hi - lo)))
        self.enforce_zero_at_ruin = enforce_zero_at_ruin

        layers: List[nn.Module] = []
        in_dim = 1
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(in_dim, width), nn.Tanh()])
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = (x - self.x_mid) / self.x_half.clamp_min(1.0e-8)
        raw = self.net(x_scaled.unsqueeze(-1)).squeeze(-1)
        if self.enforce_zero_at_ruin:
            return x * raw
        return raw


def make_beta_policy(cfg: ProblemConfig, train: TrainingConfig, device: torch.device) -> ScalarMLP:
    return ScalarMLP(
        (cfg.x_train_lo, cfg.x_train_hi), train.width, train.hidden_layers
    ).to(device)


def make_value_network(cfg: ProblemConfig, train: TrainingConfig, device: torch.device) -> ScalarMLP:
    return ScalarMLP(
        (cfg.x_train_lo, cfg.x_train_hi),
        train.width,
        train.hidden_layers,
        enforce_zero_at_ruin=True,
    ).to(device)


def make_derivative_network(cfg: ProblemConfig, train: TrainingConfig, device: torch.device) -> ScalarMLP:
    return ScalarMLP(
        (cfg.x_train_lo, cfg.x_train_hi), train.width, train.hidden_layers
    ).to(device)


def initialize_uniform_policy(policy: ScalarMLP) -> None:
    """Make beta(x)=0 initially without destroying hidden-layer gradients."""

    final_linear = [m for m in policy.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.zeros_(final_linear.weight)
    nn.init.zeros_(final_linear.bias)


def beta_and_spatial_derivative(
    policy: nn.Module, x: torch.Tensor, beta_clip: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate beta and d beta / dx elementwise without parameter gradients."""

    with torch.enable_grad():
        x_req = x.detach().requires_grad_(True)
        beta_raw = policy(x_req)
        beta = beta_raw.clamp(-beta_clip, beta_clip)
        beta_x = torch.autograd.grad(beta.sum(), x_req, create_graph=False)[0]
    return beta.detach(), beta_x.detach()


# -----------------------------------------------------------------------------
# Exploratory simulation and BEL target
# -----------------------------------------------------------------------------


@torch.no_grad()
def simulate_policy_targets(
    cfg: ProblemConfig,
    policy: nn.Module,
    x0: torch.Tensor,
    n_paths: int,
    *,
    need_derivative: bool = True,
    generator: Optional[torch.Generator] = None,
    d_w: Optional[torch.Tensor] = None,
    bridge_u: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Simulate value and derivative targets from many initial states.

    The exploratory drift is mu(x)-E_pi[a].  For the tangent process,

        d/dx [mu(x)-E_pi_beta(x)[a]]
          = mu'(x)-Var_pi(a) beta'(x).

    The Brownian-bridge test reduces missed boundary crossings between grid
    points.  The returned derivative target retains the BEL kernel used in the
    uncorrected whole-space construction; the fixed-policy BVP below is included precisely to reveal any
    remaining killed-boundary or time-discretization bias.
    """

    device = x0.device
    dtype = x0.dtype
    b_states = x0.numel()
    n_steps = int(math.ceil(cfg.horizon / cfg.dt))
    dt = cfg.dt

    x = x0.reshape(-1, 1).expand(b_states, n_paths).clone()
    alive = x > cfg.ruin
    value_paths = torch.zeros_like(x)

    if need_derivative:
        tangent = torch.ones_like(x)
        stochastic_integral = torch.zeros_like(x)
        derivative_paths = torch.zeros_like(x)
    else:
        tangent = stochastic_integral = derivative_paths = None

    if d_w is not None:
        if d_w.shape != (b_states, n_paths, n_steps):
            raise ValueError("d_w has the wrong shape")
    if bridge_u is not None:
        if bridge_u.shape != (b_states, n_paths, n_steps):
            raise ValueError("bridge_u has the wrong shape")

    policy.eval()
    for k in range(n_steps):
        t_left = k * dt
        t_right = (k + 1) * dt
        alive_f = alive.to(dtype)

        if need_derivative:
            beta, beta_x = beta_and_spatial_derivative(policy, x, cfg.beta_clip)
        else:
            beta = policy(x).clamp(-cfg.beta_clip, cfg.beta_clip)
            beta_x = None

        mean_a, var_a, _, reward = truncated_exponential_stats_torch(
            beta, cfg.a_max, cfg.lambda_reg
        )
        sigma = cfg.sigma_torch(x).clamp_min(1.0e-7)
        drift = cfg.mu_torch(x) - mean_a

        # Left-point quadrature for the value target.
        value_paths.add_(
            math.exp(-cfg.discount * t_left) * reward * dt * alive_f
        )

        if d_w is None:
            dw_k = math.sqrt(dt) * torch.randn(
                b_states, n_paths, device=device, dtype=dtype, generator=generator
            )
        else:
            dw_k = d_w[:, :, k]

        if need_derivative:
            # BEL stochastic integral and full tangent update use left values.
            stochastic_integral.add_((tangent / sigma) * dw_k * alive_f)
            drift_x = cfg.mu_prime_torch(x) - var_a * beta_x
            tangent = tangent + drift_x * tangent * dt + cfg.sigma_prime_torch(x) * tangent * dw_k

        x_new = x + drift * dt + sigma * dw_k

        # Approximate within-step ruin for constant sigma using a Brownian bridge.
        both_positive = alive & (x_new > cfg.ruin)
        distance_old = (x - cfg.ruin).clamp_min(0.0)
        distance_new = (x_new - cfg.ruin).clamp_min(0.0)
        crossing_prob = torch.exp(
            -2.0 * distance_old * distance_new / (sigma.square() * dt)
        ).clamp(0.0, 1.0)
        if bridge_u is None:
            u_k = torch.rand(
                b_states, n_paths, device=device, dtype=dtype, generator=generator
            )
        else:
            u_k = bridge_u[:, :, k]
        survived_bridge = u_k > crossing_prob
        alive_new = both_positive & survived_bridge

        if need_derivative:
            # Right-point BEL integrand.  This avoids the undefined N_0.
            x_reward = torch.where(alive_new, x_new, torch.zeros_like(x_new))
            beta_right = policy(x_reward).clamp(-cfg.beta_clip, cfg.beta_clip)
            _, _, _, reward_right = truncated_exponential_stats_torch(
                beta_right, cfg.a_max, cfg.lambda_reg
            )
            n_t = stochastic_integral / max(t_right, 1.0e-12)
            derivative_paths.add_(
                math.exp(-cfg.discount * t_right)
                * reward_right
                * n_t
                * dt
                * alive_new.to(dtype)
            )

        alive = alive_new
        x = torch.where(alive, x_new, torch.zeros_like(x_new))
        if need_derivative:
            tangent = tangent * alive.to(dtype)
            stochastic_integral = stochastic_integral * alive.to(dtype)

    value_mean = value_paths.mean(dim=1)
    derivative_mean = derivative_paths.mean(dim=1) if need_derivative else None
    return value_mean, derivative_mean, value_paths


def sample_training_states(
    cfg: ProblemConfig,
    train: TrainingConfig,
    batch: int,
    device: torch.device,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    # A uniform draw over [0.05, 3] severely undersamples the sharp derivative
    # boundary layer.  This mixture still covers the whole domain while putting
    # most regression states where V_x changes rapidly.
    full = cfg.x_train_lo + (cfg.x_train_hi - cfg.x_train_lo) * torch.rand(
        batch, device=device, generator=generator
    )
    boundary_hi = min(train.boundary_sampling_hi, cfg.x_train_hi)
    boundary = cfg.x_train_lo + (boundary_hi - cfg.x_train_lo) * torch.rand(
        batch, device=device, generator=generator
    )
    choose_boundary = torch.rand(
        batch, device=device, generator=generator
    ) < train.boundary_sampling_fraction
    return torch.where(choose_boundary, boundary, full)


@torch.no_grad()
def simulate_crn_finite_difference_targets(
    cfg: ProblemConfig,
    train: TrainingConfig,
    policy: nn.Module,
    x0: torch.Tensor,
    n_paths: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Boundary-aware numerical derivative diagnostic using paired rollouts.

    This is not labeled as a BEL estimator.  It differentiates the discretized
    killed value by a centered finite difference and uses exactly the same
    Brownian increments and bridge uniforms at x+h and x-h.  The mode is useful
    for diagnosing whether disagreement comes from the two-network
    parameterization or from the killed-boundary BEL target itself.
    """

    b_states = x0.numel()
    n_steps = int(math.ceil(cfg.horizon / cfg.dt))
    h = train.crn_fd_step
    x_minus = torch.clamp(x0 - h, min=cfg.ruin + 1.0e-4)
    x_plus = x0 + h
    denominator = x_plus - x_minus

    d_w_base = math.sqrt(cfg.dt) * torch.randn(
        b_states,
        n_paths,
        n_steps,
        device=x0.device,
        dtype=x0.dtype,
        generator=generator,
    )
    bridge_base = torch.rand(
        b_states,
        n_paths,
        n_steps,
        device=x0.device,
        dtype=x0.dtype,
        generator=generator,
    )
    x_stacked = torch.cat((x_minus, x_plus), dim=0)
    d_w_stacked = torch.cat((d_w_base, d_w_base), dim=0)
    bridge_stacked = torch.cat((bridge_base, bridge_base), dim=0)
    value_stacked, _, _ = simulate_policy_targets(
        cfg,
        policy,
        x_stacked,
        n_paths,
        need_derivative=False,
        d_w=d_w_stacked,
        bridge_u=bridge_stacked,
    )
    value_minus = value_stacked[:b_states]
    value_plus = value_stacked[b_states:]
    value_center = 0.5 * (value_plus + value_minus)
    derivative = (value_plus - value_minus) / denominator
    return value_center, derivative


@torch.no_grad()
def build_evaluation_targets(
    cfg: ProblemConfig,
    train: TrainingConfig,
    policy: nn.Module,
    x0: torch.Tensor,
    n_paths: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if train.derivative_target_method == "bel":
        value, derivative, _ = simulate_policy_targets(
            cfg, policy, x0, n_paths, generator=generator
        )
        if derivative is None:
            raise RuntimeError("BEL target was unexpectedly absent")
        return value, derivative
    if train.derivative_target_method == "crn_fd":
        return simulate_crn_finite_difference_targets(
            cfg, train, policy, x0, n_paths, generator=generator
        )
    raise ValueError(
        "derivative_target_method must be either 'bel' or 'crn_fd'"
    )


# -----------------------------------------------------------------------------
# Neural policy evaluation and improvement
# -----------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    value_net: ScalarMLP
    derivative_net: ScalarMLP
    history: Dict[str, List[float]]
    best_validation_loss: float


def derivative_of_value(
    value_net: nn.Module, x: torch.Tensor, *, create_graph: bool
) -> torch.Tensor:
    x_req = x if x.requires_grad else x.detach().requires_grad_(True)
    values = value_net(x_req)
    return torch.autograd.grad(
        values.sum(), x_req, create_graph=create_graph, retain_graph=create_graph
    )[0]


def train_policy_evaluation(
    cfg: ProblemConfig,
    train: TrainingConfig,
    policy: nn.Module,
    device: torch.device,
    *,
    seed: int,
    epochs: int,
    batch_states: int,
    paths_per_state: int,
    patience: int,
    verbose_prefix: str,
) -> EvaluationResult:
    seed_everything(seed)
    value_net = make_value_network(cfg, train, device)
    derivative_net = make_derivative_network(cfg, train, device)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(
        list(value_net.parameters()) + list(derivative_net.parameters()),
        lr=train.pe_lr,
    )

    history: Dict[str, List[float]] = {
        "value": [],
        "derivative": [],
        "consistency": [],
        "validation": [],
    }

    # Fixed validation data make early-stopping comparisons meaningful.
    val_gen = torch.Generator(device=device)
    val_gen.manual_seed(seed + 100_003)
    n_boundary_val = max(
        2, int(round(train.validation_states * train.boundary_sampling_fraction))
    )
    n_outer_val = max(2, train.validation_states - n_boundary_val)
    boundary_hi = min(train.boundary_sampling_hi, cfg.x_train_hi)
    x_val = torch.cat(
        (
            torch.linspace(
                cfg.x_train_lo, boundary_hi, n_boundary_val, device=device
            ),
            torch.linspace(
                boundary_hi, cfg.x_train_hi, n_outer_val + 1, device=device
            )[1:],
        )
    )
    with torch.no_grad():
        value_val, derivative_val = build_evaluation_targets(
            cfg,
            train,
            policy,
            x_val,
            train.validation_paths_per_state,
            generator=val_gen,
        )

    best_loss = float("inf")
    best_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    stale = 0

    train_gen = torch.Generator(device=device)
    train_gen.manual_seed(seed + 17)

    for epoch in range(epochs):
        value_net.train()
        derivative_net.train()
        x_batch = sample_training_states(
            cfg, train, batch_states, device, generator=train_gen
        )
        with torch.no_grad():
            value_target, derivative_target = build_evaluation_targets(
                cfg,
                train,
                policy,
                x_batch,
                paths_per_state,
                generator=train_gen,
            )

        x_req = x_batch.detach().requires_grad_(True)
        value_pred = value_net(x_req)
        derivative_pred = derivative_net(x_req)
        value_x = derivative_of_value(value_net, x_req, create_graph=True)

        loss_value = torch.mean((value_pred - value_target) ** 2)
        loss_derivative = torch.mean((derivative_pred - derivative_target) ** 2)
        loss_consistency = torch.mean((value_x - derivative_pred.detach()) ** 2)
        total_loss = (
            loss_value
            + loss_derivative
            + train.consistency_weight * loss_consistency
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(value_net.parameters()) + list(derivative_net.parameters()), 10.0
        )
        optimizer.step()

        history["value"].append(float(loss_value.detach().cpu()))
        history["derivative"].append(float(loss_derivative.detach().cpu()))
        history["consistency"].append(float(loss_consistency.detach().cpu()))

        # Validate every epoch; targets remain fixed.
        value_net.eval()
        derivative_net.eval()
        x_val_req = x_val.detach().requires_grad_(True)
        value_val_pred = value_net(x_val_req)
        derivative_val_pred = derivative_net(x_val_req)
        value_x_val = derivative_of_value(
            value_net, x_val_req, create_graph=False
        )
        val_loss_tensor = (
            torch.mean((value_val_pred - value_val) ** 2)
            + torch.mean((derivative_val_pred - derivative_val) ** 2)
            + train.consistency_weight
            * torch.mean((value_x_val - derivative_val_pred.detach()) ** 2)
        )
        val_loss = float(val_loss_tensor.detach().cpu())
        history["validation"].append(val_loss)

        if val_loss < best_loss - train.pe_min_delta:
            best_loss = val_loss
            best_state = {
                "V": copy.deepcopy(value_net.state_dict()),
                "P": copy.deepcopy(derivative_net.state_dict()),
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("Policy evaluation did not create a checkpoint")
    value_net.load_state_dict(best_state["V"])
    derivative_net.load_state_dict(best_state["P"])
    value_net.eval()
    derivative_net.eval()

    for parameter in policy.parameters():
        parameter.requires_grad_(True)

    print(
        f"{verbose_prefix}: epochs={len(history['validation'])}, "
        f"best validation={best_loss:.3e}"
    )
    return EvaluationResult(value_net, derivative_net, history, best_loss)


def diagnostic_state_grid(
    cfg: ProblemConfig,
    train: TrainingConfig,
    n_states: int,
    device: torch.device,
) -> torch.Tensor:
    """Deterministic grid with quadratic refinement near the ruin boundary."""

    n_boundary = max(3, int(round(0.85 * n_states)))
    n_outer = max(2, n_states - n_boundary)
    boundary_hi = min(train.boundary_sampling_hi, cfg.x_train_hi)
    u = torch.linspace(0.0, 1.0, n_boundary, device=device)
    boundary = cfg.x_train_lo + (boundary_hi - cfg.x_train_lo) * u.square()
    outer = torch.linspace(
        boundary_hi, cfg.x_train_hi, n_outer + 1, device=device
    )[1:]
    return torch.cat((boundary, outer))


@torch.no_grad()
def generate_static_evaluation_dataset(
    cfg: ProblemConfig,
    train: TrainingConfig,
    policy: nn.Module,
    device: torch.device,
    *,
    n_states: int,
    paths_per_state: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate relatively low-noise targets once, in state chunks."""

    states = diagnostic_state_grid(cfg, train, n_states, device)
    value_parts: List[torch.Tensor] = []
    derivative_parts: List[torch.Tensor] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    chunk = train.final_dataset_chunk_states
    for start in range(0, states.numel(), chunk):
        x_chunk = states[start : start + chunk]
        value, derivative = build_evaluation_targets(
            cfg,
            train,
            policy,
            x_chunk,
            paths_per_state,
            generator=generator,
        )
        value_parts.append(value)
        derivative_parts.append(derivative)
    return states, torch.cat(value_parts), torch.cat(derivative_parts)


def train_static_final_evaluation(
    cfg: ProblemConfig,
    train: TrainingConfig,
    policy: nn.Module,
    device: torch.device,
    *,
    seed: int,
    verbose_prefix: str,
) -> EvaluationResult:
    """High-accuracy final evaluation used for the evaluation diagnostic.

    Online Monte Carlo targets are appropriate for the PIA loop, but they make
    it unnecessarily hard to tell approximation error from target noise in a
    evaluation figure.  Here, targets are generated once on a boundary-refined
    grid with substantially more paths, after which ordinary mini-batch
    regression is run to convergence.  The fixed-policy BVP remains evaluation
    only and is not used in this training routine.
    """

    seed_everything(seed)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    target_started = time.time()
    x_train, value_train, derivative_train = generate_static_evaluation_dataset(
        cfg,
        train,
        policy,
        device,
        n_states=train.final_dataset_states,
        paths_per_state=train.final_dataset_paths,
        seed=seed + 101,
    )
    x_val, value_val, derivative_val = generate_static_evaluation_dataset(
        cfg,
        train,
        policy,
        device,
        n_states=train.final_validation_states,
        paths_per_state=train.final_validation_paths,
        seed=seed + 202,
    )
    target_seconds = time.time() - target_started

    value_net = make_value_network(cfg, train, device)
    derivative_net = make_derivative_network(cfg, train, device)
    optimizer = torch.optim.Adam(
        list(value_net.parameters()) + list(derivative_net.parameters()),
        lr=train.pe_lr,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 303)

    history: Dict[str, List[float]] = {
        "value": [],
        "derivative": [],
        "consistency": [],
        "validation": [],
    }
    best_loss = float("inf")
    best_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    stale_checks = 0

    for step in range(train.final_optimizer_steps):
        indices = torch.randint(
            0,
            x_train.numel(),
            (train.final_optimizer_batch,),
            device=device,
            generator=generator,
        )
        x_batch = x_train[indices].detach().requires_grad_(True)
        value_pred = value_net(x_batch)
        derivative_pred = derivative_net(x_batch)
        value_x = derivative_of_value(value_net, x_batch, create_graph=True)
        loss_value = torch.mean((value_pred - value_train[indices]) ** 2)
        loss_derivative = torch.mean(
            (derivative_pred - derivative_train[indices]) ** 2
        )
        loss_consistency = torch.mean(
            (value_x - derivative_pred.detach()) ** 2
        )
        total_loss = (
            loss_value
            + loss_derivative
            + train.consistency_weight * loss_consistency
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(value_net.parameters()) + list(derivative_net.parameters()), 10.0
        )
        optimizer.step()
        history["value"].append(float(loss_value.detach().cpu()))
        history["derivative"].append(float(loss_derivative.detach().cpu()))
        history["consistency"].append(float(loss_consistency.detach().cpu()))

        if step % train.final_validation_every != 0:
            continue

        value_net.eval()
        derivative_net.eval()
        x_val_req = x_val.detach().requires_grad_(True)
        value_val_pred = value_net(x_val_req)
        derivative_val_pred = derivative_net(x_val_req)
        value_x_val = derivative_of_value(
            value_net, x_val_req, create_graph=False
        )
        validation = (
            torch.mean((value_val_pred - value_val) ** 2)
            + torch.mean((derivative_val_pred - derivative_val) ** 2)
            + train.consistency_weight
            * torch.mean((value_x_val - derivative_val_pred.detach()) ** 2)
        )
        val_float = float(validation.detach().cpu())
        history["validation"].append(val_float)
        if val_float < best_loss - train.pe_min_delta:
            best_loss = val_float
            best_state = {
                "V": copy.deepcopy(value_net.state_dict()),
                "P": copy.deepcopy(derivative_net.state_dict()),
            }
            stale_checks = 0
        else:
            stale_checks += 1
        value_net.train()
        derivative_net.train()

        if stale_checks >= train.final_validation_checks_patience:
            break

    if best_state is None:
        raise RuntimeError("Static final evaluation did not create a checkpoint")
    value_net.load_state_dict(best_state["V"])
    derivative_net.load_state_dict(best_state["P"])
    value_net.eval()
    derivative_net.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(True)

    print(
        f"{verbose_prefix}: target generation={target_seconds:.1f}s, "
        f"optimizer steps={len(history['value'])}, best validation={best_loss:.3e}"
    )
    return EvaluationResult(value_net, derivative_net, history, best_loss)


def fit_improved_policy(
    cfg: ProblemConfig,
    train: TrainingConfig,
    old_policy: ScalarMLP,
    derivative_net: ScalarMLP,
    device: torch.device,
    *,
    seed: int,
) -> ScalarMLP:
    candidate = make_beta_policy(cfg, train, device)
    candidate.load_state_dict(copy.deepcopy(old_policy.state_dict()))
    optimizer = torch.optim.Adam(candidate.parameters(), lr=train.pi_lr)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stale = 0
    derivative_net.eval()

    for _ in range(train.pi_epochs):
        x_batch = sample_training_states(
            cfg, train, train.pi_batch_states, device, generator=generator
        )
        with torch.no_grad():
            p_hat = derivative_net(x_batch)
            target = ((1.0 - p_hat) / cfg.lambda_reg).clamp(
                -cfg.beta_clip, cfg.beta_clip
            )
        pred = candidate(x_batch)
        loss = torch.mean((pred - target) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_float = float(loss.detach().cpu())
        if loss_float < best_loss - 1.0e-7:
            best_loss = loss_float
            best_state = copy.deepcopy(candidate.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= train.pi_patience:
                break

    if best_state is None:
        raise RuntimeError("Policy improvement did not create a checkpoint")
    candidate.load_state_dict(best_state)
    candidate.eval()
    return candidate


@torch.no_grad()
def paired_policy_difference(
    cfg: ProblemConfig,
    old_policy: nn.Module,
    candidate: nn.Module,
    train: TrainingConfig,
    device: torch.device,
    *,
    seed: int,
) -> Tuple[float, float, float, float]:
    n_steps = int(math.ceil(cfg.horizon / cfg.dt))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    shape = (1, train.acceptance_paths, n_steps)
    d_w = math.sqrt(cfg.dt) * torch.randn(
        *shape, device=device, generator=generator
    )
    bridge_u = torch.rand(*shape, device=device, generator=generator)
    x0 = torch.tensor([train.x0_acceptance], device=device)

    old_mean_t, _, old_paths = simulate_policy_targets(
        cfg,
        old_policy,
        x0,
        train.acceptance_paths,
        need_derivative=False,
        d_w=d_w,
        bridge_u=bridge_u,
    )
    new_mean_t, _, new_paths = simulate_policy_targets(
        cfg,
        candidate,
        x0,
        train.acceptance_paths,
        need_derivative=False,
        d_w=d_w,
        bridge_u=bridge_u,
    )
    differences = (new_paths[0] - old_paths[0]).detach().cpu().numpy()
    mean_difference = float(np.mean(differences))
    se_difference = float(np.std(differences, ddof=1) / math.sqrt(differences.size))
    return (
        float(old_mean_t.item()),
        float(new_mean_t.item()),
        mean_difference,
        se_difference,
    )


def blend_policy_parameters(
    cfg: ProblemConfig,
    train: TrainingConfig,
    old_policy: ScalarMLP,
    candidate: ScalarMLP,
    alpha: float,
    device: torch.device,
) -> ScalarMLP:
    blended = make_beta_policy(cfg, train, device)
    blended.load_state_dict(copy.deepcopy(old_policy.state_dict()))
    with torch.no_grad():
        for dst, src in zip(blended.parameters(), candidate.parameters()):
            dst.mul_(1.0 - alpha).add_(src, alpha=alpha)
    blended.eval()
    return blended


def run_policy_iteration(
    cfg: ProblemConfig,
    train: TrainingConfig,
    device: torch.device,
) -> Tuple[ScalarMLP, List[Dict[str, float]]]:
    seed_everything(train.base_seed)
    policy = make_beta_policy(cfg, train, device)
    initialize_uniform_policy(policy)
    policy.eval()
    history: List[Dict[str, float]] = []

    for iteration in range(train.pia_iterations):
        evaluation = train_policy_evaluation(
            cfg,
            train,
            policy,
            device,
            seed=train.base_seed + 1_000 * iteration,
            epochs=train.pe_epochs,
            batch_states=train.pe_batch_states,
            paths_per_state=train.pe_paths_per_state,
            patience=train.pe_patience,
            verbose_prefix=f"PI {iteration:02d} evaluation",
        )
        candidate = fit_improved_policy(
            cfg,
            train,
            policy,
            evaluation.derivative_net,
            device,
            seed=train.base_seed + 1_000 * iteration + 71,
        )

        chosen = policy
        chosen_alpha = 0.0
        chosen_stats: Optional[Tuple[float, float, float, float]] = None
        for alpha in (1.0, 0.5, 0.25, 0.10):
            proposal = (
                candidate
                if alpha == 1.0
                else blend_policy_parameters(
                    cfg, train, policy, candidate, alpha, device
                )
            )
            stats = paired_policy_difference(
                cfg,
                policy,
                proposal,
                train,
                device,
                seed=train.base_seed + 20_000 + iteration,
            )
            _, _, difference, se_difference = stats
            if difference > train.acceptance_z * se_difference:
                chosen = proposal
                chosen_alpha = alpha
                chosen_stats = stats
                break

        if chosen_stats is None:
            chosen_stats = paired_policy_difference(
                cfg,
                policy,
                policy,
                train,
                device,
                seed=train.base_seed + 20_000 + iteration,
            )

        old_mean, new_mean, difference, se_difference = chosen_stats
        policy = chosen
        policy.eval()
        row = {
            "iteration": float(iteration),
            "old_value": old_mean,
            "accepted_value": new_mean,
            "paired_difference": difference,
            "paired_se": se_difference,
            "accepted_alpha": chosen_alpha,
            "pe_validation_loss": evaluation.best_validation_loss,
        }
        history.append(row)
        print(
            f"PI {iteration:02d} acceptance: old={old_mean:.6f}, "
            f"new={new_mean:.6f}, diff={difference:.3e} "
            f"(paired SE {se_difference:.3e}), alpha={chosen_alpha:g}"
        )

    return policy, history


# -----------------------------------------------------------------------------
# Fixed-policy BVP reference (evaluation only)
# -----------------------------------------------------------------------------


@torch.no_grad()
def policy_numpy(
    policy: nn.Module,
    x: np.ndarray,
    cfg: ProblemConfig,
    device: torch.device,
) -> np.ndarray:
    shape = np.asarray(x).shape
    x_clipped = np.clip(np.asarray(x, dtype=float), cfg.x_train_lo, cfg.x_train_hi)
    x_tensor = torch.as_tensor(
        x_clipped.reshape(-1), dtype=torch.float32, device=device
    )
    beta = policy(x_tensor).clamp(-cfg.beta_clip, cfg.beta_clip)
    return beta.detach().cpu().numpy().reshape(shape)


def solve_fixed_policy_reference(
    cfg: ProblemConfig,
    policy: nn.Module,
    device: torch.device,
) -> object:
    """Solve the infinite-horizon linear policy-evaluation ODE.

        0.5 sigma^2 V'' + (mu-E[a]) V' - c V + r^pi = 0,
        V(0)=0,  V'(x_reference_hi)=0.

    The terminal Neumann condition is imposed far beyond the plotted interval.
    """

    def ode(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        beta = policy_numpy(policy, x, cfg, device)
        mean_a, _, _, reward = truncated_exponential_stats_numpy(
            beta, cfg.a_max, cfg.lambda_reg
        )
        drift = cfg.mu_numpy(x) - mean_a
        sigma = cfg.sigma_numpy(x)
        value = y[0]
        gradient = y[1]
        second = 2.0 * (
            cfg.discount * value - drift * gradient - reward
        ) / (sigma**2)
        return np.vstack((gradient, second))

    def boundary(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.array([left[0], right[1]])

    mesh = np.linspace(cfg.ruin, cfg.x_reference_hi, 700)
    beta_far = float(policy_numpy(policy, np.array([cfg.x_train_hi]), cfg, device)[0])
    _, _, _, reward_far = truncated_exponential_stats_numpy(
        np.array([beta_far]), cfg.a_max, cfg.lambda_reg
    )
    plateau = float(reward_far[0] / cfg.discount)
    guess_value = plateau * (1.0 - np.exp(-mesh))
    guess_gradient = plateau * np.exp(-mesh)
    guess = np.vstack((guess_value, guess_gradient))
    solution = solve_bvp(
        ode,
        boundary,
        mesh,
        guess,
        tol=2.0e-8,
        max_nodes=50_000,
        verbose=0,
    )
    if not solution.success:
        raise RuntimeError(f"BVP reference failed: {solution.message}")
    return solution


# -----------------------------------------------------------------------------
# Multi-seed diagnostic, metrics, and plot
# -----------------------------------------------------------------------------


def evaluate_networks_on_grid(
    value_net: nn.Module,
    derivative_net: nn.Module,
    grid: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = torch.as_tensor(grid, dtype=torch.float32, device=device).requires_grad_(True)
    value = value_net(x)
    value_x = derivative_of_value(value_net, x, create_graph=False)
    with torch.no_grad():
        derivative = derivative_net(x.detach())
    return (
        value.detach().cpu().numpy(),
        value_x.detach().cpu().numpy(),
        derivative.detach().cpu().numpy(),
    )


def rms(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error))))


def run_final_consistency_experiment(
    cfg: ProblemConfig,
    train: TrainingConfig,
    final_policy: ScalarMLP,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[Dict[str, float]], object]:
    grid = np.linspace(cfg.x_plot_lo, cfg.x_plot_hi, train.plot_points)
    reference = solve_fixed_policy_reference(cfg, final_policy, device)
    reference_value, reference_gradient = reference.sol(grid)

    values: List[np.ndarray] = []
    value_gradients: List[np.ndarray] = []
    derivative_predictions: List[np.ndarray] = []
    metric_rows: List[Dict[str, float]] = []

    for replicate in range(train.final_eval_replicates):
        seed = train.base_seed + 50_000 + replicate * 997
        result = train_static_final_evaluation(
            cfg,
            train,
            final_policy,
            device,
            seed=seed,
            verbose_prefix=f"Final policy evaluation seed {seed}",
        )
        value, value_x, derivative = evaluate_networks_on_grid(
            result.value_net, result.derivative_net, grid, device
        )
        values.append(value)
        value_gradients.append(value_x)
        derivative_predictions.append(derivative)
        metric_rows.append(
            {
                "seed": float(seed),
                "rms_P_minus_dV": rms(derivative - value_x),
                "max_P_minus_dV": float(np.max(np.abs(derivative - value_x))),
                "rms_P_minus_reference": rms(derivative - reference_gradient),
                "rms_dV_minus_reference": rms(value_x - reference_gradient),
                "rms_V_minus_reference": rms(value - reference_value),
                "best_validation_loss": result.best_validation_loss,
            }
        )

    arrays = {
        "value": np.stack(values),
        "value_x": np.stack(value_gradients),
        "P": np.stack(derivative_predictions),
        "reference_value": reference_value,
        "reference_gradient": reference_gradient,
    }
    return grid, arrays, metric_rows, reference


def mean_and_ci(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(samples, axis=0)
    if samples.shape[0] == 1:
        return mean, np.zeros_like(mean)
    se = np.std(samples, axis=0, ddof=1) / math.sqrt(samples.shape[0])
    return mean, 1.96 * se


def save_metrics(
    output_dir: Path,
    cfg: ProblemConfig,
    train: TrainingConfig,
    policy_history: Sequence[Dict[str, float]],
    metric_rows: Sequence[Dict[str, float]],
) -> None:
    metric_path = output_dir / "consistency_metrics.csv"
    fieldnames = list(metric_rows[0].keys())
    with metric_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    summary: Dict[str, Dict[str, float]] = {}
    for key in fieldnames:
        if key == "seed":
            continue
        data = np.array([row[key] for row in metric_rows], dtype=float)
        summary[key] = {
            "mean": float(np.mean(data)),
            "sample_sd": float(np.std(data, ddof=1)) if data.size > 1 else 0.0,
        }

    payload = {
        "problem": asdict(cfg),
        "training": asdict(train),
        "policy_iteration": list(policy_history),
        "replicate_metrics": list(metric_rows),
        "summary": summary,
    }
    with (output_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def plot_consistency_figure(
    output_dir: Path,
    cfg: ProblemConfig,
    train: TrainingConfig,
    grid: np.ndarray,
    arrays: Dict[str, np.ndarray],
    metric_rows: Sequence[Dict[str, float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_mean, p_ci = mean_and_ci(arrays["P"])
    vx_mean, vx_ci = mean_and_ci(arrays["value_x"])
    reference = arrays["reference_gradient"]

    discrepancy = np.abs(arrays["P"] - arrays["value_x"])
    p_error = np.abs(arrays["P"] - reference[None, :])
    vx_error = np.abs(arrays["value_x"] - reference[None, :])
    discrepancy_mean, discrepancy_ci = mean_and_ci(discrepancy)
    p_error_mean, _ = mean_and_ci(p_error)
    vx_error_mean, _ = mean_and_ci(vx_error)

    rms_discrepancy = np.mean([row["rms_P_minus_dV"] for row in metric_rows])
    max_discrepancy = np.mean([row["max_P_minus_dV"] for row in metric_rows])
    rms_p_reference = np.mean(
        [row["rms_P_minus_reference"] for row in metric_rows]
    )
    rms_vx_reference = np.mean(
        [row["rms_dV_minus_reference"] for row in metric_rows]
    )

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 11,
            "legend.fontsize": 9.3,
            "figure.dpi": 140,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.25))

    ax = axes[0]
    ax.plot(grid, reference, color="black", linewidth=2.2, label="Fixed-policy BVP reference")
    target_label = "uncorrected BEL audit" if train.derivative_target_method == "bel" else "paired CRN finite difference"
    ax.plot(
        grid,
        p_mean,
        color="#1f77b4",
        linewidth=2.0,
        label=rf"Independent network $P_\eta$ ({target_label} target)",
    )
    ax.fill_between(grid, p_mean - p_ci, p_mean + p_ci, color="#1f77b4", alpha=0.18)
    ax.plot(grid, vx_mean, color="#d62728", linewidth=1.8, linestyle="--", label=r"Autodiff $\partial_xV_\theta$")
    ax.fill_between(grid, vx_mean - vx_ci, vx_mean + vx_ci, color="#d62728", alpha=0.15)
    ax.set_xlabel(r"Wealth $x$")
    ax.set_ylabel("Derivative")
    ax.set_title("(a) Derivative estimates under one frozen policy")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=True)

    ax = axes[1]
    floor = 1.0e-6
    ax.plot(grid, np.maximum(discrepancy_mean, floor), color="#9467bd", linewidth=2.0, label=r"$|P_\eta-\partial_xV_\theta|$")
    ax.fill_between(
        grid,
        np.maximum(discrepancy_mean - discrepancy_ci, floor),
        np.maximum(discrepancy_mean + discrepancy_ci, floor),
        color="#9467bd",
        alpha=0.18,
    )
    ax.plot(grid, np.maximum(p_error_mean, floor), color="#1f77b4", linewidth=1.6, label=r"$|P_\eta-V_x^{\rm ref}|$")
    ax.plot(grid, np.maximum(vx_error_mean, floor), color="#d62728", linewidth=1.6, linestyle="--", label=r"$|\partial_xV_\theta-V_x^{\rm ref}|$")
    ax.set_yscale("log")
    ax.set_xlabel(r"Wealth $x$")
    ax.set_ylabel("Absolute error")
    ax.set_title("(b) Discrepancy and benchmark errors")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper right", frameon=True)
    metrics_text = (
        f"mean RMS(P-dV) = {rms_discrepancy:.3e}\n"
        f"mean max(P-dV) = {max_discrepancy:.3e}\n"
        f"mean RMS(P-ref) = {rms_p_reference:.3e}\n"
        f"mean RMS(dV-ref) = {rms_vx_reference:.3e}"
    )
    ax.text(
        0.03,
        0.04,
        metrics_text,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8.6,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.90, "edgecolor": "0.75"},
    )

    case_label = "state-dependent drift" if cfg.coefficient_case == "state_dependent" else "constant drift"
    fig.suptitle(
        "Two-network derivative consistency under a frozen policy",
        y=1.015,
        fontsize=12.4,
    )
    fig.text(
        0.5,
        0.965,
        f"{case_label}; {target_label} target; mean and 95% CI across seeds",
        ha="center",
        va="top",
        fontsize=9.7,
        color="0.30",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(output_dir / "derivative_consistency.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "derivative_consistency.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# State-conditional policy and multi-initial-state path diagnostics
# -----------------------------------------------------------------------------


def evaluate_policy_at_wealth_levels(
    cfg: ProblemConfig,
    policy: nn.Module,
    wealth_levels: Sequence[float],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """Evaluate beta(x), the policy density, and policy moments.

    The variable x here is the *current state*.  Calling these curves
    "different initial-wealth policies" would be imprecise because the learned
    feedback policy is state conditional: pi(a | x).
    """

    wealth = np.asarray(wealth_levels, dtype=float)
    if wealth.ndim != 1 or wealth.size == 0:
        raise ValueError("At least one policy wealth level is required")
    if np.any(wealth < cfg.x_train_lo) or np.any(wealth > cfg.x_train_hi):
        raise ValueError(
            "Policy wealth levels must lie inside the trained interval "
            f"[{cfg.x_train_lo}, {cfg.x_train_hi}]"
        )

    actions = np.linspace(0.0, cfg.a_max, 501)
    beta = policy_numpy(policy, wealth, cfg, device)
    mean_a, variance_a, log_z, reward = truncated_exponential_stats_numpy(
        beta, cfg.a_max, cfg.lambda_reg
    )
    log_density = beta[:, None] * actions[None, :] - log_z[:, None]
    density = np.exp(log_density)
    entropy = (reward - mean_a) / cfg.lambda_reg
    kl_from_uniform = math.log(cfg.a_max) - entropy

    rows: List[Dict[str, float]] = []
    for i, x_value in enumerate(wealth):
        rows.append(
            {
                "wealth": float(x_value),
                "beta": float(beta[i]),
                "mean_dividend_rate": float(mean_a[i]),
                "dividend_variance": float(variance_a[i]),
                "policy_entropy": float(entropy[i]),
                "kl_from_uniform": float(max(kl_from_uniform[i], 0.0)),
            }
        )
    return actions, density, rows


def plot_policy_density_diagnostic(
    output_dir: Path,
    cfg: ProblemConfig,
    policy: nn.Module,
    wealth_levels: Sequence[float],
    device: torch.device,
) -> List[Dict[str, float]]:
    """Plot pi(a|x) and return its numerical summary at selected states."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actions, density, rows = evaluate_policy_at_wealth_levels(
        cfg, policy, wealth_levels, device
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = plt.cm.viridis(np.linspace(0.10, 0.88, len(rows)))
    for row, pdf, color in zip(rows, density, colors):
        label = (
            rf"$x={row['wealth']:g}$: $\beta={row['beta']:.3f}$, "
            rf"$\mathbb{{E}}[a|x]={row['mean_dividend_rate']:.3f}$"
        )
        ax.plot(actions, pdf, linewidth=2.1, color=color, label=label)
    ax.axhline(
        1.0 / cfg.a_max,
        color="0.35",
        linewidth=1.2,
        linestyle=":",
        label="Uniform density",
    )
    ax.set_xlabel(r"Dividend rate $a$")
    ax.set_ylabel(r"Policy density $\pi(a\mid x)$")
    ax.set_title("Learned exploratory policy at selected current wealth levels")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, fontsize=9.0)
    fig.text(
        0.5,
        0.01,
        r"Positive $\beta(x)=(1-P(x))/\lambda$ tilts the policy toward larger dividends; "
        r"$\beta(x)=0$ is uniform.",
        ha="center",
        va="bottom",
        fontsize=9.0,
        color="0.30",
    )
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 1.0))
    fig.savefig(output_dir / "policy_density_by_wealth.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "policy_density_by_wealth.pdf", bbox_inches="tight")
    plt.close(fig)
    return rows


@torch.no_grad()
def simulate_action_sampled_wealth_paths(
    cfg: ProblemConfig,
    policy: nn.Module,
    initial_wealths: Sequence[float],
    n_paths: int,
    device: torch.device,
    *,
    seed: int,
    horizon: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate paths by sampling one dividend action at every Euler step.

    Conditional on X_k, the code draws A_k from the truncated-exponential
    feedback policy pi(.|X_k) and applies

        X_{k+1} = X_k + [mu(X_k)-A_k] dt + sigma(X_k) Delta W_k.

    This action-sampled path diagnostic is intentionally distinct from the
    policy-averaged exploratory dynamics used elsewhere in the script.  Ruin
    is absorbing.  Conditional on the Euler endpoints, a Brownian-bridge test
    catches zero crossings that would otherwise be missed between time-grid
    points.  For the present constant-volatility examples this is exact for
    each frozen-coefficient Euler step.  If a path leaves the trained state
    interval, the feedback policy is held fixed at its nearest endpoint
    instead of extrapolating the neural network.
    """

    if n_paths < 2:
        raise ValueError("trajectory n_paths must be at least two")
    x0 = np.asarray(initial_wealths, dtype=float)
    if x0.ndim != 1 or x0.size == 0:
        raise ValueError("At least one trajectory initial wealth is required")
    if np.any(x0 <= cfg.ruin):
        raise ValueError("Trajectory initial wealth must be above the ruin boundary")
    if np.any(x0 < cfg.x_train_lo) or np.any(x0 > cfg.x_train_hi):
        raise ValueError(
            "Trajectory initial wealths must lie inside the trained interval "
            f"[{cfg.x_train_lo}, {cfg.x_train_hi}]"
        )

    simulation_horizon = cfg.horizon if horizon is None else float(horizon)
    if simulation_horizon <= 0.0:
        raise ValueError("trajectory horizon must be positive")
    n_steps = int(math.ceil(simulation_horizon / cfg.dt))
    times = np.minimum(np.arange(n_steps + 1) * cfg.dt, simulation_horizon)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    action_generator = torch.Generator(device=device)
    action_generator.manual_seed(seed + 1_000_003)

    x = torch.as_tensor(x0, dtype=torch.float32, device=device).reshape(-1, 1)
    x = x.expand(-1, n_paths).clone()
    alive = x > cfg.ruin
    paths = torch.zeros(
        x0.size, n_paths, n_steps + 1, dtype=torch.float32, device=device
    )
    paths[:, :, 0] = x
    policy.eval()

    for k in range(n_steps):
        dt_k = min(cfg.dt, simulation_horizon - k * cfg.dt)
        policy_state = x.clamp(min=cfg.x_train_lo, max=cfg.x_train_hi)
        beta = policy(policy_state).clamp(-cfg.beta_clip, cfg.beta_clip)
        sampled_action = sample_truncated_exponential_torch(
            beta,
            cfg.a_max,
            generator=action_generator,
        )
        sigma = cfg.sigma_torch(x).clamp_min(1.0e-7)
        drift = cfg.mu_torch(x) - sampled_action
        d_w = math.sqrt(dt_k) * torch.randn(
            x.shape, dtype=x.dtype, device=device, generator=generator
        )
        proposal = x + drift * dt_k + sigma * d_w

        both_positive = alive & (proposal > cfg.ruin)
        distance_old = (x - cfg.ruin).clamp_min(0.0)
        distance_new = (proposal - cfg.ruin).clamp_min(0.0)
        crossing_probability = torch.exp(
            -2.0 * distance_old * distance_new / (sigma.square() * dt_k)
        ).clamp(0.0, 1.0)
        bridge_uniform = torch.rand(
            x.shape, dtype=x.dtype, device=device, generator=generator
        )
        alive = both_positive & (bridge_uniform > crossing_probability)
        x = torch.where(alive, proposal, torch.zeros_like(proposal))
        paths[:, :, k + 1] = x

    return times, paths.detach().cpu().numpy()


def summarize_wealth_paths(
    cfg: ProblemConfig,
    times: np.ndarray,
    paths: np.ndarray,
    initial_wealths: Sequence[float],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for i, initial_wealth in enumerate(initial_wealths):
        terminal = paths[i, :, -1]
        ruined = terminal <= cfg.ruin
        surviving_terminal = terminal[~ruined]
        if surviving_terminal.size:
            surviving_mean = float(np.mean(surviving_terminal))
            surviving_q05 = float(np.quantile(surviving_terminal, 0.05))
            surviving_median = float(np.quantile(surviving_terminal, 0.50))
            surviving_q95 = float(np.quantile(surviving_terminal, 0.95))
        else:
            surviving_mean = surviving_q05 = surviving_median = surviving_q95 = float("nan")
        rows.append(
            {
                "initial_wealth": float(initial_wealth),
                "horizon": float(times[-1]),
                "n_paths": float(paths.shape[1]),
                "ruin_probability_by_horizon": float(np.mean(ruined)),
                "survival_probability_to_horizon": float(np.mean(~ruined)),
                "terminal_mean": float(np.mean(terminal)),
                "terminal_q05": float(np.quantile(terminal, 0.05)),
                "terminal_median": float(np.quantile(terminal, 0.50)),
                "terminal_q95": float(np.quantile(terminal, 0.95)),
                "surviving_terminal_mean": surviving_mean,
                "surviving_terminal_q05": surviving_q05,
                "surviving_terminal_median": surviving_median,
                "surviving_terminal_q95": surviving_q95,
            }
        )
    return rows


def plot_wealth_path_diagnostic(
    output_dir: Path,
    cfg: ProblemConfig,
    times: np.ndarray,
    paths: np.ndarray,
    initial_wealths: Sequence[float],
) -> List[Dict[str, float]]:
    """Plot survivor wealth quantiles together with cumulative ruin risk."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summarize_wealth_paths(cfg, times, paths, initial_wealths)
    n_panels = len(rows)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.4 * n_panels, 4.25), sharex=True, squeeze=False
    )
    for i, (row, ax) in enumerate(zip(rows, axes[0])):
        survivor_paths = np.where(paths[i] > cfg.ruin, paths[i], np.nan)
        q05, q50, q95 = np.nanquantile(
            survivor_paths, [0.05, 0.50, 0.95], axis=0
        )
        ruin_probability = np.mean(paths[i] <= cfg.ruin, axis=0)
        ax.fill_between(
            times,
            q05,
            q95,
            color="#9ecae1",
            alpha=0.50,
            label="Survivor 5%-95% quantiles",
        )
        ax.plot(
            times,
            q50,
            color="#08519c",
            linewidth=2.0,
            label="Survivor median",
        )
        ax.plot(times, q05, color="#3182bd", linewidth=1.0, linestyle="--")
        ax.plot(times, q95, color="#3182bd", linewidth=1.0, linestyle="--")
        ax.axhline(cfg.ruin, color="#b22222", linewidth=1.1, linestyle=":")
        risk_axis = ax.twinx()
        risk_axis.plot(
            times,
            ruin_probability,
            color="#d95f02",
            linewidth=1.8,
            linestyle="-.",
            label="Cumulative ruin probability",
        )
        risk_axis.set_ylim(-0.02, 1.02)
        risk_axis.tick_params(axis="y", colors="#d95f02")
        if i == n_panels - 1:
            risk_axis.set_ylabel("Cumulative ruin probability", color="#d95f02")
        ax.set_title(
            rf"$X_0={row['initial_wealth']:g}$; "
            rf"$\widehat{{\mathbb{{P}}}}(\tau\leq T)={row['ruin_probability_by_horizon']:.3f}$"
        )
        ax.set_xlabel(r"Time $t$")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.set_ylabel(r"Wealth $X_t$")
            wealth_handles, wealth_labels = ax.get_legend_handles_labels()
            risk_handles, risk_labels = risk_axis.get_legend_handles_labels()
            ax.legend(
                wealth_handles + risk_handles,
                wealth_labels + risk_labels,
                loc="upper left",
                frameon=True,
                fontsize=8.4,
            )

    fig.suptitle(
        "Wealth under the learned policy (one sampled dividend action per Euler step)",
        y=1.015,
        fontsize=12.4,
    )
    fig.text(
        0.5,
        0.01,
        "Blue bands condition on survival; orange shows ruin risk; the policy is held fixed beyond its trained state interval.",
        ha="center",
        va="bottom",
        fontsize=9.0,
        color="0.30",
    )
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.96))
    fig.savefig(output_dir / "wealth_paths_by_initial_wealth.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "wealth_paths_by_initial_wealth.pdf", bbox_inches="tight")
    plt.close(fig)
    return rows


def write_rows_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_policy_and_trajectory_diagnostics(
    output_dir: Path,
    cfg: ProblemConfig,
    policy: nn.Module,
    device: torch.device,
    *,
    policy_wealth_levels: Sequence[float],
    trajectory_initial_wealths: Sequence[float],
    trajectory_paths: int,
    trajectory_seed: int,
    trajectory_horizon: Optional[float],
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    policy_rows = plot_policy_density_diagnostic(
        output_dir, cfg, policy, policy_wealth_levels, device
    )
    times, paths = simulate_action_sampled_wealth_paths(
        cfg,
        policy,
        trajectory_initial_wealths,
        trajectory_paths,
        device,
        seed=trajectory_seed,
        horizon=trajectory_horizon,
    )
    trajectory_rows = plot_wealth_path_diagnostic(
        output_dir, cfg, times, paths, trajectory_initial_wealths
    )
    write_rows_csv(output_dir / "policy_diagnostics.csv", policy_rows)
    write_rows_csv(output_dir / "trajectory_summary.csv", trajectory_rows)
    with (output_dir / "diagnostic_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "problem": asdict(cfg),
                "policy_diagnostics": policy_rows,
                "trajectory_diagnostics": trajectory_rows,
                "trajectory_seed": trajectory_seed,
                "simulation_note": (
                    "One dividend action sampled from pi(.|X_k) at every Euler "
                    "step; absorbing ruin; Brownian-bridge "
                    "within-step crossing correction; wealth quantiles conditional "
                    "on survival with cumulative ruin probability shown separately; "
                    "constant endpoint continuation outside the trained policy interval. "
                    "The action RNG is separate so the Brownian and bridge random "
                    "streams match the policy-averaged variant for the same seed."
                ),
            },
            handle,
            indent=2,
        )
    return policy_rows, trajectory_rows


def load_policy_checkpoint(
    path: Path, device: torch.device
) -> Tuple[ProblemConfig, TrainingConfig, ScalarMLP]:
    """Load a checkpoint created by this script or its immediate predecessor."""

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions predating the weights_only argument.
        checkpoint = torch.load(path, map_location=device)
    cfg = ProblemConfig(**checkpoint["problem"])
    stored_training = checkpoint["training"]
    known_training_fields = TrainingConfig.__dataclass_fields__
    train = TrainingConfig(
        **{key: value for key, value in stored_training.items() if key in known_training_fields}
    )
    policy = make_beta_policy(cfg, train, device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return cfg, train, policy


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = args if args is not None else parse_args()
    device = choose_device(args.device)

    if args.load_policy is None:
        cfg = ProblemConfig(coefficient_case=args.case)
        train = TrainingConfig.quick() if args.quick else TrainingConfig()
        final_policy: Optional[ScalarMLP] = None
        policy_history: List[Dict[str, float]] = []
    else:
        if not args.load_policy.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {args.load_policy}")
        cfg, train, loaded_policy = load_policy_checkpoint(args.load_policy, device)
        final_policy = loaded_policy
        policy_history = []

    stored_target = train.derivative_target_method
    train.derivative_target_method = args.derivative_target
    if args.load_policy is not None and stored_target != args.derivative_target:
        print(
            f"Loaded policy training target: {stored_target}; "
            f"requested final-evaluation target: {args.derivative_target}."
        )
    if args.consistency_weight is not None:
        train.consistency_weight = args.consistency_weight
    if args.replicates is not None:
        train.final_eval_replicates = args.replicates
    if train.final_eval_replicates < 1:
        raise ValueError("--replicates must be positive")
    trajectory_paths = (
        args.trajectory_paths
        if args.trajectory_paths is not None
        else (3_000 if args.quick else 20_000)
    )
    if trajectory_paths < 2:
        raise ValueError("--trajectory-paths must be at least two")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Profile: {args.profile}")
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Problem: {cfg.coefficient_case}, c={cfg.discount}, T={cfg.horizon}, dt={cfg.dt}")
    print(
        f"Training states: boundary-refined mixture on [{cfg.x_train_lo}, {cfg.x_train_hi}], "
        f"derivative target={train.derivative_target_method}, "
        f"consistency weight={train.consistency_weight}"
    )

    print("Training dynamics: relaxed, with the policy-averaged action in the drift.")
    if train.derivative_target_method == "bel":
        print(
            "Derivative mode: uncorrected whole-space BEL audit; "
            "this mode does not supply a killed-boundary correction."
        )
    print(
        "Numerical settings and policy continuation are preserved from the source "
        "implementations; BVP and wealth diagnostics clamp policy inputs to the "
        "training interval, whereas training evaluates the network directly."
    )
    started = time.time()
    if final_policy is None:
        final_policy, policy_history = run_policy_iteration(cfg, train, device)
        torch.save(
            {
                "problem": asdict(cfg),
                "training": asdict(train),
                "policy_state_dict": final_policy.state_dict(),
            },
            args.output_dir / "final_policy.pt",
        )
    else:
        print(f"Loaded frozen policy: {args.load_policy.resolve()}")

    policy_rows: List[Dict[str, float]] = []
    trajectory_rows: List[Dict[str, float]] = []
    if args.wealth_diagnostics:
        policy_rows, trajectory_rows = save_policy_and_trajectory_diagnostics(
            args.output_dir,
            cfg,
            final_policy,
            device,
            policy_wealth_levels=args.policy_wealth_levels,
            trajectory_initial_wealths=args.trajectory_initial_wealths,
            trajectory_paths=trajectory_paths,
            trajectory_seed=args.trajectory_seed,
            trajectory_horizon=args.trajectory_horizon,
        )

    metric_rows: List[Dict[str, float]] = []
    if not args.diagnostics_only:
        grid, arrays, metric_rows, _ = run_final_consistency_experiment(
            cfg, train, final_policy, device
        )
        save_metrics(args.output_dir, cfg, train, policy_history, metric_rows)
        plot_consistency_figure(
            args.output_dir, cfg, train, grid, arrays, metric_rows
        )

    elapsed = time.time() - started
    if policy_rows:
        print("\nPolicy diagnostics")
    for row in policy_rows:
        print(
            f"  x={row['wealth']:g}: beta={row['beta']:.4f}, "
            f"E[a|x]={row['mean_dividend_rate']:.4f}, "
            f"KL(uniform)={row['kl_from_uniform']:.3e}"
        )
    if trajectory_rows:
        print("\nTrajectory diagnostics")
    for row in trajectory_rows:
        print(
            f"  X0={row['initial_wealth']:g}: "
            f"P(ruin by T)={row['ruin_probability_by_horizon']:.4f}, "
            f"terminal median={row['terminal_median']:.4f}"
        )
    if metric_rows:
        print("\nReplicate derivative metrics")
        for row in metric_rows:
            print(
                f"  seed={int(row['seed'])}: RMS(P-dV)={row['rms_P_minus_dV']:.3e}, "
                f"max(P-dV)={row['max_P_minus_dV']:.3e}, "
                f"RMS(P-ref)={row['rms_P_minus_reference']:.3e}, "
                f"RMS(dV-ref)={row['rms_dV_minus_reference']:.3e}"
            )
    print(f"Completed in {elapsed:.1f} seconds.")
    if args.wealth_diagnostics:
        print(f"Saved {args.output_dir / 'policy_density_by_wealth.png'}")
        print(f"Saved {args.output_dir / 'wealth_paths_by_initial_wealth.png'}")
        print(f"Saved {args.output_dir / 'policy_diagnostics.csv'}")
        print(f"Saved {args.output_dir / 'trajectory_summary.csv'}")
    if metric_rows:
        print(f"Saved {args.output_dir / 'derivative_consistency.png'}")
        print(f"Saved {args.output_dir / 'consistency_metrics.csv'}")
        print(f"Saved {args.output_dir / 'experiment_summary.json'}")


if __name__ == "__main__":
    main(_CLI_ARGS)

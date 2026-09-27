# Copyright (c) 2026 Gaozhan Wang. See LICENSE.
# Modification note: historical scripts consolidated; numerical code preserved.
"""Historical policy iteration using the uncorrected stopped-BEL estimator.

This module preserves the uploaded implementation for reproducibility. It does
not repair the missing killing-boundary contribution or the other known issues
listed in legacy/README.md. It is not the corrected killed-semigroup experiment.

The public functions and their original defaults are retained. The two CLI
entrypoints pass their own explicit historical settings. Importing this module
does not train a model, generate plots, or run an experiment.
"""

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, Dict, List

import numpy as np
import torch
import torch.nn as nn

# ------------------------------ Device & seeds --------------------------------
def get_device(dev: Optional[Union[str, torch.device]] = None) -> torch.device:
    if dev is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev) if isinstance(dev, str) else dev

def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)

# ------------------------------ Simple MLPs -----------------------------------
def mlp_1in(width: int = 64, out_dim: int = 1) -> nn.Module:
    net = nn.Sequential(
        nn.Linear(1, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, out_dim)
    )
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return net

# ------------------------------ Policy: β(x) ----------------------------------
class BetaPolicy(nn.Module):
    """
    Policy parameter β(x) -> truncated exponential on [0, A_max]:
        π(a|x) ∝ exp( β(x) * a ),  a ∈ [0, A_max]
    """
    def __init__(self, width: int = 64):
        super().__init__()
        self.net = mlp_1in(width=width, out_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [...], returns β: [...]
        return self.net(x.unsqueeze(-1)).squeeze(-1)

# ------------------------ Truncated exponential helpers ------------------------
def _safe_expm1(z: torch.Tensor) -> torch.Tensor:
    # torch.expm1 is already stable near 0; keep wrapper for readability.
    return torch.expm1(z)

def trunc_exp_Z(beta: torch.Tensor, A: float, eps: float = 1e-12) -> torch.Tensor:
    """
    Z(β) = ∫_0^A exp(β a) da = (exp(βA)-1)/β,  with limit A as β→0.
    """
    b = beta
    bA = b * A
    near0 = b.abs() < 1e-6
    Z = torch.empty_like(b)
    Z[near0] = A
    Z[~near0] = _safe_expm1(bA[~near0]) / (b[~near0] + eps)
    return Z.clamp_min(eps)

def trunc_exp_logZ(beta: torch.Tensor, A: float) -> torch.Tensor:
    return torch.log(trunc_exp_Z(beta, A))

def trunc_exp_Ea(beta: torch.Tensor, A: float, eps: float = 1e-12) -> torch.Tensor:
    """
    E[a] under π(a) ∝ exp(β a) on [0,A]:
      E[a] = [ (A e^{βA})/β - (e^{βA})/β^2 + 1/β^2 ] / Z(β),
    with limit A/2 as β→0.
    """
    b = beta
    near0 = b.abs() < 1e-6
    Ea = torch.empty_like(b)

    # limit β→0
    Ea[near0] = 0.5 * A

    # general case
    b_nz = b[~near0]
    e = torch.exp((b_nz * A).clamp(-50.0, 50.0))
    Z = trunc_exp_Z(b_nz, A, eps=eps)

    term = (A * e) / (b_nz + eps) - e / ((b_nz + eps) ** 2) + 1.0 / ((b_nz + eps) ** 2)
    Ea[~near0] = term / Z
    return Ea

def trunc_exp_Elogpi(beta: torch.Tensor, A: float) -> torch.Tensor:
    """
    log π(a) = β a - log Z(β). So E[log π] = β E[a] - log Z(β).
    """
    return beta * trunc_exp_Ea(beta, A) - trunc_exp_logZ(beta, A)

def trunc_exp_sample(beta: torch.Tensor, A: float, U: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Inverse-CDF sampling with U~Unif(0,1):
      F(a)= (e^{βa}-1)/(e^{βA}-1)  =>  a = (1/β) log(1 + U (e^{βA}-1)).
    limit β→0: a = U*A.
    beta, U must be same shape.
    """
    b = beta
    near0 = b.abs() < 1e-6
    a = torch.empty_like(b)

    a[near0] = U[near0] * A

    b_nz = b[~near0]
    U_nz = U[~near0]
    eA = torch.exp((b_nz * A).clamp(-50.0, 50.0))
    a[~near0] = torch.log1p(U_nz * (eA - 1.0)) / (b_nz + eps)
    return a.clamp(0.0, A)

# ------------------------------ Problem spec ----------------------------------
@dataclass
class DividendRuinSpec:
    # action set
    A_max: float = 1.0
    # entropy temperature
    lambda_reg: float = 1.0
    # discount c > 0
    discount: float = 5.0
    # horizon cap T
    T: float = 2.0
    # time step
    dt: float = 1e-2
    # ruin boundary
    x_ruin: float = 0.0

    # drift and volatility
    mu: Callable[[torch.Tensor], torch.Tensor] = None
    mu_prime: Callable[[torch.Tensor], torch.Tensor] = None
    sigma: Callable[[torch.Tensor], torch.Tensor] = None
    sigma_prime: Callable[[torch.Tensor], torch.Tensor] = None

    def __post_init__(self):
        # defaults: mu(x)=1, sigma(x)=1
        if self.mu is None:
            self.mu = lambda x: 0.0*x + 1.0
        if self.mu_prime is None:
            self.mu_prime = lambda x: torch.full_like(x, 0.0)
        if self.sigma is None:
            self.sigma = lambda x: torch.ones_like(x)
        if self.sigma_prime is None:
            self.sigma_prime = lambda x: torch.zeros_like(x)

# ------------------------------ Vectorized simulation --------------------------
@torch.no_grad()
def simulate_value_and_kernel(
    spec: DividendRuinSpec,
    beta_policy: nn.Module,
    x0: torch.Tensor,                 # [B]
    n_paths: int,
    dW: Optional[torch.Tensor] = None, # [B, n_paths, K] or [n_paths, K]
    U: Optional[torch.Tensor] = None,  # [B, n_paths, K] or [n_paths, K]
    beta_clip: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      J_mean  [B]  : MC mean of value target  ∫ e^{-ct} H dt until T or ruin
      Jp_mean [B]  : MC mean of BEL derivative target (same integrand * N_t)
      J_samps [B, n_paths] : per-path accumulated returns (for CI / error bars)
    """
    device = x0.device
    B = x0.shape[0]
    K = int(math.ceil(spec.T / spec.dt))
    dt = spec.dt

    # shape helpers
    x = x0.view(B, 1).expand(B, n_paths).clone()
    gradX = torch.ones_like(x)                 # ∂X/∂x
    S = torch.zeros_like(x)                    # ∫ σ^{-1} gradX dW
    alive = (x > spec.x_ruin)

    # accumulators
    J = torch.zeros_like(x)
    Jp = torch.zeros_like(x)

    # CRN seeds
    if dW is None:
        dW = math.sqrt(dt) * torch.randn(B, n_paths, K, device=device)
    else:
        if dW.dim() == 2:
            dW = dW.unsqueeze(0).expand(B, -1, -1).contiguous()
        assert dW.shape == (B, n_paths, K)

    if U is None:
        U = torch.rand(B, n_paths, K, device=device)
    else:
        if U.dim() == 2:
            U = U.unsqueeze(0).expand(B, -1, -1).contiguous()
        assert U.shape == (B, n_paths, K)

    # time loop (vectorized over B*n_paths)
    for k in range(K):
        t = (k + 1) * dt
        disc = math.exp(-spec.discount * t)

        # policy parameter at current x (ignore dead paths via mask)
        beta = beta_policy(x).clamp(-beta_clip, beta_clip)

        # analytic H(x, beta): E[a] - λ E[log π]
        Ea = trunc_exp_Ea(beta, spec.A_max)
        Elogpi = trunc_exp_Elogpi(beta, spec.A_max)
        H = Ea - spec.lambda_reg * Elogpi

        # BEL kernel N_t
        sig = spec.sigma(x).clamp_min(1e-6)
        inv_sig = 1.0 / sig

        dWk = dW[:, :, k]
        # update gradX (discrete tangent flow; we ignore derivative of policy)
        gradX = gradX + spec.mu_prime(x) * gradX * dt + spec.sigma_prime(x) * gradX * dWk

        # update S = ∫ σ^{-1} gradX dW
        S = S + inv_sig * gradX * dWk
        Nk = S / max(t, 1e-6)

        # accumulate only while alive
        alive_f = alive.to(x.dtype)
        J  = J  + disc * H * dt * alive_f
        Jp = Jp + disc * H * Nk * dt * alive_f

        # sample action (CRN via U)
        ak = trunc_exp_sample(beta, spec.A_max, U[:, :, k])

        # evolve X
        x = x + (spec.mu(x) - ak) * dt + sig * dWk

        # ruin update
        alive = alive & (x > spec.x_ruin)

    # per-path returns for CI
    J_samps = J.clone()
    J_mean = J.mean(dim=1)
    Jp_mean = Jp.mean(dim=1)
    return J_mean, Jp_mean, J_samps

@torch.no_grad()
def evaluate_policy_crn(
    spec: DividendRuinSpec,
    beta_policy: nn.Module,
    x0: float,
    n_paths: int,
    seed: int,
) -> Tuple[float, float]:
    """
    Returns (mean, std_error) of value at x0 under current policy,
    using fixed CRN (dW,U) for reproducibility and low-variance comparisons.
    """
    device = next(beta_policy.parameters()).device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    K = int(math.ceil(spec.T / spec.dt))
    dW = math.sqrt(spec.dt) * torch.randn(n_paths, K, device=device, generator=gen)
    U  = torch.rand(n_paths, K, device=device, generator=gen)

    x0_t = torch.tensor([x0], device=device)
    J_mean, _, J_samps = simulate_value_and_kernel(spec, beta_policy, x0_t, n_paths, dW=dW, U=U)
    vals = J_samps[0].detach().cpu().numpy()
    mean = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / math.sqrt(n_paths))
    return mean, se

# ------------------------------ Training utilities -----------------------------
def sample_x(batch: int, x_domain: Tuple[float, float], mode: Tuple[str, Dict], device: torch.device) -> torch.Tensor:
    lo, hi = x_domain
    m, opts = mode
    if m == "uniform":
        x_min = float(opts.get("x_min", lo))
        x_max = float(opts.get("x_max", hi))
        return x_min + (x_max - x_min) * torch.rand(batch, device=device)
    if m == "fixed":
        return torch.full((batch,), float(opts["x0"]), device=device)
    raise ValueError("x0_mode must be ('uniform', {...}) or ('fixed', {...})")

def polyak_interpolate_(dst: nn.Module, src: nn.Module, alpha: float) -> None:
    """dst <- (1-alpha)*dst + alpha*src  (in-place)."""
    with torch.no_grad():
        for p_dst, p_src in zip(dst.parameters(), src.parameters()):
            p_dst.data.mul_(1.0 - alpha).add_(p_src.data, alpha=alpha)

# ------------------------------ Main PIA routine -------------------------------
def run_policy_iteration(
    spec: DividendRuinSpec,
    x0_eval: float,
    x_domain: Tuple[float, float],
    x0_mode: Tuple[str, Dict],
    # PI controls
    pia_iters: int = 8,
    # policy evaluation (V and P nets)
    pe_epochs: int = 200,
    pe_batch: int = 256,
    pe_mc_paths: int = 128,
    pe_lr: float = 5e-4,
    pe_weight_decay: float = 0.0,
    pe_patience: int = 15,
    pe_min_delta: float = 1e-4,
    # policy improvement (beta net)
    pi_epochs: int = 200,
    pi_batch: int = 256,
    pi_lr: float = 5e-4,
    pi_weight_decay: float = 0.0,
    pi_patience: int = 15,
    pi_min_delta: float = 1e-4,
    beta_clip: float = 10.0,
    # evaluation for acceptance
    eval_paths: int = 4000,
    eval_seed_base: int = 1234,
    accept_z: float = 1.0,  # ~1-sigma
    # nets
    width: int = 64,
    # misc
    device: Optional[Union[str, torch.device]] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
):
    """
    Returns dict with history + trained models.
    """
    device = get_device(device)
    set_seed(seed)

    beta = BetaPolicy(width=width).to(device)
    V = mlp_1in(width=width, out_dim=1).to(device)   # value approx
    P = mlp_1in(width=width, out_dim=1).to(device)   # derivative approx (BEL target)

    hist = {
        "pi_iter": [],
        "loss_V": [],
        "loss_P": [],
        "loss_beta": [],
        "Vx0_mean": [],
        "Vx0_se": [],
        "accepted_alpha": [],
    }

    # For plotting: track per-epoch losses inside each PI iteration
    per_iter_curves: List[Dict[str, List[float]]] = []

    # fixed evaluation CRN for acceptance comparisons
    # use distinct seeds per PI iteration but same across candidate/old at that iter
    def eval_at_x0(policy_net: nn.Module, iter_idx: int) -> Tuple[float, float]:
        return evaluate_policy_crn(spec, policy_net, x0=x0_eval, n_paths=eval_paths, seed=eval_seed_base + 17 * iter_idx)

    for k in range(pia_iters):
        curves = {"V": [], "P": [], "B": []}

        # -------------------- Policy Evaluation: fit V and P --------------------
        V.train(); P.train(); beta.eval()

        opt_VP = torch.optim.Adam(
            list(V.parameters()) + list(P.parameters()),
            lr=pe_lr,
            weight_decay=pe_weight_decay
        )

        best = float("inf")
        patience = 0

        # small fixed validation set for early stopping stability
        x_val = sample_x(batch=256, x_domain=x_domain, mode=("uniform", {"x_min": x_domain[0], "x_max": x_domain[1]}), device=device)
        # fixed CRN for val so loss is comparable across epochs
        gen = torch.Generator(device=device); gen.manual_seed(999 + k)
        K = int(math.ceil(spec.T / spec.dt))
        dW_val = math.sqrt(spec.dt) * torch.randn(256, pe_mc_paths, K, device=device, generator=gen)
        U_val  = torch.rand(256, pe_mc_paths, K, device=device, generator=gen)

        for ep in range(pe_epochs):
            x_b = sample_x(pe_batch, x_domain, x0_mode, device)

            # MC targets (no grad)
            with torch.no_grad():
                J_b, Jp_b, _ = simulate_value_and_kernel(spec, beta, x_b, pe_mc_paths, beta_clip=beta_clip)

            predV = V(x_b.unsqueeze(-1)).squeeze(-1)
            predP = P(x_b.unsqueeze(-1)).squeeze(-1)

            lossV = torch.mean((predV - J_b) ** 2)
            lossP = torch.mean((predP - Jp_b) ** 2)
            loss = lossV + lossP

            opt_VP.zero_grad(set_to_none=True)
            loss.backward()
            opt_VP.step()

            curves["V"].append(float(lossV.detach().cpu()))
            curves["P"].append(float(lossP.detach().cpu()))

            # early stopping based on fixed validation loss
            with torch.no_grad():
                Jv, Jpv, _ = simulate_value_and_kernel(spec, beta, x_val, pe_mc_paths, dW=dW_val, U=U_val, beta_clip=beta_clip)
                pv = V(x_val.unsqueeze(-1)).squeeze(-1)
                pp = P(x_val.unsqueeze(-1)).squeeze(-1)
                val_loss = torch.mean((pv - Jv) ** 2) + torch.mean((pp - Jpv) ** 2)
                val = float(val_loss.detach().cpu())

            if val < best - pe_min_delta:
                best = val
                patience = 0
                best_state = {"V": V.state_dict(), "P": P.state_dict()}
            else:
                patience += 1
                if patience >= pe_patience:
                    if verbose:
                        print(f"[PI {k}] PE early stop at epoch {ep+1} (val={best:.3e}).")
                    break

        # restore best
        if "best_state" in locals():
            V.load_state_dict(best_state["V"])
            P.load_state_dict(best_state["P"])

        # -------------------- Policy Improvement: train beta_new -----------------
        beta_old = BetaPolicy(width=width).to(device)
        beta_old.load_state_dict(beta.state_dict())

        beta_new = BetaPolicy(width=width).to(device)
        beta_new.load_state_dict(beta.state_dict())
        beta_new.train()
        V.eval(); P.eval()

        opt_B = torch.optim.Adam(beta_new.parameters(), lr=pi_lr, weight_decay=pi_weight_decay)

        bestB = float("inf")
        patienceB = 0

        # fixed validation x for beta fit
        x_valB = x_val.clone()

        for ep in range(pi_epochs):
            x_b = sample_x(pi_batch, x_domain, x0_mode, device)

            with torch.no_grad():
                # beta*(x) = (1 - V'(x))/λ  ≈ (1 - P(x))/λ
                p_hat = P(x_b.unsqueeze(-1)).squeeze(-1)
                beta_star = ((1.0 - p_hat) / spec.lambda_reg).clamp(-beta_clip, beta_clip)

            beta_pred = beta_new(x_b)
            lossB = torch.mean((beta_pred - beta_star) ** 2)

            opt_B.zero_grad(set_to_none=True)
            lossB.backward()
            opt_B.step()

            curves["B"].append(float(lossB.detach().cpu()))

            # early stopping on validation beta-fit
            with torch.no_grad():
                p_val = P(x_valB.unsqueeze(-1)).squeeze(-1)
                beta_star_val = ((1.0 - p_val) / spec.lambda_reg).clamp(-beta_clip, beta_clip)
                beta_val_pred = beta_new(x_valB)
                valB = float(torch.mean((beta_val_pred - beta_star_val) ** 2).detach().cpu())

            if valB < bestB - pi_min_delta:
                bestB = valB
                patienceB = 0
                bestB_state = beta_new.state_dict()
            else:
                patienceB += 1
                if patienceB >= pi_patience:
                    if verbose:
                        print(f"[PI {k}] PI-beta early stop at epoch {ep+1} (valB={bestB:.3e}).")
                    break

        if "bestB_state" in locals():
            beta_new.load_state_dict(bestB_state)

        # -------------------- Acceptance step (monotone, CRN + CI) --------------
        beta_new.eval(); beta_old.eval()

        old_mean, old_se = eval_at_x0(beta_old, k)
        cand_mean, cand_se = eval_at_x0(beta_new, k)

        # We'll accept if cand_mean - old_mean > accept_z * sqrt(se_old^2 + se_cand^2)
        thresh = accept_z * math.sqrt(old_se**2 + cand_se**2)

        accepted_alpha = 0.0
        if cand_mean >= old_mean + thresh:
            beta.load_state_dict(beta_new.state_dict())
            accepted_alpha = 1.0
        else:
            # backtracking on convex combination of parameters
            alphas = [0.5, 0.25, 0.1]
            accepted = False
            for a in alphas:
                beta_mix = BetaPolicy(width=width).to(device)
                beta_mix.load_state_dict(beta_old.state_dict())
                polyak_interpolate_(beta_mix, beta_new, alpha=a)

                mix_mean, mix_se = eval_at_x0(beta_mix, k)
                thresh_mix = accept_z * math.sqrt(old_se**2 + mix_se**2)
                if mix_mean >= old_mean + thresh_mix:
                    beta.load_state_dict(beta_mix.state_dict())
                    accepted_alpha = a
                    cand_mean, cand_se = mix_mean, mix_se
                    accepted = True
                    break
            if not accepted:
                # keep old
                beta.load_state_dict(beta_old.state_dict())
                cand_mean, cand_se = old_mean, old_se

        # record history
        hist["pi_iter"].append(k)
        hist["loss_V"].append(float(np.mean(curves["V"]) if curves["V"] else np.nan))
        hist["loss_P"].append(float(np.mean(curves["P"]) if curves["P"] else np.nan))
        hist["loss_beta"].append(float(np.mean(curves["B"]) if curves["B"] else np.nan))
        hist["Vx0_mean"].append(float(cand_mean))
        hist["Vx0_se"].append(float(cand_se))
        hist["accepted_alpha"].append(float(accepted_alpha))
        per_iter_curves.append(curves)

        if verbose:
            print(
                f"[PI {k}] V(x0) old={old_mean:.6f}±{old_se:.6f}  "
                f"cand={cand_mean:.6f}±{cand_se:.6f}  "
                f"alpha={accepted_alpha}"
            )

    out = dict(
        spec=spec,
        beta=beta,
        V=V,
        P=P,
        hist=hist,
        curves=per_iter_curves,
        device=str(device),
    )
    return out


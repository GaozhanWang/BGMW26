# Copyright (c) 2026 Gaozhan Wang. See LICENSE.
# Modification note: shared plotting functions with generic output filenames.
"""Plots for the historical policy-iteration outputs; no training on import."""
from pathlib import Path
from typing import Dict, Tuple, Union
import numpy as np
import torch
from .policy_iteration import DividendRuinSpec, trunc_exp_Ea, trunc_exp_Z, trunc_exp_sample

def plot_results(out: Dict, x0_eval: float, x_plot: Tuple[float, float] = (-1.0, 2.0), n_grid: int = 200, output_dir: Union[str, Path] = "."):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    hist = out["hist"]
    curves = out["curves"]
    beta = out["beta"]
    spec: DividendRuinSpec = out["spec"]
    device = next(beta.parameters()).device

    # 1) Loss curves per PI iteration
    plt.figure()
    for k, c in enumerate(curves):
        if len(c["V"]) > 0:
            plt.plot(c["V"], label=f"V loss (PI {k})")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Value-fit loss decay per PI iteration")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "value_loss.png", dpi=200)
    plt.close()

    plt.figure()
    for k, c in enumerate(curves):
        if len(c["P"]) > 0:
            plt.plot(c["P"], label=f"P loss (PI {k})")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Historical stopped-BEL derivative-fit loss")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "derivative_loss.png", dpi=200)
    plt.close()

    plt.figure()
    for k, c in enumerate(curves):
        if len(c["B"]) > 0:
            plt.plot(c["B"], label=f"β loss (PI {k})")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Policy (β) supervised-fit loss decay per PI iteration")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "policy_fit_loss.png", dpi=200)
    plt.close()

    # 2) Policy improvement: V(x0) vs PI iteration (with error bars)
    it = np.array(hist["pi_iter"])
    v = np.array(hist["Vx0_mean"])
    se = np.array(hist["Vx0_se"])

    plt.figure()
    plt.errorbar(it, v, yerr=se, fmt="o-")
    plt.xlabel("Policy iteration k")
    plt.ylabel(f"MC value at x0={x0_eval}")
    plt.title("Policy improvement curve ")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "policy_improvement.png", dpi=200)
    plt.close()

    # 3) Expected action E[a|x] under initial and final policies
    #    We'll reconstruct an "initial" policy by re-running PI with pia_iters=0? No;
    #    Instead, approximate initial by beta_net at k=0 if you saved it.
    #    Here we only plot final policy E[a|x] and β(x). (You can save beta snapshots if desired.)
    xg = torch.linspace(x_plot[0], x_plot[1], n_grid, device=device)
    with torch.no_grad():
        betag = beta(xg).clamp(-10.0, 10.0)
        Eag = trunc_exp_Ea(betag, spec.A_max).detach().cpu().numpy()
        betag_np = betag.detach().cpu().numpy()
        x_np = xg.detach().cpu().numpy()

    plt.figure()
    plt.plot(x_np, betag_np)
    plt.xlabel("x")
    plt.ylabel("β(x)")
    plt.title("Final policy parameter β(x)")
    plt.tight_layout()
    plt.savefig(output_dir / "policy_parameter.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x_np, Eag)
    plt.xlabel("x")
    plt.ylabel("E[a | x]")
    plt.title("Final policy expected action")
    plt.tight_layout()
    plt.savefig(output_dir / "mean_dividend.png", dpi=200)
    plt.close()

    # 4) Optional: value function estimate V(x) on a grid
    Vnet = out["V"].to(device).eval()
    with torch.no_grad():
        Vg = Vnet(xg.unsqueeze(-1)).squeeze(-1).detach().cpu().numpy()
    plt.figure()
    plt.plot(x_np, Vg)
    plt.xlabel("x")
    plt.ylabel("V_net(x)")
    plt.title("Value network after final training step")
    plt.tight_layout()
    plt.savefig(output_dir / "value_network.png", dpi=200)
    plt.close()

    print(f"Saved training diagnostic plots to {output_dir}")

def plot_policy_pdf_by_wealth(out: dict, x_levels=None, out_png: str = "policy_density.png"):
    """Plot the learned dividend-policy density at fixed wealth levels."""
    if x_levels is None:
        x_levels = [0.5, 1.0, 1.5]
    import matplotlib.pyplot as plt
    import torch
    import numpy as np

    # Extract needed components from the training output dict
    beta_policy = out["beta"]
    spec = out["spec"]
    device = next(beta_policy.parameters()).device
    
    A = spec.A_max
    
    # Create an action grid [0, A_max] for the x-axis of our PDF plot
    a_grid = torch.linspace(0.0, A, 200, device=device)
    a_np = a_grid.cpu().numpy()

    plt.figure(figsize=(7.0, 4.2))

    # Evaluate the policy without tracking gradients
    beta_policy.eval()
    with torch.no_grad():
        for x_val in x_levels:
            # 1. Feed the fixed wealth level into the policy network to get beta(x)
            x_t = torch.tensor([x_val], device=device, dtype=torch.float32)
            beta = beta_policy(x_t)
            
            # 2. Calculate the normalization constant Z(beta)
            # Using your existing safe Z helper function
            Z = trunc_exp_Z(beta, A)
            
            # 3. Compute the PDF: pi(a|x) = exp(beta * a) / Z
            # beta is shape [1], a_grid is shape [200], so beta * a_grid broadcasts naturally
            pdf_vals = torch.exp(beta * a_grid) / Z
            
            # 4. Plot
            pdf_np = pdf_vals.cpu().numpy()
            plt.plot(a_np, pdf_np, label=f"Wealth x = {x_val}")

    plt.xlabel("Dividend Level (a)")
    plt.ylabel(r"Probability Density $\pi(a|x)$")
    plt.title("Learned Dividend Distribution by Wealth Level")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    
    print(f"Saved policy PDF plot to {out_png}")

@torch.no_grad()
def plot_wealth_quantiles(out: dict, x0_start: float = 1.0, n_paths: int = 5000, out_png: str = "wealth_quantiles.png"):
    """
    Simulates the wealth process under the returned historical policy and plots 
    the 5%, 50%, and 95% quantiles over time.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import math

    spec = out["spec"]
    beta_policy = out["beta"]
    device = next(beta_policy.parameters()).device

    T = spec.T
    dt = spec.dt
    K = int(math.ceil(T / dt))

    # Initialize tensor to store paths. Shape: [n_paths, K+1]
    X_paths = torch.zeros((n_paths, K + 1), device=device)
    x = torch.full((n_paths,), float(x0_start), device=device)
    X_paths[:, 0] = x

    alive = (x > spec.x_ruin)
    
    # Forward simulation loop
    beta_policy.eval()
    for k in range(K):
        # 1. Get policy parameter beta(x)
        beta = beta_policy(x).clamp(-10.0, 10.0)
        
        # 2. Sample dividend action from the truncated exponential policy
        U = torch.rand(n_paths, device=device)
        ak = trunc_exp_sample(beta, spec.A_max, U)
        
        # 3. Environment step (Euler-Maruyama)
        dW = math.sqrt(dt) * torch.randn(n_paths, device=device)
        x_new = x + (spec.mu(x) - ak) * dt + spec.sigma(x) * dW
        
        # 4. Check ruin boundary (if ruined, stay at 0)
        alive = alive & (x_new > spec.x_ruin)
        x = torch.where(alive, x_new, torch.zeros_like(x_new))
        
        # 5. Record state
        X_paths[:, k + 1] = x

    # Compute quantiles across the batch dimension
    X_paths_np = X_paths.cpu().numpy()
    q5 = np.percentile(X_paths_np, 5, axis=0)
    q50 = np.percentile(X_paths_np, 50, axis=0)
    q95 = np.percentile(X_paths_np, 95, axis=0)
    
    t_grid = np.linspace(0, T, K + 1)

    # Plotting
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(t_grid, q50, label='Median Wealth (50%)', color='blue')
    plt.plot(t_grid, q95, label='95% Quantile', color='green', linestyle='--')
    plt.plot(t_grid, q5, label='5% Quantile', color='orange', linestyle='--')
    
    # Fill between the 5% and 95% quantiles to show the distribution spread
    plt.fill_between(t_grid, q5, q95, color='gray', alpha=0.15, label='90% Probability Region')

    # Mark the ruin boundary
    plt.axhline(y=spec.x_ruin, color='red', linestyle=':', label='Ruin Boundary (x=0)')

    plt.xlabel('Time (t)')
    plt.ylabel('Wealth Process ($X_t$)')
    plt.title(f'Simulated Learned-Policy Wealth Trajectories (Starting $X_0$ = {x0_start})')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    
    print(f"Saved wealth quantiles plot to {out_png}")

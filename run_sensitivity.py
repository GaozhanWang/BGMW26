#!/usr/bin/env python3
# Copyright (c) 2026 Gaozhan Wang. All rights reserved.
# See LICENSE for permissions and conditions.
"""Small fixed-policy sensitivity study accompanying the BEL consistency diagnostic.

Run beside ``fixed_policy.py``::

    python run_sensitivity.py
    python run_sensitivity.py --quick --output-dir smoke_results
    python run_sensitivity.py --plot-only --output-dir results/sensitivity

This tests deterministic horizon truncation and neural initialization/width
sensitivity for ONE frozen state-independent policy with constant coefficients.
It does not test Monte Carlo sampling, Euler timesteps, entropy-policy changes,
or the convergence/robustness of the full policy-iteration algorithm.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

# Small dense CPU networks: avoid severe BLAS oversubscription by default.
# Explicit thread settings supplied by the caller are respected.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

import fixed_policy as core


SCOPE = ("Frozen state-independent Gibbs policy; constant coefficients; exact "
         "killed-survival/gradient targets with deterministic time quadrature. "
         "Across-seed SD measures neural initialization variability only.")
METRICS = ("V_rmse", "V_relative_l2", "P_rmse", "P_relative_l2",
           "dV_rmse", "dV_relative_l2", "consistency_rmse",
           "consistency_relative_l2")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative_l2(error, reference):
    denominator = float(np.linalg.norm(reference))
    if denominator == 0:
        raise ValueError("Relative L2 is undefined for a zero reference.")
    return float(np.linalg.norm(error) / denominator)


def errors(value, derivative, autodiff, ref_value, ref_derivative):
    result = {}
    for name, estimate, reference in (("V", value, ref_value),
                                       ("P", derivative, ref_derivative),
                                       ("dV", autodiff, ref_derivative)):
        result[name + "_rmse"] = core.rms(estimate - reference)
        result[name + "_relative_l2"] = relative_l2(estimate - reference, reference)
    result["consistency_rmse"] = core.rms(derivative - autodiff)
    # Normalize both derivative comparisons by the same exact derivative norm.
    result["consistency_relative_l2"] = relative_l2(
        derivative - autodiff, ref_derivative)
    return result


def study_targets(cfg, train, constants, grid, output):
    ref_value, ref_derivative = core.bvp_reference(grid, cfg, constants)
    rows = []
    for horizon in train["horizons"]:
        problem = replace(cfg, horizon=horizon)
        value, derivative = core.representation_targets(
            grid, problem, train["config"], constants)
        bound = abs(constants["regularized_reward"]) / cfg.discount * np.exp(
            -cfg.discount * horizon)
        rows.append({"horizon": horizon,
                     "V_rmse": core.rms(value - ref_value),
                     "V_relative_l2": relative_l2(value - ref_value, ref_value),
                     "P_rmse": core.rms(derivative - ref_derivative),
                     "P_relative_l2": relative_l2(derivative - ref_derivative, ref_derivative),
                     "V_max_error": float(np.max(np.abs(value - ref_value))),
                     "V_tail_bound": float(bound)})
    save_csv(output / "horizon_sensitivity.csv", rows)

    target_value, target_derivative = core.representation_targets(
        grid, cfg, train["config"], constants)
    training_grid = core.make_training_grid(cfg, train["config"])
    full_grid = np.unique(np.concatenate((grid, training_grid)))
    coarse = core.representation_targets(full_grid, cfg, train["config"], constants)
    refined = core.representation_targets(
        full_grid, cfg,
        replace(train["config"], quadrature_nodes=2 * train["config"].quadrature_nodes),
        constants)
    verification = {
        "T2_value_max_error": float(np.max(abs(target_value - ref_value))),
        "T2_derivative_max_error": float(np.max(abs(target_derivative - ref_derivative))),
        "quadrature_nodes": train["config"].quadrature_nodes,
        "refined_quadrature_nodes": 2 * train["config"].quadrature_nodes,
        "refinement_max_value_difference": float(np.max(abs(coarse[0] - refined[0]))),
        "refinement_max_derivative_difference": float(np.max(abs(coarse[1] - refined[1]))),
        "refinement_domain": [float(full_grid[0]), float(full_grid[-1])],
        "tail_bound_check_tolerance": 1e-8,
        "value_tail_bounds_pass": all(r["V_max_error"] <= r["V_tail_bound"] + 1e-8
                                      for r in rows),
        "T2_reference_check_pass": bool(np.max(abs(target_derivative - ref_derivative)) < 1e-6),
        "quadrature_refinement_check_pass": bool(np.max(abs(coarse[1] - refined[1])) < 1e-5),
    }
    save_json(output / "verification.json", verification)
    return rows, (ref_value, ref_derivative), verification


def train_one(cfg, train, seed, grid, references, constants, sources, checkpoint):
    config = {"problem": asdict(cfg), "training": asdict(train),
              "seed": seed, "sources": sources}
    config_hash = digest(config)
    metadata_path = checkpoint.with_suffix(".json")
    array_path = checkpoint.with_suffix(".npz")
    if metadata_path.exists() and array_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("config_hash") == config_hash:
            with np.load(array_path, allow_pickle=False) as arrays:
                if str(arrays["config_hash"].item()) != config_hash:
                    raise ValueError(f"Checkpoint hash disagreement: {array_path}")
            print(f"Resume width={train.hidden_width}, seed={seed}", flush=True)
            return metadata["metrics"]
        print(f"Configuration changed; recompute {checkpoint.name}", flush=True)

    started = time.perf_counter()
    training_grid = core.make_training_grid(cfg, train)
    value_target, derivative_target = core.representation_targets(
        training_grid, cfg, train, constants)
    with threadpool_limits(limits=1):
        q_network = core.fit_network(training_grid, value_target / training_grid,
                                     cfg, train, seed)
        p_network = core.fit_network(training_grid, derivative_target,
                                     cfg, train, seed + 100_003)
    quotient, quotient_derivative = core.network_output_and_derivative(q_network, grid, cfg)
    value = grid * quotient
    autodiff = quotient + grid * quotient_derivative
    derivative, _ = core.network_output_and_derivative(p_network, grid, cfg)
    row = {"width": train.hidden_width, "layers": train.hidden_layers, "seed": seed,
           **errors(value, derivative, autodiff, *references),
           "elapsed_seconds": time.perf_counter() - started,
           "Q_iterations": int(q_network.n_iter_), "P_iterations": int(p_network.n_iter_),
           "Q_training_loss": float(q_network.loss_), "P_training_loss": float(p_network.loss_),
           "Q_iteration_cap_reached": bool(q_network.n_iter_ >= train.max_iterations),
           "P_iteration_cap_reached": bool(p_network.n_iter_ >= train.max_iterations)}
    # Write arrays first, then the completed metadata: partial runs never count as done.
    temporary = array_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, config_hash=np.array(config_hash), grid=grid,
                        reference_value=references[0], reference_derivative=references[1],
                        V=value, P=derivative, dV=autodiff,
                        training_grid=training_grid, value_target=value_target,
                        derivative_target=derivative_target)
    temporary.replace(array_path)
    save_json(metadata_path, {"config_hash": config_hash, "config": config, "metrics": row})
    print(f"width={train.hidden_width}, seed={seed}: "
          f"P RMSE={row['P_rmse']:.3e}, dV RMSE={row['dV_rmse']:.3e}", flush=True)
    return row


def summarize(rows, widths):
    summary = []
    for width in widths:
        group = [r for r in rows if r["width"] == width]
        row = {"width": width, "layers": group[0]["layers"], "replicates": len(group)}
        for metric in METRICS + ("elapsed_seconds",):
            values = np.asarray([r[metric] for r in group])
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        row["runs_reaching_iteration_cap"] = sum(
            r["Q_iteration_cap_reached"] or r["P_iteration_cap_reached"] for r in group)
        summary.append(row)
    return summary


def save_latex(path, horizon_rows, summary):
    lines = [r"% Relative L2 entries are percentages; SD is sample SD across initialization seeds.",
             r"\begin{tabular}{c|cc}", r"$T$ & $V$ RelL2 (\%) & $V_x$ RelL2 (\%) \\",
             r"\hline"]
    for row in horizon_rows:
        lines.append(f"{row['horizon']:g} & {100*row['V_relative_l2']:.3e} & "
                     f"{100*row['P_relative_l2']:.3e} " + r"\\")
    lines += [r"\end{tabular}", "", r"\begin{tabular}{c|ccc}",
              r"Width & $V$ RelL2 (\%) & $P$ RelL2 (\%) & $\partial_x V$ RelL2 (\%) \\",
              r"\hline"]
    for row in summary:
        cells = [str(row["width"])]
        for key in ("V_relative_l2", "P_relative_l2", "dV_relative_l2"):
            cells.append(f"${100*row[key+'_mean']:.3f} \\pm {100*row[key+'_sd']:.3f}$")
        lines.append(" & ".join(cells) + " " + r"\\")
    lines += [r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_results(output, horizon_rows, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
        horizons = [r["horizon"] for r in horizon_rows]
        axes[0].semilogy(horizons,
                         [max(100*r["V_relative_l2"], 1e-12) for r in horizon_rows],
                         marker="o", color="#1f77b4", label=r"Value $V_T$", linewidth=1.8)
        axes[0].set(xlabel="Truncation horizon T", ylabel="Relative L2 error (%)",
                    title="(a) truncation error")
        axes[0].set_xticks(horizons)
        positions = np.arange(len(summary))
        for offset, key, label, color in ((-.18, "P_rmse", r"Independent $P$", "#1f77b4"),
                                          (.18, "dV_rmse", r"Autodiff $\partial_x V$", "#d62728")):
            axes[1].bar(positions+offset, [r[key+"_mean"] for r in summary], width=.34,
                        yerr=[r[key+"_sd"] for r in summary], capsize=4,
                        label=label, color=color, alpha=.85)
        axes[1].set_xticks(positions, [str(r["width"]) for r in summary])
        axes[1].set(xlabel="Hidden-layer width", ylabel="Derivative RMSE",
                    title="(b) NN Initialization sensitivity")
        for ax in axes:
            ax.grid(alpha=.23, axis="y")
            ax.set_axisbelow(True)
            ax.legend(frameon=False)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"fixed_policy_sensitivity.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", type=float, nargs="+", default=[.2, .5, 1., 2.])
    parser.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot-only", action="store_true",
                        help="Replot saved CSV summaries without training or changing numerical data")
    parser.add_argument("--output-dir", type=Path, default=Path("results/sensitivity"))
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--quadrature-nodes", type=int)
    args = parser.parse_args()
    for name in ("horizons", "widths"):
        values = getattr(args, name)
        if any(v <= 0 or not np.isfinite(v) for v in values):
            parser.error(f"--{name} must contain positive finite values")
        setattr(args, name, sorted(set(values)))
    for name in ("replicates", "max_iterations", "quadrature_nodes"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0 or args.seed + 100_003 + 997*((args.replicates or 5)-1) >= 2**32:
        parser.error("Seed sequence must fit uint32")
    return args


def main():
    args = parse_args()
    if args.plot_only:
        output = args.output_dir.resolve()
        tables = []
        integer_fields = {"width", "layers", "replicates", "runs_reaching_iteration_cap"}
        for filename in ("horizon_sensitivity.csv", "neural_sensitivity_summary.csv"):
            with (output / filename).open(newline="", encoding="utf-8") as handle:
                rows = [{key: int(value) if key in integer_fields else float(value)
                         for key, value in row.items()} for row in csv.DictReader(handle)]
            if not rows:
                raise ValueError(f"Cannot plot empty table: {output / filename}")
            tables.append(rows)
        plot_results(output, *tables)
        print(f"Replotted saved results without training: {output}", flush=True)
        return
    cfg = core.ProblemConfig(horizon=2.)
    train = core.TrainingConfig.quick() if args.quick else core.TrainingConfig()
    train.base_seed = args.seed
    for name in ("replicates", "max_iterations", "quadrature_nodes"):
        if getattr(args, name) is not None:
            setattr(train, name, getattr(args, name))
    if not 0 < cfg.x_train_lo < cfg.x_plot_lo < cfg.x_plot_hi <= cfg.x_train_hi:
        raise ValueError("Require a positive evaluation domain inside the training domain")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    sources = {"companion_sha256": file_hash(core.__file__), "wrapper_sha256": file_hash(__file__)}
    constants = core.fixed_policy_constants(cfg)
    grid = np.linspace(cfg.x_plot_lo, cfg.x_plot_hi, train.plot_points)
    study = {"horizons": args.horizons, "config": train}
    horizon_rows, references, verification = study_targets(cfg, study, constants, grid, output)
    rows = []
    for width in args.widths:
        current = replace(train, hidden_width=width)
        for index in range(train.replicates):
            seed = train.base_seed + 997*index
            rows.append(train_one(cfg, current, seed, grid, references, constants, sources,
                                  checkpoint_dir / f"width_{width}_seed_{seed}"))
            save_csv(output / "neural_sensitivity_runs.csv", rows)
    summary = summarize(rows, args.widths)
    save_csv(output / "neural_sensitivity_summary.csv", summary)
    save_latex(output / "sensitivity_tables.tex", horizon_rows, summary)
    plot_results(output, horizon_rows, summary)
    import matplotlib
    payload = {"scope": SCOPE, "problem": asdict(cfg), "training_base": asdict(train),
               "horizons": args.horizons, "widths": args.widths, "quick_profile": args.quick,
               "constants": constants, "sources": sources,
               "environment": {"python": sys.version, "platform": platform.platform(),
                               "numpy": np.__version__, "scipy": scipy.__version__,
                               "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                               "fit_threadpool_limit": 1,
                               "threads": {k: os.environ.get(k) for k in
                                           ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
               "metrics": {"relative_l2": "Euclidean error norm / exact-reference norm on uniform diagnostic grid",
                           "consistency_relative_l2": "Norm(P-dV) / norm(exact derivative)",
                           "dispersion": "Sample SD across neural initialization seeds; not a confidence interval",
                           "training": "Independent Q and P; V=xQ; Q fitted to V_target/x; L-BFGS; no consistency penalty",
                           "optimizer_note": "Iteration cap flag is a diagnostic, not a guarantee of convergence; companion suppresses optimizer warnings"},
               "verification": verification, "summary": summary}
    save_json(output / "study_summary.json", payload)
    for key, value in verification.items():
        if key.endswith("_pass") and not value:
            print(f"CHECK REQUIRES REVIEW: {key}", flush=True)
    print(f"Completed fixed-policy study: {output}", flush=True)


if __name__ == "__main__":
    main()

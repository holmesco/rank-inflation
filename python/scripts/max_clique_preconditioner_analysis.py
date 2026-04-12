"""
Max Clique Preconditioner Analysis Script
========================================
Runs the max clique pipeline (SDP solve + analytic center certification) across
outlier ratios, comparing three preconditioners:
  - Diagonal (MFCG_DP)
  - SparseLDLT (MFCG_LRP)
  - SparseLDLT_ZL (MFCG_LRP)

Usage:
    python max_clique_preconditioner_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from examples.max_clique.max_clique import generate_dataset, MaxCliqueProblem
from ranktools import AnalyticCenterParams, LinearSolverType, LowRankPrecondMethod

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_OUTRAT = 10
N_TRIALS_PER_OUTRAT = 5
OUTRAT_MIN = 0.1
OUTRAT_MAX = 0.90

M_ASSOC = 130
N1 = 130
N2O = 13
SIGMA = 0.01
PCFILE = "/workspace/python/examples/bun10k.ply"

SEED = 0

# ---------------------------------------------------------------------------
# Preconditioners to compare
# ---------------------------------------------------------------------------

PRECONDITIONERS = {
    # "Diagonal": (LinearSolverType.MFCG_DP, None),
    "SparseLDLT": (LinearSolverType.MFCG_LRP, LowRankPrecondMethod.SparseLDLT),
    "SparseLDLT_ZL": (
        LinearSolverType.MFCG_LRP,
        LowRankPrecondMethod.SparseLDLT_ZL,
    ),
}


def make_ac_params(
    lin_solver: LinearSolverType,
    precond_method: LowRankPrecondMethod | None = None,
) -> AnalyticCenterParams:
    """Create default AnalyticCenterParams for a solver/preconditioner pair."""
    params = AnalyticCenterParams()
    params.verbose = True
    params.lin_solver = lin_solver
    params.lin_solve_max_iter = 200
    params.lin_solve_tol = 1e-4

    params.lrp_params.tau = 1e-5
    params.delta_init = 1e-7
    params.delta_min = 1e-8

    params.rescale_lin_sys = False
    params.perturb_constraints = False
    params.adaptive_perturb = False

    if precond_method is not None:
        params.lrp_params.method = precond_method

    return params


def run_analysis(
    n_outrat: int = N_OUTRAT,
    outrat_min: float = OUTRAT_MIN,
    outrat_max: float = OUTRAT_MAX,
    n_trials_per_outrat: int = N_TRIALS_PER_OUTRAT,
    seed: int = SEED,
) -> pd.DataFrame:
    """Run sweep over outlier ratio and compare AC preconditioners."""
    np.random.seed(seed)

    outrats = np.logspace(np.log10(outrat_min), np.log10(outrat_max), n_outrat)

    T_21 = np.eye(4)
    T_21[:3, :3] = Rotation.random().as_matrix()
    T_21[:3, 3] = np.random.uniform(-5, 5, size=3)

    records: list[dict] = []

    for i_outrat, outrat in enumerate(outrats):
        print(f"\n{'=' * 60}")
        print(f"Outlier ratio {i_outrat + 1}/{n_outrat}  |  outrat = {outrat:.4f}")
        print(f"{'=' * 60}")

        for i_trial in range(n_trials_per_outrat):
            print(f"\n  Trial {i_trial + 1}/{n_trials_per_outrat}")

            clipper, _ = generate_dataset(PCFILE, M_ASSOC, N1, N2O, outrat, SIGMA, T_21)

            # Solve SDP once per trial
            prob = MaxCliqueProblem(clipper, params=make_ac_params(LinearSolverType.MFCG_DP))
            _, u_sdp, sdp_rank, sdp_time, eig_ratio = prob.solve_sdp()

            n_constraints = len(prob.As)
            sdp_cost = -(u_sdp.T @ prob.M @ u_sdp).item()

            for precond_name, (lin_solver, precond_method) in PRECONDITIONERS.items():
                print(f"\n--- Preconditioner: {precond_name} ---")

                params = make_ac_params(lin_solver, precond_method)
                prob_precond = MaxCliqueProblem(clipper, params=params)
                result = prob_precond.certify_candidate(u_sdp, cost=sdp_cost)

                records.append(
                    {
                        "outrat": outrat,
                        "trial": i_trial,
                        "preconditioner": precond_name,
                        "lin_solver": lin_solver.name,
                        "n_constraints": n_constraints,
                        "sdp_time_s": sdp_time,
                        "sdp_rank": sdp_rank,
                        "ac_time_s": result.solver_time,
                        "certified": result.certified,
                        "min_eig": result.min_eig,
                        "complementarity": result.complementarity,
                        "eig_ratio": eig_ratio,
                    }
                )

    return pd.DataFrame(records)


DEFAULT_CSV = "/workspace/python/results/max_clique_preconditioner_analysis.csv"


def plot_runtime_vs_outrat(csv_path: str = DEFAULT_CSV) -> None:
    """Plot runtime versus outlier ratio for the three preconditioners."""
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 5))

    outrats = np.array(sorted(df["outrat"].unique()))
    has_multi_trial = "trial" in df.columns and (df.groupby("outrat")["trial"].nunique().max() > 1)

    if has_multi_trial:
        x = np.arange(len(outrats))
        series = []

        sdp = df.drop_duplicates(subset=["outrat", "trial"])[["outrat", "sdp_time_s"]]
        series.append(("Interior Point", [sdp[sdp["outrat"] == o]["sdp_time_s"].values for o in outrats]))

        for name in PRECONDITIONERS:
            sub = df[df["preconditioner"] == name]
            series.append((name, [sub[sub["outrat"] == o]["ac_time_s"].values for o in outrats]))

        width = 0.8 / max(len(series), 1)
        offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * width

        for i, (label, data) in enumerate(series):
            pos = x + offsets[i]
            bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True, showfliers=False)
            color = f"C{i}"
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.35)
                patch.set_edgecolor(color)
            for median in bp["medians"]:
                median.set_color(color)
                median.set_linewidth(1.5)
            ax.plot([], [], color=color, label=label)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{o:.3f}" for o in outrats], rotation=45)
    else:
        sdp = df.drop_duplicates(subset="outrat")[["outrat", "sdp_time_s"]].sort_values("outrat")
        ax.plot(sdp["outrat"], sdp["sdp_time_s"], marker="s", label="Interior Point")

        markers = {"Diagonal": "^", "SparseLDLT": "D", "SparseLDLT_ZL": "o"}
        for name, marker in markers.items():
            sub = df[df["preconditioner"] == name].sort_values("outrat")
            if sub.empty:
                continue
            ax.plot(sub["outrat"], sub["ac_time_s"], marker=marker, label=name)

    ax.set_xlabel("Outlier ratio")
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Runtime by Outlier Ratio (Preconditioner Comparison)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(csv_path.replace(".csv", "_runtime.png"), dpi=150)
    plt.show()


def plot_runtime_scatter_vs_constraints(csv_path: str = DEFAULT_CSV) -> None:
    """Scatter runtime against number of constraints."""
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 5))

    sdp = df.drop_duplicates(subset=["outrat", "trial"])
    ax.scatter(sdp["n_constraints"], sdp["sdp_time_s"], marker="s", alpha=0.8, label="Interior Point")

    markers = {"Diagonal": "^", "SparseLDLT": "D", "SparseLDLT_ZL": "o"}
    for name, marker in markers.items():
        sub = df[df["preconditioner"] == name]
        if sub.empty:
            continue
        ax.scatter(sub["n_constraints"], sub["ac_time_s"], marker=marker, alpha=0.7, label=name)

    ax.set_xlabel("Number of constraints")
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Runtime Scatter vs. Number of Constraints")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(csv_path.replace(".csv", "_runtime_scatter.png"), dpi=150)
    plt.show()


def generate_plots(out_path: str) -> None:
    plot_runtime_vs_outrat(out_path)
    plot_runtime_scatter_vs_constraints(out_path)


if __name__ == "__main__":
    df = run_analysis()

    print("\n\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))

    out_path = "/workspace/python/results/max_clique_preconditioner_analysis.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    generate_plots(out_path)

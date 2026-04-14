"""
Max Clique Analysis Script
===========================
Runs the max clique pipeline (SDP solve + analytic center certification) across
a sweep of outlier ratios and linear solver types, collecting timing and
problem-size statistics into a pandas DataFrame.

This script will use MOSEK to solve for the IP solution (which is rank 1) 

Usage:
    python max_clique_analysis.py
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from examples.max_clique.max_clique import generate_dataset, MaxCliqueProblem
from ranktools import AnalyticCenterParams, LinearSolverType, LowRankPrecondMethod

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Number of outlier-ratio values to sweep
N_OUTRAT = 10

# Number of randomized trials to run per outlier ratio
N_TRIALS_PER_OUTRAT = 10

# Range of outlier ratios (log-spaced between these bounds)
OUTRAT_MIN = 0.1
OUTRAT_MAX = 0.90

# Number of trials (from the low-outrat end) for 
# which slow solver is also run.  Set to N_OUTRAT 
# to run them for every trial.
N_FULL_MAT = 0

# Dataset parameters (mirroring max_clique.py __main__)
M_ASSOC = 130       # total number of associations
N1 = 130            # model points in view 1
N2O = 13            # outlier points in view 2
SIGMA = 0.01        # uniform noise [m]
PCFILE = "/workspace/python/examples/bun10k.ply"

# Random seed for reproducibility
SEED = 0

# ---------------------------------------------------------------------------
# Solver bookkeeping
# ---------------------------------------------------------------------------

# All solver types to benchmark.  MFCG variants are always run; CG and LDLT
# are restricted to the first N_FULL_MAT outlier-ratio trials.
ALL_SOLVERS = {
    "MFCG_LRP": LinearSolverType.MFCG_LRP,
    # "MFCG_DP":  LinearSolverType.MFCG_DP,
    # "CG":       LinearSolverType.CG,
    # "LDLT":     LinearSolverType.LDLT,
}

SLOW_SOLVERS = {"MFCG_DP"}


def make_ac_params(solver_type: LinearSolverType) -> AnalyticCenterParams:
    """Return a default AnalyticCenterParams with the given linear solver."""
    params = AnalyticCenterParams()
    params.verbose = True
    params.lin_solver = solver_type
    params.lin_solve_max_iter = 400
    params.lin_solve_tol = 1e-5
    params.lrp_params.tau = 1e-6
    params.delta_init = 1e-6
    params.delta_min = 1e-8
    # turn off rescaling
    params.rescale_lin_sys = False
    # Turn off perturbations:
    params.perturb_constraints = False
    params.perturb_cost = False
    params.adaptive_perturb = True
    params.cost_perturb = 1e-4
    # Set preconditioner
    params.lrp_params.method = LowRankPrecondMethod.SparseLDLT
    
    return params


def run_analysis(
    n_outrat: int = N_OUTRAT,
    outrat_min: float = OUTRAT_MIN,
    outrat_max: float = OUTRAT_MAX,
    n_trials_per_outrat: int = N_TRIALS_PER_OUTRAT,
    n_full_mat: int = N_FULL_MAT,
    seed: int = SEED,
) -> pd.DataFrame:
    """Run the full max-clique analysis sweep.

    Parameters
    ----------
    n_outrat : int
        Number of outlier-ratio values to test.
    outrat_min, outrat_max : float
        Bounds of the log-spaced outlier-ratio sweep.
    n_trials_per_outrat : int
        Number of randomized trials to run for each outlier ratio.
    n_full_mat : int
        Number of trials (from the low end) for which CG and LDLT solvers
        are also run.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per (outrat, solver) combination with columns:
        outrat, solver, n_constraints, sdp_time_s, ac_time_s,
        certified, min_eig, complementarity, sdp_rank.
    """
    np.random.seed(seed)

    # Log-spaced outlier ratios
    outrats = np.logspace(np.log10(outrat_min), np.log10(outrat_max), n_outrat)

    # Random ground-truth pose (fixed across trials for consistency)
    T_21 = np.eye(4)
    T_21[:3, :3] = Rotation.random().as_matrix()
    T_21[:3, 3] = np.random.uniform(-5, 5, size=3)

    records: list[dict] = []

    for i_outrat, outrat in enumerate(outrats):
        print(f"\n{'='*60}")
        print(f"Outlier ratio {i_outrat+1}/{n_outrat}  |  outrat = {outrat:.4f}")
        print(f"{'='*60}")

        for i_trial in range(n_trials_per_outrat):
            print(f"\n  Trial {i_trial+1}/{n_trials_per_outrat}")

            # ---- Generate dataset --------------------------------------------
            clipper, Agt = generate_dataset(
                PCFILE, M_ASSOC, N1, N2O, outrat, SIGMA, T_21
            )

            # ---- Solve the SDP relaxation ------------------------------------
            # Use MFCG params for the problem setup (solver only matters for AC)
            prob = MaxCliqueProblem(clipper, params=make_ac_params(LinearSolverType.MFCG_DP))

            X_sdp, u_sdp, sdp_rank, sdp_time, eig_ratio = prob.solve_sdp()

            n_constraints = len(prob.As)
            sdp_cost = -(u_sdp.T @ prob.M @ u_sdp).item()

            # ---- Certify with each requested solver --------------------------
            for solver_name, solver_enum in ALL_SOLVERS.items():
                # Skip expensive full-matrix solvers beyond the first n_full_mat
                if solver_name in SLOW_SOLVERS and i_outrat >= n_full_mat:
                    continue

                print(f"\n--- Solver: {solver_name} ---")
                params = make_ac_params(solver_enum)
                prob_solver = MaxCliqueProblem(clipper, params=params)

                result = prob_solver.certify_candidate(u_sdp, cost=sdp_cost)

                records.append(
                    {
                        "outrat": outrat,
                        "trial": i_trial,
                        "solver": solver_name,
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

    df = pd.DataFrame(records)
    return df


DEFAULT_CSV = "/workspace/python/results/max_clique_analysis.csv"


def plot_runtime_vs_constraints(csv_path: str = DEFAULT_CSV) -> None:
    """Load results CSV and plot runtime vs number of constraints.

        Five series are shown:
      - Interior Point  (SDP solve time, one per outrat)
      - LDLT            (analytic-center time)
      - CG              (analytic-center time)
            - MFCG_LRP        (analytic-center time)
            - MFCG_DP         (analytic-center time)

    Parameters
    ----------
    csv_path : str
        Path to the CSV produced by ``run_analysis``.
    """
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 5))

    has_multi_trial = "trial" in df.columns and (df.groupby("outrat")["trial"].nunique().max() > 1)

    if has_multi_trial:
        outrats = np.array(sorted(df["outrat"].unique()))
        n_outrat = len(outrats)
        x = np.arange(n_outrat)

        # --- Interior Point (SDP) ---
        sdp = df.drop_duplicates(subset=["outrat", "trial"])[["outrat", "sdp_time_s"]]
        sdp_data = [sdp[sdp["outrat"] == o]["sdp_time_s"].values for o in outrats]

        series = [("Interior Point", sdp_data)]
        for solver_name in ["LDLT", "CG", "MFCG_LRP", "MFCG_DP"]:
            sub = df[df["solver"] == solver_name]
            if sub.empty:
                continue
            series.append(
                (
                    solver_name,
                    [sub[sub["outrat"] == o]["ac_time_s"].values for o in outrats],
                )
            )

        width = 0.8 / max(len(series), 1)
        offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * width

        for i, (label, data) in enumerate(series):
            pos = x + offsets[i]
            bp = ax.boxplot(
                data,
                positions=pos,
                widths=width * 0.9,
                patch_artist=True,
                showfliers=False,
            )
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
        ax.set_xlabel("Outlier ratio")

    else:
        # --- Interior Point (SDP) ---
        # One SDP time per outrat; take the first occurrence per outrat
        sdp = df.drop_duplicates(subset="n_constraints")["n_constraints"].to_frame()
        sdp["sdp_time_s"] = df.drop_duplicates(subset="n_constraints")["sdp_time_s"].values
        sdp = sdp.sort_values("n_constraints")
        ax.plot(
            sdp["n_constraints"],
            sdp["sdp_time_s"],
            marker="s",
            label="Interior Point",
        )

        # --- AC solvers ---
        solver_styles = {
            "LDLT":     {"marker": "^"},
            "CG":       {"marker": "o"},
            "MFCG_LRP": {"marker": "D"},
            "MFCG_DP":  {"marker": "v"},
        }
        for solver_name, style in solver_styles.items():
            sub = df[df["solver"] == solver_name].sort_values("n_constraints")
            if sub.empty:
                continue
            ax.plot(
                sub["n_constraints"],
                sub["ac_time_s"],
                label=solver_name,
                **style,
            )

        ax.set_xlabel("Number of constraints")
        ax.set_xscale("log")
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Runtime by Outlier Ratio (Lovasz Theta SDP)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(csv_path.replace(".csv", "_runtime_box.png"), dpi=150)
    plt.show()


def plot_eig_ratio_vs_constraints(csv_path: str = DEFAULT_CSV) -> None:
    """Load results CSV and plot eigenvalue ratio vs number of constraints.

    One series is shown:
      - Eigenvalue Ratio (SDP eig_ratio, one per outrat)

    Parameters
    ----------
    csv_path : str
        Path to the CSV produced by ``run_analysis``.
    """
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 5))

    has_multi_trial = "trial" in df.columns and (df.groupby("outrat")["trial"].nunique().max() > 1)

    if has_multi_trial:
        outrats = np.array(sorted(df["outrat"].unique()))
        eig = df.drop_duplicates(subset=["outrat", "trial"])[["outrat", "eig_ratio"]]
        data = [eig[eig["outrat"] == o]["eig_ratio"].values for o in outrats]
        ax.boxplot(data, labels=[f"{o:.3f}" for o in outrats], showfliers=False)
        ax.set_xlabel("Outlier ratio")
        plt.setp(ax.get_xticklabels(), rotation=45)
    else:
        eig = df.drop_duplicates(subset="n_constraints")[["n_constraints", "eig_ratio"]]
        eig = eig.sort_values("n_constraints")
        ax.plot(
            eig["n_constraints"],
            eig["eig_ratio"],
            marker="o",
            label="Eigenvalue Ratio",
        )
        ax.set_xlabel("Number of constraints")
        ax.set_xscale("log")

    ax.set_ylabel("Eigenvalue ratio")
    ax.set_title("Eigenvalue Ratio by Outlier Ratio")
    ax.set_yscale("log")
    if not has_multi_trial:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(csv_path.replace(".csv", "_eig_ratio.png"), dpi=150)
    plt.show()


def plot_runtime_scatter_vs_constraints(csv_path: str = DEFAULT_CSV) -> None:
    """Load results CSV and create a scatter plot of runtime vs constraints.

    Parameters
    ----------
    csv_path : str
        Path to the CSV produced by ``run_analysis``.
    """
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    # SDP runtime appears once per solver row, so deduplicate by trial instance
    sdp = df.drop_duplicates(subset=["outrat", "trial"])
    ax.scatter(
        sdp["n_constraints"],
        sdp["sdp_time_s"],
        marker="s",
        alpha=0.8,
        label="Interior Point",
    )

    for solver_name, marker in [("LDLT", "^"), ("CG", "o"), ("MFCG_LRP", "D"), ("MFCG_DP", "v")]:
        sub = df[df["solver"] == solver_name]
        if sub.empty:
            continue
        ax.scatter(
            sub["n_constraints"],
            sub["ac_time_s"],
            marker=marker,
            alpha=0.7,
            label=solver_name,
        )

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

def generate_plots(out_path):
    # Generate the runtime plot
    plot_runtime_vs_constraints(out_path)

    # Generate runtime scatter plot
    plot_runtime_scatter_vs_constraints(out_path)

    # Generate eigenvalue ratio plot
    plot_eig_ratio_vs_constraints(out_path)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = run_analysis()

    print("\n\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))

    # Persist to CSV for later inspection
    out_path = "/workspace/python/results/max_clique_analysis_lrp.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    generate_plots(out_path)

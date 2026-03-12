"""
Max Clique Analysis Script
===========================
Runs the max clique pipeline (SDP solve + analytic center certification) across
a sweep of outlier ratios and linear solver types, collecting timing and
problem-size statistics into a pandas DataFrame.

Usage:
    python max_clique_analysis.py
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from max_clique import generate_dataset, MaxCliqueProblem
from ranktools import AnalyticCenterParams, LinearSolverType

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Number of outlier-ratio values to sweep
N_OUTRAT = 10

# Range of outlier ratios (log-spaced between these bounds)
OUTRAT_MIN = 0.1
OUTRAT_MAX = 0.98

# Number of trials (from the low-outrat end) for which the expensive
# full-matrix solvers (CG, LDLT) are also run.  Set to N_OUTRAT to
# run them for every trial.
N_FULL_MAT = 0

# Dataset parameters (mirroring max_clique.py __main__)
M_ASSOC = 100       # total number of associations
N1 = 100            # model points in view 1
N2O = 10            # outlier points in view 2
SIGMA = 0.01        # uniform noise [m]
PCFILE = "/workspace/python/examples/bun10k.ply"

# Random seed for reproducibility
SEED = 0

# ---------------------------------------------------------------------------
# Solver bookkeeping
# ---------------------------------------------------------------------------

# All solver types to benchmark.  MFCG is always run; CG and LDLT are
# restricted to the first N_FULL_MAT outlier-ratio trials.
ALL_SOLVERS = {
    "MFCG": LinearSolverType.MFCG,
    "CG":   LinearSolverType.CG,
    "LDLT": LinearSolverType.LDLT,
}

FULL_MAT_SOLVERS = {"CG", "LDLT"}


def make_ac_params(solver_type: LinearSolverType) -> AnalyticCenterParams:
    """Return a default AnalyticCenterParams with the given linear solver."""
    params = AnalyticCenterParams()
    params.verbose = True
    params.check_cert = True
    params.delta_min = 1e-9
    params.delta_dec = 0.6
    params.max_iter = 50
    params.lin_solver = solver_type
    params.lin_solve_max_iter = 200
    params.lin_solve_tol = 1e-4
    return params


def run_analysis(
    n_outrat: int = N_OUTRAT,
    outrat_min: float = OUTRAT_MIN,
    outrat_max: float = OUTRAT_MAX,
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

    for i_trial, outrat in enumerate(outrats):
        print(f"\n{'='*60}")
        print(f"Trial {i_trial+1}/{n_outrat}  |  outrat = {outrat:.4f}")
        print(f"{'='*60}")

        # ---- Generate dataset ------------------------------------------------
        clipper, Agt = generate_dataset(
            PCFILE, M_ASSOC, N1, N2O, outrat, SIGMA, T_21
        )

        # ---- Solve the SDP relaxation (once per outrat) ----------------------
        # Use MFCG params for the problem setup (solver only matters for AC)
        prob = MaxCliqueProblem(clipper, params=make_ac_params(LinearSolverType.MFCG))

        t_sdp_start = time.time()
        X_sdp, u_sdp, sdp_rank = prob.solve_sdp()
        sdp_time = time.time() - t_sdp_start

        n_constraints = len(prob.As)
        sdp_cost = -(u_sdp.T @ prob.M @ u_sdp).item()

        # ---- Certify with each requested solver ------------------------------
        for solver_name, solver_enum in ALL_SOLVERS.items():
            # Skip expensive full-matrix solvers beyond the first n_full_mat
            if solver_name in FULL_MAT_SOLVERS and i_trial >= n_full_mat:
                continue

            print(f"\n--- Solver: {solver_name} ---")
            params = make_ac_params(solver_enum)
            prob_solver = MaxCliqueProblem(clipper, params=params)

            result, ac_time = prob_solver.certify_candidate(u_sdp, cost=sdp_cost, delta=1e-5)

            records.append(
                {
                    "outrat": outrat,
                    "solver": solver_name,
                    "n_constraints": n_constraints,
                    "sdp_time_s": sdp_time,
                    "sdp_rank": sdp_rank,
                    "ac_time_s": ac_time,
                    "certified": result.certified,
                    "min_eig": result.min_eig,
                    "complementarity": result.complementarity,
                }
            )

    df = pd.DataFrame(records)
    return df


DEFAULT_CSV = "/workspace/python/results/max_clique_analysis.csv"


def plot_runtime_vs_constraints(csv_path: str = DEFAULT_CSV) -> None:
    """Load results CSV and plot runtime vs number of constraints.

    Four series are shown:
      - Interior Point  (SDP solve time, one per outrat)
      - LDLT            (analytic-center time)
      - CG              (analytic-center time)
      - MFCG            (analytic-center time)

    Parameters
    ----------
    csv_path : str
        Path to the CSV produced by ``run_analysis``.
    """
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    # --- Interior Point (SDP) ---
    # One SDP time per outrat; take the first occurrence per outrat
    sdp = df.drop_duplicates(subset="n_constraints")[["n_constraints", "sdp_time_s"]]
    sdp = sdp.sort_values("n_constraints")
    ax.plot(
        sdp["n_constraints"],
        sdp["sdp_time_s"],
        marker="s",
        label="Interior Point",
    )

    # --- AC solvers ---
    solver_styles = {
        "LDLT": {"marker": "^"},
        "CG":   {"marker": "o"},
        "MFCG": {"marker": "D"},
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
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Runtime vs. Number of Constraints (Lovasz Theta SDP)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(csv_path.replace(".csv", "_runtime.png"), dpi=150)
    plt.show()


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
    out_path = "/workspace/python/results/max_clique_analysis.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Generate the runtime plot
    plot_runtime_vs_constraints(out_path)

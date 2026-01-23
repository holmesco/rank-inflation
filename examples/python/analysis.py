import numpy as np
from matplotlib import pyplot as plt

from .bunny_test import BunnyProb 
import pandas as pd
from datetime import datetime

def run_outlier_trials(outlier_rates=(0.95, 0.97, 0.99), n_trials=5, m=200, n1=200, n2o=0, sigma=0.01, check_lovasz_theta=True):
    rows = []
    for outr in outlier_rates:
        for seed in range(n_trials):
            prob = BunnyProb(m=m, n1=n1, n2o=n2o, outrat=outr, sigma=sigma, seed=seed)
            clique, info, cert = prob.solve_clipperplus(check_lovasz_theta=check_lovasz_theta)
            
            clique_size = len(clique)

            if hasattr(info, "__dict__"):
                info_dict = dict(info.__dict__)
            else:
                info_dict = {k: getattr(info, k) for k in dir(info) if not k.startswith("_") and not callable(getattr(info, k))}
            valid = prob.check_solution(clique)
            
            row = {"outlier_rate": outr, "seed": seed, "clique_size": clique_size, "cert": cert, "valid": valid}
            row.update(info_dict)
            rows.append(row)

    return pd.DataFrame(rows)

def plot_lt_opt_time(csv_file, save_path=None, show=False):
    """
    Load a CSV produced by run_outlier_trials and plot a box plot of lp_opt_time vs outlier_rate.
    csv_file: path to CSV
    save_path: optional path to save the figure (PNG/PDF)
    show: whether to call plt.show()
    Returns the matplotlib Figure.
    """
    df = pd.read_csv(csv_file)
    if "lt_opt_time" not in df.columns or "outlier_rate" not in df.columns:
        raise ValueError("CSV must contain 'lt_opt_time' and 'outlier_rate' columns")

    # Ensure outlier_rate is treated as a categorical variable for plotting order
    df["outlier_rate_cat"] = pd.Categorical(df["outlier_rate"], categories=sorted(df["outlier_rate"].unique()))

    fig, ax = plt.subplots(figsize=(8, 6))
    # Use pandas boxplot grouped by the categorical values
    df.boxplot(column="lt_opt_time", by="outlier_rate_cat", ax=ax, grid=False)
    ax.set_xlabel("Outlier rate")
    ax.set_ylabel("LT optimization time (seconds)")
    ax.set_title("LT optimization time by outlier rate")
    plt.suptitle("")  # remove the automatic suptitle from pandas
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig

def plot_time_vs_constraints(csv_file, save_path=None, show=False):
    """
    Load a CSV produced by run_outlier_trials and plot a box plot of lp_opt_time vs outlier_rate.
    csv_file: path to CSV
    save_path: optional path to save the figure (PNG/PDF)
    show: whether to call plt.show()
    Returns the matplotlib Figure.
    """
    df = pd.read_csv(csv_file)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # Use pandas boxplot grouped by the categorical values
    df.plot.scatter(x="lt_num_constraints", y="lt_opt_time",ax=ax, grid=True)
    ax.set_xlabel("number of constraints")
    ax.set_ylabel("LT optimization time (seconds)")
    ax.set_title("LT optimization time by number of constraints")
    plt.suptitle("")  # remove the automatic suptitle from pandas
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig

def plot_time_vs_clique_size(csv_file, save_path=None, show=False):
    """
    Load a CSV produced by run_outlier_trials and plot a box plot of lp_opt_time vs outlier_rate.
    csv_file: path to CSV
    save_path: optional path to save the figure (PNG/PDF)
    show: whether to call plt.show()
    Returns the matplotlib Figure.
    """
    df = pd.read_csv(csv_file)
    df = df[df["lt_problem_size"]>0]
    df["clq_ratio"] = df["clique_size"]/df["lt_problem_size"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # Use pandas boxplot grouped by the categorical values
    df.plot.scatter(x="clq_ratio", y="lt_opt_time",ax=ax, grid=True)
    ax.set_xlabel("Max Clique Size / LT Problem Size")
    ax.set_ylabel("LT optimization time (seconds)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title("LT optimization time by clique size")
    plt.suptitle("")  # remove the automatic suptitle from pandas
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def generate_data():
    outlier_rates = (0.5,0.6, 0.7, 0.8, 0.9)
    df = run_outlier_trials(outlier_rates=outlier_rates)
    print(df)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outname = f"bunny_outlier_trials_lower_{ts}.csv"
    df.to_csv(outname, index=False)
    print(f"Saved timestamped CSV: {outname}")


if __name__ == "__main__":
    # plot_lt_opt_time("bunny_outlier_trials_20260123_032904.csv", save_path="lt_opt_time_boxplot.png")
    plot_time_vs_constraints("combined.csv", save_path="time_vs_constraints.png")
    # plot_time_vs_clique_size("combined.csv", save_path="time_vs_clq_size.png")
    
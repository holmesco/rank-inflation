import os
import time

import numpy as np
import matplotlib.pylab as plt
import pickle
import pandas as pd

from cert_tools.sdp_solvers import solve_sdp_fusion, adjust_Q
from ranktools import AnalyticCenterParams, AnalyticCenter, solve_sdp_mosek, AnalyticCenterResult, LinearSolverType

np.set_printoptions(precision=2)

root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")


def run_analytic_center(
    Q,
    Constraints,
    x_cand,
    X_ip,
    **kwargs
    ):
    # Set parameters
    params = AnalyticCenterParams()
    params.verbose = True
    params.check_cert= True
    params.delta_min = 1e-7
    params.max_iter = 50
    params.lin_solver = LinearSolverType.MFCG
    delta = 1e-5
    As, bs = [], []
    for constraint in Constraints:
        A, b = constraint
        As.append(A)
        bs.append(b)
    cost  = (x_cand.T @ Q @ x_cand).item()
    if hasattr(Q, 'todense'):
        Q = np.array(Q.todense())
    # Adjust cost matrix to improve numerical conditioning (as we do for SDP)
    Q_adj, scale, offset = adjust_Q(Q)
    cost_adj = (x_cand.T @ Q_adj @ x_cand).item()
    
    ac = AnalyticCenter(C=Q_adj, rho=cost_adj, A=As, b=bs, params=params)
    # Run certifier
    t1 = time.time()
    result = ac.certify(x_cand, delta)
    time_ac = (time.time() - t1)
    print(f"------- time for AC: {time_ac*1e3:.0f} ms")
    print(f"AC Result: certified={result.certified}  min_eig={result.min_eig:.6e}  complementarity={result.complementarity:.6e}")
    
    # DEBUG
    # check complementarity of inflated solution
    print(f"Complementarity of inflated solution: {np.trace(result.H @ result.X)}")
    print(f"Cost of inflated solution: {np.trace(Q_adj @ result.X)}, Actual cost: {cost_adj}")
    
    return result, time_ac

def check_candidate(H, x_cand):
    if H is not None:
        eigs = np.linalg.eigvalsh(H)[:3]
        print("minimum eigenvalues:", eigs)
        if np.min(eigs) < -1e-5:
            print("Warning: SDP Dual Variable is not PSD")
            return False
        comp = x_cand.T @ H @ x_cand
        print("complementarity:", comp)
        if np.linalg.norm(comp) > 1e-5:
            print("Warning: candidate solution is not complementary to the SDP certificate")
            return False
    else:
        print("Warning: SDP didn't solve, cannot check candidate solution")
        return False
    return True

def compare_solvers(data : dict):
    t1 = time.time()
    X, info = solve_sdp_fusion(
        data["Q"], data["Constraints"], verbose=True
    )
    time_ip = info["time"]
    print(f"------- time for SDP: {time_ip*1e3:.0f} ms")
    optimal_cost = np.trace(data["Q"] @ X)
    print(f"SDP optimal cost: {optimal_cost}")
    # Get rank of SDP solution
    eigs = np.linalg.eigvalsh(X)
    rank = np.sum(eigs > 1e-6*eigs[-1])
    print(f"Rank of SDP solution: {rank} (matrix size: {X.shape[0]})")
        
    # Check if SDP certifies the candidate solution
    certified_ip = check_candidate(info["H"], data["x_cand"])
    # Run centering certifier
    res_ac, time_ac = run_analytic_center(**data, X_ip=X)
    result = dict(cert_ip=certified_ip, 
                  cert_ac=res_ac.certified,
                  time_ip=info["time"],
                  time_ac=time_ac,
                  n_iters_ac=len(data["Constraints"]),)
    
    return result

def run_all_probs():
    fnames = [
        "test_prob_10Gc.pkl",
        "test_prob_10G.pkl",
        "test_prob_10Lc.pkl",
        "test_prob_10L.pkl",
        "test_prob_11Gc.pkl",
        "test_prob_11G.pkl",
        "test_prob_11Lc.pkl",
        "test_prob_11L.pkl",
        "test_prob_12Gc.pkl",
        "test_prob_12G.pkl",
        "test_prob_12Lc.pkl",
        "test_prob_12L.pkl",
        "test_prob_13Gc.pkl",
        "test_prob_13G.pkl",
        "test_prob_13Lc.pkl",
        "test_prob_13L.pkl",
        "test_prob_14G.pkl",
        "test_prob_15G.pkl",
        "test_prob_16Gc.pkl",
        "test_prob_16G.pkl",
        "test_prob_16Lc.pkl",
        "test_prob_16L.pkl",
        "test_prob_1.pkl",
        "test_prob_2.pkl",
        "test_prob_3.pkl",
        "test_prob_4.pkl",
        "test_prob_5.pkl",
        "test_prob_6.pkl",
        "test_prob_7.pkl",
        "test_prob_8Gc.pkl",
        "test_prob_8G.pkl",
        "test_prob_8L1c.pkl",
        "test_prob_8L1.pkl",
        "test_prob_8L2c.pkl",
        "test_prob_8L2.pkl",
        "test_prob_9c.pkl",
        "test_prob_9Gc.pkl",
        "test_prob_9G.pkl",
        "test_prob_9L1c.pkl",
        "test_prob_9L1.pkl",
        "test_prob_9Lc.pkl",
        "test_prob_9L.pkl",
        # "test_prob_9.pkl", remove because solution accuracy too low
    ]
        
    results = []
    for fname in fnames:
        print(f"Running on {fname}...")
        fname_full = os.path.join(root_dir, "examples", fname)
        with open(fname_full, "rb") as f:
            data = pickle.load(f)
        if len(data["Constraints"]) > 100:
            print("Warning: skipping problem with more than 100 constraints for speed")
            continue
        result = compare_solvers(data)
        result["problem"] = fname
        results.append(result)
    # convert results to a dataframe and save
    df = pd.DataFrame(results)
    return df

def plot_results(df):
    # Filter to only include rows where cert_ip agrees with cert_ac
    df_filtered = df[df["cert_ip"] == df["cert_ac"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = df_filtered["cert_ip"].map({True: "green", False: "red"})
    ax.scatter(df_filtered["n_iters_ac"], df_filtered["time_ip"] * 1e3, label="Interior Point", marker="o", c=colors)
    ax.scatter(df_filtered["n_iters_ac"], df_filtered["time_ac"] * 1e3, label="Analytic Center", marker="x",c=colors)
    ax.set_xlabel("Number of Constraints")
    ax.set_ylabel("Time (ms)")
    ax.set_yscale("log")
    ax.set_title("Solver Time vs Number of Constraints")
    ax.legend()

    results_dir = os.path.join(root_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(os.path.join(results_dir, "solver_time_comparison.pdf"))
    plt.show()
    
    print(df)

if __name__ == "__main__":
    df = run_all_probs()
    plot_results(df)

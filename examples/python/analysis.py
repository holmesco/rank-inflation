import os
import numpy as np
import scipy.sparse as sp
from scipy.spatial.transform import Rotation
import networkx as nx
import matplotlib.pyplot as plt
from time import time
import pandas as pd
import seaborn as sns

from src.cgraph.cgraph import ConsistencyGraphProb, generate_bunny_dataset, get_affinity_from_points, prune_affinity, PARAMS_SCS_DFLT
from src.cgraph.rank_reduction import get_low_rank_factor
from src.cgraph.utils import generate_affinity




def solve_prob(affinity):
    # Generate problem
    prob = ConsistencyGraphProb(affinity=affinity)
    # Set up SCS options
    scs_params = PARAMS_SCS_DFLT
    scs_params['verbose'] = False
    # Solve
    homog = True
    return prob.solve_scs_sparse(setup_kwargs=scs_params, homog=homog)


def run_analysis(analysis=1):

    outputs = []
    if analysis == 1:
        n = 1000
        overlap = 5
        csizes = np.logspace(np.log10(10), np.log10(200), 10, dtype=int)
        for csize in csizes:
            print(f"Running with clique size {csize}")
            X, info = solve_prob(generate_affinity(
                n=n, csize=csize, overlap=overlap))
            # sort output data
            output = info['info']
            output['csize'] = csize
            output['n'] = n
            output['overlap'] = overlap
            outputs.append(output)

    df = pd.DataFrame(outputs)
    df.to_csv(f"examples/results/analysis_{analysis}.csv")


def run_analysis_pp(analysis=1):
    df = pd.read_csv(f"examples/results/analysis_{analysis}.csv")
    df['cone_time_avg'] = df['cone_time'] / (df['n'] / df['csize'])
    if analysis == 1:
        df.plot(x='csize', y=['solve_time', 'cone_time',
                'cone_time_avg', 'lin_sys_time'], loglog=True)
        plt.title('Analysis 1 - Graph size = 1000')
        plt.ylabel('Time (ms)')
        plt.xlabel('Clique Size')
        plt.show()


def run_prune_analysis():
    """Analyze how pruning via k-core changes the properties of the problem"""

    # outrates = np.linspace(0.1, 0.95, 30)
    outrates = 1-np.logspace(np.log10(0.1), np.log10(0.95), 30)

    data = []
    clq_size_data = []
    for outrate in outrates:
        affinity, clipper = get_affinity(outrate, mult=5, threshold=0.5)
        for pruned in [False, True]:
            if pruned:
                # run clipper to get lower bound on max clique size
                clipper.solve()
                soln = clipper.get_solution()
                max_clq_sz = len(soln.nodes)
                # prune the matrix
                affinity_pr, _ = prune_affinity(affinity, max_clq_sz)
            else:
                affinity_pr = affinity
                max_clq_sz = 0

            # Create graph problems to run symbolic factorization
            prob = ConsistencyGraphProb(affinity=affinity_pr)
            # Get clique sizes
            clq_sizes = [len(c) for c in prob.cliques]
            num_vars_p = np.sum([n*(n+1)/2 for n in clq_sizes])
            # Get the number of constraints in the original problem.
            num_cons = (affinity_pr.shape[0]**2 - affinity_pr.nnz)/2 + 1

            data.append(dict(n1=affinity.shape[0],
                             n2=affinity_pr.shape[0],
                             max_clq=np.max(clq_sizes),
                             mean_clq=np.mean(clq_sizes),
                             min_clq=np.min(clq_sizes),
                             num_clqs=len(clq_sizes),
                             num_vars_p=num_vars_p,
                             num_cons=num_cons,
                             pruned=pruned,
                             outrate=outrate,
                             soln_clq_sz=max_clq_sz,
                             )
                        )
            for sz in clq_sizes:
                clq_size_data.append(dict(outrate=outrate,
                                          clq_size=sz,
                                          pruned=pruned))

    df = pd.DataFrame(data)
    df.to_csv(f"examples/results/pruning_analysis_summary.csv")
    df = pd.DataFrame(clq_size_data)
    df.to_csv(f"examples/results/pruning_analysis_raw.csv")


def prune_analysis_pp(filename="examples/results/pruning_analysis_summary.csv", rawfile="examples/results/pruning_analysis_raw.csv"):
    """Generate plots for pruning analysis"""
    df = pd.read_csv(filename)

    fig, axs = plt.subplots(4, 1)
    sns.lineplot(df, x="outrate", y="num_clqs",
                 hue="pruned", markers=True, ax=axs[0])
    axs[0].set_xlabel('outlier rate')
    axs[0].set_ylabel('num cliques')
    sns.lineplot(df, x="outrate", y="max_clq",
                 hue="pruned", markers=True, ax=axs[1])
    axs[1].set_xlabel('outlier rate')
    axs[1].set_ylabel('max SDP var')
    # sns.lineplot(df, x="outrate", y="mean_clq",
    #              hue="pruned", markers=True, ax=axs[2])
    # axs[2].set_xlabel('outlier rate')
    # axs[2].set_ylabel('avg clique size')
    sns.lineplot(df, x="outrate", y="n2",
                 hue="pruned", markers=True, ax=axs[2])
    axs[2].set_xlabel('outlier rate')
    axs[2].set_ylabel('problem size')
    sns.lineplot(df, x="outrate", y="num_cons",
                 hue="pruned", markers=True, ax=axs[3])
    axs[3].set_xlabel('outlier rate')
    axs[3].set_ylabel('num constraints')
    axs[3].set_yscale('log')

    # # Make violin plots
    # df_raw = pd.read_csv(rawfile)
    # plt.figure()
    # sns.violinplot(df_raw, x='outrate', y='clq_size', hue="pruned")
    # plt.xlabel('Clique Size Dist')
    # plt.ylabel('Outlier Rate')

    plt.show()


def get_affinity(outrat, mult=2, threshold=False):
    # Set up common variables for tests
    m = 100*mult
    n1 = 100*mult
    n2o = 10*mult
    sigma = 0.01
    pcfile = 'examples/data/bun10k.ply'
    T_21 = np.eye(4)
    T_21[0:3, 0:3] = Rotation.random().as_matrix()
    T_21[0:3, 3] = np.random.uniform(low=-5, high=5, size=(3,))
    # Generate dataset
    D1, D2, Agt, A = generate_bunny_dataset(
        pcfile, m, n1, n2o, outrat, sigma, T_21
    )
    # Generate affinity
    affinity, clipper = get_affinity_from_points(
        D1, D2, A, threshold=threshold)

    return affinity, clipper


if __name__ == "__main__":

    # Analysis on clique runtime
    # run_analysis()
    # run_analysis_pp()

    # Analysis on the effect of pruning
    run_prune_analysis()
    prune_analysis_pp()


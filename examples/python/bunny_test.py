import numpy as np
from scipy.spatial.transform import Rotation
from time import time
from matplotlib import pyplot as plt

from .utils import generate_bunny_dataset, get_affinity_from_points 
from clipperpluspy import ClipperParams, find_clique
import argparse

class BunnyProb:
    def __init__(
        self, m=100, n1=100, n2o=10, outrat=0.9, sigma=0.01, seed=0
    ):
        self.outrat = outrat
        # Set up common variables for tests
        pcfile = "/workspace/examples/data/bun10k.ply"
        T_21 = np.eye(4)
        T_21[0:3, 0:3] = Rotation.random().as_matrix()
        T_21[0:3, 3] = np.random.uniform(low=-5, high=5, size=(3,))
        # Generate dataset
        np.random.seed(seed)
        D1, D2, Agt, A = generate_bunny_dataset(pcfile, m, n1, n2o, outrat, sigma, T_21)
        # Generate affinity
        self.affinity, self.clipper = get_affinity_from_points(D1, D2, A)
        # Generate a solution vector
        x = np.zeros((A.shape[0], 1))
        self.inlier_set = set()
        for i, a in enumerate(A):
            # Find any associations that are in the GT set
            if np.any(np.sum(a == Agt, axis=1) == 2):
                x[i] = 1
                self.inlier_set.add(i)
            else:
                x[i] = 0
        self.x = x
        
        
    def get_prec_recall(self, inlier_clique):
        # Report information
        inliers = np.zeros(self.affinity.shape[0])
        inliers[inlier_clique] = 1
        true_pos = np.sum(self.x.squeeze() * inliers)
        all_pos = np.sum(inliers)
        all_true_pos = np.sum(self.x)
        precision = true_pos / all_pos
        recall = true_pos / all_true_pos

        return precision, recall
    
    def check_solution(self, inlier_clique):
        solution_set = set(inlier_clique)
        valid_inliers = 0
        # check that solution set is valid 
        for i in solution_set:
            valid = True
            for j in solution_set:
                if i > j:
                    if self.affinity[i, j] == 0:
                        valid = False
                        print(f"Solution set is not a clique: edge ({i},{j}) does not exist!")
            if valid:
                valid_inliers += 1
        
        print("Number of intended inliers: ", len(self.inlier_set))
        print(f"Number of inliers returned (valid/total): {valid_inliers} / {len(solution_set)}")

    def solve_clipper(self):
        """Solve SDP using CLIPPER formulation"""
        t0 = time()
        self.clipper.solve_as_msrc_sdr()
        t1 = time()

        soln = self.clipper.get_solution()
        inliers = np.zeros(self.affinity.shape[0])
        inliers[soln.nodes] = 1

        return inliers, t1 - t0
    
    def solve_clipperplus(self, check_lovasz_theta=False):
        # parameter
        params = ClipperParams()
        params.check_lovasz_theta=check_lovasz_theta
        params.cuhallar_params.options="/workspace/parameters/cuhallar_params_inexact.cfg"
        # run 
        csize, clique, cert = find_clique(self.affinity.todense(),params)
        print(f"Clipper+ returned certificate {cert}")
        return clique
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bunny test parameters")
    parser.add_argument("--n1", type=int, default=1000, help="number of points in set 1")
    parser.add_argument("--m", type=int, default=1000, help="m parameter")
    parser.add_argument("--n2o", type=int, default=0, help="n2o parameter")
    parser.add_argument("--outlier-rate", "--outrat", dest="outlier_rate", type=float, default=0.1, help="outlier rate")
    parser.add_argument("--sigma", type=float, default=0.0, help="noise sigma")
    args = parser.parse_args()

    n1 = args.n1
    m = args.m
    n2o = args.n2o
    outlier_rate = args.outlier_rate
    sigma = args.sigma
    # Define problem
    prob = BunnyProb(m, n1, n2o, outlier_rate,sigma)
    plt.imshow (prob.affinity.todense())
    plt.title("Affinity Matrix")
    plt.savefig("/workspace/examples/python/bunny_affinity.png")
    # solve
    print("Setup complete, running clipper")
    clique = prob.solve_clipperplus(True)
    prob.check_solution(clique)
    
    
    
    

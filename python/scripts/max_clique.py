import time
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy.special import gammainc
from scipy.sparse import csc_array

import clipperpy
from cert_tools.sdp_solvers import solve_sdp_fusion, adjust_Q
from ranktools import AnalyticCenter, AnalyticCenterParams, LinearSolverType


def randsphere(m,n,r):
    """Draw random points from within a sphere."""
    X = np.random.randn(m, n)
    s2 = np.sum(X**2, axis=1)
    X = X * np.tile((r*(gammainc(n/2,s2/2)**(1/n)) / np.sqrt(s2)).reshape(-1,1),(1,n))
    return X
        
def generate_dataset(pcfile, m, n1, n2o, outrat, sigma, T_21):
        """Generate a dataset for the registration problem.
        
        Parameters
        ----------
        pcfile : str
            Path to the point cloud file.
        m : int
            Total number of associations in the problem.
        n1 : int
            Number of points used on model (i.e., seen in view 1).
        n2o : int
            Number of outliers in data (i.e., seen in view 2).
        outrat : float
            Outlier ratio of initial association set.
        sigma : float
            Uniform noise [m] range.
        T_21 : np.ndarray
            Ground truth transformation from view 1 to view 2.
            
            Returns
            -------
            D1 : np.ndarray
                Model points in view 1.
            D2 : np.ndarray
                Data points in view 2.
            Agt : np.ndarray
                Ground truth associations.
            A : np.ndarray
                Initial association set.
        """
        pcd = o3d.io.read_point_cloud(pcfile)

        n2 = n1 + n2o # number of points in view 2
        noa = round(m * outrat) # number of outlier associations
        nia = m - noa # number of inlier associations

        if nia > n1:
            raise ValueError("Cannot have more inlier associations "
                            "than there are model points. Increase"
                            "the number of points to sample from the"
                            "original point cloud model.")

        # Downsample from the original point cloud, sample randomly
        I = np.random.choice(len(pcd.points), n1, replace=False)
        D1 = np.asarray(pcd.points)[I,:].T

        # Rotate into view 2 using ground truth transformation
        D2 = T_21[0:3,0:3] @ D1 + T_21[0:3,3].reshape(-1,1)
        # Add noise uniformly sampled from a sigma cube around the true point
        eta = np.random.uniform(low=-sigma/2., high=sigma/2., size=D2.shape)
        # Add noise to view 2
        D2 += eta

        # Add outliers to view 2
        R = 1 # Radius of sphere
        O2 = randsphere(n2o,3,R).T + D2.mean(axis=1).reshape(-1,1)
        D2 = np.hstack((D2,O2))

        # Correct associations to draw from
        # NOTE: These are the exact correponsdences between views
        Agood = np.tile(np.arange(n1).reshape(-1,1),(1,2))

        # Incorrect association to draw from
        #NOTE: Picks any other correspondence than the correct one
        Abad = np.zeros((n1*n2 - n1, 2))
        itr = 0
        for i in range(n1):
            for j in range(n2):
                if i == j:
                    continue
                Abad[itr,:] = [i, j]
                itr += 1

        # Sample good and bad associations to satisfy total
        # num of associations with the requested outlier ratio
        IAgood = np.random.choice(Agood.shape[0], nia, replace=False)
        IAbad = np.random.choice(Abad.shape[0], noa, replace=False)
        A = np.concatenate((Agood[IAgood,:],Abad[IAbad,:])).astype(np.int32)

        # Ground truth associations
        Agt = Agood[IAgood,:]
        
        # Get clipper object
        # Define invariant function    
        iparams = clipperpy.invariants.EuclideanDistanceParams()
        iparams.sigma = 0.01
        iparams.epsilon = 0.02
        invariant = clipperpy.invariants.EuclideanDistance(iparams)
        # Define rounding strategy
        params = clipperpy.Params()
        params.rounding = clipperpy.Rounding.DSD_HEU
        # define clipper object
        clipper = clipperpy.CLIPPER(invariant, params)
        # Get pairwise consistency matrix
        clipper.score_pairwise_consistency(D1, D2, A) 
            
        return (clipper, Agt)

class MaxCliqueProblem:
    def __init__(self, clipper, threshold=0.0):
        self.clipper = clipper
        # Get affinity matrix from clipper
        M = clipper.get_affinity_matrix()
        if threshold > 0.0:
            M = (M > threshold).astype(float)
            # Set constraint and affinity matrix to thresholded values.
            clipper.set_matrix_data(M=M, C=M)
        self.M = M
        # Get constraints for problem
        self.As, self.bs = self.get_constraints()
        # set up parameters for analytic center
        self.params = AnalyticCenterParams()
        self.params.verbose = True
        self.params.check_cert= True
        self.params.delta_min = 1e-7
        self.params.max_iter = 50
        self.params.lin_solver = LinearSolverType.CG
        

    def get_constraints(self):
        """Get the constraints of the maximum clique problem.
        
        Parameters
        ----------
        M : np.ndarray
            Affinity matrix of the problem.
            
        Returns
        -------
        constraints : list of scipy.sparse.csc_array
            List of sparse matrices representing the constraints of the problem.
        values : np.ndarray
            Values corresponding to the constraints (e.g., 0 for non-edges, 1 for trace constraint).
        """
        # Find indices where M is zero and j > i
        rows, cols = np.where((self.M == 0) & (np.triu(np.ones(self.M.shape, dtype=bool), k=1)))
        constraints = []
        for r, c in zip(rows, cols):
            sparse_mat = csc_array(([1.0,1.0], ([r,c], [c,r])), shape=self.M.shape)
            constraints.append(sparse_mat)
        values = np.array([0.0] * len(constraints))
        # add the trace constraint
        sparse_identity = csc_array(np.eye(self.M.shape[0]))
        constraints.append(sparse_identity)
        values = np.append(values, 1.0)
        
        return constraints, values
    
    def certify_candidate(self, x_cand, delta=1e-5):
        """Certify the optimality of a candidate solution to the maximum clique problem.
        
        Parameters
        ----------
        x_cand : np.ndarray
            Candidate solution to certify.
        delta : float
            Tolerance for certification.
            
        Returns
        -------
        result : AnalyticCenterResult
            Result of the certification process, including whether the solution is certified, 
            minimum eigenvalue of the dual variable, and complementarity measure.
        time_ac : float
            Time taken for the analytic center certification process.
        """
        # Get cost of candidate solution
        cost = -(x_cand.T @ self.M @ x_cand).item()
        print(f"Cost of candidate solution: {cost}")
        print(f"Number of constraints: {len(self.As)}")
        # Run certifier
        ac = AnalyticCenter(C=-self.M, rho=cost, A=self.As, b=self.bs, params=self.params)
        print("Running analytic center certifier...")
        t1 = time.time()
        result = ac.certify(x_cand, delta)
        time_ac = (time.time() - t1)
        print(f"------- time for AC: {time_ac*1e3:.0f} ms")
        print(f"AC Result: certified={result.certified}  min_eig={result.min_eig:.6e}  complementarity={result.complementarity:.6e}")
        
        # DEBUG
        # check complementarity of inflated solution
        print(f"Complementarity of inflated solution: {np.trace(result.H @ result.X)}")
        print(f"Complementarity of candidate solution: {x_cand.T @ result.H @ x_cand}")
        print(f"Cost of inflated solution: {np.trace(-self.M @ result.X)}, Actual cost: {cost}")
        
        return result, time_ac
    
    def solve_clipper(self):
        """Solve the maximum clique problem using Clipper and certify the solution."""
        # Solve using Clipper
        self.clipper.solve()
        # Retrieve the normalized solution
        soln = self.clipper.get_solution()
        return soln

    def solve_and_certify(self):
        """Solve the maximum clique problem and certify optimality.
        
        Parameters
        ----------
        clipper : clipperpy.CLIPPER
            Clipper object with pairwise consistency matrix already scored.
            
        Returns
        -------
        Ain : np.ndarray
            Selected associations.
        cert : bool
            Whether the solution is certifiably optimal.
        """
        soln = self.solve_clipper()
        u = soln.u / np.linalg.norm(soln.u)
        # Certify solution
        result, time_ac = self.certify_candidate(u)
        
    def solve_sdp(self):
        
        """Solve the maximum clique SDP relaxation using MOSEK's Fusion solver."""
        n = self.M.shape[0]

        # Build the cost matrix Q = -M (we minimize, so negate for max clique)
        Q = -self.M

        # Build constraint list in the format expected by solve_sdp_fusion
        # Each constraint is (A_i, b_i) such that <A_i, X> = b_i
        constraints = []
        for A_i, b_i in zip(self.As, self.bs):
            constraints.append((A_i, b_i))

        # Solve SDP: min <Q, X> s.t. <A_i, X> = b_i, X >= 0
        t1 = time.time()
        X_sol, info = solve_sdp_fusion(
            Q=Q,
            Constraints=constraints,
            verbose=True
        )
        time_sdp = time.time() - t1
        print(f"SDP solve time: {time_sdp*1e3:.0f} ms")

        # Extract rank-1 solution via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(X_sol)
        # Leading eigenvector (largest eigenvalue)
        u = eigvecs[:, -1] * np.sqrt(np.maximum(eigvals[-1], 0.0))
        # Normalize
        u = u / np.linalg.norm(u)

        # Report SDP cost
        sdp_cost = np.trace(Q @ X_sol)
        print(f"SDP cost: {sdp_cost}")
        print(f"Rank of SDP solution (tol=1e-6): {np.sum(eigvals > 1e-6)}")

        return X_sol, u
        
        

    
if __name__ == "__main__":
    np.random.seed(0)
    # Build a bunny dataset
    m = 100      # total number of associations in problem
    n1 = 100     # number of points used on model (i.e., seen in view 1)
    n2o = 10     # number of outliers in data (i.e., seen in view 2)
    outrat = 0.5 # outlier ratio of initial association set
    sigma = 0.01  # uniform noise [m] range
    pcfile = '/workspace/python/examples/bun10k.ply'  # Object file
    # Random pose transormation
    T_21 = np.eye(4)
    T_21[0:3,0:3] = Rotation.random().as_matrix()
    T_21[0:3,3] = np.random.uniform(low=-5, high=5, size=(3,))
    # Generate data
    clipper, Agt = generate_dataset(pcfile, m, n1, n2o, outrat, sigma, T_21)
    
    prob = MaxCliqueProblem(clipper)
    prob.solve_and_certify()
    prob.solve_sdp()

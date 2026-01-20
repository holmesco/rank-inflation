import time
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy.special import gammainc

import clipperpy


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
        
        return (D1, D2, Agt, A)

def get_err(T, That):
    Terr = np.linalg.inv(T) @ That
    rerr = abs(np.arccos(min(max(((Terr[0:3,0:3]).trace() - 1) / 2, -1.0), 1.0)))
    terr = np.linalg.norm(Terr[0:3,3])
    return (rerr, terr)

def draw_registration_result(source, target, transformation):
    import copy
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
    
if __name__ == "__main__":
    np.random.seed(0)
    # Build a bunny dataset
    m = 100      # total number of associations in problem
    n1 = 100     # number of points used on model (i.e., seen in view 1)
    n2o = 10     # number of outliers in data (i.e., seen in view 2)
    outrat = 0.9 # outlier ratio of initial association set
    sigma = 0.01  # uniform noise [m] range
    pcfile = 'examples/data/bun10k.ply'  # Object file
    # Random pose transormation
    T_21 = np.eye(4)
    T_21[0:3,0:3] = Rotation.random().as_matrix()
    T_21[0:3,3] = np.random.uniform(low=-5, high=5, size=(3,))
    # Generate data
    D1, D2, Agt, A = generate_dataset(pcfile, m, n1, n2o, outrat, sigma, T_21)
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
    # Solve using Clipper
    # clipper.solve()
    # Solve with SDP relaxation
    clipper.solve_as_msrc_sdr()
    Ain = clipper.get_selected_associations()

    p = np.isin(Ain, Agt)[:,0].sum() / Ain.shape[0]
    r = np.isin(Ain, Agt)[:,0].sum() / Agt.shape[0]
    print(f"CLIPPER selected {Ain.shape[0]} inliers from {A.shape[0]} "
        f"putative associations (precision {p:.2f}, recall {r:.2f})")
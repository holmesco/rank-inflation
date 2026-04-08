import torch
import numpy as np

from examples.utils.stereo_camera_model import StereoCameraModel
from examples.utils.stereo_utils import get_gt_setup
from examples.utils.keypoint_tools import get_inv_cov_weights
from examples.utils.lie_algebra import se3_exp, se3_inv, se3_log

from examples.mat_weight_loc.lieopt_pose_est import LieOptPoseEstimator


def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)

class StereoLocalization:
    def __init__(self, batch_size=1, N_map=50, device="cuda:0"):
        # Default dtype
        torch.set_default_dtype(torch.float64)
        torch.autograd.set_detect_anomaly(True)
        self.device = device
        # Set seed
        set_seed(0)
        # Store vars
        self.batch_size = batch_size
        self.N_map = N_map
        # Set up test problem
        
        r_v0s, C_v0s, r_ls = get_gt_setup(
            N_map=50, N_batch=batch_size, traj_type="circle"
        )
        r_v0s = torch.tensor(r_v0s)
        C_v0s = torch.tensor(C_v0s)
        r_ls = torch.tensor(r_ls)[None, :, :].expand(batch_size, -1, -1)
        # Define Stereo Camera
        stereo_cam = StereoCameraModel(0.0, 0.0, 484.5, 0.24).cuda()
        # Frame tranform from vehicle to camera (sensor)
        pert = 0.01
        xi_pert = torch.tensor([[pert, pert, pert, pert, pert, pert]])
        T_s_v = se3_exp(xi_pert)[0]

        # Generate image coordinates (in vehicle frame)
        cam_coords_v = torch.bmm(C_v0s, r_ls - r_v0s)
        cam_coords_v = torch.concat(
            [cam_coords_v, torch.ones(batch_size, 1, r_ls.size(2))], dim=1
        )

        # Source coords in vehicle frame
        src_coords_v = torch.concat(
            [r_ls, torch.ones(batch_size, 1, r_ls.size(2))], dim=1
        )
        # Map to camera frame
        cam_coords = T_s_v[None, :, :].bmm(cam_coords_v)
        src_coords = T_s_v[None, :, :].bmm(src_coords_v)
        # Create transformation matrix
        zeros = torch.zeros(batch_size, 1, 3).type_as(r_v0s)  # Bx1x3
        one = torch.ones(batch_size, 1, 1).type_as(r_v0s)  # Bx1x1
        r_0v_v = -C_v0s.bmm(r_v0s)
        trans_cols = torch.cat([r_0v_v, one], dim=1)  # Bx4x1
        rot_cols = torch.cat([C_v0s, zeros], dim=1)  # Bx4x3
        T_trg_src = torch.cat([rot_cols, trans_cols], dim=2)  # Bx4x4
        # Store values
        self.keypoints_3D_src = src_coords.cuda()
        self.keypoints_3D_trg = cam_coords.cuda()
        self.T_trg_src = T_trg_src
        self.stereo_cam = stereo_cam
        # Generate Scalar Weights
        self.weights = torch.ones(
            self.keypoints_3D_src.size(0), 1, self.keypoints_3D_src.size(2)
        ).cuda()
        self.stereo_cam = stereo_cam
        self.T_s_v = T_s_v.cuda()
        # Initialize local pose estimator
        self.estimator : LieOptPoseEstimator = LieOptPoseEstimator(self.T_s_v, N_batch=batch_size, N_map=N_map)
        self.estimator.to(self.device)
        # Get inverse covariance weights
        # Get matrix weights - assuming 0.5 pixel std dev
        valid = self.weights > 0
        self.inv_cov_weights, cov = get_inv_cov_weights(
            self.keypoints_3D_trg, valid, self.stereo_cam
        )
        
    def run_estimator(self, T_init):
        # Run estimator
        T_trg_src = self.estimator(
            self.keypoints_3D_src,
            self.keypoints_3D_trg,
            self.weights,
            T_init,
            self.inv_cov_weights,
            verbose=True,
        )
        return T_trg_src
    
    def estimator_ground_truth_test(self):
        # Test with ground truth initialization
        T_trg_src = self.run_estimator(self.T_trg_src.cuda())
        # Check that the difference is small
        diff = se3_log(se3_inv(T_trg_src.cpu()).bmm(self.T_trg_src)).numpy()
        np.testing.assert_allclose(diff, np.zeros((1, 6)), atol=1e-7)
        # Define perturbation
        pert = 0.5
        xi_pert = torch.tensor([[pert, pert, pert, pert, pert, pert]])
        T_pert = se3_exp(xi_pert)
        T_init = T_pert.bmm(self.T_trg_src)
        # Test with perturbed starting point
        T_trg_src = self.run_estimator(T_init.cuda())
        # Check that the difference is small
        diff = se3_log(se3_inv(T_trg_src.cpu()).bmm(self.T_trg_src)).numpy()
        np.testing.assert_allclose(diff, np.zeros((1, 6)), atol=1e-8)
        
if __name__ == "__main__":
    stereo_loc = StereoLocalization(batch_size=1, N_map=50)
    stereo_loc.estimator_ground_truth_test()
import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from examples.utils.stereo_camera_model import StereoCameraModel
from examples.utils.stereo_utils import get_gt_setup
from examples.utils.keypoint_tools import get_inv_cov_weights
from examples.utils.lie_algebra import se3_exp, se3_inv, se3_log

from examples.mat_weight_loc.lieopt_pose_est import LieOptPoseEstimator
from examples.mat_weight_loc.stereo_cert import StereoPoseCertifier


def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)

class StereoLocalizationProblem:
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
            N_map=50, N_batch=batch_size, traj_type="circle", n_turns=0.25
        )
        r_v0s = torch.tensor(r_v0s)
        C_v0s = torch.tensor(C_v0s)
        r_ls = torch.tensor(r_ls)[None, :, :].expand(batch_size, -1, -1)
        # Define Stereo Camera
        stereo_cam = StereoCameraModel(0.0, 0.0, 484.5, 0.24).cuda()
        # Frame tranform from vehicle to camera (sensor)
        pert = 0.0 # Set to zero for now
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
        # Intialize certifier class
        self.certifier = StereoPoseCertifier(self.T_s_v,
                                             self.keypoints_3D_src,
                                             self.keypoints_3D_trg,
                                             self.weights,
                                             self.inv_cov_weights)
        
        
    def run_estimator(self, T_init, verbose=True):
        # Run estimator
        T_trg_src = self.estimator(
            self.keypoints_3D_src,
            self.keypoints_3D_trg,
            self.weights,
            T_init,
            self.inv_cov_weights,
            verbose=verbose,
        )
        return T_trg_src

    def plot_targ_frames_3d(self, T_ests, is_global_min, T_inits=None, axis_scale=0.2, title=None):
        """Plot target frames in 3D from estimated transforms.

        Args:
            T_ests (torch.Tensor): Batched transforms of shape (N, 4, 4).
            is_global_min (array-like): Boolean mask of shape (N,) where True indicates global minima.
            T_inits (torch.Tensor | None): Optional batched initialization transforms of shape (N, 4, 4).
            axis_scale (float): Length of frame axes used in the visualization.
            title (str | None): Optional figure title.
        """
        if T_ests.ndim != 3 or T_ests.shape[1:] != (4, 4):
            raise ValueError("T_ests must have shape (N, 4, 4).")

        is_global_min = np.asarray(is_global_min, dtype=bool)
        if is_global_min.shape[0] != T_ests.shape[0]:
            raise ValueError("is_global_min must have shape (N,) matching T_ests.")

        if T_inits is not None:
            if T_inits.ndim != 3 or T_inits.shape[1:] != (4, 4):
                raise ValueError("T_inits must have shape (N, 4, 4).")
            if T_inits.shape[0] != T_ests.shape[0]:
                raise ValueError("T_inits must have the same batch size as T_ests.")

        # Plots should use the inverse transform T_source_target
        T_ests_plot = torch.linalg.inv(T_ests)
        T_ests_np = T_ests_plot.detach().cpu().numpy()

        if T_inits is not None:
            T_inits_plot = torch.linalg.inv(T_inits)
            T_inits_np = T_inits_plot.detach().cpu().numpy()
        else:
            T_inits_np = None

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        init_global_color = "tab:green"
        init_local_color = "tab:red"
        opt_global_color = "black"
        opt_local_color = "tab:purple"

        # Plot initializations with transparency and color by convergence result
        if T_inits_np is not None:
            for i in range(T_inits_np.shape[0]):
                T0 = T_inits_np[i]
                R0 = T0[:3, :3]
                t0 = T0[:3, 3]
                c0 = init_global_color if is_global_min[i] else init_local_color

                ax.scatter(t0[0], t0[1], t0[2], c=c0, s=35, alpha=0.5)

                ex0 = R0[:, 0] * axis_scale
                ey0 = R0[:, 1] * axis_scale
                ez0 = R0[:, 2] * axis_scale

                ax.quiver(t0[0], t0[1], t0[2], ex0[0], ex0[1], ex0[2], color=c0, linewidth=1.0, alpha=0.5)
                ax.quiver(t0[0], t0[1], t0[2], ey0[0], ey0[1], ey0[2], color=c0, linewidth=1.0, alpha=0.4)
                ax.quiver(t0[0], t0[1], t0[2], ez0[0], ez0[1], ez0[2], color=c0, linewidth=1.0, alpha=0.3)

        # Plot optimized poses: global in black, local in orange
        for i in range(T_ests_np.shape[0]):
            T = T_ests_np[i]
            R = T[:3, :3]
            t = T[:3, 3]
            color = opt_global_color if is_global_min[i] else opt_local_color

            ax.scatter(t[0], t[1], t[2], c=color, s=40)

            ex = R[:, 0] * axis_scale
            ey = R[:, 1] * axis_scale
            ez = R[:, 2] * axis_scale

            ax.quiver(t[0], t[1], t[2], ex[0], ex[1], ex[2], color=color, linewidth=1.0)
            ax.quiver(t[0], t[1], t[2], ey[0], ey[1], ey[2], color=color, linewidth=1.0, alpha=0.75)
            ax.quiver(t[0], t[1], t[2], ez[0], ez[1], ez[2], color=color, linewidth=1.0, alpha=0.5)

        # Plot source keypoints (3D map points)
        if self.keypoints_3D_src.ndim == 3 and self.keypoints_3D_src.shape[1] >= 3:
            keypoints_src = self.keypoints_3D_src[0, :3, :].detach().cpu().numpy().T
            ax.scatter(
                keypoints_src[:, 0],
                keypoints_src[:, 1],
                keypoints_src[:, 2],
                c="tab:blue",
                s=12,
                alpha=0.6,
                label="keypoints_3D_src",
            )

        if T_ests_np.shape[0] > 0:
            xyz = T_ests_np[:, :3, 3]
            if T_inits_np is not None and T_inits_np.shape[0] > 0:
                xyz = np.vstack([xyz, T_inits_np[:, :3, 3]])
            mins = xyz.min(axis=0)
            maxs = xyz.max(axis=0)
            span = np.maximum(maxs - mins, 1e-6)
            pad = 0.15 * span
            ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
            ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
            ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(title if title is not None else "Estimated target frames")
        ax.set_aspect("equal", adjustable="box")

        ax.scatter([], [], [], c=init_global_color, s=35, alpha=0.5, label="init → global")
        ax.scatter([], [], [], c=init_local_color, s=35, alpha=0.5, label="init → local")
        ax.scatter([], [], [], c=opt_global_color, s=40, label="optimized global")
        ax.scatter([], [], [], c=opt_local_color, s=40, label="optimized local")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def get_random_inits(self, radius, N_batch=10, seed=0):
        """Generate random pose initializations similar to stereo_cal.get_random_inits."""
        set_seed(seed)
        r_v0s = []
        C_v0s = []

        for _ in range(N_batch):
            # random locations on sphere
            r_ = np.random.random((3, 1)) - 0.5
            r = radius * r_ / np.linalg.norm(r_)
            r_v0s += [r]

            # random orientation pointing at origin
            z = -r / np.linalg.norm(r)
            y = np.random.randn(3, 1)
            y = y - y.T @ z * z
            y = y / np.linalg.norm(y)
            x = np.cross(y[:, 0], z[:, 0])[:, None]
            C_v0s += [np.hstack([x, y, z]).T]

        r_v0s = np.stack(r_v0s)
        C_v0s = np.stack(C_v0s)

        return r_v0s, C_v0s

    
    
    def run_initializations_and_certify(self, N_init=10, seed=0, plot_results=False, plot_axis_scale=0.2):
        """Generate N random initializations, run the estimator, certify each solution, and print a DataFrame.

        Args:
            N_init (int): Number of random initializations.
            init_perturb (float): Std-dev of random se(3) perturbation used to initialize each run.
            seed (int): Random seed for reproducibility.
            plot_results (bool): If True, plot estimated target frames colored by local/global minima.
            plot_axis_scale (float): Axis length for plotted frame triads.

        Returns:
            pd.DataFrame: Per-run results including cost, certification status, and certifier metrics.
        """
        if self.batch_size != 1:
            raise ValueError("run_initializations_and_certify currently supports batch_size=1.")

        radius = torch.linalg.norm(self.T_trg_src[0, :3, 3]).item()
        r_v0s_init, C_v0s_init = self.get_random_inits(
            radius=radius,
            N_batch=N_init,
            seed=seed,
        )
        r_v0s_init = torch.tensor(
            r_v0s_init, dtype=self.T_trg_src.dtype, device=self.T_trg_src.device
        )
        C_v0s_init = torch.tensor(
            C_v0s_init, dtype=self.T_trg_src.dtype, device=self.T_trg_src.device
        )

        zeros = torch.zeros(N_init, 1, 3, dtype=self.T_trg_src.dtype, device=self.T_trg_src.device)
        one = torch.ones(N_init, 1, 1, dtype=self.T_trg_src.dtype, device=self.T_trg_src.device)
        r_0v_v = -C_v0s_init.bmm(r_v0s_init)
        trans_cols = torch.cat([r_0v_v, one], dim=1)
        rot_cols = torch.cat([C_v0s_init, zeros], dim=1)
        T_inits = torch.cat([rot_cols, trans_cols], dim=2)

        t_sdp_start = time.perf_counter()
        _, info_sdp, _ = self.certifier.solve_sdp(verbose=False)
        t_sdp_end = time.perf_counter()
        sdp_wall_time_s = t_sdp_end - t_sdp_start
        sdp_solver_time_s = info_sdp[0].get("time", np.nan) if len(info_sdp) > 0 else np.nan
        

        C = self.certifier.Cs[0].detach().cpu().numpy()
        results = []
        T_est_list = []

        for i in range(N_init):
            T_init = T_inits[i : i + 1].to(self.device)
            T_est = self.run_estimator(T_init, verbose=False)
            T_est_list.append(T_est.detach())
            x_cand = self.certifier.transform_to_x(T_est)[0].detach().cpu().numpy()

            optimal_cost = float((x_cand.T @ C @ x_cand).item())
            cert_result = self.certifier.certify_solution(
                x_cand,
                verbose=False,
                cost=optimal_cost,
            )

            results.append(
                {
                    "init": i,
                    "optimal_cost": optimal_cost,
                    "certified": bool(cert_result.certified),
                    "min_eig": float(cert_result.min_eig),
                    "complementarity": float(cert_result.complementarity),
                    "certifier_time_s": float(cert_result.solver_time),
                }
            )

        T_ests = torch.cat(T_est_list, dim=0)

        df = pd.DataFrame(results)
        print(df)
        print(f"SDP solve wall time: {sdp_wall_time_s:.6f} s")
        print(f"SDP reported solver time: {sdp_solver_time_s:.6f} s")

        global_mask = df["certified"]
        local_mask = ~global_mask

        global_avg_cert_time = (
            df.loc[global_mask, "certifier_time_s"].mean() if global_mask.any() else np.nan
        )
        local_avg_cert_time = (
            df.loc[local_mask, "certifier_time_s"].mean() if local_mask.any() else np.nan
        )

        print(f"Average certifier time (global minima): {global_avg_cert_time:.6f} s")
        print(f"Average certifier time (local minima): {local_avg_cert_time:.6f} s")

        global_costs = df.loc[global_mask, "optimal_cost"]
        local_costs = df.loc[local_mask, "optimal_cost"]

        global_cost_mean = global_costs.mean() if global_mask.any() else np.nan
        global_cost_std = global_costs.std(ddof=0) if global_mask.any() else np.nan
        print(f"Global minima cost mean: {global_cost_mean:.6e}")
        print(f"Global minima cost std: {global_cost_std:.6e}")

        # Use mean + std as a scalar comparison threshold for local minima costs
        global_cost_threshold = global_cost_mean + global_cost_std
        if local_mask.any() and global_mask.any():
            all_local_greater = bool((local_costs > global_cost_threshold).all())
            print(
                f"All local minima costs > (global mean + std = {global_cost_threshold:.6e}): {all_local_greater}"
            )
        elif local_mask.any() and not global_mask.any():
            print("Cannot compare local minima costs to global mean+std (no global minima found).")
        else:
            print("No local minima found to compare against global mean+std threshold.")
        

        if plot_results:
            fig_cost, ax_cost = plt.subplots(figsize=(9, 4))
            idx = df["init"].to_numpy()
            ax_cost.scatter(
                idx[global_mask.to_numpy()],
                global_costs.to_numpy(),
                c="black",
                s=35,
                label="global minima cost",
            )
            if local_mask.any():
                ax_cost.scatter(
                    idx[local_mask.to_numpy()],
                    local_costs.to_numpy(),
                    c="tab:purple",
                    s=35,
                    label="local minima cost",
                )

            if global_mask.any():
                ax_cost.axhline(global_cost_mean, color="tab:green", linestyle="-", linewidth=1.5, label="global mean")
                ax_cost.axhline(global_cost_mean + global_cost_std, color="tab:green", linestyle="--", linewidth=1.0, label="global mean ± std")
                ax_cost.axhline(global_cost_mean - global_cost_std, color="tab:green", linestyle="--", linewidth=1.0)

            ax_cost.set_xlabel("initialization index")
            ax_cost.set_ylabel("optimal cost")
            ax_cost.set_title("Cost by trial with global mean/std")
            ax_cost.legend(loc="best")
            ax_cost.grid(alpha=0.25)
            plt.tight_layout()

            is_global_min = df["certified"].to_numpy(dtype=bool)
            self.plot_targ_frames_3d(
                T_ests,
                is_global_min,
                T_inits=T_inits,
                axis_scale=plot_axis_scale,
                title="Initializations and optimized target frames",
            )

        return df
    
    def test_estimator_ground_truth(self):
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
        # check that the certifier sdp solve gets the same solution
        X, info, T_trg_src_sdp = self.certifier.solve_sdp(verbose=True)
        # Check that the difference is small
        diff = se3_log(se3_inv(T_trg_src_sdp.cpu()).bmm(self.T_trg_src)).numpy()
        np.testing.assert_allclose(diff, np.zeros((1, 6)), atol=1e-6)
        # Run Certifier on SDP output
        x_cand = self.certifier.transform_to_x(T_trg_src_sdp)
        result = self.certifier.certify_solution(x_cand[0])
        np.testing.assert_equal(result.certified, True)
        # Test with output of solver
        x_cand = self.certifier.transform_to_x(T_trg_src)
        result = self.certifier.certify_solution(x_cand[0])
        np.testing.assert_equal(result.certified, True)
        
if __name__ == "__main__":
    stereo_loc = StereoLocalizationProblem(batch_size=1, N_map=50)
    # stereo_loc.certifier.check_constraint_linear_independence()
    # stereo_loc.test_estimator_ground_truth()
    df = stereo_loc.run_initializations_and_certify(N_init=100, plot_results=True, seed=0)
    
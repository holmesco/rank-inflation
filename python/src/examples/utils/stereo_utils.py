#!/bin/bash/python
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import theseus as th
import torch
from pylgmath.so3.operations import vec2rot


class StereoCamera:
    def __init__(
        c, f_u=200.0, f_v=200.0, c_u=0.0, c_v=0.0, b=0.05, sigma_u=0.5, sigma_v=0.5
    ):
        c.f_u = f_u
        c.f_v = f_v
        c.c_u = c_u
        c.c_v = c_v
        c.b = b
        c.sigma_u = sigma_u
        c.sigma_v = sigma_v

    def get_intrinsic_mat(c, M=None):
        c.M = np.array(
            [
                [c.f_u, 0.0, c.c_u, 0.0],
                [0.0, c.f_v, 0.0, c.c_v, 0.0],
                [c.f_u, 0.0, c.c_u, -c.f_u * c.b],
                [0.0, c.f_v, 0.0, c.c_v, 0.0],
            ]
        )

    def forward(c, p_inC):
        """forward camera model, points to pixels"""
        z = p_inC[2, :]
        x = p_inC[0, :] / z
        y = p_inC[1, :] / z
        assert all(z > 0), "Negative depth in data"
        # noise
        noise = np.random.randn(4, len(x))
        # pixel measurements
        ul = c.f_u * x + c.c_u + c.sigma_u * noise[0, :]
        vl = c.f_v * y + c.c_v + c.sigma_v * noise[1, :]
        ur = ul - c.f_u * c.b / z + c.sigma_u * noise[2, :]
        vr = vl + c.sigma_v * noise[3, :]

        return ul, vl, ur, vr

    def inverse(c, pixel_meass: torch.Tensor):
        """inverse camera model, pixels to points. Assumes inputs are torch tensors.
        Output meas is a torch tensor of shape (N_batch, 3, N) where N is the number of measurements.
        Output weights is a torch tensor of shape (N_batch, 3, 3, N) where N is the number of measurements.
        """
        # unpack pixel measurements
        ul = pixel_meass[:, 0, :]
        vl = pixel_meass[:, 1, :]
        ur = pixel_meass[:, 2, :]
        vr = pixel_meass[:, 3, :]

        # compute disparity
        d = ul - ur
        # assert torch.all(d >= 0), "Negative disparity in data"
        # If noise set to zero, covariance matrix is identity. Otherwise, compute covariance matrix using Jacobian of inverse camera model and pixel noise covariance.
        if c.sigma_u == 0 or c.sigma_v == 0:
            Sigma = torch.eye(3)
        else:
            Sigma = torch.zeros((3, 3))
            # Define pixel covariance
            Sigma[0, 0] = c.sigma_u**2
            Sigma[1, 1] = c.sigma_v**2
            Sigma[2, 2] = 2 * c.sigma_u**2
            Sigma[0, 2] = c.sigma_u**2
            Sigma[2, 0] = c.sigma_u**2

        # compute euclidean measurement coordinates
        ratio = c.b / d
        x = (ul - c.c_u) * ratio
        y = (vl - c.c_v) * ratio * c.f_u / c.f_v
        z = ratio * c.f_u
        meas = torch.stack([x, y, z], dim=1)
        # compute weights
        G = torch.zeros((ul.size(0), ul.size(1), 3, 3), dtype=torch.double)
        # Define G
        G[:, :, 0, 0] = z / c.f_u
        G[:, :, 1, 1] = z / c.f_v
        G[:, :, 0, 2] = -x * z / c.f_u / c.b
        G[:, :, 1, 2] = -y * z / c.f_v / c.b
        G[:, :, 2, 2] = -(z**2) / c.f_u / c.b

        # Covariance matrix (matrix mult last two dims)
        Sigma = Sigma.expand((ul.size(0), ul.size(1), 3, 3))
        Cov = torch.einsum("bnij,bnjk,bnlj->bnil", G, Sigma, G)

        # Check if any of the matrices are not full rank
        ranks = torch.linalg.matrix_rank(Cov)
        if torch.any(ranks < 3):
            warnings.warn("At least one covariance matrix is not full rank")
            Cov = torch.eye(3, dtype=torch.double).expand_as(Cov)
        # Compute weights by inverting covariance matrices
        W = torch.linalg.inv(Cov)
        # Symmetrize
        weights = 0.5 * (W + W.transpose(-2, -1))

        return meas, weights


def get_gt_setup(
    traj_type="circle",  # Trajectory format [clusters,circle]
    N_batch=1,  # Number of poses
    N_map=10,  # number of landmarks
    offs=np.array([[0, 0, 2]]).T,  # offset between poses and landmarks
    n_turns=0.1,  # (circle) number of turns around the cluster
    lm_bound=1.0,  # Bounding box of uniform landmark distribution.
):
    """Used to generate a trajectory of ground truth pose data"""

    # Ground Truth Map Points
    # Cluster at the origin
    r_l = lm_bound * (np.random.rand(3, N_map) - 0.5)
    # Ground Truth Poses
    r_p0s = []
    C_p0s = []
    if traj_type == "clusters":
        # Ground Truth Poses
        for i in range(N_batch):
            r_p0s += [0.2 * np.random.randn(3, 1)]
            aaxis = np.zeros((3, 1))
            aaxis[2, 1] = 0.5 * np.random.randn(1)[0]
            C_p0s += vec2rot(aaxis_ba=aaxis)
        # Offset from the origin
        r_l = r_l + offs
    elif traj_type == "circle":
        # GT poses equally spaced along n turns of a circle
        radius = np.linalg.norm(offs)
        assert radius > 0.2, "Radius of trajectory circle should be larger"
        if N_batch > 1:
            delta_phi = n_turns * 2 * np.pi / (N_batch - 1)
        else:
            delta_phi = n_turns * 2 * np.pi
        phi = delta_phi
        for i in range(N_batch):
            # Location
            r = radius * np.array([[np.cos(phi), np.sin(phi), 0]]).T
            r_p0s += [r]
            # Z Axis points at origin
            z = -r / np.linalg.norm(r)
            x = np.array([[0.0, 0.0, 1.0]]).T
            y = skew(z) @ x
            C_p0s += [np.hstack([x, y, z]).T]
            # Update angle
            phi = (phi + delta_phi) % (2 * np.pi)
    r_p0s = np.stack(r_p0s)
    C_p0s = np.stack(C_p0s)

    return r_p0s, C_p0s, r_l


def get_prob_data(camera=StereoCamera(), N_map=30, N_batch=1):
    # get ground truth information
    r_p0s, C_p0s, r_l = get_gt_setup(N_map=N_map, N_batch=N_batch)

    # generate measurements
    pixel_meass = []
    r_ls = []
    for i in range(N_batch):
        r_p = r_p0s[i]
        C_p0 = C_p0s[i]
        r_ls += [r_l]
        r_l_inC = C_p0 @ (r_l - r_p)
        pixel_meass += [camera.forward(r_l_inC)]
    pixel_meass = np.stack(pixel_meass)
    r_ls = np.stack(r_ls)

    # Convert to torch tensors
    r_p0s = torch.tensor(r_p0s)
    C_p0s = torch.tensor(C_p0s)
    r_ls = torch.tensor(r_ls)
    pixel_meass = torch.tensor(pixel_meass)

    return r_p0s, C_p0s, r_ls, pixel_meass


def kron(A, B):
    # kronecker workaround for matrices
    # https://github.com/pytorch/pytorch/issues/74442
    return (A[:, None, :, None] * B[None, :, None, :]).reshape(
        A.shape[0] * B.shape[0], A.shape[1] * B.shape[1]
    )

term_crit_def = {
    "max_iter": 500,
    "tol_grad_sq": 1e-10,
    "tol_loss": 1e-10,
}  # Optimization termination criteria


# THESEUS OPTIMIZATION
def build_theseus_layer(N_map, N_batch=1, opt_kwargs_in={}):
    """Build theseus layer for stereo problem

    Args:
        cam_torch (StereoCamera): _description_
        r_l (_type_): _description_
        pixel_meas (_type_): _description_
    """
    # Optimization variables
    r_p0s = th.Point3(name="r_p0s")
    C_p0s = th.SO3(name="C_p0s")
    # Auxillary (data) variables (pixel measurements and landmarks)
    meas = th.Variable(torch.zeros(N_batch, 3, N_map), name="meas")
    weights = th.Variable(torch.zeros(N_batch, N_map, 3, 3), name="weights")
    r_ls = th.Variable(torch.zeros(N_batch, 3, N_map), name="r_ls")

    # Define cost function
    def error_fn(optim_vars, aux_vars):
        C_p0s, r_p0s = optim_vars
        meas, weights, r_ls = aux_vars

        # Get error for each map point
        errors = []
        for j in range(meas.shape[-1]):
            # get measurement
            meas_j = meas[:, :, j]
            # get weight (N_batch,N_map, 3, 3)
            W_ij = weights[:, j, :, :]
            # NOTE we want the transpose of the cholesky factor
            W_ij_half = torch.linalg.cholesky(W_ij).transpose(-2, -1)
            # get measurement in camera frame (N_batch, 3)
            r_jp_in0 = r_ls.tensor[:, :, j] - r_p0s.tensor
            r_jp_inp = torch.einsum("bij,bj->bi", C_p0s.tensor, r_jp_in0)
            # get error
            err = meas_j - r_jp_inp
            # Multiply by weight matrix.
            err_w = torch.einsum("bij,bj->bi", W_ij_half, err)
            errors += [err_w]
        # Stack errors (N_batch, 3 * N_map)
        error_stack = torch.cat(errors, dim=1)
        return error_stack

    objective = th.Objective()
    optim_vars = [C_p0s, r_p0s]
    aux_vars = [meas, weights, r_ls]
    cost_function = th.AutoDiffCostFunction(
        optim_vars=optim_vars,
        dim=N_map * 3,
        err_fn=error_fn,
        aux_vars=aux_vars,
        cost_weight=th.ScaleCostWeight(np.sqrt(2)),
        name="registration_cost",
    )
    objective.add(cost_function)

    # Build layer
    opt_kwargs = {
        "abs_err_tolerance": 1e-8,
        "rel_err_tolerance": 1e-8,
        "max_iterations": 100,
    }
    opt_kwargs.update(opt_kwargs_in)
    layer = th.TheseusLayer(th.GaussNewton(objective, **opt_kwargs))
    # layer = th.TheseusLayer(th.Dogleg(objective, **opt_kwargs))
    # layer = th.TheseusLayer(th.LevenbergMarquardt(objective, **opt_kwargs))

    return layer



def skew(vec):
    if vec.shape == (3,):
        vec = np.expand_dims(vec, axis=1)
    assert vec.shape == (3, 1), "Input vector must have shape (3,1)"
    return np.array(
        [
            [0.0, -vec[2, 0], vec[1, 0]],
            [vec[2, 0], 0.0, -vec[0, 0]],
            [-vec[1, 0], vec[0, 0], 0.0],
        ]
    )

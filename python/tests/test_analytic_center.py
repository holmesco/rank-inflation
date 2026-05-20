"""Tests for AnalyticCenterPyTorch using Lovasz theta fixtures."""

from __future__ import annotations

import pytest
import torch
import scipy.sparse as sp
import numpy as np
from torch.profiler import profile, ProfilerActivity, record_function


from ranktools import LinearSolverType, solve_sdp_mosek, SDPResult

from ranktools_pytorch.analytic_center_torch import (
    AnalyticCenterPyTorch,
    defaultAnalyicCenterParams,
)
from ranktools_pytorch.lin_alg_torch import LowRankPrecond
from ranktools_pytorch.utils import symmetrize_dense, symmetrize_sparse_to_torch
from .fixtures import (
    clique1_adj,
    clique2_adj,
    clique3_adj,
    clique4_adj,
    make_lovasz_test_case,
    load_problem_from_file,
)

LOVASZ_TESTS = [
    (clique1_adj, [1, 3, 4, 6, 7, 8], "Clique1"),
    (clique2_adj, [0, 2, 3, 5, 6, 8, 9], "Clique2"),
    (clique3_adj, [4, 10, 13, 14, 15, 16, 17, 18], "Clique3_Large20x20"),
    (clique4_adj, [0, 1, 2], "Clique4_Disconnected"),
]

# Turn off gradients
torch.set_grad_enabled(False)


@pytest.mark.parametrize("adj,clique,name", LOVASZ_TESTS)
def test_certify_lovasz_theta_direct_solve(adj, clique, name):
    sdp = make_lovasz_test_case(adj, clique, name)

    # Recover Analytic center from Mosek
    b = sdp.b.numpy()
    C = sdp.C.numpy()
    result: SDPResult = solve_sdp_mosek(
        C, sdp.A_mosek, b
    )  # Just to check Mosek can solve it
    X = result.X
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    rank = np.sum(S > S[0] * 1e-6)
    Y_0 = torch.Tensor(U[:, :rank] @ np.diag(np.sqrt(S[:rank])))

    params = defaultAnalyicCenterParams()
    params.verbose = True
    params.max_iter = 25
    params.delta = 1e-5
    params.lrp_params.tau = 1e-5
    params.perturb_cost = True
    params.perturb_constraints = True
    params.eps_cost = 1e-5
    params.eps_constr = 1e-5
    params.early_stop_cert = True
    params.adaptive_perturb = True
    params.eps_mult_min = 1e-4
    params.delta = 1e-5
    params.lin_solver = LinearSolverType.LDLT

    certifier = AnalyticCenterPyTorch(
        C=sdp.C,
        rho=sdp.rho,
        A_list=sdp.A,
        b=sdp.b,
        params=params,
    )

    result = certifier.certify(Y_0)

    assert result.X.shape == (sdp.dim, sdp.dim)
    assert result.H.shape == (sdp.dim, sdp.dim)
    assert result.multipliers.shape == (len(sdp.A) + 1,)
    assert result.violation.shape == (len(sdp.A) + 1,)
    assert torch.isfinite(result.X).all()
    assert torch.isfinite(result.H).all()
    assert torch.isfinite(result.violation).all()
    assert isinstance(result.certified, bool)
    assert result.solver_time >= 0.0
    assert result.num_iters >= 0

    assert result.certified, ValueError(f"Certification failed")
    assert result.min_eig > -params.tol_cert_psd, ValueError(
        f"Minimum eigenvalue negative: {result.min_eig}"
    )
    assert result.complementarity < params.tol_cert_complementarity, ValueError(
        f"Complementarity gap too large: {result.complementarity}"
    )


@pytest.mark.parametrize("adj,clique,name", LOVASZ_TESTS)
def test_certify_lovasz_theta_cg_solve(adj, clique, name):
    sdp = make_lovasz_test_case(adj, clique, name)

    # Recover Analytic center from Mosek
    b = sdp.b.numpy()
    C = sdp.C.numpy()
    result: SDPResult = solve_sdp_mosek(
        C, sdp.A_mosek, b
    )  # Just to check Mosek can solve it
    X = result.X
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    rank = np.sum(S > S[0] * 1e-5)
    Y_0 = torch.Tensor(U[:, :rank] @ np.diag(np.sqrt(S[:rank])))

    params = defaultAnalyicCenterParams()
    params.verbose = True
    params.max_iter = 25
    params.delta = 1e-5
    params.lrp_params.tau = 1e-5
    params.perturb_cost = True
    params.perturb_constraints = True
    params.eps_cost = 1e-5
    params.eps_constr = 1e-5
    params.early_stop_cert = True
    params.adaptive_perturb = True
    params.eps_mult_min = 1e-4
    params.delta = 1e-5
    params.lin_solver = LinearSolverType.MFCG_LRP

    certifier = AnalyticCenterPyTorch(
        C=sdp.C,
        rho=sdp.rho,
        A_list=sdp.A,
        b=sdp.b,
        params=params,
    )

    result = certifier.certify(Y_0)

    assert result.certified, ValueError(f"Certification failed")
    assert result.min_eig > -params.tol_cert_psd, ValueError(
        f"Minimum eigenvalue negative: {result.min_eig}"
    )
    assert result.complementarity < params.tol_cert_complementarity, ValueError(
        f"Complementarity gap too large: {result.complementarity}"
    )

@pytest.mark.parametrize("adj,clique,name", LOVASZ_TESTS)
def test_certify_lovasz_theta_cg_solve_gpu(adj, clique, name):
    sdp = make_lovasz_test_case(adj, clique, name)

    # Recover Analytic center from Mosek
    b = sdp.b.numpy()
    C = sdp.C.numpy()
    result: SDPResult = solve_sdp_mosek(
        C, sdp.A_mosek, b
    )  # Just to check Mosek can solve it
    X = result.X
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    rank = np.sum(S > S[0] * 1e-5)
    Y_0 = torch.Tensor(U[:, :rank] @ np.diag(np.sqrt(S[:rank])))

    params = defaultAnalyicCenterParams()
    params.verbose = True
    params.max_iter = 25
    params.delta = 1e-5
    params.lrp_params.tau = 1e-5
    params.perturb_cost = True
    params.perturb_constraints = True
    params.eps_cost = 1e-5
    params.eps_constr = 1e-5
    params.early_stop_cert = True
    params.adaptive_perturb = True
    params.eps_mult_min = 1e-4
    params.delta = 1e-5
    params.lin_solver = LinearSolverType.MFCG_LRP
    

    certifier = AnalyticCenterPyTorch(
        C=sdp.C,
        rho=sdp.rho,
        A_list=sdp.A,
        b=sdp.b,
        params=params,
        main_gpu=True,
    )

    result = certifier.certify(Y_0)

    assert result.certified, ValueError(f"Certification failed")
    assert result.min_eig > -params.tol_cert_psd, ValueError(
        f"Minimum eigenvalue negative: {result.min_eig}"
    )
    assert result.complementarity < params.tol_cert_complementarity, ValueError(
        f"Complementarity gap too large: {result.complementarity}"
    )



# TESTS ON STANARD RANK 1 PROBLEMS
problem_names = ["test_prob_10G"]


@pytest.mark.parametrize("name", problem_names)
def test_certify_R1_testset_cg_solve(name):
    sdp = load_problem_from_file(name)
    Y_0 = sdp.soln

    params = defaultAnalyicCenterParams()
    params.verbose = True
    params.max_iter = 25
    params.delta = 1e-5
    params.lrp_params.tau = 1e-5
    params.perturb_cost = True
    params.perturb_constraints = True
    params.eps_cost = 1e-5
    params.eps_constr = 1e-5
    params.early_stop_cert = True
    params.adaptive_perturb = True
    params.eps_mult_min = 1e-4
    params.delta = 1e-5
    params.lin_solver = LinearSolverType.MFCG_LRP

    certifier = AnalyticCenterPyTorch(
        C=sdp.C,
        rho=sdp.rho,
        A_list=sdp.A,
        b=sdp.b,
        params=params,
    )

    result = certifier.certify(Y_0)

    assert result.certified, ValueError(f"Certification failed")
    assert result.min_eig > -params.tol_cert_psd, ValueError(
        f"Minimum eigenvalue negative: {result.min_eig}"
    )
    assert result.complementarity < params.tol_cert_complementarity, ValueError(
        f"Complementarity gap too large: {result.complementarity}"
    )

"""Tests for AnalyticCenterPyTorch using Lovasz theta fixtures."""

from __future__ import annotations

import pytest
import torch
import scipy.sparse as sp

from ranktools_pytorch.analytic_center_torch import (
    AnalyticCenterPyTorch,
    defaultAnalyicCenterParams,
)
from ranktools_pytorch.lin_alg_torch import LowRankPrecond
from .fixtures import (
    clique1_adj,
    clique2_adj,
    clique3_adj,
    clique4_adj,
    make_lovasz_test_case,
)

LOVASZ_TESTS = [
    (clique1_adj, [1, 3, 4, 6, 7, 8], "Clique1"),
    (clique2_adj, [0, 2, 3, 5, 6, 8, 9], "Clique2"),
    (clique3_adj, [4, 10, 13, 14, 15, 16, 17, 18], "Clique3_Large20x20"),
    (clique4_adj, [0, 1, 2], "Clique4_Disconnected"),
]


def _symmetrize_dense(mat: torch.Tensor) -> torch.Tensor:
    return mat + mat.T - torch.diag(mat.diag())


def _symmetrize_sparse_to_torch(mat: sp.spmatrix) -> torch.Tensor:
    dense = torch.tensor(mat.toarray(), dtype=torch.float64)
    return _symmetrize_dense(dense)


def _build_explicit_B(
    X: torch.Tensor, A_list, C: torch.Tensor, scale: float
) -> torch.Tensor:
    mats = [_symmetrize_sparse_to_torch(A) for A in A_list]
    mats.append(C.to(dtype=torch.float64))

    m = len(mats)
    B = torch.zeros((m, m), dtype=torch.float64)
    for i in range(m):
        for j in range(m):
            B[i, j] = torch.trace(mats[i].T @ X @ mats[j] @ X) * scale
    return B


def _build_rhs_d(
    X: torch.Tensor,
    A_list,
    C: torch.Tensor,
    b: torch.Tensor,
    rho: float,
    eps_mult: float,
    eps_constr: float,
    eps_cost: float,
    perturb_constraints: bool,
    perturb_cost: bool,
) -> torch.Tensor:
    dtype = X.dtype
    device = X.device
    m = len(A_list) + 1
    d = torch.zeros(m, dtype=dtype, device=device)

    C_sym = C + C.T - torch.diag(torch.diagonal(C))
    trace_C = torch.diagonal(C_sym).sum()

    for i in range(len(A_list)):
        A_i = A_list[i]
        trace_Ai = float(A_i.diagonal().sum())
        A_i_dense = torch.from_numpy(A_i.toarray()).to(dtype=dtype, device=device)
        A_i_sym = A_i_dense + A_i_dense.T - torch.diag(torch.diagonal(A_i_dense))
        trace_Ai_X = (A_i_sym * X).sum()

        if perturb_constraints:
            val = b[i] + eps_mult * eps_constr * trace_Ai
        else:
            val = b[i]

        violation = trace_Ai_X - val
        d[i] = trace_Ai_X + violation

    trace_C_X = (C_sym * X).sum()
    if perturb_cost:
        val_cost = rho + eps_mult * eps_cost * trace_C
    else:
        val_cost = rho
    violation_cost = trace_C_X - val_cost
    d[-1] = trace_C_X + violation_cost

    return d


@pytest.mark.parametrize("adj,clique,name", LOVASZ_TESTS)
def test_certify_lovasz_theta(adj, clique, name):
    sdp = make_lovasz_test_case(adj, clique, name)
    Y_0 = sdp.make_solution(1)

    params = defaultAnalyicCenterParams()
    params.verbose = True
    params.max_iter = 25
    params.delta = 1e-3
    params.lrp_params.tau = 1e-3

    certifier = AnalyticCenterPyTorch(
        C=sdp.C,
        rho=sdp.rho,
        A_list=sdp.A,
        b=sdp.b,
        params=params,
        device=torch.device("cpu"),
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
    assert result.min_eig > -params.tol_cert_psd, ValueError(f"Minimum eigenvalue negative: {result.min_eig}")
    assert result.complementarity < params.tol_cert_complementarity, ValueError(f"Complementarity gap too large: {result.complementarity}")
    

# @pytest.mark.parametrize("adj,clique,name", LOVASZ_TESTS)
def test_solve_for_multipliers_matches_explicit():
    adj, clique, name = LOVASZ_TESTS[0]
    sdp = make_lovasz_test_case(adj, clique, name)
    Y_0 = sdp.make_solution(1)
    tau = 1e-5
    X = Y_0 @ Y_0.T + tau * torch.eye(sdp.dim, dtype=torch.float64)

    params = defaultAnalyicCenterParams()
    params.verbose = False
    params.perturb_constraints = False
    params.perturb_cost = False
    params.reuse_multipliers = True
    params.lin_solve_max_iter = 200
    params.lin_solve_tol = 1e-14
    params.lrp_params.tau = tau

    certifier = AnalyticCenterPyTorch(
        C=sdp.C,
        rho=sdp.rho,
        A_list=sdp.A,
        b=sdp.b,
        params=params,
        device=torch.device("cpu"),
    )
    certifier.precond = LowRankPrecond(
        U=Y_0,
        A_list=sdp.A,
        C=sdp.C,
        tau=tau,
        method="DenseLDLT",
    )
    # certifier.multipliers_stored_ = torch.zeros(len(sdp.A) + 1, dtype=torch.float64)
    certifier.multipliers_stored_ = None

    multipliers = certifier._solve_for_multipliers(X, eps_mult=1.0, Y_0=Y_0)

    d = _build_rhs_d(
        X=X,
        A_list=sdp.A,
        C=sdp.C,
        b=sdp.b,
        rho=sdp.rho,
        eps_mult=1.0,
        eps_constr=params.eps_constr,
        eps_cost=params.eps_cost,
        perturb_constraints=params.perturb_constraints,
        perturb_cost=params.perturb_cost,
    )
    B_explicit = _build_explicit_B(X, sdp.A, sdp.C, scale=1.0)

    torch.testing.assert_close(
        B_explicit @ multipliers,
        d,
        rtol=1e-6,
        atol=1e-8,
    )
    torch.testing.assert_close(
        certifier.multipliers_stored_,
        multipliers,
        rtol=0.0,
        atol=0.0,
    )

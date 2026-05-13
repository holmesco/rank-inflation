"""Tests for matrix-free linear system operators."""

from __future__ import annotations

import pytest
import torch

from ranktools_pytorch.lin_alg_torch import (
    MatrixFreeLagrangeOperator,
    SparseLDLTPreconditioner,
)
from .fixtures import (
    clique1_adj,
    clique2_adj,
    clique3_adj,
    clique4_adj,
    make_lovasz_test_case,
)


def _symmetrize_dense(mat: torch.Tensor) -> torch.Tensor:
    return mat + mat.T - torch.diag(mat.diag())


def _symmetrize_sparse_to_torch(mat) -> torch.Tensor:
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


def _condition_number_from_eigvals(matrix: torch.Tensor) -> torch.Tensor:
    eigvals = torch.linalg.eigvals(matrix)
    abs_eigvals = eigvals.abs()
    min_abs = abs_eigvals.min()
    max_abs = abs_eigvals.max()
    return max_abs / min_abs


@pytest.mark.parametrize(
    "adj,clique,name",
    [
        (clique1_adj, [1, 3, 4, 6, 7, 8], "Clique1"),
        (clique2_adj, [0, 2, 3, 5, 6, 8, 9], "Clique2"),
        (clique3_adj, [4, 10, 13, 14, 15, 16, 17, 18], "Clique3_Large20x20"),
        (clique4_adj, [0, 1, 2], "Clique4_Disconnected"),
    ],
)
def test_matrix_free_operator_matches_explicit(adj, clique, name):
    sdp = make_lovasz_test_case(adj, clique, name)
    Y_0 = sdp.make_solution(1)
    X = Y_0 @ Y_0.T + 1e-3 * torch.eye(
        sdp.dim, dtype=torch.float64
    )  # Perturb to ensure PSD

    scale = 1.0
    lin_op = MatrixFreeLagrangeOperator(X=X, A_list=sdp.A, C=sdp.C, scale=scale)

    B_explicit = _build_explicit_B(X, sdp.A, sdp.C, scale)
    m = B_explicit.shape[0]

    B_implicit = torch.zeros_like(B_explicit)
    for i in range(m):
        e_i = torch.zeros(m, dtype=torch.float64)
        e_i[i] = 1.0
        B_implicit[:, i] = lin_op.matvec(e_i)

    torch.testing.assert_close(
        B_implicit,
        B_explicit,
        rtol=1e-8,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "adj,clique,name",
    [
        (clique1_adj, [1, 3, 4, 6, 7, 8], "Clique1"),
        (clique2_adj, [0, 2, 3, 5, 6, 8, 9], "Clique2"),
        (clique3_adj, [4, 10, 13, 14, 15, 16, 17, 18], "Clique3_Large20x20"),
        (clique4_adj, [0, 1, 2], "Clique4_Disconnected"),
    ],
)
def test_sparse_ldlt_preconditioner_conditioning(adj, clique, name):
    sdp = make_lovasz_test_case(adj, clique, name)
    Y_0 = sdp.make_solution(1)
    X = Y_0 @ Y_0.T + 1e-3 * torch.eye(sdp.dim, dtype=torch.float64)

    scale = 1.0
    B_explicit = _build_explicit_B(X, sdp.A, sdp.C, scale)
    m = B_explicit.shape[0]

    precond = SparseLDLTPreconditioner(X=X, A_list=sdp.A, C=sdp.C, tau=1e-3)

    PB = torch.zeros_like(B_explicit)
    for i in range(m):
        PB[:, i] = precond.solve(B_explicit[:, i])

    cond_est = _condition_number_from_eigvals(PB)
    assert cond_est < 1.2, f"Preconditioned operator poorly conditioned: {cond_est}"

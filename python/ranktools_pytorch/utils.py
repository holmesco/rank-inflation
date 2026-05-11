"""
Utility functions for PyTorch implementation.
"""

from typing import List, Tuple
import torch
import scipy.sparse as sp


def line_search_factorization(
    Z: torch.Tensor,
    dZ: torch.Tensor,
    alpha_init: float = 1.0,
    alpha_min: float = 1e-10,
    reduction_factor: float = 0.8,
) -> Tuple[float, torch.Tensor]:
    """
    Line search to maintain PSD constraint using Cholesky factorization.

    Finds the largest step size alpha in [0, 1] such that Z + alpha * dZ
    remains positive definite.

    Args:
        Z: Current matrix (n × n), must be PSD
        dZ: Search direction (n × n)
        alpha_init: Initial step size (default: 1.0)
        alpha_min: Minimum step size before failure (default: 1e-10)
        reduction_factor: Backtracking factor (default: 0.8)

    Returns:
        (alpha, L) - step size and Cholesky factor L of Z_new
    """
    alpha = alpha_init
    device = Z.device
    dtype = Z.dtype

    while alpha > alpha_min:
        Z_new = Z + alpha * dZ
        try:
            # Fast Cholesky factorization on GPU
            L = torch.linalg.cholesky(Z_new)
            return alpha, L
        except RuntimeError:
            # Not PSD, reduce step size
            alpha *= reduction_factor

    raise RuntimeError(
        f"Line search failed: cannot maintain PSDness. "
        f"Final alpha = {alpha:.6e}, min_alpha = {alpha_min:.6e}"
    )


def eval_constraints(
    X: torch.Tensor,
    A_list: List[sp.spmatrix],
    b: torch.Tensor,
    C: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    """
    Evaluate constraint violations at X.

    For each constraint A_i:
        violation_i = tr(A_i^T @ X) - b_i

    For cost constraint:
        violation_m = tr(C^T @ X) - rho

    Args:
        X: Current primal solution (n × n)
        A_list: List of m-1 sparse constraint matrices
        b: Constraint RHS values (m-1,)
        C: Cost matrix (n × n)
        rho: Cost constraint value

    Returns:
        Violation vector (m,)
    """
    device = X.device
    dtype = X.dtype
    m = len(A_list) + 1

    violations = torch.zeros(m, dtype=dtype, device=device)

    # Evaluate sparse constraints
    for i in range(len(A_list)):
        A_i_dense = torch.from_numpy(A_list[i].toarray()).to(dtype=dtype, device=device)
        violations[i] = (A_i_dense * X).sum() - b[i]

    # Evaluate cost constraint
    violations[-1] = (C * X).sum() - rho

    return violations


def build_constraint_matrix_scipy(
    A_list: List[sp.spmatrix],
    C_np: callable,
    n: int,
) -> sp.csr_matrix:
    """
    Build vectorized constraint matrix A_bar for scipy sparse operations.

    Creates matrix of shape (n^2, m) with columns:
    [vec(A_1), vec(A_2), ..., vec(A_m), vec(C)]

    Args:
        A_list: List of sparse constraint matrices
        C_np: Cost matrix as numpy array
        n: Dimension of matrices

    Returns:
        Sparse constraint matrix
    """
    m = len(A_list) + 1
    triplets = []

    # Vectorize each constraint
    for k in range(len(A_list)):
        A_k = A_list[k].tocoo()
        for i, j, v in zip(A_k.row, A_k.col, A_k.data):
            row = i * n + j
            col = k
            triplets.append((row, col, v))

    # Vectorize cost
    C_sparse = sp.csr_matrix(C_np)
    C_coo = C_sparse.tocoo()
    for i, j, v in zip(C_coo.row, C_coo.col, C_coo.data):
        row = i * n + j
        col = len(A_list)
        triplets.append((row, col, v))

    if triplets:
        rows, cols, vals = zip(*triplets)
    else:
        rows, cols, vals = [], [], []

    A_bar = sp.csr_matrix((vals, (rows, cols)), shape=(n * n, m))
    return A_bar


def sparse_upper_triangular_to_symmetric(A_sparse: sp.spmatrix) -> sp.csr_matrix:
    """
    Convert upper-triangular sparse matrix to full symmetric matrix.

    Args:
        A_sparse: Upper-triangular sparse matrix

    Returns:
        Full symmetric matrix (A + A^T - diag(A))
    """
    return A_sparse + A_sparse.T - sp.diags(A_sparse.diagonal())


def get_multipliers_newton(
    X: torch.Tensor,
    Y_0: torch.Tensor,
    A_list: List[sp.spmatrix],
    b: torch.Tensor,
    C: torch.Tensor,
    rho: float,
    eps_mult: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Lagrange multipliers for the Newton step (simplified version).

    This is a placeholder for the full Newton system solve.
    In practice, this would solve the KKT system to get multipliers.

    Args:
        X: Current primal solution
        Y_0: Initial solution
        A_list: Constraint matrices
        b: Constraint RHS
        C: Cost matrix
        rho: Cost constraint value
        eps_mult: Perturbation multiplier

    Returns:
        (multipliers, violation) - multiplier vector and constraint violations
    """
    # Placeholder: would implement full KKT solve
    # For now, return approximate multipliers
    violation = eval_constraints(X, A_list, b, C, rho)
    m = len(A_list) + 1
    multipliers = torch.ones(m, dtype=X.dtype, device=X.device)

    return multipliers, violation

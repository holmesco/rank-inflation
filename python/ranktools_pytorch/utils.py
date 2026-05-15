"""
Utility functions for PyTorch implementation.
"""

from typing import List, Tuple
import torch
import scipy.sparse as sp


def line_search_factorization(
    X: torch.Tensor,
    dX: torch.Tensor,
    alpha_init: float = 1.0,
    alpha_min: float = 1e-10,
    reduction_factor: float = 0.8,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Line search to maintain PSD constraint using Cholesky factorization.

    Finds the largest step size alpha in [0, 1] such that X + alpha * dX
    remains positive definite.

    Args:
        X: Current matrix (n × n), must be PSD
        dX: Search direction (n × n)
        alpha_init: Initial step size (default: 1.0)
        alpha_min: Minimum step size before failure (default: 1e-10)
        reduction_factor: Backtracking factor (default: 0.8)

    Returns:
        (alpha, X_new, L) - step size, updated matrix, and Cholesky factor L
    """
    alpha = alpha_init
    device = X.device
    dtype = X.dtype

    while alpha > alpha_min:
        X_new = X + alpha * dX
        try:
            # Fast Cholesky factorization on GPU
            L = torch.linalg.cholesky(X_new)
            return alpha, X_new, L
        except RuntimeError:
            # Not PSD, reduce step size
            alpha *= reduction_factor

    # Minimum step size reached
    # Need to handle this case better
    X_new = X + alpha_min * dX 
    L = None
    return alpha_min, X_new, L


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

    # TODO note sure why we are converting to dense here. We should be able to do sparse-dense multiplication on GPU directly. Once running should test this.
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


def symmetrize_dense(mat: torch.Tensor) -> torch.Tensor:
    """Symmetrize a dense matrix by copying upper triangle to lower."""
    return mat + mat.T - torch.diag(mat.diag())


def symmetrize_sparse_to_torch(mat: sp.spmatrix) -> torch.Tensor:
    """Convert sparse matrix to dense torch tensor and symmetrize."""
    dense = torch.tensor(mat.toarray(), dtype=torch.float64)
    return symmetrize_dense(dense)

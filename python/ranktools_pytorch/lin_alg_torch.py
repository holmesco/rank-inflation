"""
Linear algebra utilities for PyTorch including matrix-free operators and preconditioners.
"""

from typing import List, Tuple, Optional, Callable
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np


def sparse_upper_dot_dense(
    A_upper_sparse: sp.spmatrix, M: torch.Tensor
) -> torch.Tensor:
    """
    Compute tr(A_upper^T @ M) where A_upper is upper-triangular sparse and M is dense.

    For symmetric matrices stored in upper-triangular format:
    - Diagonal elements contribute once: M[i,i] * A[i,i]
    - Upper off-diagonal elements contribute twice: (M[i,j] + M[j,i]) * A[i,j]

    Args:
        A_upper_sparse: Upper-triangular sparse matrix (scipy format)
        M: Dense matrix on GPU/CPU

    Returns:
        Scalar tensor result
    """
    A_coo = A_upper_sparse.tocoo()

    result = 0.0
    for i, j, v in zip(A_coo.row, A_coo.col, A_coo.data):
        if i == j:
            # Diagonal element
            result += v * M[i, i].item()
        elif j > i:
            # Upper triangular element (symmetric)
            result += v * (M[i, j].item() + M[j, i].item())

    return torch.tensor(result, dtype=M.dtype, device=M.device)


class MatrixFreeLagrangeOperator:
    """
    Matrix-free linear operator for the Lagrange multiplier system.

    Implements B * y = A_bar^T (X ⊗ X) A_bar * y without explicitly forming
    the dense matrix. Complexity: O(n^2 * m) where n is SDP dimension, m = #constraints.
    """

    def __init__(
        self,
        X: torch.Tensor,
        A_list: List[sp.spmatrix],
        C: torch.Tensor,
        scale: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize matrix-free operator.

        Args:
            X: Current primal solution (n × n, symmetric, dense)
            A_list: List of m sparse constraint matrices (scipy sparse format, upper-triangular)
            C: Cost matrix (n × n, dense)
            scale: Scaling factor for the operator (perturbation parameter)
            device: Torch device (CPU or GPU)
        """
        self.X = X.to(device)
        self.C = C.to(device)
        self.A_list = A_list
        self.scale = scale
        self.device = device
        self.n = X.shape[0]
        self.m = len(A_list) + 1  # +1 for cost constraint

    def matvec(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute B * y = A_bar^T (X ⊗ X) A_bar * y.

        Args:
            y: Vector of multipliers (m,)

        Returns:
            Result vector (m,)
        """
        y = y.to(self.device)

        # Step 1: Build S = sum_i A_i * y_i + C * y_{m-1}
        # Start with cost term
        S = y[-1] * self.C

        # Add constraint terms (convert sparse to dense on GPU)
        for i in range(len(self.A_list)):
            A_i_dense = torch.from_numpy(self.A_list[i].toarray()).to(
                dtype=torch.float64, device=self.device
            )
            S = S + y[i] * A_i_dense

        # Ensure S is symmetric (since A_i are symmetric, copy lower to upper)
        S = S + S.T - torch.diag(S.diag())

        # Step 2: Compute X @ S @ X (batched matrix multiplications on GPU)
        XS = self.X @ S
        XSX = XS @ self.X.T

        # Step 3: Compute output traces tr(A_i^T @ XSX) for each constraint
        output = torch.zeros(self.m, dtype=torch.float64, device=self.device)

        for i in range(len(self.A_list)):
            output[i] = sparse_upper_dot_dense(self.A_list[i], XSX) * self.scale

        # Cost constraint trace
        output[-1] = (self.C * XSX).sum() * self.scale

        return output


class SparseLDLTPreconditioner:
    """
    Sparse LDLT preconditioner for the Lagrange multiplier system.

    Uses scipy.sparse for sparse factorization (on CPU), provides interface
    for preconditioning on GPU via forward/backward solves.

    Implementation follows equation 23 of Zhang and Lavaei 2017.
    """

    def __init__(
        self,
        X: torch.Tensor,
        A_list: List[sp.spmatrix],
        C: torch.Tensor,
        tau: float = 1e-5,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize and build sparse LDLT preconditioner.

        Args:
            X: Current primal solution (n × n)
            A_list: List of m sparse constraint matrices (upper-triangular)
            C: Cost matrix (n × n)
            tau: Diagonal perturbation parameter
            device: Torch device
        """
        self.X = X
        self.A_list = A_list
        self.C = C
        self.tau = tau
        self.device = device
        self.n = X.shape[0]
        self.m = len(A_list) + 1
        self.is_initialized = False

        self.ldlt_factor = None
        self._build_preconditioner()

    def _build_preconditioner(self):
        """Build the sparse LDLT factorization of the augmented system."""
        # Step 1: Build constraint matrix A_bar = [vec(A1) ... vec(Am) vec(C)]
        A_bar = self._build_constraint_matrix()

        # Step 2: Build top-right block V = A_bar^T @ (U ⊗ Z)
        # For simplicity, use low-rank approximation if needed
        V_sparse = self._build_top_right_block()

        # Step 3: Assemble augmented system (stays sparse)
        # [tau^2 * (A_bar^T A_bar),  tau * V;
        #  tau * V^T,                -tau^2 * I]
        tau2 = self.tau**2
        AtA = A_bar.T @ A_bar

        # Build triplet format for efficiency
        row_indices = []
        col_indices = []
        values = []

        # Top-left block: tau^2 * AtA
        AtA_coo = AtA.tocoo()
        for i, j, v in zip(AtA_coo.row, AtA_coo.col, AtA_coo.data):
            row_indices.append(i)
            col_indices.append(j)
            values.append(tau2 * v)

        # Top-right block: tau * V and bottom-left: tau * V^T
        V_coo = V_sparse.tocoo()
        for i, j, v in zip(V_coo.row, V_coo.col, V_coo.data):
            row_indices.append(i)
            col_indices.append(self.m + j)
            values.append(self.tau * v)
            # Symmetric: V^T
            row_indices.append(self.m + j)
            col_indices.append(i)
            values.append(self.tau * v)

        # Bottom-right block: -tau^2 * I
        r_dim = V_sparse.shape[1]
        for i in range(r_dim):
            row_indices.append(self.m + i)
            col_indices.append(self.m + i)
            values.append(-tau2)

        # Assemble sparse matrix
        n_sys = self.m + r_dim
        augmented_sys = sp.csr_matrix(
            (values, (row_indices, col_indices)), shape=(n_sys, n_sys)
        )
        augmented_sys = augmented_sys.tocsc()
        augmented_sys.sum_duplicates()

        # Factorize with sparse LDLT
        try:
            self.ldlt_factor = spla.splu(
                augmented_sys, permc_spec="COLAMD", diag_pivot_thresh=0.1
            )
            self.is_initialized = True
        except Exception as e:
            raise RuntimeError(f"Sparse LDLT factorization failed: {e}")

    def _build_constraint_matrix(self) -> sp.csr_matrix:
        """
        Build vectorized constraint matrix A_bar = [vec(A1) ... vec(Am) vec(C)].

        Returns:
            Sparse matrix of shape (n^2, m)
        """
        n = self.n
        m_const = self.m - 1  # Number of linear constraints

        triplets = []

        # Vectorize each constraint matrix
        for k in range(m_const):
            A_k = self.A_list[k].tocoo()
            for i, j, v in zip(A_k.row, A_k.col, A_k.data):
                row = i * n + j
                col = k
                triplets.append((row, col, v))

        # Vectorize cost matrix C
        C_np = self.C.cpu().numpy()
        C_sparse = sp.csr_matrix(C_np)
        C_coo = C_sparse.tocoo()
        for i, j, v in zip(C_coo.row, C_coo.col, C_coo.data):
            row = i * n + j
            col = m_const
            triplets.append((row, col, v))

        rows, cols, vals = zip(*triplets)
        A_bar = sp.csr_matrix((vals, (rows, cols)), shape=(n * n, self.m))
        return A_bar

    def _build_top_right_block(self) -> sp.csr_matrix:
        """
        Build top-right block V = A_bar^T @ (U ⊗ Z).

        For simplicity, returns zero matrix (can be improved with low-rank approx).

        Returns:
            Sparse matrix of shape (m, r)
        """
        # Placeholder: low-rank structure not used in basic version
        return sp.csr_matrix((self.m, 0))

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        """
        Apply preconditioner: x = M^{-1} @ b.

        Args:
            b: Right-hand side vector (m,) on GPU

        Returns:
            Solution vector (m,) on GPU
        """
        if not self.is_initialized:
            raise RuntimeError("Preconditioner not initialized")

        # Move to CPU for sparse solve
        b_np = b.cpu().numpy()

        # Solve with sparse LDLT
        x_np = self.ldlt_factor.solve(b_np)

        # Move back to original device
        x = torch.from_numpy(x_np).to(dtype=torch.float64, device=self.device)

        return x

"""
Linear algebra utilities for PyTorch including matrix-free operators and preconditioners.
"""

from typing import List, Tuple, Optional, Callable
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import numpy as np
from torch.profiler import record_function

from ranktools_pytorch.certificate import build_adjoint, build_adjoint_batched
from ranktools import LowRankPrecondMethod


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


class KKTMatrixOperator:
    """
    Matrix-free linear operator for the Lagrange multiplier system.

    Implements B * y = A_bar^T (X ⊗ X) A_bar * y without explicitly forming
    the dense matrix. Complexity: O(n^2 * m) where n is SDP dimension, m = #constraints.
    """

    def __init__(
        self,
        X: torch.Tensor,
        A_batch: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize matrix-free operator.

        Args:
            X: Current primal solution (n × n, symmetric, dense)
            A_list: List of m sparse constraint matrices (scipy sparse format, upper-triangular)
            C: Cost matrix (n × n, dense), only upper-triangular part is used
            scale: Scaling factor for the operator (perturbation parameter)
            device: Torch device (CPU or GPU)
        """
        self.X = X.to(device)
        self.scale = scale
        self.device = device
        self.n = X.shape[0]
        self.m = A_batch.shape[0]
        self.A_batch = A_batch.to(device)

    def matvec(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute B * y = A_bar^T (X ⊗ X) A_bar * y.

        Args:
            y: Vector of multipliers (m,)

        Returns:
            Result vector (m,)
        """
        return self._matvec_batched(y)

    def _matvec_batched(self, y: torch.Tensor) -> torch.Tensor:
        # Build Adjoint
        S = build_adjoint_batched(y, self.A_batch, scale=1.0)
        # Compute X @ S @ X
        XS = self.X @ S
        XSX = XS @ self.X.T
        # Compute output traces tr(A_i^T @ XSX) = A_i dot XSX for each constraint using batch
        out = (self.A_batch * XSX).sum(dim=(1, 2)) * self.scale
        return out

    def matvec_loop(self, y: torch.Tensor) -> torch.Tensor:
        y = y.to(self.device)

        # Build Adjoint
        S = build_adjoint(y, self.A_list, torch.triu(self.C), scale=1.0)

        # Step 2: Compute X @ S @ X
        XS = self.X @ S
        XSX = XS @ self.X.T

        # Step 3: Compute output traces tr(A_i^T @ XSX) for each constraint
        output = torch.zeros(self.m, dtype=torch.float64, device=self.device)

        for i in range(len(self.A_list)):
            output[i] = sparse_upper_dot_dense(self.A_list[i], XSX) * self.scale

        # Cost constraint trace
        C_sym = self.C + self.C.T - torch.diag(torch.diagonal(self.C))
        output[-1] = (C_sym * XSX).sum() * self.scale

        return output


class DiagonalPreconditioner:
    """
    Diagonal preconditioner for the matrix-free Lagrange multiplier system.

    Mirrors the C++ MultiplierDiagPreconditioner by using
    diag_i = tr((A_i X) (A_i X)) and diag_cost = tr((C X) (C X)).
    """

    def __init__(
        self,
        X: Optional[torch.Tensor] = None,
        A_list: Optional[List[sp.spmatrix]] = None,
        C: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.inv_diag: Optional[torch.Tensor] = None
        self.is_initialized = False
        self.scale = scale
        self.device = device
        self.X: Optional[torch.Tensor] = X
        self.A_list: Optional[List[sp.spmatrix]] = A_list
        self.C: Optional[torch.Tensor] = C

        if X is not None and A_list is not None and C is not None:
            self.compute_from_data(X, A_list, C, scale)

    @staticmethod
    def _symmetrize_dense_upper(mat: torch.Tensor) -> torch.Tensor:
        upper = torch.triu(mat)
        return upper + upper.T - torch.diag(torch.diagonal(upper))

    def compute(self, op: KKTMatrixOperator) -> "DiagonalPreconditioner":
        """Compute diagonal preconditioner from a matrix-free operator."""
        return self.compute_from_data(op.X, op.A_list, op.C, op.scale)

    def compute_from_data(
        self,
        X: Optional[torch.Tensor],
        A_list: Optional[List[sp.spmatrix]] = None,
        C: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> "DiagonalPreconditioner":
        """Compute diagonal preconditioner from problem data."""
        if X is not None:
            self.X = X
        if A_list is not None:
            self.A_list = A_list
        if C is not None:
            self.C = C

        if self.X is None or self.A_list is None or self.C is None:
            raise RuntimeError("Preconditioner data not set")

        X = self.X.to(self.device)
        C = self.C.to(self.device)
        A_list = self.A_list
        self.scale = scale

        m = len(A_list) + 1
        inv_diag = torch.zeros(m, dtype=X.dtype, device=self.device)

        for i, A_i in enumerate(A_list):
            A_i_dense = torch.from_numpy(A_i.toarray()).to(
                dtype=X.dtype, device=self.device
            )
            A_i_sym = self._symmetrize_dense_upper(A_i_dense)
            AX = A_i_sym @ X
            diag_val = torch.trace(AX @ AX) * self.scale
            inv_diag[i] = 1.0 / diag_val if diag_val != 0 else 1.0

        C_sym = self._symmetrize_dense_upper(C)
        CX = C_sym @ X
        diag_cost = torch.trace(CX @ CX) * self.scale
        inv_diag[m - 1] = 1.0 / diag_cost if diag_cost != 0 else 1.0

        self.inv_diag = inv_diag
        self.is_initialized = True
        return self

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        if not self.is_initialized or self.inv_diag is None:
            raise RuntimeError("Preconditioner not initialized")
        return self.inv_diag * b


class LowRankPrecond:
    """
    Low-rank preconditioner for the Lagrange multiplier system.

    Implements the DenseLDLT and SparseLDLT variants of the augmented-system
    preconditioner from Zhang and Lavaei (2017). The construction mirrors the
    C++ implementation in lin_alg_tools.hpp.
    """

    def __init__(
        self,
        A_list: List[sp.spmatrix],
        C: torch.Tensor,
        tau: float = 1e-5,
        method: LowRankPrecondMethod = LowRankPrecondMethod.DenseLDLT,
        use_approx: bool = False,
        device: torch.device = torch.device("cpu"),
        solve_device: torch.device = torch.device("cpu"),
    ):
        self.A_list = A_list
        self.C = torch.triu(C).to(device)
        self.tau = tau
        self.method = method
        self.use_approx = use_approx
        self.device = device
        # Device where the main solve is taking place
        self.solve_device = solve_device
        self.m = len(A_list) + 1
        self.scale = 1.0
        self.is_initialized = False

        # Intialize storage
        self._sparse_factor = None
        self._dense_ldlt = None
        self._dense_lu = None
        self.Sys = None

    def build_preconditioner(self, U: torch.Tensor) -> None:
        # Store the low-rank factor U on the specified device
        self.U = U.to(self.device)
        self.n = U.shape[0]
        self.r = U.shape[1]
        # Build the augmented system once
        self.build_aug_sys()
        # Build the preconditioner
        if self.method == LowRankPrecondMethod.DenseLDLT:
            self.build_ldlt_dense()
        elif self.method == LowRankPrecondMethod.DenseQR:
            self.build_lu_dense()
        elif self.method == LowRankPrecondMethod.SparseLDLT:
            self.build_ldlt_sparse()
        else:
            raise NotImplementedError(
                f"LowRankPrecond only supports DenseLDLT and SparseLDLT. "
                f"Got method={self.method}."
            )

    def build_lu_dense(self) -> None:
        if self.Sys is None:
            raise RuntimeError("LowRankPrecond: Augmented system not built.")

        # Perform LDLT factorization (returns LD factorization and pivots)
        LU, pivots, info = torch.linalg.lu_factor_ex(self.Sys)
        if torch.any(info != 0):
            raise RuntimeError(f"Dense LDLT factorization failed with info={info}")
        # Move factorization to solve device if different from construction device
        if self.solve_device != self.device:
            LU = LU.to(self.solve_device)
            pivots = pivots.to(self.solve_device)
        # Store the factorization for use in solves
        self._dense_lu = (LU, pivots)
        # Mark initialized
        self.is_initialized = True

    def build_ldlt_dense(self) -> None:
        if self.Sys is None:
            raise RuntimeError("LowRankPrecond: Augmented system not built.")

        # Perform LDLT factorization (returns LD factorization and pivots)
        LD, pivots, info = torch.linalg.ldl_factor_ex(self.Sys)
        if torch.any(info != 0):
            raise RuntimeError(f"Dense LDLT factorization failed with info={info}")
        # Move factorization to solve device if different from construction device
        if self.solve_device != self.device:
            LD = LD.to(self.solve_device)
            pivots = pivots.to(self.solve_device)
        # Store the factorization for use in solves
        self._dense_ldlt = (LD, pivots)
        # Mark initialized
        self.is_initialized = True

    def build_ldlt_sparse(self) -> None:
        if self.U is None or self.C is None or self.A_list is None:
            raise RuntimeError("LowRankPrecond: Problem data not set.")

        A_bar = self.build_constraint_matrix(self.A_list, self.C)
        V_sparse = self.build_top_right_sparse(self.U, self.tau)
        V_sparse = V_sparse * self.tau
        V_sparse = V_sparse.tocsc()

        rdim = V_sparse.shape[1]
        nsys = self.m + rdim
        tau2 = self.tau**2

        AtA = A_bar.T @ A_bar

        row_indices = []
        col_indices = []
        values = []

        AtA_coo = AtA.tocoo()
        for i, j, v in zip(AtA_coo.row, AtA_coo.col, AtA_coo.data):
            row_indices.append(i)
            col_indices.append(j)
            values.append(tau2 * v)

        V_coo = V_sparse.tocoo()
        for i, j, v in zip(V_coo.row, V_coo.col, V_coo.data):
            row_indices.append(i)
            col_indices.append(self.m + j)
            values.append(v)
            row_indices.append(self.m + j)
            col_indices.append(i)
            values.append(v)

        for i in range(rdim):
            row_indices.append(self.m + i)
            col_indices.append(self.m + i)
            values.append(-tau2)

        augmented_sys = sp.csr_matrix(
            (values, (row_indices, col_indices)), shape=(nsys, nsys)
        ).tocsc()
        augmented_sys.sum_duplicates()

        try:
            self._sparse_factor = spla.splu(
                augmented_sys, permc_spec="COLAMD", diag_pivot_thresh=0.1
            )
            self.is_initialized = True
        except Exception as exc:
            raise RuntimeError(f"Sparse LDLT factorization failed: {exc}")

    def build_aug_sys(self) -> None:
        if self.U is None or self.C is None or self.A_list is None:
            raise RuntimeError("LowRankPrecond: Problem data not set.")

        A_bar = self.build_constraint_matrix(self.A_list, self.C)
        V = self.build_top_right(self.U, self.tau)

        tau2 = self.tau**2
        AtA = torch.from_numpy((A_bar.T @ A_bar).toarray()).to(
            dtype=torch.float64, device=self.device
        )
        V = torch.from_numpy(V).to(dtype=torch.float64, device=self.device)

        n_sys = self.m + V.shape[1]
        self.Sys = torch.zeros(
            (n_sys, n_sys), dtype=torch.float64, device=self.U.device
        )
        self.Sys[: self.m, : self.m] = AtA * tau2
        self.Sys[self.m :, : self.m] = V.T * self.tau
        self.Sys[: self.m, self.m :] = V * self.tau
        self.Sys[self.m :, self.m :] = -torch.eye(V.shape[1]) * tau2

    @staticmethod
    def build_constraint_matrix(
        A_list: List[sp.spmatrix], C: torch.Tensor
    ) -> sp.csr_matrix:
        dim = C.shape[0]
        ncons = len(A_list) + 1

        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []

        for i, A in enumerate(A_list):
            A_coo = A.tocoo()
            for r, c, v in zip(A_coo.row, A_coo.col, A_coo.data):
                if c > r:
                    row_idx = c * dim + r
                    rows.append(row_idx)
                    cols.append(i)
                    vals.append(v)
                    row_idx = r * dim + c
                    rows.append(row_idx)
                    cols.append(i)
                    vals.append(v)
                elif c == r:
                    row_idx = c * dim + r
                    rows.append(row_idx)
                    cols.append(i)
                    vals.append(v)

        C_np = C.detach().cpu().numpy()
        for r in range(dim):
            for c in range(r, dim):
                v = C_np[r, c]
                if v != 0.0:
                    row_idx = c * dim + r
                    rows.append(row_idx)
                    cols.append(ncons - 1)
                    vals.append(v)
                    if c != r:
                        row_idx = r * dim + c
                        rows.append(row_idx)
                        cols.append(ncons - 1)
                        vals.append(v)

        if not rows:
            return sp.csr_matrix((dim * dim, ncons))

        return sp.csr_matrix((vals, (rows, cols)), shape=(dim * dim, ncons))

    def build_top_right(self, U: torch.Tensor, tau: float) -> np.ndarray:
        U_np = U.detach().cpu().numpy()
        dim, rank = U_np.shape
        top_right = np.zeros((self.m, (rank + dim) * rank), dtype=np.float64)

        s = np.sqrt(2.0 * tau)
        uaiu_size = rank * rank
        aiu_size = dim * rank

        for i, A in enumerate(self.A_list):
            A_sym = A + A.T - sp.diags(A.diagonal())
            AiU = A_sym @ U_np
            UAiU = U_np.T @ AiU
            top_right[i, :uaiu_size] = UAiU.reshape(-1, order="F")
            top_right[i, uaiu_size:] = AiU.reshape(-1, order="F") * s

        C_sym = self._symmetrize_dense(self.C)
        CU = C_sym @ U_np
        UCU = U_np.T @ CU
        top_right[self.m - 1, :uaiu_size] = UCU.reshape(-1, order="F")
        top_right[self.m - 1, uaiu_size:] = CU.reshape(-1, order="F") * s

        return top_right

    def build_top_right_sparse(self, U: torch.Tensor, tau: float) -> sp.csr_matrix:
        top_right_dense = self.build_top_right(U, tau)
        vmax = np.max(np.abs(top_right_dense)) if top_right_dense.size else 0.0
        drop_tol = max(1e-14, 1e-12 * vmax)
        if drop_tol > 0.0:
            top_right_dense[np.abs(top_right_dense) < drop_tol] = 0.0
        return sp.csr_matrix(top_right_dense)

    @staticmethod
    def _symmetrize_dense(C: torch.Tensor) -> np.ndarray:
        C_np = C.detach().cpu().numpy()
        C_upper = np.triu(C_np)
        return C_upper + C_upper.T - np.diag(np.diag(C_upper))

    @record_function("LowRankPrecond.solve")
    def solve(self, b: torch.Tensor) -> torch.Tensor:
        if not self.is_initialized:
            raise RuntimeError("Preconditioner not initialized")

        if self.method == LowRankPrecondMethod.DenseLDLT:
            b_t = b / self.scale
            x_t = self.solveDenseLDLT(b_t)
            return x_t
        elif self.method == LowRankPrecondMethod.DenseQR:
            b_t = b / self.scale
            x_t = self.solveDenseLU(b_t)
            return x_t
        elif self.method == LowRankPrecondMethod.SparseLDLT:
            b_np = b.detach().cpu().numpy() / self.scale
            x_np = self.solveSparseLDLT(b_np)
            return torch.from_numpy(x_np).to(dtype=torch.float64, device=self.device)
        else:
            raise NotImplementedError(
                f"LowRankPrecond only supports DenseLDLT and SparseLDLT. "
                f"Got method={self.method}."
            )

    def solveDenseLDLT(self, b: torch.Tensor) -> torch.Tensor:
        # Expand dimension of b
        sysdim = self.Sys.shape[0]
        rhs = torch.zeros(sysdim, dtype=b.dtype, device=b.device)
        rhs[: self.m] = b.reshape(-1)
        # Solve and extract solution for multipliers
        LD, pivots = self._dense_ldlt
        x = torch.linalg.ldl_solve(LD, pivots, rhs[:, None])
        return x[: self.m].squeeze(-1)

    def solveDenseLU(self, b: torch.Tensor) -> torch.Tensor:
        # Expand dimension of b
        sysdim = self.Sys.shape[0]
        rhs = torch.zeros(sysdim, dtype=b.dtype, device=b.device)
        rhs[: self.m] = b.reshape(-1)
        # Solve and extract solution for multipliers
        LU, pivots = self._dense_lu
        x = torch.linalg.lu_solve(LU, pivots, rhs[:, None])
        return x[: self.m].squeeze(-1)

    def solveSparseLDLT(self, b_np: np.ndarray) -> np.ndarray:
        if self._sparse_factor is None:
            raise RuntimeError("Sparse LDLT factorization not initialized")

        sysdim = self._sparse_factor.L.shape[0]
        rhs = np.zeros(sysdim, dtype=np.float64)
        rhs[: self.m] = b_np
        return self._sparse_factor.solve(rhs)[: self.m]

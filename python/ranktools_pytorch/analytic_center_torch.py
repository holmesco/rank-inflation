"""
PyTorch implementation of analytic center certifier using MFCG_LRP solver.

This reimplements the C++ AnalyticCenter class with MFCG_LRP + Sparse LDLT
preconditioner for GPU acceleration of dense matrix operations.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import time
import torch
import scipy.sparse as sp
from torch.autograd.profiler import record_function


from .lin_alg_torch import KKTMatrixOperator, LowRankPrecond
from .solvers import ConjugateGradientSolver
from .certificate import (
    eval_certificate,
    check_certificate_psd,
    build_adjoint,
    build_adjoint_batched,
    compute_complementarity,
)
from .utils import (
    line_search_factorization,
    symmetrize_dense,
    symmetrize_sparse_to_torch,
)
from ranktools import AnalyticCenterParams, LinearSolverType, LowRankPrecondMethod


def defaultAnalyicCenterParams() -> AnalyticCenterParams:
    params = AnalyticCenterParams()

    # Verbosity
    params.verbose = True

    # Rank / step tolerances
    params.tol_rank_sol = 1.0e-4
    params.tol_step_norm = 1.0e-8
    params.max_iter = 50

    # Linear system scaling
    params.rescale_lin_sys = False
    params.rescaling_factor = 1.0e-5
    params.lin_solver = LinearSolverType.MFCG_LRP
    params.reuse_multipliers = True

    # Linear independence check
    params.tol_indep_constr = 1.0e-3
    params.check_indep_constr = False
    params.delta = 1.0e-5

    # Adaptive perturbation
    params.perturb_constraints = False
    params.perturb_cost = True
    params.eps_cost = 1.0e-5
    params.eps_constr = 1.0e-5
    params.adaptive_perturb = True
    params.eps_mult_min = 1.0e-2
    params.eps_inc_step_thresh = 0.1
    params.eps_inc = 2.0
    params.eps_dec_step_thresh = 0.9
    params.eps_dec = 0.6

    # Iterative solver
    params.lin_solve_max_iter = 500
    params.lin_solve_tol = 1.0e-5

    # Low-rank preconditioner params
    params.lrp_params.tau = 1.0e-5
    params.lrp_params.method = LowRankPrecondMethod.DenseLDLT
    params.lrp_params.use_approx = False
    params.lrp_params.ldlt_zero_thresh = 1.0e-14

    # Line search
    params.enable_line_search = True
    params.ln_search_red_factor = 0.8
    params.alpha_init = 1.0
    params.alpha_min = 1.0e-10

    # Early stop parameters
    params.early_stop_cert = True
    params.tol_cert_psd = 1.0e-5
    params.tol_cert_complementarity = 1.0e-5
    params.tol_cert_primal_feas = 1.0e-5
    params.early_stop_angle = False
    params.max_angle = 1.0e-2
    params.use_cert_centrality_metric = False
    params.tol_cert_centrality = 1.0e-5

    return params


@dataclass
class AnalyticCenterResult:
    """Result of analytic center computation."""

    X: torch.Tensor  # Primal solution (n × n)
    H: torch.Tensor  # Certificate matrix (n × n)
    multipliers: torch.Tensor  # Lagrange multipliers (m,)
    violation: torch.Tensor  # Constraint violations (m,)
    certified: bool  # Whether solution is certified
    min_eig: float  # Minimum eigenvalue of H
    complementarity: float  # First-order optimality measure
    solver_time: float  # Total solve time in seconds
    num_iters: int  # Number of centering iterations


class AnalyticCenterPyTorch:
    """
    PyTorch implementation of analytic center certifier.

    Solves the SDP:
        min  tr(C @ X)
        s.t. tr(A_i @ X) = b_i  for i = 1, ..., m
             X ⪰ 0

    Using interior point methods with analytic centering and
    MFCG_LRP (Matrix-Free Conjugate Gradient with Low-Rank Preconditioner).
    """

    def __init__(
        self,
        C: torch.Tensor,
        rho: float,
        A_list: List[sp.spmatrix],
        b: torch.Tensor,
        params: AnalyticCenterParams = defaultAnalyicCenterParams(),
        main_gpu: bool = False,
        precond_gpu: bool = False,
        batch_constraints: bool = False,
    ):
        """
        Initialize analytic center certifier.

        Args:
            C: Cost matrix (n × n, dense) only upper-triangular part is used
            rho: Constraint value for cost (scalar)
            A_list: List of m constraint matrices (sparse, upper-triangular)
            b: Right-hand side for constraints (m,)
            params: Algorithm parameters
            main_gpu: Whether to run main computations on GPU
            precond_gpu: Whether to run preconditioner computations on GPU"""
        # flags
        self.main_gpu = main_gpu
        self.precond_gpu = precond_gpu
        self.batch_constraints = batch_constraints
        if main_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.C = torch.triu(C).to(device=self.device, dtype=torch.float64)
        self.rho = rho
        self.A_list = [sp.triu(A) for A in A_list]
        self.b = b.to(device=self.device, dtype=torch.float64)
        self.params = params or AnalyticCenterParams()
        self.dim = C.shape[0]
        self.m = len(A_list) + 1  # +1 for cost constraint

        # If batching, pre-build the constraint and cost matrices into batched tensors
        self.A_batch: None | torch.Tensor = None
        self.b_batch: None | torch.Tensor = None
        if batch_constraints:
            if params.verbose:
                print("Building batched constraint matrix...")
            A_batch = torch.zeros(
                (self.m, self.dim, self.dim), dtype=torch.float64, device=self.device
            )
            for i, A in enumerate(range(len(A_list))):
                A_batch[i] = torch.from_numpy(self.A_list[i].toarray()).to(
                    dtype=torch.float64, device=self.device
                )
            A_batch[self.m - 1] = self.C
            # Symmetrize the upper triangular matrices to get the full constraint matrices
            self.A_batch = symmetrize_dense(A_batch)
            # rhs of constraints
            self.b_batch = torch.hstack(
                [
                    self.b,
                    torch.tensor(self.rho, dtype=torch.float64, device=self.device),
                ]
            )

        # CG solver (reused for each iteration)
        self.cg_solver = ConjugateGradientSolver(
            max_iter=self.params.lin_solve_max_iter,
            tol=self.params.lin_solve_tol,
        )
        # Initialize preconditioner
        precond_device = torch.device("cuda") if precond_gpu else torch.device("cpu")
        self.precond = LowRankPrecond(
            A_list=self.A_list,
            C=self.C,
            tau=self.params.lrp_params.tau,
            method="DenseLDLT",
            device=precond_device,
        )
        # Stored multipliers
        self.multipliers_stored_: None | torch.Tensor = None

    def certify(
        self,
        Y_0: torch.Tensor,
    ) -> AnalyticCenterResult:
        """
        Compute certificate for the solution Y_0 @ Y_0^T.

        This is the main entry point. Runs centering iterations to find
        the analytic center, then evaluates the optimality certificate.

        Args:
            Y_0: Low-rank initial solution (n × r)

        Returns:
            AnalyticCenterResult with certificate and statistics
        """
        import time

        start_time = time.time()

        Y_0 = Y_0.to(device=self.device, dtype=torch.float64)

        # Get analytic center
        X, multipliers, violation = self._get_analytic_center(Y_0)

        # Build certificate matrix
        multipliers = multipliers / multipliers[-1]
        H = build_adjoint(multipliers, self.A_list, self.C)

        # Evaluate certificate
        min_eig, complementarity = eval_certificate(H, Y_0)

        # Check certification
        is_psd = min_eig > -self.params.tol_cert_psd
        certified = (
            is_psd and complementarity.item() <= self.params.tol_cert_complementarity
        )

        elapsed = time.time() - start_time

        result = AnalyticCenterResult(
            X=X,
            H=H,
            multipliers=multipliers,
            violation=violation,
            certified=certified,
            min_eig=min_eig.item(),
            complementarity=complementarity.item(),
            solver_time=elapsed,
            num_iters=getattr(self, "_num_iters", 0),
        )

        if self.params.verbose:
            self._print_result(result)

        return result

    @record_function("GetAnalyticCenter")
    def _get_analytic_center(
        self,
        Y_0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute analytic center via interior point iterations.

        Args:
            Y_0: Initial low-rank solution (n × r)

        Returns:
            (X, multipliers, violation) - analytic center, Lagrange multipliers,
            and constraint violations at the returned solution
        """
        if self.params.verbose:
            print("Starting analytic center iterations...")

        # Initialize from low-rank
        X = Y_0 @ Y_0.T
        X = X.to(device=self.device)

        # Ensure PSD via line search
        try:
            alpha, X, L = line_search_factorization(
                X, self.params.delta * torch.eye(self.dim, device=self.device)
            )
        except RuntimeError:
            if self.params.verbose:
                print("Warning: Initial point not PSD, using shifted version")
            X = X + self.params.delta * torch.eye(self.dim, device=self.device)
        assert alpha == 1.0, ValueError("Line search failed to find PSD initial point")

        # Set up preconditioner
        if self.params.lin_solver == LinearSolverType.MFCG_LRP:
            precond_start = time.perf_counter()
            self.precond.build_preconditioner(U=Y_0)
            precond_elapsed = time.perf_counter() - precond_start
            if self.params.verbose:
                print(f"Preconditioner build time: {precond_elapsed:.6f} seconds")

        # Main centering loop
        eps_mult = 1.0  # Perturbation multiplier
        cert_complementarity = False
        cert_psd = False
        small_step = False
        for iter_idx in range(self.params.max_iter):
            # Get multipliers (solve KKT system via CG)
            multipliers, violation = self._solve_for_multipliers(X, eps_mult)
            violation_norm = violation.norm()
            barrier = 1 / multipliers[-1]
            # Compute the dual matrix (adjoint) for the Newton step
            if self.batch_constraints:
                S = build_adjoint_batched(multipliers, self.A_batch, scale=1.0)
            else:
                S = build_adjoint(multipliers, self.A_list, self.C)
            # Define the Newton step
            delta_X = X - X @ S @ X
            delta_X_norm = delta_X.norm()
            #  Line search for PSDness
            with record_function("LineSearch"):
                L_prev = L.clone()
                alpha, X_new, L = line_search_factorization(
                    X,
                    delta_X,
                    alpha_init=self.params.alpha_init,
                    alpha_min=self.params.alpha_min,
                )
            # Update solution
            X = X_new
            # Construct the certificate matrix by rescaling the adjoint
            H = S * barrier
            with record_function("CertificateMetrics"):
                # compute complementarity
                complementarity = compute_complementarity(H, Y_0)
                # compute centrality metric
                centrality = (
                    L_prev.T @ S @ L_prev - torch.eye(self.dim, device=self.device)
                ).norm()
                # Compute deviation angle
                cos_angle = (Y_0.T @ X @ Y_0).trace() / Y_0.norm() ** 2 / X.norm()
                angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0)).item()

            #  Print iteration info
            # TODO Logging should not use print statements because they can slow down GPU code.
            if self.params.verbose and iter_idx % 10 == 0:
                print(
                    f"{'Iter':>5} {'Violation':>12} {'StepNorm':>12} "
                    f"{'Alpha':>12} {'EpsMult':>12} {'Barrier':>12} "
                    f"{'Centrality':>12} {'Angle':>12} {'Compl':>12}"
                )

            if self.params.verbose:
                print(
                    f"{iter_idx+1:5d} {violation_norm:12.6e} {delta_X_norm:12.6e} "
                    f"{alpha:12.6e} {eps_mult:12.6e} {barrier:12.6e} "
                    f"{centrality:12.6e} {angle:12.6e} {complementarity:12.6e}"
                )

            # Early stopping with certificate
            if self.params.early_stop_cert and iter_idx > 0:
                cert_complementarity = (
                    complementarity < self.params.tol_cert_complementarity
                )
                with record_function("CheckCertificatePSD"):
                    cert_psd = check_certificate_psd(H, tol=self.params.tol_cert_psd)

            # Check step size
            small_step = delta_X_norm < self.params.tol_step_norm and (
                eps_mult <= self.params.eps_mult_min or not self.params.adaptive_perturb
            )

            # Check stopping conditions
            if (cert_complementarity and cert_psd) or (small_step):
                break

            # Update perturbation multiplier adaptively
            if self.params.adaptive_perturb:
                if alpha < self.params.eps_inc_step_thresh:
                    eps_mult = eps_mult * self.params.eps_inc
                elif alpha > self.params.eps_dec_step_thresh:
                    eps_mult = max(
                        eps_mult * self.params.eps_dec, self.params.eps_mult_min
                    )

        return X, multipliers, violation

    @record_function("SolveForMultipliers")
    def _solve_for_multipliers(self, X: torch.Tensor, eps_mult: float) -> torch.Tensor:
        """
        Solve for Lagrange multipliers using CG with matrix-free operator.

        This solves: B * lambda = d, where:
        - B is the matrix-free Lagrange multiplier operator
        - d is the right-hand side from the Newton system

        Args:
            X: Current primal solution

        Returns:
            Multipliers (m,)
        """
        # Construct RHS (matches C++ build_ac_system)
        d, violation = self.build_rhs_vector(X, eps_mult)

        if self.params.lin_solver == LinearSolverType.LDLT:
            # Build dense B matrix for direct solve
            B = self._build_explicit_B(X, scale=1.0)
            # Solve directly via Cholesky
            L = torch.linalg.cholesky(B)
            multipliers = torch.cholesky_solve(d.unsqueeze(1), L).squeeze(1)
        elif self.params.lin_solver == LinearSolverType.MFCG_LRP:
            # Define matrix-free operator for CG
            linop = KKTMatrixOperator(
                X=X,
                A_list=self.A_list,
                C=self.C,
                device=self.device,
                A_batched=self.A_batch if self.batch_constraints else None,
            )
            # Solve via CG
            assert self.precond is not None and self.precond.is_initialized, ValueError(
                "Preconditioner was not defined or was not initialized properly."
            )
            cg_result = self.cg_solver.solve(
                b=d,
                matvec_fn=linop.matvec,
                precond_solve_fn=self.precond.solve,
            )
            multipliers = cg_result.solution
            # Update stored multipliers
            if self.params.reuse_multipliers:
                self.multipliers_stored_ = multipliers

        return multipliers, violation

    @record_function("BuildRHSVector")
    def build_rhs_vector(
        self, X: torch.Tensor, eps_mult: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build RHS vector and violations for the KKT system.

        Args:
            X: Current primal solution
            eps_mult: Perturbation multiplier

        Returns:
            (d, violation)
        """
        dtype = X.dtype
        device = X.device
        m = len(self.A_list) + 1
        d = torch.zeros(m, dtype=dtype, device=device)

        if not self.batch_constraints:
            violation = torch.zeros(m, dtype=dtype, device=device)
            for i in range(len(self.A_list)):
                A_i = self.A_list[i]
                trace_Ai = float(A_i.diagonal().sum())
                A_i_dense = torch.from_numpy(A_i.toarray()).to(
                    dtype=dtype, device=device
                )
                A_i_sym = (
                    A_i_dense + A_i_dense.T - torch.diag(torch.diagonal(A_i_dense))
                )
                trace_Ai_X = (A_i_sym * X).sum()
                # Get adjusted rhs of constraint equation
                if self.params.perturb_constraints:
                    val = self.b[i] + eps_mult * self.params.eps_constr * trace_Ai
                else:
                    val = self.b[i]
                # Get violation
                violation[i] = trace_Ai_X - val
                # Update KKT rhs
                d[i] = trace_Ai_X + violation[i]

            # Get LHS of constraint equation
            C_sym = self.C + self.C.T - torch.diag(torch.diagonal(self.C))
            trace_C = torch.diagonal(C_sym).sum()
            trace_C_X = (C_sym * X).sum()
            # Get adjusted rhs of constraint equation
            if self.params.perturb_cost:
                val_cost = self.rho + eps_mult * self.params.eps_cost * trace_C
            else:
                val_cost = self.rho
            # Get violation
            violation[-1] = trace_C_X - val_cost
            # Update KKT rhs
            d[-1] = trace_C_X + violation[-1]
        else:
            # Batched constraints: self.A_batch already symmetrized (includes cost)
            A_batch = self.A_batch
            if A_batch is None:
                raise ValueError("A_batch was not initialized for batched constraints.")
            # Compute tr(A_i X) for all i in batch
            trace_AX = (A_batch * X).sum(dim=(1, 2))

            # Build perturbed rhs for constraints and cost
            if self.params.perturb_constraints or self.params.perturb_cost:
                trace_A = torch.diagonal(A_batch, dim1=1, dim2=2).sum(dim=1)
                b_batch = self.b_batch.clone()
                if self.params.perturb_constraints:
                    b_batch[:-1] = (
                        b_batch[:-1] + eps_mult * self.params.eps_constr * trace_A[:-1]
                    )
                if self.params.perturb_cost:
                    b_batch[-1] = (
                        b_batch[-1] + eps_mult * self.params.eps_cost * trace_A[-1]
                    )
            else:
                b_batch = self.b_batch

            # Violations and KKT rhs
            violation = trace_AX - b_batch
            d = trace_AX + violation

        return d, violation

    def _build_explicit_B(self, X: torch.Tensor, scale: float) -> torch.Tensor:
        """Build the explicit B matrix for direct solves (used in LDLT option).
        Matrix is given by B_ij = tr(A_i X A_j X) * scale, where A_i are the constraint and cost matrices.
        """
        if self.batch_constraints:
            A_batch = self.A_batch
            if A_batch is None:
                raise ValueError("A_batch was not initialized for batched constraints.")
            # A_batch already contains symmetrized constraints and cost
            # Compute AX for all constraints in batch
            AX = torch.matmul(A_batch, X)
            # B_ij = tr(A_i X A_j X) = sum_{k,l} (AX_i)_{k,l} * (AX_j)_{l,k}
            B = torch.einsum("ikl,jlk->ij", AX, AX) * scale
        else:
            mats = [symmetrize_sparse_to_torch(A) for A in self.A_list]
            mats.append(symmetrize_dense(self.C))

            m = len(mats)
            B = torch.zeros((m, m), dtype=torch.float64)
            for i in range(m):
                for j in range(m):
                    B[i, j] = torch.trace(mats[i].T @ X @ mats[j] @ X) * scale
        return B

    def _print_result(self, result: AnalyticCenterResult):
        """Print final result summary."""
        print("\n" + "=" * 60)
        print("Analytic Center Certification Result")
        print("=" * 60)
        print(f"Solver time: {result.solver_time:.6f} seconds")
        print(f"Centering iterations: {result.num_iters}")
        print(f"Minimum eigenvalue of H: {result.min_eig:.6e}")
        print(f"Complementarity: {result.complementarity:.6e}")
        print(f"Max constraint violation: {result.violation.max().item():.6e}")
        print(f"Certified: {result.certified}")
        print("=" * 60 + "\n")

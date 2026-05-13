"""
PyTorch implementation of analytic center certifier using MFCG_LRP solver.

This reimplements the C++ AnalyticCenter class with MFCG_LRP + Sparse LDLT
preconditioner for GPU acceleration of dense matrix operations.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import scipy.sparse as sp

from .lin_alg_torch import MatrixFreeLagrangeOperator, LowRankPrecond
from .solvers import ConjugateGradientSolver
from .certificate import (
    eval_certificate,
    check_certificate_psd,
    build_adjoint,
    compute_complementarity,
)
from .utils import (
    line_search_factorization,
    eval_constraints,
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
    params.lin_solver = LinearSolverType.LDLT
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
    params.lrp_params.method = LowRankPrecondMethod.SparseLDLT
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
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize analytic center certifier.

        Args:
            C: Cost matrix (n × n, dense) only upper-triangular part is used
            rho: Constraint value for cost (scalar)
            A_list: List of m constraint matrices (sparse, upper-triangular)
            b: Right-hand side for constraints (m,)
            params: Algorithm parameters
            device: Torch device (cuda or cpu)
        """
        self.C = torch.triu(C).to(device=device, dtype=torch.float64)
        self.rho = rho
        self.A_list = A_list  # Keep as scipy sparse
        self.b = b.to(device=device, dtype=torch.float64)
        self.params = params or AnalyticCenterParams()
        self.device = device

        self.n = C.shape[0]
        self.m = len(A_list) + 1  # +1 for cost constraint

        # CG solver (reused for each iteration)
        self.cg_solver = ConjugateGradientSolver(
            max_iter=self.params.lin_solve_max_iter,
            tol=self.params.lin_solve_tol,
            verbose=False,
        )
        # preconditioner
        self.precond: None | LowRankPrecond = None

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
        X, multipliers = self._get_analytic_center(Y_0)

        # Build certificate matrix
        H = build_adjoint(multipliers, self.A_list, self.C)
        H = (H + H.T) / 2  # Ensure symmetry

        # Evaluate certificate
        min_eig, complementarity = eval_certificate(H, Y_0)

        # Check certification
        is_psd = check_certificate_psd(H, tol=self.params.tol_cert_psd)
        certified = (
            is_psd and complementarity.item() <= self.params.tol_cert_complementarity
        )

        # Evaluate constraint violations at solution
        violation = eval_constraints(X, self.A_list, self.b, self.C, self.rho)

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

    def _get_analytic_center(
        self,
        Y_0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute analytic center via interior point iterations.

        Args:
            Y_0: Initial low-rank solution (n × r)

        Returns:
            (X, multipliers) - analytic center and Lagrange multipliers
        """
        # Initialize from low-rank
        X = Y_0 @ Y_0.T
        X = X.to(device=self.device)

        # Ensure PSD via line search
        try:
            alpha, X, L = line_search_factorization(
                X, self.params.delta * torch.eye(self.n, device=self.device)
            )
        except RuntimeError:
            if self.params.verbose:
                print("Warning: Initial point not PSD, using shifted version")
            X = X + self.params.delta * torch.eye(self.n, device=self.device)

        eps_mult = 1.0  # Perturbation multiplier
        multipliers = torch.ones(self.m, device=self.device, dtype=torch.float64)

        # Set up preconditioner
        self.precond = LowRankPrecond(
            U=Y_0,
            A_list=self.A_list,
            C=self.C,
            tau=self.params.precond_tau,
            device=self.device,
        )

        # Main centering loop
        cert_complementarity = False
        cert_psd = False
        small_step = False
        for iter_idx in range(self.params.max_iter):
            # Step 1: Evaluate constraints
            violation = eval_constraints(X, self.A_list, self.b, self.C, self.rho)
            violation_norm = violation.norm()

            # Step 2: Get multipliers (solve KKT system via CG)
            multipliers = self._solve_for_multipliers(X, eps_mult)
            barrier = 1 / multipliers[-1]
            # Step 3: Compute Newton step direction
            S = build_adjoint(multipliers, self.A_list, self.C)
            S = (S + S.T) / 2

            # delta_X = X - X @ S @ X (standard centering direction)
            delta_X = X - X @ S @ X
            delta_X_norm = delta_X.norm()

            # Step 4: Line search for PSDness
            alpha, X_new, L = line_search_factorization(
                X,
                delta_X,
                alpha_init=self.params.alpha_init,
                alpha_min=self.params.alpha_min,
            )
            # Step 5: Update solution
            X = X_new

            # Step 6: Print iteration info
            # TODO Logging should not use print statements because they can slow down GPU code.
            if self.params.verbose:
                print(
                    f"{'Iter':>5} {'Violation':>12} {'StepNorm':>12} "
                    f"{'Alpha':>12} {'EpsMult':>12}"
                )

            if self.params.verbose:
                print(
                    f"{iter_idx+1:5d} {violation_norm:12.6e} {delta_X_norm:12.6e} "
                    f"{alpha:12.6e} {eps_mult:12.6e}"
                )

            # Early stopping with certificate
            if self.params.early_stop_cert and iter_idx > 0:
                # Construct the certificate matrix by rescaling the multipliers
                H = build_adjoint(multipliers * barrier, self.A_list, self.C)
                complementarity = compute_complementarity(H, Y_0)
                cert_complementarity = (
                    complementarity < self.params.tol_cert_complementarity
                )
                cert_psd = check_certificate_psd(H, tol=self.params.tol_cert_psd)

            # Check step size
            small_step = (
                delta_X_norm < self.params.tol_step_norm
                and alpha <= self.params.alpha_min
            )

            # Check stopping conditions
            if (cert_complementarity and cert_psd) or (small_step):
                break

        return X, multipliers

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
        # Construct matrix-free operator
        scale = self.params.eps_cost  # Perturbation scale
        matvec_fn = MatrixFreeLagrangeOperator(
            X=X,
            A_list=self.A_list,
            C=self.C,
            scale=scale,
            device=self.device,
        ).matvec

        # Construct RHS (matches C++ build_ac_system)
        # d_i = tr(A_i X) + (tr(A_i X) - val_i)
        # val_i = b_i (+ eps_mult * eps_constr * tr(A_i) if perturb_constraints)
        # val_cost = rho (+ eps_mult * eps_cost * tr(C) if perturb_cost)
        dtype = X.dtype
        device = X.device
        m = len(self.A_list) + 1
        d = torch.zeros(m, dtype=dtype, device=device)

        # Precompute trace(C)
        trace_C = torch.diagonal(self.C).sum()

        for i in range(len(self.A_list)):
            A_i = self.A_list[i]
            trace_Ai = float(A_i.diagonal().sum())
            A_i_dense = torch.from_numpy(A_i.toarray()).to(dtype=dtype, device=device)
            trace_Ai_X = (A_i_dense * X).sum()

            if self.params.perturb_constraints:
                val = self.b[i] + eps_mult * self.params.eps_constr * trace_Ai
            else:
                val = self.b[i]

            violation = trace_Ai_X - val
            d[i] = trace_Ai_X + violation

        trace_C_X = (self.C * X).sum()
        if self.params.perturb_cost:
            val_cost = self.rho + eps_mult * self.params.eps_cost * trace_C
        else:
            val_cost = self.rho
        violation_cost = trace_C_X - val_cost
        d[-1] = trace_C_X + violation_cost

        # Solve via CG
        assert self.precond is not None, ValueError("Preconditioner was not defined.")
        multipliers, num_iters = self.cg_solver.solve(
            b=d,
            matvec_fn=matvec_fn,
            precond_solve_fn=self.precond.solve,
        )

        return multipliers

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

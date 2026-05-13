"""
PyTorch implementation of analytic center certifier using MFCG_LRP solver.

This reimplements the C++ AnalyticCenter class with MFCG_LRP + Sparse LDLT
preconditioner for GPU acceleration of dense matrix operations.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import scipy.sparse as sp

from .lin_alg_torch import MatrixFreeLagrangeOperator, SparseLDLTPreconditioner
from .solvers import ConjugateGradientSolver
from .certificate import eval_certificate, check_certificate_psd, build_adjoint
from .utils import (
    line_search_factorization,
    eval_constraints,
)


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

# TODO this should be using the existing AnalyticCenterParams
@dataclass
class AnalyticCenterParams:
    """Parameters for analytic center computation."""

    # Verbosity
    verbose: bool = True

    # Convergence tolerances
    tol_step_norm: float = 1e-8
    max_iter: int = 50

    # Iterative linear solver parameters
    lin_solve_max_iter: int = 500
    lin_solve_tol: float = 1e-5

    # Preconditioner parameters
    precond_tau: float = 1e-5

    # Line search parameters
    enable_line_search: bool = True
    ln_search_red_factor: float = 0.8
    alpha_init: float = 1.0
    alpha_min: float = 1e-10

    # Early stopping for certificate
    early_stop_cert: bool = True
    tol_cert_psd: float = 1e-5
    tol_cert_complementarity: float = 1e-5

    # Perturbation parameters
    delta: float = 1e-5
    eps_cost: float = 1e-5
    eps_constr: float = 1e-5


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
        params: Optional[AnalyticCenterParams] = None,
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
        # TODO lost support for custom perturbation on solution. 
        try:
            alpha, L = line_search_factorization(
                X, self.params.delta * torch.eye(self.n, device=self.device)
            )
        except RuntimeError:
            if self.params.verbose:
                print("Warning: Initial point not PSD, using shifted version")
            X = X + self.params.delta * torch.eye(self.n, device=self.device)

        eps_mult = 1.0  # Perturbation multiplier
        multipliers = torch.ones(self.m, device=self.device, dtype=torch.float64)

        # Main centering loop
        for iter_idx in range(self.params.max_iter):
            # Step 1: Evaluate constraints
            violation = eval_constraints(X, self.A_list, self.b, self.C, self.rho)
            violation_norm = violation.norm()

            # Step 2: Get multipliers (solve KKT system via CG)
            multipliers = self._solve_for_multipliers(X)

            # Step 3: Compute Newton step direction
            S = build_adjoint(multipliers, self.A_list, self.C)
            S = (S + S.T) / 2

            # dZ = Z - Z @ S @ Z (standard centering direction)
            dZ = X - X @ S @ X
            dZ_norm = dZ.norm()

            # Step 4: Line search for PSDness
            try:
                alpha, L = line_search_factorization(
                    X,
                    dZ,
                    alpha_init=self.params.alpha_init,
                    alpha_min=self.params.alpha_min,
                )
            except RuntimeError as e:
                if self.params.verbose:
                    print(f"Line search failed at iteration {iter_idx}: {e}")
                break

            # Step 5: Update solution
            # TODO This matrix was already computed in the line search. Should reuse it instead of recomputing the sum
            X = X + alpha * dZ

            # Step 6: Print iteration info
            # TODO Logging should not use print statements because they can slow down GPU code.
            if self.params.verbose and (iter_idx + 1) % 10 == 1:
                print(
                    f"{'Iter':>5} {'Violation':>12} {'StepNorm':>12} "
                    f"{'Alpha':>12} {'EpsMult':>12}"
                )

            if self.params.verbose:
                print(
                    f"{iter_idx+1:5d} {violation_norm:12.6e} {dZ_norm:12.6e} "
                    f"{alpha:12.6e} {eps_mult:12.6e}"
                )

            # Step 7: Check convergence
            if dZ_norm < self.params.tol_step_norm:
                if self.params.verbose:
                    print(f"Converged at iteration {iter_idx + 1}")
                self._num_iters = iter_idx + 1
                break

            # Step 8: Early stopping with certificate
            if self.params.early_stop_cert and iter_idx > 0:
                # TODO why are we rebuilding the adjoint here? We already have S. Also this should be divided by the barrier parameter.
                H = build_adjoint(multipliers, self.A_list, self.C)
                H = (H + H.T) / 2
                min_eig, complementarity = eval_certificate(H, Y_0)

                if (
                    check_certificate_psd(H, tol=self.params.tol_cert_psd)
                    and complementarity.item() <= self.params.tol_cert_complementarity
                ):
                    if self.params.verbose:
                        print(
                            f"Certificate found! Stopping at iteration {iter_idx + 1}"
                        )
                    self._num_iters = iter_idx + 1
                    break
        else:
            self._num_iters = self.params.max_iter

        return X, multipliers

    def _solve_for_multipliers(self, X: torch.Tensor, Y0: torch.Tensor) -> torch.Tensor:
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

        # Construct preconditioner
        # TODO: Preconditioner should only occur once overall.
        try:
            precond = SparseLDLTPreconditioner(
                X=X,
                A_list=self.A_list,
                C=self.C,
                tau=self.params.precond_tau,
                device=self.device,
            )
            precond_solve_fn = precond.solve
        except RuntimeError as e:
            if self.params.verbose:
                print(
                    f"Preconditioner construction failed: {e}, using no preconditioner"
                )
            precond_solve_fn = None

        # Construct RHS (simplified: just violation vector)
        d = eval_constraints(X, self.A_list, self.b, self.C, self.rho)
        d = d.to(device=self.device)

        # Solve via CG
        multipliers, num_iters = self.cg_solver.solve(
            b=d,
            matvec_fn=matvec_fn,
            precond_solve_fn=precond_solve_fn,
        )

        if self.params.verbose and False:  # Disabled by default
            print(f"  CG: {num_iters} iterations")

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

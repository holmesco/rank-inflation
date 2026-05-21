"""
Conjugate Gradient solver implementation with optional preconditioner.
"""

from dataclasses import dataclass
from typing import Callable, Optional
import torch


@dataclass
class ConjugateGradientResult:
    """Result of a Conjugate Gradient solve."""

    solution: torch.Tensor
    num_iterations: int
    final_residual: torch.Tensor


class ConjugateGradientSolver:
    """
    Preconditioned Conjugate Gradient solver for B * x = b.

    Implements CG with optional preconditioner M:
    - Standard CG: M = I
    - Preconditioned CG: M^{-1} is applied at each iteration

    The algorithm avoids explicit matrix formation via matrix-free
    matvec operations.
    """

    def __init__(
        self,
        max_iter: int = 500,
        tol: float = 1e-5,
        verbose: bool = False,
    ):
        """
        Initialize CG solver.

        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance (residual norm)
            verbose: Print iteration statistics
        """
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def solve(
        self,
        b: torch.Tensor,
        matvec_fn: Callable[[torch.Tensor], torch.Tensor],
        precond_solve_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        x_init: Optional[torch.Tensor] = None,
    ) -> ConjugateGradientResult:
        """
        Solve the linear system B * x = b using CG.

        Args:
            b: Right-hand side vector (n,)
            matvec_fn: Function that computes y = B @ x (matrix-free matvec)
            precond_solve_fn: Optional function that computes y = M^{-1} @ x
            x_init: Initial guess (default: zero vector)

        Returns:
            ConjugateGradientResult with solution vector, iteration count, and final residual norm
        """

        # Initialize solution vector
        if x_init is None:
            x = torch.zeros_like(b)
        else:
            x = x_init.clone()

        # Compute initial residual: r = B @ x - b
        r = matvec_fn(x) - b
        # Default behaviour for preconditioner is just to clone
        if precond_solve_fn is None:
            precond_solve_fn = lambda v: v.clone()  # Identity preconditioner
        # Apply preconditioner to residual: y = M^{-1} @ r
        y = precond_solve_fn(r)
        # Initialize search direction: p = y
        p = -y.clone()
        # Store initial residual norm for relative tolerance check
        b_norm = b.norm()
        r_norm = r.norm()

        # Main CG loop (Convergence: ||B@x-b|| < tol * ||b||)
        k = 1
        while k < self.max_iter and r_norm > self.tol * b_norm:
            # Compute A @ p
            Ap = matvec_fn(p)

            # Compute step size: alpha = (r^T @ y) / (p^T @ A @ p)
            rTy = torch.dot(r, y)
            pTAp = torch.dot(p, Ap)
            alpha = rTy / pTAp

            # Update solution: x = x + alpha * p
            x = x + alpha * p

            # Update residual: r = r + alpha * A @ p
            r_new = r + alpha * Ap

            # Apply preconditioner to new residual: y_new = M^{-1} @ r_new
            y_new = precond_solve_fn(r_new)

            # Compute residual norm for convergence check
            r_norm = r_new.norm()

            # Compute beta for next search direction: beta = (r_new^T @ y_new) / (r^T @ y)
            rTy_new = torch.dot(r_new, y_new)
            beta = rTy_new / rTy

            # Update search direction: p = -y_new + beta * p
            p = -y_new + beta * p

            # Update for next iteration
            r = r_new
            y = y_new
            k = k+1
            # Print iteration info
            if self.verbose:
                print(
                    f"CG iter {k+1:4d}: residual norm = {r_norm.item():.6e}, "
                    f"alpha = {alpha.item():.6e}, beta = {beta.item():.6e}"
                )

        if self.verbose:
            print(
                f"CG: Stopped at iteration {k+1}, "
                f"residual norm = {r_norm.item():.6e}"
            )
        return ConjugateGradientResult(
            solution=x,
            num_iterations=k + 1,
            final_residual=r_norm,
        )

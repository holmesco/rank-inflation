import torch
from ranktools_pytorch.solvers import ConjugateGradientSolver
import pytest


def generate_psd_matrix(n, seed=42):
    torch.manual_seed(seed)
    A = torch.randn(n, n)
    return A @ A.t() + torch.eye(n)*1e-3  # ensure positive definiteness

def generate_well_conditioned_psd_matrix(n, seed=42, max_cond=2.0):
    torch.manual_seed(seed)

    # Random orthogonal basis Q
    M = torch.randn(n, n, dtype=torch.float64)
    Q, _ = torch.linalg.qr(M)

    # Positive eigenvalues in [1, max_cond] => cond_2(B) <= max_cond
    eigvals = torch.linspace(1.0, float(max_cond), n, dtype=torch.float64)

    # SPD matrix with controlled spectrum
    B = Q @ torch.diag(eigvals) @ Q.T
    B = 0.5 * (B + B.T)  # numerical symmetry cleanup

    # Keep output dtype consistent with existing tests
    return B.to(dtype=torch.float32)

def test_cg_solver_matches_inverse():
    n = 100
    B = generate_well_conditioned_psd_matrix(n)
    b = torch.randn(n)

    # Matrix-vector multiplication function
    def matvec(x):
        return B @ x

    cg = ConjugateGradientSolver(max_iter=1000, tol=1e-10, verbose=True)
    result = cg.solve(b, matvec)
    x_cg = result.solution
    num_iter = result.num_iterations

    # Residual
    r = b - B @ x_cg
    print(
        f"CG converged in {num_iter} iterations with residual norm {r.norm().item():.2e}"
    )

    # Direct solution
    x_true = torch.linalg.solve(B, b)

    # Check that solutions are close
    assert torch.allclose(
        x_cg, x_true, atol=1e-6
    ), f"CG solution does not match direct inverse. Max diff: {(x_cg - x_true).abs().max()}"

    
def test_cg_solver_with_jacobi_preconditioner():
    n = 100
    B = generate_psd_matrix(n, seed=123)
    b = torch.randn(n)

    # Make B diagonally dominante
    B = B + torch.eye(n) * B.trace()

    def matvec(x):
        return B @ x

    # Jacobi preconditioner: M = diag(B)
    diag_B = B.diag()

    def jacobi_precond(x):
        return x / diag_B

    cg = ConjugateGradientSolver(max_iter=1000, tol=1e-8)
    result = cg.solve(b, matvec, precond_solve_fn=jacobi_precond)
    x_cg = result.solution
    num_iter = result.num_iterations

    # Direct solution
    x_true = torch.linalg.solve(B, b)

    # Check that solutions are close
    assert torch.allclose(
        x_cg, x_true, atol=1e-6
    ), f"Preconditioned CG solution does not match direct inverse. Max diff: {(x_cg - x_true).abs().max()}"

    # Check that convergence is achieved in fewer iterations than unpreconditioned (optional, for demonstration)
    cg_no_precond = ConjugateGradientSolver(max_iter=1000, tol=1e-8)
    result_no_precond = cg_no_precond.solve(b, matvec)
    num_iter_no_precond = result_no_precond.num_iterations
    assert (
        num_iter <= num_iter_no_precond
    ), "Preconditioner did not improve or match convergence speed."

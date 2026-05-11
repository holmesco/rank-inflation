"""
Basic tests for analytic center PyTorch implementation.
"""

import pytest
import torch
import numpy as np
import scipy.sparse as sp
from ranktools_pytorch import AnalyticCenterPyTorch, AnalyticCenterParams


def create_test_problem(n: int = 10, m: int = 5):
    """
    Create a simple test SDP problem.

    Returns:
        (C, rho, A_list, b, Y_0)
    """
    # Random symmetric cost matrix
    C_np = np.random.randn(n, n)
    C_np = (C_np + C_np.T) / 2
    C = torch.from_numpy(C_np).double()

    # Cost constraint
    rho = 1.0

    # Random constraint matrices (stored as sparse upper-triangular)
    A_list = []
    for _ in range(m):
        A_dense = np.random.randn(n, n)
        A_sym = (A_dense + A_dense.T) / 2
        # Store only upper triangle
        A_sparse = sp.triu(sp.csr_matrix(A_sym))
        A_list.append(A_sparse)

    # Constraint RHS
    b = torch.ones(m)

    # Initial low-rank solution
    Y_0 = np.random.randn(n, 2)
    Y_0 = torch.from_numpy(Y_0).double()

    return C, rho, A_list, b, Y_0


class TestAnalyticCenterPyTorch:
    """Test suite for AnalyticCenterPyTorch."""

    def test_initialization(self):
        """Test that certifier initializes correctly."""
        C, rho, A_list, b, Y_0 = create_test_problem(10, 5)

        certifier = AnalyticCenterPyTorch(C, rho, A_list, b)

        assert certifier.n == 10
        assert certifier.m == 6  # 5 constraints + 1 cost

    def test_small_problem(self):
        """Test certification on a small problem."""
        C, rho, A_list, b, Y_0 = create_test_problem(n=5, m=2)

        params = AnalyticCenterParams(
            verbose=True,
            max_iter=10,
            lin_solve_max_iter=100,
            lin_solve_tol=1e-4,
        )

        certifier = AnalyticCenterPyTorch(C, rho, A_list, b, params=params)
        result = certifier.certify(Y_0)

        # Check that result is valid
        assert result.X.shape == (5, 5)
        assert result.H.shape == (5, 5)
        assert result.multipliers.shape == (3,)
        assert result.solver_time > 0
        assert result.num_iters > 0

    def test_gpu_vs_cpu(self):
        """Test that CPU and GPU give similar results."""
        C, rho, A_list, b, Y_0 = create_test_problem(n=5, m=2)

        params = AnalyticCenterParams(
            verbose=False,
            max_iter=5,
            lin_solve_max_iter=50,
        )

        # CPU
        certifier_cpu = AnalyticCenterPyTorch(
            C, rho, A_list, b, params=params, device=torch.device("cpu")
        )
        result_cpu = certifier_cpu.certify(Y_0)

        # GPU (if available)
        if torch.cuda.is_available():
            certifier_gpu = AnalyticCenterPyTorch(
                C, rho, A_list, b, params=params, device=torch.device("cuda")
            )
            result_gpu = certifier_gpu.certify(Y_0)

            # Compare results (should be very close)
            assert torch.allclose(result_cpu.X, result_gpu.X.cpu(), atol=1e-6)
            assert torch.allclose(result_cpu.H, result_gpu.H.cpu(), atol=1e-6)
        else:
            pytest.skip("CUDA not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

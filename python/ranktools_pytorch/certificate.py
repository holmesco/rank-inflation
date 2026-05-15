"""
Certificate evaluation utilities.
"""

from typing import Tuple
import torch

from ranktools_pytorch.utils import symmetrize_dense

def eval_certificate(
    H: torch.Tensor,
    Y_0: torch.Tensor,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate the certificate matrix for optimality.

    Computes:
    1. Minimum eigenvalue of H (should be >= -tol_psd for PSDness)
    2. Complementarity condition (first-order optimality): tr(Y_0^T @ H @ Y_0)

    Args:
        H: Certificate matrix (n × n), typically rescaled adjoint of multipliers
        Y_0: Initial solution matrix (n × r), used for complementarity check
        scale: Scaling factor for certificate

    Returns:
        (min_eig, complementarity)
        - min_eig: Minimum eigenvalue of H
        - complementarity: Norm of Y_0^T @ H @ Y_0 (should be near zero)
    """
    device = H.device
    dtype = H.dtype

    # Compute eigenvalues of H (most expensive step: O(n^3))
    eigvals = torch.linalg.eigvalsh(H)
    min_eig = eigvals[0]

    # Compute complementarity: ||Y_0^T @ H @ Y_0||
    # This checks the first-order optimality condition
    complementarity = compute_complementarity(H, Y_0)

    return min_eig, complementarity


def compute_complementarity(H: torch.Tensor, Y_0: torch.Tensor) -> torch.Tensor:
    """
    Compute the complementarity condition: ||Y_0^T @ H @ Y_0||.

    Args:
        H: Certificate matrix (n × n)
        Y_0: Initial solution matrix (n × r)

    Returns:
        Complementarity norm (should be near zero)
    """
    HY = H @ Y_0
    YtHY = Y_0.T @ HY
    return YtHY.trace()


def check_certificate_psd(
    H: torch.Tensor,
    tol: float = 1e-5,
) -> bool:
    """
    Quick PSD check using Cholesky factorization.

    Args:
        H: Certificate matrix (n × n)
        tol: Tolerance for perturbation to ensure strict PSDness

    Returns:
        True if H + tol*I is positive definite
    """
    try:
        H_pert = H + tol * torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
        torch.linalg.cholesky(H_pert)
        return True
    except RuntimeError:
        return False


def build_adjoint(
    multipliers: torch.Tensor,
    A_list: list,
    C: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Build the adjoint (dual matrix) from multipliers.

    Computes: S = scale * (sum_i multiplier_i * A_i + multiplier_m * C)

    Args:
        multipliers: Lagrange multipliers (m,)
        A_list: List of m-1 sparse constraint matrices, assumed upper triangular
        C: Cost matrix (n × n), assumed upper triangular
        scale: Scaling factor

    Returns:
        Dual matrix S (n × n), dense and symmetrized
    """
    device = C.device
    dtype = C.dtype

    # Start with cost term
    S = multipliers[-1] * C

    # Add constraint terms (convert sparse to dense on device)
    for i in range(len(A_list)):
        A_i_dense = torch.from_numpy(A_list[i].toarray()).to(dtype=dtype, device=device)
        S = S + multipliers[i] * A_i_dense

    # Symmetrize the upper triangular matrix
    S = symmetrize_dense(torch.triu(S)) * scale
    
    return S

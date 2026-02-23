"""
Smoke-test / usage example for the ranktools Python bindings.

Run after building:
    cd /workspace
    cmake -S . -B build -DBUILD_PYTHON_BINDINGS=ON
    cmake --build build -j
    cd bindings/python
    python test_analytic_center.py
"""

import numpy as np
import scipy.sparse as sp

# The compiled module is placed next to this file by CMake
import ranktools


def make_simple_sdp(n: int):
    """
    Build a trivial SDP:
      min  trace(I * X)
      s.t. trace(X) = 1,  X >= 0
    The analytic center of { X psd | trace(X)=1 } is X* = I/n.
    """
    C = np.eye(n)
    rho = 1.0  # optimal value trace(I * I/n) = 1

    # Single constraint: trace(X) = 1  →  A = I (upper-tri stored)
    A_dense = np.eye(n)
    A_sparse = sp.csc_matrix(np.triu(A_dense))
    b = [1.0]

    return C, rho, [A_sparse], b


def test_params():
    """Verify default parameter construction and mutation."""
    p = ranktools.AnalyticCenterParams()
    assert p.verbose is True
    p.verbose = False
    assert p.verbose is False
    print("[PASS] AnalyticCenterParams")


def test_construction():
    """Construct an AnalyticCenter and verify dimensions."""
    n = 4
    C, rho, A, b = make_simple_sdp(n)
    ac = ranktools.AnalyticCenter(C, rho, A, b)
    assert ac.dim == n
    assert ac.m == len(b) + 1  # constraints + cost
    print(f"[PASS] AnalyticCenter construction  dim={ac.dim}  m={ac.m}")


def test_eval_constraints():
    n = 4
    C, rho, A, b = make_simple_sdp(n)
    ac = ranktools.AnalyticCenter(C, rho, A, b)

    X = np.eye(n) / n
    v = ac.eval_constraints(X)
    assert v.shape == (ac.m,)
    # trace(I/n) - 1 == 0, cost trace(I * I/n) - 1 == 0
    np.testing.assert_allclose(v, 0.0, atol=1e-12)
    print("[PASS] eval_constraints")


def test_get_analytic_center():
    n = 4
    C, rho, A, b = make_simple_sdp(n)
    params = ranktools.AnalyticCenterParams()
    params.verbose = False
    params.max_iter_ac = 100
    ac = ranktools.AnalyticCenter(C, rho, A, b, params)

    # Start from identity (feasible)
    Y0 = np.eye(n) / np.sqrt(n)
    X, mult = ac.get_analytic_center(Y0, delta_obj=1e-7)
    assert X.shape == (n, n)
    # X should be close to I/n
    np.testing.assert_allclose(X, np.eye(n) / n, atol=1e-4)
    print("[PASS] get_analytic_center")


def test_build_and_check_certificate():
    n = 4
    C, rho, A, b = make_simple_sdp(n)
    params = ranktools.AnalyticCenterParams()
    params.verbose = False
    ac = ranktools.AnalyticCenter(C, rho, A, b, params)

    Y0 = np.eye(n) / np.sqrt(n)
    X, mult = ac.get_analytic_center(Y0, delta_obj=1e-7)

    H = ac.build_certificate_from_dual(mult)
    assert H.shape == (n, n)

    min_eig, first_order = ac.check_certificate(H, Y0)
    print(f"[PASS] certificate  min_eig={min_eig:.6e}  1st_order={first_order:.6e}")


def test_certify():
    n = 4
    C, rho, A, b = make_simple_sdp(n)
    params = ranktools.AnalyticCenterParams()
    params.verbose = False
    ac = ranktools.AnalyticCenter(C, rho, A, b, params)

    Y0 = np.eye(n) / np.sqrt(n)
    result = ac.certify(Y0, delta=1e-7)

    assert isinstance(result, ranktools.AnalyticCenterResult)
    assert result.X.shape == (n, n)
    print(f"[PASS] certify  certified={result.certified}  min_eig={result.min_eig:.6e}")


def test_solve_sdp_mosek():
    """Verify solve_sdp_mosek returns a valid primal/dual solution."""
    n = 4
    C, rho, A, b = make_simple_sdp(n)

    result = ranktools.solve_sdp_mosek(C, A, b, verbose=False)

    assert isinstance(result, ranktools.SDPResult)
    assert result.X.shape == (n, n)
    assert result.y.shape == (len(b),)
    assert result.S.shape == (n, n)

    # Primal feasibility: trace(A_0 X) == b_0 == 1
    np.testing.assert_allclose(np.trace(result.X), b[0], atol=1e-4)
    # Objective value should equal trace(C @ X) = trace(X) ≈ 1
    np.testing.assert_allclose(result.obj_value, b[0], atol=1e-4)
    # Dual matrix S should be PSD
    eigs = np.linalg.eigvalsh(result.S)
    assert eigs.min() >= -1e-6, f"Dual S not PSD, min eig={eigs.min():.3e}"
    print(f"[PASS] solve_sdp_mosek  obj={result.obj_value:.6f}")


def test_rank_reduction():
    """Verify that rank_reduction reduces rank while preserving constraints."""
    n = 4
    C, rho, A, b = make_simple_sdp(n)

    # Start from a rank-n solution: V = I/sqrt(n)
    V0 = np.eye(n) / np.sqrt(n)

    params = ranktools.RankReductionParams()
    params.verbose = False
    params.targ_rank = 1

    V_red = ranktools.rank_reduction(A, V0, params)
    assert V_red.ndim == 2
    assert V_red.shape[0] == n
    assert V_red.shape[1] <= params.targ_rank

    # Constraint trace(X) = 1 must still hold
    X_red = V_red @ V_red.T
    np.testing.assert_allclose(np.trace(X_red), 1.0, atol=1e-6)
    print(f"[PASS] rank_reduction  output_rank={V_red.shape[1]}")


if __name__ == "__main__":
    test_params()
    test_construction()
    test_eval_constraints()
    test_get_analytic_center()
    test_build_and_check_certificate()
    test_certify()
    test_solve_sdp_mosek()
    test_rank_reduction()
    print("\nAll tests passed!")

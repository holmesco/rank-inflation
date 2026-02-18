"""
Smoke-test / usage example for the sdptools Python bindings.

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
import sdptools


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
    p = sdptools.AnalyticCenterParams()
    assert p.verbose is True
    p.verbose = False
    assert p.verbose is False
    print("[PASS] AnalyticCenterParams")


def test_construction():
    """Construct an AnalyticCenter and verify dimensions."""
    n = 4
    C, rho, A, b = make_simple_sdp(n)
    ac = sdptools.AnalyticCenter(C, rho, A, b)
    assert ac.dim == n
    assert ac.m == len(b) + 1  # constraints + cost
    print(f"[PASS] AnalyticCenter construction  dim={ac.dim}  m={ac.m}")


def test_eval_constraints():
    n = 4
    C, rho, A, b = make_simple_sdp(n)
    ac = sdptools.AnalyticCenter(C, rho, A, b)

    X = np.eye(n) / n
    v = ac.eval_constraints(X)
    assert v.shape == (ac.m,)
    # trace(I/n) - 1 == 0, cost trace(I * I/n) - 1 == 0
    np.testing.assert_allclose(v, 0.0, atol=1e-12)
    print("[PASS] eval_constraints")


def test_get_analytic_center():
    n = 4
    C, rho, A, b = make_simple_sdp(n)
    params = sdptools.AnalyticCenterParams()
    params.verbose = False
    params.max_iter_ac = 100
    ac = sdptools.AnalyticCenter(C, rho, A, b, params)

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
    params = sdptools.AnalyticCenterParams()
    params.verbose = False
    ac = sdptools.AnalyticCenter(C, rho, A, b, params)

    Y0 = np.eye(n) / np.sqrt(n)
    X, mult = ac.get_analytic_center(Y0, delta_obj=1e-7)

    H = ac.build_certificate_from_dual(mult)
    assert H.shape == (n, n)

    min_eig, first_order = ac.check_certificate(H, Y0)
    print(f"[PASS] certificate  min_eig={min_eig:.6e}  1st_order={first_order:.6e}")


def test_certify():
    n = 4
    C, rho, A, b = make_simple_sdp(n)
    params = sdptools.AnalyticCenterParams()
    params.verbose = False
    ac = sdptools.AnalyticCenter(C, rho, A, b, params)

    Y0 = np.eye(n) / np.sqrt(n)
    result = ac.certify(Y0, delta=1e-7)

    assert isinstance(result, sdptools.AnalyticCenterResult)
    assert result.X.shape == (n, n)
    print(f"[PASS] certify  certified={result.certified}  min_eig={result.min_eig:.6e}")


if __name__ == "__main__":
    test_params()
    test_construction()
    test_eval_constraints()
    test_get_analytic_center()
    test_build_and_check_certificate()
    test_certify()
    print("\nAll tests passed!")

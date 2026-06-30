"""
Python bindings for the RankTools library (AnalyticCenter, rank_reduction, solve_sdp_mosek)
"""
from __future__ import annotations
import numpy
import scipy.sparse
import typing
__all__: list[str] = ['AnalyticCenter', 'AnalyticCenterParams', 'AnalyticCenterResult', 'CG', 'DenseLDLT', 'DenseLU', 'DenseQR', 'DirectInverse', 'LDLT', 'LinearSolverType', 'LowRankPrecondMethod', 'LowRankPrecondParams', 'MFCG_DP', 'MFCG_LRP', 'MaxCliqueCertifier', 'RankReductionParams', 'SDPResult', 'SparseLDLT', 'SparseLDLT_ZL', 'SparseQR', 'rank_reduction', 'solve_sdp_mosek']
class AnalyticCenter:
    params: AnalyticCenterParams
    def __init__(self, C: numpy.ndarray, rho: float, A: list[scipy.sparse.csc_matrix], b: list[float], params: ... = ...) -> None:
        """
        Construct an AnalyticCenter problem.
        
        Parameters
        ----------
        C : numpy.ndarray (n, n)
            Cost matrix.
        rho : float
            Optimal cost value (scalar offset).
        A : list of scipy.sparse.csc_matrix or numpy.ndarray
            Constraint matrices (each n×n, upper-triangular storage).
        b : list of float
            Right-hand side values for trace(A_i X) = b_i.
        params : AnalyticCenterParams, optional
            Algorithm parameters.
        """
    def build_adjoint(self, coeffs: numpy.ndarray) -> numpy.ndarray:
        """
        Build the adjoint matrix sum_i coeffs[i] * A_i + coeffs[-1] * C.
        """
    def build_certificate_from_dual(self, multipliers: numpy.ndarray) -> numpy.ndarray:
        """
        Backward-compatible alias for build_adjoint(multipliers).
        """
    def certify(self, Y_0: numpy.ndarray, perturb: typing.Any = None) -> AnalyticCenterResult:
        """
        Run analytic centering to certify the local solution Y_0.
        
        If perturb is provided, it is used as the initial perturbation matrix.
        Otherwise, params.delta * Identity is used as the fallback.
        
        Parameters
        ----------
        Y_0 : numpy.ndarray (n, r)
            Initial low-rank factor.
        perturb : numpy.ndarray (n, n), optional
            Initial perturbation matrix. If not provided, uses params.delta * Identity.
        
        Returns
        -------
        AnalyticCenterResult
        """
    def check_certificate(self, H: numpy.ndarray, Y: numpy.ndarray) -> tuple[float, float]:
        """
        Check global optimality of a solution.
        
        Returns whether the certificate matrix is PSD and the complementarity
        of the provided solution.
        
        Returns
        -------
        tuple(min_eig, complementarity)
        """
    def eval_certificate(self, H: numpy.ndarray, Y: numpy.ndarray) -> tuple[float, float]:
        """
        Evaluate the optimality certificate.
        
        Returns the minimum eigenvalue of the certificate matrix and the
        evaluation of the certificate matrix at the solution (first order
        condition).
        
        Parameters
        ----------
        H : numpy.ndarray (n, n)
            Certificate matrix.
        Y : numpy.ndarray (n, r)
            Low-rank factor of the solution.
        
        Returns
        -------
        tuple(min_eig, first_order_cond)
        """
    def eval_constraints(self, X: numpy.ndarray) -> numpy.ndarray:
        """
        Evaluate constraint violations at X.
        """
    def export_problem(self, file_path: str, problem_name: str, solution: numpy.ndarray) -> None:
        """
        Export the current problem to a text file.
        
        The file format matches `load_problem_from_file` in the C++ test helpers.
        
        Parameters
        ----------
        file_path : str
            Destination file path.
        problem_name : str
            Value written to the `name` field.
        solution : numpy.ndarray
            Solution matrix written in the `soln` block.
        """
    def get_analytic_center(self, Y_0: numpy.ndarray, perturb: typing.Any = None) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Compute the analytic center starting from Y_0.
        
        If perturb is provided, it is used as the initial perturbation matrix.
        Otherwise, params.delta * Identity is used as the fallback.
        
        Parameters
        ----------
        Y_0 : numpy.ndarray (n, r)
            Initial point (low-rank factor; X_0 = Y_0 @ Y_0.T).
        perturb : numpy.ndarray (n, n), optional
            Initial perturbation matrix. If not provided, uses params.delta * Identity.
        
        Returns
        -------
        tuple(X, multipliers)
            X : numpy.ndarray (n, n)  — centered primal solution.
            multipliers : numpy.ndarray (m,) — optimal dual multipliers.
        """
    @property
    def dim(self) -> int:
        ...
    @property
    def m(self) -> int:
        ...
class AnalyticCenterParams:
    adaptive_perturb: bool
    alpha_init: float
    alpha_min: float
    check_indep_constr: bool
    delta: float
    early_stop_angle: bool
    early_stop_cert: bool
    enable_line_search: bool
    eps_constr: float
    eps_cost: float
    eps_dec: float
    eps_dec_step_thresh: float
    eps_inc: float
    eps_inc_step_thresh: float
    eps_mult_min: float
    lin_solve_max_iter: int
    lin_solve_tol: float
    lin_solver: LinearSolverType
    ln_search_red_factor: float
    lrp_params: LowRankPrecondParams
    max_angle: float
    max_iter: int
    perturb_constraints: bool
    perturb_cost: bool
    rescale_lin_sys: bool
    rescaling_factor: float
    reuse_multipliers: bool
    tau_lrp: float
    tol_cert_centrality: float
    tol_cert_complementarity: float
    tol_cert_primal_feas: float
    tol_cert_psd: float
    tol_indep_constr: float
    tol_rank_sol: float
    tol_step_norm: float
    use_cert_centrality_metric: bool
    verbose: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
class AnalyticCenterResult:
    def __repr__(self) -> str:
        ...
    @property
    def H(self) -> numpy.ndarray:
        ...
    @property
    def X(self) -> numpy.ndarray:
        ...
    @property
    def certified(self) -> bool:
        ...
    @property
    def complementarity(self) -> float:
        ...
    @property
    def min_eig(self) -> float:
        ...
    @property
    def multipliers(self) -> numpy.ndarray:
        ...
    @property
    def solver_time(self) -> float:
        ...
    @property
    def violation(self) -> numpy.ndarray:
        ...
class LinearSolverType:
    """
    Linear solver type for the analytic center step.
    
    Members:
    
      LDLT : Cholesky-based LDLT solver
    
      CG : Conjugate gradient solver
    
      MFCG_DP : Matrix-free conjugate gradient solver with diagonal preconditioner
    
      MFCG_LRP : Matrix-free conjugate gradient solver with low-rank preconditioner
    """
    CG: typing.ClassVar[LinearSolverType]  # value = <LinearSolverType.CG: 1>
    LDLT: typing.ClassVar[LinearSolverType]  # value = <LinearSolverType.LDLT: 0>
    MFCG_DP: typing.ClassVar[LinearSolverType]  # value = <LinearSolverType.MFCG_DP: 2>
    MFCG_LRP: typing.ClassVar[LinearSolverType]  # value = <LinearSolverType.MFCG_LRP: 3>
    __members__: typing.ClassVar[dict[str, LinearSolverType]]  # value = {'LDLT': <LinearSolverType.LDLT: 0>, 'CG': <LinearSolverType.CG: 1>, 'MFCG_DP': <LinearSolverType.MFCG_DP: 2>, 'MFCG_LRP': <LinearSolverType.MFCG_LRP: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class LowRankPrecondMethod:
    """
    Method for building the low-rank preconditioner.
    
    Members:
    
      DenseLDLT : Dense LDLT factorization approach
    
      SparseLDLT : Sparse LDLT factorization approach using alternate top-right formulation
    
      SparseLDLT_ZL : Sparse LDLT factorization approach using the Zhang-Lavaei (2017) top-right formulation
    
      DenseQR : Dense QR factorization approach
    
      SparseQR : Sparse QR factorization approach
    
      DenseLU : Dense LU factorization approach
    
      DirectInverse : Direct inverse approach
    """
    DenseLDLT: typing.ClassVar[LowRankPrecondMethod]  # value = <LowRankPrecondMethod.DenseLDLT: 0>
    DenseLU: typing.ClassVar[LowRankPrecondMethod]  # value = <LowRankPrecondMethod.DenseLU: 5>
    DenseQR: typing.ClassVar[LowRankPrecondMethod]  # value = <LowRankPrecondMethod.DenseQR: 3>
    DirectInverse: typing.ClassVar[LowRankPrecondMethod]  # value = <LowRankPrecondMethod.DirectInverse: 6>
    SparseLDLT: typing.ClassVar[LowRankPrecondMethod]  # value = <LowRankPrecondMethod.SparseLDLT: 1>
    SparseLDLT_ZL: typing.ClassVar[LowRankPrecondMethod]  # value = <LowRankPrecondMethod.SparseLDLT_ZL: 2>
    SparseQR: typing.ClassVar[LowRankPrecondMethod]  # value = <LowRankPrecondMethod.SparseQR: 4>
    __members__: typing.ClassVar[dict[str, LowRankPrecondMethod]]  # value = {'DenseLDLT': <LowRankPrecondMethod.DenseLDLT: 0>, 'SparseLDLT': <LowRankPrecondMethod.SparseLDLT: 1>, 'SparseLDLT_ZL': <LowRankPrecondMethod.SparseLDLT_ZL: 2>, 'DenseQR': <LowRankPrecondMethod.DenseQR: 3>, 'SparseQR': <LowRankPrecondMethod.SparseQR: 4>, 'DenseLU': <LowRankPrecondMethod.DenseLU: 5>, 'DirectInverse': <LowRankPrecondMethod.DirectInverse: 6>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class LowRankPrecondParams:
    ldlt_zero_thresh: float
    method: LowRankPrecondMethod
    tau: float
    use_approx: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
class MaxCliqueCertifier:
    params: AnalyticCenterParams
    def __init__(self, M: numpy.ndarray, rho: float, params: ... = ...) -> None:
        """
        Construct a MaxCliqueCertifier for the maximum-clique / Lovasz-theta SDP.
        
        Behaves exactly like AnalyticCenter, except that the constraint matrices A and
        right-hand side b are not supplied by the caller. They are constructed from the
        cost matrix M: there is one constraint per non-edge enforcing that the
        corresponding off-diagonal entry of the solution is zero, plus a trace
        constraint fixing the trace to one. The non-edges are taken to be the
        off-diagonal (i, j) indices of M whose entries are exactly equal to zero.
        
        Parameters
        ----------
        M : numpy.ndarray (n, n)
            Cost matrix. Its off-diagonal zero entries define the non-edge constraints.
        rho : float
            Optimal cost value (scalar offset).
        params : AnalyticCenterParams, optional
            Algorithm parameters.
        """
    def build_adjoint(self, coeffs: numpy.ndarray) -> numpy.ndarray:
        """
        Build the adjoint matrix sum_i coeffs[i] * A_i + coeffs[-1] * C.
        """
    def build_certificate_from_dual(self, multipliers: numpy.ndarray) -> numpy.ndarray:
        """
        Backward-compatible alias for build_adjoint(multipliers).
        """
    def certify(self, Y_0: numpy.ndarray, perturb: typing.Any = None) -> AnalyticCenterResult:
        """
        Run analytic centering to certify the local solution Y_0.
        
        If perturb is provided, it is used as the initial perturbation matrix.
        Otherwise, params.delta * Identity is used as the fallback.
        
        Parameters
        ----------
        Y_0 : numpy.ndarray (n, r)
            Initial low-rank factor.
        perturb : numpy.ndarray (n, n), optional
            Initial perturbation matrix. If not provided, uses params.delta * Identity.
        
        Returns
        -------
        AnalyticCenterResult
        """
    def check_certificate(self, H: numpy.ndarray, Y: numpy.ndarray) -> tuple[float, float]:
        """
        Check global optimality of a solution.
        
        Returns whether the certificate matrix is PSD and the complementarity
        of the provided solution.
        
        Returns
        -------
        tuple(min_eig, complementarity)
        """
    def eval_certificate(self, H: numpy.ndarray, Y: numpy.ndarray) -> tuple[float, float]:
        """
        Evaluate the optimality certificate.
        
        Returns the minimum eigenvalue of the certificate matrix and the
        evaluation of the certificate matrix at the solution (first order
        condition).
        
        Parameters
        ----------
        H : numpy.ndarray (n, n)
            Certificate matrix.
        Y : numpy.ndarray (n, r)
            Low-rank factor of the solution.
        
        Returns
        -------
        tuple(min_eig, first_order_cond)
        """
    def eval_constraints(self, X: numpy.ndarray) -> numpy.ndarray:
        """
        Evaluate constraint violations at X.
        """
    def export_problem(self, file_path: str, problem_name: str, solution: numpy.ndarray) -> None:
        """
        Export the current problem to a text file.
        
        The file format matches `load_problem_from_file` in the C++ test helpers.
        
        Parameters
        ----------
        file_path : str
            Destination file path.
        problem_name : str
            Value written to the `name` field.
        solution : numpy.ndarray
            Solution matrix written in the `soln` block.
        """
    def get_analytic_center(self, Y_0: numpy.ndarray, perturb: typing.Any = None) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Compute the analytic center starting from Y_0.
        
        If perturb is provided, it is used as the initial perturbation matrix.
        Otherwise, params.delta * Identity is used as the fallback.
        
        Parameters
        ----------
        Y_0 : numpy.ndarray (n, r)
            Initial point (low-rank factor; X_0 = Y_0 @ Y_0.T).
        perturb : numpy.ndarray (n, n), optional
            Initial perturbation matrix. If not provided, uses params.delta * Identity.
        
        Returns
        -------
        tuple(X, multipliers)
            X : numpy.ndarray (n, n)  — centered primal solution.
            multipliers : numpy.ndarray (m,) — optimal dual multipliers.
        """
    @property
    def dim(self) -> int:
        ...
    @property
    def m(self) -> int:
        ...
class RankReductionParams:
    eig_tol: float
    max_iter: int
    null_tol: float
    targ_rank: int
    verbose: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
class SDPResult:
    def __repr__(self) -> str:
        ...
    @property
    def S(self) -> numpy.ndarray:
        ...
    @property
    def X(self) -> numpy.ndarray:
        ...
    @property
    def obj_value(self) -> float:
        ...
    @property
    def y(self) -> numpy.ndarray:
        ...
def rank_reduction(As: list[scipy.sparse.csc_matrix], V_init: numpy.ndarray, params: ... = ...) -> numpy.ndarray:
    """
    Reduce the rank of an SDP solution.
    
    Implements the rank reduction algorithm from:
      Lemon, So & Ye, "Low-rank semidefinite programming: Theory and
      applications", Foundations and Trends in Optimization, 2016.
    
    Parameters
    ----------
    As : list of scipy.sparse matrices
        Constraint matrices (each n×n, upper-triangular storage).
    V_init : numpy.ndarray (n, r)
        Initial low-rank factor; the SDP solution is X = V_init @ V_init.T.
    params : RankReductionParams, optional
        Algorithm parameters.
    
    Returns
    -------
    numpy.ndarray (n, r')
        Reduced low-rank factor V such that V @ V.T satisfies all constraints
        with rank r' <= r.
    """
def solve_sdp_mosek(C: numpy.ndarray, As: list[scipy.sparse.csc_matrix], b: list[float], verbose: bool = True) -> SDPResult:
    """
    Solve a semidefinite program using MOSEK.
    
    Solves the primal SDP:
        min   trace(C @ X)
        s.t.  trace(A_i @ X) = b_i  for i = 0 ... m-1
              X >= 0  (positive semidefinite)
    
    Parameters
    ----------
    C : numpy.ndarray (n, n)
        Cost matrix.
    As : list of scipy.sparse matrices
        Constraint matrices (each n x n).
    b : list of float
        Right-hand side values.
    verbose : bool, optional
        Enable MOSEK solver output (default True).
    
    Returns
    -------
    SDPResult
        Struct with fields:
          X         : numpy.ndarray (n, n) — primal solution.
          y         : numpy.ndarray (m,)   — dual multipliers for equality constraints.
          S         : numpy.ndarray (n, n) — dual PSD matrix C - sum_i y_i A_i.
          obj_value : float                — objective value trace(C @ X).
    """
CG: LinearSolverType  # value = <LinearSolverType.CG: 1>
DenseLDLT: LowRankPrecondMethod  # value = <LowRankPrecondMethod.DenseLDLT: 0>
DenseLU: LowRankPrecondMethod  # value = <LowRankPrecondMethod.DenseLU: 5>
DenseQR: LowRankPrecondMethod  # value = <LowRankPrecondMethod.DenseQR: 3>
DirectInverse: LowRankPrecondMethod  # value = <LowRankPrecondMethod.DirectInverse: 6>
LDLT: LinearSolverType  # value = <LinearSolverType.LDLT: 0>
MFCG_DP: LinearSolverType  # value = <LinearSolverType.MFCG_DP: 2>
MFCG_LRP: LinearSolverType  # value = <LinearSolverType.MFCG_LRP: 3>
SparseLDLT: LowRankPrecondMethod  # value = <LowRankPrecondMethod.SparseLDLT: 1>
SparseLDLT_ZL: LowRankPrecondMethod  # value = <LowRankPrecondMethod.SparseLDLT_ZL: 2>
SparseQR: LowRankPrecondMethod  # value = <LowRankPrecondMethod.SparseQR: 4>

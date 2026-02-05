#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

namespace SDPTools {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Triplet = Eigen::Triplet<double>;
using SpMatrix = Eigen::SparseMatrix<double>;
using ScalarFunc = std::function<double(double)>;

// Result of solving a linear system using rank-revealing QR decomposition.
// Contains both the least-squares particular solution and the nullspace basis.
struct QRResult {
  Vector solution;
  Matrix nullspace_basis;
  int rank;
  Vector R_diagonal;
  double residual_norm;
  Eigen::ColPivHouseholderQR<Matrix> qr_decomp;
};

// Get the particular solution and null space of a system of linear equations
// using rank revealing QR decomposition This formulation is designed for dens
// matrices
QRResult get_soln_qr_dense(const Matrix& A, const Vector& b,
                           const double threshold);

// Compute the rank of a dense matrix with rank-revealing QR
int get_rank(const Matrix& Y, const double threshold);

// Use LDL^T factorization to recover a low-rank factor from a psd matrix.
// Note: This is efficient when the rank is low compared to the size of the
// matrix.
Matrix recover_lowrank_factor(const Matrix& A);

// Bisection line search to find root of scalar function df
double bisection_line_search(const ScalarFunc& df, double alpha_low,
                             double alpha_high, double tol);

inline double neglogdet(const Matrix& X) { return -std::log((X).determinant()); }

enum class RetractionMethod {
  GradientDescent,
  GaussNewton,
  ExactNewton,
};

struct RankInflateParams {
  // NOTE: for tolerances on rank and nullspace: pivot added to rank if R_ii >
  // tol * R_max

  // Verbosity
  bool verbose = true;
  // Desired rank
  int max_sol_rank = 1;
  // Max number of iterations
  int max_iter = 10;
  // threshold for checking rank of the solution
  // (does not affect convergence, just for display)
  double tol_rank_sol = 1.0E-5;
  // Threshold for checking rank of the Jacobian
  double tol_rank_jac = 1.0E-7;

  // Retraction solve parameters
  // -----------------------

  // Retraction Method
  RetractionMethod retraction_method = RetractionMethod::ExactNewton;
  // Max iterations for retraction
  int max_iter_retract = 100;
  // Tollerance for QR decomposition of Jacobian matrix
  // NOTE: Controls accuracy of GN solve. If made too small then can get
  // unstable solutions.
  double tol_jac_qr = 1.0E-7;
  // tolerance for constraint norm satisfaction.
  double tol_violation = 1.0E-10;

  // Second order correction
  // -----------------------
  // Enable second-order correction term
  bool enable_sec_ord_corr = false;
  // tolerance on violation to include constraint in hessian construction
  double tol_viol_hess = 1e-18;
  // tolerance on null space for linear solve
  // If set too small then null space direction can't be found for perturbation
  double tol_null_corr = 1e-8;

  // Line search
  // ----------------
  // Line search enable
  bool enable_line_search = true;
  // Line search sufficient decrease parameter (should be between zero and one)
  double ln_search_suff_dec = 1e-4;
  // Line search reduction factor (should be between zero and one)
  double ln_search_red_factor = 0.5;
  // Line search initialization
  double alpha_init = 1.0;
  // Line search lower bound
  double alpha_min = 1e-14;

  // Geodesic step parameters
  // ----------------
  // If true enables second order geodesic step
  // NOTE: This seems to introduce numerical issues! Don't use for now.
  bool second_ord_geo = false;
  // Tangent space/geodesic step size (wrt Frobenius norm)
  double eps_geodesic = 1.0E-2;

  // Analytic Center parameters
  // -------------------------
  // tolerance for step size (terminate if below)
  double tol_step_norm_ac = 1e-8;
  // threshold for QR solve.
  double qr_thresh_ac = 1e-8;
  // reduce violation in centering step
  bool reduce_violation_ac = false;
  // max number of iterations for centering
  int max_iter_ac = 50;
  // delta parameter for centering
  double delta_ac = 1e-6;
  // line search (bisection) parameters for centering
  // NOTE: line search param will be certain to 1/2^k for k = ls_iter_ac
  double tol_bisect_ac = 1e-6;
};

class RankInflation {
 public:
  // dimension of the sdp
  // NOTE: if required we could template on size to make things faster
  int dim;
  // number of constraints + cost if required
  int m;
  // cost matrix
  const Matrix C_;
  // optimal cost value
  const double rho_;
  // constraint matrices
  const std::vector<Eigen::SparseMatrix<double>>& A_;
  // constraint values
  const std::vector<double>& b_;
  // parameters
  RankInflateParams params_;
  // Store QR decomposition results
  mutable QRResult qr_jacobian;
  mutable QRResult qr_hessian;

  // Constructor
  RankInflation(const Matrix& C, double rho,
                const std::vector<Eigen::SparseMatrix<double>>& A,
                const std::vector<double>& b, RankInflateParams params);

  // Evaluate constraints (and cost if enabled) and compute the gradients
  Vector eval_constraints(const Matrix& Y, Matrix& Jac) const;

  // Eval constraints (without Jacobian)
  Vector eval_constraints(const Matrix& Y) const {
    auto dummy_jac = Matrix(m, dim * Y.cols());
    return eval_constraints(Y, dummy_jac);
  }

  // Inflate the solution to a desired rank
  // Returns the inflated solution as well as the corresponding Jacobian
  std::pair<Matrix, Matrix> inflate_solution(const Matrix& Y_0) const;

  // Perform the retraction step onto the manifold and returns the final
  // violation. Jac is filled with the Jacobian of the constraints at the
  // retracted point.
  Vector retraction(Matrix& Y, Matrix& Jac) const;

  // Perform the retraction step onto the manifold and returns the final
  // violation.
  Vector retraction(Matrix& Y) const {
    auto dummy_jac = Matrix(m, dim * Y.cols());
    return retraction(Y, dummy_jac);
  }

  // Build the optimality certificate for the problem using the KKT Jacobian
  // and the (high rank) solution
  Matrix build_certificate(const Matrix& Jac, const Matrix& Y) const;

  // Check global optimality of a solution
  // Returns the minimum eigenvalue of the certificate matrix and the evaluation
  // of the certificate matrix at the solution (first order condition)
  std::pair<double, double> check_certificate(const Matrix& H,
                                              const Matrix& Y) const;

  // Implement backtracking line search to find step size
  double backtrack_line_search(const Matrix& Y, const Matrix& dY,
                               const Matrix& Jac) const;

  // Build second-order correction Hessian.
  // Note: The actual correction Hessian is H = kron(I_r, H_r)
  // where I_r is the identity matrix of size r (rank) and H_r is what gets
  // returned by this function
  Matrix build_sec_ord_corr_hessian(const Vector& violation) const;

  // Build weighted sum of constraint matrices: sum_i A_i * lambda_i
  // If the coefficient falls below `tol` then the corresponding constraint is
  // not added to the sum
  SpMatrix build_wt_sum_constraints(const Vector& coeffs,
                                    double tol = 0.0) const;

  // Compute the second order correction Hessian projected into a given basis
  std::pair<Matrix, Vector> build_proj_corr_grad_hess(
      const Vector& violation, const Matrix& basis,
      const Vector& delta_n) const;

  // Compute the components of a the second order geodesic step
  // Returns a pair (V,W) such that the geodesic step is given by
  //   Y' = Y + alpha*V + alpha^2*W
  // If `second_order` is false then W is not a valid matrix and should not be
  // used
  std::pair<Matrix, Matrix> get_geodesic_step(int rank,
                                              bool second_order = true) const;

  // ----------- ANALYTIC CENTERING METHODS ------------
  // Centering method to compute the analytic center of the current
  // feasible region starting from X_0
  Matrix get_analytic_center(const Matrix& X_0) const;

  // Helper function to build the system of equations for the analytic center
  std::tuple<Matrix, Vector, Vector> get_analytic_center_system(
      const Matrix& Z, const Matrix& X) const;

  double get_analytic_center_objective(const Matrix& X) const {
    auto I = Matrix::Identity(X.rows(), X.cols());
    return neglogdet(X + I * params_.delta_ac);
  }

  // Perform bisection line search to find optimal step size for analytic
  // center
  double analytic_center_bisect(const Matrix& Z, const Matrix& Aw) const;

  // Line search function and derivative for analytic center
  // NOTE: it was shown in Boyd that this function is convex, so simple
  // bisection on the derivative is sufficient
  std::pair<ScalarFunc, ScalarFunc> analytic_center_line_search_func(
      const Matrix& Z, const Matrix& Aw) const;
};

}  // namespace SDPTools
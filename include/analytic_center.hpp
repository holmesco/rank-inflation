#pragma once
#include "utils.hpp"

namespace SDPTools {

struct AnalyticCenterResult {
  // Analytic center solution
  Matrix X;
  // Certificate matrix at the solution
  Matrix H;
  // Optimal multipliers for the constraints at the solution
  Vector multipliers;
  // Constraint violation at the solution
  Vector violation;
  // Whether the solution is certified by the certificate matrix
  bool certified;
  // Minimum eigenvalue of the certificate matrix
  double min_eig;
  // Complementarity condition (first order condition)
  double complementarity;
};

struct AnalyticCenterParams {
  // NOTE: for tolerances on rank and nullspace: pivot added to rank if R_ii >
  // tol * R_max

  // Verbosity
  bool verbose = true;
  // Max number of iterations
  int max_iter = 10;
  // threshold for checking rank of the solution
  // (does not affect convergence, just for display)
  double tol_rank_sol = 1.0E-6;

  // Analytic Center parameters
  // -------------------------
  // tolerance for step size (terminate if below)
  double tol_step_norm_ac = 1e-8;
  // reduce violation in centering step
  bool reduce_violation_ac = true;
  // max number of iterations for centering
  int max_iter_ac = 50;

  // max number of iterations for adaptive centering
  int max_iter_adaptive_ac = 20;
  // initial delta for centering (should be large enough to ensure good
  // convergence even for low rank solutions, but not too large to cause slow
  // convergence)
  double delta_init_ac = 1e-7;
  // final delta for centering (should be small to get close to boundary, but
  // not too small to cause numerical issues)
  double delta_min_ac = 1e-9;
  // update factor for adjusting delta in adaptive centering (should be between
  // zero and one, smaller values lead to more conservative updates)
  double adapt_factor_ac = 0.5;
  // enable for certificate check during centering
  // NOTE: can be used to terminate centering early if the certificate is PSD
  // within tolerance, which can be a good heuristic to avoid unnecessary
  // centering steps when the solution is already close to optimal
  bool check_cert_ac = true;

  // Line search
  // ----------------
  // line search enable for analytic center
  bool enable_line_search_ac = false;
  // Line search sufficient decrease parameter (should be between zero and one)
  double ln_search_suff_dec = 1e-4;
  // Line search reduction factor (should be between zero and one)
  double ln_search_red_factor = 0.5;
  // Line search initialization
  double alpha_init = 1.0;
  // Line search lower bound
  double alpha_min = 1e-14;
  // line search (bisection) parameters for centering
  // NOTE: line search param will be certain to 1/2^k for k = ls_iter_ac
  double tol_bisect_ac = 1e-6;

  // Certificate parameters
  // -------------------------
  // tolerance for checking PSDness of certificate matrix
  double tol_cert_psd = 1e-5;
  // tolerance for checking first order condition of certificate matrix
  double tol_cert_first_order = 1e-5;
};

class AnalyticCenter {
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
  AnalyticCenterParams params_;

  // Constructor
  AnalyticCenter(const Matrix& C, double rho,
                 const std::vector<Eigen::SparseMatrix<double>>& A,
                 const std::vector<double>& b, AnalyticCenterParams params);

  // Evaluate constraints (and cost if enabled) and compute the gradients
  Vector eval_constraints(const Matrix& Y) const;

  // Run analytic centering algorithm starting to certify local solution Y_0.
  // Returns the final result, including the centered primal solution, the
  // certificate matrix, the optimal multipliers, and the certificate
  // information (minimum eigenvalue and complementarity).
  AnalyticCenterResult certify(const Matrix& Y_0, double delta) const;
  
  // Centering method to compute the analytic center of the current
  // feasible region starting from X_0.
  // Delta represents a perturbation parameter to ensure we stay in the interior
  // of the PSD cone even when the solution is low rank. If delta is zero then
  // no perturbation is applied.
  std::pair<Matrix, Vector> get_analytic_center(
      const Matrix& Y_0, double delta_obj = 0.0,
      double delta_constraint = 0.0) const;

  // Build the optimality certificate for the problem using the optimal
  // multipliers
  Matrix build_certificate_from_dual(const Vector& multipliers) const;

  // Check global optimality of a solution
  // Returns the minimum eigenvalue of the certificate matrix and the evaluation
  // of the certificate matrix at the solution (first order condition)
  std::pair<double, double> check_certificate(const Matrix& H,
                                              const Matrix& Y) const;


 protected:
  // Build weighted sum of constraint matrices: sum_i A_i * lambda_i
  // If the coefficient falls below `tol` then the corresponding constraint is
  // not added to the sum
  SpMatrix build_adjoint(const Vector& coeffs, double tol = 0.0) const;

  // Adaptive centering method to compute the analytic center of the SDP.
  // The method starts with a large delta parameter to ensure good convergence
  // even for low rank solutions, and then reduces delta adaptively until it
  // reaches the desired value.
  Matrix get_analytic_center_adaptive(const Matrix& X_0) const;

  // Builds and solves the system of equations for the analytic center step,
  // returning the optimal multipliers and the current violation of constraints
  std::pair<Vector, Vector> solve_analytic_center_system(
      const Matrix& Z, const Matrix& X, double delta_constraint) const;

  double get_analytic_center_objective(const Matrix& X, double delta) const {
    auto I = Matrix::Identity(X.rows(), X.cols());
    return -logdet(X + I * delta);
  }

  // Analytic center backtracking line search
  std::pair<double, double> analytic_center_backtrack(const Matrix& Z,
                                                      const Matrix& Aw) const;

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
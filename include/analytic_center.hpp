#pragma once
#include <chrono>

#include "utils.hpp"

namespace RankTools {

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

enum class LinearSolverType { LDLT, CG };

std::string print_solver(LinearSolverType solver) {
  switch (solver) {
    case LinearSolverType::LDLT:
      return "LDLT";
    case LinearSolverType::CG:
      return "CG";
    default:
      return "Unknown";
  }
}

struct AnalyticCenterParams {
  // Verbosity
  bool verbose = true;
  // threshold for checking rank of the solution
  // (does not affect convergence, just for display)
  double tol_rank_sol = 1.0E-6;
  // tolerance for step size (terminate if below)
  double tol_step_norm = 1e-8;
  // reduce violation in centering step
  bool reduce_violation = true;
  // max number of iterations for centering
  int max_iter = 50;
  // Rescale linear system for centering
  // This rescaling is consistent with the system in Sremac 2021
  bool rescale_lin_sys = true;
  // Select linear solver for centering step
  LinearSolverType lin_solver = LinearSolverType::LDLT;

  // Adaptive Perturbation Parameters
  // -------------------------
  // enable adaptive perturbation for centering
  bool adaptive_perturb = true;
  // final delta for centering (should be small to get close to boundary, but
  // not too small to cause numerical issues)
  double delta_min = 1e-9;
  // Max step size for increasing perturbation. If the step size is above this
  // threshold, then we consider that the step size is too large and we increase
  // the perturbation parameter to encourage more central steps.
  double delta_inc_step_max = 0.1;
  // update factor for adjusting delta in adaptive centering
  double delta_inc = 2.0;
  // Min step size for decreasing perturbation. If the step size is below this
  // threshold, then we consider that the step size is sufficiently small and we
  // decrease the perturbation parameter to allow for more aggressive steps
  // towards the boundary.
  double delta_dec_step_min = 0.9;
  // update factor for adjusting delta in adaptive centering
  double delta_dec = 0.6;

  // Line search
  // ----------------
  // line search enable for analytic center
  bool enable_line_search = true;
  // Line search sufficient decrease parameter (should be between zero and one)
  double ln_search_suff_dec = 1e-4;
  // Line search reduction factor (should be between zero and one)
  double ln_search_red_factor = 0.5;
  // Line search initialization
  double alpha_init = 1.0;
  // Line search lower bound
  double alpha_min = 1e-10;
  // line search (bisection) parameters for centering
  // NOTE: line search param will be certain to 1/2^k for k = ls_iter_ac
  double tol_bisect = 1e-6;

  // Certificate parameters
  // -------------------------
  // enable for certificate check during centering
  // NOTE: can be used to terminate centering early if the certificate is PSD
  // within tolerance, which can be a good heuristic to avoid unnecessary
  // centering steps when the solution is already close to optimal
  bool check_cert = true;
  // tolerance for checking PSDness of certificate matrix
  double tol_cert_psd = 1e-5;
  // tolerance for checking first order condition of certificate matrix
  double tol_cert_complementarity = 1e-5;
  // primal feasibility tolerance for certificate check (i.e., tolerance for
  // violation of constraints)
  double tol_cert_primal_feas = 1e-5;
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
  AnalyticCenterResult certify(const Matrix& Y_0, double delta_init) const;

  // Centering method to compute the analytic center of the current
  // feasible region starting from X_0.
  // Delta represents a perturbation parameter to ensure we stay in the interior
  // of the PSD cone even when the solution is low rank. If delta is zero then
  // no perturbation is applied.
  std::pair<Matrix, Vector> get_analytic_center(const Matrix& Y_0,
                                                double delta_init) const;

  // Build the optimality certificate for the problem using the optimal
  // multipliers
  Matrix build_certificate_from_dual(const Vector& multipliers) const;

  // Check global optimality of a solution
  // Returns the minimum eigenvalue of the certificate matrix and the evaluation
  // of the certificate matrix at the solution (first order condition)
  std::pair<double, double> check_certificate(const Matrix& H,
                                              const Matrix& Y) const;

 protected:
  // Previous multipliers for iterative linear system solvers
  mutable Vector prev_multipliers_;

  // Build weighted sum of constraint matrices: sum_i A_i * lambda_i
  // If the coefficient falls below `tol` then the corresponding constraint is
  // not added to the sum
  SpMatrix build_adjoint(const Vector& coeffs, double tol = 0.0) const;

  // Builds and solves the system of equations for the analytic center step,
  // returning the optimal multipliers and the current violation of constraints
  std::pair<Vector, Vector> get_multipliers(const Matrix& Z,
                                            double delta) const;

  // Intermediate representation of the analytic center linear system
  struct ACSystem {
    Matrix H;                     // LHS matrix (m x m)
    Vector d;                     // RHS vector (m)
    Vector violation;             // constraint violation (m)
    std::vector<Matrix> AZ;       // A_i * Z products
    std::vector<double> A_trace;  // diagonal traces of A_i
  };

  // Constructs the linear system (H, d, violation) for the analytic center step
  ACSystem build_ac_system(const Matrix& Z, double delta) const;

  // Solves the linear system H * multipliers = d using the configured solver
  Vector solve_ac_system(const ACSystem& system) const;

  double get_analytic_center_objective(const Matrix& X, double delta) const {
    auto I = Matrix::Identity(X.rows(), X.cols());
    return -logdet(X + I * delta);
  }

  // Line search to ensure PSDness of the solution for the analytic center step
  double line_search_psd(Matrix& Z, const Matrix& dZ) const;

  // Line search based on determinant increase for analytic center
  std::pair<double, double> line_search_det(const Matrix& Z,
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

}  // namespace RankTools
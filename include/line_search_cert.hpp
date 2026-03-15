#pragma once
#include <Eigen/IterativeLinearSolvers>
#include <chrono>

#include "matrix_free_methods.hpp"
#include "utils.hpp"

namespace RankTools {

struct LineCertifierResult {
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

struct LineCertifierParams {
  // Verbosity
  bool verbose = true;
  // threshold for checking rank of the solution
  // (does not affect convergence, just for display)
  double tol_rank_sol = 1.0E-6;
  // tolerance for step size (terminate if below)
  double tol_step_norm = 1e-8;
  // max number of iterations for centering
  int max_iter = 50;
  // Select linear solver for centering step
  LinearSolverType lin_solver = LinearSolverType::LDLT;

  // Iterative Linear Solve Parameters
  // -----------------
  // Max iterations for iterative linear solvers
  int lin_solve_max_iter = 100;
  // Tolerance for iterative linear solvers
  double lin_solve_tol = 1e-5;

  // Certificate parameters
  // -------------------------
  // tolerance for checking PSDness of certificate matrix
  double tol_cert_psd = 1e-5;
  // tolerance for checking first order condition of certificate matrix
  double tol_cert_complementarity = 1e-5;
  // primal feasibility tolerance for certificate check (i.e., tolerance for
  // violation of constraints)
  double tol_cert_primal_feas = 1e-5;
};

class LineCertifier {
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
  // factorization of solution matrix
  const Matrix Y_;
  // factorization of the interior solution matrix for scaling in line search
  Matrix V_;

  // parameters
  LineCertifierParams params_;

  // Constructor
  LineCertifier(const Matrix& C, double rho,
                const std::vector<Eigen::SparseMatrix<double>>& A,
                const std::vector<double>& b, const Matrix& Y,
                LineCertifierParams params);

  LineCertifierResult certify(const Matrix& Y_0, double delta_init) const;

  // Build the optimality certificate for the problem using the optimal
  // multipliers
  Matrix build_certificate_from_dual(const Vector& multipliers) const;

  // Check global optimality of a solution
  // Returns the minimum eigenvalue of the certificate matrix and the evaluation
  // of the certificate matrix at the solution (first order condition)
  std::pair<double, double> eval_certificate(const Matrix& H,
                                             const Matrix& Y) const;

  // Check global optimality of a solution
  // Returns the whether certificate matrix is PSD and complementarity
  // of the provided solution.
  std::pair<double, double> check_certificate(const Matrix& H,
                                              const Matrix& Y) const;

  // Get the factorization of the primal matrix
  // We do this by computing the null space of the solution matrix in a one-time
  // computation
  Matrix get_interior_factor(const Matrix& Y) const;

  // Retrieve the scaling diagonal  for the line search
  Diagonal get_scaling_diagonal(double alpha) const;

 protected:
  // Previous multipliers for iterative linear system solvers
  mutable Vector prev_multipliers_;

  // Build weighted sum of constraint matrices: sum_i A_i * lambda_i
  // If the coefficient falls below `tol` then the corresponding constraint is
  // not added to the sum
  SpMatrix build_adjoint(const Vector& coeffs, double tol = 0.0) const;
};

}  // namespace RankTools
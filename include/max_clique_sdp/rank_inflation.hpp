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

// Result of solving a linear system using rank-revealing QR decomposition.
// Contains both the least-squares particular solution and the nullspace basis.
struct QRResult {
  Vector solution;
  Matrix nullspace_basis;
  int rank;
};

// Get the particular solution and null space of a system of linear equations
// using rank revealing QR decomposition This formulation is designed for dens
// matrices
QRResult get_soln_qr_dense(const Matrix& A, const Vector& b,
                           const double threshold);

// Compute the rank of a dense matrix with rank-revealing QR
int get_rank(const Matrix& Y, const double threshold);

struct RankInflateParams {
  // Verbosity
  bool verbose = true;
  // Include cost value in the constraint list
  bool enable_cost_constraint = true;
  // Desired rank
  int target_rank = 1;
  // Enable for increasing rank (for debugging)
  bool enable_inc_rank = true;
  // Max number of iterations
  int max_iter = 50;
  // Line search enable
  bool enable_line_search = true;
  // Line search sufficient decrease parameter (should be between zero and one)
  double ln_search_suff_dec = 1e-4;
  // Line search reduction factor (should be between zero and one)
  double ln_search_red_factor = 0.5;
  // Line search initialization
  double alpha_init = 1.0;
  // Line search lower bound 
  double alpha_min = 1e-4;
  // Nullspace step size (wrt Frobenius norm)
  double step_frac_null = 1E-2;
  // tolerance for constraint norm satisfaction.
  double tol_violation = 1.0E-6;
  // threshold for checking rank of the solution
  // NOTE: pivot added to rank if R_ii > thresh * R_max
  double rank_thresh_sol = 1.0E-3;
  // threshold for computing rank of solution null space
  double rank_thresh_null = 1.0E-12;
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

  // Constructor
  RankInflation(const Matrix& C, double rho,
                const std::vector<Eigen::SparseMatrix<double>>& A,
                const std::vector<double>& b, RankInflateParams params);

  // Evaluate constraints (and cost if enabled) and compute the gradients
  Vector eval_constraints(const Matrix& Y,
                          std::unique_ptr<Matrix>* Jac = nullptr) const;

  // Inflate the solution to a desired rank
  Matrix inflate_solution(const Matrix& Y_0,
                          std::unique_ptr<Matrix>* Jac_final = nullptr) const;

  // Build the optimality certificate for the problem using the KKT Jacobian and
  // the (high rank) solution
  Matrix build_certificate(const Matrix& Jac, const Matrix& Y) const;

  // Check global optimality of a solution
  // Returns the minimum eigenvalue of the certificate matrix and the evaluation
  // of the certificate matrix at the solution (first order condition)
  std::pair<double, double> check_certificate(const Matrix& H,
                                              const Matrix& Y) const;
  
  // Implement backtracking line search to find step size
  double backtrack_line_search(const Matrix& Y, const Matrix& dY, const Matrix& Jac) const;
};

}  // namespace SDPTools
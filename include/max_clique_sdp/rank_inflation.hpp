#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <memory>
#include <cassert>
#include <iostream>

namespace SDPTools {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Triplet = Eigen::Triplet<double>;

// Result of solving a linear system using rank-revealing QR decomposition.
// Contains both the least-squares particular solution and the nullspace basis.
struct QRResult {
    Vector solution_particular;
    Matrix nullspace_basis;
    int rank;
};

// Get the particular solution and null space of a system of linear equations using rank revealing QR decomposition
// This formulation is designed for dens matrices
QRResult get_soln_qr_dense(const Matrix& A, const Vector& b);

struct RankInflateParams {
  // Verbosity
    bool verbose = true;
  // Include cost value in the constraint list
  bool use_cost_constraint = true;
  // Desired rank
  int target_rank=1;
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
  RankInflation(const Matrix& C, double rho, const std::vector<Eigen::SparseMatrix<double>>& A,
                const std::vector<double>& b, RankInflateParams params);

  // Evaluate constraints (and cost if enabled) and compute the gradients
  Vector eval_constraints(const Matrix& Y, std::shared_ptr<Matrix> grad=nullptr) const;

  // Inflate the solution to a desired rank
  // Matrix inflate_solution(const Matrix& y_0) const;

};



}  // namespace SDPTools
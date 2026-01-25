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

// ---- Symmetric matrix vectorization helper functions ----

// Converts a symmetric matrix to a vectorized form (unique elements)
Vector vec_symm(const Matrix& A);

// Converts a vectorized form back to a symmetric matrix
Matrix unvec_symm(const Vector& v, int dim);

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
  const float rho_;
  // constraint matrices
  const std::vector<Eigen::SparseMatrix<double>>& A_;
  // constraint values
  const std::vector<float>& b_;
  // parameters
  RankInflateParams params_;

  // Constructor
  RankInflation(const Matrix& C, float rho, const std::vector<Eigen::SparseMatrix<double>>& A,
                const std::vector<float>& b, RankInflateParams params);

  // Evaluate constraints (and cost if enabled) and compute the gradients
  Vector eval_constraints(const Matrix& Y, Matrix& grad) const;

  // Inflate the solution to a desired rank
  // Matrix inflate_solution(const Matrix& y_0) const;

};



}  // namespace SDPTools
#include "max_clique_sdp/rank_inflation.hpp"

namespace SDPTools {

RankInflation::RankInflation(const Matrix& C, float rho,
                             const std::vector<Eigen::SparseMatrix<double>>& A,
                             const std::vector<float>& b,
                             RankInflateParams params)
    : C_(C), A_(A), rho_(rho), b_(b), params_(params) {
  // dimension of the SDP
  dim = C.rows();
  // number of constraints to enforce during inflation
  m = params.use_cost_constraint ? A.size() + 1 : A.size();
}

Vector RankInflation::eval_constraints(const Matrix& Y, Matrix& grad) const {
  // Size assertions
  int r = params_.target_rank;
  assert(Y.rows() == dim);
  assert(Y.cols() == r);
  assert(grad.rows() == m);
  assert(grad.cols() == dim * r);
  // Create vectorized version of Y
  const Vector Y_vec = Y.reshaped();
  // Loop through constraints, evaluating gradient and constraint value
  Vector result(m);
  for (int i = 0; i < m; i++) {
    // compute vectorized gradient (Eigen stores in column major order)
    Vector grad_vec(dim*r);
    float constraint_value;
    if (i < A_.size()) {
      // Constraints
      // NOTE: Converting to DENSE here. Optimize this later
      grad_vec = (A_[i].selfadjointView<Eigen::Upper>() * Y).reshaped();
      constraint_value = b_[i];
    } else {
      // Cost "constraint"
      assert(params_.use_cost_constraint);
      grad_vec = (C_ * Y).reshaped();
      constraint_value = rho_;
    }
    // Store gradient
    grad.row(i) = 2.0 * grad_vec;
    // evaluate product
    result(i) = grad_vec.dot(Y_vec) - constraint_value;
  }
  return result;
}

}  // namespace SDPTools
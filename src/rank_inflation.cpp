#include "max_clique_sdp/rank_inflation.hpp"

namespace SDPTools {

RankInflation::RankInflation(const Matrix& C, double rho,
                             const std::vector<Eigen::SparseMatrix<double>>& A,
                             const std::vector<double>& b,
                             RankInflateParams params)
    : C_(C), A_(A), rho_(rho), b_(b), params_(params) {
  // dimension of the SDP
  dim = C.rows();
  // number of constraints to enforce during inflation
  m = params.use_cost_constraint ? A.size() + 1 : A.size();
}

Vector RankInflation::eval_constraints(
    const Matrix& Y, std::shared_ptr<Matrix> grad) const {
  // dimension assertions
  int r = params_.target_rank;
  assert(Y.rows() == dim);
  assert(Y.cols() == r);
  if (grad != nullptr) {
    assert(grad->rows() == m);
    assert(grad->cols() == dim * r);
  }
  // Create vectorized version of Y
  const Vector Y_vec = Y.reshaped();
  // Loop through constraints, evaluating gradient and constraint value
  Vector result(m);
  for (int i = 0; i < m; i++) {
    // compute vectorized gradient (Eigen stores in column major order)
    Vector grad_vec(dim * r);
    double constraint_value;
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
    if (grad != nullptr) {
      grad->row(i) = 2.0 * grad_vec;
    }
    // evaluate product
    result(i) = grad_vec.dot(Y_vec) - constraint_value;
  }
  return result;
}

QRResult get_soln_qr_dense(const Matrix& A, const Vector& b) {
  // 1. Perform Column Pivoted Householder QR (Rank-Revealing)
  Eigen::ColPivHouseholderQR<Matrix> qr(A);

  int m = A.rows();
  int n = A.cols();
  int r = qr.rank();

  QRResult result;
  result.rank = r;

  // 2. Find the particular solution
  // If the system is inconsistent, this provides a least-squares solution
  result.solution_particular = qr.solve(b);

  // 3. Characterize the Null Space
  if (r >= n) {
    // Full column rank: Null space is just the zero vector
    result.nullspace_basis = Matrix::Zero(n, 0);
  } else {
    // Extract R and the permutation matrix P
    // R is stored in the upper triangle of the matrixQR
    Matrix R = qr.matrixQR().triangularView<Eigen::Upper>();
    Matrix P = qr.colsPermutation();

    // Partition R: R11 is (r x r), R12 is (r x n-r)
    Matrix R11 = R.block(0, 0, r, r);
    Matrix R12 = R.block(0, r, r, n - r);

    // Compute -R11^{-1} * R12
    // We use triangular solve for better numerical stability
    Matrix top = -R11.inverse() * R12;

    // Construct the basis in the permuted space: [ -R11^-1 * R12 ; I ]
    Matrix basisPermuted(n, n - r);
    basisPermuted.block(0, 0, r, n - r) = top;
    basisPermuted.block(r, 0, n - r, n - r) = Matrix::Identity(n - r, n - r);

    // Transform back to original space using the permutation matrix P
    result.nullspace_basis = P * basisPermuted;
  }

  return result;
}

}  // namespace SDPTools
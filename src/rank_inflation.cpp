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

Vector RankInflation::eval_constraints(const Matrix& Y,
                                       std::shared_ptr<Matrix> grad) const {
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

Matrix RankInflation::inflate_solution(const Matrix& Y_0) const {
  // convenience definitions
  int r_targ = params_.target_rank;
  // Create initial solution by padding with zeros
  assert(Y_0.rows() == dim && "Initial solution has the wrong number of rows");
  assert(Y_0.cols() <= r_targ &&
         "Initial solution has rank higher than target");
  Matrix zpad = Matrix::Zero(dim, r_targ - Y_0.cols());
  Matrix Y(dim, r_targ);
  Y << Y_0, zpad;

  // *** Main Loop ***
  // Initialize
  int n_iter = 0;
  bool converged = false;
  auto grad = std::make_shared<Matrix>(m, dim * r_targ);
  int r = get_rank(Y, params_.rank_thresh_sol);
  auto violation = eval_constraints(Y, grad);
  bool rank_increase = false;
  // Loop
  while (n_iter < params_.max_iter && !converged) {
    // Solve linear system to get update
    auto result =
    get_soln_qr_dense(*grad, -violation, params_.rank_thresh_null);
    // dimension of the solution space
    int nulldim = result.nullspace_basis.cols();
    // Print outputs
    if (params_.verbose) {
      if (n_iter == 0) {
      std::printf("%6s %6s %18s %10s %6s\n", "Iter", "Rank", "ViolationNorm",
            "NullDim", "RankUp");
      }
      char rank_up = rank_increase ? 'T' : 'F';
      std::printf("%6d %6d %18.6e %10d %6c\n", n_iter, r, violation.norm(),
            nulldim, rank_up);
      rank_increase = false; // reset
    }
    
    // Add solution to matrix (Newton step)
    Matrix Y_corrected =
        Y + params_.step_corr * result.solution.reshaped(dim, r_targ);
    // if rank not high enough, try to increase
    if (r < r_targ) {
      rank_increase = true;
      // Get random (normalized) matrix from nullspace
      Eigen::VectorXd alpha =
          Eigen::VectorXd::Random(nulldim);  // values in [-1,1]
      Matrix N = (result.nullspace_basis * alpha).reshaped(dim, r_targ);
      double norm_N = N.norm();
      if (norm_N > 0) {
        N /= norm_N;
      }
      // Add to solution
      Y_corrected.noalias() = (1 - params_.step_frac_null) * Y_corrected +
                              params_.step_frac_null * N;
    }
    // Update solution
    Y = Y_corrected;
    // Evaluate the constraints
    violation = eval_constraints(Y, grad);
    // Get new rank of Y
    r = get_rank(Y, params_.rank_thresh_sol);
    // Check convergence
    converged = r >= r_targ && violation.norm() < params_.tol_violation;
    // update
    n_iter++;
  }

  return Y;
}

int get_rank(const Matrix& Y, const double threshold) {
  // 1. Perform Column Pivoted Householder QR (Rank-Revealing)
  Eigen::ColPivHouseholderQR<Matrix> qr(Y);
  qr.setThreshold(threshold);
  return qr.rank();
}

QRResult get_soln_qr_dense(const Matrix& A, const Vector& b,
                           const double threshold) {
  // Perform Column Pivoted Householder QR (Rank-Revealing)
  Eigen::ColPivHouseholderQR<Matrix> qr(A);
  // Set threshold for nullspace
  qr.setThreshold(threshold);
  // get dimensions
  int m = A.rows();
  int n = A.cols();
  int r = qr.rank();

  QRResult result;
  result.rank = r;

  // 2. Find the particular solution
  // If the system is inconsistent, this provides a least-squares solution
  result.solution = qr.solve(b);

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
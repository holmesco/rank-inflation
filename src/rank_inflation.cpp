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
  m = params.enable_cost_constraint ? A.size() + 1 : A.size();
}

Vector RankInflation::eval_constraints(const Matrix& Y,
                                       std::unique_ptr<Matrix>* Jac) const {
  // dimension assertions
  int r = params_.target_rank;
  assert(Y.rows() == dim);
  assert(Y.cols() == r);
  if (Jac != nullptr) {
    assert((*Jac)->rows() == m);
    assert((*Jac)->cols() == dim * r);
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
      assert(params_.enable_cost_constraint);
      grad_vec = (C_ * Y).reshaped();
      constraint_value = rho_;
    }
    // Store gradient
    if (Jac != nullptr) {
      (*Jac)->row(i) = 2.0 * grad_vec;
    }
    // evaluate product
    result(i) = grad_vec.dot(Y_vec) - constraint_value;
  }
  return result;
}

Matrix RankInflation::inflate_solution(
    const Matrix& Y_0, std::unique_ptr<Matrix>* Jac_final) const {
  // convenience definitions
  int r_targ = params_.target_rank;
  // Create initial solution by padding with zeros
  assert(Y_0.rows() == dim && "Initial solution has the wrong number of rows");
  assert(Y_0.cols() <= r_targ &&
         "Initial solution has rank higher than target");
  Matrix zpad = Matrix::Zero(dim, r_targ - Y_0.cols());
  Matrix Y(dim, r_targ);
  Y << Y_0, zpad;

  // DEBUG
  std::vector<double> vals(b_.begin(), b_.end());
  vals.push_back(rho_);
  Vector constraint_val = Vector::Map(vals.data(), vals.size());

  // *** Main Loop ***
  // Initialize
  int n_iter = 0;
  bool converged = false;
  auto Jac = std::make_unique<Matrix>(m, dim * r_targ);
  int r = get_rank(Y, params_.rank_thresh_sol);
  auto violation = eval_constraints(Y, &Jac);
  bool rank_increase = false;
  // Loop
  while (n_iter < params_.max_iter && !converged) {
    // Solve linear system to get update
    auto result = get_soln_qr_dense(*Jac, -violation, params_.rank_thresh_null);
    // dimension of the solution space
    int nulldim = result.nullspace_basis.cols();

    // Get the Gauss-Newton step
    Matrix dY = result.solution.reshaped(dim, r_targ);
    // Line search
    double alpha;
    if (params_.enable_line_search) {
      alpha = RankInflation::backtrack_line_search(Y, dY, *Jac);
    } else {
      alpha = 1.0;
    }
    // Apply update
    Matrix Y_corrected = Y + alpha * dY;

    // // DEBUG
    // Vector viol_quad = eval_constraints(result.solution.reshaped(dim,
    // r_targ)) +
    //                    constraint_val;
    // std::cout << "Viol_quad (debug): " << std::endl
    //           << viol_quad.transpose() << std::endl;
    // std::cout << "Norm squared of dY: " << std::pow(dY.norm(), 2) <<
    // std::endl;

    // if rank not high enough, try to increase
    if (r < r_targ && params_.enable_inc_rank) {
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
    // Evaluate the current solution (violation and rank)
    violation = eval_constraints(Y, &Jac);
    r = get_rank(Y, params_.rank_thresh_sol);
    // Check convergence
    converged = r >= r_targ && violation.norm() < params_.tol_violation;
    // update
    n_iter++;
    // Print outputs
    if (params_.verbose) {
      if (n_iter % 10 == 1) {
        std::printf("%6s %6s %18s %10s %10s %6s\n", "Iter", "Rank",
                    "ViolationNorm", "NullDim", "Alpha", "RankUp");
      }
      char rank_up = rank_increase ? 'T' : 'F';
      std::printf("%6d %6d %18.6e %10d %10.3e %6c\n", n_iter, r,
                  violation.norm(), nulldim, alpha, rank_up);
      rank_increase = false;  // reset

      // DEBUGGING
      //  std::cout << "Violation: " << violation.transpose() << std::endl;
      //  std::cout << "Step Norm: " << dY.norm() << std::endl;
    }
  }
  if (Jac_final != nullptr) {
    // transfer ownership of the unique pointer to the outer scope
    *Jac_final = std::move(Jac);
  }

  return Y;
}

double RankInflation::backtrack_line_search(const Matrix& Y, const Matrix& dY,
                                            const Matrix& Jac) const {
  // Initial step size
  double alpha = params_.alpha_init;
  // Current violation
  Vector violation = eval_constraints(Y);
  double norm_viol = violation.norm();
  // Parameters for backtracking
  const double beta =
      params_.ln_search_red_factor;             // step size reduction factor
  const double c = params_.ln_search_suff_dec;  // sufficient decrease parameter

  // Backtracking loop
  while (true) {
    // Compute new candidate solution
    Matrix Y_new = Y + alpha * dY;
    // Evaluate constraints at new solution
    Vector violation_new = eval_constraints(Y_new);
    double norm_viol_new = violation_new.norm();
    // Check Armijo condition
    if (norm_viol_new <= norm_viol + c * alpha * (Jac * dY.reshaped()).norm()) {
      break;  // Sufficient decrease achieved
    }
    // Reduce step size
    alpha *= beta;
    // Prevent too small step sizes
    if (alpha < params_.alpha_min) {
      alpha = params_.alpha_min;
      break;  // Stop if step size is too small
    }
  }
  return alpha;
}

SpMatrix RankInflation::build_wt_sum_constraints(const Vector& coeffs,
                                                 double tol) const {
  // 1. Calculate the weighted sum of the upper-triangular parts
  SpMatrix upperSum = coeffs[0] * A_[0];
  for (size_t i = 1; i < A_.size(); ++i) {
    if (std::abs(coeffs(i)) > tol) {
      upperSum += coeffs[i] * A_[i];
    }
  }

  // 2. Reflect the upper triangle into the lower triangle to get the full
  // matrix .selfadjointView<Eigen::Upper>() treats the matrix as symmetric and
  // the assignment to a SparseMatrix fills in the missing entries.
  SpMatrix fullMatrix = upperSum.selfadjointView<Eigen::Upper>();

  fullMatrix.makeCompressed();
  return fullMatrix;
}

Matrix RankInflation::build_sec_ord_corr_hessian(
    const Vector& violation) const {
  // build weighted sum of constraint matrices with violation values
  auto f_A = build_wt_sum_constraints(violation, params_.tol_viol_hess);
  // add cost term if required
  Matrix hess_corr;
  if (params_.enable_cost_constraint) {
    hess_corr = C_ * violation(violation.size() - 1) + f_A;
  } else {
    hess_corr = f_A;
  }
  return hess_corr;
}

std::pair<Matrix, Vector> RankInflation::build_proj_corr_grad_hess(
    const Vector& violation, const Matrix& basis, const Vector& delta_n) const {
  // Get rank of current solution
  int r = basis.rows() / dim;
  // Get reduced hessian
  Matrix H_r = build_sec_ord_corr_hessian(violation);
  // Produce B^T * (I_r \otimes H_r) * B and B^T *
  int p = basis.cols();
  Matrix hess = Matrix::Zero(p, p);
  Vector grad = Vector::Zero(p);
  // Instead of I \otimes H, iterate through block diagonal structure
  for (int i = 0; i < r; ++i) {
    // Extract block row "i" from basis and compute hessian
    auto B_i = basis.block(i * dim, 0, dim, p);
    hess += B_i.transpose() * H_r * B_i;
    // extract block row "i" from input vector and compute grad
    auto delta_n_i = delta_n.segment(i * dim, dim);
    grad += B_i.transpose() * H_r * delta_n_i;
  }
  return {hess, grad};
}

Matrix RankInflation::build_certificate(const Matrix& Jac,
                                        const Matrix& Y) const {
  // Get components of stationarity condition
  Vector vecCY;
  Matrix vecAY;
  if (params_.enable_cost_constraint) {
    // vec(C*Y) is last row of jacobian, split it off
    vecCY = Jac.bottomRows(1).transpose();
    vecAY = Jac.topRows(Jac.rows() - 1);
  } else {
    vecCY = (C_ * Y).reshaped();
    vecAY = Jac;
  }
  // Solve for Lagrange multipliers
  auto result =
      get_soln_qr_dense(vecAY.transpose(), -vecCY, params_.rank_thresh_null);
  Vector lagrange = result.solution;
  // Build the certificate matrix
  auto H_sp = build_wt_sum_constraints(lagrange);
  // Add cost matrix and store as dense
  Matrix H = C_ + H_sp;

  return H;
}

std::pair<double, double> RankInflation::check_certificate(
    const Matrix& H, const Matrix& Y) const {
  // Evaluate the stationarity condition
  double first_order_norm = (H * Y).norm();
  // Check Eigenvalues
  // Use SelfAdjointEigenSolver for symmetric matrices
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
  double min_eig = es.eigenvalues().minCoeff();

  return {min_eig, first_order_norm};
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

    // print residual of the solution basis
    // std::cout << "QR Residual:" << (A*result.solution-b).norm() << std::endl;
  }

  return result;
}

}  // namespace SDPTools
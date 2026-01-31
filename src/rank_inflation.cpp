#include "rank_inflation.hpp"

namespace SDPTools {

RankInflation::RankInflation(const Matrix& C, double rho,
                             const std::vector<Eigen::SparseMatrix<double>>& A,
                             const std::vector<double>& b,
                             RankInflateParams params)
    : C_(C), A_(A), rho_(rho), b_(b), params_(params) {
  // dimension of the SDP
  dim = C.rows();
  // number of constraints to enforce during inflation
  m = A.size() + 1;
}

Vector RankInflation::eval_constraints(const Matrix& Y,
                                       std::unique_ptr<Matrix>* Jac) const {
  // dimension assertions
  int r = params_.max_sol_rank;
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
  int r_max = params_.max_sol_rank;
  // Create initial solution by padding with zeros
  assert(Y_0.rows() == dim && "Initial solution has the wrong number of rows");
  assert(Y_0.cols() <= r_max && "Initial solution has rank higher than target");
  Matrix zpad = Matrix::Zero(dim, r_max - Y_0.cols());
  Matrix Y(dim, r_max);
  Y << Y_0, zpad;

  // DEBUG
  std::vector<double> vals(b_.begin(), b_.end());
  vals.push_back(rho_);
  Vector constraint_val = Vector::Map(vals.data(), vals.size());

  // *** Main Loop ***
  // Initialize
  int n_iter = 0;
  bool converged = false;
  auto Jac = std::make_unique<Matrix>(m, dim * r_max);
  int r = get_rank(Y, params_.tol_rank_sol);
  auto violation = eval_constraints(Y, &Jac);
  bool rank_increase = false;
  // Loop
  while (n_iter < params_.max_iter && !converged) {
    // ---------- RETRACTION -----------
    // QR decomposition of Jacobian (and GN solve)
    QRResult qr_jacobian =
        get_soln_qr_dense(*Jac, -violation, params_.tol_null_qr);
    QRResult qr_hessian;
    Vector delta;
    switch (params_.retraction_method) {
      case RetractionMethod::ExactNewton: {
        // Build exact Hessian
        Matrix Hess = (*Jac).transpose() * (*Jac);
        // Add correction term along diagonal blocks
        Matrix H_corr = build_sec_ord_corr_hessian(violation);
        for (int i = 0; i < r_max; i++) {
          Hess.block(i * dim, i * dim, dim, dim).noalias() += H_corr;
        }
        // build gradient
        Vector grad = -(*Jac).transpose() * violation;
        // Solve system
        qr_hessian = get_soln_qr_dense(Hess, grad, params_.tol_null_qr);
        // define step
        // Debug: compute minimum eigenvalue of Hessian
        // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hess);
        // double min_eig = es.eigenvalues().minCoeff();
        // std::cout << "Minimum Eigenvalue of Hessian: " << min_eig <<
        // std::endl;
        break;
      }
      case RetractionMethod::GaussNewton: {
        // get GN solution
        Vector& delta_gn = qr_jacobian.solution;
        // Second order correction
        if (params_.enable_sec_ord_corr) {
          // Get system of equations for second order correction
          auto [hess, grad] = build_proj_corr_grad_hess(
              violation, qr_jacobian.nullspace_basis, delta_gn);
          // Solve new system
          qr_hessian = get_soln_qr_dense(hess, -grad, params_.tol_null_corr);
          // reconstruct tangent-space solution
          auto delta_corr = qr_jacobian.nullspace_basis * qr_hessian.solution;
          // combine normal and tangent componenets
          delta = delta_gn + delta_corr;
        } else {
          delta = delta_gn;
        }
        break;
      }
      case RetractionMethod::GradientDescent: {
        // Get QR Decomposition of Jac
        qr_jacobian = get_soln_qr_dense(*Jac, -violation, params_.tol_null_qr);
        delta = -(*Jac) * violation;
        break;
      }
    }

    // ----- LINE SEARCH -----
    Matrix dY = delta.reshaped(dim, r_max);
    double alpha;
    if (params_.enable_line_search) {
      alpha = RankInflation::backtrack_line_search(Y, dY, *Jac);
    } else {
      alpha = 1.0;
    }
    // Apply update
    Matrix Y_plus = Y + alpha * dY;

    // ------------ TANGENT STEP -----------------
    // Check if the Jabobian is exactly rank deficient by one.
    bool jac_rank_check = check_jac_rank(
        qr_jacobian.R_diagonal, params_.rank_def_thresh, params_.tol_null_qr);
    // Check if the Jacobian has full row rank, if not add perturbation to
    // solution
    if (!jac_rank_check && violation.norm() < params_.tol_violation &&
        params_.enable_inc_rank) {
      rank_increase = true;
      Matrix N;
      if (params_.enable_sec_ord_corr &&
          qr_hessian.nullspace_basis.cols() > 0) {
        // Add perturbation from the inner nullspace
        int nulldim = qr_hessian.nullspace_basis.cols();
        // Get random matrix from nullspace
        Vector phi = Vector::Random(nulldim);  // values in [-1,1]
        N = (qr_jacobian.nullspace_basis * qr_hessian.nullspace_basis * phi)
                .reshaped(dim, r_max);
      } else if (qr_jacobian.nullspace_basis.cols() > 0) {
        // Add perturbation from the outer nullspace
        int nulldim = qr_jacobian.nullspace_basis.cols();
        // Get random matrix from nullspace
        Vector phi = Vector::Random(nulldim);  // values in [-1,1]
        N = (qr_jacobian.nullspace_basis * phi).reshaped(dim, r_max);
      } else {
        // Add random perturbation
        N = Matrix::Random(dim, r_max);
      }
      // Normalize
      double norm_N = N.norm();
      if (norm_N > 0) {
        N /= norm_N;
      }
      double stepsize = n_iter > 0 ? params_.eps_null : params_.eps_null_init;
      // Add to solution
      Y_plus.noalias() = Y_plus + stepsize * N;
    }
    // Update solution
    Y = Y_plus;
    // Evaluate the current solution (violation and rank)
    violation = eval_constraints(Y, &Jac);
    r = get_rank(Y, params_.tol_rank_sol);
    // Check convergence
    converged = jac_rank_check && violation.norm() < params_.tol_violation;
    // update
    n_iter++;
    // Print outputs
    if (params_.verbose) {
      // compute grad and step norms for printing
      Vector grad = (*Jac).transpose() * violation;
      double step_norm = alpha * dY.norm();
      double grad_norm = grad.norm();

      if (n_iter % 10 == 1) {
        std::printf("%6s %6s %18s %10s %10s %10s %6s\n", "Iter", "JacRank",
                    "ViolationNorm", "Alpha", "StepNorm", "GradNorm", "RankUp");
      }
      char rank_up = rank_increase ? 'T' : 'F';
      std::printf("%6d %6d %18.6e %10.3e %10.6e %10.6e %6c\n", n_iter,
                  qr_jacobian.rank, violation.norm(), alpha, step_norm,
                  grad_norm, rank_up);
      rank_increase = false;  // reset
      std::cout << "Grad: " << grad.transpose() << std::endl;
      std::cout << "sol : " << std::endl << Y.reshaped().transpose() << std::endl;
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
  double viol_cost = violation.dot(violation);
  // Parameters for backtracking
  const double beta =
      params_.ln_search_red_factor;             // step size reduction factor
  const double c = params_.ln_search_suff_dec;  // sufficient decrease parameter
  const Vector grad = Jac * violation;
  const double expected_decrease = grad.transpose().dot(dY.reshaped());
  // Backtracking loop
  while (true) {
    // Compute new candidate solution
    Matrix Y_new = Y + alpha * dY;
    // Evaluate constraints at new solution
    Vector violation_new = eval_constraints(Y_new);
    double viol_cost_new = violation_new.dot(violation_new);
    // Check Armijo condition (applies to our actual cost )
    if (viol_cost_new <= viol_cost + c * alpha * expected_decrease) {
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
  hess_corr = C_ * violation(violation.size() - 1) + f_A;

  return hess_corr * 2;
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
  // vec(C*Y) is last row of jacobian, split it off
  Vector vecCY = Jac.bottomRows(1).transpose();
  Matrix vecAY = Jac.topRows(Jac.rows() - 1);

  // Solve for Lagrange multipliers
  auto result =
      get_soln_qr_dense(vecAY.transpose(), -vecCY, params_.tol_null_qr);
  // print diagonal and residual
  std::cout << "Lagrange solve residual norm: " << result.residual_norm
            << std::endl;
  std::cout << "R diagonal: " << result.R_diagonal.transpose() << std::endl;

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
  }
  // Store additional information
  result.R_diagonal = qr.matrixQR().diagonal();
  result.residual_norm = (A * result.solution - b).norm();
  return result;
}

bool RankInflation::check_jac_rank(const Vector& R_diag, double thresh_rank_def,
                                   double thresh_rank) const {
  // Sort the elements along the diagonal of the R matrix
  std::vector<double> vals(R_diag.size());
  for (int i = 0; i < R_diag.size(); ++i) {
    vals[i] = std::abs(R_diag(i));
  }
  std::sort(vals.begin(), vals.end());
  // This ratio should be small to satisfy rank deficiency
  double ratio = vals[A_.size()] / vals[A_.size() - 1];
  // Return true if ratio is below threshold and the other values are above the
  // general threshold
  // std::cout << "rank check diagonal:" << R_diag.transpose() << std::endl;

  return ratio < thresh_rank_def && vals[A_.size() - 1] >= thresh_rank;
}

}  // namespace SDPTools
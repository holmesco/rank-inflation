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

Vector RankInflation::eval_constraints(const Matrix& Y, Matrix& Jac) const {
  // dimension assertions
  int r = Y.cols();
  assert(Y.rows() == dim);
  assert(Jac.rows() == m);
  assert(Jac.cols() == dim * r);

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
    Jac.row(i) = 2.0 * grad_vec;
    // evaluate product
    result(i) = grad_vec.dot(Y_vec) - constraint_value;
  }
  return result;
}

std::pair<Matrix, Matrix> RankInflation::inflate_solution(
    const Matrix& Y_0) const {
  // convenience definitions
  int r_max = params_.max_sol_rank;
  // Create initial solution by padding with zeros
  assert(Y_0.rows() == dim && "Initial solution has the wrong number of rows");
  assert(Y_0.cols() <= r_max && "Initial solution has rank higher than target");
  Matrix zpad = Matrix::Zero(dim, r_max - Y_0.cols());
  Matrix Y(dim, r_max);
  Y << Y_0, zpad;

  // *** Main Loop ***
  // Initialize
  int n_iter = 0;
  auto Jac = Matrix(m, Y.cols() * dim);
  // Loop
  while (n_iter < params_.max_iter) {
    // Apply perturbation
    if (n_iter > 0) {
      if (params_.verbose) {
        std::cout << n_iter << "  PERTURBATION" << std::endl;
      }
      // Get geodesic step
      auto [V, W] = get_geodesic_step(Y.cols(), params_.second_ord_geo);
      if (params_.second_ord_geo) {
        Y.noalias() = Y + params_.eps_geodesic * V +
                      std::pow(params_.eps_geodesic, 2) * W;
      } else {
        Y.noalias() = Y + params_.eps_geodesic * V;
      }
      if (params_.verbose) {
        std::cout << "Perturbed Solution Rank: "
                  << get_rank(Y, params_.tol_rank_sol) << std::endl;
      }
    }

    // Apply Retraction
    if (params_.verbose) {
      std::cout << n_iter << "  RETRACTION" << std::endl;
    }
    auto violation = RankInflation::retraction(Y, Jac);

    // Check if the Jabobian is exactly rank deficient by one.
    int jac_rank = get_rank(Jac.topRows(A_.size()), params_.tol_rank_jac);
    if (jac_rank >= A_.size() && violation.norm() < params_.tol_violation) {
      if (params_.verbose) {
        std::cout << "CONVERGED!" << std::endl;
      }
      break;
    }
    // Print outputs
    if (params_.verbose) {
      std::cout << "Jacobian Rank: " << jac_rank << std::endl;
    }
    // update
    n_iter++;
  }

  return {Y, Jac};
}

Vector RankInflation::retraction(Matrix& Y, Matrix& Jac) const {
  // Initialize
  int r = Y.cols();
  int n_iter = 0;
  Vector violation;
  // Loop
  while (n_iter < params_.max_iter_retract) {
    // Evaluate violation and get jacobian
    violation = eval_constraints(Y, Jac);
    // QR decomposition of Jacobian (run before checking convergence to get QR)
    qr_jacobian = get_soln_qr_dense(Jac, -violation, params_.tol_jac_qr);
    // Check for convergence
    if (violation.norm() < params_.tol_violation) break;
    // Define retraction step
    Vector delta;
    switch (params_.retraction_method) {
      case RetractionMethod::ExactNewton: {
        // Build exact Hessian
        Matrix Hess = Jac.transpose() * Jac;
        // Add correction term along diagonal blocks
        Matrix H_corr = build_sec_ord_corr_hessian(violation);
        for (int i = 0; i < r; i++) {
          Hess.block(i * dim, i * dim, dim, dim).noalias() += H_corr;
        }
        // build gradient
        Vector grad = Jac.transpose() * violation;
        // Solve system
        qr_hessian = get_soln_qr_dense(Hess, -grad, params_.tol_jac_qr);
        // Define step
        delta = qr_hessian.solution;
        // Debug: compute minimum eigenvalue of Hessian
        // Eigen::SelfAdjointEigenSolver<Matrix> es(Hess);
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
          delta = qr_jacobian.solution;
        }
        break;
      }
      case RetractionMethod::GradientDescent: {
        // Get QR Decomposition of Jac
        qr_jacobian = get_soln_qr_dense(Jac, -violation, params_.tol_jac_qr);
        delta = -Jac.transpose() * violation;
        break;
      }
    }

    // ----- LINE SEARCH -----
    Matrix dY = delta.reshaped(dim, r);
    double alpha;
    if (params_.enable_line_search) {
      alpha = RankInflation::backtrack_line_search(Y, dY, Jac);
    } else {
      alpha = 1.0;
    }
    // Apply update
    Y.noalias() = Y + alpha * dY;
    n_iter++;
    // Print Status
    if (params_.verbose) {
      // compute grad and step norms for printing
      Vector grad = Jac.transpose() * violation;
      double step_norm = alpha * dY.norm();
      double grad_norm = grad.norm();

      if (n_iter % 10 == 1) {
        std::printf("%6s %18s %10s %10s %10s\n", "Iter", "ViolationNorm",
                    "Alpha", "StepNorm", "GradNorm");
      }
      std::printf("%6d %18.6e %10.3e %10.6e %10.6e\n", n_iter, violation.norm(),
                  alpha, step_norm, grad_norm);
      // std::cout << "Grad: " << grad.transpose() << std::endl;
      // std::cout << "sol : " << std::endl << Y.reshaped().transpose() <<
      // std::endl;
    }
  }

  return violation;
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
  const Vector grad = Jac.transpose() * violation;
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

std::pair<Matrix, Matrix> RankInflation::get_geodesic_step(
    int rank, bool second_order) const {
  // Get normalized direction in the tangent space
  int nulldim = qr_jacobian.nullspace_basis.cols();
  Vector phi = Vector::Random(nulldim);  // values in [-1,1]
  Matrix V = (qr_jacobian.nullspace_basis * phi).reshaped(dim, rank);
  V /= V.norm();
  // Get second order component
  Matrix W(dim, rank);
  if (second_order) {
    Vector rhs(m);
    for (int i = 0; i < m; i++) {
      if (i < A_.size()) {
        rhs(i) = -(V.transpose() * A_[i].selfadjointView<Eigen::Upper>() * V)
                      .trace();
      } else {
        rhs(i) = -(V.transpose() * C_ * V).trace();
      }
    }
    W = qr_jacobian.qr_decomp.solve(rhs).reshaped(dim, rank);
    std::cout << "Threshold " << qr_jacobian.qr_decomp.threshold() << std::endl;
    std::cout << "R_diag: " << qr_jacobian.qr_decomp.matrixR().diagonal()
              << std::endl;
    std::cout << "Norm of W: " << W.norm() << std::endl;
    std::cout << "Norm of RHS: " << rhs.norm() << std::endl;
  }

  return {V, W};
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
      get_soln_qr_dense(vecAY.transpose(), -vecCY, params_.tol_rank_jac);
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
  Eigen::SelfAdjointEigenSolver<Matrix> es(H);
  double min_eig = es.eigenvalues().minCoeff();

  return {min_eig, first_order_norm};
}

int get_rank(const Matrix& mat, const double threshold) {
  // 1. Perform Column Pivoted Householder QR (Rank-Revealing)
  Eigen::ColPivHouseholderQR<Matrix> qr(mat);
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
  result.qr_decomp = qr;
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

Matrix RankInflation::get_analytic_center(const Matrix& X_0) const {
  // Initialize
  double delta = params_.delta_ac;
  bool converged = false;
  int n_iter = 0;
  Matrix X = X_0;
  double f_val = get_analytic_center_objective(X);
  // Main loop
  while (n_iter < params_.max_iter_ac) {
    // Define perturbed version of X
    Matrix Z = X + Matrix::Identity(dim, dim) * delta;
    // Get system of equations
    auto [C, d, violation] = get_analytic_center_system(Z, X);
    // Solve with QR factorization
    Vector rhs;
    if (params_.reduce_violation_ac) {
      rhs = d + violation;
    } else {
      rhs = d;
    }
    QRResult result = get_soln_qr_dense(C.selfadjointView<Eigen::Upper>(), rhs,
                                        params_.qr_thresh_ac);
    // std::cout << "R diag AC: " << result.R_diagonal.transpose() << std::endl;
    // std::cout << "AC Solve Residual Norm: " << result.residual_norm
    //           << std::endl;
    // std::cout << "violation" << std::endl << violation.transpose() << std::endl;

    // Get step direction
    auto Aw_sp = build_wt_sum_constraints(result.solution);
    Matrix Aw = C_ * result.solution(m - 1) + Aw_sp;
    // Line search to find optimal step size
    double alpha = 1.0;
    double f_val_dec = 0.0;
    if (params_.enable_line_search_ac) {
      std::tie(alpha, f_val_dec) = analytic_center_backtrack(Z, Aw);
      f_val -= f_val_dec;
    }
    // Update step
    auto deltaX = Z - Z * Aw * Z;
    X.noalias() = X + alpha * deltaX;
    // Objective value
    if (!params_.enable_line_search_ac) {
      // Evaluate objective at new solution
      f_val = get_analytic_center_objective(X);
    }
    // Print results
    if (params_.verbose) {
      int sol_rank = get_rank(X, params_.tol_rank_sol);
      if (n_iter % 10 == 0) {
        std::printf("%6s %6s %18s %10s %8s %8s\n", "Iter", "SolRank",
                    "ViolationNorm", "StepNorm", "Alpha", "Obj Val.");
      }
      std::printf("%6d %6d %18.6e %10.6e %8.3e %8.3e\n", n_iter, sol_rank,
                  violation.norm(), deltaX.norm(), alpha, f_val);
    }
    // Increment
    n_iter++;
    // Stopping Condition
    if (deltaX.norm() < params_.tol_step_norm_ac) break;
  }
  return X;
}

std::tuple<Matrix, Vector, Vector> RankInflation::get_analytic_center_system(
    const Matrix& Z, const Matrix& X) const {
  // Construct AZ matrices and rhs of linear system
  Vector d(m);
  Vector violation(m);
  std::vector<Matrix> AZ;
  for (int i = 0; i < m; i++) {
    // compute violation
    if (i < A_.size()) {
      AZ.push_back(A_[i].selfadjointView<Eigen::Upper>() * Z);
      violation(i) =
          (A_[i].selfadjointView<Eigen::Upper>() * X).trace() - b_[i];
    } else {
      AZ.push_back(C_ * Z);
      violation(i) = (C_ * X).trace() - rho_;
    }
    d(i) = AZ.back().trace();
  }
  // Construct the LHS matrix
  Matrix C(m, m);
  for (int i = 0; i < m; i++) {
    for (int j = i; j < m; j++) {
      C(i, j) = (AZ[i] * AZ[j]).trace();
    }
  }
  return {C, d, violation};
}

std::pair<double, double> RankInflation::analytic_center_backtrack(
    const Matrix& Z, const Matrix& Aw) const {
  // NOTE: Should make this function generic for any line search function
  //  Initial step size
  double alpha = params_.alpha_init;
  // Current objective and gradient
  auto [f, df] = analytic_center_line_search_func(Z, Aw);
  double df_0 = df(0.0);
  double f_val = f(0.0);
  // Backtracking parameters
  const double beta =
      params_.ln_search_red_factor;             // step size reduction factor
  const double c = params_.ln_search_suff_dec;  // sufficient decrease parameter
  // Backtracking loop
  while (true) {
    // Evaluate objective at new solution
    double f_val_new = f(alpha);
    // Check Armijo condition
    if (f_val_new <= f_val + c * alpha * df_0) {
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
  return {alpha, f(alpha)};
}

double RankInflation::analytic_center_bisect(const Matrix& Z,
                                             const Matrix& Aw) const {
  // Create the objective function and deriviative
  auto [f, df] = analytic_center_line_search_func(Z, Aw);
  // Bisection parameters
  double alpha_low = 0.0;
  double alpha_high = 1.0;
  double tol = params_.tol_bisect_ac;
  // Perform bisection line search
  return bisection_line_search(df, alpha_low, alpha_high, tol);
}

std::pair<ScalarFunc, ScalarFunc>
RankInflation::analytic_center_line_search_func(const Matrix& Z,
                                                const Matrix& Aw) const {
  // Cholesky decomposition of augmented solution
  Eigen::LLT<Matrix> cholZ(Z);
  if (cholZ.info() == Eigen::NumericalIssue) {
    throw std::runtime_error("Matrix Z is not positive definite.");
  }
  Matrix L = cholZ.matrixL();
  // Compute eigenvalues of L^T * Aw * L
  Matrix M = L.transpose() * Aw * L;
  Eigen::SelfAdjointEigenSolver<Matrix> es(M);
  if (es.info() != Eigen::Success) {
    throw std::runtime_error("Eigenvalue decomposition failed.");
  }
  Vector eigs = es.eigenvalues();

  // Define the line search functions
  auto f = [eigs](double alpha) {
    double val = 0.0;
    for (int i = 0; i < eigs.size(); i++) {
      val -= std::log(1.0 + alpha * (1 - eigs(i)));
    }
    if (std::isnan(val)) {
      val = std::numeric_limits<double>::infinity();
    }
    return val;
  };
  // Define the derivative function
  auto df = [eigs](double alpha) {
    double val = 0.0;
    for (int i = 0; i < eigs.size(); i++) {
      val -= (1 - eigs(i)) / (1.0 + alpha * (1 - eigs(i)));
    }
    return val;
  };

  return {f, df};
}

double bisection_line_search(const ScalarFunc& df, double alpha_low,
                             double alpha_high, double tol) {
  // Ensure that the upper bound is valid
  while (df(alpha_high) < 0) {
    alpha_low = alpha_high;
    alpha_high *= 2.0;  // expand search interval
  }
  // Bisection loop
  double alpha_mid;
  while ((alpha_high - alpha_low) > tol) {
    alpha_mid = 0.5 * (alpha_low + alpha_high);
    if (df(alpha_mid) > 0) {
      alpha_high = alpha_mid;
    } else {
      alpha_low = alpha_mid;
    }
  }
  return 0.5 * (alpha_low + alpha_high);
}

Matrix recover_lowrank_factor(const Matrix& A) {
  // Use LDLT decomposition to get low-rank factors
  // NOTE: LDLT is used because it is stable for semi-definite matrices and
  // will effectively terminate when it encounters a max pivot that is
  // numerically zero.
  Eigen::LDLT<Matrix> ldlt(A);
  Matrix L = ldlt.matrixL();
  Vector D = ldlt.vectorD();
  // Determine rank based on positive pivots
  int rank = 0;
  for (int i = 0; i < D.size(); i++) {
    if (D(i) > 1e-18) {
      rank++;
    } else {
      break;
    }
  }
  // Build low-rank factor
  Matrix Y(A.rows(), rank);
  for (int i = 0; i < rank; i++) {
    Y.col(i) = L.col(i) * std::sqrt(D(i));
  }
  // Unpivot the factors
  auto P = ldlt.transpositionsP();
  Y = P.transpose() * Y;

  return Y;
}

}  // namespace SDPTools
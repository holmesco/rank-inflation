#include "analytic_center.hpp"

namespace SDPTools {

AnalyticCenter::AnalyticCenter(
    const Matrix& C, double rho,
    const std::vector<Eigen::SparseMatrix<double>>& A,
    const std::vector<double>& b, AnalyticCenterParams params)
    : C_(C), A_(A), rho_(rho), b_(b), params_(params) {
  // dimension of the SDP
  dim = C.rows();
  // number of constraints to enforce during inflation
  m = A.size() + 1;
}

Vector AnalyticCenter::eval_constraints(const Matrix& X) const {
  // Loop through constraints, evaluating gradient and constraint value
  Vector result(m);
  for (int i = 0; i < m; i++) {
    if (i < A_.size()) {
      // Constraints
      // NOTE: Converting to DENSE here. Optimize this later
      result(i) = (A_[i].selfadjointView<Eigen::Upper>() * X).trace() - b_[i];
    } else {
      // Cost "constraint"
      result(i) = (C_ * X).trace() - rho_;
    }
  }
  return result;
}

SpMatrix AnalyticCenter::build_adjoint(const Vector& coeffs, double tol) const {
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

Matrix AnalyticCenter::build_certificate_from_dual(
    const Vector& multipliers) const {
  // build weighted sum of constraint matrices with multiplier values
  auto f_A = build_adjoint(multipliers, 0.0);
  // add cost term if required
  Matrix H;
  H = C_ + f_A;

  return H;
}

std::pair<double, double> AnalyticCenter::check_certificate(
    const Matrix& H, const Matrix& Y) const {
  // Evaluate the stationarity condition
  double first_order_norm = (Y.transpose() * H * Y).norm();
  // Check Eigenvalues
  // Use SelfAdjointEigenSolver for symmetric matrices
  Eigen::SelfAdjointEigenSolver<Matrix> es(H);
  double min_eig = es.eigenvalues().minCoeff();

  return {min_eig, first_order_norm};
}

AnalyticCenterResult AnalyticCenter::certify(const Matrix& Y_0,
                                             double delta) const {
  // Run analtyic center solve
  auto [X, mult_scaled] = get_analytic_center(Y_0, delta, 0.0);
  auto H = build_certificate_from_dual(mult_scaled);
  // Check certificate at the final solution
  auto [min_eig, complementarity] = check_certificate(H, Y_0);
  // Return the final solution, multipliers and certificate information
  AnalyticCenterResult result;
  result.X = X;
  result.multipliers = mult_scaled;
  result.H = H;
  result.violation = eval_constraints(X);
  result.certified = (min_eig >= -params_.tol_cert_psd) &&
                     (complementarity <= params_.tol_cert_first_order);
  result.min_eig = min_eig;
  result.complementarity = complementarity;
  return result;
}

Matrix AnalyticCenter::get_analytic_center_adaptive(const Matrix& X_0) const {
  double delta = params_.delta_init_ac;
  auto X = X_0;
  Vector multipliers(m);
  while (delta >= params_.delta_min_ac) {
    // Compute analytic center for current delta
    std::tie(X, multipliers) = get_analytic_center(X, delta);
    // Update delta
    delta *= params_.adapt_factor_ac;
    if (params_.verbose) {
      std::cout << "-------------------------------- " << std::endl;
      std::cout << "Adapting Delta: " << delta << std::endl;
      std::cout << "-------------------------------- " << std::endl;
    }
  }
  return X;
}

std::pair<Matrix, Vector> AnalyticCenter::get_analytic_center(
    const Matrix& Y_0, double delta_obj, double delta_constraint) const {
  // Initialize
  int n_iter = 0;
  Matrix X = Y_0 * Y_0.transpose();
  // Define perturbed version of X
  Matrix Z = X + Matrix::Identity(dim, dim) * delta_obj;
  double f_val = get_analytic_center_objective(X, delta_obj);
  Vector mult_scaled(m - 1);
  // Optimality certificate
  Matrix H;
  double complementarity = std::nan("");
  double min_eig = std::nan("");
  // Main loop
  while (n_iter < params_.max_iter_ac) {
    // Get system of equations
    auto [multipliers, violation] =
        solve_analytic_center_system(Z, X, delta_constraint);
    // compute scaled multipliers for certificate checking
    mult_scaled = multipliers.segment(0, m - 1);
    if (std::abs(multipliers(m - 1)) > 0) {
      mult_scaled /= multipliers(m - 1);
    }
    // Get step direction
    auto Aw_sp = build_adjoint(multipliers);
    Matrix Aw = C_ * multipliers(m - 1) + Aw_sp;
    // Line search to find optimal step size
    double alpha = 1.0;
    double f_val_dec = 0.0;
    if (params_.enable_line_search_ac) {
      std::tie(alpha, f_val_dec) = analytic_center_backtrack(Z, Aw);
      f_val -= f_val_dec;
    }
    // Update step
    auto deltaX = Z - Z * Aw * Z;
    X.noalias() += alpha * deltaX;
    Z.noalias() += alpha * deltaX;
    // Certificate Checking (Early stopping condition if the certificate is PSD)
    if (params_.check_cert_ac) {
      // Build certificate matrix
      H = build_certificate_from_dual(mult_scaled);
      // Check complementarity condition for first order optimality
      complementarity = (Y_0.transpose() * H * Y_0).norm();
      if (complementarity <= params_.tol_cert_first_order) {
        // if first order condition is satisfied, check eigenvalues of
        // certificate matrix
        Eigen::SelfAdjointEigenSolver<Matrix> es(H);
        min_eig = es.eigenvalues().minCoeff();
        if (min_eig >= -params_.tol_cert_psd) {
          if (params_.verbose) {
            std::cout << "Certificate Found! Stopping centering." << std::endl;
          }
          break;
        }
      }
    }
    // Print results
    if (params_.verbose) {
      // Objective value
      if (!params_.enable_line_search_ac) {
        // Evaluate objective at new solution
        f_val = get_analytic_center_objective(X, delta_obj);
      }
      // get rank of solution
      int sol_rank = get_rank(X, params_.tol_rank_sol);
      if (n_iter % 10 == 0) {
        std::printf("%6s %6s %12s %12s %12s %12s %8s\n", "Iter", "SolRank",
                    "ViolNorm", "StepNorm", "Complement.", "MinEig",
                    "Obj Val.");
      }
      std::printf("%6d %6d %12.6e %12.6e %12.6e %12.6e %8.3e\n", n_iter,
                  sol_rank, violation.norm(), deltaX.norm(), complementarity,
                  min_eig, f_val);
    }
    // Increment
    n_iter++;
    // Stopping Condition
    if (deltaX.norm() < params_.tol_step_norm_ac) break;
  }

  return {X, mult_scaled};
}

std::pair<Vector, Vector> AnalyticCenter::solve_analytic_center_system(
    const Matrix& Z, const Matrix& X, double delta_constraint) const {
  // Construct AZ matrices, violation vector and d vector
  Vector d(m);
  Vector violation(m);
  std::vector<Matrix> AZ;
  std::vector<double> A_trace;
  for (int i = 0; i < m; i++) {
    // compute violation
    if (i < A_.size()) {
      AZ.push_back(A_[i].selfadjointView<Eigen::Upper>() * Z);
      violation(i) =
          (A_[i].selfadjointView<Eigen::Upper>() * X).trace() - b_[i];
      A_trace.push_back(A_[i].diagonal().sum());
    } else {
      AZ.push_back(C_ * Z);
      violation(i) = (C_ * X).trace() - rho_;
      A_trace.push_back(C_.diagonal().sum());
    }
    d(i) = AZ.back().trace();
    // add constraint perturbation
    if (delta_constraint > 0.0) {
      d(i) -= delta_constraint * A_trace.back();
    }
  }
  // Construct the LHS matrix
  Matrix H(m, m);
  for (int i = 0; i < m; i++) {
    for (int j = i; j < m; j++) {
      H(i, j) = (AZ[i] * AZ[j]).trace();
    }
  }

  // Construct the RHS vector
  Vector rhs;
  if (params_.reduce_violation_ac) {
    rhs = d + violation;
  } else {
    rhs = d;
  }
  // Solve the linear equation with LDLT decomposition (includes pivoting by
  // default) Note: This method is preferred because the system is PSD, but
  // can be ill-conditioned, especially in early iterations
  Eigen::LDLT<Eigen::MatrixXd> ldlt(H.selfadjointView<Eigen::Upper>());
  // Check for success (critical for rank-deficient cases)
  if (ldlt.info() == Eigen::NumericalIssue || !ldlt.isPositive()) {
    std::cout << "The matrix is not PSD or has severe numerical issues."
              << std::endl;
  }
  Vector multipliers = ldlt.solve(rhs);

  return {multipliers, violation};
}

std::pair<double, double> AnalyticCenter::analytic_center_backtrack(
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

double AnalyticCenter::analytic_center_bisect(const Matrix& Z,
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
AnalyticCenter::analytic_center_line_search_func(const Matrix& Z,
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
}  // namespace SDPTools
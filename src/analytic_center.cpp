#include "rank_inflation.hpp"

namespace SDPTools {

Matrix RankInflation::get_analytic_center_adaptive(const Matrix& X_0) const {
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

std::pair<Matrix, Vector> RankInflation::get_analytic_center(
    const Matrix& X_0, double delta_obj, double delta_constraint) const {
  // Initialize
  int n_iter = 0;
  Matrix X = X_0;
  double f_val = get_analytic_center_objective(X, delta_obj);
  Vector mult_scaled(m - 1);
  // Main loop
  while (n_iter < params_.max_iter_ac) {
    // Define perturbed version of X
    Matrix Z = X + Matrix::Identity(dim, dim) * delta_obj;
    // Get system of equations
    Vector multipliers(m);
    Vector violation(m);
    std::tie(multipliers, violation) =
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
    X.noalias() = X + alpha * deltaX;
    // Certificate Checking (Early stopping condition if the certificate is PSD)
    if (params_.check_cert_ac) {
      auto H = build_certificate_from_dual(mult_scaled);
      auto [min_eig, first_ord_cond] = check_certificate(H, X);
      if (params_.verbose) {
        std::cout << "Minimum Eigenvalue of Certificate: " << min_eig
                  << std::endl;
        std::cout << "First Order Condition Norm: " << first_ord_cond
                  << std::endl;
      }
      if (min_eig >= -params_.tol_cert_psd &&
          first_ord_cond <= params_.tol_cert_first_order) {
        if (params_.verbose) {
          std::cout
              << "Certificate Found! Stopping centering."
              << std::endl;
        }
        break;
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

  return {X, mult_scaled};
}

std::pair<Vector, Vector> RankInflation::solve_analytic_center_system(
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
}  // namespace SDPTools
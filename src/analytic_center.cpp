#include "analytic_center.hpp"

namespace RankTools {

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
                                             double delta_init) const {
  // Run analtyic center solve
  auto [X, mult_scaled] = get_analytic_center(Y_0, delta_init);
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

std::pair<Matrix, Vector> AnalyticCenter::get_analytic_center(
    const Matrix& Y_0, double delta_init) const {
  // Initialize
  int n_iter = 0;
  double delta = delta_init;
  Matrix Z = Y_0 * Y_0.transpose() + Matrix::Identity(dim, dim) * delta;
  double f_val = logdet(Z);
  Vector mult_scaled(m - 1);
  // Optimality certificate
  Matrix H;
  double complementarity = std::nan("");
  double min_eig = std::nan("");
  double barrier_param = std::nan("");
  // Main loop
  while (n_iter < params_.max_iter) {
    // Get system of equations
    auto [multipliers, violation] = solve_analytic_center_system(Z, delta);

    // get the barrier parameter value
    barrier_param = multipliers(m - 1);
    if (barrier_param <= 0) {
      std::cerr << "Warning: Barrier parameter is non-positive: " +
                       std::to_string(barrier_param)
                << std::endl;
    }
    // compute scaled multipliers for certificate checking
    mult_scaled = multipliers.segment(0, m - 1) / barrier_param;

    // Get step direction
    auto Aw_sp = build_adjoint(multipliers);
    Matrix Aw = C_ * multipliers(m - 1) + Aw_sp;
    auto deltaZ = Z - Z * Aw * Z;
    // Line search to find step that ensures PSDness of the solution
    // NOTE: Could replace with exact line search based on determinant increase,
    // but this backtracking
    double alpha = 1.0;
    if (params_.enable_line_search) {
      alpha = line_search_psd(Z, deltaZ);
    } else {
      Z.noalias() += alpha * deltaZ;
    }
    // Certificate Checking (Early stopping condition if the certificate is PSD)
    if (params_.check_cert) {
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
    // Perturbation update for next iteration (if enabled)
    if (params_.adaptive_perturb) {
      if (alpha < params_.delta_inc_step_max) {
        delta = delta * params_.delta_inc;
      } else if (alpha > params_.delta_dec_step_min) {
        delta = delta * params_.delta_dec;
      }
      // Ensure delta does not go below minimum threshold
      delta = std::max(delta, params_.delta_min);
    }

    n_iter++;
    // Print results
    if (params_.verbose) {
      // Objective value
      f_val = logdet(Z);
      // get rank of solution
      int sol_rank = get_rank(Z, params_.tol_rank_sol);
      if (n_iter % 10 == 1) {
        std::printf("%6s %6s %12s %12s %12s %12s %12s %12s %12s %8s\n", "Iter",
                    "SolRank", "ViolNorm", "StepNorm", "Complement.", "MinEig",
                    "BarrParam", "Alpha", "Delta", "Obj Val.");
      }
      std::printf(
          "%6d %6d %12.6e %12.6e %12.6e %12.6e %12.6e %12.6e %12.6e %8.3e\n",
          n_iter, sol_rank, violation.norm(), deltaZ.norm(), complementarity,
          min_eig, barrier_param, alpha, delta, f_val);
    }
    // Check final stopping condition
    if (deltaZ.norm() < params_.tol_step_norm)
      if (params_.adaptive_perturb) {
        if (delta <= params_.delta_min) {
          if (params_.verbose) {
            std::cout << "Step norm below threshold and minimum perturbation "
                         "reached. Stopping centering."
                      << std::endl;
          }
          break;
        }
      } else {
        if (params_.verbose) {
          std::cout << "Step norm below threshold. Stopping centering."
                    << std::endl;
        }
        break;
      }
  }

  return {Z, mult_scaled};
}

std::pair<Vector, Vector> AnalyticCenter::solve_analytic_center_system(
    const Matrix& Z, double delta) const {
  // Construct AZ matrices, violation vector and d vector
  Vector d(m);
  Vector violation(m);
  std::vector<Matrix> AZ;
  // TODO trace value should be precomputed for efficiency
  std::vector<double> A_trace;
  for (int i = 0; i < m; i++) {
    // compute violation
    if (i < A_.size()) {
      AZ.push_back(A_[i].selfadjointView<Eigen::Upper>() * Z);
      A_trace.push_back(A_[i].diagonal().sum());
      violation(i) = (A_[i].selfadjointView<Eigen::Upper>() * Z).trace() -
                     b_[i] - delta * A_trace[i];
    } else {
      AZ.push_back(C_ * Z);
      A_trace.push_back(C_.diagonal().sum());
      violation(i) = (C_ * Z).trace() - rho_ - delta * A_trace[i];
    }
    // "perturbed" RHS of the linera system
    d(i) = AZ[i].trace();
    if (params_.reduce_violation) {
      d(i) += violation(i);
    }
  }

  // Construct the LHS matrix
  Matrix H(m, m);
  for (int i = 0; i < m; i++) {
    for (int j = i; j < m; j++) {
      H(i, j) = (AZ[i] * AZ[j]).trace();
    }
  }

  // Solve the linear equation with LDLT decomposition (includes pivoting by
  // default) Note: This method is preferred because the system is PSD, but
  // can be ill-conditioned, especially in early iterations.
  Eigen::LDLT<Eigen::MatrixXd> ldlt(H.selfadjointView<Eigen::Upper>());
  // Check for success (critical for rank-deficient cases)
  if (ldlt.info() == Eigen::NumericalIssue) {
    std::cout << "The matrix is has severe numerical issues." << std::endl;
  }
  if (ldlt.isPositive() == false) {
    std::cout << "The matrix is not positive semidefinite." << std::endl;
  }
  // Solve the linear system.
  Vector multipliers = ldlt.solve(d);
#ifdef DEBUG
  // print information about the linear system
  Eigen::SelfAdjointEigenSolver<Matrix> es_Z(Z);
  double min_eig_Z = es_Z.eigenvalues().minCoeff();
  std::cout << "Minimum eigenvalue of Z: " << min_eig_Z << std::endl;
  // Print the diagonal of the LDLT decomposition to check for small or negative
  // pivots
  Vector D = ldlt.vectorD();
  std::cout << "Diagonal of D in LDLT decomposition: " << D.transpose()
            << std::endl;
  // print residual of the linear system solution
  Vector residual = H.selfadjointView<Eigen::Upper>() * multipliers - d;
  std::cout << "Residual norm of linear system solution: " << residual.norm()
            << std::endl;
#endif

  return {multipliers, violation};
}

double AnalyticCenter::line_search_psd(Matrix& Z, const Matrix& dZ) const {
  // NOTE: Should make this function generic for any line search function
  //  Initial step size
  double alpha = params_.alpha_init;
  // Backtracking parameters
  const double beta =
      params_.ln_search_red_factor;  // step size reduction factor
  // Backtracking loop
  Matrix Z_new = Z + alpha * dZ;
  Eigen::LDLT<Eigen::MatrixXd> ldlt(Z_new);
  while (!ldlt.isPositive()) {
    if (ldlt.info() == Eigen::NumericalIssue) {
      std::cout << "LINESEARCH: The matrix is has severe numerical issues."
                << std::endl;
    }
    // Reduce step size
    alpha *= beta;
    // Prevent too small step sizes
    if (alpha < params_.alpha_min) {
      alpha = params_.alpha_min;
      break;  // Stop if step size is too small
    }
    Z_new = Z + alpha * dZ;
    ldlt.compute(Z_new);
  }
  // update Z with the new value that ensures PSDness
  Z = Z_new;
  return alpha;
}

std::pair<double, double> AnalyticCenter::line_search_det(
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
  double tol = params_.tol_bisect;
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
}  // namespace RankTools
#include "analytic_center.hpp"

namespace RankTools {

AnalyticCenter::AnalyticCenter(
    const Matrix& C, double rho,
    const std::vector<Eigen::SparseMatrix<double>>& A,
    const std::vector<double>& b, AnalyticCenterParams params)
    : C_(C),
      A_(A),
      rho_(rho),
      b_(b),
      params_(params),
      dim(C.rows()),
      m(A.size() + 1),
      lr_solver(nullptr) {}

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

std::pair<double, double> AnalyticCenter::eval_certificate(
    const Matrix& H, const Matrix& Y) const {
  // Evaluate the stationarity condition
  double complementarity = (Y.transpose() * H * Y).norm();
  // Check Eigenvalues
  // Use SelfAdjointEigenSolver for symmetric matrices
  Eigen::SelfAdjointEigenSolver<Matrix> es(H);
  double min_eig = es.eigenvalues().minCoeff();

  return {min_eig, complementarity};
}

std::pair<double, double> AnalyticCenter::check_certificate(
    const Matrix& H, const Matrix& Y) const {
  // Evaluate the stationarity condition
  double complementarity = (Y.transpose() * H * Y).norm();
  // PSD Check
  Eigen::LLT<Matrix> llt(H + Matrix::Identity(H.rows(), H.cols()) *
                                 params_.tol_cert_psd);
  bool psd = (llt.info() == Eigen::Success);

  return {psd, complementarity};
}

AnalyticCenterResult AnalyticCenter::certify(const Matrix& Y_0,
                                             double delta_init) const {
  // Run analtyic center solve
  auto [X, mult_scaled] = get_analytic_center(Y_0, delta_init);
  auto H = build_certificate_from_dual(mult_scaled);
  // Check certificate at the final solution
  auto [min_eig, complementarity] = eval_certificate(H, Y_0);
  // Return the final solution, multipliers and certificate information
  AnalyticCenterResult result;
  result.X = X;
  result.multipliers = mult_scaled;
  result.H = H;
  result.violation = eval_constraints(X);
  result.certified = (min_eig >= -params_.tol_cert_psd) &&
                     (complementarity <= params_.tol_cert_complementarity);
  result.min_eig = min_eig;
  result.complementarity = complementarity;
  return result;
}

std::pair<Matrix, Vector> AnalyticCenter::get_analytic_center(
    const Matrix& Y_0, double delta_init) const {
  // Initialize
  int n_iter = 0;
  double delta = delta_init;
  Matrix Z = Y_0 * Y_0.transpose();
  // store rank of candidate solution
  rank_init = Y_0.cols();
  auto [alpha, L] = line_search_factorization(
      Z, Matrix::Identity(dim, dim) * delta);  // Initial line search to ensure
                                               // PSDness of the starting point
  Vector mult_scaled(m - 1);
  // Optimality certificate
  Matrix H;
  double complementarity = std::nan("");
  double min_eig = std::nan("");
  double barrier_param = std::nan("");
  double cent_metric = std::nan("");
  double angle = std::nan("");
  // Main loop
  while (n_iter < params_.max_iter) {
    // Get system of equations
    auto [multipliers, violation] = get_multipliers(Z, Y_0, delta);

    // get the barrier parameter value
    barrier_param = 1 / multipliers(m - 1);
    // compute scaled multipliers for certificate checking
    mult_scaled = multipliers.segment(0, m - 1) * barrier_param;

    // Dual matrix (adjoint of constraints with multipliers + cost term )
    // TODO: This is not the most efficient way to implement this, should start
    // with dense and add sparse to it.
    auto adjoint_sp = build_adjoint(multipliers);
    Matrix adjoint = C_ * multipliers(m - 1) + adjoint_sp;
    if (params_.rescale_lin_sys) {
      adjoint = adjoint / delta;
    }
    // Get Newton step direction towards analytic center
    auto deltaZ = Z - Z * adjoint * Z;
    // Line search to find step that ensures PSDness of the solution
    // NOTE: Could replace with exact line search based on determinant increase,
    // but this backtracking
    alpha = 1.0;
    std::tie(alpha, L) = line_search_factorization(Z, deltaZ);
    // Compute centrality metric from He et al. 1997
    // When this is below 1, the dual matrix is guaranteed to be PSD, so this
    // can be used as an early stopping condition for the centering iterations.
    cent_metric =
        (L.transpose() * adjoint * L - Matrix::Identity(dim, dim)).norm();
    // Compute the angle between the current solution and the initial solution
    // as a measure of deviation from the initial solution for early stopping.
    double cos_angle = (Y_0.transpose() * Z * Y_0).trace() /
                       (Y_0.transpose() * Y_0).norm() / Z.norm();
    angle = std::acos(cos_angle);
    // Print update
    n_iter++;
    if (params_.verbose) {
      if (n_iter == 1) {
        std::cout << "Starting Analytic Center Iterations..." << std::endl;
        std::cout << "Running with Linear Solver: "
                  << print_solver(params_.lin_solver) << std::endl;
      }
      if (n_iter % 10 == 1) {
        std::printf("%6s %12s %12s %12s %12s %12s %12s %12s %12s\n", "Iter",
                    "ViolNorm", "StepNorm", "Complement.", "BarrParam", "Alpha",
                    "Delta", "CentMetric", "Angle (rad)");
      }
      std::printf(
          "%6d %12.6e %12.6e %12.6e %12.6e %12.6e %12.6e %12.6e %12.6e\n",
          n_iter, violation.norm(), deltaZ.norm(), complementarity,
          barrier_param, alpha, delta, cent_metric, angle);
    }

    // Certificate Checking (Early stopping condition if the certificate is PSD)
    if (params_.early_stop_cert) {
      if (params_.use_cert_centrality_metric) {
        if (cent_metric <= 1 + params_.tol_cert_centrality) {
          if (params_.verbose) {
            std::cout << "Certificate Centrality Metric below threshold! "
                         "Stopping centering."
                      << std::endl;
          }
          break;
        }
      } else {
        // Build certificate matrix
        H = build_certificate_from_dual(mult_scaled);
        // Check complementarity condition for first order optimality
        complementarity = (Y_0.transpose() * H * Y_0).norm();
        if (complementarity <= params_.tol_cert_complementarity) {
          // Use a cholesky decomposition for a quick check of PSDness.
          Eigen::LLT<Matrix> llt(H + Matrix::Identity(H.rows(), H.cols()) *
                                         params_.tol_cert_psd);
          if (llt.info() == Eigen::Success) {
            if (params_.verbose) {
              std::cout << "Certificate Found! Stopping centering."
                        << std::endl;
            }
            break;
          }
        }
      }
    }
    // Early stop for deviation from the solution
    if (params_.early_stop_angle) {
      if (angle > params_.max_angle) {
        if (params_.verbose) {
          std::cout
              << "Solution has deviated by an angle of " << angle
              << " radians from the initial solution, which is above the "
              << "threshold of " << params_.max_angle
              << " radians. "
                 "Stopping centering. Candidate was not at the analytic center."
              << std::endl;
        }
        break;
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

AnalyticCenter::LinSysData AnalyticCenter::build_ac_system(const Matrix& X,
                                                           double delta) const {
  // Initialize system data
  LinSysData sys(X, dim, m);
  const int a_size = static_cast<int>(A_.size());
#ifdef RANKTOOLS_PARALLEL
#pragma omp parallel for schedule(dynamic)
#endif
  for (int i = 0; i < m; i++) {
    double val;
    if (i < a_size) {
      sys.AX[i] = A_[i].selfadjointView<Eigen::Upper>() * X;
      sys.A_trace[i] = A_[i].diagonal().sum();
      val = b_[i] + delta * sys.A_trace[i];
    } else {
      sys.AX[i] = C_.selfadjointView<Eigen::Upper>() * X;
      sys.A_trace[i] = C_.diagonal().sum();
      val = rho_ + delta * sys.A_trace[i];
    }
    // Set RHS of the system
    sys.d(i) = sys.AX[i].trace();
    // compute violation
    sys.violation(i) = sys.d(i) - val;
    // adjusted rhs to include violation if enabled
    if (params_.reduce_violation) {
      sys.d(i) += sys.violation(i);
    }
  }
  // Rescaling for linear system
  const double scale = params_.rescale_lin_sys ? (1.0 / delta) : 1.0;
  // Only build the LHS matrix if not using matrix-free solver.

  if (params_.lin_solver == LinearSolverType::MFCG_DP ||
      params_.lin_solver == LinearSolverType::MFCG_LRP) {
    // If using matrix-free solver, build the matrix-free operator for the LHS
    sys.B_mf = std::make_unique<MultiplierLinSys>(X, A_, C_, sys.AX, scale);
  } else {
    sys.B.setZero();
    for (int i = 0; i < m; i++) {
      for (int j = i; j < m; j++) {
        sys.B(i, j) = scale * (sys.AX[i] * sys.AX[j]).trace();
      }
    }
  }
  // Return system information and metrics
  return sys;
}

Vector AnalyticCenter::solve_ac_system(const LinSysData& sys,
                                       const Matrix& Y_0) const {
  Vector multipliers(m);

  if (params_.lin_solver == LinearSolverType::CG) {
    // Solve the linear system with Conjugate Gradient (CG) method
    // Note: CG is more scalable for large problems, but may require more
    // careful tuning of parameters and preconditioning for convergence,
    // especially in early iterations when the system can be ill-conditioned.
    Eigen::ConjugateGradient<Matrix, Eigen::Upper> cg;
    // Set paramters
    cg.setMaxIterations(
        params_.lin_solve_max_iter);  // Set a maximum number of iterations
    cg.setTolerance(params_.lin_solve_tol);  // Set a convergence tolerance
    cg.compute(sys.B);
    if (cg.info() != Eigen::Success) {
      std::cout << "Decomposition failed: " << cg.error() << std::endl;
    }

    // Use previous multipliers as initial guess if available to speed up
    // convergence
    if (prev_multipliers_.size() == m) {
      multipliers = cg.solveWithGuess(sys.d, prev_multipliers_);
    } else {
      multipliers = cg.solve(sys.d);
    }
    prev_multipliers_ = multipliers;  // Store the multipliers for the next
                                      // iteration's initial guess
    if (cg.info() != Eigen::Success) {
      std::cout << "Solving failed: " << cg.error() << std::endl;
    }
  } else if (params_.lin_solver == LinearSolverType::MFCG_DP) {
    // Solve the linear system with Matrix-Free Conjugate Gradient (MFCG) method
    // with diagonal preconditioner.
    MultiplierLinSys& lin_op = *(sys.B_mf);
    Eigen::ConjugateGradient<MultiplierLinSys, Eigen::Upper | Eigen::Lower,
                             MultiplierDiagPreconditioner>
        solver;
    // Set solve parameters
    solver.setMaxIterations(
        params_.lin_solve_max_iter);  // Set a maximum number of iterations
    solver.setTolerance(params_.lin_solve_tol);  // Set a convergence tolerance

    solver.compute(lin_op);
    if (solver.info() != Eigen::Success) {
      std::cout << "Decomposition failed: " << solver.error() << std::endl;
    }

    // Use previous multipliers as initial guess if available to speed up
    // convergence
    if (prev_multipliers_.size() == m) {
      multipliers = solver.solveWithGuess(sys.d, prev_multipliers_);
    } else {
      multipliers = solver.solve(sys.d);
    }
    prev_multipliers_ = multipliers;  // Store the multipliers for the next
                                      // iteration's initial guess
    if (solver.info() != Eigen::Success) {
      std::cout << "Solving failed: " << solver.error() << std::endl;
    }
  } else if (params_.lin_solver == LinearSolverType::MFCG_LRP) {
    // Solve the linear system with Matrix-Free Conjugate Gradient (MFCG) method
    // with low rank preconditioner.
    MultiplierLinSys& lin_op = *(sys.B_mf);
    // To avoid repeatedly initializing the preconditioner in each iteration, we
    // maintain a stored instance of the solver.
    if (!lr_solver) {
      lr_solver = std::make_unique<Eigen::ConjugateGradient<
          MultiplierLinSys, Eigen::Upper | Eigen::Lower, LowRankPrecond>>();
      // Initialize the preconditioner
      LowRankPrecond& lr_precond = lr_solver->preconditioner();
      lr_precond.initialize(Y_0, A_, C_, params_.lin_solve_precond_perturb);
    }
    auto& solver = *lr_solver;  // convenience definition
    // Set solve parameters
    solver.setMaxIterations(
        params_.lin_solve_max_iter);  // Set a maximum number of iterations
    solver.setTolerance(params_.lin_solve_tol);  // Set a convergence tolerance
    // Call compute
    solver.compute(lin_op);
    if (solver.info() != Eigen::Success) {
      std::cout << "Solver Failed: " << solver.error() << std::endl;
    }
    // Use previous multipliers as initial guess if available to speed up
    // convergence
    if (prev_multipliers_.size() == m) {
      multipliers = solver.solveWithGuess(sys.d, prev_multipliers_);
    } else {
      multipliers = solver.solve(sys.d);
    }
    prev_multipliers_ = multipliers;  // Store the multipliers for the next
                                      // iteration's initial guess
    if (solver.info() != Eigen::Success) {
      std::cout << "Solving failed: " << solver.error() << std::endl;
    }
  } else if (params_.lin_solver == LinearSolverType::LDLT) {
    // Solve the linear equation with LDLT decomposition (includes pivoting by
    // default) Note: This method is preferred because the system is PSD, but
    // can be ill-conditioned, especially in early iterations.
    Eigen::LDLT<Eigen::MatrixXd> solver(sys.B.selfadjointView<Eigen::Upper>());
    // Check for success (critical for rank-deficient cases)
    if (params_.verbose) {
      if (solver.info() == Eigen::NumericalIssue) {
        std::cout << "LDLT SOLVE: The matrix is has severe numerical issues."
                  << std::endl;
      }
      if (solver.isPositive() == false) {
        std::cout << "LDLT SOLVE: The matrix is not positive semidefinite."
                  << std::endl;
      }
    }
    // Solve the linear system.
    multipliers = solver.solve(sys.d);
  }
#ifdef DEBUG
  // print information about the linear system
  Eigen::SelfAdjointEigenSolver<Matrix> es_Z_dbg(
      sys.B.selfadjointView<Eigen::Upper>());
  double min_eig_B = es_Z_dbg.eigenvalues().minCoeff();
  std::cout << "Minimum eigenvalue of B: " << min_eig_B << std::endl;
  // print residual of the linear system solution
  Vector residual = sys.B.selfadjointView<Eigen::Upper>() * multipliers - sys.d;
  std::cout << "Residual norm of linear system solution: " << residual.norm()
            << std::endl;
#endif

  return multipliers;
}

std::pair<Vector, Vector> AnalyticCenter::get_multipliers(const Matrix& Z,
                                                          const Matrix& Y_0,
                                                          double delta) const {
  // Build the system of equations for the current solution
#ifdef TIMING
  // start a timer for building the system
  auto start = std::chrono::high_resolution_clock::now();
#endif
  auto sys = build_ac_system(Z, delta);

#ifdef TIMING
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken to build AC system: " << elapsed.count()
            << " seconds" << std::endl;
#endif
  // solve the system to get the multipliers
  Vector multipliers = solve_ac_system(sys, Y_0);
#ifdef TIMING
  // print time taken to solve the system
  auto end_solve = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_solve = end_solve - end;
  std::cout << "Time taken to solve AC system: " << elapsed_solve.count()
            << " seconds" << std::endl;
#endif
#ifdef DEBUG
  // print information about Z
  Eigen::SelfAdjointEigenSolver<Matrix> es_Z(Z);
  double min_eig_Z = es_Z.eigenvalues().minCoeff();
  std::cout << "Minimum eigenvalue of Z: " << min_eig_Z << std::endl;
#endif

  return {multipliers, sys.violation};
}

std::pair<double, Matrix> AnalyticCenter::line_search_factorization(
    Matrix& Z, const Matrix& dZ) const {
  //  Initial step size
  double alpha = params_.alpha_init;
  // Backtracking parameters
  const double beta =
      params_.ln_search_red_factor;  // step size reduction factor
  // Backtracking loop
  Matrix Z_new = Z + alpha * dZ;
  Eigen::LDLT<Eigen::MatrixXd> solver(Z_new);
  while (!solver.isPositive()) {
    if (solver.info() == Eigen::NumericalIssue) {
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
    solver.compute(Z_new);
  }
  // update Z with the new value that ensures PSDness
  Z = Z_new;
  // Return the Cholesky factorization of the new Z for use in the next
  // iteration's linear system
  Eigen::MatrixXd L_chol = solver.matrixL();
  L_chol.noalias() = L_chol * solver.vectorD().cwiseSqrt().asDiagonal();
  L_chol.noalias() = solver.transpositionsP().transpose() * L_chol;

  return {alpha, L_chol};
}

}  // namespace RankTools
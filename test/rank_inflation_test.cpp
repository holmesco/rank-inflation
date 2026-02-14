/*
c++ tests for rank inflation
*/
#include "test_harness.hpp"
#include "interior_point_sdp.hpp"


using namespace SDPTools;

// Fixture Class
class InflationParamTest : public ::testing::TestWithParam<SDPTestProblem> {};


// ------------------  TESTS -----------------------
// Test constraint evaluation and gradient function
TEST_P(InflationParamTest, EvalFuncAndGrad) {
  const auto& sdp = GetParam();
  // parameters
  RankInflateParams params;
  params.verbose = true;
  params.max_sol_rank = 2;
  // generate problem
  auto problem = sdp.make(params);
  // Test vector at actual solution
  Matrix Y = sdp.make_solution(params.max_sol_rank);
  auto Jac = Matrix(problem.m, problem.params_.max_sol_rank * sdp.dim);
  // Call evaluation function
  auto output = problem.eval_constraints(Y, Jac);
  // evaluation and gradient should be near zero
  // std::cout << "Evaluation: " << std::endl << output << std::endl;
  const double tol = 1e-6;
  ASSERT_EQ(output.size(), problem.m);
  for (int i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output(i), 0.0, tol) << "constraint " << i;
  }
  // Perturb solution and check Jacobian via finite differences
  Y += 0.01 * Matrix::Random(sdp.dim, 2);
  output = problem.eval_constraints(Y, Jac);
  std::cout << "Jac: " << std::endl << Jac << std::endl;
  // Numerical directional derivative check
  const double eps = 1e-8;
  int r = problem.params_.max_sol_rank;
  int vec_size = r * sdp.dim;
  const double deriv_tol = 1e-5;
  Eigen::MatrixXd ident = Eigen::MatrixXd::Identity(vec_size, vec_size);
  for (int i = 0; i < vec_size; ++i) {
    Eigen::VectorXd delta_vec = ident.col(i);
    Matrix Y2 = Y + eps * delta_vec.reshaped(sdp.dim, r);
    auto output2 = problem.eval_constraints(Y2);
    Eigen::VectorXd num_deriv = (output2 - output) / eps;
    Eigen::VectorXd anal_dir = Jac * delta_vec;
    for (int j = 0; j < problem.m; ++j) {
      EXPECT_NEAR(num_deriv(j), anal_dir(j), deriv_tol)
          << "directional derivative mismatch at constraint " << j
          << " for direction " << i;
    }
  }
}

// Test RRQR Solve
TEST_P(InflationParamTest, RRQRSolve) {
  const auto& sdp = GetParam();
  // parameters
  RankInflateParams params;
  params.verbose = true;
  params.max_sol_rank = 2;
  // generate problem
  auto problem = sdp.make(params);  // Test vector at actual solution
  Matrix Y = sdp.make_solution(params.max_sol_rank);
  auto Jac = Matrix(problem.m, problem.params_.max_sol_rank * sdp.dim);
  // Call evaluation function
  auto output = problem.eval_constraints(Y, Jac);
  // Apply QR decomposition
  QRResult soln = get_soln_qr_dense(Jac, -output, problem.params_.tol_rank_jac);
  // solution should be zero
  const double tol = 1e-6;
  ASSERT_EQ(soln.solution.size(), problem.params_.max_sol_rank * sdp.dim);
  for (int i = 0; i < soln.solution.size(); ++i) {
    EXPECT_NEAR(soln.solution(i), 0.0, tol) << "row " << i;
  }
  // Check for nullspace, if exists add to solution and verify small change in
  // output
  int nulldim = soln.nullspace_basis.cols();
  if (nulldim > 0) {
    std::cout << "Nullspace dimension: " << nulldim << ". Testing nullspace... "
              << std::endl;
    // Construct delta in the nullspace
    Eigen::VectorXd alpha =
        Eigen::VectorXd::Random(nulldim);  // values in [-1,1]
    double alpha_norm = alpha.norm();
    if (alpha_norm > 0) alpha /= alpha_norm;
    Matrix dY = (soln.nullspace_basis * alpha)
                    .reshaped(sdp.dim, problem.params_.max_sol_rank);
    // Add delta to solution
    Matrix Y_plus = Y + dY;
    // Evaluate constraints at new solution
    Vector output_Y_plus = problem.eval_constraints(Y_plus);
    Vector output_dY = problem.eval_constraints(dY);
    // Constraint value
    std::vector<double> vals(sdp.b.begin(), sdp.b.end());
    vals.push_back(sdp.rho);
    Vector constraint_val = Vector::Map(vals.data(), vals.size());
    // linear component of the new output
    Vector output_linear =
        output_Y_plus - output - (output_dY + constraint_val);
    // Should evaluate to zero
    for (int i = 0; i < output_linear.size(); ++i) {
      EXPECT_NEAR(output_linear(i), 0.0, tol) << "row " << i;
    }
  }
}

TEST_P(InflationParamTest, GradDescentRetraction) {
  const auto& sdp = GetParam();
  // parameters
  RankInflateParams params;
  params.verbose = true;
  params.max_sol_rank = 3;
  params.retraction_method = RetractionMethod::GradientDescent;
  params.max_iter = 20;
  params.alpha_min = 1e-12;
  // generate problem
  auto problem = sdp.make(params);
  // Get actual solution
  Matrix Y = sdp.make_solution(params.max_sol_rank);
  // Add perturbation to solution
  Eigen::MatrixXd perturb =
      Eigen::MatrixXd::Random(sdp.dim, params.max_sol_rank) * 1.0E-1;
  Y += perturb;
  // Get initial violation
  auto viol_init = problem.eval_constraints(Y);
  // Call inflation
  problem.retraction(Y);
  // recompute violation
  auto viol_retr = problem.eval_constraints(Y);
  EXPECT_LT(viol_retr.norm(), viol_init.norm())
      << "Retraction did not reduce cost";
}

TEST_P(InflationParamTest, ExactNewtonRetraction) {
  const auto& sdp = GetParam();
  // parameters
  RankInflateParams params;
  params.verbose = true;
  params.max_sol_rank = 3;
  params.retraction_method = RetractionMethod::ExactNewton;
  params.max_iter = 20;
  params.alpha_min = 1e-12;
  // generate problem
  auto problem = sdp.make(params);
  // Get actual solution
  Matrix Y = sdp.make_solution(params.max_sol_rank);
  // Add perturbation to solution
  Eigen::MatrixXd perturb =
      Eigen::MatrixXd::Random(sdp.dim, params.max_sol_rank) * 1.0E-1;
  Y += perturb;
  // Get initial violation
  auto viol_init = problem.eval_constraints(Y);
  // Call inflation
  problem.retraction(Y);
  // recompute violation
  auto viol_retr = problem.eval_constraints(Y);
  EXPECT_LT(viol_retr.norm(), viol_init.norm())
      << "Retraction did not reduce cost";
}

TEST_P(InflationParamTest, GaussNewtonRetraction) {
  const auto& sdp = GetParam();
  // parameters
  RankInflateParams params;
  params.verbose = true;
  params.max_sol_rank = 3;
  params.retraction_method = RetractionMethod::GaussNewton;
  params.max_iter = 20;
  params.alpha_min = 1e-12;
  // generate problem
  auto problem = sdp.make(params);
  // Get actual solution
  Matrix Y = sdp.make_solution(params.max_sol_rank);
  // Add perturbation to solution
  Eigen::MatrixXd perturb =
      Eigen::MatrixXd::Random(sdp.dim, params.max_sol_rank) * 1.0E-1;
  Y += perturb;
  // Get initial violation
  auto viol_init = problem.eval_constraints(Y);
  // Call inflation
  problem.retraction(Y);
  // recompute violation
  auto viol_retr = problem.eval_constraints(Y);
  EXPECT_LT(viol_retr.norm(), viol_init.norm())
      << "Retraction did not reduce cost";
}

TEST_P(InflationParamTest, GeodesicStep) {
  const auto& sdp = GetParam();
  // parameters
  int rank = 3;
  RankInflateParams params;
  params.verbose = true;
  params.max_sol_rank = rank;
  params.retraction_method = RetractionMethod::GaussNewton;
  // generate problem
  auto problem = sdp.make(params);
  // Get actual solution
  Matrix Y = sdp.make_solution(params.max_sol_rank);
  // Add perturbation to solution
  Eigen::MatrixXd perturb =
      Eigen::MatrixXd::Random(sdp.dim, params.max_sol_rank) * 1.0E-1;
  Y += perturb;
  // get jacobian and run QR decomposition
  auto Jac = Matrix(problem.m, sdp.dim * rank);
  auto viol = problem.eval_constraints(Y, Jac);
  Eigen::ColPivHouseholderQR<Matrix> qr(Jac);
  problem.qr_jacobian = get_soln_qr_dense(Jac, Vector::Zero(problem.m), 1e-10);
  // Take a geodesic step
  double alpha = 1e-2;
  auto [V, W] = problem.get_geodesic_step(Y.cols());
  auto Y_1 = Y + alpha * V;
  auto Y_2 = Y + alpha * V + std::pow(alpha, 2) * W;

  // Evaluate the violation
  auto viol_1 = problem.eval_constraints(Y_1);
  auto viol_2 = problem.eval_constraints(Y_2);
  // Print violation norms
  std::cout << "First order norm: " << (viol_1 - viol).norm() << std::endl;
  std::cout << "Second order norm: " << (viol_2 - viol).norm() << std::endl;
  // Check norm (should definitely decrease)
  EXPECT_LT((viol_2 - viol).norm(), (viol_1 - viol).norm())
      << "Norm of violation was worse with second order geodesic step";
}

// CERTIFICATE TESTS
TEST_P(InflationParamTest, CertWithInteriorPointSolution) {
  const auto& sdp = GetParam();
  // parameters
  RankInflateParams params;
  params.verbose = true;
  params.max_sol_rank = sdp.dim;
  // generate problem
  RankInflation problem = sdp.make(params);
  // Get interior point solution
  auto mosek_soln = solve_sdp_mosek(sdp.C, sdp.A, sdp.b, false);
  // Show eigenvalues of primal and dual solutions
  std::cout << "Eigenvalues of Primal Solution: " << std::endl
            << Eigen::SelfAdjointEigenSolver<Matrix>(mosek_soln.X).eigenvalues().transpose()
            << std::endl;
  std::cout << "Eigenvalues of Dual Solution: " << std::endl
            << Eigen::SelfAdjointEigenSolver<Matrix>(mosek_soln.S).eigenvalues().transpose()
            << std::endl;
  // Low rank approximation of primal and dual solutions
  auto Y_mosek = get_positive_eigspace(mosek_soln.X, 1e-5);
  auto Y_dual = get_positive_eigspace(mosek_soln.S, 1e-5);
  int r = Y_mosek.cols();
  int s = Y_dual.cols();
  // check strict complementarity
  std::cout << "r: " << r << ", s: " << s << std::endl;
  EXPECT_EQ(r + s, sdp.dim) << "Strict complementarity does not hold";
  // check objective value
  std::cout << "Primal objective at Mosek solution: " << std::endl
            << (Y_mosek.transpose() * sdp.C * Y_mosek).trace() << std::endl;
  EXPECT_NEAR((Y_mosek.transpose() * sdp.C * Y_mosek).trace(), sdp.rho, 1e-5)
      << "SDP not Tight";
  // Get jacobian and violation
  auto Jac = Matrix(problem.m, Y_mosek.cols() * sdp.dim); 
  auto violation = problem.eval_constraints(Y_mosek, Jac);
  std::cout << "Violation at IP : " << violation.norm()
            << std::endl;
  // Build certificate
  auto H = problem.build_certificate_from_primal(Jac, Y_mosek);
  // Compare certificate with optimal dual variable from Mosek
  std::cout << "Norm of difference between certificate and Mosek dual: " << std::endl
            << (H - mosek_soln.S).norm() << std::endl;
  // std::cout << "Certificate Matrix: " << std::endl << H << std::endl;
  // check certificate on high rank solution
  auto [min_eig_hr, first_ord_cond_hr] = problem.check_certificate(H, Y_mosek);
  std::cout << "Certificate on High Rank Solution: " << std::endl;
  std::cout << "Minimum Eigenvalue of Certificate: " << min_eig_hr << std::endl;
  std::cout << "First Order Condition Norm: " << first_ord_cond_hr << std::endl;
  // check certificate on initial solution
  auto [min_eig, first_ord_cond] = problem.check_certificate(H, sdp.make_solution(params.max_sol_rank));
  std::cout << "Certificate on Initial Solution: " << std::endl;
  std::cout << "Minimum Eigenvalue of Certificate: " << min_eig << std::endl;
  std::cout << "First Order Condition Norm: " << first_ord_cond << std::endl;
}

TEST_P(InflationParamTest, CertWithAnalyticCenter) {
  const auto& sdp = GetParam();
  // parameters
  RankInflateParams params;
  params.verbose = true;
  params.max_sol_rank = sdp.dim;
  auto delta = 1e-7;
  // generate problem
  RankInflation problem = sdp.make(params);
  // get current solution
  Matrix Y_0 = sdp.make_solution(params.max_sol_rank);
  // Run rank inflation, without inflation (target rank is 1)
  auto [X, multipliers] = problem.get_analytic_center(Y_0 * Y_0.transpose(), delta);
  std::cout << "Eigenvalues of Center: " << std::endl
            << Eigen::SelfAdjointEigenSolver<Matrix>(X).eigenvalues().transpose()
            << std::endl;
  // Recover low rank solution
  Matrix Y = get_positive_eigspace(X, params.tol_rank_sol);
  auto Jac = Matrix(problem.m, Y.cols() * sdp.dim);
  auto violation = problem.eval_constraints(Y, Jac);
  std::cout << "rank of recovered solution: " << get_rank(Y, 1e-5) << std::endl;
  std::cout << "Violation at Analytic Center: " << violation.norm()
            << std::endl;
  // Build the certificate matrix
  auto H = problem.build_certificate_from_dual(multipliers);
  // check certificate on high rank solution
  auto [min_eig_hr, first_ord_cond_hr] = problem.check_certificate(H, Y);
  std::cout << "Cost at High Rank Solution: " << (sdp.C * X).trace() << std::endl;
  std::cout << "Minimum Eigenvalue of Certificate: " << min_eig_hr << std::endl;
  std::cout << "First Order Condition Norm at High Rank Solution: " << first_ord_cond_hr << std::endl;
  // check certificate on initial solution
  auto [min_eig, first_ord_cond] = problem.check_certificate(H, Y_0);
  std::cout << "First Order Condition Norm at Rank 1 Solution: " << first_ord_cond << std::endl;
}

//  ANALYTIC CENTER TESTS
TEST_P(InflationParamTest, AnalyticCenter) {
  const auto& sdp = GetParam();
  // parameters
  RankInflateParams params;
  params.verbose = true;
  double delta = 1e-7;
  // generate problem
  auto problem = sdp.make(params);
  auto Y = sdp.soln;
  // Compute Analyic center starting from low rank solution
  auto X0 = Y * Y.transpose();
  auto [X, multipliers] = problem.get_analytic_center(X0, delta);
  // Compute analytic center objecive value
  double obj_0 = problem.get_analytic_center_objective(X0, delta);
  double obj_star = problem.get_analytic_center_objective(X, delta);
  // check that the center is PSD
  Eigen::SelfAdjointEigenSolver<Matrix> es(X);
  for (int i = 0; i < es.eigenvalues().size(); ++i) {
    std::cout << "Eigenvalue " << i << ": " << es.eigenvalues()(i) << std::endl;
    EXPECT_GE(es.eigenvalues()(i), -delta) << "Analytic center is not PSD";
  }
  // Check objective decrease
  std::cout << "Analytic Center Objective initially: " << obj_0 << std::endl;
  std::cout << "Analytic Center Objective solution: " << obj_star
            << std::endl;
  EXPECT_LT(obj_star, obj_0) << "Analytic center objective did not improve";
  // Check for rank increase
  auto rank_0 = Y.cols();
  auto Y_sol = get_positive_eigspace(X, params.tol_rank_sol);
  auto rank_star = Y_sol.cols();
  std::cout << "Rank at Init: " << rank_0 << ", Rank at Center: " << rank_star
            << std::endl;
  EXPECT_GE(rank_star, rank_0) << "Rank did not increase at analytic center";
  // Compare with rank of Interior point solution
  auto mosek_soln = solve_sdp_mosek(sdp.C, sdp.A, sdp.b, false);
  auto Y_mosek = get_positive_eigspace(mosek_soln.X, params.tol_rank_sol);
  auto rank_mosek = Y_mosek.cols();
  std::cout << "Rank at IP Solution: " << rank_mosek << std::endl;
  EXPECT_EQ(rank_star, rank_mosek) << "Rank at center is not equal to Mosek solution rank";

  // Norm of the difference between the IP solution and the center solution should be small
  double diff_norm = (Y_sol * Y_sol.transpose() - Y_mosek*Y_mosek.transpose()).norm();
  std::cout << "Norm of difference between center and Mosek solution: " << diff_norm << std::endl;
  EXPECT_NEAR(diff_norm, 0.0, 1e-4) << "Analytic center solution is not close to Mosek solution";
}


TEST_P(InflationParamTest, MosekSolve) {
  const auto& sdp = GetParam();
  // solve sdp using Mosek to get dual solution
  
  auto mosek_soln = solve_sdp_mosek(sdp.C, sdp.A, sdp.b);
  //print objective value
  std::cout << "Mosek Primal Objective: " << mosek_soln.obj_value << std::endl;
  // print the solution rank
  Eigen::SelfAdjointEigenSolver<Matrix> es(mosek_soln.X);
  std::cout << "Mosek Solution Eigenvalues: " << std::endl << es.eigenvalues() << std::endl;
  std::cout << "Mosek Solution Rank: " << get_rank(mosek_soln.X, 1e-6) << std::endl;
  // Check that objective matches rho 
  EXPECT_NEAR(mosek_soln.obj_value, sdp.rho, 1e-6) << "Mosek objective does not match expected value at analytic center"; 
}

TEST(AnalyticCenter, LineSearchFunctions) {
  // Generate random PSD matrix
  int dim = 5;
  // generate random orthogonal matrix A
  Matrix tmp = Matrix::Random(dim, dim);
  Eigen::HouseholderQR<Matrix> qr(tmp);
  Matrix A = qr.householderQ() * Matrix::Identity(dim, dim);
  // generate PSD matrix Z
  Matrix Z = A.transpose() * A * 5.0;
  // Generate random direction
  Matrix Aw = Matrix::Random(dim, dim);
  Aw = 0.5 * (Aw + Aw.transpose());  // symmetrize
  // Create step of the proper form
  Matrix dZ = Z - Z * Aw * Z;
  // parameters
  RankInflateParams params;
  params.verbose = true;
  RankInflation problem(Matrix::Zero(dim, dim), 0.0, {}, {}, params);
  double delta = 1e-6;
  // Generate functions
  auto [f, df] = problem.analytic_center_line_search_func(Z, Aw);
  // Test at several step sizes
  std::vector<double> alphas = {1e-4, 1e-2, 1e-1, 0.5, 1.0};
  const double tol = 1e-7;
  // Value checks
  for (double alpha : alphas) {
    double f_expected = -logdet(Z + alpha * dZ);
    if (std::isinf(f_expected)) {
      continue;  // skip infinite values
    }
    double f_val = f(alpha) - logdet(Z);
    EXPECT_NEAR(f_val, f_expected, tol)
        << "Line search function value mismatch at alpha = " << alpha;
  }
  // Derivative (finite-difference) checks
  for (double alpha : alphas) {
    double f_val = f(alpha);
    if (std::isinf(f_val)) {
      continue;  // skip infinite values
    }
    double df_val = df(alpha);
    double f_val_plus = f(alpha + tol);
    double num_df = (f_val_plus - f_val) / tol;
    EXPECT_NEAR(df_val, num_df, tol*100)
        << "Line search derivative mismatch at alpha = " << alpha;
  }
}

TEST(InflationParamTest, LowRankRecovery) {
  int dim = 3;
  double r1 = 0.5;
  double r2 = 0.5;
  double d = 0.5;
  // Start from skewed solution
  auto weights = Vector::Ones(dim - 1).eval();
  weights(0) = 10.0;
  weights /= weights.sum();
  auto Y0 = make_two_sphere_soln(r1, r2, d, weights);
  // recompute center
  auto X0 = Y0 * Y0.transpose();
  auto Y = recover_lowrank_factor(X0, 1e-10);
  // Compare to original solution
  auto diff = (X0 - Y * Y.transpose()).norm();
  const double tol = 1e-8;
  EXPECT_NEAR(diff, 0.0, tol);
  EXPECT_TRUE(Y.cols() == Y0.cols());
}



INSTANTIATE_TEST_SUITE_P(
    RankInflationSuite, InflationParamTest,
    ::testing::Values(
        // CASE 1
        make_lovasz_test_case(clique1_adj, {1, 3, 4, 6, 7, 8}, "Clique1"),
        // CASE 2
        make_lovasz_test_case(clique2_adj, {0, 2, 3, 5, 6, 8, 9}, "Clique2"),
        // CASE 3 (The 20x20 Matrix)
        make_lovasz_test_case(clique3_adj, {4, 10, 13, 14, 15, 16, 17, 18},
                              "Clique3_Large20x20"),
        // CASE 4
        make_lovasz_test_case(clique4_adj, {0, 1, 2}, "Clique4_Disconnected"),
        // CASE 5: Two Sphere Intersection
        make_two_sphere_sdp(5, 1.0, 1.0, 1.5)),
    // This helper function names the tests based on the 'test_name'
    // field
    [](const ::testing::TestParamInfo<InflationParamTest::ParamType>& info) {
      return info.param.name;
    });

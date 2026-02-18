/*
c++ tests for rank inflation
*/
#include "interior_point_sdp.hpp"
#include "test_harness.hpp"

using namespace SDPTools;

// Fixture Class
class AnalyticCentParamTest : public ::testing::TestWithParam<SDPTestProblem> {
};


TEST_P(AnalyticCentParamTest, MosekSolve) {
  const auto& sdp = GetParam();
  // solve sdp using Mosek to get dual solution

  auto mosek_soln = solve_sdp_mosek(sdp.C, sdp.A, sdp.b);
  // print objective value
  std::cout << "Mosek Primal Objective: " << mosek_soln.obj_value << std::endl;
  // print the solution rank
  Eigen::SelfAdjointEigenSolver<Matrix> es(mosek_soln.X);
  std::cout << "Mosek Solution Eigenvalues: " << std::endl
            << es.eigenvalues() << std::endl;
  std::cout << "Mosek Solution Rank: " << get_rank(mosek_soln.X, 1e-6)
            << std::endl;
  // Check that objective matches rho
  EXPECT_NEAR(mosek_soln.obj_value, sdp.rho, 1e-6)
      << "Mosek objective does not match expected value at analytic center";
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
  AnalyticCenterParams params;
  params.verbose = true;
  AnalyticCenter problem(Matrix::Zero(dim, dim), 0.0, {}, {}, params);
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
    EXPECT_NEAR(df_val, num_df, tol * 100)
        << "Line search derivative mismatch at alpha = " << alpha;
  }
}

TEST(AnalyticCentParamTest, LowRankRecovery) {
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

TEST_P(AnalyticCentParamTest, PrimalSolution) {
  const auto& sdp = GetParam();
  // parameters
  AnalyticCenterParams params;
  params.verbose = true;
  params.check_cert_ac = false; // Turn off early stopping based on certificate for testing purposes
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
  std::cout << "Analytic Center Objective solution: " << obj_star << std::endl;
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
  EXPECT_EQ(rank_star, rank_mosek)
      << "Rank at center is not equal to Mosek solution rank";

  // Norm of the difference between the IP solution and the center solution
  // should be small
  // double diff_norm =
  //     (Y_sol * Y_sol.transpose() - Y_mosek * Y_mosek.transpose()).norm();
  double diff_norm = (X - mosek_soln.X).norm();
  std::cout << "Norm of difference between center and Mosek solution: "
            << diff_norm << std::endl;
  EXPECT_NEAR(diff_norm, 0.0, 1e-4)
      << "Analytic center solution is not close to Mosek solution";
}

TEST_P(AnalyticCentParamTest, CertAtCenter) {
  const auto& sdp = GetParam();
  // parameters
  AnalyticCenterParams params;
  params.verbose = true;
  params.check_cert_ac = false; // Turn off early stopping based on certificate for testing purposes
  auto delta = 1e-7;
  // generate problem
  AnalyticCenter problem = sdp.make(params);
  // get current solution
  Matrix Y_0 = sdp.make_solution(1);
  // Run rank inflation, without inflation (target rank is 1)
  auto [X, multipliers] =
      problem.get_analytic_center(Y_0 * Y_0.transpose(), delta);
  std::cout
      << "Eigenvalues of Center: " << std::endl
      << Eigen::SelfAdjointEigenSolver<Matrix>(X).eigenvalues().transpose()
      << std::endl;
  // Recover low rank solution
  Matrix Y = get_positive_eigspace(X, params.tol_rank_sol);
  std::cout << "rank of recovered solution: " << get_rank(Y, 1e-5) << std::endl;
  auto violation = problem.eval_constraints(X);
  std::cout << "Violation at Analytic Center: " << violation.norm()
            << std::endl;
  // Build the certificate matrix
  auto H = problem.build_certificate_from_dual(multipliers);
  // check certificate on high rank solution
  auto [min_eig_hr, first_ord_cond_hr] = problem.check_certificate(H, Y);
  std::cout << "Cost at High Rank Solution: " << (sdp.C * X).trace()
            << std::endl;
  std::cout << "Minimum Eigenvalue of Certificate: " << min_eig_hr << std::endl;
  std::cout << "First Order Condition Norm at High Rank Solution: "
            << first_ord_cond_hr << std::endl;
  // check certificate on initial solution
  auto [min_eig, first_ord_cond] = problem.check_certificate(H, Y_0);
  std::cout << "First Order Condition Norm at Rank 1 Solution: "
            << first_ord_cond << std::endl;
}

TEST_P(AnalyticCentParamTest, CertEarlyStopping) {
  const auto& sdp = GetParam();
  // parameters
  AnalyticCenterParams params;
  params.verbose = true;
  params.check_cert_ac = true; // Turn off early stopping based on certificate for testing purposes
  auto delta = 1e-7;
  // generate problem
  AnalyticCenter problem = sdp.make(params);
  // get current solution
  Matrix Y_0 = sdp.make_solution(1);
  // Run rank inflation, without inflation (target rank is 1)
  auto [X, multipliers] =
      problem.get_analytic_center(Y_0 * Y_0.transpose(), delta);
  std::cout
      << "Eigenvalues of Center: " << std::endl
      << Eigen::SelfAdjointEigenSolver<Matrix>(X).eigenvalues().transpose()
      << std::endl;
  // Recover low rank solution
  Matrix Y = get_positive_eigspace(X, params.tol_rank_sol);
  std::cout << "rank of recovered solution: " << get_rank(Y, 1e-5) << std::endl;
  auto violation = problem.eval_constraints(X);
  std::cout << "Violation at Analytic Center: " << violation.norm()
            << std::endl;
  // Build the certificate matrix
  auto H = problem.build_certificate_from_dual(multipliers);
  // check certificate on high rank solution
  auto [min_eig_hr, first_ord_cond_hr] = problem.check_certificate(H, Y);
  std::cout << "Cost at High Rank Solution: " << (sdp.C * X).trace()
            << std::endl;
  std::cout << "Minimum Eigenvalue of Certificate: " << min_eig_hr << std::endl;
  std::cout << "First Order Condition Norm at High Rank Solution: "
            << first_ord_cond_hr << std::endl;
  // check certificate on initial solution
  auto [min_eig, first_ord_cond] = problem.check_certificate(H, Y_0);
  std::cout << "First Order Condition Norm at Rank 1 Solution: "
            << first_ord_cond << std::endl;
}


INSTANTIATE_TEST_SUITE_P(
    AnalyticCenterSuite, AnalyticCentParamTest,
    ::testing::Values(
        // CASE 1
        make_lovasz_test_case(clique1_adj, {1, 3, 4, 6, 7, 8}, "Clique1"),
        // CASE 2
        make_lovasz_test_case(clique2_adj, {0, 2, 3, 5, 6, 8, 9}, "Clique2"),
        // CASE 3 (The 20x20 Matrix)
        make_lovasz_test_case(clique3_adj, {4, 10, 13, 14, 15, 16, 17, 18},
                              "Clique3_Large20x20"),
        // CASE 4
        make_lovasz_test_case(clique4_adj, {0, 1, 2}, "Clique4_Disconnected")),
        // // CASE 5: Two Sphere Intersection (removed because not tight)
        // make_two_sphere_sdp(5, 1.0, 1.0, 1.5)),
    // This helper function names the tests based on the 'test_name'
    // field
    [](const ::testing::TestParamInfo<AnalyticCentParamTest::ParamType>& info) {
      return info.param.name;
    });
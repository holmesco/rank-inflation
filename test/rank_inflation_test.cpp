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

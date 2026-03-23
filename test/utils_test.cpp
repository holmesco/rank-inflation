/*
Tests for utility functions.
*/
#include "circle_problem.hpp"
#include "interior_point_sdp.hpp"
#include "lovasz_theta_problems.hpp"

using namespace RankTools;

// Fixture Class
class UtilsParamTest : public ::testing::TestWithParam<SDPTestProblem> {};

TEST(UtilsParamTest, LowRankRecovery) {
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

TEST_P(UtilsParamTest, MosekSolve) {
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

INSTANTIATE_TEST_SUITE_P(
    UtilsSuite, UtilsParamTest,
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
    [](const ::testing::TestParamInfo<UtilsParamTest::ParamType>& info) {
      return info.param.name;
    });
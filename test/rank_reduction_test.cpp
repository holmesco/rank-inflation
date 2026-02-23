/*
c++ tests for rank inflation
*/
#include "rank_reduction.hpp"

#include "interior_point_sdp.hpp"
#include "test_harness.hpp"

using namespace RankTools;

// Fixture Class
class RankReductionParamTest : public ::testing::TestWithParam<SDPTestProblem> {
};

TEST_P(RankReductionParamTest, MosekSolve) {
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

TEST_P(RankReductionParamTest, RankReduction) {
  const auto& sdp = GetParam();
  // parameters
  RankReductionParams params;
  params.verbose = true;
  params.targ_rank = -1;
  // Rank 1 solution
  auto V_opt = sdp.make_solution(1);
  // Solve optimization with mosek to get initial solution
  auto mosek_soln = solve_sdp_mosek(sdp.C, sdp.A, sdp.b);
  auto V_init = get_positive_eigspace(mosek_soln.X, 0.0);
  std::cout << "Initial solution rank: " << V_init.cols() << std::endl;
  // Run rank reduction
  auto V = rank_reduction(sdp.A, V_init, params);
  int r = V.cols();
  std::cout << "Rank after reduction: " << r << std::endl;
  // Check that rank is reduced to target
  EXPECT_LE(r, V_init.cols())
      << "Rank after reduction is not less than initial rank";
  // Check that solution is still feasible
  for (size_t i = 0; i < sdp.A.size(); ++i) {
    double constraint_val =
        (V.transpose() * sdp.A[i].selfadjointView<Eigen::Upper>() * V).trace();
    EXPECT_NEAR(constraint_val, sdp.b[i], 1e-5)
        << "Reduced solution does not satisfy constraint " << i;
  }
  // Check that objective value is preserved
  double obj_val = (V.transpose() * sdp.C * V).trace();
  EXPECT_NEAR(obj_val, mosek_soln.obj_value, 1e-5)
      << "Objective value after reduction does not match Mosek solution "
         "objective";
}

INSTANTIATE_TEST_SUITE_P(
    RankReductionTestSuite, RankReductionParamTest,
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
    [](const ::testing::TestParamInfo<RankReductionParamTest::ParamType>&
           info) { return info.param.name; });
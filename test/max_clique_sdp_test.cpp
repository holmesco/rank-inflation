/*
c++ tests for max clique with SDP
*/

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "max_clique_sdp/max_clique_sdp.hpp"

using namespace clipperplus;

#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include "clipperplus/clipperplus_clique.h"

// 1. Data structure to bundle the input and expected outputs
struct CliqueTestCase
{
    Eigen::MatrixXd adj;
    std::vector<int> expected_clique;
    clipperplus::CERTIFICATE expected_cert;
    std::string test_name;
};

// 2. The Fixture Class
class MaxCliqueParamTest : public ::testing::TestWithParam<CliqueTestCase>
{
};

// 3. The Single Generalized Test Function
TEST_P(MaxCliqueParamTest, FindsMaxClique)
{
    const auto &params = GetParam();
    // Generate problem
    auto graph = Graph(params.adj);
    auto problem = MaxCliqueProblem(graph);
    // run optimization
    auto solution = problem.optimize_cuhallar(params.expected_clique);
    // Check cost
    Eigen::RowVectorXd col_sums = solution.Y.colwise().sum();
    double cost = col_sums.array().square().sum();
    std::cout << "Max Clique: " << cost << std::endl;
    ASSERT_NEAR(cost, solution.primal_opt, 1e-3);

    // Test rank reduction
    // Run until reach rank 1 solution, regardless of singular value
    auto V = RankReduction::rank_reduction(problem.abs_edges, solution.Y, 1e-5, 1e-5, 1e-9, 1);
    std::cout << "Reduced rank solution matrix V: \n"
              << V.transpose() << "\n"
              << std::endl;

    // Get associated clique
    Eigen::VectorXd soln = V.col(0);
    auto clique = problem.soln_to_clique(soln);
    std::cout << "Reduced rank solution clique nodes: ";
    for (auto node : clique)
    {
        std::cout << node << " ";
    }
    std::cout << std::endl;
    // Check assertions
    ASSERT_TRUE(graph.is_clique(clique));
    // Assertions with descriptive messages
    EXPECT_EQ(clique.size(), params.expected_clique.size())
        << "Mismatched size in test case: " << params.test_name;
    // NOTE: The following check is commented out because the max clique may not be unique.
    // EXPECT_EQ(clique, params.expected_clique)
    //     << "Mismatched clique members in test case: " << params.test_name;
}

// 4. The Parameter Suite
INSTANTIATE_TEST_SUITE_P(
    MaxCliqueSDPSuite,
    MaxCliqueParamTest,
    ::testing::Values(
        // CASE 1
        CliqueTestCase{
            (Eigen::MatrixXd(10, 10) << 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0).finished(),
            {1, 3, 4, 6, 7, 8},
            clipperplus::CERTIFICATE::NONE,
            "Clique1_Standard"},
        // CASE 2
        CliqueTestCase{
            (Eigen::MatrixXd(10, 10) << 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0).finished(),
            {0, 2, 3, 5, 6, 8, 9},
            clipperplus::CERTIFICATE::CHROMATIC_BOUND,
            "Clique2_Chromatic"},
        // CASE 3 (The 20x20 Matrix)
        CliqueTestCase{
            (Eigen::MatrixXd(20, 20) << 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1,
             1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
             0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
             1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
             0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
             1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1,
             0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
             0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0)
                .finished(),
            {4, 10, 13, 14, 15, 16, 17, 18},
            clipperplus::CERTIFICATE::NONE,
            "Clique3_Large20x20"},
        // CASE 4
        CliqueTestCase{
            (Eigen::MatrixXd(5, 5) << 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0).finished(),
            {0, 1, 2},
            clipperplus::CERTIFICATE::HEURISTIC,
            "Clique4_Disconnected"}),
    // This helper function names the tests based on the 'test_name' field
    [](const ::testing::TestParamInfo<MaxCliqueParamTest::ParamType> &info)
    {
        return info.param.test_name;
    });

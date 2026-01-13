#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include "clipperplus/clipperplus_clique.h"

// 1. Data structure to bundle the input and expected outputs
struct CliqueTestCase {
    Eigen::MatrixXd adj;
    std::vector<int> expected_clique;
    clipperplus::CERTIFICATE expected_cert;
    std::string test_name; 
};

// 2. The Fixture Class
class ClipperPlusParamTest : public ::testing::TestWithParam<CliqueTestCase> {};

// 3. The Single Generalized Test Function
TEST_P(ClipperPlusParamTest, FindsCorrectMaxClique) {
    const auto& params = GetParam();

    // Execute logic
    auto [clique, certificate] = clipperplus::find_clique(params.adj);
    std::sort(clique.begin(), clique.end());

    // Assertions with descriptive messages
    auto graph = clipperplus::Graph(params.adj);
    ASSERT_TRUE(graph.is_clique(clique)) 
        << "Returned set is not a clique in test case: " << params.test_name;
    ASSERT_EQ(clique.size(), params.expected_clique.size()) 
        << "Mismatched size in test case: " << params.test_name;
    ASSERT_EQ(clique, params.expected_clique) 
        << "Mismatched clique members in test case: " << params.test_name;
    ASSERT_EQ(certificate, params.expected_cert) 
        << "Mismatched certificate in test case: " << params.test_name;
}

// 4. The Parameter Suite 
INSTANTIATE_TEST_SUITE_P(
    ClipperPlusSuite,
    ClipperPlusParamTest,
    ::testing::Values(
        // CASE 1
        CliqueTestCase{
            (Eigen::MatrixXd(10,10) << 0,0,1,1,1,1,1,0,1,0, 0,0,1,1,1,0,1,1,1,1, 1,1,0,1,0,1,1,1,0,1, 1,1,1,0,1,1,1,1,1,1, 1,1,0,1,0,0,1,1,1,1, 1,0,1,1,0,0,1,1,1,1, 1,1,1,1,1,1,0,1,1,0, 0,1,1,1,1,1,1,0,1,1, 1,1,0,1,1,1,1,1,0,1, 0,1,1,1,1,1,0,1,1,0).finished(),
            {1, 3, 4, 6, 7, 8},
            clipperplus::CERTIFICATE::NONE,
            "Clique1_Standard"
        },
        // CASE 2
        CliqueTestCase{
            (Eigen::MatrixXd(10,10) << 0,0,1,1,1,1,1,1,1,1, 0,0,1,1,1,1,1,1,1,1, 1,1,0,1,1,1,1,1,1,1, 1,1,1,0,1,1,1,0,1,1, 1,1,1,1,0,1,0,1,1,0, 1,1,1,1,1,0,1,1,1,1, 1,1,1,1,0,1,0,1,1,1, 1,1,1,0,1,1,1,0,1,1, 1,1,1,1,1,1,1,1,0,1, 1,1,1,1,0,1,1,1,1,0).finished(),
            {0, 2, 3, 5, 6, 8, 9},
            clipperplus::CERTIFICATE::CHROMATIC_BOUND,
            "Clique2_Chromatic"
        },
        // CASE 3 (The 20x20 Matrix)
        CliqueTestCase{
            (Eigen::MatrixXd(20,20) << 
                0,0,1,0,0,1,1,1,0,0,1,1,0,1,1,1,1,1,0,1, 0,0,1,1,1,1,1,1,0,1,1,1,1,1,1,0,1,0,1,1,
                1,1,0,1,0,1,0,1,1,1,0,1,1,0,0,1,1,1,1,0, 0,1,1,0,0,1,1,1,1,0,1,0,1,1,1,0,1,1,1,1,
                0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 1,1,1,1,0,0,1,0,1,0,1,1,1,0,0,1,1,1,1,1,
                1,1,0,1,1,1,0,1,1,0,1,1,1,1,0,1,1,1,1,1, 1,1,1,1,1,0,1,0,1,0,1,1,1,1,1,1,1,0,0,0,
                0,0,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,0,1,0, 0,1,1,0,1,0,0,0,1,0,1,1,0,0,1,1,0,0,1,1,
                1,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1, 1,1,1,0,1,1,1,1,1,1,1,0,0,0,1,1,0,1,1,1,
                0,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1, 1,1,0,1,1,0,1,1,1,0,1,0,0,0,1,1,1,1,1,1,
                1,1,0,1,1,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1, 1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,
                1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,1,0,1,1,0, 1,0,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,1,1,
                0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1, 1,1,0,1,1,1,1,0,0,1,1,1,1,1,1,0,0,1,1,0
            ).finished(),
            {4, 10, 13, 14, 15, 16, 17, 18},
            clipperplus::CERTIFICATE::NONE,
            "Clique3_Large20x20"
        },
        // CASE 4
        CliqueTestCase{
            (Eigen::MatrixXd(5,5) << 0,1,1,0,0, 1,0,1,0,0, 1,1,0,0,0, 0,0,0,0,1, 0,0,0,1,0).finished(),
            {0, 1, 2},
            clipperplus::CERTIFICATE::HEURISTIC,
            "Clique4_Disconnected"
        }
    ),
    // This helper function names the tests based on the 'test_name' field
    [](const ::testing::TestParamInfo<ClipperPlusParamTest::ParamType>& info) {
        return info.param.test_name;
    }
);
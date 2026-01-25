/*
c++ tests for rank inflation
*/
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <max_clique_sdp/rank_inflation.hpp>
#include <utility>

using namespace SDPTools;
using Edge = std::pair<int, int>;
using Triplet = Eigen::Triplet<double>;

// Compute edges from adjacency, only provide upper triangle indices
std::pair<std::vector<Edge>, std::vector<Edge>> get_edges(const Matrix& adj) {
  std::vector<Edge> edges;
  std::vector<Edge> nonedges;
  for (int i = 0; i < adj.rows(); i++) {
    for (int j = i+1; j < adj.rows(); j++) {
      if (adj(i, j) > 0.0) {
        edges.push_back({i, j});
      } else {
        nonedges.push_back({i, j});
      }
    }
  }
  return {edges, nonedges};
}

// Convert adjancency to rank inflation problem
std::vector<Eigen::SparseMatrix<double>> get_lovasz_constraints(int dim, std::vector<Edge> nonedges) {
  // generate constraints
  std::vector<Eigen::SparseMatrix<double>> A;
  std::vector<float> b;
  for (auto edge : nonedges) {
    // define sparse matrix
    A.emplace_back(dim, dim);
    std::vector<Triplet> tripletList;
    tripletList.push_back(Triplet(edge.first, edge.second, 1.0));
    A.back().setFromTriplets(tripletList.begin(), tripletList.end());
  }
  // Trace constraint
  A.emplace_back(dim,dim);
  A.back().setIdentity();
  return A;
}

// 1. Data structure to bundle the input and expected outputs
struct LovascThetaTestCase {
  Eigen::MatrixXd adj;
  std::vector<int> expected_clique;
  std::string test_name;
};

// 2. The Fixture Class
class LovascThetaParamTest
    : public ::testing::TestWithParam<LovascThetaTestCase> {};

TEST_P(LovascThetaParamTest, EvalConstraints) {
  const auto& test_params = GetParam();
  // get info from adjacency
  auto [edges, nonedges] = get_edges(test_params.adj);
  int dim = test_params.adj.rows();
  // Generate constraints
  auto A = get_lovasz_constraints(dim, nonedges);
  auto b = std::vector<float>(A.size(), 0.0);
  b.back() = 1.0;
  // generate cost
  Matrix C = Matrix::Ones(dim, dim);
  float rho = test_params.expected_clique.size();
  // parameters
  RankInflateParams params;
  params.use_cost_constraint = true;
  params.verbose = true;
  params.target_rank = 2;
  // generate problem
  auto problem = RankInflation(C, rho, A, b, params);
  // Test vector at actual solution
  Matrix Y = Matrix::Zero(dim, 2);
  std::vector<int> clique = test_params.expected_clique;
  float clq_num = clique.size();
  for (int i : clique) {
    Y(i, 0) = std::sqrt(1/clq_num);
  }
  auto grad = Matrix(problem.m, problem.params_.target_rank * dim);
  // Call evaluation function
  auto output = problem.eval_constraints(Y, grad);
  // evaluation and gradient should be near zero
  std::cout << "Evaluation: " << std::endl << output << std::endl;
  const double tol = 1e-6;
  ASSERT_EQ(output.size(), problem.m);
  for (int i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output(i), 0.0, tol) << "constraint " << i;
  }
  ASSERT_EQ(grad.rows(), problem.m);
  ASSERT_EQ(grad.cols(), problem.params_.target_rank * dim);
  // for (int i = 0; i < grad.rows(); ++i) {
  //   for (int j = 0; j < grad.cols(); ++j) {
  //     EXPECT_NEAR(grad(i, j), 0.0, tol) << "grad(" << i << "," << j << ")";
  //   }
  // }
  std::cout << "Grad: " << std::endl << grad << std::endl;
}

// 4. The Parameter Suite
INSTANTIATE_TEST_SUITE_P(
    RankInflationSuite, LovascThetaParamTest,
    ::testing::Values(
        // CASE 1
        LovascThetaTestCase{
            (Eigen::MatrixXd(10, 10) << 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1,
             1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1,
             1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1,
             1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0)
                .finished(),
            {1, 3, 4, 6, 7, 8},
            "Clique1_Standard"},
        // CASE 2
        LovascThetaTestCase{
            (Eigen::MatrixXd(10, 10) << 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
             1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1,
             1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0)
                .finished(),
            {0, 2, 3, 5, 6, 8, 9},
            "Clique2_Chromatic"},
        // CASE 3 (The 20x20 Matrix)
        LovascThetaTestCase{
            (Eigen::MatrixXd(20, 20) << 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0,
             1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
             0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1,
             1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
             1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
             1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1,
             0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,
             1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0,
             0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1,
             1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1,
             1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
             1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1,
             1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1,
             1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
             0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0)
                .finished(),
            {4, 10, 13, 14, 15, 16, 17, 18},
            "Clique3_Large20x20"},
        // CASE 4
        LovascThetaTestCase{(Eigen::MatrixXd(5, 5) << 0, 1, 1, 0, 0, 1, 0, 1, 0,
                             0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0)
                                .finished(),
                            {0, 1, 2},
                            "Clique4_Disconnected"}),
    // This helper function names the tests based on the 'test_name' field
    [](const ::testing::TestParamInfo<LovascThetaParamTest::ParamType>& info) {
      return info.param.test_name;
    });

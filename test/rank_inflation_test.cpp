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
    for (int j = i + 1; j < adj.rows(); j++) {
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
std::vector<Eigen::SparseMatrix<double>> get_lovasz_constraints(
    int dim, std::vector<Edge> nonedges) {
  // generate constraints
  std::vector<Eigen::SparseMatrix<double>> A;
  std::vector<double> b;
  for (auto edge : nonedges) {
    // define sparse matrix
    A.emplace_back(dim, dim);
    std::vector<Triplet> tripletList;
    tripletList.push_back(Triplet(edge.first, edge.second, 1.0));
    A.back().setFromTriplets(tripletList.begin(), tripletList.end());
  }
  // Trace constraint
  A.emplace_back(dim, dim);
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

//Test constraint evaluation and gradient function
TEST_P(LovascThetaParamTest, EvalConstraints) {
  const auto& test_params = GetParam();
  // get info from adjacency
  auto [edges, nonedges] = get_edges(test_params.adj);
  int dim = test_params.adj.rows();
  // Generate constraints
  auto A = get_lovasz_constraints(dim, nonedges);
  auto b = std::vector<double>(A.size(), 0.0);
  b.back() = 1.0;
  // generate cost
  Matrix C = Matrix::Ones(dim, dim);
  double rho = test_params.expected_clique.size();
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
  double clq_num = clique.size();
  for (int i : clique) {
    Y(i, 0) = std::sqrt(1 / clq_num);
  }
  auto grad = std::make_shared<Matrix>(problem.m, problem.params_.target_rank * dim);
  // Call evaluation function
  auto output = problem.eval_constraints(Y, grad);
  // evaluation and gradient should be near zero
  // std::cout << "Evaluation: " << std::endl << output << std::endl;
  const double tol = 1e-6;
  ASSERT_EQ(output.size(), problem.m);
  for (int i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output(i), 0.0, tol) << "constraint " << i;
  }

  // std::cout << "Grad: " << std::endl << grad << std::endl;
  // Numerical directional derivative check
  const double eps = 1e-6;
  int r = problem.params_.target_rank;
  int vec_size = r * dim;
  Eigen::VectorXd delta_vec = Eigen::VectorXd::Random(vec_size);
  delta_vec.normalize();
  delta_vec *= eps;
  // Map delta into a dim x r matrix (Eigen is column-major by default)
  Eigen::Map<Matrix> deltaY(delta_vec.data(), dim, r);
  Matrix Y2 = Y + deltaY;
  auto output2 = problem.eval_constraints(Y2);
  Eigen::VectorXd num_deriv = (output2 - output) / eps;
  Eigen::VectorXd anal_dir = *grad * delta_vec / eps;
  const double deriv_tol = 1e-5;
  for (int i = 0; i < problem.m; ++i) {
    EXPECT_NEAR(num_deriv(i), anal_dir(i), deriv_tol)
        << "directional derivative mismatch at constraint " << i;
  }
}

// Test RRQR Solve
TEST_P(LovascThetaParamTest, RRQRSolve) {
  const auto& test_params = GetParam();
  // get info from adjacency
  auto [edges, nonedges] = get_edges(test_params.adj);
  int dim = test_params.adj.rows();
  // Generate constraints
  auto A = get_lovasz_constraints(dim, nonedges);
  auto b = std::vector<double>(A.size(), 0.0);
  b.back() = 1.0;
  // generate cost
  Matrix C = Matrix::Ones(dim, dim);
  double rho = test_params.expected_clique.size();
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
  double clq_num = clique.size();
  for (int i : clique) {
    Y(i, 0) = std::sqrt(1 / clq_num);
  }
  auto grad = std::make_shared<Matrix>(problem.m, problem.params_.target_rank * dim);
  // Call evaluation function
  auto output = problem.eval_constraints(Y, grad);
  // Apply QR decomposition
  QRResult soln = get_soln_qr_dense(*grad, -output);
  // solution should be zero
  const double tol = 1e-6;
  ASSERT_EQ(soln.solution_particular.size(), problem.params_.target_rank * dim);
  for (int i = 0; i < soln.solution_particular.size(); ++i) {
    EXPECT_NEAR(soln.solution_particular(i), 0.0, tol) << "row " << i;
  }
  // Check for nullspace, if exists add to solution and verify small change in output
  int nulldim = soln.nullspace_basis.cols();
  if (nulldim > 0){
    std::cout << "Nullspace dimension: " << nulldim << ". Testing nullspace... " << std::endl;
    // Construct delta in the nullspace
    Eigen::VectorXd alpha = Eigen::VectorXd::Random(nulldim); // values in [-1,1]
    double alpha_norm = alpha.norm();
    if (alpha_norm > 0) alpha /= alpha_norm;
    Matrix dY = (soln.nullspace_basis * alpha).reshaped(dim, problem.params_.target_rank);
    // Add delta to solution
    Matrix Y_plus = Y + dY;
    // Evaluate constraints at new solution
    Vector output_Y_plus = problem.eval_constraints(Y_plus);
    Vector output_dY = problem.eval_constraints(dY);
    // Constraint value
    std::vector<double> vals(b.begin(), b.end());
    vals.push_back(rho);
    Vector constraint_val = Vector::Map(vals.data(), vals.size());
    // linear component of the new output
    Vector output_linear = output_Y_plus - output - (output_dY + constraint_val);
    // Should evaluate to zero
    for (int i = 0; i < output_linear.size(); ++i) {
      EXPECT_NEAR(output_linear(i), 0.0, tol) << "row " << i;
    }

    
  }
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

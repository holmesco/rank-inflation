/*
c++ tests for rank inflation
*/
#include "rank_inflation.hpp"

#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
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

// Test constraint evaluation and gradient function
TEST_P(LovascThetaParamTest, EvalFuncAndGrad) {
  const auto& test_params = GetParam();
  // get info from adjacency
  auto [edges, nonedges] = get_edges(test_params.adj);
  int dim = test_params.adj.rows();
  // Generate constraints
  auto A = get_lovasz_constraints(dim, nonedges);
  auto b = std::vector<double>(A.size(), 0.0);
  b.back() = 1.0;
  // generate cost
  Matrix C = -Matrix::Ones(dim, dim);
  double rho = -static_cast<double>(test_params.expected_clique.size());
  // parameters
  RankInflateParams params;
  params.verbose = true;
  params.max_sol_rank = 2;
  // generate problem
  auto problem = RankInflation(C, rho, A, b, params);
  // Test vector at actual solution
  Matrix Y = Matrix::Zero(dim, 2);
  std::vector<int> clique = test_params.expected_clique;
  double clq_num = clique.size();
  for (int i : clique) {
    Y(i, 0) = std::sqrt(1 / clq_num);
  }
  auto Jac =
      std::make_unique<Matrix>(problem.m, problem.params_.max_sol_rank * dim);
  // Call evaluation function
  auto output = problem.eval_constraints(Y, &Jac);
  // evaluation and gradient should be near zero
  // std::cout << "Evaluation: " << std::endl << output << std::endl;
  const double tol = 1e-6;
  ASSERT_EQ(output.size(), problem.m);
  for (int i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output(i), 0.0, tol) << "constraint " << i;
  }
  // Perturb solution and check Jacobian via finite differences
  Y += 0.01 * Matrix::Random(dim, 2);
  output = problem.eval_constraints(Y, &Jac);
  std::cout << "Jac: " << std::endl << *Jac << std::endl;
  // Numerical directional derivative check
  const double eps = 1e-8;
  int r = problem.params_.max_sol_rank;
  int vec_size = r * dim;
  const double deriv_tol = 1e-5;
  Eigen::MatrixXd ident = Eigen::MatrixXd::Identity(vec_size, vec_size);
  for (int i = 0; i < vec_size; ++i) {
    Eigen::VectorXd delta_vec = ident.col(i);
    Matrix Y2 = Y + eps * delta_vec.reshaped(dim, r);
    auto output2 = problem.eval_constraints(Y2);
    Eigen::VectorXd num_deriv = (output2 - output) / eps;
    Eigen::VectorXd anal_dir = *Jac * delta_vec;
    for (int j = 0; j < problem.m; ++j) {
      EXPECT_NEAR(num_deriv(j), anal_dir(j), deriv_tol)
          << "directional derivative mismatch at constraint " << j
          << " for direction " << i;
    }
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
  Matrix C = -Matrix::Ones(dim, dim);
  double rho = -static_cast<double>(test_params.expected_clique.size());
  // parameters
  RankInflateParams params;
  params.verbose = true;
  params.max_sol_rank = 2;
  // generate problem
  auto problem = RankInflation(C, rho, A, b, params);
  // Test vector at actual solution
  Matrix Y = Matrix::Zero(dim, 2);
  std::vector<int> clique = test_params.expected_clique;
  double clq_num = clique.size();
  for (int i : clique) {
    Y(i, 0) = std::sqrt(1 / clq_num);
  }
  auto Jac =
      std::make_unique<Matrix>(problem.m, problem.params_.max_sol_rank * dim);
  // Call evaluation function
  auto output = problem.eval_constraints(Y, &Jac);
  // Apply QR decomposition
  QRResult soln =
      get_soln_qr_dense(*Jac, -output, problem.params_.rank_def_thresh);
  // solution should be zero
  const double tol = 1e-6;
  ASSERT_EQ(soln.solution.size(), problem.params_.max_sol_rank * dim);
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
                    .reshaped(dim, problem.params_.max_sol_rank);
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
    Vector output_linear =
        output_Y_plus - output - (output_dY + constraint_val);
    // Should evaluate to zero
    for (int i = 0; i < output_linear.size(); ++i) {
      EXPECT_NEAR(output_linear(i), 0.0, tol) << "row " << i;
    }
  }
}

TEST_P(LovascThetaParamTest, GradDescentRetraction) {
  const auto& test_params = GetParam();
  // get info from adjacency
  auto [edges, nonedges] = get_edges(test_params.adj);
  int dim = test_params.adj.rows();
  // Generate constraints
  auto A = get_lovasz_constraints(dim, nonedges);
  auto b = std::vector<double>(A.size(), 0.0);
  b.back() = 1.0;
  // generate cost
  Matrix C = -Matrix::Ones(dim, dim);
  double rho = -static_cast<double>(test_params.expected_clique.size());
  // parameters
  RankInflateParams params;
  params.verbose = true;
  params.max_sol_rank = 3;
  params.retraction_method = RetractionMethod::GaussNewton;
  params.max_iter = 10000;
  params.alpha_min = 1e-12;
  // generate problem
  auto problem = RankInflation(C, rho, A, b, params);
  // Get actual solution
  Matrix Y = Matrix::Zero(dim, params.max_sol_rank);
  std::vector<int> clique = test_params.expected_clique;
  double clq_num = clique.size();
  for (int i : clique) {
    Y(i, 0) = std::sqrt(1 / clq_num);
  }
  // Add perturbation to solution
  Eigen::MatrixXd perturb =
      Eigen::MatrixXd::Random(dim, params.max_sol_rank) * 1.0E-1;
  Y += perturb;
  // Call inflation
  auto Y_ = problem.inflate_solution(Y);
}

// Test Second Order Correction
TEST_P(LovascThetaParamTest, SecondOrdCorrection) {
  const auto& test_params = GetParam();
  // get info from adjacency
  auto [edges, nonedges] = get_edges(test_params.adj);
  int dim = test_params.adj.rows();
  // Generate constraints
  auto A = get_lovasz_constraints(dim, nonedges);
  auto b = std::vector<double>(A.size(), 0.0);
  b.back() = 1.0;
  // generate cost
  Matrix C = -Matrix::Ones(dim, dim);
  double rho = -static_cast<double>(test_params.expected_clique.size());
  // parameters
  RankInflateParams params;
  params.verbose = true;
  params.max_sol_rank = 2;
  // generate problem
  auto problem = RankInflation(C, rho, A, b, params);
  // Test vector at actual solution
  Matrix Y = Matrix::Zero(dim, 2);
  std::vector<int> clique = test_params.expected_clique;
  double clq_num = clique.size();
  for (int i : clique) {
    Y(i, 0) = std::sqrt(1 / clq_num);
  }
  // Add perturbation to solution
  // Eigen::MatrixXd perturb = Eigen::MatrixXd::Random(dim, 2) * 0.1;
  // Y += perturb;
  Y *= 1.1;
  // Call evaluation function
  auto Jac =
      std::make_unique<Matrix>(problem.m, problem.params_.max_sol_rank * dim);
  auto output = problem.eval_constraints(Y, &Jac);
  // Apply QR decomposition
  QRResult result =
      get_soln_qr_dense(*Jac, -output, problem.params_.rank_def_thresh);
  // Gauss Newton part of the step
  auto delta_gn = result.solution;
  // Get system of equations for second order correction
  auto [hess, grad] = problem.build_proj_corr_grad_hess(
      output, result.nullspace_basis, delta_gn);
  // Solve new system
  QRResult corr_result =
      get_soln_qr_dense(hess, -grad, problem.params_.tol_null_corr);
  // reconstruct solution
  auto delta_corr = result.nullspace_basis * corr_result.solution;
  auto delta = delta_gn + delta_corr;
  // Evaluate
  auto viol_gn =
      problem.eval_constraints(Y + delta_gn.reshaped(dim, params.max_sol_rank));
  auto viol =
      problem.eval_constraints(Y + delta.reshaped(dim, params.max_sol_rank));
  // print norm of violations
  std::cout << "Norm of violation after GN step: " << viol_gn.norm()
            << ", after SOC step: " << viol.norm() << std::endl;
  EXPECT_TRUE(viol.norm() <= viol_gn.norm())
      << "Second order correction did not reduce constraint violation";
}

// Test Rank Inflation
TEST_P(LovascThetaParamTest, RankInflation) {
  const auto& test_params = GetParam();
  // get info from adjacency
  auto [edges, nonedges] = get_edges(test_params.adj);
  int dim = test_params.adj.rows();
  // Generate constraints
  auto A = get_lovasz_constraints(dim, nonedges);
  auto b = std::vector<double>(A.size(), 0.0);
  b.back() = 1.0;
  // generate cost
  Matrix C = -Matrix::Ones(dim, dim);
  double rho = -static_cast<double>(test_params.expected_clique.size());
  // parameters
  RankInflateParams params;
  params.verbose = true;
  params.max_sol_rank = dim;
  // generate problem
  auto problem = RankInflation(C, rho, A, b, params);
  // get current soluition
  Matrix Y_0 = Matrix::Zero(dim, 1);
  std::vector<int> clique = test_params.expected_clique;
  double clq_num = clique.size();
  for (int i : clique) {
    Y_0(i, 0) = std::sqrt(1 / clq_num);
  }
  // Run rank inflation, without inflation (target rank is 1)
  auto Y = problem.inflate_solution(Y_0);
  // Check solution rank
  int r = get_rank(Y, 1.0E-5);
  EXPECT_TRUE(r >= params.max_sol_rank) << "Did not acheive target rank";
  // Check constraint tolerance
  auto viol = problem.eval_constraints(Y);
  EXPECT_TRUE(viol.norm() <= params.tol_violation)
      << "Did not acheive target constraint violation";
}

// Test Certificate
TEST_P(LovascThetaParamTest, Certificate) {
  const auto& test_params = GetParam();
  // get info from adjacency
  auto [edges, nonedges] = get_edges(test_params.adj);
  int dim = test_params.adj.rows();
  // Generate constraints
  auto A = get_lovasz_constraints(dim, nonedges);
  auto b = std::vector<double>(A.size(), 0.0);
  b.back() = 1.0;
  // generate cost
  Matrix C = -Matrix::Ones(dim, dim);
  double rho = -static_cast<double>(test_params.expected_clique.size());
  // parameters
  RankInflateParams params;
  params.verbose = true;
  params.max_sol_rank = dim;
  // generate problem
  auto problem = RankInflation(C, rho, A, b, params);
  // get current solution
  std::vector<int> clique = test_params.expected_clique;
  double clq_num = clique.size();
  auto Y_0 = Matrix::Zero(dim, 1).eval();
  for (int i : clique) {
    Y_0(i, 0) = std::sqrt(1 / clq_num);
  }
  // Run rank inflation, without inflation (target rank is 1)
  auto Jac = std::make_unique<Matrix>(problem.m, dim * params.max_sol_rank);
  auto Y = problem.inflate_solution(Y_0, &Jac);
  // std::cout << "dot product of Y_0 and Y: "
  //           << (Y_0.transpose() * Y).norm() / Y.norm() / Y_0.norm()
  //           << std::endl;
  // std::cout << "Initial solution: " << std::endl << Y_0 << std::endl;
  // std::cout << "Inflated solution: " << std::endl << Y << std::endl;
  // std::cout << "Jacobian: " << std::endl << *Jac << std::endl;
  // Build certificate
  auto H = problem.build_certificate(*Jac, Y);
  // std::cout << "Certificate Matrix: " << std::endl << H << std::endl;
  // check certificate on high rank solution
  auto [min_eig_hr, first_ord_cond_hr] = problem.check_certificate(H, Y);
  std::cout << "Certificate on High Rank Solution: " << std::endl;
  std::cout << "Minimum Eigenvalue of Certificate: " << min_eig_hr << std::endl;
  std::cout << "First Order Condition Norm: " << first_ord_cond_hr << std::endl;
  std::cout << "Cost at High Rank Solution: " << std::endl
            << (Y.transpose() * C * Y).trace() << std::endl;
  // check certificate on initial solution
  auto [min_eig, first_ord_cond] = problem.check_certificate(H, Y_0);
  std::cout << "Certificate on Initial Solution: " << std::endl;
  std::cout << "Minimum Eigenvalue of Certificate: " << min_eig << std::endl;
  std::cout << "First Order Condition Norm: " << first_ord_cond << std::endl;
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

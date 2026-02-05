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

// Test case data structure
struct SDPTestProblem {
  int dim;     // matrix dimension
  Matrix C;    // cost
  double rho;  // scalar offset
  std::vector<Eigen::SparseMatrix<double>> A;
  std::vector<double> b;
  Matrix soln;
  std::string name;

  // Retrieve zero padded solution for testing.
  Matrix make_solution(int rank) const {
    Matrix zpad = Matrix::Zero(dim, rank - soln.cols());
    Matrix Y(dim, rank);
    Y << soln, zpad;
    return Y;
  }

  RankInflation make(const RankInflateParams& params) const {
    return RankInflation(C, rho, A, b, params);
  }
};

// Fixture Class
class InflationParamTest : public ::testing::TestWithParam<SDPTestProblem> {};

// ----------- Lovasc Theta Helper Functions ----------------

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

SDPTestProblem make_lovasz_test_case(const Eigen::MatrixXd& adj,
                                     std::vector<int> clique,
                                     std::string name) {
  int dim = adj.rows();
  auto [edges, nonedges] = get_edges(adj);

  SDPTestProblem sdp;
  // get cost and optimal solution
  sdp.dim = dim;
  sdp.C = -Matrix::Ones(dim, dim);
  sdp.rho = -static_cast<double>(clique.size());
  // get constraints
  sdp.A = get_lovasz_constraints(dim, nonedges);
  sdp.b.assign(sdp.A.size(), 0.0);
  sdp.b.back() = 1.0;

  // get solution
  sdp.soln = Vector::Zero(dim);
  double s = std::sqrt(1.0 / clique.size());
  for (int i : clique) {
    sdp.soln(i, 0) = s;
  }
  sdp.name = "LovaszTheta_" + name;
  return sdp;
}

// ------------ Lovasz-Theta Data Matrices ------------
static Matrix clique1_adj =
    (Matrix(10, 10) << 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1,
     1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
     0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
     0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1,
     1, 1, 0, 1, 1, 0)
        .finished();
static Matrix clique2_adj =
    (Eigen::MatrixXd(10, 10) << 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1,
     1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
     0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
     1, 1, 1, 0, 1, 1, 1, 1, 0)
        .finished();
static Matrix clique3_adj =
    (Eigen::MatrixXd(20, 20) << 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1,
     1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1,
     1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1,
     1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1,
     1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1,
     1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1,
     1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0,
     1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
     1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1,
     1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
     1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1,
     0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1,
     1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1,
     1, 1, 1, 1, 0, 0, 1, 1, 0)
        .finished();
static Matrix clique4_adj = (Eigen::MatrixXd(5, 5) << 0, 1, 1, 0, 0, 1, 0, 1, 0,
                             0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0)
                                .finished();

// Get factorized solution to two sphere problem
// It is assumed that weights is normalized and its length is 1-dim
Matrix make_two_sphere_soln(double r1, double r2, double d, Vector weights) {
  int n = weights.size() + 1;
  // Feasible point on intersection
  double alpha = (r1 * r1 - r2 * r2 + d * d) / (2 * d);
  double beta = std::sqrt(r1 * r1 - alpha * alpha);

  // Construct Y matrix
  auto Y = Matrix::Zero(n + 1, n - 1).eval();
  for (int i = 0; i < n - 1; i++) {
    Y(0, i) = alpha;
    Y(i + 1, i) = beta;
    Y(n, i) = 1.0;
  }

  return Y * weights.cwiseSqrt().asDiagonal();
}
// Generate an n-dimensional intersection problem between two spheres of radius
// r1 and r2 that are spaced d distance apart along the first axis
SDPTestProblem make_two_sphere_sdp(int n, double r1, double r2, double d) {
  assert(d < r1 + r2 &&
         "distance must be strictly less than the sum of the two radii");
  Eigen::VectorXd c1 = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd c2 = Eigen::VectorXd::Zero(n);
  c2(0) = d;  // shift along x-axis

  int dim = n + 1;  // homogenizing variable t

  SDPTestProblem sdp;
  sdp.dim = dim;

  auto make_Q = [dim, n](const Eigen::VectorXd& c, double r) {
    Eigen::SparseMatrix<double> A(dim, dim);
    std::vector<Eigen::Triplet<double>> T;
    for (int i = 0; i < n; ++i) T.emplace_back(i, i, 1.0);  // x^T x
    for (int i = 0; i < n; ++i) {
      T.emplace_back(i, n, -c(i));
      T.emplace_back(n, i, -c(i));
    }
    T.emplace_back(n, n, c.squaredNorm() - r * r);
    A.setFromTriplets(T.begin(), T.end());
    return A;
  };

  sdp.A.push_back(make_Q(c1, r1));
  sdp.b.push_back(0.0);

  // sdp.C = Matrix::Zero(dim, dim);
  // sdp.rho = 0.0;
  // sdp.A.push_back(make_Q(c2, r2));
  // sdp.b.push_back(0.0);

  // Set second sphere as cost
  sdp.C = make_Q(c2, r2);
  sdp.rho = 0.0;

  // t^2 = 1
  Eigen::SparseMatrix<double> At(dim, dim);
  At.insert(n, n) = 1.0;
  sdp.A.push_back(At);
  sdp.b.push_back(1.0);

  // Generate low rank solution
  auto weights = Vector::Zero(n - 1).eval();
  weights(0) = 1.0;
  sdp.soln = make_two_sphere_soln(r1, r2, d, weights);

  sdp.name = "TwoSphereDim" + std::to_string(n);

  return sdp;
}

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

// Test Certificate
TEST_P(InflationParamTest, Certificate) {
  const auto& sdp = GetParam();
  // parameters
  RankInflateParams params;
  params.verbose = true;
  params.max_sol_rank = sdp.dim;
  // generate problem
  RankInflation problem = sdp.make(params);
  // get current solution
  Matrix Y_0 = sdp.make_solution(params.max_sol_rank);
  // Run rank inflation, without inflation (target rank is 1)
  auto [Y, Jac] = problem.inflate_solution(Y_0);
  // Build certificate
  auto H = problem.build_certificate(Jac, Y);
  // std::cout << "Certificate Matrix: " << std::endl << H << std::endl;
  // check certificate on high rank solution
  auto [min_eig_hr, first_ord_cond_hr] = problem.check_certificate(H, Y);
  std::cout << "Certificate on High Rank Solution: " << std::endl;
  std::cout << "Minimum Eigenvalue of Certificate: " << min_eig_hr << std::endl;
  std::cout << "First Order Condition Norm: " << first_ord_cond_hr << std::endl;
  std::cout << "Cost at High Rank Solution: " << std::endl
            << (Y.transpose() * sdp.C * Y).trace() << std::endl;
  // check certificate on initial solution
  auto [min_eig, first_ord_cond] = problem.check_certificate(H, Y_0);
  std::cout << "Certificate on Initial Solution: " << std::endl;
  std::cout << "Minimum Eigenvalue of Certificate: " << min_eig << std::endl;
  std::cout << "First Order Condition Norm: " << first_ord_cond << std::endl;
}

TEST_P(InflationParamTest, CertWithCenter) {
  const auto& sdp = GetParam();
  // parameters
  RankInflateParams params;
  params.verbose = true;
  params.max_sol_rank = sdp.dim;
  params.delta_ac = 1e-7;
  // generate problem
  RankInflation problem = sdp.make(params);
  // get current solution
  Matrix Y_0 = sdp.make_solution(params.max_sol_rank);
  // Run rank inflation, without inflation (target rank is 1)
  auto X = problem.get_analytic_center(Y_0 * Y_0.transpose());
  Matrix Y = recover_lowrank_factor(X);
  auto Jac = Matrix(problem.m, Y.cols() * sdp.dim);
  auto violation = problem.eval_constraints(Y, Jac);
  // Build certificate
  auto H = problem.build_certificate(Jac, Y);
  // std::cout << "Certificate Matrix: " << std::endl << H << std::endl;
  // check certificate on high rank solution
  auto [min_eig_hr, first_ord_cond_hr] = problem.check_certificate(H, Y);
  std::cout << "Certificate on High Rank Solution: " << std::endl;
  std::cout << "Minimum Eigenvalue of Certificate: " << min_eig_hr << std::endl;
  std::cout << "First Order Condition Norm: " << first_ord_cond_hr << std::endl;
  std::cout << "Cost at High Rank Solution: " << std::endl
            << (Y.transpose() * sdp.C * Y).trace() << std::endl;
  // check certificate on initial solution
  auto [min_eig, first_ord_cond] = problem.check_certificate(H, Y_0);
  std::cout << "Certificate on Initial Solution: " << std::endl;
  std::cout << "Minimum Eigenvalue of Certificate: " << min_eig << std::endl;
  std::cout << "First Order Condition Norm: " << first_ord_cond << std::endl;
}

// Test Analytic Centering when one initialized column is zero
TEST_P(InflationParamTest, AnalyticCenter) {
  const auto& sdp = GetParam();
  // parameters
  RankInflateParams params;
  params.verbose = true;
  params.delta_ac = 1e-7;
  // generate problem
  auto problem = sdp.make(params);
  auto Y = sdp.soln;
  // Compute Analyic center starting from low rank solution
  auto X0 = Y * Y.transpose();
  auto X = problem.get_analytic_center(X0);
  // Compute analytic center objecive value
  double obj_0 = problem.get_analytic_center_objective(X0);
  double obj_star = problem.get_analytic_center_objective(X);
  // Check objective decrease
  std::cout << "Analytic Center Objective initially: " << obj_0 << std::endl;
  std::cout << "Analytic Center Objective at Low Rank Init: " << obj_star
            << std::endl;
  EXPECT_LT(obj_star, obj_0) << "Analytic center objective did not improve";
  // Check for rank increase
  auto rank_0 = get_rank(X0, 1e-6);
  auto rank_star = get_rank(X, 1e-6);
  std::cout << "Rank at Init: " << rank_0 << ", Rank at Center: " << rank_star
            << std::endl;
  EXPECT_GE(rank_star, rank_0) << "Rank did not increase at analytic center";
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
  auto Y = recover_lowrank_factor(X0);
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

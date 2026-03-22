/*
Test file for Analytic Center implementation.
*/
#include "circle_problem.hpp"
#include "generic_sdp_problems.hpp"
#include "interior_point_sdp.hpp"
#include "lin_alg_tools.hpp"
#include "lovasz_theta_problems.hpp"

using namespace RankTools;

// Fixture Classes
class LovazsParamTest : public ::testing::TestWithParam<SDPTestProblem> {};

class GenericParamTest : public ::testing::TestWithParam<SDPTestProblem> {};

// Test that centering works as expected
TEST_P(LovazsParamTest, PrimalSolution) {
  const auto& sdp = GetParam();
  // parameters
  AnalyticCenterParams params;
  params.verbose = true;
  params.check_cert = false;  // Turn off early stopping based on certificate
                              // for testing purposes
  params.max_iter = 50;
  params.rescale_lin_sys = false;
  double delta = 1e-7;
  // generate problem
  auto problem = sdp.make_testable(params);
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
  EXPECT_NEAR(diff_norm, 0.0, 1.0)
      << "Analytic center solution is not close to Mosek solution";
}

// Test early stopping based on certificate found.
TEST_P(LovazsParamTest, CertEarlyStopping) {
  const auto& sdp = GetParam();
  // parameters
  AnalyticCenterParams params;
  params.verbose = true;
  params.check_cert = true;
  params.rescale_lin_sys = true;
  params.lin_solver = LinearSolverType::MFCG_DP;
  auto delta = 1e-5;
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
  auto [min_eig_hr, first_ord_cond_hr] = problem.eval_certificate(H, Y);
  std::cout << "Cost at High Rank Solution: " << (sdp.C * X).trace()
            << std::endl;
  std::cout << "Minimum Eigenvalue of Certificate: " << min_eig_hr << std::endl;
  std::cout << "First Order Condition Norm at High Rank Solution: "
            << first_ord_cond_hr << std::endl;
  // check certificate on initial solution
  auto [min_eig, first_ord_cond] = problem.eval_certificate(H, Y_0);
  std::cout << "First Order Condition Norm at Rank 1 Solution: "
            << first_ord_cond << std::endl;
}

// Test the Fixed perturbation and verify that the method converges.
TEST_P(LovazsParamTest, CertifyFixedPerturb) {
  const auto& sdp = GetParam();
  // parameters
  AnalyticCenterParams params;
  params.verbose = true;
  params.check_cert = true;  // Turn off early stopping based on certificate for
                             // testing purposes
  params.adaptive_perturb =
      false;  // Turn off adaptive perturbation for testing purposes
  auto delta = 1e-7;
  // generate problem
  AnalyticCenter problem = sdp.make(params);
  // get current solution
  Matrix Y_0 = sdp.make_solution(1);
  // Run certification method
  auto result = problem.certify(Y_0, delta);
  // check that the solution is certified
  EXPECT_TRUE(result.certified) << "Analytic center failed to certify solution";
  std::cout << "Minimum Eigenvalue of Certificate: " << result.min_eig
            << std::endl;
  std::cout << "Complementarity (First Order Condition): "
            << result.complementarity << std::endl;
}

// Test adapative perturbation. Ensure that we converge to a certificate.
TEST_P(LovazsParamTest, CertifyAdaptivePerturb) {
  const auto& sdp = GetParam();
  // parameters
  AnalyticCenterParams params;
  params.verbose = true;
  params.check_cert = false;  // Turn off early stopping based on certificate
                              // for testing purposes
  params.adaptive_perturb =
      true;  // Turn on adaptive perturbation for testing purposes
  params.delta_min = 1e-9;
  params.max_iter = 100;
  // use rescaling to be consistent with the system in Sremac 2021
  params.rescale_lin_sys = true;
  auto delta = 1e-5;
  // generate problem
  AnalyticCenter problem = sdp.make(params);
  // get current solution
  Matrix Y_0 = sdp.make_solution(1);
  // Run certification method
  auto result = problem.certify(Y_0, delta);
  // check that the solution is certified
  EXPECT_TRUE(result.certified) << "Analytic center failed to certify solution";
  std::cout << "Minimum Eigenvalue of Certificate: " << result.min_eig
            << std::endl;
  std::cout << "Complementarity (First Order Condition): "
            << result.complementarity << std::endl;
}

// Test the LDLT linear solver on the system for computing multipliers
TEST_P(LovazsParamTest, CertifyLDLT) {
  const auto& sdp = GetParam();
  // parameters
  AnalyticCenterParams params;
  params.verbose = true;
  params.check_cert = false;  // Turn off early stopping based on certificate
                              // for testing purposes
  params.adaptive_perturb =
      true;  // Turn on adaptive perturbation for testing purposes
  params.delta_min = 1e-8;
  params.max_iter = 50;
  // use rescaling to be consistent with the system in Sremac 2021
  params.rescale_lin_sys = true;
  params.lin_solver = LinearSolverType::LDLT;  // Use Conjugate Gradient solver
  auto delta = 1e-5;
  // generate problem
  AnalyticCenter problem = sdp.make(params);
  // get current solution
  Matrix Y_0 = sdp.make_solution(1);
  // Run certification method
  auto result = problem.certify(Y_0, delta);
  // check that the solution is certified
  EXPECT_TRUE(result.certified) << "Analytic center failed to certify solution";
  std::cout << "Minimum Eigenvalue of Certificate: " << result.min_eig
            << std::endl;
  std::cout << "Complementarity (First Order Condition): "
            << result.complementarity << std::endl;
}

// Test the Conjugate Gradient linear solver on the system for computing
// multipliers
TEST_P(LovazsParamTest, CertifyConjGrad) {
  const auto& sdp = GetParam();
  // parameters
  AnalyticCenterParams params;
  params.verbose = true;
  params.check_cert = false;  // Turn off early stopping based on certificate
                              // for testing purposes
  params.adaptive_perturb =
      true;  // Turn on adaptive perturbation for testing purposes
  params.delta_min = 1e-8;
  params.max_iter = 50;
  // use rescaling to be consistent with the system in Sremac 2021
  params.rescale_lin_sys = true;
  params.lin_solver = LinearSolverType::CG;  // Use Conjugate Gradient solver
  auto delta = 1e-5;
  // generate problem
  AnalyticCenter problem = sdp.make(params);
  // get current solution
  Matrix Y_0 = sdp.make_solution(1);
  // Run certification method
  auto result = problem.certify(Y_0, delta);
  // check that the solution is certified
  EXPECT_TRUE(result.certified) << "Analytic center failed to certify solution";
  std::cout << "Minimum Eigenvalue of Certificate: " << result.min_eig
            << std::endl;
  std::cout << "Complementarity (First Order Condition): "
            << result.complementarity << std::endl;
}

// Test the Cholesky factorization used for line search and certificate
// checking.
TEST(General, CholeskyFactorization) {
  // Load a test problem
  auto sdp = make_lovasz_test_case(clique1_adj, {1, 3, 4, 6, 7, 8}, "Clique1");
  // parameters
  AnalyticCenterParams params;
  params.verbose = true;
  params.rescale_lin_sys =
      true;  // Use rescaling for consistency with Sremac 2021
  auto delta = 1e-7;
  // generate problem
  auto problem = sdp.make_testable(params);
  // get current solution
  Matrix Y_0 = sdp.make_solution(1);
  Matrix X = Y_0 * Y_0.transpose();
  auto [alpha, L] = problem.line_search_factorization(
      X, Matrix::Identity(problem.dim, problem.dim) * delta);
  // Verify that L*L^T = X
  Matrix reconstructed = L * L.transpose();
  double reconstruction_error = (reconstructed - X).norm();
  EXPECT_NEAR(reconstruction_error, 0.0, 1e-10) << "L*L^T does not equal X";
}

// Test the matrix-free operator product
TEST(MatrixFree, Product) {
  // Load a test problem
  auto sdp = make_lovasz_test_case(clique1_adj, {1, 3, 4, 6, 7, 8}, "Clique1");
  // parameters
  AnalyticCenterParams params;
  params.verbose = true;
  params.rescale_lin_sys =
      true;  // Use rescaling for consistency with Sremac 2021
  auto delta = 1e-7;
  // generate problem
  auto problem = sdp.make_testable(params);
  // get current solution
  Matrix Y_0 = sdp.make_solution(1);
  Matrix X = Y_0 * Y_0.transpose();
  auto [alpha, L] = problem.line_search_factorization(
      X, Matrix::Identity(problem.dim, problem.dim) * delta);
  // Build the explicit system to get the true diagonal of B
  auto system = problem.build_ac_system(X, L, delta);
  // Build the matrix-free operator
  MultiplierLinSys lin_op(system.LAL, 1 / delta);
  // Test on columns of identity
  auto Id = Matrix::Identity(sdp.dim, sdp.dim);
  for (int i = 0; i < sdp.dim; i++) {
    // Compute the matrix-free product
    Vector mf_product = lin_op * Id.col(i);
    // Compute the explicit product using the system matrix
    Vector explicit_product =
        system.B.selfadjointView<Eigen::Upper>() * Id.col(i);
    // Compare the results
    double tol = 1e-12;
    ASSERT_EQ(mf_product.size(), explicit_product.size());
    for (int j = 0; j < mf_product.size(); ++j) {
      double val = std::abs(explicit_product(j));
      EXPECT_NEAR(mf_product(j), explicit_product(j), tol + tol * val)
          << "Matrix-free product does not match explicit product at index ()"
          << i << " , " << j << ")";
    }
  }
}

TEST(MatrixFree, DiagonalPreconditioner) {
  // Load a test problem
  auto sdp = make_lovasz_test_case(clique1_adj, {1, 3, 4, 6, 7, 8}, "Clique1");
  // parameters
  AnalyticCenterParams params;
  params.verbose = true;
  params.rescale_lin_sys = true;
  auto delta = 1e-5;
  // generate problem
  auto problem = sdp.make_testable(params);
  // get current solution
  Matrix Y_0 = sdp.make_solution(1);
  Matrix X = Y_0 * Y_0.transpose();
  auto [alpha, L] = problem.line_search_factorization(
      X, Matrix::Identity(problem.dim, problem.dim) * delta);
  // Build the explicit system to get the true diagonal of B
  auto system = problem.build_ac_system(X, L, delta);
  // Build the matrix-free operator and preconditioner
  MultiplierLinSys lin_op(system.LAL, 1 / delta);
  MultiplierDiagPreconditioner precond;
  precond.compute(lin_op);
  // Check that the preconditioner computed successfully
  EXPECT_EQ(precond.info(), Eigen::Success);
  // Verify the preconditioner diagonal matches the true diagonal of B
  // The preconditioner stores 1/diag, so applying it to e_i gives 1/B(i,i)
  double tol = 1e-10;
  for (int i = 0; i < problem.m; ++i) {
    // Extract true diagonal from the explicit system (upper triangle storage)
    double B_ii = system.B(i, i);
    // Apply preconditioner to a unit vector to extract 1/B(i,i)
    Vector e_i = Vector::Zero(problem.m);
    e_i(i) = 1.0;
    Vector precond_ei = precond.solve(e_i);
    double precond_diag = precond_ei(i);
    double expected_inv = (B_ii > 1e-14) ? 1.0 / B_ii : 1.0;
    EXPECT_NEAR(precond_diag, expected_inv, tol)
        << "Preconditioner diagonal does not match B(" << i << "," << i << ")";
  }

  // Verify that the preconditioned CG solves the system correctly
  Eigen::ConjugateGradient<MultiplierLinSys, Eigen::Upper | Eigen::Lower,
                           MultiplierDiagPreconditioner>
      pcg;
  pcg.compute(lin_op);
  EXPECT_EQ(pcg.info(), Eigen::Success);
  // Solve B * x = d using preconditioned CG
  Vector rhs = Vector::Random(problem.m);
  Vector x_pcg = pcg.solve(rhs);
  EXPECT_EQ(pcg.info(), Eigen::Success)
      << "Preconditioned CG failed to converge";
  // Check residual
  Vector residual = lin_op * x_pcg - rhs;
  EXPECT_LT(residual.norm() / rhs.norm(), 1e-6)
      << "Preconditioned CG residual too large";
}

// Test the low-rank preconditioner and verify that it improves conditioning of
// the system.
TEST_P(LovazsParamTest, LowRankPrecond) {
  const auto& sdp = GetParam();
  // parameters
  AnalyticCenterParams params;
  params.verbose = true;
  params.rescale_lin_sys = false;
  auto delta = 1e-8;

  // Build system at perturbed rank-1 solution X = Y Y^T + delta I
  auto problem = sdp.make_testable(params);

  Matrix Y_0 = sdp.make_solution(1);
  Matrix X = Y_0 * Y_0.transpose();
  auto [alpha, L] = problem.line_search_factorization(
      X, Matrix::Identity(problem.dim, problem.dim) * delta);
  EXPECT_NEAR((X - Y_0 * Y_0.transpose() -
               Matrix::Identity(problem.dim, problem.dim) * delta)
                  .norm(),
              0.0, 1e-10);

  // Explicit system matrix B
  auto system = problem.build_ac_system(X, L, delta);
  Matrix B = system.B.selfadjointView<Eigen::Upper>();

  // Inspect conditioning of the original system matrix B.
  Eigen::SelfAdjointEigenSolver<Matrix> es_B(B);
  ASSERT_EQ(es_B.info(), Eigen::Success);
  auto eigvals_B = es_B.eigenvalues();
  const double max_eig_B = eigvals_B.maxCoeff();
  const double min_eig_B = eigvals_B.minCoeff();
  const double cond_B = max_eig_B / std::max(min_eig_B, 1e-16);
  std::cout << "cond(B): " << cond_B << std::endl;

  // Build low-rank preconditioner with rank = 1 at the perturbed solution
  auto precond = LowRankPrecond();
  precond.initialize(X, sdp.A, sdp.C, 1, true);
  // Test descomposition
  auto [U, W0, tau] = precond.decompose_soln(X);
  EXPECT_NEAR(tau, delta, 1e-10);
  // Verify that W0 + U*U.transpose() == X
  Matrix reconstructed = W0 + U * U.transpose();
  double reconstruction_error = (reconstructed - X).norm();
  EXPECT_NEAR(reconstruction_error, 0.0, 1e-10)
      << "W0 + U*U^T does not equal X";

  // Compute preconditioning precursers
  EXPECT_EQ(precond.info(), Eigen::Success);

  // Build explicit preconditioned operator PB by applying P to each column of
  // B. For a good preconditioner, PB should be well-conditioned.
  Matrix PB(problem.m, problem.m);
  for (int i = 0; i < problem.m; ++i) {
    PB.col(i) = precond.solve(B.col(i));
  }

  // Inspect spectrum of PB.
  Eigen::EigenSolver<Matrix> es(PB);
  ASSERT_EQ(es.info(), Eigen::Success);
  auto eigvals = es.eigenvalues();

  double min_abs_eig = std::numeric_limits<double>::infinity();
  double max_abs_eig = 0.0;
  for (int i = 0; i < eigvals.size(); ++i) {
    const double abs_eig = std::abs(eigvals(i));
    min_abs_eig = std::min(min_abs_eig, abs_eig);
    max_abs_eig = std::max(max_abs_eig, abs_eig);
  }

  // Condition number estimate from eigenvalue magnitudes.
  const double cond_est = max_abs_eig / min_abs_eig;
  std::cout << "cond(PB): " << cond_est << std::endl;

  // Well-preconditioned systems should have condition number close to 1.
  EXPECT_LT(cond_est, 10.0)
      << "Preconditioned operator PB is poorly conditioned.";
}

TEST_P(LovazsParamTest, CertifyMFCGDiagPrecond) {
  const auto& sdp = GetParam();
  // parameters
  AnalyticCenterParams params;
  params.verbose = true;
  params.check_cert = false;  // Turn off early stopping based on certificate
                              // for testing purposes
  params.adaptive_perturb =
      true;  // Turn on adaptive perturbation for testing purposes
  params.delta_min = 1e-8;
  params.max_iter = 50;
  // use rescaling to be consistent with the system in Sremac 2021
  params.rescale_lin_sys = true;
  params.lin_solver =
      LinearSolverType::MFCG_DP;  // Use Conjugate Gradient solver with diagonal preconditioner
  auto delta = 1e-5;
  // generate problem
  AnalyticCenter problem = sdp.make(params);
  // get current solution
  Matrix Y_0 = sdp.make_solution(1);
  // Run certification method
  auto result = problem.certify(Y_0, delta);
  // check that the solution is certified
  EXPECT_TRUE(result.certified) << "Analytic center failed to certify solution";
  std::cout << "Minimum Eigenvalue of Certificate: " << result.min_eig
            << std::endl;
  std::cout << "Complementarity (First Order Condition): "
            << result.complementarity << std::endl;
}

// Run with the low-rank preconditioner and the Conjugate Gradient solver.
// Note that this only works when the candidate solution is at the analytic center,
// so we use the Mosek solution as the candidate solution to certify.
TEST_P(GenericParamTest, Certify_MFCG_LRP_Global) {
  const auto& sdp = GetParam();
  // Solve using Mosek to get analytic center solution
  auto mosek_soln = solve_sdp_mosek(sdp.C, sdp.A, sdp.b);
  auto Y_mosek = get_positive_eigspace(mosek_soln.X, 1e-3);
  auto rank_mosek = Y_mosek.cols();
  std::cout << "Rank at IP Solution: " << rank_mosek << std::endl;
  auto X_mosek = Y_mosek * Y_mosek.transpose();
    
  // parameters
  AnalyticCenterParams params;
  params.verbose = true;
  params.check_cert = true;  // Turn off early stopping based on certificate
  // for testing purposes
  params.adaptive_perturb =
  true;  // Turn on adaptive perturbation for testing purposes
  params.delta_min = 1e-7;
  params.max_iter = 50;
  // Turn off rescaling (preconditioner should deal with this)
  params.rescale_lin_sys = false;
  params.lin_solver =
  LinearSolverType::MFCG_LRP;  // Use Conjugate Gradient solver
  
  auto delta = 1e-5;
  // generate problem
  AnalyticCenter problem = sdp.make(params);
  // Update cost based on the mosek solution
  problem.rho_ = (sdp.C * X_mosek).trace();
  // // Check mosek solution
  // std::cout << "Cost diff of solution: " << problem.rho_ - mosek_soln.obj_value << std::endl;
  // std::cout << "Violation of solution: " << problem.eval_constraints(X_mosek).transpose() << std::endl;
  // std::cout << "Complementarity of solution: " << (mosek_soln.S * X_mosek).trace() << std::endl;
  // Run certification method
  auto result = problem.certify(Y_mosek, delta);
  // check that the solution is certified
  EXPECT_TRUE(result.certified) << "Analytic center failed to certify solution";

  std::cout << "Minimum Eigenvalue of Certificate: " << result.min_eig
            << std::endl;
  std::cout << "Complementarity (First Order Condition): "
            << result.complementarity << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    AnalyticCenterSuite, LovazsParamTest,
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
    [](const ::testing::TestParamInfo<LovazsParamTest::ParamType>& info) {
      return info.param.name;
    });

static std::vector<SDPTestProblem> get_small_exported_cases() {
  auto all = ExportedSDPProblems::make_exported_sdp_test_problems();
  std::vector<SDPTestProblem> selected;
  selected.reserve(30);

  for (const auto& sdp : all) {
    if (sdp.A.size() <= 100) {
      selected.push_back(sdp);
    }
    if (selected.size() >= 30) {
      break;
    }
  }
  if (selected.empty()) {
    const std::size_t n = std::min<std::size_t>(all.size(), 5);
    selected.insert(selected.end(), all.begin(), all.begin() + n);
  }

  return selected;
}

INSTANTIATE_TEST_SUITE_P(
    AnalyticCenterSuite, GenericParamTest,
    ::testing::ValuesIn(get_small_exported_cases()),
    [](const ::testing::TestParamInfo<SDPTestProblem>& info) {
      std::string s = info.param.name;
      for (char& c : s) {
        if (!std::isalnum(static_cast<unsigned char>(c))) {
          c = '_';
        }
      }
      return s;
    });

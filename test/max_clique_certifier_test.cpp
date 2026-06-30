/*
Test file for the MaxCliqueCertifier.

The MaxCliqueCertifier behaves exactly like AnalyticCenter, except that the
constraint matrices A and right-hand side b are built internally from the cost
matrix M (one zero-constraint per off-diagonal zero of M, plus a trace
constraint). These tests verify that, for the Lovasz-theta / max-clique
problems, certifying with MaxCliqueCertifier produces identical output to an
AnalyticCenter constructed with the equivalent, explicitly supplied constraints.
*/
#include "interior_point_sdp.hpp"
#include "lin_alg_tools.hpp"
#include "lovasz_theta_problems.hpp"
#include "max_clique_certifier.hpp"

using namespace RankTools;

// Fixture parametrized over the Lovasz-theta test problems.
class MaxCliqueParamTest : public ::testing::TestWithParam<SDPTestProblem> {};

namespace {

// Build a max-clique cost matrix whose off-diagonal zero pattern encodes the
// graph non-edges of the given Lovasz-theta problem. Starting from a fully
// connected cost (-1 everywhere, no off-diagonal zeros), we zero out the
// entries that correspond to the non-edge constraints of the problem. The last
// constraint of the problem is the trace (identity) constraint and is skipped.
//
// build_problem inside MaxCliqueCertifier reconstructs one constraint per
// off-diagonal zero of this matrix, so the constraints it generates match the
// problem's non-edge constraints exactly.
Matrix make_max_clique_cost(const SDPTestProblem& sdp) {
  Matrix M = -Matrix::Ones(sdp.dim, sdp.dim);
  for (std::size_t k = 0; k + 1 < sdp.A.size(); ++k) {
    for (int col = 0; col < sdp.A[k].outerSize(); ++col) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(sdp.A[k], col); it;
           ++it) {
        M(it.row(), it.col()) = 0.0;
        M(it.col(), it.row()) = 0.0;
      }
    }
  }
  return M;
}

}  // namespace

// Verify that the constraints MaxCliqueCertifier builds internally from the cost
// matrix match the constraints of the equivalent AnalyticCenter problem.
TEST_P(MaxCliqueParamTest, BuildsEquivalentConstraints) {
  const auto& sdp = GetParam();
  Matrix M = make_max_clique_cost(sdp);

  AnalyticCenterParams params;
  params.verbose = false;

  AnalyticCenter reference(M, sdp.rho, sdp.A, sdp.b, params);
  MaxCliqueCertifier certifier(M, sdp.rho, params);

  // Same number of constraints (+ cost) and same right-hand side count.
  ASSERT_EQ(certifier.m, reference.m);
  ASSERT_EQ(certifier.A_.size(), reference.A_.size());
  ASSERT_EQ(certifier.b_.size(), reference.b_.size());

  // The constraints are generated in the same (ascending) order, so they should
  // match element-by-element.
  for (std::size_t k = 0; k < reference.A_.size(); ++k) {
    EXPECT_NEAR((Matrix(certifier.A_[k]) - Matrix(reference.A_[k])).norm(), 0.0,
                1e-12)
        << "Constraint " << k << " differs";
    EXPECT_NEAR(certifier.b_[k], reference.b_[k], 1e-12)
        << "RHS " << k << " differs";
  }
}

// Main test: run certify with MaxCliqueCertifier and verify the output matches
// the AnalyticCenter class. This mirrors AnalyticCenter's Certify_MFCG_LRP_Global
// test: we obtain the analytic-center candidate from Mosek, then certify the
// same candidate with both classes and compare every output field.
TEST_P(MaxCliqueParamTest, CertifyMatchesAnalyticCenter) {
  const auto& sdp = GetParam();
  // Cost matrix whose off-diagonal zero pattern encodes the graph non-edges.
  Matrix M = make_max_clique_cost(sdp);

  // Solve using Mosek to get the analytic-center solution to certify.
  auto mosek_soln = solve_sdp_mosek(M, sdp.A, sdp.b);
  auto Y_mosek = get_positive_eigspace(mosek_soln.X, 1e-3);
  std::cout << "Rank at IP Solution: " << Y_mosek.cols() << std::endl;
  const double rho_star = (M * mosek_soln.X).trace();

  // parameters (matching Certify_MFCG_LRP_Global)
  AnalyticCenterParams params;
  params.verbose = true;
  params.early_stop_cert = true;
  params.adaptive_perturb = true;
  params.eps_mult_min = 1e-2;
  params.max_iter = 50;
  params.lin_solver = LinearSolverType::MFCG_LRP;
  params.lrp_params.tau = 1e-5;
  params.lrp_params.method = LowRankPrecondMethod::SparseLDLT;
  params.delta = 1e-5;

  // Reference: plain AnalyticCenter built from the explicit constraints.
  AnalyticCenter reference(M, sdp.rho, sdp.A, sdp.b, params);
  reference.rho_ = rho_star;

  // Subject under test: constraints built internally from the cost matrix.
  MaxCliqueCertifier certifier(M, sdp.rho, params);
  certifier.rho_ = rho_star;

  auto ref_result = certifier.certify(Y_mosek);  // exercise certifier first
  auto base_result = reference.certify(Y_mosek);

  // Both classes should produce identical certification output.
  const double tol = 1e-10;
  EXPECT_NEAR((ref_result.X - base_result.X).norm(), 0.0, tol)
      << "Certified primal solutions differ";
  EXPECT_NEAR((ref_result.H - base_result.H).norm(), 0.0, tol)
      << "Certificate matrices differ";
  EXPECT_NEAR((ref_result.multipliers - base_result.multipliers).norm(), 0.0,
              tol)
      << "Multipliers differ";
  EXPECT_NEAR(ref_result.violation.norm(), base_result.violation.norm(), tol)
      << "Constraint violations differ";
  EXPECT_NEAR(ref_result.min_eig, base_result.min_eig, tol)
      << "Minimum eigenvalues differ";
  EXPECT_NEAR(ref_result.complementarity, base_result.complementarity, tol)
      << "Complementarity values differ";
  EXPECT_EQ(ref_result.certified, base_result.certified)
      << "Certification status differs";

  // The Mosek solution is globally optimal for this problem, so it should
  // certify with both classes.
  EXPECT_TRUE(ref_result.certified)
      << "MaxCliqueCertifier failed to certify solution";
  EXPECT_TRUE(base_result.certified)
      << "AnalyticCenter failed to certify solution";

  std::cout << "Minimum Eigenvalue of Certificate: " << ref_result.min_eig
            << std::endl;
  std::cout << "Complementarity (First Order Condition): "
            << ref_result.complementarity << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    MaxCliqueSuite, MaxCliqueParamTest,
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
    [](const ::testing::TestParamInfo<MaxCliqueParamTest::ParamType>& info) {
      return info.param.name;
    });

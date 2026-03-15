#include "line_search_cert.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>

#include "generic_sdp_problems.hpp"
#include "test_harness.hpp"

using namespace RankTools;

class LineCertifierTestable : public LineCertifier {
 public:
  using LineCertifier::build_adjoint;
  using LineCertifier::LineCertifier;
};

class LineSearchParamTest : public ::testing::TestWithParam<SDPTestProblem> {};

static std::vector<SDPTestProblem> get_small_exported_cases() {
  auto all = ExportedSDPProblems::make_exported_sdp_test_problems();
  std::vector<SDPTestProblem> selected;
  selected.reserve(10);

  for (const auto& sdp : all) {
    if (sdp.dim <= 25) {
      selected.push_back(sdp);
    }
    if (selected.size() >= 10) {
      break;
    }
  }

  if (selected.empty()) {
    const std::size_t n = std::min<std::size_t>(all.size(), 5);
    selected.insert(selected.end(), all.begin(), all.begin() + n);
  }

  return selected;
}

static LineCertifierParams make_params() {
  LineCertifierParams p;
  p.verbose = false;
  p.tol_cert_psd = 1e-9;
  p.tol_cert_complementarity = 1e-8;
  return p;
}

TEST_P(LineSearchParamTest, InteriorFactorContainsCandidateColumns) {
  const auto& sdp = GetParam();
  auto params = make_params();

  LineCertifier cert(sdp.C, sdp.rho, sdp.A, sdp.b, sdp.soln, params);
  Matrix V = cert.get_interior_factor(sdp.soln);

  ASSERT_EQ(V.rows(), sdp.soln.rows());
  ASSERT_EQ(V.cols(), sdp.soln.rows());
  ASSERT_GE(V.cols(), sdp.soln.cols());

  const double diff = (V.leftCols(sdp.soln.cols()) - sdp.soln).norm();
  EXPECT_NEAR(diff, 0.0, 1e-10);
}

TEST_P(LineSearchParamTest, ScalingDiagonalHasExpectedPattern) {
  const auto& sdp = GetParam();
  auto params = make_params();

  LineCertifier cert(sdp.C, sdp.rho, sdp.A, sdp.b, sdp.soln, params);
  const double alpha = 0.2;
  auto D = cert.get_scaling_diagonal(alpha);
  Vector diag = D.diagonal();

  ASSERT_EQ(diag.size(), sdp.dim);
  const int k = std::min<int>(sdp.soln.cols(), sdp.dim);
  for (int i = 0; i < k; ++i) {
    EXPECT_NEAR(diag(i), 1.0 - alpha, 1e-12);
  }
  for (int i = k; i < sdp.dim; ++i) {
    EXPECT_NEAR(diag(i), alpha, 1e-12);
  }
}

TEST_P(LineSearchParamTest, BuildAdjointMatchesManualSum) {
  const auto& sdp = GetParam();
  auto params = make_params();

  LineCertifierTestable cert(sdp.C, sdp.rho, sdp.A, sdp.b, sdp.soln, params);

  Vector lambda = Vector::Zero(static_cast<int>(sdp.A.size()));
  for (int i = 0; i < lambda.size(); ++i) {
    lambda(i) = 0.01 * static_cast<double>(i + 1);
  }

  Matrix expected = Matrix::Zero(sdp.dim, sdp.dim);
  for (int i = 0; i < static_cast<int>(sdp.A.size()); ++i) {
    expected += lambda(i) * Matrix(sdp.A[i]);
  }

  Matrix actual = Matrix(cert.build_adjoint(lambda));
  EXPECT_NEAR((actual - expected).norm(), 0.0, 1e-9);
}

TEST_P(LineSearchParamTest, CertificateFromDualMatchesCostPlusAdjoint) {
  const auto& sdp = GetParam();
  auto params = make_params();

  LineCertifierTestable cert(sdp.C, sdp.rho, sdp.A, sdp.b, sdp.soln, params);

  Vector lambda = Vector::Zero(static_cast<int>(sdp.A.size()));
  for (int i = 0; i < lambda.size(); ++i) {
    lambda(i) = (i % 2 == 0 ? 1.0 : -1.0) * 0.005 * static_cast<double>(i + 1);
  }

  Matrix H = cert.build_certificate_from_dual(lambda);
  Matrix expected = sdp.C + Matrix(cert.build_adjoint(lambda));
  EXPECT_NEAR((H - expected).norm(), 0.0, 1e-9);
}

TEST_P(LineSearchParamTest, EvalAndCheckCertificateAgreeOnComplementarity) {
  const auto& sdp = GetParam();
  auto params = make_params();

  LineCertifier cert(sdp.C, sdp.rho, sdp.A, sdp.b, sdp.soln, params);

  Matrix H = Matrix::Identity(sdp.dim, sdp.dim);
  auto [min_eig, comp_eval] = cert.eval_certificate(H, sdp.soln);
  auto [psd_flag_as_double, comp_check] = cert.check_certificate(H, sdp.soln);

  EXPECT_NEAR(min_eig, 1.0, 1e-12);
  EXPECT_NEAR(comp_eval, comp_check, 1e-12);
  EXPECT_EQ(static_cast<int>(psd_flag_as_double), 1);
}

INSTANTIATE_TEST_SUITE_P(
    ExportedProblems, LineSearchParamTest,
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

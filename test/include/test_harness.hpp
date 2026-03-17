// This file defines the test harness for SDP unit tests, including the test case
// data structure and a subclass of AnalyticCenter that promotes protected
// methods to public for testing purposes.

#pragma once

#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#include "analytic_center.hpp"
#include "rank_inflation.hpp"

using namespace RankTools;

// Subclass that promotes every protected method to public so that unit tests
// can exercise internal logic without friend declarations in production code.
class AnalyticCenterTestable : public AnalyticCenter {
 public:
  using AnalyticCenter::analytic_center_bisect;
  using AnalyticCenter::analytic_center_line_search_func;
  using AnalyticCenter::AnalyticCenter;  // inherit all constructors
  using AnalyticCenter::build_ac_system;
  using AnalyticCenter::build_adjoint;
  using AnalyticCenter::get_analytic_center_objective;
  using AnalyticCenter::get_multipliers;
  using AnalyticCenter::solve_ac_system;
};

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

  AnalyticCenter make(const AnalyticCenterParams& params) const {
    return AnalyticCenter(C, rho, A, b, params);
  }

  inline RankInflation make(const RankInflateParams& params) const {
    return RankInflation(C, rho, A, b, params);
  }

  // Returns a testable subclass that exposes protected methods as public.
  inline auto make_testable(const AnalyticCenterParams& params) const {
    return AnalyticCenterTestable(C, rho, A, b, params);
  }
};

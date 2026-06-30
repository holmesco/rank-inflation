#pragma once
#include "analytic_center.hpp"

namespace RankTools {

// Certifier for the maximum clique / Lovasz-theta SDP.
//
// This class behaves exactly like AnalyticCenter, except that the constraint
// matrices A and right-hand side b are not supplied by the caller. Instead they
// are constructed from the cost matrix M following the same logic as
// get_lovasz_constraints: there is one constraint per non-edge enforcing that
// the corresponding off-diagonal entry of the solution is zero, plus a trace
// constraint fixing the trace to one. The non-edges are taken to be the
// off-diagonal indices of M whose entries are exactly equal to zero.
class MaxCliqueCertifier : public AnalyticCenter {
 public:
  MaxCliqueCertifier(const Matrix& M, double rho, AnalyticCenterParams params)
      : MaxCliqueCertifier(M, rho, params, build_problem(M)) {}

 private:
  // Constraint matrices A and right-hand side b for the SDP.
  struct Problem {
    std::vector<SpMatrix> A;
    std::vector<double> b;
  };

  // Delegated-to constructor that forwards the pre-built constraints to the base
  // class. This allows build_problem to be evaluated exactly once.
  MaxCliqueCertifier(const Matrix& M, double rho, AnalyticCenterParams params,
                     Problem problem)
      : AnalyticCenter(M, rho, problem.A, problem.b, params) {}

  // Build the constraint matrices and right-hand side from the cost matrix M.
  // The non-edges are the off-diagonal (i, j) indices (upper triangle) where
  // M(i, j) == 0. Each non-edge yields a constraint with a single unit entry at
  // (i, j) and a right-hand side of zero. A final identity matrix enforces the
  // trace constraint, with a right-hand side of one.
  static Problem build_problem(const Matrix& M) {
    const int dim = static_cast<int>(M.rows());
    Problem problem;
#ifdef RANKTOOLS_PARALLEL
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < dim; ++i) {
      for (int j = i + 1; j < dim; ++j) {
        if (M(i, j) == 0.0) {
          problem.A.emplace_back(dim, dim);
          std::vector<Triplet> triplets;
          triplets.push_back(Triplet(i, j, 1.0));
          problem.A.back().setFromTriplets(triplets.begin(), triplets.end());
          problem.b.push_back(0.0);
        }
      }
    }
    // Trace constraint.
    problem.A.emplace_back(dim, dim);
    problem.A.back().setIdentity();
    problem.b.push_back(1.0);
    return problem;
  }
};

}  // namespace RankTools

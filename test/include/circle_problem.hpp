#pragma once
#include "test_harness.hpp"

// ----------- Two Sphere Helper Functions --------------

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

  // Make the sphere constraint. Sphere centered at c with radius r is defined
  // by the quadratic constraint: A . X == r^2, where A is [I, -c; -c^T, c^T c]
  auto make_constraint = [dim, n](const Eigen::VectorXd& c) {
    Eigen::SparseMatrix<double> A(dim, dim);
    std::vector<Eigen::Triplet<double>> T;
    for (int i = 0; i < n; ++i) T.emplace_back(i, i, 1.0);  // x^T x
    for (int i = 0; i < n; ++i) {
      T.emplace_back(i, n, -c(i));
      T.emplace_back(n, i, -c(i));
    }
    T.emplace_back(n, n, c.squaredNorm());
    A.setFromTriplets(T.begin(), T.end());
    return A;
  };

  sdp.A.push_back(make_constraint(c1));
  sdp.b.push_back(r1 * r1);
  // sdp.A.push_back(make_constraint(c2));
  // sdp.b.push_back(r2*r2);
  // Set cost to second constraint.
  sdp.C = make_constraint(c2);
  sdp.rho = r2 * r2;

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
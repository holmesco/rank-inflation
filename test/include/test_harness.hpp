#pragma once

#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <utility>

#include "rank_inflation.hpp"

using namespace SDPTools;

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

// ----------- Lovasc Theta Helper Functions ----------------

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
  
  // Make the sphere constraint. Sphere centered at c with radius r is defined by the quadratic constraint:
  // A . X == r^2, where A is [I, -c; -c^T, c^T c]
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
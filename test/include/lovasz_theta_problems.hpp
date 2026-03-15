#pragma once
#include "test_harness.hpp"

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
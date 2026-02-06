#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

namespace SDPTools {
// Simplifying declarations for commonly used Eigen types in this project. This
// also allows us to easily change the underlying types if needed.
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Triplet = Eigen::Triplet<double>;
using SpMatrix = Eigen::SparseMatrix<double>;
using ScalarFunc = std::function<double(double)>;

// Result of solving a linear system using rank-revealing QR decomposition.
// Contains both the least-squares particular solution and the nullspace basis.
struct QRResult {
  Vector solution;
  Matrix nullspace_basis;
  int rank;
  Vector R_diagonal;
  double residual_norm;
  Eigen::ColPivHouseholderQR<Matrix> qr_decomp;
};

// Get the particular solution and null space of a system of linear equations
// using rank revealing QR decomposition This formulation is designed for dens
// matrices
QRResult get_soln_qr_dense(const Matrix& A, const Vector& b,
                           const double threshold);

// Compute the rank of a dense matrix with rank-revealing QR
int get_rank(const Matrix& Y, const double threshold);

// Use LDL^T factorization to recover a low-rank factor from a psd matrix.
// threshold: diagonal threshold for determining rank.
// Note: This is efficient when the rank is low compared to the size of the
// matrix.
Matrix recover_lowrank_factor(const Matrix& A, double threshold);

// Bisection line search to find root of scalar function df
double bisection_line_search(const ScalarFunc& df, double alpha_low,
                             double alpha_high, double tol);

inline double logdet(const Matrix& X) {
  // Compute log determinant via Cholesky decomposition
  Eigen::LLT<Matrix> lltOfX(X);
  if (lltOfX.info() != Eigen::Success) {
    return -std::numeric_limits<double>::infinity();
  }
  const Matrix& L = lltOfX.matrixL();
  double val = 2.0 * L.diagonal().array().log().sum();
  return val;
}
}  // namespace SDPTools
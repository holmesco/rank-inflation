// /workspace/src/interior_point_sdp.cpp
// Requires MOSEK Fusion C++ API and Eigen.
// Example usage:
//   Eigen::MatrixXd C = ...; // n x n
//   std::vector<Eigen::SparseMatrix<double>> As = ...; // each n x n
//   Eigen::VectorXd b = ...; // m
//   Eigen::MatrixXd X = solve_sdp(C, As, b);

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <memory>
#include <vector>

#include "fusion.h"  // MOSEK Fusion C++ header

namespace SDPTools {


// New API: returns primal X, equality constraint multipliers y, and PSD dual
// matrix S.
struct SDPResult {
  Eigen::MatrixXd X;  // primal
  Eigen::VectorXd y;  // multipliers for trace(A_i X) = b_i
  Eigen::MatrixXd S;  // dual matrix for PSD cone (C - sum_i y_i A_i)
};

/**
 * @brief Converts an Eigen::SparseMatrix<double> to a mosek::fusion::Matrix
 * (sparse).
 *
 * @param eigen_mat The input Eigen sparse matrix.
 * @return A shared pointer to the resulting MOSEK Fusion sparse matrix.
 */
mosek::fusion::Matrix::t eigenToMosekSparse(const Eigen::SparseMatrix<double, Eigen::ColMajor>& eigenMat);

/**
 * Converts a dense Eigen::MatrixXd to a mosek::fusion::Matrix::t
 */
mosek::fusion::Matrix::t eigenToMosekDense(const Eigen::MatrixXd& eigenMat);
}  // namespace SDPTools
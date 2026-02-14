#include "utils.hpp"

namespace SDPTools {

double bisection_line_search(const ScalarFunc& df, double alpha_low,
                             double alpha_high, double tol) {
  // Ensure that the upper bound is valid
  while (df(alpha_high) < 0) {
    alpha_low = alpha_high;
    alpha_high *= 2.0;  // expand search interval
  }
  // Bisection loop
  double alpha_mid;
  while ((alpha_high - alpha_low) > tol) {
    alpha_mid = 0.5 * (alpha_low + alpha_high);
    if (df(alpha_mid) > 0) {
      alpha_high = alpha_mid;
    } else {
      alpha_low = alpha_mid;
    }
  }
  return 0.5 * (alpha_low + alpha_high);
}

Matrix recover_lowrank_factor(const Matrix& A, double threshold) {
  // Use LDLT decomposition to get low-rank factors
  // NOTE: LDLT is used because it is stable for semi-definite matrices and
  // will effectively terminate when it encounters a max pivot that is
  // numerically zero.
  Eigen::LDLT<Matrix> ldlt(A);
  Matrix L = ldlt.matrixL();
  Vector D = ldlt.vectorD();
  // Determine rank based on positive pivots
  int rank = 0;
  for (int i = 0; i < D.size(); i++) {
    if (D(i) > threshold) {
      rank++;
    } else {
      break;
    }
  }
  // Build low-rank factor
  Matrix Y(A.rows(), rank);
  for (int i = 0; i < rank; i++) {
    Y.col(i) = L.col(i) * std::sqrt(D(i));
  }
  // Unpivot the factors
  auto P = ldlt.transpositionsP();
  Y = P.transpose() * Y;

  return Y;
}

Matrix get_positive_eigspace(const Matrix& mat, double threshold) {
  Eigen::SelfAdjointEigenSolver<Matrix> solver(mat);

  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("Eigenvalue decomposition failed.");
  }

  // Eigenvalues are sorted in increasing order
  const Vector& evals = solver.eigenvalues();
  const Matrix& evecs = solver.eigenvectors();

  // 1. Count positive eigenvalues
  int positiveCount = 0;
  for (int i = evals.size() - 1; i >= 0; --i) {
    if (evals[i] > threshold) {  // relative threshold to handle scale
      positiveCount++;
    } else {
      break;
    }
  }

  if (positiveCount == 0) return Matrix(mat.rows(), 0);

  // 2. Extract the positive eigenvalues and take their square root
  // .tail() gets the last 'positiveCount' elements (the largest ones)
  Vector sqrtEvals = evals.tail(positiveCount).array().sqrt();

  // 3. Extract the corresponding eigenvectors
  Matrix topEvecs = evecs.rightCols(positiveCount);

  // 4. Weight the columns: Each column i is multiplied by sqrt(lambda_i)
  // Using .asDiagonal() is computationally efficient in Eigen
  return topEvecs * sqrtEvals.asDiagonal();
}

int get_rank(const Matrix& mat, const double threshold) {
  // 1. Perform Column Pivoted Householder QR (Rank-Revealing)
  Eigen::ColPivHouseholderQR<Matrix> qr(mat);
  qr.setThreshold(threshold);
  return qr.rank();
}

QRResult get_soln_qr_dense(const Matrix& A, const Vector& b,
                           const double threshold) {
  // Perform Column Pivoted Householder QR (Rank-Revealing)
  Eigen::ColPivHouseholderQR<Matrix> qr(A);
  // Set threshold for nullspace
  qr.setThreshold(threshold);
  // get dimensions
  int m = A.rows();
  int n = A.cols();
  int r = qr.rank();

  QRResult result;
  result.qr_decomp = qr;
  result.rank = r;

  // 2. Find the particular solution
  // If the system is inconsistent, this provides a least-squares solution
  result.solution = qr.solve(b);

  // 3. Characterize the Null Space
  if (r >= n) {
    // Full column rank: Null space is just the zero vector
    result.nullspace_basis = Matrix::Zero(n, 0);
  } else {
    // Extract R and the permutation matrix P
    // R is stored in the upper triangle of the matrixQR
    Matrix R = qr.matrixQR().triangularView<Eigen::Upper>();
    Matrix P = qr.colsPermutation();

    // Partition R: R11 is (r x r), R12 is (r x n-r)
    Matrix R11 = R.block(0, 0, r, r);
    Matrix R12 = R.block(0, r, r, n - r);

    // Compute -R11^{-1} * R12
    // We use triangular solve for better numerical stability
    Matrix top = -R11.inverse() * R12;

    // Construct the basis in the permuted space: [ -R11^-1 * R12 ; I ]
    Matrix basisPermuted(n, n - r);
    basisPermuted.block(0, 0, r, n - r) = top;
    basisPermuted.block(r, 0, n - r, n - r) = Matrix::Identity(n - r, n - r);

    // Transform back to original space using the permutation matrix P
    result.nullspace_basis = P * basisPermuted;
  }
  // Store additional information
  result.R_diagonal = qr.matrixQR().diagonal();
  result.residual_norm = (A * result.solution - b).norm();
  return result;
}

}  // namespace SDPTools
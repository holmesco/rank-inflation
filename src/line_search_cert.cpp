#include "line_search_cert.hpp"

namespace RankTools {

LineCertifier::LineCertifier(const Matrix& C, double rho,
                             const std::vector<Eigen::SparseMatrix<double>>& A,
                             const std::vector<double>& b, const Matrix& Y,
                             LineCertifierParams params)
    : dim(C.rows()),
      m(A.size() + 1),
      C_(C),
      rho_(rho),
      A_(A),
      b_(b),
      Y_(Y),
      params_(params) {
  // Initialize previous multipliers to zero
  prev_multipliers_ = Vector::Zero(m);
  // Get the interior solution factorization
  V_ = get_interior_factor(Y);
}

Matrix LineCertifier::get_interior_factor(const Matrix& Y) const {
  Eigen::JacobiSVD<Matrix> svd(Y, Eigen::ComputeThinU | Eigen::ComputeFullV);
  int rank = Y.cols();
  Matrix nullspace = svd.matrixV().rightCols(dim - rank);
  auto factor = Matrix(Y.rows(), Y.cols() + nullspace.cols());
  factor << Y, nullspace;

  return factor;
}

Diagonal LineCertifier::get_scaling_diagonal(double alpha) const {
  Vector diag = Vector::Constant(dim, alpha);
  const int k = std::min<int>(Y_.cols(), dim);
  diag.head(k).setConstant(1.0 - alpha);
  return diag.asDiagonal();
}

Matrix LineCertifier::build_certificate_from_dual(
    const Vector& multipliers) const {
  // build weighted sum of constraint matrices with multiplier values
  auto f_A = build_adjoint(multipliers, 0.0);
  // add cost term if required
  Matrix H;
  H = C_ + f_A;

  return H;
}

std::pair<double, double> LineCertifier::eval_certificate(
    const Matrix& H, const Matrix& Y) const {
  // Evaluate the stationarity condition
  double complementarity = (Y.transpose() * H * Y).norm();
  // Check Eigenvalues
  // Use SelfAdjointEigenSolver for symmetric matrices
  Eigen::SelfAdjointEigenSolver<Matrix> es(H);
  double min_eig = es.eigenvalues().minCoeff();

  return {min_eig, complementarity};
}

std::pair<double, double> LineCertifier::check_certificate(
    const Matrix& H, const Matrix& Y) const {
  // Evaluate the stationarity condition
  double complementarity = (Y.transpose() * H * Y).norm();
  // PSD Check
  Eigen::LLT<Matrix> llt(H + Matrix::Identity(H.rows(), H.cols()) *
                                 params_.tol_cert_psd);
  bool psd = (llt.info() == Eigen::Success);

  return {psd, complementarity};
}

SpMatrix LineCertifier::build_adjoint(const Vector& coeffs, double tol) const {
  // 1. Calculate the weighted sum of the upper-triangular parts
  SpMatrix upperSum = coeffs[0] * A_[0];
  for (size_t i = 1; i < A_.size(); ++i) {
    if (std::abs(coeffs(i)) > tol) {
      upperSum += coeffs[i] * A_[i];
    }
  }

  // 2. Reflect the upper triangle into the lower triangle to get the full
  // matrix .selfadjointView<Eigen::Upper>() treats the matrix as symmetric and
  // the assignment to a SparseMatrix fills in the missing entries.
  SpMatrix fullMatrix = upperSum.selfadjointView<Eigen::Upper>();

  fullMatrix.makeCompressed();
  return fullMatrix;
}

}  // namespace RankTools
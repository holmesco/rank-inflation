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
  // Precompute the constraint-factor products
  VAV_.reserve(m);
  for (size_t i = 0; i < A_.size(); ++i) {
    VAV_.emplace_back(V_.transpose() * A_[i] * V_);
  }
  VAV_.emplace_back(V_.transpose() * C_ * V_);
}

LineCertifierResult LineCertifier::certify(double alpha) const {
  double complementarity = std::nan("");
  double min_eig = std::nan("");
  double barrier_param = std::nan("");
  Vector mult_scaled(m - 1);
  bool certified = false;
  Matrix H;
  int n_iter = 0;
  while (n_iter < params_.max_iter && alpha >= params_.alpha_min) {
    // Get the certificate multipliers
    QRResult qr_result = get_multipliers(alpha);
    Vector multipliers = qr_result.solution;

    // get the barrier parameter value
    barrier_param = 1 / multipliers(m - 1);
    // compute scaled multipliers for certificate checking
    mult_scaled = multipliers.segment(0, m - 1) * barrier_param;

    // Build the certificate matrix from the dual solution
    H = build_certificate_from_dual(mult_scaled);
    // Check complementarity condition for first order optimality
    complementarity = (Y_.transpose() * H * Y_).norm();
    if (complementarity <= params_.tol_cert_complementarity) {
      // Use a cholesky decomposition for a quick check of PSDness.
      Eigen::LLT<Matrix> llt(H + Matrix::Identity(H.rows(), H.cols()) *
                                     params_.tol_cert_psd);
      if (llt.info() == Eigen::Success) {
        if (params_.verbose) {
          std::cout << "Certificate Found! Stopping centering." << std::endl;
        }
        certified = true;
        break;
      }
    }

    if (params_.verbose) {
      std::tie(min_eig, complementarity) = eval_certificate(H, Y_);
      if (n_iter % 10 == 0) {
        std::printf("%6s %12s %16s %16s %16s %16s\n", "Iter", "Alpha",
                    "Complementarity", "LinSys Res", "BarrParam", "Min Eig");
      }
      std::printf("%6d %12.6e %16.6e %16.6e %16.6e %16.6e\n", n_iter, alpha,
                  complementarity, qr_result.residual_norm, barrier_param,
                  min_eig);
    }

    // Failed convergence, bisect the scaling parameter
    alpha *= 0.5;
    n_iter++;
  }

  // Check the certificate at the current solution
  std::tie(min_eig, complementarity) = eval_certificate(H, Y_);

  return LineCertifierResult{H, mult_scaled, certified, min_eig,
                             complementarity};
}

QRResult LineCertifier::get_multipliers(double alpha) const {
  // Get diagonal scaling matrix
  const Diagonal sqrtD =
      get_scaling_diagonal(alpha).diagonal().cwiseSqrt().asDiagonal();
  // Build the linear multiplier system matrix (n(n-1/2) x m)
  const Eigen::Index symm_size = static_cast<Eigen::Index>(dim * (dim + 1) / 2);
  Matrix system_matrix(symm_size, static_cast<Eigen::Index>(VAV_.size()));
  for (size_t i = 0; i < VAV_.size(); ++i) {
    const Matrix scaled = (sqrtD * VAV_[i] * sqrtD).eval();
    // const Matrix scaled = (VAV_[i]).eval();
    system_matrix.col(static_cast<Eigen::Index>(i)) = vec_symm(scaled);
  }
  // Build the right-hand side of the system
  Vector rhs = vec_symm(Matrix::Identity(dim, dim));
  // Vector rhs = vec_symm(get_scaling_diagonal(alpha).inverse());

  // solve linear system
  auto result = get_soln_qr_dense(system_matrix, rhs, alpha / 10.0);

  return result;
}

Matrix LineCertifier::get_interior_factor(const Matrix& Y) const {
  Eigen::JacobiSVD<Matrix> svd(Y.transpose(),
                               Eigen::ComputeThinU | Eigen::ComputeFullV);
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
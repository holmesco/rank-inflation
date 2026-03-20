#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

// Forward declaration
class MultiplierLinSys;

// Diagonal preconditioner for MultiplierLinSys.
class MultiplierDiagPreconditioner {
 public:
  typedef double Scalar;
  typedef Eigen::VectorXd Vector;

  MultiplierDiagPreconditioner() : is_initialized_(false) {}

  // Eigen's CG calls compute(mat) with the matrix-free operator
  MultiplierDiagPreconditioner& compute(const MultiplierLinSys& op) {
    inv_diag_.resize(op.LAL_.cols());
    for (int i = 0; i < op.LAL_.cols(); ++i) {
      double diag_val = op.LAL_.col(i).array().square().sum() * op.scale_;
      inv_diag_(i) = (diag_val != Scalar(0)) ? 1.0 / diag_val : 1.0;
    }
    is_initialized_ = true;
    return *this;
  }

  // Apply the preconditioner: element-wise multiply by 1/diag
  template <typename Rhs>
  Eigen::VectorXd solve(const Eigen::MatrixBase<Rhs>& b) const {
    return inv_diag_.cwiseProduct(b);
  }

  Eigen::ComputationInfo info() const {
    return is_initialized_ ? Eigen::Success : Eigen::InvalidInput;
  }

 private:
  Vector inv_diag_;
  bool is_initialized_;
};

// Low Rank Preconditioner for the Lagrange multiplier system.
// Definition of preconditioner follows  Loraine (Habibi 2017)
class ApproxLowRankPrecond {
 public:
  typedef double Scalar;
  using Vector = Eigen::VectorXd;
  using Matrix = Eigen::MatrixXd;
  using SpMatrix = Eigen::SparseMatrix<double>;

  ApproxLowRankPrecond(const Matrix& X, const std::vector<SpMatrix>& As,
                       const Matrix& C, int rank)
      : is_initialized_(false),
        X_(X),
        C_(C),
        As_(As),
        rank_(rank),
        dim(C.cols()),
        ncons(As.size() + 1) {}

  // Eigen's CG calls compute(mat) with the matrix-free operator
  ApproxLowRankPrecond& compute() {
    // decompose eigenspace
    auto [U, W0, tau] = decompose_soln(X_);
    // store tau^2
    tau_sq_ = std::pow(tau, 2);
    // construct V_tilde
    V_tilde_ = build_V_tilde(U, W0);
    // Construct Schur complement matrix and decompose it
    Matrix Theta = Matrix::Identity(rank_ * dim, rank_ * dim) * tau_sq_ +
                   (V_tilde_.transpose() * V_tilde_);
    // Prefactorize Theta
    Theta_ldlt_.compute(Theta);
    // Flag that we have initialized
    is_initialized_ = true;
    return *this;
  }

  // Apply the preconditioner: Use SMW formula in (14) of Habibi et al. (2017)
  template <typename Rhs>
  Eigen::VectorXd solve(const Eigen::MatrixBase<Rhs>& b) const {
    // mult by Vt
    auto Vtb = V_tilde_.transpose() * b;
    auto y = Theta_ldlt_.solve(Vtb);
    auto result = (b - V_tilde_ * y) / tau_sq_;

    return result;
  }

  // Build the V_tilde matrix = A_bar^T(U otimes Z)
  Matrix build_V_tilde(const Matrix& U, const Matrix& W0) const {
    // construct Gamma matrix s.t. Z Z^T = (2W0 + U U^T) = X + W0
    Eigen::LLT<Matrix> llt(X_ + W0);
    Matrix Z = llt.matrixL();
    // build V
    auto V = Matrix(ncons, rank_ * dim);
    for (int i = 0; i < ncons; i++) {
      Matrix mat;
      if (i < ncons - 1) {
        mat = Z.transpose() * As_[i].selfadjointView<Eigen::Upper>() * U;
      } else {
        mat = Z.transpose() * C_.selfadjointView<Eigen::Upper>() * U;
      }
      V.row(i) = Eigen::Map<Vector>(mat.data(), mat.size());
    }
    return V;
  }

  // Decompose the solution in to high and low eigenvalue matrices
  std::tuple<Matrix, Matrix, double> decompose_soln(const Matrix& X) const {
    // Eigendecomposition of solution
    Eigen::SelfAdjointEigenSolver<Matrix> es(X);
    Vector eigenvalues = es.eigenvalues();
    Matrix eigenvectors = es.eigenvectors();
    // Select smallest eigenvalue as tau
    // NOTE: Other options could be used here: mineig(X) <= tau < top_eigs
    double tau = eigenvalues(0);
    // Construct U matrix (large-eigenvalue, low rank, part)
    Vector top_eigs = eigenvalues.tail(rank_).array() - tau;
    Matrix U =
        eigenvectors.rightCols(rank_) * top_eigs.cwiseSqrt().asDiagonal();
    // Replace top eigenvalues with tau
    eigenvalues.segment(dim - rank_, rank_).setConstant(tau);
    // Build W0 (low-eigenvalue part)
    auto W0 =
        eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();

    return {U, W0, tau};
  }

  Eigen::ComputationInfo info() const {
    return is_initialized_ ? Eigen::Success : Eigen::InvalidInput;
  }

 private:
  bool is_initialized_;
  const Matrix& X_;
  const Matrix& C_;
  const std::vector<Eigen::SparseMatrix<double>>& As_;
  int rank_;
  int dim;
  int ncons;

  // Stored matrices and values
  Matrix V_tilde_;
  Eigen::LDLT<Matrix> Theta_ldlt_;
  double tau_sq_;
};

// Low Rank Preconditioner for the Lagrange multiplier system.
// Definition of preconditioner follows equation 23 of Zhang and Lavaei 2017
// use_approx applies the approximation ZZ^T = 2tau I instead of Z Z^T = X + W0,
// which is cheaper to compute, but in our case may be less effective at
// improving conditioning.
class LowRankPrecond {
 public:
  typedef double Scalar;
  using Vector = Eigen::VectorXd;
  using Matrix = Eigen::MatrixXd;
  using SpMatrix = Eigen::SparseMatrix<double>;

  // Default constructor - no problem data passed
  LowRankPrecond()
      : is_initialized_(false),
        X_(nullptr),
        C_(nullptr),
        As_(nullptr),
        rank_(1),
        dim(0),
        ncons(0),
        use_approx_(false) {}

  // Initialize the preconditioner with problem data
  void init(const Matrix& X, const std::vector<SpMatrix>& As, const Matrix& C,
            int rank, bool use_approx = false) {
    X_ = &X;
    C_ = &C;
    As_ = &As;
    rank_ = rank;
    dim = C.cols();
    ncons = As.size() + 1;
    use_approx_ = use_approx;
  }

  // Eigen's CG calls compute(mat) with the matrix-free operator
  LowRankPrecond& compute(const MultiplierLinSys& op) {
    if (!X_ || !C_ || !As_) {
      throw std::runtime_error(
          "LowRankPrecond: Problem data not set. Call init() before "
          "compute().");
    }
    // compute constraint matrix
    build_constraint_mat();
    // decompose eigenspace
    auto [U, W0, tau] = decompose_soln(*X_);
    // Build sparse augmented system - Eqn 23 in Zhang and Lavaei 2017
    auto Sys = Matrix(ncons + rank_ * dim, ncons + rank_ * dim);
    Sys.block(0, 0, ncons, ncons) =
        A_bar_.transpose() * A_bar_ * std::pow(tau, 2.0);
    Sys.block(0, ncons, ncons, rank_ * dim) = build_top_right(U, W0, tau) * tau;
    Sys.block(ncons, ncons, rank_ * dim, rank_ * dim) =
        -Matrix::Identity(rank_ * dim, rank_ * dim) * std::pow(tau, 2);

    // Prefactorize
    Factor.compute(Sys.selfadjointView<Eigen::Upper>());
    // Flag that we have initialized
    is_initialized_ = true;
    return *this;
  }

  // Apply the preconditioner by solving the augmented system
  template <typename Rhs>
  Eigen::VectorXd solve(const Eigen::MatrixBase<Rhs>& b) const {
    // Augment rhs
    auto rhs = Vector(ncons + rank_ * dim);
    rhs << b, Vector::Zero(rank_ * dim);
    // Apply LDLT inverse
    auto result = Factor.solve(rhs);
    return result.segment(0, ncons);
  }

  // Build the top right matrix = A_bar^T(U otimes Z)
  Matrix build_top_right(const Matrix& U, const Matrix& W0, double tau) const {
    // Build the Z matrix s.t. Z Z^T = (2W0 + U U^T) = X + W0
    Matrix Z;
    if (!use_approx_) {
      Eigen::LLT<Matrix> llt(*X_ + W0);
      Z = llt.matrixL();
    } else {
      // if approximation set then ZZ^T = 2tau I
      // Note: we rely on eigen to optimize this below.
      Z = Matrix::Identity(dim, dim) * std::sqrt(2 * tau);
    }

    // Build the top right block of the augmented system matrix
    auto top_right = Matrix(ncons, rank_ * dim);
    for (int i = 0; i < ncons; i++) {
      Matrix mat;
      if (i < ncons - 1) {
        mat = Z.transpose() * (*As_)[i].selfadjointView<Eigen::Upper>() * U;
      } else {
        mat = Z.transpose() * (*C_).selfadjointView<Eigen::Upper>() * U;
      }
      top_right.row(i) = Eigen::Map<Vector>(mat.data(), mat.size());
    }
    return top_right;
  }

  // Construct the vectorized constraint matrix
  void build_constraint_mat() {
    // NOTE: Currently we set this up as a dense matrix. Leveraging sparsity is
    // a future todo
    A_bar_.resize(dim * dim, ncons);
    A_bar_.setZero();
    for (int i = 0; i < ncons; i++) {
      if (i < ncons - 1) {
        // Use an InnerIterator to move through non-zeros only
        for (int k = 0; k < (*As_)[i].outerSize(); ++k) {
          for (Eigen::SparseMatrix<double>::InnerIterator it((*As_)[i], k); it;
               ++it) {
            // Calculate vectorized row index: col * total_rows + row
            if (it.col() > it.row()) {
              // make symmetric
              int row_idx = it.col() * dim + it.row();
              A_bar_(row_idx, i) = it.value();
              row_idx = it.row() * dim + it.col();
              A_bar_(row_idx, i) = it.value();

            } else if (it.col() == it.row()) {
              int row_idx = it.col() * dim + it.row();
              A_bar_(row_idx, i) = it.value();
            }
          }
        }
      } else {
        // For dense matrix, just map
        A_bar_.col(i) = Eigen::Map<const Vector>(C_->data(), C_->size());
      }
    }
  }
  // Decompose the solution in to high and low eigenvalue matrices
  std::tuple<Matrix, Matrix, double> decompose_soln(const Matrix& X) const {
    // Eigendecomposition of solution
    Eigen::SelfAdjointEigenSolver<Matrix> es(X);
    Vector eigenvalues = es.eigenvalues();
    Matrix eigenvectors = es.eigenvectors();
    // Select smallest eigenvalue as tau
    // NOTE: Other options could be used here: mineig(X) <= tau < top_eigs
    double tau = eigenvalues(0);
    // Construct U matrix (large-eigenvalue, low-rank part)
    Vector top_eigs = eigenvalues.tail(rank_).array() - tau;
    Matrix U =
        eigenvectors.rightCols(rank_) * top_eigs.cwiseSqrt().asDiagonal();
    // Replace top eigenvalues with tau
    eigenvalues.segment(dim - rank_, rank_).setConstant(tau);
    // Build W0 (low-eigenvalue part)
    auto W0 =
        eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();

    return {U, W0, tau};
  }

  Eigen::ComputationInfo info() const {
    return is_initialized_ ? Eigen::Success : Eigen::InvalidInput;
  }

 private:
  bool is_initialized_;
  const Matrix* X_;
  const Matrix* C_;
  const std::vector<Eigen::SparseMatrix<double>>* As_;
  int rank_;
  int dim;
  int ncons;
  bool use_approx_;

  // built matrices
  Matrix A_bar_;
  Eigen::LDLT<Matrix> Factor;
};

// Add preconditioner types to RankTools namespace for use in analytic center
// and other algorithms
namespace RankTools {
// Preconditioner types
enum class PreconditionerType {
  Diagonal,
  LowRank,
  FixedLowRank,
};
}  // namespace RankTools

#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <unsupported/Eigen/KroneckerProduct>

// declare class for matrix-free linear system
class MultiplierLinSys;
using Eigen::SparseMatrix;

namespace Eigen {
namespace internal {
// MultiplierLinSys looks-like a dense matrix, so we inheret its traits:
template <>
struct traits<MultiplierLinSys> : public traits<Eigen::SparseMatrix<double>> {};
}  // namespace internal
}  // namespace Eigen

// Multiplier Linear System for use in iterative solvers. This class allows us
// to use Eigen's iterative solvers with a custom matrix-vector product defined
// by the sparse matrix of the problem, without explicitly forming the dense
// matrix.
class MultiplierLinSys : public Eigen::EigenBase<MultiplierLinSys> {
 public:
  // Required typedefs, constants, and method:
  typedef double Scalar;
  typedef double RealScalar;
  typedef int StorageIndex;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };
  Index rows() const { return static_cast<Eigen::Index>(LAL_.cols()); }
  Index cols() const { return static_cast<Eigen::Index>(LAL_.cols()); }

  template <typename Rhs>
  Eigen::Product<MultiplierLinSys, Rhs, Eigen::AliasFreeProduct> operator*(
      const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<MultiplierLinSys, Rhs, Eigen::AliasFreeProduct>(
        *this, x.derived());
  }
  // precomputed matrix where each col is vec(L^T * A_i * L)
  const Eigen::MatrixXd& LAL_;
  // perturbation parameter
  double scale_;

  // API to set the data of the linear system (the cost matrix and the
  // constraint matrices)
  MultiplierLinSys(const Eigen::MatrixXd& LAL, double scale)
      : LAL_(LAL), scale_(scale) {}
};

// Implementation of MultiplierLinSys * Eigen::DenseVector though a
// specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {

template <typename Rhs>
struct generic_product_impl<MultiplierLinSys, Rhs, SparseShape, DenseShape,
                            GemvProduct>  // GEMV stands for matrix-vector
    : generic_product_impl_base<MultiplierLinSys, Rhs,
                                generic_product_impl<MultiplierLinSys, Rhs>> {
  typedef typename Product<MultiplierLinSys, Rhs>::Scalar Scalar;

  // Custom implementation of the matrix-vector product for our linear system.
  // Computes a matrix vector product y = B*x = LAL_^T * (LAL_ * x) without
  // explicitly forming the dense matrix.
  // Complexity: O(n^2 m) where n is the dimension of the SDP and m is the
  // number of constraints.
  template <typename Dest>
  static void scaleAndAddTo(Dest& dst, const MultiplierLinSys& lhs,
                            const Rhs& rhs, const Scalar& alpha) {
    // Rescale the input
    auto rhs_scaled = rhs * alpha * lhs.scale_;
    // Mult by matrix then by transpose
    auto y = lhs.LAL_ * rhs_scaled;
    dst += lhs.LAL_.transpose() * y;
  }
};

}  // namespace internal
}  // namespace Eigen

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
        mat = Z.transpose() * As_[i] * U;
      } else {
        mat = Z.transpose() * C_ * U;
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
class LowRankPrecond {
 public:
  typedef double Scalar;
  using Vector = Eigen::VectorXd;
  using Matrix = Eigen::MatrixXd;
  using SpMatrix = Eigen::SparseMatrix<double>;

  LowRankPrecond(const Matrix& X, const std::vector<SpMatrix>& As,
                 const Matrix& C, int rank)
      : is_initialized_(false),
        X_(X),
        C_(C),
        As_(As),
        rank_(rank),
        dim(C.cols()),
        ncons(As.size() + 1) {}

  // Eigen's CG calls compute(mat) with the matrix-free operator
  LowRankPrecond& compute() {
    // compute constraint matrix
    build_constraint_mat();
    // decompose eigenspace
    auto [U, W0, tau] = decompose_soln(X_);
    // Build sparse augmented system - Eqn 23
    auto Sys = Matrix(ncons + rank_ * dim, ncons + rank_ * dim);
    Sys.block(0, 0, ncons, ncons) =
        A_bar_.transpose() * A_bar_ * std::pow(tau, 2.0);
    Sys.block(0, ncons, ncons, rank_ * dim) = build_V_tilde(U, W0) * tau;
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

  // Build the top right matrix in the augmented preconditioner system
  Matrix build_top_right(const Matrix& U) const {
    auto top_right = Matrix(ncons, rank_ * dim);
    for (int i = 0; i < ncons; i++) {
      Matrix mat;
      if (i < ncons - 1) {
        mat = As_[i] * U;
      } else {
        mat = C_ * U;
      }
      top_right.row(i) = Eigen::Map<Vector>(mat.data(), mat.size());
    }
    return top_right;
  }

  // Build the V_tilde matrix = A_bar^T(U otimes Z)
  Matrix build_V_tilde(const Matrix& U, const Matrix& W0) const {
    // construct Z matrix s.t. Z Z^T = (2W0 + U U^T) = X + W0
    Eigen::LLT<Matrix> llt(U * U.transpose() + 2 * W0);
    Matrix Z = llt.matrixL();
    // build V
    auto V = Matrix(ncons, rank_ * dim);
    for (int i = 0; i < ncons; i++) {
      Matrix mat;
      if (i < ncons - 1) {
        mat = Z.transpose() * As_[i] * U;
      } else {
        mat = Z.transpose() * C_ * U;
      }
      V.row(i) = Eigen::Map<Vector>(mat.data(), mat.size());
    }
    return V;
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
        for (int k = 0; k < As_[i].outerSize(); ++k) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(As_[i], k); it;
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
        A_bar_.col(i) = Eigen::Map<const Vector>(C_.data(), C_.size());
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

  Matrix get_lin_sys_mat() const {
    Matrix XkronX = Eigen::kroneckerProduct(X_, X_);
    return A_bar_.transpose() * XkronX * A_bar_;
  }

 private:
  bool is_initialized_;
  const Matrix& X_;
  const Matrix& C_;
  const std::vector<Eigen::SparseMatrix<double>>& As_;
  int rank_;
  int dim;
  int ncons;

  // built matrices
  Matrix A_bar_;
  Eigen::LDLT<Matrix> Factor;
};

namespace RankTools {

// Enumeration for linear solver types
enum class LinearSolverType { LDLT, CG, MFCG };

// Nice printing for the linear solver types for debugging and display purposes
inline std::string print_solver(LinearSolverType solver) {
  switch (solver) {
    case LinearSolverType::LDLT:
      return "LDLT Direct Solver";
    case LinearSolverType::CG:
      return "Conjugate Gradient";
    case LinearSolverType::MFCG:
      return "Matrix-Free Conjugate Gradient";
    default:
      return "Unknown";
  }
}
}  // namespace RankTools

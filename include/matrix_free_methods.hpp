#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

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
  Index rows() const { return static_cast<Eigen::Index>(num_constraints_); }
  Index cols() const { return static_cast<Eigen::Index>(num_constraints_); }

  template <typename Rhs>
  Eigen::Product<MultiplierLinSys, Rhs, Eigen::AliasFreeProduct> operator*(
      const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<MultiplierLinSys, Rhs, Eigen::AliasFreeProduct>(
        *this, x.derived());
  }
  // Stored values for the linear system
  const int num_constraints_;
  // cost matrix
  const Eigen::MatrixXd& C_;
  // constraint matrices
  const std::vector<Eigen::SparseMatrix<double>>& As_; 
  // Cholesky factor of primal solution,
  const Eigen::MatrixXd& L_;
  // precomputed L^T * A_i * L products for efficient matrix-vector product computation
  const std::vector<Eigen::MatrixXd>& LAL_;
  // perturbation parameter 
  double delta_;

  // API to set the data of the linear system (the cost matrix and the
  // constraint matrices)
  MultiplierLinSys(const Eigen::MatrixXd& C,
                   const std::vector<Eigen::SparseMatrix<double>>& As,
                   const Eigen::MatrixXd& L,
                   const std::vector<Eigen::MatrixXd>& LAL,
                   double delta)
      : num_constraints_(As.size() + 1),
        C_(C),
        As_(As),
        L_(L),
        LAL_(LAL),
        delta_(delta) {}

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
  // Computes dst(i) = alpha * tr(A_i * X * S * X) / delta where
  // S = sum_j rhs(j) * A_j (with A_last = C).
  // Uses precomputed AXt_[i] = (A_i * X)^T to avoid redundant work.
  template <typename Dest>
  static void scaleAndAddTo(Dest& dst, const MultiplierLinSys& lhs,
                            const Rhs& rhs, const Scalar& alpha) {
    // Build the weighted sum S = sum_i rhs(i) * A_i + rhs(last) * C
    // (only upper triangle is filled; used via selfadjointView below)
    Eigen::MatrixXd S = rhs(rhs.size() - 1) * lhs.C_;
    for (size_t i = 0; i < lhs.As_.size(); ++i) {
      S += rhs(i) * lhs.As_[i];
    }
    // two O(n^3) multiply: LSL = L^t * S_sym * L
    Eigen::MatrixXd LSL =
        lhs.L_.transpose() * S.selfadjointView<Eigen::Upper>() * lhs.L_ * (alpha / lhs.delta_);
    // dst(i) = tr(A_i * X * S * X) * alpha / delta 
    //        = tr(L^T * A_i * L * L^T * S * L) * alpha / delta, where L is the Cholesky factor of X
    //        = vec(LAL_[i])^T * vec(LSL) *alpha/delta
    // O(n^3)
    const int m = lhs.num_constraints_;
    for (int i = 0; i < m; ++i) {
      Eigen::Map<const Eigen::VectorXd> axt(lhs.LAL_[i].data(),
                                            lhs.LAL_[i].size());
      Eigen::Map<const Eigen::VectorXd> sx(LSL.data(), LSL.size());
      dst(i) += axt.dot(sx);
    }
  }
};

}  // namespace internal
}  // namespace Eigen


// Diagonal preconditioner for MultiplierLinSys.
// Diagonal entry i is B(i,i) = tr(A_i * X * A_i * X) / delta = tr((A_i*X)^2)
// / delta. Computed in O(n^2) per entry via tr(M^2) = sum_jk M_jk * M_kj =
// <M, M^T>_F.
class MultiplierDiagPreconditioner {
 public:
  typedef double Scalar;
  typedef Eigen::VectorXd Vector;

  MultiplierDiagPreconditioner() : is_initialized_(false) {}

  // Eigen's CG calls compute(mat) with the matrix-free operator
  MultiplierDiagPreconditioner& compute(const MultiplierLinSys& op) {
    const int m = op.num_constraints_;
    inv_diag_.resize(m);

    const int a_size = static_cast<int>(op.As_.size());
    for (int i = 0; i < m; ++i) {
      // tr(Ai * X * Ai * X) = tr (L^T * A_i * L * L^T * A_i * L) = tr((L^T * A_i * L)^2) = sum_jk (L^T * A_i * L)_jk^2
      double diag_val = op.LAL_[i].array().square().sum() / op.delta_;
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

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
struct traits<MultiplierLinSys>
    : public Eigen::internal::traits<Eigen::MatrixXd> {};
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
  const Eigen::MatrixXd& C_;
  const std::vector<Eigen::SparseMatrix<double>>& As_;
  const Eigen::MatrixXd* X_;
  double delta_;

  // API to set the data of the linear system (the cost matrix and the
  // constraint matrices)
  MultiplierLinSys(const Eigen::MatrixXd& C,
                   const std::vector<Eigen::SparseMatrix<double>>& As,
                   const Eigen::MatrixXd& X, double delta)
      : num_constraints_(As.size() + 1),
        C_(C),
        As_(As),
        X_(&X),
        delta_(delta) {}

  void setX(const Eigen::MatrixXd& new_X) { X_ = &new_X; }
  void setDelta(double new_delta) { delta_ = new_delta; }
};

// Implementation of MultiplierLinSys * Eigen::DenseVector though a
// specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {

template <typename Rhs>
struct generic_product_impl<MultiplierLinSys, Rhs, DenseShape, DenseShape,
                            GemvProduct>  // GEMV stands for matrix-vector
    : generic_product_impl_base<MultiplierLinSys, Rhs,
                                generic_product_impl<MultiplierLinSys, Rhs>> {
  typedef typename Product<MultiplierLinSys, Rhs>::Scalar Scalar;

  // Custom implementation of the matrix-vector product for our linear system.
  // Computes P = X * (sum_i rhs(i) * A_i + rhs(last) * C) * X, then dst(i) =
  // trace(A_i * S) / delta for each constraint i and dst(last) = trace(C * S) /
  // delta for the cost term.
  template <typename Dest>
  static void scaleAndAddTo(Dest& dst, const MultiplierLinSys& lhs,
                            const Rhs& rhs, const Scalar& alpha) {
    // First construct the weighted sum of constraints and cost:
    Eigen::MatrixXd S = rhs(rhs.size() - 1) * lhs.C_;
    for (size_t i = 0; i < lhs.As_.size(); ++i) {
      S += rhs(i) * lhs.As_[i];
    }
    // Form dense product
    Eigen::MatrixXd P = (*lhs.X_) * S.selfadjointView<Eigen::Upper>() *
                        (*lhs.X_) * alpha / lhs.delta_;
    // Compute the trace of A_i * P for each constraint
    // For sparse A (upper triangular storage):
    for (size_t i = 0; i < lhs.As_.size(); ++i) {
      const auto& A = lhs.As_[i];
      for (int k = 0; k < A.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
          if (it.row() == it.col())
            dst(i) += it.value() * P(it.row(), it.col());
          else
            dst(i) += 2.0 * it.value() * P(it.row(), it.col());
        }
      }
    }
    // Trace of C * P for the last entry
    Eigen::MatrixXd C_full = lhs.C_.selfadjointView<Eigen::Upper>();
    dst(rhs.size() - 1) += C_full.cwiseProduct(P).sum();
  }
};

}  // namespace internal
}  // namespace Eigen

// Diagonal preconditioner for MultiplierLinSys.
// Diagonal entry i is B(i,i) = tr(A_i * X * A_i * X) / delta = tr((A_i*X)^2) /
// delta. Computed in O(n^2) per entry via tr(M^2) = sum_jk M_jk * M_kj = <M,
// M^T>_F.
class MultiplierDiagPreconditioner {
 public:
  typedef double Scalar;
  typedef Eigen::VectorXd Vector;

  MultiplierDiagPreconditioner() : is_initialized_(false) {}

  // Eigen's CG calls compute(mat) with the matrix-free operator
  MultiplierDiagPreconditioner& compute(const MultiplierLinSys& op) {
    const int m = op.num_constraints_;
    const auto& X = *op.X_;
    inv_diag_.resize(m);

    const int a_size = static_cast<int>(op.As_.size());
    for (int i = 0; i < m; ++i) {
      Eigen::MatrixXd AiX;
      if (i < a_size) {
        AiX = op.As_[i].selfadjointView<Eigen::Upper>() * X;
      } else {
        AiX = op.C_ * X;  // last entry corresponds to the cost matrix
      }
      // tr((AiX)^2) = sum_jk (AiX)_jk * (AiX)_kj = <AiX, AiX^T>_F
      double diag_val = AiX.cwiseProduct(AiX.transpose()).sum() / op.delta_;
      inv_diag_(i) = (diag_val > 1e-14) ? 1.0 / diag_val : 1.0;
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

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
  const Eigen::MatrixXd& C_;
  const std::vector<Eigen::SparseMatrix<double>>& As_;
  const Eigen::MatrixXd* X_;
  const std::vector<Eigen::MatrixXd>& AX_;
  const std::vector<Eigen::MatrixXd>& AXt_;
  double delta_;

  // API to set the data of the linear system (the cost matrix and the
  // constraint matrices)
  MultiplierLinSys(const Eigen::MatrixXd& C,
                   const std::vector<Eigen::SparseMatrix<double>>& As,
                   const Eigen::MatrixXd& X,
                   const std::vector<Eigen::MatrixXd>& AX,
                   const std::vector<Eigen::MatrixXd>& AXt,
                    double delta)
      : num_constraints_(As.size() + 1),
        C_(C),
        As_(As),
        X_(&X),
        AX_(AX),
        AXt_(AXt),
        delta_(delta) {}

  void setX(const Eigen::MatrixXd& new_X) { X_ = &new_X; }
  void setDelta(double new_delta) { delta_ = new_delta; }

  // Compute the diagonal of the matrix for preconditioner.
  Eigen::VectorXd diagonal() const {
    const int m = num_constraints_;
    const auto& X = *X_;
    Eigen::VectorXd diag(m);
    const int a_size = static_cast<int>(As_.size());
    for (int i = 0; i < m; ++i) {
      Eigen::MatrixXd AiX;
      if (i < a_size) {
        AiX = As_[i].selfadjointView<Eigen::Upper>() * X;
      } else {
        AiX = C_ * X;  // last entry corresponds to the cost matrix
      }
      // tr((AiX)^2) = sum_jk (AiX)_jk * (AiX)_kj = <AiX, AiX^T>_F
      diag(i) = AiX.cwiseProduct(AiX.transpose()).sum() / delta_;
    }
    return diag;
  }
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
    // Single O(n^3) multiply: SX = S_sym * X
    Eigen::MatrixXd SX =
        S.selfadjointView<Eigen::Upper>() * (*lhs.X_) * (alpha / lhs.delta_);
    // dst(i) = <AXt_[i], SX>_F = tr(A_i * X * S * X) * alpha / delta
    const int m = lhs.num_constraints_;
    for (int i = 0; i < m; ++i) {
      Eigen::Map<const Eigen::VectorXd> axt(lhs.AXt_[i].data(),
                                           lhs.AXt_[i].size());
      Eigen::Map<const Eigen::VectorXd> sx(SX.data(), SX.size());
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
    const auto& X = *op.X_;
    inv_diag_.resize(m);

    const int a_size = static_cast<int>(op.As_.size());
    for (int i = 0; i < m; ++i) {
      // tr((AiX)^2) = sum_jk (AiX)_jk * (AiX)_kj = <AiX, AiX^T>_F
      double diag_val = op.AX_[i].cwiseProduct(op.AXt_[i]).sum() / op.delta_;
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

#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#include "preconditioner.hpp"

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

#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

// -------------- SOLVER ENUM TYPES ---------------

// Enumeration for linear solver types
enum class LinearSolverType { LDLT, CG, MFCG_DP, MFCG_LRP };

// Nice printing for the linear solver types for debugging and display purposes
inline std::string print_solver(LinearSolverType solver) {
  switch (solver) {
    case LinearSolverType::LDLT:
      return "LDLT Direct Solver";
    case LinearSolverType::CG:
      return "Conjugate Gradient";
    case LinearSolverType::MFCG_DP:
      return "Matrix-Free Conjugate Gradient - Diagonal Preconditioner";
    case LinearSolverType::MFCG_LRP:
      return "Matrix-Free Conjugate Gradient - Low Rank Preconditioner";
    default:
      return "Unknown";
  }
}

// --------------- MATRIX-FREE METHODS -----------

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
  Index rows() const { return static_cast<Eigen::Index>(ncons); }
  Index cols() const { return static_cast<Eigen::Index>(ncons); }

  template <typename Rhs>
  Eigen::Product<MultiplierLinSys, Rhs, Eigen::AliasFreeProduct> operator*(
      const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<MultiplierLinSys, Rhs, Eigen::AliasFreeProduct>(
        *this, x.derived());
  }
  const Eigen::MatrixXd& X_;
  const std::vector<Eigen::SparseMatrix<double>>& As_;
  const Eigen::MatrixXd& C_;
  const std::vector<Eigen::MatrixXd>&
      AX_;  // precomputed A_i * X and C * X products
  const int ncons;
  // perturbation parameter
  double scale_;

  // API to set the data of the linear system (the cost matrix and the
  // constraint matrices)
  MultiplierLinSys(const Eigen::MatrixXd& X,
                   const std::vector<Eigen::SparseMatrix<double>>& As,
                   const Eigen::MatrixXd& C,
                   const std::vector<Eigen::MatrixXd>& AX, double scale)
      : X_(X), As_(As), C_(C), AX_(AX), ncons(As.size() + 1), scale_(scale) {}
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
  // Computes a matrix vector product y = B*x = A_bar^T (X kron X) A_bar *x
  // without explicitly forming the dense matrix. Complexity: O(n^2 m) where n
  // is the dimension of the SDP and m is the number of constraints.
  template <typename Dest>
  static void scaleAndAddTo(Dest& dst, const MultiplierLinSys& lhs,
                            const Rhs& rhs, const Scalar& alpha) {
    // Construct product S = sum_i A_i * y_i
    Eigen::MatrixXd S = rhs(rhs.size() - 1) * lhs.C_;
    for (int i = 0; i < rhs.size() - 1; i++) {
      S += rhs(i) * lhs.As_[i];
    }
    // Compute (S*X)^T = X*S
    Eigen::MatrixXd XS = lhs.X_ * S.selfadjointView<Eigen::Upper>();
    // Compute trace(Ai X S X) = trace((X*S)^T A_i X) = vec(X*S)^T vec(A_i X) =
    // vec(X*S)^T AX_[i]
    for (int i = 0; i < lhs.ncons; i++) {
      Eigen::Map<const Eigen::VectorXd> sxt(XS.data(), XS.size());
      Eigen::Map<const Eigen::VectorXd> ax(lhs.AX_[i].data(),
                                           lhs.AX_[i].size());
      dst(i) += sxt.dot(ax) * alpha * lhs.scale_;
      ;
    }
  }
};

}  // namespace internal
}  // namespace Eigen

// ----------- PRECONDITIONERS ------------------

// Diagonal preconditioner for MultiplierLinSys.
class MultiplierDiagPreconditioner {
 public:
  typedef double Scalar;
  typedef Eigen::VectorXd Vector;
  typedef Eigen::MatrixXd Matrix;

  MultiplierDiagPreconditioner() : is_initialized_(false) {}

  // Eigen's CG calls compute(mat) with the matrix-free operator
  MultiplierDiagPreconditioner& compute(const MultiplierLinSys& op) {
    inv_diag_.resize(op.ncons);
    for (int i = 0; i < op.ncons; ++i) {
      double diag_val = (op.AX_[i] * op.AX_[i]).trace() * op.scale_;
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

struct LowRankPrecondParams {
  // Diagonal perturbation parameter.
  double tau = 1e-5;
  // Flag for using the sparse factorization approach (true) vs the dense
  // approach (false) to building the preconditioner. The sparse approach is
  // cheaper to build, but may be less effective at improving conditioning.
  bool use_sparse_factor = true;
  // Flag for using the approximation ZZ^T = 2tau I instead of Z Z^T = X + W0,
  // which is cheaper to compute, but in our case may be less effective at
  // improving conditioning.
  bool use_approx = false;
  // LDLT Threshold for treating small eigenvalues as zero. This is used in the
  // dense
  double ldlt_zero_thresh = 1e-14;
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

  // Low rank preconditioner parameter set
  LowRankPrecondParams params_;

  // Default constructor - no problem data passed
  LowRankPrecond(LowRankPrecondParams params = LowRankPrecondParams())
      : is_initialized_(false),
        U_(nullptr),
        C_(nullptr),
        As_(nullptr),
        dim(0),
        ncons(0),
        scale_(1.0),
        params_(params) {}

  // Initialize the preconditioner with problem data
  void initialize(const Matrix& U, const std::vector<SpMatrix>& As,
                  const Matrix& C) {
    // Initialize variables with problem data
    U_ = &U;
    C_ = &C;
    As_ = &As;
    rank_ = U.cols();
    dim = C.cols();
    ncons = As.size() + 1;
    // Call function to build the preconditioner
    if (params_.use_sparse_factor) {
      build_preconditioner_sparse();
    } else {
      build_preconditioner_dense();
    }
  }

  void set_scale(double scale) { scale_ = scale; }

  // Eigen's CG calls compute(mat) with the matrix-free operator
  LowRankPrecond& compute(const MultiplierLinSys& op) {
    // Store scaling factor of linear operator
    scale_ = op.scale_;
    if (is_initialized_) {
      // If already initialized, just return
      return *this;
    }
    // Call the internal build function that does the actual work
    if (params_.use_sparse_factor) {
      return build_preconditioner_sparse();
    } else {
      return build_preconditioner_dense();
    }
  }

  // Internal compute function that does the actual work of building the
  // preconditioner. This version builds the dense augmented system and
  // factorizes it with LDLT. We could
  LowRankPrecond& build_preconditioner_dense() {
    if (!U_ || !C_ || !As_) {
      throw std::runtime_error(
          "LowRankPrecond: Problem data not set. Call init() before "
          "compute().");
    }
    // compute constraint matrix
    build_constraint_mat();
    // Set W0 residual
    auto W0 = Matrix::Identity(dim, dim) * params_.tau;
    // Build sparse augmented system - Eqn 23 in Zhang and Lavaei 2017
    auto Sys = Matrix(ncons + rank_ * dim, ncons + rank_ * dim);
    Sys.block(0, 0, ncons, ncons) =
        (A_bar_.transpose() * A_bar_).template triangularView<Eigen::Upper>() *
        std::pow(params_.tau, 2.0);
    Sys.block(0, ncons, ncons, rank_ * dim) =
        build_top_right(*U_, W0, params_.tau) * params_.tau;
    Sys.block(ncons, ncons, rank_ * dim, rank_ * dim) =
        -Matrix::Identity(rank_ * dim, rank_ * dim) * std::pow(params_.tau, 2);
  
    // Prefactorize (LDLT)
    Factor.compute(Sys.selfadjointView<Eigen::Upper>());
    // Flag that we have initialized to eigen
    is_initialized_ = (Factor.info() == Eigen::Success);
    if (!is_initialized_) {
      std::cerr << "LDLT factorization failed. Info: " << Factor.info()
                << std::endl;
    }

    return *this;
  }

  // Internal compute function that does the actual work of building the
  // preconditioner. This version builds the augmented system sparsely and
  // factorizes it with SimplicialLDLT. We could
  LowRankPrecond& build_preconditioner_sparse() {
    if (!U_ || !C_ || !As_) {
      throw std::runtime_error(
          "LowRankPrecond: Problem data not set. Call init() before "
          "compute().");
    }
    // compute constraint matrix
    build_constraint_mat();

    const Matrix W0 = Matrix::Identity(dim, dim) * params_.tau;
    const Matrix top_right_dense =
        build_top_right(*U_, W0, params_.tau) * params_.tau;

    const int rdim = rank_ * dim;
    const int nsys = ncons + rdim;
    const double tau2 = std::pow(params_.tau, 2.0);

    // Top-left block: tau^2 * (A_bar^T A_bar)
    SpMatrix AtA = A_bar_.transpose() * A_bar_;

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<size_t>(AtA.nonZeros()) +
                  static_cast<size_t>(2 * ncons * rdim + rdim));

    for (int k = 0; k < AtA.outerSize(); ++k) {
      for (SpMatrix::InnerIterator it(AtA, k); it; ++it) {
        trips.emplace_back(it.row(), it.col(), tau2 * it.value());
      }
    }

    // Top-right and symmetric bottom-left blocks
    for (int i = 0; i < ncons; ++i) {
      for (int j = 0; j < rdim; ++j) {
        const double v = top_right_dense(i, j);
        if (v != 0.0) {
          trips.emplace_back(i, ncons + j, v);
          trips.emplace_back(ncons + j, i, v);
        }
      }
    }

    // Bottom-right block: -tau^2 * I
    for (int i = 0; i < rdim; ++i) {
      trips.emplace_back(ncons + i, ncons + i, -tau2);
    }

    SpMatrix Sys(nsys, nsys);
    Sys.setFromTriplets(trips.begin(), trips.end());
    Sys.makeCompressed();

    SparseFactor.compute(Sys);
    is_initialized_ = (SparseFactor.info() == Eigen::Success);

    return *this;
  }

  // Apply the preconditioner by solving the augmented system
  // that is, invert the preconditioner matrix and apply it to the input vector
  template <typename Rhs>
  Eigen::VectorXd solve(const Eigen::MatrixBase<Rhs>& b) const {
    auto rhs = Vector(ncons + rank_ * dim);
    rhs << b / scale_, Vector::Zero(rank_ * dim);

    Vector result;
    if (params_.use_sparse_factor) {
      result = SparseFactor.solve(rhs);
    } else {
      // result = Factor.solve(rhs);
      result = ldlt_pseudo_solve(rhs);
    }
    return result.segment(0, ncons);
  }

  // Build the top right matrix = A_bar^T(U otimes Z)
  Matrix build_top_right(const Matrix& U, const Matrix& W0, double tau) const {
    // Build the Z matrix s.t. Z Z^T = (2W0 + U U^T) = X + W0
    Matrix Z;
    if (!params_.use_approx) {
      Eigen::LLT<Matrix> llt(U * U.transpose() + 2 * W0);
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
    // Build sparse constraint matrix using triplet format for efficiency
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(dim * dim);  // Conservative estimate of non-zeros

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
              triplets.emplace_back(row_idx, i, it.value());
              row_idx = it.row() * dim + it.col();
              triplets.emplace_back(row_idx, i, it.value());

            } else if (it.col() == it.row()) {
              int row_idx = it.col() * dim + it.row();
              triplets.emplace_back(row_idx, i, it.value());
            }
          }
        }
      } else {
        // For dense matrix C, vectorize it
        for (int row = 0; row < dim; ++row) {
          for (int col = 0; col < dim; ++col) {
            double val = (*C_)(row, col);
            if (val != 0.0) {
              int row_idx = col * dim + row;
              triplets.emplace_back(row_idx, i, val);
            }
          }
        }
      }
    }

    // Construct sparse matrix from triplets
    A_bar_.resize(dim * dim, ncons);
    A_bar_.setFromTriplets(triplets.begin(), triplets.end());
    A_bar_.makeCompressed();
  }

  Eigen::ComputationInfo info() const {
    return is_initialized_ ? Eigen::Success : Eigen::InvalidInput;
  }

 private:
  bool is_initialized_;
  const Matrix* U_;
  const Matrix* C_;
  const std::vector<Eigen::SparseMatrix<double>>* As_;
  int rank_;
  int dim;
  int ncons;
  double scale_;

  // built matrices
  Eigen::SparseMatrix<double> A_bar_;
  Eigen::LDLT<Matrix> Factor;
  Eigen::SimplicialLDLT<SpMatrix> SparseFactor;

  /// @brief Solve the system using the LDLT factorization, treating small
  /// eigenvalues as zero to handle potential singularity. This effectively
  /// applies a pseudo-inverse of the factorized matrix.
  /// @param b right-hand side vector to solve for
  /// @return solution vector after applying the pseudo-inverse of the LDLT
  /// factorization
  Eigen::VectorXd ldlt_pseudo_solve(const Eigen::VectorXd& b) const {
    const auto& L = Factor.matrixL();
    const auto& P = Factor.transpositionsP();
    const auto& D = Factor.vectorD();  // diagonal of D

    // Step 1: apply permutation
    Vector rhs = P.transpose() * b;

    // Step 2: forward solve L z = rhs
    Vector z = L.solve(rhs);

    // Step 3: apply pseudo-inverse of D
    for (int i = 0; i < D.size(); ++i) {
      if (std::abs(D[i]) > params_.ldlt_zero_thresh) {
        z[i] /= D[i];
      } else {
        z[i] = 0.0;  // project out nullspace
      }
    }

    // Step 4: backward solve L^T y = z
    Eigen::VectorXd y = L.transpose().solve(z);

    // Step 5: undo permutation
    return P * y;
  }
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

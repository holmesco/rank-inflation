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

inline double sparse_upper_dot_dense(const SparseMatrix<double>& A_upper,
                                     const Eigen::MatrixXd& M) {
  double sum = 0.0;
  for (int k = 0; k < A_upper.outerSize(); ++k) {
    for (SparseMatrix<double>::InnerIterator it(A_upper, k); it; ++it) {
      const int r = it.row();
      const int c = it.col();
      if (r == c) {
        sum += it.value() * M(r, c);
      } else if (c > r) {
        sum += it.value() * (M(r, c) + M(c, r));
      }
    }
  }
  return sum;
}

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
  const int ncons;
  // perturbation parameter
  double scale_;

  // API to set the data of the linear system (the cost matrix and the
  // constraint matrices)
  MultiplierLinSys(const Eigen::MatrixXd& X,
                   const std::vector<Eigen::SparseMatrix<double>>& As,
                   const Eigen::MatrixXd& C, double scale)
      : X_(X), As_(As), C_(C), ncons(As.size() + 1), scale_(scale) {}
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
    // NOTE: Cannot parallelize this loop because of common S
    Eigen::MatrixXd S = rhs(rhs.size() - 1) * lhs.C_;
    for (int i = 0; i < rhs.size() - 1; i++) {
      S += rhs(i) * lhs.As_[i];
    }
    // Compute X*S*X once, then each output coordinate is a sparse-dense dot.
    Eigen::MatrixXd XSX = lhs.X_ * S.selfadjointView<Eigen::Upper>() * lhs.X_;
#ifdef RANKTOOLS_PARALLEL
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < lhs.ncons - 1; i++) {
      dst(i) += sparse_upper_dot_dense(lhs.As_[i], XSX) * alpha * lhs.scale_;
    }
    dst(lhs.ncons - 1) += lhs.C_.cwiseProduct(XSX).sum() * alpha * lhs.scale_;
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
    for (int i = 0; i < op.ncons - 1; ++i) {
      const Eigen::MatrixXd AX =
          op.As_[i].selfadjointView<Eigen::Upper>() * op.X_;
      const double diag_val = (AX * AX).trace() * op.scale_;
      inv_diag_(i) = (diag_val != Scalar(0)) ? 1.0 / diag_val : 1.0;
    }
    const Eigen::MatrixXd CX = op.C_.selfadjointView<Eigen::Upper>() * op.X_;
    const double diag_cost = (CX * CX).trace() * op.scale_;
    inv_diag_(op.ncons - 1) = (diag_cost != Scalar(0)) ? 1.0 / diag_cost : 1.0;
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

enum class LowRankPrecondMethod { DenseLDLT, SparseLDLT, DenseQR, SparseQR };

struct LowRankPrecondParams {
  // Diagonal perturbation parameter.
  double tau = 1e-5;
  // Flag for using the sparse factorization approach (true) vs the dense
  // approach (false) to building the preconditioner. The sparse approach is
  // cheaper to build, but may be less effective at improving conditioning.
  LowRankPrecondMethod method = LowRankPrecondMethod::SparseLDLT;
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
    if (params_.method == LowRankPrecondMethod::SparseLDLT) {
      build_ldlt_sparse();
    } else if (params_.method == LowRankPrecondMethod::DenseLDLT) {
      build_ldlt_dense();
    } else if (params_.method == LowRankPrecondMethod::DenseQR) {
      build_qr_dense();
    } else if (params_.method == LowRankPrecondMethod::SparseQR) {
      build_qr_sparse();
    } else {
      throw std::runtime_error(
          "LowRankPrecond: Unknown preconditioning method.");
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
    if (params_.method == LowRankPrecondMethod::SparseLDLT) {
      build_ldlt_sparse();
    } else if (params_.method == LowRankPrecondMethod::DenseLDLT) {
      build_ldlt_dense();
    } else if (params_.method == LowRankPrecondMethod::DenseQR) {
      build_qr_dense();
    } else if (params_.method == LowRankPrecondMethod::SparseQR) {
      build_qr_sparse();
    } else {
      throw std::runtime_error(
          "LowRankPrecond: Unknown preconditioning method.");
    }
    return *this;
  }

  // Internal compute function that does the actual work of building the
  // preconditioner. This version builds the dense augmented system and
  // factorizes it with LDLT. We could
  LowRankPrecond& build_ldlt_dense() {
    if (!U_ || !C_ || !As_) {
      throw std::runtime_error(
          "LowRankPrecond: Problem data not set. Call init() before "
          "compute().");
    }
    // compute constraint matrix
    build_constraint_mat();
    // Build sparse augmented system - Eqn 23 in Zhang and Lavaei 2017
    auto Sys = Matrix(ncons + rank_ * dim, ncons + rank_ * dim);
    Sys.block(0, 0, ncons, ncons) =
        (A_bar_.transpose() * A_bar_).template triangularView<Eigen::Upper>() *
        std::pow(params_.tau, 2.0);
    Sys.block(0, ncons, ncons, rank_ * dim) =
        build_top_right(*U_, params_.tau) * params_.tau;
    Sys.block(ncons, ncons, rank_ * dim, rank_ * dim) =
        -Matrix::Identity(rank_ * dim, rank_ * dim) * std::pow(params_.tau, 2);
    // Prefactorize (LDLT)
    LDLTDenseFactor.compute(Sys.selfadjointView<Eigen::Upper>());
    // Flag that we have initialized to eigen
    is_initialized_ = (LDLTDenseFactor.info() == Eigen::Success);
    if (!is_initialized_) {
      std::cerr << "LDLT factorization failed. Info: " << LDLTDenseFactor.info()
                << std::endl;
    }

    return *this;
  }

  // Internal compute function that does the actual work of building the
  // preconditioner. This version builds the augmented system sparsely and
  // factorizes it with SimplicialLDLT. We could
  LowRankPrecond& build_ldlt_sparse() {
    if (!U_ || !C_ || !As_) {
      throw std::runtime_error(
          "LowRankPrecond: Problem data not set. Call init() before "
          "compute().");
    }
    // compute constraint matrix
    build_constraint_mat();

    const Matrix W0 = Matrix::Identity(dim, dim) * params_.tau;
    const Matrix top_right_dense =
        build_top_right(*U_, params_.tau) * params_.tau;

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

    LDLTSparseFactor.compute(Sys);
    is_initialized_ = (LDLTSparseFactor.info() == Eigen::Success);

    return *this;
  }

  // Internal compute function that does the actual work of building the
  // preconditioner. This version builds the dense qr factorization
  LowRankPrecond& build_qr_dense() {
    if (!U_ || !C_ || !As_) {
      throw std::runtime_error(
          "LowRankPrecond: Problem data not set. Call init() before "
          "compute().");
    }
    // compute constraint matrix
    build_constraint_mat();
    Matrix A_bar_dense = A_bar_;
    // build the V = A_bar^T(U otimes Z) matrix
    auto V_dense = build_top_right(*U_, params_.tau);
    // build least squares factored matrix
    auto Sys = Eigen::MatrixXd(A_bar_dense.rows() + V_dense.cols(),
                               A_bar_dense.cols());
    Sys.topRows(A_bar_dense.rows()) = A_bar_dense * params_.tau;
    Sys.bottomRows(V_dense.cols()) = V_dense.transpose();
    // Factorize with dense QR
    QRDenseFactor.compute(Sys);
    is_initialized_ = (QRDenseFactor.info() == Eigen::Success);

    return *this;
  }

  // Internal compute function that does the actual work of building the
  // preconditioner. This version builds the sparse qr factorization
  LowRankPrecond& build_qr_sparse() {
    if (!U_ || !C_ || !As_) {
      throw std::runtime_error(
          "LowRankPrecond: Problem data not set. Call init() before "
          "compute().");
    }
    // compute constraint matrix
    build_constraint_mat();

    // build the V = A_bar^T(U otimes Z) matrix as dense
    auto V_dense = build_top_right(*U_, params_.tau);

    // Sparsify V_dense by collecting non-zero entries
    std::vector<Eigen::Triplet<double>> V_trips;
    V_trips.reserve(static_cast<size_t>(V_dense.rows() * V_dense.cols() *
                                        0.1));  // estimate 10% density
    for (int i = 0; i < V_dense.rows(); ++i) {
      for (int j = 0; j < V_dense.cols(); ++j) {
        if (V_dense(i, j) != 0.0) {
          V_trips.emplace_back(i, j, V_dense(i, j));
        }
      }
    }
    SpMatrix V_sparse(V_dense.rows(), V_dense.cols());
    V_sparse.setFromTriplets(V_trips.begin(), V_trips.end());
    V_sparse.makeCompressed();

    // Build sparse augmented system: [tau * A_bar; V_sparse^T]
    const int nrows_total = A_bar_.rows() + V_sparse.cols();
    const int ncols = A_bar_.cols();
    const double tau = params_.tau;

    std::vector<Eigen::Triplet<double>> sys_trips;
    sys_trips.reserve(
        static_cast<size_t>(A_bar_.nonZeros() + V_sparse.nonZeros()));

    // Top block: tau * A_bar_
    for (int k = 0; k < A_bar_.outerSize(); ++k) {
      for (SpMatrix::InnerIterator it(A_bar_, k); it; ++it) {
        sys_trips.emplace_back(it.row(), it.col(), tau * it.value());
      }
    }

    // Bottom block: V_sparse^T (transpose is handled by swapping indices)
    for (int k = 0; k < V_sparse.outerSize(); ++k) {
      for (SpMatrix::InnerIterator it(V_sparse, k); it; ++it) {
        sys_trips.emplace_back(A_bar_.rows() + it.col(), it.row(), it.value());
      }
    }

    SpMatrix Sys(nrows_total, ncols);
    Sys.setFromTriplets(sys_trips.begin(), sys_trips.end());
    Sys.makeCompressed();

    // Factorize with sparse QR
    QRSparseFactor.compute(Sys);
    is_initialized_ = (QRSparseFactor.info() == Eigen::Success);

    return *this;
  }

  // Apply the preconditioner by solving the augmented system
  // that is, invert the preconditioner matrix and apply it to the input vector
  template <typename Rhs>
  Eigen::VectorXd solve(const Eigen::MatrixBase<Rhs>& b) const {
    Vector result(ncons);
    if (params_.method == LowRankPrecondMethod::SparseLDLT) {
      auto rhs = Vector(ncons + rank_ * dim);
      rhs << b / scale_, Vector::Zero(rank_ * dim);
      result = LDLTSparseFactor.solve(rhs).head(ncons).eval();
    } else if (params_.method == LowRankPrecondMethod::DenseLDLT) {
      auto rhs = Vector(ncons + rank_ * dim);
      rhs << b / scale_, Vector::Zero(rank_ * dim);
      result = LDLTDenseFactor.solve(rhs).head(ncons).eval();
    } else if (params_.method == LowRankPrecondMethod::DenseQR) {
      // Solve Sys^T Sys x = b via R^T R x = b, where Sys = [A; V] = Q R is the
      // dense QR factorization of the augmented system matrix
      // Check Rank
      int rank = QRDenseFactor.rank();
      assert(rank == ncons &&
             "Column rank deficiency detected in QR factorization of low rank "
             "preconditioner.");
      auto c = QRDenseFactor.colsPermutation().inverse() * b / scale_;
      // Make a mutable copy of R to enable solve operations
      Eigen::MatrixXd R_copy =
          QRDenseFactor.matrixR().topLeftCorner(rank, rank);
      auto R = R_copy.template triangularView<Eigen::Upper>();
      // Solve R^T y = c
      Eigen::VectorXd y = R.transpose().solve(c);
      // Solve R z = y
      Eigen::VectorXd z = R.solve(y);
      // Unpermute the result
      result = QRDenseFactor.colsPermutation() * z;
    } else if (params_.method == LowRankPrecondMethod::SparseQR) {
      // Solve Sys^T Sys x = b via R^T R x = b, where Sys = [A; V] = Q R is the
      // dense QR factorization of the augmented system matrix
      // Check Rank
      int rank = QRSparseFactor.rank();
      assert(rank == ncons &&
             "Column rank deficiency detected in QR factorization of low rank "
             "preconditioner.");
      auto c = QRSparseFactor.colsPermutation().inverse() * b / scale_;
      // Make a mutable copy of R to enable solve operations
      Eigen::MatrixXd R_copy =
          QRSparseFactor.matrixR().topLeftCorner(rank, rank);
      auto R = R_copy.template triangularView<Eigen::Upper>();
      // Solve R^T y = c
      Eigen::VectorXd y = R.transpose().solve(c);
      // Solve R z = y
      Eigen::VectorXd z = R.solve(y);
      // Unpermute the result
      result = QRSparseFactor.colsPermutation() * z;
    } else {
      throw std::runtime_error(
          "LowRankPrecond: Unknown preconditioning method.");
    }
    return result;
  }

  // Build the top right matrix, V = A_bar^T(U otimes Z)
  Matrix build_top_right(const Matrix& U, double tau) const {
    // Init Matrix
    Matrix top_right(ncons, rank_ * dim);
    // define convenience variables
    const bool approx = params_.use_approx;
    const double s = std::sqrt(2.0 * tau);
    // if not approximating, store transpose of Z for computations
    // if approximating with Z Z^T = 2tau I, then we can skip building Z and
    // just use s = sqrt(2tau) as a scaling factor
    Matrix Zt;
    if (!approx) {
      Eigen::LLT<Matrix> llt(U * U.transpose() +
                             2 * Matrix::Identity(dim, dim) * tau);
      if (llt.info() != Eigen::Success) {
        throw std::runtime_error(
            "LowRankPrecond: Cholesky factorization failed when building Z "
            "matrix for top right block of preconditioner.");
      }
      Zt = llt.matrixL().transpose();
    }

#ifdef RANKTOOLS_PARALLEL
#pragma omp parallel for schedule(dynamic)
#endif
    // compute products, vectorize and store in top right block.
    for (int i = 0; i < ncons - 1; ++i) {
      Matrix AiU =
          (*As_)[i].selfadjointView<Eigen::Upper>() * U;  // dim x rank_
      if (approx) {
        // Zt = I*s
        top_right.row(i) =
            Eigen::Map<const Eigen::RowVectorXd>(AiU.data(), AiU.size()) * s;
      } else {
        Matrix mat = Zt * AiU;  // dim x rank_
        top_right.row(i) =
            Eigen::Map<const Eigen::RowVectorXd>(mat.data(), mat.size());
      }
    }

    // Last row uses C
    {
      Matrix CU = (*C_).selfadjointView<Eigen::Upper>() * U;
      if (approx) {
        top_right.row(ncons - 1) =
            Eigen::Map<const Eigen::RowVectorXd>(CU.data(), CU.size()) * s;
      } else {
        Matrix mat = Zt * CU;
        top_right.row(ncons - 1) =
            Eigen::Map<const Eigen::RowVectorXd>(mat.data(), mat.size());
      }
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

  // built constraint matrix A_bar = [vec(A1) ... vec(Am) vec(C)]
  Eigen::SparseMatrix<double> A_bar_;
  // Factorizations
  Eigen::LDLT<Matrix> LDLTDenseFactor;
  Eigen::SimplicialLDLT<SpMatrix> LDLTSparseFactor;
  Eigen::SparseQR<SpMatrix, Eigen::COLAMDOrdering<int>> QRSparseFactor;
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> QRDenseFactor;

  /// @brief Solve the system using the LDLT factorization, treating small
  /// eigenvalues as zero to handle potential singularity. This effectively
  /// applies a pseudo-inverse of the factorized matrix.
  /// @param b right-hand side vector to solve for
  /// @return solution vector after applying the pseudo-inverse of the LDLT
  /// factorization
  Eigen::VectorXd ldlt_pseudo_solve(const Eigen::VectorXd& b) const {
    const auto& L = LDLTDenseFactor.matrixL();
    const auto& P = LDLTDenseFactor.transpositionsP();
    const auto& D = LDLTDenseFactor.vectorD();  // diagonal of D

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

#include "interior_point_sdp.hpp"

using namespace mosek::fusion;
using namespace monty;  // Fusion uses monty for Matrix types
using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
namespace SDPTools {

Matrix::t eigenToMosekSparse(const SpMat& eigenMat) {
    // 1. Ensure the matrix is in compressed format (CCS)
    if (!eigenMat.isCompressed()) {
        throw std::runtime_error("Input Eigen sparse matrix must be in compressed column format (CCS).");
    }

    // 2. Extract dimensions
    int nrows = (int)eigenMat.rows();
    int ncols = (int)eigenMat.cols();
    int nnz  = (int)eigenMat.nonZeros();

    // Get triplet data from Eigen
    std::vector<double> values(nnz);
    std::vector<int> rows(nnz);  // row indices for ColMajor 
    std::vector<int> cols(nnz); // column pointers for ColMajor
    for (int k = 0; k < eigenMat.outerSize(); ++k) {
        // k is the column index for ColMajor (default)
        for (SpMat::InnerIterator it(eigenMat, k); it; ++it) {
            rows.push_back(it.row());    // Row index
            cols.push_back(it.col());    // Row index
            values.push_back(it.value());    // Row index
        }
    }
    // 4. Create MOSEK ndarrays from Eigen's raw pointers
    // Note: Fusion uses shared_ptr style ndarrays for memory management
    auto m_vals  = new_array_ptr<double>(values);
    auto m_rows  = new_array_ptr<int>(rows);
    auto m_cols  = new_array_ptr<int>(cols);

    // 5. Construct the Fusion Sparse Matrix (Compressed Column Format)
    return Matrix::sparse(nrows, ncols, m_rows, m_cols, m_vals);
}

 
mosek::fusion::Matrix::t eigenToMosekDense(const Eigen::MatrixXd& eigenMat) {
    int rows = (int)eigenMat.rows();
    int cols = (int)eigenMat.cols();
    // Column-major storage: element (i,j) at index j*rows + i
    std::vector<double> data(rows * cols, 0.0);
    for (int j = 0; j < cols; ++j)
      for (int i = 0; i < rows; ++i)
        data[j * rows + i] = eigenMat(i, j);
    // Create a contiguous array for MOSEK (column-major order)
    auto m_data = new_array_ptr<double>(data);

    // 3. Construct the Fusion Dense Matrix.
    // This overload takes (rows, cols, data_array).
    return Matrix::dense(rows, cols, m_data);
}


// Solve and return primal and duals
SDPResult solve_sdp_mosek(const Eigen::MatrixXd& C,
                          const std::vector<Eigen::SparseMatrix<double>>& As,
                          const Eigen::VectorXd& b) {
  const int n = static_cast<int>(C.rows());
  if (C.cols() != n) throw std::runtime_error("C must be square");
  if (static_cast<int>(As.size()) != b.size())
    throw std::runtime_error("As and b size mismatch");
  const int m = static_cast<int>(As.size());

  // Create model
  Model::t M = new Model("sdp_with_duals");
  M->setLogHandler([](const std::string& msg) { /* no-op */ });

  // PSD variable
  Variable::t X = M->variable("X", n, Domain::inPSDCone(n));

  // Objective
  auto Cmat = eigenToMosekDense(C);
  M->objective("obj", ObjectiveSense::Minimize, Expr::dot(Cmat, X));

  // Constraints and keep references to get duals
  std::vector<Constraint::t> cons;
  cons.reserve(m);
  for (int k = 0; k < m; ++k) {
    auto Amat = eigenToMosekSparse(As[k]);
    Constraint::t c = M->constraint("c" + std::to_string(k), Expr::dot(Amat, X),
                                    Domain::equalsTo(b(k)));
    cons.push_back(c);
  }
  // Enable verbose logging to stdout
  M->setLogHandler(
      [](const std::string& msg) { std::cout << msg << std::flush; });
  // Solve
  M->solve();

  // Primal X
  auto xvals = X->level();
  if (static_cast<int>(xvals->size()) != n * n)
    throw std::runtime_error("unexpected solution length");
  Eigen::MatrixXd Xsol(n, n);
  int idx = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) Xsol(i, j) = (*xvals)[idx++];
  Eigen::MatrixXd Xsym = 0.5 * (Xsol + Xsol.transpose());

  // Duals for equality constraints
  Eigen::VectorXd y(m);
  for (int k = 0; k < m; ++k) {
    // constraint dual is scalar (equality)
    auto dv = cons[k]->dual();
    if (static_cast<int>(dv->size()) != 1)
      throw std::runtime_error("unexpected dual size for constraint");
    y(k) = (*dv)[0];
  }

  // Dual matrix for PSD cone (same layout as X->level())
  auto svals = X->dual();
  if (static_cast<int>(svals->size()) != n * n)
    throw std::runtime_error("unexpected dual matrix length");
  Eigen::MatrixXd Ssol(n, n);
  idx = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) Ssol(i, j) = (*svals)[idx++];
  Eigen::MatrixXd Ssym = 0.5 * (Ssol + Ssol.transpose());

  SDPResult res;
  res.X = std::move(Xsym);
  res.y = std::move(y);
  res.S = std::move(Ssym);
  return res;
}
}  // namespace SDPTools
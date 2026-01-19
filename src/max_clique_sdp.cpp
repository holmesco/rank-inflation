#include "max_clique_sdp/lovasz_theta_sdp.hpp"

namespace bp = boost::process;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

namespace clipperplus {

LovaszThetaProblem::LovaszThetaProblem(const Graph &graph_in,
                                       CuhallarParams cuhallar_params_in)
    : graph(graph_in), cuhallar_params(cuhallar_params_in) {
  size = graph.size();
  // generate list of edges and nonedges
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph.is_edge(i, j)) {
        edges.push_back({i, j});
      } else {
        nonedges.push_back({i, j});
      }
    }
  }
  // Determine whether to use the sparse formulation based on number of
  // constraints that will appear in the problem
  if (size + edges.size() < nonedges.size()) {
    use_sparse = true;
  } else {
    use_sparse = false;
  }
}

int LovaszThetaProblem::build_dense_mc_hslr_problem(
    const std::string &filepath) const {
  // Formulation of the problem used when the graph is dense. Corresponds to the
  // standard primal Lovasz Theta formulation get sizes
  int n = graph.size();
  int m = nonedges.size();
  std::ofstream ofs(filepath);
  // check fail
  if (ofs.fail()) {
    std::string error_msg = "File not found or could not be opened: ";
    error_msg += filepath;
    throw std::runtime_error(error_msg);
  }
  // parameters: num constraints, problem size
  ofs << m << " " << n << std::endl;
  // constraint vector
  for (int i = 0; i < m; ++i) {
    ofs << "0.0 ";
  }
  ofs << std::endl;
  // Trace value
  ofs << "1.0" << std::endl << std::endl;

  // objective
  ofs << "0 LR" << std::endl;
  for (int i = 0; i < n; ++i) {
    ofs << "1.0 ";
  }
  ofs << "; -1.0" << std::endl;

  // constraints
  for (int i = 0; i < m; ++i) {
    Edge e = edges[i];
    ofs << std::endl;
    ofs << i + 1 << " SP" << std::endl;
    ofs << edges[i].first + 1 << " " << edges[i].second + 1 << " 1.0"
        << std::endl;
  }

  // Close file
  ofs.close();
  std::cout << "Successfully wrote model file." << std::endl;

  return 0;
}

int LovaszThetaProblem::build_sparse_mc_hslr_problem(
    const std::string &filepath) const {
  // Sparse Formulation implements the problem in dual form, to leverage the
  // fact that there are fewer edges than non-edges

  // Our variable size is now (n+2) because we need to add a dual variable
  // for the trace constraint.
  int n = graph.size() + 1;
  // "constraints" actually enumerate the variables in the solution matrix
  int m = edges.size() + graph.size();
  std::ofstream ofs(filepath);
  // check fail
  if (ofs.fail()) {
    std::string error_msg = "File not found or could not be opened: ";
    error_msg += filepath;
    throw std::runtime_error(error_msg);
  }
  // parameters: num constraints, problem size
  ofs << m << " " << n << std::endl;
  // constraint vector (diagonal cost terms)
  for (int i = 0; i < graph.size(); i++) {
    ofs << "-1.0 ";
  }
  // constraint vector (off-diagonal cost terms)
  for (Edge edge : edges) {
    ofs << "-2.0 ";
  }
  ofs << std::endl;
  // Trace value (set large to "remove" the constraint)
  float trace_val_pri = n * 100.0;
  float trace_val_dual = trace_val_pri;
  ofs << trace_val_pri << std::endl << std::endl;

  // objective (implements trace constraint in the dual)
  ofs << "0 SP" << std::endl;
  ofs << graph.size() + 1 << " " << graph.size() + 1 << " " << trace_val_dual
      << std::endl
      << std::endl;

  // matrix count
  int mat_num = 1;
  // constraints (diagonal elements)
  for (int i = 0; i < graph.size(); i++) {
    ofs << mat_num << " SP" << std::endl;
    ofs << i + 1 << " " << i + 1 << " 1.0 " << std::endl;
    ofs << n << " " << n << " -1.0 " << std::endl << std::endl;

    mat_num++;
  }
  // constraints (offdiagonal elements)
  for (Edge edge : edges) {
    ofs << mat_num << " SP" << std::endl;
    ofs << edge.first + 1 << " " << edge.second + 1 << " 1.0" << std::endl
        << std::endl;
    mat_num++;
  }

  // Close file
  ofs.close();
  std::cout << "Successfully wrote model file." << std::endl;

  return 0;
}

int LovaszThetaProblem::build_initialization_file(
    const std::string &filepath, const std::vector<Node> &init_clique) const {
  std::ofstream ofs(filepath);
  // check fail
  if (ofs.fail()) {
    std::string error_msg = "File not found or could not be opened: ";
    error_msg += filepath;
    throw std::runtime_error(error_msg);
  }

  // Build initialization vector
  Vector init_soln = clique_to_soln(init_clique, size);

  // Write to file as CSV
  if (use_sparse) {
    int n = size + 1;
    int cols = 1;
    Matrix M;
    double frob;
    do {
      M = Matrix::Random(n, cols);
      frob = M.norm();
    } while (frob == 0.0);
    // ensure Frobenius norm strictly less than 1
    if (frob >= 1.0) {
      M *= 0.999 / frob;
    }
    // write as CSV (rows)
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < cols; ++j) {
        ofs << M(i, j);
        if (j < cols - 1)
          ofs << ",";
      }
      ofs << std::endl;
    }
    ofs << std::endl;

  } else {
    for (int i = 0; i < init_soln.size(); ++i) {
      ofs << init_soln(i);
      if (i < init_soln.size() - 1)
        ofs << std::endl;
    }
    ofs << std::endl;
  }

  // Close file
  ofs.close();
  std::cout << "Successfully wrote initialization file." << std::endl;

  return 0;
}

LovaszThetaSolution LovaszThetaProblem::optimize_cuhallar(
    const std::vector<int> &init_clique) const {
  // Set up input and output files
  std::string input_file = cuhallar_params.input_file;
  std::string init_file = cuhallar_params.init_file;
  std::string primal_out = cuhallar_params.primal_out;
  std::string dual_out = cuhallar_params.dual_out;
  std::string options = cuhallar_params.options;
  // Generate the problem description
  if (use_sparse) {
    options = "/workspace/parameters/cuhallar_params_sparse.cfg";
    build_sparse_mc_hslr_problem(input_file);
  } else {
    build_dense_mc_hslr_problem(input_file);
  }
  // Set up initial clique file if provided
  build_initialization_file(init_file, init_clique);
  // Launch the child process
  bp::child c(bp::search_path("cuHallar"), "-i", input_file, "-p", primal_out,
              "-d", dual_out, "-w", init_file, "-c", options);
  // Wait for the process to exit and get the exit code
  c.wait();
  int result = c.exit_code();
  std::cout << "Process finished with code: " << result << std::endl;

  // retrieve solution from output files
  LovaszThetaSolution solution = retrieve_cuhallar_solution();

  return solution;
}

LovaszThetaSolution LovaszThetaProblem::retrieve_cuhallar_solution() const {
  // Read primal and dual outputs into Eigen types
  LovaszThetaSolution solution;
  std::ifstream primal_file(cuhallar_params.primal_out);
  std::ifstream dual_file(cuhallar_params.dual_out);

  if (!primal_file.is_open() || !dual_file.is_open()) {
    std::cerr << "Error opening output files." << std::endl;
    return solution;
  }

  // Read primal matrix (CSV) into a temporary vector of rows
  std::string line;
  std::vector<std::vector<double>> temp_rows;
  while (std::getline(primal_file, line)) {
    if (line.empty())
      continue;
    std::vector<double> row;
    std::stringstream ss(line);
    std::string value;
    while (std::getline(ss, value, ',')) {
      row.push_back(std::stod(value));
    }
    if (!row.empty())
      temp_rows.push_back(std::move(row));
  }

  // Convert to Matrix
  if (!temp_rows.empty()) {
    size_t r = temp_rows.size();
    size_t c = temp_rows[0].size();
    Matrix Y(r, c);
    for (size_t i = 0; i < r; ++i) {
      for (size_t j = 0; j < c; ++j) {
        Y(static_cast<int>(i), static_cast<int>(j)) = temp_rows[i][j];
      }
    }
    solution.Y = std::move(Y);
  } else {
    solution.Y = Matrix(); // empty
  }

  // Read dual vector (CSV single line or comma-separated)
  std::vector<double> temp_dual;
  if (std::getline(dual_file, line)) {
    std::stringstream ss(line);
    std::string value;
    while (std::getline(ss, value, ',')) {
      temp_dual.push_back(std::stod(value));
    }
  }

  // Convert to Vector
  if (!temp_dual.empty()) {
    Vector lag(static_cast<int>(temp_dual.size()));
    for (size_t i = 0; i < temp_dual.size(); ++i) {
      lag(static_cast<int>(i)) = temp_dual[i];
    }
    solution.lagrange = std::move(lag);
  } else {
    solution.lagrange = Vector(); // empty
  }

  primal_file.close();
  dual_file.close();

  return solution;
}

std::vector<int> LovaszThetaProblem::soln_to_clique(const Vector &soln) const {
  // Convert to abs array
  auto sol_array = soln.array().abs();
  // Identify the clique indices
  double mid_val = (sol_array.maxCoeff() - sol_array.minCoeff()) / 2;
  std::vector<int> clique;
  for (int i = 0; i < sol_array.size(); i++) {
    if (sol_array(i) > mid_val) {
      clique.push_back(i);
    }
  }
  return clique;
}

Vector LovaszThetaProblem::clique_to_soln(const std::vector<int> clique,
                                          int size) const {
  assert(clique.size() > 0);
  double factor = 1 / std::sqrt(clique.size());
  Eigen::VectorXd soln = Eigen::VectorXd::Zero(size);
  for (int i; i < clique.size(); i++) {
    soln(i) = factor;
  }
  return soln;
}

} // namespace clipperplus
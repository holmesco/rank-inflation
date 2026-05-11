#include <yaml-cpp/yaml.h>

#include <Eigen/Sparse>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "analytic_center.hpp"

namespace {

using namespace RankTools;

LinearSolverType parse_linear_solver(const YAML::Node& node,
                                     const std::string& key) {
  const std::string value = node[key].as<std::string>();
  if (value == "LDLT") return LinearSolverType::LDLT;
  if (value == "CG") return LinearSolverType::CG;
  if (value == "MFCG_DP") return LinearSolverType::MFCG_DP;
  if (value == "MFCG_LRP") return LinearSolverType::MFCG_LRP;
  throw std::runtime_error("Invalid lin_solver value in YAML: " + value);
}

LowRankPrecondMethod parse_lrp_method(const YAML::Node& node,
                                      const std::string& key) {
  const std::string value = node[key].as<std::string>();
  if (value == "DenseLDLT") return LowRankPrecondMethod::DenseLDLT;
  if (value == "SparseLDLT") return LowRankPrecondMethod::SparseLDLT;
  if (value == "SparseLDLT_ZL") return LowRankPrecondMethod::SparseLDLT_ZL;
  if (value == "DenseQR") return LowRankPrecondMethod::DenseQR;
  if (value == "SparseQR") return LowRankPrecondMethod::SparseQR;
  throw std::runtime_error("Invalid lrp_params.method value in YAML: " + value);
}

void apply_param_overrides(const YAML::Node& node, AnalyticCenterParams& params,
                           const std::string& scope = "") {
  if (!node || !node.IsMap()) {
    return;
  }

  auto key_exists = [&](const char* key) {
    return node[key] && !node[key].IsNull();
  };

  if (key_exists("verbose")) params.verbose = node["verbose"].as<bool>();
  if (key_exists("tol_rank_sol"))
    params.tol_rank_sol = node["tol_rank_sol"].as<double>();
  if (key_exists("tol_step_norm"))
    params.tol_step_norm = node["tol_step_norm"].as<double>();
  if (key_exists("max_iter")) params.max_iter = node["max_iter"].as<int>();
  if (key_exists("rescale_lin_sys"))
    params.rescale_lin_sys = node["rescale_lin_sys"].as<bool>();
  if (key_exists("rescaling_factor"))
    params.rescaling_factor = node["rescaling_factor"].as<double>();
  if (key_exists("lin_solver"))
    params.lin_solver = parse_linear_solver(node, "lin_solver");
  if (key_exists("reuse_multipliers"))
    params.reuse_multipliers = node["reuse_multipliers"].as<bool>();

  if (key_exists("tol_indep_constr"))
    params.tol_indep_constr = node["tol_indep_constr"].as<double>();
  if (key_exists("check_indep_constr"))
    params.check_indep_constr = node["check_indep_constr"].as<bool>();
  if (key_exists("delta")) params.delta = node["delta"].as<double>();

  if (key_exists("perturb_constraints"))
    params.perturb_constraints = node["perturb_constraints"].as<bool>();
  if (key_exists("perturb_cost"))
    params.perturb_cost = node["perturb_cost"].as<bool>();
  if (key_exists("eps_cost")) params.eps_cost = node["eps_cost"].as<double>();
  if (key_exists("eps_constr"))
    params.eps_constr = node["eps_constr"].as<double>();
  if (key_exists("adaptive_perturb"))
    params.adaptive_perturb = node["adaptive_perturb"].as<bool>();
  if (key_exists("eps_mult_min"))
    params.eps_mult_min = node["eps_mult_min"].as<double>();
  if (key_exists("eps_inc_step_thresh"))
    params.eps_inc_step_thresh = node["eps_inc_step_thresh"].as<double>();
  if (key_exists("eps_inc")) params.eps_inc = node["eps_inc"].as<double>();
  if (key_exists("eps_dec_step_thresh"))
    params.eps_dec_step_thresh = node["eps_dec_step_thresh"].as<double>();
  if (key_exists("eps_dec")) params.eps_dec = node["eps_dec"].as<double>();

  if (key_exists("lin_solve_max_iter"))
    params.lin_solve_max_iter = node["lin_solve_max_iter"].as<int>();
  if (key_exists("lin_solve_tol"))
    params.lin_solve_tol = node["lin_solve_tol"].as<double>();

  if (key_exists("enable_line_search"))
    params.enable_line_search = node["enable_line_search"].as<bool>();
  if (key_exists("ln_search_red_factor"))
    params.ln_search_red_factor = node["ln_search_red_factor"].as<double>();
  if (key_exists("alpha_init"))
    params.alpha_init = node["alpha_init"].as<double>();
  if (key_exists("alpha_min"))
    params.alpha_min = node["alpha_min"].as<double>();

  if (key_exists("early_stop_cert"))
    params.early_stop_cert = node["early_stop_cert"].as<bool>();
  if (key_exists("tol_cert_psd"))
    params.tol_cert_psd = node["tol_cert_psd"].as<double>();
  if (key_exists("tol_cert_complementarity"))
    params.tol_cert_complementarity =
        node["tol_cert_complementarity"].as<double>();
  if (key_exists("tol_cert_primal_feas"))
    params.tol_cert_primal_feas = node["tol_cert_primal_feas"].as<double>();
  if (key_exists("early_stop_angle"))
    params.early_stop_angle = node["early_stop_angle"].as<bool>();
  if (key_exists("max_angle"))
    params.max_angle = node["max_angle"].as<double>();
  if (key_exists("use_cert_centrality_metric"))
    params.use_cert_centrality_metric =
        node["use_cert_centrality_metric"].as<bool>();
  if (key_exists("tol_cert_centrality"))
    params.tol_cert_centrality = node["tol_cert_centrality"].as<double>();

  if (node["lrp_params"] && node["lrp_params"].IsMap()) {
    const YAML::Node lrp = node["lrp_params"];
    if (lrp["tau"]) params.lrp_params.tau = lrp["tau"].as<double>();
    if (lrp["method"])
      params.lrp_params.method = parse_lrp_method(lrp, "method");
    if (lrp["use_approx"])
      params.lrp_params.use_approx = lrp["use_approx"].as<bool>();
    if (lrp["ldlt_zero_thresh"])
      params.lrp_params.ldlt_zero_thresh = lrp["ldlt_zero_thresh"].as<double>();
  }

  if (node["tau_lrp"]) params.lrp_params.tau = node["tau_lrp"].as<double>();

  (void)scope;
}

AnalyticCenterParams load_params_from_yaml(const std::filesystem::path& path) {
  AnalyticCenterParams params;
  if (path.empty()) {
    return params;
  }

  const std::string path_str = path.string();
  const YAML::Node root = YAML::LoadFile(path_str);
  apply_param_overrides(root, params);

  return params;
}

void expect_key(std::istream& in, const std::string& expected,
                const std::string& file_name) {
  std::string key;
  if (!(in >> key)) {
    throw std::runtime_error("Unexpected end-of-file in " + file_name +
                             ", expected key: " + expected);
  }
  if (key != expected) {
    throw std::runtime_error("Malformed problem file " + file_name +
                             ": expected key '" + expected + "', got '" + key +
                             "'");
  }
}

struct LoadedProblem {
  int dim = 0;
  Matrix C;
  double rho = 0.0;
  std::vector<Eigen::SparseMatrix<double>> A;
  std::vector<double> b;
  Matrix soln;
  std::string name;
};

LoadedProblem load_problem_file(const std::filesystem::path& path) {
  const std::string path_str = path.string();
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open problem file: " + path_str);
  }

  LoadedProblem sdp;

  expect_key(in, "name", path_str);
  in >> sdp.name;

  expect_key(in, "dim", path_str);
  in >> sdp.dim;

  expect_key(in, "C", path_str);
  int c_count = 0;
  in >> c_count;
  if (c_count != sdp.dim * sdp.dim) {
    throw std::runtime_error("Invalid C size in " + path_str);
  }

  sdp.C.resize(sdp.dim, sdp.dim);
  for (int i = 0; i < sdp.dim; ++i) {
    for (int j = 0; j < sdp.dim; ++j) {
      if (!(in >> sdp.C(i, j))) {
        throw std::runtime_error("Failed reading C entries from " + path_str);
      }
    }
  }
  // remove homogenization offset
  sdp.C(0, 0) = 0.0;
  // Rescale C to have norm 1, to avoid numerical issues in testing.
  sdp.C /= sdp.C.norm();

  expect_key(in, "constraints", path_str);
  int n_constraints = 0;
  in >> n_constraints;

  sdp.A.clear();
  sdp.b.clear();
  sdp.A.reserve(static_cast<std::size_t>(n_constraints));
  sdp.b.reserve(static_cast<std::size_t>(n_constraints));

  for (int k = 0; k < n_constraints; ++k) {
    expect_key(in, "A", path_str);
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    in >> rows >> cols >> nnz;

    Eigen::SparseMatrix<double> A(rows, cols);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<std::size_t>(nnz));

    for (int t = 0; t < nnz; ++t) {
      int r = 0;
      int c = 0;
      double v = 0.0;
      if (!(in >> r >> c >> v)) {
        throw std::runtime_error("Failed reading triplet entries from " +
                                 path_str);
      }
      triplets.emplace_back(r, c, v);
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
    sdp.A.push_back(std::move(A));

    expect_key(in, "b", path_str);
    double bval = 0.0;
    in >> bval;
    sdp.b.push_back(bval);
  }

  expect_key(in, "soln", path_str);
  int soln_rows = 0;
  int soln_cols = 0;
  int soln_count = 0;
  in >> soln_rows >> soln_cols >> soln_count;

  if (soln_rows * soln_cols != soln_count) {
    throw std::runtime_error("Invalid solution shape/count in " + path_str);
  }

  sdp.soln.resize(soln_rows, soln_cols);
  for (int i = 0; i < soln_rows; ++i) {
    for (int j = 0; j < soln_cols; ++j) {
      if (!(in >> sdp.soln(i, j))) {
        throw std::runtime_error("Failed reading solution entries from " +
                                 path_str);
      }
    }
  }

  sdp.rho = (sdp.soln.transpose() * sdp.C * sdp.soln).trace();
  return sdp;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <problem_file> [params_yaml_file]\n";
    return 1;
  }

  try {
    const std::filesystem::path problem_path(argv[1]);
    const auto sdp = load_problem_file(problem_path);

    AnalyticCenterParams params;
    if (argc == 3) {
      params = load_params_from_yaml(argv[2]);
    }

    AnalyticCenter problem(sdp.C, sdp.rho, sdp.A, sdp.b, params);
    auto result = problem.certify(sdp.soln);

    std::cout << "Certified: " << (result.certified ? "true" : "false") << '\n';
    std::cout << "Minimum eigenvalue: " << result.min_eig << '\n';
    std::cout << "Complementarity: " << result.complementarity << '\n';
    std::cout << "Solver time (s): " << result.solver_time << '\n';
    return result.certified ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}

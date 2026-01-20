#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>
#include <boost/process.hpp>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>

// include graph and rank reduction code.
#include "clipperplus/clipperplus_graph.h"
#include "max_clique_sdp/rank_reduction.hpp"

namespace clipperplus {
struct CuhallarParams {
  std::string input_file = "/workspace/tmp/mc_hslr_problem.txt";
  std::string init_file = "/workspace/tmp/mc_initialization.txt";
  std::string primal_out = "/workspace/tmp/primal_out.txt";
  std::string dual_out = "/workspace/tmp/dual_out.txt";
  std::string options = "/workspace/parameters/cuhallar_params.cfg";
  std::string options_sparse = "/workspace/parameters/cuhallar_params_sparse.cfg";
  std::string log_file = "/workspace/tmp/log.txt";
  
};

struct CuhallarStats {
  double primal_obj;
  double dual_obj;
  double pd_gap;
  double primal_infeasibility;
  int adap_fista_calls;
  int acg_iterations;
  int fw_calls;
  double primal_val_unscaled;
  double run_time_seconds;
};

// Struct to store the low rank solution of a max clique SDP problem
struct LovaszThetaSolution {
  // primal solution
  Eigen::MatrixXd Y;
  // lagrange multipliers
  Eigen::VectorXd lagrange;
  // primal optimum
  float primal_opt;
  // solver stats
  CuhallarStats stats;
};

// Object to encode the Lovasz-theta SDP problem.
class LovaszThetaProblem {

public:
  // problem size
  int size;
  // list of edges (for sparse formulation)
  std::vector<Edge> edges;
  // list of non-edges (for dense formulation)
  std::vector<Edge> nonedges;
  // flag use sparse formulation
  bool use_sparse;
  // problem consistency graph
  const Graph &graph;
  // parameters for CuHALLaR optimizer
  CuhallarParams cuhallar_params;

  // Construct a LovaszThetaProblem from a graph
  LovaszThetaProblem(const Graph &graph_in,
                     CuhallarParams params_in = CuhallarParams());

  // Run max clique optimization and get the solution
  LovaszThetaSolution optimize_cuhallar(
      const std::vector<Node> &init_clique = std::vector<Node>()) const;

  // Convert a solution vector to a list of indices representing the optimal
  // clique
  std::vector<int> soln_to_clique(const Eigen::VectorXd &soln) const;

  // Convert a vector of indices representing a clique to a feasible vector for
  // our problem.
  Eigen::VectorXd clique_to_soln(const std::vector<int> clique, int size) const;

private:
  // Build dense max clique formulation in hslr format for cuhallar
  int build_dense_mc_hslr_problem(const std::string &filepath) const;

  // Build sparse max clique formulation in hslr format for cuhallar
  int build_sparse_mc_hslr_problem(const std::string &filepath) const;

  // Build initialization file
  int build_initialization_file(const std::string &filepath,
                                const std::vector<Node> &init_clique) const;

  // Retrieve the solution from the cuhallar output files
  // Assumes that outputs are in the root directory with default names
  LovaszThetaSolution
  retrieve_cuhallar_solution(std::vector<std::string> log) const;

  // Helper for parsing info from cuhallar
  CuhallarStats
  parse_solver_output(const std::vector<std::string> &lines) const;
};

} // namespace clipperplus
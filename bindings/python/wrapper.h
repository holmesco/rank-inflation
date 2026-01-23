/**
 * @file wrappers.h
 * @brief Wrapper for Clipperplus' C++ functions with parameters passed by
 * reference
 */

#pragma once

#include "clipperplus/clipperplus_clique.h"
#include "clipperplus/clique_optimization.h"

#include <pybind11/pybind11.h>

class Wrapper {
public:
  static std::tuple<long, std::vector<int>, std::string> find_clique_wrapper(
      const Eigen::MatrixXd &adj,
      clipperplus::ClipperParams params = clipperplus::ClipperParams(),
      std::shared_ptr<clipperplus::SolutionInfo> info = nullptr) {
    auto [clique, certificate] = clipperplus::find_clique(adj, params, info);
    std::string cert_str = clipperplus::to_string(certificate);
    return std::make_tuple((long)clique.size(), clique, cert_str);
  }

  static std::vector<int>
  find_heuristic_clique_wrapper(const Eigen::MatrixXd &adj,
                                std::vector<int> &clique) {
    clique = clipperplus::find_heuristic_clique(adj);
    return clique;
  }

  static std::tuple<int, unsigned long, std::vector<long>>
  clique_optimization_wrapper(const Eigen::MatrixXd &M,
                              const Eigen::VectorXd &u0,
                              const clipperplus::ClipperOptParams params) {
    std::vector<long> clique = clipperplus::clique_optimization(M, u0, params);
    return std::make_tuple(1, clique.size(), clique);
  }
};

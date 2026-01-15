#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <iostream>

#include "clipperplus/clipperplus_graph.h"
#include "clipperplus/clipperplus_heuristic.h"
#include "clipperplus/clique_optimization.h"
#include "clipperplus/utils.h"
#include "max_clique_sdp/max_clique_sdp.hpp"

namespace clipperplus {
struct ClipperParams {
  // Local optimization parameters
  ClipperOptParams optim_params = ClipperOptParams();
  // Whether to check Lovasz-theta SDP bound after optimization
  bool check_lovasz_theta = true;
  // CuHallar parameters
  CuhallarParams cuhallar_params = CuhallarParams();
  // Rank Reduction parameters
  RankReduction::RankRedParams rank_red_params = RankReduction::RankRedParams();
};

enum class CERTIFICATE {
  NONE,
  HEURISTIC,
  CORE_BOUND,
  CHROMATIC_BOUND,
  LOVASZ_THETA_BOUND,
  LOVASZ_THETA_SOLN_BOUND,
};

inline std::string to_string(CERTIFICATE cert) {
  switch (cert) {
  case CERTIFICATE::NONE:
    return "NONE";
  case CERTIFICATE::HEURISTIC:
    return "HEURISTIC";
  case CERTIFICATE::CORE_BOUND:
    return "CORE_BOUND";
  case CERTIFICATE::CHROMATIC_BOUND:
    return "CHROMATIC_BOUND";
  case CERTIFICATE::LOVASZ_THETA_BOUND:
    return "LOVASZ_THETA_BOUND";
  default:
    return "UNKNOWN";
  }
}

std::pair<std::vector<Node>, CERTIFICATE> find_clique(const Graph &graph, ClipperParams params = ClipperParams());

} // namespace clipperplus

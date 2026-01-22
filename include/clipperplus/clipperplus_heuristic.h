#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <Eigen/Dense>
#include <unordered_set>

#include "clipperplus/clipperplus_graph.h"

namespace clipperplus
{
    // Estimate the maximum clique in a graph via heuristic algorithm.
    // See Algorithm 1 of Fathian and Summers (2024)
    std::vector<Node> find_heuristic_clique(
        const clipperplus::Graph &graph,
        int upper_bound = -1,
        int lower_bound = 0);

    int estimate_chromatic_number_welsh_powell(const Graph &graph);

}

#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/process.hpp>
#include <cmath>
#include <algorithm>

// include graph and rank reduction code.
#include "clipperplus/clipperplus_graph.h"
#include "max_clique_sdp/rank_reduction.hpp"

namespace clipperplus
{
    // Object to store the low rank solution of a max clique SDP problem
    struct MaxCliqueSolution
    {
        // primal solution
        Eigen::MatrixXd Y;
        // lagrange multipliers
        Eigen::VectorXd lagrange;
        // primal optimum
        float primal_opt;
    };

    // Object to encode the maximum clique problem.
    class MaxCliqueProblem
    {

    public:
        int size;
        std::vector<Edge> abs_edges;

        const Graph &graph;

        MaxCliqueProblem(const Graph &graph_in);

        // Run max clique optimization and get the solution
        MaxCliqueSolution optimize_cuhallar();

    private:
        // Build max clique problem in hslr format for cuhallar
        int build_mc_hslr_problem(const std::string &filepath);

        // Retrieve the solution from the cuhallar output files
        // Assumes that outputs are in the root directory with default names
        MaxCliqueSolution retrieve_cuhallar_solution();
    };


}
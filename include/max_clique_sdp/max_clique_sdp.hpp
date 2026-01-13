#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/process.hpp>
#include <cmath>
#include <algorithm>
#include <cassert>

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
        MaxCliqueSolution optimize_cuhallar(const std::vector<Node>& init_clique = std::vector<Node>()) const;

        // Convert a solution vector to a list of indices representing the optimal clique
        std::vector<int> soln_to_clique(const Eigen::VectorXd& soln) const;
    
        // Convert a vector of indices representing a clique to a feasible vector for our problem.
        Eigen::VectorXd clique_to_soln(const std::vector<int> clique, int size) const;
        
    private:
        // Build max clique problem in hslr format for cuhallar
        int build_mc_hslr_problem(const std::string &filepath) const;

        // Build initialization file
        int build_initialization_file(const std::string &filepath, const std::vector<Node>& init_clique) const;

        // Retrieve the solution from the cuhallar output files
        // Assumes that outputs are in the root directory with default names
        MaxCliqueSolution retrieve_cuhallar_solution() const;

    };


}
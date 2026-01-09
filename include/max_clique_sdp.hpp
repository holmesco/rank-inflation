#pragma once

#include "clipperplus/clipperplus_graph.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/process.hpp>
#include <cmath>
#include <algorithm>

namespace clipperplus
{

    struct MaxCliqueSolution
    {
        // primal solution
        Eigen::MatrixXd Y;
        // lagrange multipliers
        Eigen::VectorXd lagrange;
        // primal optimum
        float primal_opt;
    };

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

    // Apply rank reduction algorithm
    Eigen::MatrixXd rank_reduction(
        const std::vector<Edge> &absent_edges,
        const Eigen::MatrixXd &V_init,
        double rank_tol = 1e-5,
        double null_tol = 1e-5,
        double eig_tol = 1e-9,
        int targ_rank = -1,
        int max_iter = -1,
        bool verbose = true);

}
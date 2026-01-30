#pragma once
#include <Eigen/Dense>
#include <utility>
#include <iostream>

namespace RankReduction
{
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
    using Node = int;
    using Edge = std::pair<Node, Node>;

    
    struct RankRedParams
    {
        // Target rank, supercedes other conditionals (ignored if -1)
        int targ_rank = 1; 
        // Tolerance on null space singular value. If no null space and target rank not set, exit.
        double null_tol = 1e-5;
        // Tolerance on eigenvalues when removing dims from SDP solution space. 
        double eig_tol = 1e-9;
        // Maximum number of iterations (-1 for unlimited)
        int max_iter = -1;
        // Set verbosity
        bool verbose = true;
    };

    // ---- Symmetric matrix vectorization helper functions ----

    // Converts a symmetric matrix to a vectorized form (unique elements)
    Vector vec_symm(const Matrix &A);

    // Converts a vectorized form back to a symmetric matrix
    Matrix unvec_symm(const Vector &v, int dim);

    // ---- Implementation of rank reduction algorithm ----

    // Retrieves the linear operator of the constraints in the reduced optimal solution space.
    // This function is currently specialized to the max clique problem.
    Matrix get_constraint_op(const std::vector<Edge> &absent_edges, const Matrix &V);

    // Compute the minimum singular vector and value of a matrix
    std::pair<Vector, double> get_min_sing_vec(const Matrix &A);

    // Updates the optimal-space, linear constraint operator
    Matrix update_constraint_op(const Matrix &vAv, const Matrix &Q_tilde, int dim);

    /* Implements the rank reduction algorithm detailed in
    Lemon, Alex, Anthony Man-Cho So, and Yinyu Ye. "Low-rank semidefinite programming: Theory and applications." Foundations and Trends in Optimization 2.1-2 (2016): 1-156.
    */
    Matrix rank_reduction(
        const std::vector<Edge> &absent_edges,
        const Matrix &V_init,
        RankRedParams params=RankRedParams());
}
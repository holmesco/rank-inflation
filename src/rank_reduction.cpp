#include "max_clique_sdp/rank_reduction.hpp"

namespace RankReduction
{

    Vector vec_symm(const Matrix &A)
    {
        int n = A.rows();
        Vector v(n * (n + 1) / 2);
        int k = 0;
        for (int j = 0; j < n; ++j)
        {
            for (int i = 0; i <= j; ++i)
            {
                if (i == j)
                    v(k++) = A(i, j);
                else
                    v(k++) = std::sqrt(2.0) * A(i, j); // Standard scaling for isometry
            }
        }
        return v;
    }

    Matrix unvec_symm(const Vector &v, int dim)
    {
        Matrix A = Matrix::Zero(dim, dim);
        int k = 0;
        for (int j = 0; j < dim; ++j)
        {
            for (int i = 0; i <= j; ++i)
            {
                if (i == j)
                    A(i, j) = v(k++);
                else
                {
                    double val = v(k++) / std::sqrt(2.0);
                    A(i, j) = val;
                    A(j, i) = val;
                }
            }
        }
        return A;
    }

    Matrix get_constraint_op(const std::vector<Edge> &absent_edges, const Matrix &V)
    {
        // Get solution space null space operator. Specialized to the Lovasz-theta problem.
        int m = absent_edges.size() + 1;
        int r = V.cols();
        int vec_dim = r * (r + 1) / 2;
        Matrix vAv(m, vec_dim);
        // add edge constraints
        for (int i = 0; i < m - 1; ++i)
        {
            auto [u, v] = absent_edges[i];
            Matrix A_sol_space = 0.5 * (V.row(u).transpose() * V.row(v) + V.row(v).transpose() * V.row(u));
            vAv.row(i) = vec_symm(A_sol_space);
        }
        // add trace constraint
        vAv.row(m - 1) = vec_symm(V.transpose() * V);
        return vAv;
    }

    std::pair<Vector, double> get_min_sing_vec(const Matrix &A)
    {
        // Note: ComputeFullV should not be necessary since A should be a skinny matrix
        Eigen::JacobiSVD<Matrix> svd(A, Eigen::ComputeFullV);
        Vector S = svd.singularValues();
        Matrix V = svd.matrixV();
        // Note: singular values provided in decreasing order, so we pick the last
        double s_min = (S.size() < A.cols()) ? 0.0 : S(S.size() - 1);
        Vector vec = V.col(V.cols() - 1);
        return {vec, s_min};
    }

    Matrix update_constraint_op(const Matrix &vAv, const Matrix &Q_tilde, int dim)
    {
        int m = vAv.rows();
        int r_new = Q_tilde.cols();
        int vec_dim_new = r_new * (r_new + 1) / 2;
        Matrix vAv_updated(m, vec_dim_new);
        // update rows of the matrix (could be parallelized)
        for (int i = 0; i < m; ++i)
        {
            Matrix vAv_mat = unvec_symm(vAv.row(i).transpose(), dim);
            vAv_updated.row(i) = vec_symm(Q_tilde.transpose() * vAv_mat * Q_tilde);
        }
        return vAv_updated;
    }

    Matrix rank_reduction(
        const std::vector<Edge> &absent_edges,
        const Matrix &V_init, int targ_rank, double null_tol, double eig_tol, int max_iter,
        bool verbose)
    {
        // get initial matrix
        Matrix V = V_init;
        // get initial rank
        int r = V.cols();
        if (verbose)
            std::cout << "Initial rank: " << r << std::endl;

        Matrix A_v = get_constraint_op(absent_edges, V);
        int n_iter = 0;

        while ((max_iter == -1 || n_iter < max_iter) && (targ_rank == -1 || r > targ_rank))
        {
            auto [vec, s_min] = get_min_sing_vec(A_v);

            if (targ_rank == -1 && s_min > null_tol)
            {
                if (verbose)
                    std::cout << "Null space has no dimension. Exiting." << std::endl;
                break;
            }

            Matrix Delta = unvec_symm(vec, r);
            Eigen::SelfAdjointEigenSolver<Matrix> es(Delta);
            Vector lambdas = es.eigenvalues();
            Matrix Q = es.eigenvectors();

            // Find max magnitude eigenvalue
            int indmax = 0;
            lambdas.array().abs().maxCoeff(&indmax);
            double max_lambda = lambdas(indmax);

            double alpha = -1.0 / max_lambda;
            Vector lambdas_red = (Vector::Ones(lambdas.size()) + alpha * lambdas);

            std::vector<int> inds;
            for (int i = 0; i < lambdas_red.size(); ++i)
            {
                if (lambdas_red(i) > eig_tol)
                    inds.push_back(i);
            }

            Matrix Q_tilde(Q.rows(), inds.size());
            for (size_t i = 0; i < inds.size(); ++i)
            {
                Q_tilde.col(i) = Q.col(inds[i]) * std::sqrt(lambdas_red(inds[i]));
            }

            A_v = update_constraint_op(A_v, Q_tilde, r);
            V = V * Q_tilde;
            r = V.cols();
            n_iter++;

            if (verbose)
            {
                std::cout << "iter: " << n_iter << ", min s-value: " << s_min << ", rank: " << r << std::endl;
            }
        }

        return V;
    }

}
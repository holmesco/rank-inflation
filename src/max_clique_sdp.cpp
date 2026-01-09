#include "max_clique_sdp.hpp"

namespace bp = boost::process;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

namespace clipperplus
{

    MaxCliqueProblem::MaxCliqueProblem(const Graph &graph_in) : graph(graph_in)
    {
        size = graph.size();
        abs_edges = graph.get_absent_edges();
    }

    int MaxCliqueProblem::build_mc_hslr_problem(const std::string &filepath)
    {
        // get absent edges in the graph
        std::vector<Edge> edges = graph.get_absent_edges();
        // get sizes
        int n = graph.size();
        int m = edges.size();
        std::ofstream ofs(filepath);
        // check fail
        if (ofs.fail())
        {
            std::cerr << "Error opening file for writing." << std::endl;
            return -1;
        }
        // parameters: num constraints, problem size
        ofs << m << " " << n << std::endl;
        // constraint vector
        for (int i = 0; i < m; ++i)
        {
            ofs << "0.0 ";
        }
        ofs << std::endl;
        // Trace value
        ofs << "1.0" << std::endl
            << std::endl;

        // objective
        ofs << "0 LR" << std::endl;
        for (int i = 0; i < n; ++i)
        {
            ofs << "1.0 ";
        }
        ofs << "; -1.0" << std::endl;

        // constraints
        for (int i = 0; i < m; ++i)
        {
            Edge e = edges[i];
            ofs << std::endl;
            ofs << i + 1 << " SP" << std::endl;
            ofs << edges[i].first + 1 << " " << edges[i].second + 1 << " 0.5" << std::endl;
        }

        // Close file
        ofs.close();
        std::cout << "Successfully wrote model file." << std::endl;

        return 0;
    }

    MaxCliqueSolution MaxCliqueProblem::optimize_cuhallar()
    {
        // To call cuHALLaR, we need to
        // generate the problem file for this graph
        std::string filepath = "mc_hslr_problem.txt";
        build_mc_hslr_problem(filepath);
        // run cuhallar
        MaxCliqueSolution solution;
        try
        {
            // Launch the child process
            bp::child c(bp::search_path("cuHallar"), "-i", filepath);

            // Wait for the process to exit and get the exit code
            c.wait();
            int result = c.exit_code();
            std::cout << "Process finished with code: " << result << std::endl;

            // retrieve solution from output files
            solution = retrieve_cuhallar_solution();
        }
        catch (const bp::process_error &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
        }

        return solution;
    }

    MaxCliqueSolution MaxCliqueProblem::retrieve_cuhallar_solution()
    {
        // Read primal and dual outputs into Eigen types
        MaxCliqueSolution solution;
        std::ifstream primal_file("primal_out.txt");
        std::ifstream dual_file("dual_out.txt");

        if (!primal_file.is_open() || !dual_file.is_open())
        {
            std::cerr << "Error opening output files." << std::endl;
            return solution;
        }

        // Read primal matrix (CSV) into a temporary vector of rows
        std::string line;
        std::vector<std::vector<double>> temp_rows;
        while (std::getline(primal_file, line))
        {
            if (line.empty())
                continue;
            std::vector<double> row;
            std::stringstream ss(line);
            std::string value;
            while (std::getline(ss, value, ','))
            {
                row.push_back(std::stod(value));
            }
            if (!row.empty())
                temp_rows.push_back(std::move(row));
        }

        // Convert to Matrix
        if (!temp_rows.empty())
        {
            size_t r = temp_rows.size();
            size_t c = temp_rows[0].size();
            Matrix Y(r, c);
            for (size_t i = 0; i < r; ++i)
            {
                for (size_t j = 0; j < c; ++j)
                {
                    Y(static_cast<int>(i), static_cast<int>(j)) = temp_rows[i][j];
                }
            }
            solution.Y = std::move(Y);
        }
        else
        {
            solution.Y = Matrix(); // empty
        }

        // Read dual vector (CSV single line or comma-separated)
        std::vector<double> temp_dual;
        if (std::getline(dual_file, line))
        {
            std::stringstream ss(line);
            std::string value;
            while (std::getline(ss, value, ','))
            {
                temp_dual.push_back(std::stod(value));
            }
        }

        // Convert to Vector
        if (!temp_dual.empty())
        {
            Vector lag(static_cast<int>(temp_dual.size()));
            for (size_t i = 0; i < temp_dual.size(); ++i)
            {
                lag(static_cast<int>(i)) = temp_dual[i];
            }
            solution.lagrange = std::move(lag);
            solution.primal_opt = solution.lagrange.size() > 0 ? solution.lagrange(0) : 0.0;
        }
        else
        {
            solution.lagrange = Vector(); // empty
            solution.primal_opt = 0.0;
        }

        primal_file.close();
        dual_file.close();

        return solution;
    }

    // --- Helper Functions to mimic diffcp.cones ---
    // Converts a symmetric matrix to a vectorized form (unique elements)
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

    // Converts a vectorized form back to a symmetric matrix
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

    // --- Implementation Logic ---

    Matrix get_constraint_op(const std::vector<Edge> &absent_edges, const Matrix &V)
    {
        // Get solution space null space operator. Specialized to the Lovasz-theta problem.
        int m = absent_edges.size() + 1;
        int r = V.cols();
        int vec_dim = r * (r + 1) / 2;
        Matrix Av(m, vec_dim);
        // add edge constraints
        for (int i = 0; i < m; ++i)
        {
            auto [u, v] = absent_edges[i];
            Matrix A_sol_space = 0.5 * (V.row(u).transpose() * V.row(v) + V.row(u).transpose() * V.row(v));
            Av.row(i) = vec_symm(A_sol_space);
        }
        // add trace constraint
        Av.row(m) = vec_symm(V.transpose() * V);
        return Av;
    }

    std::pair<Vector, double> get_min_sing_vec(const Matrix &A)
    {
        Eigen::JacobiSVD<Matrix> svd(A, Eigen::ComputeFullV);
        Vector S = svd.singularValues();
        Matrix V = svd.matrixV();

        double s_min = (S.size() < A.cols()) ? 0.0 : S(S.size() - 1);
        Vector vec = V.col(V.cols() - 1);
        return {vec, s_min};
    }

    Matrix update_constraint_op(const Matrix &Av, const Matrix &Q_tilde, int dim)
    {
        int m = Av.rows();
        int r_new = Q_tilde.cols();
        int vec_dim_new = r_new * (r_new + 1) / 2;
        Matrix Av_updated(m, vec_dim_new);

        for (int i = 0; i < m; ++i)
        {
            Matrix A = unvec_symm(Av.row(i).transpose(), dim);
            Matrix updatedA = Q_tilde.transpose() * A * Q_tilde;
            Av_updated.row(i) = vec_symm(updatedA);
        }
        return Av_updated;
    }

    // --- Main Algorithm ---

    Matrix rank_reduction(
        const std::vector<Edge> &absent_edges,
        const Matrix &V_init, double rank_tol, double null_tol, double eig_tol, int targ_rank, int max_iter,
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
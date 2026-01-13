#include "max_clique_sdp/max_clique_sdp.hpp"

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

    int MaxCliqueProblem::build_mc_hslr_problem(const std::string &filepath) const
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
            std::string error_msg = "File not found or could not be opened: ";
            error_msg += filepath;
            throw std::runtime_error(error_msg);
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

    int MaxCliqueProblem::build_initialization_file(const std::string &filepath, const std::vector<Node> &init_clique) const
    {
        std::ofstream ofs(filepath);
        // check fail
        if (ofs.fail())
        {
            std::string error_msg = "File not found or could not be opened: ";
            error_msg += filepath;
            throw std::runtime_error(error_msg);
        }

        // Build initialization vector
        Vector init_soln = clique_to_soln(init_clique, size);

        // Write to file as CSV
        for (int i = 0; i < init_soln.size(); ++i)
        {
            ofs << init_soln(i);
            if (i < init_soln.size() - 1)
                ofs << std::endl;
        }
        ofs << std::endl;

        // Close file
        ofs.close();
        std::cout << "Successfully wrote initialization file." << std::endl;

        return 0;
    }

    MaxCliqueSolution MaxCliqueProblem::optimize_cuhallar(const std::vector<int> &init_clique) const
    {
        // Set up input and output files
        std::string input_file = "/workspace/tmp/mc_hslr_problem.txt";
        std::string init_file = "/workspace/tmp/mc_initialization.txt";
        std::string primal_out = "/workspace/tmp/primal_out.txt";
        std::string dual_out = "/workspace/tmp/dual_out.txt";
        std::string options = "/workspace/parameters/cuhallar_params.cfg";
        // To call cuHALLaR, we need to generate the problem file for this graph
        build_mc_hslr_problem(input_file);
        // Set up initial clique file if provided
        build_initialization_file(init_file, init_clique);
        // Set up output file paths
        // Launch the child process
        bp::child c(bp::search_path("cuHallar"), "-i", input_file, "-p", primal_out, "-d", dual_out, "-w", init_file, "-c", options);

        // Wait for the process to exit and get the exit code
        c.wait();
        int result = c.exit_code();
        std::cout << "Process finished with code: " << result << std::endl;

        // retrieve solution from output files
        MaxCliqueSolution solution = retrieve_cuhallar_solution();

        return solution;
    }

    MaxCliqueSolution MaxCliqueProblem::retrieve_cuhallar_solution() const
    {
        // Read primal and dual outputs into Eigen types
        MaxCliqueSolution solution;
        std::ifstream primal_file("/workspace/tmp/primal_out.txt");
        std::ifstream dual_file("/workspace/tmp/dual_out.txt");

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

    std::vector<int> MaxCliqueProblem::soln_to_clique(const Vector &soln) const
    {
        // Convert to abs array
        auto sol_array = soln.array().abs();
        // Identify the clique indices
        double mid_val = (sol_array.maxCoeff() - sol_array.minCoeff()) / 2;
        std::vector<int> clique;
        for (int i = 0; i < sol_array.size(); i++)
        {
            if (sol_array(i) > mid_val)
            {
                clique.push_back(i);
            }
        }
        return clique;
    }

    Vector MaxCliqueProblem::clique_to_soln(const std::vector<int> clique, int size) const
    {
        assert(clique.size() > 0);
        double factor = 1 / std::sqrt(clique.size());
        Eigen::VectorXd soln = Eigen::VectorXd::Zero(size);
        for (int i; i < clique.size(); i++)
        {
            soln(i) = factor;
        }
        return soln;
    }

}
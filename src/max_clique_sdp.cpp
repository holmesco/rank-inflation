#include "max_clique_sdp.hpp"
#include <iostream>
#include <fstream>
#include <boost/process.hpp>


namespace bp = boost::process;

namespace clipperplus
{

int build_mc_hslr_problem(const Graph &graph, const std::string &filepath) {
    // get absent edges in the graph
    std::vector<Edge> edges = graph.get_absent_edges();
    // get sizes
    int n = graph.size();
    int m = edges.size();
    std::ofstream ofs(filepath);
    // check fail
    if (ofs.fail()) {
        std::cerr << "Error opening file for writing." << std::endl;
        return -1;
    }
    // parameters
    // ofs << "# Num Constraints, SDP Size" << std::endl;
    ofs << m << " " << n << std::endl;
    // constraint vector
    // ofs << "# Constraint Vector" << std::endl;
    for (int i = 0; i < m; ++i) {
        ofs << "0.0 ";
    }
    ofs << std::endl;
    // Trace value
    // ofs << "# Trace Value:" << std::endl
    ofs << "1.0" << std::endl << std::endl;

    // objective
    // ofs << "# Objective Matrix" << std::endl;
    ofs << "0 LR" << std::endl;
    for (int i = 0; i < n; ++i) {
        ofs << "1.0 ";
    }
    ofs << "; -1.0" << std::endl;

    // constraints
    for (int i = 0; i < m; ++i) {
        Edge e = edges[i];
        ofs << std::endl; 
        // ofs << "# Constraint Matrix " << i+1 << std::endl;
        ofs << i+1 << " SP" << std::endl;
        ofs << edges[i].first+1 << " " << edges[i].second+1 <<  " 0.5" << std::endl;
    }
    
    // Close file
    ofs.close();
    std::cout << "Successfully wrote model file." << std::endl;

    return 0;
}

int optimize_cuhallar(const Graph &graph){
    // To call cuHALLaR, we need to 
    // generate the problem file for this graph
    std::string filepath = "mc_hslr_problem.txt";
    build_mc_hslr_problem(graph, filepath);
    // define input stream
    bp::ipstream out_stream; 
    // run cuhallar
    try {
        // 2. Launch the child process
        // "ls -lh" is the command; std_out > out_stream redirects output
        bp::child c(bp::search_path("cuHallar"), "-i", filepath, bp::std_out > out_stream);

        std::string line;
        // 3. Read the output line by line
        while (c.running() && std::getline(out_stream, line) && !line.empty()) {
            std::cout << "Captured: " << line << std::endl;
        }

        // 4. Wait for the process to exit and get the exit code
        c.wait();
        int result = c.exit_code();
        
        std::cout << "Process finished with code: " << result << std::endl;
        
    } catch (const bp::process_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}

}
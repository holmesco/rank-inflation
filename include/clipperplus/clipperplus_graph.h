#pragma once

#include <Eigen/Dense>
#include <memory.h>
#include <vector>


namespace clipperplus
{

using Node = int;
using Edge = std::pair<Node, Node>;
using Neighborlist = std::vector<Node>;


class Graph
{
public:
    // Create a graph
    Graph() = default;

    // Build a Graph object from an adjacency matrix
    Graph(Eigen::MatrixXd adj_matrix);

    // static Graph from_list(const std::vector<Neighborlist> &adj_list);

    // number of nodes in the graph
    int size() const;

    // Return the degree of a node
    int degree(Node v) const;

    // Return the degrees of the nodes in the graph
    std::vector<int> degrees() const;

    // Return a list of nodes representing the neighbors of v
    const std::vector<Node> &neighbors(Node v) const;

    inline bool is_edge(Node u, Node v) const
    {
        return adj_matrix(u, v) != 0;
    }

    // Merge with graph g by combining edges
    void merge(const Graph &g);
    Graph induced(const std::vector<Node> &nodes) const;

    int max_core_number() const;
    const std::vector<int> &get_core_numbers() const;
    const std::vector<Node> &get_core_ordering() const;
    
    const Eigen::MatrixXd &get_adj_matrix() const;

    // Get the list of absent edges in the graph. Used for max clique problem formulation
    std::vector<Edge> get_absent_edges() const;

    // Check if a set of nodes form a clique
    bool is_clique(std::vector<Node> clique) const;

private:
    // Calculate the k-cores of the nodes in the graphs. Updates the 'kcore' and 'kcore_ordering'
    void calculate_kcores() const;

private:
    Eigen::MatrixXd adj_matrix;
    std::vector<Neighborlist> adj_list;

    mutable std::vector<Node> kcore_ordering;
    mutable std::vector<int> kcore;
};


}

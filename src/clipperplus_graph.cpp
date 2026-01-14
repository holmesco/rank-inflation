
#include "clipperplus/clipperplus_graph.h"

namespace clipperplus
{

    Graph::Graph(Eigen::MatrixXd adj) : adj_matrix(std::move(adj)), adj_list(adj_matrix.rows())
    {
        int nnodes = adj_matrix.rows();
        for (int i = 0; i < nnodes; ++i)
        {
            for (int j = 0; j < nnodes; ++j)
            {
                if (adj_matrix(i, j) != 0)
                {
                    adj_list[i].push_back(j);
                }
            }
        }
    }

    std::vector<Edge> Graph::get_absent_edges() const
    {
        auto absent_edges = std::vector<Edge>();
        for (int i = 0; i < size(); i++)
        {
            for (int j = i + 1; j < size(); j++)
            {
                if (!is_edge(i, j))
                {
                    absent_edges.push_back({i, j});
                }
            }
        }
        return absent_edges;
    }

    bool Graph::is_clique(std::vector<Node> clique) const
    {
        for (int i = 0; i < clique.size(); i++)
        {
            for (int j = i + 1; j < clique.size(); j++)
            {
                if (!is_edge(clique[i], clique[j]))
                {
                    return false;
                }
            }
        }
        return true;
    }

    const std::vector<Node> &Graph::neighbors(Node v) const
    {
        assert(v < adj_list.size());
        return adj_list[v];
    }

    int Graph::degree(Node v) const
    {
        assert(v < adj_list.size());
        return adj_list[v].size();
    }

    std::vector<int> Graph::degrees() const
    {
        int nnodes = adj_list.size();
        std::vector<int> degrees(adj_list.size());
        for (int i = 0; i < nnodes; ++i)
        {
            degrees[i] = degree(i);
        }
        return degrees;
    }

    void Graph::merge(const Graph &g)
    {
        assert(adj_list.size() == g.adj_list.size());
        for (int i = 0; i < adj_list.size(); ++i)
        {
            adj_list[i].insert(
                adj_list[i].end(),
                g.adj_list[i].begin(), g.adj_list[i].end());

            adj_matrix(i, g.adj_list[i]).setOnes();
        }

        kcore.clear();
        kcore_ordering.clear();
    }

    Graph Graph::induced(const std::vector<Node> &nodes) const
    {
        int n = size();

        auto g = Graph();
        g.adj_matrix = adj_matrix(nodes, nodes);

        std::vector<int> keep(size(), -1);
        for (Node i = 0; i < nodes.size(); ++i)
        {
            keep[nodes[i]] = i;
        }

        g.adj_list = std::vector<Neighborlist>(nodes.size());
        for (Node v = 0; v < n; ++v)
        {
            if (keep[v] < 0)
            {
                continue;
            }

            for (auto u : neighbors(v))
            {
                if (keep[u] >= 0)
                {
                    g.adj_list[keep[v]].push_back((Node)keep[u]);
                }
            }
        }

        return g;
    }

    int Graph::size() const
    {
        return adj_list.size();
    }

    int Graph::max_core_number() const
    {
        if (kcore.empty())
        {
            calculate_kcores();
        }
        return kcore[kcore_ordering.back()];
    }

    const std::vector<int> &Graph::get_core_numbers() const
    {
        if (kcore.empty())
        {
            calculate_kcores();
        }
        return kcore;
    }

    const std::vector<Node> &Graph::get_core_ordering() const
    {
        if (kcore.empty())
        {
            calculate_kcores();
        }
        return kcore_ordering;
    }

    std::pair<std::vector<Node>, std::vector<Node>>
    Graph::get_pruned_vertices(int min_kcore) const
    {
        int n = size();
        std::vector<int> core_number = get_core_numbers();
        std::vector<int> keep, keep_pos(n, -1);
        for (Node i = 0, j = 0; i < n; ++i)
        {
            if (core_number[i] >= min_kcore)
            {
                keep.push_back(i);
                keep_pos[i] = j++;
            }
        }
        return {keep, keep_pos};
    }

    const Eigen::MatrixXd &Graph::get_adj_matrix() const
    {
        return adj_matrix;
    }

    void Graph::calculate_kcores() const
    {
        int n = size();
        auto degree = degrees();
        auto max_degree = *std::max_element(degree.begin(), degree.end());

        // prepare for bucket sort by degree
        // pos[i] is the position of v_i on the sorted array
        // If you consider `pos` as an permutation `order` essentially is pos^-1
        std::vector<int> pos(n, 0);
        kcore_ordering.resize(n);

        // bin[d] is the starting index in the sorted array for nodes with degree d
        std::vector<int> bin(max_degree + 1, 0);
        for (auto d : degree)
        {
            ++bin[d];
        }

        int start = 0;
        for (int d = 0; d < max_degree + 1; ++d)
        {
            auto num = bin[d];
            bin[d] = start;
            start += num;
        }

        // bucket sort:
        for (int v = 0; v < n; ++v)
        {
            pos[v] = bin[degree[v]];
            kcore_ordering[pos[v]] = v;
            ++bin[degree[v]];
        }

        for (int d = max_degree; d > 0; --d)
        {
            bin[d] = bin[d - 1];
        }
        bin[0] = 0;

        // iteratively remove edges from v with lowest degree (effectively removing v)
        for (auto v : kcore_ordering)
        {
            for (auto u : neighbors(v))
            { // remove e = (v, u)
                if (degree[v] >= degree[u])
                {
                    continue;
                }

                // update sorted array: pos, order bin
                // find first element in sorted array with d[w] = d[u]
                auto pos_w = bin[degree[u]];
                auto w = kcore_ordering[pos_w];

                // swap their pose and order
                if (w != u)
                {
                    kcore_ordering[pos[u]] = w;
                    kcore_ordering[pos[w]] = u;
                    std::swap(pos[u], pos[w]);
                }
                // shift bucket start index to the right (drop out u)
                ++bin[degree[u]];
                // decrement the degree (kcore) of u
                --degree[u];
            }
        }

        kcore = std::move(degree);
    }

}

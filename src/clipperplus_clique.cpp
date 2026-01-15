/*
computes a maximal clique in graph, and certifies if it's maximum clique

Author: kaveh fathian (kavehfathian@gmail.com)
 */

#include "clipperplus/clipperplus_heuristic.h"
#include "clipperplus/clipperplus_clique.h"
#include "max_clique_sdp/max_clique_sdp.hpp"
#include "max_clique_sdp/rank_reduction.hpp"

namespace clipperplus
{

    std::pair<std::vector<Node>, CERTIFICATE> find_clique(const Graph &graph, ClipperParams params)
    {
        int n = graph.size();
        // generate the chromatic number upper bound
        auto chromatic_welsh = estimate_chromatic_number_welsh_powell(graph);
        // generate the k-core upper bound
        auto k_core_bound = graph.max_core_number() + 1;
        // Get the max clique from the heuristic algorithm
        auto heuristic_clique = find_heuristic_clique(graph);
        if (heuristic_clique.size() == std::min({k_core_bound, chromatic_welsh}))
        {
            return {heuristic_clique, CERTIFICATE::HEURISTIC};
        }
        // prune the graph based on core numbers
        auto [keep, keep_pos] = graph.get_pruned_vertices(heuristic_clique.size());

        // get the pruned adjacency matrix and augment with identity
        Eigen::MatrixXd M_pruned = graph.get_adj_matrix()(keep, keep);
        M_pruned.diagonal().setOnes();

        // initialize using a vector orthogonal to the heuristic clique
        Eigen::VectorXd u0 = Eigen::VectorXd::Ones(keep.size());
        for (auto v : heuristic_clique)
        {
            assert(keep_pos[v] >= 0);
            u0(keep_pos[v]) = 0;
        }
        u0.normalize();
        // Run optimization
        auto clique_optim_pruned = clipperplus::clique_optimization(M_pruned, u0, params.optim_params);
        std::vector<Node> optimal_clique;
        if (clique_optim_pruned.size() < heuristic_clique.size())
        {
            // if heuristic clique is larger, return it
            optimal_clique = heuristic_clique;
        }
        else
        {
            // map back to original graph nodes
            for (auto v : clique_optim_pruned)
            {
                assert(v >= 0 && v < keep.size());
                optimal_clique.push_back(keep[v]);
            }
        }

        auto certificate = CERTIFICATE::NONE;
        if (optimal_clique.size() == k_core_bound)
        {
            certificate = CERTIFICATE::CORE_BOUND;
        }
        else if (optimal_clique.size() == chromatic_welsh)
        {
            certificate = CERTIFICATE::CHROMATIC_BOUND;
        }

        // Lovasz-Theta SDP Optimization
        if (certificate == CERTIFICATE::NONE && params.check_lovasz_theta)
        {
            // Reprune based on current largest clique
            auto [keep_lt, keep_pos_lt] = graph.get_pruned_vertices(optimal_clique.size());
            // Generate reduced graph
            auto graph_sdp = Graph(graph.get_adj_matrix()(keep_lt, keep_lt));
            // run optimization
            auto max_clique_prob = LovaszThetaProblem(graph_sdp, params.cuhallar_params);
            auto soln = max_clique_prob.optimize_cuhallar(optimal_clique);
            // Check if LT bound is satisfied
            // NOTE: Hardcoded 1 because the LT bound is continuous, but still upper bounding
            // we obtain a valid certificate as long as this inequality holds.
            if (soln.primal_opt < optimal_clique.size() + 1.0)
            {
                certificate = CERTIFICATE::LOVASZ_THETA_BOUND;
            }
            else
            { // SDP potentially found a larger clique.
                // Apply rank reduction, if necessary
                Eigen::VectorXd V;
                if (soln.Y.cols() > 1)
                {
                    auto non_edges = graph_sdp.get_absent_edges();
                    V = RankReduction::rank_reduction(non_edges, soln.Y, params.rank_red_params);
                }
                else
                {
                    V = soln.Y;
                }
                // Get clique from solution
                auto clique_sdp = max_clique_prob.soln_to_clique(V);
                // Check if we found a better solution.
                if (clique_sdp.size() > optimal_clique.size() && graph_sdp.is_clique(clique_sdp))
                {
                    // map back to original graph nodes
                    optimal_clique.clear();
                    for (auto v : clique_sdp)
                    {
                        optimal_clique.push_back(keep[v]);
                    }
                    // re-check for certificate
                    if (soln.primal_opt < optimal_clique.size() + 1.0)
                    {
                        certificate = CERTIFICATE::LOVASZ_THETA_SOLN_BOUND;
                    }
                }
            }
        }

        return {optimal_clique, certificate};
    }

}
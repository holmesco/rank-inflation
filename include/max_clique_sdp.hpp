#pragma once

#include "clipperplus/clipperplus_graph.h"
#include <string>

namespace clipperplus
{
    
int build_mc_hslr_problem(const Graph& graph, const std::string &filepath) ;
    
int build_mc_hslr_problem(const Graph& graph) {
        return build_mc_hslr_problem(graph, "hslr_model.txt");
    }

int optimize_cuhallar(const Graph &graph);

}
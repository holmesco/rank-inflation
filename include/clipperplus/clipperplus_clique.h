#pragma once

#include <iostream>
#include <chrono>
#include <Eigen/Dense>

#include "clipperplus/clique_optimization.h"
#include "clipperplus/clipperplus_heuristic.h"
#include "clipperplus/utils.h"

namespace clipperplus
{

    
    enum class CERTIFICATE
    {
        NONE,
        HEURISTIC,
        CORE_BOUND,
        CHROMATIC_BOUND,
        LOVASZ_THETA_BOUND,
        LOVASZ_THETA_SOLN,
    };
    
    inline std::string to_string(CERTIFICATE cert)
    {
        switch (cert)
        {
        case CERTIFICATE::NONE:
            return "NONE";
        case CERTIFICATE::HEURISTIC:
            return "HEURISTIC";
        case CERTIFICATE::CORE_BOUND:
            return "CORE_BOUND";
        case CERTIFICATE::CHROMATIC_BOUND:
            return "CHROMATIC_BOUND";
        case CERTIFICATE::LOVASZ_THETA_BOUND:
            return "LOVASZ_THETA_BOUND";
        default:
            return "UNKNOWN";
        }
    }
    
    std::pair<std::vector<Node>, CERTIFICATE> find_clique(const Graph &graph, bool check_sdp=false);

}

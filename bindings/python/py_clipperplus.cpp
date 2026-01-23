/**
 * @file py_clipper.cpp
 * @brief Python bindings for CLIPPER
 * @author Parker Lusk <plusk@mit.edu>
 * @date 28 January 2021
 */

#include <cstdint>
#include <sstream>

#include <Eigen/Dense>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "wrapper.h"

namespace py = pybind11;
using namespace pybind11::literals;

// ----------------------------------------------------------------------------

PYBIND11_MODULE(clipperpluspy, m) {
  m.doc() = "CLIPPER+ is an algorithm for finding maximal cliques in "
            "unweighted graphs for outlier-robust global registration.";
  m.attr("__version__") = CLIPPERPLUS_VERSION;

  /* ---- Parameter Structures ----- */

  // ClipperOptParams
  py::class_<clipperplus::ClipperOptParams>(m, "ClipperOptParams")
      .def(py::init<>())
      .def_readwrite("tol_u", &clipperplus::ClipperOptParams::tol_u)
      .def_readwrite("tol_F", &clipperplus::ClipperOptParams::tol_F)
      .def_readwrite("maxiniters", &clipperplus::ClipperOptParams::maxiniters)
      .def_readwrite("maxoliters", &clipperplus::ClipperOptParams::maxoliters)
      .def_readwrite("beta", &clipperplus::ClipperOptParams::beta)
      .def_readwrite("sigma", &clipperplus::ClipperOptParams::sigma)
      .def_readwrite("maxlsiters", &clipperplus::ClipperOptParams::maxlsiters)
      .def_readwrite("minalpha", &clipperplus::ClipperOptParams::minalpha)
      .def_readwrite("maxalpha", &clipperplus::ClipperOptParams::maxalpha)
      .def_readwrite("eps", &clipperplus::ClipperOptParams::eps);
  // RankRedParams
  py::class_<RankReduction::RankRedParams>(m, "RankRedParams")
      .def(py::init<>())
      .def_readwrite("targ_rank", &RankReduction::RankRedParams::targ_rank)
      .def_readwrite("null_tol", &RankReduction::RankRedParams::null_tol)
      .def_readwrite("eig_tol", &RankReduction::RankRedParams::eig_tol)
      .def_readwrite("max_iter", &RankReduction::RankRedParams::max_iter)
      .def_readwrite("verbose", &RankReduction::RankRedParams::verbose);
  // CuhallarParams
  py::class_<clipperplus::CuhallarParams>(m, "CuhallarParams")
      .def(py::init<>())
      .def_readwrite("input_file", &clipperplus::CuhallarParams::input_file)
      .def_readwrite("init_file", &clipperplus::CuhallarParams::init_file)
      .def_readwrite("primal_out", &clipperplus::CuhallarParams::primal_out)
      .def_readwrite("dual_out", &clipperplus::CuhallarParams::dual_out)
      .def_readwrite("options_sparse",
                     &clipperplus::CuhallarParams::options_sparse)
      .def_readwrite("enable_sparse",
                     &clipperplus::CuhallarParams::enable_sparse)
      .def_readwrite("options", &clipperplus::CuhallarParams::options);
  // ClipperParams
  py::class_<clipperplus::ClipperParams>(m, "ClipperParams")
      .def(py::init<>())
      .def_readwrite("optim_params", &clipperplus::ClipperParams::optim_params)
      .def_readwrite("check_lovasz_theta",
                     &clipperplus::ClipperParams::check_lovasz_theta)
      .def_readwrite("cuhallar_params",
                     &clipperplus::ClipperParams::cuhallar_params)
      .def_readwrite("rank_red_params",
                     &clipperplus::ClipperParams::rank_red_params);

  // SolutionInfo
  py::class_<clipperplus::SolutionInfo,
             std::shared_ptr<clipperplus::SolutionInfo>>(m, "SolutionInfo")
      .def(py::init<>())
      .def_readwrite("min_kcore", &clipperplus::SolutionInfo::min_kcore)
      .def_readwrite("primal_opt", &clipperplus::SolutionInfo::primal_opt)
      .def_readwrite("dual_opt", &clipperplus::SolutionInfo::dual_opt)
      .def_readwrite("lt_opt_time", &clipperplus::SolutionInfo::lt_opt_time)
      .def_readwrite("lt_problem_size",
                     &clipperplus::SolutionInfo::lt_problem_size)
      .def_readwrite("lt_num_constraints",
                     &clipperplus::SolutionInfo::lt_num_constraints)
      .def_readwrite("local_opt_size",
                     &clipperplus::SolutionInfo::local_opt_size);

  m.def("find_clique", &Wrapper::find_clique_wrapper, "adj"_a,
        "params"_a = clipperplus::ClipperParams(), "info"_a = nullptr,
        "Find the densest subgraph of a weighted adjacency matrix.");
  m.def("find_heuristic_clique", &Wrapper::find_heuristic_clique_wrapper,
        "adj"_a, "clique"_a, "Find a heuristic maximum clique in a graph.");
  m.def("clique_optimization", &Wrapper::clique_optimization_wrapper, "M"_a,
        "u0"_a, "params"_a = clipperplus::ClipperOptParams(),
        "Run original clipper on pruned graph");
}

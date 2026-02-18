// --------------------------------------------------------------------------
// pybind11 bindings for SDPTools::AnalyticCenter and supporting types
// --------------------------------------------------------------------------
//
// The C++ AnalyticCenter class stores *references* to the constraint matrices
// (A) and RHS vector (b).  To make that safe from Python we introduce a small
// prevent wrapper (PyAnalyticCenter) that *owns* copies of A and b and
// forwards every public method to the real object.
// --------------------------------------------------------------------------

#include <pybind11/eigen.h>  // automatic Eigen <-> numpy conversion
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // std::vector / std::pair conversion

#include "analytic_center.hpp"

namespace py = pybind11;
using namespace SDPTools;

// ---------------------------------------------------------------------------
// Prevent: prevent the reference-to-temporary problem.
// ---------------------------------------------------------------------------
class PyAnalyticCenter {
 public:
  PyAnalyticCenter(const Matrix& C, double rho,
                   std::vector<Eigen::SparseMatrix<double>> A,
                   std::vector<double> b, AnalyticCenterParams params)
      : A_(std::move(A)),
        b_(std::move(b)),
        ac_(C, rho, A_, b_, std::move(params)) {}

  // ---- forwarded public API ----

  int dim() const { return ac_.dim; }
  int m() const { return ac_.m; }

  Vector eval_constraints(const Matrix& X) const {
    return ac_.eval_constraints(X);
  }

  AnalyticCenterResult certify(const Matrix& Y_0, double delta) const {
    return ac_.certify(Y_0, delta);
  }

  std::pair<Matrix, Vector> get_analytic_center(
      const Matrix& Y_0, double delta_obj = 0.0,
      double delta_constraint = 0.0) const {
    return ac_.get_analytic_center(Y_0, delta_obj, delta_constraint);
  }

  Matrix build_certificate_from_dual(const Vector& multipliers) const {
    return ac_.build_certificate_from_dual(multipliers);
  }

  std::pair<double, double> check_certificate(const Matrix& H,
                                              const Matrix& Y) const {
    return ac_.check_certificate(H, Y);
  }

  // Expose the stored params so users can inspect them.
  const AnalyticCenterParams& params() const { return ac_.params_; }

 private:
  // Owned copies — these outlive the AnalyticCenter reference members.
  std::vector<Eigen::SparseMatrix<double>> A_;
  std::vector<double> b_;
  AnalyticCenter ac_;
};

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(sdptools, m) {
  m.doc() = "Python bindings for the SDPTools AnalyticCenter solver";

  // ---- AnalyticCenterParams ----
  py::class_<AnalyticCenterParams>(m, "AnalyticCenterParams")
      .def(py::init<>())
      .def_readwrite("verbose", &AnalyticCenterParams::verbose)
      .def_readwrite("max_iter", &AnalyticCenterParams::max_iter)
      .def_readwrite("tol_rank_sol", &AnalyticCenterParams::tol_rank_sol)
      // Analytic Center
      .def_readwrite("tol_step_norm_ac",
                     &AnalyticCenterParams::tol_step_norm_ac)
      .def_readwrite("reduce_violation_ac",
                     &AnalyticCenterParams::reduce_violation_ac)
      .def_readwrite("max_iter_ac", &AnalyticCenterParams::max_iter_ac)
      .def_readwrite("max_iter_adaptive_ac",
                     &AnalyticCenterParams::max_iter_adaptive_ac)
      .def_readwrite("delta_init_ac", &AnalyticCenterParams::delta_init_ac)
      .def_readwrite("delta_min_ac", &AnalyticCenterParams::delta_min_ac)
      .def_readwrite("adapt_factor_ac", &AnalyticCenterParams::adapt_factor_ac)
      .def_readwrite("check_cert_ac", &AnalyticCenterParams::check_cert_ac)
      // Line search
      .def_readwrite("enable_line_search_ac",
                     &AnalyticCenterParams::enable_line_search_ac)
      .def_readwrite("ln_search_suff_dec",
                     &AnalyticCenterParams::ln_search_suff_dec)
      .def_readwrite("ln_search_red_factor",
                     &AnalyticCenterParams::ln_search_red_factor)
      .def_readwrite("alpha_init", &AnalyticCenterParams::alpha_init)
      .def_readwrite("alpha_min", &AnalyticCenterParams::alpha_min)
      .def_readwrite("tol_bisect_ac", &AnalyticCenterParams::tol_bisect_ac)
      // Certificate
      .def_readwrite("tol_cert_psd", &AnalyticCenterParams::tol_cert_psd)
      .def_readwrite("tol_cert_first_order",
                     &AnalyticCenterParams::tol_cert_first_order)
      .def("__repr__", [](const AnalyticCenterParams& p) {
        return "<AnalyticCenterParams max_iter_ac=" +
               std::to_string(p.max_iter_ac) +
               " verbose=" + (p.verbose ? "True" : "False") + ">";
      });

  // ---- AnalyticCenterResult ----
  py::class_<AnalyticCenterResult>(m, "AnalyticCenterResult")
      .def_readonly("X", &AnalyticCenterResult::X)
      .def_readonly("H", &AnalyticCenterResult::H)
      .def_readonly("multipliers", &AnalyticCenterResult::multipliers)
      .def_readonly("violation", &AnalyticCenterResult::violation)
      .def_readonly("certified", &AnalyticCenterResult::certified)
      .def_readonly("min_eig", &AnalyticCenterResult::min_eig)
      .def_readonly("complementarity", &AnalyticCenterResult::complementarity)
      .def("__repr__", [](const AnalyticCenterResult& r) {
        return "<AnalyticCenterResult certified=" +
               std::string(r.certified ? "True" : "False") +
               " min_eig=" + std::to_string(r.min_eig) + ">";
      });

  // ---- AnalyticCenter (via PyAnalyticCenter wrapper) ----
  py::class_<PyAnalyticCenter>(m, "AnalyticCenter")
      .def(py::init<const Matrix&, double,
                    std::vector<Eigen::SparseMatrix<double>>,
                    std::vector<double>, AnalyticCenterParams>(),
           py::arg("C"), py::arg("rho"), py::arg("A"), py::arg("b"),
           py::arg("params") = AnalyticCenterParams(),
           R"pbdoc(
Construct an AnalyticCenter problem.

Parameters
----------
C : numpy.ndarray (n, n)
    Cost matrix.
rho : float
    Optimal cost value (scalar offset).
A : list of scipy.sparse.csc_matrix or numpy.ndarray
    Constraint matrices (each n×n, upper-triangular storage).
b : list of float
    Right-hand side values for trace(A_i X) = b_i.
params : AnalyticCenterParams, optional
    Algorithm parameters.
)pbdoc")
      .def_property_readonly("dim", &PyAnalyticCenter::dim)
      .def_property_readonly("m", &PyAnalyticCenter::m)
      .def_property_readonly("params", &PyAnalyticCenter::params)
      .def("eval_constraints", &PyAnalyticCenter::eval_constraints,
           py::arg("X"), "Evaluate constraint violations at X.")
      .def("certify", &PyAnalyticCenter::certify, py::arg("Y_0"),
           py::arg("delta"),
           R"pbdoc(
Run analytic centering to certify the local solution Y_0.

Parameters
----------
Y_0 : numpy.ndarray (n, r)
    Initial low-rank factor.
delta : float
    Perturbation parameter.

Returns
-------
AnalyticCenterResult
)pbdoc")
      .def("get_analytic_center", &PyAnalyticCenter::get_analytic_center,
           py::arg("Y_0"), py::arg("delta_obj") = 0.0,
           py::arg("delta_constraint") = 0.0,
           R"pbdoc(
Compute the analytic center starting from Y_0.

Parameters
----------
Y_0 : numpy.ndarray (n, r)
    Initial point (low-rank factor; X_0 = Y_0 @ Y_0.T).
delta_obj : float, optional
    Perturbation added to objective (default 0).
delta_constraint : float, optional
    Perturbation added to constraints (default 0).

Returns
-------
tuple(X, multipliers)
    X : numpy.ndarray (n, n)  — centered primal solution.
    multipliers : numpy.ndarray (m,) — optimal dual multipliers.
)pbdoc")
      .def("build_certificate_from_dual",
           &PyAnalyticCenter::build_certificate_from_dual,
           py::arg("multipliers"),
           "Build the certificate matrix H from dual multipliers.")
      .def("check_certificate", &PyAnalyticCenter::check_certificate,
           py::arg("H"), py::arg("Y"),
           R"pbdoc(
Check global optimality of a solution.

Returns
-------
tuple(min_eig, first_order_cond)
)pbdoc");
}

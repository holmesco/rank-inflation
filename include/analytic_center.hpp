#pragma once
#include <Eigen/IterativeLinearSolvers>
#include <chrono>
#include <filesystem>

#include "lin_alg_tools.hpp"
#include "utils.hpp"

namespace RankTools {

enum class SolverStatus {
  Success,
  MaxIterReached,
  StepTolerance,
  SolverFailure,
  PreSolveFailure,
  Unknown,
};

struct AnalyticCenterResult {
  // Analytic center solution
  Matrix X;
  // Certificate matrix at the solution
  Matrix H;
  // Optimal multipliers for the constraints at the solution
  Vector multipliers;
  // Constraint violation at the solution
  Vector violation;
  // Whether the solution is certified by the certificate matrix
  bool certified;
  // Minimum eigenvalue of the certificate matrix
  double min_eig;
  // Complementarity condition (first order condition)
  double complementarity;
  // Solver time
  double solver_time;
};

struct AnalyticCenterParams {
  // Verbosity
  bool verbose = true;
  // threshold for checking rank of the solution
  // (does not affect convergence, just for display)
  double tol_rank_sol = 1.0E-4;
  // tolerance for step size (terminate if below)
  double tol_step_norm = 1e-8;
  // max number of iterations for centering
  int max_iter = 50;
  // Rescale KKT System by fixed factor. Rescaling is akin to scaling the log
  // det objective by delta and can improve conditioning. NOTE: This should only
  // be required if using the diagonal preconditioner with the CG method.
  bool rescale_lin_sys = false;
  double rescaling_factor = 1e-5;
  // Select linear solver for centering step
  LinearSolverType lin_solver = LinearSolverType::LDLT;
  // For iterative solvers, choose whether to reuse multipliers
  bool reuse_multipliers = true;

  // Linear Independence Check
  // -------------------------
  // tolerance for checking linear independence of constraints (used in problem
  // validation)
  double tol_indep_constr = 1e-3;
  // flag to enable checking linear independence of constraints (used in problem
  // validation)
  bool check_indep_constr = false;
  // initial perturbation value for centering/certification
  double delta = 1e-5;

  // Adaptive Perturbation Parameters
  // -------------------------
  // Flag to turn on perturbation of the constraints by delta
  bool perturb_constraints = false;
  // Flag to turn on perturbation of the cost by delta
  bool perturb_cost = true;
  // Initial perturbation of cost constraint
  double eps_cost = 1e-5;
  // Initial perturbation of other constraints
  double eps_constr = 1e-5;
  // enable adaptive perturbation for centering
  bool adaptive_perturb = true;
  // final value for multiplier applied to perturbation of cost and constraints
  double eps_mult_min = 1e-2;
  // Threshold for increasing perturbation. If the step size alpha from the line
  // search is below this threshold, then the perturbation is increased for the
  // next iteration.
  double eps_inc_step_thresh = 0.1;
  // Factor for increasing perturbation. If the step size alpha from the line
  // search is below eps_inc_step_thresh, then the perturbation is multiplied by
  // this factor for the next iteration.
  double eps_inc = 2.0;
  // Threshold for decreasing perturbation. If the step size alpha from the line
  // search is above this threshold, then the perturbation is decreased for the
  // next iteration.
  double eps_dec_step_thresh = 0.9;
  // Factor for decreasing perturbation. If the step size alpha from the line
  // search is above eps_dec_step_thresh, then the perturbation is multiplied by
  // this factor for the next iteration.
  double eps_dec = 0.6;

  // Iterative Linear Solve Parameters
  // -----------------
  // max number of iterations for iterative linear solvers (CG and MFCG)
  int lin_solve_max_iter = 500;
  // tolerance for iterative linear solvers (CG and MFCG)
  double lin_solve_tol = 1e-5;
  // Low rank preconditioner parameters
  LowRankPrecondParams lrp_params = LowRankPrecondParams();

  // Line search
  // ----------------
  // line search enable for analytic center
  bool enable_line_search = true;
  // Line search reduction factor (should be between zero and one)
  double ln_search_red_factor = 0.8;
  // Line search initialization
  double alpha_init = 1.0;
  // Line search lower bound
  double alpha_min = 1e-10;

  // Early stop parameters
  // -------------------------
  // enable for certificate check during centering
  // NOTE: can be used to terminate centering early if the certificate is PSD
  // within tolerance, which can be a good heuristic to avoid unnecessary
  // centering steps when the solution is already close to optimal
  bool early_stop_cert = true;
  // tolerance for checking PSDness of certificate matrix
  double tol_cert_psd = 1e-5;
  // tolerance for checking first order condition of certificate matrix
  double tol_cert_complementarity = 1e-5;
  // primal feasibility tolerance for certificate check (i.e., tolerance for
  // violation of constraints)
  double tol_cert_primal_feas = 1e-5;
  // Early stopping condition for deviation from the candidate solution.
  // Violation of condition implies certificate failure.
  // NOTE: This only detects if the solution is not the analytic center, which
  // is a looser condition than if it is not globally optimal
  bool early_stop_angle = false;
  // Maximum allowable angle between the current solution and the candidate
  // solution for early stopping (in radians).
  double max_angle = 1e-2;
  // use the centrality metric from He et al. 1997
  bool use_cert_centrality_metric = false;
  // centrality metric tolerance
  double tol_cert_centrality = 1e-5;
};

class AnalyticCenter {
 public:
  // dimension of the sdp
  // NOTE: if required we could template on size to make things faster
  int dim;
  // number of constraints + cost if required
  int m;
  // cost matrix
  const Matrix C_;

 protected:
  // Stored copies of constraints and RHS. These may be filtered in the
  // constructor if dependent constraints are detected.
  std::vector<Eigen::SparseMatrix<double>> A_storage_;
  std::vector<double> b_storage_;

 public:
  // optimal cost value
  // NOTE: this is non-const because we may want to change it after the problem
  // is defined.
  double rho_;
  // constraint matrices
  const std::vector<Eigen::SparseMatrix<double>>& A_;
  // constraint values
  const std::vector<double>& b_;
  // parameters
  AnalyticCenterParams params_;

  // Constructor
  AnalyticCenter(const Matrix& C, double rho,
                 const std::vector<Eigen::SparseMatrix<double>>& A,
                 const std::vector<double>& b, AnalyticCenterParams params);

  // Evaluate constraints (and cost if enabled) and compute the gradients
  Vector eval_constraints(const Matrix& Y) const;

  // Run analytic centering algorithm starting to certify local solution Y_0.
  // Returns the final result, including the centered primal solution, the
  // certificate matrix, the optimal multipliers, and the certificate
  // information (minimum eigenvalue and complementarity).
  // If perturb is provided, it will be used as the initial perturbation matrix;
  // otherwise, params_.delta * Identity is used as the fallback.
  AnalyticCenterResult certify(const Matrix& Y_0,
                               const Matrix* perturb = nullptr) const;

  // Centering method to compute the analytic center of the current
  // feasible region starting from X_0.
  // The initial perturbation is taken from params_.delta. Delta is used
  // to ensure we stay in the interior of the PSD cone even when the solution
  // is low rank. If delta is zero then no perturbation is applied.
  // If perturb is provided, it will be used as the initial perturbation matrix;
  // otherwise, params_.delta * Identity is used as the fallback.
  // Returns the centered solution and the scaled multipliers for certificate
  // checking.
  std::pair<Matrix, Vector> get_analytic_center(
      const Matrix& Y_0, const Matrix* perturb = nullptr) const;

  // Build weighted sum of constraint matrices: sum_i A_i * lambda_i
  // Note: to build certificate for SDP, the multipliers should be rescaled by
  // the last multiplier (corresponding to the cost constraint)
  Matrix build_adjoint(const Vector& coeffs) const;

  // Check global optimality of a solution
  // Returns the minimum eigenvalue of the certificate matrix and the evaluation
  // of the certificate matrix at the solution (first order condition)
  std::pair<double, double> eval_certificate(const Matrix& H,
                                             const Matrix& Y) const;

  // Check global optimality of a solution
  // Returns the whether certificate matrix is PSD and complementarity
  // of the provided solution.
  std::pair<double, double> check_certificate(const Matrix& H,
                                              const Matrix& Y) const;

  // Export the current problem to a text file following the format expected
  // by `load_problem_from_file` in test/include/generic_sdp_problems.hpp.
  // `problem_name` is written as the "name" field. `solution` is an
  // optional matrix describing a known solution (can be empty with 0 rows).
  void export_problem(const std::filesystem::path& file_path,
                      const std::string& problem_name,
                      const Matrix& solution) const;

 protected:
  // Previous multipliers for iterative linear system solvers
  mutable Vector prev_multipliers_;
  // Matrix Free, Low Rank Preconditioned Conjugate Gradient solver
  // We store this so that we can reuse the preconditioner.
  mutable std::unique_ptr<Eigen::ConjugateGradient<
      MultiplierLinSys, Eigen::Upper | Eigen::Lower, LowRankPrecond>>
      lr_solver;
  // Expected rank of the initial solution
  mutable int rank_init;

  // Builds and solves the system of equations for the analytic center step,
  // returning the optimal multipliers and the current violation of constraints
  std::pair<Vector, Vector> get_multipliers(const Matrix& Z, const Matrix& Y_0,
                                            double eps_mult) const;

  // Intermediate representation of the analytic center linear system
  // Note: it may be more efficient to use references here.
  struct LinSysData {
    Matrix B;          // LHS matrix (m x m)
    Vector d;          // RHS vector (m)
    Vector violation;  // constraint violation (m)
    SpMatrix A_bar;
    std::unique_ptr<MultiplierLinSys>
        B_mf;  // Matrix-free operator for B (if using matrix-free solver)
    const Matrix&
        X_;  // current primal solution (for building matrix-free operator)
    double scale_;  // scaling factor for linear system (for rescaling)

    LinSysData(const Matrix& X, int dim, int m)
        : B(Matrix(m, m)),
          d(Vector(m)),
          violation(Vector(m)),
          A_bar(SpMatrix(dim * dim, m)),
          B_mf(nullptr),
          X_(X) {}
  };

  // Constructs the linear system (H, d, violation) for the analytic center step
  LinSysData build_ac_system(const Matrix& X, double eps_mult) const;

  // Solves the linear system H * multipliers = d using the configured solver
  Vector solve_ac_system(const LinSysData& system, const Matrix& Y_0) const;

  double get_analytic_center_objective(const Matrix& X, double delta) const {
    auto I = Matrix::Identity(X.rows(), X.cols());
    return -logdet(X + I * delta);
  }

  // Line search to ensure PSDness of the solution for the analytic center step
  // This function will update Z with the new solution after line search.
  // Returns the final step size alpha used for the update and the Cholesky
  // factorization of the updated solution for free reuse in the next iteration.
  std::pair<double, Matrix> line_search_factorization(Matrix& Z,
                                                      const Matrix& dZ) const;
};

}  // namespace RankTools
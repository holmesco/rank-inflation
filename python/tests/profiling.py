"""Tests for AnalyticCenterPyTorch using Lovasz theta fixtures."""

from __future__ import annotations

import torch
import numpy as np


from ranktools import LinearSolverType, solve_sdp_mosek, SDPResult

from ranktools_pytorch.analytic_center_torch import (
    AnalyticCenterPyTorch,
    defaultAnalyicCenterParams,
    LowRankPrecondMethod,
)
from ranktools_pytorch.lin_alg_torch import LowRankPrecond
from tests.fixtures import (
    clique1_adj,
    clique2_adj,
    clique3_adj,
    clique4_adj,
    make_lovasz_test_case,
    load_problem_from_file,
)

LOVASZ_TESTS = [
    (clique1_adj, [1, 3, 4, 6, 7, 8], "Clique1"),
    (clique2_adj, [0, 2, 3, 5, 6, 8, 9], "Clique2"),
    (clique3_adj, [4, 10, 13, 14, 15, 16, 17, 18], "Clique3_Large20x20"),
    (clique4_adj, [0, 1, 2], "Clique4_Disconnected"),
]

# Turn off gradients
torch.set_grad_enabled(False)


def profile_certify(adj, clique, name):
    sdp = make_lovasz_test_case(adj, clique, name)

    # Recover Analytic center from Mosek
    b = sdp.b.numpy()
    C = sdp.C.numpy()
    result_sdp: SDPResult = solve_sdp_mosek(
        C, sdp.A_mosek, b
    )  # Just to check Mosek can solve it
    X = result_sdp.X
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    rank = np.sum(S > S[0] * 1e-5)
    Y_0 = torch.Tensor(U[:, :rank] @ np.diag(np.sqrt(S[:rank])))

    params = defaultAnalyicCenterParams()
    params.verbose = False
    params.max_iter = 25
    params.delta = 1e-5
    params.lrp_params.tau = 1e-5
    params.perturb_cost = True
    params.perturb_constraints = True
    params.eps_cost = 1e-5
    params.eps_constr = 1e-5
    params.early_stop_cert = True
    params.adaptive_perturb = True
    params.eps_mult_min = 1e-4
    params.delta = 1e-5
    params.lin_solver = LinearSolverType.MFCG_LRP
    params.lrp_params.method = LowRankPrecondMethod.DenseLU

    certifier = AnalyticCenterPyTorch(
        C=sdp.C,
        rho=sdp.rho,
        A_list=sdp.A,
        b=sdp.b,
        params=params,
        main_gpu=True,
    )
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./trace_dir"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        result = None
        for _ in range(2):
            result = certifier.certify(Y_0)
            prof.step()

    assert result.certified, ValueError(f"Certification failed")
    assert result.min_eig > -params.tol_cert_psd, ValueError(
        f"Minimum eigenvalue negative: {result.min_eig}"
    )
    assert result.complementarity < params.tol_cert_complementarity, ValueError(
        f"Complementarity gap too large: {result.complementarity}"
    )
    print(f"Profiling complete. Time: {result.solver_time:.4f} seconds")


if __name__ == "__main__":
    profile_certify(*LOVASZ_TESTS[0])

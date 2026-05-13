"""Tests for AnalyticCenterPyTorch using Lovasz theta fixtures."""

from __future__ import annotations

import pytest
import torch

from ranktools_pytorch.analytic_center_torch import (
    AnalyticCenterPyTorch,
    defaultAnalyicCenterParams,
)
from .fixtures import (
    clique1_adj,
    clique2_adj,
    clique3_adj,
    clique4_adj,
    make_lovasz_test_case,
)


@pytest.mark.parametrize(
    "adj,clique,name",
    [
        (clique1_adj, [1, 3, 4, 6, 7, 8], "Clique1"),
        (clique2_adj, [0, 2, 3, 5, 6, 8, 9], "Clique2"),
        (clique3_adj, [4, 10, 13, 14, 15, 16, 17, 18], "Clique3_Large20x20"),
        (clique4_adj, [0, 1, 2], "Clique4_Disconnected"),
    ],
)
def test_certify_lovasz_theta(adj, clique, name):
    sdp = make_lovasz_test_case(adj, clique, name)
    Y_0 = sdp.make_solution(1)

    params = defaultAnalyicCenterParams()
    params.verbose = False
    params.max_iter = 25

    certifier = AnalyticCenterPyTorch(
        C=sdp.C,
        rho=sdp.rho,
        A_list=sdp.A,
        b=sdp.b,
        params=params,
        device=torch.device("cpu"),
    )

    result = certifier.certify(Y_0)

    assert result.X.shape == (sdp.dim, sdp.dim)
    assert result.H.shape == (sdp.dim, sdp.dim)
    assert result.multipliers.shape == (len(sdp.A) + 1,)
    assert result.violation.shape == (len(sdp.A) + 1,)
    assert torch.isfinite(result.X).all()
    assert torch.isfinite(result.H).all()
    assert torch.isfinite(result.violation).all()
    assert isinstance(result.certified, bool)
    assert result.solver_time >= 0.0
    assert result.num_iters >= 0

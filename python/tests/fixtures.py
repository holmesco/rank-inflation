"""Test fixtures mirroring the C++ SDP test harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import json

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class SDPTestProblem:
    """Test case data structure mirroring C++ SDPTestProblem."""

    dim: int
    C: torch.Tensor
    rho: float
    A: List[sp.spmatrix]
    b: torch.Tensor
    soln: torch.Tensor
    name: str
    soln_is_global: bool
    A_mosek: List[sp.spmatrix] | None = None

    def make_solution(self, rank: int) -> torch.Tensor:
        """Retrieve zero padded solution for testing."""
        if rank < self.soln.shape[1]:
            raise ValueError(
                f"Requested rank {rank} is smaller than solution rank {self.soln.shape[1]}"
            )
        zpad = torch.zeros((self.dim, rank - self.soln.shape[1]), dtype=self.soln.dtype)
        return torch.cat([self.soln, zpad], dim=1)


# ----------- Lovasz Theta Helper Functions ----------------

Edge = Tuple[int, int]


def get_edges(adj: np.ndarray) -> Tuple[List[Edge], List[Edge]]:
    """Compute edges from adjacency, only provide upper triangle indices."""
    edges: List[Edge] = []
    nonedges: List[Edge] = []
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j] > 0.0:
                edges.append((i, j))
            else:
                nonedges.append((i, j))
    return edges, nonedges


def get_lovasz_constraints(dim: int, nonedges: Sequence[Edge]) -> List[sp.spmatrix]:
    """Convert adjacency to Lovasz theta constraints (sparse upper-triangular)."""
    A: List[sp.spmatrix] = []
    for i, j in nonedges:
        mat = sp.coo_matrix(([1.0], ([i], [j])), shape=(dim, dim)).tocsr()
        A.append(mat)
    # Trace constraint
    A.append(sp.identity(dim, format="csr"))
    return A

def get_lovasz_constraints_mosek(dim: int, nonedges: Sequence[Edge]) -> List[sp.spmatrix]:
    """Convert adjacency to Lovasz theta constraints (sparse upper-triangular)."""
    A: List[sp.spmatrix] = []
    for i, j in nonedges:
        mat = sp.coo_matrix(([1.0,1.0], ([i,j], [j,i])), shape=(dim, dim)).tocsc()
        A.append(mat)
    # Trace constraint
    A.append(sp.identity(dim, format="csr"))
    return A

def make_lovasz_test_case(
    adj: np.ndarray, clique: Sequence[int], name: str
) -> SDPTestProblem:
    dim = adj.shape[0]
    _, nonedges = get_edges(adj)

    C = -torch.ones((dim, dim), dtype=torch.float64)
    rho = -float(len(clique))

    A = get_lovasz_constraints(dim, nonedges)
    A_mosek = get_lovasz_constraints_mosek(dim, nonedges)
    b = torch.zeros(len(A), dtype=torch.float64)
    b[-1] = 1.0

    soln = torch.zeros((dim, 1), dtype=torch.float64)
    s = np.sqrt(1.0 / len(clique))
    for idx in clique:
        soln[idx, 0] = s

    return SDPTestProblem(
        dim=dim,
        C=C,
        rho=rho,
        A=A,
        A_mosek=A_mosek,
        b=b,
        soln=soln,
        name=f"LovaszTheta_{name}",
        soln_is_global=True,
    )


# ------------ Lovasz-Theta Data Matrices ------------


def _load_lovasz_matrices() -> dict:
    fixture_dir = Path(__file__).resolve().parent
    path = fixture_dir / "data" / "lovasz_theta_matrices.json"
    return json.loads(path.read_text())


_LOVASZ_DATA = _load_lovasz_matrices()

clique1_adj = np.array(_LOVASZ_DATA["clique1_adj"], dtype=float).reshape(10, 10)
clique2_adj = np.array(_LOVASZ_DATA["clique2_adj"], dtype=float).reshape(10, 10)
clique3_adj = np.array(_LOVASZ_DATA["clique3_adj"], dtype=float).reshape(20, 20)
clique4_adj = np.array(_LOVASZ_DATA["clique4_adj"], dtype=float).reshape(5, 5)


# ----------- Exported SDP Problems ----------------


def _expect_key(tokens: Iterable[str], expected: str, file_name: str) -> str:
    try:
        key = next(tokens)
    except StopIteration as exc:
        raise RuntimeError(
            f"Unexpected end-of-file in {file_name}, expected key: {expected}"
        ) from exc
    if key != expected:
        raise RuntimeError(
            f"Malformed problem file {file_name}: expected key '{expected}', got '{key}'"
        )
    return key


def _find_problem_file(problem_name: str) -> Path:
    file_name = f"{problem_name}.txt"
    fixture_dir = Path(__file__).resolve().parent
    repo_root = fixture_dir.parents[2]

    candidates = [
        repo_root / "test" / "data" / file_name,
        repo_root / "data" / file_name,
        Path.cwd() / "test" / "data" / file_name,
        Path.cwd() / ".." / "test" / "data" / file_name,
        Path.cwd() / ".." / ".." / "test" / "data" / file_name,
        Path.cwd() / "data" / file_name,
        Path.cwd() / ".." / "data" / file_name,
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    tried = "\n".join(f"  - {p}" for p in candidates)
    raise RuntimeError(
        f"Could not find problem file for '{problem_name}'. Tried:\n{tried}"
    )


def load_problem_from_file(problem_name: str) -> SDPTestProblem:
    path = _find_problem_file(problem_name)
    tokens = iter(path.read_text().split())

    _expect_key(tokens, "name", path.as_posix())
    name = next(tokens)

    _expect_key(tokens, "dim", path.as_posix())
    dim = int(next(tokens))

    _expect_key(tokens, "C", path.as_posix())
    c_count = int(next(tokens))
    if c_count != dim * dim:
        raise RuntimeError(f"Invalid C size in {path}")

    C_entries = [float(next(tokens)) for _ in range(c_count)]
    C = np.array(C_entries, dtype=float).reshape(dim, dim)

    # remove homogenization offset
    C[0, 0] = 0.0
    # Rescale C to have norm 1, to avoid numerical issues in testing.
    C /= np.linalg.norm(C)

    _expect_key(tokens, "constraints", path.as_posix())
    n_constraints = int(next(tokens))

    A_list: List[sp.spmatrix] = []
    b_list: List[float] = []

    for _ in range(n_constraints):
        _expect_key(tokens, "A", path.as_posix())
        rows = int(next(tokens))
        cols = int(next(tokens))
        nnz = int(next(tokens))

        r_idx = []
        c_idx = []
        values = []
        for _ in range(nnz):
            r_idx.append(int(next(tokens)))
            c_idx.append(int(next(tokens)))
            values.append(float(next(tokens)))

        A = sp.coo_matrix((values, (r_idx, c_idx)), shape=(rows, cols)).tocsr()
        A_list.append(A)

        _expect_key(tokens, "b", path.as_posix())
        b_list.append(float(next(tokens)))

    _expect_key(tokens, "soln", path.as_posix())
    soln_rows = int(next(tokens))
    soln_cols = int(next(tokens))
    soln_count = int(next(tokens))
    if soln_rows * soln_cols != soln_count:
        raise RuntimeError(f"Invalid solution shape/count in {path}")

    soln_entries = [float(next(tokens)) for _ in range(soln_count)]
    soln = np.array(soln_entries, dtype=float).reshape(soln_rows, soln_cols)

    soln_t = torch.tensor(soln, dtype=torch.float64)
    C_t = torch.tensor(C, dtype=torch.float64)

    rho = float((soln_t.T @ C_t @ soln_t).trace())
    soln_is_global = "L" not in name

    return SDPTestProblem(
        dim=dim,
        C=C_t,
        rho=rho,
        A=A_list,
        b=torch.tensor(b_list, dtype=torch.float64),
        soln=soln_t,
        name=name,
        soln_is_global=soln_is_global,
    )


def make_exported_sdp_test_problems() -> List[SDPTestProblem]:
    """Load the set of exported SDP test problems (excluding known bad cases)."""
    problem_names = [
        "test_prob_1",
        "test_prob_10G",
        "test_prob_10Gc",
        "test_prob_10L",
        "test_prob_10Lc",
        "test_prob_11G",
        "test_prob_11Gc",
        "test_prob_11L",
        "test_prob_11Lc",
        "test_prob_12G",
        "test_prob_12Gc",
        "test_prob_12L",
        "test_prob_12Lc",
        "test_prob_13G",
        "test_prob_13Gc",
        "test_prob_13L",
        "test_prob_13Lc",
        "test_prob_14G",
        "test_prob_15G",
        "test_prob_16G",
        "test_prob_16L",
        "test_prob_2",
        "test_prob_3",
        "test_prob_4",
        "test_prob_5",
        "test_prob_6",
        "test_prob_7",
        "test_prob_8G",
        "test_prob_8L1",
        "test_prob_8L1c",
        "test_prob_8L2",
        "test_prob_8L2c",
        "test_prob_9",
        "test_prob_9G",
        "test_prob_9Gc",
        "test_prob_9L",
        "test_prob_9L1",
        "test_prob_9L1c",
        "test_prob_9Lc",
        "test_prob_9c",
    ]

    out: List[SDPTestProblem] = []
    for name in problem_names:
        if name in {"test_prob_2", "test_prob_4", "test_prob_7"}:
            continue
        out.append(load_problem_from_file(name))
    return out


__all__ = [
    "SDPTestProblem",
    "get_edges",
    "get_lovasz_constraints",
    "make_lovasz_test_case",
    "clique1_adj",
    "clique2_adj",
    "clique3_adj",
    "clique4_adj",
    "load_problem_from_file",
    "make_exported_sdp_test_problems",
]

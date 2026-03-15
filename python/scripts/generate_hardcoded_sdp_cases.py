from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SparseMat:
    rows: int
    cols: int
    triplets: list[tuple[int, int, float]] = field(default_factory=list)


@dataclass
class DenseMat:
    rows: int
    cols: int
    values_row_major: list[float]


def _parse_shape_csv(shape_s: str) -> list[int]:
    return [int(x) for x in shape_s.split(",") if x]


def _parse_values_csv(values_s: str) -> list[float]:
    if not values_s:
        return []
    return [float(x) for x in values_s.split(",") if x]


def _parse_dense_block(lines: list[str], i: int) -> tuple[DenseMat, int]:
    ndim_line = lines[i + 1]
    shape_line = lines[i + 2]
    values_line = lines[i + 3]
    end_line = lines[i + 4]

    if not ndim_line.startswith("NDIM|"):
        raise ValueError(f"Expected NDIM after BEGIN_DENSE at line {i + 2}")
    if not shape_line.startswith("SHAPE|"):
        raise ValueError(f"Expected SHAPE after BEGIN_DENSE at line {i + 3}")
    if not values_line.startswith("VALUES|"):
        raise ValueError(f"Expected VALUES after BEGIN_DENSE at line {i + 4}")
    if end_line != "END_DENSE":
        raise ValueError(f"Expected END_DENSE at line {i + 5}")

    ndim = int(ndim_line.split("|", 1)[1])
    shape = _parse_shape_csv(shape_line.split("|", 1)[1])
    vals = _parse_values_csv(values_line.split("|", 1)[1])

    if ndim == 1:
        rows, cols = shape[0], 1
    elif ndim == 2:
        rows, cols = shape[0], shape[1]
    else:
        raise ValueError(f"Only 1D/2D dense blocks supported (ndim={ndim})")

    return DenseMat(rows=rows, cols=cols, values_row_major=vals), i + 5


def _parse_sparse_block(lines: list[str], i: int) -> tuple[SparseMat, int]:
    shape_line = lines[i + 1]
    nnz_line = lines[i + 2]
    triplet_hdr = lines[i + 3]

    if not shape_line.startswith("SHAPE|"):
        raise ValueError(f"Expected SHAPE after BEGIN_SPARSE at line {i + 2}")
    if not nnz_line.startswith("NNZ|"):
        raise ValueError(f"Expected NNZ after BEGIN_SPARSE at line {i + 3}")
    if not triplet_hdr.startswith("TRIPLETS|"):
        raise ValueError(f"Expected TRIPLETS header at line {i + 4}")

    shape = _parse_shape_csv(shape_line.split("|", 1)[1])
    nnz = int(nnz_line.split("|", 1)[1])
    out = SparseMat(rows=shape[0], cols=shape[1])

    j = i + 4
    for _ in range(nnz):
        r_s, c_s, v_s = lines[j].split(",")
        out.triplets.append((int(r_s), int(c_s), float(v_s)))
        j += 1

    if lines[j] != "END_SPARSE":
        raise ValueError(f"Expected END_SPARSE at line {j + 1}")

    return out, j + 1


def parse_export_file(path: Path) -> dict:
    lines = path.read_text().splitlines()

    q_dense: DenseMat | None = None
    q_sparse: SparseMat | None = None
    x_dense: DenseMat | None = None
    A_dense: dict[int, DenseMat] = {}
    A_sparse: dict[int, SparseMat] = {}
    b_vals: dict[int, float] = {}

    re_a_block = re.compile(r"^BEGIN_(DENSE|SPARSE)\|root/Constraints/(\d+)/0$")
    re_b_line = re.compile(r"^SCALAR\|root/Constraints/(\d+)/1\|(.+)$")

    i = 0
    while i < len(lines):
        line = lines[i]

        if line == "BEGIN_DENSE|root/Q":
            q_dense, i = _parse_dense_block(lines, i)
            continue
        if line == "BEGIN_SPARSE|root/Q":
            q_sparse, i = _parse_sparse_block(lines, i)
            continue
        if line == "BEGIN_DENSE|root/x_cand":
            x_dense, i = _parse_dense_block(lines, i)
            continue

        m = re_a_block.match(line)
        if m:
            kind = m.group(1)
            idx = int(m.group(2))
            if kind == "DENSE":
                d, i = _parse_dense_block(lines, i)
                A_dense[idx] = d
            else:
                s, i = _parse_sparse_block(lines, i)
                A_sparse[idx] = s
            continue

        mb = re_b_line.match(line)
        if mb:
            idx = int(mb.group(1))
            b_vals[idx] = float(mb.group(2))
            i += 1
            continue

        i += 1

    if q_dense is None and q_sparse is None:
        raise ValueError(f"No root/Q found in {path}")
    if x_dense is None:
        raise ValueError(f"No root/x_cand found in {path}")

    n_cons = max([*A_dense.keys(), *A_sparse.keys(), *b_vals.keys()], default=-1) + 1

    constraints: list[tuple[DenseMat | SparseMat, float]] = []
    for k in range(n_cons):
        A = A_sparse.get(k)
        if A is None:
            A = A_dense.get(k)
        if A is None:
            raise ValueError(f"Missing A for constraint {k} in {path}")
        if k not in b_vals:
            raise ValueError(f"Missing b for constraint {k} in {path}")
        constraints.append((A, b_vals[k]))

    return {
        "name": path.stem,
        "Q_dense": q_dense,
        "Q_sparse": q_sparse,
        "x_cand": x_dense,
        "constraints": constraints,
    }


def _fmt(v: float) -> str:
    return f"{v:.17g}"


def _emit_dense_values(vals: list[float], indent: str = "      ") -> str:
    chunks = []
    line = indent
    for i, v in enumerate(vals):
        token = _fmt(v)
        if i < len(vals) - 1:
            token += ", "
        if len(line) + len(token) > 110:
            chunks.append(line.rstrip())
            line = indent + token
        else:
            line += token
    if line.strip():
        chunks.append(line.rstrip())
    return "\n".join(chunks)


def _emit_fill_dense(var: str, d: DenseMat) -> str:
    vals = _emit_dense_values(d.values_row_major)
    return (
        f"    fill_dense({var}, {d.rows}, {d.cols}, std::vector<double>{{\n"
        f"{vals}\n"
        f"    }});\n"
    )


def _emit_sparse_to_dense_add(target: str, s: SparseMat) -> str:
    lines = [f"    {target}.setZero({s.rows}, {s.cols});"]
    for r, c, v in s.triplets:
        lines.append(f"    {target}({r}, {c}) += {_fmt(v)};")
    return "\n".join(lines) + "\n"


def _emit_sparse_constraint(s: SparseMat) -> str:
    lines = [
        "    {",
        f"      Eigen::SparseMatrix<double> A({s.rows}, {s.cols});",
        "      std::vector<Eigen::Triplet<double>> T;",
        f"      T.reserve({len(s.triplets)});",
    ]
    for r, c, v in s.triplets:
        lines.append(f"      T.emplace_back({r}, {c}, {_fmt(v)});")
    lines.extend(
        [
            "      A.setFromTriplets(T.begin(), T.end());",
            "      sdp.A.push_back(A);",
            "    }",
        ]
    )
    return "\n".join(lines) + "\n"


def _emit_dense_constraint(d: DenseMat) -> str:
    rows, cols = d.rows, d.cols
    vals = d.values_row_major
    lines = [
        "    {",
        f"      Eigen::SparseMatrix<double> A({rows}, {cols});",
        "      std::vector<Eigen::Triplet<double>> T;",
    ]
    nnz = 0
    for r in range(rows):
        base = r * cols
        for c in range(cols):
            if vals[base + c] != 0.0:
                nnz += 1
    lines.append(f"      T.reserve({nnz});")
    for r in range(rows):
        base = r * cols
        for c in range(cols):
            v = vals[base + c]
            if v != 0.0:
                lines.append(f"      T.emplace_back({r}, {c}, {_fmt(v)});")
    lines.extend(
        [
            "      A.setFromTriplets(T.begin(), T.end());",
            "      sdp.A.push_back(A);",
            "    }",
        ]
    )
    return "\n".join(lines) + "\n"


def emit_case(case: dict) -> str:
    name = case["name"]
    q_dense: DenseMat | None = case["Q_dense"]
    q_sparse: SparseMat | None = case["Q_sparse"]
    x: DenseMat = case["x_cand"]
    constraints = case["constraints"]

    dim = q_dense.rows if q_dense is not None else q_sparse.rows

    lines: list[str] = []
    lines.append(f"inline SDPTestProblem make_{name}() {{")
    lines.append("  SDPTestProblem sdp;")
    lines.append(f"  sdp.name = \"{name}\";")
    lines.append(f"  sdp.dim = {dim};")
    lines.append("")

    lines.append("  // C (from exported Q)")
    lines.append("  sdp.C = Matrix::Zero(sdp.dim, sdp.dim);")
    if q_dense is not None:
        lines.append(_emit_fill_dense("sdp.C", q_dense).rstrip())
    else:
        lines.append(_emit_sparse_to_dense_add("sdp.C", q_sparse).rstrip())
    lines.append("")

    lines.append("  // Constraints")
    lines.append("  sdp.A.clear();")
    lines.append("  sdp.b.clear();")
    lines.append(f"  sdp.A.reserve({len(constraints)});")
    lines.append(f"  sdp.b.reserve({len(constraints)});")
    for A, b in constraints:
        if isinstance(A, SparseMat):
            lines.append(_emit_sparse_constraint(A).rstrip())
        else:
            lines.append(_emit_dense_constraint(A).rstrip())
        lines.append(f"  sdp.b.push_back({_fmt(b)});")
    lines.append("")

    lines.append("  // Candidate solution (x_cand)")
    lines.append(_emit_fill_dense("sdp.soln", x).rstrip())
    lines.append("")
    lines.append("  // rho from candidate and C")
    lines.append("  sdp.rho = (sdp.soln.transpose() * sdp.C * sdp.soln).trace();")
    lines.append("  return sdp;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def generate_header(cases: list[dict], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <cstddef>")
    lines.append("#include <Eigen/Dense>")
    lines.append("#include <Eigen/Sparse>")
    lines.append("#include <stdexcept>")
    lines.append("#include <vector>")
    lines.append("")
    lines.append("namespace RankTools {")
    lines.append("namespace ExportedSDPProblems {")
    lines.append("")
    lines.append("inline void fill_dense(Matrix& M, int rows, int cols, const std::vector<double>& vals) {")
    lines.append("  if (static_cast<int>(vals.size()) != rows * cols) {")
    lines.append("    throw std::runtime_error(\"fill_dense size mismatch\");")
    lines.append("  }")
    lines.append("  M.resize(rows, cols);")
    lines.append("  for (int r = 0; r < rows; ++r) {")
    lines.append("    for (int c = 0; c < cols; ++c) {")
    lines.append("      M(r, c) = vals[static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) +")
    lines.append("                     static_cast<std::size_t>(c)];")
    lines.append("    }")
    lines.append("  }")
    lines.append("}")
    lines.append("")

    for case in cases:
        lines.append(emit_case(case))

    lines.append("inline std::vector<SDPTestProblem> make_exported_sdp_test_problems() {")
    lines.append("  std::vector<SDPTestProblem> out;")
    lines.append(f"  out.reserve({len(cases)});")
    for case in cases:
        lines.append(f"  out.push_back(make_{case['name']}());")
    lines.append("  return out;")
    lines.append("}")
    lines.append("")
    lines.append("}  // namespace ExportedSDPProblems")
    lines.append("}  // namespace RankTools")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate hard-coded SDPTestProblem C++ header from exported txt files"
    )
    parser.add_argument("--input-dir", type=Path, default=Path("/workspace/python/examples_export"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/test/include/generated_exported_sdp_problems.hpp"),
    )
    args = parser.parse_args()

    txt_files = sorted(args.input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {args.input_dir}")

    cases = [parse_export_file(p) for p in txt_files]
    generate_header(cases, args.output)
    print(f"Generated {len(cases)} cases -> {args.output}")


if __name__ == "__main__":
    main()

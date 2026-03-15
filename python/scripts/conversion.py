"""Convert problem `.pkl` files into exactly one `.txt` file each.

The output is line-oriented and C++-friendly:
- Dense arrays are exported with shape + flattened values.
- Sparse matrices are preserved in COO form (row, col, value triplets).
- Nested dict/list/tuple structure is serialized recursively.

Usage:
	python3 python/scripts/conversion.py
	python3 python/scripts/conversion.py --input-dir /workspace/python/examples --output-dir /workspace/python/examples_export
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
	import scipy.sparse as sp
except Exception:  # pragma: no cover - optional dependency guard
	sp = None


def _is_sparse(x: Any) -> bool:
	return sp is not None and sp.issparse(x)


def _as_float_array(x: Any) -> np.ndarray:
	return np.asarray(x, dtype=np.float64)


def _fmt_float(v: Any) -> str:
	return f"{float(v):.17g}"


def _write_dense(name: str, arr: np.ndarray, out) -> None:
	a = np.asarray(arr)
	if a.ndim == 0:
		out.write(f"SCALAR|{name}|{_fmt_float(a.item())}\n")
		return

	a = _as_float_array(a)
	shape = ",".join(str(int(s)) for s in a.shape)
	flat = a.ravel(order="C")
	vals = ",".join(_fmt_float(v) for v in flat)
	out.write(f"BEGIN_DENSE|{name}\n")
	out.write(f"NDIM|{a.ndim}\n")
	out.write(f"SHAPE|{shape}\n")
	out.write(f"VALUES|{vals}\n")
	out.write("END_DENSE\n")


def _write_sparse(name: str, mat: Any, out) -> None:
	coo = mat.tocoo()
	out.write(f"BEGIN_SPARSE|{name}\n")
	out.write(f"SHAPE|{int(coo.shape[0])},{int(coo.shape[1])}\n")
	out.write(f"NNZ|{int(coo.nnz)}\n")
	out.write("TRIPLETS|row,col,value\n")
	for r, c, v in zip(coo.row, coo.col, coo.data):
		out.write(f"{int(r)},{int(c)},{_fmt_float(v)}\n")
	out.write("END_SPARSE\n")


def _sanitize_key(k: Any) -> str:
	return str(k).replace("|", "_").replace("\n", " ").strip()


def _write_obj(name: str, obj: Any, out) -> None:
	if _is_sparse(obj):
		_write_sparse(name, obj, out)
		return

	if isinstance(obj, np.ndarray):
		_write_dense(name, obj, out)
		return

	if isinstance(obj, (int, float, np.integer, np.floating)):
		out.write(f"SCALAR|{name}|{_fmt_float(obj)}\n")
		return

	if isinstance(obj, (bool, np.bool_)):
		out.write(f"BOOL|{name}|{int(obj)}\n")
		return

	if isinstance(obj, str):
		out.write(f"STRING|{name}|{json.dumps(obj)}\n")
		return

	if obj is None:
		out.write(f"NONE|{name}\n")
		return

	if isinstance(obj, dict):
		out.write(f"BEGIN_DICT|{name}|{len(obj)}\n")
		for k in sorted(obj.keys(), key=lambda x: str(x)):
			child_name = f"{name}/{_sanitize_key(k)}"
			_write_obj(child_name, obj[k], out)
		out.write("END_DICT\n")
		return

	if isinstance(obj, (list, tuple)):
		out.write(f"BEGIN_LIST|{name}|{len(obj)}\n")
		for i, item in enumerate(obj):
			child_name = f"{name}/{i}"
			_write_obj(child_name, item, out)
		out.write("END_LIST\n")
		return

	# Last resort: write repr so no data key is silently dropped.
	out.write(f"REPR|{name}|{json.dumps(repr(obj))}\n")


def convert_one_pickle(pkl_path: Path, out_root: Path) -> None:
	with pkl_path.open("rb") as f:
		data = pickle.load(f)

	problem_name = pkl_path.stem
	out_root.mkdir(parents=True, exist_ok=True)
	out_path = out_root / f"{problem_name}.txt"

	with out_path.open("w") as out:
		out.write("FORMAT|rank_inflation_export_v1\n")
		out.write(f"PROBLEM|{problem_name}\n")
		out.write(f"SOURCE_PICKLE|{pkl_path.name}\n")

		if isinstance(data, dict):
			_write_obj("root", data, out)
		else:
			_write_obj("root", data, out)


def convert_all(input_dir: Path, output_dir: Path) -> None:
	pkl_files = sorted(input_dir.glob("*.pkl"))
	if not pkl_files:
		raise FileNotFoundError(f"No .pkl files found in {input_dir}")

	output_dir.mkdir(parents=True, exist_ok=True)

	print(f"Found {len(pkl_files)} pickle files in {input_dir}")
	for pkl_path in pkl_files:
		print(f"Converting {pkl_path.name} -> {pkl_path.stem}.txt ...")
		convert_one_pickle(pkl_path, output_dir)
	print(f"Done. Export written to: {output_dir}")


def main() -> None:
	default_input = Path(__file__).resolve().parents[1] / "examples"
	default_output = Path(__file__).resolve().parents[1] / "examples_export"

	parser = argparse.ArgumentParser(description="Convert problem .pkl files to C++-friendly CSV/TXT")
	parser.add_argument("--input-dir", type=Path, default=default_input, help="Directory containing .pkl files")
	parser.add_argument("--output-dir", type=Path, default=default_output, help="Directory for exported CSV/TXT files")
	args = parser.parse_args()

	input_dir = args.input_dir.resolve()
	output_dir = args.output_dir.resolve()

	if not input_dir.exists() or not input_dir.is_dir():
		raise NotADirectoryError(f"Input directory does not exist or is not a directory: {input_dir}")

	convert_all(input_dir=input_dir, output_dir=output_dir)


if __name__ == "__main__":
	main()

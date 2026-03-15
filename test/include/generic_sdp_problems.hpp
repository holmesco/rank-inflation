// Defines a set of generic SDP problems for testing.
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_harness.hpp"

namespace RankTools {
namespace ExportedSDPProblems {


inline void expect_key(std::istream& in, const std::string& expected,
                       const std::string& file_name) {
  std::string key;
  if (!(in >> key)) {
    throw std::runtime_error("Unexpected end-of-file in " + file_name +
                             ", expected key: " + expected);
  }
  if (key != expected) {
    throw std::runtime_error("Malformed problem file " + file_name +
                             ": expected key '" + expected + "', got '" + key +
                             "'");
  }
}

inline std::filesystem::path find_problem_file(
    const std::string& problem_name) {
  const std::string file = problem_name + ".txt";

  const auto header_dir = std::filesystem::path(__FILE__).parent_path();
  std::vector<std::filesystem::path> candidates{
      header_dir / ".." / "data" / file,
      std::filesystem::current_path() / "test" / "data" / file,
      std::filesystem::current_path() / ".." / "test" / "data" / file,
      std::filesystem::current_path() / ".." / ".." / "test" / "data" / file,
      std::filesystem::current_path() / "data" / file,
      std::filesystem::current_path() / ".." / "data" / file,
  };

  for (const auto& path : candidates) {
    if (std::filesystem::exists(path)) {
      return std::filesystem::canonical(path);
    }
  }

  std::ostringstream oss;
  oss << "Could not find problem file for '" << problem_name << "'. Tried:\n";
  for (const auto& p : candidates) {
    oss << "  - " << p.string() << "\n";
  }
  throw std::runtime_error(oss.str());
}

inline SDPTestProblem load_problem_from_file(const std::string& problem_name) {
  const auto path = find_problem_file(problem_name);
  const std::string path_str = path.string();

  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open problem file: " + path_str);
  }

  SDPTestProblem sdp;

  expect_key(in, "name", path_str);
  in >> sdp.name;

  expect_key(in, "dim", path_str);
  in >> sdp.dim;

  expect_key(in, "C", path_str);
  int c_count = 0;
  in >> c_count;
  if (c_count != sdp.dim * sdp.dim) {
    throw std::runtime_error("Invalid C size in " + path_str);
  }

  sdp.C.resize(sdp.dim, sdp.dim);
  for (int i = 0; i < sdp.dim; ++i) {
    for (int j = 0; j < sdp.dim; ++j) {
      if (!(in >> sdp.C(i, j))) {
        throw std::runtime_error("Failed reading C entries from " + path_str);
      }
    }
  }

  expect_key(in, "constraints", path_str);
  int n_constraints = 0;
  in >> n_constraints;

  sdp.A.clear();
  sdp.b.clear();
  sdp.A.reserve(static_cast<std::size_t>(n_constraints));
  sdp.b.reserve(static_cast<std::size_t>(n_constraints));

  for (int k = 0; k < n_constraints; ++k) {
    expect_key(in, "A", path_str);
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    in >> rows >> cols >> nnz;

    Eigen::SparseMatrix<double> A(rows, cols);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<std::size_t>(nnz));

    for (int t = 0; t < nnz; ++t) {
      int r = 0;
      int c = 0;
      double v = 0.0;
      if (!(in >> r >> c >> v)) {
        throw std::runtime_error("Failed reading triplet entries from " +
                                 path_str);
      }
      triplets.emplace_back(r, c, v);
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
    sdp.A.push_back(std::move(A));

    expect_key(in, "b", path_str);
    double bval = 0.0;
    in >> bval;
    sdp.b.push_back(bval);
  }

  expect_key(in, "soln", path_str);
  int soln_rows = 0;
  int soln_cols = 0;
  int soln_count = 0;
  in >> soln_rows >> soln_cols >> soln_count;

  if (soln_rows * soln_cols != soln_count) {
    throw std::runtime_error("Invalid solution shape/count in " + path_str);
  }

  sdp.soln.resize(soln_rows, soln_cols);
  for (int i = 0; i < soln_rows; ++i) {
    for (int j = 0; j < soln_cols; ++j) {
      if (!(in >> sdp.soln(i, j))) {
        throw std::runtime_error("Failed reading solution entries from " +
                                 path_str);
      }
    }
  }

  sdp.rho = (sdp.soln.transpose() * sdp.C * sdp.soln).trace();
  return sdp;
}

inline SDPTestProblem make_test_prob_1() {
  return load_problem_from_file("test_prob_1");
}
inline SDPTestProblem make_test_prob_10G() {
  return load_problem_from_file("test_prob_10G");
}
inline SDPTestProblem make_test_prob_10Gc() {
  return load_problem_from_file("test_prob_10Gc");
}
inline SDPTestProblem make_test_prob_10L() {
  return load_problem_from_file("test_prob_10L");
}
inline SDPTestProblem make_test_prob_10Lc() {
  return load_problem_from_file("test_prob_10Lc");
}
inline SDPTestProblem make_test_prob_11G() {
  return load_problem_from_file("test_prob_11G");
}
inline SDPTestProblem make_test_prob_11Gc() {
  return load_problem_from_file("test_prob_11Gc");
}
inline SDPTestProblem make_test_prob_11L() {
  return load_problem_from_file("test_prob_11L");
}
inline SDPTestProblem make_test_prob_11Lc() {
  return load_problem_from_file("test_prob_11Lc");
}
inline SDPTestProblem make_test_prob_12G() {
  return load_problem_from_file("test_prob_12G");
}
inline SDPTestProblem make_test_prob_12Gc() {
  return load_problem_from_file("test_prob_12Gc");
}
inline SDPTestProblem make_test_prob_12L() {
  return load_problem_from_file("test_prob_12L");
}
inline SDPTestProblem make_test_prob_12Lc() {
  return load_problem_from_file("test_prob_12Lc");
}
inline SDPTestProblem make_test_prob_13G() {
  return load_problem_from_file("test_prob_13G");
}
inline SDPTestProblem make_test_prob_13Gc() {
  return load_problem_from_file("test_prob_13Gc");
}
inline SDPTestProblem make_test_prob_13L() {
  return load_problem_from_file("test_prob_13L");
}
inline SDPTestProblem make_test_prob_13Lc() {
  return load_problem_from_file("test_prob_13Lc");
}
inline SDPTestProblem make_test_prob_14G() {
  return load_problem_from_file("test_prob_14G");
}
inline SDPTestProblem make_test_prob_15G() {
  return load_problem_from_file("test_prob_15G");
}
inline SDPTestProblem make_test_prob_16G() {
  return load_problem_from_file("test_prob_16G");
}
inline SDPTestProblem make_test_prob_16Gc() {
  return load_problem_from_file("test_prob_16Gc");
}
inline SDPTestProblem make_test_prob_16L() {
  return load_problem_from_file("test_prob_16L");
}
inline SDPTestProblem make_test_prob_16Lc() {
  return load_problem_from_file("test_prob_16Lc");
}
inline SDPTestProblem make_test_prob_2() {
  return load_problem_from_file("test_prob_2");
}
inline SDPTestProblem make_test_prob_3() {
  return load_problem_from_file("test_prob_3");
}
inline SDPTestProblem make_test_prob_4() {
  return load_problem_from_file("test_prob_4");
}
inline SDPTestProblem make_test_prob_5() {
  return load_problem_from_file("test_prob_5");
}
inline SDPTestProblem make_test_prob_6() {
  return load_problem_from_file("test_prob_6");
}
inline SDPTestProblem make_test_prob_7() {
  return load_problem_from_file("test_prob_7");
}
inline SDPTestProblem make_test_prob_8G() {
  return load_problem_from_file("test_prob_8G");
}
inline SDPTestProblem make_test_prob_8Gc() {
  return load_problem_from_file("test_prob_8Gc");
}
inline SDPTestProblem make_test_prob_8L1() {
  return load_problem_from_file("test_prob_8L1");
}
inline SDPTestProblem make_test_prob_8L1c() {
  return load_problem_from_file("test_prob_8L1c");
}
inline SDPTestProblem make_test_prob_8L2() {
  return load_problem_from_file("test_prob_8L2");
}
inline SDPTestProblem make_test_prob_8L2c() {
  return load_problem_from_file("test_prob_8L2c");
}
inline SDPTestProblem make_test_prob_9() {
  return load_problem_from_file("test_prob_9");
}
inline SDPTestProblem make_test_prob_9G() {
  return load_problem_from_file("test_prob_9G");
}
inline SDPTestProblem make_test_prob_9Gc() {
  return load_problem_from_file("test_prob_9Gc");
}
inline SDPTestProblem make_test_prob_9L() {
  return load_problem_from_file("test_prob_9L");
}
inline SDPTestProblem make_test_prob_9L1() {
  return load_problem_from_file("test_prob_9L1");
}
inline SDPTestProblem make_test_prob_9L1c() {
  return load_problem_from_file("test_prob_9L1c");
}
inline SDPTestProblem make_test_prob_9Lc() {
  return load_problem_from_file("test_prob_9Lc");
}
inline SDPTestProblem make_test_prob_9c() {
  return load_problem_from_file("test_prob_9c");
}

inline std::vector<SDPTestProblem> make_exported_sdp_test_problems() {
  static constexpr std::array<const char*, 43> kProblemNames = {
      "test_prob_1",    "test_prob_10G", "test_prob_10Gc", "test_prob_10L",
      "test_prob_10Lc", "test_prob_11G", "test_prob_11Gc", "test_prob_11L",
      "test_prob_11Lc", "test_prob_12G", "test_prob_12Gc", "test_prob_12L",
      "test_prob_12Lc", "test_prob_13G", "test_prob_13Gc", "test_prob_13L",
      "test_prob_13Lc", "test_prob_14G", "test_prob_15G",  "test_prob_16G",
      "test_prob_16Gc", "test_prob_16L", "test_prob_16Lc", "test_prob_2",
      "test_prob_3",    "test_prob_4",   "test_prob_5",    "test_prob_6",
      "test_prob_7",    "test_prob_8G",  "test_prob_8Gc",  "test_prob_8L1",
      "test_prob_8L1c", "test_prob_8L2", "test_prob_8L2c", "test_prob_9",
      "test_prob_9G",   "test_prob_9Gc", "test_prob_9L",   "test_prob_9L1",
      "test_prob_9L1c", "test_prob_9Lc", "test_prob_9c"};

  std::vector<SDPTestProblem> out;
  out.reserve(kProblemNames.size());
  for (const char* name : kProblemNames) {
    out.push_back(load_problem_from_file(name));
  }
  return out;
}

}  // namespace ExportedSDPProblems
}  // namespace RankTools

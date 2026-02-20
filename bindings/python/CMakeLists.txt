###############################################################################
# Python bindings for AnalyticCenter (pybind11)
###############################################################################

# Fetch pybind11 so we don't require a pip install at configure time
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(sdptools_py
  analytic_center_bindings.cpp
)

target_link_libraries(sdptools_py PRIVATE rank_inflation)

# Put the .so next to the source so `import sdptools` works from this dir.
# BUILD_RPATH: used when importing directly from the build/source tree.
# INSTALL_RPATH: $ORIGIN means "same directory as sdptools.so" — used after
#   pip install, where librank_inflation.so is installed alongside sdptools.so.
set_target_properties(sdptools_py PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_NAME "sdptools"
  BUILD_RPATH "${CMAKE_BINARY_DIR}"
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH FALSE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Install the module into the wheel (used by scikit-build-core)
install(TARGETS sdptools_py DESTINATION .)
# Bundle librank_inflation.so into the wheel so $ORIGIN resolves it
install(TARGETS rank_inflation LIBRARY DESTINATION .)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/sdptools.pyi"
  DESTINATION . OPTIONAL)

###############################################################################
# Auto-generate .pyi stubs with pybind11-stubgen after the module is built
###############################################################################

# Use the same Python interpreter that pybind11 resolved, falling back to
# whatever `python3` is on PATH.  Invoking via `python -m pybind11_stubgen`
# avoids the problem where the stub-gen script lives inside a conda/venv bin
# dir that CMake's find_program cannot see.
if(NOT DEFINED Python3_EXECUTABLE)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

# Verify pybind11_stubgen is actually importable by that interpreter
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import pybind11_stubgen"
  RESULT_VARIABLE _STUBGEN_CHECK
  OUTPUT_QUIET ERROR_QUIET
)

if(_STUBGEN_CHECK EQUAL 0)
  message(STATUS "pybind11-stubgen found (via ${Python3_EXECUTABLE})")
  add_custom_command(
    TARGET sdptools_py
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}:$ENV{PYTHONPATH}"
      ${Python3_EXECUTABLE} -m pybind11_stubgen
        sdptools
        --numpy-array-remove-parameters # remove problematic array dimensions
        -o "${CMAKE_CURRENT_SOURCE_DIR}"
        --ignore-all-errors
    COMMENT "Generating Python stubs for sdptools with pybind11-stubgen"
    VERBATIM
  )
else()
  message(WARNING
    "pybind11_stubgen is not importable by ${Python3_EXECUTABLE} — "
    "stubs will not be generated. "
    "Install it with: pip install pybind11-stubgen"
  )
endif()

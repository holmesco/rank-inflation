cmake_minimum_required(VERSION 3.15)
project(rank_tools VERSION 0.0.0)

# set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type." FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

###############################################################################
# Options
###############################################################################

option(BUILD_TESTS "Build testsuite" ON)
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python bindings" OFF)
option(ENABLE_DEBUG "Print additional debugging information" OFF)
option(ENABLE_TIMING "Print out timing data" OFF)
option(ENABLE_PARALLEL "Enable OpenMP parallelization of hot loops" ON)

###############################################################################
# Dependencies
###############################################################################

include(FetchContent)

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen Version: ${EIGEN3_VERSION_STRING} ${EIGEN3_VERSION}")
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)
message(STATUS "Boost Version: ${Boost_VERSION}")

# ----- MOSEK ------ 
# Define MOSEK paths (Update these to your actual paths)
set(MOSEK_DIR "/opt/mosek/11.1/tools/platform/linux64x86")
set(MOSEK_INCLUDE_DIR "${MOSEK_DIR}/h")
set(MOSEK_BIN_DIR "${MOSEK_DIR}/bin")
# include dirs and add lib
include_directories(${MOSEK_INCLUDE_DIR})
find_library(MOSEK_LIB mosek64 PATHS ${MOSEK_BIN_DIR})
find_library(FUSION_LIB fusion64 PATHS ${MOSEK_BIN_DIR})

if(BUILD_TESTS)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/e4fdb87e76b9fc4b01c54ad81aea19d6e994b994.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()


###############################################################################
# Targets
###############################################################################


# source files
file(GLOB RANK_TOOLS_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(rank_tools SHARED)
target_sources(rank_tools PRIVATE ${RANK_TOOLS_SOURCE_FILES})

# headers:
target_include_directories(rank_tools PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

# link external libraries:
target_link_libraries(rank_tools PRIVATE
    Boost::filesystem 
    Boost::system)
target_link_libraries(rank_tools PUBLIC Eigen3::Eigen)
target_link_libraries(rank_tools PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(rank_tools PUBLIC ${FUSION_LIB} ${MOSEK_LIB})

# debug flag  
if(ENABLE_DEBUG)
        message(STATUS "Enabling debug flag")
        target_compile_definitions(rank_tools PRIVATE DEBUG)
endif()

if(ENABLE_TIMING)
  message(STATUS "Enabling timing flag")
  target_compile_definitions(rank_tools PRIVATE TIMING)
endif()

if(ENABLE_PARALLEL)
  message(STATUS "Enabling OpenMP parallelization of hot loops")
  target_compile_definitions(rank_tools PRIVATE RANKTOOLS_PARALLEL)
endif()

# version
set_target_properties(rank_tools PROPERTIES
  VERSION ${PROJECT_VERSION})
target_compile_definitions(rank_tools PUBLIC RANK_TOOLS_VERSION="${PROJECT_VERSION}")


###############################################################################
# Extras
###############################################################################

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings/python)
endif()

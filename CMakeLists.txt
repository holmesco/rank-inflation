cmake_minimum_required(VERSION 3.10)
project(rank_inflation VERSION 0.0.0)

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


if(BUILD_TESTS)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/e4fdb87e76b9fc4b01c54ad81aea19d6e994b994.zip
  )
endif()

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


###############################################################################
# Targets
###############################################################################


# source files
file(GLOB RANK_INF_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(rank_inflation SHARED)
target_sources(rank_inflation PRIVATE ${RANK_INF_SOURCE_FILES})

# headers:
target_include_directories(rank_inflation PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

# link external libraries:
target_link_libraries(rank_inflation PRIVATE
    Boost::filesystem 
    Boost::system)
target_link_libraries(rank_inflation PUBLIC Eigen3::Eigen)
target_link_libraries(rank_inflation PUBLIC OpenMP::OpenMP_CXX)

# debug flag  
if(DEBUG_FLAG)
        message(STATUS "Enabling debug flag")
        target_compile_definitions(rank_inflation PRIVATE DEBUG)
endif()

# version
set_target_properties(rank_inflation PROPERTIES
  VERSION ${PROJECT_VERSION})
target_compile_definitions(rank_inflation PUBLIC RANK_INF_VERSION="${PROJECT_VERSION}")


###############################################################################
# Extras
###############################################################################

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

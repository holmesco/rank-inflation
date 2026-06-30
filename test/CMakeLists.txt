include(GoogleTest)

file(GLOB TEST_INCLUDE_FILES "${CMAKE_SOURCE_DIR}/test/include")

add_executable(analytic_center_test main.cpp analytic_center_test.cpp)
target_include_directories(analytic_center_test PRIVATE ${TEST_INCLUDE_FILES})
target_link_libraries(analytic_center_test rank_tools gtest pthread Eigen3::Eigen)
gtest_add_tests(TARGET analytic_center_test)

add_executable(max_clique_certifier_test main.cpp max_clique_certifier_test.cpp)
target_include_directories(max_clique_certifier_test PRIVATE ${TEST_INCLUDE_FILES})
target_link_libraries(max_clique_certifier_test rank_tools gtest pthread Eigen3::Eigen)
gtest_add_tests(TARGET max_clique_certifier_test)

# add_executable(rank_reduction_test main.cpp rank_reduction_test.cpp)
# target_include_directories(rank_reduction_test PRIVATE ${TEST_INCLUDE_FILES})
# target_link_libraries(rank_reduction_test rank_tools gtest pthread Eigen3::Eigen)
# gtest_add_tests(TARGET rank_reduction_test)

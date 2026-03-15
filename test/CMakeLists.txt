include(GoogleTest)

file(GLOB TEST_INCLUDE_FILES "${CMAKE_SOURCE_DIR}/test/include")

add_executable(rank_inflation_test main.cpp rank_inflation_test.cpp)
target_include_directories(rank_inflation_test PRIVATE ${TEST_INCLUDE_FILES})
target_link_libraries(rank_inflation_test rank_tools gtest pthread Eigen3::Eigen)
gtest_add_tests(TARGET rank_inflation_test)

add_executable(analytic_center_test main.cpp analytic_center_test.cpp)
target_include_directories(analytic_center_test PRIVATE ${TEST_INCLUDE_FILES})
target_link_libraries(analytic_center_test rank_tools gtest pthread Eigen3::Eigen)
gtest_add_tests(TARGET analytic_center_test)

add_executable(rank_reduction_test main.cpp rank_reduction_test.cpp)
target_include_directories(rank_reduction_test PRIVATE ${TEST_INCLUDE_FILES})
target_link_libraries(rank_reduction_test rank_tools gtest pthread Eigen3::Eigen)
gtest_add_tests(TARGET rank_reduction_test)

add_executable(line_search_cert_test main.cpp line_search_cert_test.cpp)
target_include_directories(line_search_cert_test PRIVATE ${TEST_INCLUDE_FILES})
target_link_libraries(line_search_cert_test rank_tools gtest pthread Eigen3::Eigen)
gtest_add_tests(TARGET line_search_cert_test)
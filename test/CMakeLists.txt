include(GoogleTest)

add_executable(rank_inflation_test main.cpp rank_inflation_test.cpp)
target_link_libraries(rank_inflation_test rank_inflation gtest pthread Eigen3::Eigen)
gtest_add_tests(TARGET rank_inflation_test)
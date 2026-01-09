include(GoogleTest)

add_executable(clipper_tests main.cpp clipperplus_test.cpp)
target_link_libraries(clipper_tests clipperplus gtest pthread Eigen3::Eigen)
gtest_add_tests(TARGET clipper_tests TEST_LIST all_clipper_tests)
set_tests_properties(${all_clipper_tests} PROPERTIES TIMEOUT 10)

add_executable(max_clique_tests main.cpp max_clique_sdp_test.cpp)
target_link_libraries(max_clique_tests clipperplus gtest pthread Eigen3::Eigen)
gtest_add_tests(TARGET max_clique_tests)
cmake_minimum_required(VERSION 3.20)

link_libraries(${PROJECT_NAME}::${PROJECT_NAME})

add_executable(test_std_gp gtest/test_vanilla_gp.cpp obs_gp.cpp cov_fnc.cpp)
add_test(NAME test_std_gp
        COMMAND test_std_gp)

add_executable(test_observe_gp_1d gtest/test_lidar_gp_1d.cpp obs_gp.cpp cov_fnc.cpp)
add_custom_command(TARGET test_observe_gp_1d
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/double ${CMAKE_CURRENT_BINARY_DIR}/double
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/float ${CMAKE_CURRENT_BINARY_DIR}/float)
add_test(NAME test_observe_gp_1d
        COMMAND test_observe_gp_1d)

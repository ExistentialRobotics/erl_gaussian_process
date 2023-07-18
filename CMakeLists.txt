cmake_minimum_required(VERSION 3.20)

project(erl_gaussian_process
        LANGUAGES CXX
        VERSION 0.1.0
        DESCRIPTION "erl_gaussian_process is a C++ implementation of Gaussian Process.")
message(STATUS "Configuring ${PROJECT_NAME} ${PROJECT_VERSION}")

if (NOT COMMAND erl_project_setup)
    find_package(erl_common REQUIRED)
    erl_project_setup()
endif ()
set(${PROJECT_NAME}_CATKIN_COMPONENTS erl_common erl_covariance)
erl_project_setup()
erl_setup_ros(CATKIN_COMPONENTS ${${PROJECT_NAME}_CATKIN_COMPONENTS})
erl_catkin_package(
        INCLUDE_DIRS include
        LIBRARIES ${PROJECT_NAME} py${PROJECT_NAME}
        CATKIN_DEPENDS ${${PROJECT_NAME}_CATKIN_COMPONENTS}
        DEPENDS ${${PROJECT_NAME}_DEPENDS})

#######################################################################################################################
# BUILD
#######################################################################################################################

########################
# define the libraries #
########################
file(GLOB_RECURSE SRC_FILES src/*.cpp)
add_library(${PROJECT_NAME} ${SRC_FILES})
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${${PROJECT_NAME}_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${${PROJECT_NAME}_INSTALL_INCLUDE_DIR}>)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC erl_common erl_covariance ${catkin_LIBRARIES})

###################
# Python bindings #
###################
erl_add_python_package(
        PYTHON_PKG_DIR ${${PROJECT_NAME}_PYTHON_DIR}/${PROJECT_NAME}
        PYBIND_MODULE_NAME py${PROJECT_NAME}
        DEPENDS_PYTHON_PKGS erl_common erl_covariance)

#######################################################################################################################
# INSTALL
#######################################################################################################################
erl_install(
        LIBRARIES ${PROJECT_NAME}
        PYBIND_MODULES py${PROJECT_NAME})

#######################################################################################################################
# TEST
#######################################################################################################################
erl_add_tests(
        LIBRARIES ${PROJECT_NAME}
)

#######################################################################################################################
# export variables
#######################################################################################################################
erl_mark_project_found()

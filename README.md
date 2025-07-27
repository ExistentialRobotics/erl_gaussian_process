# erl_gaussian_process

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ROS1](https://img.shields.io/badge/ROS1-noetic-blue)](http://wiki.ros.org/)
[![ROS2](https://img.shields.io/badge/ROS2-humble-blue)](https://docs.ros.org/)

**`erl_gaussian_process` is a C++ library for Gaussian process regression and applications of Gaussian processes.**

## Available Functionalities

- [Vanilla Gaussian Process](include/erl_gaussian_process/vanilla_gp.hpp) - Standard Gaussian process regression
- [Lidar Gaussian Process 2D](include/erl_gaussian_process/lidar_gp_2d.hpp) - Multi-partition Gaussian process regression for 2D lidar data
- [Noisy Input Gaussian Process](include/erl_gaussian_process/noisy_input_gp.hpp) - Gaussian process regression with noisy input
- [Range Sensor Gaussian Process 3D](include/erl_gaussian_process/range_sensor_gp_3d.hpp) - 3D Gaussian process for range sensor data
- [Sparse Pseudo Input Gaussian Process](include/erl_gaussian_process/sparse_pseudo_input_gp.hpp) - Sparse GP using pseudo inputs
- [Batch GP Update Torch](include/erl_gaussian_process/batch_gp_update_torch.hpp) - Batch GP updates using PyTorch backend
- [SPGP Occupancy Map](include/erl_gaussian_process/spgp_occupancy_map.hpp) - Sparse GP for occupancy mapping
- [Mapping](include/erl_gaussian_process/mapping.hpp) - General mapping utilities to transform input and output data

## Getting Started

### Prerequisites

- C++17 compatible compiler
- CMake 3.24 or higher

### Create Workspace

```bash
cd <your_workspace>
mkdir -p src
vcs import --input https://raw.githubusercontent.com/ExistentialRobotics/erl_gaussian_process/refs/head/main/erl_gaussian_process.repos src
```

### Dependencies

- [erl_cmake_tools](https://github.com/ExistentialRobotics/erl_cmake_tools)
- [erl_common](https://github.com/ExistentialRobotics/erl_common)
- [erl_covariance](https://github.com/ExistentialRobotics/erl_covariance)
- [erl_geometry](https://github.com/ExistentialRobotics/erl_geometry)

```bash
# Ubuntu 20.04
wget -qO - https://raw.githubusercontent.com/ExistentialRobotics/erl_common/refs/heads/main/scripts/setup_ubuntu_20.04.bash | bash
wget -qO - https://raw.githubusercontent.com/ExistentialRobotics/erl_geometry/refs/heads/main/scripts/setup_ubuntu_20.04.bash | bash
# Ubuntu 22.04, 24.04
wget -qO - https://raw.githubusercontent.com/ExistentialRobotics/erl_common/refs/heads/main/scripts/setup_ubuntu_22.04_24.04.bash | bash
wget -qO - https://raw.githubusercontent.com/ExistentialRobotics/erl_geometry/refs/heads/main/scripts/setup_ubuntu_22.04_24.04.bash | bash
```

### Docker Option

The easiest way to get started is to use the provided [Docker files](https://github.com/ExistentialRobotics/erl_geometry/tree/main/docker), which contains all dependencies.

### Use as a standard CMake package

```bash
cd <your_workspace>
touch CMakeLists.txt
```

Add the following lines to your `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.24)
project(<your_project_name>)
add_subdirectory(src/erl_cmake_tools)
add_subdirectory(src/erl_common)
add_subdirectory(src/erl_covariance)
add_subdirectory(src/erl_geometry)
add_subdirectory(src/erl_gaussian_process)
```

Then run the following commands:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j`nproc`
```

### Use as a ROS package

```bash
cd <your_workspace>/src
catkin build erl_gaussian_process # for ROS1
colcon build --packages-up-to erl_gaussian_process # for ROS2
```

### Install as a Python package

- Make sure you have installed all dependencies.
- Make sure you have the correct Python environment activated, `pipenv` is recommended.

```bash
cd <your_workspace>
for package in erl_cmake_tools erl_common erl_covariance erl_geometry erl_gaussian_process; do
    cd src/$package
    pip install . --verbose
    cd ../..
done
```

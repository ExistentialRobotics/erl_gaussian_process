#include "erl_common/pybind11.hpp"
#include "erl_gaussian_process/lidar_gp_2d.hpp"
#include "erl_gaussian_process/noisy_input_gp.hpp"

using namespace erl::common;
using namespace erl::gaussian_process;

void
BindVanillaGaussianProcess(const py::module &m);

void
BindMapping(const py::module &m);

void
BindLidarGaussianProcess2D(const py::module &m);

void
BindNoisyInputGaussianProcess(const py::module &m);

void
BindRangeSensorGaussianProcess3D(const py::module &m);

PYBIND11_MODULE(PYBIND_MODULE_NAME, m) {
    m.doc() = "Python 3 Interface of erl_gaussian_process";

    BindVanillaGaussianProcess(m);
    BindMapping(m);
    BindLidarGaussianProcess2D(m);
    BindNoisyInputGaussianProcess(m);
    BindRangeSensorGaussianProcess3D(m);
}

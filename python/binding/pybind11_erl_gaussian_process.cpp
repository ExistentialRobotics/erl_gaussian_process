#include "erl_common/pybind11.hpp"

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

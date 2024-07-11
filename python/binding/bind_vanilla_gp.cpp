#include "erl_common/pybind11.hpp"
#include "erl_gaussian_process/vanilla_gp.hpp"

using namespace erl::common;
using namespace erl::gaussian_process;

void
BindVanillaGaussianProcess(const py::module &m) {
    auto py_vanilla_gp = py::class_<VanillaGaussianProcess>(m, "VanillaGaussianProcess");

    py::class_<VanillaGaussianProcess::Setting, YamlableBase, std::shared_ptr<VanillaGaussianProcess::Setting>>(py_vanilla_gp, "Setting")
        .def(py::init<>())
        .def_readwrite("kernel_type", &VanillaGaussianProcess::Setting::kernel_type)
        .def_readwrite("kernel", &VanillaGaussianProcess::Setting::kernel)
        .def_readwrite("max_num_samples", &VanillaGaussianProcess::Setting::max_num_samples)
        .def_readwrite("auto_normalize", &VanillaGaussianProcess::Setting::auto_normalize);

    py_vanilla_gp.def(py::init<std::shared_ptr<VanillaGaussianProcess::Setting>>(), py::arg("setting").none(false))
        .def_property_readonly("is_trained", &VanillaGaussianProcess::IsTrained)
        .def_property_readonly("setting", &VanillaGaussianProcess::GetSetting)
        .def("reset", &VanillaGaussianProcess::Reset)
        .def(
            "train",
            [](VanillaGaussianProcess &self,
               const Eigen::Ref<const Eigen::MatrixXd> &mat_x_train,
               const Eigen::Ref<const Eigen::VectorXd> &vec_y,
               const Eigen::Ref<const Eigen::VectorXd> &vec_var_y) -> bool {
                const long dim = mat_x_train.rows();
                const long n = mat_x_train.cols();
                self.Reset(n, dim);
                self.GetTrainInputSamplesBuffer().topLeftCorner(dim, n) = mat_x_train;
                self.GetTrainOutputSamplesBuffer().head(n) = vec_y;
                self.GetTrainOutputSamplesVarianceBuffer().head(n) = vec_var_y;
                return self.Train(n);
            },
            py::arg("mat_x_train"),
            py::arg("vec_y"),
            py::arg("vec_var_y"))
        .def(
            "test",
            [](const VanillaGaussianProcess &gp, const Eigen::Ref<const Eigen::MatrixXd> &mat_x_test) {
                Eigen::VectorXd vec_f_out, vec_var_out;
                vec_f_out.resize(mat_x_test.cols());
                vec_var_out.resize(mat_x_test.cols());
                bool success = gp.Test(mat_x_test, vec_f_out, vec_var_out);
                return py::make_tuple(success, vec_f_out, vec_var_out);
            },
            py::arg("mat_x_test"));
}

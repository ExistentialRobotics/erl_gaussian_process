#include "erl_common/pybind11.hpp"
#include "erl_gaussian_process/noisy_input_gp.hpp"

using namespace erl::common;
using namespace erl::gaussian_process;

template<typename Dtype>
void
BindNoisyInputGaussianProcessImpl(const py::module &m, const char *name) {
    using T = NoisyInputGaussianProcess<Dtype>;
    using MatrixX = Eigen::MatrixX<Dtype>;
    using VectorX = Eigen::VectorX<Dtype>;

    auto py_noisy_input_gp = py::class_<T, std::shared_ptr<T>>(m, name);

    py::class_<typename T::Setting, YamlableBase, std::shared_ptr<typename T::Setting>>(py_noisy_input_gp, "Setting")
        .def(py::init<>())
        .def_readwrite("kernel_type", &T::Setting::kernel_type)
        .def_readwrite("kernel_setting_type", &T::Setting::kernel_setting_type)
        .def_readwrite("kernel", &T::Setting::kernel)
        .def_readwrite("max_num_samples", &T::Setting::max_num_samples)
        .def_readwrite("no_gradient_observation", &T::Setting::no_gradient_observation);

    py_noisy_input_gp.def(py::init<std::shared_ptr<typename T::Setting>>(), py::arg("setting").none(false))
        .def_property_readonly("is_trained", &T::IsTrained)
        .def_property_readonly("setting", &T::template GetSetting<typename T::Setting>)
        .def("reset", &T::Reset, py::arg("max_num_samples"), py::arg("x_dim"))
        .def_property_readonly("num_train_samples", &T::GetNumTrainSamples)
        .def_property_readonly("num_train_samples_with_grad", &T::GetNumTrainSamplesWithGrad)
        .def_property_readonly("kernel", &T::GetKernel)
        .def_property_readonly("x_train", &T::GetTrainInputSamplesBuffer)
        .def_property_readonly("y_train", &T::GetTrainOutputSamplesBuffer)
        .def_property_readonly("grad_train", &T::GetTrainOutputGradientSamplesBuffer)
        .def_property_readonly("grad_flag", &T::GetTrainGradientFlagsBuffer)
        .def_property_readonly("var_x_train", &T::GetTrainInputSamplesVarianceBuffer)
        .def_property_readonly("var_y_train", &T::GetTrainOutputValueSamplesVarianceBuffer)
        .def_property_readonly("var_grad_train", &T::GetTrainOutputGradientSamplesVarianceBuffer)
        .def_property_readonly("k_train", &T::GetKtrain)
        .def_property_readonly("alpha", &T::GetAlpha)
        .def_property_readonly("cholesky_k_train", &T::GetCholeskyDecomposition)
        .def_property_readonly("memory_usage", &T::GetMemoryUsage)
        .def(
            "train",
            [](T &self,
               const Eigen::Ref<const MatrixX> &mat_x_train,
               const Eigen::Ref<const Eigen::VectorXl> &vec_grad_flag,
               const Eigen::Ref<const VectorX> &vec_y,
               const Eigen::Ref<const VectorX> &vec_var_x,
               const Eigen::Ref<const VectorX> &vec_var_y,
               const Eigen::Ref<const VectorX> &vec_var_grad) {
                const long num_train_samples = mat_x_train.cols();
                const long x_dim = mat_x_train.rows();
                self.Reset(num_train_samples, x_dim);
                self.GetTrainInputSamplesBuffer().topLeftCorner(x_dim, num_train_samples) = mat_x_train;
                self.GetTrainGradientFlagsBuffer().head(num_train_samples) = vec_grad_flag;
                self.GetTrainOutputValueSamplesVarianceBuffer().head(vec_y.size()) = vec_y;
                self.GetTrainInputSamplesVarianceBuffer().head(num_train_samples) = vec_var_x;
                self.GetTrainOutputValueSamplesVarianceBuffer().head(num_train_samples) = vec_var_y;
                self.GetTrainOutputGradientSamplesVarianceBuffer().head(num_train_samples) = vec_var_grad;
                self.Train(num_train_samples);
            },
            py::arg("mat_x_train"),
            py::arg("vec_grad_flag"),
            py::arg("vec_y"),
            py::arg("vec_var_x"),
            py::arg("vec_var_y"),
            py::arg("vec_var_grad"))
        .def(
            "test",
            [](const T &self, const Eigen::Ref<const MatrixX> &mat_x_test) {
                MatrixX mat_f_out, mat_var_out, mat_cov_out;
                const long &dim = mat_x_test.rows();
                const long &n = mat_x_test.cols();
                mat_f_out.resize(dim + 1, n);
                mat_var_out.resize(dim + 1, n);
                mat_cov_out.resize(dim * (dim + 1) / 2, n);
                self.Test(mat_x_test, mat_f_out, mat_var_out, mat_cov_out);
                return py::make_tuple(mat_f_out, mat_var_out, mat_cov_out);
            },
            py::arg("mat_x_test"));
}

void
BindNoisyInputGaussianProcess(const py::module &m) {
    BindNoisyInputGaussianProcessImpl<double>(m, "NoisyInputGaussianProcessD");
    BindNoisyInputGaussianProcessImpl<float>(m, "NoisyInputGaussianProcessF");
}

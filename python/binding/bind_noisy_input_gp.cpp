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

    py::class_<typename T::Setting, YamlableBase, std::shared_ptr<typename T::Setting>>(
        py_noisy_input_gp,
        "Setting")
        .def(py::init<>())
        .def_readwrite("kernel_type", &T::Setting::kernel_type)
        .def_readwrite("kernel_setting_type", &T::Setting::kernel_setting_type)
        .def_readwrite("kernel", &T::Setting::kernel)
        .def_readwrite("max_num_samples", &T::Setting::max_num_samples)
        .def_readwrite("no_gradient_observation", &T::Setting::no_gradient_observation);

    py::class_<typename T::TestResult, std::shared_ptr<typename T::TestResult>>(
        py_noisy_input_gp,
        "TestResult")
        .def_property_readonly("num_test", &T::TestResult::GetNumTest)
        .def_property_readonly("k_test", &T::TestResult::GetKtest)
        .def(
            "get_mean",
            [](const typename T::TestResult &self, const long y_index, const bool parallel) {
                VectorX vec_f_out(self.GetNumTest());
                self.GetMean(y_index, vec_f_out, parallel);
                return vec_f_out;
            },
            py::arg("y_index"),
            py::arg("parallel"))
        .def(
            "get_mean",
            [](const typename T::TestResult &self, const long index, const long y_index) {
                Dtype f;
                self.GetMean(index, y_index, f);
                return f;
            },
            py::arg("index"),
            py::arg("y_index"))
        .def(
            "get_gradient",
            [](const typename T::TestResult &self, const long y_index, const bool parallel) {
                MatrixX mat_grad_out(self.GetDimX(), self.GetNumTest());
                self.GetGradient(y_index, mat_grad_out, parallel);
                return mat_grad_out;
            },
            py::arg("y_index"),
            py::arg("parallel"))
        .def(
            "get_gradient",
            [](const typename T::TestResult &self, const long index, const long y_index) {
                VectorX vec_grad_out(self.GetDimX());
                self.GetGradient(index, y_index, vec_grad_out.data());
                return vec_grad_out;
            },
            py::arg("index"),
            py::arg("y_index"))
        .def(
            "get_mean_variance",
            [](const typename T::TestResult &self, const bool parallel) {
                VectorX vec_var_out(self.GetNumTest());
                self.GetMeanVariance(vec_var_out, parallel);
                return vec_var_out;
            },
            py::arg("parallel"))
        .def(
            "get_mean_variance",
            [](const typename T::TestResult &self, const long index) {
                Dtype var;
                self.GetMeanVariance(index, var);
                return var;
            },
            py::arg("index"))
        .def(
            "get_gradient_variance",
            [](const typename T::TestResult &self, const bool parallel) {
                MatrixX mat_var_out(self.GetDimX(), self.GetNumTest());
                self.GetGradientVariance(mat_var_out, parallel);
                return mat_var_out;
            },
            py::arg("parallel"))
        .def(
            "get_gradient_variance",
            [](const typename T::TestResult &self, const long index) {
                VectorX vec_var_out(self.GetDimX());
                self.GetGradientVariance(index, vec_var_out.data());
                return vec_var_out;
            },
            py::arg("index"))
        .def(
            "get_covariance",
            [](const typename T::TestResult &self, const bool parallel) {
                long dim = self.GetDimX();
                MatrixX mat_cov_out(dim * (dim + 1) / 2, self.GetNumTest());
                self.GetCovariance(mat_cov_out, parallel);
                return mat_cov_out;
            },
            py::arg("parallel"))
        .def("get_covariance", [](const typename T::TestResult &self, const long index) {
            const long dim = self.GetDimX();
            VectorX vec_cov_out(dim * (dim + 1) / 2);
            self.GetCovariance(index, vec_cov_out.data());
            return vec_cov_out;
        });

    py::class_<typename T::TrainSet>(py_noisy_input_gp, "TrainSet")
        .def_readwrite("x_dim", &T::TrainSet::x_dim)
        .def_readwrite("y_dim", &T::TrainSet::y_dim)
        .def_readwrite("num_samples", &T::TrainSet::num_samples)
        .def_readwrite("num_samples_with_grad", &T::TrainSet::num_samples_with_grad)
        .def_readwrite("x", &T::TrainSet::x)
        .def_readwrite("y", &T::TrainSet::y)
        .def_readwrite("grad", &T::TrainSet::grad)
        .def_readwrite("var_x", &T::TrainSet::var_x)
        .def_readwrite("var_y", &T::TrainSet::var_y)
        .def_readwrite("var_grad", &T::TrainSet::var_grad)
        .def_readwrite("grad_flag", &T::TrainSet::grad_flag);

    py_noisy_input_gp
        .def(py::init<std::shared_ptr<typename T::Setting>>(), py::arg("setting").none(false))
        .def_property_readonly("setting", &T::template GetSetting<typename T::Setting>)
        .def_property_readonly("is_trained", &T::IsTrained)
        .def_property_readonly("using_reduced_rank_kernel", &T::UsingReducedRankKernel)
        .def_property("kernel_origin", &T::GetKernelCoordOrigin, &T::SetKernelCoordOrigin)
        .def("reset", &T::Reset, py::arg("max_num_samples"), py::arg("x_dim"), py::arg("y_dim"))
        .def_property_readonly("kernel", &T::GetKernel)
        .def_property_readonly("train_set", [](const T &self) { return self.GetTrainSet(); })
        .def_property_readonly("k_train", &T::GetKtrain)
        .def_property_readonly("alpha", &T::GetAlpha)
        .def_property_readonly("cholesky_k_train", &T::GetCholeskyDecomposition)
        .def_property_readonly("memory_usage", &T::GetMemoryUsage)
        .def("update_ktrain", &T::UpdateKtrain)
        .def(
            "train",
            [](T &self,
               const Eigen::Ref<const MatrixX> &mat_x_train,
               const Eigen::Ref<const MatrixX> &mat_y_train,
               const Eigen::Ref<const MatrixX> &mat_grad_train,
               const Eigen::Ref<const Eigen::VectorXl> &vec_grad_flag,
               const Eigen::Ref<const VectorX> &vec_var_x,
               const Eigen::Ref<const VectorX> &vec_var_y,
               const Eigen::Ref<const VectorX> &vec_var_grad) {
                const long x_dim = mat_x_train.rows();
                const long y_dim = mat_y_train.cols();
                const long num_train_samples = mat_x_train.cols();
                self.Reset(num_train_samples, x_dim, y_dim);
                typename T::TrainSet &train_set = self.GetTrainSet();
                train_set.x.topLeftCorner(x_dim, num_train_samples) = mat_x_train;
                train_set.y.topLeftCorner(num_train_samples, y_dim) = mat_y_train;
                train_set.grad.topLeftCorner(x_dim * y_dim, num_train_samples) = mat_grad_train;
                train_set.var_x.head(num_train_samples) = vec_var_x;
                train_set.var_y.head(num_train_samples) = vec_var_y;
                train_set.var_grad.head(num_train_samples) = vec_var_grad;
                train_set.grad_flag.head(num_train_samples) = vec_grad_flag;
                train_set.x_dim = x_dim;
                train_set.y_dim = y_dim;
                train_set.num_samples = num_train_samples;
                train_set.num_samples_with_grad = vec_grad_flag.count();
                return self.Train();
            },
            py::arg("mat_x_train"),
            py::arg("mat_y_train"),
            py::arg("mat_grad_train"),
            py::arg("vec_grad_flag"),
            py::arg("vec_var_x"),
            py::arg("vec_var_y"),
            py::arg("vec_var_grad"))
        .def("test", &T::Test, py::arg("mat_x_test"), py::arg("predict_gradient"));
}

void
BindNoisyInputGaussianProcess(const py::module &m) {
    BindNoisyInputGaussianProcessImpl<double>(m, "NoisyInputGaussianProcessD");
    BindNoisyInputGaussianProcessImpl<float>(m, "NoisyInputGaussianProcessF");
}

#include "erl_common/pybind11.hpp"
#include "erl_gaussian_process/vanilla_gp.hpp"

using namespace erl::common;
using namespace erl::gaussian_process;

template<typename Dtype>
void
BindVanillaGaussianProcessImpl(const py::module &m, const char *name) {
    using T = VanillaGaussianProcess<Dtype>;
    using MatrixX = Eigen::MatrixX<Dtype>;
    using VectorX = Eigen::VectorX<Dtype>;
    auto py_vanilla_gp = py::class_<T>(m, name);

    py::class_<typename T::Setting, YamlableBase, std::shared_ptr<typename T::Setting>>(
        py_vanilla_gp,
        "Setting")
        .def(py::init<>())
        .def_readwrite("kernel_type", &T::Setting::kernel_type)
        .def_readwrite("kernel_setting_type", &T::Setting::kernel_setting_type)
        .def_readwrite("kernel", &T::Setting::kernel)
        .def_readwrite("max_num_samples", &T::Setting::max_num_samples);

    py::class_<typename T::TestResult, std::shared_ptr<typename T::TestResult>>(
        py_vanilla_gp,
        "TestResult")
        .def_property_readonly("num_test", &T::TestResult::GetNumTest)
        .def_property_readonly("k_test", &T::TestResult::GetKtest)
        .def(
            "get_mean",
            [](const typename T::TestResult &self, long y_index, bool parallel) {
                VectorX vec_f_out(self.GetNumTest());
                self.GetMean(y_index, vec_f_out, parallel);
                return vec_f_out;
            },
            py::arg("y_index"),
            py::arg("parallel"))
        .def(
            "get_mean",
            [](const typename T::TestResult &self, long index, long y_index) {
                Dtype f;
                self.GetMean(index, y_index, f);
                return f;
            },
            py::arg("index"),
            py::arg("y_index"))
        .def(
            "get_variance",
            [](const typename T::TestResult &self, bool parallel) {
                VectorX vec_var_out(self.GetNumTest());
                self.GetVariance(vec_var_out, parallel);
                return vec_var_out;
            },
            py::arg("parallel"))
        .def(
            "get_variance",
            [](const typename T::TestResult &self, long index) {
                Dtype var;
                self.GetVariance(index, var);
                return var;
            },
            py::arg("index"));

    py::class_<typename T::TrainSet>(py_vanilla_gp, "TrainSet")
        .def_readwrite("x_dim", &T::TrainSet::x_dim)
        .def_readwrite("y_dim", &T::TrainSet::y_dim)
        .def_readwrite("num_samples", &T::TrainSet::num_samples)
        .def_readwrite("x", &T::TrainSet::x)
        .def_readwrite("y", &T::TrainSet::y)
        .def_readwrite("var", &T::TrainSet::var);

    py_vanilla_gp
        .def(py::init<std::shared_ptr<typename T::Setting>>(), py::arg("setting").none(false))
        .def_property_readonly("is_trained", &T::IsTrained)
        .def_property_readonly("setting", &T::GetSetting)
        .def("reset", &T::Reset)
        .def(
            "train",
            [](T &self,
               const Eigen::Ref<const MatrixX> &mat_x_train,
               const Eigen::Ref<const MatrixX> &mat_y_train,
               const Eigen::Ref<const VectorX> &vec_var_y) -> bool {
                const long n = mat_x_train.cols();
                const long x_dim = mat_x_train.rows();
                const long y_dim = mat_y_train.cols();
                self.Reset(n, x_dim, y_dim);
                auto &train_set = self.GetTrainSet();
                train_set.x.topLeftCorner(x_dim, n) = mat_x_train;
                train_set.y.topLeftCorner(n, y_dim) = mat_y_train;
                train_set.var.head(n) = vec_var_y;
                train_set.x_dim = x_dim;
                train_set.y_dim = y_dim;
                train_set.num_samples = n;
                return self.Train();
            },
            py::arg("mat_x_train"),
            py::arg("mat_y_train"),
            py::arg("vec_var_y"))
        .def("test", &T::Test, py::arg("mat_x_test"));
}

void
BindVanillaGaussianProcess(const py::module &m) {
    BindVanillaGaussianProcessImpl<double>(m, "VanillaGaussianProcessD");
    BindVanillaGaussianProcessImpl<float>(m, "VanillaGaussianProcessF");
}

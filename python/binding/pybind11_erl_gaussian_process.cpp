#include "erl_common/pybind11.hpp"
#include "erl_gaussian_process/lidar_gp_1d.hpp"
#include "erl_gaussian_process/mapping.hpp"
#include "erl_gaussian_process/vanilla_gp.hpp"
#include "erl_gaussian_process/noisy_input_gp.hpp"

using namespace erl::common;
using namespace erl::gaussian_process;

void
BindVanillaGaussianProcess(py::module &m) {
    auto py_vanilla_gp = py::class_<VanillaGaussianProcess>(m, "VanillaGaussianProcess");

    py::class_<VanillaGaussianProcess::Setting, YamlableBase, std::shared_ptr<VanillaGaussianProcess::Setting>>(py_vanilla_gp, "Setting")
        .def(py::init<>())
        .def_property_readonly("kernel", [](const VanillaGaussianProcess::Setting &setting) { return setting.kernel; })
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
               const Eigen::Ref<const Eigen::VectorXd> &vec_var_y) {
                long dim = mat_x_train.rows();
                long n = mat_x_train.cols();
                self.Reset(n, dim);
                self.GetTrainInputSamplesBuffer().topLeftCorner(dim, n) = mat_x_train;
                self.GetTrainOutputSamplesBuffer().head(n) = vec_y;
                self.GetTrainOutputSamplesVarianceBuffer().head(n) = vec_var_y;
                self.Train(n);
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
                gp.Test(mat_x_test, vec_f_out, vec_var_out);
                return py::make_tuple(vec_f_out, vec_var_out);
            },
            py::arg("mat_x_test"));
}

void
BindMapping(py::module &m) {
    auto py_mapping = py::class_<Mapping, std::shared_ptr<Mapping>>(m, ERL_AS_STRING(Mapping));

    py::enum_<Mapping::Type>(py_mapping, "Type", py::arithmetic(), "Type of mapping.")
        .value(Mapping::GetTypeName(Mapping::Type::kIdentity), Mapping::Type::kIdentity)
        .value(Mapping::GetTypeName(Mapping::Type::kInverse), Mapping::Type::kInverse)
        .value(Mapping::GetTypeName(Mapping::Type::kInverseSqrt), Mapping::Type::kInverseSqrt)
        .value(Mapping::GetTypeName(Mapping::Type::kExp), Mapping::Type::kExp)
        .value(Mapping::GetTypeName(Mapping::Type::kTanh), Mapping::Type::kTanh)
        .value(Mapping::GetTypeName(Mapping::Type::kSigmoid), Mapping::Type::kSigmoid)
        .value(Mapping::GetTypeName(Mapping::Type::kUnknown), Mapping::Type::kUnknown)
        .export_values();

    py::class_<Mapping::Setting, std::shared_ptr<Mapping::Setting>>(py_mapping, "Setting")
        .def_readwrite("type", &Mapping::Setting::type)
        .def_readwrite("scale", &Mapping::Setting::scale);

    py_mapping.def(py::init(py::overload_cast<>(&Mapping::Create)))
        .def(py::init(py::overload_cast<std::shared_ptr<Mapping::Setting>>(&Mapping::Create)), py::arg("setting"))
        .def_property_readonly("setting", &Mapping::GetSetting)
        .def_property_readonly("map", [](const std::shared_ptr<Mapping> &mapping) { return mapping->m_map_; })
        .def_property_readonly("inv", [](const std::shared_ptr<Mapping> &mapping) { return mapping->m_inv_; });
}

void
BindLidarGaussianProcess1D(py::module &m) {

    using T = LidarGaussianProcess1D;

    auto py_lidar_gp = py::class_<T, std::shared_ptr<T>>(m, ERL_AS_STRING(LidarGaussianProcess1D));

    // TrainBuffer
    auto py_train_buffer = py::class_<T::TrainBuffer>(py_lidar_gp, "TrainBuffer");

    // TrainBuffer::Setting
    py::class_<T::TrainBuffer::Setting, YamlableBase, std::shared_ptr<T::TrainBuffer::Setting>>(py_train_buffer, "Setting")
        .def(py::init<>())
        .def_readwrite("valid_range_min", &T::TrainBuffer::Setting::valid_range_min)
        .def_readwrite("valid_range_max", &T::TrainBuffer::Setting::valid_range_max)
        .def_readwrite("valid_angle_min", &T::TrainBuffer::Setting::valid_angle_min)
        .def_readwrite("valid_angle_max", &T::TrainBuffer::Setting::valid_angle_max)
        .def_property_readonly("mapping", [](const std::shared_ptr<T::TrainBuffer::Setting> &setting) { return setting->mapping; });

    py_train_buffer.def("__len__", &T::TrainBuffer::Size)
        .def_readwrite("angles", &T::TrainBuffer::vec_angles)
        .def_readwrite("distances", &T::TrainBuffer::vec_ranges)
        .def_readwrite("mapped_distances", &T::TrainBuffer::vec_mapped_distances)
        .def_readwrite("local_directions", &T::TrainBuffer::mat_direction_local)
        .def_readwrite("xy_locals", &T::TrainBuffer::mat_xy_local)
        .def_readwrite("global_directions", &T::TrainBuffer::mat_direction_global)
        .def_readwrite("xy_globals", &T::TrainBuffer::mat_xy_global)
        .def_readwrite("max_distance", &T::TrainBuffer::max_distance)
        .def_readwrite("position", &T::TrainBuffer::position)
        .def_readwrite("rotation", &T::TrainBuffer::rotation);

    // Setting
    py::class_<T::Setting, YamlableBase, std::shared_ptr<T::Setting>>(py_lidar_gp, "Setting")
        .def(py::init<>())
        .def_readwrite("group_size", &T::Setting::group_size)
        .def_readwrite("overlap_size", &T::Setting::overlap_size)
        .def_readwrite("boundary_margin", &T::Setting::boundary_margin)
        .def_readwrite("init_variance", &T::Setting::init_variance)
        .def_readwrite("sensor_range_var", &T::Setting::sensor_range_var)
        .def_readwrite("max_valid_distance_var", &T::Setting::max_valid_distance_var)
        .def_readwrite("occ_test_temperature", &T::Setting::occ_test_temperature)
        .def_readwrite("train_buffer", &T::Setting::train_buffer)
        .def_readwrite("gp", &T::Setting::gp);

    py_lidar_gp.def(py::init<>(&T::Create), py::arg("setting").none(false))
        .def_property_readonly("is_trained", &T::IsTrained)
        .def_property_readonly("setting", &T::GetSetting)
        .def_property_readonly("train_buffer", &T::GetTrainBuffer)
        .def("global_to_local_so2", &T::GlobalToLocalSo2, py::arg("vec_global"))
        .def("local_to_global_so2", &T::LocalToGlobalSo2, py::arg("vec_local"))
        .def("global_to_local_se2", &T::GlobalToLocalSe2, py::arg("vec_global"))
        .def("local_to_global_se2", &T::LocalToGlobalSe2, py::arg("vec_local"))
        .def("reset", &T::Reset)
        .def("train", &T::Train, py::arg("angles"), py::arg("distances"), py::arg("pose"))
        .def(
            "test",
            [](const T &gp, const Eigen::Ref<const Eigen::VectorXd> &thetas, bool un_map) {
                Eigen::VectorXd fs(thetas.size()), vars(thetas.size());
                gp.Test(thetas, fs, vars, un_map);
                return py::make_tuple(fs, vars);
            },
            py::arg("thetas"),
            py::arg("un_map") = true)
        .def(
            "compute_occ",
            [](const T &gp, double angle, double r) {
                double occ;
                Eigen::Scalard scalar_angle, f, var;
                scalar_angle << angle;
                auto success = gp.ComputeOcc(scalar_angle, r, f, var, occ);
                return py::make_tuple(success, f[0], var[0], occ);
            },
            py::arg("angle"),
            py::arg("r"));
}

void
BindNoisyInputGaussianProcess(py::module &m) {
    using T = NoisyInputGaussianProcess;

    auto py_noisy_input_gp = py::class_<T, std::shared_ptr<T>>(m, ERL_AS_STRING(NoisyInputGaussianProcess));

    py::class_<T::Setting, YamlableBase, std::shared_ptr<T::Setting>>(py_noisy_input_gp, "Setting")
        .def(py::init<>())
        .def_readwrite("kernel", &T::Setting::kernel);

    py_noisy_input_gp.def(py::init<>([]() { return std::make_shared<T>(); }))
        .def(py::init<>([](std::shared_ptr<T::Setting> setting) { return std::make_shared<T>(std::move(setting)); }), py::arg("setting").none(false))
        .def_property_readonly("is_trained", &T::IsTrained)
        .def_property_readonly("setting", &T::GetSetting)
        .def("reset", &T::Reset)
        .def(
            "train",
            [](T &self,
               const Eigen::Ref<const Eigen::MatrixXd> &mat_x_train,
               const Eigen::Ref<const Eigen::VectorXb> &vec_grad_flag,
               const Eigen::Ref<const Eigen::VectorXd> &vec_y,
               const Eigen::Ref<const Eigen::VectorXd> &vec_var_x,
               const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
               const Eigen::Ref<const Eigen::VectorXd> &vec_var_grad) {
                long num_train_samples = mat_x_train.cols();
                long x_dim = mat_x_train.rows();
                self.Reset(num_train_samples, x_dim);
                self.GetTrainInputSamplesBuffer().topLeftCorner(x_dim, num_train_samples) = mat_x_train;
                self.GetTrainGradientFlagsBuffer().head(num_train_samples) = vec_grad_flag;
                self.GetTrainOutputValueSamplesVarianceBuffer().head(vec_y.size()) = vec_y;
                self.GetTrainInputSamplesVarianceBuffer().head(num_train_samples) = vec_var_x;
                self.GetTrainOutputValueSamplesVarianceBuffer().head(num_train_samples) = vec_var_y;
                self.GetTrainOutputGradientSamplesVarianceBuffer().head(num_train_samples) = vec_var_grad;
                self.Train(num_train_samples, vec_grad_flag.head(num_train_samples).cast<long>().sum());
            },
            py::arg("mat_x_train"),
            py::arg("vec_grad_flag"),
            py::arg("vec_y"),
            py::arg("vec_var_x"),
            py::arg("vec_var_y"),
            py::arg("vec_var_grad"))
        .def(
            "test",
            [](const T &self, const Eigen::Ref<const Eigen::MatrixXd> &mat_x_test) {
                Eigen::MatrixXd mat_f_out, mat_var_out, mat_cov_out;
                long dim = mat_x_test.rows();
                long n = mat_x_test.cols();
                mat_f_out.resize(dim + 1, n);
                mat_var_out.resize(dim + 1, n);
                mat_cov_out.resize(dim * (dim + 1) / 2, n);
                self.Test(mat_x_test, mat_f_out, mat_var_out, mat_cov_out);
                return py::make_tuple(mat_f_out, mat_var_out, mat_cov_out);
            },
            py::arg("mat_x_test"));
}

PYBIND11_MODULE(PYBIND_MODULE_NAME, m) {
    m.doc() = "Python 3 Interface of erl_gaussian_process";

    BindVanillaGaussianProcess(m);
    BindMapping(m);
    BindLidarGaussianProcess1D(m);
    BindNoisyInputGaussianProcess(m);
}

#include "erl_common/pybind11.hpp"
#include "erl_gaussian_process/lidar_gp_2d.hpp"

using namespace erl::common;
using namespace erl::gaussian_process;

template<typename Dtype>
void
BindLidarGaussianProcess2DImpl(const py::module &m, const char *name) {

    using T = LidarGaussianProcess2D<Dtype>;

    auto py_lidar_gp = py::class_<T, std::shared_ptr<T>>(m, name);

    // Setting
    py::class_<typename T::Setting, YamlableBase, std::shared_ptr<typename T::Setting>>(py_lidar_gp, "Setting")
        .def(py::init<>())
        .def_readwrite("group_size", &T::Setting::group_size)
        .def_readwrite("overlap_size", &T::Setting::overlap_size)
        .def_readwrite("margin", &T::Setting::margin)
        .def_readwrite("init_variance", &T::Setting::init_variance)
        .def_readwrite("sensor_range_var", &T::Setting::sensor_range_var)
        .def_readwrite("max_valid_range_var", &T::Setting::max_valid_range_var)
        .def_readwrite("occ_test_temperature", &T::Setting::occ_test_temperature)
        .def_readwrite("lidar_frame", &T::Setting::lidar_frame)
        .def_readwrite("gp", &T::Setting::gp)
        .def_readwrite("mapping", &T::Setting::mapping);

    py_lidar_gp.def(py::init<std::shared_ptr<typename T::Setting>>(), py::arg("setting").none(false))
        .def_property_readonly("is_trained", &T::IsTrained)
        .def_property_readonly("setting", &T::GetSetting)
        .def_property_readonly("gps", &T::GetGps)
        .def_property_readonly("angle_partitions", &T::GetAnglePartitions)
        .def_property_readonly("sensor_frame", &T::GetSensorFrame)
        .def("global_to_local_so2", &T::GlobalToLocalSo2, py::arg("dir_global"))
        .def("local_to_global_so2", &T::LocalToGlobalSo2, py::arg("dir_local"))
        .def("global_to_local_se2", &T::GlobalToLocalSe2, py::arg("xy_global"))
        .def("local_to_global_se2", &T::LocalToGlobalSe2, py::arg("xy_local"))
        .def("reset", &T::Reset)
        .def("store_data", &T::StoreData, py::arg("rotation"), py::arg("translation"), py::arg("ranges"))
        .def("train", &T::Train, py::arg("rotation"), py::arg("translation"), py::arg("ranges"), py::arg("repartition_on_hit_rays"))
        .def(
            "test",
            [](const T &gp, const Eigen::Ref<const Eigen::VectorX<Dtype>> &angles, const bool angles_are_local, const bool un_map) {
                Eigen::VectorX<Dtype> fs(angles.size()), vars(angles.size());
                bool success = gp.Test(angles, angles_are_local, fs, vars, un_map);
                return py::make_tuple(success, fs, vars);
            },
            py::arg("angles"),
            py::arg("angles_are_local"),
            py::arg("un_map"))
        .def(
            "compute_occ",
            [](const T &gp, const Dtype angle, const Dtype r) {
                Dtype occ;
                Eigen::Scalar<Dtype> scalar_angle, f, var;
                scalar_angle << angle;
                bool success = gp.ComputeOcc(scalar_angle, r, f, var, occ);
                return py::make_tuple(success, f[0], var[0], occ);
            },
            py::arg("angle"),
            py::arg("r"));
}

void
BindLidarGaussianProcess2D(const py::module &m) {
    BindLidarGaussianProcess2DImpl<double>(m, "LidarGaussianProcess2Dd");
    BindLidarGaussianProcess2DImpl<float>(m, "LidarGaussianProcess2Df");
}

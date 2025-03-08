#include "erl_common/pybind11.hpp"
#include "erl_gaussian_process/range_sensor_gp_3d.hpp"

using namespace erl::common;
using namespace erl::gaussian_process;

template<typename Dtype>
void
BindRangeSensorGaussianProcess3DImpl(const py::module &m, const char *name) {

    using T = RangeSensorGaussianProcess3D<Dtype>;

    auto py_range_sensor_gp_3d = py::class_<T, std::shared_ptr<T>>(m, name);

    // Setting
    py::class_<typename T::Setting, YamlableBase, std::shared_ptr<typename T::Setting>>(py_range_sensor_gp_3d, "Setting")
        .def(py::init<>())
        .def_readwrite("row_group_size", &T::Setting::row_group_size)
        .def_readwrite("row_overlap_size", &T::Setting::row_overlap_size)
        .def_readwrite("row_margin", &T::Setting::row_margin)
        .def_readwrite("col_group_size", &T::Setting::col_group_size)
        .def_readwrite("col_overlap_size", &T::Setting::col_overlap_size)
        .def_readwrite("col_margin", &T::Setting::col_margin)
        .def_readwrite("init_variance", &T::Setting::init_variance)
        .def_readwrite("sensor_range_var", &T::Setting::sensor_range_var)
        .def_readwrite("max_valid_range_var", &T::Setting::max_valid_range_var)
        .def_readwrite("occ_test_temperature", &T::Setting::occ_test_temperature)
        .def_readwrite("sensor_frame_type", &T::Setting::sensor_frame_type)
        .def_readwrite("sensor_frame", &T::Setting::sensor_frame)
        .def_readwrite("gp", &T::Setting::gp)
        .def_readwrite("mapping", &T::Setting::mapping);

    py_range_sensor_gp_3d.def(py::init<std::shared_ptr<typename T::Setting>>(), py::arg("setting").none(false))
        .def_property_readonly("is_trained", &T::IsTrained)
        .def_property_readonly("setting", &T::GetSetting)
        .def_property_readonly(
            "gps",
            [](const T &self) {
                py::list gps;
                auto cpp_gps = self.GetGps();
                for (long i = 0; i < cpp_gps.rows(); ++i) {
                    py::list row;
                    for (long j = 0; j < cpp_gps.cols(); ++j) { row.append(cpp_gps(i, j)); }
                    gps.append(row);
                }
                return gps;
            })
        .def_property_readonly("row_partitions", &T::GetRowPartitions)
        .def_property_readonly("col_partitions", &T::GetColPartitions)
        .def_property_readonly("sensor_frame", &T::GetSensorFrame)
        .def_property_readonly("mapping", &T::GetMapping)
        .def("global_to_local_so3", &T::GlobalToLocalSo3, py::arg("dir_global"))
        .def("local_to_global_so3", &T::LocalToGlobalSo3, py::arg("dir_local"))
        .def("global_to_local_se3", &T::GlobalToLocalSe3, py::arg("xyz_global"))
        .def("local_to_global_se3", &T::LocalToGlobalSe3, py::arg("xyz_local"))
        .def("compute_frame_coords", &T::ComputeFrameCoords, py::arg("xyz_frame"))
        .def("reset", &T::Reset)
        .def("store_data", &T::StoreData, py::arg("rotation"), py::arg("translation"), py::arg("ranges"))
        .def("train", &T::Train, py::arg("rotation"), py::arg("translation"), py::arg("ranges"))
        .def(
            "test",
            [](const T &gp, const Eigen::Ref<const Eigen::Matrix3X<Dtype>> &directions, const bool directions_are_local, const bool un_map) {
                Eigen::VectorX<Dtype> fs(directions.cols()), vars(directions.cols());
                bool success = gp.Test(directions, directions_are_local, fs, vars, un_map);
                return py::make_tuple(success, fs, vars);
            },
            py::arg("directions"),
            py::arg("directions_are_local"),
            py::arg("un_map"));
}

void
BindRangeSensorGaussianProcess3D(const py::module &m) {
    BindRangeSensorGaussianProcess3DImpl<double>(m, "RangeSensorGaussianProcess3Dd");
    BindRangeSensorGaussianProcess3DImpl<float>(m, "RangeSensorGaussianProcess3Df");
}

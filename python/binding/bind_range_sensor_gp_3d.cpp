#include "erl_common/pybind11.hpp"
#include "erl_gaussian_process/range_sensor_gp_3d.hpp"

using namespace erl::common;
using namespace erl::gaussian_process;

template<typename Dtype>
void
BindRangeSensorGaussianProcess3DImpl(const py::module &m, const char *name) {

    using T = RangeSensorGaussianProcess3D<Dtype>;
    using Vector3 = Eigen::Vector3<Dtype>;
    using VectorX = Eigen::VectorX<Dtype>;
    auto py_range_sensor_gp_3d = py::class_<T, std::shared_ptr<T>>(m, name);

    // Setting
    py::class_<typename T::Setting, YamlableBase, std::shared_ptr<typename T::Setting>>(
        py_range_sensor_gp_3d,
        "Setting")
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
        .def_readwrite("sensor_frame_setting_type", &T::Setting::sensor_frame_setting_type)
        .def_readwrite("sensor_frame", &T::Setting::sensor_frame)
        .def_readwrite("gp", &T::Setting::gp)
        .def_readwrite("mapping", &T::Setting::mapping);

    py::class_<typename T::TestResult, std::shared_ptr<typename T::TestResult>>(
        py_range_sensor_gp_3d,
        "TestResult")
        .def_property_readonly("num_test", &T::TestResult::GetNumTest)
        .def("get_ktest", &T::TestResult::GetKtest, py::arg("index"))
        .def(
            "get_mean",
            [](const typename T::TestResult &self, bool parallel) {
                VectorX vec_f_out(self.GetNumTest());
                Eigen::VectorXb success;
                {
                    py::gil_scoped_release release;
                    success = self.GetMean(vec_f_out, parallel);
                }
                return py::make_tuple(success, vec_f_out);
            },
            py::arg("parallel"))
        .def(
            "get_mean",
            [](const typename T::TestResult &self, long index) {
                Dtype f;
                bool success = self.GetMean(index, f);
                return py::make_tuple(success, f);
            })
        .def(
            "get_variance",
            [](const typename T::TestResult &self, bool parallel) {
                VectorX vec_var_out(self.GetNumTest());
                Eigen::VectorXb success;
                {
                    py::gil_scoped_release release;
                    success = self.GetVariance(vec_var_out, parallel);
                }
                return py::make_tuple(success, vec_var_out);
            },
            py::arg("parallel"))
        .def("get_variance", [](const typename T::TestResult &self, long index) {
            Dtype var;
            bool success = self.GetVariance(index, var);
            return py::make_tuple(success, var);
        });

    py_range_sensor_gp_3d
        .def(py::init<std::shared_ptr<typename T::Setting>>(), py::arg("setting").none(false))
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
        .def("reset", &T::Reset)
        .def(
            "store_data",
            &T::StoreData,
            py::arg("rotation"),
            py::arg("translation"),
            py::arg("ranges"))
        .def("train", &T::Train, py::arg("rotation"), py::arg("translation"), py::arg("ranges"))
        .def("search_partition", &T::SearchPartition, py::arg("frame_coords"))
        .def(
            "test",
            &T::Test,
            py::arg("directions"),
            py::arg("directions_are_local"),
            py::arg("un_map"))
        .def(
            "compute_occ",
            [](const T &gp, const Vector3 &pos_local) {
                Dtype range_pred, occ, dist_pos;
                bool success = gp.ComputeOcc(pos_local, dist_pos, range_pred, occ);
                py::dict out;
                out["success"] = success;
                out["dist_pos"] = dist_pos;
                out["range_pred"] = range_pred;
                out["occ"] = occ;
                return out;
            },
            py::arg("pos_local"));
}

void
BindRangeSensorGaussianProcess3D(const py::module &m) {
    BindRangeSensorGaussianProcess3DImpl<double>(m, "RangeSensorGaussianProcess3Dd");
    BindRangeSensorGaussianProcess3DImpl<float>(m, "RangeSensorGaussianProcess3Df");
}

#include "erl_common/pybind11.hpp"
#include "erl_gaussian_process/lidar_gp_2d.hpp"

using namespace erl::common;
using namespace erl::gaussian_process;

template<typename Dtype>
void
BindLidarGaussianProcess2DImpl(const py::module &m, const char *name) {

    using T = LidarGaussianProcess2D<Dtype>;
    using Vector2 = Eigen::Vector2<Dtype>;
    using VectorX = Eigen::VectorX<Dtype>;

    auto py_lidar_gp = py::class_<T, std::shared_ptr<T>>(m, name);

    // Setting
    py::class_<typename T::Setting, YamlableBase, std::shared_ptr<typename T::Setting>>(
        py_lidar_gp,
        "Setting")
        .def(py::init<>())
        .def_readwrite("partition_on_hit_rays", &T::Setting::partition_on_hit_rays)
        .def_readwrite("symmetric_partitions", &T::Setting::symmetric_partitions)
        .def_readwrite("group_size", &T::Setting::group_size)
        .def_readwrite("overlap_size", &T::Setting::overlap_size)
        .def_readwrite("margin", &T::Setting::margin)
        .def_readwrite("init_variance", &T::Setting::init_variance)
        .def_readwrite("sensor_range_var", &T::Setting::sensor_range_var)
        .def_readwrite("max_valid_range_var", &T::Setting::max_valid_range_var)
        .def_readwrite("occ_test_temperature", &T::Setting::occ_test_temperature)
        .def_readwrite("sensor_frame", &T::Setting::sensor_frame)
        .def_readwrite("gp", &T::Setting::gp)
        .def_readwrite("mapping", &T::Setting::mapping);

    py::class_<typename T::TestResult, std::shared_ptr<typename T::TestResult>>(
        py_lidar_gp,
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

    py_lidar_gp
        .def(py::init<std::shared_ptr<typename T::Setting>>(), py::arg("setting").none(false))
        .def_property_readonly("is_trained", &T::IsTrained)
        .def_property_readonly("setting", &T::GetSetting)
        .def_property_readonly("gps", &T::GetGps)
        .def_property_readonly("angle_partitions", &T::GetAnglePartitions)
        .def_property_readonly("sensor_frame", &T::GetSensorFrame)
        .def("reset", &T::Reset)
        .def(
            "store_data",
            &T::StoreData,
            py::arg("rotation"),
            py::arg("translation"),
            py::arg("ranges"))
        .def("partition_on_angles", &T::PartitionOnAngles)
        .def("partition_on_hit_rays", &T::PartitionOnHitRays)
        .def("train", &T::Train, py::arg("rotation"), py::arg("translation"), py::arg("ranges"))
        .def("search_partition", &T::SearchPartition, py::arg("angle_local"))
        .def("test", &T::Test, py::arg("angles"), py::arg("angles_are_local"), py::arg("un_map"))
        .def(
            "compute_occ",
            [](T &gp, const Vector2 &pos_local) {
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
BindLidarGaussianProcess2D(const py::module &m) {
    BindLidarGaussianProcess2DImpl<double>(m, "LidarGaussianProcess2Dd");
    BindLidarGaussianProcess2DImpl<float>(m, "LidarGaussianProcess2Df");
}

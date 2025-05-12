#include "erl_common/pybind11.hpp"
#include "erl_gaussian_process/lidar_gp_2d.hpp"

using namespace erl::common;
using namespace erl::gaussian_process;

template<typename Dtype>
void
BindLidarGaussianProcess2DImpl(const py::module &m, const char *name) {

    using T = LidarGaussianProcess2D<Dtype>;
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
                Eigen::VectorXb success = self.GetMean(vec_f_out, parallel);
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
                Eigen::VectorXb success = self.GetVariance(vec_var_out, parallel);
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
        .def("global_to_local_so2", &T::GlobalToLocalSo2, py::arg("dir_global"))
        .def("local_to_global_so2", &T::LocalToGlobalSo2, py::arg("dir_local"))
        .def("global_to_local_se2", &T::GlobalToLocalSe2, py::arg("xy_global"))
        .def("local_to_global_se2", &T::LocalToGlobalSe2, py::arg("xy_local"))
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
            [](const T &gp, const Dtype angle_local, const Dtype r) {
                Dtype range_pred, occ;
                Eigen::Scalar<Dtype> scalar_angle;
                scalar_angle << angle_local;
                bool success = gp.ComputeOcc(scalar_angle, r, range_pred, occ);
                return py::make_tuple(success, range_pred, occ);
            },
            py::arg("angle_local"),
            py::arg("r"));
}

void
BindLidarGaussianProcess2D(const py::module &m) {
    BindLidarGaussianProcess2DImpl<double>(m, "LidarGaussianProcess2Dd");
    BindLidarGaussianProcess2DImpl<float>(m, "LidarGaussianProcess2Df");
}

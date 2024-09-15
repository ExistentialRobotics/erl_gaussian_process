#pragma once

#include "mapping.hpp"
#include "vanilla_gp.hpp"

#include "erl_common/eigen.hpp"
#include "erl_common/yaml.hpp"
#include "erl_geometry/depth_frame_3d.hpp"
#include "erl_geometry/lidar_frame_3d.hpp"

#include <memory>

namespace erl::gaussian_process {

    class RangeSensorGaussianProcess3D {

    public:
        struct Setting : public common::Yamlable<Setting> {
            long row_group_size = 24;   // number of points in each group for each row, including the overlap ones
            long row_overlap_size = 6;  // number of points in the overlap region for each row
            long row_margin = 0;
            long col_group_size = 8;    // number of elevation points in each group, including the overlap ones
            long col_overlap_size = 2;  // number of points in the overlap region
            long col_margin = 0;
            double init_variance = 1e6;        // large value to initialize variance result in case of computation failure
            double sensor_range_var = 0.01;    // variance of the sensor range measurement
            double max_valid_range_var = 0.1;  // if the distance variance is greater than this threshold, the prediction is invalid and should be discarded
            double occ_test_temperature = 30;  // OCC test is a tanh function, this controls the slope around 0
            std::string range_sensor_frame_type = "erl::geometry::LidarFrame3D";                   // type of the range sensor frame
            std::string range_sensor_frame_setting_type = "erl::geometry::LidarFrame3D::Setting";  // type of the range sensor frame setting
            std::shared_ptr<geometry::RangeSensorFrame3D::Setting> range_sensor_frame = std::make_shared<geometry::LidarFrame3D::Setting>();
            std::shared_ptr<VanillaGaussianProcess::Setting> gp = std::make_shared<VanillaGaussianProcess::Setting>();
            std::shared_ptr<Mapping::Setting> mapping = []() {
                auto mapping_setting = std::make_shared<Mapping::Setting>();
                mapping_setting->type = Mapping::Type::kInverseSqrt;
                mapping_setting->scale = 1.0;
                return mapping_setting;
            }();
        };

        inline static const volatile bool kSettingRegistered = common::YamlableBase::Register<Setting>();

    protected:
        bool m_trained_ = false;
        std::shared_ptr<Setting> m_setting_ = nullptr;
        Eigen::MatrixX<std::shared_ptr<VanillaGaussianProcess>> m_gps_ = {};
        std::vector<std::tuple<long, long, double, double>> m_row_partitions_ = {};
        std::vector<std::tuple<long, long, double, double>> m_col_partitions_ = {};
        std::shared_ptr<geometry::RangeSensorFrame3D> m_range_sensor_frame_ = nullptr;
        std::shared_ptr<Mapping> m_mapping_ = nullptr;
        Eigen::MatrixXd m_mapped_distances_ = {};

    public:
        explicit RangeSensorGaussianProcess3D(std::shared_ptr<Setting> setting);

        [[nodiscard]] bool
        IsTrained() const {
            return m_trained_;
        }

        [[nodiscard]] std::shared_ptr<Setting>
        GetSetting() const {
            return m_setting_;
        }

        [[nodiscard]] const Eigen::MatrixX<std::shared_ptr<VanillaGaussianProcess>> &
        GetGps() const {
            return m_gps_;
        }

        [[nodiscard]] const std::vector<std::tuple<long, long, double, double>> &
        GetRowPartitions() const {
            return m_row_partitions_;
        }

        [[nodiscard]] const std::vector<std::tuple<long, long, double, double>> &
        GetColPartitions() const {
            return m_col_partitions_;
        }

        [[nodiscard]] std::shared_ptr<const geometry::RangeSensorFrame3D>
        GetRangeSensorFrame() const {
            return m_range_sensor_frame_;
        }

        [[nodiscard]] std::shared_ptr<const Mapping>
        GetMapping() const {
            return m_mapping_;
        }

        [[nodiscard]] Eigen::Vector3d
        GlobalToLocalSo3(const Eigen::Vector3d &dir_global) const {
            return m_range_sensor_frame_->WorldToFrameSo3(dir_global);
        }

        [[nodiscard]] Eigen::Vector3d
        LocalToGlobalSo3(const Eigen::Vector3d &dir_local) const {
            return m_range_sensor_frame_->FrameToWorldSo3(dir_local);
        }

        [[nodiscard]] Eigen::Vector3d
        GlobalToLocalSe3(const Eigen::Vector3d &xyz_global) const {
            return m_range_sensor_frame_->WorldToFrameSe3(xyz_global);
        }

        [[nodiscard]] Eigen::Vector3d
        LocalToGlobalSe3(const Eigen::Vector3d &xyz_local) const {
            return m_range_sensor_frame_->FrameToWorldSe3(xyz_local);
        }

        [[nodiscard]] Eigen::Vector2d
        ComputeFrameCoords(const Eigen::Vector3d &xyz_frame) const {
            return m_range_sensor_frame_->ComputeFrameCoords(xyz_frame);
        }

        void
        Reset();

        bool
        StoreData(const Eigen::Matrix3d &rotation, const Eigen::Vector3d &translation, Eigen::MatrixXd ranges);

        [[nodiscard]] bool
        Train(const Eigen::Matrix3d &rotation, const Eigen::Vector3d &translation, Eigen::MatrixXd ranges);

        [[nodiscard]] bool
        Test(
            const Eigen::Ref<const Eigen::Matrix3Xd> &directions,
            bool directions_are_local,
            Eigen::Ref<Eigen::VectorXd> vec_ranges,
            Eigen::Ref<Eigen::VectorXd> vec_ranges_var,
            bool un_map) const;

        bool
        ComputeOcc(const Eigen::Vector3d &dir_local, double r, Eigen::Ref<Eigen::Scalard> range_pred, Eigen::Ref<Eigen::Scalard> range_pred_var, double &occ)
            const;

        [[nodiscard]] bool
        operator==(const RangeSensorGaussianProcess3D &other) const;

        [[nodiscard]] bool
        operator!=(const RangeSensorGaussianProcess3D &other) const {
            return !(*this == other);
        }

        [[nodiscard]] bool
        Write(const std::string &filename) const;

        [[nodiscard]] bool
        Write(std::ostream &s) const;

        [[nodiscard]] bool
        Read(const std::string &filename);

        [[nodiscard]] bool
        Read(std::istream &s);
    };
}  // namespace erl::gaussian_process

// ReSharper disable CppInconsistentNaming
template<>
struct YAML::convert<erl::gaussian_process::RangeSensorGaussianProcess3D::Setting> {
    static Node
    encode(const erl::gaussian_process::RangeSensorGaussianProcess3D::Setting &rhs) {
        Node node;
        node["row_group_size"] = rhs.row_group_size;
        node["row_overlap_size"] = rhs.row_overlap_size;
        node["row_margin"] = rhs.row_margin;
        node["col_group_size"] = rhs.col_group_size;
        node["col_overlap_size"] = rhs.col_overlap_size;
        node["col_margin"] = rhs.col_margin;
        node["init_variance"] = rhs.init_variance;
        node["sensor_range_var"] = rhs.sensor_range_var;
        node["max_valid_range_var"] = rhs.max_valid_range_var;
        node["occ_test_temperature"] = rhs.occ_test_temperature;
        node["range_sensor_frame_type"] = rhs.range_sensor_frame_type;
        if (rhs.range_sensor_frame_type == demangle(typeid(erl::geometry::LidarFrame3D).name())) {
            node["range_sensor_frame"] = std::dynamic_pointer_cast<erl::geometry::LidarFrame3D::Setting>(rhs.range_sensor_frame);
        } else if (rhs.range_sensor_frame_type == demangle(typeid(erl::geometry::DepthFrame3D).name())) {
            node["range_sensor_frame"] = std::dynamic_pointer_cast<erl::geometry::DepthFrame3D::Setting>(rhs.range_sensor_frame);
        } else {
            ERL_FATAL("Unknown range_sensor_frame_type: {}", rhs.range_sensor_frame_type);
        }
        node["row_margin"] = rhs.row_margin;
        node["col_margin"] = rhs.col_margin;
        node["gp"] = rhs.gp;
        node["mapping"] = rhs.mapping;
        return node;
    }

    static bool
    decode(const Node &node, erl::gaussian_process::RangeSensorGaussianProcess3D::Setting &rhs) {
        if (!node.IsMap()) { return false; }
        rhs.row_group_size = node["row_group_size"].as<int>();
        rhs.row_overlap_size = node["row_overlap_size"].as<int>();
        rhs.row_margin = node["row_margin"].as<long>();
        rhs.col_group_size = node["col_group_size"].as<int>();
        rhs.col_overlap_size = node["col_overlap_size"].as<int>();
        rhs.col_margin = node["col_margin"].as<long>();
        rhs.init_variance = node["init_variance"].as<double>();
        rhs.sensor_range_var = node["sensor_range_var"].as<double>();
        rhs.max_valid_range_var = node["max_valid_range_var"].as<double>();
        rhs.occ_test_temperature = node["occ_test_temperature"].as<double>();
        rhs.range_sensor_frame_type = node["range_sensor_frame_type"].as<std::string>();
        rhs.range_sensor_frame_setting_type = node["range_sensor_frame_setting_type"].as<std::string>();
        rhs.range_sensor_frame = erl::common::YamlableBase::Create<erl::geometry::RangeSensorFrame3D::Setting>(rhs.range_sensor_frame_setting_type);
        if (!rhs.range_sensor_frame->FromYamlNode(node["range_sensor_frame"])) { return false; }
        rhs.row_margin = node["row_margin"].as<long>();
        rhs.col_margin = node["col_margin"].as<long>();
        rhs.gp = node["gp"].as<std::shared_ptr<erl::gaussian_process::VanillaGaussianProcess::Setting>>();
        rhs.mapping = node["mapping"].as<std::shared_ptr<erl::gaussian_process::Mapping::Setting>>();
        return true;
    }
};

// ReSharper restore CppInconsistentNaming

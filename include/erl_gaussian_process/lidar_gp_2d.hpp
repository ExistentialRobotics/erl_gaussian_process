#pragma once

#include "mapping.hpp"
#include "vanilla_gp.hpp"

#include "erl_common/eigen.hpp"
#include "erl_common/yaml.hpp"
#include "erl_geometry/lidar_frame_2d.hpp"

#include <memory>

namespace erl::gaussian_process {

    class LidarGaussianProcess2D {

    public:
        struct Setting : public common::Yamlable<Setting> {
            long group_size = 26;              // number of points in each group, including the overlap ones.
            long overlap_size = 6;             // number of points in the overlap region.
            long margin = 1;                   // points closed to margin will not be used for test because it is difficult to estimate gradient for them.
            double init_variance = 1e6;        // large value to initialize variance result in case of computation failure.
            double sensor_range_var = 0.01;    // variance of the sensor range measurement.
            double max_valid_range_var = 0.1;  // if the distance variance is greater than this threshold, this prediction is invalid and should be discarded.
            double occ_test_temperature = 30;  // OCC Test is a tanh function, this controls the slope around 0.
            std::shared_ptr<geometry::LidarFrame2D::Setting> lidar_frame = std::make_shared<geometry::LidarFrame2D::Setting>();  // parameters of lidar frame
            std::shared_ptr<VanillaGaussianProcess::Setting> gp = std::make_shared<VanillaGaussianProcess::Setting>();  // parameters of local GP regression
            std::shared_ptr<Mapping::Setting> mapping = []() {
                auto mapping_setting = std::make_shared<Mapping::Setting>();
                mapping_setting->type = Mapping::Type::kInverseSqrt;
                mapping_setting->scale = 1.0;
                return mapping_setting;
            }();
        };

    protected:
        bool m_trained_ = false;
        std::shared_ptr<Setting> m_setting_ = nullptr;
        std::vector<std::shared_ptr<VanillaGaussianProcess>> m_gps_;
        std::vector<std::tuple<long, long, double, double>> m_angle_partitions_;
        std::shared_ptr<geometry::LidarFrame2D> m_lidar_frame_ = nullptr;
        std::shared_ptr<Mapping> m_mapping_ = nullptr;
        Eigen::VectorXd m_mapped_distances_ = {};

    public:
        explicit LidarGaussianProcess2D(std::shared_ptr<Setting> setting);

        [[nodiscard]] bool
        IsTrained() const {
            return m_trained_;
        }

        [[nodiscard]] std::shared_ptr<Setting>
        GetSetting() const {
            return m_setting_;
        }

        [[nodiscard]] const std::vector<std::shared_ptr<VanillaGaussianProcess>> &
        GetGps() const {
            return m_gps_;
        }

        [[nodiscard]] const std::vector<std::tuple<long, long, double, double>> &
        GetAnglePartitions() const {
            return m_angle_partitions_;
        }

        [[nodiscard]] std::shared_ptr<const geometry::LidarFrame2D>
        GetLidarFrame() const {
            return m_lidar_frame_;
        }

        [[nodiscard]] Eigen::Vector2d
        GlobalToLocalSo2(const Eigen::Vector2d &dir_global) const {
            return m_lidar_frame_->WorldToFrameSo2(dir_global);
        }

        [[nodiscard]] Eigen::Vector2d
        LocalToGlobalSo2(const Eigen::Vector2d &dir_local) const {
            return m_lidar_frame_->FrameToWorldSo2(dir_local);
        }

        [[nodiscard]] Eigen::Vector2d
        GlobalToLocalSe2(const Eigen::Vector2d &xy_global) const {
            return m_lidar_frame_->WorldToFrameSe2(xy_global);
        }

        [[nodiscard]] Eigen::Vector2d
        LocalToGlobalSe2(const Eigen::Vector2d &xy_local) const {
            return m_lidar_frame_->FrameToWorldSe2(xy_local);
        }

        void
        Reset();

        [[nodiscard]] bool
        StoreData(const Eigen::Matrix2d &rotation, const Eigen::Vector2d &translation, Eigen::VectorXd ranges);

        [[nodiscard]] bool
        Train(const Eigen::Matrix2d &rotation, const Eigen::Vector2d &translation, Eigen::VectorXd ranges, bool repartition_on_hit_rays);

        [[nodiscard]] bool
        Test(
            const Eigen::Ref<const Eigen::VectorXd> &angles,
            bool angles_are_local,
            Eigen::Ref<Eigen::VectorXd> vec_ranges,
            Eigen::Ref<Eigen::VectorXd> vec_ranges_var,
            bool un_map,
            bool parallel) const;

        [[nodiscard]] bool
        ComputeOcc(
            const Eigen::Ref<const Eigen::Scalard> &angle,
            double r,
            Eigen::Ref<Eigen::Scalard> range_pred,
            Eigen::Ref<Eigen::Scalard> range_pred_var,
            double &occ) const;  // return false if failed to compute occ
    };
}  // namespace erl::gaussian_process

// ReSharper disable CppInconsistentNaming
template<>
struct YAML::convert<erl::gaussian_process::LidarGaussianProcess2D::Setting> {
    static Node
    encode(const erl::gaussian_process::LidarGaussianProcess2D::Setting &rhs) {
        Node node;
        node["group_size"] = rhs.group_size;
        node["overlap_size"] = rhs.overlap_size;
        node["margin"] = rhs.margin;
        node["init_variance"] = rhs.init_variance;
        node["sensor_range_var"] = rhs.sensor_range_var;
        node["max_valid_range_var"] = rhs.max_valid_range_var;
        node["occ_test_temperature"] = rhs.occ_test_temperature;
        node["lidar_frame"] = rhs.lidar_frame;
        node["gp"] = rhs.gp;
        node["mapping"] = rhs.mapping;
        return node;
    }

    static bool
    decode(const Node &node, erl::gaussian_process::LidarGaussianProcess2D::Setting &rhs) {
        if (!node.IsMap()) { return false; }
        rhs.group_size = node["group_size"].as<long>();
        rhs.overlap_size = node["overlap_size"].as<long>();
        rhs.margin = node["margin"].as<long>();
        rhs.init_variance = node["init_variance"].as<double>();
        rhs.sensor_range_var = node["sensor_range_var"].as<double>();
        rhs.max_valid_range_var = node["max_valid_range_var"].as<double>();
        rhs.occ_test_temperature = node["occ_test_temperature"].as<double>();
        rhs.lidar_frame = node["lidar_frame"].as<std::shared_ptr<erl::geometry::LidarFrame2D::Setting>>();
        rhs.gp = node["gp"].as<std::shared_ptr<erl::gaussian_process::VanillaGaussianProcess::Setting>>();
        rhs.mapping = node["mapping"].as<std::shared_ptr<erl::gaussian_process::Mapping::Setting>>();
        return true;
    }
};

// ReSharper restore CppInconsistentNaming

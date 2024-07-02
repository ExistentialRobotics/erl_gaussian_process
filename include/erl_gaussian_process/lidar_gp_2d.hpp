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
        /*struct TrainBuffer {
            struct Setting : public common::Yamlable<Setting> {
                double valid_range_min = 0.2;
                double valid_range_max = 30.0;
                double valid_angle_min = -135. / 180. * M_PI;
                double valid_angle_max = 135. / 180. * M_PI;
                std::shared_ptr<Mapping::Setting> mapping = []() {
                    auto mapping_setting = std::make_shared<Mapping::Setting>();
                    mapping_setting->type = Mapping::Type::kInverseSqrt;
                    mapping_setting->scale = 1.0;
                    return mapping_setting;
                }();
            };

            std::shared_ptr<Setting> setting = nullptr;
            std::shared_ptr<Mapping> mapping = nullptr;

            // data
            Eigen::VectorXd vec_angles_original = {};
            Eigen::VectorXd vec_ranges_original = {};
            Eigen::VectorXb vec_mask_hit = {};
            Eigen::VectorXd vec_angles = {};
            Eigen::VectorXd vec_ranges = {};
            Eigen::VectorXd vec_mapped_distances = {};
            Eigen::Matrix2Xd mat_direction_local = {};
            Eigen::Matrix2Xd mat_xy_local = {};
            Eigen::Matrix2Xd mat_direction_global = {};
            Eigen::Matrix2Xd mat_xy_global = {};
            double max_distance = 0.;
            Eigen::Vector2d position = Eigen::Vector2d::Zero();
            Eigen::Matrix2d rotation = Eigen::Matrix2d::Identity();

            TrainBuffer()
                : TrainBuffer(std::make_shared<Setting>()) {}

            explicit TrainBuffer(std::shared_ptr<Setting> setting)
                : setting(std::move(setting)) {}

            [[nodiscard]] long
            Size() const {
                return vec_angles.size();
            }

            /**
             * Store new training data to this TrainBuffer.
             * @param vec_new_angles new angle data in radian, assumed in ascending order.
             * @param vec_new_distances new distance data.
             * @param mat_new_pose 2x3 matrix of 2D rotation and translation.
             * @return true if the data is stored successfully, otherwise false.
             #1#
            bool
            Store(
                const Eigen::Ref<const Eigen::VectorXd> &vec_new_angles,
                const Eigen::Ref<const Eigen::VectorXd> &vec_new_distances,
                const Eigen::Ref<const Eigen::Matrix23d> &mat_new_pose);

            [[nodiscard]] Eigen::Vector2d
            GlobalToLocalSo2(const Eigen::Vector2d &vec_global) const {
                return rotation.transpose() * vec_global;
            }

            [[nodiscard]] Eigen::Vector2d
            LocalToGlobalSo2(const Eigen::Vector2d &vec_local) const {
                return rotation * vec_local;
            }

            [[nodiscard]] Eigen::Vector2d
            GlobalToLocalSe2(const Eigen::Vector2d &vec_global) const {
                return rotation.transpose() * (vec_global - position);
            }

            [[nodiscard]] Eigen::Vector2d
            LocalToGlobalSe2(const Eigen::Vector2d &vec_local) const {
                return rotation * vec_local + position;
            }
        };*/

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
        // std::vector<double> m_partitions_;
        // TrainBuffer m_train_buffer_;

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

        /*[[nodiscard]] TrainBuffer &
        GetTrainBuffer() {
            return m_train_buffer_;
        }*/

        [[nodiscard]] std::shared_ptr<const geometry::LidarFrame2D>
        GetLidarFrame() const {
            return m_lidar_frame_;
        }

        [[nodiscard]] Eigen::Vector2d
        GlobalToLocalSo2(const Eigen::Vector2d &vec_global) const {
            // return m_train_buffer_.GlobalToLocalSo2(vec_global);
            return m_lidar_frame_->WorldToFrameSo2(vec_global);
        }

        [[nodiscard]] Eigen::Vector2d
        LocalToGlobalSo2(const Eigen::Vector2d &vec_local) const {
            // return m_train_buffer_.LocalToGlobalSo2(vec_local);
            return m_lidar_frame_->FrameToWorldSo2(vec_local);
        }

        [[nodiscard]] Eigen::Vector2d
        GlobalToLocalSe2(const Eigen::Vector2d &vec_global) const {
            // return m_train_buffer_.GlobalToLocalSe2(vec_global);
            return m_lidar_frame_->WorldToFrameSe2(vec_global);
        }

        [[nodiscard]] Eigen::Vector2d
        LocalToGlobalSe2(const Eigen::Vector2d &vec_local) const {
            // return m_train_buffer_.LocalToGlobalSe2(vec_local);
            return m_lidar_frame_->FrameToWorldSe2(vec_local);
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
/*template<>
struct YAML::convert<erl::gaussian_process::LidarGaussianProcess2D::TrainBuffer::Setting> {
    static Node
    encode(const erl::gaussian_process::LidarGaussianProcess2D::TrainBuffer::Setting &setting) {
        Node node;
        node["valid_range_min"] = setting.valid_range_min;
        node["valid_range_max"] = setting.valid_range_max;
        node["valid_angle_min"] = setting.valid_angle_min;
        node["valid_angle_max"] = setting.valid_angle_max;
        node["mapping"] = setting.mapping;
        return node;
    }

    static bool
    decode(const Node &node, erl::gaussian_process::LidarGaussianProcess2D::TrainBuffer::Setting &setting) {
        if (!node.IsMap()) { return false; }
        setting.valid_range_min = node["valid_range_min"].as<double>();
        setting.valid_range_max = node["valid_range_max"].as<double>();
        setting.valid_angle_min = node["valid_angle_min"].as<double>();
        setting.valid_angle_max = node["valid_angle_max"].as<double>();
        setting.mapping = node["mapping"].as<std::shared_ptr<erl::gaussian_process::Mapping::Setting>>();
        return true;
    }
};*/

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

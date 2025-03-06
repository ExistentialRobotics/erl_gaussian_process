#pragma once

#include "init.hpp"
#include "mapping.hpp"
#include "vanilla_gp.hpp"

#include "erl_common/yaml.hpp"
#include "erl_geometry/depth_frame_3d.hpp"
#include "erl_geometry/lidar_frame_3d.hpp"

#include <memory>

namespace erl::gaussian_process {

    template<typename Dtype>
    class RangeSensorGaussianProcess3D {
    public:
        using Gp = VanillaGaussianProcess<Dtype>;
        using RangeSensorFrame = geometry::RangeSensorFrame3D<Dtype>;
        using LidarFrame = geometry::LidarFrame3D<Dtype>;
        using DepthFrame = geometry::DepthFrame3D<Dtype>;
        using MappingDtype = Mapping<Dtype>;
        using Scalar = Eigen::Matrix<Dtype, 1, 1>;
        using Matrix3 = Eigen::Matrix3<Dtype>;
        using Matrix3X = Eigen::Matrix3X<Dtype>;
        using Vector2 = Eigen::Vector2<Dtype>;
        using Vector3 = Eigen::Vector3<Dtype>;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;

        struct Setting : common::Yamlable<Setting> {
            long row_group_size = 24;   // number of points in each group for each row, including the overlap ones
            long row_overlap_size = 6;  // number of points in the overlap region for each row
            long row_margin = 0;
            long col_group_size = 8;    // number of elevation points in each group, including the overlap ones
            long col_overlap_size = 2;  // number of points in the overlap region
            long col_margin = 0;
            Dtype init_variance = 1e6;        // large value to initialize variance result in case of computation failure
            Dtype sensor_range_var = 0.01;    // variance of the sensor range measurement
            Dtype max_valid_range_var = 0.1;  // if the distance variance is greater than this threshold, the prediction is invalid and should be discarded
            Dtype occ_test_temperature = 30;  // OCC test is a tanh function, this controls the slope around 0
            std::string range_sensor_frame_type = type_name<LidarFrame>();                            // type of the range sensor frame
            std::string range_sensor_frame_setting_type = type_name<typename LidarFrame::Setting>();  // type of the range sensor frame setting
            std::shared_ptr<typename RangeSensorFrame::Setting> range_sensor_frame = std::make_shared<typename LidarFrame::Setting>();
            std::shared_ptr<typename Gp::Setting> gp = std::make_shared<typename Gp::Setting>();
            std::shared_ptr<typename MappingDtype::Setting> mapping = []() {
                auto mapping_setting = std::make_shared<typename MappingDtype::Setting>();
                mapping_setting->type = MappingType::kInverseSqrt;
                mapping_setting->scale = 1.0;
                return mapping_setting;
            }();

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

    private:
        inline static const std::string kFileHeader = fmt::format("# {}", type_name<RangeSensorGaussianProcess3D>());

    protected:
        bool m_trained_ = false;
        std::shared_ptr<Setting> m_setting_ = nullptr;
        Eigen::MatrixX<std::shared_ptr<Gp>> m_gps_ = {};
        std::vector<std::tuple<long, long, Dtype, Dtype>> m_row_partitions_ = {};
        std::vector<std::tuple<long, long, Dtype, Dtype>> m_col_partitions_ = {};
        std::shared_ptr<RangeSensorFrame> m_range_sensor_frame_ = nullptr;
        std::shared_ptr<MappingDtype> m_mapping_ = nullptr;
        MatrixX m_mapped_distances_ = {};

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

        [[nodiscard]] const Eigen::MatrixX<std::shared_ptr<Gp>> &
        GetGps() const {
            return m_gps_;
        }

        [[nodiscard]] const std::vector<std::tuple<long, long, Dtype, Dtype>> &
        GetRowPartitions() const {
            return m_row_partitions_;
        }

        [[nodiscard]] const std::vector<std::tuple<long, long, Dtype, Dtype>> &
        GetColPartitions() const {
            return m_col_partitions_;
        }

        [[nodiscard]] std::shared_ptr<const RangeSensorFrame>
        GetRangeSensorFrame() const {
            return m_range_sensor_frame_;
        }

        [[nodiscard]] std::shared_ptr<const MappingDtype>
        GetMapping() const {
            return m_mapping_;
        }

        [[nodiscard]] Vector3
        GlobalToLocalSo3(const Vector3 &dir_global) const {
            return m_range_sensor_frame_->WorldToFrameSo3(dir_global);
        }

        [[nodiscard]] Vector3
        LocalToGlobalSo3(const Vector3 &dir_local) const {
            return m_range_sensor_frame_->FrameToWorldSo3(dir_local);
        }

        [[nodiscard]] Vector3
        GlobalToLocalSe3(const Vector3 &xyz_global) const {
            return m_range_sensor_frame_->WorldToFrameSe3(xyz_global);
        }

        [[nodiscard]] Vector3
        LocalToGlobalSe3(const Vector3 &xyz_local) const {
            return m_range_sensor_frame_->FrameToWorldSe3(xyz_local);
        }

        [[nodiscard]] Vector2
        ComputeFrameCoords(const Vector3 &xyz_frame) const {
            return m_range_sensor_frame_->ComputeFrameCoords(xyz_frame);
        }

        void
        Reset();

        bool
        StoreData(const Matrix3 &rotation, const Vector3 &translation, MatrixX ranges);

        [[nodiscard]] bool
        Train(const Matrix3 &rotation, const Vector3 &translation, MatrixX ranges);

        [[nodiscard]] bool
        Test(
            const Eigen::Ref<const Matrix3X> &directions,
            bool directions_are_local,
            Eigen::Ref<VectorX> vec_ranges,
            Eigen::Ref<VectorX> vec_ranges_var,
            bool un_map) const;

        bool
        ComputeOcc(const Vector3 &dir_local, Dtype r, Eigen::Ref<Scalar> range_pred, Eigen::Ref<Scalar> range_pred_var, Dtype &occ) const;

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

    using RangeSensorGaussianProcess3Dd = RangeSensorGaussianProcess3D<double>;
    using RangeSensorGaussianProcess3Df = RangeSensorGaussianProcess3D<float>;
}  // namespace erl::gaussian_process

#include "range_sensor_gp_3d.tpp"

template<>
struct YAML::convert<erl::gaussian_process::RangeSensorGaussianProcess3Dd::Setting>
    : erl::gaussian_process::RangeSensorGaussianProcess3Dd::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gaussian_process::RangeSensorGaussianProcess3Df::Setting>
    : erl::gaussian_process::RangeSensorGaussianProcess3Df::Setting::YamlConvertImpl {};

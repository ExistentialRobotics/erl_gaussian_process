#pragma once

#include "init.hpp"
#include "mapping.hpp"
#include "vanilla_gp.hpp"

#include "erl_geometry/lidar_frame_2d.hpp"

#include <memory>

namespace erl::gaussian_process {

    template<typename Dtype>
    class LidarGaussianProcess2D {
    public:
        using Gp = VanillaGaussianProcess<Dtype>;
        using MappingDtype = Mapping<Dtype>;
        using Scalar = Eigen::Matrix<Dtype, 1, 1>;
        using Matrix2 = Eigen::Matrix2<Dtype>;
        using Vector2 = Eigen::Vector2<Dtype>;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;
        using LidarFrame2D = geometry::LidarFrame2D<Dtype>;

        struct Setting : common::Yamlable<Setting> {
            long group_size = 26;             // number of points in each group, including the overlap ones.
            long overlap_size = 6;            // number of points in the overlap region.
            long margin = 1;                  // points closed to margin will not be used for test because it is difficult to estimate gradient for them.
            Dtype init_variance = 1e6;        // large value to initialize variance result in case of computation failure.
            Dtype sensor_range_var = 0.01;    // variance of the sensor range measurement.
            Dtype max_valid_range_var = 0.1;  // if the distance variance is greater than this threshold, this prediction is invalid and should be discarded.
            Dtype occ_test_temperature = 30;  // OCC Test is a tanh function, this controls the slope around 0.
            std::shared_ptr<typename LidarFrame2D::Setting> sensor_frame = std::make_shared<typename LidarFrame2D::Setting>();  // parameters of lidar frame
            std::shared_ptr<typename Gp::Setting> gp = std::make_shared<typename Gp::Setting>();  // parameters of local GP regression
            std::shared_ptr<typename MappingDtype::Setting> mapping = []() -> std::shared_ptr<typename MappingDtype::Setting> {
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
        inline static const volatile bool kSettingRegistered = common::YamlableBase::Register<Setting>();
        inline static const std::string kFileHeader = fmt::format("# {}", type_name<LidarGaussianProcess2D>());

    protected:
        bool m_trained_ = false;
        std::shared_ptr<Setting> m_setting_ = nullptr;
        std::vector<std::shared_ptr<Gp>> m_gps_;
        std::vector<std::tuple<long, long, Dtype, Dtype>> m_angle_partitions_;
        std::shared_ptr<LidarFrame2D> m_sensor_frame_ = nullptr;
        std::shared_ptr<MappingDtype> m_mapping_ = nullptr;
        VectorX m_mapped_distances_ = {};

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

        [[nodiscard]] const std::vector<std::shared_ptr<Gp>> &
        GetGps() const {
            return m_gps_;
        }

        [[nodiscard]] const std::vector<std::tuple<long, long, Dtype, Dtype>> &
        GetAnglePartitions() const {
            return m_angle_partitions_;
        }

        [[nodiscard]] std::shared_ptr<const LidarFrame2D>
        GetSensorFrame() const {
            return m_sensor_frame_;
        }

        [[nodiscard]] Vector2
        GlobalToLocalSo2(const Vector2 &dir_global) const {
            return m_sensor_frame_->DirWorldToFrame(dir_global);
        }

        [[nodiscard]] Vector2
        LocalToGlobalSo2(const Vector2 &dir_local) const {
            return m_sensor_frame_->DirFrameToWorld(dir_local);
        }

        [[nodiscard]] Vector2
        GlobalToLocalSe2(const Vector2 &xy_global) const {
            return m_sensor_frame_->PosWorldToFrame(xy_global);
        }

        [[nodiscard]] Vector2
        LocalToGlobalSe2(const Vector2 &xy_local) const {
            return m_sensor_frame_->PosFrameToWorld(xy_local);
        }

        void
        Reset();

        [[nodiscard]] bool
        StoreData(const Matrix2 &rotation, const Vector2 &translation, VectorX ranges);

        void
        RepartitionOnHitRays();

        [[nodiscard]] bool
        Train(const Matrix2 &rotation, const Vector2 &translation, VectorX ranges);

        [[nodiscard]] bool
        Test(const Eigen::Ref<const VectorX> &angles, bool angles_are_local, Eigen::Ref<VectorX> vec_ranges, Eigen::Ref<VectorX> vec_ranges_var, bool un_map)
            const;

        [[nodiscard]] bool
        ComputeOcc(
            const Eigen::Ref<const Scalar> &angle_local,
            Dtype r,
            Eigen::Ref<Scalar> range_pred,
            Eigen::Ref<Scalar> range_pred_var,
            Dtype &occ) const;  // return false if failed to compute occ

        [[nodiscard]] bool
        operator==(const LidarGaussianProcess2D &other) const;

        [[nodiscard]] bool
        operator!=(const LidarGaussianProcess2D &other) const;

        [[nodiscard]] bool
        Write(const std::string &filename) const;

        [[nodiscard]] bool
        Write(std::ostream &s) const;

        [[nodiscard]] bool
        Read(const std::string &filename);

        [[nodiscard]] bool
        Read(std::istream &s);
    };

    using LidarGaussianProcess2Dd = LidarGaussianProcess2D<double>;
    using LidarGaussianProcess2Df = LidarGaussianProcess2D<float>;
}  // namespace erl::gaussian_process

#include "lidar_gp_2d.tpp"

template<>
struct YAML::convert<erl::gaussian_process::LidarGaussianProcess2Dd::Setting> : erl::gaussian_process::LidarGaussianProcess2Dd::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gaussian_process::LidarGaussianProcess2Df::Setting> : erl::gaussian_process::LidarGaussianProcess2Df::Setting::YamlConvertImpl {};

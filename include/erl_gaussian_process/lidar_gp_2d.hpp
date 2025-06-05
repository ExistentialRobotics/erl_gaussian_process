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
        using GpSetting = typename Gp::Setting;
        using MappingDtype = Mapping<Dtype>;
        using MappingSetting = typename MappingDtype::Setting;
        using Scalar = Eigen::Matrix<Dtype, 1, 1>;
        using Matrix2 = Eigen::Matrix2<Dtype>;
        using Vector2 = Eigen::Vector2<Dtype>;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;
        using LidarFrame2D = geometry::LidarFrame2D<Dtype>;
        using LidarFrameSetting = typename LidarFrame2D::Setting;

        struct Setting : public common::Yamlable<Setting> {
            // if true, partitions are created based on hit rays. otherwise, partitions are created
            // based on angles.
            bool partition_on_hit_rays = false;
            // if true, partitions are symmetric. otherwise, partitions are asymmetric.
            bool symmetric_partitions = true;
            // number of points in each group, including the overlap ones.
            long group_size = 26;
            // number of points in the overlap region.
            long overlap_size = 6;
            // points closed to margin will not be used for test because it is challenging to
            // estimate gradient for them.
            long margin = 1;
            // large value to initialize the variance prediction in case of computation failure.
            Dtype init_variance = 1e6f;
            // variance of the sensor range measurement.
            Dtype sensor_range_var = 0.01f;
            // if sensor_frame->discontinuity_detection is true, this value is used when a
            // discontinuity is detected in the sensor frame.
            Dtype discontinuity_var = 10.0f;
            // if the distance variance is greater than this threshold, the prediction is invalid
            // and should be discarded.
            Dtype max_valid_range_var = 0.1f;
            // OCC Test is a tanh function, this controls the slope around 0.
            Dtype occ_test_temperature = 30.0f;
            // parameters of lidar frame
            std::shared_ptr<LidarFrameSetting> sensor_frame = std::make_shared<LidarFrameSetting>();
            // parameters of local GP regression
            std::shared_ptr<GpSetting> gp = std::make_shared<GpSetting>();
            std::shared_ptr<MappingSetting> mapping = []() -> std::shared_ptr<MappingSetting> {
                auto mapping_setting = std::make_shared<MappingSetting>();
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

        class TestResult {
        protected:
            const LidarGaussianProcess2D *m_gp_;
            std::vector<const Gp *> m_gps_;
            std::vector<VectorX> m_k_test_vec_;
            std::vector<std::pair<const Dtype *, long>> m_alpha_vec_;
            std::vector<VectorX> m_alpha_test_vec_;
            std::shared_ptr<MappingDtype> m_mapping_;
            bool m_reduced_rank_kernel_ = false;

        public:
            TestResult(
                const LidarGaussianProcess2D *gp,
                const Eigen::Ref<const VectorX> &angles,
                bool angles_are_local,
                const std::shared_ptr<MappingDtype> &mapping);

            [[nodiscard]] long
            GetNumTest() const;

            [[nodiscard]] const VectorX &
            GetKtest(long index) const;

            [[nodiscard]] Eigen::VectorXb
            GetMean(Eigen::Ref<VectorX> vec_f_out, bool parallel) const;

            [[nodiscard]] bool
            GetMean(long index, Dtype &f) const;

            [[nodiscard]] Eigen::VectorXb
            GetVariance(Eigen::Ref<VectorX> vec_var_out, bool parallel) const;

            [[nodiscard]] bool
            GetVariance(long index, Dtype &var) const;

        protected:
            void
            PrepareAlphaTest(long index);
        };

    protected:
        std::shared_ptr<Setting> m_setting_ = nullptr;
        bool m_trained_ = false;
        std::vector<std::shared_ptr<Gp>> m_gps_;
        std::vector<std::tuple<long, long, Dtype, Dtype>> m_angle_partitions_;
        std::shared_ptr<LidarFrame2D> m_sensor_frame_ = nullptr;
        std::shared_ptr<MappingDtype> m_mapping_ = nullptr;
        VectorX m_mapped_distances_ = {};

    public:
        explicit LidarGaussianProcess2D(std::shared_ptr<Setting> setting);

        [[nodiscard]] bool
        IsTrained() const;

        [[nodiscard]] std::shared_ptr<Setting>
        GetSetting() const;

        [[nodiscard]] const std::vector<std::shared_ptr<Gp>> &
        GetGps() const;

        [[nodiscard]] const std::vector<std::tuple<long, long, Dtype, Dtype>> &
        GetAnglePartitions() const;

        [[nodiscard]] std::shared_ptr<const LidarFrame2D>
        GetSensorFrame() const;

        [[nodiscard]] std::shared_ptr<const MappingDtype>
        GetMapping() const;

        [[nodiscard]] Vector2
        GlobalToLocalSo2(const Vector2 &dir_global) const;

        [[nodiscard]] Vector2
        LocalToGlobalSo2(const Vector2 &dir_local) const;

        [[nodiscard]] Vector2
        GlobalToLocalSe2(const Vector2 &xy_global) const;

        [[nodiscard]] Vector2
        LocalToGlobalSe2(const Vector2 &xy_local) const;

        void
        Reset();

        [[nodiscard]] bool
        StoreData(const Matrix2 &rotation, const Vector2 &translation, VectorX ranges);

        /**
         * Create partitions based on angles.
         */
        void
        PartitionOnAngles();

        /**
         * Create partitions based on hit rays.
         */
        void
        PartitionOnHitRays();

        [[nodiscard]] bool
        Train(const Matrix2 &rotation, const Vector2 &translation, VectorX ranges);

        [[nodiscard]] long
        SearchPartition(Dtype angle_local) const;

        [[nodiscard]] std::shared_ptr<TestResult>
        Test(const Eigen::Ref<const VectorX> &angles, bool angles_are_local, bool un_map) const;

        /**
         * Compute the occupancy of a point in the local frame.
         * @param angle_local Ray angle in the local frame.
         * @param r Distance between the sensor and the surface point.
         * @param range_pred Range prediction by the GP.
         * @param occ Occupancy prediction.
         * @return if the computation is successful.
         */
        [[nodiscard]] bool
        ComputeOcc(const Scalar &angle_local, Dtype r, Dtype &range_pred, Dtype &occ) const;

        [[nodiscard]] bool
        operator==(const LidarGaussianProcess2D &other) const;

        [[nodiscard]] bool
        operator!=(const LidarGaussianProcess2D &other) const;

        [[nodiscard]] bool
        Write(std::ostream &s) const;

        [[nodiscard]] bool
        Read(std::istream &s);
    };

    using LidarGaussianProcess2Dd = LidarGaussianProcess2D<double>;
    using LidarGaussianProcess2Df = LidarGaussianProcess2D<float>;
}  // namespace erl::gaussian_process

#include "lidar_gp_2d.tpp"

template<>
struct YAML::convert<erl::gaussian_process::LidarGaussianProcess2Dd::Setting>
    : erl::gaussian_process::LidarGaussianProcess2Dd::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gaussian_process::LidarGaussianProcess2Df::Setting>
    : erl::gaussian_process::LidarGaussianProcess2Df::Setting::YamlConvertImpl {};

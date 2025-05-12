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

        struct Setting : public common::Yamlable<Setting> {
            // number of points in each group for each row, including the overlap ones
            long row_group_size = 24;
            // number of points in the overlap region for each row
            long row_overlap_size = 6;
            long row_margin = 0;
            // number of elevation points in each group, including the overlap ones
            long col_group_size = 8;
            // number of points in the overlap region
            long col_overlap_size = 2;
            long col_margin = 0;
            // large value to initialize variance results in case of computation failure
            Dtype init_variance = 1e6f;
            // variance of the sensor range measurement
            Dtype sensor_range_var = 0.01f;
            // if the distance variance is greater than this threshold, the prediction is invalid
            // and should be discarded
            Dtype max_valid_range_var = 0.1f;
            // OCC test is a tanh function, this controls the slope around 0
            Dtype occ_test_temperature = 30.0f;
            // type of the range sensor frame
            std::string sensor_frame_type = type_name<LidarFrame>();
            // type of the range sensor frame setting
            std::string sensor_frame_setting_type = type_name<typename LidarFrame::Setting>();
            std::shared_ptr<typename RangeSensorFrame::Setting> sensor_frame =
                std::make_shared<typename LidarFrame::Setting>();
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

        class TestResult {
        protected:
            const RangeSensorGaussianProcess3D *m_gp_;
            std::vector<const Gp *> m_gps_;
            std::vector<VectorX> m_k_test_vec_;
            std::vector<std::pair<const Dtype *, long>> m_alpha_vec_;
            std::vector<VectorX> m_alpha_test_vec_;
            std::shared_ptr<MappingDtype> m_mapping_ = nullptr;
            bool m_reduced_rank_kernel_ = false;

        public:
            TestResult(
                const RangeSensorGaussianProcess3D *gp,
                const Eigen::Ref<const Matrix3X> &directions,
                bool directions_are_local,
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
        bool m_trained_ = false;
        std::shared_ptr<Setting> m_setting_ = nullptr;
        Eigen::MatrixX<std::shared_ptr<Gp>> m_gps_ = {};
        std::vector<std::tuple<long, long, Dtype, Dtype>> m_row_partitions_ = {};
        std::vector<std::tuple<long, long, Dtype, Dtype>> m_col_partitions_ = {};
        std::shared_ptr<RangeSensorFrame> m_sensor_frame_ = nullptr;
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
        GetSensorFrame() const {
            return m_sensor_frame_;
        }

        [[nodiscard]] std::shared_ptr<const MappingDtype>
        GetMapping() const {
            return m_mapping_;
        }

        [[nodiscard]] Vector3
        GlobalToLocalSo3(const Vector3 &dir_global) const {
            return m_sensor_frame_->DirWorldToFrame(dir_global);
        }

        [[nodiscard]] Vector3
        LocalToGlobalSo3(const Vector3 &dir_local) const {
            return m_sensor_frame_->DirFrameToWorld(dir_local);
        }

        [[nodiscard]] Vector3
        GlobalToLocalSe3(const Vector3 &xyz_global) const {
            return m_sensor_frame_->PosWorldToFrame(xyz_global);
        }

        [[nodiscard]] Vector3
        LocalToGlobalSe3(const Vector3 &xyz_local) const {
            return m_sensor_frame_->PosFrameToWorld(xyz_local);
        }

        [[nodiscard]] Vector2
        ComputeFrameCoords(const Vector3 &xyz_frame) const {
            return m_sensor_frame_->ComputeFrameCoords(xyz_frame);
        }

        void
        Reset();

        bool
        StoreData(const Matrix3 &rotation, const Vector3 &translation, MatrixX ranges);

        [[nodiscard]] bool
        Train(const Matrix3 &rotation, const Vector3 &translation, MatrixX ranges);

        [[nodiscard]] std::pair<long, long>
        SearchPartition(const Vector2 &frame_coords) const;

        [[nodiscard]] std::shared_ptr<TestResult>
        Test(const Eigen::Ref<const Matrix3X> &directions, bool directions_are_local, bool un_map)
            const;

        bool
        ComputeOcc(const Vector3 &dir_local, Dtype r, Dtype &range_pred, Dtype &occ) const;

        [[nodiscard]] bool
        operator==(const RangeSensorGaussianProcess3D &other) const;

        [[nodiscard]] bool
        operator!=(const RangeSensorGaussianProcess3D &other) const {
            return !(*this == other);
        }

        [[nodiscard]] bool
        Write(std::ostream &s) const;

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

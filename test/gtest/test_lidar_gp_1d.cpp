#include "../cov_fnc.cpp"
#include "../obs_gp.cpp"
#include "../obs_gp.h"

#include "erl_common/binary_file.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_gaussian_process/lidar_gp_1d.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <iostream>

using namespace erl::common;
using namespace erl::gaussian_process;

constexpr double kMaxRange = 30.;
constexpr double kMinRange = 0.2;
constexpr double kSensorOffsetX = 0.08;  // sensor x-offset in the robot frame (IMU frame).
constexpr double kSensorOffsetY = 0.;    // sensor y-offset in the robot frame (IMU frame).

using TrainDataFrame = struct TrainDataFrame {
    Eigen::VectorXd angles;
    Eigen::VectorXd distances;
    std::vector<double> pose_matlab;
    Eigen::RMatrix23d pose_numpy;
    Eigen::Vector2d position;  // 2D position
    Eigen::Matrix2d rotation;  // 2D rotation

    Eigen::VectorXd x;  // angle
    Eigen::VectorXd y;  // distance

    TrainDataFrame(double *pa, double *pr, const double *pose_ptr, const int numel) {
        angles.resize(numel);
        distances.resize(numel);
        std::copy_n(pa, numel, angles.begin());
        std::copy_n(pr, numel, distances.begin());

        Eigen::Matrix23d pose;
        // clang-format off
        pose << pose_ptr[0], pose_ptr[2], pose_ptr[4],
                pose_ptr[1], pose_ptr[3], pose_ptr[5];
        // clang-format on
        position = pose.col(0);
        rotation = pose.block<2, 2>(0, 1);

        Eigen::Vector2d sensor_offset = rotation * Eigen::Vector2d{kSensorOffsetX, kSensorOffsetY};

        pose_matlab.clear();
        pose_matlab.insert(pose_matlab.begin(), pose_ptr, pose_ptr + 6);
        pose_matlab[0] += sensor_offset[0];
        pose_matlab[1] += sensor_offset[1];

        pose_numpy.topLeftCorner<2, 2>() = rotation;
        pose_numpy.col(2) = position + sensor_offset;

        int cnt = 0;
        x.resize(numel);
        y.resize(numel);
        for (int i = 0; i < numel; ++i) {
            if ((distances[i] > kMaxRange) || (distances[i] < kMinRange)) { continue; }

            x[cnt] = angles[i];
            y[cnt] = distances[i];
            cnt++;
        }
    }
};

template<typename T>
void
ReadVar(char *&data_ptr, T &var) {
    var = reinterpret_cast<T *>(data_ptr)[0];
    data_ptr += sizeof(T);
}

template<typename T>
void
ReadPtr(char *&data_ptr, const size_t n, T *&ptr) {
    ptr = reinterpret_cast<T *>(data_ptr);
    data_ptr += sizeof(T) * n;
}

class TrainDataLoader {
    std::vector<TrainDataFrame> m_data_frames_;

public:
    explicit TrainDataLoader(const char *path) {
        auto data = LoadBinaryFile<char>(path);

        char *data_ptr = data.data();
        const auto data_ptr_begin = data_ptr;
        const size_t data_size = data.size();
        int numel;
        double *pa, *pr, *pose_ptr;
        std::size_t pose_size;

        while (data_ptr < data_ptr_begin + data_size) {
            ReadVar(data_ptr, numel);
            ReadPtr(data_ptr, numel, pa);
            ReadPtr(data_ptr, numel, pr);
            ReadVar(data_ptr, pose_size);
            ReadPtr(data_ptr, pose_size, pose_ptr);
            m_data_frames_.emplace_back(pa, pr, pose_ptr, numel);
        }
    }

    TrainDataFrame
    operator[](const size_t i) {
        return m_data_frames_[i];
    }

    [[nodiscard]] size_t
    Size() const {
        return m_data_frames_.size();
    }
};

#define DEFAULT_OBSGP_SCALE_PARAM double(0.5)
#define DEFAULT_OBSGP_NOISE_PARAM double(0.01)
#define DEFAULT_OBSGP_OVERLAP_SZ  6
#define DEFAULT_OBSGP_GROUP_SZ    20
#define DEFAULT_OBSGP_MARGIN      double(0.0175)
#define GPISMAP_OBS_VAR_THRE      double(0.1)

TEST(ERL_GAUSSIAN_PROCESS, LidarGaussianProcess2D) {
    std::filesystem::path path = __FILE__;
    path = path.parent_path().parent_path();
    path /= "double/train.dat";
    auto train_data_loader = TrainDataLoader(path.string().c_str());

    auto setting = std::make_shared<LidarGaussianProcess2D::Setting>();
    setting->group_size = DEFAULT_OBSGP_GROUP_SZ + DEFAULT_OBSGP_OVERLAP_SZ;
    setting->overlap_size = DEFAULT_OBSGP_OVERLAP_SZ;
    setting->boundary_margin = DEFAULT_OBSGP_MARGIN;
    setting->init_variance = 1.e6;
    setting->sensor_range_var = DEFAULT_OBSGP_NOISE_PARAM;
    setting->max_valid_range_var = GPISMAP_OBS_VAR_THRE;
    setting->gp->kernel->alpha = 1.;
    setting->gp->kernel->scale = DEFAULT_OBSGP_SCALE_PARAM;
    setting->train_buffer->mapping->type = Mapping::Type::kIdentity;
    std::cout << *setting << std::endl;

    auto lidar_gp = LidarGaussianProcess2D::Create(setting);
    ObsGp1D obs_gp;

    auto df = train_data_loader[0];

    Logging::Info("Train:");
    auto n = static_cast<int>(df.x.size());
    ReportTime<std::chrono::microseconds>("LidarGaussianProcess2D", 10, false, [&] { lidar_gp->Train(df.x, df.y, Eigen::Matrix23d::Zero()); });
    ReportTime<std::chrono::microseconds>("ObsGp1D", 10, false, [&] { obs_gp.Train(df.x.data(), df.y.data(), &n); });

    ASSERT_STD_VECTOR_EQUAL("m_partitions_", lidar_gp->GetPartitions(), obs_gp.m_range_);

    auto gps = lidar_gp->GetGps();
    for (size_t i = 0; i < gps.size(); ++i) {
        std::stringstream ss;
        ss << "m_gps_[" << i << "]->m_x_:";
        Eigen::MatrixXd mat_x_gt = obs_gp.m_gps_[i]->m_x_;
        Eigen::MatrixXd mat_x_ans = gps[i]->GetTrainInputSamplesBuffer().topLeftCorner(mat_x_gt.rows(), mat_x_gt.cols());
        ASSERT_EIGEN_MATRIX_EQUAL(ss.str().c_str(), mat_x_ans, mat_x_gt);
    }

    for (size_t i = 0; i < gps.size(); ++i) {
        std::stringstream ss;
        ss << "m_gps_[" << i << "]->m_l_:";
        Eigen::MatrixXd mat_l_gt = obs_gp.m_gps_[i]->m_l_;
        Eigen::MatrixXd mat_l_ans = gps[i]->GetCholeskyDecomposition().topLeftCorner(mat_l_gt.rows(), mat_l_gt.cols());
        ASSERT_EIGEN_MATRIX_EQUAL(ss.str(), mat_l_ans, mat_l_gt);
    }

    for (size_t i = 0; i < gps.size(); ++i) {
        std::stringstream ss;
        ss << "m_gps_[" << i << "]->m_alpha_:";
        Eigen::VectorXd alpha_gt = obs_gp.m_gps_[i]->m_alpha_;
        Eigen::VectorXd alpha_ans = gps[i]->GetTrainOutputSamplesBuffer().head(alpha_gt.size());
        ASSERT_EIGEN_VECTOR_EQUAL(ss.str(), alpha_ans, alpha_gt);
    }

    for (size_t i = 1; i < train_data_loader.Size(); ++i) {
        df = train_data_loader[i];
        Eigen::VectorXd ans_f, ans_var, gt_f, gt_var;
        ans_f.resize(df.x.size());
        ans_var.resize(df.x.size());
        gt_f.setConstant(df.x.size(), 0.);
        gt_var.setConstant(gt_f.size(), 0.);
        Logging::Info("test[", i, "]:");
        ReportTime<std::chrono::microseconds>("LidarGaussianProcess2D", 10, false, [&] { lidar_gp->Test(df.x, ans_f, ans_var, true); });
        ReportTime<std::chrono::microseconds>("ObsGp1D", 10, false, [&] { obs_gp.Test(df.x.transpose(), gt_f, gt_var); });
#ifdef NDEBUG
        ASSERT_EIGEN_VECTOR_NEAR("f", ans_f, gt_f, 1e-15);
        ASSERT_EIGEN_VECTOR_NEAR("var", ans_var, gt_var, 1e-15);
#else
        ASSERT_EIGEN_VECTOR_EQUAL("f", ans_f, gt_f);
        ASSERT_EIGEN_VECTOR_EQUAL("var", ans_var, gt_var);
#endif
    }
}

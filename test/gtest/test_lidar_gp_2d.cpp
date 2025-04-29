#include "erl_common/binary_file.hpp"
#include "erl_common/plplot_fig.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_covariance/ornstein_uhlenbeck.hpp"
#include "erl_gaussian_process/lidar_gp_2d.hpp"

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
// #define DEFAULT_OBSGP_MARGIN      double(0.0175)
#define GPISMAP_OBS_VAR_THRE double(0.1)
#define ANGLE_MIN            double(-135.0 / 180.0 * M_PI)
#define ANGLE_MAX            2.33874  // double(135.0 / 180.0 * M_PI)

TEST(LidarGaussianProcess2D, Build) {
    std::filesystem::path path = __FILE__;
    path = path.parent_path().parent_path();
    path /= "double/train.dat";
    auto train_data_loader = TrainDataLoader(path.string().c_str());

    auto df = train_data_loader[0];
    auto n = static_cast<int>(df.x.size());

    using LidarGp2D = LidarGaussianProcess2D<double>;
    auto setting = std::make_shared<LidarGp2D::Setting>();
    setting->group_size = DEFAULT_OBSGP_GROUP_SZ + DEFAULT_OBSGP_OVERLAP_SZ;
    setting->overlap_size = DEFAULT_OBSGP_OVERLAP_SZ;
    setting->margin = 1;
    setting->init_variance = 1.0e6;
    setting->sensor_range_var = DEFAULT_OBSGP_NOISE_PARAM;
    setting->max_valid_range_var = GPISMAP_OBS_VAR_THRE;
    setting->sensor_frame->valid_range_min;
    setting->sensor_frame->num_rays = n;
    setting->sensor_frame->angle_min = df.angles[0];
    setting->sensor_frame->angle_max = df.angles[n - 1];
    setting->gp->kernel_type = type_name<erl::covariance::OrnsteinUhlenbeck1d>();
    setting->gp->kernel->scale = DEFAULT_OBSGP_SCALE_PARAM;
    setting->mapping->type = MappingType::kIdentity;
    setting->partition_on_hit_rays = false;
    setting->symmetric_partitions = false;
    std::cout << *setting << std::endl;

    auto lidar_gp = std::make_shared<LidarGp2D>(setting);
    Logging::Info("Train:");
    ReportTime<std::chrono::microseconds>("LidarGaussianProcess2D", 1, false, [&] {
        (void) lidar_gp->Train(Eigen::Matrix2d::Identity(), Eigen::Vector2d::Zero(), df.distances);
    });

    Eigen::VectorXd distance_pred(df.distances.size()), distance_pred_var;
    ReportTime<std::chrono::microseconds>("LidarGaussianProcess2D", 1, false, [&] {
        (void) lidar_gp->Test(df.angles, false, distance_pred, distance_pred_var, true);
    });

    Eigen::VectorXd error = distance_pred - df.distances;

    PlplotFig fig(640, 480, true);
    PlplotFig::LegendOpt legend_opt(3, {"train", "pred", "error"});
    legend_opt.SetTextColors({PlplotFig::Color0::Red, PlplotFig::Color0::Green, PlplotFig::Color0::Yellow})
        .SetStyles({PL_LEGEND_LINE, PL_LEGEND_LINE, PL_LEGEND_LINE})
        .SetLineColors(legend_opt.text_colors)
        .SetLineStyles({1, 1, 1})
        .SetLineWidths({1.0, 1.0, 1.0})
        .SetBoxStyle(PL_LEGEND_BOUNDING_BOX)
        .SetBgColor0(PlplotFig::Color0::Gray)
        .SetLegendBoxLineColor0(PlplotFig::Color0::White);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(
            df.angles.minCoeff() - 0.1,
            df.angles.maxCoeff() + 0.1,
            std::min(df.distances.minCoeff(), distance_pred.minCoeff()) - 0.1,
            std::max(df.distances.maxCoeff(), distance_pred.maxCoeff()) + 0.1)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .SetCurrentColor(PlplotFig::Color0::Red)
        .SetLineStyle(1)
        .DrawLine(n, df.angles.data(), df.distances.data())
        .SetCurrentColor(PlplotFig::Color0::Green)
        .DrawLine(n, df.angles.data(), distance_pred.data())
        .SetAxisLimits(df.angles.minCoeff() - 0.1, df.angles.maxCoeff() + 0.1, error.minCoeff() - 0.1, error.maxCoeff() + 0.1)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(
            PlplotFig::AxisOpt::Off(),
            PlplotFig::AxisOpt::Off().DrawTopRightEdge().DrawTopRightTickLabels().DrawTickMajor().DrawTickMinor().DrawPerpendicularTickLabels())
        .SetAxisLabelY("error", true)
        .SetCurrentColor(PlplotFig::Color0::Yellow)
        .DrawLine(n, df.angles.data(), error.data())
        .Legend(legend_opt);

    double mae = error.cwiseAbs().mean();
    ERL_INFO("mean absolute error: {}", mae);

    cv::imshow(test_info_->name(), fig.ToCvMat());
    cv::waitKey(0);

    ASSERT_TRUE(mae < 0.022);
}

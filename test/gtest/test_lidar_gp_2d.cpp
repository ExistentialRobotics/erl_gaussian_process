#include "erl_common/binary_file.hpp"
#include "erl_common/block_timer.hpp"
#include "erl_common/plplot_fig.hpp"
#include "erl_common/serialization.hpp"
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
        std::copy_n(pa, numel, angles.data());
        std::copy_n(pr, numel, distances.data());

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

constexpr double OBSGP_SCALE_PARAM = 0.05;
constexpr double OBSGP_NOISE_PARAM = 0.01;
constexpr double OBSGP_DISCON_NOISE_PARAM = 100.0;
constexpr long OBSGP_OVERLAP_SZ = 6;
constexpr long OBSGP_GROUP_SZ = 20;
constexpr long OBSGP_MARGIN = 1;
constexpr double GPISMAP_OBS_VAR_THRE = 0.1;
constexpr double ANGLE_MIN = -135.0 / 180.0 * M_PI;
constexpr double ANGLE_MAX = 2.33874;  // double(135.0 / 180.0 * M_PI)
constexpr double RANGE_MIN = 0.1;
constexpr double RANGE_MAX = 30.0;

TEST(LidarGaussianProcess2D, Build) {
    GTEST_PREPARE_OUTPUT_DIR();
    std::filesystem::path path = ERL_GAUSSIAN_PROCESS_ROOT_DIR;
    path /= "data/double/train.dat";
    auto train_data_loader = TrainDataLoader(path.string().c_str());

    auto df = train_data_loader[0];
    auto n = static_cast<int>(df.x.size());

    using LidarGp2D = LidarGaussianProcess2D<double>;
    auto setting = std::make_shared<LidarGp2D::Setting>();
    setting->group_size = OBSGP_GROUP_SZ + OBSGP_OVERLAP_SZ;
    setting->overlap_size = OBSGP_OVERLAP_SZ;
    setting->margin = OBSGP_MARGIN;
    setting->init_variance = 1.0e6;
    setting->sensor_range_var = OBSGP_NOISE_PARAM;
    setting->discontinuity_var = OBSGP_DISCON_NOISE_PARAM;
    setting->max_valid_range_var = GPISMAP_OBS_VAR_THRE;
    setting->sensor_frame->valid_range_min = RANGE_MIN;
    setting->sensor_frame->valid_range_max = RANGE_MAX;
    setting->sensor_frame->angle_min = df.angles[0];
    setting->sensor_frame->angle_max = df.angles[n - 1];
    setting->sensor_frame->num_rays = n;
    setting->gp->kernel_type = type_name<erl::covariance::OrnsteinUhlenbeck1d>();
    setting->gp->kernel->scale = OBSGP_SCALE_PARAM;
    setting->mapping->type = MappingType::kIdentity;
    setting->partition_on_hit_rays = false;
    setting->symmetric_partitions = false;
    std::cout << *setting << std::endl;

    auto lidar_gp = std::make_shared<LidarGp2D>(setting);
    Logging::Info("Train:");
    {
        ERL_BLOCK_TIMER_MSG("gp.Train");
        ASSERT_TRUE(
            lidar_gp->Train(Eigen::Matrix2d::Identity(), Eigen::Vector2d::Zero(), df.distances));
    }

    Eigen::VectorXd distance_pred(df.distances.size()), distance_pred_var(distance_pred.size());
    {
        ERL_BLOCK_TIMER_MSG("gp.Test");
        auto test_result = lidar_gp->Test(df.angles, false /*angles_are_local*/, true /*un_map*/);
        Eigen::VectorXb success = test_result->GetMean(distance_pred, true /*parallel*/);
        ASSERT_TRUE(success.any());
        success = test_result->GetVariance(distance_pred_var, true);
        ASSERT_TRUE(success.any());
    }

    Eigen::VectorXd error = distance_pred - df.distances;
    double mae = error.cwiseAbs().mean();
    ERL_INFO("mean absolute error: {}", mae);

    PlplotFig fig(640, 480, true);
    PlplotFig::LegendOpt legend_opt(2, {"train", "pred"});
    legend_opt.SetTextColors({PlplotFig::Color0::Red, PlplotFig::Color0::Green})
        .SetStyles({PL_LEGEND_LINE, PL_LEGEND_LINE})
        .SetLineColors(legend_opt.text_colors)
        .SetLineStyles({1, 1})
        .SetLineWidths({1.0, 1.0})
        .SetBoxStyle(PL_LEGEND_BOUNDING_BOX)
        .SetBgColor0(PlplotFig::Color0::Gray)
        .SetLegendBoxLineColor0(PlplotFig::Color0::Black);
    fig.Clear(1.0, 1.0, 1.0)
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(
            df.angles.minCoeff() - 0.1,
            df.angles.maxCoeff() + 0.1,
            std::min(df.distances.minCoeff(), distance_pred.minCoeff()) - 0.1,
            std::max(df.distances.maxCoeff(), distance_pred.maxCoeff()) + 0.1)
        .SetCurrentColor(PlplotFig::Color0::Black)
        .DrawAxesBox(PlplotFig::AxisOpt(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .SetCurrentColor(PlplotFig::Color0::Red)
        .SetLineStyle(1)
        .SetPenWidth(3)
        .DrawLine(n, df.angles.data(), df.distances.data())
        .SetCurrentColor(PlplotFig::Color0::Green)
        .SetPenWidth(2)
        .DrawLine(n, df.angles.data(), distance_pred.data())
        .Legend(legend_opt);

    try {
        cv::imshow(fmt::format("{}: pred", test_info_->name()), fig.ToCvMat());
    } catch (const std::exception &e) { ERL_WARN("Failed to show image: {}", e.what()); }
    cv::imwrite(test_output_dir / "lidar_gp_2d_pred.png", fig.ToCvMat());

    legend_opt.SetTexts({"error", "variance"})
        .SetTextColors({PlplotFig::Color0::Blue, PlplotFig::Color0::Magenta})
        .SetLineColors(legend_opt.text_colors);
    fig.Clear(1.0, 1.0, 1.0)
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(
            df.angles.minCoeff() - 0.1,
            df.angles.maxCoeff() + 0.1,
            error.minCoeff() - 0.1,
            error.maxCoeff() + 0.1)
        .SetCurrentColor(PlplotFig::Color0::Black)
        .DrawAxesBox(
            PlplotFig::AxisOpt().DrawTopRightEdge(),
            PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelY("error")
        .SetCurrentColor(PlplotFig::Color0::Blue)
        .SetPenWidth(1)
        .DrawLine(n, df.angles.data(), error.data())
        .SetCurrentColor(PlplotFig::Color0::Black)
        .SetAxisLimits(
            df.angles.minCoeff() - 0.1,
            df.angles.maxCoeff() + 0.1,
            distance_pred_var.minCoeff() - 0.01,
            distance_pred_var.maxCoeff() + 0.01)
        .DrawAxesBox(
            PlplotFig::AxisOpt::Off(),
            PlplotFig::AxisOpt::Off()
                .DrawTopRightEdge()
                .DrawTopRightTickLabels()
                .DrawTickMajor()
                .DrawTickMinor()
                .DrawPerpendicularTickLabels())
        .SetAxisLabelY("variance", true)
        .SetPenWidth(1)
        .SetCurrentColor(PlplotFig::Color0::Magenta)
        .DrawLine(n, df.angles.data(), distance_pred_var.data())
        .Legend(legend_opt);

    try {
        cv::imshow(fmt::format("{}: error", test_info_->name()), fig.ToCvMat());
        cv::waitKey(1000);
    } catch (const std::exception &e) { ERL_WARN("Failed to show image: {}", e.what()); }
    cv::imwrite(test_output_dir / "lidar_gp_2d_error.png", fig.ToCvMat());

    // ASSERT_TRUE(mae < 0.022);  // 0.02135875277600203 (when discontinuity_detection is off)
    ASSERT_TRUE(mae < 0.08);  // 0.07934015284014925

    ASSERT_TRUE(Serialization<LidarGp2D>::Write("lidar_gp_2d.bin", lidar_gp));
    LidarGp2D lidar_gp_read(std::make_shared<LidarGp2D::Setting>());
    ASSERT_TRUE(Serialization<LidarGp2D>::Read("lidar_gp_2d.bin", &lidar_gp_read));
    EXPECT_TRUE(*lidar_gp == lidar_gp_read);
}

int
main(int argc, char *argv[]) {
    Init();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

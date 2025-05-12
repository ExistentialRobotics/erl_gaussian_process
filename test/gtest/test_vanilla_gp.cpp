#include "erl_common/block_timer.hpp"
#include "erl_common/plplot_fig.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_covariance/radial_bias_function.hpp"
#include "erl_gaussian_process/vanilla_gp.hpp"

using namespace erl::common;
using namespace erl::gaussian_process;

constexpr double kNoiseVar = 0.001;

TEST(VanillaGaussianProcess, SingleInputSingleOutput) {
    GTEST_PREPARE_OUTPUT_DIR();
    constexpr long n = 100;
    const auto setting = std::make_shared<VanillaGaussianProcessD::Setting>();
    setting->kernel->scale = 0.5;
    setting->kernel->x_dim = 1;
    setting->kernel_type = type_name<erl::covariance::RadialBiasFunction1d>();
    setting->max_num_samples = n;
    VanillaGaussianProcessD gp(setting);

    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(n, 0, 2 * M_PI);
    Eigen::VectorXd y = x.unaryExpr([](const double a) { return std::sin(a); });

    {
        ERL_BLOCK_TIMER_MSG("gp.Train");
        gp.Reset(n, 1, 1);
        auto &train_set = gp.GetTrainSet();
        train_set.x.row(0).head(n) = x.transpose();
        train_set.y.col(0).head(n) = y;
        train_set.var.head(n).setConstant(kNoiseVar);
        train_set.num_samples = n;
        ASSERT_TRUE(gp.Train());
    }

    constexpr long n_test = 200;
    Eigen::VectorXd x_test = Eigen::VectorXd::LinSpaced(n_test, 0, 2 * M_PI);
    Eigen::VectorXd y_test = x_test.unaryExpr([](const double a) { return std::sin(a); });
    Eigen::VectorXd y_pred(n_test);
    {
        ERL_BLOCK_TIMER_MSG("gp.Test");
        auto test_result = gp.Test(x_test.transpose());
        ASSERT_TRUE(test_result != nullptr);
        test_result->GetMean(0, y_pred, true);
    }
    Eigen::VectorXd error = y_pred - y_test;

    PlplotFig fig(640, 480, true);
    PlplotFig::LegendOpt legend_opt(4, {"train", "test g.t.", "prediction", "error"});
    legend_opt
        .SetTextColors(
            {PlplotFig::Color0::Red,
             PlplotFig::Color0::Green,
             PlplotFig::Color0::Blue,
             PlplotFig::Color0::Yellow})
        .SetStyles({PL_LEGEND_LINE, PL_LEGEND_LINE, PL_LEGEND_LINE, PL_LEGEND_LINE})
        .SetLineColors(legend_opt.text_colors)
        .SetLineStyles({1, 1, 2, 1})
        .SetLineWidths({1.0, 1.0, 1.0, 1.0})
        .SetBoxStyle(PL_LEGEND_BOUNDING_BOX)
        .SetBgColor0(PlplotFig::Color0::Gray)
        .SetLegendBoxLineColor0(PlplotFig::Color0::White);
    fig.Clear()  //
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(-0.1, 2 * M_PI + 0.1, -1.1, 1.1)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(
            PlplotFig::AxisOpt().DrawTopRightEdge(),
            PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .SetCurrentColor(PlplotFig::Color0::Red)
        .SetLineStyle(1)
        .DrawLine(n, x.data(), y.data())
        .SetCurrentColor(PlplotFig::Color0::Green)
        .SetLineStyle(1)
        .DrawLine(n_test, x_test.data(), y_test.data())
        .SetCurrentColor(PlplotFig::Color0::Blue)
        .SetLineStyle(2)
        .DrawLine(n_test, x_test.data(), y_pred.data())
        .SetAxisLimits(-0.1, 2 * M_PI + 0.1, error.minCoeff() - 0.001, error.maxCoeff() + 0.001)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(
            PlplotFig::AxisOpt::Off(),
            PlplotFig::AxisOpt::Off()
                .DrawTopRightEdge()
                .DrawTopRightTickLabels()
                .DrawTickMajor()
                .DrawTickMinor()
                .DrawPerpendicularTickLabels())
        .SetAxisLabelY("error", true)
        .SetCurrentColor(PlplotFig::Color0::Yellow)
        .SetLineStyle(1)
        .DrawLine(n_test, x_test.data(), error.data())
        .Legend(legend_opt);

    cv::imshow(test_info_->name(), fig.ToCvMat());
    cv::imwrite(test_output_dir / "test.png", fig.ToCvMat());
    cv::waitKey(1000);

    const double mae = error.cwiseAbs().mean();
    ERL_INFO("mean absolute error: {}.", mae);  // 0.00024246430481069056
    ASSERT_TRUE(mae < 3.0e-4);

    ASSERT_TRUE(Serialization<VanillaGaussianProcessD>::Write("vanilla_gp.bin", gp));
    VanillaGaussianProcessD gp_read(std::make_shared<VanillaGaussianProcessD::Setting>());
    ASSERT_TRUE(Serialization<VanillaGaussianProcessD>::Read("vanilla_gp.bin", gp_read));
    EXPECT_TRUE(gp == gp_read);
}

TEST(VanillaGaussianProcess, MultiInputSingleOutput) {
    GTEST_PREPARE_OUTPUT_DIR();
    auto compute_z = [](const Eigen::VectorXd &x,
                        const Eigen::VectorXd &y,
                        Eigen::VectorXd &z,
                        Eigen::Matrix2Xd &pts) {
        for (long xi = 0, i = 0; xi < x.size(); ++xi) {
            for (long yi = 0; yi < y.size(); ++yi, ++i) {
                pts.col(i) << x[xi], y[yi];
                z[i] = 2 * std::sin(10.0 * x[xi]) * std::cos(10.0 * y[yi]);
            }
        }
    };

    constexpr double x_min = -1.0;
    constexpr double x_max = 1.0;
    constexpr double y_min = -1.0;
    constexpr double y_max = 1.0;
    constexpr long n = 50;
    const auto setting = std::make_shared<VanillaGaussianProcessD::Setting>();
    setting->kernel->scale = 0.1;
    setting->kernel->x_dim = 2;
    setting->kernel_type = type_name<erl::covariance::RadialBiasFunction2d>();
    setting->max_num_samples = n * n;
    VanillaGaussianProcessD gp(setting);

    // Eigen::VectorXd x = Eigen::VectorXd::Random(n).array() * (x_max - x_min) + x_min;
    // Eigen::VectorXd y = Eigen::VectorXd::Random(n).array() * (y_max - y_min) + y_min;
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(n, x_min, x_max);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(n, y_min, y_max);
    Eigen::Matrix2Xd pts(2, n * n);
    Eigen::VectorXd z(n * n);
    compute_z(x, y, z, pts);

    {
        ERL_BLOCK_TIMER_MSG("gp.Train");
        gp.Reset(pts.cols(), 2, 1);
        auto &train_set = gp.GetTrainSet();
        train_set.x.topLeftCorner(2, pts.cols()) = pts;
        train_set.y.col(0).head(pts.cols()) = z;
        train_set.var.head(pts.cols()).setConstant(kNoiseVar);
        train_set.num_samples = pts.cols();
        ASSERT_TRUE(gp.Train());
    }

    constexpr long n_test = 100;
    x = Eigen::VectorXd::LinSpaced(n_test, x_min, x_max);
    y = Eigen::VectorXd::LinSpaced(n_test, y_min, y_max);
    pts.resize(2, n_test * n_test);
    Eigen::VectorXd z_test(n_test * n_test);
    compute_z(x, y, z_test, pts);
    Eigen::VectorXd z_pred(z_test.size());
    {
        ERL_BLOCK_TIMER_MSG("gp.Test");
        auto test_result = gp.Test(pts);
        ASSERT_TRUE(test_result != nullptr);
        test_result->GetMean(0, z_pred, true);
    }
    Eigen::MatrixXd error = z_pred - z_test;

    PlplotFig fig(640, 480, true);
    PlplotFig::ShadesOpt shades_opt;
    shades_opt.SetColorLevels(z_pred.data(), n_test, n_test, 127);
    PlplotFig::ColorBarOpt color_bar_opt;
    color_bar_opt.SetLabelOpts({PL_COLORBAR_LABEL_BOTTOM})
        .SetLabelTexts({"z"})
        .AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()  //
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(
            PlplotFig::AxisOpt().DrawTopRightEdge(),
            PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .SetColorMap(1, PlplotFig::ColorMap::Jet)
        .Shades(z_pred.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": prediction"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "prediction.png", fig.ToCvMat());

    shades_opt.SetColorLevels(error.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"error"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(
            PlplotFig::AxisOpt().DrawTopRightEdge(),
            PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .SetColorMap(1, PlplotFig::ColorMap::Jet)
        .Shades(error.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": error"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "error.png", fig.ToCvMat());

    cv::waitKey(1000);

    const double mae = error.cwiseAbs().mean();
    ERL_INFO("mean absolute error: {}.", mae);  // 0.0005035569336460338
    ASSERT_TRUE(mae < 5.1e-4);

    ASSERT_TRUE(Serialization<VanillaGaussianProcessD>::Write("vanilla_gp.bin", gp));
    VanillaGaussianProcessD gp_read(std::make_shared<VanillaGaussianProcessD::Setting>());
    ASSERT_TRUE(Serialization<VanillaGaussianProcessD>::Read("vanilla_gp.bin", gp_read));
    EXPECT_TRUE(gp == gp_read);
}

TEST(VanillaGaussianProcess, MultiInputMultiOutput) {
    GTEST_PREPARE_OUTPUT_DIR();
    auto compute_z = [](const Eigen::VectorXd &x,
                        const Eigen::VectorXd &y,
                        Eigen::VectorXd &z1,
                        Eigen::VectorXd &z2,
                        Eigen::Matrix2Xd &pts) {
        for (long xi = 0, i = 0; xi < x.size(); ++xi) {
            for (long yi = 0; yi < y.size(); ++yi, ++i) {
                pts.col(i) << x[xi], y[yi];
                z1[i] = 2 * std::sin(10.0 * x[xi]) * std::cos(10.0 * y[yi]);
                z2[i] = 3 * (std::sin(10.0 * x[xi]) + std::cos(10.0 * y[yi]));
            }
        }
    };

    constexpr double x_min = -1.0;
    constexpr double x_max = 1.0;
    constexpr double y_min = -1.0;
    constexpr double y_max = 1.0;
    constexpr long n = 50;
    const auto setting = std::make_shared<VanillaGaussianProcessD::Setting>();
    setting->kernel->scale = 5 * x_max / static_cast<double>(n);
    setting->kernel->x_dim = 2;
    setting->kernel_type = type_name<erl::covariance::RadialBiasFunction2d>();
    setting->max_num_samples = n * n;
    VanillaGaussianProcessD gp(setting);

    // Eigen::VectorXd x = Eigen::VectorXd::Random(n).array() * (x_max - x_min) + x_min;
    // Eigen::VectorXd y = Eigen::VectorXd::Random(n).array() * (y_max - y_min) + y_min;
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(n, x_min, x_max);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(n, y_min, y_max);
    Eigen::Matrix2Xd pts(2, n * n);
    Eigen::VectorXd z1(n * n);
    Eigen::VectorXd z2(n * n);
    compute_z(x, y, z1, z2, pts);

    {
        ERL_BLOCK_TIMER_MSG("gp.Train");
        gp.Reset(pts.cols(), 2, 2);
        auto &train_set = gp.GetTrainSet();
        train_set.x.topLeftCorner(2, pts.cols()) = pts;
        train_set.y.col(0).head(pts.cols()) = z1;
        train_set.y.col(1).head(pts.cols()) = z2;
        train_set.var.head(pts.cols()).setConstant(kNoiseVar);
        train_set.num_samples = pts.cols();
        ASSERT_TRUE(gp.Train());
    }

    constexpr long n_test = 100;
    x = Eigen::VectorXd::LinSpaced(n_test, x_min, x_max);
    y = Eigen::VectorXd::LinSpaced(n_test, y_min, y_max);
    pts.resize(2, n_test * n_test);
    Eigen::VectorXd z1_test(n_test * n_test);
    Eigen::VectorXd z2_test(n_test * n_test);
    compute_z(x, y, z1_test, z2_test, pts);
    Eigen::VectorXd z1_pred(n_test * n_test);
    Eigen::VectorXd z2_pred(n_test * n_test);
    {
        ERL_BLOCK_TIMER_MSG("gp.Test");
        auto test_result = gp.Test(pts);
        ASSERT_TRUE(test_result != nullptr);
        test_result->GetMean(0, z1_pred, true);
        test_result->GetMean(1, z2_pred, true);
    }
    Eigen::VectorXd error1 = z1_pred - z1_test;
    Eigen::VectorXd error2 = z2_pred - z2_test;

    PlplotFig fig(640, 480, true);
    fig.SetColorMap(1, PlplotFig::ColorMap::Jet);
    PlplotFig::ShadesOpt shades_opt;
    shades_opt.SetColorLevels(z1_pred.data(), n_test, n_test, 127);
    PlplotFig::ColorBarOpt color_bar_opt;
    color_bar_opt.SetLabelOpts({PL_COLORBAR_LABEL_BOTTOM})
        .SetLabelTexts({"z1"})
        .AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(
            PlplotFig::AxisOpt().DrawTopRightEdge(),
            PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .Shades(z1_pred.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": prediction z1"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "prediction_z1.png", fig.ToCvMat());

    shades_opt.SetColorLevels(z2_pred.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"z2"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(
            PlplotFig::AxisOpt().DrawTopRightEdge(),
            PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .Shades(z2_pred.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": prediction z2"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "prediction_z2.png", fig.ToCvMat());

    shades_opt.SetColorLevels(error1.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"error1"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(
            PlplotFig::AxisOpt().DrawTopRightEdge(),
            PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .Shades(error1.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": error z1"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "error_z1.png", fig.ToCvMat());

    shades_opt.SetColorLevels(error2.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"error2"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(
            PlplotFig::AxisOpt().DrawTopRightEdge(),
            PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .Shades(error2.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": error z2"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "error_z2.png", fig.ToCvMat());

    cv::waitKey(1000);

    const double mae1 = error1.cwiseAbs().mean();
    const double mae2 = error2.cwiseAbs().mean();
    ERL_INFO("mean absolute error: {}, {}.", mae1, mae2);
    ASSERT_TRUE(mae1 < 5.1e-4);  // 0.0005035569336460478
    ASSERT_TRUE(mae2 < 1.2e-3);  // 0.0011257545588707807

    ASSERT_TRUE(Serialization<VanillaGaussianProcessD>::Write("vanilla_gp.bin", gp));
    VanillaGaussianProcessD gp_read(std::make_shared<VanillaGaussianProcessD::Setting>());
    ASSERT_TRUE(Serialization<VanillaGaussianProcessD>::Read("vanilla_gp.bin", gp_read));
    EXPECT_TRUE(gp == gp_read);
}

int
main(int argc, char *argv[]) {
    Init();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

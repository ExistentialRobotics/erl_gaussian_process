#include "erl_common/plplot_fig.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_covariance/radial_bias_function.hpp"
#include "erl_gaussian_process/vanilla_gp.hpp"
using namespace erl::common;
using namespace erl::gaussian_process;

constexpr double kKernelScale = 0.5;
constexpr double kNoiseVar = 0.001;

TEST(VanillaGaussianProcess, SingleInputSingleOutput) {
    constexpr long n = 100;
    const auto setting = std::make_shared<VanillaGaussianProcessD::Setting>();
    setting->kernel->scale = kKernelScale;
    setting->kernel->x_dim = 1;
    setting->kernel_type = type_name<erl::covariance::RadialBiasFunction1d>();
    setting->max_num_samples = n;
    VanillaGaussianProcessD gp(setting);

    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(n, 0, 2 * M_PI);
    Eigen::VectorXd y = x.unaryExpr([](const double a) { return std::sin(a); });

    ReportTime<std::chrono::microseconds>("ans", 10, false, [&]() -> void {
        gp.Reset(n, 1, 1);
        auto &train_set = gp.GetTrainSet();
        train_set.x.row(0).head(n) = x.transpose();
        train_set.y.col(0).head(n) = y;
        train_set.var.head(n).setConstant(kNoiseVar);
        train_set.num_samples = n;
        ASSERT_TRUE(gp.Train());
    });

    constexpr long n_test = 200;
    Eigen::VectorXd x_test = Eigen::VectorXd::LinSpaced(n_test, 0, 2 * M_PI);
    Eigen::VectorXd y_test = x_test.unaryExpr([](const double a) { return std::sin(a); });
    Eigen::VectorXd y_pred(n_test);
    ReportTime<std::chrono::microseconds>("ans", 10, false, [&]() -> void {
        Eigen::VectorXd vec_var_out;
        ASSERT_TRUE(gp.Test(x_test.transpose(), {0}, y_pred.transpose(), vec_var_out));
    });
    Eigen::VectorXd error = y_pred - y_test;

    PlplotFig fig(640, 480, true);
    PlplotFig::LegendOpt legend_opt(4, {"train", "test g.t.", "prediction", "error"});
    legend_opt.SetTextColors({PlplotFig::Color0::Red, PlplotFig::Color0::Green, PlplotFig::Color0::Blue, PlplotFig::Color0::Yellow})
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
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
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
            PlplotFig::AxisOpt::Off().DrawTopRightEdge().DrawTopRightTickLabels().DrawTickMajor().DrawTickMinor().DrawPerpendicularTickLabels())
        .SetAxisLabelY("error", true)
        .SetCurrentColor(PlplotFig::Color0::Yellow)
        .SetLineStyle(1)
        .DrawLine(n_test, x_test.data(), error.data())
        .Legend(legend_opt);

    cv::imshow(test_info_->name(), fig.ToCvMat());
    cv::waitKey(0);

    const double mae = error.cwiseAbs().mean();
    ERL_INFO("mean absolute error: {}.", mae);
    ASSERT_TRUE(mae < 1.0e-3);
}

TEST(VanillaGaussianProcess, MultiInputSingleOutput) {
    auto compute_z = [](const Eigen::VectorXd &x, const Eigen::VectorXd &y, Eigen::MatrixXd &z, Eigen::Matrix2Xd &pts) {
        for (long yi = 0, i = 0; yi < y.size(); ++yi) {
            for (long xi = 0; xi < x.size(); ++xi, ++i) {
                pts.col(i) << x[xi], y[yi];
                z(xi, yi) = 2 * std::sin(10.0 * x[xi]) * std::cos(10.0 * y[yi]);
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
    Eigen::MatrixXd z(n, n);
    compute_z(x, y, z, pts);

    ReportTime<std::chrono::microseconds>("ans", 1, false, [&]() -> void {
        gp.Reset(pts.cols(), 2, 1);
        auto &train_set = gp.GetTrainSet();
        train_set.x.topLeftCorner(2, pts.cols()) = pts;
        train_set.y.col(0).head(pts.cols()) = z.reshaped(pts.cols(), 1);
        train_set.var.head(pts.cols()).setConstant(kNoiseVar);
        train_set.num_samples = pts.cols();
        ASSERT_TRUE(gp.Train());
    });

    constexpr long n_test = 100;
    x = Eigen::VectorXd::LinSpaced(n_test, x_min, x_max);
    y = Eigen::VectorXd::LinSpaced(n_test, y_min, y_max);
    pts.resize(2, n_test * n_test);
    Eigen::MatrixXd z_test(n_test, n_test);
    compute_z(x, y, z_test, pts);
    Eigen::VectorXd z_pred_vec(z_test.size());
    ReportTime<std::chrono::microseconds>("ans", 1, false, [&]() -> void {
        Eigen::VectorXd vec_var_out;
        ASSERT_TRUE(gp.Test(pts, {0}, z_pred_vec.transpose(), vec_var_out));
    });
    Eigen::MatrixXd z_pred = z_pred_vec.reshaped(n_test, n_test);
    Eigen::MatrixXd error = z_pred - z_test;

    PlplotFig fig(640, 480, true);
    PlplotFig::ShadesOpt shades_opt;
    shades_opt.SetColorLevels(z_pred_vec.data(), n_test, n_test, 127);
    PlplotFig::ColorBarOpt color_bar_opt;
    color_bar_opt.SetLabelOpts({PL_COLORBAR_LABEL_BOTTOM}).SetLabelTexts({"z"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()  //
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .SetColorMap(1, PlplotFig::ColorMap::Jet)
        .Shades(z_pred_vec.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": prediction"), fig.ToCvMat());

    shades_opt.SetColorLevels(error.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"error"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .SetColorMap(1, PlplotFig::ColorMap::Jet)
        .Shades(error.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": error"), fig.ToCvMat());

    cv::waitKey(0);

    const double mae = error.cwiseAbs().mean();
    ERL_INFO("mean absolute error: {}.", mae);
    ASSERT_TRUE(mae < 2.0e-3);
}

TEST(VanillaGaussianProcess, MultiInputMultiOutput) {
    auto compute_z = [](const Eigen::VectorXd &x, const Eigen::VectorXd &y, Eigen::MatrixXd &z, Eigen::Matrix2Xd &pts) {
        for (long yi = 0, i = 0; yi < y.size(); ++yi) {
            for (long xi = 0; xi < x.size(); ++xi, ++i) {
                pts.col(i) << x[xi], y[yi];
                z(0, i) = 2 * std::sin(10.0 * x[xi]) * std::cos(10.0 * y[yi]);
                z(1, i) = 3 * (std::sin(10.0 * x[xi]) + std::cos(10.0 * y[yi]));
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
    Eigen::MatrixXd z(2, n * n);
    compute_z(x, y, z, pts);

    ReportTime<std::chrono::microseconds>("ans", 1, false, [&]() -> void {
        gp.Reset(pts.cols(), 2, 2);
        auto &train_set = gp.GetTrainSet();
        train_set.x.topLeftCorner(2, pts.cols()) = pts;
        train_set.y.topLeftCorner(pts.cols(), 2) = z.transpose();
        train_set.var.head(pts.cols()).setConstant(kNoiseVar);
        train_set.num_samples = pts.cols();
        ASSERT_TRUE(gp.Train());
    });

    constexpr long n_test = 100;
    x = Eigen::VectorXd::LinSpaced(n_test, x_min, x_max);
    y = Eigen::VectorXd::LinSpaced(n_test, y_min, y_max);
    pts.resize(2, n_test * n_test);
    Eigen::MatrixXd z_test(2, n_test * n_test);
    compute_z(x, y, z_test, pts);
    Eigen::MatrixXd z_pred_mat(2, z_test.cols());
    ReportTime<std::chrono::microseconds>("ans", 1, false, [&]() -> void {
        Eigen::VectorXd vec_var_out;
        ASSERT_TRUE(gp.Test(pts, {0, 1}, z_pred_mat, vec_var_out));
    });
    Eigen::MatrixXd z1_pred = z_pred_mat.row(0).reshaped(n_test, n_test);
    Eigen::MatrixXd z2_pred = z_pred_mat.row(1).reshaped(n_test, n_test);
    Eigen::MatrixXd error1 = (z_pred_mat.row(0) - z_test.row(0)).reshaped(n_test, n_test);
    Eigen::MatrixXd error2 = (z_pred_mat.row(1) - z_test.row(1)).reshaped(n_test, n_test);

    PlplotFig fig(640, 480, true);
    fig.SetColorMap(1, PlplotFig::ColorMap::Jet);
    PlplotFig::ShadesOpt shades_opt;
    shades_opt.SetColorLevels(z1_pred.data(), n_test, n_test, 127);
    PlplotFig::ColorBarOpt color_bar_opt;
    color_bar_opt.SetLabelOpts({PL_COLORBAR_LABEL_BOTTOM}).SetLabelTexts({"z1"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .Shades(z1_pred.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": prediction z1"), fig.ToCvMat());

    shades_opt.SetColorLevels(z2_pred.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"z2"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .Shades(z2_pred.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": prediction z2"), fig.ToCvMat());

    shades_opt.SetColorLevels(error1.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"error1"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .Shades(error1.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": error z1"), fig.ToCvMat());

    shades_opt.SetColorLevels(error2.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"error2"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .Shades(error2.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": error z2"), fig.ToCvMat());

    cv::waitKey(0);

    const double mae1 = error1.cwiseAbs().mean();
    const double mae2 = error2.cwiseAbs().mean();
    ERL_INFO("mean absolute error: {}, {}.", mae1, mae2);
    ASSERT_TRUE(mae1 < 2.0e-3);
    ASSERT_TRUE(mae2 < 2.0e-3);
}

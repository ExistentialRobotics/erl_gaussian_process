#include "erl_common/plplot_fig.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_covariance/radial_bias_function.hpp"
#include "erl_gaussian_process/noisy_input_gp.hpp"
using namespace erl::common;
using namespace erl::gaussian_process;

constexpr double kKernelScale = 0.1;
constexpr double kNoiseVar = 0.0001;

TEST(NoisyInputGaussianProcess, SingleInputSingleOutputWithGradientObservation) {
    constexpr long n = 100;
    const auto setting = std::make_shared<NoisyInputGaussianProcessD::Setting>();
    setting->kernel->scale = kKernelScale;
    setting->kernel->x_dim = 1;
    setting->kernel_type = type_name<erl::covariance::RadialBiasFunction1d>();
    setting->max_num_samples = n;
    setting->no_gradient_observation = false;
    NoisyInputGaussianProcessD gp(setting);

    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(n, 0, 2 * M_PI);
    Eigen::VectorXd y = x.unaryExpr([](const double a) { return std::sin(a); });
    Eigen::VectorXd grad = x.unaryExpr([](const double a) { return std::cos(a); });

    ReportTime<std::chrono::microseconds>("ans", 1, false, [&]() -> void {
        gp.Reset(n, 1, 1);
        auto &train_set = gp.GetTrainSet();
        train_set.x.row(0).head(n) = x.transpose();
        train_set.y.col(0).head(n) = y;
        train_set.grad.row(0).head(n) = grad.transpose();
        train_set.var_x.head(n).setConstant(kNoiseVar);
        train_set.var_y.head(n).setConstant(kNoiseVar);
        train_set.var_grad.head(n).setConstant(kNoiseVar);
        train_set.grad_flag.head(n).setConstant(1);
        train_set.num_samples = n;
        train_set.num_samples_with_grad = n;
        ASSERT_TRUE(gp.Train());
    });

    constexpr long n_test = 200;
    Eigen::VectorXd x_test = Eigen::VectorXd::LinSpaced(n_test, 0, 2 * M_PI);
    Eigen::VectorXd y_test = x_test.unaryExpr([](const double a) { return std::sin(a); });
    Eigen::VectorXd grad_test = x_test.unaryExpr([](const double a) { return std::cos(a); });
    Eigen::MatrixXd y_pred(2, n_test);
    ReportTime<std::chrono::microseconds>("ans", 1, false, [&]() -> void {
        Eigen::MatrixXd mat_var_out, mat_cov_out;
        ASSERT_TRUE(gp.Test(x_test.transpose(), {{0, true}}, y_pred, mat_var_out, mat_cov_out));
    });
    Eigen::VectorXd error = Eigen::VectorXd(y_pred.row(0).transpose()) - y_test;
    Eigen::VectorXd grad_error = y_pred.row(1).transpose() - grad_test;

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
    fig.Clear()
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
        .DrawLine(n_test, x_test.data(), Eigen::VectorXd(y_pred.row(0).transpose()).data())
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
    cv::imshow(test_info_->name() + std::string(": y"), fig.ToCvMat());

    legend_opt.SetTexts({"train grad", "test grad g.t.", "prediction grad", "error grad"})
        .SetTextColors({PlplotFig::Color0::Red, PlplotFig::Color0::Green, PlplotFig::Color0::Blue, PlplotFig::Color0::Yellow})
        .SetStyles({PL_LEGEND_LINE, PL_LEGEND_LINE, PL_LEGEND_LINE, PL_LEGEND_LINE})
        .SetLineColors(legend_opt.text_colors)
        .SetLineStyles({1, 1, 2, 1})
        .SetLineWidths({1.0, 1.0, 1.0, 1.0})
        .SetBoxStyle(PL_LEGEND_BOUNDING_BOX)
        .SetBgColor0(PlplotFig::Color0::Gray)
        .SetLegendBoxLineColor0(PlplotFig::Color0::White);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(-0.1, 2 * M_PI + 0.1, -1.1, 1.1)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("grad")
        .SetCurrentColor(PlplotFig::Color0::Red)
        .SetLineStyle(1)
        .DrawLine(n, x.data(), grad.data())
        .SetCurrentColor(PlplotFig::Color0::Green)
        .SetLineStyle(1)
        .DrawLine(n_test, x_test.data(), grad_test.data())
        .SetCurrentColor(PlplotFig::Color0::Blue)
        .SetLineStyle(2)
        .DrawLine(n_test, x_test.data(), Eigen::VectorXd(y_pred.row(1).transpose()).data())
        .SetAxisLimits(-0.1, 2 * M_PI + 0.1, error.minCoeff() - 0.001, error.maxCoeff() + 0.001)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(
            PlplotFig::AxisOpt::Off(),
            PlplotFig::AxisOpt::Off().DrawTopRightEdge().DrawTopRightTickLabels().DrawTickMajor().DrawTickMinor().DrawPerpendicularTickLabels())
        .SetAxisLabelY("error", true)
        .SetCurrentColor(PlplotFig::Color0::Yellow)
        .SetLineStyle(1)
        .DrawLine(n_test, x_test.data(), grad_error.data())
        .Legend(legend_opt);
    cv::imshow(test_info_->name() + std::string(": grad"), fig.ToCvMat());

    cv::waitKey(0);

    const double mae = error.cwiseAbs().mean();
    const double mae_grad = grad_error.cwiseAbs().mean();
    ERL_INFO("mean absolute error: {}, {}.", mae, mae_grad);
    ASSERT_TRUE(mae < 0.015);
    ASSERT_TRUE(mae_grad < 0.007);
}

TEST(NoisyInputGaussianProcess, SingleInputSingleOutputWithoutGradientObservation) {
    constexpr long n = 100;
    const auto setting = std::make_shared<NoisyInputGaussianProcessD::Setting>();
    setting->kernel->scale = kKernelScale;
    setting->kernel->x_dim = 1;
    setting->kernel_type = type_name<erl::covariance::RadialBiasFunction1d>();
    setting->max_num_samples = n;
    setting->no_gradient_observation = true;
    NoisyInputGaussianProcessD gp(setting);

    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(n, 0, 2 * M_PI);
    Eigen::VectorXd y = x.unaryExpr([](const double a) { return std::sin(a); });
    Eigen::VectorXd grad = x.unaryExpr([](const double a) { return std::cos(a); });

    ReportTime<std::chrono::microseconds>("ans", 10, false, [&]() -> void {
        gp.Reset(n, 1, 1);
        auto &train_set = gp.GetTrainSet();
        train_set.x.row(0).head(n) = x.transpose();
        train_set.y.col(0).head(n) = y;
        // train_set.grad.row(0).head(n) = grad.transpose();
        train_set.var_x.head(n).setConstant(kNoiseVar);
        train_set.var_y.head(n).setConstant(kNoiseVar);
        // train_set.var_grad.head(n).setConstant(kNoiseVar);
        train_set.grad_flag.head(n).setConstant(0);
        train_set.num_samples = n;
        train_set.num_samples_with_grad = 0;
        ASSERT_TRUE(gp.Train());
    });

    constexpr long n_test = 200;
    Eigen::VectorXd x_test = Eigen::VectorXd::LinSpaced(n_test, 0, 2 * M_PI);
    Eigen::VectorXd y_test = x_test.unaryExpr([](const double a) { return std::sin(a); });
    Eigen::VectorXd grad_test = x_test.unaryExpr([](const double a) { return std::cos(a); });
    Eigen::MatrixXd y_pred(2, n_test);
    ReportTime<std::chrono::microseconds>("ans", 10, false, [&]() -> void {
        Eigen::MatrixXd mat_var_out, mat_cov_out;
        ASSERT_TRUE(gp.Test(x_test.transpose(), {{0, true}}, y_pred, mat_var_out, mat_cov_out));
    });
    Eigen::VectorXd error = Eigen::VectorXd(y_pred.row(0).transpose()) - y_test;
    Eigen::VectorXd grad_error = y_pred.row(1).transpose() - grad_test;

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
    fig.Clear()
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
        .DrawLine(n_test, x_test.data(), Eigen::VectorXd(y_pred.row(0).transpose()).data())
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
    cv::imshow(test_info_->name() + std::string(": y"), fig.ToCvMat());

    legend_opt.SetTexts({"train grad", "test grad g.t.", "prediction grad", "error grad"})
        .SetTextColors({PlplotFig::Color0::Red, PlplotFig::Color0::Green, PlplotFig::Color0::Blue, PlplotFig::Color0::Yellow})
        .SetStyles({PL_LEGEND_LINE, PL_LEGEND_LINE, PL_LEGEND_LINE, PL_LEGEND_LINE})
        .SetLineColors(legend_opt.text_colors)
        .SetLineStyles({1, 1, 2, 1})
        .SetLineWidths({1.0, 1.0, 1.0, 1.0})
        .SetBoxStyle(PL_LEGEND_BOUNDING_BOX)
        .SetBgColor0(PlplotFig::Color0::Gray)
        .SetLegendBoxLineColor0(PlplotFig::Color0::White);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(-0.1, 2 * M_PI + 0.1, -1.1, 1.1)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("grad")
        .SetCurrentColor(PlplotFig::Color0::Red)
        .SetLineStyle(1)
        .DrawLine(n, x.data(), grad.data())
        .SetCurrentColor(PlplotFig::Color0::Green)
        .SetLineStyle(1)
        .DrawLine(n_test, x_test.data(), grad_test.data())
        .SetCurrentColor(PlplotFig::Color0::Blue)
        .SetLineStyle(2)
        .DrawLine(n_test, x_test.data(), Eigen::VectorXd(y_pred.row(1).transpose()).data())
        .SetAxisLimits(-0.1, 2 * M_PI + 0.1, error.minCoeff() - 0.001, error.maxCoeff() + 0.001)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(
            PlplotFig::AxisOpt::Off(),
            PlplotFig::AxisOpt::Off().DrawTopRightEdge().DrawTopRightTickLabels().DrawTickMajor().DrawTickMinor().DrawPerpendicularTickLabels())
        .SetAxisLabelY("error", true)
        .SetCurrentColor(PlplotFig::Color0::Yellow)
        .SetLineStyle(1)
        .DrawLine(n_test, x_test.data(), grad_error.data())
        .Legend(legend_opt);
    cv::imshow(test_info_->name() + std::string(": grad"), fig.ToCvMat());

    cv::waitKey(0);

    const double mae = error.cwiseAbs().mean();
    const double mae_grad = grad_error.cwiseAbs().mean();
    ERL_INFO("mean absolute error: {}, {}.", mae, mae_grad);
    ASSERT_TRUE(mae < 0.016);
    ASSERT_TRUE(mae_grad < 0.021);
}

TEST(NoisyInputGaussianProcess, MultiInputSingleOutputWithGradientObservation) {
    auto compute_values =
        [](const Eigen::VectorXd &x, const Eigen::VectorXd &y, Eigen::MatrixXd &z, Eigen::MatrixXd &grad_x, Eigen::MatrixXd &grad_y, Eigen::Matrix2Xd &pts) {
            for (long yi = 0, i = 0; yi < y.size(); ++yi) {
                for (long xi = 0; xi < x.size(); ++xi, ++i) {
                    pts.col(i) << x[xi], y[yi];
                    z(xi, yi) = 2 * std::sin(10.0 * x[xi]) * std::cos(10.0 * y[yi]);
                    grad_x(xi, yi) = 20 * std::cos(10.0 * x[xi]) * std::cos(10.0 * y[yi]);
                    grad_y(xi, yi) = -20 * std::sin(10.0 * x[xi]) * std::sin(10.0 * y[yi]);
                }
            }
        };

    constexpr double x_min = -1.0;
    constexpr double x_max = 1.0;
    constexpr double y_min = -1.0;
    constexpr double y_max = 1.0;
    constexpr long n = 50;
    const auto setting = std::make_shared<NoisyInputGaussianProcessD::Setting>();
    setting->kernel->scale = 5 * x_max / static_cast<double>(n);
    setting->kernel->x_dim = 2;
    setting->kernel_type = type_name<erl::covariance::RadialBiasFunction2d>();
    setting->max_num_samples = n * n;
    setting->no_gradient_observation = false;
    NoisyInputGaussianProcessD gp(setting);

    // Eigen::VectorXd x = Eigen::VectorXd::Random(n).array() * (x_max - x_min) + x_min;
    // Eigen::VectorXd y = Eigen::VectorXd::Random(n).array() * (y_max - y_min) + y_min;
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(n, x_min, x_max);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(n, y_min, y_max);
    Eigen::Matrix2Xd pts(2, n * n);
    Eigen::MatrixXd z(n, n);
    Eigen::MatrixXd grad_x(n, n);
    Eigen::MatrixXd grad_y(n, n);
    compute_values(x, y, z, grad_x, grad_y, pts);

    ReportTime<std::chrono::microseconds>("ans", 1, false, [&]() -> void {
        gp.Reset(pts.cols(), 2, 1);
        auto &train_set = gp.GetTrainSet();
        train_set.x.topLeftCorner(2, pts.cols()) = pts;
        train_set.y.col(0).head(pts.cols()) = z.reshaped(pts.cols(), 1);
        train_set.grad.row(0).head(pts.cols()) = grad_x.reshaped(1, pts.cols());
        train_set.grad.row(1).head(pts.cols()) = grad_y.reshaped(1, pts.cols());
        const long n_pts = pts.cols();
        train_set.var_x.head(n_pts).setConstant(kNoiseVar);
        train_set.var_y.head(n_pts).setConstant(kNoiseVar);
        train_set.var_grad.head(n_pts).setConstant(kNoiseVar);
        train_set.grad_flag.head(n_pts).setConstant(1);
        train_set.num_samples = n_pts;
        train_set.num_samples_with_grad = n_pts;
        ASSERT_TRUE(gp.Train());
    });

    constexpr long n_test = 100;
    x = Eigen::VectorXd::LinSpaced(n_test, x_min, x_max);
    y = Eigen::VectorXd::LinSpaced(n_test, y_min, y_max);
    pts.resize(2, n_test * n_test);
    Eigen::MatrixXd z_test(n_test, n_test);
    Eigen::MatrixXd grad_x_test(n_test, n_test);
    Eigen::MatrixXd grad_y_test(n_test, n_test);
    compute_values(x, y, z_test, grad_x_test, grad_y_test, pts);
    Eigen::MatrixXd pred(3, z_test.size());
    ReportTime<std::chrono::microseconds>("ans", 1, false, [&]() -> void {
        Eigen::MatrixXd mat_var_out, mat_cov_out;
        ASSERT_TRUE(gp.Test(pts, {{0, true}}, pred, mat_var_out, mat_cov_out));
    });
    Eigen::MatrixXd z_pred = pred.row(0).reshaped(n_test, n_test);
    Eigen::MatrixXd grad_x_pred = pred.row(1).reshaped(n_test, n_test);
    Eigen::MatrixXd grad_y_pred = pred.row(2).reshaped(n_test, n_test);
    Eigen::MatrixXd error = z_pred - z_test;
    Eigen::MatrixXd grad_x_error = grad_x_pred - grad_x_test;
    Eigen::MatrixXd grad_y_error = grad_y_pred - grad_y_test;

    PlplotFig fig(640, 480, true);
    PlplotFig::ShadesOpt shades_opt;
    shades_opt.SetColorLevels(z_pred.data(), n_test, n_test, 127);
    PlplotFig::ColorBarOpt color_bar_opt;
    color_bar_opt.SetLabelOpts({PL_COLORBAR_LABEL_BOTTOM}).SetLabelTexts({"z"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .SetColorMap(1, PlplotFig::ColorMap::Jet)
        .Shades(z_pred.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": prediction"), fig.ToCvMat());

    shades_opt.SetColorLevels(grad_x_pred.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"grad_x"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .SetColorMap(1, PlplotFig::ColorMap::Jet)
        .Shades(grad_x_pred.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": grad_x"), fig.ToCvMat());

    shades_opt.SetColorLevels(grad_y_pred.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"grad_y"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .SetColorMap(1, PlplotFig::ColorMap::Jet)
        .Shades(grad_y_pred.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": grad_y"), fig.ToCvMat());

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

    shades_opt.SetColorLevels(grad_x_error.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"grad_x_error"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .SetColorMap(1, PlplotFig::ColorMap::Jet)
        .Shades(grad_x_error.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": grad_x_error"), fig.ToCvMat());

    shades_opt.SetColorLevels(grad_y_error.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"grad_y_error"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .SetColorMap(1, PlplotFig::ColorMap::Jet)
        .Shades(grad_y_error.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": grad_y_error"), fig.ToCvMat());

    cv::waitKey(0);

    const double mae = error.cwiseAbs().mean();
    const double mae_grad_x = grad_x_error.cwiseAbs().mean();
    const double mae_grad_y = grad_y_error.cwiseAbs().mean();
    ERL_INFO("mean absolute error: {}, {}, {}.", mae, mae_grad_x, mae_grad_y);
    ASSERT_TRUE(mae < 1.0e-5);
    ASSERT_TRUE(mae_grad_x < 1.6e-4);
    ASSERT_TRUE(mae_grad_y < 2.1e-4);
}

TEST(NoisyInputGaussianProcess, MultiInputSingleOutputWithoutGradientObservation) {
    auto compute_values =
        [](const Eigen::VectorXd &x, const Eigen::VectorXd &y, Eigen::MatrixXd &z, Eigen::MatrixXd &grad_x, Eigen::MatrixXd &grad_y, Eigen::Matrix2Xd &pts) {
            for (long yi = 0, i = 0; yi < y.size(); ++yi) {
                for (long xi = 0; xi < x.size(); ++xi, ++i) {
                    pts.col(i) << x[xi], y[yi];
                    z(xi, yi) = 2 * std::sin(10.0 * x[xi]) * std::cos(10.0 * y[yi]);
                    grad_x(xi, yi) = 20 * std::cos(10.0 * x[xi]) * std::cos(10.0 * y[yi]);
                    grad_y(xi, yi) = -20 * std::sin(10.0 * x[xi]) * std::sin(10.0 * y[yi]);
                }
            }
        };

    constexpr double x_min = -1.0;
    constexpr double x_max = 1.0;
    constexpr double y_min = -1.0;
    constexpr double y_max = 1.0;
    constexpr long n = 50;
    const auto setting = std::make_shared<NoisyInputGaussianProcessD::Setting>();
    setting->kernel->scale = 5 * x_max / static_cast<double>(n);
    setting->kernel->x_dim = 2;
    setting->kernel_type = type_name<erl::covariance::RadialBiasFunction2d>();
    setting->max_num_samples = n * n;
    setting->no_gradient_observation = true;
    NoisyInputGaussianProcessD gp(setting);

    // Eigen::VectorXd x = Eigen::VectorXd::Random(n).array() * (x_max - x_min) + x_min;
    // Eigen::VectorXd y = Eigen::VectorXd::Random(n).array() * (y_max - y_min) + y_min;
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(n, x_min, x_max);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(n, y_min, y_max);
    Eigen::Matrix2Xd pts(2, n * n);
    Eigen::MatrixXd z(n, n);
    Eigen::MatrixXd grad_x(n, n);
    Eigen::MatrixXd grad_y(n, n);
    compute_values(x, y, z, grad_x, grad_y, pts);

    ReportTime<std::chrono::microseconds>("ans", 1, false, [&]() -> void {
        gp.Reset(pts.cols(), 2, 1);
        auto &train_set = gp.GetTrainSet();
        train_set.x.topLeftCorner(2, pts.cols()) = pts;
        train_set.y.col(0).head(pts.cols()) = z.reshaped(pts.cols(), 1);
        const long n_pts = pts.cols();
        train_set.var_x.head(n_pts).setConstant(kNoiseVar);
        train_set.var_y.head(n_pts).setConstant(kNoiseVar);
        train_set.grad_flag.head(n_pts).setConstant(0);
        train_set.num_samples = n_pts;
        train_set.num_samples_with_grad = 0;
        ASSERT_TRUE(gp.Train());
    });

    constexpr long n_test = 100;
    x = Eigen::VectorXd::LinSpaced(n_test, x_min, x_max);
    y = Eigen::VectorXd::LinSpaced(n_test, y_min, y_max);
    pts.resize(2, n_test * n_test);
    Eigen::MatrixXd z_test(n_test, n_test);
    Eigen::MatrixXd grad_x_test(n_test, n_test);
    Eigen::MatrixXd grad_y_test(n_test, n_test);
    compute_values(x, y, z_test, grad_x_test, grad_y_test, pts);
    Eigen::MatrixXd pred(3, z_test.size());
    ReportTime<std::chrono::microseconds>("ans", 1, false, [&]() -> void {
        Eigen::MatrixXd mat_var_out, mat_cov_out;
        ASSERT_TRUE(gp.Test(pts, {{0, true}}, pred, mat_var_out, mat_cov_out));
    });
    Eigen::MatrixXd z_pred = pred.row(0).reshaped(n_test, n_test);
    Eigen::MatrixXd grad_x_pred = pred.row(1).reshaped(n_test, n_test);
    Eigen::MatrixXd grad_y_pred = pred.row(2).reshaped(n_test, n_test);
    Eigen::MatrixXd error = z_pred - z_test;
    Eigen::MatrixXd grad_x_error = grad_x_pred - grad_x_test;
    Eigen::MatrixXd grad_y_error = grad_y_pred - grad_y_test;

    PlplotFig fig(640, 480, true);
    PlplotFig::ShadesOpt shades_opt;
    shades_opt.SetColorLevels(z_pred.data(), n_test, n_test, 127);
    PlplotFig::ColorBarOpt color_bar_opt;
    color_bar_opt.SetLabelOpts({PL_COLORBAR_LABEL_BOTTOM}).SetLabelTexts({"z"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .SetColorMap(1, PlplotFig::ColorMap::Jet)
        .Shades(z_pred.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": prediction"), fig.ToCvMat());

    shades_opt.SetColorLevels(grad_x_pred.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"grad_x"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .SetColorMap(1, PlplotFig::ColorMap::Jet)
        .Shades(grad_x_pred.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": grad_x"), fig.ToCvMat());

    shades_opt.SetColorLevels(grad_y_pred.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"grad_y"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .SetColorMap(1, PlplotFig::ColorMap::Jet)
        .Shades(grad_y_pred.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": grad_y"), fig.ToCvMat());

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

    shades_opt.SetColorLevels(grad_x_error.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"grad_x_error"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .SetColorMap(1, PlplotFig::ColorMap::Jet)
        .Shades(grad_x_error.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": grad_x_error"), fig.ToCvMat());

    shades_opt.SetColorLevels(grad_y_error.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"grad_y_error"}).AddColorMap(0, shades_opt.color_levels, 10);
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(x_min, x_max, y_min, y_max)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
        .SetAxisLabelX("x")
        .SetAxisLabelY("y")
        .SetColorMap(1, PlplotFig::ColorMap::Jet)
        .Shades(grad_y_error.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": grad_y_error"), fig.ToCvMat());

    cv::waitKey(0);

    const double mae = error.cwiseAbs().mean();
    const double mae_grad_x = grad_x_error.cwiseAbs().mean();
    const double mae_grad_y = grad_y_error.cwiseAbs().mean();
    ERL_INFO("mean absolute error: {}, {}, {}.", mae, mae_grad_x, mae_grad_y);
    ASSERT_TRUE(mae < 1.0e-5);
    ASSERT_TRUE(mae_grad_x < 1.6e-4);
    ASSERT_TRUE(mae_grad_y < 2.1e-4);
}

TEST(NoisyInputGaussianProcess, MultiInputMultiOutputWithGradientObservation) {
    auto compute_values = [](const Eigen::VectorXd &x,
                             const Eigen::VectorXd &y,
                             Eigen::MatrixXd &z1,
                             Eigen::MatrixXd &z2,
                             Eigen::MatrixXd &grad1_x,
                             Eigen::MatrixXd &grad1_y,
                             Eigen::MatrixXd &grad2_x,
                             Eigen::MatrixXd &grad2_y,
                             Eigen::Matrix2Xd &pts) {
        for (long yi = 0, i = 0; yi < y.size(); ++yi) {
            for (long xi = 0; xi < x.size(); ++xi, ++i) {
                pts.col(i) << x[xi], y[yi];
                z1(xi, yi) = 2 * std::sin(10.0 * x[xi]) * std::cos(10.0 * y[yi]);
                z2(xi, yi) = 3 * (std::sin(10.0 * x[xi]) + std::cos(10.0 * y[yi]));
                grad1_x(xi, yi) = 20 * std::cos(10.0 * x[xi]) * std::cos(10.0 * y[yi]);
                grad1_y(xi, yi) = -20 * std::sin(10.0 * x[xi]) * std::sin(10.0 * y[yi]);
                grad2_x(xi, yi) = 30 * std::cos(10.0 * x[xi]);
                grad2_y(xi, yi) = -30 * std::sin(10.0 * y[yi]);
            }
        }
    };

    constexpr double x_min = -1.0;
    constexpr double x_max = 1.0;
    constexpr double y_min = -1.0;
    constexpr double y_max = 1.0;
    constexpr long n = 50;
    const auto setting = std::make_shared<NoisyInputGaussianProcessD::Setting>();
    setting->kernel->scale = 5 * x_max / static_cast<double>(n);
    setting->kernel->x_dim = 2;
    setting->kernel_type = type_name<erl::covariance::RadialBiasFunction2d>();
    setting->max_num_samples = n * n;
    setting->no_gradient_observation = false;
    NoisyInputGaussianProcessD gp(setting);

    // Eigen::VectorXd x = Eigen::VectorXd::Random(n).array() * (x_max - x_min) + x_min;
    // Eigen::VectorXd y = Eigen::VectorXd::Random(n).array() * (y_max - y_min) + y_min;
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(n, x_min, x_max);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(n, y_min, y_max);
    Eigen::Matrix2Xd pts(2, n * n);
    Eigen::MatrixXd z1(n, n);
    Eigen::MatrixXd z2(n, n);
    Eigen::MatrixXd grad1_x(n, n);
    Eigen::MatrixXd grad1_y(n, n);
    Eigen::MatrixXd grad2_x(n, n);
    Eigen::MatrixXd grad2_y(n, n);
    compute_values(x, y, z1, z2, grad1_x, grad1_y, grad2_x, grad2_y, pts);

    ReportTime<std::chrono::microseconds>("ans", 1, false, [&]() -> void {
        gp.Reset(pts.cols(), 2, 2);
        auto &train_set = gp.GetTrainSet();
        train_set.x.topLeftCorner(2, pts.cols()) = pts;
        train_set.y.col(0).head(pts.cols()) = z1.reshaped(pts.cols(), 1);
        train_set.y.col(1).head(pts.cols()) = z2.reshaped(pts.cols(), 1);
        train_set.grad.row(0).head(pts.cols()) = grad1_x.reshaped(1, pts.cols());
        train_set.grad.row(1).head(pts.cols()) = grad1_y.reshaped(1, pts.cols());
        train_set.grad.row(2).head(pts.cols()) = grad2_x.reshaped(1, pts.cols());
        train_set.grad.row(3).head(pts.cols()) = grad2_y.reshaped(1, pts.cols());
        const long n_pts = pts.cols();
        train_set.var_x.head(n_pts).setConstant(kNoiseVar);
        train_set.var_y.head(n_pts).setConstant(kNoiseVar);
        train_set.var_grad.head(n_pts).setConstant(kNoiseVar);
        train_set.grad_flag.head(n_pts).setConstant(1);
        train_set.num_samples = n_pts;
        train_set.num_samples_with_grad = n_pts;
        ASSERT_TRUE(gp.Train());
    });

    constexpr long n_test = 100;
    x = Eigen::VectorXd::LinSpaced(n_test, x_min, x_max);
    y = Eigen::VectorXd::LinSpaced(n_test, y_min, y_max);
    pts.resize(2, n_test * n_test);
    Eigen::MatrixXd z1_test(n_test, n_test);
    Eigen::MatrixXd z2_test(n_test, n_test);
    Eigen::MatrixXd grad1_x_test(n_test, n_test);
    Eigen::MatrixXd grad1_y_test(n_test, n_test);
    Eigen::MatrixXd grad2_x_test(n_test, n_test);
    Eigen::MatrixXd grad2_y_test(n_test, n_test);
    compute_values(x, y, z1_test, z2_test, grad1_x_test, grad1_y_test, grad2_x_test, grad2_y_test, pts);
    Eigen::MatrixXd pred(6, z1_test.size());
    ReportTime<std::chrono::microseconds>("ans", 1, false, [&]() -> void {
        Eigen::MatrixXd mat_var_out, mat_cov_out;
        ASSERT_TRUE(gp.Test(pts, {{0, true}, {1, true}}, pred, mat_var_out, mat_cov_out));
    });

    PlplotFig fig(640, 480, true);
    for (long d = 0; d < 2; ++d) {
        long i = d * 3;
        Eigen::MatrixXd z_pred = pred.row(i).reshaped(n_test, n_test);
        Eigen::MatrixXd grad_x_pred = pred.row(i + 1).reshaped(n_test, n_test);
        Eigen::MatrixXd grad_y_pred = pred.row(i + 2).reshaped(n_test, n_test);
        Eigen::MatrixXd error = z_pred - (d == 0 ? z1_test : z2_test);
        Eigen::MatrixXd grad_x_error = grad_x_pred - (d == 0 ? grad1_x_test : grad2_x_test);
        Eigen::MatrixXd grad_y_error = grad_y_pred - (d == 0 ? grad1_y_test : grad2_y_test);

        PlplotFig::ShadesOpt shades_opt;
        shades_opt.SetColorLevels(z_pred.data(), n_test, n_test, 127);
        PlplotFig::ColorBarOpt color_bar_opt;
        color_bar_opt.SetLabelOpts({PL_COLORBAR_LABEL_BOTTOM}).SetLabelTexts({"z"}).AddColorMap(0, shades_opt.color_levels, 10);
        fig.Clear()
            .SetMargin(0.15, 0.85, 0.15, 0.85)
            .SetAxisLimits(x_min, x_max, y_min, y_max)
            .SetCurrentColor(PlplotFig::Color0::White)
            .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
            .SetAxisLabelX("x")
            .SetAxisLabelY("y")
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(z_pred.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": pred{}", d + 1), fig.ToCvMat());

        shades_opt.SetColorLevels(grad_x_pred.data(), n_test, n_test, 127);
        color_bar_opt.SetLabelTexts({"grad_x"}).AddColorMap(0, shades_opt.color_levels, 10);
        fig.Clear()
            .SetMargin(0.15, 0.85, 0.15, 0.85)
            .SetAxisLimits(x_min, x_max, y_min, y_max)
            .SetCurrentColor(PlplotFig::Color0::White)
            .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
            .SetAxisLabelX("x")
            .SetAxisLabelY("y")
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(grad_x_pred.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": grad{}_x", d + 1), fig.ToCvMat());

        shades_opt.SetColorLevels(grad_y_pred.data(), n_test, n_test, 127);
        color_bar_opt.SetLabelTexts({"grad_y"}).AddColorMap(0, shades_opt.color_levels, 10);
        fig.Clear()
            .SetMargin(0.15, 0.85, 0.15, 0.85)
            .SetAxisLimits(x_min, x_max, y_min, y_max)
            .SetCurrentColor(PlplotFig::Color0::White)
            .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
            .SetAxisLabelX("x")
            .SetAxisLabelY("y")
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(grad_y_pred.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": grad{}_y", d + 1), fig.ToCvMat());

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
        cv::imshow(test_info_->name() + fmt::format(": error{}", d + 1), fig.ToCvMat());

        shades_opt.SetColorLevels(grad_x_error.data(), n_test, n_test, 127);
        color_bar_opt.SetLabelTexts({"grad_x_error"}).AddColorMap(0, shades_opt.color_levels, 10);
        fig.Clear()
            .SetMargin(0.15, 0.85, 0.15, 0.85)
            .SetAxisLimits(x_min, x_max, y_min, y_max)
            .SetCurrentColor(PlplotFig::Color0::White)
            .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
            .SetAxisLabelX("x")
            .SetAxisLabelY("y")
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(grad_x_error.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": grad{}_x_error", d + 1), fig.ToCvMat());

        shades_opt.SetColorLevels(grad_y_error.data(), n_test, n_test, 127);
        color_bar_opt.SetLabelTexts({"grad_y_error"}).AddColorMap(0, shades_opt.color_levels, 10);
        fig.Clear()
            .SetMargin(0.15, 0.85, 0.15, 0.85)
            .SetAxisLimits(x_min, x_max, y_min, y_max)
            .SetCurrentColor(PlplotFig::Color0::White)
            .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
            .SetAxisLabelX("x")
            .SetAxisLabelY("y")
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(grad_y_error.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": grad{}_y_error", d + 1), fig.ToCvMat());

        cv::waitKey(0);

        const double mae = error.cwiseAbs().mean();
        const double mae_grad_x = grad_x_error.cwiseAbs().mean();
        const double mae_grad_y = grad_y_error.cwiseAbs().mean();
        ERL_INFO("{}, mean absolute error: {}, {}, {}.", d + 1, mae, mae_grad_x, mae_grad_y);
        if (d == 0) {
            ASSERT_TRUE(mae < 1.0e-5);
            ASSERT_TRUE(mae_grad_x < 1.6e-4);
            ASSERT_TRUE(mae_grad_y < 2.1e-4);
        } else {
            ASSERT_TRUE(mae < 4.0e-5);
            ASSERT_TRUE(mae_grad_x < 4.7e-4);
            ASSERT_TRUE(mae_grad_y < 8.0e-4);
        }
    }
}

TEST(NoisyInputGaussianProcess, MultiInputMultiOutputWithoutGradientObservation) {
    auto compute_values = [](const Eigen::VectorXd &x,
                             const Eigen::VectorXd &y,
                             Eigen::MatrixXd &z1,
                             Eigen::MatrixXd &z2,
                             Eigen::MatrixXd &grad1_x,
                             Eigen::MatrixXd &grad1_y,
                             Eigen::MatrixXd &grad2_x,
                             Eigen::MatrixXd &grad2_y,
                             Eigen::Matrix2Xd &pts) {
        for (long yi = 0, i = 0; yi < y.size(); ++yi) {
            for (long xi = 0; xi < x.size(); ++xi, ++i) {
                pts.col(i) << x[xi], y[yi];
                z1(xi, yi) = 2 * std::sin(10.0 * x[xi]) * std::cos(10.0 * y[yi]);
                z2(xi, yi) = 3 * (std::sin(10.0 * x[xi]) + std::cos(10.0 * y[yi]));
                grad1_x(xi, yi) = 20 * std::cos(10.0 * x[xi]) * std::cos(10.0 * y[yi]);
                grad1_y(xi, yi) = -20 * std::sin(10.0 * x[xi]) * std::sin(10.0 * y[yi]);
                grad2_x(xi, yi) = 30 * std::cos(10.0 * x[xi]);
                grad2_y(xi, yi) = -30 * std::sin(10.0 * y[yi]);
            }
        }
    };

    constexpr double x_min = -1.0;
    constexpr double x_max = 1.0;
    constexpr double y_min = -1.0;
    constexpr double y_max = 1.0;
    constexpr long n = 50;
    const auto setting = std::make_shared<NoisyInputGaussianProcessD::Setting>();
    setting->kernel->scale = 5 * x_max / static_cast<double>(n);
    setting->kernel->x_dim = 2;
    setting->kernel_type = type_name<erl::covariance::RadialBiasFunction2d>();
    setting->max_num_samples = n * n;
    setting->no_gradient_observation = true;
    NoisyInputGaussianProcessD gp(setting);

    // Eigen::VectorXd x = Eigen::VectorXd::Random(n).array() * (x_max - x_min) + x_min;
    // Eigen::VectorXd y = Eigen::VectorXd::Random(n).array() * (y_max - y_min) + y_min;
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(n, x_min, x_max);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(n, y_min, y_max);
    Eigen::Matrix2Xd pts(2, n * n);
    Eigen::MatrixXd z1(n, n);
    Eigen::MatrixXd z2(n, n);
    Eigen::MatrixXd grad1_x(n, n);
    Eigen::MatrixXd grad1_y(n, n);
    Eigen::MatrixXd grad2_x(n, n);
    Eigen::MatrixXd grad2_y(n, n);
    compute_values(x, y, z1, z2, grad1_x, grad1_y, grad2_x, grad2_y, pts);

    ReportTime<std::chrono::microseconds>("ans", 1, false, [&]() -> void {
        gp.Reset(pts.cols(), 2, 2);
        auto &train_set = gp.GetTrainSet();
        train_set.x.topLeftCorner(2, pts.cols()) = pts;
        train_set.y.col(0).head(pts.cols()) = z1.reshaped(pts.cols(), 1);
        train_set.y.col(1).head(pts.cols()) = z2.reshaped(pts.cols(), 1);
        // train_set.grad.row(0).head(pts.cols()) = grad1_x.reshaped(1, pts.cols());
        // train_set.grad.row(1).head(pts.cols()) = grad1_y.reshaped(1, pts.cols());
        // train_set.grad.row(2).head(pts.cols()) = grad2_x.reshaped(1, pts.cols());
        // train_set.grad.row(3).head(pts.cols()) = grad2_y.reshaped(1, pts.cols());
        const long n_pts = pts.cols();
        train_set.var_x.head(n_pts).setConstant(kNoiseVar);
        train_set.var_y.head(n_pts).setConstant(kNoiseVar);
        // train_set.var_grad.head(n_pts).setConstant(kNoiseVar);
        train_set.grad_flag.head(n_pts).setConstant(0);
        train_set.num_samples = n_pts;
        train_set.num_samples_with_grad = 0;
        ASSERT_TRUE(gp.Train());
    });

    constexpr long n_test = 100;
    x = Eigen::VectorXd::LinSpaced(n_test, x_min, x_max);
    y = Eigen::VectorXd::LinSpaced(n_test, y_min, y_max);
    pts.resize(2, n_test * n_test);
    Eigen::MatrixXd z1_test(n_test, n_test);
    Eigen::MatrixXd z2_test(n_test, n_test);
    Eigen::MatrixXd grad1_x_test(n_test, n_test);
    Eigen::MatrixXd grad1_y_test(n_test, n_test);
    Eigen::MatrixXd grad2_x_test(n_test, n_test);
    Eigen::MatrixXd grad2_y_test(n_test, n_test);
    compute_values(x, y, z1_test, z2_test, grad1_x_test, grad1_y_test, grad2_x_test, grad2_y_test, pts);
    Eigen::MatrixXd pred(6, z1_test.size());
    ReportTime<std::chrono::microseconds>("ans", 1, false, [&]() -> void {
        Eigen::MatrixXd mat_var_out, mat_cov_out;
        ASSERT_TRUE(gp.Test(pts, {{0, true}, {1, true}}, pred, mat_var_out, mat_cov_out));
    });

    PlplotFig fig(640, 480, true);
    for (long d = 0; d < 2; ++d) {
        long i = d * 3;
        Eigen::MatrixXd z_pred = pred.row(i).reshaped(n_test, n_test);
        Eigen::MatrixXd grad_x_pred = pred.row(i + 1).reshaped(n_test, n_test);
        Eigen::MatrixXd grad_y_pred = pred.row(i + 2).reshaped(n_test, n_test);
        Eigen::MatrixXd error = z_pred - (d == 0 ? z1_test : z2_test);
        Eigen::MatrixXd grad_x_error = grad_x_pred - (d == 0 ? grad1_x_test : grad2_x_test);
        Eigen::MatrixXd grad_y_error = grad_y_pred - (d == 0 ? grad1_y_test : grad2_y_test);

        PlplotFig::ShadesOpt shades_opt;
        shades_opt.SetColorLevels(z_pred.data(), n_test, n_test, 127);
        PlplotFig::ColorBarOpt color_bar_opt;
        color_bar_opt.SetLabelOpts({PL_COLORBAR_LABEL_BOTTOM}).SetLabelTexts({"z"}).AddColorMap(0, shades_opt.color_levels, 10);
        fig.Clear()
            .SetMargin(0.15, 0.85, 0.15, 0.85)
            .SetAxisLimits(x_min, x_max, y_min, y_max)
            .SetCurrentColor(PlplotFig::Color0::White)
            .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
            .SetAxisLabelX("x")
            .SetAxisLabelY("y")
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(z_pred.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": pred{}", d + 1), fig.ToCvMat());

        shades_opt.SetColorLevels(grad_x_pred.data(), n_test, n_test, 127);
        color_bar_opt.SetLabelTexts({"grad_x"}).AddColorMap(0, shades_opt.color_levels, 10);
        fig.Clear()
            .SetMargin(0.15, 0.85, 0.15, 0.85)
            .SetAxisLimits(x_min, x_max, y_min, y_max)
            .SetCurrentColor(PlplotFig::Color0::White)
            .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
            .SetAxisLabelX("x")
            .SetAxisLabelY("y")
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(grad_x_pred.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": grad{}_x", d + 1), fig.ToCvMat());

        shades_opt.SetColorLevels(grad_y_pred.data(), n_test, n_test, 127);
        color_bar_opt.SetLabelTexts({"grad_y"}).AddColorMap(0, shades_opt.color_levels, 10);
        fig.Clear()
            .SetMargin(0.15, 0.85, 0.15, 0.85)
            .SetAxisLimits(x_min, x_max, y_min, y_max)
            .SetCurrentColor(PlplotFig::Color0::White)
            .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
            .SetAxisLabelX("x")
            .SetAxisLabelY("y")
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(grad_y_pred.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": grad{}_y", d + 1), fig.ToCvMat());

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
        cv::imshow(test_info_->name() + fmt::format(": error{}", d + 1), fig.ToCvMat());

        shades_opt.SetColorLevels(grad_x_error.data(), n_test, n_test, 127);
        color_bar_opt.SetLabelTexts({"grad_x_error"}).AddColorMap(0, shades_opt.color_levels, 10);
        fig.Clear()
            .SetMargin(0.15, 0.85, 0.15, 0.85)
            .SetAxisLimits(x_min, x_max, y_min, y_max)
            .SetCurrentColor(PlplotFig::Color0::White)
            .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
            .SetAxisLabelX("x")
            .SetAxisLabelY("y")
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(grad_x_error.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": grad{}_x_error", d + 1), fig.ToCvMat());

        shades_opt.SetColorLevels(grad_y_error.data(), n_test, n_test, 127);
        color_bar_opt.SetLabelTexts({"grad_y_error"}).AddColorMap(0, shades_opt.color_levels, 10);
        fig.Clear()
            .SetMargin(0.15, 0.85, 0.15, 0.85)
            .SetAxisLimits(x_min, x_max, y_min, y_max)
            .SetCurrentColor(PlplotFig::Color0::White)
            .DrawAxesBox(PlplotFig::AxisOpt().DrawTopRightEdge(), PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
            .SetAxisLabelX("x")
            .SetAxisLabelY("y")
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(grad_y_error.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": grad{}_y_error", d + 1), fig.ToCvMat());

        cv::waitKey(0);

        const double mae = error.cwiseAbs().mean();
        const double mae_grad_x = grad_x_error.cwiseAbs().mean();
        const double mae_grad_y = grad_y_error.cwiseAbs().mean();
        ERL_INFO("{}, mean absolute error: {}, {}, {}.", d + 1, mae, mae_grad_x, mae_grad_y);
        if (d == 0) {
            ASSERT_TRUE(mae < 2.6e-4);
            ASSERT_TRUE(mae_grad_x < 0.015);
            ASSERT_TRUE(mae_grad_y < 0.011);
        } else {
            ASSERT_TRUE(mae < 5.5e-4);
            ASSERT_TRUE(mae_grad_x < 0.030);
            ASSERT_TRUE(mae_grad_y < 0.026);
        }
    }
}

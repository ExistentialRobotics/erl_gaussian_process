#include "erl_common/block_timer.hpp"
#include "erl_common/plplot_fig.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_covariance/radial_bias_function.hpp"
#include "erl_gaussian_process/noisy_input_gp.hpp"

using namespace erl::common;
using namespace erl::gaussian_process;

constexpr double kNoiseVar = 0.0001;

TEST(NoisyInputGaussianProcess, SingleInputSingleOutputWithGradientObservation) {
    GTEST_PREPARE_OUTPUT_DIR();
    constexpr long n = 100;
    const auto setting = std::make_shared<NoisyInputGaussianProcessD::Setting>();
    setting->kernel->scale = 0.2;
    setting->kernel->x_dim = 1;
    setting->kernel_type = type_name<erl::covariance::RadialBiasFunction1d>();
    setting->max_num_samples = n;
    setting->no_gradient_observation = false;
    NoisyInputGaussianProcessD gp(setting);

    auto compute_values = [](const Eigen::VectorXd &x, Eigen::VectorXd &y, Eigen::VectorXd &grad) {
        y = x.unaryExpr([](const double a) { return std::sin(2 * a); });
        grad = x.unaryExpr([](const double a) { return 2 * std::cos(2 * a); });
    };

    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(n, 0, 2 * M_PI);
    Eigen::VectorXd y(n);
    Eigen::VectorXd grad(n);
    compute_values(x, y, grad);

    {
        ERL_BLOCK_TIMER_MSG("gp.Train()");
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
    }

    constexpr long n_test = 200;
    Eigen::VectorXd x_test = Eigen::VectorXd::LinSpaced(n_test, 0, 2 * M_PI);
    Eigen::VectorXd y_test(n_test);
    Eigen::VectorXd grad_test(n_test);
    compute_values(x_test, y_test, grad_test);
    Eigen::VectorXd y_pred(n_test);
    Eigen::VectorXd grad_pred(n_test);
    {
        ERL_BLOCK_TIMER_MSG("gp.Test()");
        auto result = gp.Test(x_test.transpose(), true);
        result->GetMean(0, y_pred, true);
        result->GetGradient(0, grad_pred.transpose(), true);
    }
    Eigen::VectorXd error = y_pred - y_test;
    Eigen::VectorXd grad_error = grad_pred - grad_test;

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
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(-0.1, 2 * M_PI + 0.1, y_test.minCoeff() - 0.1, y_test.maxCoeff() + 0.1)
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
    cv::imshow(test_info_->name() + std::string(": y"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "y.png", fig.ToCvMat());

    legend_opt.SetTexts({"train grad", "test grad g.t.", "prediction grad", "error grad"})
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
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(-0.1, 2 * M_PI + 0.1, grad_test.minCoeff() - 0.1, grad_test.maxCoeff() + 0.1)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(
            PlplotFig::AxisOpt().DrawTopRightEdge(),
            PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
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
        .DrawLine(n_test, x_test.data(), grad_pred.data())
        .SetAxisLimits(
            -0.1,
            2 * M_PI + 0.1,
            grad_error.minCoeff() - 0.001,
            grad_error.maxCoeff() + 0.001)
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
        .DrawLine(n_test, x_test.data(), grad_error.data())
        .Legend(legend_opt);
    cv::imshow(test_info_->name() + std::string(": grad"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "grad.png", fig.ToCvMat());
    cv::waitKey(1000);

    const double mae = error.cwiseAbs().mean();
    const double mae_grad = grad_error.cwiseAbs().mean();
    ERL_INFO("mean absolute error: {}, {}.", mae, mae_grad);
    // mean absolute error: 8.523327884661321e-06, 0.0001228380577847092.  (scale=0.5)
    // mean absolute error: 6.453961399301211e-06, 0.0001125665082426853.  (scale=0.4)
    // mean absolute error: 4.4761251597013675e-06, 8.851481085195954e-05. (scale=0.3)
    // mean absolute error: 4.1624286843223515e-06, 7.139121709502966e-05. (scale=0.2)
    // mean absolute error: 1.756325489369356e-05, 0.00034785637964318994. (scale=0.1)
    ASSERT_TRUE(mae < 1.0e-5);
    ASSERT_TRUE(mae_grad < 1.0e-4);

    ASSERT_TRUE(Serialization<NoisyInputGaussianProcessD>::Write("noisy_input_gp.bin", &gp));
    NoisyInputGaussianProcessD gp_read(std::make_shared<NoisyInputGaussianProcessD::Setting>());
    ASSERT_TRUE(Serialization<NoisyInputGaussianProcessD>::Read("noisy_input_gp.bin", &gp_read));
    ASSERT_TRUE(gp == gp_read);
}

TEST(NoisyInputGaussianProcess, SingleInputSingleOutputWithoutGradientObservation) {
    GTEST_PREPARE_OUTPUT_DIR();
    constexpr long n = 100;
    const auto setting = std::make_shared<NoisyInputGaussianProcessD::Setting>();
    setting->kernel->scale = 0.2;
    setting->kernel->x_dim = 1;
    setting->kernel_type = type_name<erl::covariance::RadialBiasFunction1d>();
    setting->max_num_samples = n;
    setting->no_gradient_observation = true;
    NoisyInputGaussianProcessD gp(setting);

    auto compute_values = [](const Eigen::VectorXd &x, Eigen::VectorXd &y, Eigen::VectorXd &grad) {
        y = x.unaryExpr([](const double a) { return std::sin(2 * a); });
        grad = x.unaryExpr([](const double a) { return 2 * std::cos(2 * a); });
    };

    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(n, 0, 2 * M_PI);
    Eigen::VectorXd y(n);
    Eigen::VectorXd grad(n);
    compute_values(x, y, grad);

    {
        ERL_BLOCK_TIMER_MSG("gp.Train()");
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
    }

    constexpr long n_test = 200;
    Eigen::VectorXd x_test = Eigen::VectorXd::LinSpaced(n_test, 0, 2 * M_PI);
    Eigen::VectorXd y_test(n_test);
    Eigen::VectorXd grad_test(n_test);
    compute_values(x_test, y_test, grad_test);
    Eigen::VectorXd y_pred(n_test);
    Eigen::VectorXd grad_pred(n_test);
    ReportTime<std::chrono::microseconds>("ans", 10, false, [&]() -> void {
        auto result = gp.Test(x_test.transpose(), true);
        result->GetMean(0, y_pred, true);
        result->GetGradient(0, grad_pred.transpose(), true);
    });
    Eigen::VectorXd error = y_pred - y_test;
    Eigen::VectorXd grad_error = grad_pred - grad_test;

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
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(-0.1, 2 * M_PI + 0.1, y_test.minCoeff() - 0.1, y_test.maxCoeff() + 0.1)
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
    cv::imshow(test_info_->name() + std::string(": y"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "y.png", fig.ToCvMat());

    legend_opt.SetTexts({"train grad", "test grad g.t.", "prediction grad", "error grad"})
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
    fig.Clear()
        .SetMargin(0.15, 0.85, 0.15, 0.85)
        .SetAxisLimits(-0.1, 2 * M_PI + 0.1, grad_test.minCoeff() - 0.1, grad_test.maxCoeff() + 0.1)
        .SetCurrentColor(PlplotFig::Color0::White)
        .DrawAxesBox(
            PlplotFig::AxisOpt().DrawTopRightEdge(),
            PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
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
        .DrawLine(n_test, x_test.data(), grad_pred.data())
        .SetAxisLimits(
            -0.1,
            2 * M_PI + 0.1,
            grad_error.minCoeff() - 0.001,
            grad_error.maxCoeff() + 0.001)
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
        .DrawLine(n_test, x_test.data(), grad_error.data())
        .Legend(legend_opt);
    cv::imshow(test_info_->name() + std::string(": grad"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "grad.png", fig.ToCvMat());
    cv::waitKey(1000);

    const double mae = error.cwiseAbs().mean();
    const double mae_grad = grad_error.cwiseAbs().mean();
    ERL_INFO("mean absolute error: {}, {}.", mae, mae_grad);
    // mean absolute error: 0.00019489369361661352, 0.003074427178772044. (scale=0.5)
    // mean absolute error: 7.377464439757659e-05, 0.0024347632450979033. (scale=0.2)
    ASSERT_TRUE(mae < 1.0e-4);
    ASSERT_TRUE(mae_grad < 0.0025);
}

TEST(NoisyInputGaussianProcess, MultiInputSingleOutputWithGradientObservation) {
    GTEST_PREPARE_OUTPUT_DIR();
    auto compute_values = [](const Eigen::VectorXd &x,
                             const Eigen::VectorXd &y,
                             Eigen::VectorXd &z,
                             Eigen::VectorXd &grad_x,
                             Eigen::VectorXd &grad_y,
                             Eigen::Matrix2Xd &pts) {
        for (long xi = 0, i = 0; xi < x.size(); ++xi) {
            for (long yi = 0; yi < y.size(); ++yi, ++i) {
                pts.col(i) << x[xi], y[yi];
                z[i] = 2 * std::sin(10.0 * x[xi]) * std::cos(5.0 * y[yi]);  // (yi, xi)
                grad_x[i] = 20 * std::cos(10.0 * x[xi]) * std::cos(5.0 * y[yi]);
                grad_y[i] = -10 * std::sin(10.0 * x[xi]) * std::sin(5.0 * y[yi]);
            }
        }
    };

    constexpr double x_min = -2.0;
    constexpr double x_max = 2.0;
    constexpr double y_min = -1.0;
    constexpr double y_max = 1.0;
    constexpr long n = 50;
    const auto setting = std::make_shared<NoisyInputGaussianProcessD::Setting>();
    setting->kernel->scale = 0.1;
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
    Eigen::VectorXd z(n * n);
    Eigen::VectorXd grad_x(n * n);
    Eigen::VectorXd grad_y(n * n);
    compute_values(x, y, z, grad_x, grad_y, pts);

    {
        ERL_BLOCK_TIMER_MSG("gp.Train()");
        gp.Reset(pts.cols(), 2, 1);
        auto &train_set = gp.GetTrainSet();
        train_set.x.topLeftCorner(2, pts.cols()) = pts;
        train_set.y.col(0).head(pts.cols()) = z;
        train_set.grad.row(0).head(pts.cols()) = grad_x.transpose();
        train_set.grad.row(1).head(pts.cols()) = grad_y.transpose();
        const long n_pts = pts.cols();
        train_set.var_x.head(n_pts).setConstant(kNoiseVar);
        train_set.var_y.head(n_pts).setConstant(kNoiseVar);
        train_set.var_grad.head(n_pts).setConstant(kNoiseVar);
        train_set.grad_flag.head(n_pts).setConstant(1);
        train_set.num_samples = n_pts;
        train_set.num_samples_with_grad = n_pts;
        ASSERT_TRUE(gp.Train());
    }

    constexpr long n_test = 100;
    x = Eigen::VectorXd::LinSpaced(n_test, x_min, x_max);
    y = Eigen::VectorXd::LinSpaced(n_test, y_min, y_max);
    pts.resize(2, n_test * n_test);
    Eigen::VectorXd z_test(n_test * n_test);
    Eigen::VectorXd grad_x_test(n_test * n_test);
    Eigen::VectorXd grad_y_test(n_test * n_test);
    compute_values(x, y, z_test, grad_x_test, grad_y_test, pts);
    Eigen::VectorXd z_pred(z_test.size());
    Eigen::Matrix2Xd gradient_pred(2, z_test.size());
    {
        ERL_BLOCK_TIMER_MSG("gp.Test()");
        auto result = gp.Test(pts, true);
        result->GetMean(0, z_pred, true);
        (void) result->GetGradient(0, gradient_pred, true);
    }
    Eigen::VectorXd grad_x_pred = gradient_pred.row(0).transpose();
    Eigen::VectorXd grad_y_pred = gradient_pred.row(1).transpose();
    Eigen::VectorXd error = z_pred - z_test;
    Eigen::VectorXd grad_x_error = grad_x_pred - grad_x_test;
    Eigen::VectorXd grad_y_error = grad_y_pred - grad_y_test;

    PlplotFig fig(640, 480, true);
    PlplotFig::ShadesOpt shades_opt;
    shades_opt.SetColorLevels(z_pred.data(), n_test, n_test, 127)
        .SetXMin(x_min)
        .SetXMax(x_max)
        .SetYMin(y_min)
        .SetYMax(y_max);
    PlplotFig::ColorBarOpt color_bar_opt;
    color_bar_opt.SetLabelOpts({PL_COLORBAR_LABEL_BOTTOM})
        .SetLabelTexts({"z"})
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
        .SetColorMap(1, PlplotFig::ColorMap::Jet)
        .Shades(z_pred.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": z"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "z.png", fig.ToCvMat());

    shades_opt.SetColorLevels(grad_x_pred.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"grad_x"}).AddColorMap(0, shades_opt.color_levels, 10);
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
        .Shades(grad_x_pred.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": grad_x"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "grad_x.png", fig.ToCvMat());

    shades_opt.SetColorLevels(grad_y_pred.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"grad_y"}).AddColorMap(0, shades_opt.color_levels, 10);
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
        .Shades(grad_y_pred.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": grad_y"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "grad_y.png", fig.ToCvMat());

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

    shades_opt.SetColorLevels(grad_x_error.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"grad_x_error"}).AddColorMap(0, shades_opt.color_levels, 10);
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
        .Shades(grad_x_error.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": grad_x_error"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "grad_x_error.png", fig.ToCvMat());

    shades_opt.SetColorLevels(grad_y_error.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"grad_y_error"}).AddColorMap(0, shades_opt.color_levels, 10);
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
        .Shades(grad_y_error.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": grad_y_error"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "grad_y_error.png", fig.ToCvMat());

    cv::waitKey(1000);

    const double mae = error.cwiseAbs().mean();
    const double mae_grad_x = grad_x_error.cwiseAbs().mean();
    const double mae_grad_y = grad_y_error.cwiseAbs().mean();
    ERL_INFO("mean absolute error: {}, {}, {}.", mae, mae_grad_x, mae_grad_y);
    // 1.8649622536143655e-05, 0.00035493687359116195, 0.00026104427319052114. (scale=0.2)
    // 1.5775626625013008e-05, 0.0003392957256590181, 0.00023618668921645218. (scale=0.15)
    // 9.516671456234042e-06, 0.00010712550862064423, 0.0002508214688791491. (scale=0.1)
    // 0.000836943693366059, 0.02159010031212598, 0.005901499265160975. (scale=0.05)
    ASSERT_TRUE(mae < 1.0e-5);
    ASSERT_TRUE(mae_grad_x < 1.1e-4);
    ASSERT_TRUE(mae_grad_y < 2.6e-4);
}

TEST(NoisyInputGaussianProcess, MultiInputSingleOutputWithoutGradientObservation) {
    GTEST_PREPARE_OUTPUT_DIR();
    auto compute_values = [](const Eigen::VectorXd &x,
                             const Eigen::VectorXd &y,
                             Eigen::VectorXd &z,
                             Eigen::VectorXd &grad_x,
                             Eigen::VectorXd &grad_y,
                             Eigen::Matrix2Xd &pts) {
        for (long xi = 0, i = 0; xi < x.size(); ++xi) {
            for (long yi = 0; yi < y.size(); ++yi, ++i) {
                pts.col(i) << x[xi], y[yi];
                z[i] = 2 * std::sin(10.0 * x[xi]) * std::cos(5.0 * y[yi]);
                grad_x[i] = 20 * std::cos(10.0 * x[xi]) * std::cos(5.0 * y[yi]);
                grad_y[i] = -10 * std::sin(10.0 * x[xi]) * std::sin(5.0 * y[yi]);
            }
        }
    };

    constexpr double x_min = -2.0;
    constexpr double x_max = 2.0;
    constexpr double y_min = -1.0;
    constexpr double y_max = 1.0;
    constexpr long n = 50;
    const auto setting = std::make_shared<NoisyInputGaussianProcessD::Setting>();
    setting->kernel->scale = 0.15;
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
    Eigen::VectorXd z(n * n);
    Eigen::VectorXd grad_x(n * n);
    Eigen::VectorXd grad_y(n * n);
    compute_values(x, y, z, grad_x, grad_y, pts);

    {
        ERL_BLOCK_TIMER_MSG("gp.Train()");
        gp.Reset(pts.cols(), 2, 1);
        auto &train_set = gp.GetTrainSet();
        train_set.x.topLeftCorner(2, pts.cols()) = pts;
        train_set.y.col(0).head(pts.cols()) = z;
        const long n_pts = pts.cols();
        train_set.var_x.head(n_pts).setConstant(kNoiseVar);
        train_set.var_y.head(n_pts).setConstant(kNoiseVar);
        train_set.grad_flag.head(n_pts).setConstant(0);
        train_set.num_samples = n_pts;
        train_set.num_samples_with_grad = 0;
        ASSERT_TRUE(gp.Train());
    }

    constexpr long n_test = 100;
    x = Eigen::VectorXd::LinSpaced(n_test, x_min, x_max);
    y = Eigen::VectorXd::LinSpaced(n_test, y_min, y_max);
    pts.resize(2, n_test * n_test);
    Eigen::VectorXd z_test(n_test * n_test);
    Eigen::VectorXd grad_x_test(n_test * n_test);
    Eigen::VectorXd grad_y_test(n_test * n_test);
    compute_values(x, y, z_test, grad_x_test, grad_y_test, pts);
    Eigen::VectorXd z_pred(z_test.size());
    Eigen::Matrix2Xd gradient_pred(2, z_test.size());
    {
        ERL_BLOCK_TIMER_MSG("gp.Test()");
        auto result = gp.Test(pts, true);
        result->GetMean(0, z_pred, true);
        (void) result->GetGradient(0, gradient_pred, true);
    }
    Eigen::VectorXd grad_x_pred = gradient_pred.row(0).transpose();
    Eigen::VectorXd grad_y_pred = gradient_pred.row(1).transpose();
    Eigen::VectorXd error = z_pred - z_test;
    Eigen::VectorXd grad_x_error = grad_x_pred - grad_x_test;
    Eigen::VectorXd grad_y_error = grad_y_pred - grad_y_test;

    PlplotFig fig(640, 480, true);
    PlplotFig::ShadesOpt shades_opt;
    shades_opt.SetColorLevels(z_pred.data(), n_test, n_test, 127)
        .SetXMin(x_min)
        .SetXMax(x_max)
        .SetYMin(y_min)
        .SetYMax(y_max);
    PlplotFig::ColorBarOpt color_bar_opt;
    color_bar_opt.SetLabelOpts({PL_COLORBAR_LABEL_BOTTOM})
        .SetLabelTexts({"z"})
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
        .SetColorMap(1, PlplotFig::ColorMap::Jet)
        .Shades(z_pred.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": z"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "z.png", fig.ToCvMat());

    shades_opt.SetColorLevels(grad_x_pred.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"grad_x"}).AddColorMap(0, shades_opt.color_levels, 10);
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
        .Shades(grad_x_pred.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": grad_x"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "grad_x.png", fig.ToCvMat());

    shades_opt.SetColorLevels(grad_y_pred.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"grad_y"}).AddColorMap(0, shades_opt.color_levels, 10);
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
        .Shades(grad_y_pred.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": grad_y"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "grad_y.png", fig.ToCvMat());

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

    shades_opt.SetColorLevels(grad_x_error.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"grad_x_error"}).AddColorMap(0, shades_opt.color_levels, 10);
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
        .Shades(grad_x_error.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": grad_x_error"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "grad_x_error.png", fig.ToCvMat());

    shades_opt.SetColorLevels(grad_y_error.data(), n_test, n_test, 127);
    color_bar_opt.SetLabelTexts({"grad_y_error"}).AddColorMap(0, shades_opt.color_levels, 10);
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
        .Shades(grad_y_error.data(), n_test, n_test, true, shades_opt)
        .ColorBar(color_bar_opt);
    cv::imshow(test_info_->name() + std::string(": grad_y_error"), fig.ToCvMat());
    cv::imwrite(test_output_dir / "grad_y_error.png", fig.ToCvMat());

    cv::waitKey(1000);

    const double mae = error.cwiseAbs().mean();
    const double mae_grad_x = grad_x_error.cwiseAbs().mean();
    const double mae_grad_y = grad_y_error.cwiseAbs().mean();
    ERL_INFO("mean absolute error: {}, {}, {}.", mae, mae_grad_x, mae_grad_y);
    // 0.0009210182354101644, 0.048791685832527174, 0.017905337511994152. (scale=0.1)
    // 0.0003368450993049195, 0.009407525172327099, 0.014184702590183184. (scale=0.15)
    ASSERT_TRUE(mae < 3.4e-4);
    ASSERT_TRUE(mae_grad_x < 0.01);
    ASSERT_TRUE(mae_grad_y < 0.015);
}

TEST(NoisyInputGaussianProcess, MultiInputMultiOutputWithGradientObservation) {
    GTEST_PREPARE_OUTPUT_DIR();
    auto compute_values = [](const Eigen::VectorXd &x,
                             const Eigen::VectorXd &y,
                             Eigen::VectorXd &z1,
                             Eigen::VectorXd &z2,
                             Eigen::VectorXd &grad1_x,
                             Eigen::VectorXd &grad1_y,
                             Eigen::VectorXd &grad2_x,
                             Eigen::VectorXd &grad2_y,
                             Eigen::Matrix2Xd &pts) {
        for (long xi = 0, i = 0; xi < x.size(); ++xi) {
            for (long yi = 0; yi < y.size(); ++yi, ++i) {
                pts.col(i) << x[xi], y[yi];
                z1[i] = 2 * std::sin(10.0 * x[xi]) * std::cos(10.0 * y[yi]);
                z2[i] = 3 * (std::sin(10.0 * x[xi]) + std::cos(10.0 * y[yi]));
                grad1_x[i] = 20 * std::cos(10.0 * x[xi]) * std::cos(10.0 * y[yi]);
                grad1_y[i] = -20 * std::sin(10.0 * x[xi]) * std::sin(10.0 * y[yi]);
                grad2_x[i] = 30 * std::cos(10.0 * x[xi]);
                grad2_y[i] = -30 * std::sin(10.0 * y[yi]);
            }
        }
    };

    constexpr double x_min = -1.0;
    constexpr double x_max = 1.0;
    constexpr double y_min = -1.0;
    constexpr double y_max = 1.0;
    constexpr long n = 50;
    const auto setting = std::make_shared<NoisyInputGaussianProcessD::Setting>();
    setting->kernel->scale = 0.15;
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
    Eigen::VectorXd z1(n * n);
    Eigen::VectorXd z2(n * n);
    Eigen::VectorXd grad1_x(n * n);
    Eigen::VectorXd grad1_y(n * n);
    Eigen::VectorXd grad2_x(n * n);
    Eigen::VectorXd grad2_y(n * n);
    compute_values(x, y, z1, z2, grad1_x, grad1_y, grad2_x, grad2_y, pts);

    {
        ERL_BLOCK_TIMER_MSG("gp.Train()");
        gp.Reset(pts.cols(), 2, 2);
        auto &train_set = gp.GetTrainSet();
        train_set.x.topLeftCorner(2, pts.cols()) = pts;
        train_set.y.col(0).head(pts.cols()) = z1;
        train_set.y.col(1).head(pts.cols()) = z2;
        train_set.grad.row(0).head(pts.cols()) = grad1_x.transpose();
        train_set.grad.row(1).head(pts.cols()) = grad1_y.transpose();
        train_set.grad.row(2).head(pts.cols()) = grad2_x.transpose();
        train_set.grad.row(3).head(pts.cols()) = grad2_y.transpose();
        const long n_pts = pts.cols();
        train_set.var_x.head(n_pts).setConstant(kNoiseVar);
        train_set.var_y.head(n_pts).setConstant(kNoiseVar);
        train_set.var_grad.head(n_pts).setConstant(kNoiseVar);
        train_set.grad_flag.head(n_pts).setConstant(1);
        train_set.num_samples = n_pts;
        train_set.num_samples_with_grad = n_pts;
        ASSERT_TRUE(gp.Train());
    }

    constexpr long n_test = 100;
    x = Eigen::VectorXd::LinSpaced(n_test, x_min, x_max);
    y = Eigen::VectorXd::LinSpaced(n_test, y_min, y_max);
    pts.resize(2, n_test * n_test);
    Eigen::VectorXd z1_test(n_test * n_test);
    Eigen::VectorXd z2_test(n_test * n_test);
    Eigen::VectorXd grad1_x_test(n_test * n_test);
    Eigen::VectorXd grad1_y_test(n_test * n_test);
    Eigen::VectorXd grad2_x_test(n_test * n_test);
    Eigen::VectorXd grad2_y_test(n_test * n_test);
    compute_values(
        x,
        y,
        z1_test,
        z2_test,
        grad1_x_test,
        grad1_y_test,
        grad2_x_test,
        grad2_y_test,
        pts);
    Eigen::MatrixX2d pred(z1_test.size(), 2);
    Eigen::Matrix4Xd gradient_pred(4, z1_test.size());
    {
        ERL_BLOCK_TIMER_MSG("gp.Test()");
        auto result = gp.Test(pts, true);
        result->GetMean(0, pred.col(0), true);
        result->GetMean(1, pred.col(1), true);
        result->GetGradient(0, gradient_pred.topRows<2>(), true);
        result->GetGradient(1, gradient_pred.bottomRows<2>(), true);
    }

    PlplotFig fig(640, 480, true);
    for (long d = 0; d < 2; ++d) {
        auto z_pred = pred.col(d);
        Eigen::VectorXd grad_x_pred = gradient_pred.row(d * 2).transpose();
        Eigen::VectorXd grad_y_pred = gradient_pred.row(d * 2 + 1).transpose();
        Eigen::VectorXd error = z_pred - (d == 0 ? z1_test : z2_test);
        Eigen::VectorXd grad_x_error = grad_x_pred - (d == 0 ? grad1_x_test : grad2_x_test);
        Eigen::VectorXd grad_y_error = grad_y_pred - (d == 0 ? grad1_y_test : grad2_y_test);

        PlplotFig::ShadesOpt shades_opt;
        shades_opt.SetColorLevels(z_pred.data(), n_test, n_test, 127)
            .SetXMin(x_min)
            .SetXMax(x_max)
            .SetYMin(y_min)
            .SetYMax(y_max);
        PlplotFig::ColorBarOpt color_bar_opt;
        color_bar_opt.SetLabelOpts({PL_COLORBAR_LABEL_BOTTOM})
            .SetLabelTexts({"z"})
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
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(z_pred.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": pred{}", d + 1), fig.ToCvMat());
        cv::imwrite(test_output_dir / fmt::format("pred{}.png", d + 1), fig.ToCvMat());

        shades_opt.SetColorLevels(grad_x_pred.data(), n_test, n_test, 127);
        color_bar_opt.SetLabelTexts({"grad_x"}).AddColorMap(0, shades_opt.color_levels, 10);
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
            .Shades(grad_x_pred.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": grad{}_x", d + 1), fig.ToCvMat());
        cv::imwrite(test_output_dir / fmt::format("grad{}_x.png", d + 1), fig.ToCvMat());

        shades_opt.SetColorLevels(grad_y_pred.data(), n_test, n_test, 127);
        color_bar_opt.SetLabelTexts({"grad_y"}).AddColorMap(0, shades_opt.color_levels, 10);
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
            .Shades(grad_y_pred.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": grad{}_y", d + 1), fig.ToCvMat());
        cv::imwrite(test_output_dir / fmt::format("grad{}_y.png", d + 1), fig.ToCvMat());

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
        cv::imshow(test_info_->name() + fmt::format(": error{}", d + 1), fig.ToCvMat());
        cv::imwrite(test_output_dir / fmt::format("error{}.png", d + 1), fig.ToCvMat());

        shades_opt.SetColorLevels(grad_x_error.data(), n_test, n_test, 127);
        color_bar_opt.SetLabelTexts({"grad_x_error"}).AddColorMap(0, shades_opt.color_levels, 10);
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
            .Shades(grad_x_error.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": grad{}_x_error", d + 1), fig.ToCvMat());
        cv::imwrite(test_output_dir / fmt::format("grad{}_x_error.png", d + 1), fig.ToCvMat());

        shades_opt.SetColorLevels(grad_y_error.data(), n_test, n_test, 127);
        color_bar_opt.SetLabelTexts({"grad_y_error"}).AddColorMap(0, shades_opt.color_levels, 10);
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
            .Shades(grad_y_error.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": grad{}_y_error", d + 1), fig.ToCvMat());
        cv::imwrite(test_output_dir / fmt::format("grad{}_y_error.png", d + 1), fig.ToCvMat());

        cv::waitKey(1000);

        const double mae = error.cwiseAbs().mean();
        const double mae_grad_x = grad_x_error.cwiseAbs().mean();
        const double mae_grad_y = grad_y_error.cwiseAbs().mean();
        ERL_INFO("{}, mean absolute error: {}, {}, {}.", d + 1, mae, mae_grad_x, mae_grad_y);
        if (d == 0) {
            // 6.205702021195462e-06, 0.00016324462241659358, 0.0002209177886253753.
            ASSERT_TRUE(mae < 1.0e-5);
            ASSERT_TRUE(mae_grad_x < 1.7e-4);
            ASSERT_TRUE(mae_grad_y < 2.3e-4);
        } else {
            // 1.1967913545722718e-05, 0.000292787449896784, 0.00034572267944076794.
            ASSERT_TRUE(mae < 1.2e-5);
            ASSERT_TRUE(mae_grad_x < 3.0e-4);
            ASSERT_TRUE(mae_grad_y < 3.5e-4);
        }
    }
}

TEST(NoisyInputGaussianProcess, MultiInputMultiOutputWithoutGradientObservation) {
    GTEST_PREPARE_OUTPUT_DIR();
    auto compute_values = [](const Eigen::VectorXd &x,
                             const Eigen::VectorXd &y,
                             Eigen::VectorXd &z1,
                             Eigen::VectorXd &z2,
                             Eigen::VectorXd &grad1_x,
                             Eigen::VectorXd &grad1_y,
                             Eigen::VectorXd &grad2_x,
                             Eigen::VectorXd &grad2_y,
                             Eigen::Matrix2Xd &pts) {
        for (long xi = 0, i = 0; xi < x.size(); ++xi) {
            for (long yi = 0; yi < y.size(); ++yi, ++i) {
                pts.col(i) << x[xi], y[yi];
                z1[i] = 2 * std::sin(10.0 * x[xi]) * std::cos(10.0 * y[yi]);
                z2[i] = 3 * (std::sin(10.0 * x[xi]) + std::cos(10.0 * y[yi]));
                grad1_x[i] = 20 * std::cos(10.0 * x[xi]) * std::cos(10.0 * y[yi]);
                grad1_y[i] = -20 * std::sin(10.0 * x[xi]) * std::sin(10.0 * y[yi]);
                grad2_x[i] = 30 * std::cos(10.0 * x[xi]);
                grad2_y[i] = -30 * std::sin(10.0 * y[yi]);
            }
        }
    };

    constexpr double x_min = -1.0;
    constexpr double x_max = 1.0;
    constexpr double y_min = -1.0;
    constexpr double y_max = 1.0;
    constexpr long n = 50;
    const auto setting = std::make_shared<NoisyInputGaussianProcessD::Setting>();
    setting->kernel->scale = 0.1;
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
    Eigen::VectorXd z1(n * n);
    Eigen::VectorXd z2(n * n);
    Eigen::VectorXd grad1_x(n * n);
    Eigen::VectorXd grad1_y(n * n);
    Eigen::VectorXd grad2_x(n * n);
    Eigen::VectorXd grad2_y(n * n);
    compute_values(x, y, z1, z2, grad1_x, grad1_y, grad2_x, grad2_y, pts);

    {
        ERL_BLOCK_TIMER_MSG("gp.Train()");
        gp.Reset(pts.cols(), 2, 2);
        auto &train_set = gp.GetTrainSet();
        train_set.x.topLeftCorner(2, pts.cols()) = pts;
        train_set.y.col(0).head(pts.cols()) = z1;
        train_set.y.col(1).head(pts.cols()) = z2;
        const long n_pts = pts.cols();
        train_set.var_x.head(n_pts).setConstant(kNoiseVar);
        train_set.var_y.head(n_pts).setConstant(kNoiseVar);
        train_set.grad_flag.head(n_pts).setConstant(0);
        train_set.num_samples = n_pts;
        train_set.num_samples_with_grad = 0;
    }
    ASSERT_TRUE(gp.Train());

    constexpr long n_test = 100;
    x = Eigen::VectorXd::LinSpaced(n_test, x_min, x_max);
    y = Eigen::VectorXd::LinSpaced(n_test, y_min, y_max);
    pts.resize(2, n_test * n_test);
    Eigen::VectorXd z1_test(n_test * n_test);
    Eigen::VectorXd z2_test(n_test * n_test);
    Eigen::VectorXd grad1_x_test(n_test * n_test);
    Eigen::VectorXd grad1_y_test(n_test * n_test);
    Eigen::VectorXd grad2_x_test(n_test * n_test);
    Eigen::VectorXd grad2_y_test(n_test * n_test);
    compute_values(
        x,
        y,
        z1_test,
        z2_test,
        grad1_x_test,
        grad1_y_test,
        grad2_x_test,
        grad2_y_test,
        pts);
    // Eigen::MatrixXd pred(6, z1_test.size());
    Eigen::MatrixX2d pred(z1_test.size(), 2);
    Eigen::Matrix4Xd gradient_pred(4, z1_test.size());
    {
        ERL_BLOCK_TIMER_MSG("gp.Test()");
        auto result = gp.Test(pts, true);
        result->GetMean(0, pred.col(0), true);
        result->GetMean(1, pred.col(1), true);
        result->GetGradient(0, gradient_pred.topRows<2>(), true);
        result->GetGradient(1, gradient_pred.bottomRows<2>(), true);
    }

    PlplotFig fig(640, 480, true);
    for (long d = 0; d < 2; ++d) {
        auto z_pred = pred.col(d);
        Eigen::VectorXd grad_x_pred = gradient_pred.row(d * 2).transpose();
        Eigen::VectorXd grad_y_pred = gradient_pred.row(d * 2 + 1).transpose();
        Eigen::VectorXd error = z_pred - (d == 0 ? z1_test : z2_test);
        Eigen::VectorXd grad_x_error = grad_x_pred - (d == 0 ? grad1_x_test : grad2_x_test);
        Eigen::VectorXd grad_y_error = grad_y_pred - (d == 0 ? grad1_y_test : grad2_y_test);

        PlplotFig::ShadesOpt shades_opt;
        shades_opt.SetColorLevels(z_pred.data(), n_test, n_test, 127)
            .SetXMin(x_min)
            .SetXMax(x_max)
            .SetYMin(y_min)
            .SetYMax(y_max);
        PlplotFig::ColorBarOpt color_bar_opt;
        color_bar_opt.SetLabelOpts({PL_COLORBAR_LABEL_BOTTOM})
            .SetLabelTexts({"z"})
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
            .SetColorMap(1, PlplotFig::ColorMap::Jet)
            .Shades(z_pred.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": pred{}", d + 1), fig.ToCvMat());
        cv::imwrite(test_output_dir / fmt::format("pred{}.png", d + 1), fig.ToCvMat());

        shades_opt.SetColorLevels(grad_x_pred.data(), n_test, n_test, 127);
        color_bar_opt.SetLabelTexts({"grad_x"}).AddColorMap(0, shades_opt.color_levels, 10);
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
            .Shades(grad_x_pred.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": grad{}_x", d + 1), fig.ToCvMat());
        cv::imwrite(test_output_dir / fmt::format("grad{}_x.png", d + 1), fig.ToCvMat());

        shades_opt.SetColorLevels(grad_y_pred.data(), n_test, n_test, 127);
        color_bar_opt.SetLabelTexts({"grad_y"}).AddColorMap(0, shades_opt.color_levels, 10);
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
            .Shades(grad_y_pred.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": grad{}_y", d + 1), fig.ToCvMat());
        cv::imwrite(test_output_dir / fmt::format("grad{}_y.png", d + 1), fig.ToCvMat());

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
        cv::imshow(test_info_->name() + fmt::format(": error{}", d + 1), fig.ToCvMat());
        cv::imwrite(test_output_dir / fmt::format("error{}.png", d + 1), fig.ToCvMat());

        shades_opt.SetColorLevels(grad_x_error.data(), n_test, n_test, 127);
        color_bar_opt.SetLabelTexts({"grad_x_error"}).AddColorMap(0, shades_opt.color_levels, 10);
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
            .Shades(grad_x_error.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": grad{}_x_error", d + 1), fig.ToCvMat());
        cv::imwrite(test_output_dir / fmt::format("grad{}_x_error.png", d + 1), fig.ToCvMat());

        shades_opt.SetColorLevels(grad_y_error.data(), n_test, n_test, 127);
        color_bar_opt.SetLabelTexts({"grad_y_error"}).AddColorMap(0, shades_opt.color_levels, 10);
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
            .Shades(grad_y_error.data(), n_test, n_test, true, shades_opt)
            .ColorBar(color_bar_opt);
        cv::imshow(test_info_->name() + fmt::format(": grad{}_y_error", d + 1), fig.ToCvMat());
        cv::imwrite(test_output_dir / fmt::format("grad{}_y_error.png", d + 1), fig.ToCvMat());

        cv::waitKey(1000);

        const double mae = error.cwiseAbs().mean();
        const double mae_grad_x = grad_x_error.cwiseAbs().mean();
        const double mae_grad_y = grad_y_error.cwiseAbs().mean();
        ERL_INFO("{}, mean absolute error: {}, {}, {}.", d + 1, mae, mae_grad_x, mae_grad_y);
        if (d == 0) {
            // 0.000250581062775504, 0.014144193031284197, 0.010989238198062933. (scale=0.1)
            ASSERT_TRUE(mae < 2.6e-4);
            ASSERT_TRUE(mae_grad_x < 0.015);
            ASSERT_TRUE(mae_grad_y < 0.011);
        } else {
            // 0.0005492868513902853, 0.02918565122522403, 0.025920746735521707. (scale=0.1)
            ASSERT_TRUE(mae < 5.5e-4);
            ASSERT_TRUE(mae_grad_x < 0.030);
            ASSERT_TRUE(mae_grad_y < 0.026);
        }
    }
}

int
main(int argc, char *argv[]) {
    Init();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

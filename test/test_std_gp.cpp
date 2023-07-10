#include <chrono>

#include "erl_common/test_helper.hpp"
#include "erl_gaussian_process/vanilla_gp.hpp"
#include "obs_gp.h"

using namespace erl::gaussian_process;

constexpr double kKernelAlpha = 1.;
constexpr double kKernelScale = 0.5;
constexpr double kNoiseSigma = 0.01;

int
main() {

    auto setting = std::make_shared<VanillaGaussianProcess::Setting>();
    setting->kernel->alpha = kKernelAlpha;
    setting->kernel->scale = kKernelScale;
    VanillaGaussianProcess std_gp(setting);
    GPou gp_ou;

    int d = 2, n = 10, m = 10;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(d, n) * 10;
    Eigen::MatrixXd Xt = Eigen::MatrixXd::Random(d, m) * 10;
    Eigen::VectorXd y = Eigen::VectorXd::Random(n) * 10;

    Eigen::VectorXd ans_f(n), gt_f, ans_var(n), gt_var;

    std::cout << "Train:" << std::endl;
    ReportTime<std::chrono::microseconds>("ans", 10, false, [&]() -> void { std_gp.Train(X, y, Eigen::VectorXd::Constant(y.size(), kNoiseSigma)); });
    ReportTime<std::chrono::microseconds>("gt", 10, false, [&]() -> void { gp_ou.Train(X, y); });

    std::cout << "test:" << std::endl;
    ReportTime<std::chrono::microseconds>("ans", 10, false, [&]() -> void { std_gp.Test(Xt, ans_f, ans_var); });
    ReportTime<std::chrono::microseconds>("gt", 10, false, [&]() -> void { gp_ou.Test(Xt, gt_f, gt_var); });

    CheckAnswers("f", ans_f, gt_f);
    CheckAnswers("var", ans_var, gt_var);

    return 0;
}

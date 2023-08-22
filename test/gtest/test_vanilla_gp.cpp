#include <gtest/gtest.h>

#include "erl_common/test_helper.hpp"
#include "erl_gaussian_process/vanilla_gp.hpp"
#include "../obs_gp.h"
#include "../obs_gp.cpp"
#include "../cov_fnc.cpp"

using namespace erl::common;
using namespace erl::gaussian_process;

constexpr double kKernelAlpha = 1.;
constexpr double kKernelScale = 0.5;
constexpr double kNoiseVar = 0.01;

TEST(ERL_GAUSSIAN_PROCESS, VanillaGaussianProcess) {
    auto setting = std::make_shared<VanillaGaussianProcess::Setting>();
    setting->kernel->alpha = kKernelAlpha;
    setting->kernel->scale = kKernelScale;
    VanillaGaussianProcess vanilla_gp(setting);
    GPou gp_ou;

    int d = 2, n = 10, m = 10;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(d, n) * 10;
    Eigen::MatrixXd Xt = Eigen::MatrixXd::Random(d, m) * 10;
    Eigen::VectorXd y = Eigen::VectorXd::Random(n) * 10;
    Eigen::VectorXd ans_f(n), gt_f, ans_var(n), gt_var;

    std::cout << "Train:" << std::endl;
    ReportTime<std::chrono::microseconds>("ans", 10, false, [&]() -> void {
        vanilla_gp.Reset(n, d);
        vanilla_gp.GetTrainInputSamplesBuffer() = X;
        vanilla_gp.GetTrainOutputSamplesBuffer() = y;
        vanilla_gp.GetTrainOutputSamplesVarianceBuffer().setConstant(kNoiseVar);
        vanilla_gp.Train(n);
    });
    ReportTime<std::chrono::microseconds>("gt", 10, false, [&]() -> void { gp_ou.Train(X, y); });

    std::cout << "test:" << std::endl;
    ReportTime<std::chrono::microseconds>("ans", 10, false, [&]() -> void { vanilla_gp.Test(Xt, ans_f, ans_var); });
    ReportTime<std::chrono::microseconds>("gt", 10, false, [&]() -> void { gp_ou.Test(Xt, gt_f, gt_var); });

    EXPECT_TRUE(ans_f.isApprox(gt_f, 1e-10));
    EXPECT_TRUE(ans_var.isApprox(gt_var, 1e-10));
}

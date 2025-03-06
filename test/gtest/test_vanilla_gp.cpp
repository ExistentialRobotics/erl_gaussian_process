#include "../cov_fnc.cpp"
#include "../obs_gp.cpp"
#include "../obs_gp.h"

#include "erl_common/test_helper.hpp"
#include "erl_gaussian_process/vanilla_gp.hpp"

#include <gtest/gtest.h>

using namespace erl::common;
using namespace erl::gaussian_process;

constexpr double kKernelAlpha = 1.;
constexpr double kKernelScale = 0.5;
constexpr double kNoiseVar = 0.01;

TEST(ERL_GAUSSIAN_PROCESS, VanillaGaussianProcess) {
    auto setting = std::make_shared<VanillaGaussianProcessD::Setting>();
    setting->kernel->alpha = kKernelAlpha;
    setting->kernel->scale = kKernelScale;
    VanillaGaussianProcessD vanilla_gp(setting);
    GPou gp_ou;

    int d = 2, n = 10, m = 10;
    Eigen::MatrixXd mat_x_train = Eigen::MatrixXd::Random(d, n) * 10;
    Eigen::MatrixXd mat_x_test = Eigen::MatrixXd::Random(d, m) * 10;
    Eigen::VectorXd y = Eigen::VectorXd::Random(n) * 10;
    Eigen::VectorXd ans_f(n), gt_f, ans_var(n), gt_var;

    std::cout << "Train:" << std::endl;
    ReportTime<std::chrono::microseconds>("ans", 10, false, [&]() -> void {
        vanilla_gp.Reset(n, d);
        vanilla_gp.GetTrainInputSamplesBuffer() = mat_x_train;
        vanilla_gp.GetTrainOutputSamplesBuffer() = y;
        vanilla_gp.GetTrainOutputSamplesVarianceBuffer().setConstant(kNoiseVar);
        ASSERT_TRUE(vanilla_gp.Train(n));
    });
    ReportTime<std::chrono::microseconds>("gt", 10, false, [&]() -> void { gp_ou.Train(mat_x_train, y); });

    std::cout << "test:" << std::endl;
    ReportTime<std::chrono::microseconds>("ans", 10, false, [&]() -> void { ASSERT_TRUE(vanilla_gp.Test(mat_x_test, ans_f, ans_var)); });
    ReportTime<std::chrono::microseconds>("gt", 10, false, [&]() -> void { gp_ou.Test(mat_x_test, gt_f, gt_var); });

    EXPECT_TRUE(ans_f.isApprox(gt_f, 1e-10));
    EXPECT_TRUE(ans_var.isApprox(gt_var, 1e-10));
}

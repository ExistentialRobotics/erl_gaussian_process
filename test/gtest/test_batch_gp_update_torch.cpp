
#ifdef ERL_USE_LIBTORCH

    #include "erl_common/block_timer.hpp"
    #include "erl_common/test_helper.hpp"
    #include "erl_covariance/radial_bias_function.hpp"
    #include "erl_gaussian_process/batch_gp_update_torch.hpp"
    #include "erl_gaussian_process/vanilla_gp.hpp"

TEST(BatchGpUpdateTorch, Basic) {
    using BatchGpUpdateTorch = erl::gaussian_process::BatchGaussianProcessUpdateTorch<float>;
    using VanillaGaussianProcessF = erl::gaussian_process::VanillaGaussianProcess<float>;

    GTEST_PREPARE_OUTPUT_DIR();
    constexpr long n = 100;
    const auto setting = std::make_shared<VanillaGaussianProcessF::Setting>();
    setting->kernel->scale = 0.5;
    setting->kernel->x_dim = 1;
    setting->kernel_type = type_name<erl::covariance::RadialBiasFunction1f>();
    setting->max_num_samples = n;
    VanillaGaussianProcessF gp(setting);

    Eigen::VectorXf x = Eigen::VectorXf::LinSpaced(n, 0, 2 * M_PI);
    Eigen::VectorXf y = x.unaryExpr([](const float a) { return std::sin(a); });
    constexpr float kNoiseVar = 0.001;
    gp.Reset(n, 1, 1);
    auto &train_set = gp.GetTrainSet();
    train_set.x.row(0).head(n) = x.transpose();
    train_set.y.col(0).head(n) = y;
    train_set.var.head(n).setConstant(kNoiseVar);
    train_set.num_samples = n;
    gp.UpdateKtrain();

    BatchGpUpdateTorch batch_update;
    batch_update.PrepareMemory(1, n, 1);

    {
        ERL_BLOCK_TIMER_MSG("gp.Solve");
        gp.Solve();
    }

    const Eigen::MatrixXf l_train = gp.GetCholeskyDecomposition();
    const Eigen::MatrixXf alpha = gp.GetAlpha();

    gp.UpdateKtrain();
    {
        ERL_BLOCK_TIMER_MSG("batch_update.Solve");
        {
            ERL_BLOCK_TIMER_MSG("batch_update.LoadGpData");
            batch_update.LoadGpData(0, gp.GetKtrainSize().first, gp.GetKtrain(), gp.GetAlpha());
        }
        {
            ERL_BLOCK_TIMER_MSG("batch_update.Solve");
            batch_update.Solve();
        }
        {
            ERL_BLOCK_TIMER_MSG("batch_update.GetGpResult");
            batch_update.GetGpResult(0, gp.GetCholeskyDecomposition(), gp.GetAlpha());
        }
    }

    const Eigen::MatrixXf l_train_batch = gp.GetCholeskyDecomposition();
    const Eigen::MatrixXf alpha_batch = gp.GetAlpha();

    EXPECT_TRUE(l_train.isApprox(l_train_batch));
    EXPECT_TRUE(alpha.isApprox(alpha_batch));
}

std::pair<double, double>
Profiling(const long num_gps) {
    using BatchGpUpdateTorch = erl::gaussian_process::BatchGaussianProcessUpdateTorch<float>;
    using VanillaGaussianProcessF = erl::gaussian_process::VanillaGaussianProcess<float>;

    GTEST_PREPARE_OUTPUT_DIR();
    constexpr long n = 100;
    const auto setting = std::make_shared<VanillaGaussianProcessF::Setting>();
    setting->kernel->scale = 0.5;
    setting->kernel->x_dim = 1;
    setting->kernel_type = type_name<erl::covariance::RadialBiasFunction1f>();
    setting->max_num_samples = n;
    VanillaGaussianProcessF gp(setting);

    Eigen::VectorXf x = Eigen::VectorXf::LinSpaced(n, 0, 2 * M_PI);
    Eigen::VectorXf y = x.unaryExpr([](const float a) { return std::sin(a); });
    constexpr float kNoiseVar = 0.001;
    gp.Reset(n, 1, 1);
    auto &train_set = gp.GetTrainSet();
    train_set.x.row(0).head(n) = x.transpose();
    train_set.y.col(0).head(n) = y;
    train_set.var.head(n).setConstant(kNoiseVar);
    train_set.num_samples = n;
    gp.UpdateKtrain();

    std::vector<VanillaGaussianProcessF> gps(num_gps, gp);

    BatchGpUpdateTorch batch_update;
    batch_update.PrepareMemory(num_gps, n, 1);
    double time_cpu, time_gpu;

    {
        ERL_BLOCK_TIMER_MSG_TIME("gp.Solve", time_cpu);
    #pragma omp parallel for default(none) shared(gps)
        for (auto &g: gps) { g.Solve(); }
    }

    #pragma omp parallel for default(none) shared(gps)
    for (auto &g: gps) { g.UpdateKtrain(); }  // reset Ktrain and alpha

    {
        ERL_BLOCK_TIMER_MSG_TIME("batch_update", time_gpu);

    #pragma omp parallel for default(none) shared(num_gps, batch_update, gps)
        for (long i = 0; i < num_gps; ++i) {
            auto &g = gps[i];
            batch_update.LoadGpData(i, g.GetKtrainSize().first, g.GetKtrain(), g.GetAlpha());
        }

        {
            ERL_BLOCK_TIMER_MSG("batch_update.Solve");
            batch_update.Solve();
        }

    #pragma omp parallel for default(none) shared(num_gps, batch_update, gps)
        for (long i = 0; i < num_gps; ++i) {
            batch_update.GetGpResult(i, gps[i].GetCholeskyDecomposition(), gps[i].GetAlpha());
        }
    }

    return {time_cpu, time_gpu};
}

TEST(BatchGpUpdateTorch, Profiling) {
    for (long num_gps: {1, 10, 100, 1000, 10000}) {
        auto [time_cpu, time_gpu] = Profiling(num_gps);
        ERL_INFO(
            "num_gps = {}, time_cpu = {:.3f} ms, time_gpu = {:.3f} ms",
            num_gps,
            time_cpu,
            time_gpu);
    }
}

#endif

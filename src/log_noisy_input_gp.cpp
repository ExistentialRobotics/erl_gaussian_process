#include "erl_gaussian_process/log_noisy_input_gp.hpp"

namespace erl::gaussian_process {

    std::shared_ptr<LogNoisyInputGaussianProcess>
    LogNoisyInputGaussianProcess::Create() {
        return std::shared_ptr<LogNoisyInputGaussianProcess>(new LogNoisyInputGaussianProcess());
    }

    std::shared_ptr<LogNoisyInputGaussianProcess>
    LogNoisyInputGaussianProcess::Create(std::shared_ptr<Setting> setting) {
        return std::shared_ptr<LogNoisyInputGaussianProcess>(new LogNoisyInputGaussianProcess(std::move(setting)));
    }

    void
    LogNoisyInputGaussianProcess::Train(
        Eigen::MatrixXd mat_x_train,
        Eigen::VectorXb vec_grad_flag,
        const Eigen::Ref<const Eigen::VectorXd> &vec_y,
        const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_x,
        const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_f,
        const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_grad) {

        // Apply the latest setting
        if (m_kernel_ == nullptr) {
            m_three_over_scale_square_ = double(3.) / (m_setting_->kernel->scale * m_setting_->kernel->scale);
            NoisyInputGaussianProcess::m_kernel_ = Covariance::Create(m_setting_->kernel);
            auto modified_kernel_setting = std::make_shared<Covariance::Setting>(*m_setting_->kernel);
            modified_kernel_setting->scale = std::sqrt(double(3.)) / m_setting_->log_lambda;
            m_kernel_ = Covariance::Create(modified_kernel_setting);
        }

        auto n = mat_x_train.cols();

        if (n <= 0) { return; }

        m_mat_x_train_ = std::move(mat_x_train);
        m_vec_grad_flag_ = std::move(vec_grad_flag);

        // GPIS
        auto ktrain =
            NoisyInputGaussianProcess::m_kernel_->ComputeKtrainWithGradient(m_mat_x_train_, m_vec_grad_flag_, vec_sigma_x, vec_sigma_f, vec_sigma_grad);

        m_mat_l_ = ktrain.llt().matrixL();
        m_vec_alpha_ = vec_y;
        m_mat_l_.triangularView<Eigen::Lower>().solveInPlace(m_vec_alpha_);
        m_mat_l_.transpose().triangularView<Eigen::Upper>().solveInPlace(m_vec_alpha_);

        // Log-GPIS
        auto log_ktrain = m_kernel_->ComputeKtrain(m_mat_x_train_);
        for (long i = 0; i < n; ++i) { log_ktrain(i, i) += vec_sigma_f[i]; }
        // auto log_ktrain = m_kernel_->ComputeKtrain(m_mat_x_train_, vec_sigma_f);
        m_mat_log_l_ = log_ktrain.llt().matrixL();
        m_vec_log_alpha_ = (vec_y.head(n).array() * (-m_setting_->log_lambda)).exp();
        m_mat_log_l_.triangularView<Eigen::Lower>().solveInPlace(m_vec_log_alpha_);
        m_mat_log_l_.transpose().triangularView<Eigen::Upper>().solveInPlace(m_vec_log_alpha_);

#if defined(BUILD_TEST)
        m_vec_y_ = vec_y;
        m_vec_sigma_grad_ = vec_sigma_grad;
        m_vec_sigma_x_ = vec_sigma_x;
        m_mat_k_train_ = ktrain;
#endif

        m_trained_ = true;
    }

    void
    LogNoisyInputGaussianProcess::Test(
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x_test,
        Eigen::Ref<Eigen::VectorXd> vec_f_out,
        Eigen::Ref<Eigen::VectorXd> vec_var_out) const {
        if (!m_trained_) { return; }

        long dim = mat_x_test.rows();
        long n = mat_x_test.cols();

        double f_threshold = std::exp(-m_setting_->log_lambda * std::abs(m_setting_->edf_threshold));

        auto ktest = NoisyInputGaussianProcess::m_kernel_->ComputeKtestWithGradient(m_mat_x_train_, m_vec_grad_flag_, mat_x_test);
        // auto log_ktest = m_kernel_->ComputeKtest(m_mat_x_train_, mat_x_test);
        auto log_ktest = m_kernel_->ComputeKtestWithGradient(m_mat_x_train_, Eigen::VectorXb::Constant(m_mat_x_train_.cols(), false), mat_x_test);

        // sdf sign and normal
        Eigen::VectorXd f_gpis = ktest.transpose() * m_vec_alpha_;
        // vec_f_out.tail(dim * n).reshaped(n, dim).rowwise().normalize();  // gradient norm is always 1. https://en.wikipedia.org/wiki/Eikonal_equation
        Eigen::VectorXd signs = f_gpis.head(n).array().unaryExpr([](double val) -> double { return val >= 0. ? 1. : double(-1.); });

        // distance
        // vec_f_out.head(n) = ((log_ktest.transpose() * m_vec_log_alpha_).array().log() / (-m_setting_->log_lambda)) * signs.array();
        // sdf = log(f) / -log_lambda, grad_sdf = grad_f / (-f * log_lambda)
        Eigen::VectorXd f_log_gpis = log_ktest.transpose() * m_vec_log_alpha_;
        for (long i = 0; i < n; ++i) {
            long gxi = i + n;
            long gyi = gxi + n;
            vec_f_out[i] = std::log(std::abs(f_log_gpis[i])) * signs[i] / -m_setting_->log_lambda;

            double &gx = vec_f_out[gxi];
            double &gy = vec_f_out[gyi];
            if (f_log_gpis[i] > f_threshold) {  // close to the surface
                gx = f_gpis[gxi];
                gy = f_gpis[gyi];
            } else {
                double d = -signs[i] / m_setting_->log_lambda * f_log_gpis[i];
                gx = f_log_gpis[gxi] * d;
                gy = f_log_gpis[gyi] * d;
            }
            double norm = std::sqrt(gx * gx + gy * gy);
            if (norm > 1.e-6) {
                gx /= norm;
                gy /= norm;
            }
        }

        // distance variance
        //        m_mat_log_l_.triangularView<Eigen::Lower>().solveInPlace(log_ktest);  // solve Lx = ktest -> x=m_l_.m_inv_() * ktest
        //        log_ktest = log_ktest.array().square();                               // we only need the diagonal elements
        //        Eigen::VectorXd u = log_ktest.colwise().sum();
        m_mat_l_.triangularView<Eigen::Lower>().solveInPlace(ktest);  // solve Lx = ktest -> x=m_l_.m_inv_() * ktest
        ktest = ktest.array().square();                               // we only need the diagonal elements
        Eigen::VectorXd u = ktest.colwise().sum();
        // std::cout << "mat_x_test.cols: " << mat_x_test.cols() << std::endl
        //           << "u.size: " << u.size() << std::endl
        //           << "vec_var_out.size: " << vec_var_out.size() << std::endl;
        // vec_var_out.head(n) = m_setting_->kernel->alpha - u.head(n).array();

        // normal variance
        // Eigen::VectorXd v = m_mat_l_.triangularView<Eigen::Lower>().solve(ktest.block(0, n, ktest.rows(), dim * n)).array().square().colwise().sum();
        // vec_var_out.tail(dim * n) = m_three_over_scale_square_ - v.tail(dim * n).array();
        // vec_var_out.tail(dim * n) = m_three_over_scale_square_ - u.tail(dim * n).array();

        vec_var_out = -u;
        vec_var_out.head(n).array() += m_setting_->kernel->alpha;
        vec_var_out.tail(dim * n).array() += m_three_over_scale_square_;
    }

    LogNoisyInputGaussianProcess::LogNoisyInputGaussianProcess()
        : LogNoisyInputGaussianProcess(std::make_shared<Setting>()) {}

    LogNoisyInputGaussianProcess::LogNoisyInputGaussianProcess(std::shared_ptr<Setting> setting)
        : NoisyInputGaussianProcess(setting),
          m_setting_(std::move(setting)) {}
}  // namespace erl::gaussian_process

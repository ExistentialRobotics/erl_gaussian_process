#include "erl_gaussian_process/noisy_input_gp.hpp"

namespace erl::gaussian_process {

    std::shared_ptr<NoisyInputGaussianProcess>
    NoisyInputGaussianProcess::Create() {
        return std::shared_ptr<NoisyInputGaussianProcess>(new NoisyInputGaussianProcess());
    }

    std::shared_ptr<NoisyInputGaussianProcess>
    NoisyInputGaussianProcess::Create(std::shared_ptr<Setting> setting) {
        return std::shared_ptr<NoisyInputGaussianProcess>(new NoisyInputGaussianProcess(std::move(setting)));
    }

    void
    NoisyInputGaussianProcess::Train(
        Eigen::MatrixXd mat_x_train,
        Eigen::VectorXb vec_grad_flag,
        const Eigen::Ref<const Eigen::VectorXd> &vec_y,
        const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_x,
        const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_f,
        const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_grad) {

        // Apply the latest setting
        if (m_kernel_ == nullptr) {
            m_three_over_scale_square_ = double(3.) / (m_setting_->kernel->scale * m_setting_->kernel->scale);
            m_kernel_ = Covariance::Create(m_setting_->kernel);
        }

        if (mat_x_train.cols() <= 0) { return; }

        m_mat_x_train_ = std::move(mat_x_train);
        m_vec_grad_flag_ = std::move(vec_grad_flag);

        auto ktrain = m_kernel_->ComputeKtrainWithGradient(m_mat_x_train_, m_vec_grad_flag_, vec_sigma_x, vec_sigma_f, vec_sigma_grad);

        m_mat_l_ = ktrain.llt().matrixL();
        m_vec_alpha_ = vec_y;
        m_mat_l_.triangularView<Eigen::Lower>().solveInPlace(m_vec_alpha_);
        m_mat_l_.transpose().triangularView<Eigen::Upper>().solveInPlace(m_vec_alpha_);

#if defined(BUILD_TEST)
        m_vec_y_ = vec_y;
        m_vec_sigma_grad_ = vec_sigma_grad;
        m_vec_sigma_x_ = vec_sigma_x;
        m_mat_k_train_ = ktrain;
#endif

        m_trained_ = true;
    }

    void
    NoisyInputGaussianProcess::Test(const Eigen::Ref<const Eigen::MatrixXd> &mat_x_test, Eigen::Ref<Eigen::VectorXd> vec_f_out, Eigen::Ref<Eigen::VectorXd> vec_var_out)
        const {
        if (!m_trained_) { return; }

        auto ktest = m_kernel_->ComputeKtestWithGradient(m_mat_x_train_, m_vec_grad_flag_, mat_x_test);
        vec_f_out = ktest.transpose() * m_vec_alpha_;
        (void) vec_f_out;
        m_mat_l_.triangularView<Eigen::Lower>().solveInPlace(ktest);
        ktest = ktest.array().square();

        Eigen::VectorXd v = ktest.colwise().sum();

        long dim = mat_x_test.rows();
        long n = mat_x_test.cols();
        vec_var_out.head(n) = m_setting_->kernel->alpha - v.head(n).array();               // variance of vec_f_out
        vec_var_out.tail(dim * n) = m_three_over_scale_square_ - v.tail(dim * n).array();  // variance of grad_f
    }

    NoisyInputGaussianProcess::NoisyInputGaussianProcess()
        : m_setting_(std::make_shared<Setting>()) {}

    NoisyInputGaussianProcess::NoisyInputGaussianProcess(std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)) {}
}  // namespace erl::gaussian_process

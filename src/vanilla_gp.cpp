#include "erl_gaussian_process/vanilla_gp.hpp"

namespace erl::gaussian_process {
    void
    VanillaGaussianProcess::Train(Eigen::MatrixXd mat_x_train, const Eigen::Ref<const Eigen::VectorXd> &vec_y, const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_y) {

        if (mat_x_train.cols() <= 0) { return; }

        // Create the kernel function with the latest setting
        if (m_kernel_ == nullptr) { m_kernel_ = Covariance::Create(m_setting_->kernel); }

        m_mat_x_train_ = std::move(mat_x_train);

        if (m_setting_->auto_normalize) {
            m_mean_ = vec_y.mean();
            m_std_ = std::max(double(1.e-6), std::sqrt(vec_y.array().square().mean() - m_mean_ * m_mean_));  // biased std
            m_vec_alpha_ = (vec_y.array() - m_mean_) / m_std_;
        } else {
            m_vec_alpha_ = vec_y;
        }

        auto ktrain = m_kernel_->ComputeKtrain(m_mat_x_train_, vec_sigma_y);

        // A = ktrain(mat_x_train, mat_x_train) + sigma * I = m_l_ @ m_l_.T
        // m_vec_alpha_ = A.m_inv_() @ vec_y
        m_mat_l_ = ktrain.llt().matrixL();
        m_mat_l_.triangularView<Eigen::Lower>().solveInPlace(m_vec_alpha_);
        m_mat_l_.transpose().triangularView<Eigen::Upper>().solveInPlace(m_vec_alpha_);

        m_trained_ = true;
    }

    void
    VanillaGaussianProcess::Test(const Eigen::Ref<const Eigen::MatrixXd> &mat_x_test, Eigen::Ref<Eigen::VectorXd> vec_f_out, Eigen::Ref<Eigen::VectorXd> vec_var_out) const {

        if (!m_trained_) { return; }

        const auto &kN = mat_x_test.cols();
        if (kN == 0) { return; }

        ERL_DEBUG_ASSERT(vec_f_out.size() == kN, "vec_f_out should be a %ld-dim vector instead of %ld.", kN, vec_f_out.size());
        ERL_DEBUG_ASSERT(vec_var_out.size() == kN, "vec_var_out should be a %ld-dim vector instead of %ld.", kN, vec_var_out.size());

        auto ktest = m_kernel_->ComputeKtest(m_mat_x_train_, mat_x_test);

        // xt is one column of mat_x_test
        // expectation of vec_f_out = ktest(xt, X) @ (ktest(X, X) + sigma * I).m_inv_() @ y
        vec_f_out = ktest.transpose() * m_vec_alpha_;
        if (m_setting_->auto_normalize) { vec_f_out.array() = vec_f_out.array() * m_std_ + m_mean_; }
        // variance of vec_f_out = ktest(xt, xt) - ktest(xt, X) @ (ktest(X, X) + sigma * I).m_inv_() @ ktest(X, xt) = ktest(xt, xt) - ktest(xt, X) @ (m_l_ @ m_l_.T).m_inv_() @ ktest(X, xt)
        m_mat_l_.triangularView<Eigen::Lower>().solveInPlace(ktest);
        ktest = ktest.array().pow(2);
        vec_var_out.array() = m_setting_->kernel->alpha - ktest.colwise().sum().array();
    }
}  // namespace erl::gaussian_process

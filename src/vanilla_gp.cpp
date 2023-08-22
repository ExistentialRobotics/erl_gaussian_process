#include "erl_gaussian_process/vanilla_gp.hpp"

namespace erl::gaussian_process {
    void
    VanillaGaussianProcess::Reset(long max_num_samples, long x_dim) {
        ERL_ASSERTM(x_dim > 0, "x_dim should be > 0.");
        long &x_dim_setting = m_setting_->kernel->x_dim;
        long &max_num_samples_setting = m_setting_->max_num_samples;
        if (max_num_samples_setting > 0 && x_dim_setting > 0) {  // memory already allocated
            ERL_ASSERTM(max_num_samples_setting >= max_num_samples, "max_num_samples should be <= %ld.", max_num_samples_setting);
        } else {
            ERL_ASSERTM(x_dim_setting <= 0 || x_dim_setting == x_dim, "x_dim should be %ld.", x_dim_setting);
            ERL_ASSERTM(AllocateMemory(max_num_samples, x_dim), "Failed to allocate memory.");
        }
        m_trained_ = false;
        std::shared_ptr<covariance::Covariance::Setting> kernel_setting = std::make_shared<covariance::Covariance::Setting>(*m_setting_->kernel);
        kernel_setting->x_dim = x_dim;  // x_dim is determined now, so we can set it now to improve performance
        m_kernel_ = covariance::Covariance::Create(kernel_setting);
        m_num_train_samples_ = 0;
        m_x_dim_ = x_dim;
    }

    void
    VanillaGaussianProcess::Train(long num_train_samples) {

        if (m_trained_) {
            ERL_WARN("The model has been trained. Please reset the model before training.");
            return;
        }

        m_num_train_samples_ = num_train_samples;
        if (m_num_train_samples_ <= 0) {
            ERL_WARN("num_train_samples = %ld, it should be > 0.", m_num_train_samples_);
            return;
        }

        // Compute kernel matrix
        auto mat_x_train = m_mat_x_train_.topLeftCorner(m_x_dim_, m_num_train_samples_);
        auto vec_var_h = m_vec_var_h_.head(m_num_train_samples_);
        std::pair<long, long> output_size = m_kernel_->ComputeKtrain(m_mat_k_train_, mat_x_train, vec_var_h);
        auto mat_ktrain = m_mat_k_train_.topLeftCorner(output_size.first, output_size.second);
        auto mat_l = m_mat_l_.topLeftCorner(output_size.first, output_size.second);
        auto vec_alpha = m_vec_alpha_.head(m_num_train_samples_);
        if (m_setting_->auto_normalize) {
            m_mean_ = vec_alpha.mean();
            m_std_ = std::max(1.e-6, std::sqrt(vec_alpha.cwiseAbs2().mean() - m_mean_ * m_mean_));  // biased std
            vec_alpha = (vec_alpha.array() - m_mean_) / m_std_;
        }
        mat_l = mat_ktrain.llt().matrixL();  // A = ktrain(mat_x_train, mat_x_train) + sigma * I = mat_l @ mat_l.T
        mat_l.triangularView<Eigen::Lower>().solveInPlace(vec_alpha);
        mat_l.transpose().triangularView<Eigen::Upper>().solveInPlace(vec_alpha);  // A.m_inv_() @ vec_alpha

        m_trained_ = true;
    }

    void
    VanillaGaussianProcess::Test(
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x_test,
        Eigen::Ref<Eigen::VectorXd> vec_f_out,
        Eigen::Ref<Eigen::VectorXd> vec_var_out) const {

        if (!m_trained_ || m_num_train_samples_ <= 0) { return; }

        long n = mat_x_test.cols();
        if (n == 0) { return; }
        ERL_ASSERTM(mat_x_test.rows() == m_x_dim_, "mat_x_test.rows() = %ld, it should be %ld.", mat_x_test.rows(), m_x_dim_);
        ERL_ASSERTM(vec_f_out.size() >= n, "vec_f_out size = %ld, it should be >= %ld.", vec_f_out.size(), n);
        std::pair<long, long> ktest_size = covariance::Covariance::GetMinimumKtestSize(m_num_train_samples_, 0, 0, n);
        Eigen::MatrixXd ktest(ktest_size.first, ktest_size.second);
        std::pair<long, long> output_size = m_kernel_->ComputeKtest(ktest, m_mat_x_train_.topLeftCorner(m_x_dim_, m_num_train_samples_), mat_x_test);
        ERL_DEBUG_ASSERT(
            (output_size.first == ktest_size.first && output_size.second == ktest_size.second),
            "output_size = (%ld, %ld), it should be (%ld, %ld).",
            output_size.first,
            output_size.second,
            ktest_size.first,
            ktest_size.second);

        // xt is one column of mat_x_test
        // expectation of vec_f_out = ktest(xt, X) @ (ktest(X, X) + sigma * I).m_inv_() @ y
        auto vec_alpha = m_vec_alpha_.head(output_size.first);
        if (m_setting_->auto_normalize) {
            for (long i = 0; i < output_size.first; ++i) { vec_f_out[i] = ktest.col(i).dot(vec_alpha) * m_std_ + m_mean_; }
        } else {
            for (long i = 0; i < output_size.second; ++i) { vec_f_out[i] = ktest.col(i).dot(vec_alpha); }
        }
        if (vec_var_out.size() == 0) { return; }  // only compute mean

        // variance of vec_f_out = ktest(xt, xt) - ktest(xt, X) @ (ktest(X, X) + sigma * I).m_inv_() @ ktest(X, xt)
        //                       = ktest(xt, xt) - ktest(xt, X) @ (m_l_ @ m_l_.T).m_inv_() @ ktest(X, xt)
        ERL_ASSERTM(vec_var_out.size() >= n, "vec_var_out size = %ld, it should be >= %ld.", vec_var_out.size(), n);
        m_mat_l_.topLeftCorner(output_size.first, output_size.first).triangularView<Eigen::Lower>().solveInPlace(ktest);
        for (long i = 0; i < ktest_size.second; ++i) { vec_var_out[i] = m_setting_->kernel->alpha - ktest.col(i).squaredNorm(); }
    }
}  // namespace erl::gaussian_process

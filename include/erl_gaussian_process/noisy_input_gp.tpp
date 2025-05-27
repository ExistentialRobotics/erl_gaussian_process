#pragma once

#include "noisy_input_gp.hpp"

#include "erl_common/serialization.hpp"
#include "erl_common/template_helper.hpp"

namespace erl::gaussian_process {
    template<typename Dtype>
    YAML::Node
    NoisyInputGaussianProcess<Dtype>::Setting::YamlConvertImpl::encode(const Setting& setting) {
        YAML::Node node;
        ERL_YAML_SAVE_ATTR(node, setting, kernel_type);
        ERL_YAML_SAVE_ATTR(node, setting, kernel_setting_type);
        ERL_YAML_SAVE_ATTR(node, setting, kernel);
        ERL_YAML_SAVE_ATTR(node, setting, max_num_samples);
        ERL_YAML_SAVE_ATTR(node, setting, no_gradient_observation);
        return node;
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::Setting::YamlConvertImpl::decode(
        const YAML::Node& node,
        Setting& setting) {
        using namespace common;

        if (!node.IsMap()) { return false; }
        ERL_YAML_LOAD_ATTR(node, setting, kernel_type);
        ERL_YAML_LOAD_ATTR(node, setting, kernel_setting_type);
        setting.kernel = YamlableBase::Create<CovarianceSetting>(setting.kernel_setting_type);
        ERL_YAML_LOAD_ATTR(node, setting, kernel);
        ERL_YAML_LOAD_ATTR(node, setting, max_num_samples);
        ERL_YAML_LOAD_ATTR(node, setting, no_gradient_observation);
        return true;
    }

    template<typename Dtype>
    NoisyInputGaussianProcess<Dtype>::TestResult::TestResult(
        const NoisyInputGaussianProcess* gp,
        const Eigen::Ref<const MatrixX>& mat_x_test,
        const bool will_predict_gradient)
        : m_gp_(NotNull(gp, true, "gp = nullptr.")),
          m_num_test_(mat_x_test.cols()),
          m_support_gradient_(will_predict_gradient),
          m_reduced_rank_kernel_(m_gp_->m_reduced_rank_kernel_),
          m_x_dim_(m_gp_->m_train_set_.x_dim),
          m_y_dim_(m_gp_->m_train_set_.y_dim) {

        ERL_DEBUG_ASSERT(m_gp_->IsTrained(), "The model has not been trained.");
        ERL_DEBUG_ASSERT(m_num_test_ > 0, "m_num_test_ = {}, it should be > 0.", m_num_test_);

        auto& [x_dim, y_dim, n_samples, num_samples_with_grad, x_train, y_train, grad_train, var_x, var_y, var_grad, grad_flag] =
            m_gp_->m_train_set_;

        // compute mean and gradient of the test queries
        const auto kernel = m_gp_->m_kernel_;
        auto [rows1, cols1] = kernel->GetMinimumKtestSize(
            n_samples,
            num_samples_with_grad,
            m_x_dim_,
            m_num_test_,
            m_support_gradient_);
        m_mat_k_test_.resize(rows1, cols1);
        auto [rows2, cols2] = kernel->ComputeKtestWithGradient(
            x_train,
            n_samples,
            grad_flag,
            mat_x_test,
            m_num_test_,
            m_support_gradient_,
            m_mat_k_test_);

#ifndef NDEBUG
        const long k_train_cols = m_gp_->m_k_train_cols_;
        const long k_test_cols = m_support_gradient_ ? m_num_test_ * (m_x_dim_ + 1) : m_num_test_;
#else
        (void) rows2, (void) cols2;
#endif
        ERL_DEBUG_ASSERT(
            rows1 == rows2 && cols1 == cols2,
            "output_size = ({}, {}), it should be ({}, {}).",
            rows2,
            cols2,
            rows1,
            cols1);
        ERL_DEBUG_ASSERT(
            m_mat_k_test_.rows() == k_train_cols,
            "ktest.rows() = {}, it should be {}.",
            m_mat_k_test_.rows(),
            k_train_cols);
        ERL_DEBUG_ASSERT(
            m_mat_k_test_.cols() >= k_test_cols,
            "ktest.cols() = {}, it should be {}.",
            m_mat_k_test_.cols(),
            k_test_cols);
    }

    template<typename Dtype>
    long
    NoisyInputGaussianProcess<Dtype>::TestResult::GetNumTest() const {
        return m_num_test_;
    }

    template<typename Dtype>
    long
    NoisyInputGaussianProcess<Dtype>::TestResult::GetDimX() const {
        return m_x_dim_;
    }

    template<typename Dtype>
    long
    NoisyInputGaussianProcess<Dtype>::TestResult::GetDimY() const {
        return m_y_dim_;
    }

    template<typename Dtype>
    const typename NoisyInputGaussianProcess<Dtype>::MatrixX&
    NoisyInputGaussianProcess<Dtype>::TestResult::GetKtest() const {
        return m_mat_k_test_;
    }

    template<typename Dtype>
    void
    NoisyInputGaussianProcess<Dtype>::TestResult::GetMean(
        const long y_index,
        Eigen::Ref<VectorX> vec_f_out,
        const bool parallel) const {
        (void) parallel;
        ERL_DEBUG_ASSERT(
            y_index >= 0 && y_index < m_y_dim_,
            "y_index = {}, it should be in [0, {}).",
            y_index,
            m_y_dim_);
        ERL_DEBUG_ASSERT(
            vec_f_out.size() >= m_num_test_,
            "vec_f_out.size() = {}, it should be >= {}.",
            vec_f_out.size(),
            m_num_test_);
        const auto alpha = m_gp_->m_mat_alpha_.col(y_index).head(m_gp_->m_k_train_cols_);
        Dtype* f = vec_f_out.data();
#pragma omp parallel for if (parallel) default(none) shared(alpha, f)
        for (long i = 0; i < m_num_test_; ++i) { f[i] = m_mat_k_test_.col(i).dot(alpha); }
    }

    template<typename Dtype>
    void
    NoisyInputGaussianProcess<Dtype>::TestResult::GetMean(
        const long index,
        const long y_index,
        Dtype& f) const {
        ERL_DEBUG_ASSERT(
            index >= 0 && index < m_num_test_,
            "index = {}, it should be in [0, {}).",
            index,
            m_num_test_);
        ERL_DEBUG_ASSERT(
            y_index >= 0 && y_index < m_y_dim_,
            "y_index = {}, it should be in [0, {}).",
            y_index,
            m_y_dim_);
        const auto alpha = m_gp_->m_mat_alpha_.col(y_index).head(m_gp_->m_k_train_cols_);
        f = m_mat_k_test_.col(index).dot(alpha);  // h_{y_index}(x_{index})
    }

    template<typename Dtype>
    Eigen::VectorXb
    NoisyInputGaussianProcess<Dtype>::TestResult::GetGradient(
        const long y_index,
        Eigen::Ref<MatrixX> mat_grad_out,
        const bool parallel) const {
        (void) parallel;
        ERL_DEBUG_ASSERT(
            m_support_gradient_,
            "m_support_gradient_ = false, it should be true to call GetGradient().");
        ERL_DEBUG_ASSERT(
            y_index >= 0 && y_index < m_y_dim_,
            "y_index = {}, it should be in [0, {}).",
            y_index,
            m_y_dim_);
        ERL_DEBUG_ASSERT(
            mat_grad_out.rows() >= m_x_dim_,
            "mat_grad_out.rows() = {}, it should be >= {}.",
            mat_grad_out.rows(),
            m_x_dim_);
        ERL_DEBUG_ASSERT(
            mat_grad_out.cols() >= m_num_test_,
            "mat_grad_out.cols() = {}, it should be >= {}.",
            mat_grad_out.cols(),
            m_num_test_);
        const auto alpha = m_gp_->m_mat_alpha_.col(y_index).head(m_gp_->m_k_train_cols_);
        Eigen::VectorXb valid_gradients(m_num_test_);
#pragma omp parallel for if (parallel) default(none) shared(alpha, mat_grad_out, valid_gradients)
        for (long index = 0; index < m_num_test_; ++index) {
            Dtype* grad = mat_grad_out.col(index).data();
            for (long j = 0, jj = index + m_num_test_; j < m_x_dim_; ++j, jj += m_num_test_) {
                *grad = m_mat_k_test_.col(jj).dot(alpha);
                if (!std::isfinite(*grad)) {
                    valid_gradients[index] = false;
                    break;
                }
                ++grad;
            }
        }
        return valid_gradients;
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::TestResult::GetGradient(
        const long index,
        const long y_index,
        Dtype* grad) const {
        ERL_DEBUG_ASSERT(
            m_support_gradient_,
            "m_support_gradient_ = false, it should be true to call GetGradient().");
        ERL_DEBUG_ASSERT(
            index >= 0 && index < m_num_test_,
            "index = {}, it should be in [0, {}).",
            index,
            m_num_test_);
        ERL_DEBUG_ASSERT(
            y_index >= 0 && y_index < m_y_dim_,
            "y_index = {}, it should be in [0, {}).",
            y_index,
            m_y_dim_);
        const auto alpha = m_gp_->m_mat_alpha_.col(y_index).head(m_gp_->m_k_train_cols_);
        for (long j = 0, jj = index + m_num_test_; j < m_x_dim_; ++j, jj += m_num_test_) {
            *grad = m_mat_k_test_.col(jj).dot(alpha);
            if (!std::isfinite(*grad)) { return false; }  // gradient is not valid
            ++grad;
        }
        return true;
    }

    template<typename Dtype>
    void
    NoisyInputGaussianProcess<Dtype>::TestResult::GetMeanVariance(
        Eigen::Ref<VectorX> vec_var_out,
        const bool parallel) const {
        (void) parallel;
        const_cast<TestResult*>(this)->PrepareAlphaTest();
        Dtype* var = vec_var_out.data();
#pragma omp parallel for if (parallel) default(none) shared(var)
        for (long i = 0; i < m_num_test_; ++i) {
            Dtype& var_i = var[i];
            var_i = m_mat_alpha_test_.col(i).squaredNorm();  // variance of h(x)
            if (m_reduced_rank_kernel_) { continue; }
            var_i = 1.0f - var_i;  // variance of h(x)
        }
    }

    template<typename Dtype>
    void
    NoisyInputGaussianProcess<Dtype>::TestResult::GetMeanVariance(long index, Dtype& var) const {
        const_cast<TestResult*>(this)->PrepareAlphaTest();
        var = m_mat_alpha_test_.col(index).squaredNorm();  // variance of h(x)
        if (m_reduced_rank_kernel_) { return; }
        var = 1.0f - var;  // variance of h(x)
    }

    template<typename Dtype>
    void
    NoisyInputGaussianProcess<Dtype>::TestResult::GetGradientVariance(
        Eigen::Ref<MatrixX> mat_var_out,
        const bool parallel) const {
        (void) parallel;
        ERL_DEBUG_ASSERT(
            m_support_gradient_,
            "m_support_gradient_ = false, it should be true to call GetGradient().");
        const_cast<TestResult*>(this)->PrepareAlphaTest();
        const Dtype scale_square = m_gp_->m_three_over_scale_square_;
        const long cols = m_mat_alpha_test_.cols();
#pragma omp parallel for if (parallel) default(none) shared(mat_var_out, scale_square, cols)
        for (long index = 0; index < m_num_test_; ++index) {
            Dtype* var = mat_var_out.col(index).data();
            for (long jj = index + m_num_test_; jj < cols; jj += m_num_test_, ++var) {
                *var = m_mat_alpha_test_.col(jj).squaredNorm();
                if (m_reduced_rank_kernel_) { continue; }
                *var = scale_square - *var;
            }
        }
    }

    template<typename Dtype>
    void
    NoisyInputGaussianProcess<Dtype>::TestResult::GetGradientVariance(const long index, Dtype* var)
        const {
        ERL_DEBUG_ASSERT(
            m_support_gradient_,
            "m_support_gradient_ = false, it should be true to call GetGradient().");
        const_cast<TestResult*>(this)->PrepareAlphaTest();
        const Dtype scale_square = m_gp_->m_three_over_scale_square_;
        const long cols = m_mat_alpha_test_.cols();
        for (long jj = index + m_num_test_; jj < cols; jj += m_num_test_, ++var) {
            *var = m_mat_alpha_test_.col(jj).squaredNorm();
            if (m_reduced_rank_kernel_) { continue; }
            *var = scale_square - *var;
        }
    }

    template<typename Dtype>
    void
    NoisyInputGaussianProcess<Dtype>::TestResult::GetCovariance(
        Eigen::Ref<MatrixX> mat_cov_out,
        const bool parallel) const {
        (void) parallel;
        ERL_DEBUG_ASSERT(
            m_support_gradient_,
            "m_support_gradient_ = false, it should be true to call GetGradient().");
        ERL_DEBUG_ASSERT(
            mat_cov_out.rows() >= m_x_dim_ * (m_x_dim_ + 1) / 2,
            "mat_cov_out.rows() = {}, it should be >= {}.",
            mat_cov_out.rows(),
            m_x_dim_ * (m_x_dim_ + 1) / 2);
        ERL_DEBUG_ASSERT(
            mat_cov_out.cols() >= m_num_test_,
            "mat_cov_out.cols() = {}, it should be >= {}.",
            mat_cov_out.cols(),
            m_num_test_);

        const_cast<TestResult*>(this)->PrepareAlphaTest();
#pragma omp parallel for if (parallel) default(none) shared(mat_cov_out)
        for (long index = 0; index < m_num_test_; ++index) {
            Dtype* cov = mat_cov_out.col(index).data();
            for (long j = 0, jj = index + m_num_test_; j < m_x_dim_; ++j, jj += m_num_test_) {
                VectorX col_jj = m_mat_alpha_test_.col(jj);
                if (!m_reduced_rank_kernel_) { col_jj = -col_jj; }
                *cov++ = col_jj.dot(m_mat_alpha_test_.col(index));  // cov(dh(x)/dx_j, h(x))
                for (long k = 0, kk = index + m_num_test_; k < j; ++k, kk += m_num_test_) {
                    *cov++ = col_jj.dot(m_mat_alpha_test_.col(kk));  // cov(dh(x)/dx_j, dh(x)/dx_k)
                }
            }
        }
    }

    template<typename Dtype>
    void
    NoisyInputGaussianProcess<Dtype>::TestResult::GetCovariance(const long index, Dtype* cov)
        const {
        // compute covariance only when TestResult is created with will_predict_gradient = true.
        // when will_predict_gradient = true, m_mat_k_test_.cols() = m_num_test_ * (x_dim + 1).
        ERL_DEBUG_ASSERT(
            m_support_gradient_,
            "m_support_gradient_ = false, it should be true to call GetGradient().");
        ERL_DEBUG_ASSERT(
            index >= 0 && index < m_num_test_,
            "index = {}, it should be in [0, {}).",
            index,
            m_num_test_);
        ERL_DEBUG_ASSERT(cov != nullptr, "cov should not be nullptr.");

        const_cast<TestResult*>(this)->PrepareAlphaTest();
        for (long j = 0, jj = index + m_num_test_; j < m_x_dim_; ++j, jj += m_num_test_) {
            VectorX col_jj = m_mat_alpha_test_.col(jj);
            if (!m_reduced_rank_kernel_) { col_jj = -col_jj; }
            *cov++ = col_jj.dot(m_mat_alpha_test_.col(index));  // cov(dh(x)/dx_j, h(x))
            for (long k = 0, kk = index + m_num_test_; k < j; ++k, kk += m_num_test_) {
                *cov++ = col_jj.dot(m_mat_alpha_test_.col(kk));  // cov(dh(x)/dx_j, dh(x)/dx_k)
            }
        }
    }

    template<typename Dtype>
    void
    NoisyInputGaussianProcess<Dtype>::TestResult::PrepareAlphaTest() {
        if (m_mat_alpha_test_.size() > 0) { return; }
        const long rows = m_mat_k_test_.rows();
        m_mat_alpha_test_ = m_gp_->m_mat_l_.topLeftCorner(rows, rows)
                                .template triangularView<Eigen::Lower>()
                                .solve(m_mat_k_test_);
    }

    template<typename Dtype>
    void
    NoisyInputGaussianProcess<Dtype>::TrainSet::Reset(
        long max_num_samples,
        long x_dim,
        long y_dim,
        const bool no_gradient_observation) {
        this->x_dim = x_dim;
        this->y_dim = y_dim;
        if (x.rows() < x_dim || x.cols() < max_num_samples) { x.resize(x_dim, max_num_samples); }
        if (y.rows() < max_num_samples || y.cols() < y_dim) { y.resize(max_num_samples, y_dim); }
        if (grad_flag.size() < max_num_samples) { grad_flag.resize(max_num_samples); }
        if (var_x.size() < max_num_samples) { var_x.resize(max_num_samples); }
        if (var_y.size() < max_num_samples) { var_y.resize(max_num_samples); }

        if (!no_gradient_observation) {
            // grad, var_grad are used when no_gradient_observation is false.
            if (grad.rows() < x_dim * y_dim || grad.cols() < max_num_samples) {
                grad.resize(x_dim * y_dim, max_num_samples);
            }
            if (var_grad.size() < max_num_samples) { var_grad.resize(max_num_samples); }
        }
        num_samples = 0;
        num_samples_with_grad = 0;
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::TrainSet::operator==(const TrainSet& other) const {
        if (x_dim != other.x_dim) { return false; }
        if (y_dim != other.y_dim) { return false; }
        if (num_samples != other.num_samples) { return false; }
        if (num_samples_with_grad != other.num_samples_with_grad) { return false; }
        if (num_samples == 0) { return true; }
        if (other.x.rows() < x_dim || other.x.cols() < num_samples) { return false; }
        if (x.topLeftCorner(x_dim, num_samples) != other.x.topLeftCorner(x_dim, num_samples)) {
            return false;
        }
        if (other.y.rows() < num_samples || other.y.cols() < y_dim) { return false; }
        if (y.topLeftCorner(num_samples, y_dim) != other.y.topLeftCorner(num_samples, y_dim)) {
            return false;
        }
        if (other.grad_flag.size() < num_samples) { return false; }
        if (grad_flag.head(num_samples) != other.grad_flag.head(num_samples)) { return false; }
        if (other.var_x.size() < num_samples) { return false; }
        if (var_x.head(num_samples) != other.var_x.head(num_samples)) { return false; }
        if (other.var_y.size() < num_samples) { return false; }
        if (var_y.head(num_samples) != other.var_y.head(num_samples)) { return false; }
        if (grad.size() == 0 && other.grad.size() == 0) { return true; }
        if (other.grad.rows() < x_dim * y_dim || other.grad.cols() < num_samples) { return false; }
        if (grad.topLeftCorner(x_dim * y_dim, num_samples) !=
            other.grad.topLeftCorner(x_dim * y_dim, num_samples)) {
            return false;
        }
        if (var_grad.size() == 0 && other.var_grad.size() == 0) { return true; }
        if (other.var_grad.size() < num_samples) { return false; }
        if (var_grad.head(num_samples) != other.var_grad.head(num_samples)) { return false; }
        return true;
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::TrainSet::Write(std::ostream& s) const {
        using namespace common;
        static const TokenWriteFunctionPairs<TrainSet> token_function_pairs = {
            {
                "x_dim",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    stream << train_set->x_dim;
                    return stream.good();
                },
            },
            {
                "y_dim",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    stream << train_set->y_dim;
                    return stream.good();
                },
            },
            {
                "num_samples",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    stream << train_set->num_samples;
                    return stream.good();
                },
            },
            {
                "num_samples_with_grad",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    stream << train_set->num_samples_with_grad;
                    return stream.good();
                },
            },
            {
                "x",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, train_set->x) && stream.good();
                },
            },
            {
                "y",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, train_set->y) && stream.good();
                },
            },
            {
                "grad",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, train_set->grad) && stream.good();
                },
            },
            {
                "var_x",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, train_set->var_x) && stream.good();
                },
            },
            {
                "var_y",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, train_set->var_y) && stream.good();
                },
            },
            {
                "var_grad",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, train_set->var_grad) &&
                           stream.good();
                },
            },
            {
                "grad_flag",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, train_set->grad_flag) &&
                           stream.good();
                },
            },
        };
        return WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::TrainSet::Read(std::istream& s) {
        using namespace common;
        static const TokenReadFunctionPairs<TrainSet> token_function_pairs = {
            {
                "x_dim",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    stream >> train_set->x_dim;
                    return stream.good();
                },
            },
            {
                "y_dim",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    stream >> train_set->y_dim;
                    return stream.good();
                },
            },
            {
                "num_samples",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    stream >> train_set->num_samples;
                    return stream.good();
                },
            },
            {
                "num_samples_with_grad",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    stream >> train_set->num_samples_with_grad;
                    return stream.good();
                },
            },
            {
                "x",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, train_set->x) && stream.good();
                },
            },
            {
                "y",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, train_set->y) && stream.good();
                },
            },
            {
                "grad",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, train_set->grad) &&
                           stream.good();
                },
            },
            {
                "var_x",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, train_set->var_x) &&
                           stream.good();
                },
            },
            {
                "var_y",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, train_set->var_y) &&
                           stream.good();
                },
            },
            {
                "var_grad",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, train_set->var_grad) &&
                           stream.good();
                },
            },
            {
                "grad_flag",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, train_set->grad_flag) &&
                           stream.good();
                },
            },
        };
        return ReadTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::TrainSet::operator!=(const TrainSet& other) const {
        return !(*this == other);
    }

    template<typename Dtype>
    NoisyInputGaussianProcess<Dtype>::NoisyInputGaussianProcess(std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)) {
        ERL_ASSERTM(m_setting_ != nullptr, "setting should not be nullptr.");
        ERL_ASSERTM(m_setting_->kernel != nullptr, "setting->kernel should not be nullptr.");
    }

    template<typename Dtype>
    NoisyInputGaussianProcess<Dtype>::NoisyInputGaussianProcess(
        const NoisyInputGaussianProcess& other)
        : m_setting_(other.m_setting_),
          m_trained_(other.m_trained_),
          m_trained_once_(other.m_trained_once_),
          m_k_train_updated_(other.m_k_train_updated_),
          m_k_train_rows_(other.m_k_train_rows_),
          m_k_train_cols_(other.m_k_train_cols_),
          m_three_over_scale_square_(other.m_three_over_scale_square_),
          m_reduced_rank_kernel_(other.m_reduced_rank_kernel_),
          m_mat_k_train_(other.m_mat_k_train_),
          m_mat_l_(other.m_mat_l_),
          m_mat_alpha_(other.m_mat_alpha_),
          m_train_set_(other.m_train_set_) {
        if (other.m_kernel_ != nullptr) {
            m_kernel_ = Covariance::CreateCovariance(m_setting_->kernel_type, m_setting_->kernel);
            if (m_reduced_rank_kernel_) {
                // rank-reduced kernel is stateful, so we need to copy the kernel.
                *std::reinterpret_pointer_cast<ReducedRankCovariance>(m_kernel_) =
                    *std::reinterpret_pointer_cast<ReducedRankCovariance>(other.m_kernel_);
            }
        }
    }

    template<typename Dtype>
    NoisyInputGaussianProcess<Dtype>&
    NoisyInputGaussianProcess<Dtype>::operator=(const NoisyInputGaussianProcess& other) {
        if (this == &other) { return *this; }
        m_setting_ = other.m_setting_;
        m_trained_ = other.m_trained_;
        m_trained_once_ = other.m_trained_once_;
        m_k_train_updated_ = other.m_k_train_updated_;
        m_k_train_rows_ = other.m_k_train_rows_;
        m_k_train_cols_ = other.m_k_train_cols_;
        m_three_over_scale_square_ = other.m_three_over_scale_square_;
        m_reduced_rank_kernel_ = other.m_reduced_rank_kernel_;
        m_mat_k_train_ = other.m_mat_k_train_;
        m_mat_l_ = other.m_mat_l_;
        m_mat_alpha_ = other.m_mat_alpha_;
        m_train_set_ = other.m_train_set_;
        if (other.m_kernel_ != nullptr) {
            m_kernel_ = Covariance::CreateCovariance(m_setting_->kernel_type, m_setting_->kernel);
            if (m_reduced_rank_kernel_) {
                // rank-reduced kernel is stateful, so we need to copy the kernel.
                *std::reinterpret_pointer_cast<ReducedRankCovariance>(m_kernel_) =
                    *std::reinterpret_pointer_cast<ReducedRankCovariance>(other.m_kernel_);
            }
        }
        return *this;
    }

    template<typename Dtype>
    template<typename T>
    std::shared_ptr<const T>
    NoisyInputGaussianProcess<Dtype>::GetSetting() const {
        return std::dynamic_pointer_cast<const T>(m_setting_);
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::IsTrained() const {
        return m_trained_;
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::UsingReducedRankKernel() const {
        return m_reduced_rank_kernel_;
    }

    template<typename Dtype>
    typename NoisyInputGaussianProcess<Dtype>::VectorX
    NoisyInputGaussianProcess<Dtype>::GetKernelCoordOrigin() const {
        if (m_reduced_rank_kernel_) {
            return std::reinterpret_pointer_cast<ReducedRankCovariance>(m_kernel_)
                ->GetCoordOrigin();
        }
        ERL_DEBUG_ASSERT(m_train_set_.x_dim > 0, "train set should be initialized first.");
        return VectorX::Zero(m_train_set_.x_dim);
    }

    template<typename Dtype>
    void
    NoisyInputGaussianProcess<Dtype>::SetKernelCoordOrigin(const VectorX& coord_origin) const {
        if (m_reduced_rank_kernel_) {
            std::reinterpret_pointer_cast<ReducedRankCovariance>(m_kernel_)->SetCoordOrigin(
                coord_origin);
        }
    }

    template<typename Dtype>
    void
    NoisyInputGaussianProcess<Dtype>::Reset(
        const long max_num_samples,
        const long x_dim,
        const long y_dim) {
        ERL_DEBUG_ASSERT(max_num_samples > 0, "max_num_samples should be > 0.");
        ERL_DEBUG_ASSERT(x_dim > 0, "x_dim should be > 0.");
        ERL_DEBUG_ASSERT(y_dim > 0, "y_dim should be > 0.");
        ERL_DEBUG_ASSERT(
            m_setting_->kernel->x_dim == -1 || m_setting_->kernel->x_dim == x_dim,
            "x_dim should be {}.",
            m_setting_->kernel->x_dim);
        ERL_DEBUG_ASSERT(
            m_setting_->max_num_samples < 0 || max_num_samples <= m_setting_->max_num_samples,
            "max_num_samples should be <= {}.",
            m_setting_->max_num_samples);

        m_train_set_.Reset(max_num_samples, x_dim, y_dim, m_setting_->no_gradient_observation);
        const bool success = AllocateMemory(max_num_samples, x_dim, y_dim);
        ERL_ASSERTM(success, "Failed to allocate memory.");
        m_trained_ = false;
        m_k_train_updated_ = false;
        m_k_train_rows_ = 0;
        m_k_train_cols_ = 0;
        m_three_over_scale_square_ = 3.0f / (m_setting_->kernel->scale * m_setting_->kernel->scale);
    }

    template<typename Dtype>
    std::shared_ptr<typename NoisyInputGaussianProcess<Dtype>::Covariance>
    NoisyInputGaussianProcess<Dtype>::GetKernel() const {
        return m_kernel_;
    }

    template<typename Dtype>
    typename NoisyInputGaussianProcess<Dtype>::TrainSet&
    NoisyInputGaussianProcess<Dtype>::GetTrainSet() {
        return m_train_set_;
    }

    template<typename Dtype>
    const typename NoisyInputGaussianProcess<Dtype>::TrainSet&
    NoisyInputGaussianProcess<Dtype>::GetTrainSet() const {
        return m_train_set_;
    }

    template<typename Dtype>
    const typename NoisyInputGaussianProcess<Dtype>::MatrixX&
    NoisyInputGaussianProcess<Dtype>::GetKtrain() const {
        return m_mat_k_train_;
    }

    template<typename Dtype>
    const typename NoisyInputGaussianProcess<Dtype>::MatrixX&
    NoisyInputGaussianProcess<Dtype>::GetAlpha() {
        return m_mat_alpha_;
    }

    template<typename Dtype>
    const typename NoisyInputGaussianProcess<Dtype>::MatrixX&
    NoisyInputGaussianProcess<Dtype>::GetCholeskyDecomposition() {
        return m_mat_l_;
    }

    template<typename Dtype>
    std::size_t
    NoisyInputGaussianProcess<Dtype>::GetMemoryUsage() const {
        std::size_t memory_usage = sizeof(NoisyInputGaussianProcess);
        if (m_setting_ != nullptr) { memory_usage += sizeof(Setting); }
        if (m_kernel_ != nullptr) { memory_usage += m_kernel_->GetMemoryUsage(); }
        memory_usage += sizeof(TrainSet);
        memory_usage += (m_train_set_.x.size() + m_train_set_.y.size() + m_train_set_.grad.size() +
                         m_train_set_.var_x.size() + m_train_set_.var_y.size() +
                         m_train_set_.var_grad.size() + m_train_set_.grad_flag.size()) *
                        sizeof(Dtype);
        memory_usage +=
            (m_mat_k_train_.size() + m_mat_l_.size() + m_mat_alpha_.size()) * sizeof(Dtype);
        return memory_usage;
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::UpdateKtrain() {
        if (m_k_train_updated_) { return true; }
        auto& [x_dim, y_dim, num_samples, num_samples_with_grad, x_train, y_train, grad_train, var_x, var_y, var_grad, grad_flag] =
            m_train_set_;
        if (num_samples <= 0) {
            ERL_WARN("num_samples = {}, it should be > 0.", num_samples);
            return false;
        }
        ERL_DEBUG_ASSERT(
            m_mat_alpha_.cols() == y_train.cols(),
            "m_mat_alpha_.cols() ({}) != y_train.cols() ({}).",
            m_mat_alpha_.cols(),
            y_train.cols());

        if (m_setting_->no_gradient_observation) {
            grad_flag.head(num_samples).setZero();
            m_mat_alpha_.topRows(num_samples) = y_train.topRows(num_samples);
            std::tie(m_k_train_rows_, m_k_train_cols_) = m_kernel_->ComputeKtrain(
                x_train,
                var_x + var_y,
                num_samples,
                m_mat_k_train_,
                m_mat_alpha_);
        } else {
            ERL_DEBUG_ASSERT(
                grad_flag.head(num_samples).count() == num_samples_with_grad,
                "grad_flag.head(num_samples).count() ({}) != num_samples_with_grad ({}).",
                grad_flag.head(num_samples).count(),
                num_samples_with_grad);
#ifndef NDEBUG
            const long m = num_samples + x_dim * num_samples_with_grad;
#endif
            ERL_DEBUG_ASSERT(
                m_mat_alpha_.rows() >= m,
                "m_mat_alpha_.rows() = {}, it should be >= {}.",
                m_mat_alpha_.rows(),
                m);

            const long* grad_flag_ptr = grad_flag.data();
            for (long d = 0; d < y_dim; ++d) {
                Dtype* alpha = m_mat_alpha_.col(d).data();
                const Dtype* y = y_train.col(d).data();
                std::memcpy(alpha, y, num_samples * sizeof(Dtype));  // h_d(x_i)
                for (long i = 0, j = num_samples; i < num_samples; ++i) {
                    if (!grad_flag_ptr[i]) { continue; }
                    const Dtype* grad_i = grad_train.col(i).data() + d * x_dim;
                    for (long k = 0, l = j++; k < x_dim; ++k, l += num_samples_with_grad) {
                        alpha[l] = grad_i[k];  // dh_d(x_i)/dx(k, i)
                    }
                }
            }

            // Compute kernel matrix
            std::tie(m_k_train_rows_, m_k_train_cols_) = m_kernel_->ComputeKtrainWithGradient(
                x_train,
                num_samples,
                grad_flag,
                var_x,
                var_y,
                var_grad,
                m_mat_k_train_,
                m_mat_alpha_);
        }
        ERL_DEBUG_ASSERT(
            !m_mat_k_train_.topLeftCorner(m_k_train_rows_, m_k_train_cols_).hasNaN(),
            "NaN in m_mat_k_train_!");
        m_k_train_updated_ = true;
        return true;
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::Train() {

        if (m_trained_) {
            ERL_WARN("The model has been trained. Please reset the model before training.");
            return false;
        }
        m_trained_ = m_trained_once_;
        if (!UpdateKtrain()) { return false; }
        // square matrix
        const auto mat_ktrain = m_mat_k_train_.topLeftCorner(m_k_train_rows_, m_k_train_cols_);
        // square lower triangular matrix
        auto&& mat_l = m_mat_l_.topLeftCorner(m_k_train_rows_, m_k_train_cols_);
        const auto alpha = m_mat_alpha_.topRows(m_k_train_cols_);  // h and gradient of h
        mat_l = mat_ktrain.llt().matrixL();
        mat_l.template triangularView<Eigen::Lower>().solveInPlace(alpha);
        mat_l.transpose().template triangularView<Eigen::Upper>().solveInPlace(alpha);
        m_trained_once_ = true;
        m_trained_ = true;
        return true;
    }

    template<typename Dtype>
    std::shared_ptr<typename NoisyInputGaussianProcess<Dtype>::TestResult>
    NoisyInputGaussianProcess<Dtype>::Test(
        const Eigen::Ref<const MatrixX>& mat_x_test,
        const bool predict_gradient) const {
        if (!m_trained_) { return nullptr; }
        return std::make_shared<TestResult>(this, mat_x_test, predict_gradient);
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::operator==(const NoisyInputGaussianProcess& other) const {
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr &&
            (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) {
            return false;
        }
        if (m_trained_ != other.m_trained_) { return false; }
        if (m_trained_once_ != other.m_trained_once_) { return false; }
        if (m_k_train_updated_ != other.m_k_train_updated_) { return false; }
        if (m_k_train_rows_ != other.m_k_train_rows_) { return false; }
        if (m_k_train_cols_ != other.m_k_train_cols_) { return false; }
        if (m_three_over_scale_square_ != other.m_three_over_scale_square_) { return false; }
        if (m_reduced_rank_kernel_ != other.m_reduced_rank_kernel_) { return false; }
        // kernel is not compared.
        if (other.m_mat_k_train_.rows() < m_k_train_rows_ ||
            other.m_mat_k_train_.cols() < m_k_train_cols_ ||
            m_mat_k_train_.topLeftCorner(m_k_train_rows_, m_k_train_cols_) !=
                other.m_mat_k_train_.topLeftCorner(m_k_train_rows_, m_k_train_cols_)) {
            return false;
        }
        if (other.m_mat_l_.rows() < m_k_train_rows_ || other.m_mat_l_.cols() < m_k_train_cols_ ||
            m_mat_l_.topLeftCorner(m_k_train_rows_, m_k_train_cols_) !=
                other.m_mat_l_.topLeftCorner(m_k_train_rows_, m_k_train_cols_)) {
            return false;
        }
        if (m_mat_alpha_.cols() != other.m_mat_alpha_.cols() ||
            other.m_mat_alpha_.rows() < m_k_train_cols_ ||
            m_mat_alpha_.topRows(m_k_train_cols_) != other.m_mat_alpha_.topRows(m_k_train_cols_)) {
            return false;
        }
        if (m_train_set_ != other.m_train_set_) { return false; }
        return true;
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::operator!=(const NoisyInputGaussianProcess& other) const {
        return !(*this == other);
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::Write(std::ostream& s) const {
        using namespace common;
        static const TokenWriteFunctionPairs<NoisyInputGaussianProcess> token_function_pairs = {
            {
                "setting",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    return gp->m_setting_->Write(stream) && stream.good();
                },
            },
            {
                "trained",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    stream << gp->m_trained_;
                    return stream.good();
                },
            },
            {
                "trained_once",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    stream << gp->m_trained_once_;
                    return stream.good();
                },
            },
            {
                "k_train_updated",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    stream << gp->m_k_train_updated_;
                    return stream.good();
                },
            },
            {
                "k_train_rows",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    stream << gp->m_k_train_rows_;
                    return stream.good();
                },
            },
            {
                "k_train_cols",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    stream << gp->m_k_train_cols_;
                    return stream.good();
                },
            },
            {
                "three_over_scale_square",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    stream.write(
                        reinterpret_cast<const char*>(&gp->m_three_over_scale_square_),
                        sizeof(Dtype));
                    return stream.good();
                },
            },
            {
                "kernel",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    stream << (gp->m_kernel_ != nullptr) << '\n';
                    if (gp->m_kernel_ != nullptr && !gp->m_kernel_->Write(stream)) { return false; }
                    return stream.good();
                },
            },
            {
                "mat_k_train",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, gp->m_mat_k_train_) &&
                           stream.good();
                },
            },
            {
                "mat_l",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, gp->m_mat_l_) && stream.good();
                },
            },
            {
                "mat_alpha",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, gp->m_mat_alpha_) && stream.good();
                },
            },
            {
                "train_set",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    return gp->m_train_set_.Write(stream) && stream.good();
                },
            },
        };
        return common::WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::Read(std::istream& s) {
        using namespace common;
        static const TokenReadFunctionPairs<NoisyInputGaussianProcess> token_function_pairs = {
            {
                "setting",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    return gp->m_setting_->Read(stream) && stream.good();
                },
            },
            {
                "trained",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    stream >> gp->m_trained_;
                    return stream.good();
                },
            },
            {
                "trained_once",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    stream >> gp->m_trained_once_;
                    return stream.good();
                },
            },
            {
                "k_train_updated",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    stream >> gp->m_k_train_updated_;
                    return stream.good();
                },
            },
            {
                "k_train_rows",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    stream >> gp->m_k_train_rows_;
                    return stream.good();
                },
            },
            {
                "k_train_cols",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    stream >> gp->m_k_train_cols_;
                    return stream.good();
                },
            },
            {
                "three_over_scale_square",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    stream.read(
                        reinterpret_cast<char*>(&gp->m_three_over_scale_square_),
                        sizeof(Dtype));
                    return stream.good();
                },
            },
            {
                "kernel",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    bool has_kernel;
                    stream >> has_kernel;
                    SkipLine(stream);
                    if (has_kernel) {
                        gp->m_kernel_ = Covariance::CreateCovariance(
                            gp->m_setting_->kernel_type,
                            gp->m_setting_->kernel);
                        if (!gp->m_kernel_->Read(stream)) { return false; }
                        const auto rank_reduced_kernel =
                            std::dynamic_pointer_cast<ReducedRankCovariance>(gp->m_kernel_);
                        gp->m_reduced_rank_kernel_ = rank_reduced_kernel != nullptr;
                        if (gp->m_reduced_rank_kernel_) {
                            rank_reduced_kernel->BuildSpectralDensities();
                        }
                    }
                    return true;
                },
            },
            {
                "mat_k_train",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, gp->m_mat_k_train_) &&
                           stream.good();
                },
            },
            {
                "mat_l",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, gp->m_mat_l_) && stream.good();
                },
            },
            {
                "mat_alpha",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, gp->m_mat_alpha_) &&
                           stream.good();
                },
            },
            {
                "train_set",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    return gp->m_train_set_.Read(stream) && stream.good();
                },
            },
        };
        return ReadTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::AllocateMemory(
        const long max_num_samples,
        const long x_dim,
        const long y_dim) {
        if (max_num_samples <= 0 || x_dim <= 0 || y_dim <= 0) { return false; }  // invalid input
        if (m_setting_->max_num_samples > 0 && max_num_samples > m_setting_->max_num_samples) {
            return false;
        }
        if (m_setting_->kernel->x_dim > 0 && x_dim != m_setting_->kernel->x_dim) { return false; }
        InitKernel();
        const auto [rows, cols] = m_kernel_->GetMinimumKtrainSize(
            max_num_samples,
            m_setting_->no_gradient_observation ? 0 : max_num_samples,
            x_dim);
        if (m_mat_k_train_.rows() < rows || m_mat_k_train_.cols() < cols) {
            m_mat_k_train_.resize(rows, cols);
        }
        if (m_mat_l_.rows() < rows || m_mat_l_.cols() < cols) { m_mat_l_.resize(rows, cols); }
        if (const long alpha_rows = std::max(
                max_num_samples * (m_setting_->no_gradient_observation ? 1 : x_dim + 1),
                cols);
            m_mat_alpha_.rows() < alpha_rows || m_mat_alpha_.cols() < y_dim) {
            m_mat_alpha_.resize(alpha_rows, y_dim);
        }
        return true;
    }

    template<typename Dtype>
    void
    NoisyInputGaussianProcess<Dtype>::InitKernel() {
        if (m_kernel_ != nullptr) { return; }
        m_kernel_ = Covariance::CreateCovariance(m_setting_->kernel_type, m_setting_->kernel);
        auto rank_reduced_kernel = std::dynamic_pointer_cast<ReducedRankCovariance>(m_kernel_);
        m_reduced_rank_kernel_ = rank_reduced_kernel != nullptr;
        if (m_reduced_rank_kernel_) { rank_reduced_kernel->BuildSpectralDensities(); }
    }

}  // namespace erl::gaussian_process

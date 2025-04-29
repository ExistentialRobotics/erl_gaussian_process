#pragma once

#include "erl_common/serialization.hpp"

namespace erl::gaussian_process {
    template<typename Dtype>
    YAML::Node
    NoisyInputGaussianProcess<Dtype>::Setting::YamlConvertImpl::encode(const Setting& setting) {
        YAML::Node node;
        node["kernel_type"] = setting.kernel_type;
        node["kernel_setting_type"] = setting.kernel_setting_type;
        node["kernel"] = setting.kernel;
        node["max_num_samples"] = setting.max_num_samples;
        node["no_gradient_observation"] = setting.no_gradient_observation;
        return node;
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::Setting::YamlConvertImpl::decode(const YAML::Node& node, Setting& setting) {
        if (!node.IsMap()) { return false; }
        setting.kernel_type = node["kernel_type"].as<std::string>();
        setting.kernel_setting_type = node["kernel_setting_type"].as<std::string>();
        setting.kernel = common::YamlableBase::Create<typename Covariance::Setting>(setting.kernel_setting_type);
        if (!setting.kernel->FromYamlNode(node["kernel"])) { return false; }
        setting.max_num_samples = node["max_num_samples"].as<long>();
        setting.no_gradient_observation = node["no_gradient_observation"].as<bool>();
        return true;
    }

    template<typename Dtype>
    void
    NoisyInputGaussianProcess<Dtype>::TrainSet::Reset(long max_num_samples, long x_dim, long y_dim, const bool no_gradient_observation) {
        this->x_dim = x_dim;
        this->y_dim = y_dim;
        if (x.rows() < x_dim || x.cols() < max_num_samples) { x.resize(x_dim, max_num_samples); }
        if (y.rows() < max_num_samples || y.cols() < y_dim) { y.resize(max_num_samples, y_dim); }
        if (grad_flag.size() < max_num_samples) { grad_flag.resize(max_num_samples); }
        if (var_x.size() < max_num_samples) { var_x.resize(max_num_samples); }
        if (var_y.size() < max_num_samples) { var_y.resize(max_num_samples); }

        if (!no_gradient_observation) {  // grad, var_grad are used when no_gradient_observation is false.
            if (grad.rows() < x_dim * y_dim || grad.cols() < max_num_samples) { grad.resize(x_dim * y_dim, max_num_samples); }
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
        if (x.topLeftCorner(x_dim, num_samples) != other.x.topLeftCorner(x_dim, num_samples)) { return false; }
        if (y.topLeftCorner(num_samples, y_dim) != other.y.topLeftCorner(num_samples, y_dim)) { return false; }
        if (grad_flag.head(num_samples) != other.grad_flag.head(num_samples)) { return false; }
        if (var_x.head(num_samples) != other.var_x.head(num_samples)) { return false; }
        if (var_y.head(num_samples) != other.var_y.head(num_samples)) { return false; }
        if (grad.size() == 0 && other.grad.size() == 0) { return true; }
        if (grad.size() != other.grad.size()) { return false; }
        if (grad.topLeftCorner(x_dim * y_dim, num_samples) != other.grad.topLeftCorner(x_dim * y_dim, num_samples)) { return false; }
        if (var_grad.size() == 0 && other.var_grad.size() == 0) { return true; }
        if (var_grad.size() != other.var_grad.size()) { return false; }
        if (var_grad.head(num_samples) != other.var_grad.head(num_samples)) { return false; }
        return true;
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::TrainSet::Write(std::ostream& s) const {
        static const std::vector<std::pair<const char*, std::function<bool(const TrainSet*, std::ostream&)>>> token_function_pairs = {
            {
                "x_dim",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    stream << train_set->x_dim;
                    return true;
                },
            },
            {
                "y_dim",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    stream << train_set->y_dim;
                    return true;
                },
            },
            {
                "num_samples",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    stream << train_set->num_samples;
                    return true;
                },
            },
            {
                "num_samples_with_grad",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    stream << train_set->num_samples_with_grad;
                    return true;
                },
            },
            {
                "x",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, train_set->x)) {
                        ERL_WARN("Failed to write x.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "y",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, train_set->y)) {
                        ERL_WARN("Failed to write y.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "grad",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, train_set->grad)) {
                        ERL_WARN("Failed to write grad.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "var_x",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, train_set->var_x)) {
                        ERL_WARN("Failed to write var_x.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "var_y",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, train_set->var_y)) {
                        ERL_WARN("Failed to write var_y.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "var_grad",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, train_set->var_grad)) {
                        ERL_WARN("Failed to write var_grad.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "grad_flag",
                [](const TrainSet* train_set, std::ostream& stream) -> bool {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, train_set->grad_flag)) {
                        ERL_WARN("Failed to write grad_flag.");
                        return false;
                    }
                    return true;
                },
            },
        };
        return common::WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::TrainSet::Read(std::istream& s) {
        static const std::vector<std::pair<const char*, std::function<bool(TrainSet*, std::istream&)>>> token_function_pairs = {
            {
                "x_dim",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    stream >> train_set->x_dim;
                    return true;
                },
            },
            {
                "y_dim",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    stream >> train_set->y_dim;
                    return true;
                },
            },
            {
                "num_samples",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    stream >> train_set->num_samples;
                    return true;
                },
            },
            {
                "num_samples_with_grad",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    stream >> train_set->num_samples_with_grad;
                    return true;
                },
            },
            {
                "x",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    common::SkipLine(stream);
                    if (!common::LoadEigenMatrixFromBinaryStream(stream, train_set->x)) {
                        ERL_WARN("Failed to read x.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "y",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    common::SkipLine(stream);
                    if (!common::LoadEigenMatrixFromBinaryStream(stream, train_set->y)) {
                        ERL_WARN("Failed to read y.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "grad",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    common::SkipLine(stream);
                    if (!common::LoadEigenMatrixFromBinaryStream(stream, train_set->grad)) {
                        ERL_WARN("Failed to read grad.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "var_x",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    common::SkipLine(stream);
                    if (!common::LoadEigenMatrixFromBinaryStream(stream, train_set->var_x)) {
                        ERL_WARN("Failed to read var_x.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "var_grad",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    common::SkipLine(stream);
                    if (!common::LoadEigenMatrixFromBinaryStream(stream, train_set->var_grad)) {
                        ERL_WARN("Failed to read var_grad.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "grad_flag",
                [](TrainSet* train_set, std::istream& stream) -> bool {
                    common::SkipLine(stream);
                    if (!common::LoadEigenMatrixFromBinaryStream(stream, train_set->grad_flag)) {
                        ERL_WARN("Failed to read grad_flag.");
                        return false;
                    }
                    return true;
                },
            },
        };
        return common::ReadTokens(s, this, token_function_pairs);
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
    NoisyInputGaussianProcess<Dtype>::NoisyInputGaussianProcess(const NoisyInputGaussianProcess& other)
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
            if (m_reduced_rank_kernel_) {  // rank-reduced kernel is stateful, so we need to copy the kernel
                *std::reinterpret_pointer_cast<covariance::ReducedRankCovariance>(m_kernel_) =
                    *std::reinterpret_pointer_cast<covariance::ReducedRankCovariance>(other.m_kernel_);
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
            if (m_reduced_rank_kernel_) {  // rank-reduced kernel is stateful, so we need to copy the kernel
                *std::reinterpret_pointer_cast<ReducedRankCovariance>(m_kernel_) = *std::reinterpret_pointer_cast<ReducedRankCovariance>(other.m_kernel_);
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
        if (m_reduced_rank_kernel_) { return std::reinterpret_pointer_cast<ReducedRankCovariance>(m_kernel_)->GetCoordOrigin(); }
        ERL_DEBUG_ASSERT(m_train_set_.x_dim > 0, "train set should be initialized first.");
        return VectorX::Zero(m_train_set_.x_dim);
    }

    template<typename Dtype>
    void
    NoisyInputGaussianProcess<Dtype>::SetKernelCoordOrigin(const VectorX& coord_origin) const {
        if (m_reduced_rank_kernel_) { std::reinterpret_pointer_cast<ReducedRankCovariance>(m_kernel_)->SetCoordOrigin(coord_origin); }
    }

    template<typename Dtype>
    void
    NoisyInputGaussianProcess<Dtype>::Reset(const long max_num_samples, const long x_dim, const long y_dim) {
        ERL_DEBUG_ASSERT(max_num_samples > 0, "max_num_samples should be > 0.");
        ERL_DEBUG_ASSERT(x_dim > 0, "x_dim should be > 0.");
        ERL_DEBUG_ASSERT(y_dim > 0, "y_dim should be > 0.");
        ERL_DEBUG_ASSERT(m_setting_->kernel->x_dim == -1 || m_setting_->kernel->x_dim == x_dim, "x_dim should be {}.", m_setting_->kernel->x_dim);
        ERL_DEBUG_ASSERT(
            m_setting_->max_num_samples < 0 || max_num_samples <= m_setting_->max_num_samples,
            "max_num_samples should be <= {}.",
            m_setting_->max_num_samples);

        m_train_set_.Reset(max_num_samples, x_dim, y_dim, m_setting_->no_gradient_observation);
        ERL_ASSERTM(AllocateMemory(max_num_samples, x_dim, y_dim), "Failed to allocate memory.");
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
        memory_usage += (m_train_set_.x.size() + m_train_set_.y.size() + m_train_set_.grad.size() + m_train_set_.var_x.size() +  //
                         m_train_set_.var_y.size() + m_train_set_.var_grad.size() + m_train_set_.grad_flag.size()) *
                        sizeof(Dtype);
        memory_usage += (m_mat_k_train_.size() + m_mat_l_.size() + m_mat_alpha_.size()) * sizeof(Dtype);
        return memory_usage;
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::UpdateKtrain() {
        if (m_k_train_updated_) { return true; }
        auto& [x_dim, y_dim, num_samples, num_samples_with_grad, x_train, y_train, grad_train, var_x, var_y, var_grad, grad_flag] = m_train_set_;
        if (num_samples <= 0) {
            ERL_WARN("num_samples = {}, it should be > 0.", num_samples);
            return false;
        }
        ERL_DEBUG_ASSERT(m_mat_alpha_.cols() == y_train.cols(), "m_mat_alpha_.cols() ({}) != y_train.cols() ({}).", m_mat_alpha_.cols(), y_train.cols());

        if (m_setting_->no_gradient_observation) {
            grad_flag.head(num_samples).setZero();
            m_mat_alpha_.topRows(num_samples) = y_train.topRows(num_samples);
            std::tie(m_k_train_rows_, m_k_train_cols_) = m_kernel_->ComputeKtrain(x_train, var_x + var_y, num_samples, m_mat_k_train_, m_mat_alpha_);
        } else {
            ERL_DEBUG_ASSERT(
                grad_flag.head(num_samples).count() == num_samples_with_grad,
                "grad_flag.head(num_samples).count() ({}) != num_samples_with_grad ({}).",
                grad_flag.head(num_samples).count(),
                num_samples_with_grad);
#ifndef NDEBUG
            const long m = num_samples + x_dim * num_samples_with_grad;
#endif
            ERL_DEBUG_ASSERT(m_mat_alpha_.rows() >= m, "m_mat_alpha_.rows() = {}, it should be >= {}.", m_mat_alpha_.rows(), m);

            const long* grad_flag_ptr = grad_flag.data();
            for (long d = 0; d < y_dim; ++d) {
                Dtype* alpha = m_mat_alpha_.col(d).data();
                const Dtype* y = y_train.col(d).data();
                std::memcpy(alpha, y, num_samples * sizeof(Dtype));  // h_d(x_i)
                for (long i = 0, j = num_samples; i < num_samples; ++i) {
                    if (!grad_flag_ptr[i]) { continue; }
                    const Dtype* grad_i = grad_train.col(i).data() + d * x_dim;
                    for (long k = 0, l = j++; k < x_dim; ++k, l += num_samples_with_grad) { alpha[l] = grad_i[k]; }  // dh_d(x_i)/dx(k, i)
                }
            }

            // Compute kernel matrix
            std::tie(m_k_train_rows_, m_k_train_cols_) =
                m_kernel_->ComputeKtrainWithGradient(x_train, num_samples, grad_flag, var_x, var_y, var_grad, m_mat_k_train_, m_mat_alpha_);
        }
        ERL_DEBUG_ASSERT(!m_mat_k_train_.topLeftCorner(m_k_train_rows_, m_k_train_cols_).hasNaN(), "NaN in m_mat_k_train_!");
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

        const auto mat_ktrain = m_mat_k_train_.topLeftCorner(m_k_train_rows_, m_k_train_cols_);  // square matrix
        auto&& mat_l = m_mat_l_.topLeftCorner(m_k_train_rows_, m_k_train_cols_);                 // square lower triangular matrix
        const auto alpha = m_mat_alpha_.topRows(m_k_train_cols_);                                // h and gradient of h
        mat_l = mat_ktrain.llt().matrixL();
        mat_l.template triangularView<Eigen::Lower>().solveInPlace(alpha);
        mat_l.transpose().template triangularView<Eigen::Upper>().solveInPlace(alpha);
        m_trained_once_ = true;
        m_trained_ = true;
        return true;
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::Test(
        const Eigen::Ref<const MatrixX>& mat_x_test,
        const std::vector<std::pair<long, bool>>& y_index_grad_pairs,
        Eigen::Ref<MatrixX> mat_f_out,
        Eigen::Ref<MatrixX> mat_var_out,
        Eigen::Ref<MatrixX> mat_cov_out) const {

        if (!m_trained_) {
            ERL_WARN("The model has not been trained.");
            return false;
        }

        const long num_test = mat_x_test.cols();
        if (num_test == 0) { return false; }

        auto& [x_dim, y_dim, num_samples, num_samples_with_grad, x_train, y_train, grad_train, var_x, var_y, var_grad, grad_flag] = m_train_set_;

        // compute mean and gradient of the test queries
        const bool predict_gradient = std::any_of(y_index_grad_pairs.begin(), y_index_grad_pairs.end(), [](const auto& pair) { return pair.second; });
        const auto [rows1, cols1] = m_kernel_->GetMinimumKtestSize(num_samples, num_samples_with_grad, x_dim, num_test, predict_gradient);
        MatrixX ktest(rows1, cols1);
        const auto [rows2, cols2] = m_kernel_->ComputeKtestWithGradient(x_train, num_samples, grad_flag, mat_x_test, num_test, predict_gradient, ktest);
        (void) rows2, (void) cols2;
        ERL_DEBUG_ASSERT(rows1 == rows2 && cols1 == cols2, "output_size = ({}, {}), it should be ({}, {}).", rows2, cols2, rows1, cols1);

        ComputeValuePrediction(ktest, num_test, y_index_grad_pairs, mat_f_out);
        ComputeCovPrediction(ktest, num_test, predict_gradient, mat_var_out, mat_cov_out);
        return true;
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::operator==(const NoisyInputGaussianProcess& other) const {
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr && (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) { return false; }
        if (m_trained_ != other.m_trained_) { return false; }
        if (m_trained_once_ != other.m_trained_once_) { return false; }
        if (m_k_train_updated_ != other.m_k_train_updated_) { return false; }
        if (m_k_train_rows_ != other.m_k_train_rows_) { return false; }
        if (m_k_train_cols_ != other.m_k_train_cols_) { return false; }
        if (m_three_over_scale_square_ != other.m_three_over_scale_square_) { return false; }
        if (m_reduced_rank_kernel_ != other.m_reduced_rank_kernel_) { return false; }
        if (m_train_set_ != other.m_train_set_) { return false; }
        if (m_mat_k_train_.rows() != other.m_mat_k_train_.rows() || m_mat_k_train_.cols() != other.m_mat_k_train_.cols() ||
            std::memcmp(m_mat_k_train_.data(), other.m_mat_k_train_.data(), m_mat_k_train_.size() * sizeof(Dtype)) != 0) {
            return false;
        }
        if (m_mat_l_.rows() != other.m_mat_l_.rows() || m_mat_l_.cols() != other.m_mat_l_.cols() ||
            std::memcmp(m_mat_l_.data(), other.m_mat_l_.data(), m_mat_l_.size() * sizeof(Dtype)) != 0) {
            return false;
        }
        if (m_mat_alpha_.size() != other.m_mat_alpha_.size() ||
            std::memcmp(m_mat_alpha_.data(), other.m_mat_alpha_.data(), m_mat_alpha_.size() * sizeof(Dtype)) != 0) {
            return false;
        }
        return true;
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::operator!=(const NoisyInputGaussianProcess& other) const {
        return !(*this == other);
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::Write(const std::string& filename) const {
        ERL_INFO("Writing {} to file: {}", type_name(*this), filename);
        std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
        std::ofstream file(filename, std::ios_base::out | std::ios_base::binary);
        if (!file.is_open()) {
            ERL_WARN("Failed to open file: {}", filename);
            return false;
        }

        const bool success = Write(file);
        file.close();
        return success;
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::Write(std::ostream& s) const {
        s << "# " << type_name(*this) << "\n# (feel free to add / change comments, but leave the first line as it is!)\n";

        static const std::vector<std::pair<const char*, std::function<bool(const NoisyInputGaussianProcess*, std::ostream&)>>> token_function_pairs = {
            {
                "setting",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    if (!gp->m_setting_->Write(stream)) {
                        ERL_WARN("Failed to write setting.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "trained",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    stream << gp->m_trained_;
                    return true;
                },
            },
            {
                "trained_once",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    stream << gp->m_trained_once_;
                    return true;
                },
            },
            {
                "ktrain_updated",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    stream << gp->m_k_train_updated_;
                    return true;
                },
            },
            {
                "ktrain_rows",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    stream << gp->m_k_train_rows_;
                    return true;
                },
            },
            {
                "ktrain_cols",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    stream << gp->m_k_train_cols_;
                    return true;
                },
            },
            {
                "three_over_scale_square",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    stream << gp->m_three_over_scale_square_;
                    return true;
                },
            },
            {
                "kernel",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    stream << (gp->m_kernel_ != nullptr) << '\n';
                    if (gp->m_kernel_ != nullptr && !gp->m_kernel_->Write(stream)) {
                        ERL_WARN("Failed to write kernel.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "mat_k_train",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, gp->m_mat_k_train_)) {
                        ERL_WARN("Failed to write mat_k_train.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "mat_l",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, gp->m_mat_l_)) {
                        ERL_WARN("Failed to write mat_l.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "mat_alpha",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, gp->m_mat_alpha_)) {
                        ERL_WARN("Failed to write mat_alpha.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "train_set",
                [](const NoisyInputGaussianProcess* gp, std::ostream& stream) -> bool {
                    if (!gp->m_train_set_.Write(stream)) {
                        ERL_WARN("Failed to write train_set.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "end_of_NoisyInputGaussianProcess",
                [](const NoisyInputGaussianProcess*, std::ostream&) -> bool { return true; },
            },
        };
        return common::WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::Read(const std::string& filename) {
        ERL_INFO("Reading {} from file: {}", type_name(*this), filename);
        std::ifstream file(filename, std::ios_base::in | std::ios_base::binary);
        if (!file.is_open()) {
            ERL_WARN("Failed to open file: {}", filename);
            return false;
        }

        const bool success = Read(file);
        file.close();
        return success;
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::Read(std::istream& s) {
        if (!s.good()) {
            ERL_WARN("Input stream is not ready for reading");
            return false;
        }

        // check if the first line is valid
        std::string line;
        std::getline(s, line);
        if (std::string file_header = fmt::format("# {}", type_name(*this));
            line.compare(0, file_header.length(), file_header) != 0) {  // check if the first line is valid
            ERL_WARN("Header does not start with \"{}\"", file_header);
            return false;
        }

        static const std::vector<std::pair<const char*, std::function<bool(NoisyInputGaussianProcess*, std::istream&)>>> token_function_pairs = {
            {
                "setting",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    common::SkipLine(stream);
                    if (!gp->m_setting_->Read(stream)) {
                        ERL_WARN("Failed to read setting.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "trained",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    stream >> gp->m_trained_;
                    return true;
                },
            },
            {
                "trained_once",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    stream >> gp->m_trained_once_;
                    return true;
                },
            },
            {
                "ktrain_updated",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    stream >> gp->m_k_train_updated_;
                    return true;
                },
            },
            {
                "ktrain_rows",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    stream >> gp->m_k_train_rows_;
                    return true;
                },
            },
            {
                "ktrain_cols",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    stream >> gp->m_k_train_cols_;
                    return true;
                },
            },
            {
                "three_over_scale_square",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    char c;
                    do { c = static_cast<char>(stream.get()); } while (stream.good() && c != '\n');
                    stream.read(reinterpret_cast<char*>(&gp->m_three_over_scale_square_), sizeof(Dtype));
                    return true;
                },
            },
            {
                "kernel",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    bool has_kernel;
                    stream >> has_kernel;
                    if (has_kernel) {
                        common::SkipLine(stream);
                        gp->m_kernel_ = Covariance::CreateCovariance(gp->m_setting_->kernel_type, gp->m_setting_->kernel);
                        if (!gp->m_kernel_->Read(stream)) {
                            ERL_WARN("Failed to read kernel.");
                            return false;
                        }
                        const auto rank_reduced_kernel = std::dynamic_pointer_cast<ReducedRankCovariance>(gp->m_kernel_);
                        gp->m_reduced_rank_kernel_ = rank_reduced_kernel != nullptr;
                        if (gp->m_reduced_rank_kernel_) { rank_reduced_kernel->BuildSpectralDensities(); }
                    }
                    return true;
                },
            },
            {
                "mat_k_train",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    common::SkipLine(stream);
                    if (!common::LoadEigenMatrixFromBinaryStream(stream, gp->m_mat_k_train_)) {
                        ERL_WARN("Failed to read mat_k_train.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "mat_l",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    common::SkipLine(stream);
                    if (!common::LoadEigenMatrixFromBinaryStream(stream, gp->m_mat_l_)) {
                        ERL_WARN("Failed to read mat_l.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "mat_alpha",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    common::SkipLine(stream);
                    if (!common::LoadEigenMatrixFromBinaryStream(stream, gp->m_mat_alpha_)) {
                        ERL_WARN("Failed to read mat_alpha.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "train_set",
                [](NoisyInputGaussianProcess* gp, std::istream& stream) -> bool {
                    if (!gp->m_train_set_.Read(stream)) {
                        ERL_WARN("Failed to read train set.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "end_of_NoisyInputGaussianProcess",
                [](NoisyInputGaussianProcess*, std::istream& stream) -> bool {
                    common::SkipLine(stream);
                    return true;
                },
            },
        };
        return common::ReadTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    NoisyInputGaussianProcess<Dtype>::AllocateMemory(const long max_num_samples, const long x_dim, const long y_dim) {
        if (max_num_samples <= 0 || x_dim <= 0 || y_dim <= 0) { return false; }  // invalid input
        if (m_setting_->max_num_samples > 0 && max_num_samples > m_setting_->max_num_samples) { return false; }
        if (m_setting_->kernel->x_dim > 0 && x_dim != m_setting_->kernel->x_dim) { return false; }
        InitKernel();
        const auto [rows, cols] = m_kernel_->GetMinimumKtrainSize(max_num_samples, m_setting_->no_gradient_observation ? 0 : max_num_samples, x_dim);
        if (m_mat_k_train_.rows() < rows || m_mat_k_train_.cols() < cols) { m_mat_k_train_.resize(rows, cols); }
        if (m_mat_l_.rows() < rows || m_mat_l_.cols() < cols) { m_mat_l_.resize(rows, cols); }
        if (const long alpha_rows = std::max(max_num_samples * (m_setting_->no_gradient_observation ? 1 : (x_dim + 1)), cols);  //
            m_mat_alpha_.rows() < alpha_rows || m_mat_alpha_.cols() < y_dim) {
            m_mat_alpha_.resize(alpha_rows, y_dim);
        }
        return true;
    }

    template<typename Dtype>
    void
    NoisyInputGaussianProcess<Dtype>::InitKernel() {
        if (m_kernel_ == nullptr) {
            m_kernel_ = Covariance::CreateCovariance(m_setting_->kernel_type, m_setting_->kernel);
            const auto rank_reduced_kernel = std::dynamic_pointer_cast<ReducedRankCovariance>(m_kernel_);
            m_reduced_rank_kernel_ = rank_reduced_kernel != nullptr;
            if (m_reduced_rank_kernel_) { rank_reduced_kernel->BuildSpectralDensities(); }
        }
    }

    template<typename Dtype>
    void
    NoisyInputGaussianProcess<Dtype>::ComputeValuePrediction(
        const MatrixX& ktest,
        const long n_test,
        const std::vector<std::pair<long, bool>>& y_index_grad_pairs,
        Eigen::Ref<MatrixX> mat_f_out) const {

        // compute value prediction
        /// ktest.T * alpha = [h(x1),...,h(xn),dh(x1)/dx_1,...,dh(xn)/dx_1,...,dh(x1)/dx_dim,...,dh(xn)/dx_dim]
#ifndef NDEBUG
        const long f_rows = std::accumulate(y_index_grad_pairs.begin(), y_index_grad_pairs.end(), 0l, [&](long acc, const std::pair<long, bool>& pair) {
            return acc + (pair.second ? 1 + m_train_set_.x_dim : 1);
        });
#endif
        ERL_DEBUG_ASSERT(mat_f_out.rows() >= f_rows, "mat_f_out.rows() = {}, it should be >= {}.", mat_f_out.rows(), f_rows);
        ERL_DEBUG_ASSERT(mat_f_out.cols() >= n_test, "mat_f_out.cols() = {}, not enough for {} test queries.", mat_f_out.cols(), n_test);
        ERL_DEBUG_ASSERT(ktest.rows() == m_k_train_cols_, "ktest.rows() = {}, it should be {}.", ktest.rows(), m_k_train_cols_);

        for (long i = 0; i < n_test; ++i) {
            Dtype* f = mat_f_out.col(i).data();
            for (const auto& [y_idx, predict_gradient]: y_index_grad_pairs) {
                ERL_DEBUG_ASSERT(y_idx >= 0 && y_idx < m_train_set_.y_dim, "y_idx = {}, it should be in [0, {}).", y_idx, m_train_set_.y_dim);
                const auto alpha = m_mat_alpha_.col(y_idx).head(m_k_train_cols_);
                *f++ = ktest.col(i).dot(alpha);  // h_d(x_i)
                if (!predict_gradient) { continue; }
                for (long j = 0, jj = i + n_test; j < m_train_set_.x_dim; ++j, jj += n_test) { *f++ = ktest.col(jj).dot(alpha); }  // dh_d(x_i)/dx(j-1, i)
            }
        }
    }

    template<typename Dtype>
    void
    NoisyInputGaussianProcess<Dtype>::ComputeCovPrediction(
        MatrixX& ktest,
        const long n_test,
        const bool predict_gradient,
        Eigen::Ref<MatrixX> mat_var_out,
        Eigen::Ref<MatrixX> mat_cov_out) const {

        const bool compute_var = mat_var_out.size() > 0;
        const bool compute_cov = mat_cov_out.size() > 0;
        if (!compute_var && !compute_cov) { return; }  // only compute mean

        const long ktest_rows = ktest.rows();

        // compute (co)variance of the test queries
        m_mat_l_.topLeftCorner(ktest_rows, ktest_rows).template triangularView<Eigen::Lower>().solveInPlace(ktest);
        const long x_dim = m_train_set_.x_dim;
        if (compute_var) {
            ERL_DEBUG_ASSERT(
                mat_var_out.rows() >= (predict_gradient ? 1 + x_dim : 1),
                "mat_var_out.rows() = {}, it should be >= {} for variance.",
                mat_var_out.rows(),
                predict_gradient ? 1 + x_dim : 1);
            ERL_DEBUG_ASSERT(mat_var_out.cols() >= n_test, "mat_var_out.cols() = {}, not enough for {} test queries.", mat_var_out.cols(), n_test);
        }
        if (!compute_cov) {  // compute variance only
            // column-wise square sum of ktest = var([h(x1),...,h(xn),dh(x1)/dx_1,...,dh(xn)/dx_1,...,dh(x1)/dx_dim,...,dh(xn)/dx_dim])
            if (m_reduced_rank_kernel_) {
                for (long i = 0; i < n_test; ++i) {
                    Dtype* var = mat_var_out.col(i).data();
                    *var = ktest.col(i).squaredNorm();  // variance of h(x)
                    if (!predict_gradient) { continue; }
                    for (long j = 1, jj = i + n_test; j <= x_dim; ++j, jj += n_test) { var[j] = ktest.col(jj).squaredNorm(); }  // variance of dh(x)/dx_j1
                }
            } else {
                for (long i = 0; i < n_test; ++i) {
                    Dtype* var = mat_var_out.col(i).data();
                    *var = 1.0f - ktest.col(i).squaredNorm();  // variance of h(x)
                    if (!predict_gradient) { continue; }
                    // variance of dh(x)/dx_j
                    for (long j = 1, jj = i + n_test; j <= x_dim; ++j, jj += n_test) { var[j] = m_three_over_scale_square_ - ktest.col(jj).squaredNorm(); }
                }
            }
            return;
        }

        if (!predict_gradient) { return; }

        // compute covariance, but only when predict_gradient is true
        ERL_DEBUG_ASSERT(
            mat_cov_out.rows() >= (x_dim + 1) * x_dim / 2,
            "mat_cov_out.rows() = {}, it should be >= {} for covariance.",
            mat_cov_out.rows(),
            (x_dim + 1) * x_dim / 2);
        ERL_DEBUG_ASSERT(mat_cov_out.cols() >= n_test, "mat_cov_out.cols() = {}, not enough for {} test queries.", mat_cov_out.cols(), n_test);
        // each column of mat_cov_out is the lower triangular part of the covariance matrix of the corresponding test query
        if (m_reduced_rank_kernel_) {
            for (long i = 0; i < n_test; ++i) {
                Dtype* var = nullptr;
                if (compute_var) {
                    var = mat_var_out.col(i).data();
                    *var = ktest.col(i).squaredNorm();  // var(h(x))
                }
                Dtype* cov = mat_cov_out.col(i).data();
                for (long j = 1, jj = i + n_test; j <= x_dim; ++j, jj += n_test) {
                    const auto& col_jj = ktest.col(jj);
                    *cov++ = col_jj.dot(ktest.col(i));                                                                   // cov(dh(x)/dx_j, h(x))
                    for (long k = 1, kk = i + n_test; k < j; ++k, kk += n_test) { *cov++ = col_jj.dot(ktest.col(kk)); }  // cov(dh(x)/dx_j, dh(x)/dx_k)
                    if (var != nullptr) { var[j] = col_jj.squaredNorm(); }                                               // var(dh(x)/dx_j)
                }
            }
            return;
        }

        for (long i = 0; i < n_test; ++i) {
            Dtype* var = nullptr;
            if (compute_var) {
                var = mat_var_out.col(i).data();
                *var = 1.0f - ktest.col(i).squaredNorm();  // var(h(x))
            }
            Dtype* cov = mat_cov_out.col(i).data();
            for (long j = 1, jj = i + n_test; j <= x_dim; ++j, jj += n_test) {
                const auto& col_jj = ktest.col(jj);
                *cov++ = -col_jj.dot(ktest.col(i));                                                                   // cov(dh(x)/dx_j, h(x))
                for (long k = 1, kk = i + n_test; k < j; ++k, kk += n_test) { *cov++ = -col_jj.dot(ktest.col(kk)); }  // cov(dh(x)/dx_j, dh(x)/dx_k)
                if (var != nullptr) { var[j] = m_three_over_scale_square_ - col_jj.squaredNorm(); }                   // var(dh(x)/dx_j)
            }
        }
    }

}  // namespace erl::gaussian_process

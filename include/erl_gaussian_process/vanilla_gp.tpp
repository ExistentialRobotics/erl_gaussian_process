#pragma once

#include "erl_common/serialization.hpp"
#include "erl_common/yaml.hpp"

namespace erl::gaussian_process {
    template<typename Dtype>
    YAML::Node
    VanillaGaussianProcess<Dtype>::Setting::YamlConvertImpl::encode(const Setting &setting) {
        YAML::Node node;
        node["kernel_type"] = setting.kernel_type;
        node["kernel_setting_type"] = setting.kernel_setting_type;
        node["kernel"] = setting.kernel->AsYamlNode();
        node["max_num_samples"] = setting.max_num_samples;
        return node;
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::Setting::YamlConvertImpl::decode(const YAML::Node &node, Setting &setting) {
        if (!node.IsMap()) { return false; }
        setting.kernel_type = node["kernel_type"].as<std::string>();
        setting.kernel_setting_type = node["kernel_setting_type"].as<std::string>();
        setting.kernel = common::YamlableBase::Create<typename Covariance::Setting>(setting.kernel_setting_type);
        if (!setting.kernel->FromYamlNode(node["kernel"])) { return false; }
        setting.max_num_samples = node["max_num_samples"].as<long>();
        return true;
    }

    template<typename Dtype>
    void
    VanillaGaussianProcess<Dtype>::TrainSet::Reset(long max_num_samples, long x_dim, long y_dim) {
        this->x_dim = x_dim;
        this->y_dim = y_dim;
        if (x.rows() < x_dim || x.cols() < max_num_samples) { x.resize(x_dim, max_num_samples); }
        if (y.rows() < max_num_samples || y.cols() < y_dim) { y.resize(max_num_samples, y_dim); }
        if (var.size() < max_num_samples) { var.resize(max_num_samples); }
        num_samples = 0;
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::TrainSet::operator==(const TrainSet &other) const {
        if (x_dim != other.x_dim) { return false; }
        if (y_dim != other.y_dim) { return false; }
        if (num_samples != other.num_samples) { return false; }
        if (num_samples == 0) { return true; }
        if (x.topLeftCorner(x_dim, num_samples) != other.x.topLeftCorner(x_dim, num_samples)) { return false; }
        if (y.topLeftCorner(num_samples, y_dim) != other.y.topLeftCorner(num_samples, y_dim)) { return false; }
        if (var.head(num_samples) != other.var.head(num_samples)) { return false; }
        return true;
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::TrainSet::operator!=(const TrainSet &other) const {
        return !(*this == other);
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::TrainSet::Write(std::ostream &s) const {
        static const std::vector<std::pair<const char *, std::function<bool(const TrainSet *, std::ostream &)>>> token_function_pairs = {
            {
                "x_dim",
                [](const TrainSet *train_set, std::ostream &stream) -> bool {
                    stream << train_set->x_dim;
                    return true;
                },
            },
            {
                "y_dim",
                [](const TrainSet *train_set, std::ostream &stream) -> bool {
                    stream << train_set->y_dim;
                    return true;
                },
            },
            {
                "num_samples",
                [](const TrainSet *train_set, std::ostream &stream) -> bool {
                    stream << train_set->num_samples;
                    return true;
                },
            },
            {
                "x",
                [](const TrainSet *train_set, std::ostream &stream) -> bool {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, train_set->x)) {
                        ERL_WARN("Failed to write x.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "y",
                [](const TrainSet *train_set, std::ostream &stream) -> bool {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, train_set->y)) {
                        ERL_WARN("Failed to write y.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "var",
                [](const TrainSet *train_set, std::ostream &stream) -> bool {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, train_set->var)) {
                        ERL_WARN("Failed to write var.");
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
    VanillaGaussianProcess<Dtype>::TrainSet::Read(std::istream &s) {
        static const std::vector<std::pair<const char *, std::function<bool(TrainSet *, std::istream &)>>> token_function_pairs = {
            {"x_dim",
             [](TrainSet *train_set, std::istream &stream) -> bool {
                 stream >> train_set->x_dim;
                 return true;
             }},
            {
                "y_dim",
                [](TrainSet *train_set, std::istream &stream) -> bool {
                    stream >> train_set->y_dim;
                    return true;
                },
            },
            {
                "num_samples",
                [](TrainSet *train_set, std::istream &stream) -> bool {
                    stream >> train_set->num_samples;
                    return true;
                },
            },
            {
                "x",
                [](TrainSet *train_set, std::istream &stream) -> bool {
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
                [](TrainSet *train_set, std::istream &stream) -> bool {
                    common::SkipLine(stream);
                    if (!common::LoadEigenMatrixFromBinaryStream(stream, train_set->y)) {
                        ERL_WARN("Failed to read y.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "var",
                [](TrainSet *train_set, std::istream &stream) -> bool {
                    common::SkipLine(stream);
                    if (!common::LoadEigenMatrixFromBinaryStream(stream, train_set->var)) {
                        ERL_WARN("Failed to read var.");
                        return false;
                    }
                    return true;
                },
            }};
        return common::ReadTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    VanillaGaussianProcess<Dtype>::VanillaGaussianProcess(std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)) {
        ERL_ASSERTM(m_setting_ != nullptr, "setting should not be nullptr.");
        ERL_ASSERTM(m_setting_->kernel != nullptr, "setting->kernel should not be nullptr.");
    }

    template<typename Dtype>
    VanillaGaussianProcess<Dtype>::VanillaGaussianProcess(const VanillaGaussianProcess &other)
        : m_setting_(other.m_setting_),
          m_trained_(other.m_trained_),
          m_trained_once_(other.m_trained_once_),
          m_k_train_updated_(other.m_k_train_updated_),
          m_k_train_rows_(other.m_k_train_rows_),
          m_k_train_cols_(other.m_k_train_cols_),
          m_reduced_rank_kernel_(other.m_reduced_rank_kernel_),
          m_mat_k_train_(other.m_mat_k_train_),
          m_mat_l_(other.m_mat_l_),
          m_mat_alpha_(other.m_mat_alpha_),
          m_train_set_(other.m_train_set_) {
        if (other.m_kernel_ != nullptr) {
            m_kernel_ = Covariance::CreateCovariance(m_setting_->kernel_type, m_setting_->kernel);
            if (m_reduced_rank_kernel_) {  // rank-reduced kernel is stateful, so we need to copy the kernel
                *std::reinterpret_pointer_cast<ReducedRankCovariance>(m_kernel_) = *std::reinterpret_pointer_cast<ReducedRankCovariance>(other.m_kernel_);
            }
        }
    }

    template<typename Dtype>
    VanillaGaussianProcess<Dtype> &
    VanillaGaussianProcess<Dtype>::operator=(const VanillaGaussianProcess &other) {
        if (this == &other) { return *this; }
        m_setting_ = other.m_setting_;
        m_trained_ = other.m_trained_;
        m_trained_once_ = other.m_trained_once_;
        m_k_train_updated_ = other.m_k_train_updated_;
        m_k_train_rows_ = other.m_k_train_rows_;
        m_k_train_cols_ = other.m_k_train_cols_;
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
    std::shared_ptr<const typename VanillaGaussianProcess<Dtype>::Setting>
    VanillaGaussianProcess<Dtype>::GetSetting() const {
        return m_setting_;
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::IsTrained() const {
        return m_trained_;
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::UsingReducedRankKernel() const {
        return m_reduced_rank_kernel_;
    }

    template<typename Dtype>
    typename VanillaGaussianProcess<Dtype>::VectorX
    VanillaGaussianProcess<Dtype>::GetKernelCoordOrigin() const {
        if (m_reduced_rank_kernel_) { return std::reinterpret_pointer_cast<ReducedRankCovariance>(m_kernel_)->GetCoordOrigin(); }
        ERL_DEBUG_ASSERT(m_train_set_.x_dim > 0, "train set should be initialized first.");
        return VectorX::Zero(m_train_set_.x_dim);
    }

    template<typename Dtype>
    void
    VanillaGaussianProcess<Dtype>::SetKernelCoordOrigin(const VectorX &coord_origin) const {
        if (m_reduced_rank_kernel_) { std::reinterpret_pointer_cast<ReducedRankCovariance>(m_kernel_)->SetCoordOrigin(coord_origin); }
    }

    template<typename Dtype>
    void
    VanillaGaussianProcess<Dtype>::Reset(const long max_num_samples, const long x_dim, const long y_dim) {
        ERL_DEBUG_ASSERT(max_num_samples > 0, "max_num_samples should be > 0.");
        ERL_DEBUG_ASSERT(x_dim > 0, "x_dim should be > 0.");
        ERL_DEBUG_ASSERT(y_dim > 0, "y_dim should be > 0.");
        ERL_DEBUG_ASSERT(m_setting_->kernel->x_dim == -1 || m_setting_->kernel->x_dim == x_dim, "x_dim should be {}.", m_setting_->kernel->x_dim);
        ERL_DEBUG_ASSERT(
            m_setting_->max_num_samples < 0 || max_num_samples <= m_setting_->max_num_samples,
            "max_num_samples should be <= {}.",
            m_setting_->max_num_samples);

        m_train_set_.Reset(max_num_samples, x_dim, y_dim);
        ERL_ASSERTM(AllocateMemory(max_num_samples, x_dim, y_dim), "Failed to allocate memory.");
        m_trained_ = false;
        m_k_train_updated_ = false;
        m_k_train_rows_ = 0;
        m_k_train_cols_ = 0;
    }

    template<typename Dtype>
    std::shared_ptr<typename VanillaGaussianProcess<Dtype>::Covariance>
    VanillaGaussianProcess<Dtype>::GetKernel() const {
        return m_kernel_;
    }

    template<typename Dtype>
    typename VanillaGaussianProcess<Dtype>::TrainSet &
    VanillaGaussianProcess<Dtype>::GetTrainSet() {
        return m_train_set_;
    }

    template<typename Dtype>
    const typename VanillaGaussianProcess<Dtype>::TrainSet &
    VanillaGaussianProcess<Dtype>::GetTrainSet() const {
        return m_train_set_;
    }

    template<typename Dtype>
    const typename VanillaGaussianProcess<Dtype>::MatrixX &
    VanillaGaussianProcess<Dtype>::GetKtrain() const {
        return m_mat_k_train_;
    }

    template<typename Dtype>
    const typename VanillaGaussianProcess<Dtype>::MatrixX &
    VanillaGaussianProcess<Dtype>::GetCholeskyDecomposition() const {
        return m_mat_l_;
    }

    template<typename Dtype>
    const typename VanillaGaussianProcess<Dtype>::MatrixX &
    VanillaGaussianProcess<Dtype>::GetAlpha() const {
        return m_mat_alpha_;
    }

    template<typename Dtype>
    std::size_t
    VanillaGaussianProcess<Dtype>::GetMemoryUsage() const {
        std::size_t memory_usage = sizeof(VanillaGaussianProcess);
        if (m_setting_ != nullptr) { memory_usage += sizeof(Setting); }
        if (m_kernel_ != nullptr) { memory_usage += m_kernel_->GetMemoryUsage(); }
        memory_usage += sizeof(TrainSet);
        memory_usage += (m_train_set_.x.size() + m_train_set_.y.size() + m_train_set_.var.size()) * sizeof(Dtype);
        memory_usage += (m_mat_k_train_.size() + m_mat_l_.size() + m_mat_alpha_.size()) * sizeof(Dtype);
        return memory_usage;
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::UpdateKtrain() {
        if (m_k_train_updated_) { return true; }
        auto &[x_dim, y_dim, num_samples, x, y, var] = m_train_set_;  // unpack
        if (num_samples <= 0) {
            ERL_WARN("num_samples = {}, it should be > 0.", num_samples);
            return false;
        }
        m_mat_alpha_.topLeftCorner(num_samples, y_dim) = y.topRows(num_samples);
        std::tie(m_k_train_rows_, m_k_train_cols_) = m_kernel_->ComputeKtrain(x, var, num_samples, m_mat_k_train_, m_mat_alpha_);
        m_k_train_updated_ = true;
        return true;
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::Train() {

        if (m_trained_) {
            ERL_WARN("The model has been trained. Please reset the model before training.");
            return false;
        }
        m_trained_ = m_trained_once_;
        if (!UpdateKtrain()) { return false; }

        const auto mat_ktrain = m_mat_k_train_.topLeftCorner(m_k_train_rows_, m_k_train_cols_);
        auto &&mat_l = m_mat_l_.topLeftCorner(m_k_train_rows_, m_k_train_cols_);
        auto alpha = m_mat_alpha_.topRows(m_k_train_cols_);
        mat_l = mat_ktrain.llt().matrixL();  // A = ktrain(mat_x_train, mat_x_train) + sigma * I = mat_l @ mat_l.T
        mat_l.template triangularView<Eigen::Lower>().solveInPlace(alpha);
        mat_l.transpose().template triangularView<Eigen::Upper>().solveInPlace(alpha);  // A.m_inv_() @ vec_alpha
        m_trained_once_ = true;
        m_trained_ = true;
        return true;
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::Test(
        const Eigen::Ref<const MatrixX> &mat_x_test,
        const std::vector<long> &y_indices,
        Eigen::Ref<MatrixX> mat_f_out,
        Eigen::Ref<VectorX> vec_var_out) const {

        if (!m_trained_) {
            ERL_WARN("The model has not been trained.");
            return false;
        }

        const long num_test = mat_x_test.cols();
        if (num_test == 0) { return false; }

        const auto &[x_dim, y_dim, num_samples, x_train, y_train, var_train] = m_train_set_;  // unpack

#ifndef NDEBUG
        const auto num_y_dims = static_cast<long>(y_indices.size());
#endif
        ERL_DEBUG_ASSERT(mat_x_test.rows() == x_dim, "mat_x_test.rows() = {}, it should be {}.", mat_x_test.rows(), x_dim);
        ERL_DEBUG_ASSERT(mat_f_out.rows() >= num_y_dims, "mat_f_out.rows() = {}, it should be >= {}.", mat_f_out.rows(), num_y_dims);
        ERL_DEBUG_ASSERT(mat_f_out.cols() >= num_test, "mat_f_out.cols() = {}, it should be >= {}.", mat_f_out.cols(), num_test);

        const auto [ktest_rows, ktest_cols] = m_kernel_->GetMinimumKtestSize(num_samples, 0, x_dim, num_test, false);
        MatrixX ktest(ktest_rows, ktest_cols);  // (num_features, num_test). Usually, num_features = num_train_samples.
        const auto [output_rows, output_cols] = m_kernel_->ComputeKtest(x_train, num_samples, mat_x_test, num_test, ktest);

        ERL_DEBUG_ASSERT(
            output_rows == ktest_rows && output_cols == ktest_cols,
            "output_size = ({}, {}), it should be ({}, {}).",
            output_rows,
            output_cols,
            ktest_rows,
            ktest_cols);

        ComputeValuePrediction(ktest, y_indices, mat_f_out);
        ComputeCovPrediction(ktest, vec_var_out);
        return true;
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::operator==(const VanillaGaussianProcess &other) const {
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr && (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) { return false; }
        if (m_trained_ != other.m_trained_) { return false; }
        if (m_trained_once_ != other.m_trained_once_) { return false; }
        if (m_k_train_updated_ != other.m_k_train_updated_) { return false; }
        if (m_k_train_rows_ != other.m_k_train_rows_) { return false; }
        if (m_k_train_cols_ != other.m_k_train_cols_) { return false; }
        if (m_reduced_rank_kernel_ != other.m_reduced_rank_kernel_) { return false; }
        if (m_train_set_ != other.m_train_set_) { return false; }
        if (m_mat_k_train_.rows() != other.m_mat_k_train_.rows() || m_mat_k_train_.cols() != other.m_mat_k_train_.cols() ||
            m_mat_k_train_.topLeftCorner(m_k_train_rows_, m_k_train_cols_) != other.m_mat_k_train_.topLeftCorner(m_k_train_rows_, m_k_train_cols_)) {
            return false;
        }
        if (m_mat_l_.rows() != other.m_mat_l_.rows() || m_mat_l_.cols() != other.m_mat_l_.cols() ||
            m_mat_l_.topLeftCorner(m_k_train_rows_, m_k_train_cols_) != other.m_mat_l_.topLeftCorner(m_k_train_rows_, m_k_train_cols_)) {
            return false;
        }
        if (m_mat_alpha_.size() != other.m_mat_alpha_.size() || m_mat_alpha_.topRows(m_k_train_cols_) != other.m_mat_alpha_.topRows(m_k_train_cols_)) {
            return false;
        }
        return true;
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::operator!=(const VanillaGaussianProcess &other) const {
        return !(*this == other);
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::Write(std::ostream &s) const {
        static const std::vector<std::pair<const char *, std::function<bool(const VanillaGaussianProcess *, std::ostream &)>>> token_function_pairs = {
            {
                "setting",
                [](const VanillaGaussianProcess *gp, std::ostream &stream) -> bool {
                    if (!gp->m_setting_->Write(stream)) {
                        ERL_WARN("Failed to write setting.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "trained",
                [](const VanillaGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream << gp->m_trained_;
                    return true;
                },
            },
            {
                "trained_once",
                [](const VanillaGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream << gp->m_trained_once_;
                    return true;
                },
            },
            {
                "ktrain_updated",
                [](const VanillaGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream << gp->m_k_train_updated_;
                    return true;
                },
            },
            {
                "ktrain_rows",
                [](const VanillaGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream << gp->m_k_train_rows_;
                    return true;
                },
            },
            {
                "ktrain_cols",
                [](const VanillaGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream << gp->m_k_train_cols_;
                    return true;
                },
            },
            {
                "kernel",
                [](const VanillaGaussianProcess *gp, std::ostream &stream) -> bool {
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
                [](const VanillaGaussianProcess *gp, std::ostream &stream) -> bool {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, gp->m_mat_k_train_)) {
                        ERL_WARN("Failed to write mat_k_train.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "mat_l",
                [](const VanillaGaussianProcess *gp, std::ostream &stream) -> bool {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, gp->m_mat_l_)) {
                        ERL_WARN("Failed to write mat_l.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "mat_alpha",
                [](const VanillaGaussianProcess *gp, std::ostream &stream) -> bool {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, gp->m_mat_alpha_)) {
                        ERL_WARN("Failed to write mat_alpha.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "train_set",
                [](const VanillaGaussianProcess *gp, std::ostream &stream) -> bool {
                    if (!gp->m_train_set_.Write(stream)) {
                        ERL_WARN("Failed to write train set.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "end_of_VanillaGaussianProcess",
                [](const VanillaGaussianProcess *, std::ostream &) -> bool { return true; },
            },
        };
        return common::WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::Read(std::istream &s) {
        static const std::vector<std::pair<const char *, std::function<bool(VanillaGaussianProcess *, std::istream &)>>> token_function_pairs = {
            {
                "setting",
                [](VanillaGaussianProcess *gp, std::istream &stream) -> bool {
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
                [](VanillaGaussianProcess *gp, std::istream &stream) -> bool {
                    stream >> gp->m_trained_;
                    return true;
                },
            },
            {
                "trained_once",
                [](VanillaGaussianProcess *gp, std::istream &stream) -> bool {
                    stream >> gp->m_trained_once_;
                    return true;
                },
            },
            {
                "ktrain_updated",
                [](VanillaGaussianProcess *gp, std::istream &stream) -> bool {
                    stream >> gp->m_k_train_updated_;
                    return true;
                },
            },
            {
                "ktrain_rows",
                [](VanillaGaussianProcess *gp, std::istream &stream) -> bool {
                    stream >> gp->m_k_train_rows_;
                    return true;
                },
            },
            {
                "ktrain_cols",
                [](VanillaGaussianProcess *gp, std::istream &stream) -> bool {
                    stream >> gp->m_k_train_cols_;
                    return true;
                },
            },
            {
                "kernel",
                [](VanillaGaussianProcess *gp, std::istream &stream) -> bool {
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
                [](VanillaGaussianProcess *gp, std::istream &stream) -> bool {
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
                [](VanillaGaussianProcess *gp, std::istream &stream) -> bool {
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
                [](VanillaGaussianProcess *gp, std::istream &stream) -> bool {
                    common::SkipLine(stream);
                    if (!common::LoadEigenMatrixFromBinaryStream(stream, gp->m_mat_alpha_)) {
                        ERL_WARN("Failed to read alpha.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "train_set",
                [](VanillaGaussianProcess *gp, std::istream &stream) -> bool {
                    if (!gp->m_train_set_.Read(stream)) {
                        ERL_WARN("Failed to read train set.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "end_of_VanillaGaussianProcess",
                [](VanillaGaussianProcess *, std::istream &stream) -> bool {
                    common::SkipLine(stream);
                    return true;
                },
            },
        };
        return common::ReadTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::AllocateMemory(const long max_num_samples, const long x_dim, const long y_dim) {
        if (max_num_samples <= 0 || x_dim <= 0 || y_dim <= 0) { return false; }  // invalid input
        if (m_setting_->max_num_samples > 0 && max_num_samples > m_setting_->max_num_samples) { return false; }
        if (m_setting_->kernel->x_dim > 0 && x_dim != m_setting_->kernel->x_dim) { return false; }
        InitKernel();
        const auto [rows, cols] = m_kernel_->GetMinimumKtrainSize(max_num_samples, 0, 0);
        if (m_mat_k_train_.rows() < rows || m_mat_k_train_.cols() < cols) { m_mat_k_train_.resize(rows, cols); }
        if (m_mat_l_.rows() < rows || m_mat_l_.cols() < cols) { m_mat_l_.resize(rows, cols); }
        if (const long alpha_rows = std::max(max_num_samples, cols);  //
            m_mat_alpha_.rows() < alpha_rows || m_mat_alpha_.cols() < y_dim) {
            m_mat_alpha_.resize(alpha_rows, y_dim);
        }
        return true;
    }

    template<typename Dtype>
    void
    VanillaGaussianProcess<Dtype>::InitKernel() {
        if (m_kernel_ == nullptr) {
            m_kernel_ = Covariance::CreateCovariance(m_setting_->kernel_type, m_setting_->kernel);
            ERL_DEBUG_ASSERT(m_kernel_ != nullptr, "failed to create kernel of type {}.", m_setting_->kernel_type);
            const auto rank_reduced_kernel = std::dynamic_pointer_cast<ReducedRankCovariance>(m_kernel_);
            m_reduced_rank_kernel_ = rank_reduced_kernel != nullptr;
            if (m_reduced_rank_kernel_) { rank_reduced_kernel->BuildSpectralDensities(); }
        }
    }

    template<typename Dtype>
    void
    VanillaGaussianProcess<Dtype>::ComputeValuePrediction(const MatrixX &ktest, const std::vector<long> &y_indices, Eigen::Ref<MatrixX> mat_f_out) const {
        // xt is one column of mat_x_test.
        // expectation: f = ktest(xt, X) @ (ktrain(X, X) + sigma * I).inv() @ y
        for (long i = 0; i < ktest.cols(); ++i) {
            Dtype *f = mat_f_out.col(i).data();
            const auto ktest_col = ktest.col(i);
            for (const long &y_idx: y_indices) {
                *f = ktest_col.dot(m_mat_alpha_.col(y_idx).head(m_k_train_cols_));  // h_d(x_i)
                ++f;
            }
        }
    }

    template<typename Dtype>
    void
    VanillaGaussianProcess<Dtype>::ComputeCovPrediction(MatrixX &ktest, Eigen::Ref<VectorX> vec_var_out) const {
        if (vec_var_out.size() == 0) { return; }  // only compute mean

        const long ktest_rows = ktest.rows();
        const long ktest_cols = ktest.cols();

        // variance of vec_f_out = ktest(xt, xt) - ktest(xt, X) @ (ktest(X, X) + sigma * I).m_inv_() @ ktest(X, xt)
        //                       = ktest(xt, xt) - ktest(xt, X) @ (m_l_ @ m_l_.T).m_inv_() @ ktest(X, xt)
        ERL_DEBUG_ASSERT(vec_var_out.size() >= ktest_cols, "vec_var_out size = {}, it should be >= {}.", vec_var_out.size(), ktest_cols);
        m_mat_l_.topLeftCorner(ktest_rows, ktest_rows).template triangularView<Eigen::Lower>().solveInPlace(ktest);
        if (m_reduced_rank_kernel_) {
            for (long i = 0; i < ktest_cols; ++i) { vec_var_out[i] = ktest.col(i).squaredNorm(); }
        } else {
            for (long i = 0; i < ktest_cols; ++i) { vec_var_out[i] = 1.0f - ktest.col(i).squaredNorm(); }
        }
    }

}  // namespace erl::gaussian_process

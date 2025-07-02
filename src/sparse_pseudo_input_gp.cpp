#include "erl_gaussian_process/sparse_pseudo_input_gp.hpp"

#include "erl_common/serialization.hpp"

#include <open3d/core/Dtype.h>

namespace erl::gaussian_process {
    template<typename Dtype>
    YAML::Node
    SparsePseudoInputGaussianProcess<Dtype>::Setting::YamlConvertImpl::encode(
        const Setting &setting) {
        YAML::Node node;
        ERL_YAML_SAVE_ATTR(node, setting, kernel_type);
        ERL_YAML_SAVE_ATTR(node, setting, kernel_setting_type);
        ERL_YAML_SAVE_ATTR(node, setting, kernel);
        ERL_YAML_SAVE_ATTR(node, setting, max_num_samples);
        ERL_YAML_SAVE_ATTR(node, setting, sparse_zero_threshold);
        ERL_YAML_SAVE_ATTR(node, setting, use_sparse);
        ERL_YAML_SAVE_ATTR(node, setting, diagonal_qm);
        return node;
    }

    template<typename Dtype>
    bool
    SparsePseudoInputGaussianProcess<Dtype>::Setting::YamlConvertImpl::decode(
        const YAML::Node &node,
        Setting &setting) {
        if (!node.IsMap()) { return false; }
        ERL_YAML_LOAD_ATTR(node, setting, kernel_type);
        ERL_YAML_LOAD_ATTR(node, setting, kernel_setting_type);
        using namespace common;
        using CovarianceSetting = typename Covariance::Setting;
        setting.kernel = YamlableBase::Create<CovarianceSetting>(setting.kernel_setting_type);
        if (!ERL_YAML_LOAD_ATTR(node, setting, kernel)) { return false; }
        ERL_YAML_LOAD_ATTR(node, setting, max_num_samples);
        ERL_YAML_LOAD_ATTR(node, setting, sparse_zero_threshold);
        ERL_YAML_LOAD_ATTR(node, setting, use_sparse);
        ERL_YAML_LOAD_ATTR(node, setting, diagonal_qm);
        return true;
    }

    template<typename Dtype>
    SparsePseudoInputGaussianProcess<Dtype>::TestResult::TestResult(
        const SparsePseudoInputGaussianProcess *gp,
        const Eigen::Ref<const MatrixX> &mat_x_test,
        const bool will_predict_gradient)
        : m_gp_(NotNull(gp, true, "gp = nullptr.")),
          m_num_test_(mat_x_test.cols()),
          m_support_gradient_(will_predict_gradient),
          m_use_sparse_(m_gp_->m_setting_->use_sparse),
          m_x_dim_(m_gp_->m_train_set_.x_dim),
          m_y_dim_(m_gp_->m_train_set_.y_dim) {

        auto &pseudo_points = m_gp_->m_pseudo_points_;
        const long n = m_num_test_ * (m_support_gradient_ ? m_x_dim_ + 1 : 1);
        if (m_use_sparse_) {
            m_sparse_mat_k_test_ = SparseMatrix(pseudo_points.cols(), n);
            if (m_support_gradient_) {
                Eigen::VectorXl grad_flags = Eigen::VectorXl::Zero(pseudo_points.cols());
                (void) m_gp_->m_kernel_->ComputeKtestWithGradientSparse(
                    pseudo_points,
                    pseudo_points.cols(),
                    grad_flags,
                    mat_x_test,
                    m_num_test_,
                    true, /*predict_gradient*/
                    m_gp_->m_setting_->sparse_zero_threshold,
                    m_sparse_mat_k_test_);
            } else {
                (void) m_gp_->m_kernel_->ComputeKtestSparse(
                    pseudo_points,
                    pseudo_points.cols(),
                    mat_x_test,
                    m_num_test_,
                    m_gp_->m_setting_->sparse_zero_threshold,
                    m_sparse_mat_k_test_);
            }
        } else {
            m_mat_k_test_.resize(pseudo_points.cols(), n);
            if (m_support_gradient_) {
                Eigen::VectorXl grad_flags = Eigen::VectorXl::Zero(pseudo_points.cols());
                (void) m_gp_->m_kernel_->ComputeKtestWithGradient(
                    pseudo_points,
                    pseudo_points.cols(),
                    grad_flags,
                    mat_x_test,
                    m_num_test_,
                    true, /*predict_gradient*/
                    m_mat_k_test_);
            } else {
                (void) m_gp_->m_kernel_->ComputeKtest(
                    pseudo_points,
                    pseudo_points.cols(),
                    mat_x_test,
                    m_num_test_,
                    m_mat_k_test_);
            }
        }

        if (m_gp_->m_setting_->diagonal_qm) {
            m_mat_alpha_ = m_gp_->m_mat_alpha_.array().colwise() / m_gp_->m_mat_qm_.col(0).array();
        } else {
            m_mat_alpha_ = m_gp_->m_mat_l_qm_
                               .template triangularView<Eigen::Lower>()  //
                               .solve(m_gp_->m_mat_alpha_);
            m_gp_->m_mat_l_qm_.transpose()
                .template triangularView<Eigen::Upper>()  //
                .solveInPlace(m_mat_alpha_);
        }

        // VectorX qm_diagonal = m_gp_->m_mat_qm_.diagonal();
        // m_mat_alpha_ = m_gp_->m_mat_alpha_.array().colwise() / qm_diagonal.array();
    }

    template<typename Dtype>
    long
    SparsePseudoInputGaussianProcess<Dtype>::TestResult::GetNumTest() const {
        return m_num_test_;
    }

    template<typename Dtype>
    long
    SparsePseudoInputGaussianProcess<Dtype>::TestResult::GetDimX() const {
        return m_x_dim_;
    }

    template<typename Dtype>
    long
    SparsePseudoInputGaussianProcess<Dtype>::TestResult::GetDimY() const {
        return m_y_dim_;
    }

    template<typename Dtype>
    void
    SparsePseudoInputGaussianProcess<Dtype>::TestResult::GetMean(
        long y_index,
        Eigen::Ref<VectorX> vec_f_out,
        const bool parallel) const {
        (void) parallel;
        ERL_DEBUG_ASSERT(
            y_index >= 0 && y_index < m_y_dim_,
            "y_index = {}, it should be in [0, {}).",
            y_index,
            m_y_dim_);
        ERL_DEBUG_ASSERT(
            vec_f_out.size() == m_num_test_,
            "vec_f_out.size() = {}, it should be {}.",
            vec_f_out.size(),
            m_num_test_);
        const auto alpha = m_mat_alpha_.col(y_index);
        Dtype *f = vec_f_out.data();
        if (m_use_sparse_) {
#pragma omp parallel for if (parallel) default(none) shared(alpha, f)
            for (long index = 0; index < m_num_test_; ++index) {
                f[index] = m_sparse_mat_k_test_.col(index).dot(alpha);
            }
        } else {
#pragma omp parallel for if (parallel) default(none) shared(alpha, f)
            for (long index = 0; index < m_num_test_; ++index) {
                f[index] = m_mat_k_test_.col(index).dot(alpha);
            }
        }
    }

    template<typename Dtype>
    void
    SparsePseudoInputGaussianProcess<Dtype>::TestResult::GetMean(long index, long y_index, Dtype &f)
        const {
        ERL_DEBUG_ASSERT(
            index >= 0 && index < m_num_test_,
            "index = {}, it should be >= 0 and < {}.",
            index,
            m_num_test_);
        ERL_DEBUG_ASSERT(
            y_index >= 0 && y_index < m_y_dim_,
            "y_index = {}, it should be >= 0 and < {}.",
            y_index,
            m_y_dim_);
        const auto alpha = m_mat_alpha_.col(y_index);
        if (m_use_sparse_) {
            f = m_sparse_mat_k_test_.col(index).dot(alpha);
        } else {
            f = m_mat_k_test_.col(index).dot(alpha);
        }
    }

    template<typename Dtype>
    Eigen::VectorXb
    SparsePseudoInputGaussianProcess<Dtype>::TestResult::GetGradient(
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
        const auto alpha = m_gp_->m_mat_alpha_.col(y_index);
        Eigen::VectorXb valid_gradients(m_num_test_);
        if (m_use_sparse_) {
#pragma omp parallel for if (parallel) default(none) shared(alpha, mat_grad_out, valid_gradients)
            for (long index = 0; index < m_num_test_; ++index) {
                Dtype *grad = mat_grad_out.col(index).data();
                for (long j = 0, jj = index + m_num_test_; j < m_x_dim_; ++j, jj += m_num_test_) {
                    *grad = m_sparse_mat_k_test_.col(jj).dot(alpha);
                    if (!std::isfinite(*grad)) {
                        valid_gradients[index] = false;
                        break;
                    }
                    ++grad;
                }
            }
        } else {
#pragma omp parallel for if (parallel) default(none) shared(alpha, mat_grad_out, valid_gradients)
            for (long index = 0; index < m_num_test_; ++index) {
                Dtype *grad = mat_grad_out.col(index).data();
                for (long j = 0, jj = index + m_num_test_; j < m_x_dim_; ++j, jj += m_num_test_) {
                    *grad = m_mat_k_test_.col(jj).dot(alpha);
                    if (!std::isfinite(*grad)) {
                        valid_gradients[index] = false;
                        break;
                    }
                    ++grad;
                }
            }
        }
        return valid_gradients;
    }

    template<typename Dtype>
    bool
    SparsePseudoInputGaussianProcess<Dtype>::TestResult::GetGradient(
        const long index,
        const long y_index,
        Dtype *grad) const {
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
        const auto alpha = m_mat_alpha_.col(y_index);
        if (m_use_sparse_) {
            for (long j = 0, jj = index + m_num_test_; j < m_x_dim_; ++j, jj += m_num_test_) {
                *grad = m_sparse_mat_k_test_.col(jj).dot(alpha);
                if (!std::isfinite(*grad)) { return false; }  // gradient is not valid
                ++grad;
            }
        } else {
            for (long j = 0, jj = index + m_num_test_; j < m_x_dim_; ++j, jj += m_num_test_) {
                *grad = m_mat_k_test_.col(jj).dot(alpha);
                if (!std::isfinite(*grad)) { return false; }  // gradient is not valid
                ++grad;
            }
        }
        return true;
    }

    template<typename Dtype>
    void
    SparsePseudoInputGaussianProcess<Dtype>::TestResult::GetVariance(
        Eigen::Ref<VectorX> vec_var_out,
        const bool parallel) const {
        (void) parallel;
        const_cast<TestResult *>(this)->PrepareForVariance();
        Dtype *var = vec_var_out.data();
#pragma omp parallel for if (parallel) default(none) shared(var)
        for (long index = 0; index < m_num_test_; ++index) {
            var[index] =
                1.0f - m_mat_beta_.col(index).squaredNorm() + m_mat_gamma_.col(index).squaredNorm();
        }
    }

    template<typename Dtype>
    void
    SparsePseudoInputGaussianProcess<Dtype>::TestResult::GetVariance(long index, Dtype &var) const {
        const_cast<TestResult *>(this)->PrepareForVariance();
        var = 1.0f - m_mat_beta_.col(index).squaredNorm() + m_mat_gamma_.col(index).squaredNorm();
    }

    template<typename Dtype>
    void
    SparsePseudoInputGaussianProcess<Dtype>::TestResult::PrepareForVariance() {
        if (m_mat_beta_.size() > 0) { return; }
        m_mat_beta_ =
            m_gp_->m_mat_l_km_.template triangularView<Eigen::Lower>().solve(m_mat_k_test_);
        m_mat_gamma_ =
            m_gp_->m_mat_l_qm_.template triangularView<Eigen::Lower>().solve(m_mat_k_test_);
    }

    template<typename Dtype>
    SparsePseudoInputGaussianProcess<Dtype>::SparsePseudoInputGaussianProcess(
        std::shared_ptr<Setting> setting,
        MatrixX pseudo_points)
        : m_setting_(std::move(setting)),
          m_pseudo_points_(std::move(pseudo_points)) {
        ERL_ASSERTM(m_setting_ != nullptr, "setting is null");
        ERL_ASSERTM(m_setting_->kernel != nullptr, "setting->kernel is null");
        ERL_ASSERTM(m_pseudo_points_.cols() > 0, "pseudo_points must have at least one column");
        ERL_ASSERTM(
            m_setting_->kernel->x_dim == -1 || m_setting_->kernel->x_dim == m_pseudo_points_.rows(),
            "setting->kernel->x_dim {} and pseudo_points.rows() {} should match.",
            m_pseudo_points_.rows());

        // init kernel
        m_kernel_ = Covariance::CreateCovariance(m_setting_->kernel_type, m_setting_->kernel);
        ERL_ASSERTM(
            m_kernel_ != nullptr,
            "failed to create kernel of type {}.",
            m_setting_->kernel_type);
        const auto rank_reduced_kernel =
            std::dynamic_pointer_cast<ReducedRankCovariance>(m_kernel_);
        m_reduced_rank_kernel_ = rank_reduced_kernel != nullptr;
        if (m_reduced_rank_kernel_) { rank_reduced_kernel->BuildSpectralDensities(); }

        // initialize matrices
        const long m = m_pseudo_points_.cols();
        m_mat_km_ = MatrixX::Zero(m, m);
        (void) m_kernel_->ComputeKtest(m_pseudo_points_, m, m_pseudo_points_, m, m_mat_km_);
        m_mat_l_km_ = m_mat_km_.llt().matrixL();
        if (m_setting_->use_sparse) {
            m_sparse_mat_km_ = m_mat_km_.sparseView(m_setting_->sparse_zero_threshold);
            m_sparse_mat_km_.makeCompressed();
        }
        if (m_setting_->diagonal_qm) {
            m_mat_qm_ = MatrixX::Ones(m, 1);
        } else {
            m_mat_qm_ = m_mat_km_;
        }

        ERL_DEBUG_ASSERT(!m_mat_km_.hasNaN(), "m_mat_km_ has NaN values.");
        ERL_DEBUG_ASSERT(
            !m_mat_l_km_.hasNaN(),
            "m_mat_l_km_ has NaN values after Cholesky decomposition.");
    }

    template<typename Dtype>
    std::shared_ptr<const typename SparsePseudoInputGaussianProcess<Dtype>::Setting>
    SparsePseudoInputGaussianProcess<Dtype>::GetSetting() const {
        return m_setting_;
    }

    template<typename Dtype>
    bool
    SparsePseudoInputGaussianProcess<Dtype>::IsTrained() const {
        return m_trained_;
    }

    template<typename Dtype>
    bool
    SparsePseudoInputGaussianProcess<Dtype>::UsingReducedRankKernel() const {
        return m_reduced_rank_kernel_;
    }

    template<typename Dtype>
    typename SparsePseudoInputGaussianProcess<Dtype>::VectorX
    SparsePseudoInputGaussianProcess<Dtype>::GetKernelCoordOrigin() const {
        if (m_reduced_rank_kernel_) {
            return std::reinterpret_pointer_cast<ReducedRankCovariance>(m_kernel_)
                ->GetCoordOrigin();
        }
        ERL_DEBUG_ASSERT(m_train_set_.x_dim > 0, "train set should be initialized first.");
        return VectorX::Zero(m_train_set_.x_dim);
    }

    template<typename Dtype>
    void
    SparsePseudoInputGaussianProcess<Dtype>::SetKernelCoordOrigin(
        const VectorX &coord_origin) const {
        if (m_reduced_rank_kernel_) {
            std::reinterpret_pointer_cast<ReducedRankCovariance>(m_kernel_)->SetCoordOrigin(
                coord_origin);
        }
    }

    template<typename Dtype>
    std::shared_ptr<typename SparsePseudoInputGaussianProcess<Dtype>::Covariance>
    SparsePseudoInputGaussianProcess<Dtype>::GetKernel() const {
        return m_kernel_;
    }

    template<typename Dtype>
    void
    SparsePseudoInputGaussianProcess<Dtype>::Reset(long max_num_samples, long x_dim, long y_dim) {
        ERL_DEBUG_ASSERT(max_num_samples > 0, "max_num_samples should be > 0.");
        ERL_DEBUG_ASSERT(
            x_dim == m_pseudo_points_.rows(),
            "x_dim should be {}.",
            m_pseudo_points_.rows());
        ERL_DEBUG_ASSERT(y_dim > 0, "y_dim should be > 0.");
        ERL_ASSERTM(
            m_setting_->max_num_samples < 0 || max_num_samples <= m_setting_->max_num_samples,
            "max_num_samples should be <= {}.",
            m_setting_->max_num_samples);

        m_mat_l_qm_updated_ = false;
        m_trained_ = false;
        const long m = m_pseudo_points_.cols();
        m_train_set_.Reset(max_num_samples, x_dim, y_dim);
        if (m_mat_alpha_.size() == 0) { m_mat_alpha_.setConstant(m, y_dim, 0); }
        ERL_ASSERTM(
            m_mat_alpha_.cols() == y_dim,
            "m_mat_alpha_ is initialized for {} dimensions, but y_dim is {}.",
            m_mat_alpha_.cols(),
            y_dim);
    }

    template<typename Dtype>
    const typename SparsePseudoInputGaussianProcess<Dtype>::MatrixX &
    SparsePseudoInputGaussianProcess<Dtype>::GetPseudoPoints() const {
        return m_pseudo_points_;
    }

    template<typename Dtype>
    const typename SparsePseudoInputGaussianProcess<Dtype>::MatrixX &
    SparsePseudoInputGaussianProcess<Dtype>::GetMatKm() const {
        return m_mat_km_;
    }

    template<typename Dtype>
    const typename SparsePseudoInputGaussianProcess<Dtype>::SparseMatrix &
    SparsePseudoInputGaussianProcess<Dtype>::GetSparseMatKm() const {
        return m_sparse_mat_km_;
    }

    template<typename Dtype>
    const typename SparsePseudoInputGaussianProcess<Dtype>::MatrixX &
    SparsePseudoInputGaussianProcess<Dtype>::GetMatLKm() const {
        return m_mat_l_km_;
    }

    template<typename Dtype>
    const typename SparsePseudoInputGaussianProcess<Dtype>::MatrixX &
    SparsePseudoInputGaussianProcess<Dtype>::GetMatQm() const {
        return m_mat_qm_;
    }

    template<typename Dtype>
    const typename SparsePseudoInputGaussianProcess<Dtype>::MatrixX &
    SparsePseudoInputGaussianProcess<Dtype>::GetMatLQm() const {
        return m_mat_l_qm_;
    }

    template<typename Dtype>
    const typename SparsePseudoInputGaussianProcess<Dtype>::MatrixX &
    SparsePseudoInputGaussianProcess<Dtype>::GetMatAlpha() const {
        return m_mat_alpha_;
    }

    template<typename Dtype>
    typename SparsePseudoInputGaussianProcess<Dtype>::TrainSet &
    SparsePseudoInputGaussianProcess<Dtype>::GetTrainSet() {
        return m_train_set_;
    }

    template<typename Dtype>
    const typename SparsePseudoInputGaussianProcess<Dtype>::TrainSet &
    SparsePseudoInputGaussianProcess<Dtype>::GetTrainSet() const {
        return m_train_set_;
    }

    template<typename Dtype>
    bool
    SparsePseudoInputGaussianProcess<Dtype>::Update(const bool parallel) {
        if (m_trained_) { return true; }
        if (m_setting_->use_sparse) { return UpdateSparse(parallel); }
        return UpdateDense(parallel);
    }

    template<typename Dtype>
    std::shared_ptr<typename SparsePseudoInputGaussianProcess<Dtype>::TestResult>
    SparsePseudoInputGaussianProcess<Dtype>::Test(
        const Eigen::Ref<const MatrixX> &mat_x_test,
        const bool predict_gradient) const {
        const_cast<SparsePseudoInputGaussianProcess *>(this)->PrepareLqm();
        return std::make_shared<TestResult>(this, mat_x_test, predict_gradient);
    }

    template<typename Dtype>
    bool
    SparsePseudoInputGaussianProcess<Dtype>::operator==(
        const SparsePseudoInputGaussianProcess &other) const {
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr &&
            (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) {
            return false;
        }
        if (m_trained_ != other.m_trained_) { return false; }
        if (m_trained_once_ != other.m_trained_once_) { return false; }
        if (m_mat_l_qm_updated_ != other.m_mat_l_qm_updated_) { return false; }
        if (m_kernel_ == nullptr && other.m_kernel_ != nullptr) { return false; }
        if (m_kernel_ != nullptr &&
            (other.m_kernel_ == nullptr || *m_kernel_ != *other.m_kernel_)) {
            return false;
        }
        if (m_reduced_rank_kernel_ != other.m_reduced_rank_kernel_) { return false; }
        if (m_pseudo_points_ != other.m_pseudo_points_) { return false; }
        using namespace common;
        if (!SafeEigenMatrixEqual(m_mat_km_, other.m_mat_km_)) { return false; }
        if (!SafeSparseEigenMatrixEqual(m_sparse_mat_km_, other.m_sparse_mat_km_)) { return false; }
        if (!SafeEigenMatrixEqual(m_mat_l_km_, other.m_mat_l_km_)) { return false; }
        if (!SafeEigenMatrixEqual(m_mat_qm_, other.m_mat_qm_)) { return false; }
        if (!SafeEigenMatrixEqual(m_mat_l_qm_, other.m_mat_l_qm_)) { return false; }
        if (!SafeEigenMatrixEqual(m_mat_alpha_, other.m_mat_alpha_)) { return false; }
        if (m_train_set_ != other.m_train_set_) { return false; }
        return true;
    }

    template<typename Dtype>
    bool
    SparsePseudoInputGaussianProcess<Dtype>::operator!=(
        const SparsePseudoInputGaussianProcess &other) const {
        return !(*this == other);
    }

    template<typename Dtype>
    bool
    SparsePseudoInputGaussianProcess<Dtype>::Write(std::ostream &s) const {
        using namespace common;
        using Self = SparsePseudoInputGaussianProcess;
        static const TokenWriteFunctionPairs<Self> token_function_pairs = {
            {
                "setting",
                [](const Self *gp, std::ostream &stream) -> bool {
                    if (!gp->m_setting_->Write(stream)) {
                        ERL_WARN("Failed to write setting.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "trained",
                [](const Self *gp, std::ostream &stream) -> bool {
                    stream << gp->m_trained_;
                    return true;
                },
            },
            {
                "trained_once",
                [](const Self *gp, std::ostream &stream) -> bool {
                    stream << gp->m_trained_once_;
                    return true;
                },
            },
            {
                "mat_l_qm_updated",
                [](const Self *gp, std::ostream &stream) -> bool {
                    stream << gp->m_mat_l_qm_updated_;
                    return true;
                },
            },
            {
                "kernel",
                [](const Self *gp, std::ostream &stream) -> bool {
                    stream << (gp->m_kernel_ != nullptr) << '\n';
                    if (gp->m_kernel_ != nullptr && !gp->m_kernel_->Write(stream)) {
                        ERL_WARN("Failed to write kernel.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "pseudo_points",
                [](const Self *gp, std::ostream &stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, gp->m_pseudo_points_) &&
                           stream.good();
                },
            },
            {
                "mat_km",
                [](const Self *gp, std::ostream &stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, gp->m_mat_km_) && stream.good();
                },
            },
            {
                "sparse_mat_km",
                [](const Self *gp, std::ostream &stream) -> bool {
                    return SaveSparseEigenMatrixToBinaryStream(stream, gp->m_sparse_mat_km_) &&
                           stream.good();
                },
            },
            {
                "mat_l_km",
                [](const Self *gp, std::ostream &stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, gp->m_mat_l_km_) && stream.good();
                },
            },
            {
                "mat_qm",
                [](const Self *gp, std::ostream &stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, gp->m_mat_qm_) && stream.good();
                },
            },
            {
                "mat_l_qm",
                [](const Self *gp, std::ostream &stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, gp->m_mat_l_qm_) && stream.good();
                },
            },
            {
                "mat_alpha",
                [](const Self *gp, std::ostream &stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, gp->m_mat_alpha_) && stream.good();
                },
            },
            {
                "train_set",
                [](const Self *gp, std::ostream &stream) -> bool {
                    return gp->m_train_set_.Write(stream) && stream.good();
                },
            },
        };
        return WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    SparsePseudoInputGaussianProcess<Dtype>::Read(std::istream &s) {
        using namespace common;
        using Self = SparsePseudoInputGaussianProcess;
        static const TokenReadFunctionPairs<Self> token_function_pairs = {
            {
                "setting",
                [](Self *self, std::istream &stream) -> bool {
                    return self->m_setting_->Read(stream) && stream.good();
                },
            },
            {
                "trained",
                [](Self *self, std::istream &stream) -> bool {
                    stream >> self->m_trained_;
                    return true;
                },
            },
            {
                "trained_once",
                [](Self *self, std::istream &stream) -> bool {
                    stream >> self->m_trained_once_;
                    return true;
                },
            },
            {
                "mat_l_qm_updated",
                [](Self *self, std::istream &stream) -> bool {
                    stream >> self->m_mat_l_qm_updated_;
                    return true;
                },
            },
            {
                "kernel",
                [](Self *self, std::istream &stream) -> bool {
                    bool has_kernel;
                    stream >> has_kernel;
                    SkipLine(stream);
                    if (!has_kernel) { return stream.good(); }
                    self->m_kernel_ = Covariance::CreateCovariance(
                        self->m_setting_->kernel_type,
                        self->m_setting_->kernel);
                    if (!self->m_kernel_->Read(stream)) { return false; }
                    const auto rank_reduced_kernel =
                        std::dynamic_pointer_cast<ReducedRankCovariance>(self->m_kernel_);
                    self->m_reduced_rank_kernel_ = rank_reduced_kernel != nullptr;
                    if (self->m_reduced_rank_kernel_) {
                        rank_reduced_kernel->BuildSpectralDensities();
                    }
                    return true;
                },
            },
            {
                "pseudo_points",
                [](Self *self, std::istream &stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, self->m_pseudo_points_) &&
                           stream.good();
                },
            },
            {
                "mat_km",
                [](Self *self, std::istream &stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, self->m_mat_km_) &&
                           stream.good();
                },
            },
            {
                "sparse_mat_km",
                [](Self *self, std::istream &stream) -> bool {
                    return LoadSparseEigenMatrixFromBinaryStream(stream, self->m_sparse_mat_km_) &&
                           stream.good();
                },
            },
            {
                "mat_l_km",
                [](Self *self, std::istream &stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, self->m_mat_l_km_) &&
                           stream.good();
                },
            },
            {
                "mat_qm",
                [](Self *self, std::istream &stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, self->m_mat_qm_) &&
                           stream.good();
                },
            },
            {
                "mat_l_qm",
                [](Self *self, std::istream &stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, self->m_mat_l_qm_) &&
                           stream.good();
                },
            },
            {
                "mat_alpha",
                [](Self *self, std::istream &stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, self->m_mat_alpha_) &&
                           stream.good();
                },
            },
            {
                "train_set",
                [](Self *self, std::istream &stream) -> bool {
                    return self->m_train_set_.Read(stream) && stream.good();
                },
            },
        };
        return ReadTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    SparsePseudoInputGaussianProcess<Dtype>::UpdateDense(const bool parallel) {
        (void) parallel;
        m_trained_ = m_trained_once_;
        if (!m_train_set_.num_samples) { return false; }  // no training samples

        // compute K_MN
        const long m = m_pseudo_points_.cols();
        MatrixX mat_kmn(m, m_train_set_.num_samples);
        (void) m_kernel_
            ->ComputeKtest(m_pseudo_points_, m, m_train_set_.x, m_train_set_.num_samples, mat_kmn);

        // update Q_M and alpha
        MatrixX mat_kmn_scaled = mat_kmn;
        auto mat_l_km = m_mat_l_km_.template triangularView<Eigen::Lower>();
        const Dtype *var = m_train_set_.var.data();
#pragma omp parallel for if (parallel) default(none) shared(mat_kmn, mat_l_km, var, mat_kmn_scaled)
        for (long i = 0; i < m_train_set_.num_samples; ++i) {
            VectorX beta = mat_kmn.col(i);
            mat_l_km.solveInPlace(beta);
            Dtype lambda = 1.0f - beta.squaredNorm();
            mat_kmn_scaled.col(i) *= 1.0f / (lambda + var[i]);
        }
        if (m_setting_->diagonal_qm) {
            m_mat_qm_ += mat_kmn_scaled.cwiseProduct(mat_kmn).rowwise().sum();
        } else {
            m_mat_qm_ += mat_kmn_scaled * mat_kmn.transpose();
        }
        m_mat_alpha_ += mat_kmn_scaled * m_train_set_.y.topRows(m_train_set_.num_samples);  // alpha

        m_trained_once_ = true;
        m_trained_ = true;

        ERL_DEBUG_ASSERT(!mat_kmn.hasNaN(), "mat_kmn has NaN values.");
        ERL_DEBUG_ASSERT(!mat_kmn_scaled.hasNaN(), "mat_kmn_scaled has NaN values.");
        ERL_DEBUG_ASSERT(!m_mat_qm_.hasNaN(), "m_mat_qm has NaN values.");
        ERL_DEBUG_ASSERT(!m_mat_alpha_.hasNaN(), "m_mat_alpha has NaN values.");

        return true;
    }

    template<typename Dtype>
    bool
    SparsePseudoInputGaussianProcess<Dtype>::UpdateSparse(const bool parallel) {
        (void) parallel;
        m_trained_ = m_trained_once_;
        if (!m_train_set_.num_samples) { return false; }  // no training samples
        const long m = m_pseudo_points_.cols();
        SparseMatrix mat_kmn(m, m_train_set_.num_samples);
        // compute K_MN
        (void) m_kernel_->ComputeKtestSparse(
            m_pseudo_points_,
            m,
            m_train_set_.x,
            m_train_set_.num_samples,
            m_setting_->sparse_zero_threshold,
            mat_kmn);

        // update Q_M
        SparseMatrix mat_kmn_scaled = mat_kmn;
        auto mat_l_km = m_mat_l_km_.template triangularView<Eigen::Lower>();
        const Dtype *var = m_train_set_.var.data();
#pragma omp parallel for if (parallel) default(none) shared(mat_kmn, mat_l_km, var, mat_kmn_scaled)
        for (long i = 0; i < m_train_set_.num_samples; ++i) {
            VectorX beta = mat_kmn.col(i).toDense();
            mat_l_km.solveInPlace(beta);
            Dtype lambda = 1.0f - beta.squaredNorm();
            mat_kmn_scaled.col(i) *= 1.0f / (lambda + var[i]);
        }
        if (m_setting_->diagonal_qm) {
            for (long i = 0; i < m_train_set_.num_samples; ++i) {
                m_mat_qm_ += mat_kmn_scaled.col(i).cwiseProduct(mat_kmn.col(i));
            }
        } else {
            m_mat_qm_ += mat_kmn_scaled * mat_kmn.transpose();
        }
        m_mat_alpha_ += mat_kmn_scaled * m_train_set_.y.topRows(m_train_set_.num_samples);  // alpha
        m_trained_once_ = true;
        m_trained_ = true;
        return true;
    }

    template<typename Dtype>
    void
    SparsePseudoInputGaussianProcess<Dtype>::PrepareLqm() {
        std::lock_guard<std::mutex> lock(m_mutex_);
        if (!m_mat_l_qm_updated_) {
            if (!m_setting_->diagonal_qm) { m_mat_l_qm_ = m_mat_qm_.llt().matrixL(); }
            m_mat_l_qm_updated_ = true;
        }
    }

    template class SparsePseudoInputGaussianProcess<double>;
    template class SparsePseudoInputGaussianProcess<float>;
}  // namespace erl::gaussian_process

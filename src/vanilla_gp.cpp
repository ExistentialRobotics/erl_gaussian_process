#include "erl_gaussian_process/vanilla_gp.hpp"

#include "erl_common/serialization.hpp"
#include "erl_common/template_helper.hpp"
#include "erl_common/yaml.hpp"

namespace erl::gaussian_process {
    template<typename Dtype>
    YAML::Node
    VanillaGaussianProcess<Dtype>::Setting::YamlConvertImpl::encode(const Setting &setting) {
        YAML::Node node;
        ERL_YAML_SAVE_ATTR(node, setting, kernel_type);
        ERL_YAML_SAVE_ATTR(node, setting, kernel_setting_type);
        ERL_YAML_SAVE_ATTR(node, setting, kernel);
        ERL_YAML_SAVE_ATTR(node, setting, max_num_samples);
        return node;
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::Setting::YamlConvertImpl::decode(
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
        return true;
    }

    template<typename Dtype>
    VanillaGaussianProcess<Dtype>::TestResult::TestResult(
        const VanillaGaussianProcess *gp,
        const Eigen::Ref<const MatrixX> &mat_x_test)
        : m_gp_(NotNull(gp, true, "gp = nullptr.")),
          m_num_test_(mat_x_test.cols()),
          m_reduced_rank_kernel_(m_gp_->m_reduced_rank_kernel_),
          m_x_dim_(m_gp_->m_train_set_.x_dim),
          m_y_dim_(m_gp_->m_train_set_.y_dim) {
        const bool success = m_gp_->ComputeKtest(mat_x_test, m_mat_k_test_);
        (void) success;
        ERL_DEBUG_ASSERT(success, "Failed to compute Ktest.");
    }

    template<typename Dtype>
    long
    VanillaGaussianProcess<Dtype>::TestResult::GetNumTest() const {
        return m_num_test_;
    }

    template<typename Dtype>
    const typename VanillaGaussianProcess<Dtype>::MatrixX &
    VanillaGaussianProcess<Dtype>::TestResult::GetKtest() const {
        return m_mat_k_test_;
    }

    template<typename Dtype>
    void
    VanillaGaussianProcess<Dtype>::TestResult::GetMean(
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
            vec_f_out.size() == m_num_test_,
            "vec_f_out.size() = {}, it should be {}.",
            vec_f_out.size(),
            m_num_test_);
        const auto alpha = m_gp_->m_mat_alpha_.col(y_index).head(m_gp_->m_k_train_cols_);
        Dtype *f = vec_f_out.data();
#pragma omp parallel for if (parallel) default(none) shared(alpha, f)
        for (long i = 0; i < m_num_test_; ++i) { f[i] = m_mat_k_test_.col(i).dot(alpha); }
    }

    template<typename Dtype>
    void
    VanillaGaussianProcess<Dtype>::TestResult::GetMean(
        const long index,
        const long y_index,
        Dtype &f) const {
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
        // f = ktest(xt, X) @ (ktrain(X, X) + sigma * I).inv() @ y
        // f = ktest(xt, X) @ alpha
        const auto alpha = m_gp_->m_mat_alpha_.col(y_index).head(m_gp_->m_k_train_cols_);
        f = m_mat_k_test_.col(index).dot(alpha);  // h_{y_index}(x_{index})
    }

    template<typename Dtype>
    void
    VanillaGaussianProcess<Dtype>::TestResult::GetVariance(
        Eigen::Ref<VectorX> vec_var_out,
        const bool parallel) const {
        (void) parallel;
        // var = ktest(xt, xt) - ktest(xt, X) @ (ktest(X, X) + sigma * I).m_inv_() @ ktest(X, xt)
        //     = ktest(xt, xt) - ktest(xt, X) @ (m_l_ @ m_l_.T).m_inv_() @ ktest(X, xt)
        const_cast<TestResult *>(this)->PrepareAlphaTest();
        Dtype *var = vec_var_out.data();
#pragma omp parallel for if (parallel) default(none) shared(var)
        for (long i = 0; i < m_num_test_; ++i) {
            Dtype &var_i = var[i];
            var_i = m_mat_alpha_test_.col(i).squaredNorm();
            if (m_reduced_rank_kernel_) { continue; }
            var_i = 1.0f - var_i;  // variance of h(x)
        }
    }

    template<typename Dtype>
    void
    VanillaGaussianProcess<Dtype>::TestResult::GetVariance(long index, Dtype &var) const {
        const_cast<TestResult *>(this)->PrepareAlphaTest();
        var = m_mat_alpha_test_.col(index).squaredNorm();
        if (m_reduced_rank_kernel_) { return; }
        var = 1.0f - var;  // variance of h(x)
    }

    template<typename Dtype>
    void
    VanillaGaussianProcess<Dtype>::TestResult::PrepareAlphaTest() {
        if (m_mat_alpha_test_.size() > 0) { return; }
        const long rows = m_mat_k_test_.rows();
        auto mat_l =
            m_gp_->m_mat_l_.topLeftCorner(rows, rows).template triangularView<Eigen::Lower>();
        m_mat_alpha_test_.resize(rows, m_mat_k_test_.cols());
#pragma omp parallel for default(none) shared(mat_l)
        for (long i = 0; i < m_mat_k_test_.cols(); ++i) {
            m_mat_alpha_test_.col(i) = mat_l.solve(m_mat_k_test_.col(i));
        }
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
        if (other.x.rows() < x_dim || other.x.cols() < num_samples) { return false; }
        if (x.topLeftCorner(x_dim, num_samples) != other.x.topLeftCorner(x_dim, num_samples)) {
            return false;
        }
        if (other.y.rows() < num_samples || other.y.cols() < y_dim) { return false; }
        if (y.topLeftCorner(num_samples, y_dim) != other.y.topLeftCorner(num_samples, y_dim)) {
            return false;
        }
        if (other.var.size() < num_samples) { return false; }
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
        using namespace common;
        static const TokenWriteFunctionPairs<TrainSet> token_function_pairs = {
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
                    return SaveEigenMatrixToBinaryStream(stream, train_set->x) && stream.good();
                },
            },
            {
                "y",
                [](const TrainSet *train_set, std::ostream &stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, train_set->y) && stream.good();
                },
            },
            {
                "var",
                [](const TrainSet *train_set, std::ostream &stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, train_set->var) && stream.good();
                },
            },
        };
        return WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::TrainSet::Read(std::istream &s) {
        using namespace common;
        static const TokenReadFunctionPairs<TrainSet> token_function_pairs = {
            {
                "x_dim",
                [](TrainSet *train_set, std::istream &stream) -> bool {
                    stream >> train_set->x_dim;
                    return true;
                },
            },
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
                    return LoadEigenMatrixFromBinaryStream(stream, train_set->x) && stream.good();
                },
            },
            {
                "y",
                [](TrainSet *train_set, std::istream &stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, train_set->y) && stream.good();
                },
            },
            {
                "var",
                [](TrainSet *train_set, std::istream &stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, train_set->var) && stream.good();
                },
            }};
        return ReadTokens(s, this, token_function_pairs);
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
            if (m_reduced_rank_kernel_) {  // rank-reduced kernel is stateful, need to copy it
                *std::reinterpret_pointer_cast<ReducedRankCovariance>(m_kernel_) =
                    *std::reinterpret_pointer_cast<ReducedRankCovariance>(other.m_kernel_);
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
            if (m_reduced_rank_kernel_) {  // rank-reduced kernel is stateful, need to copy it
                *std::reinterpret_pointer_cast<ReducedRankCovariance>(m_kernel_) =
                    *std::reinterpret_pointer_cast<ReducedRankCovariance>(other.m_kernel_);
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
        if (m_reduced_rank_kernel_) {
            return std::reinterpret_pointer_cast<ReducedRankCovariance>(m_kernel_)
                ->GetCoordOrigin();
        }
        ERL_DEBUG_ASSERT(m_train_set_.x_dim > 0, "train set should be initialized first.");
        return VectorX::Zero(m_train_set_.x_dim);
    }

    template<typename Dtype>
    void
    VanillaGaussianProcess<Dtype>::SetKernelCoordOrigin(const VectorX &coord_origin) const {
        if (m_reduced_rank_kernel_) {
            std::reinterpret_pointer_cast<ReducedRankCovariance>(m_kernel_)->SetCoordOrigin(
                coord_origin);
        }
    }

    template<typename Dtype>
    void
    VanillaGaussianProcess<Dtype>::Reset(
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

        m_train_set_.Reset(max_num_samples, x_dim, y_dim);
        ERL_ASSERTM(AllocateMemory(max_num_samples, x_dim, y_dim), "Failed to allocate memory.");
        m_trained_ = false;
        m_k_train_updated_ = false;
        m_k_train_rows_ = 0;
        m_k_train_cols_ = 0;
    }

    template<typename Dtype>
    std::pair<long, long>
    VanillaGaussianProcess<Dtype>::GetKtrainSize() const {
        return {m_k_train_rows_, m_k_train_cols_};
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
    typename VanillaGaussianProcess<Dtype>::MatrixX &
    VanillaGaussianProcess<Dtype>::GetKtrain() {
        return m_mat_k_train_;
    }

    template<typename Dtype>
    const typename VanillaGaussianProcess<Dtype>::MatrixX &
    VanillaGaussianProcess<Dtype>::GetCholeskyDecomposition() const {
        return m_mat_l_;
    }

    template<typename Dtype>
    typename VanillaGaussianProcess<Dtype>::MatrixX &
    VanillaGaussianProcess<Dtype>::GetCholeskyDecomposition() {
        return m_mat_l_;
    }

    template<typename Dtype>
    const typename VanillaGaussianProcess<Dtype>::MatrixX &
    VanillaGaussianProcess<Dtype>::GetAlpha() const {
        return m_mat_alpha_;
    }

    template<typename Dtype>
    typename VanillaGaussianProcess<Dtype>::MatrixX &
    VanillaGaussianProcess<Dtype>::GetAlpha() {
        return m_mat_alpha_;
    }

    template<typename Dtype>
    std::size_t
    VanillaGaussianProcess<Dtype>::GetMemoryUsage() const {
        std::size_t memory_usage = sizeof(VanillaGaussianProcess);
        if (m_setting_ != nullptr) { memory_usage += sizeof(Setting); }
        if (m_kernel_ != nullptr) { memory_usage += m_kernel_->GetMemoryUsage(); }
        memory_usage += sizeof(TrainSet);
        memory_usage += (m_train_set_.x.size() + m_train_set_.y.size() + m_train_set_.var.size()) *
                        sizeof(Dtype);
        memory_usage +=
            (m_mat_k_train_.size() + m_mat_l_.size() + m_mat_alpha_.size()) * sizeof(Dtype);
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
        std::tie(m_k_train_rows_, m_k_train_cols_) =
            m_kernel_->ComputeKtrain(x, var, num_samples, m_mat_k_train_, m_mat_alpha_);
        m_k_train_updated_ = true;
        return true;
    }

    template<typename Dtype>
    void
    VanillaGaussianProcess<Dtype>::Solve() {
        const auto mat_ktrain = m_mat_k_train_.topLeftCorner(m_k_train_rows_, m_k_train_cols_);
        auto &&mat_l = m_mat_l_.topLeftCorner(m_k_train_rows_, m_k_train_cols_);
        auto alpha = m_mat_alpha_.topRows(m_k_train_cols_);
        // A = ktrain(mat_x_train, mat_x_train) + sigma * I = mat_l @ mat_l.T
        mat_l = mat_ktrain.llt().matrixL();
        // A.inv() @ vec_alpha
        mat_l.template triangularView<Eigen::Lower>().solveInPlace(alpha);
        mat_l.transpose().template triangularView<Eigen::Upper>().solveInPlace(alpha);
        m_trained_once_ = true;
        m_trained_ = true;
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
        Solve();
        return true;
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::ComputeKtest(
        const Eigen::Ref<const MatrixX> &mat_x_test,
        MatrixX &mat_k_test) const {
        if (!m_trained_) { return false; }

        auto &[x_dim, y_dim, n_samples, x_train, y_train, var_x] = m_train_set_;
        const long num_test = mat_x_test.cols();
        if (num_test == 0) {
            ERL_WARN("num_test = {}, it should be > 0.", num_test);
            return false;
        }

        auto size0 = m_kernel_->GetMinimumKtestSize(n_samples, 0, x_dim, num_test, false);
        mat_k_test.resize(size0.first, size0.second);
        auto size1 = m_kernel_->ComputeKtest(x_train, n_samples, mat_x_test, num_test, mat_k_test);
        (void) size1;
        ERL_DEBUG_ASSERT(size0 == size1, "output size is ({}), it should be ({}).", size0, size1);
        ERL_DEBUG_ASSERT(
            mat_k_test.rows() == m_k_train_cols_,
            "m_mat_k_test_.rows() = {}, it should be {}.",
            mat_k_test.rows(),
            m_k_train_cols_);
        ERL_DEBUG_ASSERT(
            mat_k_test.cols() == num_test,
            "m_mat_k_test_.cols() = {}, it should be {}.",
            mat_k_test.cols(),
            num_test);

        return true;
    }

    template<typename Dtype>
    std::shared_ptr<typename VanillaGaussianProcess<Dtype>::TestResult>
    VanillaGaussianProcess<Dtype>::Test(const Eigen::Ref<const MatrixX> &mat_x_test) const {
        if (!m_trained_) { return nullptr; }
        return std::make_shared<TestResult>(this, mat_x_test);
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::operator==(const VanillaGaussianProcess &other) const {
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
        if (m_reduced_rank_kernel_ != other.m_reduced_rank_kernel_) { return false; }
        if (m_train_set_ != other.m_train_set_) { return false; }
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
        using namespace common;
        static const TokenWriteFunctionPairs<VanillaGaussianProcess> token_function_pairs = {
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
                "k_train_updated",
                [](const VanillaGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream << gp->m_k_train_updated_;
                    return true;
                },
            },
            {
                "k_train_rows",
                [](const VanillaGaussianProcess *gp, std::ostream &stream) -> bool {
                    stream << gp->m_k_train_rows_;
                    return true;
                },
            },
            {
                "k_train_cols",
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
                    return SaveEigenMatrixToBinaryStream(stream, gp->m_mat_k_train_) &&
                           stream.good();
                },
            },
            {
                "mat_l",
                [](const VanillaGaussianProcess *gp, std::ostream &stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, gp->m_mat_l_) && stream.good();
                },
            },
            {
                "mat_alpha",
                [](const VanillaGaussianProcess *gp, std::ostream &stream) -> bool {
                    return SaveEigenMatrixToBinaryStream(stream, gp->m_mat_alpha_) && stream.good();
                },
            },
            {
                "train_set",
                [](const VanillaGaussianProcess *gp, std::ostream &stream) -> bool {
                    return gp->m_train_set_.Write(stream) && stream.good();
                },
            },
        };
        return WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::Read(std::istream &s) {
        using namespace common;
        static const TokenReadFunctionPairs<VanillaGaussianProcess> token_function_pairs = {
            {
                "setting",
                [](VanillaGaussianProcess *self, std::istream &stream) -> bool {
                    return self->m_setting_->Read(stream) && stream.good();
                },
            },
            {
                "trained",
                [](VanillaGaussianProcess *self, std::istream &stream) -> bool {
                    stream >> self->m_trained_;
                    return true;
                },
            },
            {
                "trained_once",
                [](VanillaGaussianProcess *self, std::istream &stream) -> bool {
                    stream >> self->m_trained_once_;
                    return true;
                },
            },
            {
                "k_train_updated",
                [](VanillaGaussianProcess *self, std::istream &stream) -> bool {
                    stream >> self->m_k_train_updated_;
                    return true;
                },
            },
            {
                "k_train_rows",
                [](VanillaGaussianProcess *self, std::istream &stream) -> bool {
                    stream >> self->m_k_train_rows_;
                    return true;
                },
            },
            {
                "k_train_cols",
                [](VanillaGaussianProcess *self, std::istream &stream) -> bool {
                    stream >> self->m_k_train_cols_;
                    return true;
                },
            },
            {
                "kernel",
                [](VanillaGaussianProcess *self, std::istream &stream) -> bool {
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
                "mat_k_train",
                [](VanillaGaussianProcess *self, std::istream &stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, self->m_mat_k_train_) &&
                           stream.good();
                },
            },
            {
                "mat_l",
                [](VanillaGaussianProcess *self, std::istream &stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, self->m_mat_l_) && stream.good();
                },
            },
            {
                "mat_alpha",
                [](VanillaGaussianProcess *self, std::istream &stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, self->m_mat_alpha_) &&
                           stream.good();
                },
            },
            {
                "train_set",
                [](VanillaGaussianProcess *self, std::istream &stream) -> bool {
                    return self->m_train_set_.Read(stream) && stream.good();
                },
            },
        };
        return ReadTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    VanillaGaussianProcess<Dtype>::AllocateMemory(
        const long max_num_samples,
        const long x_dim,
        const long y_dim) {
        if (max_num_samples <= 0 || x_dim <= 0 || y_dim <= 0) { return false; }  // invalid input
        if (m_setting_->max_num_samples > 0 && max_num_samples > m_setting_->max_num_samples) {
            return false;
        }
        if (m_setting_->kernel->x_dim > 0 && x_dim != m_setting_->kernel->x_dim) { return false; }
        InitKernel();
        const auto [rows, cols] = m_kernel_->GetMinimumKtrainSize(max_num_samples, 0, 0);
        if (m_mat_k_train_.rows() < rows || m_mat_k_train_.cols() < cols) {
            m_mat_k_train_.resize(rows, cols);
        }
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
            ERL_ASSERTM(
                m_kernel_ != nullptr,
                "failed to create kernel of type {}.",
                m_setting_->kernel_type);
            const auto rank_reduced_kernel =
                std::dynamic_pointer_cast<ReducedRankCovariance>(m_kernel_);
            m_reduced_rank_kernel_ = rank_reduced_kernel != nullptr;
            if (m_reduced_rank_kernel_) { rank_reduced_kernel->BuildSpectralDensities(); }
        }
    }

    template class VanillaGaussianProcess<double>;
    template class VanillaGaussianProcess<float>;
}  // namespace erl::gaussian_process

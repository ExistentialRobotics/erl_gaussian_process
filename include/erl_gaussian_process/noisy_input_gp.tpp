#pragma once

template<typename Dtype>
YAML::Node
NoisyInputGaussianProcess<Dtype>::Setting::YamlConvertImpl::encode(const Setting &setting) {
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
NoisyInputGaussianProcess<Dtype>::Setting::YamlConvertImpl::decode(const YAML::Node &node, Setting &setting) {
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
NoisyInputGaussianProcess<Dtype>::NoisyInputGaussianProcess(const NoisyInputGaussianProcess &other)
    : m_x_dim_(other.m_x_dim_),
      m_num_train_samples_(other.m_num_train_samples_),
      m_num_train_samples_with_grad_(other.m_num_train_samples_with_grad_),
      m_trained_(other.m_trained_),
      m_three_over_scale_square_(other.m_three_over_scale_square_),
      m_setting_(other.m_setting_),
      m_reduced_rank_kernel_(other.m_reduced_rank_kernel_),
      m_mat_x_train_(other.m_mat_x_train_),
      m_vec_y_train_(other.m_vec_y_train_),
      m_mat_grad_train_(other.m_mat_grad_train_),
      m_mat_k_train_(other.m_mat_k_train_),
      m_mat_l_(other.m_mat_l_),
      m_vec_grad_flag_(other.m_vec_grad_flag_),
      m_vec_alpha_(other.m_vec_alpha_),
      m_vec_var_x_(other.m_vec_var_x_),
      m_vec_var_h_(other.m_vec_var_h_),
      m_vec_var_grad_(other.m_vec_var_grad_) {
    if (other.m_kernel_ != nullptr) {
        m_kernel_ = Covariance::CreateCovariance(m_setting_->kernel_type, m_setting_->kernel);
        if (m_reduced_rank_kernel_) {  // rank-reduced kernel is stateful, so we need to copy the kernel
            *std::reinterpret_pointer_cast<covariance::ReducedRankCovariance>(m_kernel_) =
                *std::reinterpret_pointer_cast<covariance::ReducedRankCovariance>(other.m_kernel_);
        }
    }
}

template<typename Dtype>
NoisyInputGaussianProcess<Dtype> &
NoisyInputGaussianProcess<Dtype>::operator=(const NoisyInputGaussianProcess &other) {
    if (this == &other) { return *this; }
    m_x_dim_ = other.m_x_dim_;
    m_num_train_samples_ = other.m_num_train_samples_;
    m_num_train_samples_with_grad_ = other.m_num_train_samples_with_grad_;
    m_trained_ = other.m_trained_;
    m_three_over_scale_square_ = other.m_three_over_scale_square_;
    m_setting_ = other.m_setting_;
    m_reduced_rank_kernel_ = other.m_reduced_rank_kernel_;
    m_mat_x_train_ = other.m_mat_x_train_;
    m_vec_y_train_ = other.m_vec_y_train_;
    m_mat_grad_train_ = other.m_mat_grad_train_;
    m_mat_k_train_ = other.m_mat_k_train_;
    m_mat_l_ = other.m_mat_l_;
    m_vec_grad_flag_ = other.m_vec_grad_flag_;
    m_vec_alpha_ = other.m_vec_alpha_;
    m_vec_var_x_ = other.m_vec_var_x_;
    m_vec_var_h_ = other.m_vec_var_h_;
    m_vec_var_grad_ = other.m_vec_var_grad_;
    if (other.m_kernel_ != nullptr) {
        m_kernel_ = Covariance::CreateCovariance(m_setting_->kernel_type, m_setting_->kernel);
        if (m_reduced_rank_kernel_) {  // rank-reduced kernel is stateful, so we need to copy the kernel
            *std::reinterpret_pointer_cast<ReducedRankCovariance>(m_kernel_) = *std::reinterpret_pointer_cast<ReducedRankCovariance>(other.m_kernel_);
        }
    }
    return *this;
}

template<typename Dtype>
typename NoisyInputGaussianProcess<Dtype>::VectorX
NoisyInputGaussianProcess<Dtype>::GetKernelCoordOrigin() const {
    if (m_reduced_rank_kernel_) { return std::reinterpret_pointer_cast<ReducedRankCovariance>(m_kernel_)->GetCoordOrigin(); }
    return VectorX::Zero(m_x_dim_);
}

template<typename Dtype>
void
NoisyInputGaussianProcess<Dtype>::SetKernelCoordOrigin(const VectorX &coord_origin) const {
    if (m_reduced_rank_kernel_) { std::reinterpret_pointer_cast<ReducedRankCovariance>(m_kernel_)->SetCoordOrigin(coord_origin); }
}

template<typename Dtype>
void
NoisyInputGaussianProcess<Dtype>::Reset(const long max_num_samples, const long x_dim) {
    ERL_ASSERTM(max_num_samples > 0, "max_num_samples should be > 0.");
    ERL_ASSERTM(x_dim > 0, "x_dim should be > 0.");
    if (m_setting_->max_num_samples > 0 && m_setting_->kernel->x_dim > 0) {  // memory already allocated
        ERL_ASSERTM(m_setting_->max_num_samples >= max_num_samples, "max_num_samples should be <= {}.", m_setting_->max_num_samples);
    } else {
        if (m_setting_->kernel->x_dim > 0) { ERL_ASSERTM(m_setting_->kernel->x_dim == x_dim, "x_dim should be {}.", m_setting_->kernel->x_dim); }
        ERL_ASSERTM(AllocateMemory(max_num_samples, x_dim), "Failed to allocate memory.");
    }
    m_trained_ = false;
    m_k_train_updated_ = false;
    m_k_train_rows_ = 0;
    m_k_train_cols_ = 0;
    InitKernel();
    m_three_over_scale_square_ = 3. * m_setting_->kernel->alpha / (m_setting_->kernel->scale * m_setting_->kernel->scale);
    m_num_train_samples_ = 0;
    m_num_train_samples_with_grad_ = 0;
    m_x_dim_ = x_dim;
}

template<typename Dtype>
std::size_t
NoisyInputGaussianProcess<Dtype>::GetMemoryUsage() const {
    std::size_t memory_usage = sizeof(NoisyInputGaussianProcess);
    if (m_setting_ != nullptr) { memory_usage += sizeof(Setting); }
    if (m_kernel_ != nullptr) { memory_usage += m_kernel_->GetMemoryUsage(); }
    memory_usage += m_mat_x_train_.size() * sizeof(Dtype);
    memory_usage += m_vec_y_train_.size() * sizeof(Dtype);
    memory_usage += m_mat_grad_train_.size() * sizeof(Dtype);
    memory_usage += m_mat_k_train_.size() * sizeof(Dtype);
    memory_usage += m_mat_l_.size() * sizeof(Dtype);
    memory_usage += m_vec_grad_flag_.size() * sizeof(long);
    memory_usage += m_vec_alpha_.size() * sizeof(Dtype);
    memory_usage += m_vec_var_x_.size() * sizeof(Dtype);
    memory_usage += m_vec_var_h_.size() * sizeof(Dtype);
    memory_usage += m_vec_var_grad_.size() * sizeof(Dtype);
    return memory_usage;
}

template<typename Dtype>
bool
NoisyInputGaussianProcess<Dtype>::UpdateKtrain(const long num_train_samples) {
    if (m_k_train_updated_) { return true; }
    m_num_train_samples_ = num_train_samples;
    if (m_num_train_samples_ <= 0) {
        ERL_WARN("num_train_samples = {}, it should be > 0.", m_num_train_samples_);
        return false;
    }
    if (m_setting_->no_gradient_observation) {
        m_vec_grad_flag_.setZero(m_num_train_samples_);
        m_vec_alpha_.head(m_num_train_samples_) = m_vec_y_train_.head(m_num_train_samples_);
        std::tie(m_k_train_rows_, m_k_train_cols_) =
            m_kernel_->ComputeKtrain(m_mat_x_train_, m_vec_var_x_ + m_vec_var_h_, m_num_train_samples_, m_mat_k_train_, m_vec_alpha_);
    } else {
        m_num_train_samples_with_grad_ = m_vec_grad_flag_.head(m_num_train_samples_).count();
        const long m = m_num_train_samples_ + m_x_dim_ * m_num_train_samples_with_grad_;
        ERL_ASSERTM(m_vec_alpha_.size() >= m, "m_vec_alpha_ should have size >= {}.", m);
        Dtype *alpha = m_vec_alpha_.data();
        Dtype *y = m_vec_y_train_.data();
        long *grad_flag = m_vec_grad_flag_.data();
        for (long i = 0, j = m_num_train_samples_; i < m_num_train_samples_; ++i) {
            alpha[i] = y[i];  // h(x_i)
            if (!grad_flag[i]) { continue; }
            Dtype *grad_i = m_mat_grad_train_.col(i).data();
            for (long k = 0, l = j++; k < m_x_dim_; ++k, l += m_num_train_samples_with_grad_) { alpha[l] = grad_i[k]; }
        }

        // Compute kernel matrix
        std::tie(m_k_train_rows_, m_k_train_cols_) = m_kernel_->ComputeKtrainWithGradient(
            m_mat_x_train_,
            m_num_train_samples_,
            m_vec_grad_flag_,
            m_vec_var_x_,
            m_vec_var_h_,
            m_vec_var_grad_,
            m_mat_k_train_,
            m_vec_alpha_);
    }
    ERL_DEBUG_ASSERT(!m_mat_k_train_.topLeftCorner(m_k_train_rows_, m_k_train_cols_).hasNaN(), "NaN in m_mat_k_train_!");
    m_k_train_updated_ = true;
    return true;
}

template<typename Dtype>
void
NoisyInputGaussianProcess<Dtype>::Train(const long num_train_samples) {

    if (m_trained_) {
        ERL_WARN("The model has been trained. Please reset the model before training.");
        return;
    }
    m_trained_ = m_trained_once_;
    if (!UpdateKtrain(num_train_samples)) { return; }

    const auto mat_ktrain = m_mat_k_train_.topLeftCorner(m_k_train_rows_, m_k_train_cols_);  // square matrix
    auto &&mat_l = m_mat_l_.topLeftCorner(m_k_train_rows_, m_k_train_cols_);                 // square matrix, lower triangular
    const auto vec_alpha = m_vec_alpha_.head(m_k_train_cols_);                               // h and gradient of h
    mat_l = mat_ktrain.llt().matrixL();
    mat_l.template triangularView<Eigen::Lower>().solveInPlace(vec_alpha);
    mat_l.transpose().template triangularView<Eigen::Upper>().solveInPlace(vec_alpha);
    m_trained_once_ = true;
    m_trained_ = true;
}

template<typename Dtype>
bool
NoisyInputGaussianProcess<Dtype>::Test(
    const Eigen::Ref<const MatrixX> &mat_x_test,
    Eigen::Ref<MatrixX> mat_f_out,
    Eigen::Ref<MatrixX> mat_var_out,
    Eigen::Ref<MatrixX> mat_cov_out,
    const bool predict_gradient) const {

    if (!m_trained_) {
        ERL_WARN("The model has not been trained.");
        return false;
    }

    const long dim = mat_x_test.rows();
    const long n = mat_x_test.cols();
    if (n == 0) { return false; }

    // compute mean and gradient of the test queries
    ERL_ASSERTM(
        mat_f_out.rows() >= (predict_gradient ? dim + 1 : 1),
        "mat_f_out.rows() = {}, it should be >= {}.",
        mat_f_out.rows(),
        predict_gradient ? dim + 1 : 1);
    ERL_ASSERTM(mat_f_out.cols() >= n, "mat_f_out.cols() = {}, not enough for {} test queries.", mat_f_out.cols(), n);

    const auto [ktest_rows, ktest_cols] = m_kernel_->GetMinimumKtestSize(m_num_train_samples_, m_num_train_samples_with_grad_, dim, n, predict_gradient);
    MatrixX ktest(ktest_rows, ktest_cols);  // (dim of train samples, dim of test queries)
    const auto [output_rows, output_cols] =
        m_kernel_->ComputeKtestWithGradient(m_mat_x_train_, m_num_train_samples_, m_vec_grad_flag_, mat_x_test, n, predict_gradient, ktest);
    (void) output_rows;
    (void) output_cols;
    ERL_DEBUG_ASSERT(
        output_rows == ktest_rows && output_cols == ktest_cols,
        "output_size = ({}, {}), it should be ({}, {}).",
        output_rows,
        output_cols,
        ktest_rows,
        ktest_cols);

    ComputeValuePrediction(ktest, dim, n, predict_gradient, mat_f_out);
    ComputeCovPrediction(ktest, dim, n, predict_gradient, mat_var_out, mat_cov_out);
    return true;
}

template<typename Dtype>
bool
NoisyInputGaussianProcess<Dtype>::operator==(const NoisyInputGaussianProcess &other) const {
    if (m_x_dim_ != other.m_x_dim_) { return false; }
    if (m_num_train_samples_ != other.m_num_train_samples_) { return false; }
    if (m_num_train_samples_with_grad_ != other.m_num_train_samples_with_grad_) { return false; }
    if (m_trained_ != other.m_trained_) { return false; }
    if (m_three_over_scale_square_ != other.m_three_over_scale_square_) { return false; }
    if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
    if (m_setting_ != nullptr && (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) { return false; }
    // m_kernel_ is set by Reset() according to m_setting_
    if (m_num_train_samples_ == 0) { return true; }  // no training samples, no need to compare the following
    if (m_mat_x_train_.rows() != other.m_mat_x_train_.rows() || m_mat_x_train_.cols() != other.m_mat_x_train_.cols() ||
        std::memcmp(m_mat_x_train_.data(), other.m_mat_x_train_.data(), m_mat_x_train_.size() * sizeof(Dtype)) != 0) {
        return false;
    }
    if (m_vec_y_train_.size() != other.m_vec_y_train_.size() ||
        std::memcmp(m_vec_y_train_.data(), other.m_vec_y_train_.data(), m_vec_y_train_.size() * sizeof(Dtype)) != 0) {
        return false;
    }
    if (m_mat_grad_train_.rows() != other.m_mat_grad_train_.rows() || m_mat_grad_train_.cols() != other.m_mat_grad_train_.cols() ||
        std::memcmp(m_mat_grad_train_.data(), other.m_mat_grad_train_.data(), m_mat_grad_train_.size() * sizeof(Dtype)) != 0) {
        return false;
    }
    if (m_mat_k_train_.rows() != other.m_mat_k_train_.rows() || m_mat_k_train_.cols() != other.m_mat_k_train_.cols() ||
        std::memcmp(m_mat_k_train_.data(), other.m_mat_k_train_.data(), m_mat_k_train_.size() * sizeof(Dtype)) != 0) {
        return false;
    }
    if (m_mat_l_.rows() != other.m_mat_l_.rows() || m_mat_l_.cols() != other.m_mat_l_.cols() ||
        std::memcmp(m_mat_l_.data(), other.m_mat_l_.data(), m_mat_l_.size() * sizeof(Dtype)) != 0) {
        return false;
    }
    if (m_vec_grad_flag_.size() != other.m_vec_grad_flag_.size() ||
        std::memcmp(m_vec_grad_flag_.data(), other.m_vec_grad_flag_.data(), m_vec_grad_flag_.size() * sizeof(long)) != 0) {
        return false;
    }
    if (m_vec_alpha_.size() != other.m_vec_alpha_.size() ||
        std::memcmp(m_vec_alpha_.data(), other.m_vec_alpha_.data(), m_vec_alpha_.size() * sizeof(Dtype)) != 0) {
        return false;
    }
    if (m_vec_var_x_.size() != other.m_vec_var_x_.size() ||
        std::memcmp(m_vec_var_x_.data(), other.m_vec_var_x_.data(), m_vec_var_x_.size() * sizeof(Dtype)) != 0) {
        return false;
    }
    if (m_vec_var_h_.size() != other.m_vec_var_h_.size() ||
        std::memcmp(m_vec_var_h_.data(), other.m_vec_var_h_.data(), m_vec_var_h_.size() * sizeof(Dtype)) != 0) {
        return false;
    }
    if (m_vec_var_grad_.size() != other.m_vec_var_grad_.size() ||
        std::memcmp(m_vec_var_grad_.data(), other.m_vec_var_grad_.data(), m_vec_var_grad_.size() * sizeof(Dtype)) != 0) {
        return false;
    }
    return true;
}

template<typename Dtype>
bool
NoisyInputGaussianProcess<Dtype>::Write(const std::string &filename) const {
    ERL_INFO("Writing {} to file: {}", demangle(typeid(*this).name()), filename);
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
NoisyInputGaussianProcess<Dtype>::Write(std::ostream &s) const {
    s << "# " << demangle(typeid(*this).name()) << std::endl  //
      << "# (feel free to add / change comments, but leave the first line as it is!)" << std::endl
      << "setting" << std::endl;
    // write setting
    if (!m_setting_->Write(s)) {
        ERL_WARN("Failed to write setting.");
        return false;
    }
    // write data
    s << "x_dim " << m_x_dim_ << std::endl
      << "num_train_samples " << m_num_train_samples_ << std::endl
      << "num_train_samples_with_grad " << m_num_train_samples_with_grad_ << std::endl
      << "trained " << m_trained_ << std::endl
      << "trained_once " << m_trained_once_ << std::endl
      << "ktrain_updated " << m_k_train_updated_ << std::endl
      << "ktrain_rows " << m_k_train_rows_ << std::endl
      << "ktrain_cols " << m_k_train_cols_ << std::endl
      << "kernel " << (m_kernel_ != nullptr) << std::endl;
    if (m_kernel_ != nullptr && !m_kernel_->Write(s)) {
        ERL_WARN("Failed to write kernel.");
        return false;
    }
    s << "three_over_scale_square" << std::endl;
    s.write(reinterpret_cast<const char *>(&m_three_over_scale_square_), sizeof(Dtype));
    // m_kernel_ is set by Reset() according to m_setting_
    s << "mat_x_train" << std::endl;
    if (!common::SaveEigenMatrixToBinaryStream(s, m_mat_x_train_)) {
        ERL_WARN("Failed to write mat_x_train.");
        return false;
    }
    s << "vec_y_train" << std::endl;
    if (!common::SaveEigenMatrixToBinaryStream(s, m_vec_y_train_)) {
        ERL_WARN("Failed to write vec_y_train.");
        return false;
    }
    s << "mat_grad_train" << std::endl;
    if (!common::SaveEigenMatrixToBinaryStream(s, m_mat_grad_train_)) {
        ERL_WARN("Failed to write mat_grad_train.");
        return false;
    }
    s << "mat_k_train" << std::endl;
    if (!common::SaveEigenMatrixToBinaryStream(s, m_mat_k_train_)) {
        ERL_WARN("Failed to write mat_k_train.");
        return false;
    }
    s << "mat_l" << std::endl;
    if (!common::SaveEigenMatrixToBinaryStream(s, m_mat_l_)) {
        ERL_WARN("Failed to write mat_l.");
        return false;
    }
    s << "vec_grad_flag" << std::endl;
    if (!common::SaveEigenMatrixToBinaryStream(s, m_vec_grad_flag_)) {
        ERL_WARN("Failed to write vec_grad_flag.");
        return false;
    }
    s << "vec_alpha" << std::endl;
    if (!common::SaveEigenMatrixToBinaryStream(s, m_vec_alpha_)) {
        ERL_WARN("Failed to write vec_alpha.");
        return false;
    }
    s << "vec_var_x" << std::endl;
    if (!common::SaveEigenMatrixToBinaryStream(s, m_vec_var_x_)) {
        ERL_WARN("Failed to write vec_var_x.");
        return false;
    }
    s << "vec_var_h" << std::endl;
    if (!common::SaveEigenMatrixToBinaryStream(s, m_vec_var_h_)) {
        ERL_WARN("Failed to write vec_var_h.");
        return false;
    }
    s << "vec_var_grad" << std::endl;
    if (!common::SaveEigenMatrixToBinaryStream(s, m_vec_var_grad_)) {
        ERL_WARN("Failed to write vec_var_grad.");
        return false;
    }
    s << "end_of_NoisyInputGaussianProcess" << std::endl;
    return s.good();
}

template<typename Dtype>
bool
NoisyInputGaussianProcess<Dtype>::Read(const std::string &filename) {
    ERL_INFO("Reading {} from file: {}", demangle(typeid(*this).name()), filename);
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
NoisyInputGaussianProcess<Dtype>::Read(std::istream &s) {
    if (!s.good()) {
        ERL_WARN("Input stream is not ready for reading");
        return false;
    }

    // check if the first line is valid
    std::string line;
    std::getline(s, line);
    if (std::string file_header = fmt::format("# {}", demangle(typeid(*this).name()));
        line.compare(0, file_header.length(), file_header) != 0) {  // check if the first line is valid
        ERL_WARN("Header does not start with \"{}\"", file_header);
        return false;
    }

    auto skip_line = [&s]() {
        char c;
        do { c = static_cast<char>(s.get()); } while (s.good() && c != '\n');
    };

    static const char *tokens[] = {
        "setting",
        "x_dim",
        "num_train_samples",
        "num_train_samples_with_grad",
        "trained",
        "trained_once",
        "ktrain_updated",
        "ktrain_rows",
        "ktrain_cols",
        "kernel",
        "three_over_scale_square",
        "mat_x_train",
        "vec_y_train",
        "mat_grad_train",
        "mat_k_train",
        "mat_l",
        "vec_grad_flag",
        "vec_alpha",
        "vec_var_x",
        "vec_var_h",
        "vec_var_grad",
        "end_of_NoisyInputGaussianProcess",
    };

    // read data
    std::string token;
    int token_idx = 0;
    while (s.good()) {
        s >> token;
        if (token.compare(0, 1, "#") == 0) {
            skip_line();  // comment line, skip forward until end of line
            continue;
        }
        // non-comment line
        if (token != tokens[token_idx]) {
            ERL_WARN("Expected token {}, got {}.", tokens[token_idx], token);  // check token
            return false;
        }
        // reading state machine
        switch (token_idx) {
            case 0: {  // setting
                skip_line();
                if (!m_setting_->Read(s)) {
                    ERL_WARN("Failed to read setting.");
                    return false;
                }
                break;
            }
            case 1: {  // x_dim
                s >> m_x_dim_;
                break;
            }
            case 2: {  // num_train_samples
                s >> m_num_train_samples_;
                break;
            }
            case 3: {  // num_train_samples_with_grad
                s >> m_num_train_samples_with_grad_;
                break;
            }
            case 4: {  // trained
                s >> m_trained_;
                break;
            }
            case 5: {  // trained_once
                s >> m_trained_once_;
                break;
            }
            case 6: {  // ktrain_updated
                s >> m_k_train_updated_;
                break;
            }
            case 7: {  // ktrain_rows
                s >> m_k_train_rows_;
                break;
            }
            case 8: {  // ktrain_cols
                s >> m_k_train_cols_;
                break;
            }
            case 9: {  // kernel
                bool has_kernel;
                s >> has_kernel;
                if (has_kernel) {
                    skip_line();  // skip the line to read the binary data section
                    m_kernel_ = Covariance::CreateCovariance(m_setting_->kernel_type, m_setting_->kernel);
                    if (!m_kernel_->Read(s)) {
                        ERL_WARN("Failed to read kernel.");
                        return false;
                    }
                    const auto rank_reduced_kernel = std::dynamic_pointer_cast<ReducedRankCovariance>(m_kernel_);
                    m_reduced_rank_kernel_ = rank_reduced_kernel != nullptr;
                    if (m_reduced_rank_kernel_) { rank_reduced_kernel->BuildSpectralDensities(); }
                }
                break;
            }
            case 10: {  // three_over_scale_square
                skip_line();
                s.read(reinterpret_cast<char *>(&m_three_over_scale_square_), sizeof(Dtype));
                break;
            }
            case 11: {  // mat_x_train
                skip_line();
                if (!common::LoadEigenMatrixFromBinaryStream(s, m_mat_x_train_)) {
                    ERL_WARN("Failed to read mat_x_train.");
                    return false;
                }
                break;
            }
            case 12: {  // vec_y_train
                skip_line();
                if (!common::LoadEigenMatrixFromBinaryStream(s, m_vec_y_train_)) {
                    ERL_WARN("Failed to read vec_y_train.");
                    return false;
                }
                break;
            }
            case 13: {  // mat_grad_train
                skip_line();
                if (!common::LoadEigenMatrixFromBinaryStream(s, m_mat_grad_train_)) {
                    ERL_WARN("Failed to read mat_grad_train.");
                    return false;
                }
                break;
            }
            case 14: {  // mat_k_train
                skip_line();
                if (!common::LoadEigenMatrixFromBinaryStream(s, m_mat_k_train_)) {
                    ERL_WARN("Failed to read mat_k_train.");
                    return false;
                }
                break;
            }
            case 15: {  // mat_l
                skip_line();
                if (!common::LoadEigenMatrixFromBinaryStream(s, m_mat_l_)) {
                    ERL_WARN("Failed to read mat_l.");
                    return false;
                }
                break;
            }
            case 16: {  // vec_grad_flag
                skip_line();
                if (!common::LoadEigenMatrixFromBinaryStream(s, m_vec_grad_flag_)) {
                    ERL_WARN("Failed to read vec_grad_flag.");
                    return false;
                }
                break;
            }
            case 17: {  // vec_alpha
                skip_line();
                if (!common::LoadEigenMatrixFromBinaryStream(s, m_vec_alpha_)) {
                    ERL_WARN("Failed to read vec_alpha.");
                    return false;
                }
                break;
            }
            case 18: {  // vec_var_x
                skip_line();
                if (!common::LoadEigenMatrixFromBinaryStream(s, m_vec_var_x_)) {
                    ERL_WARN("Failed to read vec_var_x.");
                    return false;
                }
                break;
            }
            case 19: {  // vec_var_h
                skip_line();
                if (!common::LoadEigenMatrixFromBinaryStream(s, m_vec_var_h_)) {
                    ERL_WARN("Failed to read vec_var_h.");
                    return false;
                }
                break;
            }
            case 20: {  // vec_var_grad
                skip_line();
                if (!common::LoadEigenMatrixFromBinaryStream(s, m_vec_var_grad_)) {
                    ERL_WARN("Failed to read vec_var_grad.");
                    return false;
                }
                break;
            }
            case 21: {  // end_of_NoisyInputGaussianProcess
                skip_line();
                return true;
            }
            default: {  // should not reach here
                ERL_FATAL("Internal error, should not reach here.");
            }
        }
        ++token_idx;
    }
    ERL_WARN("Failed to read NoisyInputGaussianProcess. Truncated file?");
    return false;  // should not reach here
}

template<typename Dtype>
bool
NoisyInputGaussianProcess<Dtype>::AllocateMemory(const long max_num_samples, const long x_dim) {
    if (max_num_samples <= 0 || x_dim <= 0) { return false; }  // invalid input
    if (m_setting_->max_num_samples > 0 && max_num_samples > m_setting_->max_num_samples) { return false; }
    if (m_setting_->kernel->x_dim > 0 && x_dim != m_setting_->kernel->x_dim) { return false; }
    InitKernel();

    if (m_setting_->no_gradient_observation) {  // y does not contain gradient information
        const auto [rows, cols] = m_kernel_->GetMinimumKtrainSize(max_num_samples, 0, x_dim);
        if (m_mat_k_train_.rows() < rows || m_mat_k_train_.cols() < cols) { m_mat_k_train_.resize(rows, cols); }
        if (m_mat_x_train_.rows() < x_dim || m_mat_x_train_.cols() < max_num_samples) { m_mat_x_train_.resize(x_dim, max_num_samples); }
        if (m_vec_y_train_.size() < max_num_samples) { m_vec_y_train_.resize(max_num_samples); }
        if (m_mat_l_.rows() < rows || m_mat_l_.cols() < cols) { m_mat_l_.resize(rows, cols); }
        if (const long alpha_size = std::max(max_num_samples, cols); m_vec_alpha_.size() < alpha_size) { m_vec_alpha_.resize(alpha_size); }
        if (m_vec_grad_flag_.size() < max_num_samples) { m_vec_grad_flag_.resize(max_num_samples); }
        if (m_vec_var_x_.size() < max_num_samples) { m_vec_var_x_.resize(max_num_samples); }
        if (m_vec_var_h_.size() < max_num_samples) { m_vec_var_h_.resize(max_num_samples); }
        // m_mat_grad_train_, m_vec_var_grad_ are not used
        // to save memory; they are not allocated
    } else {
        const auto [rows, cols] = m_kernel_->GetMinimumKtrainSize(max_num_samples, max_num_samples, x_dim);
        if (m_mat_k_train_.rows() < rows || m_mat_k_train_.cols() < cols) { m_mat_k_train_.resize(rows, cols); }
        if (m_mat_x_train_.rows() < x_dim || m_mat_x_train_.cols() < max_num_samples) { m_mat_x_train_.resize(x_dim, max_num_samples); }
        if (m_vec_y_train_.size() < max_num_samples) { m_vec_y_train_.resize(max_num_samples); }
        if (m_mat_grad_train_.rows() < x_dim || m_mat_grad_train_.cols() < max_num_samples) { m_mat_grad_train_.resize(x_dim, max_num_samples); }
        if (m_mat_l_.rows() < rows || m_mat_l_.cols() < cols) { m_mat_l_.resize(rows, cols); }
        if (const long alpha_size = std::max(max_num_samples * (x_dim + 1), cols); m_vec_alpha_.size() < alpha_size) { m_vec_alpha_.resize(alpha_size); }
        if (m_vec_grad_flag_.size() < max_num_samples) { m_vec_grad_flag_.resize(max_num_samples); }
        if (m_vec_var_x_.size() < max_num_samples) { m_vec_var_x_.resize(max_num_samples); }
        if (m_vec_var_h_.size() < max_num_samples) { m_vec_var_h_.resize(max_num_samples); }
        if (m_vec_var_grad_.size() < max_num_samples) { m_vec_var_grad_.resize(max_num_samples); }
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
    const MatrixX &ktest,
    const long dim,
    const long n_test,
    const long predict_gradient,
    Eigen::Ref<MatrixX> mat_f_out) const {
    // compute value prediction
    /// ktest.T * m_vec_alpha_ = [h(x1),...,h(xn),dh(x1)/dx_1,...,dh(xn)/dx_1,...,dh(x1)/dx_dim,...,dh(xn)/dx_dim]
    auto vec_alpha = m_vec_alpha_.head(m_num_train_samples_);
    if (predict_gradient) {
        for (long i = 0; i < n_test; ++i) {
            Dtype *f = mat_f_out.col(i).data();
            f[0] = ktest.col(i).dot(vec_alpha);                                                                      // h(x)
            for (long j = 1, jj = i + n_test; j <= dim; ++j, jj += n_test) { f[j] = ktest.col(jj).dot(vec_alpha); }  // dh(x)/dx_j
        }
        return;
    }
    for (long i = 0; i < n_test; ++i) { mat_f_out(0, i) = ktest.col(i).dot(vec_alpha); }  // h(x)
}

template<typename Dtype>
void
NoisyInputGaussianProcess<Dtype>::ComputeCovPrediction(
    const MatrixX &ktest,
    const long dim,
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
    if (compute_var) {
        ERL_DEBUG_ASSERT(
            mat_var_out.rows() >= (predict_gradient ? 1 + dim : 1),
            "mat_var_out.rows() = {}, it should be >= {} for variance.",
            mat_var_out.rows(),
            predict_gradient ? 1 + dim : 1);
        ERL_DEBUG_ASSERT(mat_var_out.cols() >= n_test, "mat_var_out.cols() = {}, not enough for {} test queries.", mat_var_out.cols(), n_test);
    }
    if (!compute_cov) {  // compute variance only
        // column-wise square sum of ktest = var([h(x1),...,h(xn),dh(x1)/dx_1,...,dh(xn)/dx_1,...,dh(x1)/dx_dim,...,dh(xn)/dx_dim])
        if (m_reduced_rank_kernel_) {
            for (long i = 0; i < n_test; ++i) {
                Dtype *var = mat_var_out.col(i).data();
                var[0] = ktest.col(i).squaredNorm();  // variance of h(x)
                if (predict_gradient) {               // variance of dh(x)/dx_j
                    for (long j = 1, jj = i + n_test; j <= dim; ++j, jj += n_test) { var[j] = ktest.col(jj).squaredNorm(); }
                }
            }
        } else {
            const Dtype alpha = m_setting_->kernel->alpha;
            for (long i = 0; i < n_test; ++i) {
                Dtype *var = mat_var_out.col(i).data();
                var[0] = alpha - ktest.col(i).squaredNorm();  // variance of h(x)
                if (predict_gradient) {                       // variance of dh(x)/dx_j
                    for (long j = 1, jj = i + n_test; j <= dim; ++j, jj += n_test) { var[j] = m_three_over_scale_square_ - ktest.col(jj).squaredNorm(); }
                }
            }
        }
    } else if (predict_gradient) {  // compute covariance, but only when predict_gradient is true
        ERL_DEBUG_ASSERT(
            mat_cov_out.rows() >= (dim + 1) * dim / 2,
            "mat_cov_out.rows() = {}, it should be >= {} for covariance.",
            mat_cov_out.rows(),
            (dim + 1) * dim / 2);
        ERL_DEBUG_ASSERT(mat_cov_out.cols() >= n_test, "mat_cov_out.cols() = {}, not enough for {} test queries.", mat_cov_out.cols(), n_test);
        // each column of mat_cov_out is the lower triangular part of the covariance matrix of the corresponding test query
        if (m_reduced_rank_kernel_) {
            for (long i = 0; i < n_test; ++i) {
                Dtype *var = nullptr;
                if (compute_var) {
                    var = mat_var_out.col(i).data();
                    var[0] = ktest.col(i).squaredNorm();  // var(h(x))
                }
                Dtype *cov = mat_cov_out.col(i).data();
                long index = 0;
                for (long j = 1, jj = i + n_test; j <= dim; ++j, jj += n_test) {
                    const auto &col_jj = ktest.col(jj);
                    cov[index++] = col_jj.dot(ktest.col(i));                                                                   // cov(dh(x)/dx_j, h(x))
                    for (long k = 1, kk = i + n_test; k < j; ++k, kk += n_test) { cov[index++] = col_jj.dot(ktest.col(kk)); }  // cov(dh(x)/dx_j, dh(x)/dx_k)
                    if (var != nullptr) { var[j] = col_jj.squaredNorm(); }                                                     // var(dh(x)/dx_j)
                }
            }
        } else {
            const Dtype alpha = m_setting_->kernel->alpha;
            for (long i = 0; i < n_test; ++i) {
                Dtype *var = nullptr;
                if (compute_var) {
                    var = mat_var_out.col(i).data();
                    var[0] = alpha - ktest.col(i).squaredNorm();  // var(h(x))
                }
                Dtype *cov = mat_cov_out.col(i).data();
                long index = 0;
                for (long j = 1, jj = i + n_test; j <= dim; ++j, jj += n_test) {
                    const auto &col_jj = ktest.col(jj);
                    cov[index++] = -col_jj.dot(ktest.col(i));                                                                   // cov(dh(x)/dx_j, h(x))
                    for (long k = 1, kk = i + n_test; k < j; ++k, kk += n_test) { cov[index++] = -col_jj.dot(ktest.col(kk)); }  // cov(dh(x)/dx_j, dh(x)/dx_k)
                    if (var != nullptr) { var[j] = m_three_over_scale_square_ - col_jj.squaredNorm(); }                         // var(dh(x)/dx_j)
                }
            }
        }
    }
}

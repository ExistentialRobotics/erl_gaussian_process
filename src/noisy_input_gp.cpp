#include "erl_gaussian_process/noisy_input_gp.hpp"

namespace erl::gaussian_process {
    void
    NoisyInputGaussianProcess::Reset(const long max_num_samples, const long x_dim) {
        ERL_ASSERTM(max_num_samples > 0, "max_num_samples should be > 0.");
        ERL_ASSERTM(x_dim > 0, "x_dim should be > 0.");
        if (m_setting_->max_num_samples > 0 && m_setting_->kernel->x_dim > 0) {  // memory already allocated
            ERL_ASSERTM(m_setting_->max_num_samples >= max_num_samples, "max_num_samples should be <= {}.", m_setting_->max_num_samples);
        } else {
            if (m_setting_->kernel->x_dim > 0) { ERL_ASSERTM(m_setting_->kernel->x_dim == x_dim, "x_dim should be {}.", m_setting_->kernel->x_dim); }
            ERL_ASSERTM(AllocateMemory(max_num_samples, x_dim), "Failed to allocate memory.");
        }
        m_trained_ = false;
        m_kernel_ = covariance::Covariance::CreateCovariance(m_setting_->kernel_type, m_setting_->kernel);
        m_three_over_scale_square_ = 3. * m_setting_->kernel->alpha / (m_setting_->kernel->scale * m_setting_->kernel->scale);
        m_num_train_samples_ = 0;
        m_num_train_samples_with_grad_ = 0;
        m_x_dim_ = x_dim;
    }

    std::size_t
    NoisyInputGaussianProcess::GetMemoryUsage() const {
        std::size_t memory_usage = sizeof(NoisyInputGaussianProcess);
        if (m_setting_ != nullptr) { memory_usage += sizeof(Setting); }
        if (m_kernel_ != nullptr) { memory_usage += m_kernel_->GetMemoryUsage(); }
        memory_usage += m_mat_x_train_.size() * sizeof(double);
        memory_usage += m_vec_y_train_.size() * sizeof(double);
        memory_usage += m_mat_grad_train_.size() * sizeof(double);
        memory_usage += m_mat_k_train_.size() * sizeof(double);
        memory_usage += m_mat_l_.size() * sizeof(double);
        memory_usage += m_vec_grad_flag_.size() * sizeof(long);
        memory_usage += m_vec_alpha_.size() * sizeof(double);
        memory_usage += m_vec_var_x_.size() * sizeof(double);
        memory_usage += m_vec_var_h_.size() * sizeof(double);
        memory_usage += m_vec_var_grad_.size() * sizeof(double);
        return memory_usage;
    }

    void
    NoisyInputGaussianProcess::Train(const long num_train_samples) {

        if (m_trained_) {
            ERL_WARN("The model has been trained. Please reset the model before training.");
            return;
        }

        m_num_train_samples_ = num_train_samples;
        if (m_num_train_samples_ <= 0) {
            ERL_WARN("num_train_samples = {}, it should be > 0.", m_num_train_samples_);
            return;
        }

        InitializeVectorAlpha();  // initialize m_vec_alpha_

        // Compute kernel matrix
        const auto [rows, cols] = m_kernel_->ComputeKtrainWithGradient(
            m_mat_x_train_,
            m_num_train_samples_,
            m_vec_grad_flag_,
            m_vec_var_x_,
            m_vec_var_h_,
            m_vec_var_grad_,
            m_mat_k_train_);
        const auto mat_ktrain = m_mat_k_train_.topLeftCorner(rows, cols);  // square matrix
        auto &&mat_l = m_mat_l_.topLeftCorner(rows, cols);                 // square matrix, lower triangular
        const auto vec_alpha = m_vec_alpha_.head(rows);                    // h and gradient of h
        mat_l = mat_ktrain.llt().matrixL();
        mat_l.triangularView<Eigen::Lower>().solveInPlace(vec_alpha);
        mat_l.transpose().triangularView<Eigen::Upper>().solveInPlace(vec_alpha);
        m_trained_ = true;
    }

    void
    NoisyInputGaussianProcess::Test(
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x_test,
        Eigen::Ref<Eigen::MatrixXd> mat_f_out,
        Eigen::Ref<Eigen::MatrixXd> mat_var_out,
        Eigen::Ref<Eigen::MatrixXd> mat_cov_out) const {

        if (!m_trained_) {
            ERL_WARN("The model has not been trained.");
            return;
        }

        const long dim = mat_x_test.rows();
        const long n = mat_x_test.cols();
        if (n == 0) { return; }

        // compute mean and gradient of the test queries
        ERL_ASSERTM(mat_f_out.rows() >= dim + 1, "mat_f_out.rows() = {}, it should be >= Dim + 1 = {}.", mat_f_out.rows(), dim + 1);
        ERL_ASSERTM(mat_f_out.cols() >= n, "mat_f_out.cols() = {}, not enough for {} test queries.", mat_f_out.cols(), n);

        const auto [ktest_rows, ktest_cols] = covariance::Covariance::GetMinimumKtestSize(m_num_train_samples_, m_num_train_samples_with_grad_, dim, n);
        Eigen::MatrixXd ktest(ktest_rows, ktest_cols);                                // (dim of train samples, dim of test queries)
        const auto [output_rows, output_cols] = m_kernel_->ComputeKtestWithGradient(  //
            m_mat_x_train_,
            m_num_train_samples_,
            m_vec_grad_flag_,
            mat_x_test,
            n,
            ktest);
        ERL_DEBUG_ASSERT(
            output_rows == ktest_rows && output_cols == ktest_cols,
            "output_size = ({}, {}), it should be ({}, {}).",
            output_rows,
            output_cols,
            ktest_rows,
            ktest_cols);

        // compute value prediction
        /// ktest.T * m_vec_alpha_ = [h(x1),...,h(xn),dh(x1)/dx_1,...,dh(xn)/dx_1,...,dh(x1)/dx_dim,...,dh(xn)/dx_dim]
        auto vec_alpha = m_vec_alpha_.head(output_rows);
        for (long i = 0; i < n; ++i) {
            mat_f_out(0, i) = ktest.col(i).dot(vec_alpha);                                                            // h(x)
            for (long j = 1, jj = i + n; j <= dim; ++j, jj += n) { mat_f_out(j, i) = ktest.col(jj).dot(vec_alpha); }  // dh(x)/dx_j
        }
        if (mat_var_out.size() == 0) { return; }  // only compute mean

        // compute (co)variance of the test queries
        m_mat_l_.topLeftCorner(output_rows, output_rows).triangularView<Eigen::Lower>().solveInPlace(ktest);
        ERL_ASSERTM(mat_var_out.rows() >= dim + 1, "mat_var_out.rows() = {}, it should be >= {} for variance.", mat_var_out.rows(), dim + 1);
        ERL_ASSERTM(mat_var_out.cols() >= n, "mat_var_out.cols() = {}, not enough for {} test queries.", mat_var_out.cols(), n);
        if (mat_cov_out.size() == 0) {  // compute variance only
            // column-wise square sum of ktest = var([h(x1),...,h(xn),dh(x1)/dx_1,...,dh(xn)/dx_1,...,dh(x1)/dx_dim,...,dh(xn)/dx_dim])
            for (long i = 0; i < n; ++i) {
                mat_var_out(0, i) = m_setting_->kernel->alpha - ktest.col(i).squaredNorm();  // variance of h(x)
                for (long j = 1, jj = i + n; j <= dim; ++j, jj += n) {                       // variance of dh(x)/dx_j
                    mat_var_out(j, i) = m_three_over_scale_square_ - ktest.col(jj).squaredNorm();
                }
            }
        } else {  // compute covariance
            long min_n_rows = (dim + 1) * dim / 2;
            ERL_ASSERTM(mat_cov_out.rows() >= min_n_rows, "mat_cov_out.rows() = {}, it should be >= {} for covariance.", mat_cov_out.rows(), min_n_rows);
            ERL_ASSERTM(mat_cov_out.cols() >= n, "mat_cov_out.cols() = {}, not enough for {} test queries.", mat_cov_out.cols(), n);
            // each column of mat_cov_out is the lower triangular part of the covariance matrix of the corresponding test query
            for (long i = 0; i < n; ++i) {
                mat_var_out(0, i) = m_setting_->kernel->alpha - ktest.col(i).squaredNorm();  // var(h(x))
                long index = 0;
                for (long j = 1, jj = i + n; j <= dim; ++j, jj += n) {
                    const auto &col_jj = ktest.col(jj);
                    mat_cov_out(index++, i) = -col_jj.dot(ktest.col(i));                                                         // cov(dh(x)/dx_j, h(x))
                    for (long k = 1, kk = i + n; k < j; ++k, kk += n) { mat_cov_out(index++, i) = -col_jj.dot(ktest.col(kk)); }  // cov(dh(x)/dx_j, dh(x)/dx_k)
                    mat_var_out(j, i) = m_three_over_scale_square_ - col_jj.squaredNorm();                                       // var(dh(x)/dx_j)
                }
            }
        }
    }

    bool
    NoisyInputGaussianProcess::operator==(const NoisyInputGaussianProcess &other) const {
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
            std::memcmp(m_mat_x_train_.data(), other.m_mat_x_train_.data(), m_mat_x_train_.size() * sizeof(double)) != 0) {
            return false;
        }
        if (m_vec_y_train_.size() != other.m_vec_y_train_.size() ||
            std::memcmp(m_vec_y_train_.data(), other.m_vec_y_train_.data(), m_vec_y_train_.size() * sizeof(double)) != 0) {
            return false;
        }
        if (m_mat_grad_train_.rows() != other.m_mat_grad_train_.rows() || m_mat_grad_train_.cols() != other.m_mat_grad_train_.cols() ||
            std::memcmp(m_mat_grad_train_.data(), other.m_mat_grad_train_.data(), m_mat_grad_train_.size() * sizeof(double)) != 0) {
            return false;
        }
        if (m_mat_k_train_.rows() != other.m_mat_k_train_.rows() || m_mat_k_train_.cols() != other.m_mat_k_train_.cols() ||
            std::memcmp(m_mat_k_train_.data(), other.m_mat_k_train_.data(), m_mat_k_train_.size() * sizeof(double)) != 0) {
            return false;
        }
        if (m_mat_l_.rows() != other.m_mat_l_.rows() || m_mat_l_.cols() != other.m_mat_l_.cols() ||
            std::memcmp(m_mat_l_.data(), other.m_mat_l_.data(), m_mat_l_.size() * sizeof(double)) != 0) {
            return false;
        }
        if (m_vec_grad_flag_.size() != other.m_vec_grad_flag_.size() ||
            std::memcmp(m_vec_grad_flag_.data(), other.m_vec_grad_flag_.data(), m_vec_grad_flag_.size() * sizeof(long)) != 0) {
            return false;
        }
        if (m_vec_alpha_.size() != other.m_vec_alpha_.size() ||
            std::memcmp(m_vec_alpha_.data(), other.m_vec_alpha_.data(), m_vec_alpha_.size() * sizeof(double)) != 0) {
            return false;
        }
        if (m_vec_var_x_.size() != other.m_vec_var_x_.size() ||
            std::memcmp(m_vec_var_x_.data(), other.m_vec_var_x_.data(), m_vec_var_x_.size() * sizeof(double)) != 0) {
            return false;
        }
        if (m_vec_var_h_.size() != other.m_vec_var_h_.size() ||
            std::memcmp(m_vec_var_h_.data(), other.m_vec_var_h_.data(), m_vec_var_h_.size() * sizeof(double)) != 0) {
            return false;
        }
        if (m_vec_var_grad_.size() != other.m_vec_var_grad_.size() ||
            std::memcmp(m_vec_var_grad_.data(), other.m_vec_var_grad_.data(), m_vec_var_grad_.size() * sizeof(double)) != 0) {
            return false;
        }
        return true;
    }

    bool
    NoisyInputGaussianProcess::Write(const std::string &filename) const {
        ERL_INFO("Writing NoisyInputGaussianProcess to file: {}", filename);
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

    static const std::string kFileHeader = "# erl::gaussian_process::NoisyInputGaussianProcess";

    bool
    NoisyInputGaussianProcess::Write(std::ostream &s) const {
        s << kFileHeader << std::endl  //
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
          << "trained " << m_trained_ << std::endl;
        s << "three_over_scale_square" << std::endl;
        s.write(reinterpret_cast<const char *>(&m_three_over_scale_square_), sizeof(m_three_over_scale_square_));
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

    bool
    NoisyInputGaussianProcess::Read(const std::string &filename) {
        ERL_INFO("Reading NoisyInputGaussianProcess from file: {}", std::filesystem::absolute(filename));
        std::ifstream file(filename.c_str(), std::ios_base::in | std::ios_base::binary);
        if (!file.is_open()) {
            ERL_WARN("Failed to open file: {}", filename.c_str());
            return false;
        }

        const bool success = Read(file);
        file.close();
        return success;
    }

    bool
    NoisyInputGaussianProcess::Read(std::istream &s) {
        if (!s.good()) {
            ERL_WARN("Input stream is not ready for reading");
            return false;
        }

        // check if the first line is valid
        std::string line;
        std::getline(s, line);
        if (line.compare(0, kFileHeader.length(), kFileHeader) != 0) {  // check if the first line is valid
            ERL_WARN("Header does not start with \"{}\"", kFileHeader.c_str());
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
                case 5: {  // three_over_scale_square
                    skip_line();
                    s.read(reinterpret_cast<char *>(&m_three_over_scale_square_), sizeof(m_three_over_scale_square_));
                    break;
                }
                case 6: {  // mat_x_train
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_mat_x_train_)) {
                        ERL_WARN("Failed to read mat_x_train.");
                        return false;
                    }
                    break;
                }
                case 7: {  // vec_y_train
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_vec_y_train_)) {
                        ERL_WARN("Failed to read vec_y_train.");
                        return false;
                    }
                    break;
                }
                case 8: {  // mat_grad_train
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_mat_grad_train_)) {
                        ERL_WARN("Failed to read mat_grad_train.");
                        return false;
                    }
                    break;
                }
                case 9: {  // mat_k_train
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_mat_k_train_)) {
                        ERL_WARN("Failed to read mat_k_train.");
                        return false;
                    }
                    break;
                }
                case 10: {  // mat_l
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_mat_l_)) {
                        ERL_WARN("Failed to read mat_l.");
                        return false;
                    }
                    break;
                }
                case 11: {  // vec_grad_flag
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_vec_grad_flag_)) {
                        ERL_WARN("Failed to read vec_grad_flag.");
                        return false;
                    }
                    break;
                }
                case 12: {  // vec_alpha
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_vec_alpha_)) {
                        ERL_WARN("Failed to read vec_alpha.");
                        return false;
                    }
                    break;
                }
                case 13: {  // vec_var_x
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_vec_var_x_)) {
                        ERL_WARN("Failed to read vec_var_x.");
                        return false;
                    }
                    break;
                }
                case 14: {  // vec_var_h
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_vec_var_h_)) {
                        ERL_WARN("Failed to read vec_var_h.");
                        return false;
                    }
                    break;
                }
                case 15: {  // vec_var_grad
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_vec_var_grad_)) {
                        ERL_WARN("Failed to read vec_var_grad.");
                        return false;
                    }
                    break;
                }
                case 16: {  // end_of_NoisyInputGaussianProcess
                    skip_line();
                    m_kernel_ = covariance::Covariance::CreateCovariance(m_setting_->kernel_type, m_setting_->kernel);
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

    bool
    NoisyInputGaussianProcess::AllocateMemory(const long max_num_samples, const long x_dim) {
        if (max_num_samples <= 0 || x_dim <= 0) { return false; }  // invalid input
        if (m_setting_->max_num_samples > 0 && max_num_samples > m_setting_->max_num_samples) { return false; }
        if (m_setting_->kernel->x_dim > 0) {
            ERL_ASSERTM(x_dim == m_setting_->kernel->x_dim, "x_dim {} does not match kernel->x_dim {}.", x_dim, m_setting_->kernel->x_dim);
        }
        const auto [rows, cols] = covariance::Covariance::GetMinimumKtrainSize(max_num_samples, max_num_samples, x_dim);
        if (m_mat_k_train_.rows() < rows || m_mat_k_train_.cols() < cols) { m_mat_k_train_.resize(rows, cols); }
        if (m_mat_x_train_.rows() < x_dim || m_mat_x_train_.cols() < max_num_samples) { m_mat_x_train_.resize(x_dim, max_num_samples); }
        if (m_vec_y_train_.size() < max_num_samples) { m_vec_y_train_.resize(max_num_samples); }
        if (m_mat_grad_train_.rows() < x_dim || m_mat_grad_train_.cols() < max_num_samples) { m_mat_grad_train_.resize(x_dim, max_num_samples); }
        if (m_mat_l_.rows() < rows || m_mat_l_.cols() < cols) { m_mat_l_.resize(rows, cols); }
        if (m_vec_alpha_.size() < max_num_samples * (x_dim + 1)) { m_vec_alpha_.resize(max_num_samples * (x_dim + 1)); }
        if (m_vec_grad_flag_.size() < max_num_samples) { m_vec_grad_flag_.resize(max_num_samples); }
        if (m_vec_var_x_.size() < max_num_samples) { m_vec_var_x_.resize(max_num_samples); }
        if (m_vec_var_h_.size() < max_num_samples) { m_vec_var_h_.resize(max_num_samples); }
        if (m_vec_var_grad_.size() < max_num_samples) { m_vec_var_grad_.resize(max_num_samples); }
        return true;
    }

    void
    NoisyInputGaussianProcess::InitializeVectorAlpha() {
        ERL_DEBUG_ASSERT(
            m_vec_alpha_.size() >= m_num_train_samples_ * (m_x_dim_ + 1),
            "m_vec_alpha_ should have size >= {}.",
            m_num_train_samples_ * (m_x_dim_ + 1));

        m_num_train_samples_with_grad_ = 0;
        for (long i = 0; i < m_num_train_samples_; ++i) {
            m_vec_alpha_[i] = m_vec_y_train_[i];  // h(x_i)
            if (m_vec_grad_flag_[i]) { ++m_num_train_samples_with_grad_; }
        }
        for (long i = 0, j = m_num_train_samples_; i < m_num_train_samples_; ++i) {
            if (!m_vec_grad_flag_[i]) { continue; }
            for (long k = 0, l = j++; k < m_x_dim_; ++k, l += m_num_train_samples_with_grad_) { m_vec_alpha_[l] = m_mat_grad_train_(k, i); }
        }
    }

}  // namespace erl::gaussian_process

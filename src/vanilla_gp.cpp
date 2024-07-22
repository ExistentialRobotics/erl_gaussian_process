#include "erl_gaussian_process/vanilla_gp.hpp"

namespace erl::gaussian_process {
    void
    VanillaGaussianProcess::Reset(const long max_num_samples, const long x_dim) {
        ERL_ASSERTM(x_dim > 0, "x_dim should be > 0.");
        if (const long &max_num_samples_setting = m_setting_->max_num_samples, &x_dim_setting = m_setting_->kernel->x_dim;
            max_num_samples_setting > 0 && x_dim_setting > 0) {  // memory already allocated
            ERL_ASSERTM(max_num_samples_setting >= max_num_samples, "max_num_samples should be <= {}.", max_num_samples_setting);
        } else {
            ERL_ASSERTM(x_dim_setting <= 0 || x_dim_setting == x_dim, "x_dim should be {}.", x_dim_setting);
            ERL_ASSERTM(AllocateMemory(max_num_samples, x_dim), "Failed to allocate memory.");
        }
        m_trained_ = false;
        if (m_kernel_ == nullptr) { m_kernel_ = covariance::Covariance::CreateCovariance(m_setting_->kernel_type, m_setting_->kernel); }
        m_num_train_samples_ = 0;
        m_x_dim_ = x_dim;
    }

    bool
    VanillaGaussianProcess::Train(const long num_train_samples) {

        if (m_trained_) {
            ERL_WARN("The model has been trained. Please reset the model before training.");
            return false;
        }

        m_num_train_samples_ = num_train_samples;
        if (m_num_train_samples_ <= 0) {
            ERL_WARN("num_train_samples = {}, it should be > 0.", m_num_train_samples_);
            return false;
        }

        // Compute kernel matrix
        // const auto mat_x_train = m_mat_x_train_.topLeftCorner(m_x_dim_, m_num_train_samples_);
        // const auto vec_var_h = m_vec_var_h_.head(m_num_train_samples_);
        const auto [rows, cols] = m_kernel_->ComputeKtrain(m_mat_x_train_, m_vec_var_h_, m_num_train_samples_, m_mat_k_train_);
        const auto mat_ktrain = m_mat_k_train_.topLeftCorner(rows, cols);
        auto &&mat_l = m_mat_l_.topLeftCorner(rows, cols);
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
        return true;
    }

    bool
    VanillaGaussianProcess::Test(
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x_test,
        Eigen::Ref<Eigen::VectorXd> vec_f_out,
        Eigen::Ref<Eigen::VectorXd> vec_var_out) const {

        if (!m_trained_ || m_num_train_samples_ <= 0) { return false; }

        const long n = mat_x_test.cols();
        if (n == 0) { return false; }
        ERL_ASSERTM(mat_x_test.rows() == m_x_dim_, "mat_x_test.rows() = {}, it should be {}.", mat_x_test.rows(), m_x_dim_);
        ERL_ASSERTM(vec_f_out.size() >= n, "vec_f_out size = {}, it should be >= {}.", vec_f_out.size(), n);
        const auto [ktest_rows, ktest_cols] = covariance::Covariance::GetMinimumKtestSize(m_num_train_samples_, 0, 0, n);
        Eigen::MatrixXd ktest(ktest_rows, ktest_cols);
        const auto [output_rows, output_cols] = m_kernel_->ComputeKtest(m_mat_x_train_, m_num_train_samples_, mat_x_test, n, ktest);
        ERL_DEBUG_ASSERT(
            (output_rows == ktest_rows && output_cols == ktest_cols),
            "output_size = ({}, {}), it should be ({}, {}).",
            output_rows,
            output_cols,
            ktest_rows,
            ktest_cols);

        // xt is one column of mat_x_test
        // expectation of vec_f_out = ktest(xt, X) @ (ktest(X, X) + sigma * I).m_inv_() @ y
        const auto vec_alpha = m_vec_alpha_.head(output_rows);
        if (m_setting_->auto_normalize) {
            for (long i = 0; i < output_rows; ++i) { vec_f_out[i] = ktest.col(i).dot(vec_alpha) * m_std_ + m_mean_; }
        } else {
            for (long i = 0; i < output_cols; ++i) { vec_f_out[i] = ktest.col(i).dot(vec_alpha); }
        }
        if (vec_var_out.size() == 0) { return true; }  // only compute mean

        // variance of vec_f_out = ktest(xt, xt) - ktest(xt, X) @ (ktest(X, X) + sigma * I).m_inv_() @ ktest(X, xt)
        //                       = ktest(xt, xt) - ktest(xt, X) @ (m_l_ @ m_l_.T).m_inv_() @ ktest(X, xt)
        ERL_ASSERTM(vec_var_out.size() >= n, "vec_var_out size = {}, it should be >= {}.", vec_var_out.size(), n);
        m_mat_l_.topLeftCorner(output_rows, output_rows).triangularView<Eigen::Lower>().solveInPlace(ktest);
        for (long i = 0; i < ktest_cols; ++i) { vec_var_out[i] = m_setting_->kernel->alpha - ktest.col(i).squaredNorm(); }
        return true;
    }

    bool
    VanillaGaussianProcess::operator==(const VanillaGaussianProcess &other) const {
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr && (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) { return false; }
        if (m_x_dim_ != other.m_x_dim_) { return false; }
        if (m_num_train_samples_ != other.m_num_train_samples_) { return false; }
        if (m_trained_ != other.m_trained_) { return false; }
        if (m_num_train_samples_ == 0) { return true; }
        if (m_mean_ != other.m_mean_) { return false; }
        if (m_std_ != other.m_std_) { return false; }
        if (m_mat_k_train_.rows() != other.m_mat_k_train_.rows() || m_mat_k_train_.cols() != other.m_mat_k_train_.cols() ||
            std::memcmp(m_mat_k_train_.data(), other.m_mat_k_train_.data(), m_mat_k_train_.size() * sizeof(double)) != 0) {
            return false;
        }
        if (m_mat_x_train_.rows() != other.m_mat_x_train_.rows() || m_mat_x_train_.cols() != other.m_mat_x_train_.cols() ||
            std::memcmp(m_mat_x_train_.data(), other.m_mat_x_train_.data(), m_mat_x_train_.size() * sizeof(double)) != 0) {
            return false;
        }
        if (m_mat_l_.rows() != other.m_mat_l_.rows() || m_mat_l_.cols() != other.m_mat_l_.cols() ||
            std::memcmp(m_mat_l_.data(), other.m_mat_l_.data(), m_mat_l_.size() * sizeof(double)) != 0) {
            return false;
        }
        if (m_vec_alpha_.size() != other.m_vec_alpha_.size() ||
            std::memcmp(m_vec_alpha_.data(), other.m_vec_alpha_.data(), m_vec_alpha_.size() * sizeof(double)) != 0) {
            return false;
        }
        if (m_vec_var_h_.size() != other.m_vec_var_h_.size() ||
            std::memcmp(m_vec_var_h_.data(), other.m_vec_var_h_.data(), m_vec_var_h_.size() * sizeof(double)) != 0) {
            return false;
        }
        return true;
    }

    bool
    VanillaGaussianProcess::Write(const std::string &filename) const {
        ERL_INFO("Writing VanillaGaussianProcess to file: {}", filename);
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

    static const std::string kFileHeader = "# erl::gaussian_process::VanillaGaussianProcess";

    bool
    VanillaGaussianProcess::Write(std::ostream &s) const {
        s << kFileHeader << std::endl  //
          << "# (feel free to add / change comments, but leave the first line as it is!)" << std::endl
          << "setting" << std::endl;
        // write setting
        if (!m_setting_->Write(s)) {
            ERL_WARN("Failed to write setting.");
            return false;
        }
        // write data
        s << "x_dim " << m_x_dim_ << std::endl  //
          << "num_train_samples " << m_num_train_samples_ << std::endl
          << "trained " << m_trained_ << std::endl;
        s << "mean" << std::endl;
        s.write(reinterpret_cast<const char *>(&m_mean_), sizeof(double));
        s << "std" << std::endl;
        s.write(reinterpret_cast<const char *>(&m_std_), sizeof(double));
        s << "mat_k_train" << std::endl;
        if (!common::SaveEigenMatrixToBinaryStream(s, m_mat_k_train_)) {
            ERL_WARN("Failed to write mat_k_train.");
            return false;
        }
        s << "mat_x_train" << std::endl;
        if (!common::SaveEigenMatrixToBinaryStream(s, m_mat_x_train_)) {
            ERL_WARN("Failed to write mat_x_train.");
            return false;
        }
        s << "mat_l" << std::endl;
        if (!common::SaveEigenMatrixToBinaryStream(s, m_mat_l_)) {
            ERL_WARN("Failed to write mat_l.");
            return false;
        }
        s << "vec_alpha" << std::endl;
        if (!common::SaveEigenMatrixToBinaryStream(s, m_vec_alpha_)) {
            ERL_WARN("Failed to write vec_alpha.");
            return false;
        }
        s << "vec_var_h" << std::endl;
        if (!common::SaveEigenMatrixToBinaryStream(s, m_vec_var_h_)) {
            ERL_WARN("Failed to write vec_var_h.");
            return false;
        }
        s << "end_of_VanillaGaussianProcess" << std::endl;
        return s.good();
    }

    bool
    VanillaGaussianProcess::Read(const std::string &filename) {
        ERL_INFO("Reading VanillaGaussianProcess from file: {}", std::filesystem::absolute(filename));
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
    VanillaGaussianProcess::Read(std::istream &s) {
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
            "trained",
            "mean",
            "std",
            "mat_k_train",
            "mat_x_train",
            "mat_l",
            "vec_alpha",
            "vec_var_h",
            "end_of_VanillaGaussianProcess",
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
                case 0: {         // setting
                    skip_line();  // skip the line to read the bindary data section
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
                case 3: {  // trained
                    s >> m_trained_;
                    break;
                }
                case 4: {         // mean
                    skip_line();  // skip the line to read the bindary data section
                    s.read(reinterpret_cast<char *>(&m_mean_), sizeof(double));
                    break;
                }
                case 5: {         // std
                    skip_line();  // skip the line to read the bindary data section
                    s.read(reinterpret_cast<char *>(&m_std_), sizeof(double));
                    break;
                }
                case 6: {         // mat_k_train
                    skip_line();  // skip the line to read the bindary data section
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_mat_k_train_)) {
                        ERL_WARN("Failed to read mat_k_train.");
                        return false;
                    }
                    break;
                }
                case 7: {         // mat_x_train
                    skip_line();  // skip the line to read the bindary data section
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_mat_x_train_)) {
                        ERL_WARN("Failed to read mat_x_train.");
                        return false;
                    }
                    break;
                }
                case 8: {         // mat_l
                    skip_line();  // skip the line to read the bindary data section
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_mat_l_)) {
                        ERL_WARN("Failed to read mat_l.");
                        return false;
                    }
                    break;
                }
                case 9: {         // vec_alpha
                    skip_line();  // skip the line to read the bindary data section
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_vec_alpha_)) {
                        ERL_WARN("Failed to read vec_alpha.");
                        return false;
                    }
                    break;
                }
                case 10: {        // vec_var_h
                    skip_line();  // skip the line to read the bindary data section
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_vec_var_h_)) {
                        ERL_WARN("Failed to read vec_var_h.");
                        return false;
                    }
                    break;
                }
                case 11: {        // end_of_VanillaGaussianProcess
                    skip_line();  // skip forward until end of line
                    m_kernel_ = covariance::Covariance::CreateCovariance(m_setting_->kernel_type, m_setting_->kernel);
                    return true;
                }
                default: {  // should not reach here
                    ERL_FATAL("Internal error, should not reach here.");
                }
            }
            ++token_idx;
        }
        ERL_WARN("Failed to read VanillaGaussianProcess. Truncated file?");
        return false;  // should not reach here
    }

}  // namespace erl::gaussian_process

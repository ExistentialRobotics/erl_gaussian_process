#pragma once

#include <functional>
#include <memory>

#include "erl_covariance/covariance.hpp"

namespace erl::gaussian_process {
    // using namespace common;
    // using namespace covariance;

    /**
     * VanillaGaussianProcess implements the standard Gaussian Process
     */
    class VanillaGaussianProcess {

    public:
        // structure for holding the parameters
        struct Setting : public common::Yamlable<Setting> {
            std::shared_ptr<covariance::Covariance::Setting> kernel = []() -> std::shared_ptr<covariance::Covariance::Setting> {
                auto setting = std::make_shared<covariance::Covariance::Setting>();
                setting->type = covariance::Covariance::Type::kOrnsteinUhlenbeck;
                setting->x_dim = 2;
                setting->alpha = 1.;
                setting->scale = 0.5;
                setting->scale_mix = 1.;
                return setting;
            }();
            long max_num_samples = 256;
            bool auto_normalize = false;
        };

    protected:
        long m_x_dim_ = 0;                                            // dimension of x
        long m_num_train_samples_ = 0;                                // number of training samples
        double m_mean_ = 0.;                                          // mean of the training output samples
        double m_std_ = 0.;                                           // standard deviation of the training output samples
        bool m_trained_ = true;                                       // true if the GP is trained
        std::shared_ptr<Setting> m_setting_ = nullptr;                // setting
        std::shared_ptr<covariance::Covariance> m_kernel_ = nullptr;  // kernel
        Eigen::MatrixXd m_mat_k_train_ = {};                          // Ktrain, avoid reallocation
        Eigen::MatrixXd m_mat_x_train_ = {};                          // x1, ..., xn
        Eigen::MatrixXd m_mat_l_ = {};                                // lower triangular matrix of the Cholesky decomposition of Ktrain
        Eigen::VectorXd m_vec_alpha_ = {};                            // h(x1)..h(xn), dh(x1)/dx1_1 .. dh(xn)/dxn_1 .. dh(x1)/dx1_dim .. dh(xn)/dxn_dim
        Eigen::VectorXd m_vec_var_h_ = {};                            // variance of y1 ... yn

    public:
        VanillaGaussianProcess()
            : m_setting_(std::make_shared<Setting>()) {}

        explicit VanillaGaussianProcess(std::shared_ptr<Setting> setting)
            : m_setting_(std::move(setting)) {
            ERL_ASSERTM(m_setting_ != nullptr, "setting should not be nullptr.");
            ERL_ASSERTM(m_setting_->kernel != nullptr, "setting->kernel should not be nullptr.");
            m_trained_ = !(m_setting_->max_num_samples > 0 && m_setting_->kernel->x_dim > 0);  // if memory is allocated, the model is ready to be trained
            if (!m_trained_) { ERL_ASSERTM(AllocateMemory(m_setting_->max_num_samples, m_setting_->kernel->x_dim), "Failed to allocate memory."); }
        }

        [[nodiscard]] std::shared_ptr<Setting>
        GetSetting() const {
            return m_setting_;
        }

        [[nodiscard]] inline bool
        IsTrained() const {
            return m_trained_;
        }

        /**
         * @brief reset the model: update flags, kernel, and allocate memory if necessary, etc.
         * @param max_num_samples maximum number of training samples
         * @param x_dim dimension of training input samples
         */
        void
        Reset(long max_num_samples, long x_dim);

        [[nodiscard]] inline long
        GetNumTrainSamples() const {
            return m_num_train_samples_;
        }

        [[nodiscard]] inline Eigen::MatrixXd &
        GetTrainInputSamplesBuffer() {
            return m_mat_x_train_;
        }

        [[nodiscard]] inline Eigen::VectorXd &
        GetTrainOutputSamplesBuffer() {
            return m_vec_alpha_;
        }

        [[nodiscard]] inline Eigen::VectorXd &
        GetTrainOutputSamplesVarianceBuffer() {
            return m_vec_var_h_;
        }

        [[nodiscard]] inline Eigen::MatrixXd
        GetKtrain() const {
            return m_mat_k_train_;
        }

        [[nodiscard]] inline Eigen::MatrixXd
        GetCholeskyDecomposition() const {
            return m_mat_l_;
        }

        void
        Train(long num_train_samples);

        void
        Test(const Eigen::Ref<const Eigen::MatrixXd> &mat_x_test, Eigen::Ref<Eigen::VectorXd> vec_f_out, Eigen::Ref<Eigen::VectorXd> vec_var_out) const;

    protected:
        inline bool
        AllocateMemory(long max_num_samples, long x_dim) {
            if (m_setting_->max_num_samples > 0 && max_num_samples > m_setting_->max_num_samples) { return false; }
            if (m_setting_->kernel->x_dim > 0 && x_dim != m_setting_->kernel->x_dim) { return false; }
            std::pair<long, long> size = covariance::Covariance::GetMinimumKtrainSize(max_num_samples, 0, 0);
            if (m_mat_k_train_.rows() < size.first || m_mat_k_train_.cols() < size.second) { m_mat_k_train_.resize(size.first, size.second); }
            if (m_mat_x_train_.rows() < x_dim || m_mat_x_train_.cols() < max_num_samples) { m_mat_x_train_.resize(x_dim, max_num_samples); }
            if (m_mat_l_.rows() < size.first || m_mat_l_.cols() < size.second) { m_mat_l_.resize(size.first, size.second); }
            if (m_vec_alpha_.size() < max_num_samples) { m_vec_alpha_.resize(max_num_samples); }
            if (m_vec_var_h_.size() < max_num_samples) { m_vec_var_h_.resize(max_num_samples); }
            return true;
        }
    };
}  // namespace erl::gaussian_process

namespace YAML {
    template<>
    struct convert<erl::gaussian_process::VanillaGaussianProcess::Setting> {
        inline static Node
        encode(const erl::gaussian_process::VanillaGaussianProcess::Setting &setting) {
            Node node;
            node["kernel"] = *setting.kernel;
            node["auto_normalize"] = setting.auto_normalize;
            return node;
        }

        inline static bool
        decode(const Node &node, erl::gaussian_process::VanillaGaussianProcess::Setting &setting) {
            if (!node.IsMap()) { return false; }
            *setting.kernel = node["kernel"].as<erl::covariance::Covariance::Setting>();
            setting.auto_normalize = node["auto_normalize"].as<bool>();
            return true;
        }
    };

    inline Emitter &
    operator<<(Emitter &out, const erl::gaussian_process::VanillaGaussianProcess::Setting &setting) {
        out << BeginMap;
        out << Key << "kernel" << Value << *setting.kernel;
        out << Key << "auto_normalize" << Value << setting.auto_normalize;
        out << EndMap;
        return out;
    }
}  // namespace YAML

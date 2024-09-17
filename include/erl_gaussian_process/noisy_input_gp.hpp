#pragma once

#include "vanilla_gp.hpp"

#include "erl_covariance/covariance.hpp"

#include <utility>

namespace erl::gaussian_process {

    class NoisyInputGaussianProcess {

    public:
        struct Setting : public common::Yamlable<Setting> {
            std::string kernel_type = "erl::covariance::Matern32<2l>";
            std::string kernel_setting_type = "erl::covariance::Covariance::Setting";
            std::shared_ptr<covariance::Covariance::Setting> kernel = std::make_shared<covariance::Covariance::Setting>();
            long max_num_samples = -1;  // maximum number of training samples, -1 means no limit
            bool no_gradient_observation = false;
        };

        inline static const volatile bool kSettingRegistered = common::YamlableBase::Register<Setting>();

    protected:
        long m_x_dim_ = 0;                                            // dimension of x
        long m_num_train_samples_ = 0;                                // number of training samples
        long m_num_train_samples_with_grad_ = 0;                      // number of training samples with gradient
        bool m_trained_ = false;                                      // true if the GP is trained
        bool m_trained_once_ = false;                                 // true if the GP is trained at least once
        bool m_k_train_updated_ = false;                              // true if Ktrain is updated
        long m_k_train_rows_ = 0;                                     // number of rows of Ktrain
        long m_k_train_cols_ = 0;                                     // number of columns of Ktrain
        double m_three_over_scale_square_ = 0.;                       // for computing normal variance
        std::shared_ptr<Setting> m_setting_ = nullptr;                // setting
        std::shared_ptr<covariance::Covariance> m_kernel_ = nullptr;  // kernel
        bool m_reduced_rank_kernel_ = false;                          // whether the kernel is rank reduced or not
        Eigen::MatrixXd m_mat_x_train_ = {};                          // x1, ..., xn
        Eigen::VectorXd m_vec_y_train_ = {};                          // h(x1), ..., h(xn)
        Eigen::MatrixXd m_mat_grad_train_ = {};                       // dh(x_j)/dx_ij for index (i, j)
        Eigen::MatrixXd m_mat_k_train_ = {};                          // Ktrain, avoid reallocation
        Eigen::MatrixXd m_mat_l_ = {};                                // lower triangular matrix of the Cholesky decomposition of Ktrain
        Eigen::VectorXl m_vec_grad_flag_ = {};                        // true if the corresponding training sample has gradient
        Eigen::VectorXd m_vec_alpha_ = {};                            // h(x1)..h(xn), dh(x1)/dx1_1 .. dh(xn)/dxn_1 .. dh(x1)/dx1_dim .. dh(xn)/dxn_dim
        Eigen::VectorXd m_vec_var_x_ = {};                            // variance of x1 ... xn
        Eigen::VectorXd m_vec_var_h_ = {};                            // variance of h(x1)..h(xn)
        Eigen::VectorXd m_vec_var_grad_ = {};                         // variance of dh(x1)/dx1_1 .. dh(xn)/dxn_1 .. dh(x1)/dx1_dim .. dh(xn)/dxn_dim

    public:
        explicit NoisyInputGaussianProcess(std::shared_ptr<Setting> setting)
            : m_setting_(std::move(setting)) {
            ERL_ASSERTM(m_setting_ != nullptr, "setting should not be nullptr.");
            ERL_ASSERTM(m_setting_->kernel != nullptr, "setting->kernel should not be nullptr.");
            if (m_setting_->max_num_samples > 0 && m_setting_->kernel->x_dim > 0) {
                ERL_ASSERTM(AllocateMemory(m_setting_->max_num_samples, m_setting_->kernel->x_dim), "Failed to allocate memory.");
            }
        }

        NoisyInputGaussianProcess(const NoisyInputGaussianProcess &other);

        NoisyInputGaussianProcess(NoisyInputGaussianProcess &&other) = default;

        NoisyInputGaussianProcess &
        operator=(const NoisyInputGaussianProcess &other);

        NoisyInputGaussianProcess &
        operator=(NoisyInputGaussianProcess &&other) = default;

        virtual ~NoisyInputGaussianProcess() = default;

        template<typename T = Setting>
        [[nodiscard]] std::shared_ptr<T>
        GetSetting() const {
            return std::dynamic_pointer_cast<T>(m_setting_);
        }

        [[nodiscard]] bool
        IsTrained() const {
            return m_trained_;
        }

        [[nodiscard]] bool
        UsingReducedRankKernel() const {
            return m_reduced_rank_kernel_;
        }

        [[nodiscard]] Eigen::VectorXd
        GetKernelCoordOrigin() const;

        void
        SetKernelCoordOrigin(const Eigen::VectorXd &coord_origin) const;

        void
        Reset(long max_num_samples, long x_dim);

        [[nodiscard]] long
        GetNumTrainSamples() const {
            return m_num_train_samples_;
        }

        [[nodiscard]] long
        GetNumTrainSamplesWithGrad() const {
            return m_num_train_samples_with_grad_;
        }

        [[nodiscard]] std::shared_ptr<covariance::Covariance>
        GetKernel() const {
            return m_kernel_;
        }

        [[nodiscard]] Eigen::MatrixXd &
        GetTrainInputSamplesBuffer() {
            return m_mat_x_train_;
        }

        [[nodiscard]] Eigen::VectorXd &
        GetTrainOutputSamplesBuffer() {
            return m_vec_y_train_;
        }

        [[nodiscard]] Eigen::MatrixXd &
        GetTrainOutputGradientSamplesBuffer() {
            return m_mat_grad_train_;
        }

        [[nodiscard]] Eigen::VectorXl &
        GetTrainGradientFlagsBuffer() {
            return m_vec_grad_flag_;
        }

        [[nodiscard]] Eigen::VectorXd &
        GetTrainInputSamplesVarianceBuffer() {
            return m_vec_var_x_;
        }

        [[nodiscard]] Eigen::VectorXd &
        GetTrainOutputValueSamplesVarianceBuffer() {
            return m_vec_var_h_;
        }

        [[nodiscard]] Eigen::VectorXd &
        GetTrainOutputGradientSamplesVarianceBuffer() {
            return m_vec_var_grad_;
        }

        [[nodiscard]] Eigen::MatrixXd
        GetKtrain() {
            return m_mat_k_train_;
        }

        [[nodiscard]] Eigen::VectorXd
        GetAlpha() {
            return m_vec_alpha_;
        }

        [[nodiscard]] Eigen::MatrixXd
        GetCholeskyDecomposition() {
            return m_mat_l_;
        }

        [[nodiscard]] virtual std::size_t
        GetMemoryUsage() const;

        bool
        UpdateKtrain(long num_train_samples);

        virtual void
        Train(long num_train_samples);

        [[nodiscard]] virtual bool
        Test(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x_test,
            Eigen::Ref<Eigen::MatrixXd> mat_f_out,
            Eigen::Ref<Eigen::MatrixXd> mat_var_out,
            Eigen::Ref<Eigen::MatrixXd> mat_cov_out) const;

        [[nodiscard]] bool
        operator==(const NoisyInputGaussianProcess &other) const;

        [[nodiscard]] bool
        operator!=(const NoisyInputGaussianProcess &other) const {
            return !(*this == other);
        }

        [[nodiscard]] virtual bool
        Write(const std::string &filename) const;

        [[nodiscard]] virtual bool
        Write(std::ostream &s) const;

        [[nodiscard]] virtual bool
        Read(const std::string &filename);

        [[nodiscard]] virtual bool
        Read(std::istream &s);

    protected:
        bool
        AllocateMemory(long max_num_samples, long x_dim);

        void
        InitKernel();
    };
}  // namespace erl::gaussian_process

// ReSharper disable CppInconsistentNaming
template<>
struct YAML::convert<erl::gaussian_process::NoisyInputGaussianProcess::Setting> {
    static Node
    encode(const erl::gaussian_process::NoisyInputGaussianProcess::Setting &setting) {
        Node node;
        node["kernel_type"] = setting.kernel_type;
        node["kernel_setting_type"] = setting.kernel_setting_type;
        node["kernel"] = setting.kernel->AsYamlNode();
        node["max_num_samples"] = setting.max_num_samples;
        node["no_gradient_observation"] = setting.no_gradient_observation;
        return node;
    }

    static bool
    decode(const Node &node, erl::gaussian_process::NoisyInputGaussianProcess::Setting &setting) {
        if (!node.IsMap()) { return false; }
        setting.kernel_type = node["kernel_type"].as<std::string>();
        setting.kernel_setting_type = node["kernel_setting_type"].as<std::string>();
        setting.kernel = erl::common::YamlableBase::Create<erl::covariance::Covariance::Setting>(setting.kernel_setting_type);
        if (!setting.kernel->FromYamlNode(node["kernel"])) { return false; }
        setting.max_num_samples = node["max_num_samples"].as<long>();
        setting.no_gradient_observation = node["no_gradient_observation"].as<bool>();
        return true;
    }
};  // namespace YAML

// ReSharper restore CppInconsistentNaming

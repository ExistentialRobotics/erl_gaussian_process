#pragma once

#include "erl_covariance/covariance.hpp"
#include "erl_covariance/reduced_rank_covariance.hpp"

#include <utility>

namespace erl::gaussian_process {

    template<typename Dtype>
    class NoisyInputGaussianProcess {
    public:
        using Covariance = covariance::Covariance<Dtype>;
        using ReducedRankCovariance = covariance::ReducedRankCovariance<Dtype>;
        using Matrix = Eigen::MatrixX<Dtype>;
        using Vector = Eigen::VectorX<Dtype>;

        struct Setting : common::Yamlable<Setting> {
            std::string kernel_type = fmt::format("erl::covariance::Matern32<2, {}>", type_name<Dtype>());
            std::string kernel_setting_type = fmt::format("erl::covariance::Covariance<{}>::Setting", type_name<Dtype>());
            std::shared_ptr<typename Covariance::Setting> kernel = std::make_shared<typename Covariance::Setting>();
            long max_num_samples = -1;  // maximum number of training samples, -1 means no limit
            bool no_gradient_observation = false;

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

    private:
        inline static const std::string kFileHeader = fmt::format("# erl::gaussian_process::NoisyInputGaussianProcess<{}>", type_name<Dtype>());

    protected:
        long m_x_dim_ = 0;                                // dimension of x
        long m_num_train_samples_ = 0;                    // number of training samples
        long m_num_train_samples_with_grad_ = 0;          // number of training samples with gradient
        bool m_trained_ = false;                          // true if the GP is trained
        bool m_trained_once_ = false;                     // true if the GP is trained at least once
        bool m_k_train_updated_ = false;                  // true if Ktrain is updated
        long m_k_train_rows_ = 0;                         // number of rows of Ktrain
        long m_k_train_cols_ = 0;                         // number of columns of Ktrain
        Dtype m_three_over_scale_square_ = 0.;            // for computing normal variance
        std::shared_ptr<Setting> m_setting_ = nullptr;    // setting
        std::shared_ptr<Covariance> m_kernel_ = nullptr;  // kernel
        bool m_reduced_rank_kernel_ = false;              // whether the kernel is rank reduced or not
        Matrix m_mat_x_train_ = {};                       // x1, ..., xn
        Vector m_vec_y_train_ = {};                       // h(x1), ..., h(xn)
        Matrix m_mat_grad_train_ = {};                    // dh(x_j)/dx_ij for index (i, j)
        Matrix m_mat_k_train_ = {};                       // Ktrain, avoid reallocation
        Matrix m_mat_l_ = {};                             // lower triangular matrix of the Cholesky decomposition of Ktrain
        Eigen::VectorXl m_vec_grad_flag_ = {};            // true if the corresponding training sample has gradient
        Vector m_vec_alpha_ = {};                         // h(x1)..h(xn), dh(x1)/dx1_1 .. dh(xn)/dxn_1 .. dh(x1)/dx1_dim .. dh(xn)/dxn_dim
        Vector m_vec_var_x_ = {};                         // variance of x1 ... xn
        Vector m_vec_var_h_ = {};                         // variance of h(x1)..h(xn)
        Vector m_vec_var_grad_ = {};                      // variance of dh(x1)/dx1_1 .. dh(xn)/dxn_1 .. dh(x1)/dx1_dim .. dh(xn)/dxn_dim

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

        [[nodiscard]] Vector
        GetKernelCoordOrigin() const;

        void
        SetKernelCoordOrigin(const Vector &coord_origin) const;

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

        [[nodiscard]] std::shared_ptr<Covariance>
        GetKernel() const {
            return m_kernel_;
        }

        [[nodiscard]] Matrix &
        GetTrainInputSamplesBuffer() {
            return m_mat_x_train_;
        }

        [[nodiscard]] Vector &
        GetTrainOutputSamplesBuffer() {
            return m_vec_y_train_;
        }

        [[nodiscard]] Matrix &
        GetTrainOutputGradientSamplesBuffer() {
            return m_mat_grad_train_;
        }

        [[nodiscard]] Eigen::VectorXl &
        GetTrainGradientFlagsBuffer() {
            return m_vec_grad_flag_;
        }

        [[nodiscard]] Vector &
        GetTrainInputSamplesVarianceBuffer() {
            return m_vec_var_x_;
        }

        [[nodiscard]] Vector &
        GetTrainOutputValueSamplesVarianceBuffer() {
            return m_vec_var_h_;
        }

        [[nodiscard]] Vector &
        GetTrainOutputGradientSamplesVarianceBuffer() {
            return m_vec_var_grad_;
        }

        [[nodiscard]] Matrix
        GetKtrain() {
            return m_mat_k_train_;
        }

        [[nodiscard]] Vector
        GetAlpha() {
            return m_vec_alpha_;
        }

        [[nodiscard]] Matrix
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
        Test(const Eigen::Ref<const Matrix> &mat_x_test, Eigen::Ref<Matrix> mat_f_out, Eigen::Ref<Matrix> mat_var_out, Eigen::Ref<Matrix> mat_cov_out) const;

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

#include "noisy_input_gp.tpp"

    using NoisyInputGaussianProcess_d = NoisyInputGaussianProcess<double>;
    using NoisyInputGaussianProcess_f = NoisyInputGaussianProcess<float>;
}  // namespace erl::gaussian_process

template<>
struct YAML::convert<erl::gaussian_process::NoisyInputGaussianProcess_d::Setting>
    : erl::gaussian_process::NoisyInputGaussianProcess_d::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gaussian_process::NoisyInputGaussianProcess_f::Setting>
    : erl::gaussian_process::NoisyInputGaussianProcess_f::Setting::YamlConvertImpl {};

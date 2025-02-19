#pragma once

#include "erl_covariance/covariance.hpp"
#include "erl_covariance/reduced_rank_covariance.hpp"

#include <memory>

namespace erl::gaussian_process {

    /**
     * VanillaGaussianProcess implements the standard Gaussian Process
     */
    template<typename Dtype>
    class VanillaGaussianProcess {
    public:
        using Covariance = covariance::Covariance<Dtype>;
        using ReducedRankCovariance = covariance::ReducedRankCovariance<Dtype>;
        using Matrix = Eigen::MatrixX<Dtype>;
        using Vector = Eigen::VectorX<Dtype>;

        // structure for holding the parameters
        struct Setting : common::Yamlable<Setting> {
            std::string kernel_type = fmt::format("erl::covariance::OrnsteinUhlenbeck<2, {}>", type_name<Dtype>());
            std::string kernel_setting_type = fmt::format("erl::covariance::Covariance<{}>::Setting", type_name<Dtype>());
            std::shared_ptr<typename Covariance::Setting> kernel = []() -> std::shared_ptr<typename Covariance::Setting> {
                auto setting = std::make_shared<typename Covariance::Setting>();
                setting->x_dim = 2;
                setting->alpha = 1.0;
                setting->scale = 0.5;
                setting->scale_mix = 1.0;
                return setting;
            }();
            long max_num_samples = 256;

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

    private:
        inline static const std::string kFileHeader = fmt::format("# erl::gaussian_process::VanillaGaussianProcess<{}>", type_name<Dtype>());

    protected:
        long m_x_dim_ = 0;                                // dimension of x
        long m_num_train_samples_ = 0;                    // number of training samples
        bool m_trained_ = true;                           // true if the GP is trained
        bool m_trained_once_ = false;                     // true if the GP is trained at least once
        bool m_k_train_updated_ = false;                  // true if Ktrain is updated
        long m_k_train_rows_ = 0;                         // number of rows of Ktrain
        long m_k_train_cols_ = 0;                         // number of columns of Ktrain
        std::shared_ptr<Setting> m_setting_ = nullptr;    // setting
        std::shared_ptr<Covariance> m_kernel_ = nullptr;  // kernel
        bool m_reduced_rank_kernel_ = false;              // whether the kernel is rank reduced or not
        Matrix m_mat_k_train_ = {};                       // Ktrain, avoid reallocation
        Matrix m_mat_x_train_ = {};                       // x1, ..., xn
        Matrix m_mat_l_ = {};                             // lower triangular matrix of the Cholesky decomposition of Ktrain
        Vector m_vec_alpha_ = {};                         // h(x1)..h(xn), dh(x1)/dx1_1 .. dh(xn)/dxn_1 .. dh(x1)/dx1_dim .. dh(xn)/dxn_dim
        Vector m_vec_var_h_ = {};                         // variance of y1 ... yn

    public:
        VanillaGaussianProcess()
            : m_setting_(std::make_shared<Setting>()) {}

        explicit VanillaGaussianProcess(std::shared_ptr<Setting> setting)
            : m_setting_(std::move(setting)) {
            ERL_ASSERTM(m_setting_ != nullptr, "setting should not be nullptr.");
            ERL_ASSERTM(m_setting_->kernel != nullptr, "setting->kernel should not be nullptr.");
            if (m_setting_->max_num_samples > 0 && m_setting_->kernel->x_dim > 0) {
                ERL_ASSERTM(AllocateMemory(m_setting_->max_num_samples, m_setting_->kernel->x_dim), "Failed to allocate memory.");
            }
        }

        VanillaGaussianProcess(const VanillaGaussianProcess &other);
        VanillaGaussianProcess(VanillaGaussianProcess &&other) = default;
        VanillaGaussianProcess &
        operator=(const VanillaGaussianProcess &other);
        VanillaGaussianProcess &
        operator=(VanillaGaussianProcess &&other) = default;

        virtual ~VanillaGaussianProcess() = default;

        [[nodiscard]] std::shared_ptr<Setting>
        GetSetting() const {
            return m_setting_;
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

        /**
         * @brief reset the model: update flags, kernel, and allocate memory if necessary, etc.
         * @param max_num_samples maximum number of training samples
         * @param x_dim dimension of training input samples
         */
        void
        Reset(long max_num_samples, long x_dim);

        [[nodiscard]] long
        GetNumTrainSamples() const {
            return m_num_train_samples_;
        }

        [[nodiscard]] Matrix &
        GetTrainInputSamplesBuffer() {
            return m_mat_x_train_;
        }

        [[nodiscard]] Vector &
        GetTrainOutputSamplesBuffer() {
            return m_vec_alpha_;
        }

        [[nodiscard]] Vector &
        GetTrainOutputSamplesVarianceBuffer() {
            return m_vec_var_h_;
        }

        [[nodiscard]] Matrix
        GetKtrain() const {
            return m_mat_k_train_;
        }

        [[nodiscard]] Matrix
        GetCholeskyDecomposition() const {
            return m_mat_l_;
        }

        [[nodiscard]] virtual std::size_t
        GetMemoryUsage() const;

        bool
        UpdateKtrain(long num_train_samples);

        [[nodiscard]] virtual bool
        Train(long num_train_samples);

        [[nodiscard]] virtual bool
        Test(const Eigen::Ref<const Matrix> &mat_x_test, Eigen::Ref<Vector> vec_f_out, Eigen::Ref<Vector> vec_var_out) const;

        [[nodiscard]] bool
        operator==(const VanillaGaussianProcess &other) const;

        [[nodiscard]] bool
        operator!=(const VanillaGaussianProcess &other) const {
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

    using VanillaGaussianProcess_d = VanillaGaussianProcess<double>;
    using VanillaGaussianProcess_f = VanillaGaussianProcess<float>;
}  // namespace erl::gaussian_process

#include "vanilla_gp.tpp"

template<>
struct YAML::convert<erl::gaussian_process::VanillaGaussianProcess_d::Setting> : erl::gaussian_process::VanillaGaussianProcess_d::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gaussian_process::VanillaGaussianProcess_f::Setting> : erl::gaussian_process::VanillaGaussianProcess_f::Setting::YamlConvertImpl {};

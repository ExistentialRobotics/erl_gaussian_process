#pragma once

#include "init.hpp"

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
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;

        struct Setting : public common::Yamlable<Setting> {
            std::string kernel_type = type_name<Covariance>();
            std::string kernel_setting_type = type_name<typename Covariance::Setting>();
            std::shared_ptr<typename Covariance::Setting> kernel =
                std::make_shared<typename Covariance::Setting>();
            long max_num_samples = 256;

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

        class TestResult {
        protected:
            const VanillaGaussianProcess *m_gp_;
            const long m_num_test_;
            const bool m_reduced_rank_kernel_;
            const long m_x_dim_;
            const long m_y_dim_;
            MatrixX m_mat_k_test_;
            MatrixX m_mat_alpha_test_;

        public:
            TestResult(
                const VanillaGaussianProcess *gp,
                const Eigen::Ref<const MatrixX> &mat_x_test);

            [[nodiscard]] long
            GetNumTest() const;

            [[nodiscard]] const MatrixX &
            GetKtest() const;

            void
            GetMean(long y_index, Eigen::Ref<VectorX> vec_f_out, bool parallel) const;

            void
            GetMean(long index, long y_index, Dtype &f) const;

            void
            GetVariance(Eigen::Ref<VectorX> vec_var_out, bool parallel) const;

            void
            GetVariance(long index, Dtype &var) const;

        protected:
            void
            PrepareForVariance(bool parallel);
        };

        struct TrainSet {
            long x_dim = 0;        // dimension of x
            long y_dim = 0;        // dimension of y
            long num_samples = 0;  // number of training samples
            MatrixX x;             // input, x1, ..., xn
            MatrixX y;             // output, h(x1), ..., h(xn)
            VectorX var;           // variance of y, assumed to be identical across dimensions of y.

            void
            Reset(long max_num_samples, long x_dim, long y_dim);

            [[nodiscard]] bool
            operator==(const TrainSet &other) const;

            [[nodiscard]] bool
            operator!=(const TrainSet &other) const;

            [[nodiscard]] bool
            Write(std::ostream &s) const;

            [[nodiscard]] bool
            Read(std::istream &s);
        };

    protected:
        std::shared_ptr<Setting> m_setting_ = nullptr;    // setting
        bool m_trained_ = false;                          // true if the GP is trained
        bool m_trained_once_ = false;                     // true if the GP is trained at least once
        bool m_k_train_updated_ = false;                  // true if Ktrain is updated
        long m_k_train_rows_ = 0;                         // number of rows of Ktrain
        long m_k_train_cols_ = 0;                         // number of columns of Ktrain
        std::shared_ptr<Covariance> m_kernel_ = nullptr;  // kernel
        bool m_reduced_rank_kernel_ = false;  // whether the kernel is rank reduced or not
        MatrixX m_mat_k_train_ = {};          // Ktrain, avoid reallocation
        MatrixX m_mat_l_ = {};  // lower triangular matrix of the Cholesky decomposition of Ktrain
        MatrixX m_mat_alpha_ = {};  // h(x1)..h(xn), dh(x1)/dx_1..dh(xn)/dx_n, ..dh(xn)/dx_dim
        TrainSet m_train_set_;      // the training set

    public:
        explicit VanillaGaussianProcess(std::shared_ptr<Setting> setting);

        VanillaGaussianProcess(const VanillaGaussianProcess &other);
        VanillaGaussianProcess(VanillaGaussianProcess &&other) = default;
        VanillaGaussianProcess &
        operator=(const VanillaGaussianProcess &other);
        VanillaGaussianProcess &
        operator=(VanillaGaussianProcess &&other) = default;

        [[nodiscard]] std::shared_ptr<const Setting>
        GetSetting() const;

        [[nodiscard]] bool
        IsTrained() const;

        [[nodiscard]] bool
        UsingReducedRankKernel() const;

        [[nodiscard]] VectorX
        GetKernelCoordOrigin() const;

        void
        SetKernelCoordOrigin(const VectorX &coord_origin) const;

        /**
         * @brief reset the model: update flags, kernel, and allocate memory if necessary, etc.
         * @param max_num_samples maximum number of training samples
         * @param x_dim dimension of training input samples
         * @param y_dim dimension of output, default is 1
         */
        void
        Reset(long max_num_samples, long x_dim, long y_dim);

        [[nodiscard]] std::pair<long, long>
        GetKtrainSize() const;

        [[nodiscard]] std::shared_ptr<Covariance>
        GetKernel() const;

        [[nodiscard]] TrainSet &
        GetTrainSet();

        [[nodiscard]] const TrainSet &
        GetTrainSet() const;

        [[nodiscard]] const MatrixX &
        GetKtrain() const;

        [[nodiscard]] MatrixX &
        GetKtrain();

        [[nodiscard]] const MatrixX &
        GetCholeskyDecomposition() const;

        [[nodiscard]] MatrixX &
        GetCholeskyDecomposition();

        [[nodiscard]] const MatrixX &
        GetAlpha() const;

        [[nodiscard]] MatrixX &
        GetAlpha();

        [[nodiscard]] std::size_t
        GetMemoryUsage() const;

        bool
        UpdateKtrain();

        void
        Solve();

        [[nodiscard]] bool
        Train();

        [[nodiscard]] bool
        ComputeKtest(const Eigen::Ref<const MatrixX> &mat_x_test, MatrixX &mat_k_test) const;

        [[nodiscard]] std::shared_ptr<TestResult>
        Test(const Eigen::Ref<const MatrixX> &mat_x_test) const;

        [[nodiscard]] bool
        operator==(const VanillaGaussianProcess &other) const;

        [[nodiscard]] bool
        operator!=(const VanillaGaussianProcess &other) const;

        [[nodiscard]] bool
        Write(std::ostream &s) const;

        [[nodiscard]] bool
        Read(std::istream &s);

    protected:
        bool
        AllocateMemory(long max_num_samples, long x_dim, long y_dim);

        void
        InitKernel();
    };

    using VanillaGaussianProcessD = VanillaGaussianProcess<double>;
    using VanillaGaussianProcessF = VanillaGaussianProcess<float>;

    extern template class VanillaGaussianProcess<double>;
    extern template class VanillaGaussianProcess<float>;
}  // namespace erl::gaussian_process

template<>
struct YAML::convert<erl::gaussian_process::VanillaGaussianProcessD::Setting>
    : erl::gaussian_process::VanillaGaussianProcessD::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gaussian_process::VanillaGaussianProcessF::Setting>
    : erl::gaussian_process::VanillaGaussianProcessF::Setting::YamlConvertImpl {};

#pragma once

#include "init.hpp"

#include "erl_covariance/covariance.hpp"
#include "erl_covariance/reduced_rank_covariance.hpp"

#include <utility>

namespace erl::gaussian_process {

    template<typename Dtype>
    class NoisyInputGaussianProcess {
    public:
        using Covariance = covariance::Covariance<Dtype>;
        using CovarianceSetting = typename Covariance::Setting;
        using ReducedRankCovariance = covariance::ReducedRankCovariance<Dtype>;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;

        struct Setting : public common::Yamlable<Setting> {
            std::string kernel_type = type_name<Covariance>();
            std::string kernel_setting_type = type_name<CovarianceSetting>();
            std::shared_ptr<CovarianceSetting> kernel = std::make_shared<CovarianceSetting>();
            long max_num_samples = -1;  // maximum number of training samples, -1 means no limit
            bool no_gradient_observation = false;

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

        /**
         * This class holds the Ktest matrix and delays the test result computation until it is
         * necessary.
         */
        class TestResult {
        protected:
            const NoisyInputGaussianProcess *m_gp_;  // pointer to the GP
            const long m_num_test_;                  // number of test samples
            const bool m_support_gradient_;          // whether support gradient prediction
            const bool m_reduced_rank_kernel_;       // whether the kernel is rank reduced or not
            const long m_x_dim_;                     // dimension of x
            const long m_y_dim_;                     // dimension of y
            MatrixX m_mat_k_test_;      // Ktest, where Ktest(i,j) = k(x_test_i, x_train_j)
            MatrixX m_mat_alpha_test_;  // L.inv() @ Ktest, where Ktrain = L @ L.T

        public:
            TestResult(
                const NoisyInputGaussianProcess *gp,
                const Eigen::Ref<const MatrixX> &mat_x_test,
                bool will_predict_gradient);

            virtual ~TestResult() = default;

            [[nodiscard]] long
            GetNumTest() const;

            [[nodiscard]] long
            GetDimX() const;

            [[nodiscard]] long
            GetDimY() const;

            [[nodiscard]] const MatrixX &
            GetKtest() const;

            /**
             * Compute the mean of the test samples.
             * @param y_index Index of the output dimension.
             * @param vec_f_out Vector to store the result.
             * @param parallel Whether to use parallel computation.
             */
            virtual void
            GetMean(long y_index, Eigen::Ref<VectorX> vec_f_out, bool parallel) const;

            /**
             * Compute the mean of the selected test sample.
             * @param index Index of the test sample.
             * @param y_index Index of the output dimension.
             * @param f Variable to store the result.
             */
            virtual void
            GetMean(long index, long y_index, Dtype &f) const;

            /**
             * Compute the gradient of the test samples.
             * @param y_index Index of the output dimension.
             * @param mat_grad_out The matrix to store the gradient result. Each column is a
             * gradient vector.
             * @param parallel Whether to use parallel computation.
             * @return A vector indicating whether the gradients are valid (true) or not (false).
             */
            virtual Eigen::VectorXb
            GetGradient(long y_index, Eigen::Ref<MatrixX> mat_grad_out, bool parallel) const;

            /**
             * Compute the gradient of the selected test sample.
             * @param index Index of the test sample.
             * @param y_index Index of the output dimension.
             * @param grad Raw pointer to store the result.
             * @return True if the gradient is valid, false otherwise.
             */
            virtual bool
            GetGradient(long index, long y_index, Dtype *grad) const;

            /**
             * Compute the variance of prediction for the test samples.
             * @param vec_var_out Vector to store the variance result.
             * @param parallel Whether to use parallel computation.
             */
            void
            GetMeanVariance(Eigen::Ref<VectorX> vec_var_out, bool parallel) const;

            /**
             * Compute the prediction variance of the selected test sample.
             * @param index Index of the selected test sample.
             * @param var Variable to store the result.
             */
            void
            GetMeanVariance(long index, Dtype &var) const;

            /**
             * Compute the gradient variance of the test samples.
             * @param mat_var_out The matrix to store the variance result. Each column is the
             * variance of the gradient.
             * @param parallel Whether to use parallel computation.
             */
            void
            GetGradientVariance(Eigen::Ref<MatrixX> mat_var_out, bool parallel) const;

            /**
             * Compute the gradient variance of the selected test sample.
             * @param index Index of the test sample.
             * @param var Raw pointer to store the result.
             */
            void
            GetGradientVariance(long index, Dtype *var) const;

            /**
             * Compute the lower triangular part of the covariance matrix except the diagonal for
             * each test sample.
             * @param mat_cov_out The matrix to store the covariance result. Each column is the
             * covariance between the mean and the gradient.
             * @param parallel Whether to use parallel computation.
             */
            void
            GetCovariance(Eigen::Ref<MatrixX> mat_cov_out, bool parallel) const;

            /**
             * Compute the lower triangular part of the covariance matrix except the diagonal for
             * the selected test sample.
             * @param index Index of the test sample.
             * @param cov Raw pointer to store the result.
             */
            void
            GetCovariance(long index, Dtype *cov) const;

        protected:
            void
            PrepareAlphaTest(bool parallel);
        };

        struct TrainSet {
            long x_dim = 0;                  // m = dimension of x
            long y_dim = 0;                  // n = dimension of y
            long num_samples = 0;            // N = number of training samples
            long num_samples_with_grad = 0;  // number of training samples with gradient
            MatrixX x;                       // input, x1, ..., xN
            MatrixX y;                       // output, each column i: h_i(x1), ..., h_i(xN)

            // gradient, each column i: dh_1(xi)/dx_1 .. dh_1(xi)/dx_m, dh_2(xi)/dx_1 ..
            // dh_2(xi)/dx_m, ... dh_n(xi)/dx_m
            MatrixX grad;
            VectorX var_x;     // variance of x1 ... xN
            VectorX var_y;     // variance of y, assumed to be identical across dimensions of y.
            VectorX var_grad;  // gradient variance, assumed to be identical across dimensions.
            Eigen::VectorXl grad_flag;  // true if the corresponding training sample has a gradient.

            void
            Reset(long max_num_samples, long x_dim, long y_dim, bool no_gradient_observation);

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
        Dtype m_three_over_scale_square_ = 0.0f;          // for computing normal variance
        std::shared_ptr<Covariance> m_kernel_ = nullptr;  // kernel
        bool m_reduced_rank_kernel_ = false;  // whether the kernel is rank reduced or not
        MatrixX m_mat_k_train_ = {};          // Ktrain, avoid reallocation
        MatrixX m_mat_l_ = {};  // lower triangular matrix of the Cholesky decomposition of Ktrain

        // col k: h_k(x1)..h_k(xn), dh_k(x1)/dx(1,1)..dh_k(xn)/dx(1,n) ..
        // dh_k(x1)/dx(d,1)..dh_k(xn)/dx(d,n)
        MatrixX m_mat_alpha_ = {};
        TrainSet m_train_set_;  // the training set

    public:
        explicit NoisyInputGaussianProcess(std::shared_ptr<Setting> setting);

        NoisyInputGaussianProcess(const NoisyInputGaussianProcess &other);
        NoisyInputGaussianProcess(NoisyInputGaussianProcess &&other) = default;
        NoisyInputGaussianProcess &
        operator=(const NoisyInputGaussianProcess &other);
        NoisyInputGaussianProcess &
        operator=(NoisyInputGaussianProcess &&other) = default;

        virtual ~NoisyInputGaussianProcess() = default;

        template<typename T = Setting>
        [[nodiscard]] std::shared_ptr<const T>
        GetSetting() const;

        [[nodiscard]] bool
        IsTrained() const;

        [[nodiscard]] bool
        UsingReducedRankKernel() const;

        [[nodiscard]] VectorX
        GetKernelCoordOrigin() const;

        void
        SetKernelCoordOrigin(const VectorX &coord_origin) const;

        void
        Reset(long max_num_samples, long x_dim, long y_dim);

        [[nodiscard]] std::shared_ptr<Covariance>
        GetKernel() const;

        [[nodiscard]] TrainSet &
        GetTrainSet();

        [[nodiscard]] const TrainSet &
        GetTrainSet() const;

        [[nodiscard]] const MatrixX &
        GetKtrain() const;

        [[nodiscard]] const MatrixX &
        GetAlpha();

        [[nodiscard]] const MatrixX &
        GetCholeskyDecomposition();

        [[nodiscard]] virtual std::size_t
        GetMemoryUsage() const;

        bool
        UpdateKtrain();

        virtual bool
        Train();

        [[nodiscard]] virtual std::shared_ptr<TestResult>
        Test(const Eigen::Ref<const MatrixX> &mat_x_test, bool predict_gradient) const;

        [[nodiscard]] bool
        operator==(const NoisyInputGaussianProcess &other) const;

        [[nodiscard]] bool
        operator!=(const NoisyInputGaussianProcess &other) const;

        [[nodiscard]] virtual bool
        Write(std::ostream &s) const;

        [[nodiscard]] virtual bool
        Read(std::istream &s);

    protected:
        bool
        AllocateMemory(long max_num_samples, long x_dim, long y_dim);

        void
        InitKernel();
    };

    using NoisyInputGaussianProcessD = NoisyInputGaussianProcess<double>;
    using NoisyInputGaussianProcessF = NoisyInputGaussianProcess<float>;
}  // namespace erl::gaussian_process

#include "noisy_input_gp.tpp"

template<>
struct YAML::convert<erl::gaussian_process::NoisyInputGaussianProcessD::Setting>
    : erl::gaussian_process::NoisyInputGaussianProcessD::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gaussian_process::NoisyInputGaussianProcessF::Setting>
    : erl::gaussian_process::NoisyInputGaussianProcessF::Setting::YamlConvertImpl {};

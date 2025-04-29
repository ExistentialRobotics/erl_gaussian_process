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
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;

        struct Setting : common::Yamlable<Setting> {
            std::string kernel_type = type_name<Covariance>();
            std::string kernel_setting_type = type_name<typename Covariance::Setting>();
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

        struct TrainSet {
            long x_dim = 0;                  // m = dimension of x
            long y_dim = 0;                  // n = dimension of y
            long num_samples = 0;            // N = number of training samples
            long num_samples_with_grad = 0;  // number of training samples with gradient
            MatrixX x;                       // input, x1, ..., xN
            MatrixX y;                       // output, each column i: h_i(x1), ..., h_i(xN)
            MatrixX grad;                    // gradient, each column i: dh_1(xi)/dx_1 .. dh_1(xi)/dx_m, dh_2(xi)/dx_1 .. dh_2(xi)/dx_m, ... dh_n(xi)/dx_m
            VectorX var_x;                   // variance of x1 ... xN
            VectorX var_y;                   // variance of y, assumed to be identical across dimensions of y.
            VectorX var_grad;                // variance of gradient, assumed to be identical across dimensions of gradient.
            Eigen::VectorXl grad_flag;       // true if the corresponding training sample has a gradient.

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
        Dtype m_three_over_scale_square_ = 0.;            // for computing normal variance
        std::shared_ptr<Covariance> m_kernel_ = nullptr;  // kernel
        bool m_reduced_rank_kernel_ = false;              // whether the kernel is rank reduced or not
        MatrixX m_mat_k_train_ = {};                      // Ktrain, avoid reallocation
        MatrixX m_mat_l_ = {};                            // lower triangular matrix of the Cholesky decomposition of Ktrain
        MatrixX m_mat_alpha_ = {};                        // col k: h_k(x1)..h_k(xn), dh_k(x1)/dx(1,1)..dh_k(xn)/dx(1,n) .. dh_k(x1)/dx(d,1)..dh_k(xn)/dx(d,n)
        TrainSet m_train_set_;                            // the training set

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

        [[nodiscard]] virtual bool
        Test(
            const Eigen::Ref<const MatrixX> &mat_x_test,
            const std::vector<std::pair<long, bool>> &y_index_grad_pairs,
            Eigen::Ref<MatrixX> mat_f_out,
            Eigen::Ref<MatrixX> mat_var_out,
            Eigen::Ref<MatrixX> mat_cov_out) const;

        [[nodiscard]] bool
        operator==(const NoisyInputGaussianProcess &other) const;

        [[nodiscard]] bool
        operator!=(const NoisyInputGaussianProcess &other) const;

        [[nodiscard]] bool
        Write(const std::string &filename) const;

        [[nodiscard]] virtual bool
        Write(std::ostream &s) const;

        [[nodiscard]] bool
        Read(const std::string &filename);

        [[nodiscard]] virtual bool
        Read(std::istream &s);

    protected:
        bool
        AllocateMemory(long max_num_samples, long x_dim, long y_dim);

        void
        InitKernel();

        void
        ComputeValuePrediction(const MatrixX &ktest, long n_test, const std::vector<std::pair<long, bool>> &y_index_grad_pairs, Eigen::Ref<MatrixX> mat_f_out)
            const;

        void
        ComputeCovPrediction(MatrixX &ktest, long n_test, bool predict_gradient, Eigen::Ref<MatrixX> mat_var_out, Eigen::Ref<MatrixX> mat_cov_out) const;
    };

    using NoisyInputGaussianProcessD = NoisyInputGaussianProcess<double>;
    using NoisyInputGaussianProcessF = NoisyInputGaussianProcess<float>;
}  // namespace erl::gaussian_process

#include "noisy_input_gp.tpp"

template<>
struct YAML::convert<erl::gaussian_process::NoisyInputGaussianProcessD::Setting> : erl::gaussian_process::NoisyInputGaussianProcessD::Setting::YamlConvertImpl {
};

template<>
struct YAML::convert<erl::gaussian_process::NoisyInputGaussianProcessF::Setting> : erl::gaussian_process::NoisyInputGaussianProcessF::Setting::YamlConvertImpl {
};

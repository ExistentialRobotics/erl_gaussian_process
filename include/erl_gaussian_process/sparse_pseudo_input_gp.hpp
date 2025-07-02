#pragma once

#include "init.hpp"
#include "vanilla_gp.hpp"

#include "erl_covariance/covariance.hpp"
#include "erl_covariance/reduced_rank_covariance.hpp"

#include <memory>

namespace erl::gaussian_process {

    /**
     * Initialization:
     *      Q_M = K_M
     *      alpha = 0
     *      L_KM = Cholesky(K_M)
     * Update:
     *      Q_M = Q_M + K_MN (Lambda + sigma^2 I)^{-1} K_MN^T
     *      alpha = alpha + K_MN (Lambda + sigma^2 I)^{-1} y_N
     *      L_QM = Cholesky(Q_M) (can be delayed until prediction)
     *   where
     *        Lambda = diag(lambda_1, lambda_2, ..., lambda_N)
     *        lambda_i = k(x_i, x_i) - k_n^T K_M^{-1} k_n
     *        (K_MN)_mn = k(x_m, x_n), x_m is the m-th pseudo point, x_n is the n-th training point.
     *        sigma^2 is the noise variance.
     *   notes:
     *      we want to update Gaussian(K_M Q_M^{-1} alpha, K_M Q_M^{-1} K_M),
     *      which is the distribution of the target values at the pseudo points.
     * Predict:
     *      y_* = k_*^T Q_M^{-1} alpha
     *      sigma_*^2 = k(x_*, x_*) - k_*^T K_M^{-1} k_* + k_*^T Q_M^{-1} k_* + sigma^2
     * @tparam Dtype floating point type, e.g. float or double.
     */
    template<typename Dtype>
    class SparsePseudoInputGaussianProcess {
    public:
        using Covariance = covariance::Covariance<Dtype>;
        using ReducedRankCovariance = covariance::ReducedRankCovariance<Dtype>;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;
        using SparseMatrix = Eigen::SparseMatrix<Dtype>;
        using TrainSet = typename VanillaGaussianProcess<Dtype>::TrainSet;

        struct Setting : public common::Yamlable<Setting> {
            std::string kernel_type = type_name<Covariance>();
            std::string kernel_setting_type = type_name<typename Covariance::Setting>();
            std::shared_ptr<typename Covariance::Setting> kernel =
                std::make_shared<typename Covariance::Setting>();
            // maximum number of samples to train the model.
            long max_num_samples = 256;
            // threshold for sparse matrix zeroing.
            Dtype sparse_zero_threshold = 1e-6f;
            // if true, use sparse matrix to speed up the computation.
            bool use_sparse = false;
            // if true, assume Q_M is diagonal.
            bool diagonal_qm = false;

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

        class TestResult {
        protected:
            const SparsePseudoInputGaussianProcess *m_gp_;
            const long m_num_test_;
            const bool m_support_gradient_;  // whether support gradient prediction
            const bool m_use_sparse_;
            const long m_x_dim_;
            const long m_y_dim_;
            MatrixX m_mat_k_test_;
            SparseMatrix m_sparse_mat_k_test_;  // for sparse kernel
            MatrixX m_mat_alpha_;               // Q_M^{-1} alpha
            MatrixX m_mat_beta_;                // L_KM^{-1} k_*
            MatrixX m_mat_gamma_;               // L_QM^{-1} k_*

        public:
            TestResult(
                const SparsePseudoInputGaussianProcess *gp,
                const Eigen::Ref<const MatrixX> &mat_x_test,
                bool will_predict_gradient);

            [[nodiscard]] long
            GetNumTest() const;

            [[nodiscard]] long
            GetDimX() const;

            [[nodiscard]] long
            GetDimY() const;

            void
            GetMean(long y_index, Eigen::Ref<VectorX> vec_f_out, bool parallel) const;

            void
            GetMean(long index, long y_index, Dtype &f) const;

            [[nodiscard]] Eigen::VectorXb
            GetGradient(long y_index, Eigen::Ref<MatrixX> mat_grad_out, bool parallel) const;

            [[nodiscard]] bool
            GetGradient(long index, long y_index, Dtype *grad) const;

            void
            GetVariance(Eigen::Ref<VectorX> vec_var_out, bool parallel) const;

            void
            GetVariance(long index, Dtype &var) const;

        protected:
            void
            PrepareForVariance();
        };

    protected:
        std::shared_ptr<Setting> m_setting_ = nullptr;
        std::mutex m_mutex_;
        // true if the GP is trained
        bool m_trained_ = false;
        // true if the GP is trained at least once
        bool m_trained_once_ = false;
        // if m_mat_l_qm_ is updated
        bool m_mat_l_qm_updated_ = false;
        std::shared_ptr<Covariance> m_kernel_ = nullptr;
        // whether the kernel is rank reduced or not
        bool m_reduced_rank_kernel_ = false;
        // [DimX, M] pseudo points
        MatrixX m_pseudo_points_{};
        // [M, M] covariance matrix of the pseudo points
        MatrixX m_mat_km_{};
        SparseMatrix m_sparse_mat_km_{};
        // [M, M] Cholesky decomposition of m_mat_km_
        MatrixX m_mat_l_km_{};
        // [M, M] Q_M
        MatrixX m_mat_qm_{};
        // [M, M] Cholesky decomposition of m_mat_qm_
        MatrixX m_mat_l_qm_{};
        // [M, DimY] alpha vector for the pseudo points
        MatrixX m_mat_alpha_{};
        // the training set
        TrainSet m_train_set_;

    public:
        SparsePseudoInputGaussianProcess() = delete;

        explicit SparsePseudoInputGaussianProcess(
            std::shared_ptr<Setting> setting,
            MatrixX pseudo_points);

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

        [[nodiscard]] std::shared_ptr<Covariance>
        GetKernel() const;

        void
        Reset(long max_num_samples, long x_dim, long y_dim);

        [[nodiscard]] const MatrixX &
        GetPseudoPoints() const;

        [[nodiscard]] const MatrixX &
        GetMatKm() const;

        [[nodiscard]] const SparseMatrix &
        GetSparseMatKm() const;

        [[nodiscard]] const MatrixX &
        GetMatLKm() const;

        [[nodiscard]] const MatrixX &
        GetMatQm() const;

        [[nodiscard]] const MatrixX &
        GetMatLQm() const;

        [[nodiscard]] const MatrixX &
        GetMatAlpha() const;

        [[nodiscard]] TrainSet &
        GetTrainSet();

        [[nodiscard]] const TrainSet &
        GetTrainSet() const;

        [[nodiscard]] bool
        Update(bool parallel);

        [[nodiscard]] std::shared_ptr<TestResult>
        Test(const Eigen::Ref<const MatrixX> &mat_x_test, bool predict_gradient) const;

        [[nodiscard]] bool
        operator==(const SparsePseudoInputGaussianProcess &other) const;

        [[nodiscard]] bool
        operator!=(const SparsePseudoInputGaussianProcess &other) const;

        [[nodiscard]] bool
        Write(std::ostream &s) const;

        [[nodiscard]] bool
        Read(std::istream &s);

    private:
        bool
        UpdateDense(bool parallel);

        bool
        UpdateSparse(bool parallel);

        void
        PrepareLqm();
    };

    using SparsePseudoInputGaussianProcessD = SparsePseudoInputGaussianProcess<double>;
    using SparsePseudoInputGaussianProcessF = SparsePseudoInputGaussianProcess<float>;

    extern template class SparsePseudoInputGaussianProcess<double>;
    extern template class SparsePseudoInputGaussianProcess<float>;
}  // namespace erl::gaussian_process

template<>
struct YAML::convert<erl::gaussian_process::SparsePseudoInputGaussianProcessD::Setting>
    : erl::gaussian_process::SparsePseudoInputGaussianProcessD::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gaussian_process::SparsePseudoInputGaussianProcessF::Setting>
    : erl::gaussian_process::SparsePseudoInputGaussianProcessF::Setting::YamlConvertImpl {};

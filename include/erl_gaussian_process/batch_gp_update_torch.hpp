#pragma once

#ifdef ERL_USE_LIBTORCH

    #include "erl_common/eigen.hpp"

    #include <torch/torch.h>

namespace erl::gaussian_process {

    /**
     * Update multiple Gaussian process models using libtorch.
     * @tparam Dtype data type, e.g., float or double.
     */
    template<typename Dtype>
    class BatchGaussianProcessUpdateTorch {
    public:
        using Matrix = Eigen::MatrixX<Dtype>;

    private:
        // (B, N, N) matrix of the covariance matrix.
        torch::Tensor m_mat_k_train_;
        // (B, N, N) matrix of the Cholesky decomposition of the covariance matrix.
        torch::Tensor m_mat_l_;
        // (B, N, Dy) matrix of the alpha matrix.
        torch::Tensor m_mat_alpha_;
        // B: number of maps
        long m_num_maps_ = 0;
        // N: k_train_size
        long m_max_k_train_size_ = 0;
        // Dy: number of output dimensions
        long m_num_y_dims_ = 0;
        // device to run the computations on, default is CUDA
        torch::Device m_device_ = torch::kCUDA;

    public:
        BatchGaussianProcessUpdateTorch() = default;

        BatchGaussianProcessUpdateTorch(const BatchGaussianProcessUpdateTorch &other) = default;
        BatchGaussianProcessUpdateTorch &
        operator=(const BatchGaussianProcessUpdateTorch &other) = default;
        BatchGaussianProcessUpdateTorch(BatchGaussianProcessUpdateTorch &&other) = default;
        BatchGaussianProcessUpdateTorch &
        operator=(BatchGaussianProcessUpdateTorch &&other) = default;

        void
        SetDevice(const torch::Device &device) {
            m_device_ = device;
        }

        /**
         *
         * @param num_gps Number of Gaussian processes to update.
         * @param max_k_train_size Max k_train size.
         * @param num_y_dims Number of output dimensions.
         */
        void
        PrepareMemory(long num_gps, long max_k_train_size, long num_y_dims);

        /**
         *
         * @param gp_idx Index of the Gaussian process instance.
         * @param k_train_size k_train size.
         * @param k_train Matrix of k_train, shape (M, M).
         * @param alpha Matrix of alpha, shape (M, Dy), where Dy is the number of output dimensions.
         */
        void
        LoadGpData(long gp_idx, long k_train_size, const Matrix &k_train, const Matrix &alpha);

        void
        Solve();

        void
        GetGpResult(long gp_idx, Matrix &l_train, Matrix &alpha) const;
    };

    extern template class BatchGaussianProcessUpdateTorch<double>;
    extern template class BatchGaussianProcessUpdateTorch<float>;

}  // namespace erl::gaussian_process

#endif

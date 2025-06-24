#ifdef ERL_USE_LIBTORCH
    #include "erl_gaussian_process/batch_gp_update_torch.hpp"

    #include "erl_common/eigen.hpp"

namespace erl::gaussian_process {

    template<typename Dtype>
    void
    BatchGaussianProcessUpdateTorch<Dtype>::PrepareMemory(
        const long num_gps,
        const long max_k_train_size,
        const long num_y_dims) {

        m_num_maps_ = num_gps;
        m_max_k_train_size_ = max_k_train_size;
        m_num_y_dims_ = num_y_dims;

        ERL_DEBUG_ASSERT(m_num_maps_ > 0, "Number of maps must be greater than 0.");
        ERL_DEBUG_ASSERT(m_max_k_train_size_ > 0, "max_k_train_size must be greater than 0.");
        ERL_DEBUG_ASSERT(m_num_y_dims_ > 0, "Number of output dimensions must be greater than 0.");

        if (!m_mat_k_train_.defined() || m_mat_k_train_.size(0) != m_num_maps_ ||
            m_mat_k_train_.size(1) != m_max_k_train_size_) {
            m_mat_k_train_ = torch::eye(
                                 m_max_k_train_size_,
                                 torch::TensorOptions()
                                     .dtype(sizeof(Dtype) == 4 ? torch::kFloat32 : torch::kFloat64)
                                     .device(torch::kCPU))
                                 .unsqueeze(0)
                                 .repeat({m_num_maps_, 1, 1});
        }

        if (!m_mat_alpha_.defined() || m_mat_alpha_.size(0) != m_num_maps_ ||
            m_mat_alpha_.size(1) != m_max_k_train_size_ || m_mat_alpha_.size(2) != m_num_y_dims_) {
            m_mat_alpha_ = torch::zeros(
                {m_num_maps_, m_max_k_train_size_, m_num_y_dims_},
                torch::TensorOptions()
                    .dtype(sizeof(Dtype) == 4 ? torch::kFloat32 : torch::kFloat64)
                    .device(torch::kCPU));
        }
    }

    template<typename Dtype>
    void
    BatchGaussianProcessUpdateTorch<Dtype>::LoadGpData(
        const long gp_idx,
        const long k_train_size,
        const Matrix &k_train,
        const Matrix &alpha) {

        auto ptr = m_mat_k_train_.mutable_data_ptr<Dtype>();
        ptr += gp_idx * m_max_k_train_size_ * m_max_k_train_size_;
        std::memcpy(ptr, k_train.data(), sizeof(Dtype) * m_max_k_train_size_ * m_max_k_train_size_);

        ptr = m_mat_alpha_.mutable_data_ptr<Dtype>();
        ptr += gp_idx * m_max_k_train_size_ * m_num_y_dims_;
        std::memcpy(ptr, alpha.data(), sizeof(Dtype) * m_max_k_train_size_ * m_num_y_dims_);

        using namespace torch::indexing;
        (void) m_mat_k_train_
            .index({gp_idx, Slice(0, k_train_size), Slice(k_train_size, m_max_k_train_size_)})
            .fill_(0.0f);
        (void) m_mat_k_train_.index({gp_idx, Slice(k_train_size, m_max_k_train_size_)}).fill_(0.0f);
        for (long i = k_train_size; i < m_max_k_train_size_; ++i) {
            m_mat_k_train_.index({gp_idx, i, i}) = 1.0f;  // set diagonal to 1.0
        }

        (void) m_mat_alpha_.index({gp_idx, Slice(k_train_size, m_max_k_train_size_)}).fill_(0.0f);
    }

    template<typename Dtype>
    void
    BatchGaussianProcessUpdateTorch<Dtype>::Solve() {
        m_mat_l_ = torch::linalg_cholesky(m_mat_k_train_.to(m_device_));
        m_mat_alpha_ = m_mat_alpha_.to(m_device_).cholesky_solve(m_mat_l_);
        m_mat_l_ = m_mat_l_.transpose(1, 2).contiguous();
        m_mat_alpha_ = m_mat_alpha_.cholesky_solve(m_mat_l_, true);
        torch::cuda::synchronize();
        m_mat_l_ = m_mat_l_.to(torch::kCPU);  // move back to CPU
        m_mat_alpha_ = m_mat_alpha_.to(torch::kCPU);
    }

    template<typename Dtype>
    void
    BatchGaussianProcessUpdateTorch<Dtype>::GetGpResult(
        const long gp_idx,
        Matrix &l_train,
        Matrix &alpha) const {

        auto *ptr = m_mat_l_.data_ptr<Dtype>();
        ptr += gp_idx * m_max_k_train_size_ * m_max_k_train_size_;
        std::memcpy(ptr, l_train.data(), sizeof(Dtype) * m_max_k_train_size_ * m_max_k_train_size_);

        ptr = m_mat_alpha_.data_ptr<Dtype>();
        ptr += gp_idx * m_max_k_train_size_ * m_num_y_dims_;
        std::memcpy(ptr, alpha.data(), sizeof(Dtype) * m_max_k_train_size_ * m_num_y_dims_);
    }

    template class BatchGaussianProcessUpdateTorch<double>;
    template class BatchGaussianProcessUpdateTorch<float>;

}  // namespace erl::gaussian_process

#endif

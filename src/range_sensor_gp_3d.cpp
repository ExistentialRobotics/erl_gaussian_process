#include "erl_gaussian_process/range_sensor_gp_3d.hpp"

#include "erl_common/angle_utils.hpp"

namespace erl::gaussian_process {

    RangeSensorGaussianProcess3D::RangeSensorGaussianProcess3D(std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)),
          m_mapping_(Mapping::Create(m_setting_->mapping)) {
        m_range_sensor_frame_ = geometry::RangeSensorFrame3D::Create(m_setting_->range_sensor_frame_type, m_setting_->range_sensor_frame);

        const Eigen::MatrixX<Eigen::Vector2d> &frame_coords = m_range_sensor_frame_->GetFrameCoords();
        const long num_rows = frame_coords.rows();
        const long row_gs = m_setting_->row_group_size;
        const long row_step = m_setting_->row_group_size - m_setting_->row_overlap_size;
        const long row_half_overlap = m_setting_->row_overlap_size / 2;
        const long num_row_groups = std::max(1l, num_rows / row_step) + 1;
        const long row_gs2 = (num_rows - (num_row_groups - 2) * row_step) / 2;
        const long num_cols = frame_coords.cols();
        const long col_gs = m_setting_->col_group_size;
        const long col_step = m_setting_->col_group_size - m_setting_->col_overlap_size;
        const long col_half_overlap = m_setting_->col_overlap_size / 2;
        const long num_col_groups = std::max(1l, num_cols / col_step) + 1;
        const long col_gs2 = (num_cols - (num_col_groups - 2) * col_step) / 2;

        m_setting_->gp->max_num_samples = m_setting_->row_group_size * m_setting_->col_group_size;  // adjust the max_num_samples
        m_setting_->gp->kernel->x_dim = 2;                                                          // row and col
        m_gps_.setConstant(num_row_groups, num_col_groups, nullptr);

        m_row_partitions_.reserve(num_row_groups);
        // first partition
        m_row_partitions_.emplace_back(0, row_gs2 + row_half_overlap, frame_coords(m_setting_->row_margin, 0)[0], frame_coords(row_gs2, 0)[0]);
        // middle partitions
        for (long i = 0; i < num_row_groups - 2; ++i) {
            const long index_left = i * row_step + row_gs2 - row_half_overlap;
            const long index_right = index_left + row_gs;
            const double coord_left = frame_coords(index_left + row_half_overlap, 0)[0];
            const double coord_right = frame_coords(index_right - row_half_overlap, 0)[0];
            m_row_partitions_.emplace_back(index_left, index_right, coord_left, coord_right);
        }
        // last partition
        m_row_partitions_.emplace_back(
            num_rows - row_gs2 - row_half_overlap,
            num_rows,
            frame_coords(num_rows - row_gs2, 0)[0],
            frame_coords(num_rows - 1 - m_setting_->row_margin, 0)[0]);

#ifndef NDEBUG
        std::vector<long> row_partition_sizes;
        for (const auto &[row_index_left, row_index_right, row_coord_left, row_coord_right]: m_row_partitions_) {
            row_partition_sizes.push_back(row_index_right - row_index_left);
        }
        ERL_DEBUG("Row partition sizes: {}", row_partition_sizes);
#endif

        m_col_partitions_.reserve(num_col_groups);
        // first partition
        m_col_partitions_.emplace_back(0, col_gs2 + col_half_overlap, frame_coords(0, m_setting_->col_margin)[1], frame_coords(0, col_gs2)[1]);
        // middle partitions
        for (long i = 0; i < num_col_groups - 2; ++i) {
            const long index_left = i * col_step + col_gs2 - col_half_overlap;
            const long index_right = index_left + col_gs;
            const double coord_left = frame_coords(0, index_left + col_half_overlap)[1];
            const double coord_right = frame_coords(0, index_right - col_half_overlap)[1];
            m_col_partitions_.emplace_back(index_left, index_right, coord_left, coord_right);
        }
        // last partition
        m_col_partitions_.emplace_back(
            num_cols - col_gs2 - col_half_overlap,
            num_cols,
            frame_coords(0, num_cols - col_gs2)[1],
            frame_coords(0, num_cols - 1 - m_setting_->col_margin)[1]);

#ifndef NDEBUG
        std::vector<long> col_partition_sizes;
        for (const auto &[col_index_left, col_index_right, col_coord_left, col_coord_right]: m_col_partitions_) {
            col_partition_sizes.push_back(col_index_right - col_index_left);
        }
        ERL_DEBUG("Col partition sizes: {}", col_partition_sizes);
#endif
    }

    void
    RangeSensorGaussianProcess3D::Reset() {
        m_trained_ = false;
    }

    bool
    RangeSensorGaussianProcess3D::StoreData(const Eigen::Matrix3d &rotation, const Eigen::Vector3d &translation, Eigen::MatrixXd ranges) {
        if (m_setting_->range_sensor_frame_type == geometry::RgbdFrame3D::GetFrameType()) {
            ranges = std::reinterpret_pointer_cast<geometry::RgbdFrame3D>(m_range_sensor_frame_)->DepthImageToDepth(ranges);
        }
        m_range_sensor_frame_->UpdateRanges(rotation, translation, std::move(ranges), false);
        m_mapped_distances_ = m_range_sensor_frame_->GetRanges().unaryExpr(m_mapping_->map);
        return m_range_sensor_frame_->IsValid();
    }

    bool
    RangeSensorGaussianProcess3D::Train(const Eigen::Matrix3d &rotation, const Eigen::Vector3d &translation, Eigen::MatrixXd ranges) {

        Reset();

        if (!StoreData(rotation, translation, std::move(ranges))) {
            ERL_DEBUG("No training data is stored.");
            return false;
        }

#pragma omp parallel for collapse(2) default(none) shared(g_print_mutex)
        for (long j = 0; j < static_cast<long>(m_col_partitions_.size()); ++j) {
            for (long i = 0; i < static_cast<long>(m_row_partitions_.size()); ++i) {
                const auto &[row_index_left, row_index_right, row_coord_left, row_coord_right] = m_row_partitions_[i];
                const auto &[col_index_left, col_index_right, col_coord_left, col_coord_right] = m_col_partitions_[j];
                std::shared_ptr<VanillaGaussianProcess> &gp = m_gps_(i, j);
                if (gp == nullptr) { gp = std::make_shared<VanillaGaussianProcess>(m_setting_->gp); }
                gp->Reset(m_setting_->gp->max_num_samples, 2);
                long cnt = 0;
                Eigen::MatrixXd &train_input_samples = gp->GetTrainInputSamplesBuffer();
                Eigen::VectorXd &train_output_samples = gp->GetTrainOutputSamplesBuffer();
                Eigen::VectorXd &train_output_samples_variance = gp->GetTrainOutputSamplesVarianceBuffer();
                const Eigen::MatrixXb &mask_hit = m_range_sensor_frame_->GetHitMask();
                const Eigen::MatrixX<Eigen::Vector2d> &frame_coords = m_range_sensor_frame_->GetFrameCoords();
                for (long c = col_index_left; c < col_index_right; ++c) {
                    for (long r = row_index_left; r < row_index_right; ++r) {
                        if (!mask_hit(r, c)) { continue; }
                        train_input_samples.leftCols(cnt) << frame_coords(r, c);
                        train_output_samples[cnt] = m_mapped_distances_(r, c);
                        train_output_samples_variance[cnt] = m_setting_->sensor_range_var;
                        ++cnt;
                    }
                }
                if (cnt > 0 && !gp->Train(cnt)) {
#pragma omp critical  // avoid multiple threads printing at the same time
                    ERL_WARN("Failed to train GP for partition ({}, {}) X ({}, {})", row_index_left, row_index_right, col_index_left, col_index_right);
                }
            }
        }

        m_trained_ = true;
        return true;
    }

    bool
    RangeSensorGaussianProcess3D::Test(
        const Eigen::Ref<const Eigen::Matrix2Xd> &frame_coords,
        Eigen::Ref<Eigen::VectorXd> vec_ranges,
        Eigen::Ref<Eigen::VectorXd> vec_ranges_var,
        const bool un_map) const {

        if (!m_trained_) { return false; }
        long n = frame_coords.cols();
        ERL_DEBUG_ASSERT(vec_ranges.size() >= n, "vec_ranges size = {}, it should be >= {}.", vec_ranges.size(), n);
        ERL_DEBUG_ASSERT(vec_ranges_var.size() >= n, "vec_ranges_var size = {}, it should be >= {}.", vec_ranges_var.size(), n);

        vec_ranges.setZero();
        vec_ranges_var.setConstant(m_setting_->init_variance);

        for (long i = 0; i < n; ++i) {
            // search for the partition
            // row
            const double row_coord = frame_coords(0, i);
            long partition_row_index = 0;
            for (; partition_row_index < static_cast<long>(m_row_partitions_.size()); ++partition_row_index) {
                if (const auto &[row_index_left, row_index_right, row_coord_left, row_coord_right] = m_row_partitions_[partition_row_index];
                    row_coord >= row_coord_left && row_coord <= row_coord_right) {
                    break;
                }
            }
            if (partition_row_index >= static_cast<long>(m_row_partitions_.size())) { continue; }
            // col
            const double col_coord = frame_coords(1, i);
            long partition_col_index = 0;
            for (; partition_col_index < static_cast<long>(m_col_partitions_.size()); ++partition_col_index) {
                if (const auto &[col_index_left, col_index_right, col_coord_left, col_coord_right] = m_col_partitions_[partition_col_index];
                    col_coord >= col_coord_left && col_coord <= col_coord_right) {
                    break;
                }
            }
            if (partition_col_index >= static_cast<long>(m_col_partitions_.size())) { continue; }

            const auto &gp = m_gps_(partition_row_index, partition_col_index);
            if (!gp->IsTrained()) { continue; }
            Eigen::Scalard f, var;
            if (!gp->Test(frame_coords.col(i), f, var)) { continue; }  // invalid test
            vec_ranges[i] = un_map ? m_mapping_->inv(f[0]) : f[0];
            vec_ranges_var[i] = var[0];
        }
        return true;
    }

    bool
    RangeSensorGaussianProcess3D::ComputeOcc(
        const Eigen::Vector2d &frame_coords,
        const double r,
        Eigen::Ref<Eigen::Scalard> range_pred,
        Eigen::Ref<Eigen::Scalard> range_pred_var,
        double &occ) const {

        if (!Test(frame_coords, range_pred, range_pred_var, false)) { return false; }
        if (range_pred_var[0] > m_setting_->max_valid_range_var) { return false; }
        const double a = r * m_setting_->occ_test_temperature;
        occ = 2.0 / (1.0 + std::exp(a * (range_pred[0] - m_mapping_->map(r)))) - 1.0;
        range_pred[0] = m_mapping_->inv(range_pred[0]);
        return true;
    }
}  // namespace erl::gaussian_process

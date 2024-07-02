#include "erl_gaussian_process/lidar_gp_2d.hpp"

namespace erl::gaussian_process {
    /*static double
    ClipAngle(double angle) {
        if (angle < -M_PI) {
            const auto n = std::floor(std::abs(angle) / M_PI);
            angle += n * M_PI * 2;
        } else if (angle >= M_PI) {
            const auto n = std::floor(angle / M_PI);
            angle -= n * M_PI * 2;
        }
        return angle;
    }*/

    /*bool
    LidarGaussianProcess2D::TrainBuffer::Store(
        const Eigen::Ref<const Eigen::VectorXd> &vec_new_angles,
        const Eigen::Ref<const Eigen::VectorXd> &vec_new_distances,
        const Eigen::Ref<const Eigen::Matrix23d> &mat_new_pose) {

        // Store sorted original data
        std::vector<int> indices(vec_new_angles.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](const int i, const int j) { return vec_new_angles[i] < vec_new_angles[j]; });
        vec_angles_original = vec_new_angles(indices);
        vec_ranges_original = vec_new_distances(indices);

        // Reset
        const auto n = vec_angles_original.size();
        vec_angles.resize(n);
        vec_ranges.resize(n);
        vec_mask_hit.setConstant(n + 1, false);  // +1 flag to mask the last valid area
        mat_direction_local.resize(2, n);
        mat_xy_local.resize(2, n);
        mat_direction_global.resize(2, n);
        mat_xy_global.resize(2, n);
        mapping = Mapping::Create(setting->mapping);

        // mat_new_pose: a flattened version of 2x3 row-major matrix array
        // p0 p1 p2
        // p3 p4 p5
        position = mat_new_pose.col(2);
        rotation = mat_new_pose.block<2, 2>(0, 0);

        max_distance = 0.;
        int cnt = 0;
        for (ssize_t i = 0; i < n; ++i) {
            const double angle = ClipAngle(vec_angles_original[i]);  // make sure angle is within [-pi, pi)
            const double &range = vec_ranges_original[i];

            if (std::isnan(range) || range <= setting->valid_range_min || range >= setting->valid_range_max) { continue; }  // valid range: [min, max)

            vec_angles[cnt] = angle;
            vec_ranges[cnt] = range;
            vec_mask_hit[i] = true;
            if (range > max_distance) { max_distance = range; }

            // local frame
            mat_direction_local.col(cnt) << std::cos(angle), std::sin(angle);
            mat_xy_local.col(cnt) = mat_direction_local.col(cnt) * range;
            // global frame
            mat_direction_global.col(cnt) = LocalToGlobalSo2(mat_direction_local.col(cnt));
            mat_xy_global.col(cnt) = LocalToGlobalSe2(mat_xy_local.col(cnt));

            cnt++;
        }

        vec_angles.conservativeResize(cnt);
        vec_ranges.conservativeResize(cnt);
        mat_xy_local.conservativeResize(2, cnt);
        mat_xy_global.conservativeResize(2, cnt);

        vec_mapped_distances = vec_ranges.unaryExpr(mapping->map);
        return cnt > 0;
    }*/

    LidarGaussianProcess2D::LidarGaussianProcess2D(std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)),
          m_mapping_(Mapping::Create(m_setting_->mapping)) {
        m_lidar_frame_ = std::make_shared<geometry::LidarFrame2D>(m_setting_->lidar_frame);

        const Eigen::VectorXd &angles = m_lidar_frame_->GetAnglesInFrame();
        const long n = angles.rows();
        const long gs = m_setting_->group_size;
        const long step = m_setting_->group_size - m_setting_->overlap_size;
        const long half_overlap = m_setting_->overlap_size / 2;
        const long num_groups = std::max(1l, n / step) + 1;
        const long gs2 = (n - (num_groups - 2) * step) / 2;

        if (n <= m_setting_->overlap_size) {
            ERL_DEBUG("LidarGaussianProcess2D: no enough samples to perform partition.");
            return;
        }

        m_setting_->gp->max_num_samples = m_setting_->group_size;  // adjust the max_num_samples
        m_setting_->gp->kernel->x_dim = 1;                         // adjust the x_dim
        m_gps_.resize(num_groups);

        m_angle_partitions_.reserve(num_groups + 1);
        // first partition
        m_angle_partitions_.emplace_back(0, gs2 + half_overlap, angles[m_setting_->margin], angles[gs2]);
        // middle partitions
        for (long i = 0; i < num_groups - 2; ++i) {
            const long index_left = i * step + gs2 - half_overlap;
            const long index_right = index_left + gs;
            const double coord_left = angles[index_left + half_overlap];
            const double coord_right = angles[index_right - half_overlap];
            m_angle_partitions_.emplace_back(index_left, index_right, coord_left, coord_right);
        }
        // last partition
        m_angle_partitions_.emplace_back(n - gs2 - half_overlap, n, angles[n - 1 - gs2], angles[n - 1 - m_setting_->margin]);
    }

    void
    LidarGaussianProcess2D::Reset() {
        m_trained_ = false;
    }

    bool
    LidarGaussianProcess2D::StoreData(const Eigen::Matrix2d &rotation, const Eigen::Vector2d &translation, Eigen::VectorXd ranges) {
        m_lidar_frame_->UpdateRanges(rotation, translation, std::move(ranges), false);
        m_mapped_distances_ = m_lidar_frame_->GetRanges().unaryExpr(m_mapping_->map);
        return m_lidar_frame_->IsValid();
    }

    bool
    LidarGaussianProcess2D::Train(
        const Eigen::Matrix2d &rotation,
        const Eigen::Vector2d &translation,
        Eigen::VectorXd ranges,
        const bool repartition_on_hit_rays) {

        Reset();

        if (!StoreData(rotation, translation, std::move(ranges))) {
            ERL_DEBUG("No training data is stored.");
            return false;
        }

        if (repartition_on_hit_rays) {
            m_angle_partitions_.clear();
            const long n = m_lidar_frame_->GetNumHitRays();
            const long s = m_setting_->group_size - m_setting_->overlap_size;
            const auto num_groups = std::max(1l, n / s) + 1;

            m_setting_->gp->max_num_samples = m_setting_->group_size;  // adjust the max_num_samples
            m_setting_->gp->kernel->x_dim = 1;                         // adjust the x_dim
            m_gps_.resize(num_groups);
            m_angle_partitions_.reserve(num_groups);

            const std::vector<long> &hit_ray_indices = m_lidar_frame_->GetHitRayIndices();
            const Eigen::VectorXd &angles_frame = m_lidar_frame_->GetAnglesInFrame();
            for (int i = 0; i < num_groups - 2; ++i) {
                long index_left = i * s;                                 // lower bound, included
                long index_right = index_left + m_setting_->group_size;  // upper bound, not included

                // hit ray indices to original ray indices
                index_left = hit_ray_indices[index_left];
                index_right = hit_ray_indices[index_right];

                m_angle_partitions_.emplace_back(index_left, index_right, angles_frame[index_left], angles_frame[index_right]);
            }

            // the last two groups
            long index_left = (num_groups - 2) * s;
            long index_right = index_left + (n - index_left + m_setting_->overlap_size) / 2;
            index_left = hit_ray_indices[index_left];
            index_right = hit_ray_indices[index_right];
            m_angle_partitions_.emplace_back(index_left, index_right, angles_frame[index_left], angles_frame[index_right]);

            index_left = index_left + (n - index_left - m_setting_->overlap_size) / 2;
            index_left = hit_ray_indices[index_left];
            index_right = hit_ray_indices[n - 1];
            m_angle_partitions_.emplace_back(index_left, index_right, angles_frame[index_left], angles_frame[index_right]);
        }

#pragma omp parallel for default(none) shared(g_print_mutex)
        for (long i = 0; i < static_cast<long>(m_angle_partitions_.size()); ++i) {
            const auto &[index_left, index_right, coord_left, coord_right] = m_angle_partitions_[i];
            std::shared_ptr<VanillaGaussianProcess> &gp = m_gps_[i];
            if (gp == nullptr) { gp = std::make_shared<VanillaGaussianProcess>(m_setting_->gp); }
            gp->Reset(m_setting_->gp->max_num_samples, 1);
            long cnt = 0;
            Eigen::MatrixXd &train_input_samples = gp->GetTrainInputSamplesBuffer();
            Eigen::VectorXd &train_output_samples = gp->GetTrainOutputSamplesBuffer();
            Eigen::VectorXd &train_output_samples_variance = gp->GetTrainOutputSamplesVarianceBuffer();
            const Eigen::VectorXb &mask_hit = m_lidar_frame_->GetHitMask();
            const Eigen::VectorXd &angles = m_lidar_frame_->GetAnglesInFrame();
            for (long j = index_left; j < index_right; ++j) {
                if (!mask_hit[j]) { continue; }
                train_input_samples(0, cnt) = angles[j];
                train_output_samples[cnt] = m_mapped_distances_[j];
                train_output_samples_variance(cnt) = m_setting_->sensor_range_var;
                ++cnt;
            }
            if (cnt > 0) { (void) gp->Train(cnt); }
        }

        /*const long s = m_setting_->group_size - m_setting_->overlap_size;
        const auto num_groups = std::max(1l, n / s) + 1;
        m_setting_->gp->max_num_samples = m_setting_->group_size;  // adjust the max_num_samples
        m_setting_->gp->kernel->x_dim = 1;                         // adjust the x_dim
        m_gps_.resize(num_groups);
        m_partitions_.reserve(num_groups + 1);
        m_partitions_.push_back(m_train_buffer_.vec_angles[0]);
        const long half_overlap_size = m_setting_->overlap_size / 2;

        for (int i = 0; i < num_groups - 2; ++i) {
            const long index_left = i * s;                                 // lower bound, included
            const long index_right = index_left + m_setting_->group_size;  // upper bound, not included

            m_partitions_.push_back(m_train_buffer_.vec_angles[index_right - half_overlap_size]);
            std::shared_ptr<VanillaGaussianProcess> &gp = m_gps_[i];
            if (gp == nullptr) { gp = std::make_shared<VanillaGaussianProcess>(m_setting_->gp); }
            gp->Reset(m_setting_->group_size, 1);
            // buffer size fits the data exactly
            gp->GetTrainInputSamplesBuffer() = m_train_buffer_.vec_angles.segment(index_left, index_right - index_left).transpose();
            gp->GetTrainOutputSamplesBuffer() = m_train_buffer_.vec_mapped_distances.segment(index_left, index_right - index_left);
            gp->GetTrainOutputSamplesVarianceBuffer().setConstant(m_setting_->sensor_range_var);
            (void) gp->Train(m_setting_->group_size);
        }

        // the last two groups
        long index_left = (num_groups - 2) * s;
        long index_right = index_left + (n - index_left + m_setting_->overlap_size) / 2;
        m_partitions_.push_back(m_train_buffer_.vec_angles(index_right - half_overlap_size));
        {
            std::shared_ptr<VanillaGaussianProcess> &gp = m_gps_[num_groups - 2];
            if (gp == nullptr) { gp = std::make_shared<VanillaGaussianProcess>(m_setting_->gp); }
            const long num_samples = index_right - index_left;
            gp->Reset(num_samples, 1);
            gp->GetTrainInputSamplesBuffer().leftCols(num_samples) = m_train_buffer_.vec_angles.segment(index_left, num_samples).transpose();
            gp->GetTrainOutputSamplesBuffer().head(num_samples) = m_train_buffer_.vec_mapped_distances.segment(index_left, num_samples);
            gp->GetTrainOutputSamplesVarianceBuffer().head(num_samples).setConstant(m_setting_->sensor_range_var);
            (void) gp->Train(num_samples);
        }

        index_left = index_left + (n - index_left - m_setting_->overlap_size) / 2;
        index_right = n;
        m_partitions_.push_back(m_train_buffer_.vec_angles(n - 1));
        {
            std::shared_ptr<VanillaGaussianProcess> &gp = m_gps_[num_groups - 1];
            if (gp == nullptr) { gp = std::make_shared<VanillaGaussianProcess>(m_setting_->gp); }
            const long num_samples = index_right - index_left;
            gp->Reset(num_samples, 1);
            gp->GetTrainInputSamplesBuffer().leftCols(num_samples) = m_train_buffer_.vec_angles.segment(index_left, num_samples).transpose();
            gp->GetTrainOutputSamplesBuffer().head(num_samples) = m_train_buffer_.vec_mapped_distances.segment(index_left, num_samples);
            gp->GetTrainOutputSamplesVarianceBuffer().head(num_samples).setConstant(m_setting_->sensor_range_var);
            (void) gp->Train(num_samples);
        }*/

        m_trained_ = true;
        return true;
    }

    bool
    LidarGaussianProcess2D::Test(
        const Eigen::Ref<const Eigen::VectorXd> &angles,
        const bool angles_are_local,
        Eigen::Ref<Eigen::VectorXd> vec_ranges,
        Eigen::Ref<Eigen::VectorXd> vec_ranges_var,
        const bool un_map,
        // ReSharper disable once CppParameterNeverUsed
        const bool parallel) const {

        if (!m_trained_) { return false; }

        const long n = angles.size();
        ERL_ASSERTM(vec_ranges.size() >= n, "vec_ranges size = {}, it should be >= {}.", vec_ranges.size(), n);
        ERL_ASSERTM(vec_ranges_var.size() >= n, "vec_ranges_var size = {}, it should be >= {}.", vec_ranges_var.size(), n);

        vec_ranges.setZero();
        vec_ranges_var.setConstant(m_setting_->init_variance);

#pragma omp parallel for if (parallel) default(none) shared(angles, angles_are_local, vec_ranges, vec_ranges_var, un_map, parallel)
        for (int i = 0; i < angles.size(); ++i) {
            Eigen::Scalard angle_local;
            angle_local[0] = angles[i];
            if (!angles_are_local) {
                const Eigen::Vector2d direction_local = m_lidar_frame_->WorldToFrameSo2({std::cos(angle_local[0]), std::sin(angle_local[0])});
                angle_local[0] = std::atan2(direction_local[1], direction_local[0]);
            }
            long partition_index = 0;
            for (; partition_index < static_cast<long>(m_angle_partitions_.size()); ++partition_index) {
                if (const auto &[index_left, index_right, coord_left, coord_right] = m_angle_partitions_[partition_index];
                    angle_local[0] >= coord_left && angle_local[0] <= coord_right) {
                    break;
                }
            }
            if (partition_index >= static_cast<long>(m_angle_partitions_.size())) { continue; }

            const auto gp = m_gps_[partition_index];
            if (!gp->IsTrained()) { continue; }
            Eigen::Scalard f, var;
            if (!gp->Test(angle_local, f, var)) { continue; }
            vec_ranges[i] = un_map ? m_mapping_->inv(f[0]) : f[0];
            vec_ranges_var[i] = var[0];

            // for (size_t j = 0; j < m_gps_.size(); ++j) {
            //     if (angles[i] < m_partitions_[j] || angles[i] > m_partitions_[j + 1]) { continue; }
            //     if (!m_gps_[j]->IsTrained()) { break; }
            //     Eigen::Scalard f, var;
            //     if (!m_gps_[j]->Test(angles.segment<1>(i), f, var)) { break; }
            //     if (un_map) {
            //         fs[i] = m_train_buffer_.mapping->inv(f[0]);
            //     } else {
            //         fs[i] = f[0];
            //     }
            //     vars[i] = var[0];
            //     break;
            // }
        }
        return true;
    }

    bool
    LidarGaussianProcess2D::ComputeOcc(
        const Eigen::Ref<const Eigen::Scalard> &angle_local,
        const double r,
        Eigen::Ref<Eigen::Scalard> range_pred,
        Eigen::Ref<Eigen::Scalard> range_pred_var,
        double &occ) const {

        if (!Test(angle_local, true, range_pred, range_pred_var, false, false)) { return false; }
        if (range_pred_var[0] > m_setting_->max_valid_range_var) { return false; }  // fail to estimate the mapped r f
        // when the r is larger, 1/r results in smaller different, we need a larger m_scale_.
        const double a = r * m_setting_->occ_test_temperature;
        occ = 2. / (1. + std::exp(a * (range_pred[0] - m_mapping_->map(r)))) - 1.;
        range_pred[0] = m_mapping_->inv(range_pred[0]);
        return true;
    }
}  // namespace erl::gaussian_process

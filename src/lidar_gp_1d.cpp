#include "erl_gaussian_process/lidar_gp_1d.hpp"

namespace erl::gaussian_process {
    static inline double
    ClipAngle(double angle) {
        if (angle < -M_PI) {
            auto n = std::floor(std::abs(angle) / M_PI);
            angle += n * M_PI * 2;
        } else if (angle >= M_PI) {
            auto n = std::floor(angle / M_PI);
            angle -= n * M_PI * 2;
        }
        return angle;
    }

    bool
    LidarGaussianProcess1D::TrainBuffer::Store(
        const Eigen::Ref<const Eigen::VectorXd> &vec_new_angles,
        const Eigen::Ref<const Eigen::VectorXd> &vec_new_distances,
        const Eigen::Ref<const Eigen::Matrix23d> &mat_new_pose) {

        // Store sorted original data
        std::vector<int> indices(vec_new_angles.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int i, int j) { return vec_new_angles[i] < vec_new_angles[j]; });
        vec_angles_original = vec_new_angles(indices);
        vec_ranges_original = vec_new_distances(indices);

        // Reset
        auto n = vec_angles_original.size();
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
            double angle = ClipAngle(vec_angles_original[i]);  // make sure angle is within [-pi, pi)
            const double &kRange = vec_ranges_original[i];

            if (std::isnan(kRange) || (kRange < setting->valid_range_min) || (kRange >= setting->valid_range_max)) { continue; }  // valid range: [min, max)

            vec_angles[cnt] = angle;
            vec_ranges[cnt] = kRange;
            vec_mask_hit[i] = true;
            if (kRange > max_distance) { max_distance = kRange; }

            // local frame
            mat_direction_local.col(cnt) << std::cos(angle), std::sin(angle);
            mat_xy_local.col(cnt) = mat_direction_local.col(cnt) * kRange;
            // global frame
            mat_direction_global.col(cnt) = LocalToGlobalSo2(mat_direction_local.col(cnt));
            mat_xy_global.col(cnt) = LocalToGlobalSe2(mat_xy_local.col(cnt));

            cnt++;
        }

        // observed_area = Aabb2D(position, max_distance);

        vec_angles.conservativeResize(cnt);
        vec_ranges.conservativeResize(cnt);
        mat_xy_local.conservativeResize(2, cnt);
        mat_xy_global.conservativeResize(2, cnt);

        vec_mapped_distances = vec_ranges.unaryExpr(mapping->m_map_);
        return cnt > 0;
    }

    std::shared_ptr<LidarGaussianProcess1D>
    LidarGaussianProcess1D::Create(std::shared_ptr<Setting> setting) {
        return std::shared_ptr<LidarGaussianProcess1D>(new LidarGaussianProcess1D(std::move(setting)));
    }

    void
    LidarGaussianProcess1D::Reset() {
        m_trained_ = false;
        m_partitions_.clear();
    }

    void
    LidarGaussianProcess1D::Train(
        const Eigen::Ref<const Eigen::VectorXd> &angles,
        const Eigen::Ref<const Eigen::VectorXd> &distances,
        const Eigen::Ref<const Eigen::Matrix23d> &pose) {

        Reset();

        if (!m_train_buffer_.Store(angles, distances, pose)) {
            ERL_DEBUG("No training data is stored.");
            return;
        }

        long n = m_train_buffer_.Size();
        if (n <= m_setting_->overlap_size) {
            ERL_DEBUG("LidarGaussianProcess1D: no enough samples to perform partition.");
            return;
        }

        auto num_groups = std::max(1l, n / (m_setting_->group_size - m_setting_->overlap_size)) + 1;
        m_setting_->gp->max_num_samples = m_setting_->group_size;  // adjust the max_num_samples
        m_setting_->gp->kernel->x_dim = 1;                         // adjust the x_dim
        m_gps_.resize(num_groups);
        // m_gps_.reserve(num_groups);
        m_partitions_.reserve(num_groups + 1);
        m_partitions_.push_back(m_train_buffer_.vec_angles[0]);
        long half_overlap_size = m_setting_->overlap_size / 2;

        for (int i = 0; i < num_groups - 2; ++i) {
            long index_left = i * (m_setting_->group_size - m_setting_->overlap_size);  // lower bound, included
            long index_right = index_left + m_setting_->group_size;                     // upper bound, not included

            m_partitions_.push_back(m_train_buffer_.vec_angles(index_right - half_overlap_size));
            std::shared_ptr<VanillaGaussianProcess> &gp = m_gps_[i];
            if (gp == nullptr) { gp = std::make_shared<VanillaGaussianProcess>(m_setting_->gp); }
            gp->Reset(m_setting_->group_size, 1);
            // buffer size fits the data exactly
            gp->GetTrainInputSamplesBuffer() = m_train_buffer_.vec_angles.segment(index_left, index_right - index_left).transpose();
            gp->GetTrainOutputSamplesBuffer() = m_train_buffer_.vec_mapped_distances.segment(index_left, index_right - index_left);
            gp->GetTrainOutputSamplesVarianceBuffer().setConstant(m_setting_->sensor_range_var);
            gp->Train(m_setting_->group_size);
        }

        // the last two groups
        long index_left = (num_groups - 2) * (m_setting_->group_size - m_setting_->overlap_size);
        long index_right = index_left + (n - index_left + m_setting_->overlap_size) / 2;
        m_partitions_.push_back(m_train_buffer_.vec_angles(index_right - half_overlap_size));
        {
            std::shared_ptr<VanillaGaussianProcess> &gp = m_gps_[num_groups - 2];
            if (gp == nullptr) { gp = std::make_shared<VanillaGaussianProcess>(m_setting_->gp); }
            long num_samples = index_right - index_left;
            gp->Reset(num_samples, 1);
            gp->GetTrainInputSamplesBuffer().leftCols(num_samples) = m_train_buffer_.vec_angles.segment(index_left, num_samples).transpose();
            gp->GetTrainOutputSamplesBuffer().head(num_samples) = m_train_buffer_.vec_mapped_distances.segment(index_left, num_samples);
            gp->GetTrainOutputSamplesVarianceBuffer().head(num_samples).setConstant(m_setting_->sensor_range_var);
            gp->Train(num_samples);
        }

        index_left = index_left + (n - index_left - m_setting_->overlap_size) / 2;
        index_right = n;
        m_partitions_.push_back(m_train_buffer_.vec_angles(n - 1));
        {
            std::shared_ptr<VanillaGaussianProcess> &gp = m_gps_[num_groups - 1];
            if (gp == nullptr) { gp = std::make_shared<VanillaGaussianProcess>(m_setting_->gp); }
            long num_samples = index_right - index_left;
            gp->Reset(num_samples, 1);
            gp->GetTrainInputSamplesBuffer().leftCols(num_samples) = m_train_buffer_.vec_angles.segment(index_left, num_samples).transpose();
            gp->GetTrainOutputSamplesBuffer().head(num_samples) = m_train_buffer_.vec_mapped_distances.segment(index_left, num_samples);
            gp->GetTrainOutputSamplesVarianceBuffer().head(num_samples).setConstant(m_setting_->sensor_range_var);
            gp->Train(num_samples);
        }

        m_trained_ = true;
    }

    void
    LidarGaussianProcess1D::Test(const Eigen::Ref<const Eigen::VectorXd> &angles, Eigen::Ref<Eigen::VectorXd> fs, Eigen::Ref<Eigen::VectorXd> vars, bool un_map)
        const {

        if (!m_trained_) { return; }
        long n = angles.size();
        ERL_ASSERTM(fs.size() >= n, "fs size = %ld, it should be >= %ld.", fs.size(), n);
        ERL_ASSERTM(vars.size() >= n, "vars size = %ld, it should be >= %ld.", vars.size(), n);

        fs.setZero();
        vars.setConstant(m_setting_->init_variance);

        double boundary_min = m_partitions_.front() + m_setting_->boundary_margin;
        double boundary_max = m_partitions_.back() - m_setting_->boundary_margin;
        for (int i = 0; i < angles.size(); ++i) {
            if ((angles[i] < boundary_min) || (angles[i] > boundary_max)) { continue; }
            for (size_t j = 0; j < m_gps_.size(); ++j) {
                if (angles[i] < m_partitions_[j] || angles[i] > m_partitions_[j + 1]) { continue; }
                if (!m_gps_[j]->IsTrained()) { continue; }
                Eigen::Scalard f, var;
                m_gps_[j]->Test(angles.segment<1>(i), f, var);
                if (un_map) {
                    fs[i] = m_train_buffer_.mapping->m_inv_(f[0]);
                } else {
                    fs[i] = f[0];
                }
                vars[i] = var[0];
                break;
            }
        }
    }

    bool
    LidarGaussianProcess1D::ComputeOcc(
        const Eigen::Ref<const Eigen::Scalard> &angle,
        double r,
        Eigen::Ref<Eigen::Scalard> f,
        Eigen::Ref<Eigen::Scalard> var,
        double &occ) const {

        Test(angle, f, var, false);
        if (var[0] > m_setting_->max_valid_distance_var) { return false; }  // fail to estimate the mapped r f
        // when the r is larger, 1/r results in smaller different, we need a larger m_scale_.
        double a = r * m_setting_->occ_test_temperature;
        occ = 2. / (1. + std::exp(a * (f[0] - m_train_buffer_.mapping->m_map_(r)))) - 1.;
        f[0] = m_train_buffer_.mapping->m_inv_(f[0]);
        return true;
    }

    LidarGaussianProcess1D::LidarGaussianProcess1D(std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)),
          m_train_buffer_(m_setting_->train_buffer) {}
}  // namespace erl::gaussian_process

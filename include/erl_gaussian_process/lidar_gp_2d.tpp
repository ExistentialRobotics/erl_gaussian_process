#pragma once

#include "erl_common/serialization.hpp"

namespace erl::gaussian_process {
    template<typename Dtype>
    YAML::Node
    LidarGaussianProcess2D<Dtype>::Setting::YamlConvertImpl::encode(const Setting &setting) {
        YAML::Node node;
        ERL_YAML_SAVE_ATTR(node, setting, partition_on_hit_rays);
        ERL_YAML_SAVE_ATTR(node, setting, symmetric_partitions);
        ERL_YAML_SAVE_ATTR(node, setting, group_size);
        ERL_YAML_SAVE_ATTR(node, setting, overlap_size);
        ERL_YAML_SAVE_ATTR(node, setting, margin);
        ERL_YAML_SAVE_ATTR(node, setting, init_variance);
        ERL_YAML_SAVE_ATTR(node, setting, sensor_range_var);
        ERL_YAML_SAVE_ATTR(node, setting, max_valid_range_var);
        ERL_YAML_SAVE_ATTR(node, setting, occ_test_temperature);
        ERL_YAML_SAVE_ATTR(node, setting, sensor_frame);
        ERL_YAML_SAVE_ATTR(node, setting, gp);
        ERL_YAML_SAVE_ATTR(node, setting, mapping);
        return node;
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::Setting::YamlConvertImpl::decode(
        const YAML::Node &node,
        Setting &setting) {
        if (!node.IsMap()) { return false; }
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, partition_on_hit_rays, bool);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, symmetric_partitions, bool);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, group_size, long);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, overlap_size, long);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, margin, long);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, init_variance, Dtype);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, sensor_range_var, Dtype);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, max_valid_range_var, Dtype);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, occ_test_temperature, Dtype);
        ERL_YAML_LOAD_ATTR(node, setting, sensor_frame);
        ERL_YAML_LOAD_ATTR(node, setting, gp);
        ERL_YAML_LOAD_ATTR(node, setting, mapping);
        return true;
    }

    template<typename Dtype>
    LidarGaussianProcess2D<Dtype>::LidarGaussianProcess2D(std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)),
          m_mapping_(MappingDtype::Create(m_setting_->mapping)) {
        m_sensor_frame_ = std::make_shared<LidarFrame2D>(m_setting_->sensor_frame);

        const VectorX &angles = m_sensor_frame_->GetAnglesInFrame();
        const long n = angles.rows();
        if (n <= m_setting_->overlap_size) {
            ERL_DEBUG("LidarGaussianProcess2D: no enough samples to perform partition.");
            return;
        }

        if (!m_setting_->partition_on_hit_rays) { PartitionOnAngles(); }
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::IsTrained() const {
        return m_trained_;
    }

    template<typename Dtype>
    std::shared_ptr<typename LidarGaussianProcess2D<Dtype>::Setting>
    LidarGaussianProcess2D<Dtype>::GetSetting() const {
        return m_setting_;
    }

    template<typename Dtype>
    const std::vector<std::shared_ptr<typename LidarGaussianProcess2D<Dtype>::Gp>> &
    LidarGaussianProcess2D<Dtype>::GetGps() const {
        return m_gps_;
    }

    template<typename Dtype>
    const std::vector<std::tuple<long, long, Dtype, Dtype>> &
    LidarGaussianProcess2D<Dtype>::GetAnglePartitions() const {
        return m_angle_partitions_;
    }

    template<typename Dtype>
    std::shared_ptr<const typename LidarGaussianProcess2D<Dtype>::LidarFrame2D>
    LidarGaussianProcess2D<Dtype>::GetSensorFrame() const {
        return m_sensor_frame_;
    }

    template<typename Dtype>
    typename LidarGaussianProcess2D<Dtype>::Vector2
    LidarGaussianProcess2D<Dtype>::GlobalToLocalSo2(const Vector2 &dir_global) const {
        return m_sensor_frame_->DirWorldToFrame(dir_global);
    }

    template<typename Dtype>
    typename LidarGaussianProcess2D<Dtype>::Vector2
    LidarGaussianProcess2D<Dtype>::LocalToGlobalSo2(const Vector2 &dir_local) const {
        return m_sensor_frame_->DirFrameToWorld(dir_local);
    }

    template<typename Dtype>
    typename LidarGaussianProcess2D<Dtype>::Vector2
    LidarGaussianProcess2D<Dtype>::GlobalToLocalSe2(const Vector2 &xy_global) const {
        return m_sensor_frame_->PosWorldToFrame(xy_global);
    }

    template<typename Dtype>
    typename LidarGaussianProcess2D<Dtype>::Vector2
    LidarGaussianProcess2D<Dtype>::LocalToGlobalSe2(const Vector2 &xy_local) const {
        return m_sensor_frame_->PosFrameToWorld(xy_local);
    }

    template<typename Dtype>
    void
    LidarGaussianProcess2D<Dtype>::Reset() {
        m_trained_ = false;
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::StoreData(
        const Matrix2 &rotation,
        const Vector2 &translation,
        VectorX ranges) {
        m_sensor_frame_->UpdateRanges(rotation, translation, std::move(ranges), false);
        m_mapped_distances_ = m_sensor_frame_->GetRanges().unaryExpr(m_mapping_->map);
        return m_sensor_frame_->IsValid();
    }

    template<typename Dtype>
    void
    LidarGaussianProcess2D<Dtype>::PartitionOnAngles() {
        const VectorX &angles = m_sensor_frame_->GetAnglesInFrame();
        const long n = angles.size();
        const long gs = m_setting_->group_size;
        const long step = m_setting_->group_size - m_setting_->overlap_size;
        const long num_groups = std::max(1l, n / step) + 1;
        const long gs2 = (n - (num_groups - 2) * step) / 2;
        const long half_overlap = m_setting_->overlap_size / 2;

        m_setting_->gp->max_num_samples = m_setting_->group_size;  // adjust the max_num_samples
        m_setting_->gp->kernel->x_dim = 1;                         // adjust the x_dim
        m_gps_.resize(num_groups);
        m_angle_partitions_.clear();
        m_angle_partitions_.reserve(num_groups);

        if (m_setting_->symmetric_partitions) {
            // first partition
            m_angle_partitions_.emplace_back(
                0,                           // index left
                gs2 + half_overlap,          // index right
                angles[m_setting_->margin],  // coord left
                angles[gs2]);                // coord right
            // middle partitions
            for (long i = 0; i < num_groups - 2; ++i) {
                const long index_left = i * step + gs2 - half_overlap;
                const long index_right = index_left + gs;
                const Dtype coord_left = angles[index_left + half_overlap];
                const Dtype coord_right = angles[index_right - half_overlap];
                m_angle_partitions_.emplace_back(index_left, index_right, coord_left, coord_right);
            }
            // last partition
            m_angle_partitions_.emplace_back(
                n - gs2 - half_overlap,
                n,
                angles[n - 1 - gs2],
                angles[n - 1 - m_setting_->margin]);
            return;
        }

        for (int i = 0; i < num_groups - 2; ++i) {
            long index_left = i * step;                              // lower bound, included
            long index_right = index_left + m_setting_->group_size;  // upper bound, not included
            m_angle_partitions_.emplace_back(
                index_left,
                index_right,
                angles[index_left],
                angles[index_right - half_overlap]);
        }
        // the last two groups
        long index_left = (num_groups - 2) * step;
        long index_right = index_left + (n - index_left + m_setting_->overlap_size) / 2;
        m_angle_partitions_.emplace_back(
            index_left,
            index_right,
            angles[index_left],
            angles[index_right - half_overlap]);
        index_left = index_left + (n - index_left - m_setting_->overlap_size) / 2;
        index_right = n;
        m_angle_partitions_
            .emplace_back(index_left, index_right, angles[index_left], angles[index_right - 1]);
    }

    template<typename Dtype>
    void
    LidarGaussianProcess2D<Dtype>::PartitionOnHitRays() {
        const VectorX &angles = m_sensor_frame_->GetAnglesInFrame();
        const long n = m_sensor_frame_->GetNumHitRays();
        if (n == 0) {
            ERL_WARN("No hit rays are stored.");
            return;
        }
        const long step = m_setting_->group_size - m_setting_->overlap_size;
        const long num_groups = std::max(1l, n / step) + 1;
        const std::vector<long> &hit_ray_indices = m_sensor_frame_->GetHitRayIndices();

        m_setting_->gp->max_num_samples = m_setting_->group_size;  // adjust the max_num_samples
        m_setting_->gp->kernel->x_dim = 1;                         // adjust the x_dim
        m_gps_.resize(num_groups);
        m_angle_partitions_.clear();
        m_angle_partitions_.reserve(num_groups);

        if (m_setting_->symmetric_partitions) {
            ERL_WARN("Symmetric partition is not implemented yet. Asymmetric partition is used.");
        }

        for (int i = 0; i < num_groups - 2; ++i) {
            long index_left = i * step;                              // lower bound, included
            long index_right = index_left + m_setting_->group_size;  // upper bound, not included
            // hit ray indices to original ray indices
            index_left = hit_ray_indices[index_left];
            index_right = hit_ray_indices[index_right];
            m_angle_partitions_
                .emplace_back(index_left, index_right, angles[index_left], angles[index_right]);
        }

        // the last two groups
        long index_left = (num_groups - 2) * step;
        long index_right = index_left + (n - index_left + m_setting_->overlap_size) / 2;
        index_left = hit_ray_indices[index_left];
        index_right = hit_ray_indices[index_right];
        m_angle_partitions_
            .emplace_back(index_left, index_right, angles[index_left], angles[index_right]);

        index_left = index_left + (n - index_left - m_setting_->overlap_size) / 2;
        index_left = hit_ray_indices[index_left];
        index_right = hit_ray_indices[n - 1] + 1;  // upper bound, not included
        m_angle_partitions_
            .emplace_back(index_left, index_right, angles[index_left], angles[index_right]);
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::Train(
        const Matrix2 &rotation,
        const Vector2 &translation,
        VectorX ranges) {

        Reset();

        if (!StoreData(rotation, translation, std::move(ranges))) {
            ERL_DEBUG("No training data is stored.");
            return false;
        }

        if (m_setting_->partition_on_hit_rays) { PartitionOnHitRays(); }

#pragma omp parallel for default(none)
        for (long i = 0; i < static_cast<long>(m_angle_partitions_.size()); ++i) {
            const auto &[index_left, index_right, coord_left, coord_right] = m_angle_partitions_[i];
            std::shared_ptr<Gp> &gp = m_gps_[i];
            if (gp == nullptr) { gp = std::make_shared<Gp>(m_setting_->gp); }
            gp->Reset(m_setting_->gp->max_num_samples, 1, 1);
            long cnt = 0;
            auto &train_set = gp->GetTrainSet();
            const Eigen::VectorXb &mask_hit = m_sensor_frame_->GetHitMask();
            const VectorX &angles = m_sensor_frame_->GetAnglesInFrame();
            for (long j = index_left; j < index_right; ++j) {
                if (!mask_hit[j]) { continue; }
                train_set.x(0, cnt) = angles[j];
                train_set.y.col(0)[cnt] = m_mapped_distances_[j];
                train_set.var[cnt] = m_setting_->sensor_range_var;
                ++cnt;
            }
            train_set.num_samples = cnt;
            if (cnt > 0) { (void) gp->Train(); }
        }

        m_trained_ = true;
        return true;
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::Test(
        const Eigen::Ref<const VectorX> &angles,
        const bool angles_are_local,
        Eigen::Ref<VectorX> vec_ranges,
        Eigen::Ref<VectorX> vec_ranges_var,
        const bool un_map) const {

        if (!m_trained_) { return false; }

        const long n = angles.size();
        ERL_DEBUG_ASSERT(
            vec_ranges.size() >= n,
            "vec_ranges size = {}, it should be >= {}.",
            vec_ranges.size(),
            n);

        vec_ranges.setZero();
        const bool compute_var = vec_ranges_var.size() > 0;
        if (compute_var) {
            ERL_DEBUG_ASSERT(
                vec_ranges_var.size() >= n,
                "vec_ranges_var size = {}, it should be >= {}.",
                vec_ranges_var.size(),
                n);
            vec_ranges_var.setConstant(m_setting_->init_variance);
        }

        for (int i = 0; i < n; ++i) {
            Scalar angle_local;
            angle_local[0] = angles[i];
            if (!angles_are_local) {
                const Vector2 direction_local = m_sensor_frame_->DirWorldToFrame(
                    {std::cos(angle_local[0]), std::sin(angle_local[0])});
                angle_local[0] = std::atan2(direction_local[1], direction_local[0]);
            }
            long partition_index = 0;
            for (; partition_index < static_cast<long>(m_angle_partitions_.size());
                 ++partition_index) {
                if (const auto &[index_left, index_right, coord_left, coord_right] =
                        m_angle_partitions_[partition_index];
                    angle_local[0] >= coord_left && angle_local[0] <= coord_right) {
                    break;
                }
            }
            if (partition_index >= static_cast<long>(m_angle_partitions_.size())) { continue; }

            const auto gp = m_gps_[partition_index];
            if (!gp->IsTrained()) { continue; }
            Scalar f, var;
            if (VectorX no_var; compute_var ? !gp->Test(angle_local, {0}, f, var)
                                            : !gp->Test(angle_local, {0}, f, no_var)) {
                continue;
            }
            vec_ranges[i] = un_map ? m_mapping_->inv(f[0]) : f[0];
            if (compute_var) { vec_ranges_var[i] = var[0]; }
        }
        return true;
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::ComputeOcc(
        const Eigen::Ref<const Scalar> &angle_local,
        const Dtype r,
        Eigen::Ref<Scalar> range_pred,
        Eigen::Ref<Scalar> range_pred_var,
        Dtype &occ) const {

        if (!Test(angle_local, true, range_pred, range_pred_var, false)) { return false; }
        if (range_pred_var[0] > m_setting_->max_valid_range_var) {
            return false;  // fail to estimate the mapped r f
        }
        // when the r is larger, 1/r results in a smaller difference. we need a larger scale.
        const Dtype a = r * m_setting_->occ_test_temperature;
        occ = 2. / (1. + std::exp(a * (range_pred[0] - m_mapping_->map(r)))) - 1.;
        range_pred[0] = m_mapping_->inv(range_pred[0]);
        return true;
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::operator==(const LidarGaussianProcess2D &other) const {
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr &&
            (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) {
            return false;
        }
        if (m_trained_ != other.m_trained_) { return false; }
        if (m_gps_.size() != other.m_gps_.size()) { return false; }
        for (std::size_t i = 0; i < m_gps_.size(); ++i) {
            if (m_gps_[i] == nullptr && other.m_gps_[i] != nullptr) { return false; }
            if (m_gps_[i] != nullptr &&
                (other.m_gps_[i] == nullptr || *m_gps_[i] != *other.m_gps_[i])) {
                return false;
            }
        }
        if (m_angle_partitions_.size() != other.m_angle_partitions_.size()) { return false; }
        for (std::size_t i = 0; i < m_angle_partitions_.size(); ++i) {
            if (m_angle_partitions_[i] != other.m_angle_partitions_[i]) { return false; }
        }
        if (m_sensor_frame_ == nullptr && other.m_sensor_frame_ != nullptr) { return false; }
        if (m_sensor_frame_ != nullptr &&
            (other.m_sensor_frame_ == nullptr || *m_sensor_frame_ != *other.m_sensor_frame_)) {
            return false;
        }
        if (m_mapped_distances_.size() != other.m_mapped_distances_.size() ||
            std::memcmp(
                m_mapped_distances_.data(),
                other.m_mapped_distances_.data(),
                m_mapped_distances_.size() * sizeof(Dtype)) != 0) {
            return false;
        }
        return true;
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::operator!=(const LidarGaussianProcess2D &other) const {
        return !(*this == other);
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::Write(std::ostream &s) const {
        static const std::vector<std::pair<
            const char *,
            std::function<bool(const LidarGaussianProcess2D *, std::ostream &)>>>
            token_function_pairs = {
                {
                    "setting",
                    [](const LidarGaussianProcess2D *gp, std::ostream &stream) {
                        return gp->m_setting_->Write(stream) && stream.good();
                    },
                },
                {
                    "trained",
                    [](const LidarGaussianProcess2D *gp, std::ostream &stream) {
                        stream << gp->m_trained_;
                        return stream.good();
                    },
                },
                {
                    "gps",
                    [](const LidarGaussianProcess2D *gp, std::ostream &stream) {
                        stream << gp->m_gps_.size() << '\n';
                        for (const auto &g: gp->m_gps_) {
                            char has_gp = static_cast<char>(g != nullptr);
                            stream.write(&has_gp, sizeof(char));
                            if (!has_gp) { continue; }
                            if (!g->Write(stream)) { return false; }
                        }
                        return stream.good();
                    },
                },
                {
                    "angle_partitions",
                    [](const LidarGaussianProcess2D *gp, std::ostream &stream) {
                        stream << gp->m_angle_partitions_.size() << '\n';
                        for (const auto &[index_left, index_right, coord_left, coord_right]:
                             gp->m_angle_partitions_) {
                            stream.write(
                                reinterpret_cast<const char *>(&index_left),
                                sizeof(index_left));
                            stream.write(
                                reinterpret_cast<const char *>(&index_right),
                                sizeof(index_right));
                            stream.write(
                                reinterpret_cast<const char *>(&coord_left),
                                sizeof(coord_left));
                            stream.write(
                                reinterpret_cast<const char *>(&coord_right),
                                sizeof(coord_right));
                        }
                        return stream.good();
                    },
                },
                {
                    "sensor_frame",
                    [](const LidarGaussianProcess2D *gp, std::ostream &stream) {
                        return gp->m_sensor_frame_->Write(stream) && stream.good();
                    },
                },
                {
                    "mapped_distances",
                    [](const LidarGaussianProcess2D *gp, std::ostream &stream) {
                        return common::SaveEigenMatrixToBinaryStream(
                                   stream,
                                   gp->m_mapped_distances_) &&
                               stream.good();
                    },
                },
            };
        return common::WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::Read(std::istream &s) {
        static const std::vector<
            std::pair<const char *, std::function<bool(LidarGaussianProcess2D *, std::istream &)>>>
            token_function_pairs = {
                {
                    "setting",
                    [](LidarGaussianProcess2D *gp, std::istream &stream) -> bool {
                        return gp->m_setting_->Read(stream) && stream.good();
                    },
                },
                {
                    "trained",
                    [](LidarGaussianProcess2D *gp, std::istream &stream) -> bool {
                        stream >> gp->m_trained_;
                        return stream.good();
                    },
                },
                {
                    "gps",
                    [](LidarGaussianProcess2D *gp, std::istream &stream) -> bool {
                        long num_gps;
                        stream >> num_gps;
                        gp->m_gps_.resize(num_gps, nullptr);
                        common::SkipLine(stream);
                        char has_gp;
                        for (long i = 0; i < num_gps; ++i) {
                            stream.read(&has_gp, sizeof(char));
                            if (!has_gp) { continue; }
                            gp->m_gps_[i] = std::make_shared<Gp>(gp->m_setting_->gp);
                            if (!gp->m_gps_[i]->Read(stream)) { return false; }
                        }
                        return stream.good();
                    },
                },
                {
                    "angle_partitions",
                    [](LidarGaussianProcess2D *gp, std::istream &stream) -> bool {
                        long num_partitions;
                        stream >> num_partitions;
                        gp->m_angle_partitions_.resize(num_partitions);
                        common::SkipLine(stream);
                        for (long i = 0; i < num_partitions; ++i) {
                            auto &[index_left, index_right, coord_left, coord_right] =
                                gp->m_angle_partitions_[i];
                            stream.read(reinterpret_cast<char *>(&index_left), sizeof(index_left));
                            stream.read(
                                reinterpret_cast<char *>(&index_right),
                                sizeof(index_right));
                            stream.read(reinterpret_cast<char *>(&coord_left), sizeof(coord_left));
                            stream.read(
                                reinterpret_cast<char *>(&coord_right),
                                sizeof(coord_right));
                        }
                        return stream.good();
                    },
                },
                {
                    "sensor_frame",
                    [](LidarGaussianProcess2D *gp, std::istream &stream) -> bool {
                        common::SkipLine(stream);
                        gp->m_sensor_frame_ =
                            std::make_shared<LidarFrame2D>(gp->m_setting_->sensor_frame);
                        return gp->m_sensor_frame_->Read(stream) && stream.good();
                    },
                },
                {
                    "mapped_distances",
                    [](LidarGaussianProcess2D *gp, std::istream &stream) -> bool {
                        common::SkipLine(stream);
                        return common::LoadEigenMatrixFromBinaryStream(
                                   stream,
                                   gp->m_mapped_distances_) &&
                               stream.good();
                    },
                },
            };
        return common::ReadTokens(s, this, token_function_pairs);
    }
}  // namespace erl::gaussian_process

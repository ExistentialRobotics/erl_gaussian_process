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
        ERL_YAML_SAVE_ATTR(node, setting, discontinuity_var);
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
        ERL_YAML_LOAD_ATTR(node, setting, partition_on_hit_rays);
        ERL_YAML_LOAD_ATTR(node, setting, symmetric_partitions);
        ERL_YAML_LOAD_ATTR(node, setting, group_size);
        ERL_YAML_LOAD_ATTR(node, setting, overlap_size);
        ERL_YAML_LOAD_ATTR(node, setting, margin);
        ERL_YAML_LOAD_ATTR(node, setting, init_variance);
        ERL_YAML_LOAD_ATTR(node, setting, sensor_range_var);
        ERL_YAML_LOAD_ATTR(node, setting, discontinuity_var);
        ERL_YAML_LOAD_ATTR(node, setting, max_valid_range_var);
        ERL_YAML_LOAD_ATTR(node, setting, occ_test_temperature);
        ERL_YAML_LOAD_ATTR(node, setting, sensor_frame);
        ERL_YAML_LOAD_ATTR(node, setting, gp);
        ERL_YAML_LOAD_ATTR(node, setting, mapping);
        return true;
    }

    template<typename Dtype>
    LidarGaussianProcess2D<Dtype>::TestResult::TestResult(
        const LidarGaussianProcess2D *gp,
        const Eigen::Ref<const VectorX> &angles,
        const bool angles_are_local,
        const std::shared_ptr<MappingDtype> &mapping)
        : m_gp_(NotNull(gp, true, "gp = nullptr.")),
          m_mapping_(mapping) {

        ERL_DEBUG_ASSERT(m_gp_->IsTrained(), "The model has not been trained.");

        m_reduced_rank_kernel_ = m_gp_->m_gps_[0]->UsingReducedRankKernel();
        const long n = angles.size();
        m_gps_.resize(n, nullptr);
        m_k_test_vec_.resize(n);
        m_alpha_vec_.resize(n);
        m_alpha_test_vec_.resize(n);

        LidarFrame2D *sensor_frame = m_gp_->m_sensor_frame_.get();
        auto &gps = m_gp_->m_gps_;

        for (long i = 0; i < n; ++i) {
            Scalar angle_local;
            angle_local[0] = angles[i];
            if (!angles_are_local) {
                const Vector2 direction_local = sensor_frame->DirWorldToFrame(
                    {std::cos(angle_local[0]), std::sin(angle_local[0])});
                angle_local[0] = std::atan2(direction_local[1], direction_local[0]);
            }
            const long idx = m_gp_->SearchPartition(angle_local[0]);
            if (idx < 0) { continue; }
            const Gp *partition_gp = gps[idx].get();
            if (!partition_gp->IsTrained()) { continue; }
            MatrixX ktest;
            const bool success = partition_gp->ComputeKtest(angle_local, ktest);
            (void) success;
            ERL_DEBUG_ASSERT(success, "Failed to compute ktest.");
            m_gps_[i] = partition_gp;
            m_k_test_vec_[i] = ktest;
            m_alpha_vec_[i] = {partition_gp->GetAlpha().col(0).data(), ktest.rows()};
        }
    }

    template<typename Dtype>
    long
    LidarGaussianProcess2D<Dtype>::TestResult::GetNumTest() const {
        return static_cast<long>(m_k_test_vec_.size());
    }

    template<typename Dtype>
    const typename LidarGaussianProcess2D<Dtype>::VectorX &
    LidarGaussianProcess2D<Dtype>::TestResult::GetKtest(const long index) const {
        return m_k_test_vec_[index];
    }

    template<typename Dtype>
    Eigen::VectorXb
    LidarGaussianProcess2D<Dtype>::TestResult::GetMean(
        Eigen::Ref<VectorX> vec_f_out,
        const bool parallel) const {
        (void) parallel;
        const auto n = static_cast<long>(m_k_test_vec_.size());
        Dtype *f = vec_f_out.data();
        Eigen::VectorXb valid(n);
#pragma omp parallel for if (parallel) default(none) shared(n, f, valid)
        for (long i = 0; i < n; ++i) { valid[i] = GetMean(i, f[i]); }
        return valid;
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::TestResult::GetMean(const long index, Dtype &f) const {
        const VectorX &ktest = m_k_test_vec_[index];
        if (ktest.size() == 0) { return false; }  // no valid test result
        const auto &[alpha_ptr, alpha_size] = m_alpha_vec_[index];
        Eigen::Map<const VectorX> alpha(alpha_ptr, alpha_size);
        f = ktest.dot(alpha);  // h(x_test)
        if (m_mapping_ != nullptr) { f = m_mapping_->inv(f); }
        return true;
    }

    template<typename Dtype>
    Eigen::VectorXb
    LidarGaussianProcess2D<Dtype>::TestResult::GetVariance(
        Eigen::Ref<VectorX> vec_var_out,
        const bool parallel) const {
        (void) parallel;
        const auto n = static_cast<long>(m_k_test_vec_.size());
        Dtype *var = vec_var_out.data();
        Eigen::VectorXb valid(n);
#pragma omp parallel for if (parallel) default(none) shared(n, var, valid)
        for (long i = 0; i < n; ++i) { valid[i] = GetVariance(i, var[i]); }
        return valid;
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::TestResult::GetVariance(const long index, Dtype &var) const {
        const_cast<TestResult *>(this)->PrepareAlphaTest(index);
        auto &alpha_test = m_alpha_test_vec_[index];
        if (alpha_test.size() == 0) { return false; }  // no valid test result
        var = alpha_test.squaredNorm();
        if (m_reduced_rank_kernel_) { return true; }
        var = 1.0f - var;  // variance of h(x_test)
        return true;
    }

    template<typename Dtype>
    void
    LidarGaussianProcess2D<Dtype>::TestResult::PrepareAlphaTest(const long index) {
        const Gp *gp = m_gps_[index];
        if (gp == nullptr) { return; }

        VectorX &alpha_test = m_alpha_test_vec_[index];
        if (alpha_test.size() > 0) { return; }

        const VectorX &ktest = m_k_test_vec_[index];
        const long rows = ktest.rows();
        const auto &mat_l = gp->GetCholeskyDecomposition().topLeftCorner(rows, rows);
        alpha_test = mat_l.template triangularView<Eigen::Lower>().solve(ktest);
    }

    template<typename Dtype>
    LidarGaussianProcess2D<Dtype>::LidarGaussianProcess2D(std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)),
          m_sensor_frame_(std::make_shared<LidarFrame2D>(m_setting_->sensor_frame)),
          m_mapping_(MappingDtype::Create(m_setting_->mapping)) {

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
    std::shared_ptr<const typename LidarGaussianProcess2D<Dtype>::MappingDtype>
    LidarGaussianProcess2D<Dtype>::GetMapping() const {
        return m_mapping_;
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
        m_sensor_frame_->UpdateRanges(rotation, translation, std::move(ranges));
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
            const Eigen::VectorXb &mask_con = m_sensor_frame_->GetContinuityMask();
            const bool discon_detection = m_setting_->sensor_frame->discontinuity_detection;
            const Dtype discon_var = m_setting_->discontinuity_var;
            const VectorX &angles = m_sensor_frame_->GetAnglesInFrame();
            for (long j = index_left; j < index_right; ++j) {
                if (!mask_hit[j]) { continue; }
                train_set.x(0, cnt) = angles[j];
                train_set.y.col(0)[cnt] = m_mapped_distances_[j];
                if (discon_detection && !mask_con[j]) {
                    train_set.var[cnt] = discon_var;
                } else {
                    train_set.var[cnt] = m_setting_->sensor_range_var;
                }
                ++cnt;
            }
            train_set.num_samples = cnt;
            if (cnt > 0) { (void) gp->Train(); }
        }

        m_trained_ = true;
        return true;
    }

    template<typename Dtype>
    long
    LidarGaussianProcess2D<Dtype>::SearchPartition(const Dtype angle_local) const {
        long idx = 0;
        const long n = m_angle_partitions_.size();
        for (; idx < n; ++idx) {
            if (auto &[idx_left, idx_right, coord_left, coord_right] = m_angle_partitions_[idx];
                angle_local >= coord_left && angle_local <= coord_right) {
                return idx;
            }
        }
        return -1;
    }

    template<typename Dtype>
    std::shared_ptr<typename LidarGaussianProcess2D<Dtype>::TestResult>
    LidarGaussianProcess2D<Dtype>::Test(
        const Eigen::Ref<const VectorX> &angles,
        const bool angles_are_local,
        const bool un_map) const {

        if (!m_trained_) { return nullptr; }
        return std::make_shared<TestResult>(
            this,
            angles,
            angles_are_local,
            un_map ? m_mapping_ : nullptr);
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::ComputeOcc(
        const Scalar &angle_local,
        const Dtype r,
        Dtype &range_pred,
        Dtype &occ) const {

        if (!m_trained_) { return false; }
        const long idx = SearchPartition(angle_local[0]);
        if (idx < 0) { return false; }
        const Gp &gp = *m_gps_[idx];
        if (!gp.IsTrained()) { return false; }
        typename Gp::TestResult test_result = *gp.Test(angle_local);

        // check the validity of range_pred first.
        // if it is not valid, we can save the cost of computing range_pred.
        Dtype range_pred_var;
        test_result.GetVariance(0, range_pred_var);
        if (range_pred_var > m_setting_->max_valid_range_var) { return false; }

        test_result.GetMean(0, 0, range_pred);
        const Dtype a = r * m_setting_->occ_test_temperature;
        occ = 2.0f / (1.0f + std::exp(a * (range_pred - m_mapping_->map(r)))) - 1.0f;
        range_pred = m_mapping_->inv(range_pred);
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
        if (m_angle_partitions_ != other.m_angle_partitions_) { return false; }
        if (m_sensor_frame_ == nullptr && other.m_sensor_frame_ != nullptr) { return false; }
        if (m_sensor_frame_ != nullptr &&
            (other.m_sensor_frame_ == nullptr || *m_sensor_frame_ != *other.m_sensor_frame_)) {
            return false;
        }
        using namespace common;
        if (!SafeEigenMatrixEqual(m_mapped_distances_, other.m_mapped_distances_)) { return false; }
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
        using namespace common;
        static const TokenWriteFunctionPairs<LidarGaussianProcess2D> token_function_pairs = {
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
                    return SaveEigenMatrixToBinaryStream(stream, gp->m_mapped_distances_) &&
                           stream.good();
                },
            },
        };
        return WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::Read(std::istream &s) {
        using namespace common;
        static const TokenReadFunctionPairs<LidarGaussianProcess2D> token_function_pairs = {
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
                    SkipLine(stream);
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
                    SkipLine(stream);
                    for (long i = 0; i < num_partitions; ++i) {
                        auto &[index_left, index_right, coord_left, coord_right] =
                            gp->m_angle_partitions_[i];
                        stream.read(reinterpret_cast<char *>(&index_left), sizeof(index_left));
                        stream.read(reinterpret_cast<char *>(&index_right), sizeof(index_right));
                        stream.read(reinterpret_cast<char *>(&coord_left), sizeof(coord_left));
                        stream.read(reinterpret_cast<char *>(&coord_right), sizeof(coord_right));
                    }
                    return stream.good();
                },
            },
            {
                "sensor_frame",
                [](LidarGaussianProcess2D *gp, std::istream &stream) -> bool {
                    gp->m_sensor_frame_ =
                        std::make_shared<LidarFrame2D>(gp->m_setting_->sensor_frame);
                    return gp->m_sensor_frame_->Read(stream) && stream.good();
                },
            },
            {
                "mapped_distances",
                [](LidarGaussianProcess2D *gp, std::istream &stream) -> bool {
                    return LoadEigenMatrixFromBinaryStream(stream, gp->m_mapped_distances_) &&
                           stream.good();
                },
            },
        };
        return ReadTokens(s, this, token_function_pairs);
    }
}  // namespace erl::gaussian_process

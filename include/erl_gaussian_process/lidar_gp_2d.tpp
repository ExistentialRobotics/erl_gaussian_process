#pragma once

#include "erl_common/serialization.hpp"

namespace erl::gaussian_process {
    template<typename Dtype>
    YAML::Node
    LidarGaussianProcess2D<Dtype>::Setting::YamlConvertImpl::encode(const Setting &setting) {
        YAML::Node node;
        node["partition_on_hit_rays"] = setting.partition_on_hit_rays;
        node["symmetric_partitions"] = setting.symmetric_partitions;
        node["group_size"] = setting.group_size;
        node["overlap_size"] = setting.overlap_size;
        node["margin"] = setting.margin;
        node["init_variance"] = setting.init_variance;
        node["sensor_range_var"] = setting.sensor_range_var;
        node["max_valid_range_var"] = setting.max_valid_range_var;
        node["occ_test_temperature"] = setting.occ_test_temperature;
        node["sensor_frame"] = setting.sensor_frame;
        node["gp"] = setting.gp;
        node["mapping"] = setting.mapping;
        return node;
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::Setting::YamlConvertImpl::decode(const YAML::Node &node, Setting &setting) {
        if (!node.IsMap()) { return false; }
        setting.partition_on_hit_rays = node["partition_on_hit_rays"].as<bool>();
        setting.symmetric_partitions = node["symmetric_partitions"].as<bool>();
        setting.group_size = node["group_size"].as<long>();
        setting.overlap_size = node["overlap_size"].as<long>();
        setting.margin = node["margin"].as<long>();
        setting.init_variance = node["init_variance"].as<Dtype>();
        setting.sensor_range_var = node["sensor_range_var"].as<Dtype>();
        setting.max_valid_range_var = node["max_valid_range_var"].as<Dtype>();
        setting.occ_test_temperature = node["occ_test_temperature"].as<Dtype>();
        setting.sensor_frame = node["sensor_frame"].as<std::shared_ptr<typename LidarFrame2D::Setting>>();
        setting.gp = node["gp"].as<std::shared_ptr<typename Gp::Setting>>();
        setting.mapping = node["mapping"].as<std::shared_ptr<typename MappingDtype::Setting>>();
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
    LidarGaussianProcess2D<Dtype>::StoreData(const Matrix2 &rotation, const Vector2 &translation, VectorX ranges) {
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
            m_angle_partitions_.emplace_back(0, gs2 + half_overlap, angles[m_setting_->margin], angles[gs2]);
            // middle partitions
            for (long i = 0; i < num_groups - 2; ++i) {
                const long index_left = i * step + gs2 - half_overlap;
                const long index_right = index_left + gs;
                const Dtype coord_left = angles[index_left + half_overlap];
                const Dtype coord_right = angles[index_right - half_overlap];
                m_angle_partitions_.emplace_back(index_left, index_right, coord_left, coord_right);
            }
            // last partition
            m_angle_partitions_.emplace_back(n - gs2 - half_overlap, n, angles[n - 1 - gs2], angles[n - 1 - m_setting_->margin]);
            return;
        }

        for (int i = 0; i < num_groups - 2; ++i) {
            long index_left = i * step;                              // lower bound, included
            long index_right = index_left + m_setting_->group_size;  // upper bound, not included
            m_angle_partitions_.emplace_back(index_left, index_right, angles[index_left], angles[index_right - half_overlap]);
        }
        // the last two groups
        long index_left = (num_groups - 2) * step;
        long index_right = index_left + (n - index_left + m_setting_->overlap_size) / 2;
        m_angle_partitions_.emplace_back(index_left, index_right, angles[index_left], angles[index_right - half_overlap]);
        index_left = index_left + (n - index_left - m_setting_->overlap_size) / 2;
        index_right = n;
        m_angle_partitions_.emplace_back(index_left, index_right, angles[index_left], angles[index_right - 1]);
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

        if (m_setting_->symmetric_partitions) { ERL_WARN("Symmetric partition is not implemented yet. Asymmetric partition is used."); }

        for (int i = 0; i < num_groups - 2; ++i) {
            long index_left = i * step;                              // lower bound, included
            long index_right = index_left + m_setting_->group_size;  // upper bound, not included
            // hit ray indices to original ray indices
            index_left = hit_ray_indices[index_left];
            index_right = hit_ray_indices[index_right];
            m_angle_partitions_.emplace_back(index_left, index_right, angles[index_left], angles[index_right]);
        }

        // the last two groups
        long index_left = (num_groups - 2) * step;
        long index_right = index_left + (n - index_left + m_setting_->overlap_size) / 2;
        index_left = hit_ray_indices[index_left];
        index_right = hit_ray_indices[index_right];
        m_angle_partitions_.emplace_back(index_left, index_right, angles[index_left], angles[index_right]);

        index_left = index_left + (n - index_left - m_setting_->overlap_size) / 2;
        index_left = hit_ray_indices[index_left];
        index_right = hit_ray_indices[n - 1] + 1;  // upper bound, not included
        m_angle_partitions_.emplace_back(index_left, index_right, angles[index_left], angles[index_right]);
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::Train(const Matrix2 &rotation, const Vector2 &translation, VectorX ranges) {

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
        ERL_DEBUG_ASSERT(vec_ranges.size() >= n, "vec_ranges size = {}, it should be >= {}.", vec_ranges.size(), n);

        vec_ranges.setZero();
        const bool compute_var = vec_ranges_var.size() > 0;
        if (compute_var) {
            ERL_DEBUG_ASSERT(vec_ranges_var.size() >= n, "vec_ranges_var size = {}, it should be >= {}.", vec_ranges_var.size(), n);
            vec_ranges_var.setConstant(m_setting_->init_variance);
        }

        for (int i = 0; i < angles.size(); ++i) {
            Scalar angle_local;
            angle_local[0] = angles[i];
            if (!angles_are_local) {
                const Vector2 direction_local = m_sensor_frame_->DirWorldToFrame({std::cos(angle_local[0]), std::sin(angle_local[0])});
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
            Scalar f, var;
            if (VectorX no_var; compute_var ? !gp->Test(angle_local, {0}, f, var) : !gp->Test(angle_local, {0}, f, no_var)) { continue; }
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
        if (range_pred_var[0] > m_setting_->max_valid_range_var) { return false; }  // fail to estimate the mapped r f
        // when the r is larger, 1/r results in smaller difference, we need a larger scale.
        const Dtype a = r * m_setting_->occ_test_temperature;
        occ = 2. / (1. + std::exp(a * (range_pred[0] - m_mapping_->map(r)))) - 1.;
        range_pred[0] = m_mapping_->inv(range_pred[0]);
        return true;
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::operator==(const LidarGaussianProcess2D &other) const {
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr && (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) { return false; }
        if (m_trained_ != other.m_trained_) { return false; }
        if (m_gps_.size() != other.m_gps_.size()) { return false; }
        for (std::size_t i = 0; i < m_gps_.size(); ++i) {
            if (m_gps_[i] == nullptr && other.m_gps_[i] != nullptr) { return false; }
            if (m_gps_[i] != nullptr && (other.m_gps_[i] == nullptr || *m_gps_[i] != *other.m_gps_[i])) { return false; }
        }
        if (m_angle_partitions_.size() != other.m_angle_partitions_.size()) { return false; }
        for (std::size_t i = 0; i < m_angle_partitions_.size(); ++i) {
            if (m_angle_partitions_[i] != other.m_angle_partitions_[i]) { return false; }
        }
        if (m_sensor_frame_ == nullptr && other.m_sensor_frame_ != nullptr) { return false; }
        if (m_sensor_frame_ != nullptr && (other.m_sensor_frame_ == nullptr || *m_sensor_frame_ != *other.m_sensor_frame_)) { return false; }
        if (m_mapped_distances_.size() != other.m_mapped_distances_.size() ||
            std::memcmp(m_mapped_distances_.data(), other.m_mapped_distances_.data(), m_mapped_distances_.size() * sizeof(Dtype)) != 0) {
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
    LidarGaussianProcess2D<Dtype>::Write(const std::string &filename) const {
        ERL_INFO("Writing LidarGaussianProcess2D to file: {}", filename);
        std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
        std::ofstream file(filename, std::ios_base::out | std::ios_base::binary);
        if (!file.is_open()) {
            ERL_WARN("Failed to open file: {}", filename);
            return false;
        }

        const bool success = Write(file);
        file.close();
        return success;
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::Write(std::ostream &s) const {
        s << "# " << type_name(*this) << "\n# (feel free to add / change comments, but leave the first line as it is!)\n";

        static const std::vector<std::pair<const char *, std::function<bool(const LidarGaussianProcess2D *, std::ostream &)>>> token_function_pairs = {
            {
                "setting",
                [](const LidarGaussianProcess2D *gp, std::ostream &stream) {
                    if (!gp->m_setting_->Write(stream)) {
                        ERL_WARN("Failed to write setting.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "trained",
                [](const LidarGaussianProcess2D *gp, std::ostream &stream) {
                    stream << gp->m_trained_;
                    return true;
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
                        if (!g->Write(stream)) {
                            ERL_WARN("Failed to write gp.");
                            return false;
                        }
                    }
                    return true;
                },
            },
            {
                "angle_partitions",
                [](const LidarGaussianProcess2D *gp, std::ostream &stream) {
                    stream << gp->m_angle_partitions_.size() << '\n';
                    for (const auto &[index_left, index_right, coord_left, coord_right]: gp->m_angle_partitions_) {
                        stream.write(reinterpret_cast<const char *>(&index_left), sizeof(index_left));
                        stream.write(reinterpret_cast<const char *>(&index_right), sizeof(index_right));
                        stream.write(reinterpret_cast<const char *>(&coord_left), sizeof(coord_left));
                        stream.write(reinterpret_cast<const char *>(&coord_right), sizeof(coord_right));
                    }
                    return true;
                },
            },
            {
                "lidar_frame",
                [](const LidarGaussianProcess2D *gp, std::ostream &stream) {
                    if (!gp->m_sensor_frame_->Write(stream)) {
                        ERL_WARN("Failed to write lidar_frame.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "mapped_distances",
                [](const LidarGaussianProcess2D *gp, std::ostream &stream) {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, gp->m_mapped_distances_)) {
                        ERL_WARN("Failed to write mapped_distances.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "end_of_LidarGaussianProcess2D",
                [](const LidarGaussianProcess2D *, std::ostream &) { return true; },
            },
        };
        return common::WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::Read(const std::string &filename) {
        ERL_INFO("Reading {} from file: {}", type_name(*this), filename);
        std::ifstream file(filename.c_str(), std::ios_base::in | std::ios_base::binary);
        if (!file.is_open()) {
            ERL_WARN("Failed to open file: {}", filename.c_str());
            return false;
        }

        const bool success = Read(file);
        file.close();
        return success;
    }

    template<typename Dtype>
    bool
    LidarGaussianProcess2D<Dtype>::Read(std::istream &s) {
        if (!s.good()) {
            ERL_WARN("Input stream is not ready for reading");
            return false;
        }

        // check if the first line is valid
        std::string line;
        std::getline(s, line);
        if (std::string file_header = fmt::format("# {}", type_name(*this));
            line.compare(0, file_header.length(), file_header) != 0) {  // check if the first line is valid
            ERL_WARN("Header does not start with \"{}\"", file_header);
            return false;
        }

        static const std::vector<std::pair<const char *, std::function<bool(LidarGaussianProcess2D *, std::istream &)>>> token_function_pairs = {
            {
                "setting",
                [](LidarGaussianProcess2D *gp, std::istream &stream) -> bool {
                    if (!gp->m_setting_->Read(stream)) {
                        ERL_WARN("Failed to read setting.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "trained",
                [](LidarGaussianProcess2D *gp, std::istream &stream) -> bool {
                    stream >> gp->m_trained_;
                    return true;
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
                        if (!gp->m_gps_[i]->Read(stream)) {
                            ERL_WARN("Failed to read gp.");
                            return false;
                        }
                    }
                    return true;
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
                        auto &[index_left, index_right, coord_left, coord_right] = gp->m_angle_partitions_[i];
                        stream.read(reinterpret_cast<char *>(&index_left), sizeof(index_left));
                        stream.read(reinterpret_cast<char *>(&index_right), sizeof(index_right));
                        stream.read(reinterpret_cast<char *>(&coord_left), sizeof(coord_left));
                        stream.read(reinterpret_cast<char *>(&coord_right), sizeof(coord_right));
                    }
                    return true;
                },
            },
            {
                "lidar_frame",
                [](LidarGaussianProcess2D *gp, std::istream &stream) -> bool {
                    common::SkipLine(stream);
                    gp->m_sensor_frame_ = std::make_shared<LidarFrame2D>(gp->m_setting_->sensor_frame);
                    if (!gp->m_sensor_frame_->Read(stream)) {
                        ERL_WARN("Failed to read lidar_frame.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "mapped_distances",
                [](LidarGaussianProcess2D *gp, std::istream &stream) -> bool {
                    common::SkipLine(stream);
                    if (!common::LoadEigenMatrixFromBinaryStream(stream, gp->m_mapped_distances_)) {
                        ERL_WARN("Failed to read mapped_distances.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "end_of_LidarGaussianProcess2D",
                [](LidarGaussianProcess2D *gp, std::istream &stream) -> bool {
                    common::SkipLine(stream);
                    gp->m_mapping_ = MappingDtype::Create(gp->m_setting_->mapping);
                    return true;
                },
            },
        };
        return common::ReadTokens(s, this, token_function_pairs);
    }
}  // namespace erl::gaussian_process

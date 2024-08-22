#include "erl_gaussian_process/lidar_gp_2d.hpp"

namespace erl::gaussian_process {
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

        m_trained_ = true;
        return true;
    }

    bool
    LidarGaussianProcess2D::Test(
        const Eigen::Ref<const Eigen::VectorXd> &angles,
        const bool angles_are_local,
        Eigen::Ref<Eigen::VectorXd> vec_ranges,
        Eigen::Ref<Eigen::VectorXd> vec_ranges_var,
        const bool un_map) const {

        if (!m_trained_) { return false; }

        const long n = angles.size();
        ERL_ASSERTM(vec_ranges.size() >= n, "vec_ranges size = {}, it should be >= {}.", vec_ranges.size(), n);
        ERL_ASSERTM(vec_ranges_var.size() >= n, "vec_ranges_var size = {}, it should be >= {}.", vec_ranges_var.size(), n);

        vec_ranges.setZero();
        vec_ranges_var.setConstant(m_setting_->init_variance);

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

        if (!Test(angle_local, true, range_pred, range_pred_var, false)) { return false; }
        if (range_pred_var[0] > m_setting_->max_valid_range_var) { return false; }  // fail to estimate the mapped r f
        // when the r is larger, 1/r results in smaller difference, we need a larger scale.
        const double a = r * m_setting_->occ_test_temperature;
        occ = 2. / (1. + std::exp(a * (range_pred[0] - m_mapping_->map(r)))) - 1.;
        range_pred[0] = m_mapping_->inv(range_pred[0]);
        return true;
    }

    bool
    LidarGaussianProcess2D::operator==(const LidarGaussianProcess2D &other) const {
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
        if (m_lidar_frame_ == nullptr && other.m_lidar_frame_ != nullptr) { return false; }
        if (m_lidar_frame_ != nullptr && (other.m_lidar_frame_ == nullptr || *m_lidar_frame_ != *other.m_lidar_frame_)) { return false; }
        if (m_mapped_distances_.size() != other.m_mapped_distances_.size() ||
            std::memcmp(m_mapped_distances_.data(), other.m_mapped_distances_.data(), m_mapped_distances_.size() * sizeof(double)) != 0) {
            return false;
        }
        return true;
    }

    bool
    LidarGaussianProcess2D::Write(const std::string &filename) const {
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

    static const std::string kFileHeader = "# erl::gaussian_process::LidarGaussianProcess2D";

    bool
    LidarGaussianProcess2D::Write(std::ostream &s) const {
        s << kFileHeader << std::endl  //
          << "# (feel free to add / change comments, but leave the first line as it is!)" << std::endl
          << "setting" << std::endl;
        // write setting
        if (!m_setting_->Write(s)) {
            ERL_WARN("Failed to write setting.");
            return false;
        }
        s << "trained " << m_trained_ << std::endl;
        s << "gps " << m_gps_.size() << std::endl;
        for (const auto &gp: m_gps_) {
            auto has_gp = static_cast<char>(gp != nullptr);
            s.write(&has_gp, sizeof(char));
            if (has_gp) {
                if (!gp->Write(s)) {
                    ERL_WARN("Failed to write gp.");
                    return false;
                }
            }
        }
        s << "angle_partitions " << m_angle_partitions_.size() << std::endl;
        for (const auto &[index_left, index_right, coord_left, coord_right]: m_angle_partitions_) {
            s.write(reinterpret_cast<const char *>(&index_left), sizeof(index_left));
            s.write(reinterpret_cast<const char *>(&index_right), sizeof(index_right));
            s.write(reinterpret_cast<const char *>(&coord_left), sizeof(coord_left));
            s.write(reinterpret_cast<const char *>(&coord_right), sizeof(coord_right));
        }
        s << "lidar_frame" << std::endl;
        if (!m_lidar_frame_->Write(s)) {
            ERL_WARN("Failed to write lidar_frame.");
            return false;
        }
        s << "mapped_distances" << std::endl;
        if (!common::SaveEigenMatrixToBinaryStream(s, m_mapped_distances_)) {
            ERL_WARN("Failed to write mapped_distances.");
            return false;
        }
        s << "end_of_LidarGaussianProcess2D" << std::endl;
        return s.good();
    }

    bool
    LidarGaussianProcess2D::Read(const std::string &filename) {
        ERL_INFO("Reading LidarGaussianProcess2D from file: {}", std::filesystem::absolute(filename));
        std::ifstream file(filename.c_str(), std::ios_base::in | std::ios_base::binary);
        if (!file.is_open()) {
            ERL_WARN("Failed to open file: {}", filename.c_str());
            return false;
        }

        const bool success = Read(file);
        file.close();
        return success;
    }

    bool
    LidarGaussianProcess2D::Read(std::istream &s) {
        if (!s.good()) {
            ERL_WARN("Input stream is not ready for reading");
            return false;
        }

        // check if the first line is valid
        std::string line;
        std::getline(s, line);
        if (line.compare(0, kFileHeader.length(), kFileHeader) != 0) {  // check if the first line is valid
            ERL_WARN("Header does not start with \"{}\"", kFileHeader.c_str());
            return false;
        }

        auto skip_line = [&s]() {
            char c;
            do { c = static_cast<char>(s.get()); } while (s.good() && c != '\n');
        };

        static const char *tokens[] = {
            "setting",
            "trained",
            "gps",
            "angle_partitions",
            "lidar_frame",
            "mapped_distances",
            "end_of_LidarGaussianProcess2D",
        };

        // read data
        std::string token;
        int token_idx = 0;
        while (s.good()) {
            s >> token;
            if (token.compare(0, 1, "#") == 0) {
                skip_line();  // comment line, skip forward until end of line
                continue;
            }
            // non-comment line
            if (token != tokens[token_idx]) {
                ERL_WARN("Expected token {}, got {}.", tokens[token_idx], token);  // check token
                return false;
            }
            // reading state machine
            switch (token_idx) {
                case 0: {  // setting
                    skip_line();
                    if (!m_setting_->Read(s)) {
                        ERL_WARN("Failed to read setting.");
                        return false;
                    }
                    break;
                }
                case 1: {  // trained
                    s >> m_trained_;
                    break;
                }
                case 2: {  // gps
                    long num_gps;
                    s >> num_gps;
                    m_gps_.resize(num_gps, nullptr);
                    skip_line();
                    for (long i = 0; i < num_gps; ++i) {
                        char has_gp;
                        s.read(&has_gp, sizeof(char));
                        if (has_gp) {
                            m_gps_[i] = std::make_shared<VanillaGaussianProcess>(m_setting_->gp);
                            if (!m_gps_[i]->Read(s)) {
                                ERL_WARN("Failed to read gp.");
                                return false;
                            }
                        }
                    }
                    break;
                }
                case 3: {  // angle_partitions
                    long num_partitions;
                    s >> num_partitions;
                    m_angle_partitions_.resize(num_partitions);
                    skip_line();
                    for (long i = 0; i < num_partitions; ++i) {
                        auto &[index_left, index_right, coord_left, coord_right] = m_angle_partitions_[i];
                        s.read(reinterpret_cast<char *>(&index_left), sizeof(index_left));
                        s.read(reinterpret_cast<char *>(&index_right), sizeof(index_right));
                        s.read(reinterpret_cast<char *>(&coord_left), sizeof(coord_left));
                        s.read(reinterpret_cast<char *>(&coord_right), sizeof(coord_right));
                    }
                    break;
                }
                case 4: {  // lidar_frame
                    skip_line();
                    m_lidar_frame_ = std::make_shared<geometry::LidarFrame2D>(m_setting_->lidar_frame);
                    if (!m_lidar_frame_->Read(s)) {
                        ERL_WARN("Failed to read lidar_frame.");
                        return false;
                    }
                    break;
                }
                case 5: {  // mapped_distances
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_mapped_distances_)) {
                        ERL_WARN("Failed to read mapped_distances.");
                        return false;
                    }
                    break;
                }
                case 6: {  // end_of_LidarGaussianProcess2D
                    skip_line();
                    m_mapping_ = Mapping::Create(m_setting_->mapping);
                    return true;
                }
                default: {  // should not reach here
                    ERL_FATAL("Internal error, should not reach here.");
                }
            }
            ++token_idx;
        }
        ERL_WARN("Failed to read LidarGaussianProcess2D. Truncated file?");
        return false;  // should not reach here
    }

}  // namespace erl::gaussian_process

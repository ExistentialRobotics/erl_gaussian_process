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
            long index_left = i * row_step + row_gs2 - row_half_overlap;
            long index_right = index_left + row_gs;
            double coord_left = frame_coords(index_left + row_half_overlap, 0)[0];
            double coord_right = frame_coords(index_right - row_half_overlap, 0)[0];
            m_row_partitions_.emplace_back(index_left, index_right, coord_left, coord_right);
        }
        // last partition
        m_row_partitions_.emplace_back(
            num_rows - row_gs2 - row_half_overlap,
            num_rows,
            frame_coords(num_rows - 1 - row_gs2, 0)[0],
            frame_coords(num_rows - 1 - m_setting_->row_margin, 0)[0]);

        m_col_partitions_.reserve(num_col_groups);
        // first partition
        m_col_partitions_.emplace_back(0, col_gs2 + col_half_overlap, frame_coords(0, m_setting_->col_margin)[1], frame_coords(0, col_gs2)[1]);
        // middle partitions
        for (long i = 0; i < num_col_groups - 2; ++i) {
            long index_left = i * col_step + col_gs2 - col_half_overlap;
            long index_right = index_left + col_gs;
            double coord_left = frame_coords(0, index_left + col_half_overlap)[1];
            double coord_right = frame_coords(0, index_right - col_half_overlap)[1];
            m_col_partitions_.emplace_back(index_left, index_right, coord_left, coord_right);
        }
        // last partition
        m_col_partitions_.emplace_back(
            num_cols - col_gs2 - col_half_overlap,
            num_cols,
            frame_coords(0, num_cols - 1 - col_gs2)[1],
            frame_coords(0, num_cols - 1 - m_setting_->col_margin)[1]);
    }

    void
    RangeSensorGaussianProcess3D::Reset() {
        m_trained_ = false;
    }

    bool
    RangeSensorGaussianProcess3D::StoreData(const Eigen::Matrix3d &rotation, const Eigen::Vector3d &translation, Eigen::MatrixXd ranges) {
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

#pragma omp parallel for collapse(2) default(none)
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
                        train_input_samples.col(cnt) << frame_coords(r, c);
                        train_output_samples[cnt] = m_mapped_distances_(r, c);
                        train_output_samples_variance[cnt] = m_setting_->sensor_range_var;
                        ++cnt;
                    }
                }
                if (cnt > 0) { (void) gp->Train(cnt); }
            }
        }

        ERL_INFO("{} x {} Gaussian processes are trained.", m_gps_.rows(), m_gps_.cols());

        m_trained_ = true;
        return true;
    }

    bool
    RangeSensorGaussianProcess3D::Test(
        const Eigen::Ref<const Eigen::Matrix3Xd> &directions,
        const bool directions_are_local,
        Eigen::Ref<Eigen::VectorXd> vec_ranges,
        Eigen::Ref<Eigen::VectorXd> vec_ranges_var,
        const bool un_map) const {

        if (!m_trained_) { return false; }

        const long n = directions.cols();
        ERL_DEBUG_ASSERT(n > 0, "directions_world is empty.");
        ERL_DEBUG_ASSERT(vec_ranges.size() >= n, "vec_ranges size = {}, it should be >= {}.", vec_ranges.size(), n);
        ERL_DEBUG_ASSERT(vec_ranges_var.size() >= n, "vec_ranges_var size = {}, it should be >= {}.", vec_ranges_var.size(), n);

        vec_ranges.setZero();
        vec_ranges_var.setConstant(m_setting_->init_variance);

        for (long i = 0; i < n; ++i) {
            Eigen::Vector3d direction_local = directions.col(i);
            if (!directions_are_local) { direction_local = m_range_sensor_frame_->WorldToFrameSo3(direction_local); }
            if (!m_range_sensor_frame_->PointIsInFrame(direction_local)) { continue; }
            const Eigen::Vector2d frame_coords = m_range_sensor_frame_->ComputeFrameCoords(direction_local);

            // search for the partition
            // row
            const double &row_coord = frame_coords.x();
            long partition_row_index = 0;
            for (; partition_row_index < static_cast<long>(m_row_partitions_.size()); ++partition_row_index) {
                if (const auto &[row_index_left, row_index_right, row_coord_left, row_coord_right] = m_row_partitions_[partition_row_index];
                    row_coord >= row_coord_left && row_coord < row_coord_right) {
                    break;
                }
            }
            if (partition_row_index >= static_cast<long>(m_row_partitions_.size())) { continue; }
            // col
            const double &col_coord = frame_coords.y();
            long partition_col_index = 0;
            for (; partition_col_index < static_cast<long>(m_col_partitions_.size()); ++partition_col_index) {
                if (const auto &[col_index_left, col_index_right, col_coord_left, col_coord_right] = m_col_partitions_[partition_col_index];
                    col_coord >= col_coord_left && col_coord <= col_coord_right) {
                    break;
                }
            }
            if (partition_col_index >= static_cast<long>(m_col_partitions_.size())) { continue; }

            const auto gp = m_gps_(partition_row_index, partition_col_index);
            if (!gp->IsTrained()) { continue; }
            Eigen::Scalard f, var;
            if (!gp->Test(frame_coords, f, var)) { continue; }  // invalid test
            vec_ranges[i] = un_map ? m_mapping_->inv(f[0]) : f[0];
            vec_ranges_var[i] = var[0];
        }
        return true;
    }

    bool
    RangeSensorGaussianProcess3D::ComputeOcc(
        const Eigen::Vector3d &dir_local,
        const double r,
        Eigen::Ref<Eigen::Scalard> range_pred,
        Eigen::Ref<Eigen::Scalard> range_pred_var,
        double &occ) const {

        if (!Test(dir_local, true, range_pred, range_pred_var, false)) { return false; }
        if (range_pred_var[0] > m_setting_->max_valid_range_var) { return false; }
        const double a = r * m_setting_->occ_test_temperature;
        occ = 2.0 / (1.0 + std::exp(a * (range_pred[0] - m_mapping_->map(r)))) - 1.0;
        range_pred[0] = m_mapping_->inv(range_pred[0]);
        return true;
    }

    bool
    RangeSensorGaussianProcess3D::operator==(const RangeSensorGaussianProcess3D &other) const {
        if (m_trained_ != other.m_trained_) { return false; }
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr && (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) { return false; }
        if (m_gps_.rows() != other.m_gps_.rows() || m_gps_.cols() != other.m_gps_.cols()) { return false; }
        for (long i = 0; i < m_gps_.size(); ++i) {
            const auto &gp = m_gps_.data()[i];
            const auto &other_gp = other.m_gps_.data()[i];
            if (gp == nullptr && other_gp != nullptr) { return false; }
            if (gp != nullptr && (other_gp == nullptr || *gp != *other_gp)) { return false; }
        }
        if (m_row_partitions_ != other.m_row_partitions_) { return false; }
        if (m_col_partitions_ != other.m_col_partitions_) { return false; }
        if (m_range_sensor_frame_ == nullptr && other.m_range_sensor_frame_ != nullptr) { return false; }
        if (m_range_sensor_frame_ != nullptr && (other.m_range_sensor_frame_ == nullptr || *m_range_sensor_frame_ != *other.m_range_sensor_frame_)) {
            return false;
        }
        if (m_mapped_distances_.rows() != other.m_mapped_distances_.rows() || m_mapped_distances_.cols() != other.m_mapped_distances_.cols() ||
            std::memcmp(m_mapped_distances_.data(), other.m_mapped_distances_.data(), m_mapped_distances_.size() * sizeof(double)) != 0) {
            return false;
        }
        return true;
    }

    bool
    RangeSensorGaussianProcess3D::Write(const std::string &filename) const {
        ERL_INFO("Writing RangeSensorGaussianProcess3D to file: {}", filename);
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

    static const std::string kFileHeader = "# erl::gaussian_process::RangeSensorGaussianProcess3D";

    bool
    RangeSensorGaussianProcess3D::Write(std::ostream &s) const {
        s << kFileHeader << std::endl  //
          << "# (feel free to add / change comments, but leave the first line as it is!)" << std::endl
          << "setting" << std::endl;
        // write setting
        if (!m_setting_->Write(s)) {
            ERL_WARN("Failed to write setting.");
            return false;
        }
        s << "trained " << m_trained_ << std::endl;
        s << "gps " << m_gps_.rows() << " " << m_gps_.cols() << std::endl;
        auto *gp_ptr = m_gps_.data();
        for (long i = 0; i < m_gps_.size(); ++i) {
            const auto gp = gp_ptr[i];
            auto has_gp = static_cast<char>(gp != nullptr);
            s.write(&has_gp, sizeof(char));
            if (has_gp) {
                if (!gp->Write(s)) {
                    ERL_WARN("Failed to write gp.");
                    return false;
                }
            }
        }
        s << "row_partitions " << m_row_partitions_.size() << std::endl;
        for (const auto &[index_left, index_right, coord_left, coord_right]: m_row_partitions_) {
            s.write(reinterpret_cast<const char *>(&index_left), sizeof(index_left));
            s.write(reinterpret_cast<const char *>(&index_right), sizeof(index_right));
            s.write(reinterpret_cast<const char *>(&coord_left), sizeof(coord_left));
            s.write(reinterpret_cast<const char *>(&coord_right), sizeof(coord_right));
        }
        s << "col_partitions " << m_col_partitions_.size() << std::endl;
        for (const auto &[index_left, index_right, coord_left, coord_right]: m_col_partitions_) {
            s.write(reinterpret_cast<const char *>(&index_left), sizeof(index_left));
            s.write(reinterpret_cast<const char *>(&index_right), sizeof(index_right));
            s.write(reinterpret_cast<const char *>(&coord_left), sizeof(coord_left));
            s.write(reinterpret_cast<const char *>(&coord_right), sizeof(coord_right));
        }
        s << "range_sensor_frame" << std::endl;
        if (!m_range_sensor_frame_->Write(s)) {
            ERL_WARN("Failed to write range_sensor_frame.");
            return false;
        }
        s << "mapped_distances" << std::endl;
        if (!common::SaveEigenMatrixToBinaryStream(s, m_mapped_distances_)) {
            ERL_WARN("Failed to write mapped_distances.");
            return false;
        }
        s << "end_of_RangeSensorGaussianProcess3D" << std::endl;
        return s.good();
    }

    bool
    RangeSensorGaussianProcess3D::Read(const std::string &filename) {
        ERL_INFO("Reading RangeSensorGaussianProcess3D from file: {}", std::filesystem::absolute(filename));
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
    RangeSensorGaussianProcess3D::Read(std::istream &s) {
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
            "row_partitions",
            "col_partitions",
            "range_sensor_frame",
            "mapped_distances",
            "end_of_RangeSensorGaussianProcess3D",
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
                    long rows, cols;
                    s >> rows >> cols;
                    skip_line();
                    m_gps_.resize(rows, cols);
                    m_gps_.setConstant(nullptr);
                    auto *gp_ptr = m_gps_.data();
                    const long num_gps = rows * cols;
                    for (long i = 0; i < num_gps; ++i) {
                        char has_gp;
                        s.read(&has_gp, sizeof(char));
                        if (has_gp) {
                            gp_ptr[i] = std::make_shared<VanillaGaussianProcess>(m_setting_->gp);
                            if (!gp_ptr[i]->Read(s)) {
                                ERL_WARN("Failed to read gp.");
                                return false;
                            }
                        }
                    }
                    break;
                }
                case 3: {  // row_partitions
                    long num_partitions;
                    s >> num_partitions;
                    m_row_partitions_.resize(num_partitions);
                    skip_line();
                    for (long i = 0; i < num_partitions; ++i) {
                        auto &[index_left, index_right, coord_left, coord_right] = m_row_partitions_[i];
                        s.read(reinterpret_cast<char *>(&index_left), sizeof(index_left));
                        s.read(reinterpret_cast<char *>(&index_right), sizeof(index_right));
                        s.read(reinterpret_cast<char *>(&coord_left), sizeof(coord_left));
                        s.read(reinterpret_cast<char *>(&coord_right), sizeof(coord_right));
                    }
                    break;
                }
                case 4: {  // col_partitions
                    long num_partitions;
                    s >> num_partitions;
                    m_col_partitions_.resize(num_partitions);
                    skip_line();
                    for (long i = 0; i < num_partitions; ++i) {
                        auto &[index_left, index_right, coord_left, coord_right] = m_col_partitions_[i];
                        s.read(reinterpret_cast<char *>(&index_left), sizeof(index_left));
                        s.read(reinterpret_cast<char *>(&index_right), sizeof(index_right));
                        s.read(reinterpret_cast<char *>(&coord_left), sizeof(coord_left));
                        s.read(reinterpret_cast<char *>(&coord_right), sizeof(coord_right));
                    }
                    break;
                }
                case 5: {  // range_sensor_frame
                    skip_line();
                    if (!m_range_sensor_frame_->Read(s)) {
                        ERL_WARN("Failed to read lidar_frame.");
                        return false;
                    }
                    break;
                }
                case 6: {  // mapped_distances
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_mapped_distances_)) {
                        ERL_WARN("Failed to read mapped_distances.");
                        return false;
                    }
                    break;
                }
                case 7: {  // end_of_RangeSensorGaussianProcess3D
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
        ERL_WARN("Failed to read RangeSensorGaussianProcess3D. Truncated file?");
        return false;  // should not reach here
    }

}  // namespace erl::gaussian_process

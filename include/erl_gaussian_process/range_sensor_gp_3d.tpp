#pragma once

#include "erl_common/block_timer.hpp"
#include "erl_common/logging.hpp"

namespace erl::gaussian_process {
    template<typename Dtype>
    YAML::Node
    RangeSensorGaussianProcess3D<Dtype>::Setting::YamlConvertImpl::encode(const Setting &setting) {
        YAML::Node node;
        ERL_YAML_SAVE_ATTR(node, setting, row_group_size);
        ERL_YAML_SAVE_ATTR(node, setting, row_overlap_size);
        ERL_YAML_SAVE_ATTR(node, setting, row_margin);
        ERL_YAML_SAVE_ATTR(node, setting, col_group_size);
        ERL_YAML_SAVE_ATTR(node, setting, col_overlap_size);
        ERL_YAML_SAVE_ATTR(node, setting, col_margin);
        ERL_YAML_SAVE_ATTR(node, setting, init_variance);
        ERL_YAML_SAVE_ATTR(node, setting, sensor_range_var);
        ERL_YAML_SAVE_ATTR(node, setting, max_valid_range_var);
        ERL_YAML_SAVE_ATTR(node, setting, occ_test_temperature);
        ERL_YAML_SAVE_ATTR(node, setting, sensor_frame_type);
        ERL_YAML_SAVE_ATTR(node, setting, sensor_frame_setting_type);
        node["sensor_frame"] = setting.sensor_frame->AsYamlNode();
        ERL_YAML_SAVE_ATTR(node, setting, gp);
        ERL_YAML_SAVE_ATTR(node, setting, mapping);
        return node;
    }

    template<typename Dtype>
    bool
    RangeSensorGaussianProcess3D<Dtype>::Setting::YamlConvertImpl::decode(
        const YAML::Node &node,
        Setting &setting) {
        if (!node.IsMap()) { return false; }
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, row_group_size, int);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, row_overlap_size, int);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, row_margin, long);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, col_group_size, int);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, col_overlap_size, int);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, col_margin, long);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, init_variance, Dtype);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, sensor_range_var, Dtype);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, max_valid_range_var, Dtype);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, occ_test_temperature, Dtype);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, sensor_frame_type, std::string);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, sensor_frame_setting_type, std::string);
        using SensorFrameSetting = typename geometry::RangeSensorFrame3D<Dtype>::Setting;
        setting.sensor_frame =
            common::YamlableBase::Create<SensorFrameSetting>(setting.sensor_frame_setting_type);
        if (!setting.sensor_frame->FromYamlNode(node["sensor_frame"])) { return false; }
        ERL_YAML_LOAD_ATTR(node, setting, gp);
        ERL_YAML_LOAD_ATTR(node, setting, mapping);
        return true;
    }

    template<typename Dtype>
    RangeSensorGaussianProcess3D<Dtype>::TestResult::TestResult(
        const RangeSensorGaussianProcess3D *gp,
        const Eigen::Ref<const Matrix3X> &directions,
        const bool directions_are_local,
        const std::shared_ptr<MappingDtype> &mapping)
        : m_gp_(NotNull(gp, true, "gp = nullptr.")),
          m_mapping_(mapping) {

        const long n = directions.cols();
        ERL_DEBUG_ASSERT(m_gp_->IsTrained(), "The model has not been trained.");
        ERL_DEBUG_ASSERT(n > 0, "directions_world is empty.");

        m_reduced_rank_kernel_ = m_gp_->m_gps_.data()[0]->UsingReducedRankKernel();
        m_gps_.resize(n, nullptr);
        m_k_test_vec_.resize(n);
        m_alpha_vec_.resize(n, {nullptr, 0});
        m_alpha_test_vec_.resize(n);

        const RangeSensorFrame *sensor_frame = m_gp_->m_sensor_frame_.get();
        auto &gps = m_gp_->m_gps_;

        for (long i = 0; i < n; ++i) {
            Vector3 direction_local = directions.col(i);
            if (!directions_are_local) {
                direction_local = sensor_frame->DirWorldToFrame(direction_local);
            }
            if (!sensor_frame->PointIsInFrame(direction_local)) { continue; }
            const Vector2 frame_coords = sensor_frame->ComputeFrameCoords(direction_local);
            auto [partition_row_index, partition_col_index] = m_gp_->SearchPartition(frame_coords);
            if (partition_row_index < 0 || partition_col_index < 0) { continue; }
            const Gp *partition_gp = gps(partition_row_index, partition_col_index).get();
            if (!partition_gp->IsTrained()) { continue; }
            MatrixX ktest;
            const bool success = partition_gp->ComputeKtest(frame_coords, ktest);
            (void) success;
            ERL_DEBUG_ASSERT(success, "Failed to compute ktest.");
            m_gps_[i] = partition_gp;
            m_k_test_vec_[i] = ktest;
            m_alpha_vec_[i] = {partition_gp->GetAlpha().col(0).data(), ktest.rows()};
        }
    }

    template<typename Dtype>
    long
    RangeSensorGaussianProcess3D<Dtype>::TestResult::GetNumTest() const {
        return static_cast<long>(m_k_test_vec_.size());
    }

    template<typename Dtype>
    const typename RangeSensorGaussianProcess3D<Dtype>::VectorX &
    RangeSensorGaussianProcess3D<Dtype>::TestResult::GetKtest(long index) const {
        return m_k_test_vec_[index];
    }

    template<typename Dtype>
    Eigen::VectorXb
    RangeSensorGaussianProcess3D<Dtype>::TestResult::GetMean(
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
    RangeSensorGaussianProcess3D<Dtype>::TestResult::GetMean(const long index, Dtype &f) const {
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
    RangeSensorGaussianProcess3D<Dtype>::TestResult::GetVariance(
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
    RangeSensorGaussianProcess3D<Dtype>::TestResult::GetVariance(const long index, Dtype &var)
        const {
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
    RangeSensorGaussianProcess3D<Dtype>::TestResult::PrepareAlphaTest(const long index) {
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
    RangeSensorGaussianProcess3D<Dtype>::RangeSensorGaussianProcess3D(
        std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)),
          m_sensor_frame_(geometry::RangeSensorFrame3D<Dtype>::Create(
              m_setting_->sensor_frame_type,
              m_setting_->sensor_frame)),
          m_mapping_(MappingDtype::Create(m_setting_->mapping)) {

        const Eigen::MatrixX<Vector2> &frame_coords = m_sensor_frame_->GetFrameCoords();
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

        m_setting_->gp->max_num_samples = m_setting_->row_group_size * m_setting_->col_group_size;
        m_setting_->gp->kernel->x_dim = 2;
        m_gps_.setConstant(num_row_groups, num_col_groups, nullptr);

        m_row_partitions_.reserve(num_row_groups);
        // first partition
        m_row_partitions_.emplace_back(
            0,
            row_gs2 + row_half_overlap,
            frame_coords(m_setting_->row_margin, 0)[0],
            frame_coords(row_gs2, 0)[0]);
        // middle partitions
        for (long i = 0; i < num_row_groups - 2; ++i) {
            long index_left = i * row_step + row_gs2 - row_half_overlap;
            long index_right = index_left + row_gs;
            Dtype coord_left = frame_coords(index_left + row_half_overlap, 0)[0];
            Dtype coord_right = frame_coords(index_right - row_half_overlap, 0)[0];
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
        m_col_partitions_.emplace_back(
            0,
            col_gs2 + col_half_overlap,
            frame_coords(0, m_setting_->col_margin)[1],
            frame_coords(0, col_gs2)[1]);
        // middle partitions
        for (long i = 0; i < num_col_groups - 2; ++i) {
            long index_left = i * col_step + col_gs2 - col_half_overlap;
            long index_right = index_left + col_gs;
            Dtype coord_left = frame_coords(0, index_left + col_half_overlap)[1];
            Dtype coord_right = frame_coords(0, index_right - col_half_overlap)[1];
            m_col_partitions_.emplace_back(index_left, index_right, coord_left, coord_right);
        }
        // last partition
        m_col_partitions_.emplace_back(
            num_cols - col_gs2 - col_half_overlap,
            num_cols,
            frame_coords(0, num_cols - 1 - col_gs2)[1],
            frame_coords(0, num_cols - 1 - m_setting_->col_margin)[1]);
    }

    template<typename Dtype>
    void
    RangeSensorGaussianProcess3D<Dtype>::Reset() {
        m_trained_ = false;
    }

    template<typename Dtype>
    bool
    RangeSensorGaussianProcess3D<Dtype>::StoreData(
        const Matrix3 &rotation,
        const Vector3 &translation,
        MatrixX ranges) {
        m_sensor_frame_->UpdateRanges(rotation, translation, std::move(ranges));
        m_mapped_distances_ = m_sensor_frame_->GetRanges().unaryExpr(m_mapping_->map);
        return m_sensor_frame_->IsValid();
    }

    template<typename Dtype>
    bool
    RangeSensorGaussianProcess3D<Dtype>::Train(
        const Matrix3 &rotation,
        const Vector3 &translation,
        MatrixX ranges) {
        ERL_BLOCK_TIMER();
        Reset();

        if (!StoreData(rotation, translation, std::move(ranges))) {
            ERL_DEBUG("No training data is stored.");
            return false;
        }

#pragma omp parallel for collapse(2) default(none)
        for (long j = 0; j < static_cast<long>(m_col_partitions_.size()); ++j) {
            for (long i = 0; i < static_cast<long>(m_row_partitions_.size()); ++i) {
                const auto &[row_index_left, row_index_right, row_coord_left, row_coord_right] =
                    m_row_partitions_[i];
                const auto &[col_index_left, col_index_right, col_coord_left, col_coord_right] =
                    m_col_partitions_[j];
                std::shared_ptr<Gp> &gp = m_gps_(i, j);
                if (gp == nullptr) { gp = std::make_shared<Gp>(m_setting_->gp); }
                gp->Reset(m_setting_->gp->max_num_samples, 2, 1);
                long cnt = 0;
                typename Gp::TrainSet &train_set = gp->GetTrainSet();
                const Eigen::MatrixXb &mask_hit = m_sensor_frame_->GetHitMask();
                const Eigen::MatrixX<Vector2> &frame_coords = m_sensor_frame_->GetFrameCoords();
                for (long c = col_index_left; c < col_index_right; ++c) {
                    for (long r = row_index_left; r < row_index_right; ++r) {
                        if (!mask_hit(r, c)) { continue; }
                        train_set.x.col(cnt) = frame_coords(r, c);
                        train_set.y.col(0)[cnt] = m_mapped_distances_(r, c);
                        train_set.var[cnt] = m_setting_->sensor_range_var;
                        ++cnt;
                    }
                }
                train_set.num_samples = cnt;
                if (cnt > 0) { (void) gp->Train(); }
            }
        }

        // ERL_INFO("{} x {} Gaussian processes are trained.", m_gps_.rows(), m_gps_.cols());

        m_trained_ = true;
        return true;
    }

    template<typename Dtype>
    std::pair<long, long>
    RangeSensorGaussianProcess3D<Dtype>::SearchPartition(const Vector2 &frame_coords) const {
        // row
        const Dtype &row_coord = frame_coords.x();
        long partition_row_index = 0;
        for (; partition_row_index < static_cast<long>(m_row_partitions_.size());
             ++partition_row_index) {
            if (const auto &[row_index_left, row_index_right, row_coord_left, row_coord_right] =
                    m_row_partitions_[partition_row_index];
                row_coord >= row_coord_left && row_coord < row_coord_right) {
                break;
            }
        }
        if (partition_row_index >= static_cast<long>(m_row_partitions_.size())) { return {-1, -1}; }
        // col
        const Dtype &col_coord = frame_coords.y();
        long partition_col_index = 0;
        for (; partition_col_index < static_cast<long>(m_col_partitions_.size());
             ++partition_col_index) {
            if (const auto &[col_index_left, col_index_right, col_coord_left, col_coord_right] =
                    m_col_partitions_[partition_col_index];
                col_coord >= col_coord_left && col_coord <= col_coord_right) {
                return {partition_row_index, partition_col_index};
            }
        }
        return {-1, -1};
    }

    template<typename Dtype>
    std::shared_ptr<typename RangeSensorGaussianProcess3D<Dtype>::TestResult>
    RangeSensorGaussianProcess3D<Dtype>::Test(
        const Eigen::Ref<const Matrix3X> &directions,
        const bool directions_are_local,
        const bool un_map) const {
        if (!m_trained_) { return nullptr; }
        return std::make_shared<TestResult>(
            this,
            directions,
            directions_are_local,
            un_map ? m_mapping_ : nullptr);
    }

    template<typename Dtype>
    bool
    RangeSensorGaussianProcess3D<Dtype>::ComputeOcc(
        const Vector3 &dir_local,
        const Dtype r,
        Dtype &range_pred,
        Dtype &occ) const {

        if (!m_trained_) { return false; }
        if (!m_sensor_frame_->PointIsInFrame(dir_local)) { return false; }
        const Vector2 frame_coords = m_sensor_frame_->ComputeFrameCoords(dir_local);
        const auto [partition_row_index, partition_col_index] = SearchPartition(frame_coords);
        if (partition_row_index < 0 || partition_col_index < 0) { return false; }
        const Gp &gp = *m_gps_(partition_row_index, partition_col_index);
        if (!gp.IsTrained()) { return false; }
        typename Gp::TestResult test_result = *gp.Test(frame_coords);

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
    RangeSensorGaussianProcess3D<Dtype>::operator==(
        const RangeSensorGaussianProcess3D &other) const {
        if (m_trained_ != other.m_trained_) { return false; }
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr &&
            (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) {
            return false;
        }
        if (m_gps_.rows() != other.m_gps_.rows() || m_gps_.cols() != other.m_gps_.cols()) {
            return false;
        }
        for (long i = 0; i < m_gps_.size(); ++i) {
            const auto &gp = m_gps_.data()[i];
            const auto &other_gp = other.m_gps_.data()[i];
            if (gp == nullptr && other_gp != nullptr) { return false; }
            if (gp != nullptr && (other_gp == nullptr || *gp != *other_gp)) { return false; }
        }
        if (m_row_partitions_ != other.m_row_partitions_) { return false; }
        if (m_col_partitions_ != other.m_col_partitions_) { return false; }
        if (m_sensor_frame_ == nullptr && other.m_sensor_frame_ != nullptr) { return false; }
        if (m_sensor_frame_ != nullptr &&
            (other.m_sensor_frame_ == nullptr || *m_sensor_frame_ != *other.m_sensor_frame_)) {
            return false;
        }
        if (m_mapped_distances_.rows() != other.m_mapped_distances_.rows() ||
            m_mapped_distances_.cols() != other.m_mapped_distances_.cols() ||
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
    RangeSensorGaussianProcess3D<Dtype>::Write(std::ostream &s) const {
        using namespace common;
        static const TokenWriteFunctionPairs<RangeSensorGaussianProcess3D> token_function_pairs = {
            {
                "setting",
                [](const RangeSensorGaussianProcess3D *self, std::ostream &stream) {
                    return self->m_setting_->Write(stream) && stream.good();
                },
            },
            {
                "trained",
                [](const RangeSensorGaussianProcess3D *self, std::ostream &stream) {
                    stream.write(reinterpret_cast<const char *>(&self->m_trained_), sizeof(bool));
                    return stream.good();
                },
            },
            {
                "gps",
                [](const RangeSensorGaussianProcess3D *self, std::ostream &stream) {
                    stream << self->m_gps_.rows() << ' ' << self->m_gps_.cols() << '\n';
                    auto *gp_ptr = self->m_gps_.data();
                    const long num_gps = self->m_gps_.rows() * self->m_gps_.cols();
                    for (long i = 0; i < num_gps; ++i) {
                        char has_gp = static_cast<char>(gp_ptr[i] != nullptr);
                        stream.write(&has_gp, sizeof(char));
                        if (!has_gp) { continue; }
                        if (!gp_ptr[i]->Write(stream)) { return false; }
                    }
                    return stream.good();
                },
            },
            {
                "row_partitions",
                [](const RangeSensorGaussianProcess3D *self, std::ostream &stream) {
                    stream << self->m_row_partitions_.size() << '\n';
                    for (auto &[index_left, index_right, coord_left, coord_right]:
                         self->m_row_partitions_) {
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
                "col_partitions",
                [](const RangeSensorGaussianProcess3D *self, std::ostream &stream) {
                    stream << self->m_col_partitions_.size() << '\n';
                    for (const auto &[index_left, index_right, coord_left, coord_right]:
                         self->m_col_partitions_) {
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
                [](const RangeSensorGaussianProcess3D *self, std::ostream &stream) {
                    return self->m_sensor_frame_->Write(stream) && stream.good();
                },
            },
            {
                "mapped_distances",
                [](const RangeSensorGaussianProcess3D *self, std::ostream &stream) {
                    return SaveEigenMatrixToBinaryStream(stream, self->m_mapped_distances_) &&
                           stream.good();
                },
            },
        };
        return WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    RangeSensorGaussianProcess3D<Dtype>::Read(std::istream &s) {
        using namespace common;
        static const TokenReadFunctionPairs<RangeSensorGaussianProcess3D> token_function_pairs = {
            {
                "setting",
                [](RangeSensorGaussianProcess3D *self, std::istream &stream) {
                    return self->m_setting_->Read(stream) && stream.good();
                },
            },
            {
                "trained",
                [](RangeSensorGaussianProcess3D *self, std::istream &stream) {
                    stream.read(reinterpret_cast<char *>(&self->m_trained_), sizeof(bool));
                    return stream.good();
                },
            },
            {
                "gps",
                [](RangeSensorGaussianProcess3D *self, std::istream &stream) {
                    long rows, cols;
                    stream >> rows >> cols;
                    SkipLine(stream);
                    self->m_gps_.setConstant(rows, cols, nullptr);
                    auto *gp_ptr = self->m_gps_.data();
                    const long num_gps = rows * cols;
                    for (long i = 0; i < num_gps; ++i) {
                        char has_gp;
                        stream.read(&has_gp, sizeof(char));
                        if (!has_gp) { continue; }
                        gp_ptr[i] = std::make_shared<Gp>(self->m_setting_->gp);
                        if (!gp_ptr[i]->Read(stream)) { return false; }
                    }
                    return stream.good();
                },
            },
            {
                "row_partitions",
                [](RangeSensorGaussianProcess3D *self, std::istream &stream) {
                    long num_partitions;
                    stream >> num_partitions;
                    self->m_row_partitions_.resize(num_partitions);
                    SkipLine(stream);
                    for (auto &[index_left, index_right, coord_left, coord_right]:
                         self->m_row_partitions_) {
                        stream.read(reinterpret_cast<char *>(&index_left), sizeof(index_left));
                        stream.read(reinterpret_cast<char *>(&index_right), sizeof(index_right));
                        stream.read(reinterpret_cast<char *>(&coord_left), sizeof(coord_left));
                        stream.read(reinterpret_cast<char *>(&coord_right), sizeof(coord_right));
                    }
                    return stream.good();
                },
            },
            {
                "col_partitions",
                [](RangeSensorGaussianProcess3D *self, std::istream &stream) {
                    long num_partitions;
                    stream >> num_partitions;
                    self->m_col_partitions_.resize(num_partitions);
                    SkipLine(stream);
                    for (auto &[index_left, index_right, coord_left, coord_right]:
                         self->m_col_partitions_) {
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
                [](RangeSensorGaussianProcess3D *self, std::istream &stream) {
                    self->m_sensor_frame_ = geometry::RangeSensorFrame3D<Dtype>::Create(
                        self->m_setting_->sensor_frame_type,
                        self->m_setting_->sensor_frame);
                    return self->m_sensor_frame_->Read(stream) && stream.good();
                },
            },
            {
                "mapped_distances",
                [](RangeSensorGaussianProcess3D *self, std::istream &stream) {
                    return LoadEigenMatrixFromBinaryStream(stream, self->m_mapped_distances_) &&
                           stream.good();
                },
            },
        };
        return ReadTokens(s, this, token_function_pairs);
    }
}  // namespace erl::gaussian_process

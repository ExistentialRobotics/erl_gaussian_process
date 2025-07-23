#include "erl_gaussian_process/spgp_occupancy_map.hpp"

#include "erl_common/serialization.hpp"

namespace erl::gaussian_process {

    template<typename Dtype, int Dim>
    YAML::Node
    SpGpOccupancyMap<Dtype, Dim>::Setting::YamlConvertImpl::encode(const Setting &setting) {
        YAML::Node node;
        ERL_YAML_SAVE_ATTR(node, setting, sp_gp);
        ERL_YAML_SAVE_ATTR(node, setting, min_distance);
        ERL_YAML_SAVE_ATTR(node, setting, max_distance);
        ERL_YAML_SAVE_ATTR(node, setting, free_points_per_meter);
        ERL_YAML_SAVE_ATTR(node, setting, free_sampling_margin);
        ERL_YAML_SAVE_ATTR(node, setting, parallel);
        ERL_YAML_SAVE_ATTR(node, setting, logodd_free);
        ERL_YAML_SAVE_ATTR(node, setting, logodd_occupied);
        ERL_YAML_SAVE_ATTR(node, setting, logodd_variance);
        return node;
    }

    template<typename Dtype, int Dim>
    bool
    SpGpOccupancyMap<Dtype, Dim>::Setting::YamlConvertImpl::decode(
        const YAML::Node &node,
        Setting &setting) {
        if (!node.IsMap()) { return false; }
        if (!ERL_YAML_LOAD_ATTR(node, setting, sp_gp)) { return false; }
        ERL_YAML_LOAD_ATTR(node, setting, min_distance);
        ERL_YAML_LOAD_ATTR(node, setting, max_distance);
        ERL_YAML_LOAD_ATTR(node, setting, free_points_per_meter);
        ERL_YAML_LOAD_ATTR(node, setting, free_sampling_margin);
        ERL_YAML_LOAD_ATTR(node, setting, parallel);
        ERL_YAML_LOAD_ATTR(node, setting, logodd_free);
        ERL_YAML_LOAD_ATTR(node, setting, logodd_occupied);
        ERL_YAML_LOAD_ATTR(node, setting, logodd_variance);
        return true;
    }

    template<typename Dtype, int Dim>
    SpGpOccupancyMap<Dtype, Dim>::SpGpOccupancyMap(
        std::shared_ptr<Setting> setting,
        MatrixX pseudo_points,
        AabbD map_boundary,
        uint64_t seed)
        : m_setting_(NotNull(std::move(setting), true, "setting is nullptr.")),
          m_sp_gp_(m_setting_->sp_gp, std::move(pseudo_points)),
          m_map_boundary_(std::move(map_boundary)),
          m_generator_(seed) {}

    template<typename Dtype, int Dim>
    void
    SpGpOccupancyMap<Dtype, Dim>::GenerateDataset(
        const Eigen::Ref<const VectorD> &sensor_position,
        const Eigen::Ref<const MatrixDX> &points,
        const std::vector<long> &point_indices,
        long max_dataset_size,
        long &num_samples,
        MatrixDX &dataset_points,
        VectorX &dataset_labels,
        std::vector<long> &hit_indices) {
        geometry::OccupancyMap<Dtype, Dim>::GenerateDataset(
            sensor_position,
            points,
            point_indices,
            m_map_boundary_,
            m_generator_,
            m_setting_->min_distance,
            m_setting_->max_distance,
            m_setting_->free_sampling_margin,
            m_setting_->free_points_per_meter,
            max_dataset_size,
            num_samples,
            dataset_points,
            dataset_labels,
            hit_indices);
    }

    template<typename Dtype, int Dim>
    bool
    SpGpOccupancyMap<Dtype, Dim>::Update(
        const Eigen::Ref<const VectorD> &sensor_position,
        const Eigen::Ref<const MatrixDX> &points,
        const std::vector<long> &point_indices,
        long &num_samples,
        MatrixDX &dataset_points,
        VectorX &dataset_labels,
        std::vector<long> &hit_indices) {

        const long max_dataset_size = m_setting_->sp_gp->max_num_samples;
        ERL_ASSERTM(
            max_dataset_size > 0,
            "max_dataset_size should be greater than 0, but got {}.",
            max_dataset_size);
        GenerateDataset(
            sensor_position,
            points,
            point_indices,
            max_dataset_size,
            num_samples,
            dataset_points,
            dataset_labels,
            hit_indices);
        if (num_samples == 0) {
            ERL_WARN("No valid points generated for update. Skipping update.");
            return false;
        }
        m_sp_gp_.Reset(num_samples, Dim, 1);
        auto &train_set = m_sp_gp_.GetTrainSet();
        train_set.x_dim = Dim;
        train_set.y_dim = 1;
        train_set.num_samples = num_samples;
        train_set.x.topLeftCorner(Dim, num_samples) = dataset_points.leftCols(num_samples);
        train_set.y.col(0).head(num_samples) =
            dataset_labels.head(num_samples).unaryExpr([this](Dtype label) {
                return label > 0 ? m_setting_->logodd_occupied : m_setting_->logodd_free;
            });
        train_set.var.head(num_samples).setConstant(m_setting_->logodd_variance);
        return m_sp_gp_.Update(m_setting_->parallel);
    }

    template<typename Dtype, int Dim>
    void
    SpGpOccupancyMap<Dtype, Dim>::Predict(
        const Eigen::Ref<const MatrixDX> &points,
        bool compute_gradient,
        bool parallel,
        VectorX &logodd,
        MatrixDX &gradient) const {
        auto test_result = m_sp_gp_.Test(points, compute_gradient);
        if (logodd.rows() < points.cols()) { logodd.resize(points.cols()); }
        test_result->GetMean(0, logodd, parallel);
        if (compute_gradient) {
            if (gradient.cols() < points.cols()) { gradient.resize(Dim, points.cols()); }
            (void) test_result->GetGradient(0, gradient, parallel);
        }
    }

    template<typename Dtype, int Dim>
    void
    SpGpOccupancyMap<Dtype, Dim>::Predict(
        const VectorD &point,
        bool compute_gradient,
        Dtype &logodd,
        VectorD &gradient) const {
        auto test_result = m_sp_gp_.Test(point, compute_gradient);
        test_result->GetMean(0, 0, logodd);
        if (compute_gradient) { (void) test_result->GetGradient(0, 0, gradient.data()); }
    }

    template<typename Dtype, int Dim>
    void
    SpGpOccupancyMap<Dtype, Dim>::PredictGradient(
        const Eigen::Ref<const MatrixDX> &points,
        bool parallel,
        MatrixDX &gradient) const {
        auto test_result = m_sp_gp_.Test(points, true);
        (void) test_result->GetGradient(0, gradient, parallel);
    }

    template<typename Dtype, int Dim>
    bool
    SpGpOccupancyMap<Dtype, Dim>::Write(std::ostream &s) const {
        using namespace common;
        static const TokenWriteFunctionPairs<SpGpOccupancyMap> token_function_pairs = {
            {
                "setting",
                [](const SpGpOccupancyMap *self, std::ostream &stream) {
                    return self->m_setting_->Write(stream) && stream.good();
                },
            },
            {
                "sp_gp",
                [](const SpGpOccupancyMap *self, std::ostream &stream) {
                    return self->m_sp_gp_.Write(stream) && stream.good();
                },
            },
            {
                "map_boundary",
                [](const SpGpOccupancyMap *self, std::ostream &stream) {
                    return stream.write(
                               reinterpret_cast<const char *>(self->m_map_boundary_.center.data()),
                               sizeof(Dtype) * Dim) &&
                           stream.write(
                               reinterpret_cast<const char *>(
                                   self->m_map_boundary_.half_sizes.data()),
                               sizeof(Dtype) * Dim) &&
                           stream.good();
                },
            },
            {
                "generator",
                [](const SpGpOccupancyMap *self, std::ostream &stream) {
                    stream << self->m_generator_ << '\n';
                    return stream.good();
                },
            },
        };
        return WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype, int Dim>
    bool
    SpGpOccupancyMap<Dtype, Dim>::Read(std::istream &s) {
        using namespace common;
        static const TokenReadFunctionPairs<SpGpOccupancyMap> token_function_pairs = {
            {
                "setting",
                [](SpGpOccupancyMap *self, std::istream &stream) {
                    return self->m_setting_->Read(stream) && stream.good();
                },
            },
            {
                "sp_gp",
                [](SpGpOccupancyMap *self, std::istream &stream) {
                    return self->m_sp_gp_.Read(stream) && stream.good();
                },
            },
            {
                "map_boundary",
                [](SpGpOccupancyMap *self, std::istream &stream) {
                    return stream.read(
                               reinterpret_cast<char *>(self->m_map_boundary_.center.data()),
                               sizeof(Dtype) * Dim) &&
                           stream.read(
                               reinterpret_cast<char *>(self->m_map_boundary_.half_sizes.data()),
                               sizeof(Dtype) * Dim) &&
                           stream.good();
                },
            },
            {
                "generator",
                [](SpGpOccupancyMap *self, std::istream &stream) {
                    stream >> self->m_generator_;
                    SkipLine(stream);
                    return stream.good();
                },
            },
        };
        return ReadTokens(s, this, token_function_pairs);
    }

    template<typename Dtype, int Dim>
    bool
    SpGpOccupancyMap<Dtype, Dim>::operator==(const SpGpOccupancyMap &other) const {
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr &&
            (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) {
            return false;
        }
        if (m_sp_gp_ != other.m_sp_gp_) { return false; }
        if (m_map_boundary_ != other.m_map_boundary_) { return false; }
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    SpGpOccupancyMap<Dtype, Dim>::operator!=(const SpGpOccupancyMap &other) const {
        return !(*this == other);
    }

    template class SpGpOccupancyMap<double, 3>;
    template class SpGpOccupancyMap<float, 3>;
    template class SpGpOccupancyMap<double, 2>;
    template class SpGpOccupancyMap<float, 2>;
}  // namespace erl::gaussian_process

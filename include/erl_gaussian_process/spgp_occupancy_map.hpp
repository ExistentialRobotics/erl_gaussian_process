#pragma once

#include "sparse_pseudo_input_gp.hpp"

#include "erl_geometry/occupancy_map.hpp"

namespace erl::gaussian_process {

    template<typename Dtype, int Dim>
    class SpGpOccupancyMap : geometry::OccupancyMap<Dtype, Dim> {
    public:
        using SpGp = SparsePseudoInputGaussianProcess<Dtype>;
        using SpGpSetting = typename SpGp::Setting;
        using AabbD = geometry::Aabb<Dtype, Dim>;
        using MatrixDX = Eigen::Matrix<Dtype, Dim, Eigen::Dynamic>;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorD = Eigen::Vector<Dtype, Dim>;
        using VectorX = Eigen::VectorX<Dtype>;

        struct Setting : common::Yamlable<Setting> {
            std::shared_ptr<SpGpSetting> sp_gp = std::make_shared<SpGpSetting>();
            // maximum distance from the sensor to consider a point as occupied.
            Dtype max_distance = 30.0f;
            // number of free points to sample per meter from the sensor.
            Dtype free_points_per_meter = 2.0f;
            // percentage margin to use when sampling free points to avoid sampling too close to the
            // surface or the sensor.
            Dtype free_sampling_margin = 0.05f;
            // if true, update the SpGp model with parallel processing.
            bool parallel = true;
            // logodd value for points that are free.
            Dtype logodd_free = -5.0f;
            // logodd value for points that are occupied.
            Dtype logodd_occupied = 5.0f;
            // variance of the logodd values.
            Dtype logodd_variance = 0.0001f;

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

    private:
        std::shared_ptr<Setting> m_setting_ = nullptr;
        SpGp m_sp_gp_;
        AabbD m_map_boundary_;
        std::mt19937_64 m_generator_;

    public:
        SpGpOccupancyMap() = delete;

        SpGpOccupancyMap(
            std::shared_ptr<Setting> setting,
            MatrixX pseudo_points,
            AabbD map_boundary,
            uint64_t seed);

        [[nodiscard]] const SpGp &
        GetSpGp() const {
            return m_sp_gp_;
        }

        /**
         * Generate a dataset of {x, y} where x is the position and y is the occupancy label (1 for
         * occupied, 0 for free).
         * @param sensor_position the position of the sensor in the world frame.
         * @param points point cloud in the world frame of the sensor measurement.
         * @param point_indices indices of the points in the point cloud that are valid for dataset.
         * If empty, all points will be used.
         * @param max_dataset_size maximum number of points in the dataset. -1 means no limit.
         * @param num_samples number of points in the dataset.
         * @param dataset_points points in the dataset.
         * @param dataset_labels labels of the points in the dataset.
         * @param hit_indices indices of the points that are occupied.
         * @return
         */
        void
        GenerateDataset(
            const Eigen::Ref<const VectorD> &sensor_position,
            const Eigen::Ref<const MatrixDX> &points,
            const std::vector<long> &point_indices,
            long max_dataset_size,
            long &num_samples,
            MatrixDX &dataset_points,
            VectorX &dataset_labels,
            std::vector<long> &hit_indices);

        bool
        Update(
            const Eigen::Ref<const VectorD> &sensor_position,
            const Eigen::Ref<const MatrixDX> &points,
            const std::vector<long> &point_indices,
            long &num_samples,
            MatrixDX &dataset_points,
            VectorX &dataset_labels,
            std::vector<long> &hit_indices);

        /**
         *
         * @param points Matrix of points in the world frame. Each column is a point.
         * @param compute_gradient If true, the gradient will be computed.
         * @param parallel If true, the computation will be parallelized.
         * @param logodd Output vector of occupancy probabilities or log-odds.
         * @param gradient Output matrix of gradients. If compute_gradient is false, this will not
         * be used.
         */
        void
        Predict(
            const Eigen::Ref<const MatrixDX> &points,
            bool compute_gradient,
            bool parallel,
            VectorX &logodd,
            MatrixDX &gradient) const;

        void
        Predict(const VectorD &point, bool compute_gradient, Dtype &logodd, VectorD &gradient)
            const;

        void
        PredictGradient(const Eigen::Ref<const MatrixDX> &points, bool parallel, MatrixDX &gradient)
            const;

        [[nodiscard]] bool
        Write(std::ostream &s) const;

        [[nodiscard]] bool
        Read(std::istream &s);

        [[nodiscard]] bool
        operator==(const SpGpOccupancyMap &other) const;

        [[nodiscard]] bool
        operator!=(const SpGpOccupancyMap &other) const;
    };

    using SpGpOccupancyMap2Df = SpGpOccupancyMap<float, 2>;
    using SpGpOccupancyMap2Dd = SpGpOccupancyMap<double, 2>;
    using SpGpOccupancyMap3Df = SpGpOccupancyMap<float, 3>;
    using SpGpOccupancyMap3Dd = SpGpOccupancyMap<double, 3>;

    extern template class SpGpOccupancyMap<double, 3>;
    extern template class SpGpOccupancyMap<float, 3>;
    extern template class SpGpOccupancyMap<double, 2>;
    extern template class SpGpOccupancyMap<float, 2>;
}  // namespace erl::gaussian_process

template<>
struct YAML::convert<erl::gaussian_process::SpGpOccupancyMap2Df::Setting>
    : erl::gaussian_process::SpGpOccupancyMap2Df::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gaussian_process::SpGpOccupancyMap2Dd::Setting>
    : erl::gaussian_process::SpGpOccupancyMap2Dd::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gaussian_process::SpGpOccupancyMap3Df::Setting>
    : erl::gaussian_process::SpGpOccupancyMap3Df::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gaussian_process::SpGpOccupancyMap3Dd::Setting>
    : erl::gaussian_process::SpGpOccupancyMap3Dd::Setting::YamlConvertImpl {};

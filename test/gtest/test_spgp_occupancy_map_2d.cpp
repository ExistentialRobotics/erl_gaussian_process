#include "erl_common/serialization.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_covariance/radial_bias_function.hpp"
#include "erl_gaussian_process/spgp_occupancy_map.hpp"
#include "erl_geometry/lidar_2d.hpp"
#include "erl_geometry/space_2d.hpp"

#include <boost/program_options.hpp>

using namespace erl::common;
using namespace erl::geometry;
using namespace erl::covariance;
using namespace erl::gaussian_process;

template<typename Dtype>
std::vector<Eigen::Vector3<Dtype>>
GenerateTrajectory(const int n = 50, const int repeats = 1) {
    const int total = n * repeats;
    std::vector<Eigen::Vector3<Dtype>> trajectory;
    trajectory.reserve(total);
    const Dtype angle_step = 2 * M_PI / n;
    constexpr Dtype a = 1.6;
    constexpr Dtype b = 1.2;
    Dtype angle = 0;
    for (int i = 0; i < n; ++i) {
        Dtype pose_angle = 0;
        if (i > 0) {
            pose_angle = std::atan2(
                trajectory[i].y() - trajectory[i - 1].y(),
                trajectory[i].x() - trajectory[i - 1].x());
        }
        trajectory.emplace_back(a * std::cos(angle), b * std::sin(angle), pose_angle);
        angle += angle_step;
    }
    for (int r = 1; r < repeats; ++r) {
        trajectory.insert(trajectory.end(), trajectory.begin(), trajectory.begin() + n);
    }
    return trajectory;
}

std::shared_ptr<Space2D>
GenerateSpace() {

    std::vector<Eigen::Vector2d> ordered_object_vertices;

    int n = 50;
    Eigen::Matrix2Xd pts_circle1(2, n);
    double angle_step = 2 * M_PI / static_cast<double>(n);
    double angle = 0;
    for (int i = 0; i < n; ++i) {
        constexpr double r1 = 0.3;
        constexpr double x1 = -1.0;
        constexpr double y1 = 0.2;
        pts_circle1.col(i) << r1 * std::cos(angle) + x1, r1 * std::sin(angle) + y1;
        angle += angle_step;
    }

    n = 100;
    Eigen::Matrix2Xd pts_circle2(2, n);
    angle_step = 2 * M_PI / static_cast<double>(n);
    angle = 0;
    for (int i = 0; i < n; ++i) {
        constexpr double r2 = 0.8;
        constexpr double x2 = 0.3;
        constexpr double y2 = 0.0;
        pts_circle2.col(i) << r2 * std::cos(angle) + x2, r2 * std::sin(angle) + y2;
        angle += angle_step;
    }

    n = 40;
    Eigen::Matrix2Xd pts_box(2, n * 4);
    constexpr double half_size = 2.0;
    const double step = 2 * half_size / static_cast<double>(n);
    double v = -half_size;
    for (int i = 0; i < n; ++i) {
        int j = i;
        pts_box.col(j) << -half_size, v;
        j += n;
        pts_box.col(j) << v, half_size;
        j += n;
        pts_box.col(j) << half_size, -v;
        j += n;
        pts_box.col(j) << -v, -half_size;
        v += step;
    }

    std::vector<Eigen::Ref<const Eigen::Matrix2Xd>> pts;
    pts.reserve(3);
    pts.emplace_back(pts_circle1);
    pts.emplace_back(pts_circle2);
    pts.emplace_back(pts_box);

    Eigen::VectorXb outside_flags(3);
    outside_flags << true, true, false;

    return std::make_shared<Space2D>(pts, outside_flags);
}

template<typename Dtype>
std::pair<GridMapInfo2D<Dtype>, Eigen::Matrix2X<Dtype>>
GenerateGridPoints(const int grid_size, const Aabb<Dtype, 2> &map_boundary) {
    GridMapInfo2D<Dtype> grid_map(
        Eigen::Vector2i(grid_size, grid_size),
        map_boundary.min(),
        map_boundary.max());
    Eigen::Matrix2X<Dtype> points = grid_map.GenerateMeterCoordinates(true);
    return {grid_map, points};
}

template<typename Dtype>
std::tuple<cv::Mat, cv::Mat, cv::Mat>
VisualizeResult(
    const SpGpOccupancyMap<Dtype, 2> &sp_gp_occupancy_map,
    const GridMapInfo2D<Dtype> &grid_map,
    const std::vector<Eigen::Vector2<Dtype>> &waypoints,
    const std::vector<Eigen::Matrix2X<Dtype>> &scanned_points,
    const Eigen::VectorX<Dtype> &prob_occupied,
    const Eigen::Matrix2X<Dtype> &gradient_grid,
    const Eigen::Matrix2X<Dtype> &surface_points,
    const Eigen::Matrix2X<Dtype> &gradient_surf) {

    const long img_size = grid_map.Shape(0);
    using MatrixX = Eigen::MatrixX<Dtype>;
    MatrixX eigen_img = Eigen::Map<const MatrixX>(prob_occupied.data(), img_size, img_size);
    cv::Mat img_prob_occupied, img_prob_occupied_rgb;
    cv::eigen2cv(eigen_img, img_prob_occupied);
    cv::normalize(img_prob_occupied, img_prob_occupied, 0, 255, cv::NORM_MINMAX);
    img_prob_occupied.convertTo(img_prob_occupied, CV_8UC1);
    cv::applyColorMap(img_prob_occupied, img_prob_occupied_rgb, cv::COLORMAP_JET);
    eigen_img = eigen_img.unaryExpr([](Dtype v) -> Dtype { return v > 0 ? 1.0f : 0.0f; });
    cv::Mat img_prob_occupied_binary;
    cv::eigen2cv(eigen_img, img_prob_occupied_binary);
    cv::normalize(img_prob_occupied_binary, img_prob_occupied_binary, 0, 255, cv::NORM_MINMAX);

    Eigen::VectorX<Dtype> gradient_norm = gradient_grid.colwise().norm();
    eigen_img = Eigen::Map<MatrixX>(gradient_norm.data(), img_size, img_size);
    cv::Mat img_gradient_norm, img_gradient_norm_rgb;
    cv::eigen2cv(eigen_img, img_gradient_norm);
    cv::normalize(img_gradient_norm, img_gradient_norm, 0, 255, cv::NORM_MINMAX);
    img_gradient_norm.convertTo(img_gradient_norm, CV_8UC1);
    cv::applyColorMap(img_gradient_norm, img_gradient_norm_rgb, cv::COLORMAP_JET);

    cv::flip(img_prob_occupied_rgb, img_prob_occupied_rgb, 0);
    cv::flip(img_prob_occupied_binary, img_prob_occupied_binary, 0);
    cv::flip(img_gradient_norm_rgb, img_gradient_norm_rgb, 0);

    const int s = 800 / img_prob_occupied_rgb.rows;
    cv::resize(
        img_prob_occupied_rgb,
        img_prob_occupied_rgb,
        cv::Size(),
        s,
        s,
        cv::INTER_LINEAR_EXACT);
    cv::resize(
        img_prob_occupied_binary,
        img_prob_occupied_binary,
        cv::Size(),
        s,
        s,
        cv::INTER_LINEAR_EXACT);
    cv::resize(
        img_gradient_norm_rgb,
        img_gradient_norm_rgb,
        cv::Size(),
        s,
        s,
        cv::INTER_LINEAR_EXACT);

    GridMapInfo2D<Dtype> grid_map_scaled(grid_map.Shape() * s, grid_map.Min(), grid_map.Max());

    // visualize scanned points
    std::vector<std::vector<cv::Point>> pts(1);
    for (std::size_t idx = 0; idx < waypoints.size(); ++idx) {
        const Eigen::Vector2<Dtype> &waypoint = waypoints[idx];
        Eigen::Vector2i px = grid_map_scaled.MeterToPixelForPoints(waypoint);
        pts[0].emplace_back(px[0], px[1]);

        const Eigen::Matrix2X<Dtype> &points = scanned_points[idx];
        for (long i = 0; i < points.cols(); ++i) {
            px = grid_map_scaled.MeterToPixelForPoints(points.col(i));
            cv::circle(
                img_prob_occupied_rgb,
                cv::Point(px[0], px[1]),
                1,
                cv::Scalar(0, 255, 0),
                -1);
        }
    }
    cv::polylines(img_prob_occupied_rgb, pts, false, cv::Scalar(0, 255, 0), 1, cv::LINE_AA, 0);

    // visualize gradient
    for (long i = 0; i < surface_points.cols(); ++i) {
        Eigen::Vector2i px = grid_map_scaled.MeterToPixelForPoints(surface_points.col(i));
        const cv::Point pt1(px[0], px[1]);
        Eigen::Vector2<Dtype> gradient = gradient_surf.col(i);
        gradient.normalize();
        px += grid_map_scaled.MeterToPixelForVectors(-gradient * 0.125);
        const cv::Point pt2(px[0], px[1]);
        cv::arrowedLine(img_prob_occupied_rgb, pt1, pt2, cv::Scalar(255, 255, 255), 2);
        cv::arrowedLine(img_gradient_norm_rgb, pt1, pt2, cv::Scalar(255, 255, 255), 2);
        cv::circle(img_gradient_norm_rgb, pt1, 1, cv::Scalar(0, 0, 255), -1);
    }
    cv::polylines(img_gradient_norm_rgb, pts, false, cv::Scalar(0, 255, 0), 1, cv::LINE_AA, 0);

    // visualize weights
    Eigen::VectorX<Dtype> mu = sp_gp_occupancy_map.GetSpGp().GetMatAlpha().col(0);
    // Apply x^0.25 to the weights for visualization
    mu = mu.unaryExpr([](Dtype v) { return v > 0 ? std::pow(v, 0.25f) : -std::pow(-v, 0.25f); });
    cv::Mat mu_color;
    cv::eigen2cv(mu, mu_color);
    cv::normalize(mu_color, mu_color, 0, 255, cv::NORM_MINMAX);
    mu_color.convertTo(mu_color, CV_8UC1);
    cv::applyColorMap(mu_color, mu_color, cv::COLORMAP_JET);
    const Eigen::Matrix2X<Dtype> &hinged_points = sp_gp_occupancy_map.GetSpGp().GetPseudoPoints();
    cv::Mat img_bhm_weights(img_prob_occupied_rgb.rows, img_prob_occupied_rgb.cols, CV_8UC3);
    img_bhm_weights.setTo(0);
    for (long i = 0; i < hinged_points.cols(); ++i) {
        Eigen::Vector2i px = grid_map_scaled.MeterToPixelForPoints(hinged_points.col(i));
        cv::circle(
            img_bhm_weights,
            cv::Point(px[0], px[1]),
            8,
            mu_color.at<cv::Vec3b>(static_cast<int>(i), 0),
            -1);
    }
    cv::polylines(img_bhm_weights, pts, false, cv::Scalar(0, 255, 0), 1, cv::LINE_AA, 0);

    cv::imshow("Probability Occupied", img_prob_occupied_rgb);
    cv::imshow("Probability Occupied (binary)", img_prob_occupied_binary);
    cv::imshow("Gradient Norm", img_gradient_norm_rgb);
    cv::imshow("Induced-point Weights", img_bhm_weights);
    cv::waitKey(10);

    return {img_prob_occupied_rgb, img_gradient_norm_rgb, img_bhm_weights};
}

template<typename Dtype>
void
TestIo(
    const SpGpOccupancyMap<Dtype, 2> &sp_gp_occupancy_map,
    const Eigen::MatrixX<Dtype> &pseudo_points,
    const Aabb<Dtype, 2> &map_boundary) {
    GTEST_PREPARE_OUTPUT_DIR();
    std::string filename = fmt::format("test_bhm_2d_{}.bin", type_name<Dtype>());
    filename = test_output_dir / filename;
    using Serializer = erl::common::Serialization<SpGpOccupancyMap<Dtype, 2>>;
    ASSERT_TRUE(Serializer::Write(filename, &sp_gp_occupancy_map));
    auto setting = std::make_shared<typename SpGpOccupancyMap<Dtype, 2>::Setting>();
    setting->sp_gp->kernel->x_dim = 2;
    setting->sp_gp->kernel_type = type_name<RadialBiasFunction<Dtype, 2>>();
    SpGpOccupancyMap<Dtype, 2> sp_gp_occupancy_map_read(setting, pseudo_points, map_boundary, 1);
    ASSERT_TRUE(Serializer::Read(filename, &sp_gp_occupancy_map_read));
    ASSERT_TRUE(sp_gp_occupancy_map == sp_gp_occupancy_map_read);
}

template<typename Dtype>
void
TestImpl2D(const int hinged_grid_size, const int test_grid_size, const std::string &config_file) {

    GTEST_PREPARE_OUTPUT_DIR();
    auto setting = std::make_shared<typename SpGpOccupancyMap<Dtype, 2>::Setting>();
    if (std::filesystem::exists(config_file)) {
        ASSERT_TRUE(setting->FromYamlFile(config_file));
    } else {
        setting->AsYamlFile(config_file);
    }

    // setting->sp_gp->kernel_type = type_name<RadialBiasFunction<Dtype, 2>>();
    // setting->sp_gp->kernel->x_dim = 2;
    // setting->sp_gp->kernel->scale = std::sqrt(0.5 / rbf_gamma);
    // setting->sp_gp->use_sparse = use_sparse;
    // setting->sp_gp->max_num_samples = max_dataset_size;

    Aabb<Dtype, 2> map_boundary(Eigen::Vector2<Dtype>(-3.0, -3.0), Eigen::Vector2<Dtype>(3.0, 3.0));
    Eigen::Matrix2X<Dtype> hinged_points =
        GenerateGridPoints(hinged_grid_size, map_boundary).second;
    SpGpOccupancyMap<Dtype, 2> sp_gp_occupancy_map(setting, hinged_points, map_boundary, 0);

    TestIo<Dtype>(sp_gp_occupancy_map, hinged_points, map_boundary);

    std::vector<Eigen::Vector3<Dtype>> trajectory = GenerateTrajectory<Dtype>(50, 1);

    const auto lidar_setting = std::make_shared<Lidar2D::Setting>();
    lidar_setting->max_angle = 135.0 / 180.0 * M_PI;   // 135 degrees
    lidar_setting->min_angle = -135.0 / 180.0 * M_PI;  // -135 degrees
    lidar_setting->num_lines = 135;
    const std::shared_ptr<Space2D> space = GenerateSpace();
    const Lidar2D lidar(lidar_setting, space);
    const Eigen::Matrix2Xd ray_dirs_frame = lidar.GetRayDirectionsInFrame();

    const auto [test_grid, test_points] = GenerateGridPoints(test_grid_size, map_boundary);
    const Eigen::Matrix2X<Dtype> surf_points = space->GetSurface()->vertices.cast<Dtype>();
    Eigen::VectorX<Dtype> predicted_logodd(test_points.cols());
    Eigen::Matrix2X<Dtype> predicted_gradient(2, test_points.cols());
    Eigen::Matrix2X<Dtype> predicted_gradient_surf(2, surf_points.cols());

    std::vector<Eigen::Vector2<Dtype>> waypoints;
    std::vector<Eigen::Matrix2X<Dtype>> scanned_points;
    long cnt = 0;
    std::filesystem::path prob_occupied_dir = test_output_dir / "prob_occupied";
    std::filesystem::create_directories(prob_occupied_dir);
    std::filesystem::path gradient_norms_dir = test_output_dir / "gradient_norms";
    std::filesystem::create_directories(gradient_norms_dir);
    std::filesystem::path bhm_weights_dir = test_output_dir / "bhm_weights";
    std::filesystem::create_directories(bhm_weights_dir);
    for (const auto &pose: trajectory) {
        auto scan = lidar.Scan(
            static_cast<double>(pose[2]),
            pose.template head<2>().template cast<double>(),
            true);
        const Eigen::Matrix2<Dtype> rotation = Eigen::Rotation2D<Dtype>(pose[2]).toRotationMatrix();
        Eigen::Matrix2X<Dtype> points(2, scan.size());
        for (long i = 0; i < scan.size(); ++i) {
            points.col(i) << scan[i] * (rotation * ray_dirs_frame.col(i).cast<Dtype>()) +
                                 pose.template head<2>();
        }
        scanned_points.push_back(points);
        waypoints.push_back(pose.template head<2>());

        long num_points = 0;
        Eigen::Matrix2X<Dtype> dataset_points;
        Eigen::VectorX<Dtype> dataset_labels;
        std::vector<long> hit_indices;

        sp_gp_occupancy_map.Update(
            pose.template head<2>(),
            points,
            std::vector<long>{},
            num_points,
            dataset_points,
            dataset_labels,
            hit_indices);

        // predict and visualize
        constexpr bool parallel = true;
        sp_gp_occupancy_map.Predict(
            test_points,
            true, /*compute_gradient*/
            parallel,
            predicted_logodd,
            predicted_gradient);
        sp_gp_occupancy_map.PredictGradient(surf_points, parallel, predicted_gradient_surf);
        auto [img_prob_occupied_rgb, img_gradient_norm_rgb, img_bhm_weights] = VisualizeResult(
            sp_gp_occupancy_map,
            test_grid,
            waypoints,
            scanned_points,
            predicted_logodd,
            predicted_gradient,
            surf_points,
            predicted_gradient_surf);

        std::string filename = fmt::format("{:04d}.png", cnt++);
        cv::imwrite(prob_occupied_dir / filename, img_prob_occupied_rgb);
        cv::imwrite(gradient_norms_dir / filename, img_gradient_norm_rgb);
        cv::imwrite(bhm_weights_dir / filename, img_bhm_weights);
    }

    cv::waitKey(0);

    TestIo<Dtype>(sp_gp_occupancy_map, hinged_points, map_boundary);
}

struct Options {
    int hinged_grid_size = 31;
    int test_grid_size = 100;
    std::string config_file = ERL_GAUSSIAN_PROCESS_ROOT_DIR "/config/spgp_occupancy_map_2d.yaml";
};

Options g_options;

TEST(SpGpOccupancyMap, 2Dd) {
    TestImpl2D<double>(g_options.hinged_grid_size, g_options.test_grid_size, g_options.config_file);
}

TEST(SpGpOccupancyMap, 2Df) {
    TestImpl2D<float>(g_options.hinged_grid_size, g_options.test_grid_size, g_options.config_file);
}

int
main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);

    try {
        namespace po = boost::program_options;
        po::options_description desc;
        // clang-format off
        desc.add_options()
            ("help,h", "Show help message")
            ("hinged-grid-size", po::value<int>(&g_options.hinged_grid_size)->default_value(g_options.hinged_grid_size), "Size of the hinged grid")
            ("test-grid-size", po::value<int>(&g_options.test_grid_size)->default_value(g_options.test_grid_size), "Size of the test grid")
            ("config-file", po::value<std::string>(&g_options.config_file)->default_value(g_options.config_file), "Path to the configuration file for SpGpOccupancyMap");
        // clang-format on
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        if (vm.count("help")) {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl << desc << std::endl;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception &e) {
        ERL_ERROR("Error parsing command line arguments: {}", e.what());
        return -1;
    }

    return RUN_ALL_TESTS();
}

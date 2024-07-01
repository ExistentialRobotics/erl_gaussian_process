#include "erl_common/opencv.hpp"
#include "erl_common/random.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_gaussian_process/range_sensor_gp_3d.hpp"
#include "erl_geometry/depth_camera_3d.hpp"
#include "erl_geometry/euler_angle.hpp"
#include "erl_geometry/lidar_3d.hpp"

#include <open3d/geometry/LineSet.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/t/geometry/RaycastingScene.h>
#include <open3d/visualization/utility/DrawGeometry.h>

TEST(RangeSensorGp3D, Lidar) {
    GTEST_PREPARE_OUTPUT_DIR();

    const auto gp_setting = std::make_shared<erl::gaussian_process::RangeSensorGaussianProcess3D::Setting>();
    const auto lidar_frame_setting = std::make_shared<erl::geometry::LidarFrame3D::Setting>();
    lidar_frame_setting->azimuth_min = -M_PI * 3 / 4;
    lidar_frame_setting->azimuth_max = M_PI * 3 / 4;
    lidar_frame_setting->num_azimuth_lines = 271;
    lidar_frame_setting->elevation_min = -M_PI / 2;
    lidar_frame_setting->elevation_max = M_PI / 2;
    lidar_frame_setting->num_elevation_lines = 91;
    gp_setting->range_sensor_frame_type = "lidar";
    gp_setting->range_sensor_frame = lidar_frame_setting;
    gp_setting->row_group_size = 10;
    gp_setting->row_overlap_size = 3;
    gp_setting->row_margin = 0;
    gp_setting->col_group_size = 10;
    gp_setting->col_overlap_size = 3;
    gp_setting->col_margin = 0;
    erl::gaussian_process::RangeSensorGaussianProcess3D gp(gp_setting);
    const auto lidar_frame = std::dynamic_pointer_cast<const erl::geometry::LidarFrame3D>(gp.GetRangeSensorFrame());

    const auto mesh = open3d::io::CreateMeshFromFile(gtest_src_dir / "replica-office-1.ply");
    const auto lidar_setting = std::make_shared<erl::geometry::Lidar3D::Setting>();
    lidar_setting->azimuth_min = lidar_frame_setting->azimuth_min;
    lidar_setting->azimuth_max = lidar_frame_setting->azimuth_max;
    lidar_setting->num_azimuth_lines = lidar_frame_setting->num_azimuth_lines;
    lidar_setting->elevation_min = lidar_frame_setting->elevation_min;
    lidar_setting->elevation_max = lidar_frame_setting->elevation_max;
    lidar_setting->num_elevation_lines = lidar_frame_setting->num_elevation_lines;
    erl::geometry::Lidar3D lidar(lidar_setting, mesh->vertices_, mesh->triangles_);

    srand(0);                                                               // NOLINT(*-msc51-cpp)
    Eigen::Vector3d rpy = (Eigen::Vector3d::Random() * 2.0).array() - 1.0;  // [-1, 1]
    rpy[0] *= M_PI_4;                                                       // roll
    rpy[1] *= M_PI_4;                                                       // pitch
    rpy[2] *= M_PI;                                                         // yaw
    // rpy.setZero();                                                          // for debug
    const Eigen::Matrix3d rotation = erl::geometry::EulerToRotation3D(rpy[0], rpy[1], rpy[2], erl::geometry::EulerAngleOrder::kRxyz);
    const Eigen::Vector3d translation = mesh->GetCenter();

    // generate training data
    Eigen::MatrixXd ranges;
    {
        erl::common::BlockTimer<std::chrono::milliseconds> timer("lidar.Scan");
        (void) timer;
        ranges = lidar.Scan(rotation, translation);
    }

    // train
    {
        erl::common::BlockTimer<std::chrono::milliseconds> timer("gp.Train");
        (void) timer;
        ASSERT_TRUE(gp.Train(rotation, translation, ranges));
    }

    // test
    constexpr long n_test = 10000;
    const Eigen::VectorXd test_azimuths = (Eigen::VectorXd::Random(n_test).array() * 2.0 - 1.0) * M_PI;
    const Eigen::VectorXd test_elevations = (Eigen::VectorXd::Random(n_test).array() * 2.0 - 1.0) * M_PI / 2;
    Eigen::Matrix3Xd directions_world(3, test_azimuths.size());
    open3d::t::geometry::RaycastingScene scene;
    scene.AddTriangles(open3d::t::geometry::TriangleMesh::FromLegacy(*mesh));
    open3d::core::Tensor rays({test_azimuths.size(), 6}, open3d::core::Dtype::Float32);
    for (long i = 0; i < test_azimuths.size(); ++i) {
        directions_world.col(i) = erl::common::AzimuthElevationToDirection(test_azimuths[i], test_elevations[i]);
        rays[i][0] = translation[0];
        rays[i][1] = translation[1];
        rays[i][2] = translation[2];
        rays[i][3] = directions_world(0, i);
        rays[i][4] = directions_world(1, i);
        rays[i][5] = directions_world(2, i);
    }
    auto result = scene.CastRays(rays);
    const Eigen::VectorXd vec_ranges_gt = Eigen::Map<Eigen::VectorXf>(result["t_hit"].GetDataPtr<float>(), test_azimuths.size()).cast<double>();
    Eigen::VectorXd vec_ranges(test_azimuths.size());
    Eigen::VectorXd vec_ranges_var(test_azimuths.size());
    {
        erl::common::BlockTimer<std::chrono::milliseconds> timer("gp.Test");
        (void) timer;
        ASSERT_TRUE(gp.Test(directions_world, false, vec_ranges, vec_ranges_var, true, true));
    }

    // cast invalid test queries and compute mse
    double mse = 0;
    std::vector<long> invalid_indices;
    Eigen::VectorXd vec_ranges_invalid;
    if (long n_invalid = (vec_ranges_var.array() > 100).count(); n_invalid > 0) {
        open3d::core::Tensor rays_invalid({n_invalid, 6}, open3d::core::Dtype::Float32);
        auto *rays_invalid_ptr = rays_invalid.GetDataPtr<float>();
        const Eigen::Vector3f translation_f = translation.cast<float>();
        for (long i = 0; i < test_azimuths.size(); ++i) {
            if (vec_ranges_var[i] <= 100) {
                mse += std::pow(vec_ranges[i] - vec_ranges_gt[i], 2);
                continue;
            }
            invalid_indices.push_back(i);
            Eigen::Vector3f ray_direction = directions_world.col(i).cast<float>();
            std::copy_n(translation_f.data(), 3, rays_invalid_ptr);
            std::copy_n(ray_direction.data(), 3, rays_invalid_ptr + 3);
            rays_invalid_ptr += 6;
        }
        mse /= static_cast<double>(test_azimuths.size() - n_invalid);
        ERL_INFO("n_invalid: {}/{}", n_invalid, test_azimuths.size());
        auto result_invalid = scene.CastRays(rays_invalid);
        vec_ranges_invalid = Eigen::Map<Eigen::VectorXf>(result_invalid["t_hit"].GetDataPtr<float>(), n_invalid).cast<double>();
    }
    ERL_INFO("mse: {}", mse);

    // visualize
    const long n_azimuths = lidar_frame->GetNumAzimuthLines();
    const long n_elevations = lidar_frame->GetNumElevationLines();

    const Eigen::MatrixX<Eigen::Vector3d> &end_points_in_world = lidar_frame->GetEndPointsInWorld();
    const auto line_set_rays = std::make_shared<open3d::geometry::LineSet>();
    line_set_rays->points_.reserve(n_azimuths * n_elevations + 1);
    line_set_rays->points_.push_back(translation);
    line_set_rays->points_.insert(line_set_rays->points_.end(), end_points_in_world.data(), end_points_in_world.data() + n_azimuths * n_elevations);
    line_set_rays->lines_.reserve(n_azimuths * n_elevations);
    for (long i = 0; i < end_points_in_world.size(); ++i) { line_set_rays->lines_.emplace_back(0, i + 1); }
    line_set_rays->PaintUniformColor({1.0, 0.5, 0.0});

    const auto point_cloud_train = std::make_shared<open3d::geometry::PointCloud>();
    point_cloud_train->points_.insert(point_cloud_train->points_.end(), line_set_rays->points_.begin() + 1, line_set_rays->points_.end());
    point_cloud_train->PaintUniformColor({0.0, 1.0, 0.0});

    const auto point_cloud_test = std::make_shared<open3d::geometry::PointCloud>();
    point_cloud_test->points_.reserve(test_azimuths.size() - vec_ranges_invalid.size());
    for (long i = 0; i < test_azimuths.size(); ++i) {
        if (vec_ranges_var[i] > 100) { continue; }
        point_cloud_test->points_.emplace_back(translation + directions_world.col(i) * vec_ranges[i]);
    }
    point_cloud_test->PaintUniformColor({1.0, 0.0, 0.0});

    const auto point_cloud_test_invalid = std::make_shared<open3d::geometry::PointCloud>();
    point_cloud_test_invalid->points_.reserve(vec_ranges_invalid.size());
    for (long i = 0; i < vec_ranges_invalid.size(); ++i) {
        point_cloud_test_invalid->points_.emplace_back(translation + directions_world.col(invalid_indices[i]) * vec_ranges_invalid[i]);
    }
    point_cloud_test_invalid->PaintUniformColor({0.0, 0.0, 1.0});

    open3d::visualization::DrawGeometries(
        {mesh,
         // line_set_rays,
         // point_cloud_train,
         point_cloud_test,
         point_cloud_test_invalid},
        test_info->name(),
        1600,
        900);
}

TEST(RangeSensorGp3D, Depth) {
    GTEST_PREPARE_OUTPUT_DIR();

    const auto gp_setting = std::make_shared<erl::gaussian_process::RangeSensorGaussianProcess3D::Setting>();
    const auto depth_frame_setting = std::make_shared<erl::geometry::DepthFrame3D::Setting>();
    gp_setting->range_sensor_frame_type = "depth";
    gp_setting->range_sensor_frame = depth_frame_setting;
    gp_setting->row_group_size = 10;
    gp_setting->row_overlap_size = 3;
    gp_setting->row_margin = 0;
    gp_setting->col_group_size = 10;
    gp_setting->col_overlap_size = 3;
    gp_setting->col_margin = 0;
    erl::gaussian_process::RangeSensorGaussianProcess3D gp(gp_setting);

    const auto mesh = open3d::io::CreateMeshFromFile(gtest_src_dir / "replica-hotel-0.ply");
    const auto depth_camera_setting = std::make_shared<erl::geometry::DepthCamera3D::Setting>();
    depth_camera_setting->image_height = depth_frame_setting->image_height;
    depth_camera_setting->image_width = depth_frame_setting->image_width;
    depth_camera_setting->camera_cx = depth_frame_setting->camera_cx;
    depth_camera_setting->camera_cy = depth_frame_setting->camera_cy;
    depth_camera_setting->camera_fx = depth_frame_setting->camera_fx;
    depth_camera_setting->camera_fy = depth_frame_setting->camera_fy;
    depth_frame_setting->camera_to_optical = erl::geometry::DepthCamera3D::CameraToOptical();
    erl::geometry::DepthCamera3D depth_camera(depth_camera_setting, mesh->vertices_, mesh->triangles_);

    srand(0);                                                               // NOLINT(*-msc51-cpp)
    Eigen::Vector3d rpy = (Eigen::Vector3d::Random() * 2.0).array() - 1.0;  // [-1, 1]
    rpy[0] *= M_PI_4;                                                       // roll
    rpy[1] *= M_PI_4;                                                       // pitch
    rpy[2] *= M_PI;                                                         // yaw
    rpy.setZero();                                                          // for debug
    rpy[1] = -M_PI_2;
    const Eigen::Matrix3d rotation = erl::geometry::EulerToRotation3D(rpy[0], rpy[1], rpy[2], erl::geometry::EulerAngleOrder::kRxyz);
    const Eigen::Vector3d translation = mesh->GetCenter();

    // generate training data
    Eigen::MatrixXd real_depths;
    {
        erl::common::BlockTimer<std::chrono::milliseconds> timer("depth_camera.Scan");
        (void) timer;
        real_depths = depth_camera.Scan(rotation, translation);
    }
    ERL_INFO("real_depths: min={}, max={}, shape={}x{}", real_depths.minCoeff(), real_depths.maxCoeff(), real_depths.rows(), real_depths.cols());
    const Eigen::MatrixXd depths = real_depths * depth_frame_setting->depth_scale;
    (void) depths;

    // // visualize depth image
    // const Eigen::MatrixX8U depth_image = (real_depths.array() / real_depths.maxCoeff() * 255).cast<uint8_t>();
    // cv::Mat depth_image_cv;
    // cv::eigen2cv(depth_image, depth_image_cv);
    // // cv::applyColorMap(depth_image_cv, depth_image_cv, cv::COLORMAP_JET);
    // cv::imshow("Depth Image", depth_image_cv);
    // cv::waitKey(0);

    // train
    {
        erl::common::BlockTimer<std::chrono::milliseconds> timer("gp.Train");
        (void) timer;
        ASSERT_TRUE(gp.Train(rotation, translation, depths));
    }

    // test
    constexpr long n_test = 10000;
    const Eigen::VectorXd test_azimuths = (Eigen::VectorXd::Random(n_test).array() * 2.0 - 1.0) * M_PI;
    const Eigen::VectorXd test_elevations = (Eigen::VectorXd::Random(n_test).array() * 2.0 - 1.0) * M_PI_2;
    Eigen::Matrix3Xd directions_world(3, test_azimuths.size());
    open3d::t::geometry::RaycastingScene scene;
    scene.AddTriangles(open3d::t::geometry::TriangleMesh::FromLegacy(*mesh));
    open3d::core::Tensor rays({test_azimuths.size(), 6}, open3d::core::Dtype::Float32);
    for (long i = 0; i < test_azimuths.size(); ++i) {
        directions_world.col(i) = erl::common::AzimuthElevationToDirection(test_azimuths[i], test_elevations[i]);
        rays[i][0] = translation[0];
        rays[i][1] = translation[1];
        rays[i][2] = translation[2];
        rays[i][3] = directions_world(0, i);
        rays[i][4] = directions_world(1, i);
        rays[i][5] = directions_world(2, i);
    }
    auto result = scene.CastRays(rays);
    const Eigen::VectorXd vec_ranges_gt = Eigen::Map<Eigen::VectorXf>(result["t_hit"].GetDataPtr<float>(), test_azimuths.size()).cast<double>();
    Eigen::VectorXd vec_ranges(test_azimuths.size());
    Eigen::VectorXd vec_ranges_var(test_azimuths.size());
    {
        erl::common::BlockTimer<std::chrono::milliseconds> timer("gp.Test");
        (void) timer;
        ASSERT_TRUE(gp.Test(directions_world, false, vec_ranges, vec_ranges_var, true, true));
    }

    // cast invalid test queries
    double mse = 0;
    std::vector<long> invalid_indices;
    Eigen::VectorXd vec_ranges_invalid;
    if (long n_invalid = (vec_ranges_var.array() > 100).count(); n_invalid > 0) {
        open3d::core::Tensor rays_invalid({n_invalid, 6}, open3d::core::Dtype::Float32);
        auto *rays_invalid_ptr = rays_invalid.GetDataPtr<float>();
        const Eigen::Vector3f translation_f = translation.cast<float>();
        for (long i = 0; i < test_azimuths.size(); ++i) {
            if (vec_ranges_var[i] <= 100) {
                mse += std::pow(vec_ranges[i] - vec_ranges_gt[i], 2);
                continue;
            }
            invalid_indices.push_back(i);
            Eigen::Vector3f ray_direction = directions_world.col(i).cast<float>();
            std::copy_n(translation_f.data(), 3, rays_invalid_ptr);
            std::copy_n(ray_direction.data(), 3, rays_invalid_ptr + 3);
            rays_invalid_ptr += 6;
        }
        mse /= static_cast<double>(test_azimuths.size() - n_invalid);
        ERL_INFO("n_invalid: {}/{}", n_invalid, test_azimuths.size());
        auto result_invalid = scene.CastRays(rays_invalid);
        vec_ranges_invalid = Eigen::Map<Eigen::VectorXf>(result_invalid["t_hit"].GetDataPtr<float>(), n_invalid).cast<double>();
    }
    ERL_INFO("mse: {}", mse);

    // visualize
    const auto depth_frame = std::dynamic_pointer_cast<const erl::geometry::DepthFrame3D>(gp.GetRangeSensorFrame());
    const long image_height = depth_frame->GetImageHeight();
    const long image_width = depth_frame->GetImageWidth();
    if (!gp.IsTrained()) { std::const_pointer_cast<erl::geometry::DepthFrame3D>(depth_frame)->UpdateRanges(rotation, translation, real_depths, false); }

    const Eigen::MatrixX<Eigen::Vector3d> &end_points_in_world = depth_frame->GetEndPointsInWorld();
    const auto line_set_rays = std::make_shared<open3d::geometry::LineSet>();
    line_set_rays->points_.reserve(image_height * image_width + 1);
    line_set_rays->points_.push_back(translation);
    line_set_rays->points_.insert(line_set_rays->points_.end(), end_points_in_world.data(), end_points_in_world.data() + image_height * image_width);
    line_set_rays->lines_.reserve(image_height * image_width);
    for (long i = 0; i < end_points_in_world.size(); ++i) { line_set_rays->lines_.emplace_back(0, i + 1); }
    line_set_rays->PaintUniformColor({1.0, 0.5, 0.0});

    const auto point_cloud_train = std::make_shared<open3d::geometry::PointCloud>();
    point_cloud_train->points_.insert(point_cloud_train->points_.end(), line_set_rays->points_.begin() + 1, line_set_rays->points_.end());
    point_cloud_train->PaintUniformColor({0.0, 1.0, 0.0});

    const auto point_cloud_test = std::make_shared<open3d::geometry::PointCloud>();
    point_cloud_test->points_.reserve(test_azimuths.size() - vec_ranges_invalid.size());
    for (long i = 0; i < test_azimuths.size(); ++i) {
        if (vec_ranges_var[i] > 100) { continue; }
        point_cloud_test->points_.emplace_back(translation + directions_world.col(i) * vec_ranges[i]);
    }
    point_cloud_test->PaintUniformColor({1.0, 0.0, 0.0});

    const auto point_cloud_test_invalid = std::make_shared<open3d::geometry::PointCloud>();
    point_cloud_test_invalid->points_.reserve(vec_ranges_invalid.size());
    for (long i = 0; i < vec_ranges_invalid.size(); ++i) {
        point_cloud_test_invalid->points_.emplace_back(translation + directions_world.col(invalid_indices[i]) * vec_ranges_invalid[i]);
    }
    point_cloud_test_invalid->PaintUniformColor({0.0, 0.0, 1.0});

    // const auto point_cloud_test_gt = std::make_shared<open3d::geometry::PointCloud>();
    // point_cloud_test_gt->points_.reserve(test_azimuths.size());
    // for (long i = 0; i < test_azimuths.size(); ++i) { point_cloud_test_gt->points_.emplace_back(translation + directions_world.col(i) * vec_ranges_gt[i]); }
    // point_cloud_test_gt->PaintUniformColor({0.0, 1.0, 1.0});

    open3d::visualization::DrawGeometries(
        {mesh,
         // line_set_rays,
         // point_cloud_train,
         point_cloud_test,
         point_cloud_test_invalid},
        test_info->name(),
        1600,
        900);
}

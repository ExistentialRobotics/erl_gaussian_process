#include "erl_common/test_helper.hpp"
#include "erl_gaussian_process/range_sensor_gp_3d.hpp"

#include <erl_common/random.hpp>
#include <erl_geometry/euler_angle.hpp>
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
    erl::gaussian_process::RangeSensorGaussianProcess3D gp(gp_setting);

    const auto mesh = open3d::io::CreateMeshFromFile(gtest_src_dir / "replica-office-1.ply");
    open3d::t::geometry::RaycastingScene scene;
    scene.AddTriangles(open3d::t::geometry::TriangleMesh::FromLegacy(*mesh));

    srand(0);                                                               // NOLINT(*-msc51-cpp)
    Eigen::Vector3d rpy = (Eigen::Vector3d::Random() * 2.0).array() - 1.0;  // [-1, 1]
    rpy[0] *= M_PI_4;                                                       // roll
    rpy[1] *= M_PI_4;                                                       // pitch
    rpy[2] *= M_PI;                                                         // yaw
    // rpy.setZero();                                                          // for debug
    const Eigen::Matrix3d rotation = erl::geometry::EulerToRotation3D(rpy[0], rpy[1], rpy[2], erl::geometry::EulerAngleOrder::kRxyz);
    const Eigen::Vector3d translation = mesh->GetCenter();

    const auto lidar_frame = std::dynamic_pointer_cast<const erl::geometry::LidarFrame3D>(gp.GetRangeSensorFrame());
    const long n_azimuths = lidar_frame->GetNumAzimuthLines();
    const long n_elevations = lidar_frame->GetNumElevationLines();
    const Eigen::MatrixX<Eigen::Vector3d> &ray_directions_in_frame = lidar_frame->GetRayDirectionsInFrame();
    open3d::core::Tensor rays({n_azimuths, n_elevations, 6}, open3d::core::Dtype::Float32);
    auto *rays_ptr = rays.GetDataPtr<float>();
    const Eigen::Vector3f translation_f = translation.cast<float>();
    for (long i = 0; i < n_azimuths; ++i) {
        for (long j = 0; j < n_elevations; ++j, rays_ptr += 6) {
            std::copy_n(translation_f.data(), 3, rays_ptr);
            Eigen::Vector3f ray_direction = (rotation * ray_directions_in_frame(i, j)).cast<float>();
            std::copy_n(ray_direction.data(), 3, rays_ptr + 3);
        }
    }
    auto result = scene.CastRays(rays);
    const Eigen::MatrixXd ranges = Eigen::Map<Eigen::MatrixXf>(result["t_hit"].GetDataPtr<float>(), n_elevations, n_azimuths).transpose().cast<double>();

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
    std::vector<Eigen::Vector3d> directions_world;
    directions_world.reserve(test_azimuths.size());
    for (long i = 0; i < test_azimuths.size(); ++i) {
        directions_world.emplace_back(erl::common::AzimuthElevationToDirection(test_azimuths[i], test_elevations[i]));
    }
    Eigen::VectorXd vec_ranges(test_azimuths.size());
    Eigen::VectorXd vec_ranges_var(test_azimuths.size());
    {
        erl::common::BlockTimer<std::chrono::milliseconds> timer("gp.Test");
        (void) timer;
        ASSERT_TRUE(gp.Test(directions_world, vec_ranges, vec_ranges_var, true, true));
    }

    // cast invalid test queries
    std::vector<long> invalid_indices;
    Eigen::VectorXd vec_ranges_invalid;
    if (long n_invalid = (vec_ranges_var.array() > 100).count(); n_invalid > 0) {
        open3d::core::Tensor rays_invalid({n_invalid, 6}, open3d::core::Dtype::Float32);
        auto *rays_invalid_ptr = rays_invalid.GetDataPtr<float>();
        for (long i = 0; i < test_azimuths.size(); ++i) {
            if (vec_ranges_var[i] <= 100) { continue; }
            invalid_indices.push_back(i);
            Eigen::Vector3f ray_direction = directions_world[i].cast<float>();
            std::copy_n(translation_f.data(), 3, rays_invalid_ptr);
            std::copy_n(ray_direction.data(), 3, rays_invalid_ptr + 3);
            rays_invalid_ptr += 6;
        }
        auto result_invalid = scene.CastRays(rays_invalid);
        vec_ranges_invalid = Eigen::Map<Eigen::VectorXf>(result_invalid["t_hit"].GetDataPtr<float>(), n_invalid).cast<double>();
    }

    // visualize
    const auto line_set_rays = std::make_shared<open3d::geometry::LineSet>();
    line_set_rays->points_.reserve(n_azimuths * n_elevations + 1);
    line_set_rays->points_.push_back(translation);
    line_set_rays->lines_.reserve(n_azimuths * n_elevations);
    for (long j = 0; j < n_elevations; ++j) {
        for (long i = 0; i < n_azimuths; ++i) {
            const Eigen::Vector3d ray_end = translation + rotation * ray_directions_in_frame(i, j) * ranges(i, j);
            line_set_rays->points_.push_back(ray_end);
            line_set_rays->lines_.push_back({0, line_set_rays->points_.size() - 1});
        }
    }
    line_set_rays->PaintUniformColor({1.0, 0.5, 0.0});

    const auto point_cloud_train = std::make_shared<open3d::geometry::PointCloud>();
    point_cloud_train->points_.insert(point_cloud_train->points_.end(), line_set_rays->points_.begin() + 1, line_set_rays->points_.end());
    point_cloud_train->PaintUniformColor({0.0, 1.0, 0.0});

    const auto point_cloud_test = std::make_shared<open3d::geometry::PointCloud>();
    const auto point_cloud_test_invalid = std::make_shared<open3d::geometry::PointCloud>();
    point_cloud_test->points_.reserve(test_azimuths.size() - vec_ranges_invalid.size());
    point_cloud_test_invalid->points_.reserve(vec_ranges_invalid.size());
    for (long i = 0; i < test_azimuths.size(); ++i) {
        if (vec_ranges_var[i] > 100) { continue; }
        point_cloud_test->points_.emplace_back(translation + directions_world[i] * vec_ranges[i]);
    }
    point_cloud_test->PaintUniformColor({1.0, 0.0, 0.0});
    for (long i = 0; i < vec_ranges_invalid.size(); ++i) {
        point_cloud_test_invalid->points_.emplace_back(translation + directions_world[invalid_indices[i]] * vec_ranges_invalid[i]);
    }
    point_cloud_test_invalid->PaintUniformColor({0.0, 0.0, 1.0});

    open3d::visualization::DrawGeometries(
        {mesh, line_set_rays, point_cloud_train, point_cloud_test, point_cloud_test_invalid},
        test_info->name(),
        1600,
        900,
        50,
        50);
}

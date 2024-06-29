#include "erl_common/test_helper.hpp"
#include "erl_gaussian_process/range_sensor_gp_3d.hpp"

#include <erl_common/random.hpp>
#include <erl_geometry/euler_angle.hpp>

#include <open3d/geometry/LineSet.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/t/geometry/RaycastingScene.h>

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
    auto ts_hit = result["t_hit"].ToFlatVector<float>();
    Eigen::MatrixXd ranges = Eigen::Map<Eigen::MatrixXf>(result["t_hit"].GetDataPtr<float>(), n_elevations, n_azimuths).transpose().cast<double>();

    // visualize
    const auto line_set_rays = std::make_shared<open3d::geometry::LineSet>();

    ASSERT_TRUE(gp.Train(rotation, translation, std::move(ranges)));
}

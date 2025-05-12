#include "erl_common/opencv.hpp"
#include "erl_common/random.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_covariance/ornstein_uhlenbeck.hpp"
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

using Dtype = double;
using RangeSensorGaussianProcess3D = std::conditional_t<
    sizeof(Dtype) == 4,
    erl::gaussian_process::RangeSensorGaussianProcess3Df,
    erl::gaussian_process::RangeSensorGaussianProcess3Dd>;
using Matrix3 = Eigen::Matrix3<Dtype>;
using Vector3 = Eigen::Vector3<Dtype>;
using Matrix3X = Eigen::Matrix<Dtype, 3, Eigen::Dynamic>;
using MatrixX = Eigen::MatrixX<Dtype>;
using VectorX = Eigen::VectorX<Dtype>;
using LidarFrame = erl::geometry::LidarFrame3D<Dtype>;
using Lidar = erl::geometry::Lidar3D<Dtype>;
using DepthFrame = erl::geometry::DepthFrame3D<Dtype>;
using DepthCamera = erl::geometry::DepthCamera3D<Dtype>;
static std::filesystem::path kProjectDir = ERL_GAUSSIAN_PROCESS_ROOT_DIR;

TEST(RangeSensorGp3D, Lidar) {
    using namespace erl::common;
    GTEST_PREPARE_OUTPUT_DIR();
    const auto gp_setting = std::make_shared<RangeSensorGaussianProcess3D::Setting>();
    const auto lidar_frame_setting = std::make_shared<LidarFrame::Setting>();
    lidar_frame_setting->azimuth_min = -M_PI * 3 / 4;
    lidar_frame_setting->azimuth_max = M_PI * 3 / 4;
    lidar_frame_setting->num_azimuth_lines = 271;
    lidar_frame_setting->elevation_min = -M_PI / 2;
    lidar_frame_setting->elevation_max = M_PI / 2;
    lidar_frame_setting->num_elevation_lines = 91;
    gp_setting->gp->kernel_type = type_name<erl::covariance::OrnsteinUhlenbeck2d>();
    gp_setting->sensor_frame_type = type_name<LidarFrame>();
    gp_setting->sensor_frame_setting_type = type_name<LidarFrame::Setting>();
    gp_setting->sensor_frame = lidar_frame_setting;
    gp_setting->row_group_size = 10;
    gp_setting->row_overlap_size = 3;
    gp_setting->row_margin = 0;
    gp_setting->col_group_size = 10;
    gp_setting->col_overlap_size = 3;
    gp_setting->col_margin = 0;
    RangeSensorGaussianProcess3D gp(gp_setting);
    const auto lidar_frame = std::dynamic_pointer_cast<const LidarFrame>(gp.GetSensorFrame());

    const std::string mesh_file = kProjectDir / "data/replica-office-1.ply";
    const auto mesh = open3d::io::CreateMeshFromFile(mesh_file);
    const auto lidar_setting = std::make_shared<Lidar::Setting>();
    lidar_setting->azimuth_min = lidar_frame_setting->azimuth_min;
    lidar_setting->azimuth_max = lidar_frame_setting->azimuth_max;
    lidar_setting->num_azimuth_lines = lidar_frame_setting->num_azimuth_lines;
    lidar_setting->elevation_min = lidar_frame_setting->elevation_min;
    lidar_setting->elevation_max = lidar_frame_setting->elevation_max;
    lidar_setting->num_elevation_lines = lidar_frame_setting->num_elevation_lines;
    Lidar lidar(lidar_setting);
    lidar.AddMesh(mesh_file);

    srand(0);                                               // NOLINT(*-msc51-cpp)
    Vector3 rpy = (Vector3::Random() * 2.0).array() - 1.0;  // [-1, 1]
    rpy[0] *= M_PI_4;                                       // roll
    rpy[1] *= M_PI_4;                                       // pitch
    rpy[2] *= M_PI;                                         // yaw
    const Matrix3 rotation =
        EulerToRotation3D(rpy[0], rpy[1], rpy[2], erl::geometry::EulerAngleOrder::kRxyz);
    const Vector3 translation = mesh->GetCenter().cast<Dtype>();

    // generate training data
    MatrixX ranges;
    {
        ERL_BLOCK_TIMER_MSG("lidar.Scan");
        ranges = lidar.Scan(rotation, translation);
    }

    // train
    {
        ERL_BLOCK_TIMER_MSG("gp.Train");
        ASSERT_TRUE(gp.Train(rotation, translation, ranges));
    }

    // test
    constexpr long n_test = 10000;
    const VectorX test_azimuths = (VectorX::Random(n_test).array() * 2.0 - 1.0) * M_PI;
    const VectorX test_elevations = (VectorX::Random(n_test).array() * 2.0 - 1.0) * M_PI / 2;
    Matrix3X directions_world(3, test_azimuths.size());
    open3d::t::geometry::RaycastingScene scene;
    scene.AddTriangles(open3d::t::geometry::TriangleMesh::FromLegacy(*mesh));
    open3d::core::Tensor rays({test_azimuths.size(), 6}, open3d::core::Dtype::Float32);
    for (long i = 0; i < test_azimuths.size(); ++i) {
        directions_world.col(i) =
            AzimuthElevationToDirection<Dtype>(test_azimuths[i], test_elevations[i]);
        rays[i][0] = translation[0];
        rays[i][1] = translation[1];
        rays[i][2] = translation[2];
        rays[i][3] = directions_world(0, i);
        rays[i][4] = directions_world(1, i);
        rays[i][5] = directions_world(2, i);
    }
    auto result = scene.CastRays(rays);
    const VectorX vec_ranges_gt =
        Eigen::Map<Eigen::VectorXf>(result["t_hit"].GetDataPtr<float>(), test_azimuths.size())
            .cast<Dtype>();
    VectorX vec_ranges(test_azimuths.size());
    Eigen::VectorXb success;
    {
        ERL_BLOCK_TIMER_MSG("gp.Test");
        auto test_result = gp.Test(directions_world, false, true);
        success = test_result->GetMean(vec_ranges, true /*parallel*/);
        ASSERT_TRUE(success.any());
    }

    // cast invalid test queries and compute mse
    double mse = 0;
    std::vector<long> invalid_indices;
    VectorX vec_ranges_invalid;
    if (long n_invalid = success.size() - success.count(); n_invalid > 0) {
        invalid_indices.reserve(n_invalid);
        open3d::core::Tensor rays_invalid({n_invalid, 6}, open3d::core::Dtype::Float32);
        auto *rays_invalid_ptr = rays_invalid.GetDataPtr<float>();
        const Eigen::Vector3f &translation_f = translation.cast<float>();
        for (long i = 0; i < test_azimuths.size(); ++i) {
            if (success[i]) {
                mse += std::pow(vec_ranges[i] - vec_ranges_gt[i], 2);
                continue;
            }
            invalid_indices.push_back(i);
            Eigen::Vector3f ray_direction = directions_world.col(i).cast<float>();
            std::copy_n(translation_f.data(), 3, rays_invalid_ptr);
            std::copy_n(ray_direction.data(), 3, rays_invalid_ptr + 3);
            rays_invalid_ptr += 6;
        }
        ERL_ASSERT(n_invalid == static_cast<long>(invalid_indices.size()));
        mse /= static_cast<double>(test_azimuths.size() - n_invalid);
        ERL_INFO("n_invalid: {}/{}", n_invalid, test_azimuths.size());
        auto result_invalid = scene.CastRays(rays_invalid);
        vec_ranges_invalid =
            Eigen::Map<Eigen::VectorXf>(result_invalid["t_hit"].GetDataPtr<float>(), n_invalid)
                .cast<Dtype>();
    }
    ERL_INFO("mse: {}", mse);  // mse: 0.0004142553492453702
    EXPECT_LE(mse, 0.00042);

    // visualize
    const long n_azimuths = lidar_frame->GetNumAzimuthLines();
    const long n_elevations = lidar_frame->GetNumElevationLines();

    const Eigen::MatrixX<Vector3> &end_points_in_world = lidar_frame->GetEndPointsInWorld();
    const auto line_set_rays = std::make_shared<open3d::geometry::LineSet>();
    line_set_rays->points_.reserve(n_azimuths * n_elevations + 1);
    line_set_rays->lines_.reserve(n_azimuths * n_elevations);
    line_set_rays->points_.emplace_back(translation.cast<double>());
    for (long i = 0; i < end_points_in_world.size(); ++i) {
        line_set_rays->points_.emplace_back(end_points_in_world.data()[i].cast<double>());
        line_set_rays->lines_.emplace_back(0, i + 1);
    }
    line_set_rays->PaintUniformColor({1.0, 0.5, 0.0});

    const auto point_cloud_train = std::make_shared<open3d::geometry::PointCloud>();
    point_cloud_train->points_.insert(
        point_cloud_train->points_.end(),
        line_set_rays->points_.begin() + 1,
        line_set_rays->points_.end());
    point_cloud_train->PaintUniformColor({0.0, 1.0, 0.0});

    const auto point_cloud_test = std::make_shared<open3d::geometry::PointCloud>();
    point_cloud_test->points_.reserve(test_azimuths.size() - vec_ranges_invalid.size());
    for (long i = 0; i < test_azimuths.size(); ++i) {
        if (!success[i]) { continue; }
        point_cloud_test->points_.emplace_back(
            (translation + directions_world.col(i) * vec_ranges[i]).cast<double>());
    }
    point_cloud_test->PaintUniformColor({1.0, 0.0, 0.0});
    ERL_INFO("Valid point cloud[red]: {} points", point_cloud_test->points_.size());

    const auto point_cloud_test_invalid = std::make_shared<open3d::geometry::PointCloud>();
    point_cloud_test_invalid->points_.reserve(vec_ranges_invalid.size());
    for (long i = 0; i < vec_ranges_invalid.size(); ++i) {
        point_cloud_test_invalid->points_.emplace_back(
            (translation + directions_world.col(invalid_indices[i]) * vec_ranges_invalid[i])
                .cast<double>());
    }
    point_cloud_test_invalid->PaintUniformColor({0.0, 0.0, 1.0});
    ERL_INFO("Invalid point cloud[blue]: {} points", point_cloud_test_invalid->points_.size());

    open3d::visualization::DrawGeometries(
        {mesh,
         // line_set_rays,
         // point_cloud_train,
         point_cloud_test,
         point_cloud_test_invalid},
        test_info->name(),
        1600,
        900);

    EXPECT_TRUE(Serialization<RangeSensorGaussianProcess3D>::Write(
        test_output_dir / "range_sensor_gp_3d_lidar.bin",
        gp));
    RangeSensorGaussianProcess3D gp_read(std::make_shared<RangeSensorGaussianProcess3D::Setting>());
    ASSERT_TRUE(Serialization<RangeSensorGaussianProcess3D>::Read(
        test_output_dir / "range_sensor_gp_3d_lidar.bin",
        gp_read));
    EXPECT_TRUE(gp == gp_read);
}

TEST(RangeSensorGp3D, Depth) {
    using namespace erl::common;
    GTEST_PREPARE_OUTPUT_DIR();

    const auto gp_setting = std::make_shared<RangeSensorGaussianProcess3D::Setting>();
    const auto depth_frame_setting = std::make_shared<DepthFrame::Setting>();
    gp_setting->gp->kernel_type = type_name<erl::covariance::OrnsteinUhlenbeck2d>();
    gp_setting->sensor_frame_type = type_name<DepthFrame>();
    gp_setting->sensor_frame_setting_type = type_name<DepthFrame::Setting>();
    gp_setting->sensor_frame = depth_frame_setting;
    gp_setting->row_group_size = 10;
    gp_setting->row_overlap_size = 3;
    gp_setting->row_margin = 0;
    gp_setting->col_group_size = 10;
    gp_setting->col_overlap_size = 3;
    gp_setting->col_margin = 0;
    std::cout << "gp_setting:\n" << *gp_setting << std::endl;
    RangeSensorGaussianProcess3D gp(gp_setting);

    const std::string mesh_file = kProjectDir / "data/replica-office-1.ply";
    const auto mesh = open3d::io::CreateMeshFromFile(mesh_file);
    const auto depth_camera_setting = std::make_shared<DepthCamera::Setting>();
    *depth_camera_setting = depth_frame_setting->camera_intrinsic;
    std::cout << "depth_camera_setting:\n" << *depth_camera_setting << std::endl;
    std::cout << "depth_frame_setting:\n" << *depth_frame_setting << std::endl;
    erl::geometry::DepthCamera3D depth_camera(depth_camera_setting);
    depth_camera.AddMesh(mesh_file);

    srand(10);                                              // NOLINT(*-msc51-cpp)
    Vector3 rpy = (Vector3::Random() * 2.0).array() - 1.0;  // [-1, 1]
    rpy[0] *= M_PI_4;                                       // roll
    rpy[1] *= M_PI_4;                                       // pitch
    rpy[2] *= M_PI;                                         // yaw
    Matrix3 cam_rotation =
        EulerToRotation3D(rpy[0], rpy[1], rpy[2], erl::geometry::EulerAngleOrder::kRxyz);
    Vector3 cam_translation = mesh->GetCenter().cast<Dtype>();

    // generate training data
    MatrixX real_depths;
    Matrix3 optical_rotation;
    Vector3 optical_translation;
    {
        ERL_BLOCK_TIMER_MSG("depth_camera.Scan");
        real_depths = depth_camera.Scan(cam_rotation, cam_translation);
        // convert camera pose to optical pose
        std::tie(optical_rotation, optical_translation) =
            depth_camera.GetOpticalPose(cam_rotation, cam_translation);
    }
    ERL_INFO(
        "real_depths: min={}, max={}, shape={}x{}",
        real_depths.minCoeff(),
        real_depths.maxCoeff(),
        real_depths.rows(),
        real_depths.cols());

    // visualize depth image
    const Eigen::MatrixX8U depth_image =
        (real_depths.array() / real_depths.maxCoeff() * 255).cast<uint8_t>();
    cv::Mat depth_image_cv;
    cv::eigen2cv(depth_image, depth_image_cv);
    // cv::applyColorMap(depth_image_cv, depth_image_cv, cv::COLORMAP_JET);
    cv::imshow("Depth Image", depth_image_cv);
    cv::waitKey(100);

    // train
    {
        ERL_BLOCK_TIMER_MSG("gp.Train");
        ASSERT_TRUE(gp.Train(optical_rotation, optical_translation, real_depths));
    }

    // test
    constexpr long n_test = 10000;
    const VectorX test_azimuths = (VectorX::Random(n_test).array() * 2.0 - 1.0) * M_PI;
    const VectorX test_elevations = (VectorX::Random(n_test).array() * 2.0 - 1.0) * M_PI_2;
    Matrix3X directions_world(3, test_azimuths.size());
    open3d::t::geometry::RaycastingScene scene;
    scene.AddTriangles(open3d::t::geometry::TriangleMesh::FromLegacy(*mesh));
    open3d::core::Tensor rays({test_azimuths.size(), 6}, open3d::core::Dtype::Float32);
    for (long i = 0; i < test_azimuths.size(); ++i) {
        directions_world.col(i) = AzimuthElevationToDirection(test_azimuths[i], test_elevations[i]);
        rays[i][0] = optical_translation[0];
        rays[i][1] = optical_translation[1];
        rays[i][2] = optical_translation[2];
        rays[i][3] = directions_world(0, i);
        rays[i][4] = directions_world(1, i);
        rays[i][5] = directions_world(2, i);
    }
    auto result = scene.CastRays(rays);
    const VectorX vec_ranges_gt =
        Eigen::Map<Eigen::VectorXf>(result["t_hit"].GetDataPtr<float>(), test_azimuths.size())
            .cast<Dtype>();
    VectorX vec_ranges(test_azimuths.size());
    Eigen::VectorXb success;
    {
        ERL_BLOCK_TIMER_MSG("gp.Test");
        auto test_result = gp.Test(directions_world, false, true);
        success = test_result->GetMean(vec_ranges, true /*parallel*/);
        ASSERT_TRUE(success.any());
    }

    // cast invalid test queries
    double mse = 0;
    std::vector<long> invalid_indices;
    VectorX vec_ranges_invalid;
    if (long n_invalid = success.size() - success.count(); n_invalid > 0) {
        open3d::core::Tensor rays_invalid({n_invalid, 6}, open3d::core::Dtype::Float32);
        auto *rays_invalid_ptr = rays_invalid.GetDataPtr<float>();
        const Eigen::Vector3f &translation_f = optical_translation.cast<float>();
        for (long i = 0; i < test_azimuths.size(); ++i) {
            if (success[i]) {
                mse += std::pow(vec_ranges[i] - vec_ranges_gt[i], 2);
                continue;
            }
            invalid_indices.push_back(i);
            const Eigen::Vector3f &ray_direction = directions_world.col(i).cast<float>();
            std::copy_n(translation_f.data(), 3, rays_invalid_ptr);
            std::copy_n(ray_direction.data(), 3, rays_invalid_ptr + 3);
            rays_invalid_ptr += 6;
        }
        mse /= static_cast<double>(test_azimuths.size() - n_invalid);
        ERL_INFO("n_invalid: {}/{}", n_invalid, test_azimuths.size());
        auto result_invalid = scene.CastRays(rays_invalid);
        vec_ranges_invalid =
            Eigen::Map<Eigen::VectorXf>(result_invalid["t_hit"].GetDataPtr<float>(), n_invalid)
                .cast<Dtype>();
    }
    ERL_INFO("mse: {}", mse);  // mse: 0.00021043860239746616
    EXPECT_LE(mse, 0.00022);

    // visualize
    const auto depth_frame = std::dynamic_pointer_cast<const DepthFrame>(gp.GetSensorFrame());
    const long image_height = depth_frame->GetImageHeight();
    const long image_width = depth_frame->GetImageWidth();
    if (!gp.IsTrained()) {
        std::const_pointer_cast<DepthFrame>(depth_frame)
            ->UpdateRanges(optical_rotation, optical_translation, real_depths);
    }

    const Eigen::MatrixX<Vector3> &end_points_in_world = depth_frame->GetEndPointsInWorld();
    const auto line_set_rays = std::make_shared<open3d::geometry::LineSet>();
    line_set_rays->points_.reserve(image_height * image_width + 1);
    line_set_rays->lines_.reserve(image_height * image_width);
    line_set_rays->points_.emplace_back(optical_translation.cast<double>());
    for (long i = 0; i < end_points_in_world.size(); ++i) {
        line_set_rays->points_.emplace_back(end_points_in_world.data()[i].cast<double>());
        line_set_rays->lines_.emplace_back(0, i + 1);
    }
    line_set_rays->PaintUniformColor({1.0, 0.5, 0.0});

    const auto point_cloud_train = std::make_shared<open3d::geometry::PointCloud>();
    point_cloud_train->points_.insert(
        point_cloud_train->points_.end(),
        line_set_rays->points_.begin() + 1,
        line_set_rays->points_.end());
    point_cloud_train->PaintUniformColor({0.0, 1.0, 0.0});

    const auto point_cloud_test = std::make_shared<open3d::geometry::PointCloud>();
    point_cloud_test->points_.reserve(test_azimuths.size() - vec_ranges_invalid.size());
    for (long i = 0; i < test_azimuths.size(); ++i) {
        if (!success[i]) { continue; }
        point_cloud_test->points_.emplace_back(
            (optical_translation + directions_world.col(i) * vec_ranges[i]).cast<double>());
    }
    point_cloud_test->PaintUniformColor({1.0, 0.0, 0.0});
    ERL_INFO("Valid point cloud[red]: {} points", point_cloud_test->points_.size());

    const auto point_cloud_test_invalid = std::make_shared<open3d::geometry::PointCloud>();
    point_cloud_test_invalid->points_.reserve(vec_ranges_invalid.size());
    for (long i = 0; i < vec_ranges_invalid.size(); ++i) {
        point_cloud_test_invalid->points_.emplace_back(
            (optical_translation + directions_world.col(invalid_indices[i]) * vec_ranges_invalid[i])
                .cast<double>());
    }
    point_cloud_test_invalid->PaintUniformColor({0.0, 0.0, 1.0});
    ERL_INFO("Invalid point cloud[blue]: {} points", point_cloud_test_invalid->points_.size());

    auto line_set_camera = open3d::geometry::LineSet::CreateCameraVisualization(
        static_cast<int>(depth_camera_setting->image_width),
        static_cast<int>(depth_camera_setting->image_height),
        depth_camera_setting->GetIntrinsicMatrix().cast<double>(),
        DepthCamera::ComputeExtrinsic(cam_rotation, cam_translation).cast<double>());
    open3d::visualization::DrawGeometries(
        {mesh,
         line_set_camera,
         // line_set_rays,
         // point_cloud_train,
         point_cloud_test,
         point_cloud_test_invalid},
        test_info->name(),
        1600,
        900);

    ASSERT_TRUE(Serialization<RangeSensorGaussianProcess3D>::Write(
        test_output_dir / "range_sensor_gp_3d_depth.bin",
        gp));
    RangeSensorGaussianProcess3D gp_read(std::make_shared<RangeSensorGaussianProcess3D::Setting>());
    ASSERT_TRUE(Serialization<RangeSensorGaussianProcess3D>::Read(
        test_output_dir / "range_sensor_gp_3d_depth.bin",
        gp_read));
    EXPECT_TRUE(gp == gp_read);
}

int
main(int argc, char *argv[]) {
    erl::gaussian_process::Init();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

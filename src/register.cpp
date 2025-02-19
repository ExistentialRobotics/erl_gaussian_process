#include "erl_gaussian_process/register.hpp"

#include "erl_gaussian_process/lidar_gp_2d.hpp"
#include "erl_gaussian_process/mapping.hpp"
#include "erl_gaussian_process/noisy_input_gp.hpp"
#include "erl_gaussian_process/range_sensor_gp_3d.hpp"
#include "erl_gaussian_process/vanilla_gp.hpp"

namespace erl::gaussian_process {

#define REGISTER(x) (void) x::Register<x>()

    const bool kRegistered = []() -> bool {
        REGISTER(NoisyInputGaussianProcess_d::Setting);
        REGISTER(NoisyInputGaussianProcess_f::Setting);
        REGISTER(RangeSensorGaussianProcess3Dd::Setting);
        REGISTER(RangeSensorGaussianProcess3Df::Setting);
        REGISTER(VanillaGaussianProcess_d::Setting);
        REGISTER(VanillaGaussianProcess_f::Setting);
        REGISTER(LidarGaussianProcess2D_d::Setting);
        REGISTER(LidarGaussianProcess2D_f::Setting);
        REGISTER(Mapping_d::Setting);
        REGISTER(Mapping_f::Setting);
        return true;
    }();

}  // namespace erl::gaussian_process

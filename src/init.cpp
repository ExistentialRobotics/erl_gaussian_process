#include "erl_gaussian_process/init.hpp"

#include "erl_covariance/init.hpp"
#include "erl_gaussian_process/lidar_gp_2d.hpp"
#include "erl_gaussian_process/mapping.hpp"
#include "erl_gaussian_process/noisy_input_gp.hpp"
#include "erl_gaussian_process/range_sensor_gp_3d.hpp"
#include "erl_gaussian_process/vanilla_gp.hpp"
#include "erl_geometry/init.hpp"

namespace erl::gaussian_process {

#define REGISTER(x) (void) x::Register<x>()

    bool initialized = false;

    bool
    Init() {
        if (initialized) { return true; }

        if (!covariance::Init()) { return false; }
        if (!geometry::Init()) { return false; }

        REGISTER(NoisyInputGaussianProcessD::Setting);
        REGISTER(NoisyInputGaussianProcessF::Setting);
        REGISTER(RangeSensorGaussianProcess3Dd::Setting);
        REGISTER(RangeSensorGaussianProcess3Df::Setting);
        REGISTER(VanillaGaussianProcessD::Setting);
        REGISTER(VanillaGaussianProcessF::Setting);
        REGISTER(LidarGaussianProcess2Dd::Setting);
        REGISTER(LidarGaussianProcess2Df::Setting);
        REGISTER(MappingD::Setting);
        REGISTER(MappingF::Setting);

        ERL_INFO("erl_gaussian_process initialized");
        initialized = true;

        return true;
    }

}  // namespace erl::gaussian_process
